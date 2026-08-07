"""Single-head FiLM-conditioned UNet for the DIVERGENT (irrotational) velocity
component only.

Same shared encoder / bottleneck / FiLM backbone and multi-res sparse-obs
conditioning (FPN) as ddpm_library's MyUNet_Helmholtz_Split_FiLM_MultiRes, but
with the divergence-free machinery removed entirely:

  - no ψ (stream-function) branch, no curl_from_streamfunction -- that's the
    solenoidal/divergence-free-specific head.
  - single output head: φ (potential) -> grad(φ). A gradient field is exactly
    the correct parameterization for a purely irrotational/divergent field
    (Helmholtz's theorem), so this isn't "divergence-free machinery" -- it's
    the curl-free counterpart, appropriate for what this model is trained on.

Trained with a single standard (non-split) DDPM schedule -- see
divergent_schedule.py -- not the dual solenoidal/irrotational
HelmholtzSplitSchedule.

Input:  (N, 5, H, W) = [x_t(2ch), mask(1ch), cond_u(1ch), cond_v(1ch)]
Output: (N, 2, H, W) = grad(phi)  (divergent velocity estimate)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Inlined building blocks (normally ddpm_library.model.*) ───────────────
# Copied verbatim so this training script has zero dependency on the
# ddpm_library package (and its ~300MB of bundled weight assets) being
# installed on the remote training box.


def sinusoidal_embedding(n: int, d: int) -> torch.Tensor:
    """Fixed sinusoidal position embedding (timestep -> vector)."""
    emb = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    t = torch.arange(n).unsqueeze(1)
    emb[:, 0::2] = torch.sin(t * wk[0::2])
    emb[:, 1::2] = torch.cos(t * wk[1::2])
    return emb


class ResBlock(nn.Module):
    """Residual block with GroupNorm + AdaGN time-embedding modulation."""

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.SiLU()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 2 * out_c))
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        ts = self.time_mlp(t_emb)
        scale, shift = ts.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial dims (H*W sequence length)."""

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class ResAttnBlock(nn.Module):
    """ResBlock optionally followed by self-attention."""

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int,
                 use_attn: bool = False, num_heads: int = 4):
        super().__init__()
        self.res = ResBlock(in_c, out_c, time_emb_dim)
        self.attn = SelfAttention2d(out_c, num_heads=num_heads) if use_attn else None

    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        if self.attn is not None:
            h = self.attn(h)
        return h


class FiLMLayer(nn.Module):
    """Spatial FiLM with GroupNorm (per-pixel modulation)."""

    def __init__(self, cond_channels, feature_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, feature_channels)
        self.scale_conv = nn.Conv2d(cond_channels, feature_channels, 1)
        self.shift_conv = nn.Conv2d(cond_channels, feature_channels, 1)
        nn.init.zeros_(self.scale_conv.weight)
        nn.init.zeros_(self.scale_conv.bias)
        nn.init.zeros_(self.shift_conv.weight)
        nn.init.zeros_(self.shift_conv.bias)

    def forward(self, h, cond):
        gamma = self.scale_conv(cond)
        beta = self.shift_conv(cond)
        return (1 + gamma) * self.norm(h) + beta


class MultiResCondEncoder(nn.Module):
    """Feature Pyramid conditioning encoder for sparse observations."""

    def __init__(self, ch=(64, 128, 256, 256)):
        super().__init__()
        pool_ch = 3
        self.proc5 = nn.Sequential(
            nn.Conv2d(pool_ch, ch[3], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        self.proc4 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[3], ch[3], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        self.proc3 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[3], ch[2], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[2], ch[2], 3, 1, 1), nn.SiLU(),
        )
        self.proc2 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[2], ch[1], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[1], ch[1], 3, 1, 1), nn.SiLU(),
        )
        self.proc1 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[1], ch[0], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[0], ch[0], 3, 1, 1), nn.SiLU(),
        )

    @staticmethod
    def _pool_sparse(obs_uv, known_mask, target_h, target_w):
        H, W = obs_uv.shape[-2:]
        kh, kw = H // target_h, W // target_w
        if kh == 1 and kw == 1:
            density = known_mask
            normalized = obs_uv
        else:
            density = F.avg_pool2d(known_mask, (kh, kw))
            pooled = F.avg_pool2d(obs_uv, (kh, kw))
            safe_density = density.clamp(min=1e-8)
            normalized = pooled / safe_density
            has_obs = (density > 1e-7).float()
            normalized = normalized * has_obs
        return torch.cat([normalized, density], dim=1)

    def forward(self, cond):
        missing_mask = cond[:, :1]
        known_mask = 1.0 - missing_mask
        obs_uv = cond[:, 1:3]

        p5 = self._pool_sparse(obs_uv, known_mask, 4, 8)
        p4 = self._pool_sparse(obs_uv, known_mask, 8, 16)
        p3 = self._pool_sparse(obs_uv, known_mask, 16, 32)
        p2 = self._pool_sparse(obs_uv, known_mask, 32, 64)
        p1 = self._pool_sparse(obs_uv, known_mask, 64, 128)

        c5 = self.proc5(p5)
        c5_up = F.interpolate(c5, size=(8, 16), mode='bilinear', align_corners=False)
        c4 = self.proc4(torch.cat([p4, c5_up], dim=1))
        c4_up = F.interpolate(c4, size=(16, 32), mode='bilinear', align_corners=False)
        c3 = self.proc3(torch.cat([p3, c4_up], dim=1))
        c3_up = F.interpolate(c3, size=(32, 64), mode='bilinear', align_corners=False)
        c2 = self.proc2(torch.cat([p2, c3_up], dim=1))
        c2_up = F.interpolate(c2, size=(64, 128), mode='bilinear', align_corners=False)
        c1 = self.proc1(torch.cat([p1, c2_up], dim=1))

        return c1, c2, c3, c4, c5


class DivergentUNet(nn.Module):
    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 5):
        super().__init__()
        self.in_channels = in_channels
        ch = [64, 128, 256, 256]

        self.cond_encoder = MultiResCondEncoder(ch=ch)

        self.film_enc1 = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])
        self.film_enc2 = FiLMLayer(cond_channels=ch[1], feature_channels=ch[1])
        self.film_enc3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[2])
        self.film_enc4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])
        self.film_mid = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])
        self.film_dec4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[2])
        self.film_dec3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[1])
        self.film_dec2_phi = FiLMLayer(cond_channels=ch[1], feature_channels=ch[0])
        self.film_dec1_phi = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])

        self.time_embed_table = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed_table.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed_table.requires_grad_(False)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.enc1 = nn.ModuleList([
            ResAttnBlock(2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)

        self.enc2 = nn.ModuleList([
            ResAttnBlock(ch[0], ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[1], time_emb_dim, use_attn=False),
        ])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)

        self.enc3 = nn.ModuleList([
            ResAttnBlock(ch[1], ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, 2, 1)

        self.enc4 = nn.ModuleList([
            ResAttnBlock(ch[2], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, 2, 1)

        self.mid = nn.ModuleList([
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])

        self.up4 = nn.ConvTranspose2d(ch[3], ch[3], 4, 2, 1)
        self.dec4 = nn.ModuleList([
            ResAttnBlock(ch[3] * 2, ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])

        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, 2, 1)
        self.dec3 = nn.ModuleList([
            ResAttnBlock(ch[2] * 2, ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[1], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # phi (potential) branch -- ONLY head. grad(phi) is curl-free by
        # construction, matching purely-divergent training targets.
        self.up2_phi = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2_phi = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        self.up1_phi = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1_phi = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        self.phi_norm = nn.GroupNorm(8, ch[0])
        self.phi_act = nn.SiLU()
        self.phi_conv = nn.Conv2d(ch[0], 1, 3, 1, 1)
        nn.init.zeros_(self.phi_conv.weight)
        nn.init.zeros_(self.phi_conv.bias)

        dx = torch.tensor([[[[0.0, -0.5, 0.5]]]])
        dy = torch.tensor([[[[0.0], [-0.5], [0.5]]]])
        self.register_buffer("_dx_kernel", dx)
        self.register_buffer("_dy_kernel", dy)

    def _grad_potential(self, phi: torch.Tensor) -> torch.Tensor:
        u_irr = F.conv2d(phi, self._dx_kernel, padding=(0, 1))
        v_irr = F.conv2d(phi, self._dy_kernel, padding=(1, 0))
        return torch.cat([u_irr, v_irr], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        x_t = x[:, :2]
        cond = x[:, 2:]

        c1, c2, c3, c4, c5 = self.cond_encoder(cond)

        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)

        h = x_t
        for block in self.enc1:
            h = block(h, t_emb)
        h = self.film_enc1(h, c1)
        skip1 = h

        h = self.down1(h)
        for block in self.enc2:
            h = block(h, t_emb)
        h = self.film_enc2(h, c2)
        skip2 = h

        h = self.down2(h)
        for block in self.enc3:
            h = block(h, t_emb)
        h = self.film_enc3(h, c3)
        skip3 = h

        h = self.down3(h)
        for block in self.enc4:
            h = block(h, t_emb)
        h = self.film_enc4(h, c4)
        skip4 = h

        h = self.down4(h)
        for block in self.mid:
            h = block(h, t_emb)
        h = self.film_mid(h, c5)

        h = self.up4(h)
        h = torch.cat([skip4, h], dim=1)
        for block in self.dec4:
            h = block(h, t_emb)
        h = self.film_dec4(h, c4)

        h = self.up3(h)
        h = torch.cat([skip3, h], dim=1)
        for block in self.dec3:
            h = block(h, t_emb)
        h = self.film_dec3(h, c3)

        h_phi = self.up2_phi(h)
        h_phi = torch.cat([skip2, h_phi], dim=1)
        for block in self.dec2_phi:
            h_phi = block(h_phi, t_emb)
        h_phi = self.film_dec2_phi(h_phi, c2)

        h_phi = self.up1_phi(h_phi)
        h_phi = torch.cat([skip1, h_phi], dim=1)
        for block in self.dec1_phi:
            h_phi = block(h_phi, t_emb)
        h_phi = self.film_dec1_phi(h_phi, c1)

        phi = self.phi_conv(self.phi_act(self.phi_norm(h_phi)))
        return self._grad_potential(phi)
