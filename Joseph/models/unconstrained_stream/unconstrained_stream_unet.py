"""Unconstrained direction network for the stream-model pipeline -- exactly
the same UNet backbone as ddpm_library.stream.stream_model.StreamFunctionUNet,
but WITHOUT the divergence-free constraint.

StreamFunctionUNet predicts a scalar stream function psi and derives velocity
as curl(psi) -- divergence-free by mathematical construction, no matter what
data it's trained on. This class skips that entirely: the backbone predicts
(u, v) directly (out_ch=2), so it can represent any vector field, divergent
or not.

Same input/output contract as StreamFunctionUNet.forward, so it's a drop-in
replacement in the rest of the stream pipeline (sampler, conditioning, fusion).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


def _num_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_fc(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class UNet(nn.Module):
    """Same backbone as ddpm_library.stream.stream_model.UNet, out_ch=2 fixed."""
    _PAD = (2, 2, 1, 1)

    def __init__(self, in_ch: int, base_ch: int = 64, time_dim: int = 256, out_ch: int = 2):
        super().__init__()
        self.time_dim = time_dim
        c = base_ch
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.enc0 = ResBlock(in_ch, c, time_dim)
        self.enc1 = ResBlock(c, c * 2, time_dim)
        self.enc2 = ResBlock(c * 2, c * 4, time_dim)
        self.enc3 = ResBlock(c * 4, c * 8, time_dim)
        self.mid = ResBlock(c * 8, c * 8, time_dim)
        self.dec3 = ResBlock(c * 8 + c * 8, c * 4, time_dim)
        self.dec2 = ResBlock(c * 4 + c * 4, c * 2, time_dim)
        self.dec1 = ResBlock(c * 2 + c * 2, c, time_dim)
        self.dec0 = ResBlock(c + c, c, time_dim)
        self.out_conv = nn.Conv2d(c, out_ch, 1)
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, t):
        x = F.pad(x, self._PAD)
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        e0 = self.enc0(x, t_emb)
        e1 = self.enc1(self.down(e0), t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)
        h = self.mid(self.down(e3), t_emb)

        h = self.up(h)
        h = self.dec3(torch.cat([h, e3], dim=1), t_emb)
        h = self.up(h)
        h = self.dec2(torch.cat([h, e2], dim=1), t_emb)
        h = self.up(h)
        h = self.dec1(torch.cat([h, e1], dim=1), t_emb)
        h = self.up(h)
        h = self.dec0(torch.cat([h, e0], dim=1), t_emb)

        h = self.out_conv(h)
        return h[:, :, 1:-1, 2:-2]


class UnconstrainedDirectionUNet(nn.Module):
    """Predicts (u, v) directly -- no psi, no curl, no divergence-free constraint."""

    def __init__(self, in_ch: int = 2, base_ch: int = 64, time_dim: int = 256, cond_ch: int = 0):
        super().__init__()
        self.in_ch = in_ch
        self.cond_ch = cond_ch
        self.backbone = UNet(in_ch=in_ch + cond_ch, base_ch=base_ch, time_dim=time_dim, out_ch=2)

    def forward(self, x, t, cond=None):
        if self.cond_ch > 0:
            if cond is None:
                raise ValueError(f"model built with cond_ch={self.cond_ch} but cond is None")
            inp = torch.cat([x, cond], dim=1)
        else:
            inp = x
        return self.backbone(inp, t)   # (B, 2, H, W) -- direct velocity, unconstrained
