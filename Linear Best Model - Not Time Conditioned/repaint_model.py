import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timestep (B,) to sinusoidal embedding (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]  # (B, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


def _num_groups(channels: int) -> int:
    """Pick the largest divisor of channels that is <= 32."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with time-step conditioning via addition."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1   = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.norm2   = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act     = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_fc(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Repaint
# ---------------------------------------------------------------------------

class Repaint(nn.Module):
    """
    2D UNet for DDPM noise prediction on 2-channel ocean current fields.
    Used as the backbone for noise-schedule ablation experiments.

    Input spatial size: (H, W) = (94, 44).
    Internally padded to (96, 48) for clean factor-of-2 downsampling:
        96 → 48 → 24 → 12 → 6 → 3  (bottleneck)
        48 → 24 → 12 → 6            (width)

    Padding applied: left=2, right=2, top=1, bottom=1 → 94→96 height, 44→48 width.
    """

    # (left, right, top, bottom) — F.pad convention is last-dim first
    _PAD  = (2, 2, 1, 1)   # → 44+4=48 wide, 94+2=96 tall
    _UPAD = (2, 2, 1, 1)   # same amounts to strip when unpadding

    def __init__(self, in_ch: int = 2, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        c = base_ch

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.enc0 = ResBlock(in_ch,  c,    time_dim)   # 96×48
        self.enc1 = ResBlock(c,      c*2,  time_dim)   # 48×24
        self.enc2 = ResBlock(c*2,    c*4,  time_dim)   # 24×12
        self.enc3 = ResBlock(c*4,    c*8,  time_dim)   # 12×6

        # Bottleneck
        self.mid  = ResBlock(c*8,    c*8,  time_dim)   # 6×3

        # Decoder  (skip connections double the input channels)
        self.dec3 = ResBlock(c*8+c*8, c*4, time_dim)   # 12×6
        self.dec2 = ResBlock(c*4+c*4, c*2, time_dim)   # 24×12
        self.dec1 = ResBlock(c*2+c*2, c,   time_dim)   # 48×24
        self.dec0 = ResBlock(c  +c,   c,   time_dim)   # 96×48

        self.out_conv = nn.Conv2d(c, in_ch, 1)

        self.down = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 94, 44) noisy field
            t: (B,) integer timesteps
        Returns:
            predicted noise: (B, 2, 94, 44)
        """
        # Pad to 96×48
        x = F.pad(x, self._PAD)

        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e0 = self.enc0(x,              t_emb)   # 96×48
        e1 = self.enc1(self.down(e0),  t_emb)   # 48×24
        e2 = self.enc2(self.down(e1),  t_emb)   # 24×12
        e3 = self.enc3(self.down(e2),  t_emb)   # 12×6

        # Bottleneck
        h = self.mid(self.down(e3), t_emb)      # 6×3

        # Decoder
        h = self.up(h)                                           # 12×6
        h = self.dec3(torch.cat([h, e3], dim=1), t_emb)

        h = self.up(h)                                           # 24×12
        h = self.dec2(torch.cat([h, e2], dim=1), t_emb)

        h = self.up(h)                                           # 48×24
        h = self.dec1(torch.cat([h, e1], dim=1), t_emb)

        h = self.up(h)                                           # 96×48
        h = self.dec0(torch.cat([h, e0], dim=1), t_emb)

        h = self.out_conv(h)

        # Unpad back to 94×44
        # _PAD = (left=2, right=2, top=1, bottom=1)
        # so strip: height [1:-1], width [2:-2]
        h = h[:, :, 1:-1, 2:-2]
        return h
