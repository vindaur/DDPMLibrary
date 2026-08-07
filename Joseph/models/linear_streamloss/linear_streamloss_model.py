"""Unconditional Repaint UNet -- IDENTICAL architecture to the research repo's
"standard linear model" (Repaint vs DPS/best-model-linear-curldiv-gaussian/
repaint_model.py). Predicts noise (eps), not x0. No conditioning channels --
this model is conditioned only at INFERENCE time (RePaint/DPS/MCG), matching
how best_model_linear.pt itself is used.
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


class Repaint(nn.Module):
    """2D UNet for DDPM noise prediction on 2-channel ocean current fields.
    Input spatial size (94,44), padded to (96,48). Identical to the research
    repo's Repaint class (unconditional linear-model architecture)."""
    _PAD = (2, 2, 1, 1)

    def __init__(self, in_ch: int = 2, base_ch: int = 64, time_dim: int = 256,
                 out_ch: int | None = None):
        super().__init__()
        self.time_dim = time_dim
        out_ch = in_ch if out_ch is None else out_ch
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
