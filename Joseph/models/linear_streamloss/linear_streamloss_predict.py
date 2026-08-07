"""Local inference wrapper for the linear+stream_function_loss checkpoint.

This model is unconditional (no obs/priors channels), so it's conditioned
only at inference time via MCG (manifold constrained gradient, Chung et al.
2022, z=0.04) -- the exact recipe already established and validated in this
research repo's mcg_infer_a (Repaint vs DPS/run_linear_vs_stream_mcg_100seeds.py),
replicated verbatim here against our new checkpoint.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from linear_streamloss_model import Repaint
from linear_streamloss_diffusion import DDPM


class LinearStreamLossPredictor:
    def __init__(self, ckpt_path, device="cpu", stride=10, step_size=0.04):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.noise_std = float(ckpt["noise_std"])
        self.model = Repaint(in_ch=2, base_ch=64, time_dim=256).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.diffusion = DDPM(T=1000, beta_schedule="linear", device=self.device,
                              noise_std=self.noise_std)
        self.stride = stride
        self.step_size = step_size

    def predict(self, x0_known, path_mask, land_mask, seed=0):
        """x0_known: (2,H,W) torch tensor, RAW (unstandardized) m/s, zero off-path.
        path_mask, land_mask: (H,W) numpy bool. Returns (2,H,W) numpy m/s."""
        device = self.device
        diffusion = self.diffusion
        H, W = x0_known.shape[1:]
        x0_known_t = x0_known.unsqueeze(0).to(device)
        known_t = torch.from_numpy(path_mask.astype(np.float32)).to(device)[None, None]
        land_t = torch.from_numpy(land_mask.astype(np.float32)).to(device)[None, None]
        ocean_t = 1.0 - land_t

        torch.manual_seed(seed)
        xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
        timesteps = list(range(0, diffusion.T, self.stride))

        for i in reversed(range(len(timesteps))):
            t_int = timesteps[i]
            t_prev_int = timesteps[i - 1] if i > 0 else 0

            xt_in = xt.detach().requires_grad_(True)
            t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

            pred_noise = self.model(xt_in, t_vec)
            ab = diffusion.alpha_bar[t_int]
            x0_hat = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
            x0_hat = x0_hat.clamp(-1.5, 1.5)

            residual = known_t * (x0_hat - x0_known_t)
            norm_sq = (residual ** 2).sum()
            grad = torch.autograd.grad(norm_sq, xt_in)[0]

            with torch.no_grad():
                xt_unknown = diffusion.p_sample_step(self.model, xt_in.detach(), t_int, t_prev_int)
                norm = norm_sq.sqrt().item() + 1e-8
                xt_unknown = xt_unknown - (self.step_size / norm) * grad.detach()

                t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
                xt_known_noisy, _ = diffusion.q_sample(x0_known_t, t_prev_t)
                xt = known_t * xt_known_noisy + (1.0 - known_t) * xt_unknown
                xt = xt * ocean_t

        return xt.squeeze(0).detach().cpu().numpy()
