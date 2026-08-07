"""Local inference wrapper for the trained UnconstrainedDirectionUNet checkpoint.

Since this model predicts (u,v) directly (no stream-function/curl, no
heteroscedastic magnitude fusion needed -- it's a single unified velocity
prediction), inference is a subsampled ancestral reverse-diffusion chain
starting from pure noise, conditioned on [sparse obs | real 13h/25h priors |
static geometry] -- the same conditioning richness as the real StreamDDPM.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from unconstrained_stream_unet import UnconstrainedDirectionUNet
from unconstrained_stream_diffusion import CosineSchedule

T_STEPS = 1000
LAGS = (13, 25)


def geometry_channels(land_mask):
    land = np.asarray(land_mask).astype(bool)
    H, W = land.shape
    ocean = ~land
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)
    dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dmax = float(dist.max())
    if dmax > 0:
        dist = dist / dmax
    dist[land] = 0.0
    return torch.from_numpy(np.stack([xs, ys, dist], axis=0))


def observation_channels(field, path_mask, legacy=True):
    pm = torch.from_numpy(np.asarray(path_mask, dtype=bool))
    obs = torch.zeros_like(field)
    obs[:, pm] = field[:, pm]
    mask = pm.float()[None]
    return torch.cat([obs, mask], dim=0)


class UnconstrainedStreamPredictor:
    def __init__(self, ckpt_path, land_mask, device="cpu", n_infer_steps=20):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cond_ch = int(ckpt.get("cond_ch", 10))
        self.data_std = float(ckpt["data_std"])
        self.net = UnconstrainedDirectionUNet(in_ch=2, base_ch=64, time_dim=256,
                                              cond_ch=self.cond_ch).to(self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.schedule = CosineSchedule(T=T_STEPS, device=self.device)
        self.land_mask = np.asarray(land_mask).astype(bool)   # (94,44)
        self.geom = geometry_channels(self.land_mask).to(self.device)
        self.n_infer_steps = n_infer_steps
        self.H, self.W = self.land_mask.shape

    def _build_inference_schedule(self, n_steps):
        step_size = T_STEPS // n_steps
        ts = list(reversed(range(step_size - 1, T_STEPS, step_size)))
        pairs = [(ts[i], ts[i + 1] if i + 1 < len(ts) else -1) for i in range(len(ts))]
        return pairs

    def _subsampled_step(self, x_t, x0_pred, t_int, t_prev_int):
        """Ancestral posterior for a (possibly non-consecutive) (t, t_prev) pair --
        same math as ddpm_library.stream.diffusion.DDPM.p_sample_step, just in
        x0-form directly (this network already outputs x0, not eps)."""
        ab = self.schedule.alpha_bars[t_int]
        if t_prev_int < 0:
            return x0_pred
        ab_prev = self.schedule.alpha_bars[t_prev_int]
        beta_eff = 1.0 - ab / ab_prev
        var = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
        coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
        coef2 = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean = coef1 * x0_pred + coef2 * x_t
        return mean + var.sqrt() * torch.randn_like(x_t)

    def predict(self, target_field_std, path_mask, priors_std, seed=None):
        """target_field_std: (2,94,44) standardized (only used to build the sparse obs
        channel -- observed cells only). priors_std: (4,94,44) standardized, RAW
        (unprojected). Returns mean (2,94,44) m/s, uncertainty (zeros)."""
        if seed is not None:
            torch.manual_seed(seed)

        obs = observation_channels(torch.from_numpy(target_field_std.astype(np.float32)),
                                    path_mask, legacy=True)                  # (3,94,44)
        priors_t = torch.from_numpy(priors_std.astype(np.float32))          # (4,94,44)
        cond = torch.cat([obs, priors_t, self.geom.cpu()], dim=0).unsqueeze(0).to(self.device)

        x = torch.randn(1, 2, self.H, self.W, device=self.device)
        pairs = self._build_inference_schedule(self.n_infer_steps)
        with torch.no_grad():
            for t_int, t_prev_int in pairs:
                t_tensor = torch.full((1,), t_int, device=self.device, dtype=torch.long)
                x0_pred = self.net(x, t_tensor, cond)
                x = self._subsampled_step(x, x0_pred, t_int, t_prev_int)
                if t_prev_int < 0:
                    break
        mean_std = x.squeeze(0).cpu().numpy()          # (2,94,44) standardized
        mean = mean_std * self.data_std                 # inverse-standardize (mean=0)
        return mean, np.zeros_like(mean)
