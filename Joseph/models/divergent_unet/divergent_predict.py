"""Local inference wrapper for the trained DivergentUNet checkpoint.

Same single-step inference convention as ddpm_library.inference.run_single_step
(voronoi warm-start, mask_xt=True splice), just using the single-head model +
standard schedule instead of the split-head model + HelmholtzSplitSchedule.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from divergent_unet import DivergentUNet
from divergent_schedule import StandardSchedule

from ddpm_library.config import FULL_H, FULL_W, OCEAN_H, OCEAN_W
from ddpm_library.inference import _pad_ocean_to_full, _crop_full_to_ocean, _voronoi_fill_2ch
from ddpm_library.rasterize import observations_to_channels

N_STEPS = 250
MIN_BETA = 1e-4
MAX_BETA = 0.02
DEFAULT_T = 50


class DivergentPredictor:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.net = DivergentUNet(n_steps=N_STEPS, time_emb_dim=256, in_channels=5).to(self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.data_std = float(ckpt["data_std"])
        self.schedule = StandardSchedule(n_steps=N_STEPS, min_beta=MIN_BETA, max_beta=MAX_BETA,
                                          device=self.device)

    def _build_model_input(self, x_t, miss_mask, known_mask, known_std):
        noise_replace = torch.randn_like(x_t)
        x_t_in = x_t * miss_mask + noise_replace * known_mask
        miss_ch = miss_mask[:, :1]
        cond_field = known_std * known_mask
        return torch.cat([x_t_in, miss_ch, cond_field], dim=1)

    def predict(self, observations, *, t=DEFAULT_T, voronoi=True, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        obs_list = list(observations)
        sparse_u, sparse_v, missing_mask = observations_to_channels(obs_list)

        uv_raw = np.stack([sparse_u, sparse_v], axis=0)          # (2,44,94) m/s
        uv_std = (uv_raw / max(self.data_std, 1e-8)).astype(np.float32)
        known_mask_44x94 = (1.0 - missing_mask).astype(np.float32)
        uv_std = uv_std * known_mask_44x94[None]

        uv_full = _pad_ocean_to_full(uv_std)                     # (2,64,128)
        miss_full = np.ones((1, FULL_H, FULL_W), dtype=np.float32)
        miss_full[0, :OCEAN_H, :OCEAN_W] = missing_mask

        known_std = torch.from_numpy(uv_full).unsqueeze(0).to(self.device)
        miss_mask = torch.from_numpy(miss_full).unsqueeze(0).to(self.device)
        known_mask = 1.0 - miss_mask

        with torch.no_grad():
            if voronoi:
                vor_std = _voronoi_fill_2ch(known_std, known_mask)
                base = known_std * known_mask + vor_std * miss_mask
            else:
                base = known_std
            t_b = torch.tensor([t], device=self.device)
            x_t, _ = self.schedule.q_sample(base, t_b)
            t_tensor = torch.full((1, 1), t, device=self.device, dtype=torch.long)
            x0_pred = self.net(self._build_model_input(x_t, miss_mask, known_mask, known_std), t_tensor)
            x_final = known_std * known_mask + x0_pred * miss_mask

        ocean = _crop_full_to_ocean(x_final).squeeze(0).cpu().numpy()   # (2,44,94) standardized
        ocean = ocean * self.data_std                                   # inverse-standardize (mean=0)
        mean = np.transpose(ocean, (1, 2, 0)).astype(np.float32)        # (44,94,2)
        uncertainty = np.zeros_like(mean)
        return mean, uncertainty
