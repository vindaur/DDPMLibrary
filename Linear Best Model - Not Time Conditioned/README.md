# Linear Best Model — NOT time conditioned

This is the plain, unconditional model — the exact original
`best_model_linear.pt` recipe (`Inference Tests and Results/best-model-linear-curldiv-gaussian/checkpoints_linear/best_model_linear.pt`).
It has **no time conditioning of any kind**: no temporal priors, no
observation channels, no geometry channels. The `Repaint` model here takes
only the noisy field `x_t` and the diffusion timestep `t` — nothing else.

For the variant that IS conditioned on the real ocean state 13h/25h earlier,
see the sibling folder `../Linear Best Model - Time Conditioned/`.

## Files

| File | Role |
|---|---|
| `dataset.py` | `OceanCurrentDataset` — loads `data.pickle` (random train/val/test split) |
| `repaint_model.py` | `Repaint` UNet — in_ch=2, base_ch=64, time_dim=256, ~15.0M params, NO cond_ch |
| `diffusion.py` | `DDPM` — beta schedules, `q_sample`, `training_loss`, `p_sample_step`, NO cond param |
| `loss_functions.py` | `curl_div_loss` structural regularizer |
| `repaint_infer.py` | `biased_walk_path` / `random_walk_path` + base RePaint sampler |
| `train.py` | Training entry point |
| `run_mcg_dps_z004.py` | MCG + DPS inference, both at z=0.04 |

## Exact hyperparameters the original checkpoint used

| Field | Value |
|---|---|
| Beta schedule | `linear` |
| T | 1000 |
| noise_std | 0.11618577688932419 |
| curl_div_weight | 0.002 |
| base_ch / time_dim | 64 / 256 |
| Training data | `data.pickle` |

Verified by loading the real checkpoint's `state_dict` into a freshly-built
`Repaint` model from these exact files: zero errors.
