# Linear Best Model — Time Conditioned

Everything needed to train, run, and evaluate the time-conditioned model —
the linear Repaint architecture additionally shown the REAL ocean state 13
hours and 25 hours before the target frame (a genuine per-pixel history
signal, not a global hour-of-day embedding), read from a chronologically
ordered dataset so those priors are never fabricated.

For the plain (non-conditioned) sibling model, see
`../Linear Best Model - Not Time Conditioned/`.

## Status: trained and evaluated

- **Training**: complete. 150 epochs run; best validation loss (0.00021)
  reached at **epoch 120** — epochs 121-150 just oscillated (0.00022-0.00026)
  with no further improvement, so training was not extended further.
- **Evaluation**: complete. MCG/DPS (z=0.04) uncertainty-calibration + RMSE run
  over 100 seeds, exactly 90 shared pixels/seed, n=10 ensemble each — same
  process used throughout this project. Results below.

---

## 1. The model

This is the same `Repaint` UNet used everywhere else in the project (2 input
channels — u/v current — padded from 94×44 to 96×48, four downsample stages
to a 6×3 bottleneck, ResBlocks with GroupNorm/SiLU and sinusoidal-timestep
conditioning via addition, skip connections on the way back up), with one
addition: an optional `cond_ch` argument (`repaint_model.py`). When
`cond_ch > 0`, the conditioning tensor is concatenated onto the noisy field
*before* the first conv (`enc0 = ResBlock(in_ch + cond_ch, c, time_dim)`), so
the network sees `[x_t ; cond]` at every denoising step. `cond_ch=0`
reproduces the original unconditional model exactly — this same file also
backs the non-conditioned sibling.

**What the conditioning signal actually is.** `chrono_dataset.py`'s
`ChronoOceanDataset` reads a `chrono_v1` pickle: one continuous, chronologically
ordered array of hourly ocean-current frames (`fields`, shape `(N, 2, H, W)`),
plus a fixed `land_mask` and train/val/test splits given as *target-frame
indices into that same array*. For a target frame at index `f`, the item
returned is:

```python
{
  "target": fields[f],                                   # (2, H, W) — the frame to denoise
  "cond":   cat([fields[f-13], fields[f-25]], dim=0),     # (4, H, W) — real priors, 13h and 25h earlier
}
```

Because every lookup — target and both priors — comes from the *same* shared
`fields` array, a prior is never fabricated or borrowed from an unrelated
sequence; it's always the actual measured state at that earlier hour. This is
distinct from (and independent of) the sparse robot-path observations used
for MCG/DPS guidance at inference time — see §3. The model conditions on
*history*; MCG/DPS conditions on *sparse present-time observations*. Both
signals are active simultaneously at inference time, but only the temporal
prior is present during training (the network never sees masked path
observations during training — RePaint doesn't require it).

**Training loss** is the same combined objective as the rest of the project:
`eps_MSE + CURL_DIV_WEIGHT * curl_div_loss(x0_hat, x0)` with
`CURL_DIV_WEIGHT = 0.002` (`diffusion.py`), where `x0_hat` is recovered from
the predicted noise via the standard DDPM closed form and `curl_div_loss`
penalizes unphysical curl/divergence in the reconstructed field
(`loss_functions.py`). Diffusion itself is a standard DDPM: `T=1000`,
linear beta schedule (`1e-4` → `0.02`), `noise_std` set to the empirical
std of ocean-cell values in the training split (not a fixed constant).

**Trained checkpoint**: `checkpoints_TimeConditioned_linear/best_model_TimeConditioned_linear.pt`
— epoch 120, val_loss 0.00021, `cond_ch=4`, `lags=[13,25]`. Every checkpoint
this project saves is self-describing (`epoch`, `model`/`optimizer`/`scheduler`
state, `val_loss`, `noise_std`, `curl_div_weight`, `schedule`, `cond_ch`,
`lags`, and the full `argparse` namespace as `args`), so downstream scripts
reconstruct the exact architecture/training recipe from the file alone.

## 2. How to train a model just like it

1. **Get a `chrono_v1` pickle.** This is *not* the same file format as the
   plain `data.pickle` used elsewhere — it needs one continuous hourly
   sequence rather than a shuffled train/val/test split, so that `fields[f-13]`
   and `fields[f-25]` are always meaningful. Build one with
   `utils/build_chrono_dataset.py` (or use the existing
   `Stride Conditional/data_chrono_raw.pickle`, 564MB — referenced by path,
   not duplicated into this folder). It must contain:
   ```python
   {
     "format": "chrono_v1",
     "fields": (N, 2, H, W) float array,          # one continuous hourly sequence
     "land_mask": (H, W) bool,                     # True = land
     "splits": {"train": [...], "val": [...], "test": [...]},  # target-frame indices into `fields`
     "lags": [13, 25],
     ...
   }
   ```
   Every index in every split must be `>= max(lags)` so both priors are always
   in range (`ChronoOceanDataset` raises if not).

2. **Pick your lags.** Default is `(13, 25)` hours — chosen because they're
   real, informative-but-not-trivial priors for this dataset. Any tuple of
   hour-offsets works as long as the pickle has enough history before every
   split index; `cond_ch = 2 * len(lags)` is derived automatically.

3. **Train**:
   ```bash
   python "Linear Best Model - Time Conditioned/train_TimeConditioned.py" \
       --pickle "Stride Conditional/data_chrono_raw.pickle" \
       --schedule linear --epochs 150
   ```
   Key defaults: `--base_ch 64 --time_dim 256 --T 1000 --batch 32 --lr 2e-4`
   (AdamW + cosine LR annealing over `--epochs`). `--schedule` is
   `linear | cosine | geometric` — this project's convention is to try all
   three noise schedules per model and keep whichever validates best; `linear`
   won here. `noise_std` isn't a flag — it's computed automatically from the
   loaded training split's ocean-cell std, so it always matches whatever
   pickle you point at.

4. **Watch validation loss, not epoch count.** Training here isn't run for a
   fixed budget — it's run until val loss plateaus/oscillates with no further
   improvement (this run: best at epoch 120 of 150, with epochs 121-150 just
   oscillating 0.00022-0.00026). `--resume <checkpoint>` continues from any
   saved checkpoint if you want to extend a run.

5. **Checkpointing is disk-frugal by design**: only `last_model_TimeConditioned_{schedule}.pt`
   (overwritten every epoch) and `best_model_TimeConditioned_{schedule}.pt`
   (overwritten only on improvement) are kept — no accumulating per-epoch
   snapshots, since each checkpoint (with optimizer/scheduler state) is
   ~180MB and remote training boxes in this project routinely ran low on disk.

6. **To reproduce the *unconditional* sibling instead**, use `cond_ch=0` /
   `cond=None` — the same `repaint_model.py`/`diffusion.py` in this folder
   already support that path exactly. The actual unconditional training/
   inference *scripts* (`train.py`, `run_mcg_dps_z004.py`) live in the sibling
   `../Linear Best Model - Not Time Conditioned/` folder, not here.

## 3. Inference types (MCG and DPS, z=0.04)

Both inference methods solve the same problem: reconstruct the full 2-channel
current field given only a sparse set of "robot path" observations (u/v known
at a handful of ocean cells, unknown everywhere else), by guiding the reverse
diffusion process toward those known values at every step. The temporal-prior
conditioning from §1 is a completely separate, simultaneous input — both
guidance methods run *on top of* the time-conditioned model exactly as they
would on the unconditional one; nothing about MCG/DPS itself changes.

At every reverse step `t`, both methods:

1. Predict noise `ε̂ = model(x_t, t, cond)` and recover the clean (Tweedie)
   estimate `x̂0 = (x_t - sqrt(1-ᾱ_t)·ε̂) / sqrt(ᾱ_t)`.
2. Compute the residual between `x̂0` and the true observation, restricted to
   the known path cells: `r = mask_known · (x̂0 - x0_obs)`.
3. Backpropagate `‖r‖²` through the model to get a gradient `∇_{x_t}‖r‖²`,
   and take a corrective step `x_{t-1} ← x_{t-1} - (z / ‖r‖) · ∇`, with the
   step size fixed at **z = 0.04** throughout this project.

**DPS (Diffusion Posterior Sampling)** stops there — gradient guidance only.

**MCG (Manifold Constrained Gradient)** adds a hard projection on top: after
the gradient-corrected ancestral step, the known path cells are forcibly
overwritten with `q(x_{t-1} | x0_obs)` — the true observation re-noised to
the `t-1` noise level — while unknown cells keep the gradient-corrected
model prediction (this is RePaint's `repaint_infer.py`, minus the `resample`
step: 1 gradient-guided pass per timestep rather than the full
reverse→merge→forward resampling loop). Everywhere else in this project MCG's
hard merge measurably outperforms DPS's gradient-only correction, and the
same held true for the plain unconditional model; time conditioning narrows
but doesn't reverse that gap (see results below).

**The evaluation methodology** used for the 100-seed results below (same
process as every other model in this project):

- **Path generation**: a directional-persistence random walk over ocean cells
  that continues stepping until exactly **90 unique cells** have been visited
  (not a fixed step count, since revisits would otherwise give variable
  coverage of 64-90 cells depending on the seed) — `biased_walk_path_fixed_coverage`.
- **100 seeds**: `range(0, 700, 7)` — one path + one target frame per seed.
- **n=10 ensemble** per seed: 10 independent reverse-diffusion draws (different
  noise seeds) from the *same* known observations, for both MCG and DPS.
- **Uncertainty calibration**: the ensemble's per-pixel spread (directional,
  magnitude, and full-vector) is compared via Pearson correlation
  (`r_dir`, `r_mag`, `r_vec`) against an "empirical" spread built from the
  nearest real historical frames matching the same observed pixels — i.e.
  does the model's uncertainty *look like* the ocean's actual variability
  given those observations, not just point-accuracy.
- **RMSE** is computed on the ensemble mean vs. ground truth, ocean cells only.

> **Note on missing scripts**: `run_mcg_dps_z004_TimeConditioned.py` (single-shot
> MCG/DPS RMSE, variable path coverage) and `uncertainty_validation_time_conditioned.py`
> (the exact script that produced the fixed-90px results below) are
> referenced by this README and by `train_TimeConditioned.py`'s docstring but
> are **currently missing from this folder** — they existed earlier in this
> project's history but are absent from the latest commit and working tree.
> The computed outputs (`results/uncertainty_validation_time_conditioned_100seeds/`)
> are intact; only the source scripts that produced them need restoring
> (e.g. from an earlier commit, or rewritten following the identical pattern
> used in `Joseph - Unsuccesful Models/LDM/uncertainty_validation_ldm.py`,
> which implements the same fixed-90px/MCG/DPS/z=0.04 methodology for a
> different model and can be adapted by swapping in `Repaint`+`cond` in place
> of the VAE/latent model).

## Files

| File | Role |
|---|---|
| `chrono_dataset.py` | `ChronoOceanDataset` — loads the chrono_v1 pickle; each item is `{"target": (2,H,W), "cond": (4,H,W)}` where `cond` = `[prev_13h(u,v), prev_25h(u,v)]`, looked up from one continuous hourly `fields` array so priors are always real history |
| `repaint_model.py` | `Repaint` UNet, extended with an optional `cond_ch` param (channel-concat before the first conv) |
| `diffusion.py` | `DDPM` class, extended with an optional `cond` passthrough in `training_loss`/`p_sample_step` |
| `loss_functions.py` | `curl_div_loss` — the structural regularizer added to eps-MSE |
| `repaint_infer.py` | `biased_walk_path` / `random_walk_path` path generators, RePaint-style merge/resample reference implementation |
| `train_TimeConditioned.py` | Training entry point (the script that produced the checkpoint below) |
| `run_mcg_dps_z004_TimeConditioned.py` | **Missing — see note in §3.** MCG/DPS (z=0.04) inference, single-shot RMSE, 100-seed sweep (variable path coverage) |
| `uncertainty_validation_time_conditioned.py` | **Missing — see note in §3.** The actual script used for the 100-seed evaluation below — exact-90-pixel shared path, n=10 ensemble, r_dir/r_mag/r_vec calibration + RMSE, saves every per-seed prediction |
| `checkpoints_TimeConditioned_linear/best_model_TimeConditioned_linear.pt` | **Trained checkpoint** (epoch 120, val_loss=0.00021) |
| `results/uncertainty_validation_time_conditioned_100seeds/` | `results.csv` (per-seed metrics), `summary.txt` (aggregates), sample prediction images |

## External data dependency (not copied here)

Both training and evaluation need `Stride Conditional/data_chrono_raw.pickle`
(564MB) — referenced by path, not duplicated into this folder. If that file
ever moves, update `--pickle` in `train_TimeConditioned.py` accordingly.

## Results (100 seeds, 0..693, exactly 90 shared pixels, n=10 ensemble, z=0.04)

| Method | r_dir | r_mag | r_vec | RMSE (mean±std) | RMSE min–max |
|---|---|---|---|---|---|
| MCG-TimeConditioned | +0.569 | +0.673 | +0.747 | 0.0414 ± 0.0155 | 0.0189–0.1019 |
| DPS-TimeConditioned | +0.579 | +0.682 | +0.755 | 0.0410 ± 0.0144 | 0.0173–0.0888 |

For context, the plain (unconditional) linear model on the same 100 seeds:
MCG/DPS-linear RMSE ≈ 0.052–0.053, r_vec ≈ 0.82. **Time conditioning cut RMSE
by roughly 20%** — the best point-accuracy of every method tested in this
project — at the cost of slightly weaker uncertainty calibration (r_vec 0.75
vs 0.82), plausibly because the temporal priors make the model genuinely more
confident, narrowing ensemble spread below what's needed to match the
empirical uncertainty pattern. Full cross-method table:
`Inference Tests and Results/Results/42_uncertainty_validation_fixed90px_100seeds_combined/rmse_combined_table_MASTER.txt`.
