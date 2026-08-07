"""Train DivergentUNet on the RAW, unprocessed chrono field.

Training target is the raw velocity field exactly as measured (non-divergence-
free -- data_chrono_raw.pickle's own `projected: False` field, just repacked/
transposed). No Helmholtz/Leray projection is applied to it, and no
divergence-free machinery exists anywhere in this pipeline:
  - single-head UNet (phi/potential branch only -- see divergent_unet.py)
  - single standard DDPM schedule, isotropic noise (see divergent_schedule.py)

Early stopping on validation loss plateau (primary stop condition per user
request), with a generous max-step safety cap so a bug can't cause an
unbounded-cost run.

Expects divergent_chrono_compact.pickle already transposed to library
orientation: fields (N, 2, 44, 94), land_mask (44, 94) True=land -- so
ddpm_library.config's OCEAN_H/W=44/94, FULL_H/W=64/128 pad/crop utilities
apply unchanged.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from divergent_unet import DivergentUNet
from divergent_schedule import StandardSchedule

# Constants inlined from ddpm_library.config (avoids needing the package --
# and its ~300MB of bundled weight assets -- installed on the remote box).
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
N_STEPS = 250
MIN_BETA = 1e-4
MAX_BETA = 0.02


def biased_walk_path(land_mask, n_steps=150, seed=None, straight_bias=0.75):
    """Inlined from ddpm_library.stream.paths (verbatim, pure numpy/self-contained).
    Robot path with directional persistence -- meanders, navigates around land."""
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape
    ocean_cells = list(zip(*np.where(~land_mask)))
    if not ocean_cells:
        raise ValueError("No ocean cells found in land_mask.")
    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c = int(start[0]), int(start[1])
    path_mask = np.zeros((H, W), dtype=bool)
    path_mask[r, c] = True
    all_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cur_dir = all_dirs[rng.integers(4)]
    visit_count = np.zeros((H, W), dtype=np.float32)
    visit_count[r, c] = 1.0
    for _ in range(n_steps - 1):
        valid = [(dr, dc) for dr, dc in all_dirs
                 if 0 <= r + dr < H and 0 <= c + dc < W and not land_mask[r + dr, c + dc]]
        if not valid:
            break
        side = (1.0 - straight_bias) / 2.0
        weights = []
        for dr, dc in valid:
            dot = dr * cur_dir[0] + dc * cur_dir[1]
            if dot == 1:
                w = straight_bias
            elif dot == 0:
                w = side
            else:
                w = side * 0.001
            nr, nc = r + dr, c + dc
            novelty = 1.0 / (1.0 + visit_count[nr, nc])
            weights.append(w * novelty)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        idx = rng.choice(len(valid), p=weights)
        dr, dc = valid[idx]
        r, c = r + dr, c + dc
        cur_dir = (dr, dc)
        visit_count[r, c] += 1.0
        path_mask[r, c] = True
    return path_mask

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "divergent_chrono_compact.pickle"
CKPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "checkpoints_divergent"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
LR = 2e-4
VAL_EVERY = 500          # steps between validation checks
VAL_BATCHES = 20         # batches per validation check (fixed seeds -> reproducible)
PATIENCE = 15            # consecutive non-improving val checks before stopping
MIN_REL_IMPROVEMENT = 0.002   # <0.2% relative improvement counts as "no improvement"
MAX_STEPS = 200_000      # safety cap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

print(f"Loading {DATA_PATH} ...", flush=True)
with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)

fields = d["fields"].astype(np.float32)          # (N, 2, 44, 94)
land_mask = np.asarray(d["land_mask"]).astype(bool)  # (44, 94) True=land
ocean_mask_np = ~land_mask
data_std = float(d["data_std"])
splits = d["splits"]
assert fields.shape[-2:] == (OCEAN_H, OCEAN_W), f"expected (44,94), got {fields.shape[-2:]}"

fields_std = fields / max(data_std, 1e-8)
fields_std[:, :, land_mask] = 0.0

train_idx = splits["train"]
val_idx = splits["val"]
print(f"train={len(train_idx)} val={len(val_idx)} test={len(splits['test'])}", flush=True)

ocean_mask_t = torch.from_numpy(ocean_mask_np.astype(np.float32)).to(device)  # (44,94)
land_mask_np = land_mask  # for biased_walk_path (True=land)


def sample_obs_mask(rng):
    """Random sparse-observation mask over the ocean, (44,94) bool, True=observed."""
    if rng.random() < 0.5:
        pct = rng.uniform(0.3, 5.0)
        ocean_idx = np.argwhere(ocean_mask_np)
        n_obs = max(1, int(len(ocean_idx) * pct / 100.0))
        sel = rng.choice(len(ocean_idx), size=n_obs, replace=False)
        mask = np.zeros_like(ocean_mask_np)
        for i, j in ocean_idx[sel]:
            mask[i, j] = True
        return mask
    else:
        n_steps = int(rng.uniform(40, 220))
        seed = int(rng.integers(0, 2**31 - 1))
        path = biased_walk_path(land_mask_np, n_steps=n_steps, seed=seed, straight_bias=0.75)
        return path & ocean_mask_np


def make_batch(frame_indices, rng):
    B = len(frame_indices)
    x0 = np.zeros((B, 2, OCEAN_H, OCEAN_W), dtype=np.float32)
    known_std = np.zeros((B, 2, OCEAN_H, OCEAN_W), dtype=np.float32)
    miss_mask = np.ones((B, 1, OCEAN_H, OCEAN_W), dtype=np.float32)
    for b, fi in enumerate(frame_indices):
        field = fields_std[fi]                     # (2,44,94)
        x0[b] = field
        obs_mask = sample_obs_mask(rng)              # (44,94) True=observed
        known_std[b] = field * obs_mask[None]
        miss_mask[b, 0] = (~obs_mask).astype(np.float32)

    return x0, known_std, miss_mask


def pad_batch_to_full(x0, known_std, miss_mask):
    B = x0.shape[0]
    x0_full = np.zeros((B, 2, FULL_H, FULL_W), dtype=np.float32)
    known_full = np.zeros((B, 2, FULL_H, FULL_W), dtype=np.float32)
    miss_full = np.ones((B, 1, FULL_H, FULL_W), dtype=np.float32)
    x0_full[:, :, :OCEAN_H, :OCEAN_W] = x0
    known_full[:, :, :OCEAN_H, :OCEAN_W] = known_std
    miss_full[:, :, :OCEAN_H, :OCEAN_W] = miss_mask
    return x0_full, known_full, miss_full


def build_model_input(x_t, miss_mask, known_mask, known_std):
    noise_replace = torch.randn_like(x_t)
    x_t_in = x_t * miss_mask + noise_replace * known_mask
    miss_ch = miss_mask[:, :1]
    cond_field = known_std * known_mask
    return torch.cat([x_t_in, miss_ch, cond_field], dim=1)


def ocean_mse(pred, target):
    """MSE over ocean cells only, on the padded (64,128) grid."""
    mask = torch.zeros(1, 1, FULL_H, FULL_W, device=pred.device)
    mask[:, :, :OCEAN_H, :OCEAN_W] = ocean_mask_t[None, None]
    diff = (pred - target) * mask
    return (diff ** 2).sum() / (mask.sum() * pred.shape[1] * pred.shape[0])


net = DivergentUNet(n_steps=N_STEPS, time_emb_dim=256, in_channels=5).to(device)
schedule = StandardSchedule(n_steps=N_STEPS, min_beta=MIN_BETA, max_beta=MAX_BETA, device=device)
opt = torch.optim.AdamW(net.parameters(), lr=LR)

print(f"Model params: {sum(p.numel() for p in net.parameters()):,}", flush=True)

train_rng = np.random.default_rng(0)
val_rng_seed = 12345

best_val = float("inf")
no_improve_count = 0
step = 0
t_start = time.time()

print("Starting training ...", flush=True)
while step < MAX_STEPS:
    step += 1
    net.train()
    frame_indices = train_rng.choice(train_idx, size=BATCH_SIZE, replace=True)
    x0, known_std, miss_mask = make_batch(frame_indices, train_rng)
    x0_full, known_full, miss_full = pad_batch_to_full(x0, known_std, miss_mask)

    x0_t = torch.from_numpy(x0_full).to(device)
    known_t = torch.from_numpy(known_full).to(device)
    miss_t = torch.from_numpy(miss_full).to(device)
    known_mask_t = 1.0 - miss_t

    t = torch.randint(0, N_STEPS, (BATCH_SIZE,), device=device)
    x_t, _ = schedule.q_sample(x0_t, t)

    model_in = build_model_input(x_t, miss_t, known_mask_t, known_t)
    t_in = t.reshape(BATCH_SIZE, 1)
    x0_pred = net(model_in, t_in)

    loss = ocean_mse(x0_pred, x0_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0:
        elapsed = time.time() - t_start
        print(f"step {step:7d}  train_loss={loss.item():.6f}  "
              f"elapsed={elapsed:.0f}s  ({elapsed/step:.3f}s/step)", flush=True)

    if step % VAL_EVERY == 0:
        net.eval()
        vrng = np.random.default_rng(val_rng_seed)
        val_losses = []
        with torch.no_grad():
            for _ in range(VAL_BATCHES):
                frame_indices = vrng.choice(val_idx, size=BATCH_SIZE, replace=True)
                x0, known_std, miss_mask = make_batch(frame_indices, vrng)
                x0_full, known_full, miss_full = pad_batch_to_full(x0, known_std, miss_mask)
                x0_t = torch.from_numpy(x0_full).to(device)
                known_t = torch.from_numpy(known_full).to(device)
                miss_t = torch.from_numpy(miss_full).to(device)
                known_mask_t = 1.0 - miss_t
                t = torch.randint(0, N_STEPS, (BATCH_SIZE,), device=device)
                x_t, _ = schedule.q_sample(x0_t, t)
                model_in = build_model_input(x_t, miss_t, known_mask_t, known_t)
                x0_pred = net(model_in, t.reshape(BATCH_SIZE, 1))
                val_losses.append(ocean_mse(x0_pred, x0_t).item())
        val_loss = float(np.mean(val_losses))
        rel_improve = (best_val - val_loss) / best_val if best_val < float("inf") else 1.0
        print(f"  [val] step {step:7d}  val_loss={val_loss:.6f}  "
              f"best={best_val:.6f}  rel_improve={rel_improve:+.4f}  "
              f"no_improve_count={no_improve_count}", flush=True)

        torch.save({"model": net.state_dict(), "step": step, "val_loss": val_loss,
                    "data_std": data_std}, CKPT_DIR / "last.pt")

        if rel_improve > MIN_REL_IMPROVEMENT:
            best_val = val_loss
            no_improve_count = 0
            torch.save({"model": net.state_dict(), "step": step, "val_loss": val_loss,
                        "data_std": data_std}, CKPT_DIR / "best.pt")
            print(f"  [val] new best, saved checkpoint", flush=True)
        else:
            no_improve_count += 1
            if no_improve_count >= PATIENCE:
                print(f"EARLY STOP at step {step}: val loss plateaued "
                      f"({PATIENCE} checks without >{MIN_REL_IMPROVEMENT*100:.2f}% improvement)",
                      flush=True)
                break

print(f"Done. best_val={best_val:.6f}  total_steps={step}  "
      f"elapsed={(time.time()-t_start)/3600:.2f}h", flush=True)
