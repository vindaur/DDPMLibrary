"""Train the unconstrained direction network -- "StreamDDPM but not
divergence-free". Same rich conditioning (sparse obs + real 13h/25h temporal
priors + static geometry, cond_ch=10) and cosine/x0-parameterized diffusion as
the original stream pipeline, but:
  - UnconstrainedDirectionUNet predicts (u,v) directly, no curl/stream-function
  - plain gaussian noise, not the spectrally-shaped div-free noise
  - no Helmholtz reprojection anywhere
  - trained on the RAW (unprocessed, non-divergence-free) field and RAW
    (unprojected) priors -- no divergence-free assumption anywhere in the
    data pipeline either.

Reuses divergent_chrono_compact.pickle (already on this box from the earlier
DivergentUNet run) but transposes it back from library orientation (44,94) to
the stream-native (94,44) orientation this architecture expects.

Early stopping on validation loss plateau, same convention as train_divergent.py.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from unconstrained_stream_unet import UnconstrainedDirectionUNet
from unconstrained_stream_diffusion import CosineSchedule

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "divergent_chrono_compact.pickle"
CKPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "checkpoints_unconstrained"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LAGS = (13, 25)
MAX_LAG = max(LAGS)
T_STEPS = 1000
BATCH_SIZE = 8
LR = 2e-4
VAL_EVERY = 500
VAL_BATCHES = 20
PATIENCE = 15
MIN_REL_IMPROVEMENT = 0.002
MAX_STEPS = 200_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)


# ─── Inlined conditioning helpers (normally ddpm_library.stream.conditioning) ──

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
    geom = np.stack([xs, ys, dist], axis=0)
    return torch.from_numpy(geom)


def observation_channels(field, path_mask, land_np, legacy=True):
    pm = torch.from_numpy(np.asarray(path_mask, dtype=bool))
    obs = torch.zeros_like(field)
    obs[:, pm] = field[:, pm]
    mask = pm.float()[None]
    if legacy:
        return torch.cat([obs, mask], dim=0)
    pm_np = pm.numpy()
    ocean_np = ~land_np
    dist = ndimage.distance_transform_edt(~pm_np).astype(np.float32)
    dist[land_np] = 0.0
    dmax = float(dist[ocean_np].max()) if ocean_np.any() and dist[ocean_np].max() > 0 else 1.0
    dist = dist / dmax
    dist[land_np] = 0.0
    dist_t = torch.from_numpy(dist)[None]
    return torch.cat([obs, mask, dist_t], dim=0)


def build_conditioning(obs_field_std, path_mask, priors_std, land_np, geom, legacy_obs=True):
    obs = observation_channels(torch.from_numpy(obs_field_std.astype(np.float32)),
                                path_mask, land_np, legacy=legacy_obs)
    priors_t = torch.from_numpy(priors_std.astype(np.float32))
    return torch.cat([obs, priors_t, geom], dim=0)


def biased_walk_path(land_mask, n_steps=150, seed=None, straight_bias=0.75):
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
            w = straight_bias if dot == 1 else (side if dot == 0 else side * 0.001)
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


def sample_path_mask(land_mask, ocean_mask, rng):
    if rng.random() < 0.5:
        pct = rng.uniform(0.3, 5.0)
        ocean_idx = np.argwhere(ocean_mask)
        n_obs = max(1, int(len(ocean_idx) * pct / 100.0))
        sel = rng.choice(len(ocean_idx), size=n_obs, replace=False)
        mask = np.zeros_like(ocean_mask)
        for i, j in ocean_idx[sel]:
            mask[i, j] = True
        return mask
    n_steps = int(rng.uniform(40, 220))
    seed = int(rng.integers(0, 2**31 - 1))
    path = biased_walk_path(land_mask, n_steps=n_steps, seed=seed, straight_bias=0.75)
    return path & ocean_mask


# ─── Data ──────────────────────────────────────────────────────────────────

print(f"Loading {DATA_PATH} ...", flush=True)
with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)

fields_lib = d["fields"].astype(np.float32)             # (N, 2, 44, 94) library orientation
land_mask_lib = np.asarray(d["land_mask"]).astype(bool)  # (44, 94)
data_std = float(d["data_std"])
splits = d["splits"]

# transpose back to stream-native (94, 44) orientation
fields = np.transpose(fields_lib, (0, 1, 3, 2))          # (N, 2, 94, 44)
land_mask = land_mask_lib.T                              # (94, 44)
ocean_mask = ~land_mask

fields_std = fields / max(data_std, 1e-8)
fields_std[:, :, land_mask] = 0.0

train_idx = splits["train"]
val_idx = splits["val"]
train_idx = train_idx[train_idx >= MAX_LAG]
val_idx = val_idx[val_idx >= MAX_LAG]
print(f"train={len(train_idx)} val={len(val_idx)} test={len(splits['test'])}", flush=True)

geom = geometry_channels(land_mask)   # (3, 94, 44) torch tensor, precomputed once
H, W = land_mask.shape
COND_CH = 3 + 2 * len(LAGS) + 3        # 3 obs + 4 priors + 3 geom = 10


def make_sample(fi, rng):
    priors = np.concatenate([fields_std[fi - L] for L in LAGS], axis=0)   # (4, 94, 44), RAW (unprojected)
    path_mask = sample_path_mask(land_mask, ocean_mask, rng)
    cond = build_conditioning(fields_std[fi], path_mask, priors, land_mask, geom, legacy_obs=True)
    x0 = torch.from_numpy(fields_std[fi].astype(np.float32))
    return x0, cond, int(path_mask.sum())


def ocean_mse(pred, target, ocean_t):
    diff = (pred - target) * ocean_t[None, None]
    return (diff ** 2).sum() / (ocean_t.sum() * pred.shape[1] * pred.shape[0])


net = UnconstrainedDirectionUNet(in_ch=2, base_ch=64, time_dim=256, cond_ch=COND_CH).to(device)
schedule = CosineSchedule(T=T_STEPS, device=device)
opt = torch.optim.AdamW(net.parameters(), lr=LR)
ocean_t = torch.from_numpy(ocean_mask.astype(np.float32)).to(device)

print(f"Model params: {sum(p.numel() for p in net.parameters()):,}  cond_ch={COND_CH}", flush=True)

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
    x0_batch = torch.zeros(BATCH_SIZE, 2, H, W)
    cond_batch = torch.zeros(BATCH_SIZE, COND_CH, H, W)
    for b in range(BATCH_SIZE):
        fi = int(train_rng.choice(train_idx))
        x0_b, cond_b, _ = make_sample(fi, train_rng)
        x0_batch[b] = x0_b
        cond_batch[b] = cond_b
    x0_batch = x0_batch.to(device)
    cond_batch = cond_batch.to(device)

    t = torch.randint(0, T_STEPS, (BATCH_SIZE,), device=device)
    x_t, _ = schedule.q_sample(x0_batch, t)
    x0_pred = net(x_t, t, cond_batch)
    loss = ocean_mse(x0_pred, x0_batch, ocean_t)

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
                x0_batch = torch.zeros(BATCH_SIZE, 2, H, W)
                cond_batch = torch.zeros(BATCH_SIZE, COND_CH, H, W)
                for b in range(BATCH_SIZE):
                    fi = int(vrng.choice(val_idx))
                    x0_b, cond_b, _ = make_sample(fi, vrng)
                    x0_batch[b] = x0_b
                    cond_batch[b] = cond_b
                x0_batch = x0_batch.to(device)
                cond_batch = cond_batch.to(device)
                t = torch.randint(0, T_STEPS, (BATCH_SIZE,), device=device)
                x_t, _ = schedule.q_sample(x0_batch, t)
                x0_pred = net(x_t, t, cond_batch)
                val_losses.append(ocean_mse(x0_pred, x0_batch, ocean_t).item())
        val_loss = float(np.mean(val_losses))
        rel_improve = (best_val - val_loss) / best_val if best_val < float("inf") else 1.0
        print(f"  [val] step {step:7d}  val_loss={val_loss:.6f}  "
              f"best={best_val:.6f}  rel_improve={rel_improve:+.4f}  "
              f"no_improve_count={no_improve_count}", flush=True)

        torch.save({"model": net.state_dict(), "step": step, "val_loss": val_loss,
                    "data_std": data_std, "cond_ch": COND_CH, "lags": LAGS},
                   CKPT_DIR / "last.pt")

        if rel_improve > MIN_REL_IMPROVEMENT:
            best_val = val_loss
            no_improve_count = 0
            torch.save({"model": net.state_dict(), "step": step, "val_loss": val_loss,
                        "data_std": data_std, "cond_ch": COND_CH, "lags": LAGS},
                       CKPT_DIR / "best.pt")
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
