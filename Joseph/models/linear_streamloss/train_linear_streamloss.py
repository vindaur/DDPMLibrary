"""Train the unconditional "linear model" architecture (Repaint UNet, linear
beta schedule, eps-prediction) but with the STREAM_FUNCTION auxiliary loss
in place of the standard curl_div loss:

    total = eps_mse(pred_noise, noise) + 0.002 * stream_function_loss(x0_hat, x0)

No masking/conditioning at training time at all -- this model is conditioned
only at INFERENCE time (RePaint/DPS/MCG), exactly matching how the real
best_model_linear.pt is used (see Repaint vs DPS/README.md).

Trained on the RAW (unprocessed, non-divergence-free) chrono field, same
philosophy as every other model built this session -- no divergence-free
projection anywhere in the data pipeline.

Early stopping on validation loss plateau.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from linear_streamloss_model import Repaint
from linear_streamloss_diffusion import DDPM

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "divergent_chrono_compact.pickle"
CKPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "checkpoints_linear_streamloss"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

T_STEPS = 1000
BATCH_SIZE = 32
LR = 2e-4
VAL_EVERY = 200
VAL_BATCHES = 20
PATIENCE = 15
MIN_REL_IMPROVEMENT = 0.002
MAX_STEPS = 200_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

print(f"Loading {DATA_PATH} ...", flush=True)
with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)

fields_lib = d["fields"].astype(np.float32)              # (N, 2, 44, 94) library orientation
land_mask_lib = np.asarray(d["land_mask"]).astype(bool)   # (44, 94)
splits = d["splits"]

# transpose back to (94, 44) -- native orientation for this architecture
fields = np.transpose(fields_lib, (0, 1, 3, 2))           # (N, 2, 94, 44), RAW (unprojected) m/s
land_mask_np = land_mask_lib.T                            # (94, 44)
ocean_mask_np = ~land_mask_np

train_idx = splits["train"]
val_idx = splits["val"]
print(f"train={len(train_idx)} val={len(val_idx)} test={len(splits['test'])}", flush=True)

ocean_train = fields[train_idx][:, :, ocean_mask_np]
noise_std = float(ocean_train.std())
print(f"noise_std: {noise_std:.5f}  (ocean pixel std, raw field, train split)", flush=True)

land_mask_t = torch.from_numpy(land_mask_np).to(device)
H, W = land_mask_np.shape

model = Repaint(in_ch=2, base_ch=64, time_dim=256).to(device)
diffusion = DDPM(T=T_STEPS, beta_schedule="linear", device=device, noise_std=noise_std)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

train_rng = np.random.default_rng(0)
val_rng_seed = 12345
best_val = float("inf")
no_improve_count = 0
step = 0
t_start = time.time()

print("Starting training ...", flush=True)
while step < MAX_STEPS:
    step += 1
    model.train()
    idx = train_rng.choice(train_idx, size=BATCH_SIZE, replace=True)
    x0 = torch.from_numpy(fields[idx]).to(device)   # (B,2,94,44) raw, unconditional

    loss, eps_loss, sf_loss = diffusion.training_loss(model, x0, land_mask_t)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 50 == 0:
        elapsed = time.time() - t_start
        print(f"step {step:7d}  loss={loss.item():.6f}  eps={eps_loss.item():.6f}  "
              f"stream_fn={sf_loss.item():.6f}  elapsed={elapsed:.0f}s  "
              f"({elapsed/step:.3f}s/step)", flush=True)

    if step % VAL_EVERY == 0:
        model.eval()
        vrng = np.random.default_rng(val_rng_seed)
        val_losses = []
        with torch.no_grad():
            for _ in range(VAL_BATCHES):
                idx = vrng.choice(val_idx, size=BATCH_SIZE, replace=True)
                x0 = torch.from_numpy(fields[idx]).to(device)
                vloss, _, _ = diffusion.training_loss(model, x0, land_mask_t)
                val_losses.append(vloss.item())
        val_loss = float(np.mean(val_losses))
        rel_improve = (best_val - val_loss) / best_val if best_val < float("inf") else 1.0
        print(f"  [val] step {step:7d}  val_loss={val_loss:.6f}  "
              f"best={best_val:.6f}  rel_improve={rel_improve:+.4f}  "
              f"no_improve_count={no_improve_count}", flush=True)

        torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss,
                    "noise_std": noise_std, "schedule": "linear", "loss_type": "stream_function"},
                   CKPT_DIR / "last.pt")

        if rel_improve > MIN_REL_IMPROVEMENT:
            best_val = val_loss
            no_improve_count = 0
            torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss,
                        "noise_std": noise_std, "schedule": "linear", "loss_type": "stream_function"},
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
