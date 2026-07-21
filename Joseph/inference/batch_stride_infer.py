"""
Batch stride inference experiment.

Runs RePaint inference at multiple stride values on a set of test samples
and records RMSE for each, so you can compare speed vs. accuracy trade-offs.

Default strides tested: 1, 5, 10, 20, 50
  stride=1  -> full 1000 steps (slowest, most accurate)
  stride=50 -> only 20 steps   (fastest, least accurate)

Usage (run from workspace root):
    python GeometricStride/batch_stride_infer.py
    python GeometricStride/batch_stride_infer.py --checkpoint GeometricStride/checkpoints/best_model.pt
    python GeometricStride/batch_stride_infer.py --strides 1 10 20 50 --n_samples 20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for dataset.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # for local diffusion.py

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path


# ---------------------------------------------------------------------------
# Quiver helper (identical to batch_repaint.py)
# ---------------------------------------------------------------------------

def plot_field(ax, u, v, land_mask, title, step=2, cmap="cool"):
    H, W = u.shape
    ax.imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    yq, xq = np.mgrid[0:H:step, 0:W:step]
    uq = u[::step, ::step]
    vq = v[::step, ::step]
    mq = np.sqrt(uq ** 2 + vq ** 2)
    mask = ~np.isnan(uq) & ~land_mask[::step, ::step]
    q = ax.quiver(
        xq[mask], yq[mask], uq[mask], vq[mask], mq[mask],
        cmap=cmap, clim=(0, np.nanpercentile(mq[mask], 98) if mask.any() else 1),
        scale=12, width=0.003, zorder=2,
    )
    plt.colorbar(q, ax=ax, label="Speed", shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")


def save_combined_plot(u_true, v_true, land_mask, path_mask,
                       stride_results, strides, T,
                       seed, sample_idx, checkpoint, out_path):
    """
    One image per sample showing all strides side by side.
    Layout: 2 rows x (n_strides + 1) columns
      Row 0: Ground Truth | Pred (stride=s1) | Pred (stride=s2) | ...
      Row 1: Path         | Error (stride=s1) | Error (stride=s2) | ...
    stride_results: dict {stride: (u_pred, v_pred, err, rmse)}
    """
    n = len(strides)
    H, W = land_mask.shape
    fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 10))

    # [0, 0] Ground Truth
    plot_field(axes[0, 0], u_true, v_true, land_mask, "Ground Truth")

    # [1, 0] Path
    axes[1, 0].imshow(
        land_mask, origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", "black"]),
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=0,
    )
    path_display = np.zeros_like(land_mask, dtype=float)
    path_display[path_mask] = 1.0
    axes[1, 0].imshow(
        path_display, origin="lower", cmap="Reds", alpha=0.8,
        extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=1, vmin=0, vmax=1,
    )
    axes[1, 0].set_title(f"Robot Path ({int(path_mask.sum())} cells\nseed={seed})", fontsize=11)
    axes[1, 0].set_xlabel("X"); axes[1, 0].set_ylabel("Y")
    ocean_p = mpatches.Patch(facecolor="white", edgecolor="gray", label="Ocean")
    path_p  = mpatches.Patch(facecolor="#d62728",                 label="Path")
    land_p  = mpatches.Patch(facecolor="black",                   label="Land")
    axes[1, 0].legend(handles=[ocean_p, path_p, land_p], loc="upper right", fontsize=8)

    for col, stride in enumerate(strides, start=1):
        u_pred, v_pred, err, rmse = stride_results[stride]
        n_steps = len(range(0, T, stride))

        # [0, col] Predicted field
        plot_field(axes[0, col], u_pred, v_pred, land_mask,
                   f"stride={stride} ({n_steps} steps)\nRMSE={rmse:.4f}")

        # [1, col] Error map
        err_plot = np.ma.masked_where(land_mask, err)
        im = axes[1, col].imshow(
            err_plot, origin="lower", cmap="hot_r", aspect="auto",
            extent=[-0.5, W - 0.5, -0.5, H - 0.5],
        )
        axes[1, col].imshow(
            land_mask, origin="lower",
            cmap=plt.matplotlib.colors.ListedColormap(["none", "black"]),
            extent=[-0.5, W - 0.5, -0.5, H - 0.5], aspect="auto", zorder=1,
        )
        plt.colorbar(im, ax=axes[1, col], label="|error| speed", shrink=0.7)
        axes[1, col].set_title(f"Error (RMSE={rmse:.4f})", fontsize=11)
        axes[1, col].set_xlabel("X"); axes[1, col].set_ylabel("Y")

    plt.suptitle(
        f"Test sample {sample_idx}  —  seed={seed}  checkpoint={os.path.basename(checkpoint)}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# RePaint inference (stride-aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def repaint(
    model, diffusion,
    x0_known, path_mask, land_mask,
    r=10, device="cpu", stride=1,
):
    H, W = x0_known.shape[1:]
    x0_known = x0_known.unsqueeze(0).to(device)

    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t  = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std
    xt = xt * ocean_t
    T  = diffusion.T

    timesteps = list(range(0, T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        for j in range(r):
            xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)

            t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known_t, _ = diffusion.q_sample(x0_known, t_prev_t)

            xt_merged = known_t * xt_known_t + (1.0 - known_t) * xt_unknown
            xt_merged = xt_merged * ocean_t

            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int, t_prev_int) * ocean_t
            else:
                xt = xt_merged

    return xt.squeeze(0).cpu().numpy()  # (2, H, W)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare RePaint inference at multiple stride values.")
    p.add_argument("--pickle",      default="data.pickle")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to checkpoint. Defaults to GeometricStride/checkpoints/best_model.pt")
    p.add_argument("--strides",     type=int, nargs="+", default=[1, 5, 10, 20, 50],
                   help="List of stride values to test.")
    p.add_argument("--n_samples",     type=int, default=10,
                   help="Number of test samples to evaluate per stride.")
    p.add_argument("--sample_start",  type=int, default=0,
                   help="Test dataset index to start from (0-based).")
    p.add_argument("--torch_seed",   type=int, default=None,
                   help="If set, seeds torch RNG before each repaint() call so all strides get identical noise.")
    p.add_argument("--path_steps",  type=int, default=150)
    p.add_argument("--resample",    type=int, default=10, help="RePaint r parameter.")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--T",           type=int, default=1000)
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--time_dim",    type=int, default=256)
    p.add_argument("--out",         default=None,
                   help="Output .txt path. Defaults to GeometricStride/results/stride_comparison.txt")
    p.add_argument("--img_dir",     default=None,
                   help="Directory for per-sample images. Defaults to Stride/results/images/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.checkpoint is None:
        args.checkpoint = os.path.join(script_dir, "checkpoints", "best_model.pt")

    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    if args.out is None:
        args.out = os.path.join(results_dir, "stride_comparison.txt")
    if args.img_dir is None:
        args.img_dir = os.path.join(results_dir, "images")
    os.makedirs(args.img_dir, exist_ok=True)

    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Strides     : {args.strides}")
    print(f"N samples   : {args.n_samples}")

    # ---- Data ----
    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    land_mask_np = test_ds.land_mask.numpy()
    ocean_mask   = ~land_mask_np

    # ---- Model ----
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    T         = ckpt_args.get("T",        args.T)

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        train_ds  = OceanCurrentDataset(args.pickle, split=0)
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())

    beta_schedule = ckpt.get("schedule", "geometric")

    print(f"Loaded      : epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}, "
          f"schedule={beta_schedule}, curl_div_weight={ckpt.get('curl_div_weight', 0.002)}, noise_std={noise_std:.5f}")

    diffusion = DDPM(T=T, beta_schedule=beta_schedule, device=device, noise_std=noise_std)

    sample_start = args.sample_start
    n_samples = min(args.n_samples, len(test_ds) - sample_start)

    # ---- Run experiment ----
    # results[stride] = list of per-sample RMSE
    results = {s: [] for s in args.strides}

    for idx in range(sample_start, sample_start + n_samples):
        x0_true  = test_ds[idx]
        true_np  = x0_true.numpy()
        path_mask = biased_walk_path(
            land_mask_np, n_steps=args.path_steps, seed=args.seed + idx
        )

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        print(f"\nSample {idx - sample_start + 1}/{n_samples}  (test idx={idx})")

        stride_data = {}
        for stride in args.strides:
            n_steps = len(range(0, T, stride))
            if args.torch_seed is not None:
                torch.manual_seed(args.torch_seed)
            x0hat = repaint(
                model, diffusion, x0_obs, path_mask, land_mask_np,
                r=args.resample, device=device, stride=stride,
            )
            rmse = float(np.sqrt(np.mean(
                (x0hat[:, ocean_mask] - true_np[:, ocean_mask]) ** 2
            )))
            results[stride].append(rmse)
            print(f"  stride={stride:3d}  ({n_steps:4d} steps)  RMSE={rmse:.4f}")

            u_pred = x0hat[0]
            v_pred = x0hat[1]
            err = np.sqrt((u_pred - true_np[0]) ** 2 + (v_pred - true_np[1]) ** 2)
            err[land_mask_np] = np.nan
            stride_data[stride] = (u_pred, v_pred, err, rmse)

        seed = args.seed + idx
        out_path = os.path.join(args.img_dir, f"result_{idx+1:02d}_all_strides.png")
        save_combined_plot(
            true_np[0].T, true_np[1].T, land_mask_np.T, path_mask.T,
            {s: (d[0].T, d[1].T, d[2].T, d[3]) for s, d in stride_data.items()},
            args.strides, T, seed, idx + 1, args.checkpoint, out_path,
        )

    # ---- Summary ----
    lines = []
    lines.append(f"Geometric schedule stride comparison")
    lines.append(f"Checkpoint       : {args.checkpoint}")
    lines.append(f"curl_div_weight  : {ckpt.get('curl_div_weight', 0.002)}")
    lines.append(f"noise_std        : {noise_std:.5f}")
    lines.append(f"N samples  : {n_samples}")
    lines.append(f"resample r : {args.resample}")
    lines.append(f"path_steps : {args.path_steps}")
    lines.append("")
    lines.append(f"{'Stride':>8}  {'Steps':>6}  {'Mean RMSE':>10}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    lines.append("-" * 56)

    for stride in sorted(results.keys()):
        rmses = results[stride]
        n_steps = len(range(0, T, stride))
        lines.append(
            f"{stride:>8}  {n_steps:>6}  {np.mean(rmses):>10.4f}  "
            f"{np.std(rmses):>8.4f}  {np.min(rmses):>8.4f}  {np.max(rmses):>8.4f}"
        )

    lines.append("")
    lines.append("Per-sample breakdown:")
    for idx in range(n_samples):
        row = "  ".join(f"s{s}={results[s][idx]:.4f}" for s in sorted(results.keys()))
        lines.append(f"  sample {idx:3d}: {row}")

    report = "\n".join(lines)
    print("\n" + report)

    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
