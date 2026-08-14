"""
run_mcg_dps_100seeds_3runs.py
=================================
Compares two guidance methods over 100 seeds (SEEDS = range(0, 700, 7),
same list used by run_all_methods.py / run_mcg_dps_ddimdps_100seeds.py),
each seed repeated 3 times (independent noise draw / stochastic sampling
trajectory per repeat) to capture run-to-run variance:

  1. MCG  z=0.04   (DPS gradient + hard re-noised merge, r=1 — Chung et al. 2022)
  2. DPS  z=0.04   (Chung et al. 2022, ancestral backbone)

Each seed fixes the ground-truth test sample (test_idx = seed % n_test) and
the robot path (biased_walk_path(seed=seed)); the 3 repeats vary only the
diffusion noise / sampling stochasticity via torch.manual_seed(seed*1000+rep).

Usage:
    python3 run_mcg_dps_100seeds_3runs.py --pickle data.pickle --checkpoint ckpt.pt \\
        --T 1000 --stride 10 --n_seeds 100 --n_repeats 3 --out_dir results/mcg_dps_100seeds_3runs
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset       import OceanCurrentDataset
from diffusion     import DDPM
from repaint_model import Repaint
from repaint_infer import biased_walk_path

SEEDS = list(range(0, 700, 7))   # 100 seeds: 0, 7, 14, ..., 693

METHODS = ["MCG z=0.04", "DPS z=0.04"]


# ── MCG (DPS gradient + hard re-noised merge, r=1) ─────────────────────────────

def mcg_infer(model, diffusion, x0_known, path_mask, land_mask,
              device="cpu", stride=1, step_size=0.04):
    H, W       = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t     = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        pred_noise = model(xt_in, t_vec)
        ab = diffusion.alpha_bar[t_int]
        x0_hat = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq  = (residual ** 2).sum()
        grad     = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            xt_unknown = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
            norm = norm_sq.sqrt().item() + 1e-8
            xt_unknown = xt_unknown - (step_size / norm) * grad.detach()

            t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known_noisy, _ = diffusion.q_sample(x0_known_t, t_prev_t)
            xt = known_t * xt_known_noisy + (1.0 - known_t) * xt_unknown
            xt = xt * ocean_t

    return xt.squeeze(0).cpu().numpy()


# ── DPS ─────────────────────────────────────────────────────────────────────

def dps_infer(model, diffusion, x0_known, path_mask, land_mask,
              device="cpu", stride=1, step_size=0.04):
    H, W       = x0_known.shape[1:]
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t    = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t     = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t    = 1.0 - land_t

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        pred_noise = model(xt_in, t_vec)
        ab     = diffusion.alpha_bar[t_int]
        x0_hat = (xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq  = (residual ** 2).sum()
        grad     = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            xt_next = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int)
            norm    = norm_sq.sqrt().item() + 1e-8
            xt_next = xt_next - (step_size / norm) * grad.detach()
            xt      = xt_next * ocean_t

    return xt.squeeze(0).cpu().numpy()


# ── Bar chart ─────────────────────────────────────────────────────────────────

def save_bar_chart(all_rmse, all_times, T, stride, n_seeds, n_repeats, out_path):
    methods = list(all_rmse.keys())
    rmse    = [np.mean(all_rmse[m]) for m in methods]
    std     = [np.std(all_rmse[m])  for m in methods]
    times   = [np.mean(all_times[m]) for m in methods]

    colors = ["#4C72B0", "#C44E52"]
    x = np.arange(len(methods))
    w = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    fig.suptitle(f"T={T} / stride={stride}  —  {n_seeds} seeds x {n_repeats} repeats", fontsize=11)

    ax = axes[0]
    bars = ax.bar(x, rmse, w, yerr=std, capsize=5, color=colors, alpha=0.85)
    ax.set_title("Mean RMSE (± 1 std)", fontsize=10)
    ax.set_ylabel("RMSE")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, max(rmse) * 1.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, times, w, color=colors, alpha=0.85)
    ax.set_title("Mean Inference Time per Run (s)", fontsize=10)
    ax.set_ylabel("Seconds")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, max(times) * 1.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--T",          type=int, default=1000)
    p.add_argument("--stride",     type=int, default=10)
    p.add_argument("--path_steps", type=int, default=150)
    p.add_argument("--base_ch",    type=int, default=64)
    p.add_argument("--time_dim",   type=int, default=256)
    p.add_argument("--n_seeds",    type=int, default=100)
    p.add_argument("--n_repeats",  type=int, default=3)
    p.add_argument("--out_dir",    default=None)
    p.add_argument("--mcg_step",   type=float, default=0.04)
    p.add_argument("--dps_step",   type=float, default=0.04)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.out_dir is None:
        args.out_dir = os.path.join(script_dir, "results",
                                    f"mcg_dps_T{args.T}_s{args.stride}_{args.n_seeds}seeds_{args.n_repeats}runs")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"T          : {args.T}  stride={args.stride}  "
          f"({len(range(0, args.T, args.stride))} diffusion steps)")
    print(f"Methods    : {len(METHODS)} ({', '.join(METHODS)})")
    print(f"N seeds    : {args.n_seeds}  x  {args.n_repeats} repeats each")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output dir : {args.out_dir}", flush=True)

    test_ds      = OceanCurrentDataset(args.pickle, split=2)
    train_ds     = OceanCurrentDataset(args.pickle, split=0)
    land_mask_np = test_ds.land_mask.numpy()
    ocean_mask   = ~land_mask_np
    n_test       = len(test_ds)

    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_ch",  args.base_ch)
    time_dim  = ckpt_args.get("time_dim", args.time_dim)
    schedule  = ckpt.get("schedule", "linear")

    model = Repaint(in_ch=2, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    noise_std = ckpt.get("noise_std", None)
    if noise_std is None:
        noise_std = float(train_ds.data[:, :, ~train_ds.land_mask].std())

    diffusion = DDPM(T=args.T, beta_schedule=schedule, device=device, noise_std=noise_std)

    print(f"Loaded     : epoch {ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}  "
          f"schedule={schedule}  noise_std={noise_std:.5f}\n", flush=True)

    seeds   = SEEDS[:args.n_seeds]
    n_total = len(seeds)

    all_rmse  = {m: [] for m in METHODS}
    all_times = {m: [] for m in METHODS}
    rows = []

    for run_i, seed in enumerate(seeds):
        sample_idx = seed % n_test
        x0_true    = test_ds[sample_idx]
        true_np    = x0_true.numpy()
        path_mask  = biased_walk_path(land_mask_np, n_steps=args.path_steps, seed=seed)

        x0_obs = x0_true.clone()
        x0_obs[:, ~torch.from_numpy(path_mask)] = 0.0

        for rep in range(args.n_repeats):
            print(f"[{run_i+1:03d}/{n_total:03d}]  seed={seed}  test_idx={sample_idx}  repeat={rep+1}/{args.n_repeats}", flush=True)
            row = [seed, sample_idx, rep]

            def run_method(name, fn, *fn_args, torch_seed, **fn_kwargs):
                torch.manual_seed(torch_seed)
                t0   = time.perf_counter()
                pred = fn(*fn_args, **fn_kwargs)
                t    = time.perf_counter() - t0
                rmse = float(np.sqrt(np.mean(
                    (pred[:, ocean_mask] - true_np[:, ocean_mask])**2)))
                all_rmse[name].append(rmse)
                all_times[name].append(t)
                print(f"  {name:<12}: RMSE={rmse:.4f}  t={t:.1f}s", flush=True)
                return rmse, t

            r, t = run_method("MCG z=0.04", mcg_infer, model, diffusion, x0_obs, path_mask, land_mask_np,
                               device=device, stride=args.stride, step_size=args.mcg_step,
                               torch_seed=seed * 1000 + rep)
            row += [r, t]
            r, t = run_method("DPS z=0.04", dps_infer, model, diffusion, x0_obs, path_mask, land_mask_np,
                               device=device, stride=args.stride, step_size=args.dps_step,
                               torch_seed=seed * 1000 + rep + 500)
            row += [r, t]

            rows.append(row)

    # ── CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    header = ["seed", "test_idx", "repeat"]
    for m in METHODS:
        key = m.replace(" ", "_").replace("=", "")
        header += [f"{key}_rmse", f"{key}_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ── Summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"2 Methods  —  T={args.T}  stride={args.stride}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Schedule   : {schedule}   noise_std={noise_std:.5f}\n")
        f.write(f"N seeds    : {n_total}   N repeats: {args.n_repeats}   "
                f"Total runs per method: {n_total * args.n_repeats}\n\n")
        f.write(f"{'Method':<14} {'Mean RMSE':>10} {'Std RMSE':>10} "
                f"{'Min':>8} {'Max':>8} {'Mean Time(s)':>13}\n")
        f.write("-" * 71 + "\n")
        for m in METHODS:
            rs = all_rmse[m]; ts = all_times[m]
            f.write(f"{m:<14} {np.mean(rs):>10.4f} {np.std(rs):>10.4f} "
                    f"{np.min(rs):>8.4f} {np.max(rs):>8.4f} {np.mean(ts):>13.2f}\n")

        # Per-seed aggregation across repeats (mean +- std over the 3 repeats)
        f.write("\nPer-seed breakdown (mean ± std over repeats):\n")
        hdr = f"  {'Seed':>6}  {'idx':>4}"
        for m in METHODS:
            hdr += f"  {m:>20}"
        f.write(hdr + "\n")
        f.write("-" * (len(hdr) + 2) + "\n")
        for idx, seed in enumerate(seeds):
            seed_rows = rows[idx * args.n_repeats:(idx + 1) * args.n_repeats]
            line = f"  {seed:6d}  {seed_rows[0][1]:4d}"
            for k, m in enumerate(METHODS):
                vals = [seed_rows[j][3 + k*2] for j in range(args.n_repeats)]
                line += f"  {np.mean(vals):>8.4f}±{np.std(vals):<8.4f}"
            f.write(line + "\n")

        f.write("\nFull per-run breakdown:\n")
        hdr = f"  {'Seed':>6}  {'idx':>4}  {'rep':>3}"
        for m in METHODS:
            hdr += f"  {m:>12}  {'t':>5}"
        f.write(hdr + "\n")
        f.write("-" * (len(hdr) + 2) + "\n")
        for row in rows:
            seed, idx, rep = row[0], row[1], row[2]
            line = f"  {seed:6d}  {idx:4d}  {rep:3d}"
            for k in range(len(METHODS)):
                rmse_v = row[3 + k*2]
                time_v = row[4 + k*2]
                line += f"  {rmse_v:12.4f}  {time_v:5.1f}"
            f.write(line + "\n")

    print(f"\nCSV saved     : {csv_path}")
    print(f"Summary saved : {summary_path}")

    # ── Bar chart
    chart_path = os.path.join(args.out_dir, "bar_chart.png")
    save_bar_chart(all_rmse, all_times, args.T, args.stride, n_total, args.n_repeats, chart_path)


if __name__ == "__main__":
    main()
