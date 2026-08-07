"""Compare DDPM, V-CNN, StreamDDPM(full), DivergentUNet, UnconstrainedStream,
and the new LinearStreamLoss model ("standard linear model" but trained with
StreamDDPM's stream_function loss instead of curl_div) on the same 100 seeds /
same real chronological ground truth / same biased-walk sparse sampling as
all earlier sweeps.
"""
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MODELS_DIR / "divergent_unet"))
sys.path.insert(0, str(MODELS_DIR / "unconstrained_stream"))
sys.path.insert(0, str(MODELS_DIR / "linear_streamloss"))
from divergent_predict import DivergentPredictor
from unconstrained_stream_predict import UnconstrainedStreamPredictor
from linear_streamloss_predict import LinearStreamLossPredictor

from ddpm_library import DDPM, VCNN, StreamDDPM
from ddpm_library.geo import grid_arrays
from ddpm_library.stream.paths import biased_walk_path

REPO = Path(r"c:\Users\Josep\Documents\GitHub\DDPMLibrary")
RESULTS = REPO / "Joseph" / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

ANGLE_SPEED_THRESHOLD = 0.02
N_SEEDS = 100
N_STEPS = 150
IMAGE_SEEDS = [0, 1, 2]

print("Loading data_chrono_raw.pickle ...")
with open(REPO / "Joseph" / "data_chrono_raw.pickle", "rb") as f:
    d = pickle.load(f)
fields = np.nan_to_num(d["fields"].astype(np.float32))   # (17040, 2, 94, 44) raw m/s
land_mask = np.asarray(d["land_mask"]).astype(bool)      # (94, 44) True=land
lags = tuple(int(x) for x in d["lags"])                  # (13, 25)
data_std = float(d["data_std"])
test_idx = d["splits"]["test"]

lat, lon = grid_arrays()

print("Loading models (cpu) ...")
ddpm = DDPM(device="cpu")
vcnn = VCNN(device="cpu")
stream = StreamDDPM(device="cpu")
divergent = DivergentPredictor(str(REPO / "Joseph" / "results" / "divergent_best.pt"), device="cpu")
unconstrained = UnconstrainedStreamPredictor(
    str(REPO / "Joseph" / "results" / "unconstrained_stream_best.pt"),
    land_mask, device="cpu", n_infer_steps=20)
linstream = LinearStreamLossPredictor(
    str(REPO / "Joseph" / "results" / "linear_streamloss_best.pt"),
    device="cpu", stride=10, step_size=0.04)
ocean_lib = stream.ocean_mask.astype(bool)   # (44, 94), matches vcnn.ocean_mask


def wrap_deg(diff_rad):
    dd = (diff_rad + np.pi) % (2 * np.pi) - np.pi
    return np.abs(np.degrees(dd))


def score(pred, gt, mask):
    diff = (pred - gt) * mask[..., None]
    mse = float((diff ** 2).sum() / (mask.sum() * 2))
    rmse = float(np.sqrt(mse))
    mag_pred = np.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2)
    mag_gt = np.sqrt(gt[..., 0] ** 2 + gt[..., 1] ** 2)
    mag_err = float(np.abs((mag_pred - mag_gt) * mask)[mask].mean())
    ang_mask = mask & (mag_gt > ANGLE_SPEED_THRESHOLD)
    if ang_mask.sum() > 0:
        ang_pred = np.arctan2(pred[..., 1], pred[..., 0])
        ang_gt = np.arctan2(gt[..., 1], gt[..., 0])
        ang_err = float(wrap_deg(ang_pred - ang_gt)[ang_mask].mean())
    else:
        ang_err = float("nan")
    return rmse, mag_err, ang_err


rows = []
t_sweep_start = time.time()
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    fi = int(rng.choice(test_idx))

    gt_model = fields[fi]                          # (2, 94, 44)
    gt_lib = np.transpose(gt_model, (2, 1, 0))       # (44, 94, 2)
    p13 = fields[fi - lags[0]]
    p25 = fields[fi - lags[1]]

    path_mask = biased_walk_path(land_mask, n_steps=N_STEPS, seed=seed, straight_bias=0.75)
    path_mask = path_mask & ~land_mask
    oi, oj = np.where(path_mask)
    n_obs = len(oi)
    obs = [
        (float(lat[b]), float(lon[a]), 0.0, float(gt_model[0, a, b]), float(gt_model[1, a, b]))
        for a, b in zip(oi, oj)
    ]

    mean_ddpm, _ = ddpm.predict(obs, single_step=True, seed=seed)
    mean_vcnn, _ = vcnn.predict(obs)
    mean_stream_full, _ = stream.predict(obs, priors=[p13, p25], seed=seed, full_field=True)
    mean_div, _ = divergent.predict(obs, seed=seed)

    target_std = gt_model / data_std
    priors_std = np.concatenate([p13 / data_std, p25 / data_std], axis=0)
    mean_unc_model, _ = unconstrained.predict(target_std, path_mask, priors_std, seed=seed)
    mean_unc = np.transpose(mean_unc_model, (2, 1, 0))   # (44,94,2)

    x0_known = np.zeros_like(gt_model)
    x0_known[:, path_mask] = gt_model[:, path_mask]
    mean_ls_model = linstream.predict(torch.from_numpy(x0_known), path_mask, land_mask, seed=seed)
    mean_ls = np.transpose(mean_ls_model, (2, 1, 0))   # (44,94,2)

    d_rmse, d_mag, d_ang = score(mean_ddpm, gt_lib, ocean_lib)
    v_rmse, v_mag, v_ang = score(mean_vcnn, gt_lib, ocean_lib)
    sf_rmse, sf_mag, sf_ang = score(mean_stream_full, gt_lib, ocean_lib)
    dv_rmse, dv_mag, dv_ang = score(mean_div, gt_lib, ocean_lib)
    u_rmse, u_mag, u_ang = score(mean_unc, gt_lib, ocean_lib)
    ls_rmse, ls_mag, ls_ang = score(mean_ls, gt_lib, ocean_lib)

    rows.append({
        "seed": seed, "frame": fi, "n_obs": n_obs,
        "ddpm_rmse": d_rmse, "ddpm_mag_err": d_mag, "ddpm_ang_err_deg": d_ang,
        "vcnn_rmse": v_rmse, "vcnn_mag_err": v_mag, "vcnn_ang_err_deg": v_ang,
        "stream_full_rmse": sf_rmse, "stream_full_mag_err": sf_mag, "stream_full_ang_err_deg": sf_ang,
        "divergentunet_rmse": dv_rmse, "divergentunet_mag_err": dv_mag, "divergentunet_ang_err_deg": dv_ang,
        "unconstrained_rmse": u_rmse, "unconstrained_mag_err": u_mag, "unconstrained_ang_err_deg": u_ang,
        "linstream_rmse": ls_rmse, "linstream_mag_err": ls_mag, "linstream_ang_err_deg": ls_ang,
    })
    elapsed = time.time() - t_sweep_start
    print(f"seed {seed:3d} frame={fi:5d} n_obs={n_obs:4d}  elapsed={elapsed:.0f}s  "
          f"DDPM[rmse={d_rmse:.4f} ang={d_ang:6.2f}]  "
          f"VCNN[rmse={v_rmse:.4f} ang={v_ang:6.2f}]  "
          f"STREAM_FULL[rmse={sf_rmse:.4f} ang={sf_ang:6.2f}]  "
          f"DIVERGENTUNET[rmse={dv_rmse:.4f} ang={dv_ang:6.2f}]  "
          f"UNCONSTRAINED[rmse={u_rmse:.4f} ang={u_ang:6.2f}]  "
          f"LINSTREAM[rmse={ls_rmse:.4f} ang={ls_ang:6.2f}]", flush=True)

    if seed in IMAGE_SEEDS:
        speed_gt = np.ma.masked_where(~ocean_lib, np.sqrt(gt_lib[..., 0]**2 + gt_lib[..., 1]**2))
        speed_ddpm = np.ma.masked_where(~ocean_lib, np.sqrt(mean_ddpm[..., 0]**2 + mean_ddpm[..., 1]**2))
        speed_vcnn = np.ma.masked_where(~ocean_lib, np.sqrt(mean_vcnn[..., 0]**2 + mean_vcnn[..., 1]**2))
        speed_stream = np.ma.masked_where(~ocean_lib, np.sqrt(mean_stream_full[..., 0]**2 + mean_stream_full[..., 1]**2))
        speed_div = np.ma.masked_where(~ocean_lib, np.sqrt(mean_div[..., 0]**2 + mean_div[..., 1]**2))
        speed_unc = np.ma.masked_where(~ocean_lib, np.sqrt(mean_unc[..., 0]**2 + mean_unc[..., 1]**2))
        speed_ls = np.ma.masked_where(~ocean_lib, np.sqrt(mean_ls[..., 0]**2 + mean_ls[..., 1]**2))
        vmax = float(speed_gt.max())

        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        ax = axes[0]
        ax.imshow(speed_gt, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.scatter(oi, oj, s=5, c="red", marker=".")
        ax.set_title(f"Ground truth (frame {fi})\n{n_obs} obs")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1]
        ax.imshow(speed_ddpm, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"DDPM  RMSE={d_rmse:.4f}")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2]
        ax.imshow(speed_vcnn, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"V-CNN  RMSE={v_rmse:.4f}")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[3]
        ax.imshow(speed_stream, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"StreamDDPM(full)  RMSE={sf_rmse:.4f}")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[4]
        ax.imshow(speed_div, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"DivergentUNet(raw)  RMSE={dv_rmse:.4f}")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[5]
        ax.imshow(speed_unc, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"Unconstrained-Stream  RMSE={u_rmse:.4f}")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[6]
        ax.imshow(speed_ls, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"Linear+StreamFnLoss  RMSE={ls_rmse:.4f}")
        ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        out = RESULTS / f"linstream_vs_all_seed{seed}.png"
        plt.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  saved {out}", flush=True)

csv_path = RESULTS / "linstream_vs_all_metrics.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved per-seed metrics: {csv_path}")

for model_name in ("ddpm", "vcnn", "stream_full", "divergentunet", "unconstrained", "linstream"):
    rmses = np.array([r[f"{model_name}_rmse"] for r in rows])
    mags = np.array([r[f"{model_name}_mag_err"] for r in rows])
    angs = np.array([r[f"{model_name}_ang_err_deg"] for r in rows])
    angs_valid = angs[~np.isnan(angs)]
    print(f"\n{model_name.upper()} over {N_SEEDS} seeds:")
    print(f"  RMSE          mean={rmses.mean():.4f}  std={rmses.std():.4f} m/s")
    print(f"  Magnitude err mean={mags.mean():.4f}  std={mags.std():.4f} m/s")
    print(f"  Angle err     mean={angs_valid.mean():.2f}  std={angs_valid.std():.2f} deg")
