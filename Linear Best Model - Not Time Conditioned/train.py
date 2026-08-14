"""
Train the Repaint UNet with:
  - Choice of noise schedule: linear, cosine, geometric
  - curl_div structural loss (weight set via CURL_DIV_WEIGHT in diffusion.py)

Usage (run from workspace root):
    python Stride/train.py --schedule geometric
    python Stride/train.py --schedule linear --epochs 150
    python Stride/train.py --schedule cosine --epochs 150 --batch 32
    python Stride/train.py --schedule geometric --resume Stride/checkpoints_geometric/best_model_geometric.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for dataset.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # for local diffusion.py

import torch
from torch.utils.data import DataLoader

from dataset        import OceanCurrentDataset
from diffusion      import DDPM, CURL_DIV_WEIGHT
from repaint_model  import Repaint


def parse_args():
    p = argparse.ArgumentParser(description="Train Repaint UNet with curl_div loss.")
    p.add_argument("--pickle",   default="data.pickle")
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "cosine", "geometric"],
                   help="Beta noise schedule to use for training.")
    p.add_argument("--epochs",   type=int,   default=150)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--base_ch",  type=int,   default=64)
    p.add_argument("--time_dim", type=int,   default=256)
    p.add_argument("--T",        type=int,   default=1000)
    p.add_argument("--save_dir", default=None,
                   help="Checkpoint directory. Defaults to Stride/checkpoints_{schedule}/")
    p.add_argument("--resume",   default=None,
                   help="Path to a checkpoint to resume training from.")
    p.add_argument("--workers",  type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.save_dir is None:
        args.save_dir = os.path.join(script_dir, f"checkpoints_{args.schedule}")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Schedule        : {args.schedule}")
    print(f"Loss            : eps_mse + {CURL_DIV_WEIGHT} * curl_div")
    print(f"Save dir        : {args.save_dir}")

    # ---- Data ----
    train_ds = OceanCurrentDataset(args.pickle, split=0)
    val_ds   = OceanCurrentDataset(args.pickle, split=1)

    land_mask = train_ds.land_mask.to(device)

    ocean_pixels = train_ds.data[:, :, ~train_ds.land_mask]
    noise_std = float(ocean_pixels.std())
    print(f"noise_std  : {noise_std:.5f}  (ocean pixel std)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Model + diffusion ----
    model = Repaint(in_ch=2, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    diffusion = DDPM(T=args.T, beta_schedule=args.schedule, device=device, noise_std=noise_std)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- Resume ----
    start_epoch = 0
    best_val    = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "val_loss" in ckpt:
            best_val = ckpt["val_loss"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.5f}")

    # ---- Training loop ----
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x0 in train_loader:
            x0 = x0.to(device)
            loss = diffusion.training_loss(model, x0, land_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                val_loss += diffusion.training_loss(model, x0, land_mask).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "noise_std": noise_std,
            "curl_div_weight": CURL_DIV_WEIGHT,
            "schedule": args.schedule,
            "args": vars(args),
        }

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt_data, os.path.join(args.save_dir, f"best_model_{args.schedule}.pt"))

        if epoch % 50 == 0:
            torch.save(
                ckpt_data,
                os.path.join(args.save_dir, f"ckpt_{args.schedule}_epoch{epoch:04d}.pt"),
            )

    print(f"\nTraining complete.  Best val loss: {best_val:.5f}")
    print(f"Best checkpoint : {args.save_dir}/best_model_{args.schedule}.pt")


if __name__ == "__main__":
    main()
