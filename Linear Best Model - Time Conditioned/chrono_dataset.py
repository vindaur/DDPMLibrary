"""
Time-conditioned dataset: each item is paired with the REAL ocean state
13 hours and 25 hours earlier (temporal priors), read from a continuous
chronological field array.

Requires a chrono_v1 pickle (built by utils/build_chrono_dataset.py /
utils/build_chrono_raw.py), e.g. "Stride Conditional/data_chrono_raw.pickle":
    {
      "format": "chrono_v1",
      "fields": (N, 2, H, W) float array, one continuous hourly sequence,
      "land_mask": (H, W) bool, True = land,
      "splits": {"train": [...], "val": [...], "test": [...]}  (target frame
                 indices into `fields`; NOT relative to each other -- a
                 target at index f always looks up fields[f-13]/fields[f-25]
                 in the SAME shared array, so priors are genuine history,
                 never a fabrication across an unrelated block boundary),
      "lags": [13, 25],
      ...
    }

This is intentionally minimal: ONLY the two temporal-prior fields are used
as conditioning (4 channels: prev_13h(u,v), prev_25h(u,v)) -- no sparse
observation channels, no static geometry. Robot-path observations for MCG/DPS
inference are handled separately, exactly as for the unconditional model
(see run_mcg_dps_z004_time_cond.py) -- this dataset only supplies the extra
INPUT signal the network sees during training.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_LAGS = (13, 25)


class ChronoOceanDataset(Dataset):
    """
    Each item is a dict:
        {
          "target": (2, H, W)   float32 -- the field to denoise (raw m/s, land=0)
          "cond":   (2*len(lags), H, W) float32 -- [prev_L0(u,v), prev_L1(u,v), ...]
        }

    Args:
        pickle_path: path to a chrono_v1 pickle.
        split:       0/"train", 1/"val", 2/"test".
        lags:        prior lags in hours/frames (default (13, 25)). Must not
                     exceed what the pickle was built with.
    """

    _SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}

    def __init__(self, pickle_path: str, split=0, lags=DEFAULT_LAGS):
        self.lags = tuple(int(l) for l in lags)
        if not self.lags:
            raise ValueError("lags must be non-empty")
        self.max_lag = max(self.lags)

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        if not (isinstance(data, dict) and data.get("format") == "chrono_v1"):
            raise ValueError(f"{pickle_path} is not a chrono_v1 pickle.")

        split_name = self._SPLIT_NAMES.get(split, split)
        if split_name not in data["splits"]:
            raise ValueError(
                f"split {split!r} -> {split_name!r} not in pickle "
                f"(have {list(data['splits'])})")
        self.split_name = split_name

        fields_np = np.nan_to_num(np.asarray(data["fields"], dtype=np.float32))  # (N,2,H,W)
        N = fields_np.shape[0]
        self.land_mask = torch.from_numpy(np.asarray(data["land_mask"], dtype=bool))
        self.fields = torch.from_numpy(fields_np)
        self.fields[:, :, self.land_mask] = 0.0

        self.valid = np.asarray(data["splits"][split_name], dtype=np.int64)
        if self.valid.size == 0:
            raise ValueError(f"split {split_name!r} has no target frames")
        if int(self.valid.min()) < self.max_lag:
            raise ValueError(
                f"split {split_name!r} contains a target index < max_lag="
                f"{self.max_lag}; requested lags {self.lags} exceed what the "
                f"pickle guarantees history for.")
        if int(self.valid.max()) >= N:
            raise ValueError(f"split {split_name!r} target index out of range for N={N}")

    @staticmethod
    def compute_stats(pickle_path: str, split=0):
        """Return (mean, std) of ocean-cell values -- matches OceanCurrentDataset's
        API for scripts that call this generically; unused by train_time_cond.py
        (which computes noise_std directly from the loaded tensor instead)."""
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        return float(data.get("data_mean", 0.0)), float(data["data_std"])

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int) -> dict:
        f = int(self.valid[idx])
        target = self.fields[f]                                       # (2, H, W)
        priors = torch.cat([self.fields[f - L] for L in self.lags], dim=0)  # (2*nlags, H, W)
        return {"target": target, "cond": priors}
