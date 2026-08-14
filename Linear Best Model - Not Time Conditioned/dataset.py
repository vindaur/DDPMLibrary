import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class OceanCurrentDataset(Dataset):
    """
    Loads ocean current vector fields from the pickle file.
    Each sample is a (2, H, W) float32 tensor — channels are u and v.
    Land pixels (originally NaN) are set to 0.
    The land_mask attribute is a (H, W) bool tensor: True = land.

    Optional normalization: pass data_mean and data_std (scalars computed
    from the training split's ocean cells) to normalize fields to unit std.
    Land cells are re-zeroed after normalization.
    """

    def __init__(
        self,
        pickle_path: str,
        split:       int   = 0,
        data_mean:   float | None = None,
        data_std:    float | None = None,
    ):
        """
        Args:
            pickle_path: path to data.pickle
            split:       0 = train, 1 = val, 2 = test
            data_mean:   subtract this before dividing by data_std (optional)
            data_std:    divide by this to reach unit variance (optional)
        """
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        arr = data[split]  # (H, W, 2, N)
        H, W, _, N = arr.shape

        # Rearrange to (N, 2, H, W)
        arr_t = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)

        # Land mask: True where NaN (same for all timesteps)
        self.land_mask = torch.from_numpy(np.isnan(arr[:, :, 0, 0]))  # (H, W)

        # Replace NaN with 0 so tensors are finite
        self.data = torch.from_numpy(np.nan_to_num(arr_t, nan=0.0))  # (N, 2, H, W)

        # Optional normalization to unit std
        self.data_mean: float | None = data_mean
        self.data_std:  float | None = data_std
        if data_mean is not None and data_std is not None:
            self.data = (self.data - data_mean) / max(float(data_std), 1e-8)
            # Re-zero land cells (they were 0, now would be -mean/std)
            self.data[:, :, self.land_mask] = 0.0

    @staticmethod
    def compute_stats(pickle_path: str, split: int = 0) -> tuple[float, float]:
        """Return (mean, std) of ocean-cell values from the given split."""
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)
        arr        = raw[split]                        # (H, W, 2, N)
        ocean_mask = ~np.isnan(arr[:, :, 0, 0])       # (H, W)
        vals       = arr[ocean_mask, :, :].flatten()   # all ocean u/v values
        vals       = vals[~np.isnan(vals)].astype(np.float64)
        return float(vals.mean()), float(vals.std())

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]  # (2, H, W)
