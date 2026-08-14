"""
RePaint-style inference for ocean current inpainting.

At inference time, we know the u/v values only along the robot's path
(a random walk on ocean grid cells).  The DDPM fills in everything else.

RePaint algorithm (per timestep t, repeated r times):
  1. Reverse step  → x_{t-1} from x_t via the model
  2. Merge         → known path cells get q(x_{t-1} | x_0_known),
                     unknown cells keep the model's prediction
  3. Resample      → if not the last iteration, go forward one step
                     x_t = q(x_t | x_{t-1}) and repeat
  4. Advance       → move to t-1 with the merged x_{t-1}
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Random-walk path generator
# ---------------------------------------------------------------------------

def random_walk_path(
    land_mask: np.ndarray,
    n_steps:   int = 150,
    seed:      int | None = None,
) -> np.ndarray:
    """
    Simulate a slow robot doing a random walk on ocean grid cells.

    Args:
        land_mask: (H, W) bool, True = land (robot cannot enter)
        n_steps:   number of steps (robot can revisit cells)
        seed:      optional RNG seed

    Returns:
        path_mask: (H, W) bool, True = cells the robot visited
    """
    rng = np.random.default_rng(seed)
    H, W = land_mask.shape

    ocean_cells = list(zip(*np.where(~land_mask)))
    if not ocean_cells:
        raise ValueError("No ocean cells found in land_mask.")

    start = ocean_cells[rng.integers(len(ocean_cells))]
    r, c = int(start[0]), int(start[1])

    path_mask = np.zeros((H, W), dtype=bool)
    path_mask[r, c] = True

    for _ in range(n_steps - 1):
        neighbors = [
            (r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < H and 0 <= c + dc < W and not land_mask[r + dr, c + dc]
        ]
        if neighbors:
            r, c = neighbors[rng.integers(len(neighbors))]
            path_mask[r, c] = True

    return path_mask


def biased_walk_path(
    land_mask:     np.ndarray,
    n_steps:       int = 150,
    seed:          int | None = None,
    straight_bias: float = 0.75,
) -> np.ndarray:
    """
    Random walk with directional persistence: the robot strongly prefers
    to continue in its current direction, producing a roughly straight path
    that still meanders and automatically navigates around land.

    Args:
        land_mask:     (H, W) bool, True = land
        n_steps:       number of steps
        seed:          optional RNG seed
        straight_bias: weight given to continuing straight (0–1).
                       0.75 → ~75% chance to continue, ~12.5% each side-step,
                       very small chance to reverse.

    Returns:
        path_mask: (H, W) bool, True = cells the robot visited
    """
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

    # Visit-count grid for exploration bonus (unvisited neighbours preferred)
    visit_count = np.zeros((H, W), dtype=np.float32)
    visit_count[r, c] = 1.0

    for _ in range(n_steps - 1):
        valid = [
            (dr, dc)
            for dr, dc in all_dirs
            if 0 <= r + dr < H and 0 <= c + dc < W and not land_mask[r + dr, c + dc]
        ]
        if not valid:
            break

        side = (1.0 - straight_bias) / 2.0
        weights = []
        for dr, dc in valid:
            dot = dr * cur_dir[0] + dc * cur_dir[1]
            if dot == 1:    # straight ahead
                w = straight_bias
            elif dot == 0:  # perpendicular
                w = side
            else:           # reverse — strongly discouraged
                w = side * 0.05

            # Exploration bonus: scale down weight if neighbour already visited
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


# ---------------------------------------------------------------------------
# RePaint inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def repaint(
    model:      torch.nn.Module,
    diffusion,                        # DDPM instance
    x0_known:   torch.Tensor,         # (2, H, W) observed field (0 outside path)
    path_mask:  np.ndarray,           # (H, W) bool, True = known path cell
    land_mask:  np.ndarray,           # (H, W) bool, True = land
    r:          int = 10,             # resampling iterations per timestep
    device:     str = "cpu",
    stride:     int = 1,              # step size through timesteps (1 = all T steps)
) -> torch.Tensor:
    """
    Run RePaint to reconstruct the full current field from sparse path observations.

    Args:
        model:     trained unconditional Repaint UNet
        diffusion: DDPM instance
        x0_known:  (2, H, W) tensor — true u/v at path cells, 0 elsewhere
        path_mask: (H, W) bool — True at cells the robot visited
        land_mask: (H, W) bool — True at land cells
        r:         RePaint resampling count (r=1 = no resampling, r=10 = paper default)
        device:    torch device string
        stride:    step size between timesteps (stride=1 uses all T steps,
                   stride=10 uses every 10th step for ~10x faster inference)

    Returns:
        x0_pred: (2, H, W) reconstructed vector field (land pixels = 0)
    """
    model.eval()
    H, W = x0_known.shape[1:]

    x0_known = x0_known.unsqueeze(0).to(device)      # (1, 2, H, W)

    # Masks as (1, 1, H, W) float for broadcasting
    known_t = torch.from_numpy(path_mask).float().to(device)[None, None]
    land_t  = torch.from_numpy(land_mask).float().to(device)[None, None]
    ocean_t = 1.0 - land_t

    # Start from pure noise scaled to data range
    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std
    xt = xt * ocean_t  # land stays 0

    T = diffusion.T

    # Build strided timestep list [0, stride, 2*stride, ..., T-1 or nearest]
    timesteps = list(range(0, T, stride))

    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        for j in range(r):
            # --- Step 1: model reverse step for unknown pixels ---
            xt_unknown = diffusion.p_sample_step(model, xt, t_int, t_prev_int)

            # --- Step 2: forward-diffuse x0_known to previous timestep ---
            t_prev_tensor = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known, _ = diffusion.q_sample(x0_known, t_prev_tensor)

            # --- Step 3: merge ---
            xt_merged = known_t * xt_known + (1.0 - known_t) * xt_unknown
            xt_merged = xt_merged * ocean_t   # keep land at 0

            # --- Step 4: resample (go forward one step) if not last iteration ---
            if j < r - 1 and t_int > 0:
                xt = diffusion.q_sample_from_prev(xt_merged, t_int)
                xt = xt * ocean_t
            else:
                xt = xt_merged

    return xt.squeeze(0).cpu()   # (2, H, W)
