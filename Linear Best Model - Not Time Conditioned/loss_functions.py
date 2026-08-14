"""
Structural auxiliary loss functions for ocean current diffusion models.

All functions are stateless and operate on (B, 2, H, W) vector field tensors.
Import this module from any training script to add structural regularisation
on top of the base epsilon-MSE loss.

Available loss modes (LOSS_MODES):
    eps               Pure epsilon-MSE only — no auxiliary term.
    curl_div          MSE between curl and divergence fields of x̂₀ and x₀.
    spectral          MSE between FFT power spectra of x̂₀ and x₀.
    okubo_weiss       MSE between Okubo-Weiss parameters of x̂₀ and x₀.
    wasserstein       Sinkhorn–Wasserstein distance between vorticity point clouds.
    stream_function   MSE between approximate stream-function fields of x̂₀ and x₀.
    strain_rate       MSE between strain-rate tensor invariants of x̂₀ and x₀.

Omitting "eps" from the loss list trains with only the listed auxiliary losses —
no MSE term is added to the total.  Example: --loss curl_div spectral

Default weights (DEFAULT_WEIGHTS):
    curl_div          0.002
    spectral          0.000002
    okubo_weiss       0.001
    wasserstein       1.0
    stream_function   0.002
    strain_rate       0.001
"""

import torch
import torch.nn.functional as F

LOSS_MODES = (
    "eps", "curl_div", "spectral", "okubo_weiss", "wasserstein",
    "stream_function", "strain_rate",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "curl_div":        0.002,
    "spectral":        0.000002,
    "okubo_weiss":     0.0000001,
    "wasserstein":     1.0,
    "stream_function": 0.002,
    "strain_rate":     0.001,
}


# ---------------------------------------------------------------------------
# Shared finite-difference helper
# ---------------------------------------------------------------------------

def _jacobian(field: torch.Tensor):
    """
    First-order spatial derivatives of a (B, 2, H, W) vector field via
    central-difference convolution.

    Returns: (du_dx, du_dy, dv_dx, dv_dy)  each (B, 1, H, W).
    """
    u = field[:, 0:1]
    v = field[:, 1:2]

    kx = torch.tensor(
        [[[[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]]],
        device=field.device,
    ) / 2.0

    ky = torch.tensor(
        [[[[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]]]],
        device=field.device,
    ) / 2.0

    return (
        F.conv2d(u, kx, padding=1),   # du_dx
        F.conv2d(u, ky, padding=1),   # du_dy
        F.conv2d(v, kx, padding=1),   # dv_dx
        F.conv2d(v, ky, padding=1),   # dv_dy
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def curl_div_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between the curl and divergence of pred and true, masked to ocean.

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask (1 = ocean, 0 = land)
    """
    def _features(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        curl = dv_dx - du_dy
        div  = du_dx + dv_dy
        return torch.cat([curl, div], dim=1)   # (B, 2, H, W)

    return F.mse_loss(_features(pred) * ocean, _features(true) * ocean).sqrt()


def spectral_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between the FFT power spectra of pred and true.
    Land pixels are zeroed before the FFT so they contribute no energy.

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask
    """
    def _features(field):
        masked = field * ocean
        Su = torch.fft.rfft2(masked[:, 0]).abs()   # (B, H, W//2+1)
        Sv = torch.fft.rfft2(masked[:, 1]).abs()
        return torch.stack([Su, Sv], dim=1)         # (B, 2, H, W//2+1)

    return F.mse_loss(_features(pred), _features(true)).sqrt()


def okubo_weiss_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between the Okubo-Weiss parameter W of pred and true, masked to ocean.

        sₙ = du/dx − dv/dy   (normal strain)
        s_s = du/dy + dv/dx  (shear strain)
        ω   = dv/dx − du/dy  (vorticity)
        W   = sₙ² + s_s² − ω²

    W < 0 → rotation-dominated (eddy cores)
    W > 0 → strain-dominated  (eddy boundaries)

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask
    """
    def _ow(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        sn = du_dx - dv_dy   # normal strain
        ss = du_dy + dv_dx   # shear strain
        w  = dv_dx - du_dy   # vorticity
        return sn**2 + ss**2 - w**2   # (B, 1, H, W)

    return F.mse_loss(_ow(pred) * ocean, _ow(true) * ocean).sqrt()


def _vorticity_cloud(
    field: torch.Tensor,
    ocean: torch.Tensor,
    max_pts: int = 64,
):
    """
    Convert the vorticity field into a weighted 2-D point cloud for geomloss.

    Keeps only the top `max_pts` ocean pixels by |ω| magnitude to keep the
    Sinkhorn computation tractable without pykeops (O(N²) CPU otherwise).

    Returns:
        coords:  (B, max_pts, 2) — normalised (row, col) coordinates in [0, 1]
        weights: (B, max_pts, 1) — |ω| weights normalised to sum 1 per sample
    """
    du_dx, du_dy, dv_dx, _ = _jacobian(field)
    curl = (dv_dx - du_dy) * ocean   # (B, 1, H, W)

    B, _, H, W = curl.shape

    rows = torch.linspace(0, 1, H, device=field.device)
    cols = torch.linspace(0, 1, W, device=field.device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")   # (H, W)
    coords_all = torch.stack([grid_r, grid_c], dim=-1)           # (H, W, 2)
    coords_all = coords_all.view(1, H * W, 2).expand(B, -1, -1)  # (B, N, 2)

    raw = curl.abs().view(B, H * W)                              # (B, N)

    # Select top-max_pts points per sample by |ω|
    K = min(max_pts, raw.shape[1])
    topk_vals, topk_idx = raw.topk(K, dim=1)                    # (B, K)

    coords = coords_all.gather(
        1, topk_idx.unsqueeze(-1).expand(-1, -1, 2)
    )                                                            # (B, K, 2)

    total = topk_vals.sum(dim=1, keepdim=True)                   # (B, 1)
    uniform = torch.full_like(topk_vals, 1.0 / K)
    weights = torch.where(total > 1e-6, topk_vals / (total + 1e-12), uniform)

    # Renormalise to sum exactly 1 (required by geomloss)
    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights.unsqueeze(-1)                              # (B, K, 1)

    return coords, weights


def stream_function_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between approximate stream-function fields of pred and true.

    For a 2-D nearly-incompressible flow the stream function ψ satisfies
    u = ∂ψ/∂y and v = −∂ψ/∂x, so ∇²ψ = ω (vorticity).  We approximate ψ
    by integrating the vorticity with a Poisson solve via the FFT:

        ψ̂(k) = ω̂(k) / (kx² + ky²)   (DC component set to 0)

    The loss is the ocean-masked MSE between ψ_pred and ψ_true.

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask (1 = ocean, 0 = land)
    """
    def _stream(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        vorticity = (dv_dx - du_dy) * ocean   # (B, 1, H, W), land zeroed
        B, _, H, W = vorticity.shape

        # Wavenumber grids for Poisson solve in frequency domain
        kx = torch.fft.fftfreq(W, device=field.device).view(1, 1, 1, W) * 2 * torch.pi
        ky = torch.fft.fftfreq(H, device=field.device).view(1, 1, H, 1) * 2 * torch.pi
        k2 = kx ** 2 + ky ** 2                # (1, 1, H, W)
        k2[..., 0, 0] = 1.0                   # avoid divide-by-zero at DC

        omega_hat = torch.fft.fft2(vorticity)          # (B, 1, H, W) complex
        psi_hat   = omega_hat / k2
        psi_hat[..., 0, 0] = 0.0                       # zero mean
        psi       = torch.fft.ifft2(psi_hat).real      # (B, 1, H, W)
        return psi

    return F.mse_loss(_stream(pred) * ocean, _stream(true) * ocean).sqrt()


def strain_rate_loss(
    pred:  torch.Tensor,
    true:  torch.Tensor,
    ocean: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between strain-rate tensor invariants of pred and true, masked to ocean.

    The 2-D strain-rate tensor S has invariants:
        I₁ = trace(S) = du/dx + dv/dy           (= divergence)
        I₂ = det(S)   = (du/dx)(dv/dy)
                       − ¼(du/dy + dv/dx)²

    Both invariants are sensitive to deformation structures (fronts, filaments)
    that are not captured by curl or divergence alone.

    Args:
        pred, true: (B, 2, H, W) vector fields
        ocean:      (1, 1, H, W) float mask (1 = ocean, 0 = land)
    """
    def _invariants(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        I1 = du_dx + dv_dy                                    # divergence (trace)
        I2 = du_dx * dv_dy - 0.25 * (du_dy + dv_dx) ** 2    # determinant
        return torch.cat([I1, I2], dim=1)                     # (B, 2, H, W)

    return F.mse_loss(_invariants(pred) * ocean, _invariants(true) * ocean).sqrt()


def wasserstein_loss(
    pred:        torch.Tensor,
    true:        torch.Tensor,
    ocean:       torch.Tensor,
    sinkhorn_fn,
) -> torch.Tensor:
    """
    Sinkhorn–Wasserstein distance between the vorticity point clouds of
    pred and true, averaged over the batch.

    Tensors are moved to CPU before the Sinkhorn call because geomloss
    without pykeops uses a pure-Python CPU backend.

    Args:
        pred, true:  (B, 2, H, W) vector fields
        ocean:       (1, 1, H, W) float mask
        sinkhorn_fn: a geomloss.SamplesLoss instance
    """
    coords_pred, w_pred = _vorticity_cloud(pred, ocean)
    coords_true, w_true = _vorticity_cloud(true, ocean)
    # geomloss without pykeops requires CPU float32 tensors
    dist = sinkhorn_fn(
        w_pred.squeeze(-1).cpu().float(), coords_pred.cpu().float(),
        w_true.squeeze(-1).cpu().float(), coords_true.cpu().float(),
    )
    return dist.mean()
