"""DDPM (linear schedule) with the stream_function auxiliary loss, in place
of the standard linear model's curl_div loss.

Identical to Repaint vs DPS/best-model-linear-curldiv-gaussian/diffusion.py's
DDPM class and training_loss method, except:
    total = eps_mse  +  STREAM_FN_WEIGHT * stream_function_loss(x0_hat, x0)
instead of curl_div_loss. Same weight (0.002, loss_functions.py's own
DEFAULT_WEIGHTS["stream_function"]).
"""
import math
import torch
import torch.nn.functional as F

STREAM_FN_WEIGHT = 0.002   # matches loss_functions.py DEFAULT_WEIGHTS["stream_function"]


def _jacobian(field: torch.Tensor):
    u = field[:, 0:1]
    v = field[:, 1:2]
    kx = torch.tensor([[[[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]]]], device=field.device) / 2.0
    ky = torch.tensor([[[[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]]]], device=field.device) / 2.0
    return (F.conv2d(u, kx, padding=1), F.conv2d(u, ky, padding=1),
            F.conv2d(v, kx, padding=1), F.conv2d(v, ky, padding=1))


def stream_function_loss(pred: torch.Tensor, true: torch.Tensor, ocean: torch.Tensor) -> torch.Tensor:
    """MSE between Poisson-recovered stream-function fields of pred and true.
    Verbatim from Stride Conditional/loss_functions.py::stream_function_loss."""
    def _stream(field):
        du_dx, du_dy, dv_dx, dv_dy = _jacobian(field)
        vorticity = (dv_dx - du_dy) * ocean
        B, _, H, W = vorticity.shape
        kx = torch.fft.fftfreq(W, device=field.device).view(1, 1, 1, W) * 2 * torch.pi
        ky = torch.fft.fftfreq(H, device=field.device).view(1, 1, H, 1) * 2 * torch.pi
        k2 = kx ** 2 + ky ** 2
        k2[..., 0, 0] = 1.0
        omega_hat = torch.fft.fft2(vorticity)
        psi_hat = omega_hat / k2
        psi_hat[..., 0, 0] = 0.0
        return torch.fft.ifft2(psi_hat).real
    return F.mse_loss(_stream(pred) * ocean, _stream(true) * ocean).sqrt()


VALID_SCHEDULES = ("linear", "cosine", "geometric", "quadratic", "sigmoid")


class DDPM:
    def __init__(self, T: int = 1000, beta_schedule: str = "linear",
                 device: str = "cpu", noise_std: float = 1.0,
                 stream_fn_weight: float = STREAM_FN_WEIGHT):
        if beta_schedule not in VALID_SCHEDULES:
            raise ValueError(f"beta_schedule must be one of {VALID_SCHEDULES}, got {beta_schedule!r}")
        self.T = T
        self.device = device
        self.noise_std = noise_std
        self.stream_fn_weight = stream_fn_weight

        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T)
        elif beta_schedule == "cosine":
            betas = self._cosine_betas(T)
        else:
            raise NotImplementedError(f"{beta_schedule} not needed for this experiment")

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=device), self.alpha_bar[:-1]])
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_one_mab = (1.0 - self.alpha_bar).sqrt()

    def _cosine_betas(self, T, s=0.008):
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = 1.0 - ab[1:] / ab[:-1]
        return betas.clamp(0, 0.999)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0) * self.noise_std
        sqrt_ab = self.sqrt_ab[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_mab[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_mab * noise, noise

    def training_loss(self, model, x0, land_mask):
        """total = eps_mse(pred_noise, noise) + stream_fn_weight * stream_function_loss(x0_hat, x0)
        (ocean-masked). Unconditional -- no cond argument, matching the standard linear model."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = model(xt, t)

        ocean = (~land_mask).float()[None, None]
        eps_loss = F.mse_loss(pred_noise * ocean, noise * ocean)

        if self.stream_fn_weight == 0.0:
            return eps_loss, eps_loss, torch.tensor(0.0)

        ab = self.alpha_bar[t][:, None, None, None]
        x0_hat = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        sf_loss = stream_function_loss(x0_hat, x0, ocean)
        total = eps_loss + self.stream_fn_weight * sf_loss
        return total, eps_loss, sf_loss

    @torch.no_grad()
    def p_sample_step(self, model, xt, t_int, t_prev_int=None):
        if t_prev_int is None:
            t_prev_int = max(t_int - 1, 0)
        B = xt.shape[0]
        t = torch.full((B,), t_int, device=self.device, dtype=torch.long)
        pred_noise = model(xt, t)

        ab = self.alpha_bar[t_int]
        ab_prev = self.alpha_bar[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=self.device)
        alpha_eff = ab / ab_prev
        beta_eff = 1.0 - alpha_eff

        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1.5, 1.5)
        if t_int == 0:
            return x0_pred

        coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
        coef2 = alpha_eff.sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean = coef1 * x0_pred + coef2 * xt
        var = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
        return mean + var.sqrt() * torch.randn_like(xt) * self.noise_std
