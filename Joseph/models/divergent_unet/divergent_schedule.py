"""Standard (non-split) DDPM noise schedule -- x0-parameterized.

Unlike ddpm_library.model.schedule.HelmholtzSplitSchedule, this does NOT
decompose the field into solenoidal/irrotational parts or use two separate
beta schedules / spectrally-shaped noise. Plain isotropic Gaussian noise,
one linear beta schedule -- appropriate since the training target here (the
divergent component) is a single, already-decomposed field, not a full
velocity field needing internal splitting.
"""
import torch


class StandardSchedule:
    def __init__(self, n_steps=1000, min_beta=1e-4, max_beta=0.02, device=None):
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(min_beta, max_beta, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.device = device
        return self

    def q_sample(self, x0, t, noise=None):
        B = x0.shape[0]
        if noise is None:
            noise = torch.randn_like(x0)
        abar = self.alpha_bars[t].reshape(B, 1, 1, 1)
        x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
        return x_t, noise

    def p_step(self, x_t, x0_pred, t, noise=None):
        """Ancestral posterior sample x_{t-1} | x_t, x0_pred."""
        B = x_t.shape[0]
        alpha_t = self.alphas[t].reshape(B, 1, 1, 1)
        abar_t = self.alpha_bars[t].reshape(B, 1, 1, 1)
        beta_t = self.betas[t].reshape(B, 1, 1, 1)

        t_prev = (t - 1).clamp(min=0)
        abar_prev = self.alpha_bars[t_prev].reshape(B, 1, 1, 1)
        abar_prev = torch.where(
            t.reshape(B, 1, 1, 1) == 0, torch.ones_like(abar_prev), abar_prev)

        coeff_x0 = (abar_prev.sqrt() * beta_t) / (1 - abar_t)
        coeff_xt = (alpha_t.sqrt() * (1 - abar_prev)) / (1 - abar_t)
        mu = coeff_x0 * x0_pred + coeff_xt * x_t

        beta_tilde = ((1 - abar_prev) / (1 - abar_t)) * beta_t
        sigma = beta_tilde.sqrt()

        if noise is None:
            noise = torch.randn_like(x_t)
        mask_t0 = (t == 0).float().reshape(B, 1, 1, 1)
        return mu + (1 - mask_t0) * sigma * noise
