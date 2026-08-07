"""x0-parameterized DDPM with a cosine schedule, PLAIN gaussian noise
(matches ddpm_library.stream.diffusion.DDPM's noise_type="gaussian" option --
the "div_free" noise type is deliberately NOT used here).
"""
import math
import torch


class CosineSchedule:
    def __init__(self, T: int = 1000, device=None):
        self.T = T
        self.device = device
        betas = self._cosine_betas(T)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _cosine_betas(T, s=0.008):
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = 1.0 - ab[1:] / ab[:-1]
        return betas.clamp(0, 0.999)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.device = device
        return self

    def q_sample(self, x0, t, noise=None):
        B = x0.shape[0]
        if noise is None:
            noise = torch.randn_like(x0)          # plain gaussian -- not div-free
        abar = self.alpha_bars[t].reshape(B, 1, 1, 1)
        x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
        return x_t, noise

    def p_step(self, x_t, x0_pred, t, noise=None):
        B = x_t.shape[0]
        alpha_t = self.alphas[t].reshape(B, 1, 1, 1)
        abar_t = self.alpha_bars[t].reshape(B, 1, 1, 1)
        beta_t = self.betas[t].reshape(B, 1, 1, 1)

        t_prev = (t - 1).clamp(min=0)
        abar_prev = self.alpha_bars[t_prev].reshape(B, 1, 1, 1)
        abar_prev = torch.where(t.reshape(B, 1, 1, 1) == 0, torch.ones_like(abar_prev), abar_prev)

        coeff_x0 = (abar_prev.sqrt() * beta_t) / (1 - abar_t)
        coeff_xt = (alpha_t.sqrt() * (1 - abar_prev)) / (1 - abar_t)
        mu = coeff_x0 * x0_pred + coeff_xt * x_t

        beta_tilde = ((1 - abar_prev) / (1 - abar_t)) * beta_t
        sigma = beta_tilde.sqrt()

        if noise is None:
            noise = torch.randn_like(x_t)          # plain gaussian -- not div-free
        mask_t0 = (t == 0).float().reshape(B, 1, 1, 1)
        return mu + (1 - mask_t0) * sigma * noise
