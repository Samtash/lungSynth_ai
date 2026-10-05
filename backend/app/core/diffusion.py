"""
DDPM forward process + training loss, and DDIM sampling for fast
inference (Section 2.1.2 / 4.3.4 of the proposal): T=1000 linear
schedule for training, ~50 deterministic DDIM steps at inference.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: corrupt x0 with noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    def training_loss(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """model_fn(x_t, t) -> predicted noise. x0 is the clean target
        intermediate-phase volume. Returns the DDPM MSE loss (Section 2.1.2)."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        x_t, noise = self.q_sample(x0, t)
        pred_noise = model_fn(x_t, t)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = "cpu",
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling (Song et al., 2021), reducing the
        1000-step DDPM chain to `num_steps` steps without retraining."""
        step_indices = torch.linspace(0, self.timesteps - 1, num_steps, device=device).long()
        step_indices = torch.flip(step_indices, dims=[0])

        x = torch.randn(shape, device=device)

        for i, t in enumerate(step_indices):
            t_batch = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            pred_noise = model_fn(x, t_batch)

            alpha_bar_t = self.alpha_bars[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i == len(step_indices) - 1:
                x = x0_pred
            else:
                t_prev = step_indices[i + 1]
                alpha_bar_prev = self.alpha_bars[t_prev]
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
                )
                dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

            if progress_cb is not None:
                progress_cb(i + 1, len(step_indices))

        return x
