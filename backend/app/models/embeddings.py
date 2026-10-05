"""
Sinusoidal timestep / phase embeddings and the FiLM conditioning layer,
matching Section 4.3.2 of the proposal.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding.

    x: (B,) float tensor of scalar values (timestep index or phase fraction).
    Returns: (B, dim) embedding.
    """
    device = x.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / half)
    args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class ConditionEmbedding(nn.Module):
    """Combines the diffusion timestep and the breathing-phase fraction
    (p = xx/100, e.g. T30 -> 0.3) into a single conditioning vector used
    for FiLM modulation at every residual block, as described in 4.3.2.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.phase_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.combine = nn.Sequential(nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, timesteps: torch.Tensor, phase_fraction: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(timesteps, self.dim))
        p_emb = self.phase_mlp(sinusoidal_embedding(phase_fraction * 1000.0, self.dim))
        return self.combine(torch.cat([t_emb, p_emb], dim=-1))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: produces per-channel scale (gamma)
    and shift (beta) from the conditioning vector and applies

        F_tilde = gamma (x) F + beta

    to a 3D feature map, elementwise per-channel (Eq. in Section 4.3.2).
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, feature_map: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return gamma * feature_map + beta
