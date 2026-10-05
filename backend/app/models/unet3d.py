"""
3D U-Net noise predictor for the phase-conditioned diffusion model
(Section 4.3.1), with FiLM conditioning at every residual block
(4.3.2) and a Phase-Aware Attention Module at the bottleneck (4.3.3).

Input: 3-channel volume = [noisy target phase x_t, clean T00 anchor,
clean T50 anchor], concatenated on the channel axis.
Output: 1-channel predicted noise, same spatial size as the input.
"""

from __future__ import annotations

import torch
from torch import nn

from app.models.embeddings import ConditionEmbedding, FiLM


class FiLMResBlock3D(nn.Module):
    """Two 3D conv layers + InstanceNorm + SiLU, FiLM-modulated by the
    (timestep, phase) conditioning vector, with a residual connection."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.film1 = FiLM(cond_dim, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.film2 = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.film1(self.norm1(self.conv1(x)), cond))
        h = self.act(self.film2(self.norm2(self.conv2(h)), cond))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class PhaseAwareAttention3D(nn.Module):
    """Cross-attention at the bottleneck (Section 4.3.3): the phase
    embedding is the query, the flattened spatial bottleneck feature map
    supplies keys and values. Spatial locations relevant to the requested
    breathing phase are amplified, others suppressed. Returns the
    attended feature map (same shape as input) plus the raw attention
    weights (for the interpretability overlay described in 4.3.3).
    """

    def __init__(self, channels: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.query_proj = nn.Linear(cond_dim, channels)
        self.kv_proj = nn.Conv3d(channels, channels * 2, kernel_size=1)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        b, c, d, h, w = x.shape
        residual = x
        xn = self.norm(x)

        kv = self.kv_proj(xn)
        k, v = kv.chunk(2, dim=1)
        k = k.reshape(b, self.num_heads, self.head_dim, d * h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, d * h * w)

        q = self.query_proj(cond).reshape(b, self.num_heads, self.head_dim, 1)

        attn_logits = torch.einsum("bhdi,bhdj->bhij", q, k) / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (b, heads, 1, d*h*w)

        out = torch.einsum("bhij,bhdj->bhdi", attn_weights, v)  # (b, heads, head_dim, 1)
        out = out.expand(-1, -1, -1, d * h * w).reshape(b, c, d, h, w)
        out = self.out_proj(out)

        spatial_weights = attn_weights.mean(dim=1).reshape(b, 1, d, h, w)
        return residual + out, spatial_weights


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        cond_dim: int = 256,
    ):
        super().__init__()
        self.cond_embed = ConditionEmbedding(cond_dim)

        chs = [base_channels * m for m in channel_mults]
        self.stem = nn.Conv3d(in_channels, chs[0], kernel_size=3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_ch = chs[0]
        for ch in chs:
            self.enc_blocks.append(FiLMResBlock3D(prev_ch, ch, cond_dim))
            self.downs.append(Downsample3D(ch))
            prev_ch = ch

        # Bottleneck
        self.bottleneck1 = FiLMResBlock3D(prev_ch, prev_ch, cond_dim)
        self.attention = PhaseAwareAttention3D(prev_ch, cond_dim)
        self.bottleneck2 = FiLMResBlock3D(prev_ch, prev_ch, cond_dim)

        # Decoder (mirrors encoder, concatenating skip connections)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_chs = list(reversed(chs))
        for ch in rev_chs:
            self.ups.append(Upsample3D(prev_ch))
            self.dec_blocks.append(FiLMResBlock3D(prev_ch + ch, ch, cond_dim))
            prev_ch = ch

        self.head = nn.Sequential(
            nn.GroupNorm(min(8, prev_ch), prev_ch),
            nn.SiLU(),
            nn.Conv3d(prev_ch, out_channels, kernel_size=3, padding=1),
        )

        self.last_attention_map: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        t00_anchor: torch.Tensor,
        t50_anchor: torch.Tensor,
        timesteps: torch.Tensor,
        phase_fraction: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.cond_embed(timesteps, phase_fraction)

        h = self.stem(torch.cat([x_t, t00_anchor, t50_anchor], dim=1))

        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        h = self.bottleneck1(h, cond)
        h, attn_map = self.attention(h, cond)
        self.last_attention_map = attn_map.detach()
        h = self.bottleneck2(h, cond)

        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = torch.nn.functional.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        return self.head(h)
