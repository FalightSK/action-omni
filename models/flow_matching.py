"""
Conditional Flow Matching action decoder.

Training objective (OT-CFM / linear interpolation):
    x_t   = (1 - t) * x_0  +  t * x_1          x_0 ~ N(0,I),  x_1 = ground-truth action
    target = x_1 - x_0                           constant-velocity field
    loss   = MSE( v_θ(x_t, t, cond),  target )

Inference (Euler ODE integration):
    x_{i+1} = x_i + (1/N) * v_θ(x_i, i/N, cond)   for i = 0 … N-1
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps a scalar timestep t ∈ [0, 1] to a (dim,) sinusoidal embedding.
    Scales t → 1000·t so the network sees a range similar to DDPM.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=t.dtype, device=t.device) / half
        )
        args = t[:, None] * freqs[None] * 1000.0   # (B, half)
        return torch.cat([args.cos(), args.sin()], dim=-1)   # (B, dim)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block with GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


# ──────────────────────────────────────────────────────────────────────────────
# Flow-matching decoder
# ──────────────────────────────────────────────────────────────────────────────

class FlowMatchingDecoder(nn.Module):
    """
    Predicts the velocity field v_θ(x_t, t, cond).

    Args:
        action_dim  : total flattened action size = action_horizon × per-step dim
        cond_dim    : conditioning vector size (VLM embedding + state)
        hidden_dim  : internal MLP width
        num_layers  : number of residual blocks
        dropout     : dropout probability
    """

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        # --- input branches ---
        self.time_emb   = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cond_proj   = nn.Linear(cond_dim, hidden_dim)

        # --- backbone ---
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # --- output ---
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

        # Zero-init output so training starts from a neutral velocity field
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_t:  torch.Tensor,   # (B, action_dim)
        t:    torch.Tensor,   # (B,)  ∈ [0, 1]
        cond: torch.Tensor,   # (B, cond_dim)
    ) -> torch.Tensor:
        h = (
            self.action_proj(x_t)
            + self.time_proj(self.time_emb(t))
            + self.cond_proj(cond)
        )
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.out_norm(h))   # (B, action_dim)

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        actions: torch.Tensor,   # (B, action_dim)   normalised ground-truth
        cond:    torch.Tensor,   # (B, cond_dim)
    ) -> torch.Tensor:
        """
        Sample t ~ U[0,1], build x_t via linear interpolation,
        return MSE loss against the constant velocity field.
        """
        B = actions.shape[0]
        device = actions.device

        t   = torch.rand(B, device=device, dtype=actions.dtype)
        x_0 = torch.randn_like(actions)
        x_t  = (1.0 - t[:, None]) * x_0 + t[:, None] * actions
        target = actions - x_0

        pred = self.forward(x_t, t, cond)
        return F.mse_loss(pred, target)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        cond:      torch.Tensor,   # (B, cond_dim)
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Euler integration from x_0 ~ N(0,I) to x_1 (predicted action).
        Returns shape (B, action_dim).
        """
        B      = cond.shape[0]
        device = cond.device
        dtype  = cond.dtype

        x  = torch.randn(B, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device, dtype=dtype)
            v = self.forward(x, t, cond)
            x = x + v * dt

        return x
