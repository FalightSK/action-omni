"""
Conditional Flow Matching action decoders.

Training objective (OT-CFM / linear interpolation):
    x_t   = (1 - t) * x_0  +  t * x_1          x_0 ~ N(0,I),  x_1 = ground-truth
    target = x_1 - x_0                           constant-velocity field
    loss   = MSE( v_θ(x_t, t, cond), target )

Inference (Euler ODE):
    x_{i+1} = x_i + (1/N) * v_θ(x_i, i/N, cond)

Two decoder implementations:

  FlowMatchingDecoder  — MLP (Exp 1, baseline)
    Treats the whole action chunk as one flat vector.
    Simple and fast but cannot model inter-step correlations.

  DiTFlowDecoder       — Diffusion Transformer (Exp 2+)
    Treats each action step as a separate token.
    Self-attention: step t+k is aware of step t+k±1 → correlated trajectories.
    Cross-attention: each action step queries the 82 VLM tokens directly at
      every denoising iteration → no pooling bottleneck.
    adaLN: time + global condition modulates every DiT block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t ∈ [0, 1] → (dim,) sinusoidal embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=t.dtype, device=t.device) / half
        )
        args = t[:, None] * freqs[None] * 1000.0
        return torch.cat([args.cos(), args.sin()], dim=-1)   # (B, dim)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder 1 — MLP (Experiment 1 baseline)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class FlowMatchingDecoder(nn.Module):
    """
    MLP velocity field — Experiment 1 baseline.

    Treats the full action chunk as a single flat vector (action_dim * horizon).
    Fast and simple; cannot model correlations between action steps.
    """

    def __init__(
        self,
        action_dim:  int,
        cond_dim:    int,
        hidden_dim:  int   = 512,
        num_layers:  int   = 6,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        self.time_emb   = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cond_proj   = nn.Linear(cond_dim,   hidden_dim)
        self.blocks      = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t, t, cond, vlm_tokens=None):   # vlm_tokens ignored
        h = (self.action_proj(x_t)
             + self.time_proj(self.time_emb(t))
             + self.cond_proj(cond))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.out_norm(h))

    def compute_loss(self, actions, cond, vlm_tokens=None):
        # actions: (B, action_dim) flat  OR  (B, H, D) — flatten internally
        if actions.ndim == 3:
            actions = actions.view(actions.shape[0], -1)
        B, device = actions.shape[0], actions.device
        t    = torch.rand(B, device=device, dtype=actions.dtype)
        x_0  = torch.randn_like(actions)
        x_t  = (1 - t[:, None]) * x_0 + t[:, None] * actions
        pred = self.forward(x_t, t, cond)
        return F.mse_loss(pred, actions - x_0)

    @torch.no_grad()
    def sample(self, cond, num_steps=10, vlm_tokens=None):
        B, device, dtype = cond.shape[0], cond.device, cond.dtype
        x  = torch.randn(B, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device, dtype=dtype)
            x = x + self.forward(x, t, cond) * dt
        return x   # (B, action_dim) flat


# ──────────────────────────────────────────────────────────────────────────────
# Decoder 2 — DiT (Experiment 2+)
# ──────────────────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    One DiT block: adaLN self-attention + cross-attention to VLM tokens + FFN.

    adaLN (adaptive LayerNorm):
      A linear projection of the combined time+condition embedding generates
      per-block (scale, shift) for each LayerNorm.  This lets the condition
      directly modulate the normalisation statistics — more expressive than
      just adding the condition to the hidden state.

    Cross-attention to VLM tokens:
      Keys and Values come from the 82 adapted VLM tokens (after SpatialAwareMLP).
      Each action step can attend to any part of the image at every denoising step,
      instead of being bottlenecked through a single pooled vector.
    """

    def __init__(
        self,
        hidden_dim:     int,
        n_heads:        int,
        cross_attn_dim: int | None = None,
        dropout:        float      = 0.0,
    ) -> None:
        super().__init__()
        # --- Self-attention ---
        self.norm1     = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        # --- Cross-attention to VLM tokens ---
        self.has_cross = cross_attn_dim is not None
        if self.has_cross:
            self.norm_ca   = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, n_heads,
                kdim=cross_attn_dim, vdim=cross_attn_dim,
                dropout=dropout, batch_first=True,
            )
        # --- FFN ---
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # --- adaLN: generates (scale, shift) per norm ---
        # 4 values for sa+ffn norms, 2 more for cross-attn norm
        n_ada = 6 if self.has_cross else 4
        self.ada_proj = nn.Linear(hidden_dim, n_ada * hidden_dim)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)

    def _mod(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        return x * (1 + scale) + shift

    def forward(
        self,
        x:          torch.Tensor,           # (B, H, hidden_dim) action tokens
        cond_emb:   torch.Tensor,           # (B, hidden_dim)    time + condition
        vlm_tokens: torch.Tensor | None,    # (B, S, cross_attn_dim)
    ) -> torch.Tensor:
        ada = self.ada_proj(cond_emb).unsqueeze(1)   # (B, 1, n_ada * hidden_dim)

        if self.has_cross:
            s1, h1, s_ca, h_ca, s2, h2 = ada.chunk(6, dim=-1)
        else:
            s1, h1, s2, h2 = ada.chunk(4, dim=-1)

        # Self-attention
        sa_out, _ = self.self_attn(
            self._mod(self.norm1(x), s1, h1),
            self._mod(self.norm1(x), s1, h1),
            self._mod(self.norm1(x), s1, h1),
        )
        x = x + sa_out

        # Cross-attention to VLM token sequence
        if self.has_cross and vlm_tokens is not None:
            ca_out, _ = self.cross_attn(
                self._mod(self.norm_ca(x), s_ca, h_ca),
                vlm_tokens, vlm_tokens,
            )
            x = x + ca_out

        # FFN
        x = x + self.ffn(self._mod(self.norm2(x), s2, h2))
        return x


class DiTFlowDecoder(nn.Module):
    """
    Diffusion Transformer velocity field — Experiment 2+.

    Architecture:
      • Each of the H action steps becomes a token (B, H, action_dim) → (B, H, hidden_dim)
      • Learnable positional encoding distinguishes horizon steps
      • Time embedding + global condition → adaLN input for every block
      • N DiT blocks with:
          - Self-attention    : step t+k sees steps t+k±n (correlated trajectory)
          - Cross-attention   : each step queries all 82 VLM tokens (direct visual access)
          - adaLN             : time + state condition modulates every normalisation
      • Output: LayerNorm → Linear(hidden_dim → action_dim) per step

    Compared to MLP decoder:
      MLP  : flat 32D → one hidden state → 32D out  (no step awareness)
      DiT  : 16 tokens attend to each other AND to 82 VLM tokens → richer velocity field
    """

    def __init__(
        self,
        action_dim:     int   = 2,
        action_horizon: int   = 16,
        cond_dim:       int   = 518,
        hidden_dim:     int   = 256,
        num_layers:     int   = 6,
        n_heads:        int   = 8,
        cross_attn_dim: int   = 512,
        dropout:        float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim     = hidden_dim

        # --- Action token input ---
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_emb     = nn.Parameter(
            torch.randn(1, action_horizon, hidden_dim) * 0.02
        )

        # --- Time embedding ---
        self.time_emb  = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # --- Global condition (VLM context + state) projection for adaLN ---
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # --- DiT blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads,
                     cross_attn_dim=cross_attn_dim,
                     dropout=dropout)
            for _ in range(num_layers)
        ])

        # --- Output ---
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        n = sum(p.numel() for p in self.parameters())
        print(f"   DiTFlowDecoder: {n:,} params  "
              f"(hidden={hidden_dim}, layers={num_layers}, heads={n_heads}, "
              f"cross_attn_dim={cross_attn_dim})")

    def forward(
        self,
        x_t:        torch.Tensor,          # (B, H, action_dim) noisy action sequence
        t:          torch.Tensor,          # (B,) ∈ [0, 1]
        cond:       torch.Tensor,          # (B, cond_dim)
        vlm_tokens: torch.Tensor | None,   # (B, S, cross_attn_dim)
    ) -> torch.Tensor:                     # (B, H, action_dim)
        # Tokenise action sequence
        h = self.action_proj(x_t) + self.pos_emb   # (B, H, hidden_dim)

        # Build combined adaLN conditioning: time + global context
        cond_emb = (
            self.time_proj(self.time_emb(t))   # (B, hidden_dim)
            + self.cond_proj(cond)             # (B, hidden_dim)
        )

        for block in self.blocks:
            h = block(h, cond_emb, vlm_tokens)

        return self.out_proj(self.out_norm(h))   # (B, H, action_dim)

    def compute_loss(
        self,
        actions:    torch.Tensor,          # (B, H, action_dim)
        cond:       torch.Tensor,          # (B, cond_dim)
        vlm_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, device = actions.shape[0], actions.device
        t    = torch.rand(B, device=device, dtype=actions.dtype)
        x_0  = torch.randn_like(actions)
        x_t  = (1 - t[:, None, None]) * x_0 + t[:, None, None] * actions
        pred = self.forward(x_t, t, cond, vlm_tokens)
        return F.mse_loss(pred, actions - x_0)

    @torch.no_grad()
    def sample(
        self,
        cond:       torch.Tensor,          # (B, cond_dim)
        num_steps:  int                = 10,
        vlm_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:                     # (B, H, action_dim)
        B, device, dtype = cond.shape[0], cond.device, cond.dtype
        x  = torch.randn(B, self.action_horizon, self.action_dim,
                         device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device, dtype=dtype)
            x = x + self.forward(x, t, cond, vlm_tokens) * dt
        return x   # (B, H, action_dim)
