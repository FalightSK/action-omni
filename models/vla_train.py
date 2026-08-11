"""
models/vla_train.py
────────────────────
VLATrainModel — combines VLMTokenAdapter with a flow-matching decoder.

This module is imported by both scripts/train.py and scripts/evaluate.py.
Moving it here (out of train.py root) removes the circular import that would
occur when inference.py imports VLATrainModel from train.py.

Decoder selection (via cfg.use_dit_decoder):
  False → FlowMatchingDecoder  (MLP, Exp01 baseline)
  True  → DiTFlowDecoder       (transformer, Exp02a+)

The DiT decoder receives two conditioning signals:
  1. cond_vec   (B, cond_dim) — pooled VLM context + state, used for adaLN
  2. vlm_tokens (B, seq, 512) — per-token spatially-aware features, used for
                                 cross-attention at every denoising step

Cache compatibility:
  v1: embed (B, 1024)         — legacy mean-pooled
  v2: embed (B, seq, 1024)    — single-layer full sequence
  v3: embed (B, L, seq, 1024) — multi-scale (Exp03)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.vla import VLMTokenAdapter
from models.flow_matching import FlowMatchingDecoder, DiTFlowDecoder


class VLATrainModel(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg     = cfg
        self.use_dit = getattr(cfg, "use_dit_decoder", False)

        self.adapter = VLMTokenAdapter(
            vlm_dim       = cfg.vlm_hidden_size,
            adapter_dim   = cfg.vlm_adapter_dim,
            lora_rank     = cfg.lora_rank,
            lora_scale    = cfg.lora_scale,
            pos_dim       = cfg.pos_enc_dim,
            readout_heads = cfg.readout_heads,
            dropout       = cfg.adapter_dropout,
            img_grid_h    = cfg.img_grid_h,
            img_grid_w    = cfg.img_grid_w,
            n_vlm_layers  = cfg.n_vlm_layers,
            fusion_mode   = getattr(cfg, "fusion_mode", "linear"),
            n_views       = getattr(cfg, "n_views", 1),
        )

        if self.use_dit:
            self.decoder = DiTFlowDecoder(
                action_dim     = cfg.action_dim,
                action_horizon = cfg.action_horizon,
                cond_dim       = cfg.cond_dim,
                hidden_dim     = cfg.dit_hidden_dim,
                num_layers     = cfg.dit_num_layers,
                n_heads        = cfg.dit_num_heads,
                cross_attn_dim = cfg.vlm_adapter_dim,   # 512 — after SpatialAwareMLP
                dropout        = cfg.decoder_dropout,
            )
        else:
            self.decoder = FlowMatchingDecoder(
                action_dim = cfg.action_dim * cfg.action_horizon,
                cond_dim   = cfg.cond_dim,
                hidden_dim = cfg.decoder_hidden_dim,
                num_layers = cfg.decoder_num_layers,
                dropout    = cfg.decoder_dropout,
            )

    def _build_cond(self, embed, state, img_mask):
        """
        Returns (cond_vec, vlm_tokens).

        cond_vec   : (B, cond_dim)   — pooled context ‖ state
        vlm_tokens : (B, seq, 512)   — per-token spatially-encoded features
                                       (used by DiT for cross-attention)
        """
        if embed.ndim == 2:                            # legacy v1 mean-pooled
            embed    = embed.unsqueeze(1)
            img_mask = torch.zeros(embed.shape[:2], dtype=torch.bool,
                                   device=embed.device)
        if img_mask is None:
            img_mask = torch.zeros(embed.shape[:2], dtype=torch.bool,
                                   device=embed.device)

        context, vlm_tokens = self.adapter(embed, img_mask, return_tokens=True)
        cond_vec = torch.cat([context, state], dim=-1)
        return cond_vec, vlm_tokens

    def forward(self, embed, state, actions, img_mask=None):
        cond_vec, vlm_tokens = self._build_cond(embed, state, img_mask)
        if self.use_dit:
            return self.decoder.compute_loss(actions, cond_vec, vlm_tokens)
        else:
            flat = actions.view(actions.shape[0], -1)
            return self.decoder.compute_loss(flat, cond_vec)

    @torch.no_grad()
    def sample(self, embed, state, num_steps=3, img_mask=None):
        cond_vec, vlm_tokens = self._build_cond(embed, state, img_mask)
        if self.use_dit:
            out = self.decoder.sample(cond_vec, num_steps, vlm_tokens)  # (B, H, D)
        else:
            out = self.decoder.sample(cond_vec, num_steps)               # (B, H*D)
        # Always return (B, H*D) flat for backward-compatible inference
        return out.reshape(out.shape[0], -1)
