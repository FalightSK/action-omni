"""
configs/pusht/exp05_weighted.py
────────────────────────────────
Experiment 05 — PushT, DiT decoder, multi-scale layers (14, 21, 28),
rank-16 LoRA, learnable WEIGHTED SUM fusion (3 scalar params).

Hypothesis: Exp03's MultiScaleFusion (3.1M params, 96× LoRA) displaced LoRA,
causing it to degenerate to near-identity (+2.8% contribution vs +533% in Exp02a).
Constraining fusion capacity to << LoRA capacity (3 params vs 32K) should:
  1. Restore LoRA as the dominant task-adaptation mechanism (contribution >> +100%)
  2. Allow multi-scale features to provide marginal gain over Exp02a (if useful)

Architecture difference from Exp03:
  MultiScaleFusion  → Linear(3072 → 1024) + LN   ≈ 3,147,776 params  ← INVALID
  WeightedLayerFusion → softmax(alpha_0, alpha_1, alpha_2)  = 3 params ← VALID

Embedding cache: reuses Exp03 cache (identical layers 14/21/28 — no re-precompute)

Decision gates:
  SR > 56% AND LoRA ≥ +100%: hypothesis confirmed — weighted fusion is valid,
    multi-scale features help. Consider as architecture improvement over Exp02a.
  SR ≈ 56% AND LoRA ≥ +100%: LoRA restored, but layer 28 already captures
    all task-relevant info — multi-scale adds no value. Stick with Exp02a.
  SR < 56% OR LoRA < +100%: early layers hurt OR something else is wrong.
    Investigate learned weights alpha to see which layers dominate.

HARD CONSTRAINTS (same as all PushT experiments):
  - state_dim=2: agent position only. NEVER add block/goal position.
  - inference_horizon=4: must match training. Cannot change at inference time.
  - fusion_mode="weighted": capacity constraint satisfied (3 params << 32K LoRA)
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


@dataclass
class PushTExp05Weighted(BaseVLAConfig):
    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_path:     str = "asset/dataset/pusht"
    model_path:       str = "asset/model/Qwen3.5-0.8B"
    output_dir:       str = "asset/runs/pusht/exp05_weighted"
    # Reuse Exp03 cache — same layers (14, 21, 28), no re-precompute needed
    embeddings_cache: str = "asset/runs/pusht/exp03_multiscale/vlm_embeddings.pt"

    # ── Task ──────────────────────────────────────────────────────────────────
    task_text: str = "Push the T-shaped block onto the T-shaped target."

    # ── Dimensions ────────────────────────────────────────────────────────────
    # Golden rule: state = agent position ONLY. Never block/goal position.
    state_dim:  int = 2    # agent (x, y) only
    action_dim: int = 2

    # ── Horizons ──────────────────────────────────────────────────────────────
    action_horizon:    int = 16
    inference_horizon: int = 4   # HARD CONSTRAINT: set at training, cannot change

    use_relative_actions: bool = True

    # ── Normalisation (PushT pixel space — same as all PushT experiments) ─────
    action_mean: tuple = (-0.871, 0.678)
    action_std:  tuple = (20.187, 20.024)
    state_mean:  tuple = (229.11, 293.31)
    state_std:   tuple = (101.85,  96.49)

    # ── VLM token config ──────────────────────────────────────────────────────
    img_seq_len: int = 82
    img_grid_h:  int = 8
    img_grid_w:  int = 8
    # Same 3 layers as Exp03 cache (built with 8/16/24 — Qwen3.5-0.8B has 24 layers,
    # hidden_states[0..24]; the old config said (14,21,28) but 28 is out of range).
    vlm_extract_layers: tuple = (8, 16, 24)

    # ── Fusion: scalar weighted sum (3 params, << 32K LoRA capacity) ──────────
    # This is the key difference from Exp03 (which used MultiScaleFusion, 3.1M params)
    fusion_mode: str = "weighted"

    # ── Architecture (identical to Exp02a/Exp03) ──────────────────────────────
    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:  int = 20
    sim_max_steps: int = 300
