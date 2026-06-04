"""
configs/pusht/exp02a_dit.py
───────────────────────────
Experiment 02a — PushT, DiT decoder, single layer (28), rank-16 LoRA, 2D state.
BEST RESULT: 56% SR (n=50), Wilson 95% CI [42%, 69%], p=0.0086 vs Exp01.

This is the reference architecture carried forward to ALOHA.

Mechanistic findings:
- LoRA: +533% val loss when zeroed → dominant task-projection (1024→16→1024)
- Cross-attention: +261% → per-step spatial querying is the load-bearing mechanism
- adaLN: +231% → global scene conditioning
- Readout: +160% → less critical because cross-attn provides a bypass
- Spatial PE: +12% → helpful but not the bottleneck

HARD CONSTRAINTS:
- state_dim=2: agent position only. NEVER add block/goal position.
- inference_horizon=4: must match training. Cannot change at inference time.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


@dataclass
class PushTExp02aDiT(BaseVLAConfig):
    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_path:     str = "asset/dataset/pusht"
    model_path:       str = "asset/model/Qwen3.5-0.8B"
    output_dir:       str = "asset/runs/pusht/exp02a_dit"
    # Exp01 and Exp02a use the same layer-28 single-layer cache — no re-run needed
    embeddings_cache: str = "asset/runs/pusht/exp01_mlp/vlm_embeddings.pt"

    # ── Task ──────────────────────────────────────────────────────────────────
    task_text: str = "Push the T-shaped block onto the T-shaped target."

    # ── Dimensions ────────────────────────────────────────────────────────────
    # Golden rule: state = agent position ONLY. Never block/goal position.
    state_dim:  int = 2    # agent (x, y) only — covariate shift eliminated
    action_dim: int = 2

    # ── Horizons ──────────────────────────────────────────────────────────────
    action_horizon:    int = 16
    inference_horizon: int = 4   # HARD CONSTRAINT: set at training, cannot change

    use_relative_actions: bool = True

    # ── Normalisation (PushT pixel space) ────────────────────────────────────
    action_mean: tuple = (-0.871, 0.678)
    action_std:  tuple = (20.187, 20.024)
    state_mean:  tuple = (229.11, 293.31)
    state_std:   tuple = (101.85,  96.49)

    # ── VLM token config (PushT: 96×96 → 8×8 grid = 64 img tokens + ~18 text) ─
    img_seq_len: int = 82
    img_grid_h:  int = 8
    img_grid_w:  int = 8
    vlm_extract_layers: tuple = (28,)

    # ── Architecture ──────────────────────────────────────────────────────────
    use_dit_decoder: bool = True   # DiT decoder
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:  int = 20
    sim_max_steps: int = 300
