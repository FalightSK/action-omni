"""
configs/pusht/exp01_mlp.py
──────────────────────────
Experiment 01 — PushT, MLP decoder, single layer (28), rank-16 LoRA.
Result: 30% SR (n=50), Wilson 95% CI [19%, 44%]

Findings:
- LoRA: +300% val loss when zeroed → task-specific re-projection
- adaLN: +569% → most critical component (MLP has no cross-attn bypass)
- Readout: +457% → critical for MLP (only path to decoder)
- Spatial PE: +6% → helpful but not bottleneck
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


@dataclass
class PushTExp01MLP(BaseVLAConfig):
    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_path:     str = "asset/dataset/pusht"
    model_path:       str = "asset/model/Qwen3.5-0.8B"
    output_dir:       str = "asset/runs/pusht/exp01_mlp"
    embeddings_cache: str = "asset/runs/pusht/exp01_mlp/vlm_embeddings.pt"

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
    use_dit_decoder: bool = False   # MLP decoder (baseline)
    lora_rank:       int  = 16

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:  int = 20
    sim_max_steps: int = 300
