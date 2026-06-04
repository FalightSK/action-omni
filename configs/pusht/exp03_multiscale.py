"""
configs/pusht/exp03_multiscale.py
──────────────────────────────────
Experiment 03 — PushT, DiT decoder, multi-scale layers (14, 21, 28), rank-16 LoRA.
Result: 44% SR (n=50), Wilson 95% CI [31%, 58%], p=0.317 vs Exp02a (NOT significant)

Key finding: MultiScaleFusion (3.1M params, 96× more than LoRA's 32K) displaced LoRA.
- LoRA: only +2.8% loss when zeroed (vs +533% in Exp02a) → effectively identity
- Fusion displaced LoRA: model memorised offline distribution → offline-online gap
- Valid fusion alternatives: scalar weighted sum (3 params), per-token gating (~66K)

CONCLUSION: Do NOT use naive large linear fusion. Exp02a is the better architecture.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


@dataclass
class PushTExp03MultiScale(BaseVLAConfig):
    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_path:     str = "asset/dataset/pusht"
    model_path:       str = "asset/model/Qwen3.5-0.8B"
    output_dir:       str = "asset/runs/pusht/exp03_multiscale"
    embeddings_cache: str = "asset/runs/pusht/exp03_multiscale/vlm_embeddings.pt"

    # ── Task ──────────────────────────────────────────────────────────────────
    task_text: str = "Push the T-shaped block onto the T-shaped target."

    # ── Dimensions ────────────────────────────────────────────────────────────
    state_dim:  int = 2
    action_dim: int = 2

    # ── Horizons ──────────────────────────────────────────────────────────────
    action_horizon:    int = 16
    inference_horizon: int = 4

    use_relative_actions: bool = True

    # ── Normalisation ─────────────────────────────────────────────────────────
    action_mean: tuple = (-0.871, 0.678)
    action_std:  tuple = (20.187, 20.024)
    state_mean:  tuple = (229.11, 293.31)
    state_std:   tuple = (101.85,  96.49)

    # ── VLM token config ──────────────────────────────────────────────────────
    img_seq_len: int = 82
    img_grid_h:  int = 8
    img_grid_w:  int = 8
    # Multi-scale: layers 8/16/24 (early/mid/final of Qwen3.5-0.8B's 24 transformer
    # layers). hidden_states has indices 0-24; the previous (14,21,28) was wrong — 28
    # is out of range. Cache was built with (8,16,24) — verify with cache metadata.
    vlm_extract_layers: tuple = (8, 16, 24)

    # ── Architecture ──────────────────────────────────────────────────────────
    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:  int = 20
    sim_max_steps: int = 300
