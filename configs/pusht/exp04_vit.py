"""
configs/pusht/exp04_vit.py
───────────────────────────
Experiment 04 — PushT, ViT-B/16 backbone ablation.
PRIORITY 1 — must run before ALOHA.

Gate question: Does language-visual pretraining (Qwen) provide better
representations than vision-only (ViT-B/16)?

If Qwen >> ViT (p < 0.05): proceed with full language-visual pretraining claim.
If Qwen ≈ ViT: reframe as efficiency paper. Design OOD for compositional language.

Architecture: Same adapter + DiT decoder as Exp02a, but backbone = ViT-B/16
instead of Qwen3.5-0.8B. ViT has no language tokens — only image tokens.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


@dataclass
class PushTExp04ViT(BaseVLAConfig):
    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_path:     str = "asset/dataset/pusht"
    model_path:       str = "asset/model/vit-b-16"   # ViT-B/16 weights
    output_dir:       str = "asset/runs/pusht/exp04_vit"
    embeddings_cache: str = "asset/runs/pusht/exp04_vit/vlm_embeddings.pt"

    # ── Task ──────────────────────────────────────────────────────────────────
    # ViT has no language input — task_text is ignored during embedding
    task_text: str = "Push the T-shaped block onto the T-shaped target."

    # ── Dimensions ────────────────────────────────────────────────────────────
    state_dim:  int = 2
    action_dim: int = 2

    action_horizon:    int = 16
    inference_horizon: int = 4

    use_relative_actions: bool = True

    # ── Normalisation ─────────────────────────────────────────────────────────
    action_mean: tuple = (-0.871, 0.678)
    action_std:  tuple = (20.187, 20.024)
    state_mean:  tuple = (229.11, 293.31)
    state_std:   tuple = (101.85,  96.49)

    # ── ViT-B/16 token config ──────────────────────────────────────────────────
    # ViT-B/16: 96×96 input → 6×6 = 36 patches (patch_size=16)
    # + 1 [CLS] token = 37 tokens total, hidden_dim=768
    vlm_hidden_size: int = 768   # ViT-B hidden dim (vs 1024 for Qwen)
    img_seq_len: int = 37        # 6×6 patches + CLS
    img_grid_h:  int = 6
    img_grid_w:  int = 6
    vlm_extract_layers: tuple = (12,)  # Final ViT-B/16 layer (12 total)

    # NOTE: vlm_adapter_dim stays at 512 for fair comparison with Exp02a
    # The adapter maps 768 → 512 the same way it maps 1024 → 512

    # ── Architecture (same as Exp02a for fair comparison) ─────────────────────
    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:  int = 50   # need n≥50 for statistical comparison
    sim_max_steps: int = 300
