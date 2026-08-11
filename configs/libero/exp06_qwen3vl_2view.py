"""
configs/libero/exp06_qwen3vl_2view.py
─────────────────────────────────────
LIBERO-Goal Exp06 — stock Qwen3-VL-2B with BOTH cameras.

The control for exp05, and the arm that makes the comparison fair. See
exp05_groot_2view.py for why the second camera is required rather than optional.

Everything except the backbone weights is inherited from exp05: same two views
in the same order, same 152-token budget, same layer-16 read, same head, same
schedule, same 50 fixed evaluation initial states. Read depth stays at 16 —
GR00T's documented `select_layer`, which is FINAL for its 16-layer stack but
INTERMEDIATE for this model's 28, so encode_vlm's final-RMSNorm correction fires
here and not for exp05. Without it the pair would differ by normalisation rather
than by weights.

The hypothesis this pair tests
──────────────────────────────
H1, but with the observation-spec confound removed:

    SR(Qwen3-VL, 2 views) >= 0.9 x SR(GR00T, 2 views)
        -> robot pretraining of the VLM is not necessary, on a
           benchmark-matched observation spec
    GR00T leads by a wide margin
        -> the single-view exp03/exp04 result (62.5% vs 68.0%) was an artifact
           of denying the pretrained backbone the views it was trained on

Both outcomes are reportable, and the second is the one worth knowing before a
reviewer finds it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.libero.exp05_groot_2view import LiberoExp05Groot2View


@dataclass
class LiberoExp06Qwen3VL2View(LiberoExp05Groot2View):
    vlm_backbone: str = "qwen3vl"
    output_dir: str = "asset/runs/libero/exp06_qwen3vl_2view"
    embeddings_cache: str = "asset/runs/libero/exp06_qwen3vl_2view/vlm_embeddings.h5"

    # Layer 16 of 28 — matched to GR00T's select_layer, not this model's own
    # last layer. Intermediate, so the final-norm correction applies.
    vlm_extract_layers: tuple = (16,)
