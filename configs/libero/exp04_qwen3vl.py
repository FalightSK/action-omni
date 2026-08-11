"""
configs/libero/exp04_qwen3vl.py
───────────────────────────────
LIBERO-Goal Exp04 — stock Qwen3-VL-2B as the frozen backbone.

The control for exp03, and the root of GR00T's ancestry chain:

    Qwen3-VL-2B  ->  Cosmos-Reason2-2B  ->  GR00T N1.7

Chapter 1 verified both hops are real finetunes, not reuse: Cosmos differs from
Qwen3-VL in 584/625 tensors, and GR00T differs from Cosmos in 476/493. Comparing
exp03 against this arm therefore spans BOTH hops — the physics/video pretraining
and the robot pretraining — which is what makes it worth running. The single hop
GR00T<-Cosmos is too weak to test (RSA 0.936, delta eta^2 -0.020); there would be
almost nothing to detect.

Read depth is deliberately NOT this model's own last layer
──────────────────────────────────────────────────────────
Qwen3-VL-2B has 28 layers, but it is read at layer 16 to match GR00T's
select_layer=16. Reading each arm at its own final layer is the exact confound
this study exists to remove — it would compare GR00T's layer 16 against
Qwen3-VL's layer 28 and attribute a depth difference to pretraining.

Because layer 16 is INTERMEDIATE for this model but FINAL for GR00T, the two
arms would otherwise come back on different scales: HF applies the language
stack's final RMSNorm only to the last hidden state. models/vla.py's encode_vlm
applies that norm to intermediate reads for exactly this reason, so this arm is
post-norm like GR00T rather than pre-norm. Without that correction the pair
would differ by normalisation rather than by weights — the same defect that made
Chapter 1's provably-identical SmolVLA pair read cosine 0.22 instead of 1.0.

Everything else is inherited from exp03's geometry unchanged: same 88-token
budget, same 8x8 image grid, same 2048 width, same head, same schedule.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.libero.exp03_groot import LiberoExp03Groot


@dataclass
class LiberoExp04Qwen3VL(LiberoExp03Groot):
    vlm_backbone: str = "qwen3vl"
    output_dir: str = "asset/runs/libero/exp04_qwen3vl"
    embeddings_cache: str = "asset/runs/libero/exp04_qwen3vl/vlm_embeddings.h5"

    # Layer 16 of 28 — matched to GR00T's select_layer, not this model's own
    # last layer. Intermediate, so encode_vlm's final-norm correction fires.
    vlm_extract_layers: tuple = (16,)
