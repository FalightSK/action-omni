"""
configs/aloha/exp06_qwen3vl_transfer.py
───────────────────────────────────────
ALOHA transfer-cube — stock Qwen3-VL-2B as the frozen backbone.

The control for exp05, and GR00T's own ancestry root:

    Qwen3-VL-2B  ->  Cosmos-Reason2-2B  ->  GR00T N1.7

Chapter 1 verified both hops are real finetunes (584/625 and 476/493 tensors
differ), so comparing this arm against exp05 spans the whole robot-pretraining
treatment rather than one weak hop.

Everything except the backbone weights is inherited from exp05: same single top
camera, same 96-token budget on a 6x9 grid, same layer-16 read, same 14-D
bimanual action space, same head, same schedule, same 200-episode protocol.

Read depth is deliberately NOT this model's own last layer. Qwen3-VL-2B has 28
layers but is read at 16 to match GR00T's select_layer; reading each arm at its
own final layer would compare layer 16 against layer 28 and attribute a depth
difference to pretraining. Because 16 is INTERMEDIATE here but FINAL for GR00T,
models/vla.py::encode_vlm applies the language stack's final RMSNorm to this
arm's read — without it the pair would differ by normalisation rather than by
weights, the defect that made Chapter 1's provably-identical SmolVLA pair read
cosine 0.22 instead of 1.0.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.aloha.exp05_groot_transfer import AlohaExp05GrootTransfer


@dataclass
class AlohaExp06Qwen3VLTransfer(AlohaExp05GrootTransfer):
    vlm_backbone: str = "qwen3vl"
    output_dir: str = "asset/runs/aloha/exp06_qwen3vl_transfer"
    embeddings_cache: str = "asset/runs/aloha/exp06_qwen3vl_transfer/vlm_embeddings.h5"

    # Layer 16 of 28 — matched to GR00T's select_layer, intermediate here.
    vlm_extract_layers: tuple = (16,)
