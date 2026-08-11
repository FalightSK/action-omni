"""
configs/libero/exp03_groot.py
─────────────────────────────
LIBERO-Goal Exp03 — GR00T N1.7's VLM subtree as the frozen backbone.

Why this pair exists
────────────────────
Exp01/exp02 (Pi-0.5 vs stock PaliGemma) answered H1, but with a confound that a
reviewer will find: our DiT head cross-attends to the text tokens directly, so
Pi-0.5's distinctive property — language fused INTO its image tokens — is made
redundant by our architecture. The head diagnostics measured exactly that:
Pi-0.5's head sends 4.2% of its cross-attention mass to the text tokens and is
unaffected when they are zeroed (1.02x loss), while PaliGemma's sends 23.8% and
nearly triples its loss without them (2.76x).

GR00T vs Qwen3-VL removes that confound. Chapter 1's text-visibility test found
GR00T, Cosmos and Qwen3-VL ALL bit-identical under any instruction change
(0.0000) — their image tokens are produced before the instruction is attended
to. Neither arm has backbone fusion for our head to render redundant, so the
head must supply it for BOTH, symmetrically.

Two further reasons this pair is the cleaner test:
  * GR00T genuinely reads ONE hidden state at a selected layer
    (groot_n17_3b/config.json select_layer=16) and feeds it to a flow-matching
    action head. That is our architecture. The "you did not use their
    architecture" objection applies to Pi-0.5's KV-consuming expert, not here.
  * GR00T does not freeze its backbone (476/493 tensors differ from Cosmos,
    tune_llm=True), so the robot-pretraining treatment is real, not nominal.

Prediction under the mechanism the diagnostics support
──────────────────────────────────────────────────────
If the Pi-0.5 result is caused by ROUTING — pretraining moving the instruction
out of a stable dedicated channel and into the drift-prone visual one — then
with no fused channel available on either arm, both heads take the text-token
route and the SR gap should be small, with both landing near or above
PaliGemma's 42%.

Falsifier: GR00T beats Qwen3-VL by a wide margin anyway, which would mean robot
pretraining helps for reasons unrelated to language routing.

Geometry
────────
Qwen3-VL-family processors emit 64 image tokens at 224x224 plus 11-16 chat
template and instruction tokens — measured 75-80 total across the ten canonical
LIBERO-Goal instructions. img_seq_len is set to 88 to leave headroom for the
held-out paraphrase tiers, which are longer; precompute zero-pads into that
fixed buffer, so a batch shorter than 88 is written consistently.

Cache is ~23 GB against PaliGemma's 71 GB, because 64 image tokens replace 256.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.libero.exp01_goal import LiberoExp01Goal


@dataclass
class LiberoExp03Groot(LiberoExp01Goal):
    vlm_backbone: str = "groot"
    output_dir: str = "asset/runs/libero/exp03_groot"
    embeddings_cache: str = "asset/runs/libero/exp03_groot/vlm_embeddings.h5"

    # 64 image tokens (8x8) + chat template + instruction; 75-80 measured, 88
    # gives headroom for the longer held-out paraphrases.
    img_seq_len: int = 88
    img_grid_h:  int = 8
    img_grid_w:  int = 8
    vlm_hidden_size: int = 2048

    # GR00T's documented read: select_layer=16 of its 16-layer stack. That is
    # its LAST layer, so hidden_states[16] is already post-final-norm and
    # encode_vlm's norm correction correctly does not fire for this arm.
    vlm_extract_layers: tuple = (16,)

    # 64 image tokens instead of 256 makes the forward far cheaper than the
    # PaliGemma arms, so the precompute batch can rise well above their 8.
    embed_batch_size: int = 32
