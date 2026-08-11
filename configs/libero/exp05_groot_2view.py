"""
configs/libero/exp05_groot_2view.py
───────────────────────────────────
LIBERO-Goal Exp05 — GR00T N1.7 with BOTH cameras.

Why this run exists
───────────────────
exp01-exp04 all read `agentview` only. That is not merely less information: it
is an asymmetry that runs in the direction of our headline result. GR00T and
Pi-0.5 were adapted on multi-view robot data, so a single exterior view puts
them OUTSIDE their trained observation spec, while PaliGemma and Qwen3-VL — which
never saw robot multi-view — are inside theirs. Every published LIBERO pipeline
feeds both views: the demos record `['robot0_eye_in_hand', 'agentview']`, and
LeRobot's conversion exposes them as `image` + `wrist_image`.

So the single-view exp03/exp04 comparison cannot cleanly test whether robot
pretraining helps; it partly tests which backbone tolerates a missing camera.
exp05/exp06 repeat that pair with the benchmark's own observation spec, making
the comparison a real hypothesis test rather than a confounded null.

This pair, not the PaliGemma pair, because GR00T's documented read is a single
hidden state at `select_layer=16` feeding a flow-matching action head — which is
what we actually built — so the "you did not reproduce their architecture"
objection does not apply here.

Geometry, measured not assumed
──────────────────────────────
The Qwen3-VL processor emits 64 image tokens per view at 224x224, so two views
give 128 image tokens and a total sequence of 141-146 across the ten canonical
instructions. img_seq_len = 152 leaves headroom for the longer held-out
paraphrase tiers; precompute zero-pads into that fixed buffer and encode_vlm
pads identically at evaluation time.

The adapter's 2-D positional encoding is tiled per view and offset by a learned
view embedding (models/vla.py::SpatialAwareMLP). Without that, the running image
index would exceed the 64-row grid table and every wrist token would clamp onto
the last row of the exterior grid. With n_views=1 that code path builds nothing
and leaves exp01-exp04 bit-identical, so the earlier runs remain reproducible.

What it does NOT fix
────────────────────
Still one suite (LIBERO-Goal's shared scene inflates language effects), still a
frozen backbone and a 19.2M head, still n=2 in the fair comparison. The
between-pair token-geometry confound (272 vs 152) also remains, so cross-family
claims stay off the table.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.libero.exp03_groot import LiberoExp03Groot


@dataclass
class LiberoExp05Groot2View(LiberoExp03Groot):
    output_dir: str = "asset/runs/libero/exp05_groot_2view"
    embeddings_cache: str = "asset/runs/libero/exp05_groot_2view/vlm_embeddings.h5"

    # agentview + robot0_eye_in_hand, in that order everywhere: the dataset
    # builds it, the eval agent rebuilds it, and the view embedding now makes
    # the order load-bearing.
    n_views: int = 2

    # 128 image tokens (2 x 64) + chat template + instruction; 141-146 measured.
    img_seq_len: int = 152

    # 152 tokens against exp03's 88 — a ~1.7x longer sequence through the DiT
    # cross-attention, so the precompute batch comes down to keep peak VRAM in
    # the same band that measured 70 samples/s for exp03.
    embed_batch_size: int = 24
