"""
configs/libero/exp02_paligemma.py
─────────────────────────────────
LIBERO-Goal Exp02 — stock PaliGemma-3B as the frozen backbone.

The control for exp01. Identical in every respect except the backbone weights:
same DiT head, same data, same read layer (18 of 18), same token budget, same
chunking, same 50 evaluation initial states. Pi-0.5 IS a finetuned PaliGemma-3B,
so this pair differs only in whether the VLM ever saw robot data.

What it tests (H1 in CHAPTER2_PLAN.md)
──────────────────────────────────────
Chapter 1 measured, offline, that robot pretraining changed Pi-0.5's
representation substantially — task/scene ratio +0.224 against this exact
checkpoint, RSA 0.818 (the lowest within-pair value in the study), and
instruction eta^2 +0.523. If those differences matter for control, Pi-0.5 should
beat this arm on success rate. If they do not, robot pretraining of the VLM is
not necessary for LIBERO-Goal, and Chapter 1's largest measured effect is
availability without necessity.

The prediction is falsifiable in both directions:
    SR(paligemma) >= 0.9 * SR(pi05)   -> pretraining not necessary
    Pi-0.5 leads by a wide margin     -> the probes understated its value

Why this is the clean comparison
────────────────────────────────
It holds the adapter fixed and varies only the backbone. It does NOT reproduce
how Pi-0.5's own action expert consumes its backbone (per-layer KV at every
layer), so it answers "is a robot-pretrained backbone worth more to OUR head"
rather than "is Pi-0.5 a better policy". The latter would confound backbone,
expert, recipe and hyperparameters at once.

One asymmetry worth stating: our head cross-attends to the full 272-token
sequence, so it reads the instruction tokens directly regardless of what the
backbone did with them. A null result here therefore means "the backbone's robot
pretraining adds nothing our adapter cannot supply", not "the instruction is
unused" — the gate already showed the instruction is load-bearing (33% -> 0%).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.libero.exp01_goal import LiberoExp01Goal


@dataclass
class LiberoExp02PaliGemma(LiberoExp01Goal):
    # Stock PaliGemma-3B — the checkpoint Pi-0.5 was initialised from.
    vlm_backbone: str = "paligemma"
    output_dir: str = "asset/runs/libero/exp02_paligemma"
    embeddings_cache: str = "asset/runs/libero/exp02_paligemma/vlm_embeddings.h5"
    # Everything else is inherited unchanged, deliberately: same 18-layer read,
    # same img_seq_len=272, same 16/8 chunking, same batch size and schedule.
    # Any divergence here would confound the backbone comparison with a training
    # difference.
