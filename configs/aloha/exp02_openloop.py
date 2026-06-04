"""
configs/aloha/exp02_openloop.py
───────────────────────────────
ALOHA Exp02 — causal test of the closed-loop hypothesis.

Identical to Exp01 (frozen Qwen + adapter + DiT) EXCEPT the action chunk is long
and executed fully open-loop: action_horizon = inference_horizon = 50, matching
SmolVLA's 50-step (1.0 s) chunk. Exp01 used 16/8 (replan every 0.16 s) -> 66% SR.

Question: does the SAME architecture collapse when forced to act open-loop like
SmolVLA? If yes, the closed-loop replan rate is a primary cause of the
Exp01 (66%) vs SmolVLA (0%) gap; if Exp02 still works, SmolVLA's failure is more
about data/cameras than the chunk length.

Needs its own embedding cache (the cached action chunks are length-50).
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.aloha.exp01_baseline import AlohaExp01


@dataclass
class AlohaExp02OpenLoop(AlohaExp01):
    output_dir:       str = "asset/runs/aloha/exp02_openloop"
    embeddings_cache: str = "asset/runs/aloha/exp02_openloop/vlm_embeddings.pt"

    # the only architectural change: long, fully open-loop action chunk
    action_horizon:    int = 50
    inference_horizon: int = 50

    # cached-embedding training barely uses the GPU -> use a bigger batch (Exp01 used 256)
    batch_size: int = 1024
