"""
configs/aloha/exp05_groot_transfer.py
─────────────────────────────────────
ALOHA transfer-cube — GR00T N1.7's VLM subtree as the frozen backbone.

Why ALOHA at all
────────────────
Chapter 2's LIBERO result is that robot pretraining of the VLM buys nothing
detectable once the observation spec is complete (GR00T 91.5% vs stock Qwen3-VL
89.0%, p = 0.40). Its weakest axis is that it rests on ONE benchmark. This pair
tests the same claim on a genuinely different problem:

    embodiment    single-arm Panda        ->  bimanual ViperX, 14 DOF
    action space  7-D OSC deltas          ->  14-D absolute joint targets
    task          pick-place / articulated ->  right->left cube handover
    cameras       2 (agentview + wrist)   ->  1 top, which is the COMPLETE spec
    language      10 instructions          ->  a single fixed instruction

Bimanual manipulation is squarely inside GR00T's pretraining distribution, so
this is a venue biased TOWARD finding a pretraining effect. A null here is
therefore much harder to dismiss than a null on LIBERO.

*Prediction:* GR00T ~= Qwen3-VL (exp06), i.e. within the ~9.4 pt detectable
margin at n=200.
*Falsifier:* GR00T leads by more than that -> pretraining does pay off on an
embodiment inside its training distribution, and the LIBERO null was
task-specific.

Why transfer-cube and not insertion
───────────────────────────────────
ACT — a fully trained specialised policy — reaches ~87.6% on this dataset via
LeRobot but only ~20% on sim insertion with human demos. Scaling by our stack's
ratio to specialised policies puts insertion at ~10-15%, where a between-arm
difference of the size we can detect would require pretraining to roughly double
the success rate. Insertion is a floor experiment; transfer-cube has an operating
range in both directions.

Geometry, measured not assumed
──────────────────────────────
The 480x640 top camera is resized to 320x240, preserving the native 4:3 rather
than distorting to square. The Qwen3-VL processor then reports
image_grid_thw = (1, 12, 18), i.e. a merged 6x9 = 54-token grid, and 77 total
tokens with this task string. img_seq_len = 96 leaves headroom; precompute
zero-pads into that buffer and encode_vlm pads identically at eval.

One camera is the complete spec here: BOTH the dataset (`meta/info.json` lists
only observation.images.top) and the live env (`gym_aloha` exposes only `top`)
have a single view, so unlike LIBERO there is no missing-camera asymmetry to
worry about.

Harness note
────────────
scripts/validate_harness_aloha.py confirms the action plumbing: replaying demo
actions tracks the recorded joint trajectory to 0.0011 rad, 16x tighter than the
demos' own action->state lag. It does NOT validate success detection, because
AlohaEnv samples the cube pose from a seed into a module-level global and this
dataset records neither pose nor seed — so a demo's initial scene cannot be
reconstructed. The external anchor is ACT at 87.6%.

Requires MUJOCO_GL=glfw at eval (dm_control), NOT wgl (that is LIBERO/robosuite).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.aloha.exp01_baseline import AlohaExp01


@dataclass
class AlohaExp05GrootTransfer(AlohaExp01):
    vlm_backbone: str = "groot"
    output_dir: str = "asset/runs/aloha/exp05_groot_transfer"
    embeddings_cache: str = "asset/runs/aloha/exp05_groot_transfer/vlm_embeddings.h5"

    # GR00T's documented read: select_layer=16 of its 16-layer stack (its last,
    # so already post-final-norm). exp06 reads the same layer of Qwen3-VL's 28,
    # where it is intermediate and encode_vlm's norm correction fires.
    vlm_extract_layers: tuple = (16,)
    vlm_hidden_size: int = 2048

    # 54 image tokens on a 6x9 merged grid + chat template + instruction = 77.
    img_seq_len: int = 96
    img_grid_h:  int = 6
    img_grid_w:  int = 9

    # n=50 gives SE 6.7 pts — too weak to resolve anything under ~19 points.
    # 200 matches the LIBERO protocol and brings SE to ~3.4 pts.
    sim_episodes: int = 200

    embed_batch_size: int = 32
