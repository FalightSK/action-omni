"""
configs/libero/exp01_baseline.py
────────────────────────────────
Experiment 01 — LIBERO-Spatial, DiT decoder, single-layer (28) rank-16 LoRA.

Same validated architecture as PushT Exp02a (56% SR) carried to a real manipulation
benchmark. Frozen Qwen3.5-0.8B backbone; only the ~16M adapter + DiT flow decoder train.

Dataset: HuggingFaceVLA/libero (lerobot v3), libero_spatial suite — 432 episodes /
52,970 frames, single 256x256 agentview camera, 8-dim proprio state, 7-dim delta-EEF
actions in [-1,1], 10 language-instructed tasks, 10 fps control.

Measured (do not guess — see the port notes):
- Qwen 256x256 -> image_grid_thw [1,16,16] -> 2x2 merge -> 8x8 = 64 image tokens.
- Longest of the 10 instructions -> total seq <= 90 tokens; img_seq_len=100 leaves margin
  and stays under the SpatialAwareMLP text-PE ceiling (128).

HARD CONSTRAINTS (same discipline as PushT):
- inference_horizon=4: baked in at training, cannot change at eval.
- state = the raw 8-dim env proprio, NOT augmented (no PushT-style prev-delta hack).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig

REPO_ID = "HuggingFaceVLA/libero"

# Canonical libero_spatial task strings (the HuggingFaceVLA/libero metadata mixes all
# four suites with no suite tag; suite membership lives here as ground truth).
LIBERO_SPATIAL_TASKS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]


def _lerobot_root() -> str:
    """Resolve the local lerobot dataset cache (honours the project's on-E: HF_HOME)."""
    root = os.environ.get("HF_LEROBOT_HOME")
    if not root:
        hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        root = os.path.join(hf, "lerobot")
    return os.path.join(root, "HuggingFaceVLA", "libero")


@dataclass
class LiberoExp01(BaseVLAConfig):
    # ── Paths ─────────────────────────────────────────────────────────────────
    model_path:       str = "Qwen/Qwen3.5-0.8B"   # HF id; resolves from HF_HOME cache
    output_dir:       str = "asset/runs/libero/exp01_baseline"
    embeddings_cache: str = "asset/runs/libero/exp01_baseline/vlm_embeddings.h5"  # .h5 => streamed (52,970 frames)
    dataset_path:     str = ""   # filled in __post_init__ (lerobot cache dir)

    # ── Task (fallback only; the loader/agent supply per-task instructions) ────
    task_text: str = "pick up the black bowl and place it on the plate"

    # ── Dimensions ────────────────────────────────────────────────────────────
    state_dim:  int = 8    # [eef_pos(3), quat->axis-angle(3), gripper_qpos(2)]
    action_dim: int = 7    # delta-EEF (6) + gripper (1), already in [-1,1]

    # ── Horizons ──────────────────────────────────────────────────────────────
    action_horizon:    int = 16
    inference_horizon: int = 4    # HARD CONSTRAINT: set at training, cannot change

    use_relative_actions: bool = False   # LIBERO actions ARE the env-step delta; command directly

    # ── Normalisation (loaded from the dataset's meta/stats.json in __post_init__)
    action_mean: tuple = (0.0,) * 7
    action_std:  tuple = (1.0,) * 7
    state_mean:  tuple = (0.0,) * 8
    state_std:   tuple = (1.0,) * 8

    # ── VLM token config (MEASURED for 256x256 Qwen3.5-0.8B; do not reuse PushT) ─
    img_seq_len: int = 100   # 64 image + <=26 text/template + margin; < 128 PE ceiling
    img_grid_h:  int = 8     # 16x16 patches / 2x2 merge = 8x8 = 64 image tokens
    img_grid_w:  int = 8
    vlm_extract_layers: tuple = (28,)   # single layer -> last_hidden_state (matches exp02a)

    # ── Architecture (identical to PushT Exp02a, the validated 56% config) ─────
    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_episodes:  int = 50    # 10 tasks x 5 eps for a fast pass; raise to 100 for headline
    sim_max_steps: int = 300
    video_fps:     int = 10

    # ── LIBERO-specific fields (live on the subclass, not the base) ────────────
    camera_key: str = "observation.images.image"   # agentview (image2 = wrist, unused)
    image_size: int = 256

    def __post_init__(self):
        if not self.dataset_path:
            self.dataset_path = _lerobot_root()
        stats_path = os.path.join(self.dataset_path, "meta", "stats.json")
        if os.path.exists(stats_path):
            s = json.load(open(stats_path))
            self.action_mean = tuple(s["action"]["mean"])
            self.action_std = tuple(s["action"]["std"])
            self.state_mean = tuple(s["observation.state"]["mean"])
            self.state_std = tuple(s["observation.state"]["std"])
        else:
            print(f"[LiberoExp01] WARN: stats.json not at {stats_path}; using unit normalisation")
