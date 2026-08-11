"""
configs/aloha/exp03_insertion.py
────────────────────────────────
ALOHA Exp03 — bimanual 14-DOF INSERTION task (peg-into-socket).

Second ALOHA manipulation task after transfer-cube (exp01). Same validated
frozen-VLM + LoRA-adapter + DiT flow-matching architecture; only the dataset,
sim env, and task instruction change. Insertion is contact-rich and requires
precise peg/socket alignment — a harder placement task than transfer-cube, so it
is a useful second data point for how far frozen-VLM control precision extends.

Dataset : lerobot/aloha_sim_insertion_human (50 episodes, 25k frames, 50 fps)
Sim eval : gym_aloha/AlohaInsertion-v0 (max_reward 4; success = reward >= 4)

Layout is identical to transfer-cube (lerobot v3.0): 4 parquet shards
(data/chunk-000/file-*.parquet) + a single top-camera mp4 with all 25k frames in
order, so data/aloha/dataset.py loads it unchanged. Per-joint 14-D normalisation
stats are read from meta/stats.json at construction.

Everything else (state_dim/action_dim 14, absolute joint targets, 320×240 → 8×10
= 80 image tokens, seq_len 103, rank-16 LoRA, DiT hidden 256) mirrors exp01.

HARD CONSTRAINTS:
  • state = environment observation (joint positions) only — never action history.
  • inference_horizon at eval == value set here.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


def _default_dataset_root() -> str:
    return str(Path(__file__).parents[2] / "asset" / "data" / "aloha_sim_insertion_human")


@dataclass
class AlohaExp03Insertion(BaseVLAConfig):
    # ── Identity / paths ────────────────────────────────────────────────────────
    repo_id:          str = "lerobot/aloha_sim_insertion_human"
    dataset_root:     str = ""                       # filled in __post_init__
    dataset_path:     str = ""                       # alias kept for base-class API
    model_path:       str = "Qwen/Qwen3.5-0.8B"      # resolved from HF cache (offline ok)
    output_dir:       str = "asset/runs/aloha/exp03_insertion"
    embeddings_cache: str = "asset/runs/aloha/exp03_insertion/vlm_embeddings.h5"

    # ── Task (canonical instruction from the dataset's meta/tasks.parquet) ───────
    task_text: str = "Insert the peg into the socket."

    # ── Sim env ─────────────────────────────────────────────────────────────────
    env_id:     str = "gym_aloha/AlohaInsertion-v0"
    camera_key: str = "observation.images.top"

    # ── Dimensions (14-DOF bimanual) ─────────────────────────────────────────────
    state_dim:  int = 14    # joint positions (env observation only)
    action_dim: int = 14    # absolute joint targets

    # ── Horizons ──────────────────────────────────────────────────────────────
    action_horizon:    int = 16   # predicted chunk (matches exp01 decoder)
    inference_horizon: int = 8    # execute before replan (absolute ⇒ no drift, ok > 4)
    use_relative_actions: bool = False   # absolute joint targets

    # ── Image tokenisation (320×240 → Qwen 8×10 merged grid = 80 img tokens) ─────
    aloha_img_h: int = 240
    aloha_img_w: int = 320
    img_seq_len: int = 103   # 80 image + text tokens ("Insert the peg into the socket." fits)
    img_grid_h:  int = 8
    img_grid_w:  int = 10
    vlm_extract_layers: tuple = (24,)   # single layer → uses last_hidden_state

    # ── Architecture (identical to exp01) ────────────────────────────────────────
    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    # ── Training ────────────────────────────────────────────────────────────────
    batch_size:  int = 256
    num_epochs:  int = 300

    # ── Simulation eval ───────────────────────────────────────────────────────────
    sim_episodes:  int = 50
    sim_max_steps: int = 400   # demos are 400+ frames; override registered 300-step cap

    # ──────────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        if not self.dataset_root:
            self.dataset_root = _default_dataset_root()
        if not self.dataset_path:
            self.dataset_path = self.dataset_root

        # Load exact per-joint normalisation stats from the dataset metadata.
        stats_path = Path(self.dataset_root) / "meta" / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            self.action_mean = tuple(float(x) for x in stats["action"]["mean"])
            self.action_std  = tuple(float(x) for x in stats["action"]["std"])
            self.state_mean  = tuple(float(x) for x in stats["observation.state"]["mean"])
            self.state_std   = tuple(float(x) for x in stats["observation.state"]["std"])
        else:
            print(f"[AlohaExp03][WARN] stats.json not found at {stats_path}; "
                  f"using unit normalisation. Precompute/eval will be miscalibrated.")
            self.action_mean = tuple([0.0] * self.action_dim)
            self.action_std  = tuple([1.0] * self.action_dim)
            self.state_mean  = tuple([0.0] * self.state_dim)
            self.state_std   = tuple([1.0] * self.state_dim)
