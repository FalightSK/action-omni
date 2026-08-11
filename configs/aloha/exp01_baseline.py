"""
configs/aloha/exp01_baseline.py
───────────────────────────────
ALOHA Exp01 — baseline carry-over of the best PushT architecture (Exp02a) to the
bimanual 14-DOF transfer-cube task.  Phase-2 gate: does the frozen-VLM + adapter +
DiT pipeline scale to physical complexity?

Dataset : lerobot/aloha_sim_transfer_cube_human (50 episodes, 20k frames, 50 fps)
Sim eval : gym_aloha/AlohaTransferCube-v0 (success = reward 4)

What changes vs PushT Exp02a:
  • action_dim / state_dim : 2 → 14  (6 arm joints + 1 normalised gripper, per arm)
  • actions                : ABSOLUTE joint targets (NOT relative deltas).  ALOHA
                             action == env step() space, so predictions pass straight
                             through with no accumulation/drift.  This is why the
                             agent does not integrate deltas like the PushT agent.
  • image                  : 480×640 top camera, resized to 320×240 → Qwen emits
                             80 image tokens on an 8×10 grid (seq_len 103).  This
                             keeps the token budget close to PushT's 82 so the DiT
                             cross-attention load matches the validated architecture.
  • normalisation          : per-joint mean/std loaded from the dataset's
                             meta/stats.json at construction (14-D, exact).

Architecture is otherwise identical to Exp02a (DiT, rank-16 LoRA, hidden 256).
LoRA rank-64 is deferred to a later ablation (ALOHA-B).

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
    return str(Path(__file__).parents[2] / "asset" / "data" / "aloha_sim_transfer_cube_human")


@dataclass
class AlohaExp01(BaseVLAConfig):
    # ── Identity / paths ────────────────────────────────────────────────────────
    repo_id:          str = "lerobot/aloha_sim_transfer_cube_human"
    dataset_root:     str = ""                       # filled in __post_init__
    dataset_path:     str = ""                       # alias kept for base-class API
    model_path:       str = "Qwen/Qwen3.5-0.8B"      # resolved from HF cache (offline ok)
    output_dir:       str = "asset/runs/aloha/exp01_baseline"
    embeddings_cache: str = "asset/runs/aloha/exp01_baseline/vlm_embeddings.h5"

    # ── Task ──────────────────────────────────────────────────────────────────
    task_text: str = "Pick up the cube with the right arm and transfer it to the left arm."

    # ── Sim env ─────────────────────────────────────────────────────────────────
    env_id:     str = "gym_aloha/AlohaTransferCube-v0"
    camera_key: str = "observation.images.top"

    # ── Dimensions (14-DOF bimanual) ─────────────────────────────────────────────
    state_dim:  int = 14    # joint positions (env observation only)
    action_dim: int = 14    # absolute joint targets

    # ── Horizons ──────────────────────────────────────────────────────────────
    action_horizon:    int = 16   # predicted chunk (matches Exp02a decoder)
    inference_horizon: int = 8    # execute before replan (absolute ⇒ no drift, ok > 4)
    use_relative_actions: bool = False   # absolute joint targets

    # ── Image tokenisation (320×240 → Qwen 8×10 merged grid = 80 img tokens) ─────
    aloha_img_h: int = 240
    aloha_img_w: int = 320
    img_seq_len: int = 103   # 80 image + 23 text tokens (deterministic for fixed task)
    img_grid_h:  int = 8
    img_grid_w:  int = 10
    vlm_extract_layers: tuple = (24,)   # single layer → uses last_hidden_state

    # ── Architecture (identical to Exp02a) ───────────────────────────────────────
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
    sim_max_steps: int = 400   # demos are 400 frames; override registered 300-step cap

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
            print(f"[AlohaExp01][WARN] stats.json not found at {stats_path}; "
                  f"using unit normalisation. Precompute/eval will be miscalibrated.")
            self.action_mean = tuple([0.0] * self.action_dim)
            self.action_std  = tuple([1.0] * self.action_dim)
            self.state_mean  = tuple([0.0] * self.state_dim)
            self.state_std   = tuple([1.0] * self.state_dim)
