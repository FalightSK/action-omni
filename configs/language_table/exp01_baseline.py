"""
configs/language_table/exp01_baseline.py
────────────────────────────────────────
Language Table Exp01 — first LANGUAGE-CONDITIONED VLA experiment.

Carries the validated frozen-Qwen + LoRA-adapter + DiT flow-matching architecture
to Google's Language Table 2D block-pushing benchmark, where the instruction
VARIES PER EPISODE (unlike PushT/ALOHA's single fixed task_text).  This is the
project's first per-sample-instruction task and its first new-command / OOD
generalization study.

Dataset : google `language_table_sim` (RLDS/TFDS), converted to local parquet by
          scripts/convert_language_table.py.  A bounded subset (~5k episodes) with
          the `separate` and `point` command types HELD OUT for the OOD eval.
Sim eval : language_table PyBullet sim (gym 0.23 API), constructed directly with a
           chosen reward_factory:
             in-dist  → block2block.BlockToBlockReward
             OOD      → separate_blocks.SeparateBlocksReward (held-out verb)
                        point2block.PointToBlockReward       (held-out verb)

What changes vs ALOHA Exp01:
  • state_dim / action_dim : 14 → 2  (effector xy state; 2D ee-delta action)
  • action representation   : the stored action IS the env-step delta, so the agent
                              commands predictions DIRECTLY (denorm, no integration,
                              like ALOHA) — use_relative_actions=False.
  • task_text               : NOT fixed — the per-episode instruction string is fed
                              to the VLM (dataset loader supplies it per sample; the
                              eval agent reads the live env instruction).
  • image                   : 320×180 (16:9) → Qwen emits 66 image tokens on a 6×11
                              merged grid (grid_thw=[1,12,22]); img_seq_len=120 leaves
                              margin for the longest instruction (measured max ~90).
  • normalisation           : 2-D action/state mean/std loaded from the converted
                              dataset's meta/stats.json at construction.

Architecture is otherwise identical to the validated DiT path (rank-16 LoRA,
adapter_dim 256/512, DiT 256-hidden 6-layer).
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


def _default_dataset_root() -> str:
    # Allow an env override so the pipeline smoke test can point at a small subset
    # (e.g. language_table_sim_smoke50) without editing the config.
    env = os.environ.get("LT_DATASET_ROOT")
    if env:
        return env
    return str(Path(__file__).parents[2] / "asset" / "data" / "language_table_sim")


@dataclass
class LTExp01(BaseVLAConfig):
    # ── Identity / paths ────────────────────────────────────────────────────────
    repo_id:          str = "google/language_table_sim"
    dataset_root:     str = ""                       # filled in __post_init__
    dataset_path:     str = ""                       # alias kept for base-class API
    model_path:       str = "Qwen/Qwen3.5-0.8B"      # resolved from HF cache (offline ok)
    output_dir:       str = "asset/runs/language_table/exp01_baseline"
    # HDF5 (streamed, low-RAM) cache — the 10% subset is far too large to hold the
    # full embedding tensor in RAM for a single torch.save(.pt). precompute.py and
    # train.py auto-detect the .h5 extension and stream/lazy-read accordingly.
    embeddings_cache: str = "asset/runs/language_table/exp01_baseline/vlm_embeddings.h5"

    # ── Task (per-episode instruction; this is only a fallback placeholder) ──────
    task_text: str = "push the blocks as instructed"

    # ── Sim env (constructed directly, NOT via gym.make) ─────────────────────────
    lt_block_mode:   str = "BLOCK_8"           # blocks.LanguageTableBlockVariants.BLOCK_8
    lt_reward_indist: str = "block2block"      # default in-distribution reward factory
    lt_reward_ood:   tuple = ("separate", "point")   # held-out OOD command types

    # ── Dimensions (2-D effector) ────────────────────────────────────────────────
    state_dim:  int = 2     # effector_translation (x, y)
    action_dim: int = 2     # 2D ee-delta setpoint

    # ── Horizons ──────────────────────────────────────────────────────────────
    action_horizon:    int = 16   # predicted chunk (matches validated decoder)
    inference_horizon: int = 4    # replan often — contact-rich pushing
    use_relative_actions: bool = False   # stored action == env-step delta, commanded directly

    # ── Image tokenisation (320×180 → Qwen 6×11 merged grid = 66 img tokens) ─────
    lt_img_w: int = 320
    lt_img_h: int = 180
    img_seq_len: int = 120   # 66 image + up to ~24 text/template tokens (+ margin)
    img_grid_h:  int = 6
    img_grid_w:  int = 11
    vlm_extract_layers: tuple = (28,)   # single layer → uses last_hidden_state

    # ── Architecture (identical to validated DiT path) ───────────────────────────
    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    # ── Training ────────────────────────────────────────────────────────────────
    batch_size:       int = 256
    num_epochs:       int = 150
    embed_batch_size: int = 16    # precompute throughput (Qwen-0.8B fits 12 GB at bs16)

    # ── Simulation eval ───────────────────────────────────────────────────────────
    sim_episodes:  int = 50
    sim_max_steps: int = 200   # generous cap for one push (data eps ≤ 96 steps)

    # ──────────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        if not self.dataset_root:
            self.dataset_root = _default_dataset_root()
        if not self.dataset_path:
            self.dataset_path = self.dataset_root

        # Load exact 2-D normalisation stats from the converted dataset metadata.
        stats_path = Path(self.dataset_root) / "meta" / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            self.action_mean = tuple(float(x) for x in stats["action"]["mean"])
            self.action_std  = tuple(float(x) for x in stats["action"]["std"])
            self.state_mean  = tuple(float(x) for x in stats["observation.state"]["mean"])
            self.state_std   = tuple(float(x) for x in stats["observation.state"]["std"])
        else:
            print(f"[LTExp01][WARN] stats.json not found at {stats_path}; "
                  f"using unit normalisation. Precompute/eval will be miscalibrated.")
            self.action_mean = tuple([0.0] * self.action_dim)
            self.action_std  = tuple([1.0] * self.action_dim)
            self.state_mean  = tuple([0.0] * self.state_dim)
            self.state_std   = tuple([1.0] * self.state_dim)
