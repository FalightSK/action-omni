"""
configs/language_table/exp03_curated.py
───────────────────────────────────────
Language Table Exp03 — CURATED validation subset (8 concepts, 50% reach + 50%
block-to-block), built by scripts/_lt_make_curated_subset.py.

Fixes the two failure modes of exp02_5instr:
  • phrasing mismatch — concepts pool ALL phrasings (push/move/slide/put ·
    to/into/towards/close to · point/touch/move-your-arm), so the sim's eval
    paraphrases are in-distribution.
  • control-data starvation — 822 episodes / ~18k frames (~4x exp02), ~100 eps per
    concept, balanced 50/50 so the easy reach task and the hard push task are both
    well-covered.

Eval uses TWO reward factories: `point` for the 4 reach targets, `block2block` for
the 4 push pairs (see scripts/_lt_eval_curated.py). Architecture identical to exp01.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


def _default_dataset_root() -> str:
    env = os.environ.get("LT_DATASET_ROOT")
    if env:
        return env
    return str(Path(__file__).parents[2] / "asset" / "data" / "language_table_sim_curated")


@dataclass
class LTExp03Curated(BaseVLAConfig):
    repo_id:          str = "google/language_table_sim"
    dataset_root:     str = ""
    dataset_path:     str = ""
    model_path:       str = "Qwen/Qwen3.5-0.8B"
    output_dir:       str = "asset/runs/language_table/exp03_curated"
    embeddings_cache: str = "asset/runs/language_table/exp03_curated/vlm_embeddings.h5"

    task_text: str = "push the blocks as instructed"

    lt_block_mode:    str = "BLOCK_8"
    lt_reward_indist: str = "block2block"
    lt_reward_ood:    tuple = ()

    # curated concepts (for eval + analysis)
    reach_targets: tuple = ("yellow pentagon", "red pentagon", "yellow star", "green star")
    push_pairs:    tuple = (("green star", "green cube"), ("yellow pentagon", "yellow star"),
                            ("yellow star", "yellow pentagon"), ("green cube", "green star"))

    state_dim:  int = 2
    action_dim: int = 2
    action_horizon:    int = 16
    inference_horizon: int = 4
    use_relative_actions: bool = False

    lt_img_w: int = 320
    lt_img_h: int = 180
    img_seq_len: int = 120
    img_grid_h:  int = 6
    img_grid_w:  int = 11
    # Was (28,), which never existed — Qwen3.5-0.8B has 24 layers. It was harmless
    # only because encode_vlm ignored the index and returned last_hidden_state,
    # i.e. layer 24. So 24 is what the existing vlm_embeddings.h5 actually
    # contains, despite that file's attrs recording 28. Stating 24 here keeps the
    # config, the cache and reality in agreement; the new layer guard in
    # data/hdf5_embeddings.py would otherwise (correctly) reject the cache.
    vlm_extract_layers: tuple = (24,)

    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int  = 256
    dit_num_layers:  int  = 6
    dit_num_heads:   int  = 8

    batch_size:       int = 256
    num_epochs:       int = 150
    embed_batch_size: int = 16

    sim_episodes:  int = 50
    sim_max_steps: int = 200

    def __post_init__(self) -> None:
        if not self.dataset_root:
            self.dataset_root = _default_dataset_root()
        if not self.dataset_path:
            self.dataset_path = self.dataset_root
        stats_path = Path(self.dataset_root) / "meta" / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            self.action_mean = tuple(float(x) for x in stats["action"]["mean"])
            self.action_std  = tuple(float(x) for x in stats["action"]["std"])
            self.state_mean  = tuple(float(x) for x in stats["observation.state"]["mean"])
            self.state_std   = tuple(float(x) for x in stats["observation.state"]["std"])
        else:
            print(f"[LTExp03][WARN] stats.json not found at {stats_path}; unit normalisation.")
            self.action_mean = tuple([0.0] * self.action_dim)
            self.action_std  = tuple([1.0] * self.action_dim)
            self.state_mean  = tuple([0.0] * self.state_dim)
            self.state_std   = tuple([1.0] * self.state_dim)
