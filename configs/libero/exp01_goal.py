"""
configs/libero/exp01_goal.py
────────────────────────────
LIBERO-Goal Exp01 — Pi-0.5's VLM as a frozen backbone, this project's DiT
flow-matching head trained on top.

Why this dataset, and why this backbone
───────────────────────────────────────
LIBERO-Goal's ten tasks share ONE fixed scene and object set and differ only in
the goal, so language is the only signal that disambiguates them. Neither of the
project's other datasets can test the instruction pathway: ALOHA has a single
instruction (no language variation at all), and curated Language Table has no
linearly recoverable action signal for any of nine backbones (R^2 <= 0.063), so
an ablation there returns "no effect" regardless of what is ablated.

The backbone is Pi-0.5 rather than the project's frozen Qwen because Chapter 1
found Pi-0.5 to be the one arm whose visual representation is conditioned on
language: on this exact dataset its IMAGE tokens carry goal identity at
eta^2 = 0.761, where every other backbone sits in a 0.16-0.24 band. That is a
representational fact measured offline. Whether it produces robust instruction
following is a closed-loop question, which is what this experiment runs.

Read depth
──────────
vlm_extract_layers is the first knob Chapter 2 varies. PaliGemma-3B has 18
layers; the depth sweep found action decodability peaking at ~50% depth for
9 of 9 arms on ALOHA and 8 of 9 on LIBERO-Goal, with Pi-0.5's own peak at 75%
here (+0.022 over its last layer). This config starts at the LAST layer so the
first run is comparable to every previously reported number; the depth ablation
then moves it and the embedding cache is regenerated. That regeneration is not
optional — data/hdf5_embeddings.py compares the cache's recorded layer against
this field and refuses a mismatch, because two caches built at different depths
are indistinguishable once loaded.

Evaluation
──────────
Training uses only LIBERO's ten canonical instruction strings. Paraphrases in
data/libero/paraphrases.py are held out entirely and substituted at eval time,
giving three tiers: canonical (in-distribution), near-paraphrase (reworded,
same content words) and far-paraphrase (different verb and/or object synonym).
Same goal, same scene, same required trajectory — only the wording changes.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from configs.base_config import BaseVLAConfig


def _default_dataset_root() -> str:
    env = os.environ.get("LIBERO_DATASET_ROOT")
    if env:
        return env
    return str(Path(__file__).parents[2] / "asset" / "data" / "libero" / "libero_goal")


@dataclass
class LiberoExp01Goal(BaseVLAConfig):
    repo_id:          str = "nvidia/LIBERO-goal"
    dataset_root:     str = ""
    dataset_path:     str = ""
    # Pi-0.5's VLM subtree, grafted into PaliGemma's HF architecture. The action
    # expert is discarded — only the perceptual half is used here.
    vlm_backbone:     str = "pi05"
    model_path:       str = "asset/models/paligemma_3b_pt_224"   # config/processor source
    output_dir:       str = "asset/runs/libero/exp01_goal"
    embeddings_cache: str = "asset/runs/libero/exp01_goal/vlm_embeddings.h5"

    task_text: str = "complete the task"      # overridden per-frame by the dataset

    # 7-D OSC delta: dx dy dz droll dpitch dyaw + binary gripper.
    # 9-D robot_states: 7 joint positions + 2 gripper finger positions.
    state_dim:  int = 9
    action_dim: int = 7
    action_horizon:    int = 16
    inference_horizon: int = 8    # HARD CONSTRAINT: must match at eval time
    # LIBERO actions are already bounded deltas in [-1, 1]; the gripper channel is
    # effectively binary. Treating them as "relative" would integrate them, which
    # is wrong — they ARE the env's step() space and pass straight through.
    use_relative_actions: bool = False

    # 128x128 native. PaliGemma is fixed at 224x224 / 256 image tokens, so the
    # processor upsamples; the grid below describes the TOKEN layout, not pixels.
    lt_img_w: int = 224
    lt_img_h: int = 224
    # Fixed total sequence length: 256 image tokens + BOS + instruction.
    # Measured through the real PaliGemma processor rather than by counting text
    # tokens (which misses the prepended BOS and cost one off-by-one already):
    #   canonical 267 · near-paraphrase 269 · far-paraphrase 270
    # 272 is the max plus headroom, so training and every eval tier produce the
    # SAME shape. Padding is to this exact length, not to the longest item in the
    # batch — a per-batch length makes the embedding cache ragged and silently
    # changes the sequence between training (short canonical strings) and OOD
    # eval (longer paraphrases).
    img_seq_len: int = 272
    text_token_budget: int = 16
    img_grid_h:  int = 16
    img_grid_w:  int = 16
    vlm_hidden_size: int = 2048   # PaliGemma-3B text width
    vlm_extract_layers: tuple = (-1,)   # last layer; Ch.2 varies this

    use_dit_decoder: bool = True
    lora_rank:       int  = 16
    dit_hidden_dim:  int = 320
    dit_num_layers:  int = 6
    dit_num_heads:   int = 8

    batch_size:       int = 128
    num_epochs:       int = 120
    embed_batch_size: int = 8     # 3B backbone at 224x224 — keep inside 16 GB

    sim_episodes:  int = 20       # per task, per instruction tier
    sim_max_steps: int = 400
    video_fps:     int = 20

    def __post_init__(self) -> None:
        if not self.dataset_root:
            self.dataset_root = _default_dataset_root()
        if not self.dataset_path:
            self.dataset_path = self.dataset_root
        super().__post_init__() if hasattr(super(), "__post_init__") else None
