"""
Language Table dataset loaders.

LanguageTableDataset          – reads the converted parquet shards (image stored as
                                JPEG bytes per row) and yields (image, state, actions,
                                task_text=<per-episode instruction>).  Used to
                                pre-compute VLM embeddings.
LanguageTableEmbeddingDataset – pre-computed VLM token sequences (re-uses the generic
                                PushTEmbeddingDataset; the cache format is
                                dataset-agnostic).

The ONLY substantive difference from AlohaDataset is that `task_text` is the
PER-EPISODE instruction string (Language Table is language-conditioned and the
command varies every episode), and frames are decoded from in-parquet JPEG bytes
rather than a side-car mp4.

Action representation: the stored `action` IS the env-step 2D delta setpoint, so it
is returned as-is (z-score normalised) — use_relative_actions is False and there is
no relative subtraction (mirrors ALOHA, NOT PushT).
"""

from __future__ import annotations

import glob
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Re-export the generic embedding dataset under a Language Table name.
from data.pusht.dataset import PushTEmbeddingDataset as LanguageTableEmbeddingDataset  # noqa: F401


def resize_frame(rgb_uint8: np.ndarray, width: int, height: int) -> Image.Image:
    """uint8 (H,W,3) RGB array → PIL image resized to (width, height).

    Single source of truth for Language Table image preprocessing — used by both
    the precompute dataset and the live eval agent so train/eval distributions match.
    """
    return Image.fromarray(np.ascontiguousarray(rgb_uint8)).resize((width, height))


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


class LanguageTableDataset(Dataset):
    """
    Each sample:
        image     – PIL.Image (resized to cfg.lt_img_w × lt_img_h, RGB)
        state     – float32 (2,)        normalised effector translation
        actions   – float32 (H, 2)      normalised 2D delta setpoints
        task_text – str                 the episode's language instruction
        idx       – global frame index
    """

    def __init__(self, config, cache_frames: bool = False) -> None:
        self.config = config
        self.H = config.action_horizon
        self.img_w = config.lt_img_w
        self.img_h = config.lt_img_h

        self.action_mean = np.array(config.action_mean, dtype=np.float32)
        self.action_std  = np.array(config.action_std,  dtype=np.float32)
        self.state_mean  = np.array(config.state_mean,  dtype=np.float32)
        self.state_std   = np.array(config.state_std,   dtype=np.float32)

        root = Path(config.dataset_root)
        shards = sorted(glob.glob(str(root / "data" / "chunk-000" / "file-*.parquet")))
        if not shards:
            raise FileNotFoundError(f"No data shards under {root/'data'/'chunk-000'}")
        df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
        df = df.sort_values("index").reset_index(drop=True)

        self.states       = np.asarray(df["observation.state"].tolist(), dtype=np.float32)  # (N,2)
        self.actions      = np.asarray(df["action"].tolist(),            dtype=np.float32)  # (N,2)
        self.episode_ids  = df["episode_index"].to_numpy(dtype=np.int64)
        self.instructions = df["instruction"].tolist()
        self.images       = df["image"].tolist()   # list of JPEG bytes (references; cheap)
        self.N = len(df)

        n_eps = len(np.unique(self.episode_ids))
        n_inst = len(set(self.instructions))
        print(f"  LanguageTableDataset: {self.N:,} frames | {n_eps:,} episodes | "
              f"{n_inst:,} unique instructions | state{self.states.shape[1]}D "
              f"action{self.actions.shape[1]}D | img {self.img_w}x{self.img_h}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        rgb = np.asarray(Image.open(io.BytesIO(self.images[idx])).convert("RGB"))
        image = resize_frame(rgb, self.img_w, self.img_h)

        state_norm = _normalize(self.states[idx], self.state_mean, self.state_std)

        ep = self.episode_ids[idx]
        last = self.actions[idx]
        acts = []
        for k in range(self.H):
            fi = idx + k
            if fi < self.N and self.episode_ids[fi] == ep:
                a = self.actions[fi]
                last = a
            else:
                a = last                                   # pad past episode end
            acts.append(_normalize(a, self.action_mean, self.action_std))
        actions_arr = np.stack(acts, axis=0)               # (H, 2)

        return {
            "image":     image,
            "state":     torch.from_numpy(state_norm.astype(np.float32)),
            "actions":   torch.from_numpy(actions_arr.astype(np.float32)),
            "task_text": self.instructions[idx],
            "idx":       int(idx),
        }
