"""
ALOHA dataset loaders (lerobot v3.0 transfer-cube).

AlohaDataset           – reads raw parquet shards + decodes the single top-camera
                         mp4 sequentially, yields (image, state, actions).  Used to
                         pre-compute VLM embeddings.
AlohaEmbeddingDataset  – pre-computed VLM token sequences (re-uses the generic
                         PushTEmbeddingDataset; the cache format is dataset-agnostic).

Why hand-rolled instead of lerobot.LeRobotDataset:
  LeRobotDataset.__getitem__ with the pyav backend aliases global indices on this
  v3.0 dataset (ds[400] returned row 0), which would corrupt the precompute cache.
  The raw layout is simple — one mp4 with all 20k frames in order (global index ==
  frame position) plus 3 parquet shards — so we read it directly and correctly.

Action representation: ABSOLUTE joint targets, z-score normalised per joint.
"""

from __future__ import annotations

import glob
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Re-export the generic embedding dataset under an ALOHA name.
from data.pusht.dataset import PushTEmbeddingDataset as AlohaEmbeddingDataset  # noqa: F401


def resize_frame(rgb_uint8: np.ndarray, width: int, height: int) -> Image.Image:
    """uint8 (H,W,3) RGB array → PIL image resized to (width, height).

    Single source of truth for ALOHA image preprocessing — used by both the
    precompute dataset and the live eval agent so train/eval distributions match.
    """
    return Image.fromarray(np.ascontiguousarray(rgb_uint8)).resize((width, height))


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


class _SequentialFrameReader:
    """Decodes a video in order. Fast for sequential access (precompute);
    transparently re-opens + skips forward if an index is requested out of order."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._open()

    def _open(self) -> None:
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.skip_frame = "DEFAULT"
        self._gen = self._frames()
        self._next = 0

    def _frames(self):
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                yield frame.to_ndarray(format="rgb24")

    def get(self, idx: int) -> np.ndarray:
        if idx < self._next:                      # backward jump → restart decode
            self.container.close()
            self._open()
        frame = None
        while self._next <= idx:
            frame = next(self._gen)
            self._next += 1
        return frame

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


class AlohaDataset(Dataset):
    """
    Each sample:
        image     – PIL.Image (resized to config.aloha_img_w × aloha_img_h, RGB)
        state     – float32 (14,)         normalised joint positions
        actions   – float32 (H, 14)       normalised ABSOLUTE joint targets
        task_text – str
        idx       – global frame index
    """

    def __init__(self, config, cache_frames: bool = False) -> None:
        # cache_frames is ignored: 20k × 480×640×3 ≈ 18 GB — far too large to hold.
        self.config = config
        self.H = config.action_horizon
        self.img_w = config.aloha_img_w
        self.img_h = config.aloha_img_h
        self.task_text = config.task_text

        self.action_mean = np.array(config.action_mean, dtype=np.float32)
        self.action_std  = np.array(config.action_std,  dtype=np.float32)
        self.state_mean  = np.array(config.state_mean,  dtype=np.float32)
        self.state_std   = np.array(config.state_std,   dtype=np.float32)

        root = Path(config.dataset_root)

        # ── Tabular data: concat the file-NNN.parquet shards, ordered by index ──
        # (episode_000000.parquet is a per-episode duplicate — exclude it.)
        shards = sorted(glob.glob(str(root / "data" / "chunk-000" / "file-*.parquet")))
        if not shards:
            raise FileNotFoundError(f"No data shards under {root/'data'/'chunk-000'}")
        df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
        df = df.sort_values("index").reset_index(drop=True)

        self.states  = np.asarray(df["observation.state"].tolist(), dtype=np.float32)  # (N,14)
        self.actions = np.asarray(df["action"].tolist(),            dtype=np.float32)  # (N,14)
        self.episode_ids = df["episode_index"].to_numpy(dtype=np.int64)
        self.N = len(df)

        # ── Video: single mp4 with all frames in order ──────────────────────────
        vids = sorted(glob.glob(str(
            root / "videos" / config.camera_key / "chunk-000" / "*.mp4")))
        if not vids:
            raise FileNotFoundError(
                f"No mp4 under {root/'videos'/config.camera_key/'chunk-000'}")
        self._reader = _SequentialFrameReader(vids[0])

        print(f"  AlohaDataset: {self.N:,} frames | state{self.states.shape[1]}D "
              f"action{self.actions.shape[1]}D | img {self.img_w}x{self.img_h}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        rgb = self._reader.get(idx)                       # (480,640,3) uint8
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
        actions_arr = np.stack(acts, axis=0)               # (H, 14)

        return {
            "image":     image,
            "state":     torch.from_numpy(state_norm.astype(np.float32)),
            "actions":   torch.from_numpy(actions_arr.astype(np.float32)),
            "task_text": self.task_text,
            "idx":       int(idx),
        }
