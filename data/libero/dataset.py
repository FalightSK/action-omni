"""
data/libero/dataset.py
──────────────────────
Raw (precompute/train-time) dataset for LIBERO-Spatial.

Reads the lerobot-v3 parquet directly with pyarrow (images are PNG bytes stored
INLINE in the parquet — `video_keys` is empty). This avoids lerobot 0.4.x's
`LeRobotDataset(episodes=[...])` construction, which hangs for minutes on this
dataset. Same pattern as the repo's aloha/LT loaders (parquet + pandas), just with
PNG-bytes images instead of a side-car mp4.

Satisfies the precompute collate contract:
__getitem__ -> {'image': PIL.Image, 'state': (8,) f32 z-scored, 'actions': (H,7) f32
z-scored, 'task_text': str, 'idx': int}.

Orientation: dataset frames are stored 180deg-rotated vs the live OffScreenRenderEnv
render. We train on the dataset frames as-is; the eval agent rotates the live render.
"""
from __future__ import annotations

import glob
import io
import os

import numpy as np
import torch
from PIL import Image

from configs.libero.exp01_baseline import LIBERO_SPATIAL_TASKS, _lerobot_root

EPS = 1e-8


def resize_frame(rgb_uint8: np.ndarray, width: int, height: int) -> Image.Image:
    """HxWx3 uint8 RGB -> PIL, resized to (width, height). Shared by precompute + eval."""
    img = Image.fromarray(np.asarray(rgb_uint8, dtype=np.uint8), mode="RGB")
    if img.size != (width, height):
        img = img.resize((width, height))
    return img


def _spatial_task_indices(root: str):
    """Map the 10 libero_spatial task strings -> their task_index values.
    tasks.parquet stores the task STRING as the row index and task_index as a column."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
    if "task" in df.columns:                       # (task, task_index) columns
        mapping = {str(r["task"]): int(r["task_index"]) for _, r in df.iterrows()}
    else:                                          # task string is the index
        mapping = {str(k): int(v) for k, v in df["task_index"].items()}
    wanted = set(LIBERO_SPATIAL_TASKS)
    idxs = {v for k, v in mapping.items() if k in wanted}
    assert len(idxs) == len(wanted), f"matched {len(idxs)}/{len(wanted)} spatial tasks"
    return idxs, {v: k for k, v in mapping.items()}


class LiberoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, cache_frames: bool = True):
        import pyarrow.dataset as pads

        self.cfg = cfg
        self.horizon = cfg.action_horizon
        self.image_size = cfg.image_size
        root = cfg.dataset_path or _lerobot_root()

        spatial_idx, idx2task = _spatial_task_indices(root)

        # Read only spatial rows, only the columns we need (filtered pyarrow scan).
        data_glob = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))
        assert data_glob, f"no parquet under {root}/data"
        dset = pads.dataset(data_glob, format="parquet")
        cols = ["observation.images.image", "observation.state", "action",
                "episode_index", "index", "task_index"]
        filt = pads.field("task_index").isin(list(spatial_idx))
        tbl = dset.to_table(columns=cols, filter=filt)

        # Sort by global frame index so episode order + chunking are correct.
        order = np.argsort(tbl.column("index").to_numpy())
        self._img = [tbl.column("observation.images.image")[int(i)].as_py() for i in order]
        self._state = np.stack([np.asarray(tbl.column("observation.state")[int(i)].as_py(), np.float32) for i in order])
        self._action = np.stack([np.asarray(tbl.column("action")[int(i)].as_py(), np.float32) for i in order])
        self._ep = tbl.column("episode_index").to_numpy()[order]
        self._task = [idx2task[int(tbl.column("task_index")[int(i)].as_py())] for i in order]
        self.N = len(order)

        # Normalisation from the config (loaded from meta/stats.json).
        self.a_mean = np.asarray(cfg.action_mean, np.float32)
        self.a_std = np.clip(np.asarray(cfg.action_std, np.float32), EPS, None)
        self.s_mean = np.asarray(cfg.state_mean, np.float32)
        self.s_std = np.clip(np.asarray(cfg.state_std, np.float32), EPS, None)

    def __len__(self):
        return self.N

    def _decode(self, cell) -> Image.Image:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        img = Image.open(io.BytesIO(b)).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size))
        return img

    def __getitem__(self, i):
        image = self._decode(self._img[i])                      # dataset frame as-is (rotated-stored)
        state = (self._state[i] - self.s_mean) / self.s_std      # (8,)

        # 16-step action chunk from consecutive frames within the same episode; pad last.
        ep = self._ep[i]
        chunk = np.empty((self.horizon, self._action.shape[1]), np.float32)
        last = self._action[i]
        for k in range(self.horizon):
            j = i + k
            if j < self.N and self._ep[j] == ep:
                last = self._action[j]
            chunk[k] = last
        actions = (chunk - self.a_mean) / self.a_std             # (H,7)

        return {
            "image": image,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "task_text": self._task[i],
            "idx": int(i),
        }
