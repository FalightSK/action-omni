"""
data/libero/dataset.py
──────────────────────
Raw (precompute-time) dataset for LIBERO-Spatial, backed by lerobot's LeRobotDataset.

Satisfies the precompute collate contract (scripts/precompute.py _custom_collate):
__getitem__ -> {'image': PIL.Image, 'state': (8,) f32 z-scored, 'actions': (H,7) f32
z-scored, 'task_text': str, 'idx': int}.

Orientation note: HuggingFaceVLA/libero frames are stored 180-degrees rotated vs the
live OffScreenRenderEnv render. We train on the DATASET frames as-is; the eval agent
(envs/libero_env.py) rotates the LIVE render 180 degrees so the two distributions match.
resize_frame() below is the single source of truth reused by the eval agent.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from configs.libero.exp01_baseline import LIBERO_SPATIAL_TASKS, REPO_ID

EPS = 1e-8


def resize_frame(rgb_uint8: np.ndarray, width: int, height: int) -> Image.Image:
    """HxWx3 uint8 RGB -> PIL, resized to (width, height). Shared by precompute + eval."""
    img = Image.fromarray(np.asarray(rgb_uint8, dtype=np.uint8), mode="RGB")
    if img.size != (width, height):
        img = img.resize((width, height))
    return img


class LiberoDataset(torch.utils.data.Dataset):
    """Frame-indexed LIBERO-Spatial dataset yielding the precompute item contract."""

    def __init__(self, cfg, cache_frames: bool = True):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        self.cfg = cfg
        self.image_key = cfg.camera_key
        self.horizon = cfg.action_horizon
        self.image_size = cfg.image_size

        meta = LeRobotDatasetMetadata(REPO_ID)
        wanted = set(LIBERO_SPATIAL_TASKS)
        episodes = [
            int(e) for e, ts in zip(meta.episodes["episode_index"], meta.episodes["tasks"])
            if any(t in wanted for t in ts)
        ]
        got = {t for ts in meta.episodes["tasks"] for t in ts if t in wanted}
        assert got == wanted, f"libero_spatial: dataset missing tasks {wanted - got}"

        # delta_timestamps makes lerobot return an H-step action chunk per frame and
        # clamp past-episode-end steps to the last real action (== repeat-last padding).
        self.ds = LeRobotDataset(
            REPO_ID, episodes=episodes,
            delta_timestamps={"action": [i / meta.fps for i in range(self.horizon)]},
        )

        # Normalisation comes from the config (loaded from meta/stats.json) so train,
        # eval, and the cached embeddings all use identical stats.
        self.a_mean = torch.tensor(cfg.action_mean, dtype=torch.float32)
        self.a_std = torch.tensor(cfg.action_std, dtype=torch.float32).clamp(min=EPS)
        self.s_mean = torch.tensor(cfg.state_mean, dtype=torch.float32)
        self.s_std = torch.tensor(cfg.state_std, dtype=torch.float32).clamp(min=EPS)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        img_chw = s[self.image_key]              # [3,256,256] float32 in [0,1]
        rgb = (img_chw * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        image = resize_frame(rgb, self.image_size, self.image_size)   # dataset frames as-is (no rotation)
        state = (s["observation.state"].float() - self.s_mean) / self.s_std      # (8,)
        actions = (s["action"].float() - self.a_mean) / self.a_std               # (H,7)
        return {
            "image": image,
            "state": state,
            "actions": actions,
            "task_text": s["task"],              # per-frame instruction (language conditioning)
            "idx": int(i),
        }
