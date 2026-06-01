"""
PushT dataset loaders.

PushTDataset         – reads raw parquet + video, yields (image, state, actions).
PushTEmbeddingDataset – reads pre-computed VLM embeddings, yields (embed, state, actions).
                        Used during actual training (much faster on M1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _decode_all_frames(video_path: Path) -> Dict[int, np.ndarray]:
    """
    Sequentially decode every frame in *video_path* and return a dict
    mapping global frame index → uint8 RGB array (H, W, 3).

    Requires PyAV built against an ffmpeg with AV1 support.
    On macOS:  brew install ffmpeg  &&  pip install av
    """
    print(f"  Decoding video: {video_path.name}  (this may take ~30 s) …")
    frames: Dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    # Ask the codec to decode every frame (no skipping)
    stream.codec_context.skip_frame = "DEFAULT"

    frame_idx = 0
    for packet in container.demux(stream):
        try:
            for frame in packet.decode():
                frames[frame_idx] = frame.to_ndarray(format="rgb24")
                frame_idx += 1
        except av.AVError:
            continue  # skip malformed packets

    container.close()
    print(f"  Decoded {len(frames):,} frames.")
    return frames


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Raw dataset (used for pre-computing VLM embeddings)
# ──────────────────────────────────────────────────────────────────────────────

class PushTDataset(Dataset):
    """
    Reads the LeRobot-format PushT dataset from disk.

    Each sample:
        image   – PIL.Image  (H=96, W=96, RGB)
        state   – float32 tensor (2,)   normalised
        actions – float32 tensor (action_horizon, 2)  normalised
        idx     – global frame index (int)
    """

    def __init__(
        self,
        dataset_path: str,
        config,
        cache_frames: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.action_horizon = config.action_horizon

        # Normalisation stats as numpy arrays.
        # state_mean/std may be 6D (Exp B) but raw state from env is always 2D
        # (agent position only); we slice to the first 2 dims here.
        self.action_mean = np.array(config.action_mean,    dtype=np.float32)
        self.action_std  = np.array(config.action_std,     dtype=np.float32)
        self.state_mean  = np.array(config.state_mean[:2], dtype=np.float32)
        self.state_std   = np.array(config.state_std[:2],  dtype=np.float32)

        # ── Load tabular data ──────────────────────────────────────────────
        data_file    = self.dataset_path / "data/chunk-000/file-000.parquet"
        self.df = pd.read_parquet(data_file)

        # Pre-build a boolean mask: is frame at position k still the same episode?
        # Used for action-horizon padding (don't cross episode boundaries).
        self.episode_ids = self.df["episode_index"].to_numpy(dtype=np.int64)

        # ── Decode video ───────────────────────────────────────────────────
        video_path = (
            self.dataset_path / "videos/observation.image/chunk-000/file-000.mp4"
        )
        self.frames: Optional[Dict[int, np.ndarray]] = None
        if cache_frames:
            self.frames = _decode_all_frames(video_path)
        else:
            self._video_path = video_path

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        global_idx  = int(row["index"])
        episode_idx = int(row["episode_index"])

        # ── Image ─────────────────────────────────────────────────────────
        if self.frames is not None:
            img_arr = self.frames[global_idx]
        else:
            img_arr = self._decode_single_frame(global_idx)
        image = Image.fromarray(img_arr)

        # ── State ─────────────────────────────────────────────────────────
        state = np.array(row["observation.state"], dtype=np.float32)
        state_norm = _normalize(state, self.state_mean, self.state_std)

        # ── Action sequence (chunked) ──────────────────────────────────────
        # If use_relative_actions: store delta = action - state_at_that_step
        actions = []
        last_action = np.array(row["action"], dtype=np.float32)
        for k in range(self.action_horizon):
            future = idx + k
            if (
                future < len(self.df)
                and self.episode_ids[future] == episode_idx
            ):
                a = np.array(self.df.iloc[future]["action"], dtype=np.float32)
                last_action = a
            else:
                a = last_action   # pad with last valid action

            if getattr(self.config, "use_relative_actions", False):
                # Use state at the future timestep as the base
                if future < len(self.df) and self.episode_ids[future] == episode_idx:
                    s = np.array(self.df.iloc[future]["observation.state"], dtype=np.float32)
                else:
                    s = state   # fallback to current state at boundary
                a = a - s       # relative delta

            actions.append(_normalize(a, self.action_mean, self.action_std))

        actions_arr = np.stack(actions, axis=0)  # (horizon, 2)

        return {
            "image":    image,
            "state":    torch.from_numpy(state_norm),
            "actions":  torch.from_numpy(actions_arr),
            "task_text": self.config.task_text,
            "idx":      global_idx,
        }

    # ------------------------------------------------------------------
    def _decode_single_frame(self, global_idx: int) -> np.ndarray:
        """Seek to a single frame by timestamp (fallback when not caching)."""
        fps = 10.0
        target_ts = global_idx / fps
        container = av.open(str(self._video_path))
        stream = container.streams.video[0]
        container.seek(int(target_ts * av.time_base), stream=stream)
        frame_arr = None
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts * stream.time_base >= target_ts - 1 / fps:
                    frame_arr = frame.to_ndarray(format="rgb24")
                    break
            if frame_arr is not None:
                break
        container.close()
        return frame_arr if frame_arr is not None else np.zeros((96, 96, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding dataset (used during training after pre-computation)
# ──────────────────────────────────────────────────────────────────────────────

class PushTEmbeddingDataset(Dataset):
    """
    Fast dataset that loads pre-computed VLM token sequences from disk.

    Supports two cache formats:
      v2 (new): full token sequences
        embeddings : (N, seq_len, vlm_hidden_size)  bfloat16
        img_masks  : (N, seq_len)                   bool
      v1 (legacy): mean-pooled single vector
        embeddings : (N, vlm_hidden_size)            bfloat16/float32

    Training code detects the format automatically via embeddings.ndim.
    """

    def __init__(self, cache_path: str) -> None:
        cache = torch.load(cache_path, weights_only=True, map_location="cpu")

        self.embeddings: torch.Tensor = cache["embeddings"].float()
        self.states:     torch.Tensor = cache["states"].float()
        self.actions:    torch.Tensor = cache["actions"].float()

        # v2 cache includes img_masks; v1 does not
        self.img_masks: torch.Tensor | None = cache.get("img_masks", None)

        assert len(self.embeddings) == len(self.states) == len(self.actions), \
            "Cache tensors have mismatched lengths."

        if self.embeddings.ndim == 4:
            fmt = "v3 (multi-scale)"
        elif self.embeddings.ndim == 3:
            fmt = "v2 (full sequences)"
        else:
            fmt = "v1 (mean-pooled)"
        print(
            f"  Loaded embedding cache [{fmt}]: {len(self.embeddings):,} samples  "
            f"| embed shape={tuple(self.embeddings.shape[1:])}"
        )
        if self.embeddings.ndim == 2:
            print("  [WARN] Cache is v1 (mean-pooled). "
                  "Re-run precompute_embeddings.py for correct training.")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "embedding": self.embeddings[idx],   # (seq, 1024) or (1024,)
            "state":     self.states[idx],
            "actions":   self.actions[idx],
        }
        if self.img_masks is not None:
            item["img_mask"] = self.img_masks[idx]   # (seq_len,) bool
        return item
