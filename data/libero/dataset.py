"""LIBERO-Goal dataset — raw demonstrations for VLM embedding precompute.

LiberoDataset          reads the 10 task HDF5 files and yields
                       {image, state, actions, task_text} per frame.
LiberoEmbeddingDataset re-exports the generic embedding reader; the cache format
                       is dataset-agnostic.

Why read the HDF5s directly rather than going through LIBERO's own loaders: the
`libero` package pins numpy 1.22 / robosuite 1.4 and lives in a separate conda
env (`vla_libero`) precisely so it cannot drag those pins into the training env.
The demonstration files are plain HDF5 with a documented layout, so training-side
code reads them with h5py alone and never imports libero.

Image convention: the HDF5s carry macros_image_convention="opengl", i.e. frames
are stored bottom-up. They are flipped once here. The live eval agent must apply
the same flip, or train and eval see vertically mirrored worlds — which is a
silent failure that looks like a bad policy rather than a bug.

State representation: the 9-D `robot_states` array is NOT "7 joint positions
plus 2 gripper" as earlier versions of this docstring claimed. Decomposed
against the demos' own observation arrays it is
    gripper_states(2) | ee_pos(3) | eef_quat(4)
— the trailing four values form a unit quaternion (norm 1.00002), and
`joint_states` holds entirely different numbers. envs/libero_env.build_state
rebuilds exactly this from the live simulator, which does not expose
`robot_states` directly.

Action representation: LIBERO actions are 7-D OSC deltas — 6 end-effector
(dx,dy,dz,droll,dpitch,dyaw) plus a binary gripper command in [-1,1], already
bounded, so they are used as-is rather than z-scored. Normalising a
half-saturated binary channel would stretch its two modes apart and make the
flow-matching target harder for no benefit.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Re-export the generic embedding dataset under a LIBERO name.
from data.pusht.dataset import PushTEmbeddingDataset as LiberoEmbeddingDataset  # noqa: F401

CAMERA = "agentview_rgb"
# Wrist camera, present in every LIBERO demo and used by pi0/pi05/GR00T. Read
# only when cfg.n_views > 1 so the single-view runs are untouched.
CAMERA_WRIST = "eye_in_hand_rgb"


def task_language(fname: str) -> str:
    """'put_the_bowl_on_the_plate_demo.hdf5' -> 'put the bowl on the plate'."""
    stem = fname[:-len("_demo.hdf5")] if fname.endswith("_demo.hdf5") else Path(fname).stem
    return stem.replace("_", " ")


# Re-exported, not redefined. The implementation lives in data/libero/frames.py
# so the closed-loop eval agent can import it without pulling in this module's
# package chain (data/__init__ -> pusht -> av, which the simulator env lacks).
# Both sides therefore run the same code rather than two copies that can drift.
from data.libero.frames import resize_frame  # noqa: E402,F401


class LiberoDataset(Dataset):
    """Flat per-frame view over every demo in LIBERO-Goal."""

    def __init__(self, config, cache_frames: bool = False) -> None:
        self.cfg = config
        root = Path(config.dataset_root)
        self.files = sorted(root.glob("*_demo.hdf5"))
        if not self.files:
            raise FileNotFoundError(f"no LIBERO demos under {root}")

        self.horizon = int(config.action_horizon)
        self.w, self.h = int(config.lt_img_w), int(config.lt_img_h)

        # Index: one row per frame -> (file idx, demo name, frame idx)
        self.index: list[tuple[int, str, int]] = []
        self.episode_ids: list[int] = []
        self.task_texts: list[str] = []
        self._lengths: dict[tuple[int, str], int] = {}
        ep = 0
        for fi, fp in enumerate(self.files):
            lang = task_language(fp.name)
            with h5py.File(fp, "r") as f:
                for dname in sorted(f["data"], key=lambda s: int(s.split("_")[1])):
                    T = f["data"][dname]["actions"].shape[0]
                    self._lengths[(fi, dname)] = T
                    for t in range(T):
                        self.index.append((fi, dname, t))
                        self.episode_ids.append(ep)
                        self.task_texts.append(lang)
                    ep += 1
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int64)
        self.n_episodes = ep
        self._handles: dict[int, h5py.File] = {}
        self._cache = {} if cache_frames else None
        # ASCII only: this console is cp874 (Thai) and cannot encode a middot, so
        # a decorative separator here would crash dataset construction itself.
        print(f"  LiberoDataset: {len(self.index):,} frames | {ep} episodes | "
              f"{len(self.files)} tasks | action_dim={config.action_dim}")

    def _f(self, fi: int) -> h5py.File:
        # one handle per file per worker; h5py handles are not fork-safe, so they
        # are opened lazily on first access inside whichever process uses them
        if fi not in self._handles:
            self._handles[fi] = h5py.File(self.files[fi], "r")
        return self._handles[fi]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        fi, dname, t = self.index[i]
        d = self._f(fi)["data"][dname]
        T = self._lengths[(fi, dname)]

        img = np.asarray(d["obs"][CAMERA][t])[::-1]      # opengl -> upright
        state = np.asarray(d["robot_states"][t], dtype=np.float32)

        # Second view. LIBERO records BOTH cameras the published pipelines use
        # (env_args camera_names = ['robot0_eye_in_hand', 'agentview']), and the
        # LeRobot conversion exposes them as image + wrist_image. exp01-exp04 read
        # the exterior view only, which puts the robot-pretrained backbones —
        # trained multi-view — outside their observation spec while leaving the
        # stock VLMs inside theirs. n_views=2 removes that asymmetry.
        if getattr(self.cfg, "n_views", 1) > 1:
            wrist = np.asarray(d["obs"][CAMERA_WRIST][t])[::-1]
            return {
                "image": [resize_frame(img, self.w, self.h),
                          resize_frame(wrist, self.w, self.h)],
                "state": torch.from_numpy(state),
                "actions": torch.from_numpy(
                    np.asarray(d["actions"][:], dtype=np.float32)[
                        np.minimum(np.arange(t, t + self.horizon), T - 1)]),
                "task_text": self.task_texts[i],
                "idx": i,
            }

        # action chunk, edge-padded at the end of the episode
        idx = np.minimum(np.arange(t, t + self.horizon), T - 1)
        actions = np.asarray(d["actions"][:], dtype=np.float32)[idx]

        return {
            "image": resize_frame(img, self.w, self.h),
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "task_text": self.task_texts[i],
            # precompute's collate keys rows by this so the streaming writer can
            # place each embedding at its own dataset row even when the loader
            # shuffles or uses multiple workers
            "idx": i,
        }

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()
