"""Build the LIBERO-Goal probe set, in the same schema as the other probes.

Why this is a separate script from build_probe.py
─────────────────────────────────────────────────
build_probe.py goes through the config registry and the project's Dataset
classes. LIBERO has neither yet — and deliberately so: this probe exists to
decide whether LIBERO-Goal is worth building a pipeline around, so building the
pipeline first would defeat the purpose. It reads the demonstration HDF5s
directly and emits a byte-compatible probe_libero_goal.h5.

What the gate is for
────────────────────
Finding 1 of the cross-backbone study recommended a cheap upfront check before
committing GPU time to any dataset: *is the action linearly recoverable from a
single observation at all, by any encoder?* Language Table failed that check —
action R² ≤ 0.063 across seven backbones — which makes it a poor testbed for
anything about the action pathway. This applies the same test to LIBERO-Goal
before it becomes the language testbed for the anatomy study.

Sampling matches the other probes: stratified over episode × phase, capped so
each contributing episode supplies several phase points rather than one frame
from a huge number of episodes. LIBERO-Goal has 500 episodes (10 tasks x 50
demos), so the 12-per-episode floor yields 200 episodes — 20 per task, evenly
spread across all ten goals.

Image convention: the HDF5s carry macros_image_convention="opengl", meaning
frames are stored bottom-up. They are flipped here so the probe holds
upright images, matching every other probe in the study.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_probe import _stratified_indices  # noqa: E402

ROOT = HERE.parents[2]
OUT_DIR = ROOT / "asset" / "analysis" / "latent_compare"
LIBERO_DIR = ROOT / "asset" / "data" / "libero" / "libero_goal"

KEY = "libero_goal"
ACTION_HORIZON = 16   # matches the ALOHA probes, so action R² is comparable
CAMERA = "agentview_rgb"


def task_language(fname: str) -> str:
    """'put_the_bowl_on_the_plate_demo.hdf5' -> 'put the bowl on the plate'.

    Matches the strings the benchmark itself reports for libero_goal (verified
    against benchmark.get_task(i).language for all ten tasks).
    """
    stem = fname[:-len("_demo.hdf5")] if fname.endswith("_demo.hdf5") else Path(fname).stem
    return stem.replace("_", " ")


def load_all() -> dict:
    """Concatenate every task's demos into flat per-frame arrays."""
    files = sorted(LIBERO_DIR.glob("*_demo.hdf5"))
    if not files:
        raise FileNotFoundError(f"no LIBERO demos under {LIBERO_DIR}")

    images, actions, states, episode, frame_in_ep, phase, instr_id = [], [], [], [], [], [], []
    instructions: list[str] = []
    ep_counter = 0

    for fp in files:
        lang = task_language(fp.name)
        instructions.append(lang)
        tid = len(instructions) - 1
        with h5py.File(fp, "r") as f:
            demos = sorted(f["data"], key=lambda s: int(s.split("_")[1]))
            for dname in demos:
                d = f["data"][dname]
                act = np.asarray(d["actions"], dtype=np.float32)          # (T, 7)
                img = np.asarray(d["obs"][CAMERA])[:, ::-1]               # (T,128,128,3) flipped
                st = np.asarray(d["robot_states"], dtype=np.float32)      # (T, 9)
                T = len(act)

                # Action chunk: the next ACTION_HORIZON actions from each frame,
                # edge-padded at the end of the episode. Same construction the
                # ALOHA probes use, so the probe targets are comparable.
                idx = np.minimum(np.arange(T)[:, None] + np.arange(ACTION_HORIZON)[None, :], T - 1)
                chunks = act[idx]                                          # (T, 16, 7)

                images.append(img)
                actions.append(chunks)
                states.append(st)
                episode.append(np.full(T, ep_counter, dtype=np.int64))
                frame_in_ep.append(np.arange(T, dtype=np.int64))
                phase.append((np.arange(T, dtype=np.float32) / max(1, T - 1)))
                instr_id.append(np.full(T, tid, dtype=np.int64))
                ep_counter += 1
        print(f"  {fp.name[:52]:<54} {len(demos)} demos", flush=True)

    return {
        "images": np.concatenate(images),
        "actions": np.concatenate(actions),
        "states": np.concatenate(states),
        "episode": np.concatenate(episode),
        "frame_in_ep": np.concatenate(frame_in_ep),
        "phase": np.concatenate(phase),
        "instr_id": np.concatenate(instr_id),
        "instructions": instructions,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400, help="target probe frames")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"reading {LIBERO_DIR}")
    D = load_all()
    n_frames = len(D["actions"])
    print(f"\ntotal: {n_frames:,} frames  {D['episode'].max()+1:,} episodes  "
          f"{len(D['instructions'])} tasks")

    rng = np.random.default_rng(a.seed)
    idxs = _stratified_indices(D["episode"], a.n, rng)
    eps = np.unique(D["episode"][idxs])
    tasks = np.unique(D["instr_id"][idxs])
    print(f"probe: {len(idxs):,} frames from {len(eps):,} episodes across {len(tasks)} tasks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"probe_{KEY}.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("images", data=D["images"][idxs], compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=D["actions"][idxs].astype(np.float32))
        f.create_dataset("states", data=D["states"][idxs].astype(np.float32))
        f.create_dataset("phase", data=D["phase"][idxs].astype(np.float32))
        f.create_dataset("episode", data=D["episode"][idxs])
        f.create_dataset("frame_in_ep", data=D["frame_in_ep"][idxs])
        f.create_dataset("instr_id", data=D["instr_id"][idxs])
        f.create_dataset("source_idx", data=idxs)
        f.attrs["dataset"] = "libero"
        f.attrs["exp_id"] = "goal"
        f.attrs["key"] = KEY
        f.attrs["instructions"] = json.dumps(D["instructions"])
        f.attrs["n_instructions"] = len(D["instructions"])
        f.attrs["img_h"] = D["images"].shape[1]
        f.attrs["img_w"] = D["images"].shape[2]
        f.attrs["camera"] = CAMERA
        f.attrs["action_horizon"] = ACTION_HORIZON

    print(f"\nwrote {out.name}  images{D['images'][idxs].shape} "
          f"actions{D['actions'][idxs].shape} states{D['states'][idxs].shape}")
    for i, s in enumerate(D["instructions"]):
        print(f"  {i}: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
