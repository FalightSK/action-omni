"""
Step 1/3 of the cross-backbone latent study — build the shared probe set.

WHY this exists as a separate step: the whole comparison is only meaningful if
every backbone sees *byte-identical* inputs. If each extraction script sampled
its own frames, differences in the latent geometry could come from the sample
rather than the model. So we freeze one probe set to disk first, and every
extractor reads from it.

The probe is stratified over the two factors we later want to explain variance
with: episode identity and normalised phase within the episode. Uniform random
sampling would over-represent whatever phase happens to dominate the frame count
(for ALOHA, the long approach segment), which would bias the temporal analysis.

Output: asset/analysis/latent_compare/probe_<dataset>.h5
  images       (N, H, W, 3) uint8   the exact PIL frame the project feeds Qwen
  actions      (N, H_a, D)  float32 normalised action chunk
  states       (N, D_s)     float32 normalised state
  phase        (N,)         float32 t / (T-1) within episode
  episode      (N,)         int64
  frame_in_ep  (N,)         int64
  instr_id     (N,)         int64   index into the `instructions` attr
  attrs: dataset, exp_id, instructions (json list), img_w, img_h
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config  # noqa: E402

OUT_DIR = ROOT / "asset" / "analysis" / "latent_compare"

# (dataset, exp_id) pairs that have a surviving best-config in the registry.
TARGETS = {
    "aloha_transfer": ("aloha", "exp01"),
    "aloha_insertion": ("aloha", "exp03"),
    "language_table": ("language_table", "exp03"),
}


def _load_raw(dataset: str, cfg):
    if dataset == "aloha":
        from data.aloha import AlohaDataset

        return AlohaDataset(cfg, cache_frames=False)
    if dataset == "language_table":
        from data.language_table import LanguageTableDataset

        return LanguageTableDataset(cfg, cache_frames=False)
    raise ValueError(f"unsupported dataset: {dataset}")


def _stratified_indices(episode_ids: np.ndarray, n_target: int, rng) -> np.ndarray:
    """Pick frames spread evenly over phase within evenly-spread episodes.

    Rationale: we want the temporal factor (phase) and the episode factor to be
    close to orthogonal in the probe, so that variance attributed to one is not
    silently carried by the other.
    """
    by_ep: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(episode_ids):
        by_ep[int(e)].append(i)

    eps = sorted(by_ep)
    # Cap episodes so each contributes several phase points rather than one frame
    # from a huge number of episodes (which would leave phase under-sampled).
    # When the dataset has few but long episodes (ALOHA: 50 episodes x ~400
    # frames), invert the split and take proportionally more frames per episode
    # so we still reach n_target.
    min_per_ep = 12
    n_eps = max(1, min(len(eps), n_target // min_per_ep))
    per_ep = max(min_per_ep, -(-n_target // n_eps))  # ceil division
    chosen_eps = [eps[i] for i in np.linspace(0, len(eps) - 1, n_eps).astype(int)]

    picked: list[int] = []
    for e in chosen_eps:
        frames = by_ep[e]
        take = min(per_ep, len(frames))
        sel = np.linspace(0, len(frames) - 1, take).astype(int)
        picked.extend(frames[s] for s in sel)

    picked = np.array(sorted(set(picked)), dtype=np.int64)
    if len(picked) > n_target:
        picked = picked[np.sort(rng.choice(len(picked), n_target, replace=False))]
    return picked


def build(key: str, n_target: int, seed: int) -> Path:
    dataset, exp_id = TARGETS[key]
    cfg = get_config(dataset, exp_id)
    ds = _load_raw(dataset, cfg)
    rng = np.random.default_rng(seed)

    episode_ids = ds.episode_ids
    idxs = _stratified_indices(episode_ids, n_target, rng)
    print(f"[{key}] {len(idxs):,} probe frames from {len(np.unique(episode_ids[idxs])):,} episodes")

    # phase = position within the episode, normalised to [0, 1]
    ep_first: dict[int, int] = {}
    ep_len: dict[int, int] = {}
    for e in np.unique(episode_ids):
        w = np.nonzero(episode_ids == e)[0]
        ep_first[int(e)] = int(w[0])
        ep_len[int(e)] = int(len(w))

    samples = [ds[int(i)] for i in idxs]

    instructions: list[str] = []
    instr_lookup: dict[str, int] = {}
    instr_id = np.zeros(len(samples), dtype=np.int64)
    for j, s in enumerate(samples):
        t = s["task_text"]
        if t not in instr_lookup:
            instr_lookup[t] = len(instructions)
            instructions.append(t)
        instr_id[j] = instr_lookup[t]

    images = np.stack([np.asarray(s["image"], dtype=np.uint8) for s in samples])
    actions = np.stack([s["actions"].numpy() for s in samples]).astype(np.float32)
    states = np.stack([s["state"].numpy() for s in samples]).astype(np.float32)

    ep_arr = episode_ids[idxs].astype(np.int64)
    frame_in_ep = np.array([int(i) - ep_first[int(e)] for i, e in zip(idxs, ep_arr)], dtype=np.int64)
    phase = np.array(
        [f / max(1, ep_len[int(e)] - 1) for f, e in zip(frame_in_ep, ep_arr)], dtype=np.float32
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"probe_{key}.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=actions)
        f.create_dataset("states", data=states)
        f.create_dataset("phase", data=phase)
        f.create_dataset("episode", data=ep_arr)
        f.create_dataset("frame_in_ep", data=frame_in_ep)
        f.create_dataset("instr_id", data=instr_id)
        f.create_dataset("source_idx", data=idxs)
        f.attrs["dataset"] = dataset
        f.attrs["exp_id"] = exp_id
        f.attrs["key"] = key
        f.attrs["instructions"] = json.dumps(instructions)
        f.attrs["n_instructions"] = len(instructions)
        f.attrs["img_h"] = images.shape[1]
        f.attrs["img_w"] = images.shape[2]

    print(
        f"[{key}] wrote {out.name}  images{images.shape} actions{actions.shape} "
        f"| {len(instructions)} unique instruction(s)"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--n", type=int, default=2400, help="probe frames per dataset")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k in a.keys:
        build(k, a.n, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
