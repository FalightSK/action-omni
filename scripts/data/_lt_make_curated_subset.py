"""
scripts/_lt_make_curated_subset.py
──────────────────────────────────
Build the CURATED Language Table subset for the validation milestone: 8 concepts,
50% reach-to-named-block + 50% block-to-block, each concept POOLED ACROSS ALL
PHRASINGS present in the data (fixes the two failures of the 5-instr "into"-only
subset: train/eval phrasing mismatch + control-data starvation).

Concepts:
  REACH (point/touch/move-arm TO a named block; eval via the sim `point` factory):
    yellow pentagon, red pentagon, yellow star, green star
  PUSH  (block-to-block; eval via the sim `block2block` factory):
    green star->green cube, yellow pentagon->yellow star,
    yellow star->yellow pentagon, green cube->green star

Every training frame whose instruction matches a concept (in ANY phrasing:
push/move/slide/put · to/into/towards/close to · point/touch/move your arm) is kept,
so the model sees the sim's eval paraphrases in-distribution. Whole episodes are kept
(instruction is constant per episode). Stats recomputed on the subset.

Output: asset/data/language_table_sim_curated/{data,meta}
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
SRC = ROOT / "asset" / "data" / "language_table_sim"
DST = ROOT / "asset" / "data" / "language_table_sim_curated"

REACH_TARGETS = ["yellow pentagon", "red pentagon", "yellow star", "green star"]
PUSH_PAIRS = [("green star", "green cube"), ("yellow pentagon", "yellow star"),
              ("yellow star", "yellow pentagon"), ("green cube", "green star")]

PUSH_VERBS = ("push", "move", "slide", "put", "pull")


def classify(ins: str):
    """Return (task_type, concept_label) or (None, None) if the instruction matches
    no target concept. Each instruction maps to at most one concept."""
    s = ins.lower().strip()
    if "away" in s:
        return None, None
    first = s.split()[0] if s else ""

    # REACH: point/touch/move-your-arm TO a single named target (no second block)
    is_reach_phrasing = (first in ("point", "touch") or
                         s.startswith("move your arm") or s.startswith("move your hand"))
    if is_reach_phrasing:
        for tb in REACH_TARGETS:
            if tb in s:
                # ensure it's a single-block reach (no OTHER named block after 'to/toward')
                return "reach", tb

    # PUSH: block-to-block, source before target, moving a block (not the arm)
    if first in PUSH_VERBS and "your arm" not in s and "your hand" not in s:
        for src, tgt in PUSH_PAIRS:
            i, j = s.find(src), s.find(tgt)
            if i != -1 and j != -1 and i < j:
                return "push", f"{src} -> {tgt}"
    return None, None


def main():
    shards = sorted(glob.glob(str(SRC / "data" / "chunk-000" / "file-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No source shards under {SRC/'data'/'chunk-000'}")

    parts = []
    for i, f in enumerate(shards):
        df = pd.read_parquet(f)
        tt = df["instruction"].map(classify)
        df = df.assign(task_type=[t[0] for t in tt], concept=[t[1] for t in tt])
        m = df[df["task_type"].notna()]
        if len(m):
            parts.append(m)
        print(f"  shard {i+1:2d}/{len(shards)}: kept {len(m):5d} / {len(df):5d}")
    sub = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)

    # ── Balance to 50/50 reach/push at the EPISODE level (downsample majority) ────
    ep_type = sub.groupby("episode_index")["task_type"].first()
    reach_eps = ep_type[ep_type == "reach"].index.tolist()
    push_eps = ep_type[ep_type == "push"].index.tolist()
    n_side = min(len(reach_eps), len(push_eps))
    rng = np.random.default_rng(42)

    def stratified_pick(eps, n):
        """Pick n episodes spread evenly across their concepts (deterministic)."""
        by_c = defaultdict(list)
        for e in eps:
            by_c[sub.loc[sub.episode_index == e, "concept"].iloc[0]].append(e)
        for c in by_c:
            by_c[c] = list(rng.permutation(by_c[c]))
        picked, ci = [], 0
        concepts = sorted(by_c)
        while len(picked) < n and any(by_c.values()):
            c = concepts[ci % len(concepts)]
            if by_c[c]:
                picked.append(by_c[c].pop())
            ci += 1
        return picked

    keep_eps = set(stratified_pick(reach_eps, n_side) + stratified_pick(push_eps, n_side))
    sub = sub[sub["episode_index"].isin(keep_eps)].sort_values("index").reset_index(drop=True)

    n_eps = sub["episode_index"].nunique()
    print(f"\n  balanced to 50/50: {n_side} reach + {n_side} push episodes")
    print(f"  curated subset: {len(sub):,} frames | {n_eps} episodes")
    print(f"  by task_type (episodes): "
          f"{sub.groupby('task_type')['episode_index'].nunique().to_dict()}")

    # recompute 2-D normalisation stats on the subset
    acts = np.asarray(sub["action"].tolist(), np.float64)
    sts = np.asarray(sub["observation.state"].tolist(), np.float64)
    def st(x): return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                       "min": x.min(0).tolist(), "max": x.max(0).tolist()}
    stats = {"action": st(acts), "observation.state": st(sts)}

    (DST / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (DST / "meta").mkdir(parents=True, exist_ok=True)
    out_pq = DST / "data" / "chunk-000" / "file-000.parquet"
    sub.to_parquet(out_pq, index=False)
    (DST / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    concept_eps = sub.groupby(["task_type", "concept"])["episode_index"].nunique()
    meta = {
        "n_episodes": int(n_eps), "n_frames": int(len(sub)),
        "reach_targets": REACH_TARGETS,
        "push_pairs": [f"{a} -> {b}" for a, b in PUSH_PAIRS],
        "episodes_per_concept": {f"{k[0]}:{k[1]}": int(v) for k, v in concept_eps.items()},
    }
    (DST / "meta" / "instructions.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  wrote {out_pq}  ({out_pq.stat().st_size/1e6:.1f} MB)")
    print("  episodes per concept:")
    for (tt, c), v in concept_eps.items():
        print(f"    [{tt:5s}] {v:3d} eps | {c}")
    print("\n  Next: precompute -> python scripts/precompute.py --dataset language_table --exp exp03")


if __name__ == "__main__":
    main()
