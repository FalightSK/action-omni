"""
scripts/_lt_make_5instr_subset.py
─────────────────────────────────
Build a STABLE, low-diversity Language Table subset restricted to a handful of
instructions, so the language-conditioned model is not spread across 13,268 unique
commands (~1.4 episodes each) but instead sees a few instructions many times.

Default = the 5 most frequent instructions in the converted dataset (each has
23–54 whole episodes). Because a Language Table instruction is constant within an
episode, filtering by instruction keeps every selected episode intact — no action
chunk is broken.

Output (mirrors the converted-dataset layout so data/language_table/dataset.py and
the LTExp02 config read it unchanged):
  asset/data/language_table_sim_5instr/
    data/chunk-000/file-000.parquet   (filtered rows, original index order)
    meta/stats.json                   (action/state mean·std·min·max RECOMPUTED on subset)
    meta/instructions.json            (the kept instructions + counts, for the record)

No internet required — reads the already-converted local parquet only.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
SRC = ROOT / "asset" / "data" / "language_table_sim"
DST = ROOT / "asset" / "data" / "language_table_sim_5instr"

# The 5 most frequent instructions (by frame count) in the converted dataset.
# Edit this list to choose a different stable instruction set.
KEEP = [
    "push the green star into the green cube",
    "push the yellow pentagon into the yellow star",
    "push the green cube into the green star",
    "push the yellow star into the yellow pentagon",
    "push the blue cube into the blue crescent",
]


def main() -> None:
    shards = sorted(glob.glob(str(SRC / "data" / "chunk-000" / "file-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No source shards under {SRC/'data'/'chunk-000'}")
    keep_set = set(KEEP)

    # ── Filter shard-by-shard (keeps peak RAM low — only matched rows are held) ──
    parts = []
    for i, f in enumerate(shards):
        df = pd.read_parquet(f)
        m = df[df["instruction"].isin(keep_set)]
        if len(m):
            parts.append(m)
        print(f"  shard {i+1:2d}/{len(shards)}: kept {len(m):5d} / {len(df):5d}")
    sub = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)

    n_eps = sub["episode_index"].nunique()
    print(f"\n  subset: {len(sub):,} frames | {n_eps} episodes | "
          f"{sub['instruction'].nunique()} instructions")

    # ── Recompute normalisation stats on the subset (2-D action + state) ─────────
    acts = np.asarray(sub["action"].tolist(), dtype=np.float64)             # (N,2)
    sts  = np.asarray(sub["observation.state"].tolist(), dtype=np.float64)  # (N,2)

    def stat(x):
        return {
            "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "min": x.min(0).tolist(),   "max": x.max(0).tolist(),
        }

    stats = {"action": stat(acts), "observation.state": stat(sts)}

    # ── Write out (mirror the converted-dataset layout) ──────────────────────────
    (DST / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (DST / "meta").mkdir(parents=True, exist_ok=True)

    out_parquet = DST / "data" / "chunk-000" / "file-000.parquet"
    sub.to_parquet(out_parquet, index=False)

    (DST / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    counts = sub["instruction"].value_counts()
    instr_meta = {
        "n_episodes": int(n_eps),
        "n_frames": int(len(sub)),
        "n_instructions": int(sub["instruction"].nunique()),
        "kept_instructions": {s: int(counts[s]) for s in KEEP},
        "source": str(SRC.name),
    }
    (DST / "meta" / "instructions.json").write_text(json.dumps(instr_meta, indent=2))

    mb = out_parquet.stat().st_size / 1e6
    print(f"\n  wrote {out_parquet}  ({mb:.1f} MB)")
    print(f"  wrote {DST/'meta'/'stats.json'}")
    print(f"  action.mean={stats['action']['mean']}  action.std={stats['action']['std']}")
    print("\n  kept instructions (frames):")
    for s in KEEP:
        print(f"    {counts.get(s,0):5d} | {s}")
    print("\n  Next: precompute the .h5 embeddings cache:")
    print("    python scripts/precompute.py --dataset language_table --exp exp02")


if __name__ == "__main__":
    main()
