"""Fold per-arm metrics_<arm>.json shards into a single metrics.json.

analyze.py parallelises by arm: the (arm, dataset) cells share nothing except
the output file, so each worker writes its own shard via --out and this folds
them back together. Serial `analyze.py` with no --out still writes metrics.json
directly and does not need this step.

Structure is {dataset: {arm: {...}}}, so merging is a two-level update. A given
(dataset, arm) must come from exactly one shard; a collision means two workers
were given overlapping --models and one silently overwrote the other, so it is
an error rather than a last-writer-wins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import ARMS  # noqa: E402

DIR = Path(__file__).resolve().parents[3] / "asset" / "analysis" / "latent_compare"


def main() -> int:
    # Look up one shard per arm in the CURRENT roster, rather than globbing
    # metrics_*.json. A glob also matches leftovers from earlier runs -- this
    # directory still holds metrics_shard_*.json from a roster that contained
    # cosmos16 and smolvlm2_16 -- and merging those would quietly reintroduce
    # arms that were deliberately removed, at read depths that no longer apply.
    shards = []
    missing = []
    for arm in ARMS:
        p = DIR / f"metrics_{arm}.json"
        (shards if p.exists() else missing).append(p)
    if missing:
        print(f"missing shards for: {', '.join(p.stem[8:] for p in missing)}",
              file=sys.stderr)
        return 1
    if not shards:
        print("no shards found", file=sys.stderr)
        return 1

    merged: dict = {}
    seen: dict[tuple[str, str], str] = {}
    for sp in shards:
        data = json.loads(sp.read_text(encoding="utf-8"))
        for key, arms in data.items():
            for arm, payload in arms.items():
                prev = seen.get((key, arm))
                if prev is not None:
                    print(f"collision: {key}/{arm} in both {prev} and {sp.name}",
                          file=sys.stderr)
                    return 1
                seen[(key, arm)] = sp.name
                merged.setdefault(key, {})[arm] = payload

    out = DIR / "metrics.json"
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    n_cells = sum(len(v) for v in merged.values())
    print(f"merged {len(shards)} shards -> {out.name}  "
          f"({len(merged)} datasets, {n_cells} cells)")
    for key in sorted(merged):
        print(f"  {key:<18} {len(merged[key])} arms: {' '.join(sorted(merged[key]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
