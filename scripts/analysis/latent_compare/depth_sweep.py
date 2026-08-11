"""Where in the language stack does control-relevant information live?

Chapter 2 of the anatomy study, tier 1 (availability).

Two published VLAs independently deleted the top half of their language stack —
SmolVLA 32 -> 16, GR00T 28 -> 16 — and neither justified the cut. This asks
whether the discarded half carried anything a controller would want.

The study already extracted every arm at three relative depths (0.50 / 0.75 /
1.00) but only ever probed the last one, so this needs no GPU and no new
forward passes: the latents are on disk.

Method is deliberately identical to analyze.py's factor probes, so the numbers
are directly comparable to the depth-1.00 column already reported: PCA to the
retained dimensionality, then 5-fold cross-validated ridge to the action chunk,
the proprioceptive state, and episode phase.

Caveat that belongs on every number here: this is *availability*, not
*necessity*. A linear probe says the information is present and linearly
readable; it does not say a trained policy uses it. This project has already
seen those two diverge — a closed-loop ALOHA ablation showed image tokens
carried the control signal while offline loss had overstated the text pathway.
Tier 2 (retrain at each depth, measure closed-loop success) is what settles it.

Output: asset/analysis/latent_compare/depth_sweep.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import _pcs, cv_r2, parallel_analysis  # noqa: E402

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"

from backbones import ARMS as MODELS  # noqa: E402
from backbones import DOC_LAYER, KEYS  # noqa: E402

# The grid, plus each arm's documented read as a separate series. "doc" is not a
# grid fraction — GR00T reads layer 16 of Cosmos's 28 (57%) — so it is carried
# alongside rather than folded in, and plotted as a marker on each arm's curve.
DEPTHS = ["d000", "d025", "d050", "d075", "d100"]
DOC_TAG = "doc"
KV_TAG = "kv"

# Every pool, not just "all".
#
# At 0% the modalities have not been mixed by any attention yet, so they are
# still in their native scales and a mean over all tokens is dominated by
# whichever has the larger norm. Measured at d000:
#     SmolVLM2   image 3.09  text 0.063     (image dominates ~49x)
#     Pi-0.5     image 1.25  text 3.40      (text dominates, Gemma scales
#                                            embeddings by sqrt(2048)=45.25)
# The imbalance INVERTS between architectures, so an "all" curve at 0% is not
# comparable across arms — it would report a modality-weighting artifact as a
# property of the language stack. Probing each pool separately costs three cheap
# ridge fits and removes the ambiguity.
POOLS = ["all", "image", "text"]

# Absolute layer index each relative depth corresponds to, so the curve can be
# read in layers rather than fractions. Derived from DOC_LAYER so it cannot
# disagree with the table the extraction used.
N_LAYERS = {arm: n for arm, (_, n, _, _) in DOC_LAYER.items()}


def load(model: str, key: str, depth: str, pool: str) -> np.ndarray | None:
    p = DIR / f"latents_{model}_{key}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        name = f"{depth}_{pool}"
        return f[name][:] if name in f else None


def _abs_layer(model: str, tag: str) -> int | None:
    """Absolute layer index behind a tag, or None where the idea does not apply.

    'doc' is not a grid fraction, so the old `float(tag[1:]) / 100 * n_layers`
    would raise on it. It reads straight out of DOC_LAYER instead — the only
    value guaranteed to match what extraction actually tapped.

    'kv' spans EVERY layer up to the documented read rather than sitting at one,
    so a single layer index would be a fiction; it returns None.
    """
    if tag == KV_TAG:
        return None
    if tag == DOC_TAG:
        return int(DOC_LAYER[model][0])
    return int(round(float(tag[1:]) / 100 * N_LAYERS[model]))


def probe(key: str) -> dict:
    with h5py.File(DIR / f"probe_{key}.h5", "r") as f:
        return {"actions": f["actions"][:], "states": f["states"][:],
                "phase": f["phase"][:]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out: dict = {}

    for key in KEYS:
        P = probe(key)
        A = P["actions"].reshape(len(P["actions"]), -1)
        S = P["states"].reshape(len(P["states"]), -1)
        ph = P["phase"].astype(np.float64)
        out[key] = {}
        print(f"\n=== {key} ===")
        print(f"{'model':<13}{'depth':<7}{'layer':>6}{'k':>5}"
              f"{'action R2':>11}{'state R2':>10}{'phase R2':>10}")

        for m in MODELS:
            rows = {}
            # DOC_TAG and KV_TAG after the grid so each is probed with exactly
            # the same protocol as the grid points and can be overlaid on it.
            for d in DEPTHS + [DOC_TAG, KV_TAG]:
                by_pool = {}
                for pool in POOLS:
                    X = load(m, key, d, pool)
                    if X is None:
                        continue
                    # Retained dimensionality is recomputed per depth and pool:
                    # k is itself depth-dependent, and freezing it at the
                    # depth-1.00 value would hand some depths more components
                    # than the data supports. It also puts the 9,216-dim KV tap
                    # and the 2,048-dim hidden tap on the same footing, so a KV
                    # advantage cannot be an artifact of raw width.
                    pa = parallel_analysis(X)
                    k = max(2, pa["n_retained"])
                    Z = _pcs(X, k)
                    by_pool[pool] = {
                        "n_retained": int(pa["n_retained"]),
                        "participation_ratio": float(pa["participation_ratio"]),
                        "r2_action": cv_r2(Z, A),
                        "r2_state": cv_r2(Z, S),
                        "r2_phase": cv_r2(Z, ph),
                        "raw_dim": int(X.shape[1]),
                    }
                if not by_pool:
                    continue
                # "all" stays the headline for backward comparability; the other
                # pools sit alongside it rather than replacing it.
                head = by_pool.get("all") or next(iter(by_pool.values()))
                r = dict(head)
                r["layer"] = _abs_layer(m, d)
                r["pools"] = by_pool
                rows[d] = r
                lay = "-" if r["layer"] is None else r["layer"]
                print(f"{m:<13}{d:<7}{lay:>6}{r['n_retained']:>5}"
                      f"{r['r2_action']:>11.3f}{r['r2_state']:>10.3f}"
                      f"{r['r2_phase']:>10.3f}", flush=True)
            if rows:
                # Restricted to the grid: "which depth is best" is a question
                # about the sweep, and letting DOC_TAG compete would answer a
                # different question (does the shipped choice beat the grid?)
                # under the same name. That comparison is reported separately.
                grid = [d for d in rows if d in DEPTHS]
                best = max(grid, key=lambda d: rows[d]["r2_action"])
                last = rows.get("d100", {}).get("r2_action")
                doc = rows.get(DOC_TAG, {}).get("r2_action")
                rows["_summary"] = {
                    "best_depth": best,
                    "best_r2_action": rows[best]["r2_action"],
                    "last_layer_r2_action": last,
                    "doc_r2_action": doc,
                    "doc_layer": _abs_layer(m, DOC_TAG) if doc is not None else None,
                    # positive => the shipped read layer leaves signal on the
                    # table relative to the best grid point
                    "best_minus_doc": (rows[best]["r2_action"] - doc)
                    if doc is not None else None,
                    # positive => an intermediate layer beat the final layer,
                    # i.e. reading at the top is leaving signal behind
                    "gain_over_last": (rows[best]["r2_action"] - last)
                    if last is not None else None,
                }
                out[key][m] = rows

    (DIR / "depth_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== best depth for action decodability ===")
    print(f"{'model':<13}" + "".join(f"{k[:15]:>26}" for k in KEYS))
    for m in MODELS:
        cells = ""
        for key in KEYS:
            s = out.get(key, {}).get(m, {}).get("_summary")
            cells += (f"{s['best_depth']:>8}{s['best_r2_action']:>9.3f}"
                      f"{s['gain_over_last']:>+9.3f}") if s else f"{'—':>26}"
        print(f"{m:<13}{cells}")

    print(f"\nwrote {DIR / 'depth_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
