"""Depth-matched control, extended to the joint task geometry.

depth_matched_control.py handles the per-dataset probe metrics. This one asks
the same question of the pooled geometry: is a robot arm's task/scene ratio
different from its base model's, once depth is held fixed?

It matters because the unmatched numbers look like an effect. GR00T's ratio is
0.50 against stock Cosmos's 0.40 — a +0.10 shift that would read as "robot
pretraining organises the latent more by task". But GR00T is read at layer 16
and Cosmos at layer 28, and the principal-angle result already showed that
comparison to be pure depth artifact. So the ratio has to be re-checked against
Cosmos truncated to 16 layers before the +0.10 can be attributed to finetuning.

Also reports within-pair RSA, which is the width-independent way to ask "how much
did this model change from its base at all".

Output: asset/analysis/latent_compare/depth_matched_geometry.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_joint import _pc_space, load_joint, rdm  # noqa: E402

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
POOL = "image"

# (finetuned, stock, note) — each robot arm against its base at BOTH depths
CONTRASTS = [
    ("groot", "cosmos", "depth-unmatched (16 vs 28)"),
    ("groot", "cosmos16", "depth-matched (16 vs 16)"),
    ("smolvla", "smolvlm2", "depth-unmatched (16 vs 32)"),
    ("smolvla", "smolvlm2_16", "depth-matched (16 vs 16)"),
    ("pi05", "paligemma", "depth-matched (18 vs 18)"),
]


def stats_for(model: str) -> tuple[float, np.ndarray]:
    """Task/scene ratio and the PC-space embedding for one arm."""
    X, y = load_joint(model, POOL)
    Z = _pc_space(X)
    cents = np.stack([Z[y == i].mean(0) for i in range(3)])
    d01 = float(np.linalg.norm(cents[0] - cents[1]))
    d02 = float(np.linalg.norm(cents[0] - cents[2]))
    d12 = float(np.linalg.norm(cents[1] - cents[2]))
    ratio = d01 / ((d02 + d12) / 2)
    return ratio, Z


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    need = sorted({m for c in CONTRASTS for m in c[:2]})
    cache: dict[str, tuple[float, np.ndarray]] = {}
    for m in need:
        try:
            cache[m] = stats_for(m)
            print(f"  loaded {m:<14} task/scene ratio = {cache[m][0]:.3f}", flush=True)
        except Exception as e:
            print(f"  SKIP {m}: {e}", flush=True)

    out: dict = {}
    print(f"\n{'contrast':<30}{'note':<26}{'ratio FT':>9}{'ratio base':>11}"
          f"{'delta':>8}{'RSA':>7}")
    for ft, stock, note in CONTRASTS:
        if ft not in cache or stock not in cache:
            continue
        rf, Zf = cache[ft]
        rs, Zs = cache[stock]
        # RSA between the pair: correlate their frame x frame distance matrices.
        # Sub-sample identically for both so the comparison is over the same
        # frame pairs and stays tractable.
        n = Zf.shape[0]
        idx = np.random.default_rng(0).choice(n, size=min(1200, n), replace=False)
        # rdm() already returns the condensed upper triangle, so these are 1-D
        # vectors of the same length and correlate directly.
        Df, Ds = rdm(Zf[idx]), rdm(Zs[idx])
        rsa = float(stats.spearmanr(Df, Ds).statistic)
        out[f"{ft}_vs_{stock}"] = {
            "note": note, "ratio_finetuned": rf, "ratio_stock": rs,
            "delta": rf - rs, "rsa_within_pair": rsa,
        }
        print(f"{ft + ' vs ' + stock:<30}{note:<26}{rf:>9.3f}{rs:>11.3f}"
              f"{rf - rs:>+8.3f}{rsa:>7.3f}")

    for fam, unm, mat in [("groot", "groot_vs_cosmos", "groot_vs_cosmos16"),
                          ("smolvla", "smolvla_vs_smolvlm2", "smolvla_vs_smolvlm2_16")]:
        if unm in out and mat in out:
            du, dm = abs(out[unm]["delta"]), abs(out[mat]["delta"])
            out.setdefault("verdicts", {})[fam] = {
                "delta_unmatched": out[unm]["delta"],
                "delta_matched": out[mat]["delta"],
                "fraction_surviving": round(dm / du, 3) if du > 1e-9 else None,
                "verdict": ("ratio shift survives depth matching" if dm > 0.5 * du
                            else "ratio shift was largely the depth difference"),
            }

    if "verdicts" in out:
        print("\n" + json.dumps(out["verdicts"], indent=2))
    (DIR / "depth_matched_geometry.json").write_text(json.dumps(out, indent=2),
                                                     encoding="utf-8")
    print(f"\nwrote {DIR / 'depth_matched_geometry.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
