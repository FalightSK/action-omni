"""RETIRED — superseded by reading every arm at its documented layer.

Do not run this. It is kept only so the retraction it produced remains
reconstructible; `backbones.ARMS` no longer contains `cosmos16` or
`smolvlm2_16`, so every pairing below will fail to load its latents.

Why it is obsolete
──────────────────
This script removed a confound by building an extra arm: stock weights truncated
to the robot arm's depth. The study now removes the same confound at the source —
`backbones.DOC_LAYER` reads each stock control at the layer its own robot
descendant consumes (SmolVLM2 at 16 of 32, Cosmos at 16 of 28), so every pair is
depth-matched by construction and a separate truncated arm would only duplicate
it.

What it established, which still stands
───────────────────────────────────────
    GR00T vs Cosmos      8.38 deg unmatched -> 0.28 deg depth-matched (3% survives)
    SmolVLA vs SmolVLM2 12.81 deg unmatched -> 0.00 deg depth-matched (0% survives)
It is the reason Finding 3 ("robot pretraining separates image and text
subspaces") was retracted: most of the effect was read depth, not finetuning.

The original documentation follows.

The problem this solved
───────────────────────
GR00T N1.7 changed two things about Cosmos-Reason2-2B at once: it finetuned the
weights on robot data, AND it deleted the top 12 of 28 language layers. Comparing
GR00T's last layer (16) against stock Cosmos's last layer (28) therefore confounds
"what robot finetuning did" with "what reading 12 layers earlier does". Either
could produce a difference in subspace angle or task geometry.

`cosmos16` removes the confound: the same stock weights, truncated to exactly
GR00T's 16 layers. GR00T vs cosmos16 differ in the finetuning and nothing else —
same width, same depth, same tokenizer, same image budget, same probe frames.

This is a stronger control than either other pair in the study has. SmolVLA also
truncates (32 → 16) and is compared against a full-depth SmolVLM2, so that pair
carries exactly the confound this script removes for GR00T.

What it reports
───────────────
For each of the three pairings, the within-pair delta in mean image/text
principal angle and in the probe R² values:

    groot vs cosmos     depth-unmatched (16 vs 28) — the conventional pairing
    groot vs cosmos16   depth-matched (16 vs 16)   — the clean causal contrast

If the two rows agree in sign and rough magnitude, the conventional pairing was
not being driven by the depth difference, which also reassures about SmolVLA's
pairing. If they disagree, the depth-matched row is the one to believe.

Output: asset/analysis/latent_compare/depth_matched.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"

KEYS = ["aloha_transfer", "aloha_insertion", "language_table"]
CONTRASTS = [
    ("groot", "cosmos", "depth-unmatched (16 vs 28)"),
    ("groot", "cosmos16", "depth-matched (16 vs 16)"),
    ("pi05", "paligemma", "depth-matched (18 vs 18)"),
    ("smolvla", "smolvlm2", "depth-unmatched (16 vs 32)"),
    ("smolvla", "smolvlm2_16", "depth-matched (16 vs 16)"),
]
METRICS = [
    ("image_text_geometry", "mean_angle_deg", "mean principal angle (deg)"),
    ("factors", "r2_action", "action R2"),
    ("factors", "r2_state", "state R2"),
]


def get(M, key, model, group, field):
    d = M.get(key, {}).get(model, {}).get(group, {})
    if group == "factors":
        d = d.get("all", {})
    v = d.get(field)
    return None if v is None else float(v)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    M = json.loads((DIR / "metrics.json").read_text(encoding="utf-8"))

    out: dict = {}
    for group, field, label in METRICS:
        print(f"\n=== {label} ===")
        print(f"{'contrast':<34}" + "".join(f"{k[:16]:>18}" for k in KEYS))
        out[field] = {}
        for ft, stock, note in CONTRASTS:
            row, cells = {}, ""
            for k in KEYS:
                a, b = get(M, k, ft, group, field), get(M, k, stock, group, field)
                if a is None or b is None:
                    row[k] = None
                    cells += f"{'—':>18}"
                    continue
                row[k] = {"finetuned": a, "stock": b, "delta": a - b}
                cells += f"{a:7.3f}/{b:6.3f}{a-b:+7.3f}"
            out[field][f"{ft}_vs_{stock}"] = {"note": note, "per_dataset": row}
            print(f"{ft + ' vs ' + stock:<22}{note[:11]:<12}" + cells)

    # headline: does the depth-matched contrast agree with the unmatched one?
    # For each family that has both an unmatched and a matched contrast, report
    # how much of the apparent effect survives once depth is held fixed. The
    # ratio matters more than the sign: a matched delta near zero means the
    # "effect" was the depth difference all along.
    ang = out["mean_angle_deg"]
    checks = {}
    for fam, unm, mat in [("groot", "groot_vs_cosmos", "groot_vs_cosmos16"),
                          ("smolvla", "smolvla_vs_smolvlm2", "smolvla_vs_smolvlm2_16")]:
        um = ang.get(unm, {}).get("per_dataset", {})
        dm = ang.get(mat, {}).get("per_dataset", {})
        rows = [(um[k]["delta"], dm[k]["delta"]) for k in KEYS if um.get(k) and dm.get(k)]
        if not rows:
            continue
        mu = sum(abs(u) for u, _ in rows) / len(rows)
        md = sum(abs(d) for _, d in rows) / len(rows)
        checks[fam] = {
            "mean_abs_delta_unmatched_deg": round(mu, 3),
            "mean_abs_delta_matched_deg": round(md, 3),
            "fraction_surviving_depth_match": round(md / mu, 3) if mu > 1e-9 else None,
            "verdict": ("effect survives depth matching" if md > 0.5 * mu else
                        "effect was largely the depth difference, not the finetuning"),
        }
    out["depth_confound_check"] = checks
    print("\n" + json.dumps(out["depth_confound_check"], indent=2))

    (DIR / "depth_matched.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {DIR / 'depth_matched.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
