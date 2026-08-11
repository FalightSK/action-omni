"""
scripts/plot_lt_why_ood.py
──────────────────────────
WHY do the held-out (OOD) commands score HIGHER than the in-distribution ones?
Evidence that it is TASK DIFFICULTY, not better extrapolation.

Argument (three panels, all from existing eval JSON):
  A. The four tasks form a ladder of PLACEMENT-PRECISION required, and SR falls
     monotonically along it.  point (reach only) > separate (push apart, any dir)
     > block2absolute (push to a region) > block2block (push onto a specific block).
     The model HAS the reach skill (point 74%) but loses success as the task demands
     more precise transport (block2block 8%).  The OOD tasks need only the easy skill.
  B. Difficulty-matched control: for the SAME task, swapping the trained verb for the
     held-out verb barely changes SR (reach: move-arm 64% ≈ point 74%).  So "OOD-ness"
     is NOT the driver — the task's success criterion is.
  C. SR vs mean-steps-to-success: easy tasks succeed fast AND often; hard tasks mostly
     time out.  Speed-of-success is an independent difficulty signal that tracks SR.

Output → docs/experiments/language_table/lt_why_ood.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
RUN = ROOT / "asset/runs/language_table/exp01_baseline"
OUT = ROOT / "docs/experiments/language_table/lt_why_ood.png"
BLUE, ORANGE = "#4C72B0", "#DD8452"

ood = json.loads((RUN / "ood_results.json").read_text())["by_reward"]
ctrl = json.loads((RUN / "control_results.json").read_text())["conditions"]

# ladder of increasing placement precision required
LADDER = [
    ("point",          "reach effector to the block\n(no manipulation)",          "OOD"),
    ("separate",       "push two blocks apart\n(any direction works)",            "OOD"),
    ("block2absolute", "push a block into a\nboard REGION",                       "in-dist"),
    ("block2block",    "push a block onto a\nSPECIFIC other block",               "in-dist"),
]
labels = [l[0] for l in LADDER]
srs = [ood[l[0]]["sr"] * 100 for l in LADDER]
crit = [l[1] for l in LADDER]
kinds = [l[2] for l in LADDER]
steps = [ood[l[0]]["mean_steps"] for l in LADDER]
cols = [ORANGE if k == "OOD" else BLUE for k in kinds]

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})

# ── A. precision ladder ─────────────────────────────────────────────────────────
x = np.arange(len(labels))
axA.bar(x, srs, color=cols, edgecolor="black", lw=1.0, width=0.66)
for xi, (sr, c, k) in enumerate(zip(srs, crit, kinds)):
    axA.annotate(f"{sr:.0f}%", (xi, sr), textcoords="offset points", xytext=(0, 5),
                 ha="center", fontsize=13, fontweight="bold")
    axA.text(xi, 4, c, ha="center", va="bottom", fontsize=8.2, color="#222")
axA.plot(x, srs, "k--o", lw=1.4, ms=5, alpha=0.7)
axA.annotate("", xy=(3.35, 96), xytext=(-0.35, 96),
             arrowprops=dict(arrowstyle="-|>", lw=2, color="#444"))
axA.text(1.5, 99, "increasing placement precision required  →", ha="center",
         fontsize=10, fontweight="bold", color="#444")
axA.set_xticks(x)
axA.set_xticklabels([f"{l}\n[{k}]" for l, k in zip(labels, kinds)], fontsize=10, fontweight="bold")
axA.set_ylabel("success rate (%)"); axA.set_ylim(0, 108)
axA.set_title("A. Success falls monotonically with the PRECISION the task demands\n"
              "model can reach (point 74%) but not precisely place (block2block 8%)",
              fontsize=11.5, fontweight="bold")
axA.grid(axis="y", alpha=0.3)

# ── B. difficulty-matched control ──────────────────────────────────────────────
def c_sr(factory, kind):
    return next(c["sr"] * 100 for c in ctrl if c["factory"] == factory and c["verb_kind"] == kind)
groups = [("reach\n(point task)", "point"), ("push-apart\n(separate task)", "separate")]
gx = np.arange(len(groups)); w = 0.36
tr = [c_sr(f, "trained") for _, f in groups]
ho = [c_sr(f, "held_out") for _, f in groups]
axB.bar(gx - w/2, tr, w, color=BLUE, edgecolor="black", lw=0.9, label="trained verb")
axB.bar(gx + w/2, ho, w, color=ORANGE, edgecolor="black", lw=0.9, label="held-out verb")
for xi, v in zip(gx - w/2, tr):
    axB.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11, fontweight="bold")
for xi, v in zip(gx + w/2, ho):
    axB.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11, fontweight="bold", color="#8a4b1f")
axB.set_xticks(gx); axB.set_xticklabels([g[0] for g in groups], fontsize=9.5, fontweight="bold")
axB.set_ylabel("success rate (%)"); axB.set_ylim(0, 100)
axB.set_title("B. Same task, verb swapped\nSR set by the TASK, not by seen/unseen verb",
              fontsize=11.5, fontweight="bold")
axB.legend(fontsize=9); axB.grid(axis="y", alpha=0.3)

# ── C. SR vs speed-of-success ──────────────────────────────────────────────────
for l, sr, st, k in zip(labels, srs, steps, kinds):
    axC.scatter(st, sr, s=180, color=(ORANGE if k == "OOD" else BLUE), edgecolor="black", lw=1.0, zorder=3)
    axC.annotate(l, (st, sr), textcoords="offset points", xytext=(6, 5), fontsize=9.5, fontweight="bold")
axC.set_xlabel("mean steps to terminate (↑ = mostly timing out = harder)")
axC.set_ylabel("success rate (%)"); axC.set_ylim(0, 100)
axC.invert_xaxis()
axC.set_title("C. Fast-success tasks succeed more\n(speed is an independent difficulty signal)",
              fontsize=11.5, fontweight="bold")
axC.grid(alpha=0.3)

fig.suptitle(
    "Why do held-out (OOD) commands beat in-distribution ones?  →  TASK DIFFICULTY, not superior extrapolation.\n"
    "The unseen-verb tasks (point, separate) only require REACH / loose pushing — the sub-skill the model already has; "
    "the in-dist tasks (block2block/absolute) require PRECISE placement, which it lacks. The verb itself is nearly free (panel B).",
    fontsize=12.5, fontweight="bold", y=1.04)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved → {OUT}")
