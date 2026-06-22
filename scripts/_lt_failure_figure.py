"""Regenerate the LT failure-mode figure from failure_modes.json with accurate,
evidence-faithful framing (the first auto-figure had an over-optimistic caption).
CPU-only; no model/sim needed."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
J = json.loads((ROOT / "asset/runs/language_table/exp01_baseline/failure_modes.json").read_text())
OUT = ROOT / "docs/experiments/language_table/lt_failure_modes.png"
THRESH = 0.05
rows = J["rows"]
d0 = np.array([r["d0"] for r in rows]); dmin = np.array([r["d_min"] for r in rows])
succ = np.array([r["success"] for r in rows], bool)
toward = np.array([r["toward"] for r in rows])
n = len(rows); sr = succ.mean()
cats = J["cats"]; med = J["median_d0"]
sr_short, sr_long = J["sr_short"], J["sr_long"]
toward_pos = float((toward > 0.01).mean())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6.0))
fig.subplots_adjust(top=0.80, bottom=0.13, wspace=0.28)

# A — failure-mode decomposition (honest)
order = ["success", "near_miss", "partial", "wrong/idle"]
labs = ["success", "near-miss\n(stall <0.10)", "partial\nprogress", "stalled far\n/ no engage"]
cs = ["#55A868", "#4C72B0", "#DD8452", "#C44E52"]
vals = [cats[k] for k in order]
ax1.bar(range(4), vals, color=cs, edgecolor="black")
for i, v in enumerate(vals):
    ax1.annotate(f"{v}\n{100*v/n:.0f}%", (i, v), textcoords="offset points",
                 xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks(range(4)); ax1.set_xticklabels(labs, fontsize=9)
ax1.set_ylim(0, max(vals) * 1.20)
ax1.set_ylabel("episodes")
ax1.set_title("A. Outcome decomposition (n=%d)\nmost failures stall FAR, few are near-misses" % n,
              fontsize=10.5, fontweight="bold")

# B — d0 vs d_min scatter, colored by success (the distance-limited-control picture)
ax2.scatter(d0[~succ], dmin[~succ], c="#C44E52", s=46, edgecolor="black", label="fail", zorder=3)
ax2.scatter(d0[succ],  dmin[succ],  c="#55A868", s=56, edgecolor="black", label="success", zorder=4)
lim = max(d0.max(), dmin.max()) * 1.05
ax2.plot([0, lim], [0, lim], color="gray", ls=":", lw=1, label="no progress (d_min=d0)")
ax2.axhline(THRESH, color="black", ls="--", lw=1.4, label=f"success tol {THRESH}")
ax2.set_xlabel("required push distance d0"); ax2.set_ylabel("closest approach d_min")
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_title("B. Closest approach vs required distance\nsuccesses cluster at small d0; long pushes stall high",
              fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=8, loc="upper left")

# C — SR vs required push distance (the decisive evidence)
ax3.bar([0, 1], [sr_short*100, sr_long*100], color=["#55A868", "#C44E52"], edgecolor="black")
for i, v in [(0, sr_short*100), (1, sr_long*100)]:
    ax3.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=13, fontweight="bold")
ax3.set_xticks([0, 1]); ax3.set_xticklabels([f"short push\n(d0<={med:.2f})", f"long push\n(d0>{med:.2f})"], fontsize=9.5)
ax3.set_ylabel("success rate (%)"); ax3.set_ylim(0, 100)
ax3.set_title("C. SR collapses with push distance\n(short %.0f%% -> long %.0f%%)" % (sr_short*100, sr_long*100),
              fontsize=10.5, fontweight="bold")

fig.suptitle(
    f"Why is LT exp01 SR only {sr*100:.0f}%?  Grounding works — the correct block is pushed toward the target in "
    f"{toward_pos*100:.0f}% of episodes —\n"
    f"but closed-loop control is DISTANCE-LIMITED: short pushes succeed {sr_short*100:.0f}%, long pushes only {sr_long*100:.0f}%; "
    f"most failures stall far short of the {THRESH} goal (median d_min={np.median(dmin[~succ]):.2f}).\n"
    f"Bottleneck = sustained long-range pushing, not language understanding.",
    fontsize=12, fontweight="bold", y=0.99)
plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved -> {OUT}")
print(f"toward_pos={toward_pos:.2f} sr_short={sr_short:.2f} sr_long={sr_long:.2f} median_fail_dmin={np.median(dmin[~succ]):.3f}")
