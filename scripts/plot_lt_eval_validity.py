"""
scripts/plot_lt_eval_validity.py
────────────────────────────────
Shows the Language Table success metric is VALID (not trivially satisfied), to
answer "did it really succeed?".

Panel A: trained policy SR vs a NO-OP (do-nothing) policy under the exact eval.
         No-op = 0% everywhere → success is earned, not given.
Panel B: steps-to-success per command (trained) — no instant wins; tasks take real
         movement.  point is fastest because it is reach-only (no manipulation).

No-op numbers from scripts/diagnose_lt_success.py (n=25/command; goal-at-reset and
no-op success were both 0/25 for every command).

Output → docs/experiments/language_table/lt_eval_validity.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
RUN = ROOT / "asset/runs/language_table/exp01_baseline"
OUT = ROOT / "docs/experiments/language_table/lt_eval_validity.png"
BLUE, GRAY = "#4C72B0", "#BBBBBB"

# from diagnose_lt_success.py (run bq8jikbcl): noop SR and pre-satisfied both 0/25
NOOP = {"block2block": 0.0, "block2absolute": 0.0, "separate": 0.0, "point": 0.0}
PRESAT = {"block2block": 0, "block2absolute": 0, "separate": 0, "point": 0}
NOOP_N = 25

ood = json.loads((RUN / "ood_results.json").read_text())["by_reward"]
cmds = [c for c in ("block2block", "block2absolute", "separate", "point") if c in ood]
trained = [ood[c]["sr"] * 100 for c in cmds]
noop = [NOOP[c] * 100 for c in cmds]
succ_steps = {c: [e["steps"] for e in ood[c]["episodes"] if e["is_success"]] for c in cmds}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

# A. trained vs no-op
x = np.arange(len(cmds)); w = 0.38
axA.bar(x - w/2, trained, w, color=BLUE, edgecolor="black", lw=0.9, label="trained policy")
axA.bar(x + w/2, noop, w, color=GRAY, edgecolor="black", lw=0.9, label=f"NO-OP (do nothing, n={NOOP_N})")
for xi, v in zip(x - w/2, trained):
    axA.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11, fontweight="bold")
for xi in x + w/2:
    axA.annotate("0%", (xi, 0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10, fontweight="bold", color="#555")
axA.set_xticks(x); axA.set_xticklabels(cmds, fontsize=10, fontweight="bold", rotation=10)
axA.set_ylabel("success rate (%)"); axA.set_ylim(0, 100)
axA.set_title("A. Trained vs do-nothing policy (same eval)\n"
              "No-op = 0% everywhere → success is EARNED, not pre-satisfied\n"
              "(goal already met at reset: 0/25 for every command)",
              fontsize=11.5, fontweight="bold")
axA.legend(fontsize=10); axA.grid(axis="y", alpha=0.3)

# B. steps-to-success
data = [succ_steps[c] for c in cmds]
parts = axB.boxplot(data, labels=cmds, showmeans=True, patch_artist=True)
for patch in parts["boxes"]:
    patch.set_facecolor("#AEC7E8")
for c, d in zip(cmds, data):
    if d:
        axB.annotate(f"min {min(d)}", (cmds.index(c) + 1, min(d)), textcoords="offset points",
                     xytext=(8, -2), fontsize=8.5, color="#333")
axB.set_ylabel("steps to success (trained successes)")
axB.set_title("B. Successes take real movement (no instant wins)\n"
              "fastest is point (reach-only); block tasks need ≥12 steps of pushing",
              fontsize=11.5, fontweight="bold")
axB.grid(axis="y", alpha=0.3)
axB.tick_params(axis="x", rotation=10)

fig.suptitle(
    "Is the Language Table success metric real?  YES — it is the library's own `succeeded` (reward>0), with reset guards.\n"
    "A do-nothing policy scores 0% on every command and no goal is satisfied at reset; successes require 5–95+ steps of motion. "
    "point looks underwhelming because it is REACH-ONLY (no block is pushed) — but it is still earned.",
    fontsize=12, fontweight="bold", y=1.04)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved → {OUT}")
