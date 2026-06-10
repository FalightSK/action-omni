"""
scripts/plot_lt_taskgen.py
──────────────────────────
Plan A result figure (honest framing).

Trained BALANCED (40%, 1 epoch on-the-fly) on block2block/absolute/relative; evaluated
on those SEEN tasks vs UNSEEN families (separate/point/between/corner).

Key point: the raw seen-vs-unseen mean is a TASK-DIFFICULTY artifact — every seen task
needs precise placement, while the unseen set is inflated by the easy reach/push-apart
tasks (separate/point). So panels group by SKILL and compare against the original 5k
(exp01, 150-epoch) model on overlapping tasks.

Reads asset/runs/language_table/exp02_balanced/taskgen_results.json
      asset/runs/language_table/exp01_baseline/ood_results.json   (for the comparison)
Output docs/experiments/language_table/lt_taskgen.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parents[1]
RES = ROOT / "asset/runs/language_table/exp02_balanced/taskgen_results.json"
OOD = ROOT / "asset/runs/language_table/exp01_baseline/ood_results.json"
OUT = ROOT / "docs/experiments/language_table/lt_taskgen.png"
BLUE, ORANGE, GREEN = "#4C72B0", "#DD8452", "#55A868"

SKILL = {"block2block": "precise", "block2absolute": "precise", "block2relative": "precise",
         "between": "precise", "corner": "medium", "separate": "easy", "point": "easy"}
SKILL_COL = {"precise": "#C44E52", "medium": "#CCB974", "easy": "#55A868"}


def main():
    j = json.loads(RES.read_text())
    tasks = j["by_task"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw={"width_ratios": [1.5, 1.1, 1.1]})

    # A. per-task SR, coloured by seen/unseen, skill noted
    labels = [t["factory"] for t in tasks]; srs = [t["sr"] * 100 for t in tasks]
    kinds = [t["kind"] for t in tasks]
    cols = [BLUE if k == "seen" else ORANGE for k in kinds]
    x = np.arange(len(labels))
    axes[0].bar(x, srs, color=cols, edgecolor="black", lw=0.9)
    for xi, (s, t) in enumerate(zip(srs, tasks)):
        axes[0].annotate(f"{s:.0f}%", (xi, s), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
        axes[0].text(xi, -6, SKILL[t["factory"]], ha="center", va="top", fontsize=8,
                     color=SKILL_COL[SKILL[t["factory"]]], fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{l}\n[{k}]" for l, k in zip(labels, kinds)], fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("success rate (%)"); axes[0].set_ylim(-16, 100); axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("A. SR by task — colour=seen/unseen, label=skill needed\nALL precise tasks fail (2-8%) whether seen or unseen; only easy tasks (separate/point) succeed",
                      fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(handles=[Patch(facecolor=BLUE, edgecolor="black", label="seen (trained)"),
                            Patch(facecolor=ORANGE, edgecolor="black", label="unseen (held-out)")], fontsize=8.5)

    # B. group by SKILL (the real axis)
    by_skill = {}
    for t in tasks:
        by_skill.setdefault(SKILL[t["factory"]], []).append(t["sr"] * 100)
    order = ["precise", "medium", "easy"]
    means = [np.mean(by_skill.get(s, [0])) for s in order]
    axes[1].bar(range(len(order)), means, color=[SKILL_COL[s] for s in order], edgecolor="black", lw=0.9)
    for i, (s, m) in enumerate(zip(order, means)):
        axes[1].annotate(f"{m:.0f}%\n(n={len(by_skill.get(s,[]))} tasks)", (i, m), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(["precise\nplacement", "medium", "easy\n(reach/spread)"], fontsize=9.5, fontweight="bold")
    axes[1].set_ylabel("mean success rate (%)"); axes[1].set_ylim(0, 100)
    axes[1].set_title("B. The REAL axis is skill/difficulty\n(precise placement fails; easy tasks succeed)", fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    # C. balanced (exp02, 40% 1ep) vs original 5k (exp01, 150ep) on overlapping tasks
    by02 = {t["factory"]: t["sr"] * 100 for t in tasks}
    ex01 = {}
    if OOD.exists():
        o = json.loads(OOD.read_text())["by_reward"]
        ex01 = {k: o[k]["sr"] * 100 for k in o}
    comp = [("block2block", "precise"), ("block2absolute", "precise"), ("separate", "easy"), ("point", "easy")]
    comp = [(c, sk) for c, sk in comp if c in by02 and c in ex01]
    cx = np.arange(len(comp)); w = 0.38
    e1 = [ex01[c] for c, _ in comp]; e2 = [by02[c] for c, _ in comp]
    axes[2].bar(cx - w/2, e1, w, color="#999999", edgecolor="black", label="exp01: 5k narrow, 150 ep")
    axes[2].bar(cx + w/2, e2, w, color=BLUE, edgecolor="black", label="exp02: balanced 40%, 1 ep")
    for xi, v in zip(cx - w/2, e1):
        axes[2].annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, fontweight="bold")
    for xi, v in zip(cx + w/2, e2):
        axes[2].annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, fontweight="bold")
    axes[2].set_xticks(cx); axes[2].set_xticklabels([c for c, _ in comp], rotation=20, ha="right", fontsize=8.5)
    axes[2].set_ylabel("success rate (%)"); axes[2].set_ylim(0, 100)
    axes[2].set_title("C. Balanced 40%/1ep vs original 5k/150ep\neasy tasks ↑ (better generalization); precise tasks ↓", fontsize=10, fontweight="bold")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Plan A — balanced 40% (block2block/absolute/relative), 1-epoch on-the-fly, then seen vs unseen task eval (n=50/task)\n"
        "Honest read: PRECISE PLACEMENT still fails (2-8%, seen OR unseen) — 12× more balanced data did NOT fix it (it's a resolution bottleneck). "
        "But untrained EASY tasks generalize BETTER than before (separate 46→70%, point 74→86%): more task diversity helps language→skill transfer.",
        fontsize=11, fontweight="bold", y=1.03)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
