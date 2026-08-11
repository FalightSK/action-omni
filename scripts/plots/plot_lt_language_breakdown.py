"""
scripts/plot_lt_language_breakdown.py
─────────────────────────────────────
Language-conditioned RESULT analysis for Language Table (mirrors the ALOHA breakdown
style).  Parses the per-episode instructions in the eval results and asks: does
success depend on the object COLOR / SHAPE the command references, and how decisively
does the model succeed?

Panels:
  A. Success rate by command type (block2block / block2absolute / separate / point)
  B. Success rate by referenced object COLOR (pooled over all commands)
  C. Success rate by referenced object SHAPE (pooled over all commands)
  D. Steps-to-success distribution per command type (how fast it solves)

Reads asset/runs/language_table/exp01_baseline/{ood_results,control_results}.json
Output → docs/experiments/language_table/lt_language_breakdown.png
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
RUN = ROOT / "asset/runs/language_table/exp01_baseline"
OUT = ROOT / "docs/experiments/language_table/lt_language_breakdown.png"

COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["cube", "moon", "star", "pentagon", "crescent"]
COL_HEX = {"red": "#C44E52", "green": "#55A868", "blue": "#4C72B0", "yellow": "#CCB974"}
CMD_HEX = {"block2block": "#4C72B0", "block2absolute": "#6FA8DC",
           "separate": "#DD8452", "point": "#E8A87C"}
_OBJ = re.compile(r"(red|green|blue|yellow)\s+(cube|moon|star|pentagon|crescent)")


def first_object(instr: str):
    m = _OBJ.search(instr.lower())
    return (m.group(1), m.group(2)) if m else (None, None)


ood = json.loads((RUN / "ood_results.json").read_text())
by = ood["by_reward"]
order = [r for r in ("block2block", "block2absolute", "separate", "point") if r in by]

# flatten episodes with command label
rows = []  # (cmd, success, steps, color, shape)
for r in order:
    for e in by[r]["episodes"]:
        c, s = first_object(e.get("instruction", ""))
        rows.append((r, bool(e["is_success"]), int(e["steps"]), c, s))

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axA, axB, axC, axD = axes.ravel()


def sr_bar(ax, keys, key_idx, palette, title, xlabel):
    srs, ns, cols = [], [], []
    for k in keys:
        ep = [r for r in rows if r[key_idx] == k]
        n = len(ep)
        sr = 100 * np.mean([r[1] for r in ep]) if n else 0
        srs.append(sr); ns.append(n); cols.append(palette.get(k, "#888"))
    x = np.arange(len(keys))
    ax.bar(x, srs, color=cols, edgecolor="black", lw=0.9, width=0.7)
    for xi, (sr, n) in enumerate(zip(srs, ns)):
        ax.annotate(f"{sr:.0f}%\n(n={n})", (xi, sr), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=10)
    ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel(xlabel)


sr_bar(axA, order, 0, CMD_HEX, "A. Success by command type\n(task difficulty differs)", "command type")
sr_bar(axB, COLORS, 3, COL_HEX, "B. Success by referenced object COLOR\n(pooled over all commands)", "object color")
sr_bar(axC, SHAPES, 4, {s: "#8172B3" for s in SHAPES},
       "C. Success by referenced object SHAPE\n(pooled over all commands)", "object shape")

# D. steps-to-success distribution per command
for r in order:
    succ_steps = [e["steps"] for e in by[r]["episodes"] if e["is_success"]]
    if succ_steps:
        axD.hist(succ_steps, bins=np.arange(0, 205, 15), alpha=0.6,
                 label=f"{r} (n={len(succ_steps)})", color=CMD_HEX.get(r, "#888"), edgecolor="black", lw=0.5)
axD.set_xlabel("steps to success"); axD.set_ylabel("# successful episodes")
axD.set_title("D. How fast does it succeed?\n(successful episodes only; fewer steps = more decisive)",
              fontsize=12, fontweight="bold")
axD.legend(fontsize=9); axD.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Qwen-DiT VLA on Language Table — success broken down by the LANGUAGE of the command (n=50/command)\n"
    "Color (B) is even (~28-40%); the rare crescent shape (C) is a weak spot (0/11, small n). "
    "Success is gated mainly by TASK type (A), not by which block is named.",
    fontsize=13, fontweight="bold", y=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved → {OUT}")
