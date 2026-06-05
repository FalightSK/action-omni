"""
scripts/plot_lt_dataset_diversity.py
────────────────────────────────────
Characterises the converted Language Table TRAINING set (mirrors the ALOHA
dataset-diversity figure).  Shows the language + episode diversity the model was
trained on, and what was deliberately HELD OUT.

Panels:
  A. Command-template distribution of the 5k training episodes (+ held-out verbs)
  B. Object vocabulary coverage — color × shape co-occurrence heatmap
  C. Episode-length distribution
  D. Instruction word-count distribution + dataset summary

Reads asset/data/language_table_sim/meta/{episodes,instructions}.json
Output → docs/experiments/language_table/lt_dataset_diversity.png
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
META = ROOT / "asset/data/language_table_sim/meta"
OUT = ROOT / "docs/experiments/language_table/lt_dataset_diversity.png"

COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["cube", "moon", "star", "pentagon", "crescent"]
_OBJ = re.compile(r"(red|green|blue|yellow)\s+(cube|moon|star|pentagon|crescent)")

eps = json.loads((META / "episodes.json").read_text())
info = json.loads((META / "instructions.json").read_text())

templates = Counter(e["template"] for e in eps)
lengths = np.array([e["n_steps"] for e in eps])
wordcounts = np.array([len(e["instruction"].split()) for e in eps])

# color × shape co-occurrence (first object referenced)
mat = np.zeros((len(COLORS), len(SHAPES)), dtype=int)
for e in eps:
    m = _OBJ.search(e["instruction"].lower())
    if m:
        mat[COLORS.index(m.group(1)), SHAPES.index(m.group(2))] += 1

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axA, axB, axC, axD = axes.ravel()

# A. template distribution
torder = [t for t, _ in templates.most_common()]
vals = [templates[t] for t in torder]
axA.bar(range(len(torder)), vals, color="#4C72B0", edgecolor="black", lw=0.8)
for i, v in enumerate(vals):
    axA.annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10, fontweight="bold")
axA.set_xticks(range(len(torder))); axA.set_xticklabels(torder, fontsize=9.5, rotation=15)
axA.set_ylabel("# training episodes")
held = info.get("skipped_holdout", 0)
axA.set_title(f"A. Command-template mix of {len(eps):,} training episodes\n"
              f"(held OUT of training: 'separate'+'point' verbs = {held} episodes skipped)",
              fontsize=12, fontweight="bold")
axA.grid(axis="y", alpha=0.3)

# B. color × shape heatmap
im = axB.imshow(mat, cmap="viridis", aspect="auto")
axB.set_xticks(range(len(SHAPES))); axB.set_xticklabels(SHAPES, fontsize=10)
axB.set_yticks(range(len(COLORS))); axB.set_yticklabels(COLORS, fontsize=10)
for i in range(len(COLORS)):
    for j in range(len(SHAPES)):
        axB.text(j, i, mat[i, j], ha="center", va="center",
                 color="white" if mat[i, j] < mat.max() * 0.6 else "black", fontsize=10, fontweight="bold")
axB.set_title("B. Object vocabulary — the Language Table block set\n(referenced color × shape; blanks = combos not in the block pool)",
              fontsize=12, fontweight="bold")
fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label="# episodes")

# C. episode length
axC.hist(lengths, bins=np.arange(0, lengths.max() + 5, 4), color="#55A868", edgecolor="black", lw=0.5)
axC.axvline(lengths.mean(), color="red", ls="--", lw=1.5, label=f"mean {lengths.mean():.0f}")
axC.axvline(np.median(lengths), color="black", ls=":", lw=1.5, label=f"median {np.median(lengths):.0f}")
axC.set_xlabel("episode length (steps)"); axC.set_ylabel("# episodes")
axC.set_title("C. Episode-length distribution", fontsize=12, fontweight="bold")
axC.legend(fontsize=9.5); axC.grid(axis="y", alpha=0.3)

# D. instruction word-count + summary
axD.hist(wordcounts, bins=np.arange(0, wordcounts.max() + 2), color="#8172B3", edgecolor="black", lw=0.5)
axD.set_xlabel("instruction length (words)"); axD.set_ylabel("# episodes")
axD.set_title("D. Instruction length + dataset summary", fontsize=12, fontweight="bold")
axD.grid(axis="y", alpha=0.3)
txt = (f"episodes: {info.get('n_episodes', len(eps)):,}\n"
       f"frames: {info.get('n_frames', 0):,}\n"
       f"unique instructions: {info.get('n_unique_instructions', 0):,}\n"
       f"vocabulary: {len(COLORS)} colors × {len(SHAPES)} shapes\n"
       f"held-out verbs: separate, point")
axD.text(0.97, 0.95, txt, transform=axD.transAxes, ha="right", va="top", fontsize=11,
         bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#888"))

fig.suptitle(
    "Google Language Table — converted training set diversity (language_table_sim subset)\n"
    "Per-episode instructions span the Language Table block set (colors × shapes) and multiple command templates; "
    "the 'separate' and 'point' verbs were withheld for the new-command generalization test.",
    fontsize=13, fontweight="bold", y=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved → {OUT}")
