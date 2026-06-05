"""
scripts/plot_lt_data_budget.py
──────────────────────────────
How much of Google Language Table did we actually train on, and how is that subset
split?  Answers the "what fraction of the data" question visually.

Panels:
  A. Subset vs full language_table_sim (5,000 of 181,020 episodes = 2.76%)
  B. Composition of the 5,000-episode subset by command template (+ the held-out
     separate/point episodes that were skipped)
  C. Data pipeline: episodes → frames → 90/10 train/val → embedding cache

Reads asset/data/language_table_sim/meta/instructions.json
Output → docs/experiments/language_table/lt_data_budget.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
META = ROOT / "asset/data/language_table_sim/meta/instructions.json"
OUT = ROOT / "docs/experiments/language_table/lt_data_budget.png"

FULL_EPISODES = 181_020        # language_table_sim train split (from TFDS info)
FULL_SHARDS = 200
MEAN_LEN = 25.6                # mean episode length (from the 1.5k-episode inventory)

info = json.loads(META.read_text())
used_eps = info["n_episodes"]            # 5000
used_frames = info["n_frames"]           # 128104
held = info.get("skipped_holdout", 0)    # 343
tmpl = info["template_counts"]
est_full_frames = int(FULL_EPISODES * MEAN_LEN)
val_frac = 0.10
val_frames = int(used_frames * val_frac)
train_frames = used_frames - val_frames
pct = 100 * used_eps / FULL_EPISODES

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 6))

# ── A. donut: used vs unused of the full sim dataset ───────────────────────────
sizes = [used_eps, FULL_EPISODES - used_eps]
axA.pie(sizes, colors=["#4C72B0", "#E3E3E3"], startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white"),
        labels=[f"used\n{used_eps:,} eps", f"unused\n{FULL_EPISODES-used_eps:,} eps"],
        labeldistance=1.12, textprops={"fontsize": 10, "fontweight": "bold"})
axA.text(0, 0, f"{pct:.1f}%\nof 181,020", ha="center", va="center", fontsize=15, fontweight="bold")
axA.set_title("A. Fraction of language_table_sim used for training\n"
              f"5,000 of 181,020 episodes  (~1 in {FULL_EPISODES//used_eps})",
              fontsize=11.5, fontweight="bold")

# ── B. subset composition by template (+ held-out) ─────────────────────────────
order = sorted(tmpl, key=tmpl.get, reverse=True)
vals = [tmpl[t] for t in order]
bars = axB.bar(range(len(order)), vals, color="#55A868", edgecolor="black", lw=0.8)
axB.bar(len(order), held, color="#C44E52", edgecolor="black", lw=0.8)
for i, v in enumerate(vals + [held]):
    axB.annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9.5, fontweight="bold")
axB.set_xticks(range(len(order) + 1))
axB.set_xticklabels(order + ["separate+point\n(HELD OUT)"], fontsize=9, rotation=20)
axB.set_ylabel("# episodes in subset")
axB.set_title(f"B. Composition of the 5,000-episode training subset\n"
              f"(trained templates in green; {held} OOD episodes removed in red)",
              fontsize=11.5, fontweight="bold")
axB.grid(axis="y", alpha=0.3)

# ── C. pipeline funnel ─────────────────────────────────────────────────────────
axC.axis("off")
steps = [
    (f"language_table_sim\n{FULL_EPISODES:,} episodes  (~{est_full_frames/1e6:.1f}M frames, {FULL_SHARDS} shards)", "#E3E3E3"),
    (f"streamed + held-out filter\n→ {used_eps:,} episodes kept  ({held} separate/point skipped)", "#AEC7E8"),
    (f"converted frames\n{used_frames:,} frames  (= {pct:.1f}% of dataset)", "#4C72B0"),
    (f"90/10 split\ntrain {train_frames:,}  |  val {val_frames:,} frames", "#55A868"),
    (f"VLM embedding cache\n{used_frames:,} × (120 × 1024) bf16  ≈ 31.5 GB", "#8172B3"),
]
y = 0.92
for i, (txt, col) in enumerate(steps):
    axC.add_patch(plt.Rectangle((0.06, y - 0.13), 0.88, 0.12, transform=axC.transAxes,
                                facecolor=col, edgecolor="black", lw=0.9))
    axC.text(0.5, y - 0.07, txt, transform=axC.transAxes, ha="center", va="center",
             fontsize=9.5, fontweight="bold" if i in (2, 3) else "normal")
    if i < len(steps) - 1:
        axC.annotate("", xy=(0.5, y - 0.135), xytext=(0.5, y - 0.16), transform=axC.transAxes,
                     arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))
    y -= 0.185
axC.set_title("C. Training data pipeline", fontsize=11.5, fontweight="bold")

fig.suptitle(
    "Language Table — training data budget:  5,000 of 181,020 episodes (2.76%, ~1 in 36) of language_table_sim\n"
    "A bounded subset chosen to keep the VLM-embedding cache tractable on a 12 GB GPU; still 100× the 50-demo ALOHA run and language-conditioned.",
    fontsize=12.5, fontweight="bold", y=1.03)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved → {OUT}")
