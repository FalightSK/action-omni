"""
scripts/plot_lt_balanced_plan.py
────────────────────────────────
Data-exploration + design viz for a BALANCED Language Table training set.

Question: build a balanced training set (equal episodes per task) sized to ~40% of the
full dataset. The natural distribution is heavily skewed (block2block 41% … point 2%),
so "balanced across all 7 tasks" and "40% of all data" CANNOT both hold — the rarest
task (point) caps an all-7 balanced set at ~14.5%. This figure shows the skew, the
feasibility trade-off, the two viable balanced designs, and the data characteristics.

Reads asset/data/language_table_sim/meta/{instructions_inventory,episodes,stats}.json
       + the converted parquet (for the action/state coverage panel)
Output → docs/experiments/language_table/lt_balanced_plan.png
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
META = ROOT / "asset/data/language_table_sim/meta"
DATA = ROOT / "asset/data/language_table_sim/data/chunk-000"
OUT = ROOT / "docs/experiments/language_table/lt_balanced_plan.png"
FULL_EPISODES = 181_020
TARGET_FRAC = 0.40

inv = json.loads((META / "instructions_inventory.json").read_text())
eps = json.loads((META / "episodes.json").read_text())
stats = json.loads((META / "stats.json").read_text())

# ── natural distribution (extrapolate the 1,500-ep scan to the full dataset) ─────
sc = inv["template_counts"]; scanned = inv["scanned_episodes"]
NAME = {"block2block": "block2block", "absolute_location": "absolute", "relative_to_block": "relative",
        "separate": "separate", "between": "between", "other": "other/corner", "point": "point"}
tasks = sorted(sc, key=sc.get, reverse=True)
frac = {t: sc[t] / scanned for t in tasks}
avail = {t: int(frac[t] * FULL_EPISODES) for t in tasks}            # est. episodes available per task
order = sorted(tasks, key=lambda t: avail[t], reverse=True)

target_eps = int(TARGET_FRAC * FULL_EPISODES)

# feasibility: max balanced total if we include the top-K tasks (by availability)
Ks, maxbal = [], []
for K in range(1, len(order) + 1):
    inc = order[:K]
    cap = min(avail[t] for t in inc)            # balanced ⇒ every task limited to the rarest included
    Ks.append(K); maxbal.append(K * cap / FULL_EPISODES * 100)

# Design A: balanced 40% across the 3 biggest tasks
A_tasks = order[:3]; A_per = target_eps // 3
# Design B: balanced across ALL 7 (capped by rarest)
B_per = min(avail.values()); B_total = B_per * len(order)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# A. natural distribution
ax = axes[0, 0]
vals = [avail[t] for t in order]
ax.bar(range(len(order)), vals, color="#4C72B0", edgecolor="black")
for i, t in enumerate(order):
    ax.annotate(f"{avail[t]:,}\n{frac[t]*100:.1f}%", (i, avail[t]), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(range(len(order))); ax.set_xticklabels([NAME[t] for t in order], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("est. episodes (of 181k)")
ax.set_title("A. Natural task distribution (extrapolated from 1,500-ep scan)\nheavily skewed — block2block 41% … point 2%",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# B. feasibility trade-off
ax = axes[0, 1]
ax.plot(Ks, maxbal, "-o", color="#C44E52", lw=2, ms=7)
ax.axhline(TARGET_FRAC * 100, color="black", ls="--", lw=1.5, label=f"40% target")
for k, mb in zip(Ks, maxbal):
    ax.annotate(f"{mb:.0f}%", (k, mb), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, fontweight="bold")
ax.fill_between(Ks, maxbal, 0, where=[mb >= 40 for mb in maxbal], alpha=0.12, color="green")
ax.set_xlabel("# tasks included (added by availability)"); ax.set_ylabel("max BALANCED size (% of data)")
ax.set_title("B. Why 40% + balanced-all-7 is impossible\n40% needs ≤3 tasks; all 7 caps at ~14.5% (point-limited)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(Ks)

# C. the two balanced designs
ax = axes[0, 2]
xA = np.arange(len(order))
barsA = [A_per if t in A_tasks else 0 for t in order]
barsB = [B_per for t in order]
w = 0.4
ax.bar(xA - w/2, barsA, w, color="#55A868", edgecolor="black", label=f"Design A: 3 tasks ×{A_per:,} = 40%")
ax.bar(xA + w/2, barsB, w, color="#DD8452", edgecolor="black", label=f"Design B: all 7 ×{B_per:,} = {B_total/FULL_EPISODES*100:.0f}%")
ax.set_xticks(xA); ax.set_xticklabels([NAME[t] for t in order], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("episodes per task (balanced)")
ax.set_title("C. Two viable BALANCED designs\nA = 40% (3 tasks, rest held out) · B = all-7 but 14.5%",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)

# D. episode length distribution (real, from converted episodes)
ax = axes[1, 0]
lens = np.array([e["n_steps"] for e in eps])
ax.hist(lens, bins=np.arange(0, lens.max() + 4, 3), color="#8172B3", edgecolor="black", lw=0.4)
ax.axvline(lens.mean(), color="red", ls="--", lw=1.5, label=f"mean {lens.mean():.0f}")
ax.axvline(np.median(lens), color="black", ls=":", lw=1.5, label=f"median {np.median(lens):.0f}")
ax.set_xlabel("episode length (steps)"); ax.set_ylabel("# episodes")
ax.set_title(f"D. Episode-length distribution\n(balanced set ≈ same; {A_per*3:,} eps × ~{lens.mean():.0f} steps ≈ {A_per*3*lens.mean()/1e6:.1f}M frames)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# E. object vocabulary coverage (color × shape referenced in instructions)
ax = axes[1, 1]
COLORS = ["red", "green", "blue", "yellow"]; SHAPES = ["cube", "moon", "star", "pentagon", "crescent"]
OBJ = re.compile(r"(red|green|blue|yellow)\s+(cube|moon|star|pentagon|crescent)")
mat = np.zeros((4, 5), int)
for e in eps:
    m = OBJ.search(e["instruction"].lower())
    if m:
        mat[COLORS.index(m.group(1)), SHAPES.index(m.group(2))] += 1
im = ax.imshow(mat, cmap="viridis", aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(SHAPES, fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels(COLORS, fontsize=9)
for i in range(4):
    for j in range(5):
        ax.text(j, i, mat[i, j], ha="center", va="center", color="white" if mat[i, j] < mat.max()*0.6 else "black", fontsize=8)
ax.set_title("E. Object vocabulary coverage\n(referenced color × shape — the LT block set)", fontsize=11, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# F. action-delta coverage (from converted parquet — representative of all tasks)
ax = axes[1, 2]
shards = sorted(glob.glob(str(DATA / "file-*.parquet")))
acts = np.concatenate([np.asarray(pd.read_parquet(f, columns=["action"])["action"].tolist(), np.float32) for f in shards[:3]])
ax.hexbin(acts[:, 0], acts[:, 1], gridsize=40, cmap="magma", bins="log", mincnt=1)
ax.set_xlabel("action Δx (m/step)"); ax.set_ylabel("action Δy (m/step)")
ax.set_aspect("equal", "box")
ax.set_title(f"F. Action-delta coverage (2D ee-delta)\nclipped ±0.03; std≈({stats['action']['std'][0]:.3f},{stats['action']['std'][1]:.3f})",
             fontsize=11, fontweight="bold")

fig.suptitle(
    "Balanced Language Table training set — design + data exploration\n"
    f"Goal: ~40% of {FULL_EPISODES:,} eps, balanced per task. RESULT: balanced-40% ⇒ 3 tasks (block2block/absolute/relative ×{A_per:,}); "
    f"all-7 balanced ⇒ only {B_total/FULL_EPISODES*100:.0f}% ({B_per:,} each, point-limited). Pick the trade-off.",
    fontsize=12.5, fontweight="bold", y=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"   Design A (40%, 3 tasks): {A_per:,} eps × 3 = {A_per*3:,} ({A_per*3/FULL_EPISODES*100:.0f}%); held out: {[NAME[t] for t in order[3:]]}")
print(f"   Design B (all 7 balanced): {B_per:,} eps × 7 = {B_total:,} ({B_total/FULL_EPISODES*100:.1f}%)")
print(f"saved → {OUT}")


if __name__ == "__main__":
    pass
main = None
