"""
Side-by-side OOD spatial-generalization heatmaps for the inference-horizon sweep:
exp01 ih=8 (closed-loop) | exp02 ih=10 | exp02 ih=50 (open-loop), + a summary bar.

Shows the same conclusion three ways: in-distribution skill shifts with replan
rate, but OOD success is a flat 0% in every config.

Output -> docs/experiments/aloha/qwen_ood_ih_compare.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parents[1]
OUT = ROOT / "docs/experiments/aloha/qwen_ood_ih_compare.png"
BLUE, RED = "#4C72B0", "#C44E52"

RUNS = [
    ("exp01  ih=8\n(closed-loop, 66%)", ROOT / "asset/runs/aloha/exp01_baseline/ood_cube_results.json"),
    ("exp02  ih=10\n(replan every 10)", ROOT / "asset/runs/aloha/exp02_openloop/ood_cube_results_ih10.json"),
    ("exp02  ih=50\n(open-loop, 86%)", ROOT / "asset/runs/aloha/exp02_openloop/ood_cube_results.json"),
]

data = []
for label, p in RUNS:
    j = json.loads(Path(p).read_text())
    xs, ys = j["xs"], j["ys"]
    g = np.zeros((len(ys), len(xs)))
    for e in j["grid"]:
        g[ys.index(e["y"]), xs.index(e["x"])] = e["sr"] * 100
    data.append((label, xs, ys, g, j["mean_sr_in_dist"], j["mean_sr_ood"]))

fig = plt.figure(figsize=(20, 5.6))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.7], wspace=0.28)

for k, (label, xs, ys, g, ind, ood) in enumerate(data):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(g, origin="lower", cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{v:+.2f}" for v in xs], fontsize=8)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{v:.2f}" for v in ys], fontsize=8)
    ax.set_xlabel("cube x (m)")
    if k == 0:
        ax.set_ylabel("cube y (m)")
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            ax.text(ix, iy, f"{g[iy, ix]:.0f}", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.add_patch(Rectangle((0.5, 0.5), 3, 3, fill=False, ec="blue", lw=2.2, ls="--"))
    ax.set_title(f"{label}\nin-dist {ind*100:.0f}%  |  OOD {ood*100:.0f}%", fontsize=10.5, fontweight="bold")

# summary bars
ax = fig.add_subplot(gs[0, 3])
labels = ["ih=8", "ih=10", "ih=50"]
ind_v = [d[4] * 100 for d in data]
ood_v = [d[5] * 100 for d in data]
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, ind_v, w, color=BLUE, label="in-distribution")
ax.bar(x + w / 2, ood_v, w, color=RED, label="OOD (0.1 m out)")
for xi, v in zip(x - w / 2, ind_v):
    ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, fontweight="bold")
for xi, v in zip(x + w / 2, ood_v):
    ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, fontweight="bold", color=RED)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("mean success rate (%)"); ax.set_ylim(0, 100)
ax.set_title("Summary: in-dist vs OOD", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Qwen-DiT OOD spatial generalization across inference horizons (5x5 cube grid, n=8/cell; dashed = trained region)\n"
    "Replan rate (ih=8/10/50) moves in-distribution skill a little -- but OOD success is a flat 0% in every case: the model interpolates, it does not extrapolate.",
    fontsize=12.5, fontweight="bold", y=1.04)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved -> {OUT}")
