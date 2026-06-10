"""Key-frame filmstrip: in-distribution success vs out-of-distribution miss."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
VID = ROOT / "asset/runs/aloha/exp01_baseline/videos_ood"
OUT = ROOT / "docs/experiments/aloha/qwen_ood_filmstrip.png"

clips = [
    (VID / "cube_x+0.10_y0.50_IN.mp4",  "IN-distribution  cube (0.10, 0.50)  ->  SUCCESS (reward 4)", "#1a7a1a"),
    (VID / "cube_x+0.30_y0.50_OOD.mp4", "OUT-of-distribution  cube (0.30, 0.50)  ->  never reaches the cube (reward 0)", "#b01818"),
]
NCOL = 6

fig, axes = plt.subplots(2, NCOL, figsize=(3.0 * NCOL, 6.4))
for r, (path, label, col) in enumerate(clips):
    frames = imageio.mimread(str(path), memtest=False)
    n = len(frames)
    idxs = np.linspace(0, n - 1, NCOL).astype(int)
    for c, fi in enumerate(idxs):
        ax = axes[r, c]
        ax.imshow(frames[fi])
        ax.set_xticks([]); ax.set_yticks([])
        pct = int(100 * fi / max(1, n - 1))
        if r == 0:
            ax.set_title(f"{pct}%  (step {fi})", fontsize=9)
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2.5)
    axes[r, 0].set_ylabel(label, fontsize=10.5, color=col, fontweight="bold", rotation=90, labelpad=12)

fig.suptitle("Qwen-DiT generalization: same policy, cube inside vs 0.1 m outside the trained region\n"
             "top: trained position, clean grasp+transfer  |  bottom: OOD position, arm reaches the trained spot and grasps empty air",
             fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved -> {OUT}")
