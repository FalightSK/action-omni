"""
scripts/plot_aloha_diversity.py
────────────────────────────────
Is the ALOHA transfer-cube dataset diverse enough to claim generalization,
or does the cube always start in ~the same place?

The parquet logs only 14-DOF joint state (no object pose), so we recover the
cube's START position by detecting the RED cube in the first frame of each of
the 50 episodes (top camera, 480x640).  Arm-motion diversity comes from the
joint trajectories directly.

Output -> docs/experiments/aloha/aloha_dataset_diversity.png

Panels:
  A  cube start positions scattered over a reference frame  (the headline)
  B  cube-position 2D density + marginal spread stats
  C  joint-space PCA: every episode's trajectory (fan-out = diverse motions)
  D  per-joint range/std across the dataset
  E  validation montage: 4 extreme-position episodes with the detected cube marked
"""

from __future__ import annotations
import glob, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from data.aloha.dataset import _SequentialFrameReader

BLUE, ORANGE, GREEN, PURPLE, RED = "#4C72B0", "#DD8452", "#55A868", "#8172B3", "#C44E52"
OUT = ROOT / "docs" / "experiments" / "aloha" / "aloha_dataset_diversity.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

cfg  = get_config("aloha", "exp01")
root = Path(cfg.dataset_root)

# ── tabular: states + episode boundaries ───────────────────────────────────────
shards = sorted(glob.glob(str(root / "data" / "chunk-000" / "file-*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
df = df.sort_values("index").reset_index(drop=True)
states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)   # (N,14)
ep_ids = df["episode_index"].to_numpy()
episodes = sorted(df["episode_index"].unique())
first_idx = {e: int(df.index[ep_ids == e][0]) for e in episodes}          # global frame idx
print(f"{len(episodes)} episodes, {len(df)} frames")

# ── decode first frame of each episode + detect red cube ───────────────────────
vid = sorted(glob.glob(str(root / "videos" / cfg.camera_key / "chunk-000" / "*.mp4")))[0]
reader = _SequentialFrameReader(vid)


def red_centroid(frame):
    """Return (u, v, n_red) for the red cube; robust median centroid."""
    hsv = mcolors.rgb_to_hsv(frame.astype(np.float32) / 255.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h < 0.045) | (h > 0.955)) & (s > 0.45) & (v > 0.25)
    ys, xs = np.where(mask)
    if len(xs) < 25:
        return np.nan, np.nan, len(xs)
    return float(np.median(xs)), float(np.median(ys)), len(xs)


cube_uv, ref_frame, frame_cache = [], None, {}
for e in episodes:
    fr = reader.get(first_idx[e])              # (480,640,3) uint8, ascending -> efficient
    if ref_frame is None:
        ref_frame = fr.copy()
    u, v, n = red_centroid(fr)
    cube_uv.append((u, v))
    if e < 4 or e % 12 == 0:                   # keep a few frames for the montage
        frame_cache[e] = fr.copy()
reader.close()
cube_uv = np.array(cube_uv)                    # (50,2)
H, W = ref_frame.shape[:2]
ok = ~np.isnan(cube_uv[:, 0])
print(f"cube detected in {ok.sum()}/{len(episodes)} first frames")
det = cube_uv[ok]
ux, uy = det[:, 0], det[:, 1]
spread_x = ux.max() - ux.min()
spread_y = uy.max() - uy.min()
print(f"cube X span {spread_x:.0f}px ({spread_x/W*100:.0f}% width), "
      f"Y span {spread_y:.0f}px ({spread_y/H*100:.0f}% height)")

# ── figure ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs = GridSpec(3, 4, figure=fig, hspace=0.34, wspace=0.30, height_ratios=[1.15, 1.0, 0.9])

# A — cube start positions over reference frame ─────────────────────────────────
ax = fig.add_subplot(gs[0, 0:2])
ax.imshow(ref_frame, alpha=0.55)
ax.scatter(ux, uy, s=70, c=np.arange(len(ux)), cmap="turbo",
           edgecolor="white", lw=1.2, zorder=4)
# bounding box of cube spread
ax.add_patch(plt.Rectangle((ux.min(), uy.min()), spread_x, spread_y,
             fill=False, ec=RED, lw=2.0, ls="--", zorder=5))
ax.set_title(f"A. Cube START position, all {ok.sum()} episodes (top camera)\n"
             f"red box = full spread of starting positions",
             fontsize=11.5, fontweight="bold")
ax.set_xlabel("image x (px)"); ax.set_ylabel("image y (px)")
ax.set_xlim(0, W); ax.set_ylim(H, 0)

# B — cube position density + marginal spread ──────────────────────────────────
ax = fig.add_subplot(gs[0, 2:4])
ax.hexbin(ux, uy, gridsize=12, cmap="Blues", mincnt=1)
ax.scatter(ux, uy, s=18, color=RED, alpha=0.8, zorder=4)
ax.scatter([ux.mean()], [uy.mean()], s=180, marker="+", color="black", lw=2.5,
           zorder=5, label="mean")
ax.set_title("B. Cube-position density\n(are they bunched or spread?)",
             fontsize=11.5, fontweight="bold")
ax.set_xlabel("image x (px)"); ax.set_ylabel("image y (px)")
ax.set_xlim(0, W); ax.set_ylim(H, 0)
ax.legend(loc="upper right", fontsize=9)
ax.text(0.02, 0.02,
        f"X spread: {spread_x:.0f}px = {spread_x/W*100:.0f}% of frame width\n"
        f"Y spread: {spread_y:.0f}px = {spread_y/H*100:.0f}% of frame height\n"
        f"std: ({ux.std():.0f}, {uy.std():.0f}) px",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.9))

# C — joint-space PCA: per-episode trajectories ────────────────────────────────
ax = fig.add_subplot(gs[1, 0:2])
# PCA via NumPy SVD (no sklearn dependency)
mu = states.mean(0)
Xc = states - mu
U, S, Vt = np.linalg.svd(Xc[::4], full_matrices=False)   # fit on a subsample
comps = Vt[:2]                                            # (2,14)
ev = (S[:2] ** 2) / (S ** 2).sum() * 100                  # explained-variance ratio
proj = Xc @ comps.T                                       # (N,2)
cmap = plt.cm.turbo
for i, e in enumerate(episodes):
    m = ep_ids == e
    ax.plot(proj[m, 0], proj[m, 1], lw=0.7, alpha=0.5, color=cmap(i / len(episodes)))
starts = np.array([proj[ep_ids == e][0] for e in episodes])
ends = np.array([proj[ep_ids == e][-1] for e in episodes])
ax.scatter(starts[:, 0], starts[:, 1], s=22, color="black", zorder=5,
           label="episode start (home)")
ax.scatter(ends[:, 0], ends[:, 1], s=22, marker="X", color=RED, zorder=5,
           label="episode end")
ax.set_title(f"C. Arm trajectories in joint-PCA space ({len(episodes)} episodes)\n"
             f"fan-out = diverse motions  (PC1+PC2 = {ev.sum():.0f}% var)",
             fontsize=11.5, fontweight="bold")
ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=0.3)

# D — per-joint range / std ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2:4])
jmin, jmax, jstd = states.min(0), states.max(0), states.std(0)
y = np.arange(14)
labels = [f"R-j{k}" for k in range(6)] + ["R-grip"] + [f"L-j{k}" for k in range(6)] + ["L-grip"]
ax.barh(y, jmax - jmin, left=jmin, color=BLUE, alpha=0.55, label="full range (min-max)")
ax.scatter(jmin + jstd, y, color=RED, s=18, zorder=5, label="+1 std from min")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_title("D. Per-joint spread across dataset\n(wide bar = that joint explores a lot)",
             fontsize=11.5, fontweight="bold")
ax.set_xlabel("joint value (normalised units)")
ax.legend(loc="lower right", fontsize=8.5); ax.grid(axis="x", alpha=0.3)

# E — validation montage: 4 extreme-position episodes with cube marked ──────────
# pick episodes at the extremes of cube position so spread is visible
order = {"left": np.nanargmin(cube_uv[:, 0]), "right": np.nanargmax(cube_uv[:, 0]),
         "top": np.nanargmin(cube_uv[:, 1]), "bottom": np.nanargmax(cube_uv[:, 1])}
picks = list(dict.fromkeys(order.values()))[:4]
# ensure we have those frames decoded; re-open reader for any missing
need = [p for p in picks if episodes[p] not in frame_cache]
if need:
    r2 = _SequentialFrameReader(vid)
    for p in sorted(need):
        frame_cache[episodes[p]] = r2.get(first_idx[episodes[p]]).copy()
    r2.close()
for col, p in enumerate(picks):
    e = episodes[p]
    axm = fig.add_subplot(gs[2, col])
    axm.imshow(frame_cache[e])
    u, v = cube_uv[p]
    if not np.isnan(u):
        axm.scatter([u], [v], s=300, facecolors="none", edgecolors="lime", lw=2.5)
        axm.plot([u], [v], "+", color="lime", ms=14, mew=2)
    side = [k for k, val in order.items() if val == p]
    axm.set_title(f"ep{e}  cube=({u:.0f},{v:.0f})  [{'/'.join(side)}]", fontsize=9)
    axm.set_xticks([]); axm.set_yticks([])

fig.suptitle(
    "ALOHA transfer-cube dataset diversity  |  50 human demos, 400 steps each  "
    "-- does the cube really start in varied places?",
    fontsize=13.5, fontweight="bold", y=0.995)
verdict = (f"Cube starts span {spread_x/W*100:.0f}% of the frame width and "
           f"{spread_y/H*100:.0f}% of the height across the {ok.sum()} episodes "
           f"(green circles in row E mark the detected cube). "
           f"Arm trajectories (C) fan out accordingly from the shared home pose.")
fig.text(0.5, 0.005, verdict, ha="center", va="bottom", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", fc="#FFF7E6", ec="0.7"))

plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {OUT}")
