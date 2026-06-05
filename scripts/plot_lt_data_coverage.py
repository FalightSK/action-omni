"""
scripts/plot_lt_data_coverage.py
────────────────────────────────
Physical / action coverage of the 5,000-episode Language Table training subset
(complements lt_dataset_diversity.py, which covers the language side).

Reads the converted parquet (state = effector xy, action = 2D delta) directly.

Panels:
  A. Effector-position density — where in the workspace the arm operates
  B. Action-delta density — the 2D setpoint distribution the model must learn
  C. Per-axis action histograms + magnitude (step sizes)
  D. Action-direction rose — directional coverage of the pushes

Output → docs/experiments/language_table/lt_data_coverage.png
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
DATA = ROOT / "asset/data/language_table_sim/data/chunk-000"
OUT = ROOT / "docs/experiments/language_table/lt_data_coverage.png"

shards = sorted(glob.glob(str(DATA / "file-*.parquet")))
df = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "reward", "episode_index"])
                for f in shards], ignore_index=True)
states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)   # (N,2)
actions = np.asarray(df["action"].tolist(), dtype=np.float32)             # (N,2)
rewards = df["reward"].to_numpy()
N = len(df)
n_eps = df["episode_index"].nunique()
pos_reward = int((rewards > 0).sum())
mag = np.linalg.norm(actions, axis=1)
moving = mag > 1e-4
ang = np.arctan2(actions[moving, 1], actions[moving, 0])

fig = plt.figure(figsize=(16, 12))
axA = fig.add_subplot(2, 2, 1)
axB = fig.add_subplot(2, 2, 2)
axC = fig.add_subplot(2, 2, 3)
axD = fig.add_subplot(2, 2, 4, projection="polar")

# ── A. effector position density ───────────────────────────────────────────────
hb = axA.hexbin(states[:, 0], states[:, 1], gridsize=45, cmap="viridis", mincnt=1)
axA.set_xlabel("effector x (m)"); axA.set_ylabel("effector y (m)")
axA.set_title(f"A. Effector-position coverage  ({N:,} frames)\n"
              f"workspace x∈[{states[:,0].min():.2f},{states[:,0].max():.2f}]  "
              f"y∈[{states[:,1].min():.2f},{states[:,1].max():.2f}]",
              fontsize=11.5, fontweight="bold")
axA.set_aspect("equal", "box")
fig.colorbar(hb, ax=axA, fraction=0.046, pad=0.04, label="# frames")

# ── B. action delta density ────────────────────────────────────────────────────
hb2 = axB.hexbin(actions[:, 0], actions[:, 1], gridsize=45, cmap="magma", mincnt=1, bins="log")
axB.axhline(0, color="white", lw=0.5, alpha=0.5); axB.axvline(0, color="white", lw=0.5, alpha=0.5)
axB.set_xlabel("action Δx (m/step)"); axB.set_ylabel("action Δy (m/step)")
axB.set_title(f"B. Action-delta coverage (log density)\n"
              f"deltas span ±{np.abs(actions).max():.3f};  {100*moving.mean():.0f}% of frames are moving",
              fontsize=11.5, fontweight="bold")
axB.set_aspect("equal", "box")
fig.colorbar(hb2, ax=axB, fraction=0.046, pad=0.04, label="log # frames")

# ── C. per-axis action + magnitude ─────────────────────────────────────────────
axC.hist(actions[:, 0], bins=80, alpha=0.55, label="Δx", color="#4C72B0")
axC.hist(actions[:, 1], bins=80, alpha=0.55, label="Δy", color="#DD8452")
axC.axvline(0, color="black", lw=0.8, ls=":")
axC.set_xlabel("action component (m/step)"); axC.set_ylabel("# frames")
axC.set_title(f"C. Per-axis action distribution\n"
              f"mean|Δ|={mag.mean():.4f}  median|Δ|={np.median(mag):.4f}  max={mag.max():.3f}",
              fontsize=11.5, fontweight="bold")
axC.legend(fontsize=10); axC.grid(axis="y", alpha=0.3)

# ── D. action-direction rose ───────────────────────────────────────────────────
nb = 24
counts, edges = np.histogram(ang, bins=nb, range=(-np.pi, np.pi))
centers = (edges[:-1] + edges[1:]) / 2
axD.bar(centers, counts, width=2 * np.pi / nb, color="#55A868", edgecolor="black", lw=0.5, alpha=0.85)
axD.set_title(f"D. Action-direction coverage\n({moving.sum():,} moving frames; all directions used)",
              fontsize=11.5, fontweight="bold", pad=18)
axD.set_theta_zero_location("E")

fig.suptitle(
    f"Language Table training subset — physical & action coverage  "
    f"({n_eps:,} episodes / {N:,} frames)\n"
    f"Reward is sparse: only {pos_reward:,} frames ({100*pos_reward/N:.2f}%) carry a positive (success) reward — "
    f"the model learns from dense action supervision, not reward.",
    fontsize=12.5, fontweight="bold", y=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved → {OUT}  | frames={N} eps={n_eps} pos_reward={pos_reward}")
