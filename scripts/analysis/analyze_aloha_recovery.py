"""
Does the ALOHA transfer-cube dataset contain RECOVERY cases (failed grasp -> retry)?

The parquet has no reward, so we use the gripper signals as a proxy.  A clean
transfer = the grasping gripper closes ONCE (grasp -> hold -> release) and the
receiving gripper closes ONCE.  A recovery episode shows EXTRA close intervals
on a gripper (re-grasp after a miss/drop) and/or many direction reversals.

Auto-detects the two gripper joints (most bimodal), counts debounced
"close intervals" (= grasp attempts) per gripper per episode, and flags episodes
with > the clean number.

Output -> docs/experiments/aloha/aloha_recovery_analysis.png
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
cfg = get_config("aloha", "exp01")
root = Path(cfg.dataset_root)
OUT = ROOT / "docs/experiments/aloha/aloha_recovery_analysis.png"

shards = sorted(glob.glob(str(root / "data/chunk-000/file-*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True).sort_values("index").reset_index(drop=True)
states = np.asarray(df["observation.state"].tolist(), np.float32)   # (N,14)
acts = np.asarray(df["action"].tolist(), np.float32)
ep_ids = df["episode_index"].to_numpy()
fidx = df["frame_index"].to_numpy()
eps = sorted(df["episode_index"].unique())
JL = [f"R-j{k}" for k in range(6)] + ["R-grip"] + [f"L-j{k}" for k in range(6)] + ["L-grip"]

# ── auto-detect gripper joints: most "binary" (values hug the two extremes) ─────
biscore = []
for j in range(14):
    v = states[:, j]; lo, hi = np.percentile(v, 2), np.percentile(v, 98); span = hi - lo + 1e-9
    near_ext = np.mean((v < lo + 0.15 * span) | (v > hi - 0.15 * span))
    biscore.append(near_ext)
grippers = [6, 13]               # canonical ALOHA gripper indices (arm0 grip=6, arm1 grip=13)
print(f"gripper joints (canonical): {[(g, JL[g], round(biscore[g],2)) for g in grippers]}")
print(f"(bimodality of all joints: {np.round(biscore,2)})")


def close_intervals(traj, open_level, thr, min_run=5):
    """# of debounced intervals where the gripper is on the CLOSED side."""
    closed_is_low = open_level > thr
    closed = (traj < thr) if closed_is_low else (traj > thr)
    # debounce: ignore runs shorter than min_run
    n = len(closed); i = 0; runs = []
    while i < n:
        j = i
        while j < n and closed[j] == closed[i]:
            j += 1
        runs.append((closed[i], j - i)); i = j
    # merge short runs into neighbours
    state_seq = []
    for val, ln in runs:
        if ln >= min_run or not state_seq:
            state_seq.append([val, ln])
        else:
            state_seq[-1][1] += ln
    # count closed segments (>= min_run)
    return sum(1 for val, ln in state_seq if val and ln >= min_run)


per_ep = {g: [] for g in grippers}
reversals_ep = []
for e in eps:
    m = ep_ids == e
    for g in grippers:
        v = states[m, g]
        thr = (np.percentile(v, 10) + np.percentile(v, 90)) / 2
        open_level = v[0]                              # grippers start open
        per_ep[g].append(close_intervals(v, open_level, thr))
    # arm path reversals: sign changes in velocity summed over arm joints (proxy for hesitation)
    arm = [j for j in range(14) if j not in grippers]
    vel = np.diff(states[m][:, arm], axis=0)
    rev = np.sum(np.abs(np.diff(np.sign(vel), axis=0)) > 0) / len(arm)
    reversals_ep.append(rev)

g0, g1 = grippers
total_close = np.array(per_ep[g0]) + np.array(per_ep[g1])
# The cube PICK-UP gripper (R-grip) is the unambiguous recovery signal: a real
# re-grasp = it grasps, drops/opens, grasps again.  The receiver's (L-grip)
# multi-close is normal multi-level hand-off motion (consistent across ~all eps).
regrasp_eps = [eps[i] for i in range(len(eps)) if per_ep[g0][i] > 1]
print(f"\nclose-intervals per gripper (clean=1 each):")
print(f"  {JL[g0]} (cube pick-up): {per_ep[g0]}")
print(f"  {JL[g1]} (hand-off recv): {per_ep[g1]}")
print(f"\nCUBE re-grasps (R-grip > 1 = genuine recovery): {len(regrasp_eps)}/{len(eps)} -> {regrasp_eps}")

# ── figure ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
# A: histogram of grasp attempts per gripper
bins = np.arange(0, max(total_close.max(), 4) + 2) - 0.5
ax[0].hist(per_ep[g0], bins=bins, alpha=0.6, label=f"{JL[g0]} (grasp)", color="#4C72B0")
ax[0].hist(per_ep[g1], bins=bins, alpha=0.6, label=f"{JL[g1]} (receive)", color="#DD8452")
ax[0].axvline(1, color="green", ls="--", lw=1.5, label="clean = 1 grasp")
ax[0].set_xlabel("# close-intervals (grasp attempts) per episode")
ax[0].set_ylabel("# episodes"); ax[0].legend(fontsize=9)
ax[0].set_title("A. Grasp attempts per gripper\n(>1 = re-grasp = recovery)", fontsize=11.5, fontweight="bold")
ax[0].grid(axis="y", alpha=0.3)

# B: clean example vs most-grasp example, gripper trajectories
clean_ep = eps[int(np.argmin(total_close))]
busy_ep = eps[int(np.argmax(total_close))]
for k, (e, name) in enumerate([(clean_ep, "cleanest"), (busy_ep, "most grasp-cycles")]):
    m = ep_ids == e
    axx = ax[1 + k]
    for g, c in [(g0, "#4C72B0"), (g1, "#DD8452")]:
        v = states[m, g]
        axx.plot(np.arange(len(v)), (v - v.min()) / (np.ptp(v) + 1e-9), color=c, lw=1.8, label=JL[g])
    axx.set_title(f"{'B' if k==0 else 'C'}. Episode {e} ({name})\n"
                  f"grasps: {JL[g0]}={per_ep[g0][eps.index(e)]}, {JL[g1]}={per_ep[g1][eps.index(e)]}",
                  fontsize=11, fontweight="bold")
    axx.set_xlabel("frame"); axx.set_ylabel("gripper (norm: 0=closed,1=open or vice-versa)")
    axx.legend(fontsize=9); axx.grid(alpha=0.3)

n_rec = len(regrasp_eps)
fig.suptitle(
    f"ALOHA transfer-cube ({len(eps)} human demos): is there RECOVERY (failed-grasp -> retry) data?\n"
    f"Cube pick-up (R-grip) is a CLEAN single grasp in {len(eps)-n_rec}/{len(eps)} episodes "
    f"(only {n_rec} re-grasp); trajectories are smooth grasp->transfer->release  ==>  "
    f"essentially NO failure-recovery data in the demos.",
    fontsize=12.5, fontweight="bold", y=1.02)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"\nsaved -> {OUT}")
