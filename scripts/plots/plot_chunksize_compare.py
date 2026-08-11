"""
scripts/plot_chunksize_compare.py
──────────────────────────────────
Full visual comparison for the ALOHA closed-loop / chunk-size study, with
EXPLICIT model attribution so it is unambiguous which point belongs to which model.

Two models share the horizon axis:
  * exp01  (BLUE, circles)  -- ONE model, trained to predict 16 / execute 8.
                              The ih = 1 / 4 / 8 / 16 points are this SAME model
                              re-evaluated while forcing it to run N steps
                              open-loop before re-planning.
  * exp02  (GREEN, star)    -- a SECOND model trained from scratch to predict &
                              execute all 50 (fully open-loop = SmolVLA's 1.0 s chunk).
  * SmolVLA (RED)           -- the SmolVLA finetune baseline (0%).

A vertical divider + shaded background band separate the exp01 region (ih<=16)
from the exp02 region (ih=50) in every horizon panel.

Saves -> docs/experiments/aloha/qwen_chunksize_sweep.png
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parents[2]
RUNS = ROOT / "asset" / "runs" / "aloha"
OUT  = ROOT / "docs" / "experiments" / "aloha" / "qwen_chunksize_sweep.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# house palette (matches scripts/analysis.py); models are colour-coded:
BLUE, ORANGE, GREEN, PURPLE, RED, GREY = (
    "#4C72B0", "#DD8452", "#55A868", "#8172B3", "#C44E52", "#B0B0B0")
C_EXP01, C_EXP02, C_SMOL = BLUE, GREEN, RED

STAGE_COLOR = {0: GREY, 1: "#F2C14E", 2: ORANGE, 4: GREEN}
STAGE_NAME  = {0: "r0 no contact", 1: "r1 touched", 2: "r2 lifted",
               4: "r4 transfer (success)"}

# x layout: a GAP before the exp02 point so the two models read as separate
XPOS = np.array([0.0, 1.0, 2.0, 3.0, 4.6])
DIV  = 3.8            # divider between exp01 region and exp02 region
XL, XR = -0.6, 5.2


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load(path: Path):
    j = json.loads(Path(path).read_text())
    eps = j["episodes"]
    n = len(eps)
    k = sum(1 for e in eps if e["is_success"])
    rewards = [int(e["max_reward"]) for e in eps]
    return {
        "n": n, "k": k, "sr": k / n,
        "cov": float(np.mean([e["max_coverage"] for e in eps])),
        "steps": float(np.mean([e["steps"] for e in eps])),
        "hist": {s: rewards.count(s) for s in (0, 1, 2, 3, 4)},
    }


SWEEP = [
    (1,  "exp01", RUNS / "exp01_baseline" / "sim_results_ih1.json"),
    (4,  "exp01", RUNS / "exp01_baseline" / "sim_results_ih4.json"),
    (8,  "exp01", RUNS / "exp01_baseline" / "sim_results.json"),     # default
    (16, "exp01", RUNS / "exp01_baseline" / "sim_results_ih16.json"),
    (50, "exp02", RUNS / "exp02_openloop" / "sim_results.json"),     # open-loop
]
data = []
for ih, model, path in SWEEP:
    d = load(path); d["ih"] = ih; d["model"] = model
    data.append(d)
    print(f"  ih={ih:<3} [{model}] SR={d['sr']*100:4.0f}%  cov={d['cov']*100:3.0f}%  "
          f"steps={d['steps']:.0f}")
SMOL = {"n": 50, "k": 0, "sr": 0.0}

E1 = [i for i, d in enumerate(data) if d["model"] == "exp01"]   # 0..3
E2 = [i for i, d in enumerate(data) if d["model"] == "exp02"][0]  # 4
xlabels = [str(d["ih"]) for d in data]
srs   = np.array([d["sr"] for d in data])
covs  = np.array([d["cov"] for d in data])
steps = np.array([d["steps"] for d in data])
lo    = np.array([wilson(d["k"], d["n"])[0] for d in data])
hi    = np.array([wilson(d["k"], d["n"])[1] for d in data])


def regions(ax, big=False):
    """Shade + divide the exp01 (left) and exp02 (right) model regions."""
    ax.set_xlim(XL, XR)
    ax.axvspan(XL, DIV, color=C_EXP01, alpha=0.06, zorder=0)
    ax.axvspan(DIV, XR, color=C_EXP02, alpha=0.09, zorder=0)
    ax.axvline(DIV, color="0.5", ls=(0, (5, 3)), lw=1.2, zorder=1)
    tr = ax.get_xaxis_transform()
    if big:
        ax.text(1.5, 0.97, "exp01 model\n(predict16 / exec8)\nsame net, eval-horizon sweep",
                transform=tr, ha="center", va="top", fontsize=8.2,
                color=C_EXP01, fontweight="bold", linespacing=1.3)
        ax.text(4.6, 0.40, "exp02 model\n(predict50 / exec50)\ntrained open-loop",
                transform=tr, ha="center", va="top", fontsize=8.2,
                color=C_EXP02, fontweight="bold", linespacing=1.3)
    else:
        ax.text(1.5, 0.965, "exp01", transform=tr, ha="center", va="top",
                fontsize=9.5, color=C_EXP01, fontweight="bold")
        ax.text(4.6, 0.965, "exp02", transform=tr, ha="center", va="top",
                fontsize=9.5, color=C_EXP02, fontweight="bold")


fig = plt.figure(figsize=(18, 10.5))
gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.26)

# ── Panel 1 — SR vs horizon (headline) ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
regions(ax, big=True)
ax.errorbar(XPOS[E1], srs[E1] * 100,
            yerr=[(srs[E1] - lo[E1]) * 100, (hi[E1] - srs[E1]) * 100],
            fmt="o-", color=C_EXP01, lw=2.4, ms=10, capsize=4, zorder=4,
            label="exp01 (one trained model)")
ax.errorbar([XPOS[E2]], [srs[E2] * 100],
            yerr=[[(srs[E2] - lo[E2]) * 100], [(hi[E2] - srs[E2]) * 100]],
            fmt="*", color=C_EXP02, ms=26, capsize=4, zorder=5,
            label="exp02 (trained open-loop @50)")
ax.axhline(0, color=C_SMOL, ls=":", lw=1.8)
ax.text(XR - 0.1, 3.0, "SmolVLA = 0%", color=C_SMOL, fontsize=9.5,
        ha="right", va="bottom", fontweight="bold")
for i, d in enumerate(data):
    c = C_EXP02 if d["model"] == "exp02" else C_EXP01
    ax.annotate(f"{d['sr']*100:.0f}%", (XPOS[i], d["sr"] * 100),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=11, fontweight="bold", color=c)
ax.set_xticks(XPOS); ax.set_xticklabels(xlabels)
ax.set_xlabel("inference_horizon  (steps executed before re-plan)")
ax.set_ylabel("Success rate (%)   n=50")
ax.set_ylim(-5, 104)
ax.set_title("1. Success RISES with chunk length\n(more closed-loop replanning = worse)",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="center left", fontsize=8.5, framealpha=0.92)
ax.grid(alpha=0.3)

# ── Panel 2 — reward-stage funnel (stacked) ────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
regions(ax)
bottoms = np.zeros(len(data))
for s in (0, 1, 2, 4):
    frac = np.array([d["hist"][s] / d["n"] for d in data]) * 100
    ax.bar(XPOS, frac, bottom=bottoms, width=0.78, color=STAGE_COLOR[s],
           edgecolor="white", lw=0.7, label=STAGE_NAME[s], zorder=3)
    bottoms += frac
ax.set_xticks(XPOS); ax.set_xticklabels(xlabels)
ax.set_xlabel("inference_horizon")
ax.set_ylabel("% of episodes reaching stage")
ax.set_ylim(0, 100)
ax.set_title("2. Where episodes end up\n(grey = never touched -> green = success)",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="lower center", fontsize=7.8, ncol=2, framealpha=0.92)

# ── Panel 3 — three-model head-to-head ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
exp01, exp02 = data[2], data[E2]
models = [
    ("exp01\n(ours, closed-loop ih=8)", exp01["sr"], wilson(exp01["k"], exp01["n"]), C_EXP01),
    ("exp02\n(ours, open-loop ih=50)",  exp02["sr"], wilson(exp02["k"], exp02["n"]), C_EXP02),
    ("SmolVLA\n(open-loop 50-step)",    SMOL["sr"],  wilson(SMOL["k"], SMOL["n"]),  C_SMOL),
]
for i, (lbl, sr, ci, col) in enumerate(models):
    ax.bar(i, sr * 100, color=col, alpha=0.88, edgecolor="white", width=0.6, zorder=3)
    ax.errorbar(i, sr * 100, yerr=[[(sr - ci[0]) * 100], [(ci[1] - sr) * 100]],
                fmt="none", ecolor="black", capsize=6, lw=1.4, zorder=4)
    ax.annotate(f"{sr*100:.0f}%", (i, sr * 100), textcoords="offset points",
                xytext=(0, 9 if sr > 0 else 5), ha="center",
                fontsize=14, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([m[0] for m in models], fontsize=9)
ax.set_ylabel("Success rate (%)   n=50")
ax.set_ylim(0, 100)
ax.set_title("3. Same architecture, open-loop @50 WINS\n=> open-loop is NOT SmolVLA's problem",
             fontsize=11.5, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# ── Panel 4 — coverage vs horizon ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
regions(ax)
ax.plot(XPOS[E1], covs[E1] * 100, "o-", color=C_EXP01, lw=2.4, ms=10, zorder=4)
ax.plot([XPOS[E2]], [covs[E2] * 100], "*", color=C_EXP02, ms=26, zorder=5)
for i, d in enumerate(data):
    c = C_EXP02 if d["model"] == "exp02" else C_EXP01
    ax.annotate(f"{d['cov']*100:.0f}%", (XPOS[i], d["cov"] * 100),
                textcoords="offset points", xytext=(0, 11), ha="center",
                fontsize=9.5, color=c, fontweight="bold")
ax.set_xticks(XPOS); ax.set_xticklabels(xlabels)
ax.set_xlabel("inference_horizon")
ax.set_ylabel("Mean max-coverage (% of reward 4)")
ax.set_ylim(0, 100)
ax.set_title("4. Task progress (partial credit) also rises",
             fontsize=11.5, fontweight="bold")
ax.grid(alpha=0.3)

# ── Panel 5 — efficiency: mean steps vs horizon ────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
regions(ax)
ax.plot(XPOS[E1], steps[E1], "o-", color=ORANGE, lw=2.4, ms=10, zorder=4)
ax.plot([XPOS[E2]], [steps[E2]], "*", color=C_EXP02, ms=26, zorder=5)
for i, d in enumerate(data):
    c = C_EXP02 if d["model"] == "exp02" else ORANGE
    ax.annotate(f"{d['steps']:.0f}", (XPOS[i], d["steps"]),
                textcoords="offset points", xytext=(0, 11), ha="center",
                fontsize=9.5, color=c, fontweight="bold")
ax.set_xticks(XPOS); ax.set_xticklabels(xlabels)
ax.set_xlabel("inference_horizon")
ax.set_ylabel("Mean episode length (steps, cap=400)")
ax.set_title("5. Longer chunks finish FASTER\n(lower = quicker, fewer stalls)",
             fontsize=11.5, fontweight="bold")
ax.grid(alpha=0.3)

# ── Panel 6 — mechanism: never-touched vs success ──────────────────────────────
ax = fig.add_subplot(gs[1, 2])
regions(ax)
never = np.array([d["hist"][0] / d["n"] for d in data]) * 100
succ = srs * 100
# solid within exp01, dashed across the model divider to exp02
ax.plot(XPOS[E1], never[E1], "o-", color=C_SMOL, lw=2.4, ms=8, label="never touched cube (r0)")
ax.plot(XPOS[[3, 4]], never[[3, 4]], "--", color=C_SMOL, lw=1.6, alpha=0.7)
ax.plot([XPOS[E2]], [never[E2]], "*", color=C_SMOL, ms=20)
ax.plot(XPOS[E1], succ[E1], "s-", color=C_EXP02, lw=2.4, ms=8, label="success (r4)")
ax.plot(XPOS[[3, 4]], succ[[3, 4]], "--", color=C_EXP02, lw=1.6, alpha=0.7)
ax.plot([XPOS[E2]], [succ[E2]], "*", color=C_EXP02, ms=20)
ax.set_xticks(XPOS); ax.set_xticklabels(xlabels)
ax.set_xlabel("inference_horizon")
ax.set_ylabel("% of episodes")
ax.set_ylim(-5, 104)
ax.set_title("6. WHY: frequent replanning -> never commits\n(re-samples flow noise mid-grasp -> jitter)",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="center right", fontsize=8.5, framealpha=0.92)
ax.grid(alpha=0.3)

fig.suptitle(
    "ALOHA transfer-cube  |  Qwen frozen-VLM VLA: closed-loop rate / chunk-size study   (n=50 each, Wilson 95% CI)",
    fontsize=13, fontweight="bold", y=1.005)

KEY = ("HOW TO READ  -  inference_horizon = env steps run open-loop before the policy re-plans.   "
       "BLUE circles = exp01: ONE model (trained predict-16 / execute-8) re-evaluated at ih = 1/4/8/16.   "
       "GREEN star = exp02: a SECOND model trained to predict & execute all 50 (open-loop = SmolVLA's 1.0 s chunk).   "
       "RED = SmolVLA finetune.\n"
       "Finding: success rises monotonically with chunk length (10 -> 42 -> 66 -> 70 -> 86%). "
       "The same architecture at SmolVLA's 50-step open-loop chunk scores its BEST (86%) -- so open-loop is NOT the cause of SmolVLA's 0%.")
fig.text(0.5, -0.045, KEY, ha="center", va="top", fontsize=9.2,
         bbox=dict(boxstyle="round,pad=0.6", fc="#F4F4F4", ec="0.7"))

plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {OUT}")
