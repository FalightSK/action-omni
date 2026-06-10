"""
'Why SmolVLA fails' figure, from the checkpoint diagnostics in
scripts/diagnose_smolvla*.py (measured, not the logged loss).

Numbers (64 fixed training frames, eval pipeline):
  ckpt   logged-train-loss   t-avg flow loss   sampled MAE (norm)
  4k     0.017               0.197             0.0522
  8k     0.007               0.266             0.0505
  12k    0.005               0.296             0.0503
  Qwen exp02 sampled MAE (same frames/units) = 0.0165
  per-t flow loss profile (12k): high near noise, low near data (min 0.10)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
OUT = ROOT / "docs/experiments/aloha/smolvla_why_fail.png"
BLUE, GREEN, RED, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#DD8452"

ckpts = ["4k", "8k", "12k"]
logged = [0.017, 0.007, 0.005]
fwd = [0.197, 0.266, 0.296]
mae = [0.0522, 0.0505, 0.0503]
QWEN_MAE = 0.0165
tgrid = np.array([.05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95])
per_t = [0.683, 0.612, 0.529, 0.463, 0.413, 0.360, 0.319, 0.286, 0.257, 0.232,
         0.214, 0.194, 0.174, 0.159, 0.145, 0.132, 0.120, 0.110, 0.101]

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

# Panel 1 — logged loss DOWN but predictions FLAT
a = ax[0]; x = np.arange(3)
a.plot(x, logged, "o-", color=BLUE, lw=2.4, ms=9, label="logged TRAIN loss (down 3.4x)")
a.set_xticks(x); a.set_xticklabels(ckpts); a.set_xlabel("checkpoint (training steps)")
a.set_ylabel("flow loss (logged)", color=BLUE); a.tick_params(axis="y", labelcolor=BLUE)
a.set_ylim(0, 0.02)
a2 = a.twinx()
a2.plot(x, mae, "s-", color=RED, lw=2.4, ms=9, label="sampled MAE (FLAT)")
a2.set_ylabel("sampled prediction MAE", color=RED); a2.tick_params(axis="y", labelcolor=RED)
a2.set_ylim(0, 0.06)
for xi, m in zip(x, mae):
    a2.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, 9), ha="center", color=RED, fontsize=9)
a.set_title("1. The loss curve lied\nlogged loss falls, real predictions don't improve", fontsize=11.5, fontweight="bold")
l1, la1 = a.get_legend_handles_labels(); l2, la2 = a2.get_legend_handles_labels()
a.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8.5)
a.grid(alpha=0.3)

# Panel 2 — SmolVLA 3x worse than Qwen
a = ax[1]
bars = a.bar([0, 1], [mae[-1], QWEN_MAE], color=[RED, GREEN], width=0.55, edgecolor="white")
a.set_xticks([0, 1]); a.set_xticklabels(["SmolVLA\n(12k, 0% SR)", "Qwen-DiT exp02\n(86% SR)"], fontsize=10)
a.set_ylabel("sampled prediction MAE vs GT (norm units)")
for xi, m in zip([0, 1], [mae[-1], QWEN_MAE]):
    a.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=12, fontweight="bold")
a.text(0.5, 0.046, f"{mae[-1]/QWEN_MAE:.1f}x worse", ha="center", color=RED, fontsize=12, fontweight="bold")
a.set_title("2. Same frames: SmolVLA predicts 3x worse\n(plateaued -- more steps won't help)", fontsize=11.5, fontweight="bold")
a.grid(axis="y", alpha=0.3)

# Panel 3 — per-t flow loss profile (true fit is mediocre)
a = ax[2]
a.plot(tgrid, per_t, "o-", color=ORANGE, lw=2.2, ms=4)
a.axhline(0.005, color=BLUE, ls="--", lw=1.5, label="logged loss (0.005)")
a.fill_between(tgrid, per_t, 0.005, alpha=0.12, color=ORANGE)
a.set_xlabel("flow-matching time t  (0=noise, 1=data)")
a.set_ylabel("velocity-field error")
a.set_title("3. Real velocity-field error (12k) is 20-130x\nthe logged 0.005 at every t", fontsize=11.5, fontweight="bold")
a.legend(fontsize=9); a.grid(alpha=0.3)

fig.suptitle(
    "Why SmolVLA gets 0% on ALOHA (root-cause diagnostics)  -  NOT a pipeline bug; NOT step count\n"
    "Cause: under-fit visual grounding (frozen SmolVLM + expert-only, 50 demos, 1 cam) -> plateaus ~3x worse than Qwen -> wrong grasp + open-loop drift",
    fontsize=12.5, fontweight="bold", y=1.04)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved -> {OUT}")
