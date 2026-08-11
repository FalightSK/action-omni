"""
scripts/plot_pred_distribution.py
──────────────────────────────────
What does each model actually PREDICT, and does its action distribution match
the data?  Compares the trained Qwen models on held-out (val) frames:

  * exp01  (closed-loop, predict 16)   -> 66% SR
  * exp02  (open-loop,  predict 50)    -> 86% SR
  * SmolVLA (open-loop, predict 50)    -> 0%   [added if --smolvla and it loads]

vs ground-truth dataset actions.  All actions denormalised to joint-target units.

Output -> docs/experiments/aloha/pred_distribution_compare.png
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from data.aloha.dataset import AlohaEmbeddingDataset
from models.vla_train import VLATrainModel

BLUE, ORANGE, GREEN, GREY, RED = "#4C72B0", "#DD8452", "#55A868", "#9A9A9A", "#C44E52"
OUT = ROOT / "docs" / "experiments" / "aloha" / "pred_distribution_compare.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
JLABELS = [f"R-j{k}" for k in range(6)] + ["R-grip"] + [f"L-j{k}" for k in range(6)] + ["L-grip"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


@torch.no_grad()
def qwen_preds(exp: str):
    """Run a trained Qwen model on its held-out val set -> (pred, gt) in joint units."""
    cfg = get_config("aloha", exp)
    ds = AlohaEmbeddingDataset(str(ROOT / cfg.embeddings_cache))
    vlen = max(1, int(len(ds) * 0.10))
    _, val = random_split(ds, [len(ds) - vlen, vlen],
                          generator=torch.Generator().manual_seed(42))
    loader = DataLoader(val, batch_size=256, shuffle=False, num_workers=0)

    model = VLATrainModel(cfg).to(device)
    ckpt = torch.load(ROOT / cfg.output_dir / "checkpoints" / "best.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    P, G = [], []
    for b in loader:
        embed = b["embedding"].float().to(device)
        state = b["state"].to(device)
        im = b.get("img_mask")
        im = im.to(device) if im is not None else None
        flat = model.sample(embed, state, num_steps=cfg.num_flow_steps, img_mask=im)
        P.append(flat.view(-1, cfg.action_horizon, cfg.action_dim).cpu().numpy())
        G.append(b["actions"].numpy())
    pred = np.concatenate(P); gt = np.concatenate(G)           # (M,H,14) normalised
    amean = np.array(cfg.action_mean, np.float32); astd = np.array(cfg.action_std, np.float32)
    print(f"  {exp}: {pred.shape[0]} val samples, H={cfg.action_horizon}")
    return pred * astd + amean, gt * astd + amean              # denormalised


print(f"device={device}")
p1, g1 = qwen_preds("exp01")        # H=16
p2, g2 = qwen_preds("exp02")        # H=50
GT = g2                              # reference GT distribution (50-step chunks)

# SmolVLA predictions (precomputed by scripts/compute_smolvla_preds.py)
smol = smol_gt = None
npz = ROOT / "asset/runs/aloha/smolvla_ft/pred_cache.npz"
if npz.exists():
    z = np.load(npz)
    smol, smol_gt = z["pred"], z["gt"]            # (N,50,14) already in joint units
    print(f"  SmolVLA: {smol.shape} loaded from cache")

# ── per-joint summary stats ─────────────────────────────────────────────────────
def mae_per_joint(pred, gt):
    return np.abs(pred - gt).reshape(-1, 14).mean(0)
mae1, mae2 = mae_per_joint(p1, g1), mae_per_joint(p2, g2)
maeS = mae_per_joint(smol, smol_gt) if smol is not None else None
std_gt = GT.reshape(-1, 14).std(0)
std1   = p1.reshape(-1, 14).std(0)
std2   = p2.reshape(-1, 14).std(0)
stdS   = smol.reshape(-1, 14).std(0) if smol is not None else None
active = np.argsort(std_gt)[::-1][:6]            # 6 most-varying joints for overlays

# ── figure ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.26, height_ratios=[1.0, 1.0, 1.0])

# Panel 1 — per-joint MAE (accuracy) ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
x = np.arange(14); w = 0.27
ax.bar(x - w, mae1, w, color=BLUE, label="exp01 (66%)")
ax.bar(x,     mae2, w, color=GREEN, label="exp02 (86%)")
if maeS is not None:
    ax.bar(x + w, maeS, w, color=RED, label="SmolVLA (0%)")
ax.set_xticks(x); ax.set_xticklabels(JLABELS, rotation=60, fontsize=7, ha="right")
ax.set_ylabel("Mean abs error vs GT (joint units)")
ax.set_title("1. Prediction accuracy per joint\n(lower = closer to demonstrated action)",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)

# Panel 2 — error vs horizon step ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
err1 = np.linalg.norm(p1 - g1, axis=-1).mean(0)   # (16,)
err2 = np.linalg.norm(p2 - g2, axis=-1).mean(0)   # (50,)
ax.plot(np.arange(1, 17), err1, "o-", color=BLUE, ms=4, label="exp01 (16 steps)")
ax.plot(np.arange(1, 51), err2, "s-", color=GREEN, ms=3, label="exp02 (50 steps)")
if smol is not None:
    errS = np.linalg.norm(smol - smol_gt, axis=-1).mean(0)   # (50,)
    ax.plot(np.arange(1, smol.shape[1] + 1), errS, "^-", color=RED, ms=3, label="SmolVLA (50 steps)")
ax.axvspan(0.5, 8.5, color=BLUE, alpha=0.07, label="exp01 executes (8)")
ax.set_xlabel("horizon step within predicted chunk")
ax.set_ylabel("L2 error vs GT (all joints)")
ax.set_title("2. Error along the chunk\n(does open-loop drift deep into the chunk?)",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

# Panel 3 — per-joint std: prediction spread vs data spread ──────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.bar(x - w, std1 / (std_gt + 1e-9), w, color=BLUE, label="exp01")
ax.bar(x,     std2 / (std_gt + 1e-9), w, color=GREEN, label="exp02")
if stdS is not None:
    ax.bar(x + w, stdS / (std_gt + 1e-9), w, color=RED, label="SmolVLA")
ax.axhline(1.0, color="black", ls="--", lw=1.3, label="= data spread (ideal)")
ax.set_xticks(x); ax.set_xticklabels(JLABELS, rotation=60, fontsize=7, ha="right")
ax.set_ylabel("predicted std / GT std")
ax.set_title("3. Prediction spread vs data\n(<1 = collapsed/cautious, >1 = noisy)",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)

# Panels 4-9 — per-joint predicted vs GT distributions (6 most active joints) ────
for k, j in enumerate(active):
    ax = fig.add_subplot(gs[1 + k // 3, k % 3])
    lo = min(GT[..., j].min(), p1[..., j].min(), p2[..., j].min())
    hi = max(GT[..., j].max(), p1[..., j].max(), p2[..., j].max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(GT[..., j].ravel(), bins=bins, density=True, color=GREY, alpha=0.55,
            label="ground truth")
    ax.hist(p1[..., j].ravel(), bins=bins, density=True, histtype="step",
            color=BLUE, lw=1.8, label="exp01")
    ax.hist(p2[..., j].ravel(), bins=bins, density=True, histtype="step",
            color=GREEN, lw=1.8, label="exp02")
    if smol is not None and j < smol.shape[2]:
        ax.hist(smol[..., j].ravel(), bins=bins, density=True, histtype="step",
                color=RED, lw=1.8, label="SmolVLA")
    ax.set_title(f"{JLABELS[j]}  (joint {j})", fontsize=10, fontweight="bold")
    ax.set_xlabel("joint target (denorm)"); ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    if k == 0:
        ax.legend(fontsize=8)

ttl = ("ALOHA transfer-cube  |  what each model PREDICTS vs the data  "
       "(held-out frames, actions denormalised to joint units)\n"
       "exp01 closed-loop 66% (blue)  -  exp02 open-loop 86% (green)  -  SmolVLA 0% (red)  -  ground-truth fill (grey)")
fig.suptitle(ttl, fontsize=13, fontweight="bold", y=1.005)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {OUT}")
