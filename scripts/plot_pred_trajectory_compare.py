"""
scripts/plot_pred_trajectory_compare.py
─────────────────────────────────────────
Direct, per-frame comparison of the PREDICTED action chunk:
SmolVLA  vs  your Qwen DiT (exp02 open-loop & exp01 closed-loop)  vs ground truth,
all evaluated on the SAME frames (the ones SmolVLA was run on: index 0,50,100,...).

For a few dynamic frames x a few active joints, plot the joint-target trajectory
across the predicted chunk.  GT and Qwen track; SmolVLA drifts.

Output -> docs/experiments/aloha/pred_trajectory_compare.png
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from data.aloha.dataset import AlohaEmbeddingDataset
from models.vla_train import VLATrainModel

BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#444444"
OUT = ROOT / "docs" / "experiments" / "aloha" / "pred_trajectory_compare.png"
JLABELS = [f"R-j{k}" for k in range(6)] + ["R-grip"] + [f"L-j{k}" for k in range(6)] + ["L-grip"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ── SmolVLA predictions (aligned to sample_idx = 0,50,100,...) ──────────────────
z = np.load(ROOT / "asset/runs/aloha/smolvla_ft/pred_cache.npz")
smol, gt = z["pred"], z["gt"]                       # (Ns,50,14) joint units
Ns = smol.shape[0]
sample_idx = np.arange(0, 20000, 50)[:Ns]
print(f"SmolVLA preds {smol.shape}, frames {sample_idx[0]}..{sample_idx[-1]}")


@torch.no_grad()
def qwen_on(exp: str, idx: np.ndarray):
    """Run a trained Qwen model on specific frame indices -> (len(idx), H, 14) joint units."""
    cfg = get_config("aloha", exp)
    ds = AlohaEmbeddingDataset(str(ROOT / cfg.embeddings_cache))
    emb = torch.stack([ds.embeddings[i].float() for i in idx])
    st = torch.stack([ds.states[i] for i in idx])
    im = (torch.stack([ds.img_masks[i] for i in idx]) if ds.img_masks is not None else None)
    model = VLATrainModel(cfg).to(device)
    ck = torch.load(ROOT / cfg.output_dir / "checkpoints" / "best.pt",
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    out = []
    for s in range(0, len(idx), 256):
        b = slice(s, s + 256)
        imb = im[b].to(device) if im is not None else None
        flat = model.sample(emb[b].to(device), st[b].to(device),
                            num_steps=cfg.num_flow_steps, img_mask=imb)
        out.append(flat.view(-1, cfg.action_horizon, cfg.action_dim).cpu().numpy())
    pred = np.concatenate(out)
    amean = np.array(cfg.action_mean, np.float32); astd = np.array(cfg.action_std, np.float32)
    del ds, emb
    return pred * astd + amean, cfg.action_horizon


p2, H2 = qwen_on("exp02", sample_idx)               # (Ns,50,14)
p1, H1 = qwen_on("exp01", sample_idx)               # (Ns,16,14)
print(f"exp02 {p2.shape}  exp01 {p1.shape}")

# ── pick the most DYNAMIC frames (largest within-chunk GT motion) & active joints
dyn = (gt.max(1) - gt.min(1)).sum(1)                # (Ns,) total joint travel in the chunk
frames = np.argsort(dyn)[::-1][:4]                  # 4 most dynamic chunks
joints = np.argsort(gt.reshape(-1, 14).std(0))[::-1][:4]   # 4 most active joints
print(f"frames {sample_idx[frames]}  joints {[JLABELS[j] for j in joints]}")

# ── figure: rows = frames, cols = joints ───────────────────────────────────────
nf, nj = len(frames), len(joints)
fig = plt.figure(figsize=(4.4 * nj, 3.4 * nf))
gs = GridSpec(nf, nj, figure=fig, hspace=0.38, wspace=0.26)
hx2 = np.arange(H2); hx1 = np.arange(H1)
for r, f in enumerate(frames):
    ep = sample_idx[f] // 400
    for c, j in enumerate(joints):
        ax = fig.add_subplot(gs[r, c])
        ax.plot(hx2, gt[f, :, j], color=GREY, lw=3.2, alpha=0.85, label="ground truth", zorder=2)
        ax.plot(hx2, p2[f, :, j], color=GREEN, lw=2.0, label="exp02 Qwen-DiT (open-loop, 86%)", zorder=4)
        ax.plot(hx1, p1[f, :, j], color=BLUE, lw=1.6, ls="--", label="exp01 Qwen-DiT (16-step, 66%)", zorder=3)
        ax.plot(hx2, smol[f, :, j], color=RED, lw=2.0, label="SmolVLA (0%)", zorder=4)
        if r == 0:
            ax.set_title(f"{JLABELS[j]}  (joint {j})", fontsize=11, fontweight="bold")
        if c == 0:
            ax.set_ylabel(f"frame {sample_idx[f]} (ep{ep})\njoint target", fontsize=9)
        if r == nf - 1:
            ax.set_xlabel("step within predicted chunk")
        ax.grid(alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(fontsize=7.5, loc="best")

fig.suptitle(
    "Predicted action chunk: SmolVLA vs your Qwen-DiT vs ground truth (same frames)\n"
    "GT (grey) & Qwen exp02 (green) track tightly; SmolVLA (red) starts off-target and drifts open-loop -> 0%",
    fontsize=13, fontweight="bold", y=1.005)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {OUT}")
