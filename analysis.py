"""
analysis.py — Comprehensive post-training model analysis.

Generates 8 figures saved to asset/result/analysis/:
  1. training_curve.png        — train/val loss + gap over epochs
  2. prediction_quality.png    — scatter, L2 histogram, per-horizon boxplot
  3. attention_heatmap.png     — AttentionReadout spatial attention on 8x8 grid
  4. action_distribution.png   — 2D density, agent-pos quiver, error heatmap
  5. embedding_pca.png         — raw VLM vs adapted embedding PCA
  6. flow_trajectory.png       — denoising path from noise → action
  7. gradient_flow.png         — gradient magnitude per trainable layer
  8. error_vs_coverage.png     — prediction error vs simulation coverage

Usage:
  python3 analysis.py
"""

from __future__ import annotations
import re, sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import VLAConfig
from data.pusht_dataset import PushTEmbeddingDataset
from train import VLATrainModel
from models.vla import VLMTokenAdapter

OUT = ROOT / "asset" / "result" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

cfg    = VLAConfig()
device = torch.device("cpu")   # CPU for analysis — deterministic, no MPS quirks
print(f"Analysis device: {device}")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt_path = ROOT / "asset" / "result" / "checkpoints" / "best.pt"
print(f"Loading {ckpt_path.name} …")
model = VLATrainModel(cfg).to(device)
ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}\n")

# ── Load validation data ──────────────────────────────────────────────────────
print("Loading dataset …")
full_ds = PushTEmbeddingDataset(str(ROOT / "asset" / "result" / "vlm_embeddings.pt"))
val_len  = max(1, int(len(full_ds) * 0.10))
train_ds, val_ds = random_split(
    full_ds, [len(full_ds) - val_len, val_len],
    generator=torch.Generator().manual_seed(42),
)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
print(f"  Val: {len(val_ds):,} samples\n")

# ── Collect predictions ───────────────────────────────────────────────────────
print("Running inference on val set …")
all_pred, all_gt, all_states, all_embeds_raw, all_adapted = [], [], [], [], []

@torch.no_grad()
def collect():
    for batch in val_loader:
        embed    = batch["embedding"].float().to(device)
        state    = batch["state"].to(device)
        actions  = batch["actions"]
        img_mask = batch.get("img_mask")
        if img_mask is not None:
            img_mask = img_mask.to(device)

        # Adapted embedding (before decoder)
        if embed.ndim == 3:
            h = model.adapter(embed, img_mask)   # (B, 512)
        else:
            h = embed
        all_adapted.append(h.cpu())
        all_embeds_raw.append(embed[:, :64].float().mean(dim=1).cpu())  # mean of image tokens

        flat = model.sample(embed, state, num_steps=cfg.num_flow_steps, img_mask=img_mask)
        pred = flat.view(-1, cfg.action_horizon, cfg.action_dim)
        all_pred.append(pred.cpu().float())
        all_gt.append(actions.float())
        all_states.append(state.cpu().float())

collect()

pred_arr   = torch.cat(all_pred).numpy()    # (N, 16, 2)
gt_arr     = torch.cat(all_gt).numpy()
state_arr  = torch.cat(all_states).numpy()  # (N, 2)
adapted    = torch.cat(all_adapted).numpy() # (N, 512)
raw_img    = torch.cat(all_embeds_raw).numpy()  # (N, 1024)

action_std  = np.array(cfg.action_std,  dtype=np.float32)
action_mean = np.array(cfg.action_mean, dtype=np.float32)
state_std   = np.array(cfg.state_std,   dtype=np.float32)
state_mean  = np.array(cfg.state_mean,  dtype=np.float32)

pred_px = pred_arr * action_std + action_mean   # (N,16,2) in pixels
gt_px   = gt_arr   * action_std + action_mean
state_px = state_arr * state_std + state_mean   # (N,2)

l2 = np.linalg.norm(pred_px - gt_px, axis=-1)  # (N,16) per-step L2 in px
print("Inference done.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Training Curve
# ═══════════════════════════════════════════════════════════════════════════════
print("[1/8] Training curve …")
log_path = Path("/tmp/train_v3.txt")
epochs_tr, train_losses, val_losses = [], [], []
if log_path.exists():
    for line in log_path.read_text().splitlines():
        m = re.search(r"Epoch\s+(\d+)/\d+.*train=([\d.]+).*val=([\d.]+)", line)
        if m:
            epochs_tr.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.plot(epochs_tr, train_losses, label="Train", color="#4C72B0", lw=1.5)
ax.plot(epochs_tr, val_losses,   label="Val",   color="#DD8452", lw=1.5)
if ckpt.get("epoch"):
    best_ep = ckpt["epoch"]
    if best_ep <= len(val_losses):
        best_val = val_losses[best_ep - 1]
        ax.axvline(best_ep, color="green", ls="--", lw=1, label=f"Best (ep {best_ep})")
        ax.scatter([best_ep], [best_val], color="green", zorder=5, s=50)
ax.set_xlabel("Epoch"); ax.set_ylabel("CFM Loss")
ax.set_title("Training & Validation Loss"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
gaps = np.array(val_losses) - np.array(train_losses)
ax.plot(epochs_tr, gaps, color="#8172B3", lw=1.5)
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.fill_between(epochs_tr, 0, gaps, alpha=0.2, color="#8172B3")
ax.set_xlabel("Epoch"); ax.set_ylabel("Val - Train Loss (overfitting gap)")
ax.set_title("Generalisation Gap"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "1_training_curve.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Prediction Quality
# ═══════════════════════════════════════════════════════════════════════════════
print("[2/8] Prediction quality …")
fig = plt.figure(figsize=(15, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 2a: scatter pred vs gt (x)
ax = fig.add_subplot(gs[0, 0])
sample_idx = np.random.choice(len(pred_px), 2000, replace=False)
ax.scatter(gt_px[sample_idx, 0, 0], pred_px[sample_idx, 0, 0],
           alpha=0.15, s=4, color="#4C72B0")
lim = max(abs(gt_px[:, 0, 0]).max(), abs(pred_px[:, 0, 0]).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
ax.set_xlabel("GT delta_x (px)"); ax.set_ylabel("Pred delta_x (px)")
ax.set_title("t+1 Predicted vs GT  (x)"); ax.grid(alpha=0.3)

# 2b: scatter pred vs gt (y)
ax = fig.add_subplot(gs[0, 1])
ax.scatter(gt_px[sample_idx, 0, 1], pred_px[sample_idx, 0, 1],
           alpha=0.15, s=4, color="#DD8452")
ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
ax.set_xlabel("GT delta_y (px)"); ax.set_ylabel("Pred delta_y (px)")
ax.set_title("t+1 Predicted vs GT  (y)"); ax.grid(alpha=0.3)

# 2c: L2 error histogram
ax = fig.add_subplot(gs[0, 2])
l2_t1 = l2[:, 0]
ax.hist(l2_t1, bins=60, color="#55A868", edgecolor="white", lw=0.3)
ax.axvline(l2_t1.mean(), color="red", lw=1.5, ls="--",
           label=f"Mean={l2_t1.mean():.1f}px")
ax.axvline(np.median(l2_t1), color="orange", lw=1.5, ls="--",
           label=f"Median={np.median(l2_t1):.1f}px")
ax.set_xlabel("L2 error (px)"); ax.set_ylabel("Count")
ax.set_title("L2 Error Distribution (t+1)"); ax.legend(); ax.grid(alpha=0.3)

# 2d: per-horizon L2 boxplot
ax = fig.add_subplot(gs[1, :2])
ax.boxplot([l2[:, h] for h in range(16)],
           positions=range(1, 17), patch_artist=True,
           boxprops=dict(facecolor="#4C72B080"),
           medianprops=dict(color="red", lw=1.5),
           flierprops=dict(marker=".", ms=1, alpha=0.2),
           showfliers=True)
ax.set_xlabel("Horizon step"); ax.set_ylabel("L2 error (px)")
ax.set_title("Per-step L2 Error Distribution"); ax.grid(axis="y", alpha=0.3)
ax.axvspan(0.5, 4.5, alpha=0.08, color="green", label="Executed steps (4)")
ax.legend()

# 2e: directional accuracy per horizon
ax = fig.add_subplot(gs[1, 2])
dir_accs = []
for h in range(16):
    pf = pred_arr[:, h]
    tf = gt_arr[:,   h]
    pn = pf / (np.linalg.norm(pf, axis=-1, keepdims=True) + 1e-8)
    tn = tf / (np.linalg.norm(tf, axis=-1, keepdims=True) + 1e-8)
    dir_accs.append((np.sum(pn * tn, axis=-1) > 0).mean() * 100)
ax.bar(range(1, 17), dir_accs, color="#4C72B0", alpha=0.8)
ax.axhline(50, color="red", ls="--", lw=1, label="Random (50%)")
ax.axvspan(0.5, 4.5, alpha=0.08, color="green")
ax.set_xlabel("Horizon step"); ax.set_ylabel("Directional Accuracy (%)")
ax.set_title("Directional Accuracy per Horizon"); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 100)

plt.suptitle("Prediction Quality Analysis", fontsize=13, y=1.01)
plt.savefig(OUT / "2_prediction_quality.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Attention Heatmap (AttentionReadout spatial weights)
# ═══════════════════════════════════════════════════════════════════════════════
print("[3/8] Attention heatmap …")

# Hook into AttentionReadout to capture weights
captured_weights = []

def _attn_hook(module, inp, out):
    # Re-run with need_weights=True to capture weights
    # inp[0]=query(B,1,512), inp[1]=key(B,82,512), inp[2]=value(B,82,512)
    pass

# Patch forward to return weights
class AttentionReadoutWithWeights(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.query = orig.query
        self.attn  = orig.attn
        self.norm  = orig.norm

    def forward(self, tokens, key_padding_mask=None):
        B = tokens.shape[0]
        Q = self.query.expand(B, -1, -1)
        out, weights = self.attn(Q, tokens, tokens,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=True)  # (B,1,82)
        return self.norm(out.squeeze(1)), weights.squeeze(1)  # weights: (B, 82)

readout_patched = AttentionReadoutWithWeights(model.adapter.readout).to(device)

# Collect attention weights across val set
all_attn_weights, all_img_masks_val = [], []
with torch.no_grad():
    for batch in val_loader:
        embed    = batch["embedding"].float().to(device)
        img_mask = batch.get("img_mask")
        if img_mask is None:
            continue
        img_mask = img_mask.to(device)

        # Run through LoRA + SpatialAwareMLP
        h = model.adapter.lora(embed)
        h = model.adapter.spatial(h, img_mask, cfg.img_grid_h, cfg.img_grid_w)
        _, weights = readout_patched(h)   # (B, 82)

        all_attn_weights.append(weights.cpu())
        all_img_masks_val.append(img_mask.cpu())
        if len(all_attn_weights) >= 5:   # enough for heatmap
            break

attn_w   = torch.cat(all_attn_weights).numpy()    # (M, 82)
img_masks_val = torch.cat(all_img_masks_val).numpy()  # (M, 82)

# Separate image vs text tokens
img_attn = []
for i in range(len(attn_w)):
    img_pos = np.where(img_masks_val[i])[0][:64]
    if len(img_pos) == 64:
        img_attn.append(attn_w[i][img_pos])
img_attn = np.array(img_attn)   # (M, 64)

mean_attn_map = img_attn.mean(0).reshape(8, 8)    # (8,8)
std_attn_map  = img_attn.std(0).reshape(8, 8)

# Text token attention
txt_attn_mean = []
for i in range(len(attn_w)):
    txt_pos = np.where(~img_masks_val[i])[0]
    txt_attn_mean.append(attn_w[i][txt_pos].mean() if len(txt_pos) > 0 else 0)
txt_mean = np.mean(txt_attn_mean)
img_mean = img_attn.mean()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im = axes[0].imshow(mean_attn_map, cmap="hot", interpolation="nearest")
axes[0].set_title(f"Mean Attention — Image Tokens (8x8)\n"
                  f"img_mean={img_mean:.4f}  txt_mean={txt_mean:.4f}")
axes[0].set_xlabel("Col"); axes[0].set_ylabel("Row")
plt.colorbar(im, ax=axes[0])

im = axes[1].imshow(std_attn_map, cmap="Blues", interpolation="nearest")
axes[1].set_title("Attention Std (variability across samples)")
axes[1].set_xlabel("Col"); axes[1].set_ylabel("Row")
plt.colorbar(im, ax=axes[1])

# Per-sample attention variance — shows if model focuses consistently
per_sample_entropy = -(img_attn * np.log(img_attn + 1e-8)).sum(axis=-1)
axes[2].hist(per_sample_entropy, bins=40, color="#4C72B0", edgecolor="white")
axes[2].axvline(np.log(64), color="red", ls="--", lw=1.5, label="Uniform (max entropy)")
axes[2].set_xlabel("Entropy of attention over 64 image tokens")
axes[2].set_ylabel("Count")
axes[2].set_title("Attention Entropy\n(low=focused, high=diffuse)")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle("AttentionReadout: Where Does the Model Look?", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "3_attention_heatmap.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Action Distribution & Decision Boundary
# ═══════════════════════════════════════════════════════════════════════════════
print("[4/8] Action distribution …")
fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 4a: 2D density of predicted actions (t+1)
ax = fig.add_subplot(gs[0, 0])
h2d, xedge, yedge, _ = ax.hist2d(
    pred_px[:, 0, 0], pred_px[:, 0, 1],
    bins=60, cmap="Blues", density=True)
ax.set_xlabel("Predicted delta_x (px)"); ax.set_ylabel("Predicted delta_y (px)")
ax.set_title("Predicted Action Density (t+1)")
ax.axhline(0, color="gray", lw=0.5, ls="--"); ax.axvline(0, color="gray", lw=0.5, ls="--")
plt.colorbar(plt.cm.ScalarMappable(cmap="Blues"), ax=ax)

# 4b: 2D density of GT actions (t+1)
ax = fig.add_subplot(gs[0, 1])
ax.hist2d(gt_px[:, 0, 0], gt_px[:, 0, 1], bins=60, cmap="Oranges", density=True)
ax.set_xlabel("GT delta_x (px)"); ax.set_ylabel("GT delta_y (px)")
ax.set_title("Ground-Truth Action Density (t+1)")
ax.axhline(0, color="gray", lw=0.5, ls="--"); ax.axvline(0, color="gray", lw=0.5, ls="--")
plt.colorbar(plt.cm.ScalarMappable(cmap="Oranges"), ax=ax)

# 4c: quiver — agent position → predicted action direction
ax = fig.add_subplot(gs[0, 2])
# Bin agent positions and average predicted direction per bin
bins = 12
x_bins = np.linspace(state_px[:, 0].min(), state_px[:, 0].max(), bins + 1)
y_bins = np.linspace(state_px[:, 1].min(), state_px[:, 1].max(), bins + 1)
qx, qy, qu, qv = [], [], [], []
for i in range(bins):
    for j in range(bins):
        mask = ((state_px[:, 0] >= x_bins[i]) & (state_px[:, 0] < x_bins[i+1]) &
                (state_px[:, 1] >= y_bins[j]) & (state_px[:, 1] < y_bins[j+1]))
        if mask.sum() > 5:
            qx.append((x_bins[i] + x_bins[i+1]) / 2)
            qy.append((y_bins[j] + y_bins[j+1]) / 2)
            qu.append(pred_px[mask, 0, 0].mean())
            qv.append(pred_px[mask, 0, 1].mean())
ax.quiver(qx, qy, qu, qv, np.sqrt(np.array(qu)**2+np.array(qv)**2),
          cmap="RdYlGn", scale=800, width=0.004)
ax.scatter(state_px[:, 0], state_px[:, 1], s=0.5, alpha=0.1, color="gray")
ax.set_xlabel("Agent X (px)"); ax.set_ylabel("Agent Y (px)")
ax.set_title("Mean Predicted Direction\nby Agent Position")
ax.set_xlim(0, 512); ax.set_ylim(0, 512); ax.invert_yaxis()

# 4d: L2 error heatmap on agent-position grid
ax = fig.add_subplot(gs[1, 0])
bins2 = 16
x_bins2 = np.linspace(0, 512, bins2 + 1)
y_bins2 = np.linspace(0, 512, bins2 + 1)
err_grid = np.full((bins2, bins2), np.nan)
for i in range(bins2):
    for j in range(bins2):
        mask = ((state_px[:, 0] >= x_bins2[i]) & (state_px[:, 0] < x_bins2[i+1]) &
                (state_px[:, 1] >= y_bins2[j]) & (state_px[:, 1] < y_bins2[j+1]))
        if mask.sum() > 3:
            err_grid[j, i] = l2[mask, 0].mean()
im = ax.imshow(err_grid, origin="upper", extent=[0,512,512,0], cmap="RdYlGn_r")
ax.set_xlabel("Agent X (px)"); ax.set_ylabel("Agent Y (px)")
ax.set_title("L2 Error Heatmap\nby Agent Position")
plt.colorbar(im, ax=ax, label="Mean L2 (px)")

# 4e: action magnitude distribution
ax = fig.add_subplot(gs[1, 1])
pred_mag = np.linalg.norm(pred_px[:, 0], axis=-1)
gt_mag   = np.linalg.norm(gt_px[:,   0], axis=-1)
ax.hist(gt_mag,   bins=50, alpha=0.6, label=f"GT   mean={gt_mag.mean():.1f}px",   density=True, color="#DD8452")
ax.hist(pred_mag, bins=50, alpha=0.6, label=f"Pred mean={pred_mag.mean():.1f}px", density=True, color="#4C72B0")
ax.set_xlabel("Action magnitude (px)"); ax.set_ylabel("Density")
ax.set_title("Action Magnitude Distribution"); ax.legend(); ax.grid(alpha=0.3)

# 4f: angular error distribution
ax = fig.add_subplot(gs[1, 2])
def safe_angle(v):
    return np.arctan2(v[:, 1], v[:, 0])
angle_pred = safe_angle(pred_px[:, 0])
angle_gt   = safe_angle(gt_px[:,   0])
angle_err  = np.abs(np.arctan2(np.sin(angle_pred - angle_gt),
                                np.cos(angle_pred - angle_gt)))
ax.hist(np.degrees(angle_err), bins=60, color="#8172B3", edgecolor="white", lw=0.3)
ax.axvline(np.degrees(angle_err.mean()), color="red", ls="--", lw=1.5,
           label=f"Mean={np.degrees(angle_err.mean()):.1f}deg")
ax.axvline(90, color="gray", ls=":", lw=1, label="90deg (random)")
ax.set_xlabel("Angular error (degrees)"); ax.set_ylabel("Count")
ax.set_title("Angular Error Distribution (t+1)"); ax.legend(); ax.grid(alpha=0.3)

plt.suptitle("Action Distribution & Decision Boundary", fontsize=13)
plt.savefig(OUT / "4_action_distribution.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Embedding PCA
# ═══════════════════════════════════════════════════════════════════════════════
print("[5/8] Embedding PCA …")
from sklearn.decomposition import PCA

n_pca = min(2000, len(adapted))
idx   = np.random.choice(len(adapted), n_pca, replace=False)

# PCA of raw image-mean VLM embeddings
pca_raw = PCA(n_components=2).fit_transform(raw_img[idx])
# PCA of adapted embeddings
pca_ada = PCA(n_components=2).fit_transform(adapted[idx])
# Color by agent position x
color_val = state_px[idx, 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sc = axes[0].scatter(pca_raw[:, 0], pca_raw[:, 1], c=color_val,
                     cmap="RdYlGn", s=4, alpha=0.5)
axes[0].set_title(f"Raw VLM Embeddings (mean img tokens)\nPCA  n={n_pca}")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
plt.colorbar(sc, ax=axes[0], label="Agent X (px)")

sc = axes[1].scatter(pca_ada[:, 0], pca_ada[:, 1], c=color_val,
                     cmap="RdYlGn", s=4, alpha=0.5)
axes[1].set_title(f"Adapted Embeddings (VLMTokenAdapter output)\nPCA  n={n_pca}")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
plt.colorbar(sc, ax=axes[1], label="Agent X (px)")

# Variance explained
pca_full_raw = PCA(n_components=20).fit(raw_img[idx])
pca_full_ada = PCA(n_components=20).fit(adapted[idx])
cumvar_raw   = np.cumsum(pca_full_raw.explained_variance_ratio_) * 100
cumvar_ada   = np.cumsum(pca_full_ada.explained_variance_ratio_) * 100
axes[2].plot(range(1, 21), cumvar_raw, "o-", label="Raw VLM (mean img)", color="#DD8452")
axes[2].plot(range(1, 21), cumvar_ada, "s-", label="Adapted",            color="#4C72B0")
axes[2].set_xlabel("Number of PCs"); axes[2].set_ylabel("Cumulative variance (%)")
axes[2].set_title("PCA Variance Explained\n(how compressed is the representation)")
axes[2].legend(); axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 100)

plt.suptitle("Embedding Space: Raw VLM vs Adapted", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "5_embedding_pca.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Flow Matching Trajectory
# ═══════════════════════════════════════════════════════════════════════════════
print("[6/8] Flow matching trajectory …")

@torch.no_grad()
def get_flow_trajectory(model, cond, n_steps=20):
    """Sample with many steps and record the path."""
    dim = cfg.action_dim * cfg.action_horizon
    x   = torch.randn(cond.shape[0], dim)
    ts  = torch.linspace(0, 1, n_steps + 1)
    traj = [x.clone()]
    for i in range(n_steps):
        t      = ts[i].expand(cond.shape[0])
        v      = model.decoder(x, t, cond)
        dt     = ts[i+1] - ts[i]
        x      = x + v * dt
        traj.append(x.clone())
    return torch.stack(traj, dim=0)  # (n_steps+1, B, dim)

# Pick 8 representative val samples
batch = next(iter(DataLoader(val_ds, batch_size=8, shuffle=True)))
embed    = batch["embedding"].float()
state    = batch["state"]
img_mask = batch.get("img_mask")
gt_flat  = batch["actions"].view(8, -1).float()

cond = model._build_cond(embed, state, img_mask)
traj = get_flow_trajectory(model, cond, n_steps=20)  # (21, 8, 32)
traj = traj.numpy()   # (21, 8, 32)
gt_np = gt_flat.numpy()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for s_idx, ax in enumerate(axes.flat):
    # Show trajectory of first 2 action dims (delta_x, delta_y at t+1)
    traj_x = traj[:, s_idx, 0] * action_std[0] + action_mean[0]
    traj_y = traj[:, s_idx, 1] * action_std[1] + action_mean[1]
    gt_x   = gt_np[s_idx, 0]   * action_std[0] + action_mean[0]
    gt_y   = gt_np[s_idx, 1]   * action_std[1] + action_mean[1]

    ax.scatter(traj_x, traj_y, c=np.arange(21), cmap="plasma", s=20, zorder=3)
    ax.plot(traj_x, traj_y, color="gray", lw=0.7, zorder=2)
    ax.scatter([traj_x[0]],  [traj_y[0]],  color="blue",  s=60, zorder=4, marker="o", label="Start (noise)")
    ax.scatter([traj_x[-1]], [traj_y[-1]], color="red",   s=60, zorder=4, marker="*", label="Final pred")
    ax.scatter([gt_x],       [gt_y],       color="green", s=60, zorder=4, marker="^", label="GT")
    ax.set_title(f"Sample {s_idx+1}", fontsize=9)
    ax.set_xlabel("delta_x (px)", fontsize=8); ax.set_ylabel("delta_y (px)", fontsize=8)
    ax.grid(alpha=0.3)
    if s_idx == 0:
        ax.legend(fontsize=7)

plt.suptitle("Flow Matching Denoising Trajectory (t+1 action, 20 steps)\n"
             "Color: blue=start(noise) → yellow=end(pred)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "6_flow_trajectory.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Gradient Flow
# ═══════════════════════════════════════════════════════════════════════════════
print("[7/8] Gradient flow …")
model.train()

# Single batch backward pass
batch    = next(iter(DataLoader(val_ds, batch_size=64, shuffle=True)))
embed    = batch["embedding"].float()
state    = batch["state"]
actions  = batch["actions"]
img_mask = batch.get("img_mask")

loss = model(embed, state, actions, img_mask)
loss.backward()
model.eval()

# Collect gradient norms by layer group
grad_data = {}
for name, param in model.named_parameters():
    if param.grad is not None and param.requires_grad:
        grad_data[name] = param.grad.norm().item()

# Group by component
groups = {
    "LoRA A":          [],
    "LoRA B":          [],
    "SpatialMLP":      [],
    "Attn Readout":    [],
    "Decoder Layer 1-2": [],
    "Decoder Layer 3-4": [],
    "Decoder Layer 5-6": [],
}
group_vals = {k: [] for k in groups}

for name, val in grad_data.items():
    if "lora.A" in name:
        group_vals["LoRA A"].append(val)
    elif "lora.B" in name:
        group_vals["LoRA B"].append(val)
    elif "spatial" in name:
        group_vals["SpatialMLP"].append(val)
    elif "readout" in name:
        group_vals["Attn Readout"].append(val)
    elif "decoder.net.layers.0" in name or "decoder.net.layers.1" in name:
        group_vals["Decoder Layer 1-2"].append(val)
    elif "decoder.net.layers.2" in name or "decoder.net.layers.3" in name:
        group_vals["Decoder Layer 3-4"].append(val)
    elif "decoder.net.layers.4" in name or "decoder.net.layers.5" in name:
        group_vals["Decoder Layer 5-6"].append(val)

labels = [k for k in group_vals if group_vals[k]]
means  = [np.mean(group_vals[k]) if group_vals[k] else 0 for k in labels]
maxes  = [np.max(group_vals[k])  if group_vals[k] else 0 for k in labels]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ["#4C72B0","#4C72B0","#DD8452","#55A868","#8172B3","#8172B3","#8172B3"][:len(labels)]
bars = axes[0].barh(labels, means, color=colors, alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Mean Gradient Norm")
axes[0].set_title("Gradient Flow: Mean |grad| per Layer Group")
axes[0].grid(axis="x", alpha=0.3)
for bar, val in zip(bars, means):
    axes[0].text(val + 1e-6, bar.get_y() + bar.get_height()/2,
                 f"{val:.2e}", va="center", fontsize=8)

# Per-parameter detail (top 20 by grad norm)
all_items = sorted(grad_data.items(), key=lambda x: x[1], reverse=True)[:20]
names_top = [n.replace("decoder.net.", "dec.").replace("adapter.", "adp.") for n, _ in all_items]
vals_top  = [v for _, v in all_items]
axes[1].barh(range(len(names_top)), vals_top, color="#4C72B0", alpha=0.8)
axes[1].set_yticks(range(len(names_top)))
axes[1].set_yticklabels(names_top, fontsize=7)
axes[1].set_xlabel("Gradient Norm")
axes[1].set_title("Top-20 Parameters by Gradient Magnitude")
axes[1].grid(axis="x", alpha=0.3)

plt.suptitle("Gradient Flow Analysis (what's actually being learned)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "7_gradient_flow.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")

# Zero grads after
for p in model.parameters():
    if p.grad is not None:
        p.grad = None


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Prediction Error vs Simulation Coverage
# ═══════════════════════════════════════════════════════════════════════════════
print("[8/8] Error vs coverage …")
sim_path = ROOT / "asset" / "result" / "sim_results.json"

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

if sim_path.exists():
    sim = json.loads(sim_path.read_text())
    eps = sim["episodes"]
    cov_list  = [e["max_coverage"]  for e in eps]
    step_list = [e["steps"]         for e in eps]
    succ_list = [e["is_success"]    for e in eps]
    replan_list = [e["replan_count"] for e in eps]

    # 8a: coverage distribution
    ax = axes[0]
    colors_ep = ["green" if s else "red" for s in succ_list]
    ax.bar(range(1, len(cov_list)+1), [c*100 for c in cov_list],
           color=colors_ep, alpha=0.8, edgecolor="white")
    ax.axhline(95, color="blue", ls="--", lw=1.5, label="95% threshold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Max Coverage (%)")
    ax.set_title(f"Coverage per Episode\n(green=success, red=fail)"
                 f"  SR={np.mean(succ_list)*100:.0f}%")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # 8b: coverage histogram
    ax = axes[1]
    ax.hist([c*100 for c in cov_list], bins=15, color="#4C72B0",
            edgecolor="white", rwidth=0.85)
    ax.axvline(95, color="red", ls="--", lw=1.5, label="95% success threshold")
    ax.axvline(np.mean(cov_list)*100, color="orange", ls="--", lw=1.5,
               label=f"Mean={np.mean(cov_list)*100:.1f}%")
    ax.set_xlabel("Max Coverage (%)"); ax.set_ylabel("Count")
    ax.set_title("Coverage Distribution\n(how close each ep gets to success)")
    ax.legend(); ax.grid(alpha=0.3)

    # 8c: coverage gap analysis — what % are within X% of success
    ax = axes[2]
    gaps_to_95 = [max(0, (0.95 - c) * 100) for c in cov_list]
    gaps_sorted = sorted(gaps_to_95)
    failed_gaps = [g for g, s in zip(gaps_to_95, succ_list) if not s]
    ax.hist(failed_gaps, bins=15, color="#DD8452", edgecolor="white",
            label=f"Failed eps (n={sum(1 for s in succ_list if not s)})")
    ax.axvline(np.mean(failed_gaps), color="red", ls="--", lw=1.5,
               label=f"Mean gap={np.mean(failed_gaps):.1f}%")
    ax.set_xlabel("Coverage gap to 95% threshold (%)"); ax.set_ylabel("Count")
    ax.set_title("Failure Gap Analysis\n(0=nearly succeeded)")
    ax.legend(); ax.grid(alpha=0.3)
else:
    for ax in axes:
        ax.text(0.5, 0.5, "sim_results.json not found", ha="center", va="center",
                transform=ax.transAxes)

plt.suptitle("Simulation Coverage Analysis", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "8_error_vs_coverage.png", dpi=140, bbox_inches="tight")
plt.close()
print("   Saved.")


print(f"\nAll 8 figures saved to {OUT}/")
print("Files:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
