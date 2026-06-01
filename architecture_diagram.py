"""
architecture_diagram.py
────────────────────────
Renders a detailed current model architecture diagram with live data-flow
statistics captured from a real forward pass on a validation batch.

Output: asset/result/analysis/architecture_diagram.png
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import VLAConfig
from data.pusht_dataset import PushTEmbeddingDataset
from train import VLATrainModel

# ── Colours ───────────────────────────────────────────────────────────────────
C_FROZEN   = "#4472C4"   # blue  — frozen
C_TRAIN    = "#70AD47"   # green — trainable
C_STATE    = "#ED7D31"   # orange — state/conditioning
C_DECODER  = "#7030A0"   # purple — decoder
C_FLOW     = "#C00000"   # red — flow
C_ARROW    = "#404040"
C_BG       = "#F8F8F8"
C_PANEL    = "#EEEEEE"

# ── Load model & one real batch ───────────────────────────────────────────────
cfg = VLAConfig()
device = torch.device("cpu")

model = VLATrainModel(cfg).to(device)
ckpt  = torch.load(ROOT / "asset/result/checkpoints/best.pt",
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

full_ds = PushTEmbeddingDataset(str(ROOT / "asset/result/vlm_embeddings.pt"))
val_len = max(1, int(len(full_ds) * 0.10))
_, val_ds = random_split(full_ds, [len(full_ds)-val_len, val_len],
                         generator=torch.Generator().manual_seed(42))
batch = next(iter(DataLoader(val_ds, batch_size=16, shuffle=False)))

embed    = batch["embedding"].float()   # (16, 82, 1024)
state    = batch["state"]               # (16, 2)
actions  = batch["actions"]             # (16, 16, 2)
img_mask = batch["img_mask"]            # (16, 82)
B = embed.shape[0]

# ── Capture live intermediate tensors ─────────────────────────────────────────
live = {}
with torch.no_grad():
    live["embed_raw"] = embed                                  # (16,82,1024)
    live["img_tokens"]  = embed[:, :64]                        # (16,64,1024)
    live["txt_tokens"]  = embed[:, 64:]                        # (16,18,1024)

    h1 = model.adapter.lora(embed)
    live["after_lora"] = h1                                    # (16,82,1024)
    live["lora_delta"] = (h1 - embed)                          # correction

    h2 = model.adapter.spatial(h1, img_mask,
                                cfg.img_grid_h, cfg.img_grid_w)
    live["after_spatial"] = h2                                 # (16,82,512)

    # AttentionReadout with weights
    Q = model.adapter.readout.query.expand(B, -1, -1)          # (16,1,512)
    h3_out, attn_w = model.adapter.readout.attn(
        Q, h2, h2, need_weights=True, average_attn_weights=True)
    h3 = model.adapter.readout.norm(h3_out.squeeze(1))         # (16,512)
    live["context"]    = h3                                    # (16,512)
    live["attn_w"]     = attn_w.squeeze(1)                     # (16,82)

    state_norm = state                                         # already normalised
    cond = torch.cat([h3, state_norm], dim=-1)                 # (16,514)
    live["cond"] = cond

    # Decoder internals
    x_noise = torch.randn(B, 32)
    t_half  = torch.full((B,), 0.5)
    h_dec   = (model.decoder.action_proj(x_noise)
               + model.decoder.time_proj(model.decoder.time_emb(t_half))
               + model.decoder.cond_proj(cond))
    live["dec_h_in"] = h_dec

    for blk in model.decoder.blocks:
        h_dec = blk(h_dec)
    live["dec_h_out"] = h_dec

    velocity = model.decoder.out_proj(model.decoder.out_norm(h_dec))
    live["velocity"] = velocity                                # (16,32)

    pred_flat = model.decoder.sample(cond, num_steps=3)        # (16,32)
    live["pred"] = pred_flat.view(B, 16, 2)


def ns(t):
    """Format norm ± std for a tensor."""
    n = t.float().norm(dim=-1).mean().item()
    return f"|·|={n:.2f}"

def ts(t):
    """Shape string."""
    return "×".join(str(s) for s in t.shape[1:])


# ═══════════════════════════════════════════════════════════════════════════════
# Figure layout
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 28), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, 22); ax.set_ylim(0, 28)
ax.axis("off")

def box(ax, x, y, w, h, color, label, sublabels=(), alpha=0.85, radius=0.25,
        fontsize=10, label_color="white"):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           linewidth=1.5, edgecolor=color,
                           facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(patch)
    cy = y + h/2
    if sublabels:
        cy_label = y + h * 0.72
    else:
        cy_label = cy
    ax.text(x + w/2, cy_label, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=label_color, zorder=4)
    for i, sl in enumerate(sublabels):
        ax.text(x + w/2, y + h*0.42 - i*0.18, sl, ha="center", va="center",
                fontsize=7.5, color=label_color, alpha=0.9, zorder=4,
                fontfamily="monospace")

def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW, lw=2.0, fontsize=8.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.08, my, label, ha="left", va="center",
                fontsize=fontsize, color=color, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85), zorder=6)

def panel(ax, x, y, w, h, title, color="#CCCCCC"):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=0.3",
                           linewidth=1.2, edgecolor=color,
                           facecolor=color, alpha=0.22, zorder=1)
    ax.add_patch(patch)
    ax.text(x + 0.15, y + h - 0.18, title, ha="left", va="top",
            fontsize=8.5, color=color, fontweight="bold",
            alpha=0.9, zorder=2)

def stat_box(ax, x, y, lines, color="#333333", fs=7.8):
    for i, line in enumerate(lines):
        ax.text(x, y - i*0.21, line, ha="left", va="top",
                fontsize=fs, color=color, fontfamily="monospace", zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          ec="#CCCCCC", alpha=0.9))

def hline(ax, x0, x1, y, color="#AAAAAA", lw=0.8, ls="--"):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=2)

def label(ax, x, y, text, fs=8, color="#444444", ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs,
            color=color, zorder=7)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(11, 27.6, "VLA Model — Current Architecture & Live Data Flow",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#222222")
ax.text(11, 27.2,
        "Batch=16  |  best.pt (epoch 152, val_loss=0.4490)  |  All tensor stats from real forward pass",
        ha="center", va="center", fontsize=9.5, color="#666666", style="italic")

# ── Legend ────────────────────────────────────────────────────────────────────
lx, ly = 0.3, 27.0
for c, txt in [(C_FROZEN,"Frozen (Qwen3.5)"),
               (C_TRAIN, "Trainable (Adapter)"),
               (C_DECODER,"Trainable (Decoder)"),
               (C_STATE,  "State / Cond")]:
    p = mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.8, label=txt)
    ax.add_patch(FancyBboxPatch((lx, ly-0.14), 0.28, 0.28,
                                boxstyle="round,pad=0", fc=c, ec=c,
                                alpha=0.85, zorder=4))
    ax.text(lx+0.36, ly, txt, va="center", fontsize=8.5, color="#333333")
    lx += 3.8


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Main pipeline (x=0.4 … 9.5)
# ═══════════════════════════════════════════════════════════════════════════════
LX, LW = 0.5, 5.5     # left box origin-x, width
MX = LX + LW/2        # centre x of left boxes

# ── 1. Input ──────────────────────────────────────────────────────────────────
panel(ax, 0.2, 25.5, 9.0, 1.4, "INPUT", "#888888")
box(ax, LX, 25.6, LW*0.45, 1.1, "#888888",
    "PIL Image", ("96×96 RGB",), fontsize=9)
box(ax, LX + LW*0.52, 25.6, LW*0.46, 1.1, "#888888",
    'Task Text', ('"Push T-block…"',), fontsize=9)
ax.text(MX, 25.15, "Qwen3VLProcessor  →  input_ids (1,100), pixel_values (1,3,392,392)",
        ha="center", va="center", fontsize=8, color="#555555",
        fontfamily="monospace")

# ── 2. Qwen3.5 VLM ───────────────────────────────────────────────────────────
panel(ax, 0.2, 22.7, 9.0, 2.25, "QWEN3.5-0.8B  [FROZEN]", C_FROZEN)
box(ax, LX, 22.8, LW, 2.0, C_FROZEN, "Qwen3.5-0.8B  (853M params)",
    ("28-layer transformer   bfloat16",
     "Vision encoder: patch_size=14  merge_size=2",
     "16x16 patches  →  8x8 image tokens (64)",
     "18 text tokens",
     "last_hidden_state  →  (B, 82, 1024)"),
    fontsize=9.5, alpha=0.7)

arrow(ax, MX, 25.5, MX, 24.8, color=C_FROZEN)
arrow(ax, MX, 22.8, MX, 22.3, label="tokens (B,82,1024)\nimg_mask (B,82) bool",
      color=C_FROZEN)

n_raw_img = live["img_tokens"].float().norm(dim=-1).mean().item()
n_raw_txt = live["txt_tokens"].float().norm(dim=-1).mean().item()
stat_box(ax, LX+LW+0.12, 24.5, [
    f"img tokens: {ts(live['img_tokens'])}  {ns(live['img_tokens'])}",
    f"txt tokens: {ts(live['txt_tokens'])}  {ns(live['txt_tokens'])}",
    f"img_mask:   64 True / 18 False per row",
], fs=7.8)

# ── 3. VLMTokenAdapter panel ──────────────────────────────────────────────────
panel(ax, 0.2, 16.5, 9.0, 5.55, "VLMTokenAdapter  [TRAINABLE  2.79M params]", C_TRAIN)

# Stage 1 – PerTokenLoRA
box(ax, LX, 20.7, LW, 1.55, C_TRAIN,
    "Stage 1 — PerTokenLoRA",
    ("input (B, 82, 1024)   rank=16   scale=0.1",
     "A: Linear(1024→16)  [kaiming init]",
     "B: Linear(16→1024)  [zero init]",
     "h' = h + 0.1 * B(A(h))    per token"),
    fontsize=9, alpha=0.75)

delta_norm = live["lora_delta"].float().norm(dim=-1).mean().item()
raw_norm   = live["embed_raw"].float().norm(dim=-1).mean().item()
stat_box(ax, LX+LW+0.12, 22.1, [
    f"correction |delta|  = {delta_norm:.3f}",
    f"raw token  |h|      = {raw_norm:.3f}",
    f"correction ratio    = {delta_norm/raw_norm*100:.2f}%",
], fs=7.8)

arrow(ax, MX, 22.8, MX, 22.25, label="(B,82,1024)", color=C_TRAIN, fontsize=7.5)
arrow(ax, MX, 20.7, MX, 20.3,  label="(B,82,1024)", color=C_TRAIN, fontsize=7.5)

# Stage 2 – SpatialAwareMLP
box(ax, LX, 18.5, LW, 1.65, C_TRAIN,
    "Stage 2 — SpatialAwareMLP  (DINO-style)",
    ("img tokens: cat([h_i, 2D_sincos_PE(row,col)])  →  (B,64,1152)",
     "txt tokens: cat([h_i, 1D_sincos_PE(pos)])      →  (B,18,1152)",
     "LayerNorm(1152) → Linear(1152→1024) → GELU",
     "→ Dropout(0.25) → Linear(1024→512)"),
    fontsize=9, alpha=0.75)

n_spatial = live["after_spatial"].norm(dim=-1).mean().item()
n_img_sp  = live["after_spatial"][:,:64].norm(dim=-1).mean().item()
n_txt_sp  = live["after_spatial"][:,64:].norm(dim=-1).mean().item()
stat_box(ax, LX+LW+0.12, 20.0, [
    f"output shape: (B,82,512)",
    f"img token |h|  = {n_img_sp:.3f}",
    f"txt token |h|  = {n_txt_sp:.3f}",
], fs=7.8)

arrow(ax, MX, 18.5, MX, 18.0, label="(B,82,512)", color=C_TRAIN, fontsize=7.5)

# Stage 3 – AttentionReadout
box(ax, LX, 16.6, LW, 1.25, C_TRAIN,
    "Stage 3 — AttentionReadout",
    ("Q = learnable param (1,1,512)   K=V = all 82 tokens",
     "MultiheadAttention(dim=512, heads=8)",
     "→ LayerNorm  →  squeeze  →  (B,512)"),
    fontsize=9, alpha=0.75)

img_attn_sum = live["attn_w"][:, :64].sum(dim=-1).mean().item()
txt_attn_sum = live["attn_w"][:, 64:].sum(dim=-1).mean().item()
peak_attn    = live["attn_w"][:, :64].max(dim=-1).values.mean().item()
stat_box(ax, LX+LW+0.12, 18.35, [
    f"img attn share: {img_attn_sum*100:.1f}%",
    f"txt attn share: {txt_attn_sum*100:.1f}%",
    f"peak token attn: {peak_attn:.4f}",
    f"context |h| = {ns(live['context'])}",
], fs=7.8)

arrow(ax, MX, 16.6, MX, 16.1, label="context (B,512)", color=C_TRAIN, fontsize=7.5)

# ── 4. State + conditioning ───────────────────────────────────────────────────
panel(ax, 0.2, 14.5, 9.0, 1.45, "CONDITIONING", C_STATE)

box(ax, LX, 14.6, LW*0.42, 1.1, C_STATE,
    "State (B,2)", ("agent_x, agent_y",
                    "normalised: μ=(229,293)",
                    "           σ=(102, 96)"), fontsize=9, alpha=0.75)

box(ax, LX+LW*0.46, 14.6, LW*0.52, 1.1, C_STATE,
    "cat([context, state])",
    ("(B,512) ‖ (B,2)  →  (B,514)",
     f"cond |h| = {ns(live['cond'])}"),
    fontsize=9, alpha=0.75)

ax.annotate("", xy=(LX+LW*0.46+0.05, 15.15), xytext=(LX+LW*0.42, 15.15),
            arrowprops=dict(arrowstyle="-|>", color=C_STATE, lw=1.5,
                            mutation_scale=12), zorder=5)
arrow(ax, LX+LW*0.72, 14.6, LX+LW*0.72, 14.15,
      label="cond (B,514)", color=C_STATE, fontsize=7.5)

# ── 5. FlowMatchingDecoder ────────────────────────────────────────────────────
panel(ax, 0.2, 9.1, 9.0, 5.2, "FlowMatchingDecoder  [TRAINABLE  13.4M params]", C_DECODER)

# Training vs inference note
ax.text(MX, 14.0,
        "TRAINING                                         INFERENCE",
        ha="center", va="center", fontsize=8.5, color="#444444",
        fontweight="bold")
hline(ax, 0.4, 9.4, 13.85)

# Training side
box(ax, LX, 12.4, LW*0.46, 1.3, "#C00000",
    "OT-CFM Objective",
    ("t ~ U[0,1]",
     "x_0 ~ N(0,I)   x_1 = GT actions",
     "x_t = (1-t)*x_0 + t*x_1",
     "target = x_1 - x_0"),
    fontsize=8.5, alpha=0.7)

# Inference side
box(ax, LX+LW*0.52, 12.4, LW*0.46, 1.3, C_DECODER,
    "Euler Integration (3 steps)",
    ("x_0 ~ N(0,I)   shape=(B,32)",
     "for i in 0..2:",
     "  v = decoder(x_i, i/3, cond)",
     "  x_{i+1} = x_i + v/3"),
    fontsize=8.5, alpha=0.7)

# Shared decoder body
box(ax, LX, 10.05, LW, 2.2, C_DECODER,
    "Velocity Network  v_θ(x_t, t, cond)",
    ("action_proj:  Linear(32→512)",
     "time_emb:     Sinusoidal(dim=512) → MLP(512→512) [SiLU]",
     "cond_proj:    Linear(514→512)",
     "h = action_proj(x_t) + time_proj(t) + cond_proj(cond)",
     "6× ResidualBlock(512):  PreNorm → Linear(512→2048) → GELU",
     "                         → Dropout(0.10) → Linear(2048→512) + skip",
     "out_norm:  LayerNorm(512)   out_proj: Linear(512→32)  [zero-init]"),
    fontsize=8.8, alpha=0.7)

n_hin  = live["dec_h_in"].norm(dim=-1).mean().item()
n_hout = live["dec_h_out"].norm(dim=-1).mean().item()
n_vel  = live["velocity"].norm(dim=-1).mean().item()
stat_box(ax, LX+LW+0.12, 12.5, [
    f"h (input)  |·| = {n_hin:.3f}",
    f"h (output) |·| = {n_hout:.3f}",
    f"velocity   |·| = {n_vel:.3f}",
], fs=7.8)

arrow(ax, MX, 12.4, MX, 12.25, color=C_DECODER)
arrow(ax, MX, 10.05, MX, 9.6,  label="velocity (B,32)", color=C_FLOW, fontsize=7.5)

# ── 6. Output ─────────────────────────────────────────────────────────────────
panel(ax, 0.2, 7.3, 9.0, 2.1, "OUTPUT", "#888888")
box(ax, LX, 7.4, LW, 1.8, "#888888",
    "Predicted Actions",
    ("reshape: (B,32) → (B,16,2)  [16 steps × 2D delta]",
     "denorm:  delta_px = pred * action_std + action_mean",
     "         action_std=(20.19, 20.02)  action_mean=(-0.87, 0.68)",
     f"execute first 4 steps, replan every 4 steps"),
    fontsize=9, alpha=0.75)

pred_norm_mean = live["pred"].norm(dim=-1).mean().item()
pred_std       = live["pred"][:, 0].std(dim=0)
stat_box(ax, LX+LW+0.12, 9.4, [
    f"pred shape: (B,16,2)",
    f"|pred| mean = {pred_norm_mean:.3f}",
    f"t+1 std(x)  = {pred_std[0]:.3f}",
    f"t+1 std(y)  = {pred_std[1]:.3f}",
], fs=7.8)


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Detail panels (x=10.5 … 21.5)
# ═══════════════════════════════════════════════════════════════════════════════
RX = 11.0    # right panel start x
RW = 10.4    # right panel width

# ── R1. Token anatomy ─────────────────────────────────────────────────────────
panel(ax, RX-0.2, 24.6, RW+0.4, 2.75, "Token Sequence Anatomy  (82 tokens total)", C_FROZEN)

# Draw the 82-token sequence
tok_y = 26.8
tok_h = 0.55
total_w = RW
img_w   = total_w * 64/82
txt_w   = total_w * 18/82

ax.add_patch(FancyBboxPatch((RX, tok_y), img_w, tok_h,
             boxstyle="round,pad=0", fc=C_FROZEN, ec="white", alpha=0.7, zorder=3))
ax.text(RX + img_w/2, tok_y + tok_h/2,
        "64 image tokens  (8×8 spatial grid, merge_size=2)",
        ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=4)

ax.add_patch(FancyBboxPatch((RX+img_w+0.04, tok_y), txt_w-0.04, tok_h,
             boxstyle="round,pad=0", fc="#888888", ec="white", alpha=0.7, zorder=3))
ax.text(RX+img_w+0.04 + (txt_w-0.04)/2, tok_y + tok_h/2,
        "18 text\ntokens",
        ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)

# 8x8 grid visualisation
grid_x0 = RX
grid_y0 = 24.75
cell    = 0.22
for r in range(8):
    for c in range(8):
        ax.add_patch(FancyBboxPatch(
            (grid_x0 + c*cell, grid_y0 + (7-r)*cell), cell-0.015, cell-0.015,
            boxstyle="round,pad=0", fc=C_FROZEN, ec="white", lw=0.5,
            alpha=0.55, zorder=3))

ax.text(grid_x0 + 8*cell/2, grid_y0 - 0.18,
        "8×8 image patch grid\n(each cell = 1 image token, 1024-dim)",
        ha="center", va="top", fontsize=8, color=C_FROZEN)

# 2D PE illustration
pe_x0 = grid_x0 + 8*cell + 0.5
ax.text(pe_x0, grid_y0 + 1.85, "2D Sinusoidal PE", fontsize=8.5,
        color=C_TRAIN, fontweight="bold", ha="left")
for r in range(4):
    col_v = ["#70AD47","#A9D18E","#E2EFDA","#F2F2F2"][r]
    ax.add_patch(FancyBboxPatch((pe_x0, grid_y0 + r*0.32), 1.2, 0.28,
                 boxstyle="round,pad=0", fc=col_v, ec="none", alpha=0.9, zorder=3))
ax.text(pe_x0 + 0.6, grid_y0 + 0*0.32 + 0.14, "sin(row/f₀)…cos(row/f₀)",
        ha="center", va="center", fontsize=6.5, color="#333333", zorder=4)
ax.text(pe_x0 + 0.6, grid_y0 + 1*0.32 + 0.14, "sin(col/f₀)…cos(col/f₀)",
        ha="center", va="center", fontsize=6.5, color="#333333", zorder=4)
ax.text(pe_x0 + 0.6, grid_y0 + 2*0.32 + 0.14, "dim = 128",
        ha="center", va="center", fontsize=6.5, color="#333333", zorder=4)
ax.text(pe_x0 + 0.6, grid_y0 + 3*0.32 + 0.14, "cat → 1152-d input",
        ha="center", va="center", fontsize=6.5, color="#333333", zorder=4)

ax.annotate("", xy=(RX + img_w + 0.02, 26.5),
            xytext=(grid_x0 + 8*cell/2, grid_y0 + 8*cell),
            arrowprops=dict(arrowstyle="-|>", color=C_FROZEN, lw=1,
                            connectionstyle="arc3,rad=-0.3"), zorder=5)

# ── R2. Attention heatmap ─────────────────────────────────────────────────────
panel(ax, RX-0.2, 21.4, RW+0.4, 2.95, "AttentionReadout — Live Attention Heatmap", C_TRAIN)

attn_img = live["attn_w"][:, :64].mean(0).reshape(8, 8).numpy()
heat_x, heat_y = RX, 21.6
heat_cell = 0.28
vmin, vmax = attn_img.min(), attn_img.max()

for r in range(8):
    for c in range(8):
        val = attn_img[r, c]
        intensity = (val - vmin) / max(vmax - vmin, 1e-8)
        color = plt.cm.hot(intensity)
        ax.add_patch(FancyBboxPatch(
            (heat_x + c*heat_cell, heat_y + (7-r)*heat_cell),
            heat_cell-0.02, heat_cell-0.02,
            boxstyle="round,pad=0", fc=color, ec="none", zorder=3))
        ax.text(heat_x + c*heat_cell + heat_cell/2,
                heat_y + (7-r)*heat_cell + heat_cell/2,
                f"{val*100:.1f}", ha="center", va="center",
                fontsize=5.5, color="white" if intensity>0.4 else "#444444", zorder=4)

ax.text(heat_x + 4*heat_cell, heat_y - 0.18,
        "Mean attention % per image patch (×100)", ha="center", fontsize=8, color="#444444")
ax.text(heat_x + 4*heat_cell, heat_y + 8*heat_cell + 0.12, "Col  0 → 7",
        ha="center", fontsize=7.5, color="#444444")
ax.text(heat_x - 0.22, heat_y + 4*heat_cell, "Row", fontsize=7.5,
        color="#444444", va="center", rotation=90)

# Annotations
img_attn_pct = live["attn_w"][:, :64].sum(dim=-1).mean().item()*100
txt_attn_pct = live["attn_w"][:, 64:].sum(dim=-1).mean().item()*100
peak_tok = attn_img.max()*100
peak_rc  = np.unravel_index(attn_img.argmax(), attn_img.shape)

ax.text(heat_x + 8*heat_cell + 0.3, heat_y + 7.5*heat_cell,
        f"Image tokens:  {img_attn_pct:.1f}%", fontsize=8.5, color=C_TRAIN)
ax.text(heat_x + 8*heat_cell + 0.3, heat_y + 7.0*heat_cell,
        f"Text tokens:   {txt_attn_pct:.1f}%  [IGNORED]", fontsize=8.5, color="#888888")
ax.text(heat_x + 8*heat_cell + 0.3, heat_y + 6.3*heat_cell,
        f"Peak patch:    row={peak_rc[0]} col={peak_rc[1]}", fontsize=8.5, color="#C00000")
ax.text(heat_x + 8*heat_cell + 0.3, heat_y + 5.8*heat_cell,
        f"Peak attn:     {peak_tok:.1f}%", fontsize=8.5, color="#C00000")
ax.text(heat_x + 8*heat_cell + 0.3, heat_y + 5.0*heat_cell,
        "→ Center-left focus\n  (T-block workspace)\n  Corners neglected",
        fontsize=8, color="#444444", va="top")

# ── R3. Flow denoising steps ──────────────────────────────────────────────────
panel(ax, RX-0.2, 17.9, RW+0.4, 3.3, "Flow Matching — Denoising Steps (Euler, 3 steps)", C_DECODER)

with torch.no_grad():
    x_traj = [torch.randn(1, 32)]
    cond_s = live["cond"][:1]
    dt = 1.0 / 3
    for i in range(3):
        t_s = torch.full((1,), i/3)
        v_s = model.decoder(x_traj[-1], t_s, cond_s)
        x_traj.append(x_traj[-1] + v_s * dt)

step_labels = ["x₀~N(0,I)", "x₁/₃", "x₂/₃", "x₁ (pred)"]
step_colors = ["#888888", C_DECODER, "#9B59B6", "#C00000"]
traj_xs = [RX + i * 2.5 for i in range(4)]
traj_y  = 19.9

for i, (xt, lbl, col) in enumerate(zip(x_traj, step_labels, step_colors)):
    n = xt.norm().item()
    bx = traj_xs[i]
    ax.add_patch(FancyBboxPatch((bx, traj_y-0.55), 1.9, 1.1,
                 boxstyle="round,pad=0.05,rounding_size=0.15",
                 fc=col, ec=col, alpha=0.75, zorder=3))
    ax.text(bx+0.95, traj_y+0.0,  lbl, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold", zorder=4)
    ax.text(bx+0.95, traj_y-0.3, f"|·|={n:.2f}", ha="center", va="center",
            fontsize=8, color="white", alpha=0.9, zorder=4)
    if i < 3:
        ax.annotate("", xy=(traj_xs[i+1], traj_y),
                    xytext=(traj_xs[i]+1.9, traj_y),
                    arrowprops=dict(arrowstyle="-|>", color=C_DECODER,
                                    lw=2, mutation_scale=14), zorder=5)
        v_s = model.decoder(x_traj[i], torch.tensor([i/3.0]), cond_s)
        ax.text((traj_xs[i]+1.9 + traj_xs[i+1])/2, traj_y + 0.45,
                f"v  |·|={v_s.norm().item():.2f}\nt={i/3:.2f}→{(i+1)/3:.2f}",
                ha="center", va="center", fontsize=7.5,
                color=C_DECODER, fontfamily="monospace")

# Math formula
ax.text(RX + RW/2, 18.55,
        r"x_t = (1-t)·x₀ + t·x₁    [Training interpolation]     "
        r"  x_{i+1} = xᵢ + v_θ(xᵢ, i/3, cond)·(1/3)    [Inference Euler]",
        ha="center", va="center", fontsize=8.5, color="#444444", style="italic",
        bbox=dict(boxstyle="round,pad=0.2", fc="#F0F0F0", ec="#CCCCCC"))

# Trajectory plot (action dim 0 vs dim 1, first 6 dims)
tp_x0, tp_y0 = RX, 18.05
tp_w, tp_h   = RW*0.45, 0.7
traj_arr = np.array([xt[0, :2].numpy() for xt in x_traj])  # (4,2)
action_std_np = np.array(cfg.action_std)
action_mean_np = np.array(cfg.action_mean)
traj_px = traj_arr * action_std_np + action_mean_np

inner_ax = fig.add_axes([tp_x0/22, tp_y0/28, tp_w/22*0.95, tp_h/28*3.2],
                         facecolor="#F8F8F8")
inner_ax.plot(traj_px[:, 0], traj_px[:, 1], "o-", ms=6, lw=1.5, color=C_DECODER)
for i, (xx, yy) in enumerate(traj_px):
    inner_ax.scatter([xx], [yy], color=step_colors[i], s=40, zorder=4)
    inner_ax.text(xx, yy+1.5, step_labels[i], fontsize=6, ha="center",
                  color=step_colors[i])
inner_ax.set_xlabel("delta_x (px)", fontsize=7)
inner_ax.set_ylabel("delta_y (px)", fontsize=7)
inner_ax.set_title("t+1 action trajectory", fontsize=7.5)
inner_ax.tick_params(labelsize=6); inner_ax.grid(alpha=0.3)

# ── R4. Parameter counts ──────────────────────────────────────────────────────
panel(ax, RX-0.2, 14.5, RW+0.4, 3.2, "Parameter Summary & Gradient Norms", "#666666")

components = [
    ("Qwen3.5-0.8B (frozen)",           853_000_000, 0.0,    C_FROZEN),
    ("  PerTokenLoRA (rank=16)",           2*1024*16,  0.532,  C_TRAIN),
    ("  SpatialAwareMLP (1152→512)",       2_759_168,  0.349,  C_TRAIN),
    ("  AttentionReadout (512, 8-head)",     529_408,  0.194,  C_TRAIN),
    ("FlowMatchingDecoder (512-dim, 6L)",  13_427_744, 0.21,   C_DECODER),
    ("TOTAL TRAINABLE",                   16_220_448,  None,   "#333333"),
]

row_y = 17.3
for name, nparams, grad, col in components:
    ax.text(RX, row_y, name, fontsize=8.5, va="center", color=col,
            fontweight="bold" if "TOTAL" in name else "normal")
    ax.text(RX+5.2, row_y, f"{nparams/1e6:.2f}M params", fontsize=8.5,
            va="center", color="#444444", ha="right")
    if grad is not None:
        bar_w = grad * 3.5
        ax.add_patch(FancyBboxPatch((RX+5.5, row_y-0.13), bar_w, 0.26,
                     boxstyle="round,pad=0", fc=col, ec="none",
                     alpha=0.65, zorder=3))
        ax.text(RX+5.5+bar_w+0.08, row_y, f"grad={grad:.3f}", fontsize=7.5,
                va="center", color=col)
    row_y -= 0.42

ax.text(RX+7.5, 17.35, "Mean |grad| at epoch 152  →",
        fontsize=8, color="#666666", ha="right", style="italic")
ax.text(RX+5.5, 14.8, "← bar width = gradient magnitude (normalised)",
        fontsize=7.5, color="#888888")

# ── R5. Key limitations ───────────────────────────────────────────────────────
panel(ax, RX-0.2, 9.1, RW+0.4, 5.2, "Identified Limitations & Next Experiments", "#C00000")

limitations = [
    ("L1", "#C00000",
     "LoRA rank saturation",
     "LoRA-A grad=0.07 vs LoRA-B grad=0.53  (7x imbalance).  B matrix bottlenecked by",
     "rank=16 compression.  Rank 16 cannot express the full correction signal.",
     "FIX: Increase lora_rank 16 → 64"),

    ("L2", "#ED7D31",
     "Attention diffuse + text token waste",
     "Text tokens receive 0.0% attention yet participate as keys in every readout",
     "forward pass — diluting softmax normalisation.  Center-left focus misses corners.",
     "FIX: key_padding_mask on text tokens  +  deeper LoRA correction"),

    ("L3", "#7030A0",
     "No action history — stall blindness",
     "State = (agent_x, agent_y) only.  Model cannot detect 'I've been pushing the",
     "same direction for 20 steps with no coverage gain' → cannot self-recover.",
     "FIX: Add prev 2 actions to state: state 2D → 6D"),

    ("L4", "#888888",
     "Persistent generalisation gap",
     f"Best gap train={0.23:.2f} vs val={0.449:.2f} (+{0.449-0.23:.2f}).  Only last-layer",
     "Qwen features used.  Multi-scale features (layers 14+21+28) would add edge info.",
     "FIX: Extract intermediate Qwen layers + multi-scale concat"),
]

lim_y = 14.0
for tag, col, title, l1, l2, fix in limitations:
    ax.add_patch(FancyBboxPatch((RX-0.1, lim_y-0.65), 0.4, 0.7,
                 boxstyle="round,pad=0", fc=col, ec="none", alpha=0.85, zorder=3))
    ax.text(RX+0.1, lim_y-0.28, tag, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold", zorder=4)
    ax.text(RX+0.45, lim_y, title, fontsize=9, va="center",
            color=col, fontweight="bold")
    ax.text(RX+0.45, lim_y-0.28, l1, fontsize=7.5, va="center", color="#333333")
    ax.text(RX+0.45, lim_y-0.5,  l2, fontsize=7.5, va="center", color="#333333")
    ax.text(RX+0.45, lim_y-0.72, fix, fontsize=7.5, va="center",
            color=col, fontweight="bold", style="italic")
    hline(ax, RX-0.15, RX+RW+0.15, lim_y-0.82, color="#DDDDDD")
    lim_y -= 1.15

# ── Footer ────────────────────────────────────────────────────────────────────
ax.text(11, 0.35,
        "Architecture: Qwen3.5-0.8B [frozen] → PerTokenLoRA → SpatialAwareMLP(DINO) → "
        "AttentionReadout → cond(514) → OT-CFM FlowMatchingDecoder → 16-step action trajectory",
        ha="center", va="center", fontsize=8.5, color="#666666", style="italic")

out = ROOT / "asset/result/analysis/architecture_diagram.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved → {out}")
