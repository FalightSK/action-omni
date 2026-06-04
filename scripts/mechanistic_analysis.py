"""
scripts/mechanistic_analysis.py
════════════════════════════════
Deep mechanistic analysis of how the Exp2a (DiT) model works.

Sections:
  1. DiT cross-attention heatmaps  — which VLM tokens each action step attends to
  2. DiT self-attention             — how action steps attend to each other
  3. LoRA contribution              — per-token residual magnitude vs raw VLM
  4. AttentionReadout focus         — which tokens the readout bottleneck selects
  5. Component ablation             — val-loss increase when each component is removed
  6. Denoising step evolution       — how cross-attention changes across 3 flow steps

Usage:
  python3 scripts/mechanistic_analysis.py          # Exp02a (default)
  python3 scripts/mechanistic_analysis.py --exp 1  # Exp01 MLP
  python3 scripts/mechanistic_analysis.py --exp 3  # Exp03 multi-scale
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parents[1]   # scripts/ → project root
sys.path.insert(0, str(ROOT))

from configs.registry import get_config_legacy
from models.vla_train import VLATrainModel
from data.pusht import PushTEmbeddingDataset

def get_config(exp: int): return get_config_legacy(exp, dataset="pusht")

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(exp: int, device: torch.device) -> tuple[VLATrainModel, object]:
    cfg = get_config(exp)
    tm  = VLATrainModel(cfg).to(device)
    ckpt_path = Path(cfg.output_dir) / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"])
    tm.eval()
    print(f"  Loaded exp={exp}  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return tm, cfg

def get_val_loader(cfg, device, n_batches=80):
    ds = PushTEmbeddingDataset(cfg.embeddings_cache)
    total = len(ds)
    val_n = int(total * 0.1)
    _, val_ds = random_split(ds, [total - val_n, val_n],
                             generator=torch.Generator().manual_seed(42))
    return DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

def run_batches(tm, loader, device, cfg, n_batches=60,
                hooks_fn=None, ablation_fn=None):
    """Run n_batches through the model, applying optional hooks/ablations."""
    losses, all_data = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            embed   = batch["embedding"].to(device)
            state   = batch["state"].to(device)
            actions = batch["actions"].to(device)
            mask    = batch.get("img_mask")
            if mask is not None:
                mask = mask.to(device)

            if ablation_fn:
                loss = ablation_fn(tm, embed, state, actions, mask, cfg)
            else:
                loss = tm(embed, state, actions, mask)
            losses.append(loss.item())
            all_data.append((embed.cpu(), mask.cpu() if mask is not None else None))
    return np.mean(losses), all_data

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1+2: DiT Attention Hooks
# ═══════════════════════════════════════════════════════════════════════════════

def capture_dit_attention(tm, loader, device, cfg, n_batches=40):
    """
    Capture cross-attention and self-attention weights from all DiT blocks
    using register_forward_hook (safe — no monkey-patching, no recursion).

    Returns:
      cross_weights: dict[layer_idx] -> (N, tgt_H, src_S) averaged over heads
      self_weights:  dict[layer_idx] -> (N, tgt_H, src_H) averaged over heads
      img_masks:     (N, vlm_S)
    """
    # Only DiTFlowDecoder has cross_attn blocks; MLP ResidualBlocks don't
    if not hasattr(tm.decoder, 'blocks') or not hasattr(tm.decoder.blocks[0], 'has_cross'):
        print("  [skip] MLP decoder — no DiT attention blocks")
        return None, None, None

    cross_store = defaultdict(list)
    self_store  = defaultdict(list)
    handles     = []

    def make_cross_hook(lidx):
        def hook(module, inp, output):
            # MHA returns (attn_output, attn_weights)
            if isinstance(output, (tuple, list)) and len(output) >= 2 and output[1] is not None:
                cross_store[lidx].append(output[1].detach().cpu().float())  # (B, tgt, src)
        return hook

    def make_self_hook(lidx):
        def hook(module, inp, output):
            if isinstance(output, (tuple, list)) and len(output) >= 2 and output[1] is not None:
                self_store[lidx].append(output[1].detach().cpu().float())   # (B, tgt, src)
        return hook

    for layer_idx, block in enumerate(tm.decoder.blocks):
        if block.has_cross:
            handles.append(block.cross_attn.register_forward_hook(make_cross_hook(layer_idx)))
        handles.append(block.self_attn.register_forward_hook(make_self_hook(layer_idx)))

    all_masks = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            embed = batch["embedding"].to(device)
            state = batch["state"].to(device)
            mask  = batch.get("img_mask")
            if mask is not None:
                all_masks.append(mask.cpu().float())
                mask = mask.to(device)
            B = embed.shape[0]
            H, D = cfg.action_horizon, cfg.action_dim
            cond_vec, vlm_tokens = tm._build_cond(embed, state, mask)
            x_t = torch.randn(B, H, D, device=device)
            t   = torch.zeros(B, device=device)
            tm.decoder.forward(x_t, t, cond_vec, vlm_tokens)

    for h in handles:
        h.remove()

    cross_weights = {k: torch.cat(v, dim=0) for k, v in cross_store.items()}
    self_weights  = {k: torch.cat(v, dim=0) for k, v in self_store.items()}
    img_mask_all  = torch.cat(all_masks, dim=0) if all_masks else None
    return cross_weights, self_weights, img_mask_all

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: LoRA Contribution
# ═══════════════════════════════════════════════════════════════════════════════

def capture_lora_contribution(tm, loader, device, cfg, n_batches=40):
    """
    Measure per-token LoRA residual magnitude: ||scale*B(A(h))|| / ||h||
    Uses hooks on LoRA's A and B projections to capture input/output.
    """
    input_store  = []   # raw h before LoRA
    output_store = []   # h after LoRA
    mask_store   = []

    # Single hook on the LoRA module captures both input (raw h) and output (h + residual)
    def hook(module, inp, out):
        input_store.append(inp[0].detach().cpu().float())   # raw VLM features (x)
        output_store.append(out.detach().cpu().float())     # x + scale*B(A(x))

    h_in  = None
    h_out = tm.adapter.lora.register_forward_hook(hook)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            embed = batch["embedding"].to(device)
            state = batch["state"].to(device)
            mask  = batch.get("img_mask")
            if mask is not None:
                mask_store.append(mask.cpu().float())
                mask = mask.to(device)
            tm._build_cond(embed, state, mask)

    h_out.remove()

    if not input_store:
        return None, None

    raw_all  = torch.cat(input_store,  dim=0)   # (N, seq, 1024) — before LoRA
    lora_all = torch.cat(output_store, dim=0)   # (N, seq, 1024) — after LoRA
    mask_all = torch.cat(mask_store,   dim=0) if mask_store else None

    residual    = lora_all - raw_all             # (N, seq, 1024) LoRA delta only
    raw_norm    = raw_all.norm(dim=-1)           # (N, seq)
    resid_norm  = residual.norm(dim=-1)          # (N, seq)
    ratio       = resid_norm / (raw_norm + 1e-8) # (N, seq)
    return ratio, mask_all

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: AttentionReadout Focus
# ═══════════════════════════════════════════════════════════════════════════════

def capture_readout_attention(tm, loader, device, cfg, n_batches=40):
    """Returns readout attention weights: (N, seq) via forward hook on attn module."""
    weight_store = []
    mask_store   = []

    def hook(module, inp, output):
        # MHA output: (attn_out, attn_weights)  weights: (B, 1, seq)
        if isinstance(output, (tuple, list)) and len(output) >= 2 and output[1] is not None:
            w = output[1].squeeze(1)   # (B, seq)
            weight_store.append(w.detach().cpu().float())

    handle = tm.adapter.readout.attn.register_forward_hook(hook)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            embed = batch["embedding"].to(device)
            state = batch["state"].to(device)
            mask  = batch.get("img_mask")
            if mask is not None:
                mask_store.append(mask.cpu().float())
                mask = mask.to(device)
            tm._build_cond(embed, state, mask)

    handle.remove()

    weights = torch.cat(weight_store, dim=0) if weight_store else None
    masks   = torch.cat(mask_store,   dim=0) if mask_store  else None
    return weights, masks

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Component Ablations (val-loss delta)
# ═══════════════════════════════════════════════════════════════════════════════

def component_ablations(tm, loader, device, cfg, n_batches=60):
    results = {}

    # --- Baseline ---
    baseline_loss, _ = run_batches(tm, loader, device, cfg, n_batches)
    results["Baseline"] = baseline_loss
    print(f"  Baseline val loss : {baseline_loss:.4f}")

    # --- Ablate LoRA (set scale=0) ---
    orig_scale = tm.adapter.lora.scale
    tm.adapter.lora.scale = 0.0
    loss, _ = run_batches(tm, loader, device, cfg, n_batches)
    results["LoRA=0"] = loss
    tm.adapter.lora.scale = orig_scale
    print(f"  LoRA zeroed       : {loss:.4f}  (Δ={loss-baseline_loss:+.4f})")

    # --- Ablate cross-attention (pass vlm_tokens=None to DiT) ---
    if hasattr(tm.decoder, 'blocks'):
        def ablate_cross(tm_, embed, state, actions, mask, cfg_):
            cond_vec, _ = tm_._build_cond(embed, state, mask)
            return tm_.decoder.compute_loss(actions, cond_vec, vlm_tokens=None)
        loss, _ = run_batches(tm, loader, device, cfg, n_batches,
                              ablation_fn=ablate_cross)
        results["No cross-attn"] = loss
        print(f"  No cross-attn     : {loss:.4f}  (Δ={loss-baseline_loss:+.4f})")

    # --- Ablate readout → mean-pool over image tokens ---
    def ablate_readout(tm_, embed, state, actions, mask, cfg_):
        # Bypass readout, use mean-pool over img tokens instead
        if embed.ndim == 2:
            embed = embed.unsqueeze(1)
        if mask is None:
            mask = torch.zeros(embed.shape[:2], dtype=torch.bool, device=embed.device)

        # Run through fusion + LoRA + spatial, then mean-pool
        if tm_.adapter.fusion is not None:
            h = tm_.adapter.fusion(embed)
        else:
            h = embed.squeeze(1) if embed.ndim == 4 else embed
        h = tm_.adapter.lora(h)
        h = tm_.adapter.spatial(h, mask, tm_.adapter.grid_h, tm_.adapter.grid_w)

        # Mean pool over image tokens instead of attention readout
        img_mask_exp = mask.unsqueeze(-1).float()
        n_img = img_mask_exp.sum(dim=1).clamp(min=1)
        context = (h * img_mask_exp).sum(dim=1) / n_img

        cond_vec = torch.cat([context, state], dim=-1)
        _, vlm_tokens = tm_._build_cond(embed, state, mask)  # still need vlm_tokens
        if tm_.use_dit:
            return tm_.decoder.compute_loss(actions, cond_vec, vlm_tokens)
        else:
            return tm_.decoder.compute_loss(actions.view(actions.shape[0],-1), cond_vec)

    loss, _ = run_batches(tm, loader, device, cfg, n_batches,
                          ablation_fn=ablate_readout)
    results["Readout→mean-pool"] = loss
    print(f"  Readout→mean-pool : {loss:.4f}  (Δ={loss-baseline_loss:+.4f})")

    # --- Ablate adaLN conditioning (cond_vec → zeros) ---
    if hasattr(tm.decoder, 'blocks'):
        def ablate_adaln(tm_, embed, state, actions, mask, cfg_):
            cond_vec, vlm_tokens = tm_._build_cond(embed, state, mask)
            zero_cond = torch.zeros_like(cond_vec)
            return tm_.decoder.compute_loss(actions, zero_cond, vlm_tokens)
        loss, _ = run_batches(tm, loader, device, cfg, n_batches,
                              ablation_fn=ablate_adaln)
        results["adaLN cond=0"] = loss
        print(f"  adaLN zeroed      : {loss:.4f}  (Δ={loss-baseline_loss:+.4f})")

    # --- Ablate spatial position encoding (use raw LoRA output, skip SpatialMLP PE) ---
    def ablate_spatial_pe(tm_, embed, state, actions, mask, cfg_):
        # Replace spatial MLP with a no-PE version: use zeros for PE
        orig_pos2d_fwd = tm_.adapter.spatial.pos2d.forward
        orig_pos1d_fwd = tm_.adapter.spatial.pos1d.forward

        tm_.adapter.spatial.pos2d.forward = lambda h, w: torch.zeros(
            h * w, tm_.adapter.spatial.pos2d.h_enc.shape[-1] * 2,
            device=next(tm_.parameters()).device, dtype=next(tm_.parameters()).dtype
        )
        tm_.adapter.spatial.pos1d.forward = lambda n: torch.zeros(
            n, tm_.adapter.spatial.pos1d.enc.shape[-1],
            device=next(tm_.parameters()).device, dtype=next(tm_.parameters()).dtype
        )
        try:
            cond_vec, vlm_tokens = tm_._build_cond(embed, state, mask)
            if tm_.use_dit:
                loss = tm_.decoder.compute_loss(actions, cond_vec, vlm_tokens)
            else:
                loss = tm_.decoder.compute_loss(actions.view(actions.shape[0],-1), cond_vec)
        finally:
            tm_.adapter.spatial.pos2d.forward = orig_pos2d_fwd
            tm_.adapter.spatial.pos1d.forward = orig_pos1d_fwd
        return loss

    loss, _ = run_batches(tm, loader, device, cfg, n_batches,
                          ablation_fn=ablate_spatial_pe)
    results["No spatial PE"] = loss
    print(f"  No spatial PE     : {loss:.4f}  (Δ={loss-baseline_loss:+.4f})")

    return results

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cross_attention(cross_weights, img_mask_all, out_dir, cfg, exp):
    """Plot DiT cross-attention heatmaps: one row per DiT layer."""
    if not cross_weights:
        return

    n_layers = len(cross_weights)
    fig, axes = plt.subplots(2, n_layers, figsize=(3.5 * n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    img_token_mean = img_mask_all.mean(dim=0)  # (seq,)
    img_positions  = (img_token_mean > 0.5).nonzero(as_tuple=True)[0]  # image token indices

    for lidx in range(n_layers):
        if lidx not in cross_weights:
            continue
        w = cross_weights[lidx]   # (N, action_H, vlm_S) averaged over heads
        # Average over batch and action steps
        w_mean = w.mean(dim=(0, 1))  # (vlm_S,)

        # Image token heatmap
        n_img = len(img_positions)
        grid_side = int(n_img ** 0.5)
        img_w = w_mean[img_positions].numpy()

        ax_top = axes[0, lidx]
        if grid_side ** 2 == n_img:
            hmap = img_w.reshape(grid_side, grid_side)
        else:
            hmap = img_w[:grid_side**2].reshape(grid_side, grid_side)
        im = ax_top.imshow(hmap, cmap="hot", vmin=0, aspect="equal")
        ax_top.set_title(f"Layer {lidx+1}\nCross-Attn\n(image tokens)", fontsize=9)
        ax_top.set_xticks([]); ax_top.set_yticks([])
        plt.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04)

        # img vs text bar
        ax_bot = axes[1, lidx]
        img_total = w_mean[img_positions].sum().item()
        txt_positions = (~img_mask_all[0].bool()).nonzero(as_tuple=True)[0]
        txt_total = w_mean[txt_positions].sum().item() if len(txt_positions) > 0 else 0
        total = img_total + txt_total + 1e-8
        bars = ax_bot.bar(["Image\ntokens", "Text\ntokens"],
                          [img_total / total * 100, txt_total / total * 100],
                          color=["#e67e22", "#3498db"])
        ax_bot.set_ylim(0, 105)
        ax_bot.set_ylabel("% attention")
        ax_bot.set_title(f"Img vs Text\n(Layer {lidx+1})", fontsize=9)
        for bar, val in zip(bars, [img_total/total*100, txt_total/total*100]):
            ax_bot.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Exp{exp}: DiT Cross-Attention to VLM Tokens (all layers)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "mech_01_cross_attention.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

def plot_self_attention(self_weights, out_dir, cfg, exp):
    """Plot DiT self-attention: action step → action step."""
    if not self_weights:
        return

    n_layers = len(self_weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(3.2 * n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]

    for lidx in range(n_layers):
        if lidx not in self_weights:
            continue
        w = self_weights[lidx]   # (N, H, H) averaged over heads
        w_mean = w.mean(dim=0).numpy()   # (H, H)
        ax = axes[lidx]
        im = ax.imshow(w_mean, cmap="Blues", aspect="equal", vmin=0)
        ax.set_title(f"Layer {lidx+1}\nSelf-Attn", fontsize=9)
        ax.set_xlabel("Key step"); ax.set_ylabel("Query step")
        if lidx == 0:
            ax.set_xticks([0, 7, 15]); ax.set_yticks([0, 7, 15])
        else:
            ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Exp{exp}: DiT Self-Attention (action steps → action steps)", fontsize=11)
    plt.tight_layout()
    path = out_dir / "mech_02_self_attention.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

def plot_lora_contribution(ratio, img_mask_all, out_dir, cfg, exp):
    """Plot per-token LoRA modification ratio."""
    if ratio is None:
        return

    mean_ratio = ratio.mean(dim=0).numpy()  # (seq,)
    img_positions = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
    txt_positions = (img_mask_all.mean(dim=0) <= 0.5).nonzero(as_tuple=True)[0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Spatial heatmap
    n_img = len(img_positions)
    grid_side = int(n_img ** 0.5)
    img_ratio = mean_ratio[img_positions]
    if grid_side ** 2 == n_img:
        hmap = img_ratio.reshape(grid_side, grid_side)
    else:
        hmap = img_ratio[:grid_side**2].reshape(grid_side, grid_side)
    im = axes[0].imshow(hmap, cmap="YlOrRd", aspect="equal")
    axes[0].set_title(f"LoRA Modification Ratio\n||residual||/||h|| per image token\n(spatial layout)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im, ax=axes[0], label="||LoRA|| / ||h||")

    # Image vs text comparison
    axes[1].bar(["Image tokens", "Text tokens"],
                [mean_ratio[img_positions].mean(), mean_ratio[txt_positions].mean()],
                color=["#e67e22", "#3498db"])
    axes[1].set_title("Mean LoRA Ratio\nImage vs Text tokens", fontsize=9)
    axes[1].set_ylabel("Mean ||residual|| / ||h||")

    # Per-position scatter (sorted)
    all_ratios = mean_ratio
    axes[2].plot(sorted(all_ratios), color="#2c3e50", linewidth=1.2)
    axes[2].axhline(mean_ratio[img_positions].mean(), color="#e67e22",
                    linestyle="--", label="img mean")
    axes[2].axhline(mean_ratio[txt_positions].mean() if len(txt_positions)>0 else 0,
                    color="#3498db", linestyle="--", label="txt mean")
    axes[2].set_title("LoRA Ratio per Token\n(sorted)", fontsize=9)
    axes[2].set_xlabel("Token rank"); axes[2].set_ylabel("||LoRA|| / ||h||")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Exp{exp}: LoRA Contribution — How Much Does LoRA Modify Each Token?", fontsize=11)
    plt.tight_layout()
    path = out_dir / "mech_03_lora_contribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

def plot_readout_attention(weights, img_mask_all, out_dir, cfg, exp):
    """Plot AttentionReadout spatial focus."""
    if weights is None:
        return

    mean_w = weights.mean(dim=0).numpy()   # (seq,)
    img_positions = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
    txt_positions = (img_mask_all.mean(dim=0) <= 0.5).nonzero(as_tuple=True)[0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Spatial heatmap of readout
    n_img = len(img_positions)
    grid_side = int(n_img ** 0.5)
    img_w = mean_w[img_positions]
    if grid_side ** 2 == n_img:
        hmap = img_w.reshape(grid_side, grid_side)
    else:
        hmap = img_w[:grid_side**2].reshape(grid_side, grid_side)
    im = axes[0].imshow(hmap, cmap="hot", aspect="equal")
    axes[0].set_title("AttentionReadout\nSpatial Focus (image tokens)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im, ax=axes[0], label="Attention weight")

    # Img vs text
    img_total = mean_w[img_positions].sum()
    txt_total = mean_w[txt_positions].sum() if len(txt_positions) > 0 else 0
    total = img_total + txt_total + 1e-8
    bars = axes[1].bar(["Image tokens\n(64)", "Text tokens\n(18)"],
                       [img_total/total*100, txt_total/total*100],
                       color=["#e67e22", "#3498db"])
    for bar, val in zip(bars, [img_total/total*100, txt_total/total*100]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, 115)
    axes[1].set_title("Readout: Image vs Text\n% attention share", fontsize=9)
    axes[1].set_ylabel("% total attention")

    # Token-level weight distribution
    axes[2].bar(range(len(mean_w)), sorted(mean_w, reverse=True),
                color=["#e67e22" if i < img_total*len(mean_w)/total else "#3498db"
                       for i in range(len(mean_w))])
    axes[2].set_title("Readout Weights\n(all 82 tokens, sorted desc.)", fontsize=9)
    axes[2].set_xlabel("Token rank"); axes[2].set_ylabel("Attention weight")
    axes[2].axvline(n_img, color="red", linestyle="--", alpha=0.7, label="img/txt boundary")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Exp{exp}: AttentionReadout — Which Tokens Does the Bottleneck Select?", fontsize=11)
    plt.tight_layout()
    path = out_dir / "mech_04_readout_attention.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

def plot_ablations(results, out_dir, cfg, exp):
    """Bar chart of val-loss delta per ablation."""
    baseline = results.pop("Baseline", None)
    if baseline is None:
        return

    labels = list(results.keys())
    deltas = [(results[k] - baseline) / baseline * 100 for k in labels]
    abs_losses = [results[k] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Relative increase
    colors = ["#e74c3c" if d > 5 else "#f39c12" if d > 1 else "#27ae60" for d in deltas]
    bars = axes[0].bar(labels, deltas, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"Exp{exp}: Val Loss Increase When Component Removed\n"
                      f"(baseline = {baseline:.4f})", fontsize=10)
    axes[0].set_ylabel("% increase in val loss")
    axes[0].set_xlabel("Ablated component")
    for bar, val in zip(bars, deltas):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.5,
                     f"+{val:.1f}%" if val >= 0 else f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=20)

    # Absolute losses
    all_labels = [f"Baseline\n({baseline:.4f})"] + [f"{k}\n({results[k]:.4f})" for k in labels]
    all_vals   = [baseline] + abs_losses
    acolors    = ["#2ecc71"] + colors
    axes[1].bar(all_labels, all_vals, color=acolors)
    axes[1].set_title("Absolute Val Loss per Configuration", fontsize=10)
    axes[1].set_ylabel("Val loss (CFM MSE)")
    axes[1].tick_params(axis='x', rotation=20)

    fig.suptitle(f"Exp{exp}: Component Attribution — What Carries the Performance?", fontsize=12)
    plt.tight_layout()
    path = out_dir / "mech_05_ablations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

def plot_summary_table(results_ablation, cross_weights, lora_ratio,
                       readout_weights, img_mask_all, out_dir, cfg, exp):
    """One-page summary: key numbers from all sections."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    baseline = results_ablation.get("Baseline", 0) if results_ablation else 0

    # 1. Cross-attention (last DiT layer avg)
    ax = axes[0, 0]
    if cross_weights:
        last_layer = max(cross_weights.keys())
        w = cross_weights[last_layer].mean(dim=(0,1)).numpy()  # (seq,)
        img_pos = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
        n_img = len(img_pos)
        gs = int(n_img**0.5)
        hmap = w[img_pos[:gs**2]].reshape(gs, gs)
        im = ax.imshow(hmap, cmap="hot", aspect="equal")
        ax.set_title(f"DiT Cross-Attn\n(Layer {last_layer+1}, image tokens)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, "N/A\n(MLP decoder)", ha="center", va="center")
        ax.set_title("DiT Cross-Attn", fontsize=9)

    # 2. LoRA heatmap
    ax = axes[0, 1]
    if lora_ratio is not None:
        mr = lora_ratio.mean(dim=0).numpy()
        img_pos = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
        n_img = len(img_pos); gs = int(n_img**0.5)
        hmap = mr[img_pos[:gs**2]].reshape(gs, gs)
        im = ax.imshow(hmap, cmap="YlOrRd", aspect="equal")
        ax.set_title("LoRA Modification\n||residual||/||h|| per token", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    # 3. Readout heatmap
    ax = axes[0, 2]
    if readout_weights is not None:
        mw = readout_weights.mean(dim=0).numpy()
        img_pos = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
        n_img = len(img_pos); gs = int(n_img**0.5)
        hmap = mw[img_pos[:gs**2]].reshape(gs, gs)
        im = ax.imshow(hmap, cmap="hot", aspect="equal")
        ax.set_title("AttentionReadout\nSpatial focus", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. Ablation bar chart
    ax = axes[1, 0]
    if results_ablation:
        bl = results_ablation.get("Baseline", 0)
        ablation_copy = {k: v for k, v in results_ablation.items() if k != "Baseline"}
        labels = list(ablation_copy.keys())
        deltas = [(ablation_copy[k] - bl) / bl * 100 for k in labels]
        colors = ["#e74c3c" if d > 5 else "#f39c12" if d > 1 else "#27ae60" for d in deltas]
        ax.bar(labels, deltas, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Component Attribution\n(% val loss increase)", fontsize=9)
        ax.set_ylabel("% Δ loss")
        ax.tick_params(axis='x', rotation=25, labelsize=7)

    # 5. Img vs Text split across all components
    ax = axes[1, 1]
    if readout_weights is not None and img_mask_all is not None:
        img_pos = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
        txt_pos = (img_mask_all.mean(dim=0) <= 0.5).nonzero(as_tuple=True)[0].numpy()
        sources = {}
        if readout_weights is not None:
            mw = readout_weights.mean(dim=0).numpy()
            sources["Readout"] = (mw[img_pos].sum(), mw[txt_pos].sum() if len(txt_pos)>0 else 0)
        if cross_weights:
            last = max(cross_weights.keys())
            w = cross_weights[last].mean(dim=(0,1)).numpy()
            sources["Cross-Attn\n(last layer)"] = (w[img_pos].sum(), w[txt_pos].sum() if len(txt_pos)>0 else 0)

        x = np.arange(len(sources))
        width = 0.35
        img_vals = [v[0]/(v[0]+v[1]+1e-8)*100 for v in sources.values()]
        txt_vals = [v[1]/(v[0]+v[1]+1e-8)*100 for v in sources.values()]
        ax.bar(x - width/2, img_vals, width, label="Image tokens", color="#e67e22")
        ax.bar(x + width/2, txt_vals, width, label="Text tokens", color="#3498db")
        ax.set_xticks(x); ax.set_xticklabels(list(sources.keys()), fontsize=8)
        ax.set_ylim(0, 110); ax.set_ylabel("% attention")
        ax.set_title("Image vs Text Token\nAttention Share", fontsize=9)
        ax.legend(fontsize=7)

    # 6. Key numbers text box
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"Experiment {exp} — Mechanistic Summary",
        "─" * 36,
        f"Model: {'DiT decoder' if getattr(cfg, 'use_dit_decoder', False) else 'MLP decoder'}",
        f"VLM layers: {cfg.vlm_extract_layers}",
        f"Adapter: LoRA r={cfg.lora_rank}, dim={cfg.vlm_adapter_dim}",
    ]
    if results_ablation:
        bl = results_ablation.get("Baseline", 0)
        lines += ["", "Component ablation (val loss):"]
        for k, v in results_ablation.items():
            if k != "Baseline":
                delta = (v - bl) / bl * 100
                arrow = "▲" if delta > 2 else "~"
                lines.append(f"  {arrow} {k}: {delta:+.1f}%")
    if readout_weights is not None and img_mask_all is not None:
        img_pos = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
        txt_pos = (img_mask_all.mean(dim=0) <= 0.5).nonzero(as_tuple=True)[0].numpy()
        mw = readout_weights.mean(dim=0).numpy()
        img_pct = mw[img_pos].sum() / (mw.sum() + 1e-8) * 100
        txt_pct = mw[txt_pos].sum() / (mw.sum() + 1e-8) * 100
        lines += ["", f"Readout: img={img_pct:.1f}%  txt={txt_pct:.1f}%"]
    if lora_ratio is not None:
        mr = lora_ratio.mean(dim=0).numpy()
        img_pos = (img_mask_all.mean(dim=0) > 0.5).nonzero(as_tuple=True)[0].numpy()
        lines.append(f"LoRA mean ratio: {mr[img_pos].mean():.4f} (img tokens)")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.8))

    fig.suptitle(f"Exp{exp}: Mechanistic Analysis Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "mech_00_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=2, choices=[1, 2, 3, 5])
    parser.add_argument("--batches", type=int, default=60,
                        help="Number of val batches to use (default 60 ≈ full val set)")
    args = parser.parse_args()

    device = torch.device("cpu")  # cpu for reproducible hook outputs
    print(f"\nMechanistic Analysis — Exp{args.exp}  (device: {device})")
    print("=" * 56)

    # Load
    print("\n[1/7] Loading model and data …")
    tm, cfg = load_model(args.exp, device)
    loader  = get_val_loader(cfg, device)
    out_dir = Path(cfg.output_dir) / "mechanistic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Section 1+2: Attention
    print("\n[2/7] Capturing DiT attention weights …")
    cross_w, self_w, img_mask = capture_dit_attention(
        tm, loader, device, cfg, n_batches=args.batches)

    # Section 3: LoRA
    print("\n[3/7] Capturing LoRA contribution …")
    lora_ratio, lora_mask = capture_lora_contribution(
        tm, loader, device, cfg, n_batches=args.batches)
    if img_mask is None and lora_mask is not None:
        img_mask = lora_mask

    # Section 4: Readout
    print("\n[4/7] Capturing AttentionReadout weights …")
    readout_w, readout_mask = capture_readout_attention(
        tm, loader, device, cfg, n_batches=args.batches)
    if img_mask is None and readout_mask is not None:
        img_mask = readout_mask

    # Section 5: Ablations
    print("\n[5/7] Running component ablations …")
    ablation_loader = get_val_loader(cfg, device)
    ablation_results = component_ablations(
        tm, ablation_loader, device, cfg, n_batches=args.batches)

    # Plots
    print(f"\n[6/7] Generating plots → {out_dir}/")

    if cross_w:
        plot_cross_attention(cross_w, img_mask, out_dir, cfg, args.exp)
    if self_w:
        plot_self_attention(self_w, out_dir, cfg, args.exp)
    plot_lora_contribution(lora_ratio, img_mask, out_dir, cfg, args.exp)
    plot_readout_attention(readout_w, img_mask, out_dir, cfg, args.exp)
    ablation_results["Baseline"] = ablation_results.get("Baseline",
        list(ablation_results.values())[0] if ablation_results else 0)
    plot_ablations(ablation_results, out_dir, cfg, args.exp)
    plot_summary_table(ablation_results, cross_w, lora_ratio,
                       readout_w, img_mask, out_dir, cfg, args.exp)

    print(f"\n[7/7] Done. All plots saved to {out_dir}/")
    print("  mech_00_summary.png        — one-page overview")
    print("  mech_01_cross_attention.png — DiT cross-attn per layer")
    print("  mech_02_self_attention.png  — DiT self-attn (action steps)")
    print("  mech_03_lora_contribution.png — LoRA residual magnitude")
    print("  mech_04_readout_attention.png — AttentionReadout spatial focus")
    print("  mech_05_ablations.png       — component attribution bar chart")


if __name__ == "__main__":
    main()
