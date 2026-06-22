"""
scripts/token_info_probe.py
═══════════════════════════
Token Information Validation — Tests whether image-scene information has leaked
into the frozen VLM's text-token hidden states via causal attention.

Background
──────────
Qwen3.5 processes tokens causally: image tokens come first, text tokens follow.
By layer 24, text-token hidden states CAN contain aggregated visual information.
The heatmaps from PushT/ALOHA/LT experiments showed "random-looking" image
cross-attention — this script tests whether that's because the policy actually
reads the scene from text-token registers instead.

Three tests implemented (ordered cheapest → most expensive):

  Test 5 — Per-head attention breakdown
    Re-plot DiT cross-attention per head and per DiT block (not averaged).
    If 1-2 heads are sharply structured while others are diffuse, the "random"
    result from averaged heatmaps is an artifact.

  Test 1 — Token-type masking (necessity probe)
    Pass only image tokens / only text tokens / all tokens through the trained
    adapter+decoder. Measure val-loss increase for each condition.
    Reason: diagnoses which token group the trained model depends on WITHOUT
    retraining. Limitation: masking at inference ≠ masking at training — a
    loss increase proves the model USES those tokens, not that they are the
    only source of information.

  Test 2 — Text-token variance + linear probe (information probe)
    On ALOHA, the task prompt is IDENTICAL for every frame. Therefore any
    frame-to-frame variance in text-token hidden states is, by construction,
    visual information that causal attention has written into those positions.
    - Compute across-frame variance of text-token states vs image-token states.
    - Train linear probes (Ridge regression) to predict joint positions from:
        (a) mean-pooled image tokens
        (b) mean-pooled text tokens
        (c) final text token only
    If probe-R² (text) ≈ probe-R² (image) → text registers carry scene info.

Usage
─────
  python scripts/token_info_probe.py --dataset aloha --exp exp01
  python scripts/token_info_probe.py --dataset aloha --exp exp01 --n-probe 5000

Output (saved to <output_dir>/token_probe/)
───────────────────────────────────────────
  t5_per_head_attention.png   — Test 5: per-head cross-attention heatmaps
  t1_masking_results.json     — Test 1: val-loss per masking condition
  t1_masking_results.png      — Test 1: bar chart
  t2_variance.png             — Test 2: token-wise variance (img vs text)
  t2_probe_results.json       — Test 2: linear probe R² per token group
  t2_probe_results.png        — Test 2: probe comparison bar chart
  summary.md                  — Human-readable conclusion
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from models.vla_train import VLATrainModel


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedding_dataset(dataset_name: str, cache_path: str):
    if dataset_name in ("pusht", "aloha"):
        from data.pusht.dataset import PushTEmbeddingDataset
        return PushTEmbeddingDataset(cache_path)
    if dataset_name == "language_table":
        from data.language_table import LanguageTableEmbeddingDataset
        return LanguageTableEmbeddingDataset(cache_path)
    raise NotImplementedError(dataset_name)


def _get_val_loader(dataset_name: str, cfg, batch_size: int = 128):
    ds = _get_embedding_dataset(dataset_name, cfg.embeddings_cache)
    val_n = max(1, int(len(ds) * 0.10))
    _, val_ds = random_split(
        ds, [len(ds) - val_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _load_model(cfg, device: torch.device) -> VLATrainModel:
    ckpt_path = Path(cfg.output_dir) / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = VLATrainModel(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — Per-head attention breakdown
# ══════════════════════════════════════════════════════════════════════════════

def run_test5_per_head_attention(model: VLATrainModel, val_loader: DataLoader,
                                 device: torch.device, cfg,
                                 out_dir: Path, n_batches: int = 10) -> None:
    """
    Why: averaged cross-attention heatmaps can hide structured per-head patterns.
    This extracts attention weights per head per DiT block, showing whether any
    individual head is sharply focused on image vs text tokens.

    Method: register forward hooks on each DiTBlock's cross-attention module,
    capture raw attention weights (before softmax dropout averaging), accumulate
    across batches, then plot per-block/per-head heatmaps.
    """
    print("\n── Test 5: Per-head cross-attention breakdown ──")

    if not hasattr(model.decoder, "blocks"):
        print("  [SKIP] Decoder has no .blocks attribute — MLP decoder, not DiT.")
        return

    n_blocks = len(model.decoder.blocks)
    n_heads  = cfg.dit_num_heads
    seq_len  = cfg.img_seq_len   # total token sequence length (img + text)

    # Storage: [block][head] → (seq_len,) accumulated attention weights
    attn_accum = [[torch.zeros(seq_len) for _ in range(n_heads)]
                  for _ in range(n_blocks)]
    attn_count = 0

    # Register hooks on each block's cross-attention.
    # nn.MultiheadAttention defaults to average_attn_weights=True → returns (B, tgt, src).
    # We need per-head weights (B, n_heads, tgt, src), so monkey-patch forward to force
    # average_attn_weights=False before registering the hook.
    hooks = []
    orig_forwards = {}          # block_idx → original forward fn (for cleanup)
    block_attn_cache = [None] * n_blocks

    def make_hook(block_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                # attn_weights: (B, n_heads, n_action_tokens, seq_len)
                block_attn_cache[block_idx] = out[1].detach().cpu()
        return hook

    for b_idx, block in enumerate(model.decoder.blocks):
        for attr in ["cross_attn", "cross_attention", "attn2"]:
            ca = getattr(block, attr, None)
            if ca is not None:
                # Patch forward to request per-head weights
                orig_fwd = ca.forward
                orig_forwards[b_idx] = (ca, orig_fwd)
                def _patched(q, k, v, _orig=orig_fwd, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    return _orig(q, k, v, **kwargs)
                ca.forward = _patched
                hooks.append(ca.register_forward_hook(make_hook(b_idx)))
                break

    if not hooks:
        print("  [SKIP] Could not find cross-attention modules in DiT blocks.")
        return

    # Run batches and collect per-head attention via hooks
    batch_num = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch_num >= n_batches:
                break
            embed    = batch["embedding"].to(device)
            state    = batch["state"].to(device)
            actions  = batch["actions"].to(device)
            img_mask = batch.get("img_mask")
            if img_mask is not None:
                img_mask = img_mask.to(device)

            # Clear cache
            for i in range(n_blocks):
                block_attn_cache[i] = None

            # Forward pass — hooks fire and fill block_attn_cache
            model(embed, state, actions, img_mask)

            for b_idx in range(n_blocks):
                aw = block_attn_cache[b_idx]  # (B, n_heads, n_action, seq_len) or None
                if aw is None:
                    continue
                # Average over batch and action tokens → (n_heads, seq_len)
                avg = aw.mean(dim=(0, 2))
                for h in range(min(n_heads, avg.shape[0])):
                    sl = min(seq_len, avg.shape[1])
                    attn_accum[b_idx][h][:sl] += avg[h, :sl]

            attn_count += 1
            batch_num  += 1

    for h in hooks:
        h.remove()
    for ca, orig_fwd in orig_forwards.values():
        ca.forward = orig_fwd

    if attn_count == 0:
        print("  [SKIP] No attention weights captured — hooks may not have fired.")
        return

    # Normalise
    for b_idx in range(n_blocks):
        for h in range(n_heads):
            attn_accum[b_idx][h] /= attn_count

    # ── Plotting ──────────────────────────────────────────────────────────────
    # One row per DiT block, one column per head; color bar = attention weight
    fig, axes = plt.subplots(n_blocks, n_heads, figsize=(n_heads * 2.5, n_blocks * 1.8),
                              squeeze=False)
    fig.suptitle("Test 5 — DiT Cross-Attention per Block × Head\n"
                 "(averaged over action tokens and batches; x-axis = VLM token index)",
                 fontsize=11)

    for b_idx in range(n_blocks):
        for h in range(n_heads):
            ax = axes[b_idx][h]
            weights = attn_accum[b_idx][h].numpy()
            ax.bar(range(len(weights)), weights, color="steelblue", alpha=0.7, width=1.0)
            ax.set_title(f"Blk{b_idx+1} H{h+1}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            # Draw a vertical line at image/text boundary
            n_img = int(img_mask[0].sum().item()) if img_mask is not None else None
            if n_img:
                ax.axvline(n_img, color="red", linewidth=0.8, alpha=0.6, linestyle="--")

    axes[-1][0].set_xlabel("VLM token index  (red = img/text boundary)", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "t5_per_head_attention.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    # Quick text summary: image-attention fraction per head
    print("\n  Image-token attention fraction per head (averaged over blocks):")
    n_img_tok = int(img_mask[0].sum().item()) if img_mask is not None else seq_len
    for h in range(n_heads):
        fracs = []
        for b_idx in range(n_blocks):
            w = attn_accum[b_idx][h].numpy()
            img_w = w[:n_img_tok].sum()
            total = w.sum()
            fracs.append(img_w / (total + 1e-9))
        print(f"    Head {h+1:2d}: img_frac = {np.mean(fracs):.3f}  "
              f"(range [{min(fracs):.3f}, {max(fracs):.3f}])")


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Token-type masking (necessity probe)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_val_loss_masked(model: VLATrainModel, val_loader: DataLoader,
                             device: torch.device, mask_mode: str,
                             n_batches: int = 60) -> float:
    """
    Compute val loss under a token-masking condition:
      "all"   — baseline (no masking)
      "image" — zero out text tokens before adapter
      "text"  — zero out image tokens before adapter

    Why zero-out rather than remove: keeps sequence length fixed so attention
    patterns remain valid. Zeroing means "no information" at those positions.
    """
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            embed    = batch["embedding"].to(device)
            state    = batch["state"].to(device)
            actions  = batch["actions"].to(device)
            img_mask = batch.get("img_mask")
            if img_mask is not None:
                img_mask = img_mask.to(device)

            if mask_mode != "all" and img_mask is not None:
                embed = embed.clone()
                if embed.ndim == 3:  # (B, S, H)
                    if mask_mode == "image":
                        # zero image tokens, keep text tokens
                        embed[img_mask.unsqueeze(-1).expand_as(embed)] = 0.0
                    elif mask_mode == "text":
                        # zero text tokens, keep image tokens
                        text_mask = ~img_mask
                        embed[text_mask.unsqueeze(-1).expand_as(embed)] = 0.0

            loss = model(embed, state, actions, img_mask)
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


def run_test1_token_masking(model: VLATrainModel, val_loader: DataLoader,
                            device: torch.device, out_dir: Path,
                            n_batches: int = 60) -> dict:
    """
    Why: directly tests which token group the trained model depends on.
    A large loss increase when zeroing image tokens → model uses image positions.
    A large loss increase when zeroing text tokens → model uses text positions.
    Both large → both pathways are active.
    Neither large → something is wrong (degenerate model).
    """
    print("\n── Test 1: Token-type masking (necessity probe) ──")
    results = {}
    for mode in ["all", "text", "image"]:
        print(f"  Running condition: {mode!r} tokens retained …", end=" ", flush=True)
        loss = _compute_val_loss_masked(model, val_loader, device, mode, n_batches)
        results[mode] = loss
        print(f"val_loss = {loss:.4f}")

    baseline = results["all"]
    results["delta_image_zeroed"] = round(results["text"] - baseline, 4)
    results["delta_text_zeroed"]  = round(results["image"] - baseline, 4)
    results["pct_image_zeroed"]   = round((results["text"] / baseline - 1) * 100, 1)
    results["pct_text_zeroed"]    = round((results["image"] / baseline - 1) * 100, 1)

    print(f"\n  Baseline val loss              : {baseline:.4f}")
    print(f"  Image tokens zeroed (text only): {results['text']:.4f}  "
          f"(+{results['pct_image_zeroed']:.0f}%)")
    print(f"  Text tokens zeroed (image only): {results['image']:.4f}  "
          f"(+{results['pct_text_zeroed']:.0f}%)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels = ["Baseline\n(all tokens)", "Image zeroed\n(text only)", "Text zeroed\n(image only)"]
    values = [results["all"], results["text"], results["image"]]
    colors = ["#4c8cbf", "#e07b39", "#5aab61"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(baseline, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Val loss (flow-matching MSE)")
    ax.set_title("Test 1 — Token-type masking: necessity probe\n"
                 "Zero-out image vs text tokens from frozen VLM hidden states")
    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    out_path = out_dir / "t1_masking_results.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    out_json = out_dir / "t1_masking_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {out_json}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Text-token variance + linear probe (information probe)
# ══════════════════════════════════════════════════════════════════════════════

def run_test2_text_variance_and_probe(cfg, device: torch.device,
                                      dataset_name: str, out_dir: Path,
                                      n_probe: int = 8000) -> dict:
    """
    Why: on ALOHA/PushT the task prompt is IDENTICAL for every frame. Therefore
    any across-frame variance in text-token hidden states is VISUAL information
    written in by causal self-attention inside the frozen VLM.

    Step A — Variance:
      Compute per-token across-frame variance. If text tokens have non-trivial
      variance despite a fixed prompt, the visual information has leaked in.
      Compare against image-token variance as reference.

    Step B — Linear probe:
      Train Ridge regression from each token group's mean-pool to predict the
      ground-truth joint positions (state vector). R² measures how much task-
      relevant information is encoded. If text-token R² ≈ image-token R²,
      the text registers contain a near-complete scene encoding.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    print("\n── Test 2: Text-token variance + linear probe (information probe) ──")

    # ── Load embedding cache ───────────────────────────────────────────────────
    cache_path = Path(cfg.embeddings_cache)
    if not cache_path.exists():
        print(f"  [ERROR] Cache not found: {cache_path}")
        return {}

    print(f"  Loading cache: {cache_path} …")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    embeds   = cache["embeddings"]   # (N, S, H) bfloat16
    img_mask = cache["img_masks"]    # (N, S) bool
    states   = cache["states"]       # (N, state_dim) float32

    N = min(n_probe, len(embeds))
    print(f"  Using {N:,} of {len(embeds):,} frames for probe")

    # Down-sample
    idx = torch.randperm(len(embeds))[:N]
    embeds   = embeds[idx].float()   # (N, S, H)
    img_mask = img_mask[idx]         # (N, S)
    states   = states[idx].numpy()   # (N, state_dim)

    S, H = embeds.shape[1], embeds.shape[2]

    # ── Step A: Variance ───────────────────────────────────────────────────────
    # For each token position compute variance across frames (mean over hidden dim)
    per_token_var = embeds.var(dim=0).mean(dim=-1).numpy()  # (S,)

    # Separate into image and text positions
    # img_mask is per-frame but since prompt is fixed, mask is the same for all frames
    img_mask_row = img_mask[0].numpy()   # (S,) — reference frame
    n_img_tok = img_mask_row.sum()
    n_txt_tok = (~img_mask_row).sum()

    img_var = per_token_var[img_mask_row].mean()
    txt_var = per_token_var[~img_mask_row].mean()
    final_txt_var = per_token_var[~img_mask_row][-1]  # last text token

    print(f"\n  Token variance (across {N} frames, averaged over hidden dim):")
    print(f"    Image tokens ({n_img_tok}): mean var = {img_var:.6f}")
    print(f"    Text tokens  ({n_txt_tok}): mean var = {txt_var:.6f}")
    print(f"    Final text token:        var = {final_txt_var:.6f}")
    print(f"    Text/Image ratio:        {txt_var / (img_var + 1e-9):.4f}")

    if txt_var / (img_var + 1e-9) > 0.01:
        print("    → NON-TRIVIAL text variance: visual info IS leaking into text tokens")
    else:
        print("    → Near-zero text variance: text tokens contain only fixed prompt info")

    # Variance plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    colors = ["steelblue" if m else "tomato" for m in img_mask_row]
    ax.bar(range(S), per_token_var, color=colors, alpha=0.7, width=1.0)
    ax.set_xlabel("VLM token index")
    ax.set_ylabel("Across-frame variance (mean over hidden dim)")
    ax.set_title("Per-token variance (blue=image, red=text)\n"
                 "Non-zero text variance = visual info leaked via causal attn")
    ax.axvline(n_img_tok, color="black", linewidth=1.2, linestyle="--")

    ax2 = axes[1]
    groups = ["Image tokens\n(avg)", "Text tokens\n(avg)", "Final text token"]
    vals   = [float(img_var), float(txt_var), float(final_txt_var)]
    bar_colors = ["steelblue", "tomato", "darkred"]
    bars = ax2.bar(groups, vals, color=bar_colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v,
                 f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Mean across-frame variance")
    ax2.set_title("Variance by token group\n(same-prompt task → var = visual info)")
    plt.suptitle("Test 2A — Token variance: fixed prompt → "
                 "any text-token variance is visual information", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / "t2_variance.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out_path}")

    # ── Step B: Linear probe ────────────────────────────────────────────────────
    print("\n  Training linear probes (Ridge) on token group features …")

    # Build feature vectors for each token group
    img_feats  = embeds[:, img_mask_row].mean(dim=1).numpy()    # (N, H) mean-pool img
    txt_feats  = embeds[:, ~img_mask_row].mean(dim=1).numpy()   # (N, H) mean-pool txt
    last_feats = embeds[:, ~img_mask_row, :][:, -1, :].numpy()  # (N, H) final txt tok
    all_feats  = embeds.mean(dim=1).numpy()                      # (N, H) mean-pool all

    target = states   # (N, state_dim) — joint positions

    probe_results = {}
    for name, feats in [("all_mean_pool", all_feats),
                         ("image_mean_pool", img_feats),
                         ("text_mean_pool", txt_feats),
                         ("final_text_token", last_feats)]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            feats, target, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y_tr)
        r2 = float(reg.score(X_te, y_te))
        probe_results[name] = round(r2, 4)
        print(f"    {name:25s}: R² = {r2:.4f}")

    # Probe comparison plot
    fig, ax = plt.subplots(figsize=(8, 4))
    probe_labels = [k.replace("_", "\n") for k in probe_results.keys()]
    probe_vals   = list(probe_results.values())
    bar_colors   = ["#4c8cbf", "#5aab61", "tomato", "darkred"]
    bars = ax.bar(probe_labels, probe_vals, color=bar_colors, edgecolor="white",
                  linewidth=1.2)
    for bar, v in zip(bars, probe_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.01,
                f"R²={v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("R² (linear probe, joint position prediction)")
    ax.set_title("Test 2B — Linear probe: which token group encodes joint positions?\n"
                 "If text R² ≈ image R² → text registers carry scene information")
    ax.set_ylim(min(min(probe_vals) - 0.1, -0.05), 1.0)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out_path = out_dir / "t2_probe_results.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    probe_results["text_image_r2_ratio"] = round(
        probe_results["text_mean_pool"] / (probe_results["image_mean_pool"] + 1e-6), 4)
    probe_results["variance"] = {
        "image_mean": round(float(img_var), 6),
        "text_mean":  round(float(txt_var), 6),
        "final_text": round(float(final_txt_var), 6),
        "text_image_ratio": round(float(txt_var / (img_var + 1e-9)), 6),
    }
    out_json = out_dir / "t2_probe_results.json"
    out_json.write_text(json.dumps(probe_results, indent=2))
    print(f"  Saved → {out_json}")
    return probe_results


# ══════════════════════════════════════════════════════════════════════════════
# Summary report
# ══════════════════════════════════════════════════════════════════════════════

def _write_summary(t1: dict, t2: dict, out_dir: Path) -> None:
    """Write a human-readable markdown summary of all test results."""

    def pct(new, base):
        return f"+{(new / base - 1)*100:.0f}%" if base > 0 else "N/A"

    baseline = t1.get("all", float("nan"))
    img_zeroed_loss = t1.get("text", float("nan"))
    txt_zeroed_loss = t1.get("image", float("nan"))

    img_r2  = t2.get("image_mean_pool", float("nan"))
    txt_r2  = t2.get("text_mean_pool",  float("nan"))
    last_r2 = t2.get("final_text_token", float("nan"))
    ratio   = t2.get("text_image_r2_ratio", float("nan"))

    var = t2.get("variance", {})
    txt_var_ratio = var.get("text_image_ratio", float("nan"))

    if txt_var_ratio > 0.05 and ratio > 0.5:
        conclusion = (
            "**CONFIRMED — Visual information IS present in text tokens.**\n"
            "Text-token variance is non-trivial (ratio={:.3f}), and linear probes "
            "decode joint positions from text tokens with R²={:.3f} vs image R²={:.3f}. "
            "The frozen VLM's causal attention has aggregated scene information into "
            "the text registers. This explains the 'random' averaged image-attention "
            "heatmaps: the policy may route spatial queries through text positions "
            "rather than individual image tokens."
        ).format(txt_var_ratio, txt_r2, img_r2)
    elif txt_var_ratio > 0.01:
        conclusion = (
            "**PARTIAL — Some visual information in text tokens, but image tokens dominate.**\n"
            "Text-token variance ratio={:.3f} (small but non-zero). Linear probe R² "
            "text={:.3f} vs image={:.3f}. Text registers carry coarse scene information; "
            "image tokens remain the primary spatial signal pathway."
        ).format(txt_var_ratio, txt_r2, img_r2)
    else:
        conclusion = (
            "**NOT CONFIRMED — Text tokens carry minimal visual information.**\n"
            "Variance ratio={:.3f} ≈ 0. Text registers contain only fixed-prompt "
            "information. The 'random' image-attention heatmap is likely an artifact "
            "of head/block averaging. Re-examine per-head results (Test 5)."
        ).format(txt_var_ratio)

    md = f"""# Token Information Probe — Results Summary

## Test 5: Per-head attention
See `t5_per_head_attention.png` — check whether individual heads show structured
patterns hidden by averaging.

## Test 1: Token-type masking (necessity probe)
| Condition | Val loss | vs baseline |
|---|---|---|
| Baseline (all tokens) | {baseline:.4f} | — |
| Image tokens zeroed (text only) | {img_zeroed_loss:.4f} | {pct(img_zeroed_loss, baseline)} |
| Text tokens zeroed (image only) | {txt_zeroed_loss:.4f} | {pct(txt_zeroed_loss, baseline)} |

**Interpretation:** A large increase when image tokens are zeroed means the model
actively USES image token positions. A large increase when text tokens are zeroed
means the model USES text token positions (which would only carry visual info if
Test 2 confirms the leak).

## Test 2A: Across-frame variance (fixed prompt → variance = visual info)
| Token group | Mean variance | Ratio vs image |
|---|---|---|
| Image tokens (avg) | {var.get("image_mean", float("nan")):.6f} | 1.000 |
| Text tokens (avg) | {var.get("text_mean", float("nan")):.6f} | {txt_var_ratio:.4f} |
| Final text token | {var.get("final_text", float("nan")):.6f} | — |

## Test 2B: Linear probe R² (joint position prediction)
| Feature | R² |
|---|---|
| All tokens (mean-pool) | {t2.get("all_mean_pool", float("nan")):.4f} |
| Image tokens (mean-pool) | {img_r2:.4f} |
| Text tokens (mean-pool) | {txt_r2:.4f} |
| Final text token | {last_r2:.4f} |
| Text/Image ratio | {ratio:.4f} |

## Conclusion
{conclusion}
"""
    out_path = out_dir / "summary.md"
    out_path.write_text(md)
    print(f"\n  Summary → {out_path}")
    print(f"\n{'═'*60}")
    print("  CONCLUSION")
    print(f"{'═'*60}")
    print(f"  Text-token variance ratio : {txt_var_ratio:.4f}")
    print(f"  Probe R² image            : {img_r2:.4f}")
    print(f"  Probe R² text             : {txt_r2:.4f}")
    print(f"  Text/image R² ratio       : {ratio:.4f}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Token information validation probe")
    parser.add_argument("--dataset",   type=str, default="aloha")
    parser.add_argument("--exp",       type=str, default="exp01")
    parser.add_argument("--n-probe",   type=int, default=8000,
                        help="Max samples for linear probe (default 8000)")
    parser.add_argument("--n-batches", type=int, default=60,
                        help="Val batches for Tests 1+5 (default 60)")
    parser.add_argument("--skip-t5",  action="store_true", help="Skip Test 5")
    parser.add_argument("--skip-t1",  action="store_true", help="Skip Test 1")
    parser.add_argument("--skip-t2",  action="store_true", help="Skip Test 2")
    args = parser.parse_args()

    cfg    = get_config(args.dataset, args.exp)
    device = cfg.get_device()
    print(f"Dataset: {args.dataset}  Exp: {args.exp}  Device: {device}")

    out_dir = Path(cfg.output_dir) / "token_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    # ── Load model + val loader (needed for Test 1 and Test 5) ────────────────
    model      = _load_model(cfg, device)
    val_loader = _get_val_loader(args.dataset, cfg, batch_size=128)

    t1_results = {}
    t2_results = {}

    # ── Test 5: per-head attention ─────────────────────────────────────────────
    if not args.skip_t5:
        run_test5_per_head_attention(
            model, val_loader, device, cfg, out_dir,
            n_batches=min(args.n_batches, 10))

    # ── Test 1: token-type masking ─────────────────────────────────────────────
    if not args.skip_t1:
        t1_results = run_test1_token_masking(
            model, val_loader, device, out_dir, n_batches=args.n_batches)

    # ── Test 2: variance + linear probe ───────────────────────────────────────
    if not args.skip_t2:
        t2_results = run_test2_text_variance_and_probe(
            cfg, device, args.dataset, out_dir, n_probe=args.n_probe)

    # ── Summary ───────────────────────────────────────────────────────────────
    if t1_results and t2_results:
        _write_summary(t1_results, t2_results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
