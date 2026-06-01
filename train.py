"""
train.py  –  Train VLMFeatureAdapter + FlowMatchingDecoder.

The frozen Qwen3.5 embeddings are read from cache.
Both the LoRA-style adapter and the flow decoder are trained end-to-end.

Usage
-----
  python train.py
  python train.py --cache asset/result/vlm_embeddings.pt --epochs 300
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config_loader import get_config
from data.pusht_dataset import PushTEmbeddingDataset
from models.vla import VLMTokenAdapter
from models.flow_matching import FlowMatchingDecoder, DiTFlowDecoder


# ── Joint model: adapter + decoder ────────────────────────────────────────────

class VLATrainModel(nn.Module):
    """
    Combines VLMTokenAdapter with a flow-matching decoder.

    Decoder selection (via cfg.use_dit_decoder):
      False → FlowMatchingDecoder  (MLP, Exp1 baseline)
      True  → DiTFlowDecoder       (transformer, Exp2+)

    The DiT decoder receives two conditioning signals:
      1. cond_vec   (B, cond_dim) — pooled VLM context + state, used for adaLN
      2. vlm_tokens (B, seq, 512) — per-token spatially-aware features, used for
                                     cross-attention at every denoising step

    Cache compatibility:
      v1: embed (B, 1024)         — legacy mean-pooled
      v2: embed (B, seq, 1024)    — single-layer full sequence
      v3: embed (B, L, seq, 1024) — multi-scale (Exp C)
    """

    def __init__(self, cfg: VLAConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.use_dit = getattr(cfg, "use_dit_decoder", False)

        self.adapter = VLMTokenAdapter(
            vlm_dim       = cfg.vlm_hidden_size,
            adapter_dim   = cfg.vlm_adapter_dim,
            lora_rank     = cfg.lora_rank,
            lora_scale    = cfg.lora_scale,
            pos_dim       = cfg.pos_enc_dim,
            readout_heads = cfg.readout_heads,
            dropout       = cfg.adapter_dropout,
            img_grid_h    = cfg.img_grid_h,
            img_grid_w    = cfg.img_grid_w,
            n_vlm_layers  = cfg.n_vlm_layers,
        )

        if self.use_dit:
            self.decoder = DiTFlowDecoder(
                action_dim     = cfg.action_dim,
                action_horizon = cfg.action_horizon,
                cond_dim       = cfg.cond_dim,
                hidden_dim     = cfg.dit_hidden_dim,
                num_layers     = cfg.dit_num_layers,
                n_heads        = cfg.dit_num_heads,
                cross_attn_dim = cfg.vlm_adapter_dim,   # 512 — after SpatialAwareMLP
                dropout        = cfg.decoder_dropout,
            )
        else:
            self.decoder = FlowMatchingDecoder(
                action_dim = cfg.action_dim * cfg.action_horizon,
                cond_dim   = cfg.cond_dim,
                hidden_dim = cfg.decoder_hidden_dim,
                num_layers = cfg.decoder_num_layers,
                dropout    = cfg.decoder_dropout,
            )

    def _build_cond(self, embed, state, img_mask):
        """
        Returns (cond_vec, vlm_tokens).

        cond_vec   : (B, cond_dim)   — pooled context ‖ state
        vlm_tokens : (B, seq, 512)   — per-token spatially-encoded features
                                       (used by DiT for cross-attention)
        """
        if embed.ndim == 2:                            # legacy v1 mean-pooled
            embed    = embed.unsqueeze(1)
            img_mask = torch.zeros(embed.shape[:2], dtype=torch.bool,
                                   device=embed.device)
        if img_mask is None:
            img_mask = torch.zeros(embed.shape[:2], dtype=torch.bool,
                                   device=embed.device)

        context, vlm_tokens = self.adapter(embed, img_mask, return_tokens=True)
        cond_vec = torch.cat([context, state], dim=-1)
        return cond_vec, vlm_tokens

    def forward(self, embed, state, actions, img_mask=None):
        cond_vec, vlm_tokens = self._build_cond(embed, state, img_mask)
        if self.use_dit:
            # DiT: actions kept as (B, H, D) — each step is a separate token
            return self.decoder.compute_loss(actions, cond_vec, vlm_tokens)
        else:
            # MLP: flatten to (B, H*D)
            flat = actions.view(actions.shape[0], -1)
            return self.decoder.compute_loss(flat, cond_vec)

    @torch.no_grad()
    def sample(self, embed, state, num_steps=3, img_mask=None):
        cond_vec, vlm_tokens = self._build_cond(embed, state, img_mask)
        if self.use_dit:
            out = self.decoder.sample(cond_vec, num_steps, vlm_tokens)  # (B, H, D)
        else:
            out = self.decoder.sample(cond_vec, num_steps)               # (B, H*D)
        # Always return (B, H*D) flat for backward-compatible inference
        return out.reshape(out.shape[0], -1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",        type=int, default=2, choices=[1, 2],
                        help="Experiment config: 1=Exp1 (baseline), 2=Exp2 B+C (default)")
    parser.add_argument("--cache",      type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.exp)
    print(f"Experiment : {args.exp}  |  output: {cfg.output_dir}")
    if args.epochs:     cfg.num_epochs  = args.epochs
    if args.batch_size: cfg.batch_size  = args.batch_size
    device = cfg.get_device()
    print(f"Device: {device}\n")

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    cache_path = Path(args.cache or cfg.embeddings_cache)
    if not cache_path.exists():
        print(f"[ERROR] Cache not found: {cache_path}")
        print("        Run: python precompute_embeddings.py")
        sys.exit(1)

    print("[1/4] Loading embedding dataset …")
    full_ds   = PushTEmbeddingDataset(str(cache_path))
    val_len   = max(1, int(len(full_ds) * 0.10))
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - val_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    print(f"   Train: {len(train_ds):,}  Val: {len(val_ds):,}  Batch: {cfg.batch_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building VLATrainModel (adapter + decoder) …")
    model  = VLATrainModel(cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total trainable params: {n_total:,}")

    # ── Optimiser (separate max LRs: adapter at 0.5×, decoder at 1×) ────
    # NOTE: do NOT set lr= here; OneCycleLR overrides it via max_lr below.
    optim = torch.optim.AdamW([
        {"params": model.adapter.parameters()},   # group 0 — adapter
        {"params": model.decoder.parameters()},   # group 1 — decoder
    ], weight_decay=cfg.weight_decay)

    total_steps = cfg.num_epochs * len(train_loader)
    pct_start   = cfg.warmup_steps / max(1, total_steps)  # warmup fraction

    # OneCycleLR: linear warmup → cosine annealing in a single scheduler.
    # max_lr is specified per param-group so the adapter stays at 0.5×.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr        = [cfg.learning_rate * 0.5, cfg.learning_rate],
        total_steps   = total_steps,
        pct_start     = pct_start,
        anneal_strategy = "cos",
        div_factor    = 25.0,      # initial_lr = max_lr / 25
        final_div_factor = 1e4,    # min_lr = initial_lr / 1e4
    )

    print(f"\n[3/4] Training {cfg.num_epochs} epochs  ({total_steps:,} steps) …")
    print(f"   Scheduler : OneCycleLR  (warmup {pct_start*100:.1f}% → cosine decay)")
    print(f"   Peak LR   : adapter={cfg.learning_rate*0.5:.1e}  decoder={cfg.learning_rate:.1e}\n")

    global_step   = 0
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        ep_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            embed    = batch["embedding"].to(device)
            state    = batch["state"].to(device)
            actions  = batch["actions"].to(device)
            img_mask = batch.get("img_mask")
            if img_mask is not None:
                img_mask = img_mask.to(device)

            loss = model(embed, state, actions, img_mask)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            scheduler.step()           # ← step every batch (OneCycleLR style)

            ep_loss     += loss.item()
            global_step += 1

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                embed    = batch["embedding"].to(device)
                state    = batch["state"].to(device)
                actions  = batch["actions"].to(device)
                img_mask = batch.get("img_mask")
                if img_mask is not None:
                    img_mask = img_mask.to(device)
                val_loss += model(embed, state, actions, img_mask).item()

        avg_train = ep_loss   / len(train_loader)
        avg_val   = val_loss  / len(val_loader)
        elapsed   = time.time() - t0
        cur_lr    = scheduler.get_last_lr()[1]   # decoder LR for display
        improved = avg_val < best_val_loss
        tag      = "  ★ best" if improved else f"  (no impr: {no_improve+1})"
        print(f"Epoch {epoch:3d}/{cfg.num_epochs}  |  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  |  "
              f"lr={cur_lr:.2e}  |  {elapsed:.1f}s{tag}")

        if improved:
            best_val_loss = avg_val
            no_improve    = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": avg_val, "cfg": cfg},
                       ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"\n[Early stop] No improvement for {cfg.early_stop_patience} epochs. Stopping.")
                break

        if epoch % cfg.save_every == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": avg_val},
                       ckpt_dir / f"epoch_{epoch:04d}.pt")

    torch.save({"epoch": cfg.num_epochs, "state_dict": model.state_dict()},
               ckpt_dir / "final.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints → {ckpt_dir}")


if __name__ == "__main__":
    main()
