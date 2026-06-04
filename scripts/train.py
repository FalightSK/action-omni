"""
scripts/train.py
─────────────────
Step 2 of 3: Train VLMTokenAdapter + FlowMatchingDecoder.

The frozen VLM embeddings are read from cache (built by scripts/precompute.py).
Both the LoRA-style adapter and the flow decoder are trained end-to-end.

Usage
─────
  python scripts/train.py --dataset pusht --exp exp02a
  python scripts/train.py --dataset pusht --exp exp02a --epochs 300
  python scripts/train.py --dataset pusht --exp exp02a --cache /path/to/cache.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from models.vla_train import VLATrainModel


def _get_embedding_dataset(dataset_name: str, cache_path: str):
    if dataset_name == "pusht":
        from data.pusht import PushTEmbeddingDataset
        return PushTEmbeddingDataset(cache_path)
    if dataset_name == "aloha":
        from data.aloha import AlohaEmbeddingDataset
        return AlohaEmbeddingDataset(cache_path)
    raise NotImplementedError(f"Dataset {dataset_name!r} not yet implemented.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VLA adapter + decoder")
    parser.add_argument("--dataset",    type=str, default="pusht",
                        help="Dataset name: pusht, aloha, language_table, ood")
    parser.add_argument("--exp",        type=str, default="exp02a",
                        help="Experiment ID: exp01, exp02a, exp03, exp04, …")
    parser.add_argument("--cache",      type=str, default=None,
                        help="Override embedding cache path")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.dataset, args.exp)
    print(f"Dataset    : {args.dataset}  |  Exp: {args.exp}")
    print(f"Output dir : {cfg.output_dir}")
    if args.epochs:     cfg.num_epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    device = cfg.get_device()
    print(f"Device: {device}\n")

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    cache_path = args.cache or cfg.embeddings_cache
    if not Path(cache_path).exists():
        print(f"[ERROR] Cache not found: {cache_path}")
        print(f"        Run: python scripts/precompute.py --dataset {args.dataset} --exp {args.exp}")
        sys.exit(1)

    print("[1/4] Loading embedding dataset …")
    full_ds  = _get_embedding_dataset(args.dataset, cache_path)
    val_len  = max(1, int(len(full_ds) * 0.10))
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - val_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )
    # num_workers=0 is optimal for in-memory embedding datasets on MPS/CPU.
    # pin_memory only helps for CUDA, not MPS unified memory.
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    print(f"   Train: {len(train_ds):,}  Val: {len(val_ds):,}  Batch: {cfg.batch_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building VLATrainModel (adapter + decoder) …")
    model   = VLATrainModel(cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total trainable params: {n_total:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optim = torch.optim.AdamW([
        {"params": model.adapter.parameters()},   # group 0 — adapter (0.5× LR)
        {"params": model.decoder.parameters()},   # group 1 — decoder (1× LR)
    ], weight_decay=cfg.weight_decay)

    total_steps = cfg.num_epochs * len(train_loader)
    pct_start   = cfg.warmup_steps / max(1, total_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr          = [cfg.learning_rate * 0.5, cfg.learning_rate],
        total_steps     = total_steps,
        pct_start       = pct_start,
        anneal_strategy = "cos",
        div_factor      = 25.0,
        final_div_factor = 1e4,
    )

    print(f"\n[3/4] Training {cfg.num_epochs} epochs  ({total_steps:,} steps) …")
    print(f"   Peak LR: adapter={cfg.learning_rate*0.5:.1e}  decoder={cfg.learning_rate:.1e}\n")

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

            optim.zero_grad(set_to_none=True)   # frees grad memory rather than zeroing
            loss = model(embed, state, actions, img_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            scheduler.step()
            ep_loss     += loss.item()
            global_step += 1

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

        avg_train = ep_loss  / len(train_loader)
        avg_val   = val_loss / len(val_loader)
        elapsed   = time.time() - t0
        cur_lr    = scheduler.get_last_lr()[1]
        improved  = avg_val < best_val_loss
        tag       = "  ★ best" if improved else f"  (no impr: {no_improve+1})"
        print(f"Epoch {epoch:3d}/{cfg.num_epochs}  |  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  |  "
              f"lr={cur_lr:.2e}  |  {elapsed:.1f}s{tag}")

        if improved:
            best_val_loss = avg_val
            no_improve    = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": avg_val, "cfg": cfg,
                        "dataset": args.dataset, "exp_id": args.exp},
                       ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"\n[Early stop] No improvement for {cfg.early_stop_patience} epochs.")
                break

        if epoch % cfg.save_every == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": avg_val},
                       ckpt_dir / f"epoch_{epoch:04d}.pt")

    torch.save({"epoch": cfg.num_epochs, "state_dict": model.state_dict()},
               ckpt_dir / "final.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints → {ckpt_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/evaluate.py --dataset {args.dataset} --exp {args.exp} --episodes 50")


if __name__ == "__main__":
    main()
