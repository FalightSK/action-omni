"""
scripts/train_live.py
─────────────────────
Cache-free training: runs the FROZEN VLM live each step instead of reading a
precomputed embedding cache. Same optimiser / schedule / checkpointing as
scripts/train.py — only the source of `embed` differs (live encode_vlm vs HDF5).

Use when the precompute cache is too large to store (e.g. LIBERO's 52,970 frames
= ~10 GB). Costs a frozen VLM forward per step (no_grad, cheap for Qwen3.5-0.8B on
a fast GPU) in exchange for zero cache on disk.

    python scripts/train_live.py --dataset libero --exp exp01 --batch 128 --epochs 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parents[1]))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel


def _raw_dataset(name, cfg):
    if name == "libero":
        from data.libero import LiberoDataset
        return LiberoDataset(cfg)
    if name == "pusht":
        from data.pusht import PushTDataset
        return PushTDataset(cfg.dataset_path, cfg, cache_frames=True)
    if name == "aloha":
        from data.aloha import AlohaDataset
        return AlohaDataset(cfg)
    if name == "language_table":
        from data.language_table import LanguageTableDataset
        return LanguageTableDataset(cfg)
    raise NotImplementedError(name)


def _collate(samples):
    """Keep PIL images + texts as lists for build_vlm_inputs; stack the rest."""
    return {
        "images": [s["image"] for s in samples],
        "task_texts": [s["task_text"] for s in samples],
        "state": torch.stack([s["state"] for s in samples]),
        "actions": torch.stack([s["actions"] for s in samples]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="libero")
    ap.add_argument("--exp", default="exp01")
    ap.add_argument("--batch", type=int, default=128)   # live path: size for GPU, not cfg
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cfg = get_config(args.dataset, args.exp)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch
    device = cfg.get_device()
    print(f"Dataset: {args.dataset} | Exp: {args.exp} | device: {device} | batch: {args.batch}")

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data (raw frames; VLM runs live) ──────────────────────────────────
    ds = _raw_dataset(args.dataset, cfg)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(42))
    dl = dict(batch_size=args.batch, collate_fn=_collate, num_workers=args.workers,
              persistent_workers=args.workers > 0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl)
    val_loader = DataLoader(val_ds, shuffle=False, **dl)
    print(f"   Train {len(train_ds)}  Val {len(val_ds)}  ({len(train_loader)} steps/epoch)")

    # ── Models: frozen VLM (encode) + trainable adapter/decoder ───────────
    vlm = VLAModel(cfg)
    vlm.vlm.to(device).eval()
    model = VLATrainModel(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Frozen VLM + {n_train:,} trainable params")

    optim = torch.optim.AdamW([
        {"params": model.adapter.parameters()},
        {"params": model.decoder.parameters()},
    ], weight_decay=cfg.weight_decay)
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[cfg.learning_rate * 0.5, cfg.learning_rate],
        total_steps=total_steps, pct_start=cfg.warmup_steps / max(1, total_steps),
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)

    @torch.no_grad()
    def encode(batch):
        inputs = vlm.build_vlm_inputs(batch["images"], batch["task_texts"], device)
        tokens, img_mask = vlm.encode_vlm(inputs)
        return tokens, img_mask

    best_val, no_improve, start_epoch, gstep = float("inf"), 0, 1, 0
    last_path = ckpt_dir / "last.pt"
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("batch_size") == cfg.batch_size:
            model.load_state_dict(ck["state_dict"]); optim.load_state_dict(ck["optim"])
            scheduler.load_state_dict(ck["scheduler"]); start_epoch = ck["epoch"] + 1
            best_val = ck.get("best_val_loss", float("inf")); gstep = ck.get("global_step", 0)
            no_improve = ck.get("no_improve", 0)
            print(f"[resume] from epoch {start_epoch} (best_val={best_val:.4f})")

    print(f"[train] {cfg.num_epochs} epochs x {len(train_loader)} steps\n")
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train(); ep_loss = 0.0; t0 = time.time()
        for batch in train_loader:
            embed, img_mask = encode(batch)
            state = batch["state"].to(device); actions = batch["actions"].to(device)
            optim.zero_grad(set_to_none=True)
            loss = model(embed, state, actions, img_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step(); scheduler.step()
            ep_loss += loss.item(); gstep += 1

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                embed, img_mask = encode(batch)
                val_loss += model(embed, batch["state"].to(device),
                                  batch["actions"].to(device), img_mask).item()
        tr, vl = ep_loss / len(train_loader), val_loss / max(1, len(val_loader))
        is_best = vl < best_val
        tag = " ★ best" if is_best else ""
        print(f"Epoch {epoch:3d}/{cfg.num_epochs} | train={tr:.4f} val={vl:.4f} | "
              f"lr={scheduler.get_last_lr()[1]:.2e} | {time.time()-t0:.0f}s{tag}", flush=True)

        if is_best:
            best_val, no_improve = vl, 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": vl, "cfg": cfg, "dataset": args.dataset,
                        "exp_id": args.exp}, ckpt_dir / "best.pt")
        else:
            no_improve += 1
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_val_loss": best_val,
                    "global_step": gstep, "no_improve": no_improve,
                    "batch_size": cfg.batch_size}, ckpt_dir / "last.pt")
        if no_improve >= cfg.early_stop_patience:
            print(f"[early-stop] no val improvement for {no_improve} epochs"); break

    print(f"[done] best val={best_val:.4f}  ckpt={ckpt_dir/'best.pt'}")


if __name__ == "__main__":
    main()
