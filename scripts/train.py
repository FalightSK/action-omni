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
import json
from contextlib import nullcontext
from dataclasses import asdict
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split
from transformers import get_cosine_with_min_lr_schedule_with_warmup

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from models.vla_train import VLATrainModel


def _get_embedding_dataset(dataset_name: str, cache_path: str, expect_layers=None,
                           protocol=None, config=None):
    # HDF5 caches are dataset-agnostic — the lazy on-disk reader handles any dataset
    # whose embeddings were streamed to .h5/.hdf5 by precompute.py (large datasets
    # like Language Table 10% that cannot fit the full embedding tensor in RAM).
    from data.hdf5_embeddings import is_hdf5_path, HDF5EmbeddingDataset
    if is_hdf5_path(cache_path):
        # expect_layers makes a stale cache fail loudly. Two caches built at
        # different depths are identical in shape and dtype, so without this a
        # depth ablation would reuse whichever was cached first and conclude
        # that depth has no effect.
        return HDF5EmbeddingDataset(cache_path, expect_layers=expect_layers,
                                    protocol=protocol, config=config)

    if protocol is not None:
        raise ValueError("Episode-holdout training requires a metadata-bearing HDF5 cache")

    if dataset_name == "pusht":
        from data.pusht import PushTEmbeddingDataset
        return PushTEmbeddingDataset(cache_path)
    if dataset_name == "aloha":
        from data.aloha import AlohaEmbeddingDataset
        return AlohaEmbeddingDataset(cache_path)
    if dataset_name == "language_table":
        from data.language_table import LanguageTableEmbeddingDataset
        return LanguageTableEmbeddingDataset(cache_path)
    raise NotImplementedError(f"Dataset {dataset_name!r} not yet implemented.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train VLA adapter + decoder")
    parser.add_argument("--dataset",    type=str, default="pusht",
                        help="Dataset name: pusht, aloha, language_table, ood")
    parser.add_argument("--exp",        type=str, default="exp02a",
                        help="Experiment ID: exp01, exp02a, exp03, exp04, …")
    parser.add_argument("--cache",      type=str, default=None,
                        help="Override embedding cache path")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Maximum training epochs; configured LR decay horizon is capped to this limit")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any last.pt and restart training from epoch 1.")
    parser.add_argument("--seed", type=int, help="Independent head-training seed")
    parser.add_argument("--split-manifest", type=Path,
                        help="Prepared episode holdout; requires --seed and a matching HDF5 cache")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved settings only; no cache encoding, training, or files written")
    args = parser.parse_args(argv)

    cfg = get_config(args.dataset, args.exp)
    from experiment_protocol import (apply_normalization, fixed_rng, load_protocol,
                                     restore_rng, rng_state, run_signature,
                                     seed_everything, seed_worker, write_json)
    protocol = load_protocol(args.split_manifest, args.dataset) if args.split_manifest else None
    if protocol:
        if args.seed is None:
            raise ValueError("--split-manifest requires an explicit --seed")
        apply_normalization(cfg, protocol)
        cfg.output_dir = str(ROOT / "asset/runs/episode_holdout" /
                             args.dataset / Path(cfg.output_dir).name / f"seed_{args.seed:03d}")
    if args.output_dir:
        cfg.output_dir = str(args.output_dir)
    if args.seed is not None:
        seed_everything(args.seed)
    print(f"Dataset    : {args.dataset}  |  Exp: {args.exp}")
    print(f"Output dir : {cfg.output_dir}")
    if args.epochs is not None: cfg.num_epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if cfg.batch_size < 1 or cfg.num_epochs < 1 or (args.workers is not None and args.workers < 0):
        raise ValueError("Epochs/batch size must be positive; worker count must be nonnegative")
    if (cfg.early_stop_patience < 1 or cfg.warmup_steps < 0
            or not 0 < cfg.min_lr_ratio <= 1
            or (cfg.lr_decay_epochs is not None and cfg.lr_decay_epochs < 1)):
        raise ValueError("Patience/decay epochs must be positive, warmup nonnegative, and min_lr_ratio in (0, 1]")
    if args.dry_run:
        print(json.dumps({"mode": "dry-run; no training", "config": asdict(cfg),
                          "training_seed": args.seed,
                          "protocol_id": protocol["protocol_id"] if protocol else None,
                          "cache": args.cache or cfg.embeddings_cache}, indent=2))
        return
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
    from data.hdf5_embeddings import is_hdf5_path
    is_h5    = is_hdf5_path(cache_path)
    full_ds  = _get_embedding_dataset(args.dataset, cache_path,
                                      expect_layers=cfg.vlm_extract_layers,
                                      protocol=protocol, config=cfg)
    if protocol:
        import numpy as np
        train_ds = Subset(full_ds, np.flatnonzero(~full_ds.validation_mask).tolist())
        val_ds = Subset(full_ds, np.flatnonzero(full_ds.validation_mask).tolist())
    else:
        val_len = max(1, int(len(full_ds) * 0.10))
        train_ds, val_ds = random_split(
            full_ds, [len(full_ds) - val_len, val_len],
            generator=torch.Generator().manual_seed(42))
    if len(train_ds) < cfg.batch_size:
        raise ValueError("Training split is smaller than one drop-last batch")
    # In-memory caches (.pt): num_workers=0 is optimal (data already in RAM).
    # HDF5 caches: reads hit disk per sample, so use multiple workers (each reopens
    # its own h5 handle) + persistent_workers to overlap disk I/O with GPU compute —
    # otherwise the GPU starves waiting on single-threaded reads.
    on_cuda = (device.type == "cuda")
    if is_h5:
        # Conservative RAM footprint: 6 workers x prefetch 2 x batch keeps prefetch
        # buffers modest (a too-large prefetch can exhaust system RAM on big batches).
        dl_kwargs = dict(num_workers=6, persistent_workers=True,
                         pin_memory=on_cuda, prefetch_factor=2)
    else:
        dl_kwargs = dict(num_workers=0, pin_memory=False)
    if args.workers is not None:
        dl_kwargs = dict(num_workers=args.workers, pin_memory=on_cuda)
        if args.workers:
            dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    val_generator = torch.Generator().manual_seed(4242) if args.seed is not None else None
    dl_kwargs["worker_init_fn"] = seed_worker if args.seed is not None else None
    # Keep worker-base-seed draws separate from permutation draws. Recreating
    # workers on resume must not advance the saved shuffle stream.
    worker_generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=train_generator is None,
                              sampler=RandomSampler(train_ds, generator=train_generator) if train_generator is not None else None,
                              drop_last=True, generator=worker_generator, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, generator=val_generator, **dl_kwargs)
    print(f"   Train: {len(train_ds):,}  Val: {len(val_ds):,}  Batch: {cfg.batch_size}"
          f"  | workers: {dl_kwargs['num_workers']}{' (HDF5)' if is_h5 else ''}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building VLATrainModel (adapter + decoder) …")
    model   = VLATrainModel(cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total trainable params: {n_total:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optim = torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": cfg.learning_rate * 0.5},
        {"params": model.decoder.parameters(), "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)

    total_steps = cfg.num_epochs * len(train_loader)
    decay_epochs = min(cfg.num_epochs, cfg.lr_decay_epochs or cfg.num_epochs)
    decay_steps = decay_epochs * len(train_loader)
    if decay_steps <= cfg.warmup_steps + 1:
        raise ValueError("LR decay horizon needs more steps than warmup_steps + 1")
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optim, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=decay_steps, min_lr_rate=cfg.min_lr_ratio)

    early_stop_start_epoch = max(1, (cfg.warmup_steps + len(train_loader) - 1) // len(train_loader))
    print(f"\n[3/4] Warmup/cosine to floor: {decay_epochs} epochs ({decay_steps:,} steps); "
          f"training maximum: {cfg.num_epochs} epochs ({total_steps:,} steps)")
    print(f"   Peak LR: adapter={cfg.learning_rate*0.5:.1e}  decoder={cfg.learning_rate:.1e}\n")
    print(f"   LR floor: {cfg.min_lr_ratio:.1%} of peak; patience "
          f"{cfg.early_stop_patience} is enabled after {cfg.warmup_steps} warmup updates "
          f"(first eligible validation: epoch {early_stop_start_epoch})\n")

    global_step   = 0
    best_val_loss = float("inf")
    no_improve    = 0
    start_epoch   = 1
    signature = run_signature(cfg, protocol, args.seed) if protocol else None
    metadata = {"cfg": cfg, "dataset": args.dataset, "exp_id": args.exp,
                "training_seed": args.seed, "protocol": protocol,
                "run_signature": signature, "cache_id": getattr(full_ds, "cache_id", None),
                "encoder_provenance": getattr(full_ds, "encoder_provenance", None),
                "lr_schedule": "warmup_cosine_min_lr", "lr_schedule_steps": decay_steps,
                "lr_decay_epochs": decay_epochs, "max_training_steps": total_steps,
                "early_stopping": "after_warmup", "early_stop_start_step": cfg.warmup_steps,
                "early_stop_start_epoch": early_stop_start_epoch}
    if protocol and args.fresh and (ckpt_dir / "last.pt").exists():
        raise FileExistsError("Use a new output directory instead of overwriting a protocol run")

    # ── Resume from a full-state checkpoint if present (crash resilience) ───────
    # GPU driver TDRs / power events can kill the process mid-run; last.pt is written
    # every epoch with model+optimizer+scheduler so a relaunch continues where it
    # stopped instead of losing all progress. Resume requires the SAME batch size
    # (the schedule length is fixed at construction).
    last_path = ckpt_dir / "last.pt"
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if (ck.get("lr_schedule") != metadata["lr_schedule"]
                or ck.get("lr_schedule_steps") != decay_steps
                or ck.get("max_training_steps") != total_steps
                or ck["cfg"].min_lr_ratio != cfg.min_lr_ratio):
            raise ValueError("Resume LR schedule mismatch; use a new output directory")
        if (ck.get("early_stopping") != metadata["early_stopping"]
                or ck.get("early_stop_start_step") != cfg.warmup_steps):
            raise ValueError("Resume stopping rule mismatch; use a new output directory")
        if protocol and (ck.get("run_signature") != signature
                         or ck.get("cache_id") != full_ds.cache_id):
            raise ValueError("Resume protocol/configuration/seed/cache mismatch")
        saved_bs = ck.get("batch_size")
        if saved_bs is not None and saved_bs != cfg.batch_size:
            print(f"[resume] last.pt batch_size={saved_bs} != current {cfg.batch_size}; "
                  f"the LR schedule would mismatch. Restarting fresh (use the same "
                  f"--batch-size to resume).")
        else:
            model.load_state_dict(ck["state_dict"])
            optim.load_state_dict(ck["optim"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch   = ck["epoch"] + 1
            best_val_loss = ck.get("best_val_loss", float("inf"))
            global_step   = ck.get("global_step", 0)
            no_improve    = ck.get("no_improve", 0)
            if args.seed is not None:
                if not all(k in ck for k in ("rng_state", "train_generator", "val_generator")):
                    raise ValueError("Seeded resume requires saved RNG and loader state")
                train_generator.set_state(ck["train_generator"].cpu())
                val_generator.set_state(ck["val_generator"].cpu())
                restore_rng(ck["rng_state"])
            print(f"[resume] continuing from epoch {start_epoch} "
                  f"(best_val={best_val_loss:.4f}) — last.pt @ epoch {ck['epoch']}")
    elif args.fresh and last_path.exists():
        print("[--fresh] ignoring existing last.pt; training from epoch 1.")

    if protocol:
        write_json(Path(cfg.output_dir) / "run_manifest.json", {
            "dataset": args.dataset, "exp_id": args.exp, "training_seed": args.seed,
            "protocol_id": protocol["protocol_id"], "run_signature": signature,
            "cache_id": full_ds.cache_id, "config": asdict(cfg),
            "train_frames": len(train_ds), "validation_frames": len(val_ds),
            "trainable_parameters": n_total, "torch_version": torch.__version__,
            "lr_schedule": metadata["lr_schedule"], "lr_schedule_steps": decay_steps,
            "lr_decay_epochs": decay_epochs, "max_training_steps": total_steps,
            "early_stopping": metadata["early_stopping"],
            "early_stop_start_step": cfg.warmup_steps,
            "early_stop_start_epoch": early_stop_start_epoch,
            "status": "running"})
    completed_epoch = start_epoch - 1
    stop_epoch = cfg.num_epochs if no_improve < cfg.early_stop_patience else completed_epoch
    for epoch in range(start_epoch, stop_epoch + 1):
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
            # Hugging Face cosine would rebound beyond its configured endpoint.
            if global_step < decay_steps:
                scheduler.step()
            ep_loss     += loss.item() * len(state)
            global_step += 1

        model.eval()
        val_loss = 0.0
        with (fixed_rng(4242) if args.seed is not None else nullcontext()), torch.no_grad():
            for batch in val_loader:
                embed    = batch["embedding"].to(device)
                state    = batch["state"].to(device)
                actions  = batch["actions"].to(device)
                img_mask = batch.get("img_mask")
                if img_mask is not None:
                    img_mask = img_mask.to(device)
                val_loss += model(embed, state, actions, img_mask).item() * len(state)

        avg_train = ep_loss / (len(train_loader) * cfg.batch_size)
        avg_val   = val_loss / len(val_ds)
        if not (torch.isfinite(torch.tensor(avg_train)) and torch.isfinite(torch.tensor(avg_val))):
            raise ValueError("Nonfinite training/validation loss; checkpoint not selected")
        completed_epoch = epoch
        elapsed   = time.time() - t0
        cur_lr    = scheduler.get_last_lr()[1]
        improved  = avg_val < best_val_loss
        counting_patience = global_step >= cfg.warmup_steps
        phase = ("warmup" if not counting_patience else
                 "lr_floor" if global_step >= decay_steps else "cosine_decay")
        no_improve = no_improve + 1 if counting_patience and not improved else 0
        tag = "  ★ best" if improved else ""
        patience = f"{no_improve}/{cfg.early_stop_patience}" if counting_patience else "deferred"
        print(f"Epoch {epoch:3d}/{cfg.num_epochs} ({phase})  |  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  |  "
              f"lr={cur_lr:.2e}  patience={patience}  |  {elapsed:.1f}s{tag}")

        if improved:
            best_val_loss = avg_val
            torch.save({**metadata, "epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": avg_val},
                       ckpt_dir / "best.pt")
        # Full-state checkpoint every epoch for crash-safe resume. Write to a temp
        # file then atomically replace, so a crash mid-write can't corrupt last.pt.
        tmp = ckpt_dir / "last.pt.tmp"
        torch.save({**metadata, "epoch": epoch, "state_dict": model.state_dict(),
                    "optim": optim.state_dict(), "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss, "global_step": global_step,
                    "no_improve": no_improve, "batch_size": cfg.batch_size,
                    "val_loss": avg_val, "cfg": cfg,
                    "rng_state": rng_state(),
                    "train_generator": train_generator.get_state() if train_generator is not None else None,
                    "val_generator": val_generator.get_state() if val_generator is not None else None}, tmp)
        tmp.replace(ckpt_dir / "last.pt")
        with (Path(cfg.output_dir) / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": epoch, "global_step": global_step,
                                      "train_loss": avg_train, "validation_loss": avg_val,
                                      "learning_rate": cur_lr, "no_improve": no_improve,
                                      "phase": phase,
                                      "seconds": elapsed}) + "\n")

        if no_improve >= cfg.early_stop_patience:
            print(f"\n[Early stop] No improvement for {cfg.early_stop_patience} epochs.")
            break

        if epoch % cfg.save_every == 0:
            torch.save({**metadata, "epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": avg_val},
                       ckpt_dir / f"epoch_{epoch:04d}.pt")

    torch.save({**metadata, "epoch": completed_epoch, "state_dict": model.state_dict()},
               ckpt_dir / "final.pt")
    if protocol:
        manifest_path = Path(cfg.output_dir) / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(status="complete", completed_epoch=completed_epoch,
                        stop_reason="patience" if no_improve >= cfg.early_stop_patience else "max_epochs",
                        optimizer_steps=global_step, best_validation_loss=best_val_loss)
        write_json(manifest_path, manifest)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints → {ckpt_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/evaluate.py --dataset {args.dataset} --exp {args.exp} --episodes 50")


if __name__ == "__main__":
    main()
