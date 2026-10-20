"""
scripts/train.py
─────────────────
Step 2 of 3: Train VLMTokenAdapter + FlowMatchingDecoder.

By default, frozen VLM embeddings are read from the cache built by
scripts/precompute.py.  ALOHA and LIBERO also support --online-encoder, which
recomputes frozen VLM features during each training/validation batch.  Both the
LoRA-style adapter and the flow decoder are trained end-to-end.

Usage
─────
  python scripts/train.py --dataset pusht --exp exp02a
  python scripts/train.py --dataset pusht --exp exp02a --epochs 300
  python scripts/train.py --dataset pusht --exp exp02a --cache /path/to/cache.pt
  python scripts/train.py --dataset aloha --exp exp05 --online-encoder
"""

from __future__ import annotations

import argparse
import json
import itertools
import math
import os
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
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
from scripts.checkpoint_io import atomic_copy_checkpoint
from scripts.run_log import log_event, write_json_atomic

_FAILURE_CONTEXT: dict = {}

def _cpu_quota_cores() -> float | None:
    """Read a Linux container CPU quota without trusting visible host CPUs."""
    try:
        quota_text, period_text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_text != "max":
            return int(quota_text) / int(period_text)
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return None


def _configure_cpu_runtime(args) -> dict:
    """Bound online-encoder CPU fan-out to container quota and loader workers."""
    quota = _cpu_quota_cores()
    requested_workers = (args.workers if args.workers is not None
                         else (6 if args.online_encoder else None))
    if args.cpu_threads is not None:
        thread_count = args.cpu_threads
    elif args.online_encoder:
        budget = max(1, math.floor(quota)) if quota is not None else 4
        thread_count = max(1, min(2, budget - 2)) if quota is not None else 4
    else:
        thread_count = torch.get_num_threads()

    effective_workers = requested_workers
    if args.online_encoder and quota is not None:
        # Keep one core for Python orchestration; DataLoader workers each run
        # PyTorch with one intra-op thread, leaving the remainder to the encoder.
        worker_budget = max(0, math.floor(quota) - thread_count - 1)
        effective_workers = min(int(requested_workers), worker_budget)
    if args.online_encoder or args.cpu_threads is not None:
        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits this before parallel work starts. The actual
            # current value is recorded so the resource decision remains visible.
            pass
    if args.online_encoder or args.workers is not None:
        args.workers = effective_workers
    try:
        cpuset = Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip()
    except OSError:
        cpuset = None
    return {
        "cpu_quota_cores": round(quota, 3) if quota is not None else None,
        "cpuset_cpus": cpuset,
        "visible_cpu_count": os.cpu_count(),
        "requested_loader_workers": requested_workers,
        "effective_loader_workers": effective_workers,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


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
    parser.add_argument("--online-encoder", action="store_true",
                        help="Encode frozen VLM features inside each train/validation batch; supports ALOHA and LIBERO")
    parser.add_argument("--profile-online-encoder", action="store_true",
                        help="Synchronously log per-microbatch CPU input-build and frozen-VLM forward timings")
    parser.add_argument("--cpu-threads", type=int,
                        help="Override PyTorch intra-op CPU threads (online mode defaults from cgroup quota)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Maximum training epochs; configured LR decay horizon is capped to this limit")
    parser.add_argument("--min-epochs", type=int, default=None,
                        help="Prevent early stopping before this epoch; patience still counts after warmup")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int,
                        help="Use this many full shuffled batches per epoch for an exact update budget")
    parser.add_argument("--max-steps", type=int,
                        help="Require epochs x steps-per-epoch to equal this optimizer-update budget")
    parser.add_argument("--warmup-steps", type=int,
                        help="Override the warmup update count")
    parser.add_argument("--lr-decay-steps", type=int,
                        help="Set the cosine-decay endpoint directly in optimizer updates")
    parser.add_argument("--checkpoint-every-steps", type=int, default=2000,
                        help="Retain two full-state recovery checkpoints at this interval")
    parser.add_argument("--probe-epochs", type=int, nargs="*", default=[],
                        help="Save inference-only snapshots at completed epochs")
    parser.add_argument("--rollout-stopping-config", type=Path,
                        help="Locked study JSON enabling simulator-success-based unbounded stopping")
    parser.add_argument("--rollout-durable-root", type=Path,
                        help="Shared durable root used by the rollout evaluator queue")
    parser.add_argument("--rollout-durable-run-dir", type=Path,
                        help="Durable mirror of this run directory for atomic last.pt and selection-checkpoint recovery")
    parser.add_argument("--rollout-study-id", type=str,
                        help="Stable study namespace for durable rollout tasks")
    parser.add_argument("--rollout-plan-id", type=str,
                        help="Full SHA-256 identity of the locked rollout study plan")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any last.pt and restart training from epoch 1.")
    parser.add_argument("--seed", type=int, help="Independent head-training seed")
    parser.add_argument("--split-manifest", type=Path,
                        help="Prepared episode holdout; requires --seed and a matching cache or raw dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=50,
                        help="Emit a live progress event every N optimizer updates")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved settings only; no cache encoding, training, or files written")
    args = parser.parse_args(argv)
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.profile_online_encoder and not args.online_encoder:
        parser.error("--profile-online-encoder requires --online-encoder")
    if (args.rollout_durable_run_dir is not None
            and not any((args.rollout_stopping_config, args.rollout_durable_root,
                         args.rollout_study_id, args.rollout_plan_id))):
        parser.error("--rollout-durable-run-dir requires rollout-stopping arguments")

    run_started = time.perf_counter()
    cfg = get_config(args.dataset, args.exp)
    rollout_config = None
    rollout_dataset_config = None
    rollout_stopping_config = None
    rollout_controller = None
    rollout_enabled_flags = (args.rollout_stopping_config, args.rollout_durable_root,
                             args.rollout_study_id, args.rollout_plan_id)
    if any(rollout_enabled_flags):
        if not all(rollout_enabled_flags):
            raise ValueError("Rollout stopping requires config, durable root, study ID, and plan ID")
        rollout_config = json.loads(args.rollout_stopping_config.read_text(encoding="utf-8"))
        if args.dataset not in {"aloha", "libero"}:
            raise ValueError("Success-stopped study supports only ALOHA and LIBERO")
        if (len(args.rollout_plan_id) != 64
                or any(character not in "0123456789abcdef" for character in args.rollout_plan_id)):
            raise ValueError("--rollout-plan-id must be a full lowercase SHA-256")
        rollout_stopping_config = rollout_config["stopping"]
        rollout_dataset_config = rollout_config["datasets"][args.dataset]
        if rollout_stopping_config.get("maximum_epochs") is not None:
            raise ValueError("Success-stopped study must not define a maximum epoch limit")
        if args.online_encoder is not True or args.split_manifest is None or args.seed is None:
            raise ValueError("Success-stopped study requires --online-encoder, --split-manifest, and --seed")
        if args.epochs is not None or args.max_steps is not None or args.fresh:
            raise ValueError("Success-stopped study disallows --epochs, --max-steps, and --fresh")
        if args.exp not in {rollout_config["backbones"]["groot"], rollout_config["backbones"]["qwen3vl"]}:
            raise ValueError("Experiment does not match a locked success-study backbone arm")
        reference_updates = int(rollout_config["reference_optimizer_updates"])
        expected_steps = int(rollout_dataset_config["steps_per_epoch"])
        reference_epochs = int(rollout_dataset_config["reference_epochs"])
        if reference_updates != expected_steps * reference_epochs:
            raise ValueError("Locked reference optimizer update geometry is inconsistent")
        if args.steps_per_epoch is not None and args.steps_per_epoch != expected_steps:
            raise ValueError("--steps-per-epoch differs from the locked success-study plan")
        args.steps_per_epoch = expected_steps
        if args.batch_size is not None and args.batch_size != int(rollout_config["effective_batch_size"]):
            raise ValueError("--batch-size differs from the locked success-study plan")
        cfg.batch_size = int(rollout_config["effective_batch_size"])
        cfg.num_epochs = reference_epochs  # LR/reference geometry only; stopping has no epoch cap.
        cfg.min_epochs = int(rollout_stopping_config["minimum_epochs"])
        cfg.warmup_steps = int(rollout_config["warmup_updates"])
        cfg.min_lr_ratio = float(rollout_config["minimum_learning_rate_ratio"])
        cfg.lr_decay_epochs = None
        if args.warmup_steps is not None and args.warmup_steps != cfg.warmup_steps:
            raise ValueError("--warmup-steps differs from the locked success-study plan")
        if args.lr_decay_steps is not None and args.lr_decay_steps != reference_updates:
            raise ValueError("--lr-decay-steps differs from the locked success-study plan")
        args.warmup_steps = cfg.warmup_steps
        args.lr_decay_steps = reference_updates
        from scripts.success_stopping import SuccessStoppingController
        rollout_controller = SuccessStoppingController(
            minimum_epochs=int(rollout_stopping_config["minimum_epochs"]),
            evaluation_every_epochs=int(rollout_stopping_config["evaluation_every_epochs"]),
            patience_epochs=int(rollout_stopping_config["patience_epochs"]),
            minimum_improvement=float(rollout_stopping_config.get("minimum_improvement", 0.0)))
    from experiment_protocol import (apply_normalization, fixed_rng, load_protocol,
                                     restore_rng, rng_state, run_signature,
                                     seed_everything, seed_worker)
    protocol = load_protocol(args.split_manifest, args.dataset) if args.split_manifest else None
    if protocol:
        if args.seed is None:
            raise ValueError("--split-manifest requires an explicit --seed")
        apply_normalization(cfg, protocol)
        cfg.output_dir = str(ROOT / "asset/runs/episode_holdout" /
                             args.dataset / Path(cfg.output_dir).name / f"seed_{args.seed:03d}")
    if args.output_dir:
        cfg.output_dir = str(args.output_dir)
    runtime_cpu = _configure_cpu_runtime(args)
    if args.seed is not None:
        seed_everything(args.seed)
    feature_mode = "online" if args.online_encoder else "cached"
    if args.online_encoder and args.dataset not in ("aloha", "libero"):
        raise ValueError("--online-encoder currently supports only aloha and libero")
    log_event("training.started", dataset=args.dataset, experiment=args.exp,
              seed=args.seed, output_dir=cfg.output_dir,
              protocol_id=protocol["protocol_id"] if protocol else None,
              feature_mode=feature_mode,
              cpu_resources=runtime_cpu,
              cache=None if args.online_encoder else (args.cache or cfg.embeddings_cache))
    if (runtime_cpu["requested_loader_workers"] is not None
            and runtime_cpu["effective_loader_workers"] is not None
            and runtime_cpu["requested_loader_workers"] > runtime_cpu["effective_loader_workers"]):
        log_event("training.loader_workers_capped", level="WARNING",
                  requested_workers=runtime_cpu["requested_loader_workers"],
                  effective_workers=runtime_cpu["effective_loader_workers"],
                  cpu_quota_cores=runtime_cpu["cpu_quota_cores"],
                  torch_threads=runtime_cpu["torch_intraop_threads"])
    print(f"Dataset    : {args.dataset}  |  Exp: {args.exp}")
    print(f"Output dir : {cfg.output_dir}")
    if args.epochs is not None: cfg.num_epochs = args.epochs
    if args.min_epochs is not None:
        if rollout_enabled_flags[0] and args.min_epochs != cfg.min_epochs:
            raise ValueError("--min-epochs differs from the locked success-study plan")
        cfg.min_epochs = args.min_epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.warmup_steps is not None: cfg.warmup_steps = args.warmup_steps
    if args.lr_decay_steps is not None:
        if args.lr_decay_steps < 1:
            raise ValueError("lr-decay-steps must be positive")
        cfg.lr_decay_epochs = None
    if (cfg.batch_size < 1 or cfg.num_epochs < 1
            or (args.steps_per_epoch is not None and args.steps_per_epoch < 1)
            or (args.max_steps is not None and args.max_steps < 1)
            or args.checkpoint_every_steps < 1
            or args.log_every_steps < 1
            or (args.cpu_threads is not None and args.cpu_threads < 1)
            or any(epoch < 1 or epoch > cfg.num_epochs for epoch in args.probe_epochs)
            or (args.workers is not None and args.workers < 0)):
        raise ValueError("Epochs/batch size must be positive; worker count must be nonnegative")
    if (cfg.early_stop_patience < 1 or cfg.warmup_steps < 0
            or not 0 < cfg.min_lr_ratio <= 1
            or (cfg.lr_decay_epochs is not None and cfg.lr_decay_epochs < 1)
            or not 0 <= cfg.min_epochs <= cfg.num_epochs):
        raise ValueError("Invalid patience/decay/warmup/LR floor; min_epochs must be between 0 and num_epochs")
    if args.dry_run:
        print(json.dumps({"mode": "dry-run; no training", "config": asdict(cfg),
                          "training_seed": args.seed,
                          "protocol_id": protocol["protocol_id"] if protocol else None,
                          "feature_mode": feature_mode,
                          "cache": None if args.online_encoder else (args.cache or cfg.embeddings_cache)}, indent=2))
        return
    device = cfg.get_device()
    print(f"Device: {device}\n")

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(cfg.output_dir) / "progress.json"
    _FAILURE_CONTEXT.update(progress_path=progress_path, dataset=args.dataset,
                            experiment=args.exp, manifest_path=None)
    progress = {"status": "starting", "stage": "startup", "dataset": args.dataset,
                "experiment": args.exp, "seed": args.seed,
                "started_utc": datetime.now(timezone.utc).isoformat()}

    def update_progress(stage: str, **fields) -> None:
        progress.update(status="running", stage=stage,
                        updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write_json_atomic(progress_path, progress)

    requested_cache = args.cache or cfg.embeddings_cache
    cache_path = None if args.online_encoder else requested_cache
    if args.online_encoder and args.cache:
        log_event("training.cache_argument_ignored", level="WARNING", cache=args.cache,
                  reason="online_encoder_mode")
    update_progress("online_data_loading" if args.online_encoder else "cache_validation",
                    feature_mode=feature_mode, cache=cache_path,
                    elapsed_s=round(time.perf_counter() - run_started, 3))

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[1/4] Loading online raw dataset …" if args.online_encoder
          else "[1/4] Loading embedding dataset …")
    data_started = time.perf_counter()
    log_event("training.dataset_load_started", cache=cache_path, feature_mode=feature_mode)
    online_data_info = None
    if args.online_encoder:
        from scripts.online_encoder import load_online_dataset, raw_collate
        full_ds, online_data_info = load_online_dataset(args.dataset, cfg, protocol)
        is_h5 = False
    else:
        if not Path(cache_path).exists():
            log_event("training.cache_missing", level="ERROR", cache=cache_path)
            print(f"[ERROR] Cache not found: {cache_path}")
            print(f"        Run: python scripts/precompute.py --dataset {args.dataset} --exp {args.exp}")
            sys.exit(1)
        from data.hdf5_embeddings import is_hdf5_path
        is_h5 = is_hdf5_path(cache_path)
        full_ds = _get_embedding_dataset(args.dataset, cache_path,
                                        expect_layers=cfg.vlm_extract_layers,
                                        protocol=protocol, config=cfg)
    log_event("training.dataset_load_complete", samples=len(full_ds),
              feature_mode=feature_mode,
              cache_type=("hdf5" if is_h5 else "memory") if not args.online_encoder else None,
              online_data=online_data_info,
              elapsed_s=round(time.perf_counter() - data_started, 3),
              cache_id=getattr(full_ds, "cache_id", None))
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
    if not len(val_ds):
        raise ValueError("Validation split must contain at least one sample")
    # In-memory caches (.pt): num_workers=0 is optimal (data already in RAM).
    # HDF5 caches: reads hit disk per sample, so use multiple workers (each reopens
    # its own h5 handle) + persistent_workers to overlap disk I/O with GPU compute —
    # otherwise the GPU starves waiting on single-threaded reads.
    on_cuda = (device.type == "cuda")
    if args.online_encoder:
        dl_kwargs = dict(num_workers=6, persistent_workers=True,
                         pin_memory=on_cuda, prefetch_factor=2)
    elif is_h5:
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
    if args.steps_per_epoch is not None and args.steps_per_epoch * cfg.batch_size > len(train_ds):
        raise ValueError("steps-per-epoch requests more unique samples than the training split contains")
    train_sampler = (RandomSampler(train_ds, generator=train_generator,
                                   num_samples=args.steps_per_epoch * cfg.batch_size
                                   if args.steps_per_epoch is not None else None)
                     if train_generator is not None else
                     RandomSampler(train_ds, num_samples=args.steps_per_epoch * cfg.batch_size)
                     if args.steps_per_epoch is not None else None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=train_generator is None and train_sampler is None,
                              sampler=train_sampler,
                              drop_last=True, generator=worker_generator,
                              collate_fn=raw_collate if args.online_encoder else None,
                              **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, generator=val_generator,
                              collate_fn=raw_collate if args.online_encoder else None,
                              **dl_kwargs)
    print(f"   Train: {len(train_ds):,}  Val: {len(val_ds):,}  Batch: {cfg.batch_size}"
          f"  | workers: {dl_kwargs['num_workers']}{' (HDF5)' if is_h5 else ''}")
    log_event("training.data_ready", train_samples=len(train_ds), validation_samples=len(val_ds),
              batch_size=cfg.batch_size, loader_workers=dl_kwargs["num_workers"],
              batches_per_epoch=len(train_loader), feature_mode=feature_mode,
              cache_type=("hdf5" if is_h5 else "memory") if not args.online_encoder else None)
    update_progress("model_initialization", train_samples=len(train_ds),
                    validation_samples=len(val_ds), batches_per_epoch=len(train_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building VLATrainModel (adapter + decoder) …")
    model_started = time.perf_counter()
    log_event("training.model_build_started")
    model   = VLATrainModel(cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total trainable params: {n_total:,}")
    log_event("training.model_build_complete", trainable_parameters=n_total,
              adapter_parameters=sum(p.numel() for p in model.adapter.parameters() if p.requires_grad),
              decoder_parameters=sum(p.numel() for p in model.decoder.parameters() if p.requires_grad),
              elapsed_s=round(time.perf_counter() - model_started, 3))

    encoder_model = None
    encoder_provenance = getattr(full_ds, "encoder_provenance", None)
    if args.online_encoder:
        # VLAModel also constructs unused randomly-initialized head modules.
        # Restore RNG state around it so the trainable adapter/decoder seed and
        # the flow-matching noise stream match the cached-feature path.
        saved_rng = rng_state()
        from scripts.online_encoder import load_frozen_encoder
        update_progress("online_backbone_loading", feature_mode=feature_mode)
        encoder_model = load_frozen_encoder(cfg, device)
        restore_rng(saved_rng)
        from experiment_protocol import encoder_code_id, file_digest, source_assets
        from scripts import online_encoder as online_encoder_module
        encoder_provenance = {"encoder_code_id": encoder_code_id(),
                              "source_assets": source_assets(cfg),
                              "online_encoder_code_sha256": file_digest(
                                  Path(online_encoder_module.__file__))}
        log_event("training.online_encoder_ready", feature_mode=feature_mode,
                  microbatch_size=cfg.embed_batch_size,
                  backbone=cfg.vlm_backbone, provenance=encoder_provenance)

    def model_batch(batch):
        if args.online_encoder:
            from scripts.online_encoder import encode_online_batch
            embed, img_mask = encode_online_batch(
                encoder_model, batch["images"], batch["task_texts"], device,
                cfg.embed_batch_size, profile=args.profile_online_encoder)
            state = batch["states"].to(device, non_blocking=on_cuda)
            actions = batch["actions"].to(device, non_blocking=on_cuda)
        else:
            embed = batch["embedding"].to(device, non_blocking=on_cuda)
            state = batch["state"].to(device, non_blocking=on_cuda)
            actions = batch["actions"].to(device, non_blocking=on_cuda)
            img_mask = batch.get("img_mask")
            if img_mask is not None:
                img_mask = img_mask.to(device, non_blocking=on_cuda)
        return embed, state, actions, img_mask

    # ── Optimiser ─────────────────────────────────────────────────────────
    optim = torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": cfg.learning_rate * 0.5},
        {"params": model.decoder.parameters(), "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)

    total_steps = (int(rollout_config["reference_optimizer_updates"])
                   if rollout_config is not None else cfg.num_epochs * len(train_loader))
    if args.max_steps is not None and total_steps != args.max_steps:
        raise ValueError(f"Training budget mismatch: {cfg.num_epochs} epochs x {len(train_loader)} batches "
                         f"= {total_steps} updates, requested {args.max_steps}")
    if args.steps_per_epoch is not None and len(train_loader) != args.steps_per_epoch:
        raise ValueError("steps-per-epoch did not produce the requested full-batch count")
    if rollout_config is not None:
        decay_steps = int(rollout_config["reference_optimizer_updates"])
        decay_epochs = decay_steps // len(train_loader)
    else:
        decay_epochs = min(cfg.num_epochs, cfg.lr_decay_epochs or cfg.num_epochs)
        decay_steps = args.lr_decay_steps or decay_epochs * len(train_loader)
    if rollout_config is None and decay_steps > total_steps:
        raise ValueError("Learning-rate decay endpoint cannot exceed the fixed training budget")
    if decay_steps <= cfg.warmup_steps + 1:
        raise ValueError("LR decay horizon needs more steps than warmup_steps + 1")
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optim, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=decay_steps, min_lr_rate=cfg.min_lr_ratio)
    log_event("training.optimizer_ready", optimizer="AdamW", total_updates=total_steps,
              batches_per_epoch=len(train_loader), batch_size=cfg.batch_size,
              warmup_updates=cfg.warmup_steps, decay_updates=decay_steps,
              log_every_steps=args.log_every_steps)

    early_stop_start_epoch = max(1, (cfg.warmup_steps + len(train_loader) - 1) // len(train_loader))
    maximum_epoch_text = "unbounded (success-rate patience)" if rollout_config is not None \
        else f"{cfg.num_epochs} epochs ({total_steps:,} steps)"
    print(f"\n[3/4] Warmup/cosine to floor: {decay_epochs} reference epochs ({decay_steps:,} steps); "
          f"training limit: {maximum_epoch_text}")
    print(f"   Peak LR: adapter={cfg.learning_rate*0.5:.1e}  decoder={cfg.learning_rate:.1e}\n")
    if rollout_config is not None:
        print(f"   Stopping: rollout success evaluations every "
              f"{rollout_stopping_config['evaluation_every_epochs']} epochs from "
              f"{rollout_stopping_config['minimum_epochs']}; patience "
              f"{rollout_stopping_config['patience_epochs']} elapsed epochs after strict improvement\n")
    else:
        print(f"   LR floor: {cfg.min_lr_ratio:.1%} of peak; patience "
              f"{cfg.early_stop_patience} is enabled after {cfg.warmup_steps} warmup updates "
              f"(first eligible validation: epoch {early_stop_start_epoch})\n")
        print(f"   Minimum training: {cfg.min_epochs} epochs\n")

    global_step   = 0
    best_val_loss = float("inf")
    no_improve    = 0
    start_epoch   = 1
    signature = run_signature(cfg, protocol, args.seed) if protocol else None
    cache_id = getattr(full_ds, "cache_id", None)
    metadata = {"cfg": cfg, "dataset": args.dataset, "exp_id": args.exp,
                "training_seed": args.seed, "protocol": protocol,
                "protocol_id": protocol["protocol_id"] if protocol else None,
                "run_signature": signature, "cache_id": cache_id,
                "feature_mode": feature_mode,
                "encoder_provenance": encoder_provenance,
                "lr_schedule": "warmup_cosine_min_lr", "lr_schedule_steps": decay_steps,
                "lr_decay_epochs": decay_epochs,
                "max_training_steps": None if rollout_config is not None else total_steps,
                "reference_optimizer_updates": total_steps if rollout_config is not None else None,
                "training_mode": ("success_stopped" if rollout_config is not None else
                                  "update_matched" if args.max_steps is not None else "epoch"),
                "steps_per_epoch": len(train_loader),
                "checkpoint_every_steps": args.checkpoint_every_steps,
                "probe_epochs": args.probe_epochs,
                "profile_online_encoder": args.profile_online_encoder,
                "cpu_resources": runtime_cpu,
                "early_stopping": ("development_success_rate" if rollout_config is not None
                                   else "after_warmup"),
                "early_stop_start_step": cfg.warmup_steps,
                "early_stop_start_epoch": early_stop_start_epoch}
    if rollout_config is not None:
        metadata.update(rollout_study_id=args.rollout_study_id,
                        rollout_plan_id=args.rollout_plan_id,
                        rollout_stopping_config=rollout_stopping_config,
                        rollout_evaluation=rollout_dataset_config["development_evaluation"])
    if protocol and args.fresh and (ckpt_dir / "last.pt").exists():
        raise FileExistsError("Use a new output directory instead of overwriting a protocol run")

    # ── Resume from a full-state checkpoint if present (crash resilience) ───────
    # GPU driver TDRs / power events can kill the process mid-run; last.pt is written
    # every epoch with model+optimizer+scheduler so a relaunch continues where it
    # stopped instead of losing all progress. Resume requires the SAME batch size
    # (the schedule length is fixed at construction).
    last_path = ckpt_dir / "last.pt"
    validation_metrics = {}
    next_recovery_step = args.checkpoint_every_steps
    rollout_state_path = None
    rollout_durable_run_dir = None
    durable_last_path = None
    if rollout_config is not None:
        rollout_state_path = (Path(args.rollout_durable_root) / "controller_state"
                              / args.rollout_plan_id[:12] / args.dataset / args.exp
                              / f"seed_{args.seed:03d}.json")
        output_dir = Path(cfg.output_dir).resolve()
        if args.rollout_durable_run_dir is not None:
            rollout_durable_run_dir = args.rollout_durable_run_dir.resolve()
        else:
            # Keep direct trainer invocations safe too: success-study outputs
            # have <study>/<dataset>/<experiment>/seed_NNN layout.
            run_root = output_dir.parents[2]
            if run_root.name != args.rollout_study_id:
                raise ValueError("Cannot derive durable run mirror; pass --rollout-durable-run-dir")
            rollout_durable_run_dir = (Path(args.rollout_durable_root).resolve()
                                       / "success_studies" / args.rollout_study_id
                                       / args.dataset / output_dir.relative_to(run_root))
        durable_base = (Path(args.rollout_durable_root).resolve() / "success_studies"
                        / args.rollout_study_id / args.dataset).resolve()
        try:
            rollout_durable_run_dir.relative_to(durable_base)
        except ValueError as exc:
            raise ValueError("Durable run directory must remain under this study/dataset mirror") from exc
        durable_last_path = rollout_durable_run_dir / "checkpoints" / "last.pt"

    recovery_revision = 0

    def durable_counterpart(local_path: Path) -> Path:
        local_path = Path(local_path).resolve()
        try:
            relative = local_path.relative_to(Path(cfg.output_dir).resolve())
        except ValueError as exc:
            raise ValueError(f"Run artifact is outside the configured output directory: {local_path}") from exc
        return rollout_durable_run_dir / relative

    def validate_recovery_identity(checkpoint: dict, source: Path) -> None:
        expected = {
            "dataset": args.dataset, "exp_id": args.exp,
            "training_seed": args.seed, "protocol_id": protocol["protocol_id"],
            "run_signature": signature, "feature_mode": feature_mode,
            "rollout_study_id": args.rollout_study_id,
            "rollout_plan_id": args.rollout_plan_id,
        }
        mismatched = [key for key, value in expected.items()
                      if checkpoint.get(key) != value]
        if mismatched:
            raise ValueError(f"Recovery checkpoint identity mismatch at {source}: {mismatched}")

    def restore_durable_recovery_if_newer() -> None:
        if rollout_controller is None or args.fresh or not durable_last_path.is_file():
            return
        from experiment_protocol import file_digest
        durable_ck = torch.load(durable_last_path, map_location="cpu", weights_only=False)
        validate_recovery_identity(durable_ck, durable_last_path)
        if not last_path.is_file():
            atomic_copy_checkpoint(durable_last_path, last_path, file_digest(durable_last_path))
            log_event("training.durable_last_restored", path=str(last_path),
                      durable_path=str(durable_last_path), epoch=durable_ck.get("epoch"),
                      optimizer_step=durable_ck.get("global_step"))
            return
        try:
            local_ck = torch.load(last_path, map_location="cpu", weights_only=False)
            validate_recovery_identity(local_ck, last_path)
        except Exception as exc:
            log_event("training.local_last_rejected", level="WARNING", path=str(last_path),
                      error=f"{type(exc).__name__}: {exc}")
            atomic_copy_checkpoint(durable_last_path, last_path, file_digest(durable_last_path))
            log_event("training.durable_last_restored", path=str(last_path),
                      durable_path=str(durable_last_path), epoch=durable_ck.get("epoch"),
                      optimizer_step=durable_ck.get("global_step"))
            return
        def order(checkpoint: dict) -> tuple[int, int, int]:
            return (int(checkpoint.get("epoch", -1)),
                    int(checkpoint.get("global_step", -1)),
                    int(checkpoint.get("recovery_revision", 0)))
        local_order, durable_order = order(local_ck), order(durable_ck)
        if durable_order > local_order:
            atomic_copy_checkpoint(durable_last_path, last_path, file_digest(durable_last_path))
            log_event("training.durable_last_restored", path=str(last_path),
                      durable_path=str(durable_last_path), epoch=durable_order[0],
                      optimizer_step=durable_order[1], recovery_revision=durable_order[2])
        elif durable_order == local_order and file_digest(last_path) != file_digest(durable_last_path):
            raise ValueError("Local and durable last.pt have conflicting contents at the same recovery revision")

    def save_recovery_checkpoint(epoch: int, val_loss: float | None) -> None:
        nonlocal recovery_revision
        recovery_revision += 1
        payload = {**metadata, "epoch": epoch, "state_dict": model.state_dict(),
                   "validation_metrics": validation_metrics,
                   "optim": optim.state_dict(), "scheduler": scheduler.state_dict(),
                   "best_val_loss": best_val_loss, "global_step": global_step,
                   "next_recovery_step": next_recovery_step,
                   "no_improve": no_improve, "batch_size": cfg.batch_size,
                   "recovery_revision": recovery_revision,
                   "val_loss": val_loss, "cfg": cfg,
                   "rng_state": rng_state(),
                   "train_generator": train_generator.get_state() if train_generator is not None else None,
                   "val_generator": val_generator.get_state() if val_generator is not None else None}
        if rollout_controller is not None:
            payload["success_stopping_state"] = rollout_controller.state_dict()
        tmp = ckpt_dir / "last.pt.tmp"
        torch.save(payload, tmp)
        tmp.replace(last_path)
        if rollout_controller is not None:
            for name in ("history.jsonl", "training_curve.png"):
                artifact = Path(cfg.output_dir) / name
                if artifact.is_file():
                    atomic_copy_checkpoint(artifact, durable_counterpart(artifact))
            durable_sha = atomic_copy_checkpoint(last_path, durable_last_path)
            log_event("training.durable_last_committed", path=str(durable_last_path),
                      sha256=durable_sha, epoch=epoch, optimizer_step=global_step,
                      recovery_revision=recovery_revision)
            write_json_atomic(rollout_state_path, {
                "schema": 1, "study_id": args.rollout_study_id,
                "plan_id": args.rollout_plan_id, "dataset": args.dataset,
                "experiment": args.exp, "training_seed": args.seed,
                "protocol_id": protocol["protocol_id"], "run_signature": signature,
                "last_committed_epoch": epoch,
                "last_committed_recovery_revision": recovery_revision,
                "success_stopping_state": rollout_controller.state_dict(),
                "updated_utc": datetime.now(timezone.utc).isoformat()})
        log_event("training.checkpoint_complete", kind="recovery", path=str(last_path),
                  bytes=last_path.stat().st_size, epoch=epoch, optimizer_step=global_step,
                  recovery_revision=recovery_revision,
                  rollout_pending=(rollout_controller.pending is not None
                                   if rollout_controller is not None else None))

    restore_durable_recovery_if_newer()
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        recovery_revision = int(ck.get("recovery_revision", 0))
        validation_metrics = ck.get("validation_metrics", {})
        if (ck.get("lr_schedule") != metadata["lr_schedule"]
                or ck.get("lr_schedule_steps") != decay_steps
                or ck.get("max_training_steps") != metadata["max_training_steps"]
                or ck["cfg"].min_lr_ratio != cfg.min_lr_ratio):
            raise ValueError("Resume LR schedule mismatch; use a new output directory")
        if (ck.get("training_mode", "epoch") != metadata["training_mode"]
                or ck.get("steps_per_epoch", len(train_loader)) != len(train_loader)
                or ck.get("checkpoint_every_steps", args.checkpoint_every_steps) != args.checkpoint_every_steps
                or ck.get("probe_epochs", args.probe_epochs) != args.probe_epochs):
            raise ValueError("Resume update/checkpoint plan mismatch; use a new output directory")
        if (ck.get("early_stopping") != metadata["early_stopping"]
                or ck.get("early_stop_start_step") != cfg.warmup_steps):
            raise ValueError("Resume stopping rule mismatch; use a new output directory")
        if rollout_controller is not None:
            if (ck.get("rollout_study_id") != args.rollout_study_id
                    or ck.get("rollout_plan_id") != args.rollout_plan_id
                    or not isinstance(ck.get("success_stopping_state"), dict)):
                raise ValueError("Resume rollout study/controller mismatch; use the original plan and checkpoint")
            rollout_controller = type(rollout_controller)(
                minimum_epochs=int(rollout_stopping_config["minimum_epochs"]),
                evaluation_every_epochs=int(rollout_stopping_config["evaluation_every_epochs"]),
                patience_epochs=int(rollout_stopping_config["patience_epochs"]),
                minimum_improvement=float(rollout_stopping_config.get("minimum_improvement", 0.0)),
                state=ck["success_stopping_state"])
            if rollout_state_path.is_file():
                cloud_state = json.loads(rollout_state_path.read_text(encoding="utf-8"))
                if (cloud_state.get("plan_id") != args.rollout_plan_id
                        or cloud_state.get("study_id") != args.rollout_study_id
                        or cloud_state.get("protocol_id") != protocol["protocol_id"]
                        or cloud_state.get("run_signature") != signature
                        or int(cloud_state.get("last_committed_epoch", -1)) > int(ck["epoch"])
                        or (int(cloud_state.get("last_committed_epoch", -1)) == int(ck["epoch"])
                            and int(cloud_state.get("last_committed_recovery_revision", 0))
                            > int(ck.get("recovery_revision", 0)))):
                    raise ValueError("Durable rollout controller state conflicts with last.pt")
            elif int(ck["epoch"]) > 0:
                log_event("training.rollout_state_mirror_missing", level="WARNING",
                          path=str(rollout_state_path), epoch=ck["epoch"])
        if ck.get("feature_mode", "cached") != feature_mode:
            raise ValueError("Resume feature mode mismatch; use the original mode or a new output directory")
        if (feature_mode == "online"
                and ck.get("encoder_provenance") != encoder_provenance):
            raise ValueError("Resume online encoder provenance mismatch; use a new output directory")
        if protocol and (ck.get("run_signature") != signature
                         or ck.get("cache_id") != cache_id):
            raise ValueError("Resume protocol/configuration/seed/cache mismatch")
        saved_bs = ck.get("batch_size")
        if saved_bs is not None and saved_bs != cfg.batch_size:
            log_event("training.resume_rejected", level="WARNING", saved_batch_size=saved_bs,
                      requested_batch_size=cfg.batch_size, reason="batch_size_schedule_mismatch")
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
            next_recovery_step = ck.get(
                "next_recovery_step",
                (global_step // args.checkpoint_every_steps + 1) * args.checkpoint_every_steps)
            no_improve    = ck.get("no_improve", 0)
            if args.seed is not None:
                if not all(k in ck for k in ("rng_state", "train_generator", "val_generator")):
                    raise ValueError("Seeded resume requires saved RNG and loader state")
                train_generator.set_state(ck["train_generator"].cpu())
                val_generator.set_state(ck["val_generator"].cpu())
                restore_rng(ck["rng_state"])
            log_event("training.resumed", epoch=start_epoch, optimizer_step=global_step,
                      checkpoint=str(last_path), best_validation_loss=best_val_loss)
            print(f"[resume] continuing from epoch {start_epoch} "
                  f"(best_val={best_val_loss:.4f}) — last.pt @ epoch {ck['epoch']}")
            if rollout_controller is not None:
                # last.pt is the recovery commit; re-mirror it before the queue
                # replays any pending idempotent rollout task.
                write_json_atomic(rollout_state_path, {
                    "schema": 1, "study_id": args.rollout_study_id,
                    "plan_id": args.rollout_plan_id, "dataset": args.dataset,
                    "experiment": args.exp, "training_seed": args.seed,
                    "protocol_id": protocol["protocol_id"], "run_signature": signature,
                    "last_committed_epoch": int(ck["epoch"]),
                    "last_committed_recovery_revision": recovery_revision,
                    "success_stopping_state": rollout_controller.state_dict(),
                    "updated_utc": datetime.now(timezone.utc).isoformat()})
    elif args.fresh and last_path.exists():
        print("[--fresh] ignoring existing last.pt; training from epoch 1.")
    elif rollout_controller is not None and rollout_state_path.is_file():
        raise FileNotFoundError(
            "Durable rollout controller exists but last.pt is missing; restore the optimizer recovery "
            "checkpoint before resuming this study")

    if protocol:
        manifest_path = Path(cfg.output_dir) / "run_manifest.json"
        _FAILURE_CONTEXT["manifest_path"] = manifest_path
        manifest_state = {
            "dataset": args.dataset, "exp_id": args.exp, "training_seed": args.seed,
            "protocol_id": protocol["protocol_id"], "run_signature": signature,
            "cache_id": cache_id, "feature_mode": feature_mode,
            "encoder_provenance": encoder_provenance, "config": asdict(cfg),
            "train_frames": len(train_ds), "validation_frames": len(val_ds),
            "trainable_parameters": n_total, "torch_version": torch.__version__,
            "profile_online_encoder": args.profile_online_encoder,
            "cpu_resources": runtime_cpu,
            "lr_schedule": metadata["lr_schedule"], "lr_schedule_steps": decay_steps,
            "lr_decay_epochs": decay_epochs,
            "max_training_steps": metadata["max_training_steps"],
            "reference_optimizer_updates": metadata["reference_optimizer_updates"],
            "training_mode": metadata["training_mode"], "steps_per_epoch": len(train_loader),
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "probe_epochs": args.probe_epochs,
            "early_stopping": metadata["early_stopping"],
            "early_stop_start_step": cfg.warmup_steps,
            "early_stop_start_epoch": early_stop_start_epoch,
            "rollout_study_id": args.rollout_study_id if rollout_config is not None else None,
            "rollout_plan_id": args.rollout_plan_id if rollout_config is not None else None,
            "rollout_stopping_config": rollout_stopping_config,
            "success_stopping_state": (rollout_controller.state_dict()
                                       if rollout_controller is not None else None),
            "status": "running", "optimizer_steps": global_step,
            "current_epoch": start_epoch - 1,
            "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json_atomic(manifest_path, manifest_state)

    def ensure_selection_snapshot(pending: dict) -> Path:
        """Materialize the immutable task checkpoint from committed last.pt."""
        from experiment_protocol import file_digest
        path = Path(pending["checkpoint_path"])
        known_sha = pending.get("checkpoint_sha256")
        if known_sha is not None:
            if not path.is_file() or file_digest(path) != known_sha:
                durable_path = durable_counterpart(path)
                atomic_copy_checkpoint(durable_path, path, known_sha)
            existing = torch.load(path, map_location="cpu", weights_only=False)
            validate_recovery_identity(existing, path)
            if (int(existing.get("epoch", -1)) != int(pending["epoch"])
                    or int(existing.get("optimizer_step", -1)) != int(pending["optimizer_step"])):
                raise ValueError("Pending immutable checkpoint epoch/step mismatch")
            return path
        if pending.get("task") is not None:
            raise ValueError("Published rollout is missing its immutable checkpoint SHA")
        recovery = torch.load(last_path, map_location="cpu", weights_only=False)
        persisted_pending = (recovery.get("success_stopping_state") or {}).get("pending")
        if (int(recovery.get("epoch", -1)) != int(pending["epoch"])
                or not isinstance(persisted_pending, dict)
                or int(persisted_pending.get("epoch", -1)) != int(pending["epoch"])
                or persisted_pending.get("checkpoint_tag") != pending["checkpoint_tag"]):
            raise ValueError("Pending rollout does not match the committed last.pt epoch")
        keys_to_omit = {"optim", "scheduler", "rng_state", "train_generator", "val_generator",
                        "best_val_loss", "no_improve", "next_recovery_step",
                        "success_stopping_state"}
        selection = {key: value for key, value in recovery.items() if key not in keys_to_omit}
        selection["optimizer_step"] = int(recovery["global_step"])
        selection["global_step"] = int(recovery["global_step"])
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=False)
            identities_match = all(existing.get(key) == selection.get(key) for key in (
                "epoch", "optimizer_step", "run_signature", "protocol_id", "feature_mode"))
            state_a = existing.get("state_dict") or {}
            state_b = selection.get("state_dict") or {}
            weights_match = (state_a.keys() == state_b.keys()
                             and all(torch.equal(state_a[key], state_b[key]) for key in state_a))
            if (not identities_match or not weights_match
                    or existing.get("validation_metrics") != selection.get("validation_metrics")):
                raise ValueError(f"Refusing to overwrite a conflicting immutable selection checkpoint: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".tmp")
            torch.save(selection, temp)
            temp.replace(path)
        checkpoint_sha = file_digest(path)
        if rollout_controller is not None:
            atomic_copy_checkpoint(path, durable_counterpart(path), checkpoint_sha)
        rollout_controller.state["pending"]["checkpoint_sha256"] = checkpoint_sha
        save_recovery_checkpoint(completed_epoch, validation_metrics.get("flow_matching_loss"))
        return path

    def restore_best_checkpoint() -> bool:
        """Recover the selected development snapshot and bind its SHA to controller state."""
        if rollout_controller is None:
            return False
        state = rollout_controller.state
        checkpoint_value = state.get("best_checkpoint_path")
        if checkpoint_value is None:
            return False
        local_path = Path(checkpoint_value)
        expected_sha = state.get("best_checkpoint_sha256")
        candidate_path = durable_counterpart(local_path)
        chosen_path = local_path if local_path.is_file() else candidate_path
        if not chosen_path.is_file():
            pending = rollout_controller.pending
            if pending is None or Path(pending["checkpoint_path"]).resolve() != local_path.resolve():
                raise FileNotFoundError(
                    f"Best development checkpoint is absent locally and durably: {local_path}")
            return False  # Rebuilt by ensure_selection_snapshot before the pending task replays.
        from experiment_protocol import file_digest
        actual_sha = file_digest(chosen_path)
        if expected_sha is not None and actual_sha != expected_sha:
            if chosen_path == local_path and candidate_path.is_file():
                durable_sha = file_digest(candidate_path)
                if durable_sha == expected_sha:
                    atomic_copy_checkpoint(candidate_path, local_path, expected_sha)
                    chosen_path, actual_sha = local_path, durable_sha
                else:
                    raise ValueError("Best checkpoint SHA differs from both local and durable artifacts")
            else:
                raise ValueError("Best checkpoint SHA differs from controller state")
        checkpoint = torch.load(chosen_path, map_location="cpu", weights_only=False)
        validate_recovery_identity(checkpoint, chosen_path)
        if int(checkpoint.get("epoch", -1)) != int(state["best_epoch"]):
            raise ValueError("Best checkpoint epoch differs from controller selection state")
        if chosen_path == candidate_path and not local_path.is_file():
            atomic_copy_checkpoint(candidate_path, local_path, actual_sha)
            chosen_path = local_path
            log_event("training.best_checkpoint_restored", path=str(local_path),
                      durable_path=str(candidate_path), epoch=state["best_epoch"],
                      sha256=actual_sha)
        changed = state.get("best_checkpoint_sha256") != actual_sha
        state["best_checkpoint_sha256"] = actual_sha
        return changed

    def process_pending_rollout(*, wait: bool = False) -> dict | None:
        if rollout_controller is None or rollout_controller.pending is None:
            return None
        from scripts.success_queue import publish_rollout_task, read_rollout_result, wait_for_rollout
        pending = rollout_controller.pending
        checkpoint_path = ensure_selection_snapshot(pending)
        task_metadata = {**metadata, "experiment": args.exp,
                         "backbone": cfg.vlm_backbone,
                         "optimizer_step": pending["optimizer_step"],
                         "checkpoint_tag": pending["checkpoint_tag"],
                         "selection_epoch": pending["epoch"]}
        task = publish_rollout_task(
            cfg, task_metadata, checkpoint_path,
            rollout_dataset_config["development_evaluation"],
            "development_selection", args.rollout_plan_id,
            args.rollout_study_id, args.rollout_durable_root)
        if pending.get("task") is None:
            rollout_controller.set_pending_task(task)
            save_recovery_checkpoint(completed_epoch, validation_metrics.get("flow_matching_loss"))
            log_event("training.rollout_submitted_async", task_id=task["task_id"],
                      evaluation_epoch=pending["epoch"], training_epoch=completed_epoch,
                      checkpoint_sha256=task["checkpoint"]["sha256"])
        timeout_s = float(rollout_stopping_config["evaluation_wait_timeout_seconds"])
        poll_s = float(rollout_stopping_config["evaluation_poll_seconds"])
        result = read_rollout_result(task, args.rollout_durable_root)
        if result is None:
            age_s = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(task["created_utc"])).total_seconds()
            if age_s >= timeout_s:
                raise TimeoutError(f"Rollout {task['task_id']} exceeded {timeout_s}s; task remains recoverable")
            if not wait:
                log_event("training.rollout_pending_async", task_id=task["task_id"],
                          evaluation_epoch=pending["epoch"], training_epoch=completed_epoch,
                          wait_seconds=round(age_s, 1))
                return None
            log_event("training.rollout_wait_started", task_id=task["task_id"],
                      evaluation_epoch=pending["epoch"], training_epoch=completed_epoch,
                      checkpoint_sha256=task["checkpoint"]["sha256"],
                      timeout_s=timeout_s - age_s, poll_s=poll_s)
            update_progress("development_rollout_wait", epoch=completed_epoch,
                            optimizer_step=global_step, evaluation_epoch=pending["epoch"],
                            task_id=task["task_id"])
            result = wait_for_rollout(task, args.rollout_durable_root, timeout_s - age_s, poll_s)
        transition = rollout_controller.observe_result(result)
        log_event("training.rollout_evaluation_complete", task_id=task["task_id"], **transition)
        with (Path(cfg.output_dir) / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": transition["epoch"],
                                      "optimizer_step": pending["optimizer_step"],
                                      "development_success_rate": transition["success_rate"],
                                      "development_sr_improved": transition["improved"],
                                      "development_best_success_rate": transition["best_success_rate"],
                                      "development_best_epoch": transition["best_epoch"],
                                      "rollout_task_id": task["task_id"]}) + "\n")
        save_recovery_checkpoint(completed_epoch, validation_metrics.get("flow_matching_loss"))
        if protocol:
            manifest_state.update(current_epoch=completed_epoch,
                                  optimizer_steps=global_step,
                                  success_stopping_state=rollout_controller.state_dict(),
                                  last_development_success_rate=transition["success_rate"],
                                  best_development_success_rate=transition["best_success_rate"],
                                  best_development_epoch=transition["best_epoch"],
                                  last_development_evaluation_task=task["task_id"],
                                  updated_utc=datetime.now(timezone.utc).isoformat())
            write_json_atomic(manifest_path, manifest_state)
        return transition

    def finalize_rollout_selection() -> dict:
        if rollout_controller is None or rollout_controller.state["best_checkpoint_path"] is None:
            raise ValueError("Cannot stop success-stopped training without a completed baseline rollout")
        if rollout_controller.pending is not None:
            raise ValueError("Cannot finalize selection with an unresolved development rollout")
        from experiment_protocol import file_digest
        source = Path(rollout_controller.state["best_checkpoint_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Best development checkpoint is missing: {source}")
        selected_path = (ckpt_dir / "selected.pt").resolve()
        source = source.resolve()
        source_sha = file_digest(source)
        expected_sha = rollout_controller.state.get("best_checkpoint_sha256")
        if expected_sha is not None and source_sha != expected_sha:
            raise ValueError("Best development checkpoint SHA differs from controller state")
        if selected_path.exists():
            if file_digest(selected_path) != source_sha:
                raise ValueError("Existing selected.pt differs from the controller's best development checkpoint")
        else:
            temp = selected_path.with_name(selected_path.name + ".tmp")
            shutil.copyfile(source, temp)
            if file_digest(temp) != source_sha:
                raise IOError("selected.pt copy failed SHA-256 verification")
            temp.replace(selected_path)
        summary = {
            "schema": 1, "study_id": args.rollout_study_id,
            "plan_id": args.rollout_plan_id, "dataset": args.dataset,
            "experiment": args.exp, "training_seed": args.seed,
            "protocol_id": protocol["protocol_id"], "run_signature": signature,
            "selection_metric": "development_success_rate",
            "selection_rule": "highest_success_rate; earliest_epoch_on_tie",
            "best_development_success_rate": rollout_controller.state["best_success_rate"],
            "selection_epoch": rollout_controller.state["best_epoch"],
            "best_checkpoint_path": str(source), "best_checkpoint_sha256": source_sha,
            "selected_checkpoint_path": str(selected_path),
            "selected_checkpoint_sha256": file_digest(selected_path),
            "stopping_epoch": completed_epoch,
            "evaluation_execution": "asynchronous_one_pending_with_cadence_backpressure",
            "last_improvement_epoch": rollout_controller.state["last_improvement_epoch"],
            "patience_epochs": rollout_controller.patience_epochs,
            "last_development_evaluation_epoch": rollout_controller.state["last_evaluated_epoch"],
        }
        write_json_atomic(Path(cfg.output_dir) / "selection_summary.json", summary)
        write_json_atomic(rollout_state_path.with_name("selection_summary.json"), summary)
        atomic_copy_checkpoint(selected_path, durable_counterpart(selected_path),
                               summary["selected_checkpoint_sha256"])
        atomic_copy_checkpoint(Path(cfg.output_dir) / "selection_summary.json",
                               durable_counterpart(Path(cfg.output_dir) / "selection_summary.json"))
        log_event("training.rollout_selection_finalized", **summary)
        return summary

    completed_epoch = start_epoch - 1
    if restore_best_checkpoint():
        save_recovery_checkpoint(completed_epoch, validation_metrics.get("flow_matching_loss"))
    stop_epoch = (None if rollout_config is not None else
                  cfg.num_epochs if completed_epoch < cfg.min_epochs
                  or no_improve < cfg.early_stop_patience else completed_epoch)
    epoch_iterator = (itertools.count(start_epoch) if rollout_config is not None
                      else range(start_epoch, stop_epoch + 1))
    epoch_limit = None if rollout_config is not None else cfg.num_epochs
    log_event("training.rollout_controller_ready" if rollout_controller is not None
              else "training.epoch_loop_ready",
              controller_state=(rollout_controller.state_dict()
                                if rollout_controller is not None else None),
              start_epoch=start_epoch, epoch_limit=epoch_limit)
    if rollout_controller is not None and rollout_controller.pending is not None:
        log_event("training.rollout_pending_recovery", level="INFO",
                  pending=rollout_controller.pending)
    rollout_stop_requested = False
    if rollout_controller is not None and rollout_controller.pending is not None:
        recovered_transition = process_pending_rollout(
            wait=rollout_controller.wait_required(completed_epoch))
        rollout_stop_requested = bool(recovered_transition and recovered_transition["stop_due"])
    if (rollout_controller is not None and not rollout_stop_requested
            and rollout_controller.evaluation_due(completed_epoch)):
        # Replay a crash after committing the previous result at a due boundary.
        rollout_controller.begin_evaluation(
            epoch=completed_epoch,
            checkpoint_path=str(ckpt_dir / f"selection_epoch_{completed_epoch:06d}.pt"),
            optimizer_step=global_step)
        save_recovery_checkpoint(completed_epoch, validation_metrics.get("flow_matching_loss"))
        recovered_transition = process_pending_rollout(
            wait=rollout_controller.wait_required(completed_epoch))
        rollout_stop_requested = bool(recovered_transition and recovered_transition["stop_due"])
    if (rollout_controller is not None and not rollout_stop_requested
            and rollout_controller.should_stop(completed_epoch)):
        rollout_stop_requested = True
    if rollout_stop_requested:
        log_event("training.early_stop", epoch=completed_epoch,
                  reason="development_success_rate_patience",
                  success_stopping_state=rollout_controller.state_dict())
        finalize_rollout_selection()
    for epoch in (() if rollout_stop_requested else epoch_iterator):
        log_event("training.epoch_started", epoch=epoch, total_epochs=epoch_limit,
                  optimizer_step=global_step, total_updates=total_steps)
        update_progress("training", epoch=epoch, total_epochs=epoch_limit,
                        optimizer_step=global_step, total_optimizer_steps=total_steps)
        model.train()
        ep_loss = 0.0
        grad_norm_sum = 0.0
        t0 = time.time()
        batch_iter = iter(train_loader)
        window_started = time.perf_counter()
        window_steps = 0
        window_loss = 0.0
        window_grad_norm = 0.0
        window_data_wait = 0.0
        window_step_time = 0.0

        for batch_index in range(1, len(train_loader) + 1):
            load_started = time.perf_counter()
            batch = next(batch_iter)
            data_wait_s = time.perf_counter() - load_started
            step_started = time.perf_counter()
            embed, state, actions, img_mask = model_batch(batch)

            optim.zero_grad(set_to_none=True)   # frees grad memory rather than zeroing
            loss = model(embed, state, actions, img_mask)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            grad_norm_sum += grad_norm
            optim.step()
            # Hugging Face cosine would rebound beyond its configured endpoint.
            if global_step < decay_steps:
                scheduler.step()
            ep_loss     += loss.item() * len(state)
            global_step += 1
            window_steps += 1
            window_loss += float(loss.detach())
            window_grad_norm += grad_norm
            window_data_wait += data_wait_s
            window_step_time += time.perf_counter() - step_started

            if global_step == 1 or global_step % args.log_every_steps == 0 or batch_index == len(train_loader):
                window_elapsed = time.perf_counter() - window_started
                updates_per_s = window_steps / max(window_elapsed, 1e-9)
                log_event("training.step_progress", epoch=epoch, total_epochs=epoch_limit,
                          feature_mode=feature_mode,
                          epoch_batch=batch_index, batches_per_epoch=len(train_loader),
                          optimizer_step=global_step, total_optimizer_steps=total_steps,
                          window_updates=window_steps,
                          mean_loss=round(window_loss / max(window_steps, 1), 6),
                          mean_grad_norm=round(window_grad_norm / max(window_steps, 1), 6),
                          learning_rate=scheduler.get_last_lr()[1],
                          mean_data_wait_s=round(window_data_wait / max(window_steps, 1), 4),
                          mean_step_s=round(window_step_time / max(window_steps, 1), 4),
                          updates_per_s=round(updates_per_s, 3),
                          elapsed_s=round(time.perf_counter() - run_started, 2),
                          eta_s=(None if rollout_config is not None else
                                 round((total_steps - global_step) / max(updates_per_s, 1e-9), 1)))
                update_progress("training", epoch=epoch, total_epochs=epoch_limit,
                                feature_mode=feature_mode,
                                optimizer_step=global_step, total_optimizer_steps=total_steps,
                                latest_loss=round(window_loss / max(window_steps, 1), 6),
                                updates_per_s=round(updates_per_s, 3),
                                eta_s=(None if rollout_config is not None else
                                       round((total_steps - global_step) / max(updates_per_s, 1e-9), 1)))
                if protocol:
                    manifest_state.update(current_epoch=epoch, optimizer_steps=global_step,
                                          latest_train_loss=round(window_loss / max(window_steps, 1), 6),
                                          updates_per_s=round(updates_per_s, 3),
                                          updated_utc=datetime.now(timezone.utc).isoformat())
                    write_json_atomic(manifest_path, manifest_state)
                window_started = time.perf_counter()
                window_steps = 0
                window_loss = 0.0
                window_grad_norm = 0.0
                window_data_wait = 0.0
                window_step_time = 0.0

        model.eval()
        val_loss = 0.0
        validation_started = time.perf_counter()
        log_event("training.validation_started", epoch=epoch, optimizer_step=global_step,
                  batches=len(val_loader), samples=len(val_ds), feature_mode=feature_mode)
        update_progress("validation", epoch=epoch, optimizer_step=global_step,
                        validation_samples=len(val_ds))
        validation_samples_done = 0
        collect_action_metrics = (epoch in args.probe_epochs or epoch == stop_epoch
                                  or (rollout_controller is not None
                                      and rollout_controller.is_evaluation_epoch(epoch)))
        action_squared_error = torch.zeros(cfg.action_dim, dtype=torch.float64)
        action_absolute_error = torch.zeros(cfg.action_dim, dtype=torch.float64)
        with (fixed_rng(4242) if args.seed is not None else nullcontext()), torch.no_grad():
            for validation_batch, batch in enumerate(val_loader, 1):
                embed, state, actions, img_mask = model_batch(batch)
                val_loss += model(embed, state, actions, img_mask).item() * len(state)
                if collect_action_metrics:
                    # Restore the flow-loss RNG stream after sampling, so adding
                    # diagnostics cannot change later loss draws or training.
                    with fixed_rng(424200 + validation_batch):
                        predicted = model.sample(embed, state, num_steps=cfg.num_flow_steps,
                                                 img_mask=img_mask).reshape_as(actions)
                    error = predicted.float() - actions.float()
                    action_squared_error += error.square().sum(dim=(0, 1)).double().cpu()
                    action_absolute_error += error.abs().sum(dim=(0, 1)).double().cpu()
                validation_samples_done += len(state)
                if validation_batch % 10 == 0 or validation_batch == len(val_loader):
                    log_event("training.validation_progress", epoch=epoch,
                              completed_batches=validation_batch, total_batches=len(val_loader),
                              completed_samples=validation_samples_done, total_samples=len(val_ds),
                              elapsed_s=round(time.perf_counter() - validation_started, 1))
                    update_progress("validation", epoch=epoch, optimizer_step=global_step,
                                    validation_batches_completed=validation_batch,
                                    validation_batches_total=len(val_loader))
        log_event("training.validation_complete", epoch=epoch,
                  feature_mode=feature_mode,
                  validation_loss=round(val_loss / len(val_ds), 6),
                  elapsed_s=round(time.perf_counter() - validation_started, 3))

        avg_train = ep_loss / (len(train_loader) * cfg.batch_size)
        avg_grad_norm = grad_norm_sum / len(train_loader)
        avg_val   = val_loss / len(val_ds)
        validation_metrics = {
            "flow_matching_loss": avg_val, "n_samples": len(val_ds),
            "protocol_id": protocol["protocol_id"] if protocol else None,
            "split": "validation_demonstrations", "validation_rng_seed": 4242,
            "action_metrics_measured": collect_action_metrics,
            "optimizer_step": global_step,
        }
        if collect_action_metrics:
            count = len(val_ds) * cfg.action_horizon
            validation_metrics.update(
                action_mse=float(action_squared_error.mean() / count),
                action_mae=float(action_absolute_error.mean() / count),
                per_dimension_action_mse=(action_squared_error / count).tolist(),
                per_dimension_action_mae=(action_absolute_error / count).tolist(),
                action_metric_space="configured_model_action_space",
                action_sampling_seed_base=424200, action_flow_steps=cfg.num_flow_steps)
            log_event("training.checkpoint_validation_metrics", epoch=epoch,
                      **validation_metrics)
        if not (torch.isfinite(torch.tensor(avg_train)) and torch.isfinite(torch.tensor(avg_val))):
            raise ValueError("Nonfinite training/validation loss; cannot continue")
        completed_epoch = epoch
        elapsed   = time.time() - t0
        cur_lr    = scheduler.get_last_lr()[1]
        improved  = avg_val < best_val_loss
        counting_patience = (rollout_controller is None
                             and global_step >= cfg.warmup_steps)
        phase = ("warmup" if global_step < cfg.warmup_steps else
                 "lr_floor" if global_step >= decay_steps else "cosine_decay")
        no_improve = no_improve + 1 if counting_patience and not improved else 0
        tag = "  ★ best val loss" if improved and rollout_controller is None else ""
        patience = ("rollout-success" if rollout_controller is not None else
                    f"{no_improve}/{cfg.early_stop_patience}" if counting_patience else "deferred")
        print(f"Epoch {epoch:3d}/{epoch_limit if epoch_limit is not None else '∞'} ({phase})  |  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  |  "
              f"lr={cur_lr:.2e}  patience={patience}  |  {elapsed:.1f}s{tag}")
        log_event("training.epoch_complete", epoch=epoch, total_epochs=epoch_limit,
                  optimizer_step=global_step, train_loss=round(avg_train, 6),
                  validation_loss=round(avg_val, 6), gradient_norm=round(avg_grad_norm, 6),
                  learning_rate=cur_lr, phase=phase,
                  no_improve=(no_improve if rollout_controller is None else None),
                  elapsed_s=round(elapsed, 3))

        if improved:
            best_val_loss = avg_val
            if rollout_controller is None:
                checkpoint_started = time.perf_counter()
                log_event("training.checkpoint_started", kind="best", epoch=epoch,
                          optimizer_step=global_step)
                torch.save({**metadata, "epoch": epoch, "global_step": global_step,
                            "optimizer_step": global_step, "state_dict": model.state_dict(),
                            "validation_metrics": validation_metrics,
                            "val_loss": avg_val},
                           ckpt_dir / "best.pt")
                log_event("training.checkpoint_complete", kind="best", path=str(ckpt_dir / "best.pt"),
                          bytes=(ckpt_dir / "best.pt").stat().st_size,
                          elapsed_s=round(time.perf_counter() - checkpoint_started, 3))
        # Keep the previous interval recovery point while last.pt remains a
        # finer-grained epoch-boundary recovery point.
        if global_step >= next_recovery_step and last_path.exists():
            previous_tmp = ckpt_dir / "previous.pt.tmp"
            shutil.copyfile(last_path, previous_tmp)
            previous_tmp.replace(ckpt_dir / "previous.pt")
            while next_recovery_step <= global_step:
                next_recovery_step += args.checkpoint_every_steps
        with (Path(cfg.output_dir) / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": epoch, "global_step": global_step,
                                      "train_loss": avg_train, "validation_loss": avg_val,
                                      "gradient_norm": avg_grad_norm,
                                      "learning_rate": cur_lr, "no_improve": no_improve,
                                      "phase": phase,
                                      "validation_metrics": validation_metrics,
                                      "seconds": elapsed}) + "\n")
        with fixed_rng(4242):
            from scripts.plot_study_losses import plot_run_losses
            curve_path = plot_run_losses(Path(cfg.output_dir))
        log_event("training.loss_curve_saved", epoch=epoch, path=str(curve_path))
        if (rollout_controller is not None and rollout_controller.pending is not None
                and rollout_controller.is_evaluation_epoch(epoch)):
            transition = process_pending_rollout(wait=rollout_controller.wait_required(epoch))
            if transition and transition["stop_due"]:
                rollout_stop_requested = True
                finalize_rollout_selection()
                break
        if rollout_controller is not None and rollout_controller.evaluation_due(epoch):
            selection_path = ckpt_dir / f"selection_epoch_{epoch:06d}.pt"
            rollout_controller.begin_evaluation(epoch=epoch,
                                                checkpoint_path=str(selection_path),
                                                optimizer_step=global_step)
        save_recovery_checkpoint(epoch, avg_val)

        if rollout_controller is None and epoch >= cfg.min_epochs \
                and no_improve >= cfg.early_stop_patience:
            print(f"\n[Early stop] No improvement for {cfg.early_stop_patience} epochs.")
            log_event("training.early_stop", epoch=epoch, optimizer_step=global_step,
                      patience=no_improve)
            break

        if rollout_controller is not None and rollout_controller.pending is not None:
            transition = process_pending_rollout(wait=rollout_controller.wait_required(epoch))
            if transition and transition["stop_due"]:
                rollout_stop_requested = True
                log_event("training.early_stop", epoch=epoch, optimizer_step=global_step,
                          reason="development_success_rate_patience",
                          best_success_rate=transition["best_success_rate"],
                          best_epoch=transition["best_epoch"],
                          last_improvement_epoch=transition["last_improvement_epoch"])
                finalize_rollout_selection()
                break

        if epoch in args.probe_epochs:
            torch.save({**metadata, "epoch": epoch, "global_step": global_step,
                        "optimizer_step": global_step, "state_dict": model.state_dict(),
                        "validation_metrics": validation_metrics,
                        "val_loss": avg_val},
                       ckpt_dir / f"probe_step_{global_step:08d}.pt")
        elif epoch % cfg.save_every == 0:
            torch.save({**metadata, "epoch": epoch, "global_step": global_step,
                        "optimizer_step": global_step, "state_dict": model.state_dict(),
                        "validation_metrics": validation_metrics,
                        "val_loss": avg_val}, ckpt_dir / f"epoch_{epoch:04d}.pt")

    final_checkpoint_started = time.perf_counter()
    update_progress("final_checkpoint", epoch=completed_epoch, optimizer_step=global_step,
                    total_optimizer_steps=total_steps)
    log_event("training.final_checkpoint_started", epoch=completed_epoch,
              optimizer_step=global_step)
    torch.save({**metadata, "epoch": completed_epoch, "global_step": global_step,
               "optimizer_step": global_step, "state_dict": model.state_dict(),
               "validation_metrics": validation_metrics,
               "val_loss": validation_metrics.get("flow_matching_loss")},
               ckpt_dir / "final.pt")
    log_event("training.final_checkpoint_complete", path=str(ckpt_dir / "final.pt"),
              bytes=(ckpt_dir / "final.pt").stat().st_size,
              elapsed_s=round(time.perf_counter() - final_checkpoint_started, 3))
    if protocol:
        manifest_state.update(status="complete", completed_epoch=completed_epoch,
                              stop_reason=("development_success_rate_patience" if rollout_stop_requested else
                                           "patience" if no_improve >= cfg.early_stop_patience else "max_epochs"),
                              optimizer_steps=global_step, best_validation_loss=best_val_loss,
                              training_seed=args.seed, training_mode=metadata["training_mode"],
                              steps_per_epoch=len(train_loader),
                              updated_utc=datetime.now(timezone.utc).isoformat())
        if rollout_controller is not None:
            manifest_state.update(success_stopping_state=rollout_controller.state_dict(),
                                  selection_summary_path=str(Path(cfg.output_dir) / "selection_summary.json"))
        write_json_atomic(manifest_path, manifest_state)

    progress.update(status="complete", stage="complete", epoch=completed_epoch,
                    optimizer_step=global_step,
                    total_optimizer_steps=(None if rollout_config is not None else total_steps),
                    best_validation_loss=best_val_loss,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_s=round(time.perf_counter() - run_started, 3))
    write_json_atomic(progress_path, progress)
    log_event("training.complete", epoch=completed_epoch, optimizer_step=global_step,
              total_optimizer_steps=(None if rollout_config is not None else total_steps),
              reference_optimizer_updates=(total_steps if rollout_config is not None else None),
              best_validation_loss=best_val_loss,
              elapsed_s=round(time.perf_counter() - run_started, 3),
              output_dir=cfg.output_dir)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints → {ckpt_dir}")
    if rollout_controller is not None:
        print(f"Development selection → {ckpt_dir / 'selected.pt'}")
        print(f"Selection summary → {Path(cfg.output_dir) / 'selection_summary.json'}")
    else:
        print(f"\nNext step:")
        print(f"  python scripts/evaluate.py --dataset {args.dataset} --exp {args.exp} --episodes 50")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        failed_utc = datetime.now(timezone.utc).isoformat()
        error = f"{type(exc).__name__}: {exc}"
        progress_path = _FAILURE_CONTEXT.get("progress_path")
        if progress_path is not None:
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
                failed_stage = progress.get("stage", "startup")
                progress.update(status="failed", failed_stage=failed_stage, stage="failed",
                                error_type=type(exc).__name__, error=str(exc), failed_utc=failed_utc)
                write_json_atomic(progress_path, progress)
            except (OSError, ValueError):
                pass
        manifest_path = _FAILURE_CONTEXT.get("manifest_path")
        if manifest_path is not None:
            try:
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest.update(status="failed", error_type=type(exc).__name__,
                                    error=str(exc), updated_utc=failed_utc, failed_utc=failed_utc)
                    write_json_atomic(manifest_path, manifest)
            except (OSError, ValueError):
                pass
        log_event("training.failed", level="ERROR", error=error,
                  dataset=_FAILURE_CONTEXT.get("dataset"),
                  experiment=_FAILURE_CONTEXT.get("experiment"),
                  progress_path=str(progress_path) if progress_path else None)
        raise
