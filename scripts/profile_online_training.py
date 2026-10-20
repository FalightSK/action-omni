"""Profile a bounded number of real online-encoder training updates.

This is a performance diagnostic, not a study run: it uses the locked study
configuration and seeded training split, performs at most two optimizer updates,
does no validation, and writes no model or scientific-result checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from transformers import get_cosine_with_min_lr_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from experiment_protocol import (apply_normalization, digest, load_protocol,
                                 restore_rng, rng_state, seed_everything, seed_worker)
from models.vla_train import VLATrainModel
from scripts.online_encoder import load_frozen_encoder, load_online_dataset, raw_collate
from scripts.run_log import log_event


def cpu_quota_cores() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return int(quota) / int(period)
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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_info(value) -> dict | None:
    if not torch.is_tensor(value):
        return None
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device)}


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                          encoding="utf-8")
    os.replace(temporary, path)


def profile(args) -> dict:
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != 2 or plan.get("feature_mode") != "online":
        raise ValueError("Expected the locked schema-2 online-encoder study plan")
    if args.backbone not in plan["backbones"]:
        raise ValueError(f"Backbone is not in the locked plan: {args.backbone}")
    locked_exp = plan["backbones"][args.backbone]
    experiment = args.exp or locked_exp
    if experiment != locked_exp:
        raise ValueError(f"{args.backbone} is locked to {locked_exp}, not {experiment}")
    if args.dataset not in plan["datasets"]:
        raise ValueError(f"Dataset is not in the locked plan: {args.dataset}")
    if args.seed not in plan["training_seeds"]:
        raise ValueError("Use a training seed listed in the locked plan")
    if args.repeats < 1 or args.repeats > 2:
        raise ValueError("--repeats must be 1 or 2; profiling is capped at two optimizer updates")
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative")

    dataset_plan = plan["datasets"][args.dataset]
    protocol_path = (ROOT / dataset_plan["split_manifest"]).resolve()
    protocol = load_protocol(protocol_path, args.dataset)
    cfg = get_config(args.dataset, experiment)
    if cfg.vlm_backbone != args.backbone:
        raise ValueError(f"Resolved config backbone is {cfg.vlm_backbone}, expected {args.backbone}")
    if getattr(cfg, "n_views", 1) != len(dataset_plan.get("camera_views", ["top"])):
        raise ValueError("Resolved camera count differs from the locked study plan")
    apply_normalization(cfg, protocol)
    cfg.batch_size = int(plan["effective_batch_size"])
    cfg.num_epochs = int(dataset_plan["reference_epochs"])
    cfg.min_epochs = int(plan["stopping"]["minimum_epochs"])
    cfg.warmup_steps = int(plan["warmup_updates"])
    cfg.min_lr_ratio = float(plan["minimum_learning_rate_ratio"])
    cfg.lr_decay_epochs = None
    if args.microbatch_size is not None:
        if args.microbatch_size < 1:
            raise ValueError("--microbatch-size must be positive")
        cfg.embed_batch_size = args.microbatch_size

    quota = cpu_quota_cores()
    cpu_budget = max(1, math.floor(quota)) if quota is not None else None
    thread_count = args.cpu_threads or (max(1, min(2, cpu_budget - 2))
                                        if cpu_budget is not None else 4)
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("Online-training profiling requires CUDA")
    device = cfg.get_device()
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA training device, resolved {device}")

    seed_everything(args.seed)
    data_started = time.perf_counter()
    full_ds, online_data_info = load_online_dataset(args.dataset, cfg, protocol)
    train_indices = np.flatnonzero(~full_ds.validation_mask).tolist()
    train_ds = Subset(full_ds, train_indices)
    if len(train_ds) < cfg.batch_size * args.repeats:
        raise ValueError("Training split is too small for the requested full seeded batches")
    sampler_generator = torch.Generator().manual_seed(args.seed)
    worker_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        sampler=RandomSampler(train_ds, generator=sampler_generator),
        drop_last=True, generator=worker_generator,
        worker_init_fn=seed_worker, collate_fn=raw_collate,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=bool(args.workers),
        prefetch_factor=2 if args.workers else None)
    data_ready_s = time.perf_counter() - data_started

    model_started = time.perf_counter()
    model = VLATrainModel(cfg).to(device)
    model_build_s = time.perf_counter() - model_started
    preserved_rng = rng_state()
    encoder_started = time.perf_counter()
    encoder = load_frozen_encoder(cfg, device)
    encoder_load_s = time.perf_counter() - encoder_started
    restore_rng(preserved_rng)
    model.train()
    encoder.eval()
    for parameter in encoder.vlm.parameters():
        if parameter.requires_grad:
            raise AssertionError("Frozen VLM unexpectedly has trainable parameters")

    optimizer = torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": cfg.learning_rate * 0.5},
        {"params": model.decoder.parameters(), "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=int(plan["reference_optimizer_updates"]),
        min_lr_rate=cfg.min_lr_ratio)

    microbatch_events: list[dict] = []
    vlm_input_shapes: list[dict] = []
    active_update = 0
    import scripts.online_encoder as online_module
    original_log_event = online_module.log_event
    original_build = encoder.build_vlm_inputs
    original_encode = encoder.encode_vlm

    def capture_log_event(event: str, *, level: str = "INFO", **fields):
        if event == "online_encoder.microbatch_timing":
            microbatch_events.append({"update": active_update, **fields})
        return original_log_event(event, level=level, **fields)

    def capture_build_inputs(images, task_texts, input_device):
        inputs = original_build(images, task_texts, input_device)
        vlm_input_shapes.append({
            "update": active_update,
            "microbatch_samples": len(images),
            "inputs": {key: info for key, value in inputs.items()
                       if (info := tensor_info(value)) is not None},
        })
        return inputs

    def capture_encode_vlm(inputs):
        tokens, image_mask = original_encode(inputs)
        if vlm_input_shapes:
            vlm_input_shapes[-1]["output_tokens"] = tensor_info(tokens)
            vlm_input_shapes[-1]["output_image_mask"] = tensor_info(image_mask)
        return tokens, image_mask

    online_module.log_event = capture_log_event
    encoder.build_vlm_inputs = capture_build_inputs
    encoder.encode_vlm = capture_encode_vlm

    output_updates = []
    iterator = iter(loader)
    try:
        for update_index in range(1, args.repeats + 1):
            active_update = update_index
            torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            wait_started = time.perf_counter()
            batch = next(iterator)
            sync(device)
            data_wait_s = time.perf_counter() - wait_started

            sync(device)
            encoding_started = time.perf_counter()
            embed, image_mask = online_module.encode_online_batch(
                encoder, batch["images"], batch["task_texts"], device,
                int(cfg.embed_batch_size), profile=True)
            sync(device)
            encoding_s = time.perf_counter() - encoding_started

            sync(device)
            targets_started = time.perf_counter()
            state = batch["states"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            sync(device)
            target_transfer_s = time.perf_counter() - targets_started

            optimizer.zero_grad(set_to_none=True)
            sync(device)
            forward_started = time.perf_counter()
            loss = model(embed, state, actions, image_mask)
            sync(device)
            forward_s = time.perf_counter() - forward_started
            loss_value = float(loss.detach().cpu())

            sync(device)
            backward_started = time.perf_counter()
            loss.backward()
            sync(device)
            backward_s = time.perf_counter() - backward_started

            sync(device)
            clip_started = time.perf_counter()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            sync(device)
            grad_clip_s = time.perf_counter() - clip_started

            sync(device)
            optimizer_started = time.perf_counter()
            optimizer.step()
            scheduler.step()
            sync(device)
            optimizer_s = time.perf_counter() - optimizer_started

            output_updates.append({
                "update": update_index,
                "sample_row_ids": batch["idxs"].tolist(),
                "task_texts": batch["task_texts"],
                "batch_size": len(batch["images"]),
                "raw_state_shape": list(batch["states"].shape),
                "raw_action_shape": list(batch["actions"].shape),
                "encoded_shape": list(embed.shape),
                "encoded_dtype": str(embed.dtype),
                "image_mask_shape": list(image_mask.shape),
                "data_wait_s": round(data_wait_s, 4),
                "online_encoder_total_s": round(encoding_s, 4),
                "target_transfer_s": round(target_transfer_s, 4),
                "adapter_decoder_forward_loss_s": round(forward_s, 4),
                "loss_backward_s": round(backward_s, 4),
                "gradient_clip_s": round(grad_clip_s, 4),
                "optimizer_and_scheduler_s": round(optimizer_s, 4),
                "optimizer_step_wall_s": round(data_wait_s + encoding_s + target_transfer_s
                                                 + forward_s + backward_s + grad_clip_s + optimizer_s, 4),
                "loss": loss_value,
                "grad_norm_before_clip": grad_norm,
                "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            })
            log_event("online_profile.update_complete", **output_updates[-1])
    finally:
        online_module.log_event = original_log_event
        encoder.build_vlm_inputs = original_build
        encoder.encode_vlm = original_encode
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        close_dataset = getattr(full_ds, "close", None)
        if close_dataset is not None:
            close_dataset()

    properties = torch.cuda.get_device_properties(device)
    try:
        cpuset = Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip()
    except OSError:
        cpuset = None
    return {
        "schema": 1,
        "status": "complete",
        "purpose": "bounded performance diagnostic only; not a scientific result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": digest(plan),
        "plan_path": str(plan_path),
        "dataset": args.dataset,
        "backbone": args.backbone,
        "experiment": experiment,
        "training_seed": args.seed,
        "protocol_id": protocol["protocol_id"],
        "feature_mode": "online",
        "effective_batch_size": cfg.batch_size,
        "updates_requested": args.repeats,
        "microbatch_size": int(cfg.embed_batch_size),
        "loader_workers": args.workers,
        "torch": {
            "version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cpu_quota_cores": round(quota, 3) if quota is not None else None,
            "cpuset_cpus": cpuset,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "vlm_parameter_dtype": str(next(encoder.vlm.parameters()).dtype),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
        },
        "dataset_load_and_raw_preprocessing_s": round(data_ready_s, 3),
        "adapter_decoder_initialization_s": round(model_build_s, 3),
        "frozen_encoder_load_s": round(encoder_load_s, 3),
        "updates": output_updates,
        "vlm_input_and_output_shapes": vlm_input_shapes,
        "encoder_microbatch_timings": microbatch_events,
        "online_raw_data": online_data_info,
        "checkpoint_written": False,
        "validation_run": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("aloha", "libero"))
    parser.add_argument("--backbone", required=True, choices=("groot", "qwen3vl"))
    parser.add_argument("--exp", help="Must match the backbone arm in the locked plan")
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "configs/runpod_success_stopped_study.json")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--microbatch-size", type=int,
                        help="Override only VLM microbatching; effective optimizer batch remains 128")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Consecutive seeded optimizer updates; maximum 2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int,
                        help="PyTorch intra-op threads; defaults to cgroup-aware 2 on a 5-core Pod")
    parser.add_argument("--output-json", type=Path,
                        help="Results path; defaults to a timestamped profile under asset/runs/online_profiles")
    args = parser.parse_args(argv)
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.output_json is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_json = ROOT / "asset/runs/online_profiles" / (
            f"{args.dataset}_{args.backbone}_seed{args.seed}_{stamp}.json")
    else:
        args.output_json = args.output_json.resolve()
    started = time.perf_counter()
    try:
        result = profile(args)
    except BaseException as exc:
        result = {
            "schema": 1, "status": "failed",
            "purpose": "bounded performance diagnostic only; not a scientific result",
            "dataset": args.dataset, "backbone": args.backbone,
            "experiment": args.exp, "training_seed": args.seed,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "checkpoint_written": False, "validation_run": False,
        }
        write_result(args.output_json, result)
        log_event("online_profile.failed", level="ERROR", output_json=str(args.output_json),
                  error_type=type(exc).__name__, error=str(exc))
        raise
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    write_result(args.output_json, result)
    log_event("online_profile.complete", output_json=str(args.output_json),
              elapsed_s=result["elapsed_s"], updates=len(result["updates"]))
    print(json.dumps({"status": result["status"], "output_json": str(args.output_json),
                      "updates": result["updates"],
                      "encoder_microbatch_timings": result["encoder_microbatch_timings"]},
                     indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
