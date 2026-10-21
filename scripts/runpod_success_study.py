"""Run one dataset-role of the unbounded, development-success-stopped study."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from configs.registry import get_config
from experiment_protocol import (apply_normalization, cache_config, digest, encoder_code_id,
                                 file_digest, load_protocol, protocol_arrays, run_signature,
                                 source_assets, write_json)
from scripts.run_log import log_event, write_json_atomic
from scripts.runpod_bootstrap import cache_source_for_plan
from scripts.success_queue import publish_rollout_task


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def prime_backbone_image_token_id(cfg) -> int:
    """Resolve the lightweight processor/config field the encoder later writes to cfg.

    ``VLAModel`` resolves this value while loading its backbone. The runner must
    set it before signing the config, otherwise the same run gets a different
    signature before and after encoder construction.
    """
    from transformers import AutoConfig, AutoProcessor
    from models.backbones import MODELS, _img_token_id

    if cfg.vlm_backbone == "groot":
        model_root = MODELS / "cosmos_reason2_2b"
        if not model_root.exists():
            model_root = MODELS / "cosmos_reason2_2b_tok"
        processor_kwargs = {"size": {"shortest_edge": 256 * 256,
                                     "longest_edge": 256 * 256}}
    elif cfg.vlm_backbone == "qwen3vl":
        model_root = MODELS / "qwen3_vl_2b"
        processor_kwargs = {"size": {"shortest_edge": 256 * 256,
                                     "longest_edge": 256 * 256}}
    else:
        raise ValueError(f"Unsupported success-study backbone: {cfg.vlm_backbone}")

    processor = AutoProcessor.from_pretrained(str(model_root), **processor_kwargs)
    backbone_config = AutoConfig.from_pretrained(str(model_root))
    cfg.image_token_id = int(_img_token_id(backbone_config, processor))
    return cfg.image_token_id


def build_jobs(plan: dict, dataset: str, run_root: Path, workers: int,
               plan_path: Path, durable_root: Path) -> list[dict]:
    if plan.get("schema") != 2 or plan.get("checkpoint_rule") != "highest_development_success_rate_earliest_tie":
        raise ValueError("Unsupported success-stopped study schema or checkpoint rule")
    feature_mode = plan.get("feature_mode", "online")
    if feature_mode not in {"online", "cached"} or plan.get("training_seeds") != [3, 4, 5]:
        raise ValueError("Success-stopped plan must use online/cached encoding and independent seeds [3,4,5]")
    if dataset not in plan.get("datasets", {}):
        raise ValueError(f"Dataset is absent from the locked success study: {dataset}")
    stopping = plan["stopping"]
    if stopping.get("maximum_epochs") is not None:
        raise ValueError("The success-stopped study must have no maximum epoch limit")
    jobs = []
    setting = plan["datasets"][dataset]
    protocol_path = ROOT / setting["split_manifest"]
    protocol = load_protocol(protocol_path, dataset)
    study_id = run_root.name
    cache_study_id, cache_plan_id, _ = cache_source_for_plan(plan, dataset, study_id, ROOT)
    validated_caches: dict[Path, dict] = {}
    for seed in plan["training_seeds"]:
        for experiment in (plan["backbones"]["groot"], plan["backbones"]["qwen3vl"]):
            cfg = get_config(dataset, experiment)
            apply_normalization(cfg, protocol)
            cfg.batch_size = int(plan["effective_batch_size"])
            cfg.num_epochs = int(setting["reference_epochs"])
            cfg.min_epochs = int(stopping["minimum_epochs"])
            cfg.warmup_steps = int(plan["warmup_updates"])
            cfg.min_lr_ratio = float(plan["minimum_learning_rate_ratio"])
            cfg.lr_decay_epochs = None
            prime_backbone_image_token_id(cfg)
            expected_views = len(setting.get("camera_views", ["top"]))
            if getattr(cfg, "n_views", 1) != expected_views:
                raise ValueError(f"{dataset}/{experiment} camera count differs from the locked plan")
            if cfg.batch_size != int(plan["effective_batch_size"]):
                raise ValueError("Resolved effective batch size differs from the locked plan")
            assets = source_assets(cfg)
            run = run_root / dataset / Path(cfg.output_dir).name / f"seed_{seed:03d}"
            durable_run = (durable_root / "success_studies" / run_root.name / dataset
                           / run.relative_to(run_root))
            cache_path = None
            if feature_mode == "cached":
                cache_path = (ROOT / "asset/cache/cached_success_study" / cache_study_id / dataset
                              / experiment / "vlm_embeddings.h5").resolve()
                cfg.embeddings_cache = str(cache_path)
                durable_cache = (durable_root / "success_studies" / cache_study_id / "feature_caches"
                                 / dataset / experiment / "vlm_embeddings.h5").resolve()
                if cache_path not in validated_caches:
                    validated_caches[cache_path] = validate_cached_cache(
                        cache_path, durable_cache, dataset=dataset, experiment=experiment,
                        backbone=cfg.vlm_backbone, study_id=cache_study_id, plan_id=cache_plan_id,
                        protocol=protocol, cfg=cfg, locked_source_assets=assets)
            run_cfg = json.loads(json.dumps({**asdict(cfg), "output_dir": str(run)}))
            command = [sys.executable, "-u", str(ROOT / "scripts/train.py"),
                       "--dataset", dataset, "--exp", experiment, "--seed", str(seed),
                       "--split-manifest", str(protocol_path)]
            if feature_mode == "online":
                command.append("--online-encoder")
            else:
                command.extend(["--cache", str(cache_path)])
            command.extend([
                       "--output-dir", str(run), "--workers", str(workers),
                       "--cpu-threads", "2",
                       "--batch-size", str(plan["effective_batch_size"]),
                       "--steps-per-epoch", str(setting["steps_per_epoch"]),
                       "--warmup-steps", str(plan["warmup_updates"]),
                       "--lr-decay-steps", str(plan["reference_optimizer_updates"]),
                       "--rollout-stopping-config", str(plan_path),
                       "--rollout-durable-root", str(durable_root),
                       "--rollout-durable-run-dir", str(durable_run),
                       "--rollout-study-id", run_root.name,
                       "--rollout-plan-id", digest(plan)])
            jobs.append({"dataset": dataset, "experiment": experiment,
                         "backbone": cfg.vlm_backbone, "training_seed": seed,
                         "feature_mode": feature_mode,
                         "protocol_id": protocol["protocol_id"], "protocol": protocol,
                         "protocol_path": protocol_path,
                         "run_signature": run_signature(cfg, protocol, seed),
                         "config": run_cfg, "run": run, "durable_run": durable_run,
                         "command": command, "cache": str(cache_path) if cache_path else None,
                         "cache_id": (validated_caches[cache_path]["cache_id"]
                                      if cache_path else None),
                         "cache_sha256": (validated_caches[cache_path]["sha256"]
                                          if cache_path else None),
                         "source_assets": assets, "settings": setting})
    return jobs


def validate_cached_cache(cache_path: Path, durable_cache: Path, *, dataset: str,
                          experiment: str, backbone: str, study_id: str, plan_id: str,
                          protocol: dict, cfg, locked_source_assets: dict) -> dict:
    """Fail closed unless the staged local HDF5 cache and its publisher manifest agree."""
    manifest_path = cache_path.with_name(cache_path.name + ".manifest.json")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cached success study requires staged cache and sidecar: {cache_path} "
            f"(manifest {manifest_path})")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema": 1, "dataset": dataset, "experiment": experiment, "backbone": backbone,
        "study_id": study_id, "plan_id": plan_id, "protocol_id": protocol["protocol_id"],
        "encoder_code_id": encoder_code_id(), "source_assets": locked_source_assets,
        "size_bytes": cache_path.stat().st_size, "local_path": str(cache_path.resolve()),
        "durable_path": str(durable_cache.resolve()),
    }
    mismatches = {key: (manifest.get(key), expected)
                  for key, expected in expected_manifest.items()
                  if manifest.get(key) != expected}
    actual_sha = file_digest(cache_path)
    if manifest.get("sha256") != actual_sha:
        mismatches["sha256"] = (manifest.get("sha256"), actual_sha)
    if mismatches:
        raise ValueError(f"Cached study manifest mismatch for {cache_path}: {mismatches}")

    import h5py
    with h5py.File(cache_path, "r") as cache:
        attrs = cache.attrs
        expected_ids, expected_validation = protocol_arrays(protocol)
        expected_samples = int(protocol["n_samples"])
        required_datasets = {"embeddings", "img_masks", "states", "actions",
                             "episode_ids", "validation_mask"}
        missing = required_datasets.difference(cache.keys())
        if missing:
            raise ValueError(f"Cached HDF5 datasets missing from {cache_path}: {sorted(missing)}")
        layer_count = len(cfg.vlm_extract_layers)
        embedding_shape = ((expected_samples, layer_count, cfg.img_seq_len, cfg.vlm_hidden_size)
                           if layer_count > 1 else
                           (expected_samples, cfg.img_seq_len, cfg.vlm_hidden_size))
        parity = json.loads(attrs.get("batch_one_parity", "{}"))
        actual = {
            "complete": bool(attrs.get("complete", False)),
            "n_samples": int(attrs.get("n_samples", -1)),
            "dataset": attrs.get("dataset"), "experiment": attrs.get("exp_id"),
            "protocol_id": attrs.get("protocol_id"), "cache_id": attrs.get("cache_id"),
            "encoder_code_id": attrs.get("encoder_code_id"),
            "source_assets": json.loads(attrs.get("source_assets", "null")),
            "cache_config": json.loads(attrs.get("cache_config", "null")),
            "full_dataset": bool(attrs.get("full_dataset", False)),
            "layers": tuple(int(x) for x in attrs.get("vlm_extract_layers", ())),
            "embedding_shape": tuple(cache["embeddings"].shape),
            "mask_shape": tuple(cache["img_masks"].shape),
            "state_shape": tuple(cache["states"].shape),
            "action_shape": tuple(cache["actions"].shape),
            "episode_ids_match": (tuple(cache["episode_ids"].shape) == (expected_samples,)
                                  and bool((cache["episode_ids"][:] == expected_ids).all())),
            "validation_mask_match": (
                tuple(cache["validation_mask"].shape) == (expected_samples,)
                and bool((cache["validation_mask"][:].astype(bool)
                          == expected_validation).all())),
            "batch_one_parity": (bool(parity.get("rows"))
                                 and parity.get("masks_equal") is True),
        }
    expected_cache = {
        "complete": True, "n_samples": expected_samples,
        "dataset": dataset, "experiment": experiment,
        "protocol_id": protocol["protocol_id"], "cache_id": manifest.get("cache_id"),
        "encoder_code_id": expected_manifest["encoder_code_id"],
        "source_assets": locked_source_assets, "cache_config": cache_config(cfg),
        "full_dataset": True,
        "layers": tuple(int(x) for x in cfg.vlm_extract_layers),
        "embedding_shape": embedding_shape,
        "mask_shape": (expected_samples, cfg.img_seq_len),
        "state_shape": (expected_samples, cfg.state_dim),
        "action_shape": (expected_samples, cfg.action_horizon, cfg.action_dim),
        "episode_ids_match": True, "validation_mask_match": True,
        "batch_one_parity": True,
    }
    hdf5_mismatches = {key: (actual.get(key), expected)
                       for key, expected in expected_cache.items()
                       if actual.get(key) != expected}
    if hdf5_mismatches:
        raise ValueError(f"Cached HDF5 metadata mismatch for {cache_path}: {hdf5_mismatches}")
    return {"cache_id": actual["cache_id"], "sha256": actual_sha,
            "source_assets": actual["source_assets"]}


def sync_file(source: Path, destination: Path) -> bool:
    """Copy a stable source file atomically; discard a copy made during a write."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        before = source.stat()
        if (destination.is_file() and destination.stat().st_size == before.st_size
                and destination.stat().st_mtime_ns >= before.st_mtime_ns):
            return False
    except OSError:
        before = source.stat()
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        after = source.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or temporary.stat().st_size != after.st_size):
            return False
        os.replace(temporary, destination)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def sync_tree(source: Path, destination: Path) -> int:
    if not source.is_dir():
        return 0
    copied = 0
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if (path.is_file() and not path.name.startswith("recovery_e")
                and relative.as_posix() != "checkpoints/last.pt"):
            copied += int(sync_file(path, destination / relative))
    return copied


def publish_code_snapshot(code_paths: list[str], hashes: dict[str, str], destination: Path) -> None:
    """Keep the exact runner/trainer/evaluator sources beside durable checkpoints."""
    for relative in code_paths:
        source = ROOT / relative
        target = destination / "code" / relative
        expected_sha = hashes[relative]
        if target.is_file() and file_digest(target) == expected_sha:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if file_digest(temporary) != expected_sha:
                raise IOError(f"Durable workflow code copy failed verification: {relative}")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


def tail(path: Path, count: int = 30) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 192 * 1024))
            return stream.read().decode("utf-8", errors="replace").splitlines()[-count:]
    except OSError:
        return []


def selected_artifacts(run: Path, job: dict) -> tuple[Path, dict]:
    summary_path = run / "selection_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Trainer did not publish a selection summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {"dataset": job["dataset"], "experiment": job["experiment"],
                "training_seed": job["training_seed"], "protocol_id": job["protocol_id"],
                "run_signature": job["run_signature"]}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Trainer selection summary identity differs from this locked policy")
    if summary.get("plan_id") is None:
        raise ValueError("Trainer selection summary lacks the locked plan digest")
    checkpoint_path_value = summary.get("selected_checkpoint_path")
    checkpoint_path = Path(checkpoint_path_value) if checkpoint_path_value else run / "checkpoints/selected.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = run / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected development checkpoint is missing: {checkpoint_path}")
    expected_sha = summary.get("selected_checkpoint_sha256", summary.get("checkpoint_sha256"))
    if not expected_sha or file_digest(checkpoint_path) != expected_sha:
        raise ValueError("Selected checkpoint hash does not match the trainer's selection summary")
    selected_epoch = summary.get("selection_epoch", summary.get("best_epoch", summary.get("selected_epoch")))
    if type(selected_epoch) is not int or selected_epoch < 1:
        raise ValueError("Selection summary lacks selected development epoch")
    selected_rate = summary.get("best_development_success_rate",
                                summary.get("best_success_rate", summary.get("selected_success_rate")))
    if (isinstance(selected_rate, bool) or not isinstance(selected_rate, (int, float))
            or not 0 <= selected_rate <= 1):
        raise ValueError("Selection summary lacks the selected development success rate")
    summary["selection_epoch"] = selected_epoch
    summary["best_development_success_rate"] = float(selected_rate)
    return checkpoint_path, summary


def publish_final(job: dict, *, plan: dict, plan_id: str, study_id: str,
                  durable_root: Path) -> dict:
    run = job["run"]
    checkpoint_path, summary = selected_artifacts(run, job)
    if summary.get("plan_id") != plan_id:
        raise ValueError("Trainer selection summary plan digest differs from the locked plan")
    checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    step = checkpoint.get("optimizer_step", checkpoint.get("global_step"))
    selection_epoch = int(summary["selection_epoch"])
    selected_sha = str(summary.get("selected_checkpoint_sha256", summary.get("checkpoint_sha256")))
    metadata = {"dataset": job["dataset"], "experiment": job["experiment"],
                "backbone": job["backbone"], "training_seed": job["training_seed"],
                "protocol_id": job["protocol_id"], "run_signature": job["run_signature"],
                "feature_mode": job["feature_mode"], "source_assets": job["source_assets"],
                "optimizer_step": step, "selection_epoch": selection_epoch,
                "selected_checkpoint_sha256": selected_sha}
    evaluation = dict(job["settings"]["final_evaluation"])
    task = publish_rollout_task(checkpoint.get("cfg"), metadata, checkpoint_path,
                                evaluation, "final_test", plan_id, study_id, durable_root)
    log_event("success_study.final_test_enqueued", task_id=task["task_id"],
              dataset=job["dataset"], experiment=job["experiment"],
              training_seed=job["training_seed"], selected_epoch=selection_epoch,
              selected_checkpoint_sha256=selected_sha)
    return task


def policy_key(job: dict) -> str:
    return f"{job['dataset']}/{job['experiment']}/seed_{job['training_seed']:03d}"


def update_job_status(status_path: Path, durable_status_path: Path, status: dict) -> None:
    status["updated_utc"] = stamp()
    write_json_atomic(status_path, status)
    sync_file(status_path, durable_status_path)


def run_training_job(job: dict, *, durable_root: Path, poll_seconds: float,
                     process_registry: dict, registry_lock: threading.Lock) -> dict:
    run = job["run"]
    run.mkdir(parents=True, exist_ok=True)
    durable_run = job["durable_run"]
    restored = sync_tree(durable_run, run)
    if restored:
        log_event("success_study.durable_state_restored", policy=f"{job['experiment']}/seed_{job['training_seed']}",
                  restored_files=restored, durable_run=str(durable_run))
    command = list(job["command"])
    command[command.index("--rollout-durable-root") + 1] = str(durable_root)
    log_path = run / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = run / "progress_runner.json"
    durable_status_path = durable_run / "progress_runner.json"
    identity = {key: job.get(key) for key in (
        "dataset", "experiment", "training_seed", "backbone", "feature_mode", "cache_id",
        "cache_sha256")}
    status = {"policy_id": policy_key(job), **identity, "stage": "starting",
              "log": str(log_path), "command": command}
    update_job_status(status_path, durable_status_path, status)
    log_event("success_study.training_started", dataset=job["dataset"],
              experiment=job["experiment"], training_seed=job["training_seed"],
              run=str(run), log=str(log_path), runner_status=str(status_path))
    sync_tree(run, durable_run)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        child_env = dict(os.environ, PYTHONIOENCODING="utf-8",
                         PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=child_env)
        with registry_lock:
            process_registry[policy_key(job)] = {"process": process, "job": job,
                                                 "log": log_path}
        status.update(stage="training", pid=process.pid, started_utc=stamp())
        update_job_status(status_path, durable_status_path, status)
        last_sync = time.monotonic()
        last_status = time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(min(max(poll_seconds, 0.5), 10))
                log.flush()
                now = time.monotonic()
                if now - last_sync >= 60:
                    copied = sync_tree(run, durable_run)
                    if copied:
                        log_event("success_study.durable_sync", policy=policy_key(job),
                                  copied_files=copied)
                    last_sync = now
                if now - last_status >= 10:
                    status.update(stage_elapsed_s=round(time.perf_counter() - started, 1),
                                  log_tail=tail(log_path, 15), pid=process.pid)
                    update_job_status(status_path, durable_status_path, status)
                    last_status = now
        finally:
            with registry_lock:
                process_registry.pop(policy_key(job), None)
    copied = sync_tree(run, durable_run)
    if process.returncode:
        status.update(stage="failed", return_code=process.returncode,
                      stage_elapsed_s=round(time.perf_counter() - started, 1),
                      log_tail=tail(log_path, 30))
        update_job_status(status_path, durable_status_path, status)
        raise subprocess.CalledProcessError(process.returncode, command)
    status.update(stage="training_complete", return_code=0,
                  stage_elapsed_s=round(time.perf_counter() - started, 1),
                  log_tail=tail(log_path, 15), pid=None)
    update_job_status(status_path, durable_status_path, status)
    log_event("success_study.training_complete", dataset=job["dataset"],
              experiment=job["experiment"], training_seed=job["training_seed"],
              elapsed_s=round(time.perf_counter() - started, 1), durable_files_copied=copied,
              tail=tail(log_path, 12))
    return {"policy_id": policy_key(job), "log": str(log_path),
            "runner_status": str(status_path), "elapsed_s": round(time.perf_counter() - started, 1)}


def aggregate_active(active: dict[Future, dict]) -> list[dict]:
    snapshots = []
    for future, job in active.items():
        path = job["run"] / "progress_runner.json"
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            snapshot = {"stage": "queued"}
        snapshots.append({"policy_id": policy_key(job), **snapshot,
                          "future_done": future.done()})
    return snapshots


def write_study_status(status: dict, local_path: Path, durable_path: Path) -> None:
    status["updated_utc"] = stamp()
    write_json_atomic(local_path, status)
    write_json_atomic(durable_path, status)


def add_completed_policy(status: dict, job: dict, task: dict) -> None:
    key = policy_key(job)
    ready = {"policy_id": key, "dataset": job["dataset"], "experiment": job["experiment"],
             "training_seed": job["training_seed"],
             "selected_checkpoint_sha256": task["selected_checkpoint_sha256"],
             "selected_epoch": task["selection_epoch"], "task_id": task["task_id"]}
    status["completed_policies"] = [entry for entry in status.get("completed_policies", [])
                                    if entry.get("policy_id") != key]
    status["completed_policies"].append(ready)
    status["pending_final_tests"] = list(dict.fromkeys(
        [*status.get("pending_final_tests", []), task["task_id"]]))


def stop_owned_processes(process_registry: dict, registry_lock: threading.Lock,
                         grace_seconds: float = 20) -> None:
    """Checkpoint durable files, then stop only trainer children this runner owns."""
    with registry_lock:
        children = list(process_registry.items())
    for _, item in children:
        sync_tree(item["job"]["run"], item["job"]["durable_run"])
        process = item["process"]
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except (OSError, ProcessLookupError):
                pass
    deadline = time.monotonic() + grace_seconds
    for _, item in children:
        process = item["process"]
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        sync_tree(item["job"]["run"], item["job"]["durable_run"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("aloha", "libero"),
                        help="This training Pod owns exactly one simulator dataset role")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_success_stopped_study.json")
    parser.add_argument("--run-root", type=Path,
                        help="Role-specific run root; never reuse the fixed-update study root")
    parser.add_argument("--durable-root", type=Path,
                        default=Path(os.environ.get("ACTIONOMNI_DURABLE_ROOT", "/workspace-global/action-omni")))
    parser.add_argument("--workers", type=int, default=1,
                        help="DataLoader workers per training process")
    parser.add_argument("--parallel-policies", type=int, default=2,
                        help="Concurrent independent training processes on this dataset Pod (maximum 2)")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--status-seconds", type=float, default=10,
                        help="Print aggregate active-policy progress at this interval")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 0 or not 1 <= args.parallel_policies <= 2 \
            or args.poll_seconds <= 0 or args.status_seconds <= 0:
        parser.error("--workers must be nonnegative; --parallel-policies must be 1 or 2; poll/status intervals must be positive")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_id = digest(plan)
    run_name = f"{plan.get('name', 'runpod_success_stopped_v1')}_{args.dataset}"
    run_root = (args.run_root or (ROOT / "asset/runs" / run_name)).resolve()
    if any("runpod_update_matched" in part for part in run_root.parts):
        parser.error("Success-stopped training must use a new role-specific run root, never the fixed-update root")
    durable_root = args.durable_root.resolve()
    plan_path = args.plan.resolve()
    jobs = build_jobs(plan, args.dataset, run_root, args.workers, plan_path, durable_root)
    if args.dry_run:
        print(json.dumps({"mode": "dry-run; no files written", "plan_id": plan_id,
                          "dataset": args.dataset, "jobs": [
                              {"dataset": j["dataset"], "experiment": j["experiment"],
                               "training_seed": j["training_seed"], "run": str(j["run"]),
                               "run_signature": j["run_signature"],
                               "feature_mode": j["feature_mode"], "cache_id": j["cache_id"],
                               "cache_sha256": j["cache_sha256"], "command": j["command"]}
                              for j in jobs], "parallel_policies": args.parallel_policies,
                          "workers_per_policy": args.workers, "cpu_threads_per_policy": 2}, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no success-stopped training was started")
    if not Path("/workspace-global").is_dir():
        raise RuntimeError("RunPod Global Volume is not mounted at /workspace-global")
    durable_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    study_id = run_root.name
    code_paths = ["scripts/runpod_success_study.py", "scripts/success_queue.py",
                  "scripts/success_stopping.py", "scripts/train.py", "scripts/checkpoint_io.py",
                  "scripts/online_encoder.py", "scripts/plot_study_losses.py",
                  "scripts/runpod_bootstrap.py",
                  "scripts/runpod_eval_worker.py", "scripts/evaluate.py", "scripts/eval_libero.py",
                  "scripts/setup_runpod_eval.sh", "scripts/stage_runpod_assets.py",
                  "scripts/analysis/paired_results.py", "scripts/analysis/runpod_paper_results.py",
                  "experiment_protocol.py", "models/vla_train.py", "models/flow_matching.py",
                  "models/vla.py", "models/backbones.py", "data/aloha/dataset.py",
                  "data/libero/dataset.py", "data/libero/frames.py"]
    code_hashes = {path: file_digest(ROOT / path) for path in code_paths}
    contract = {"plan_id": plan_id, "plan": plan, "dataset_role": args.dataset,
                "jobs": [{key: job[key] for key in ("dataset", "experiment", "backbone",
                                                    "training_seed", "protocol_id", "run_signature",
                                                    "feature_mode", "cache_id", "cache_sha256")}
                         for job in jobs],
                "workflow_code": code_hashes}
    lock_path = run_root / "locked_success_study.json"
    if lock_path.is_file() and json.loads(lock_path.read_text(encoding="utf-8")) != contract:
        raise ValueError("Locked success-stopped plan/code changed; use a fresh role-specific run root")
    if not lock_path.is_file():
        write_json(lock_path, contract)
    status_path = run_root / "study_status.json"
    durable_study_root = durable_root / "success_studies" / study_id
    durable_study_root.mkdir(parents=True, exist_ok=True)
    durable_status_path = durable_study_root / "study_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("study_id") != study_id or status.get("plan_id") != plan_id:
            raise ValueError("Existing study status belongs to another plan or study ID")
    else:
        status = {"study_id": study_id, "plan_id": plan_id, "dataset_role": args.dataset,
                  "started_utc": stamp(), "total_policies": len(jobs),
                  "completed_policies": [], "pending_final_tests": []}
    status.update(status="running", stage="starting", dataset_role=args.dataset,
                  total_policies=len(jobs), parallel_policies=args.parallel_policies,
                  workers_per_policy=args.workers, active_policies=[],
                  pending_policies=[policy_key(job) for job in jobs], updated_utc=stamp())
    publish_code_snapshot(code_paths, code_hashes, durable_study_root)
    write_json_atomic(durable_study_root / "locked_success_study.json", contract)
    write_study_status(status, status_path, durable_status_path)
    log_event("success_study.started", study_id=study_id, plan_id=plan_id,
              dataset=args.dataset, policies=len(jobs), parallel_policies=args.parallel_policies,
              workers_per_policy=args.workers, cpu_threads_per_policy=2, run_root=str(run_root),
              durable_root=str(durable_root))
    process_registry: dict[str, dict] = {}
    registry_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=args.parallel_policies,
                                  thread_name_prefix="action-omni-train")
    pending_jobs = list(jobs)
    active: dict[Future, dict] = {}
    failures: list[dict] = []
    last_status_log = time.monotonic()
    try:
        while pending_jobs or active:
            while pending_jobs and len(active) < args.parallel_policies and not failures:
                job = pending_jobs.pop(0)
                status["pending_policies"] = [policy_key(item) for item in pending_jobs]
                manifest_path = job["run"] / "selection_summary.json"
                if manifest_path.is_file():
                    try:
                        task = publish_final(job, plan=plan, plan_id=plan_id,
                                             study_id=study_id, durable_root=durable_root)
                        add_completed_policy(status, job, task)
                        log_event("success_study.policy_ready", task_id=task["task_id"],
                                  policy_id=policy_key(job), selected_epoch=task["selection_epoch"],
                                  resumed=True)
                    except BaseException as exc:
                        failures.append({"policy_id": policy_key(job),
                                         "error": f"{type(exc).__name__}: {exc}"})
                        status.update(status="failed", stage="failed_waiting_active",
                                      error=failures[-1]["error"], failed_policy=policy_key(job))
                        log_event("success_study.policy_failed", level="ERROR",
                                  policy_id=policy_key(job), error_type=type(exc).__name__,
                                  error=str(exc))
                        break
                else:
                    future = executor.submit(run_training_job, job, durable_root=durable_root,
                                              poll_seconds=args.poll_seconds,
                                              process_registry=process_registry,
                                              registry_lock=registry_lock)
                    active[future] = job
                    log_event("success_study.policy_submitted", policy_id=policy_key(job),
                              parallel_active=len(active), log=str(job["run"] / "train.log"))
                status["active_policies"] = aggregate_active(active)
                write_study_status(status, status_path, durable_status_path)

            if active:
                done, _ = wait(active, timeout=min(args.status_seconds, 10),
                               return_when=FIRST_COMPLETED)
            else:
                done = set()
            for future in done:
                job = active.pop(future)
                try:
                    result = future.result()
                    task = publish_final(job, plan=plan, plan_id=plan_id,
                                         study_id=study_id, durable_root=durable_root)
                    add_completed_policy(status, job, task)
                    log_event("success_study.policy_ready", task_id=task["task_id"],
                              policy_id=policy_key(job), selected_epoch=task["selection_epoch"],
                              selected_checkpoint_sha256=task["selected_checkpoint_sha256"],
                              training_elapsed_s=result["elapsed_s"])
                except BaseException as exc:
                    failure = {"policy_id": policy_key(job),
                               "error": f"{type(exc).__name__}: {exc}"}
                    failures.append(failure)
                    per_job_status = job["run"] / "progress_runner.json"
                    try:
                        runner_status = json.loads(per_job_status.read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        runner_status = {}
                    runner_status.update(policy_id=policy_key(job), stage="failed",
                                         error=failure["error"], updated_utc=stamp())
                    update_job_status(per_job_status, job["durable_run"] / "progress_runner.json",
                                      runner_status)
                    status.update(status="failed", stage="failed_waiting_active",
                                  failed_policy=policy_key(job), error=failure["error"])
                    log_event("success_study.policy_failed", level="ERROR",
                              policy_id=policy_key(job), error_type=type(exc).__name__,
                              error=str(exc), active_pids=[item.get("pid")
                                  for item in aggregate_active(active) if item.get("pid")])
            if time.monotonic() - last_status_log >= args.status_seconds:
                status["active_policies"] = aggregate_active(active)
                status["completed_policy_count"] = len(status.get("completed_policies", []))
                status["pending_policies"] = [policy_key(job) for job in pending_jobs]
                write_study_status(status, status_path, durable_status_path)
                log_event("success_study.progress", study_id=study_id,
                          active=status["active_policies"],
                          completed_policy_count=status["completed_policy_count"],
                          pending_policies=status["pending_policies"], failures=failures)
                last_status_log = time.monotonic()

            if failures and not active:
                break

        status["active_policies"] = aggregate_active(active)
        status["pending_policies"] = [policy_key(job) for job in pending_jobs]
        if failures:
            status.update(status="failed", stage="failed", failures=failures,
                          failed_utc=stamp(), completed_utc=stamp())
            write_study_status(status, status_path, durable_status_path)
            return 1
        status.update(status="training_complete_evaluation_pending", stage="training_complete",
                      completed_utc=stamp(), updated_utc=stamp())
        write_study_status(status, status_path, durable_status_path)
        log_event("success_study.training_role_complete", study_id=study_id,
                  dataset=args.dataset, final_tests=len(status["pending_final_tests"]))
    except (KeyboardInterrupt, SystemExit):
        status.update(status="interrupted", stage="stopping_owned_trainers",
                      active_policies=aggregate_active(active), interrupted_utc=stamp())
        write_study_status(status, status_path, durable_status_path)
        log_event("success_study.interrupted", level="WARNING", study_id=study_id,
                  active_pids=[item.get("pid") for item in aggregate_active(active) if item.get("pid")])
        stop_owned_processes(process_registry, registry_lock)
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    except BaseException as exc:
        status.update(status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(), failed_utc=stamp(),
                      active_policies=aggregate_active(active))
        write_study_status(status, status_path, durable_status_path)
        log_event("success_study.failed", level="ERROR", study_id=study_id,
                  error_type=type(exc).__name__, error=str(exc))
        stop_owned_processes(process_registry, registry_lock)
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
