"""Run one or both dataset workers of the seed-locked camera-ready study.

RunPod Global Volume keeps durable outputs; active feature caches and training
I/O stay on each Pod's local disk. Re-runs reuse validated caches and resume.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from experiment_protocol import (apply_normalization, cache_config, digest,
                                 encoder_code_id, file_digest, load_protocol,
                                 run_signature, source_assets, write_json)
from scripts.run_experiment import ensure_cache
from scripts.run_log import log_event, write_json_atomic


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def build_jobs(plan: dict, run_root: Path, datasets: list[str], workers: int) -> list[dict]:
    if plan.get("schema") != 1 or plan.get("checkpoint_rule") != "fixed_update_endpoint":
        raise ValueError("Unsupported RunPod study schema or checkpoint rule")
    seeds = plan["training_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Study requires three distinct, recorded nonnegative training seeds")
    if plan["effective_batch_size"] != 128:
        raise ValueError("This study is locked to effective batch size 128")
    jobs = []
    for dataset in datasets:
        setting = plan["datasets"][dataset]
        protocol_path = ROOT / setting["split_manifest"]
        protocol = load_protocol(protocol_path, dataset)
        for exp in (plan["backbones"]["groot"], plan["backbones"]["qwen3vl"]):
            cfg = get_config(dataset, exp)
            apply_normalization(cfg, protocol)
            cfg.batch_size = plan["effective_batch_size"]
            cfg.num_epochs = setting["epochs"]
            cfg.min_epochs = setting["epochs"]  # fixed horizon; validation never shortens it
            cfg.warmup_steps = plan["warmup_updates"]
            cfg.lr_decay_epochs = None
            if getattr(cfg, "n_views", 1) != len(setting.get("camera_views", ["top"])):
                raise ValueError(f"{dataset}/{exp} camera count differs from the locked plan")
            if cfg.num_epochs * setting["steps_per_epoch"] != plan["optimizer_updates"]:
                raise ValueError(f"{dataset} epoch geometry does not equal the fixed update budget")
            if setting["probe_epochs"] != [step // setting["steps_per_epoch"]
                                            for step in setting["probe_updates"]]:
                raise ValueError(f"{dataset} probe updates do not align with completed epochs")
            assets = source_assets(cfg)
            identity = {"protocol_id": protocol["protocol_id"], "config": cache_config(cfg),
                        "encoder_code_id": encoder_code_id(), "source_assets": assets}
            cache = ROOT / "asset/cache" / dataset / cfg.vlm_backbone / digest(identity) / "vlm_embeddings.h5"
            for seed in seeds:
                run = run_root / dataset / Path(cfg.output_dir).name / f"seed_{seed:03d}"
                run_cfg = json.loads(json.dumps({**asdict(cfg), "output_dir": str(run)}))
                train = [sys.executable, "-u", str(ROOT / "scripts/train.py"),
                         "--dataset", dataset, "--exp", exp, "--seed", str(seed),
                         "--split-manifest", str(protocol_path), "--cache", str(cache),
                         "--output-dir", str(run), "--workers", str(workers),
                         "--epochs", str(setting["epochs"]), "--min-epochs", str(setting["epochs"]),
                         "--batch-size", "128", "--steps-per-epoch", str(setting["steps_per_epoch"]),
                         "--max-steps", str(plan["optimizer_updates"]),
                         "--warmup-steps", str(plan["warmup_updates"]),
                         "--lr-decay-steps", str(plan["optimizer_updates"]),
                         "--checkpoint-every-steps", str(plan["checkpoint_every_updates"]),
                         "--probe-epochs", *map(str, setting["probe_epochs"])]
                jobs.append({"dataset": dataset, "experiment": exp, "backbone": cfg.vlm_backbone,
                             "training_seed": seed, "protocol_id": protocol["protocol_id"],
                             "run_signature": run_signature(cfg, protocol, seed),
                             "config": run_cfg, "cache": str(cache), "cache_identity": identity,
                             "cache_config": cfg, "protocol": protocol, "run": str(run),
                             "train": train, "settings": setting})
    return jobs


def result_path(run: Path, step: int, final: bool) -> Path:
    label = "final" if final else f"step_{step:08d}"
    return run / f"evaluation_{label}.json"


def verify_policy(job: dict, total_steps: int) -> None:
    run = Path(job["run"])
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    final = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    if (manifest.get("status") != "complete" or manifest.get("training_seed") != job["training_seed"]
            or manifest.get("protocol_id") != job["protocol_id"]
            or manifest.get("run_signature") != job["run_signature"]
            or manifest.get("optimizer_steps") != total_steps
            or final.get("training_seed") != job["training_seed"]
            or final.get("global_step") != total_steps
            or final.get("run_signature") != job["run_signature"]):
        raise ValueError(f"Policy did not reach its exact locked endpoint: {run}")
    for step in job["settings"]["probe_updates"]:
        if not (run / "checkpoints" / f"probe_step_{step:08d}.pt").is_file():
            raise FileNotFoundError(f"Missing probe checkpoint at update {step}: {run}")


def evaluation_jobs(job: dict, total_steps: int) -> list[tuple[int, bool, Path, list[str]]]:
    run = Path(job["run"])
    settings = job["settings"]
    specs = [(step, False) for step in settings["probe_updates"]] + [(total_steps, True)]
    output = []
    for step, final in specs:
        checkpoint = run / "checkpoints" / ("final.pt" if final else f"probe_step_{step:08d}.pt")
        result = result_path(run, step, final)
        if job["dataset"] == "aloha":
            count = settings["final_episodes"] if final else settings["probe_episodes"]
            seed_offset = settings["final_seed_offset"] if final else settings["probe_seed_offset"]
            command = [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"),
                       "--dataset", "aloha", "--exp", job["experiment"],
                       "--checkpoint", str(checkpoint), "--episodes", str(count),
                       "--seed-offset", str(seed_offset), "--render-samples", "0",
                       "--no-video", "--output", str(result)]
        else:
            count = settings["final_episodes_per_task"] if final else settings["probe_episodes_per_task"]
            command = [sys.executable, "-u", str(ROOT / "scripts/eval_libero.py"),
                       "--exp", job["experiment"], "--checkpoint", checkpoint.name,
                       "--episodes", str(count), "--conditions", "canonical", "--videos", "0",
                       "--seed", "0", "--output-dir", str(run), "--output", str(result)]
        output.append((step, final, result, command))
    return output


def checkpoint_candidates(job: dict, total_steps: int) -> list[tuple[int, bool, Path]]:
    """Checkpoint files that are part of the locked evaluation ladder."""
    run = Path(job["run"])
    settings = job["settings"]
    checkpoints = [(step, False, run / "checkpoints" / f"probe_step_{step:08d}.pt")
                   for step in settings["probe_updates"]]
    checkpoints.append((total_steps, True, run / "checkpoints" / "final.pt"))
    return checkpoints


def checkpoint_eval_spec(job: dict, step: int, final: bool) -> dict:
    settings = job["settings"]
    if job["dataset"] == "aloha":
        return {"episodes": settings["final_episodes"] if final else settings["probe_episodes"],
                "seed_offset": settings["final_seed_offset"] if final else settings["probe_seed_offset"],
                "render_samples": 0, "videos": False,
                "capture_trajectory": bool(final)}
    return {"episodes_per_task": settings["final_episodes_per_task"] if final
            else settings["probe_episodes_per_task"],
            "tasks": settings.get("tasks", list(range(10))), "conditions": ["canonical"],
            "environment_seed": 0, "videos": 0}


def _study_component(value: str) -> str:
    """Keep external task paths inside their namespaced durable directories."""
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in str(value))
    if safe in {"", ".", ".."}:
        raise ValueError(f"Unsafe study path component: {value!r}")
    return safe


def _checkpoint_identity(checkpoint: dict, job: dict, step: int, *, online_encoder: bool) -> None:
    expected = {"dataset": job["dataset"], "exp_id": job["experiment"],
                "training_seed": job["training_seed"], "protocol_id": job["protocol_id"],
                "run_signature": job["run_signature"]}
    mismatches = {key: (checkpoint.get(key), value) for key, value in expected.items()
                  if checkpoint.get(key) != value}
    actual_step = checkpoint.get("optimizer_step", checkpoint.get("global_step"))
    if actual_step != step:
        mismatches["optimizer_step"] = (actual_step, step)
    if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
        mismatches["state_dict"] = ("missing or empty", "nonempty mapping")
    cfg = checkpoint.get("cfg")
    if cfg is None or getattr(cfg, "vlm_backbone", None) != job["backbone"]:
        mismatches["backbone"] = (getattr(cfg, "vlm_backbone", None), job["backbone"])
    feature_mode = checkpoint.get("feature_mode")
    expected_mode = "online" if online_encoder else "cached"
    if feature_mode is not None and feature_mode != expected_mode:
        mismatches["feature_mode"] = (feature_mode, expected_mode)
    if online_encoder and feature_mode != "online":
        mismatches["feature_mode"] = (feature_mode, "online")
    if mismatches:
        raise ValueError(f"Checkpoint identity mismatch: {mismatches}")


def _atomic_copy_checkpoint(source: Path, target: Path, expected_sha256: str) -> None:
    """Publish a complete checkpoint on the shared volume before task metadata."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source.stat().st_size or file_digest(temporary) != expected_sha256:
            raise IOError(f"Durable checkpoint copy verification failed: {source} -> {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_checkpoint_task(job: dict, *, step: int, final: bool, source: Path,
                            durable_root: Path, study_id: str, plan_id: str,
                            online_encoder: bool) -> dict:
    """Verify, durably copy, then atomically publish one evaluation task."""
    checkpoint = torch.load(source, map_location="meta", weights_only=False)
    _checkpoint_identity(checkpoint, job, step, online_encoder=online_encoder)
    stat = source.stat()
    sha256 = file_digest(source)
    validation_metrics = checkpoint.get("validation_metrics")
    validation_metrics_sha256 = digest(validation_metrics) if isinstance(validation_metrics, dict) else None
    dataset = _study_component(job["dataset"])
    exp = _study_component(job["experiment"])
    seed = int(job["training_seed"])
    label = "final" if final else f"probe_step_{step:08d}"
    plan_namespace = _study_component(plan_id[:12])
    task_id = f"{study_id}/{plan_namespace}/{dataset}/{exp}/seed_{seed:03d}/{label}"
    checkpoint_rel = Path("evaluation_checkpoints") / study_id / plan_namespace / dataset / exp / f"seed_{seed:03d}" / f"{label}.pt"
    checkpoint_dst = durable_root / checkpoint_rel
    existing_task = durable_root / "evaluation_tasks" / study_id / plan_namespace / dataset / exp / f"seed_{seed:03d}" / f"{label}.task.json"
    if existing_task.is_file():
        try:
            old = json.loads(existing_task.read_text(encoding="utf-8"))
            if old.get("task_id") == task_id and old.get("checkpoint", {}).get("sha256") == sha256:
                if checkpoint_dst.is_file() and checkpoint_dst.stat().st_size == stat.st_size \
                        and file_digest(checkpoint_dst) == sha256:
                    return old
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    _atomic_copy_checkpoint(source, checkpoint_dst, sha256)
    task = {"schema": 1, "status": "ready", "task_id": task_id,
            "created_utc": datetime.now(timezone.utc).isoformat(), "study_id": study_id,
            "plan_id": plan_id, "dataset": job["dataset"], "experiment": job["experiment"],
            "backbone": job["backbone"], "training_seed": job["training_seed"],
            "checkpoint_tag": label, "is_final": final, "optimizer_step": step,
            "protocol_id": job["protocol_id"], "run_signature": job["run_signature"],
            "feature_mode": "online" if online_encoder else "cached",
            "checkpoint": {"path": checkpoint_rel.as_posix(), "size_bytes": stat.st_size,
                           "sha256": sha256, "optimizer_step": step},
            "validation_metrics": validation_metrics,
            "validation_metrics_sha256": validation_metrics_sha256,
            "evaluation": checkpoint_eval_spec(job, step, final),
            "source_assets": job["cache_identity"]["source_assets"],
            "trajectory_path": ((Path("evaluation_trajectories") / study_id / plan_namespace
                                 / dataset / exp / f"seed_{seed:03d}" / f"{label}.npz").as_posix()
                                if job["dataset"] == "aloha" and final else None),
            "result_path": (Path("evaluation_results") / study_id / plan_namespace / dataset / exp
                            / f"seed_{seed:03d}" / f"{label}.json").as_posix()}
    write_json_atomic(existing_task, task)
    log_event("study.evaluation_task_published", task_id=task_id, checkpoint=str(checkpoint_dst),
              sha256=sha256, optimizer_step=step, feature_mode=task["feature_mode"])
    return task


def sync_file(src: Path, dst: Path) -> bool:
    """Copy one changed file and check that its complete size reached storage."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (dst.is_file() and dst.stat().st_size == src.stat().st_size
                and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns):
            return False
    except OSError:
        pass
    # RunPod Global Volume is object-backed and rejects timestamp preservation.
    shutil.copyfile(src, dst)
    if dst.stat().st_size != src.stat().st_size:
        raise IOError(f"Durable copy size mismatch: {src} -> {dst}")
    return True


def sync_durable(source: Path, destination: Path, *, include_logs: bool = True) -> int:
    """Copy changed files to durable storage; optionally defer growing logs until stage end."""
    if not source.exists():
        return 0
    return sum(sync_file(src, destination / src.relative_to(source))
               for src in source.rglob("*") if src.is_file()
               and (include_logs or "logs" not in src.relative_to(source).parts))


def log_tail(path: Path, lines: int = 40) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 256 * 1024))
            tail = stream.read().decode("utf-8", errors="replace")
        return tail.splitlines()[-lines:]
    except OSError:
        return []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_update_matched_study.json")
    ap.add_argument("--run-root", type=Path,
                    help="Local run directory (defaults to a dataset-specific directory for one-dataset workers)")
    ap.add_argument("--datasets", nargs="+", choices=("aloha", "libero"), default=["aloha", "libero"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--train-only", action="store_true",
                    help="Train and publish complete checkpoint tasks; leave evaluation to the evaluator Pod")
    ap.add_argument("--online-encoder", action="store_true",
                    help="Encode raw images on the training Pod instead of building/validating an embedding cache")
    ap.add_argument("--durable-root", type=Path,
                    default=Path(os.environ.get("ACTIONOMNI_DURABLE_ROOT", "/workspace-global/action-omni")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.workers < 0 or len(set(args.datasets)) != len(args.datasets):
        ap.error("Workers must be nonnegative and datasets must be distinct")
    os.chdir(ROOT)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    default_run_root = ROOT / "asset/runs/runpod_update_matched_v1"
    if args.run_root is None:
        run_root = (default_run_root.with_name(default_run_root.name + f"_{args.datasets[0]}")
                    if len(args.datasets) == 1 else default_run_root).resolve()
    else:
        run_root = args.run_root.resolve()
    durable_root = args.durable_root.resolve()
    jobs = build_jobs(plan, run_root, args.datasets, args.workers)
    if args.online_encoder:
        for job in jobs:
            job["train"].append("--online-encoder")
    if args.dry_run:
        print(json.dumps({"mode": "dry-run; no files written", "jobs": [
            {k: v for k, v in job.items() if k not in {"protocol", "cache_config", "cache_identity"}}
            for job in jobs]}, indent=2))
        return
    if not torch_cuda_available():
        raise RuntimeError("CUDA is unavailable; no RunPod training stage was started")
    global_mount = Path("/workspace-global")
    if not global_mount.is_dir():
        raise RuntimeError(f"RunPod Global Volume is not mounted at {global_mount}")
    durable_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    code_paths = ["scripts/runpod_update_matched_study.py", "scripts/runpod_eval_worker.py",
                  "scripts/run_log.py", "scripts/online_encoder.py",
                  "scripts/prepare_runpod_study.py", "scripts/prepare_protocol.py",
                  "scripts/train.py",
                  "scripts/evaluate.py", "scripts/eval_libero.py", "scripts/precompute.py",
                  "scripts/analysis/paired_results.py", "scripts/analysis/runpod_paper_results.py",
                  "scripts/plot_training.py",
                  "scripts/plot_study_losses.py",
                  "scripts/run_experiment.py", "experiment_protocol.py", "models/vla_train.py",
                  "models/flow_matching.py", "models/vla.py", "models/backbones.py",
                  "data/aloha/dataset.py", "data/libero/dataset.py", "data/libero/frames.py",
                  "data/hdf5_embeddings.py"]
    contract = {"plan": plan, "plan_id": digest(plan),
                "feature_mode": "online" if args.online_encoder else "cached",
                "train_only": args.train_only, "datasets": args.datasets,
                "training_seeds": plan["training_seeds"],
                "jobs": [{k: job[k] for k in ("dataset", "experiment", "backbone", "training_seed",
                                               "protocol_id", "run_signature")}
                         for job in jobs],
                "workflow_code": {p: file_digest(ROOT / p) for p in code_paths}}
    lock = run_root / "locked_study.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != contract:
        raise ValueError("Locked RunPod plan/code changed; use a new run root to avoid mixing results")
    if not lock.exists():
        write_json(lock, contract)

    status_path = run_root / "study_status.json"
    state = {"plan_id": contract["plan_id"],
             "feature_mode": "online" if args.online_encoder else "cached",
             "train_only": args.train_only, "status": "running", "started_utc": stamp(),
             "training_seeds": plan["training_seeds"], "datasets": args.datasets,
             "completed_policies": [], "total_policies": len(jobs)}
    study_id = _study_component(run_root.name)
    log_event("study.started", run_root=str(run_root), durable_root=str(durable_root),
              datasets=args.datasets, policies=len(jobs), optimizer_updates=plan["optimizer_updates"],
              effective_batch_size=plan["effective_batch_size"], train_only=args.train_only,
              feature_mode="online" if args.online_encoder else "cached")

    def save() -> None:
        state["updated_utc"] = stamp()
        write_json(status_path, state)

    def sync_results(*, include_logs: bool = False) -> None:
        if durable_root.exists():
            started = time.perf_counter()
            copied = sync_durable(run_root, durable_root / "runs" / run_root.name,
                                  include_logs=include_logs)
            sync_file(args.plan.resolve(), durable_root / "configs" / args.plan.name)
            if copied:
                log_event("study.durable_sync", copied_files=copied,
                          elapsed_s=round(time.perf_counter() - started, 3))

    def execute(stage: str, command: list[str], job: dict) -> None:
        state.update(stage=stage, current_policy={k: job[k] for k in
                     ("dataset", "experiment", "training_seed", "run_signature")})
        logs = run_root / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{stamp()}_{stage}_{job['dataset']}_{job['experiment']}_seed{job['training_seed']:03d}.log"
        state["log"] = str(log)
        stage_started = time.perf_counter()
        state.update(stage_started_utc=stamp(), stage_elapsed_s=0,
                     stage_log_tail=[], latest_event=None, latest_progress=None)
        save()
        sync_results()
        log_event("study.stage_started", stage=stage, log=str(log),
                  dataset=job.get("dataset"), experiment=job.get("experiment"),
                  training_seed=job.get("training_seed"), command=command)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   MUJOCO_GL="glfw" if job["dataset"] == "aloha" else "egl")
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            read_offset = 0
            event_buffer = ""
            last_durable_sync = stage_started
            checkpoint_observations: dict[str, tuple[int, int, int]] = {}
            checkpoint_attempts: dict[str, tuple[int, int]] = {}

            def publish_training_tasks(*, force: bool = False) -> None:
                if not args.train_only or stage != "training":
                    return
                for step, final, checkpoint_path in checkpoint_candidates(job, plan["optimizer_updates"]):
                    key = str(checkpoint_path)
                    try:
                        stat = checkpoint_path.stat()
                    except OSError:
                        continue
                    previous = checkpoint_observations.get(key)
                    same_count = (previous[2] + 1 if previous and
                                  previous[:2] == (stat.st_size, stat.st_mtime_ns) else 1)
                    checkpoint_observations[key] = (stat.st_size, stat.st_mtime_ns, same_count)
                    if not force and same_count < 2:
                        continue
                    stat_key = (stat.st_size, stat.st_mtime_ns)
                    if not force and checkpoint_attempts.get(key) == stat_key:
                        continue
                    checkpoint_attempts[key] = stat_key
                    try:
                        publish_checkpoint_task(job, step=step, final=final, source=checkpoint_path,
                                                durable_root=durable_root, study_id=study_id,
                                                plan_id=contract["plan_id"],
                                                online_encoder=args.online_encoder)
                    except Exception as exc:
                        if force:
                            raise
                        log_event("study.checkpoint_not_ready", level="WARNING",
                                  path=str(checkpoint_path), optimizer_step=step,
                                  error_type=type(exc).__name__, error=str(exc))

            def capture_events(text: str, *, final: bool = False) -> None:
                nonlocal event_buffer
                combined = event_buffer + text
                lines = combined.splitlines()
                if not final and combined and not combined.endswith(("\n", "\r")):
                    event_buffer = lines.pop() if lines else combined
                else:
                    event_buffer = ""
                for line in lines:
                    try:
                        child_event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "event" in child_event:
                        state["latest_event"] = child_event
                        if any(key in child_event for key in
                               ("completed_samples", "optimizer_step", "epoch", "validation_loss")):
                            state["latest_progress"] = child_event

            while process.poll() is None:
                time.sleep(15)
                stream.flush()
                try:
                    with log.open("r", encoding="utf-8", errors="replace") as reader:
                        reader.seek(read_offset)
                        new_text = reader.read()
                        read_offset = reader.tell()
                    capture_events(new_text)
                except OSError:
                    pass
                state["stage_elapsed_s"] = round(time.perf_counter() - stage_started, 1)
                state["stage_log_tail"] = log_tail(log, 8)
                save()
                publish_training_tasks()
                if time.perf_counter() - last_durable_sync >= 60:
                    sync_results()
                    last_durable_sync = time.perf_counter()
            try:
                with log.open("r", encoding="utf-8", errors="replace") as reader:
                    reader.seek(read_offset)
                    capture_events(reader.read(), final=True)
            except OSError:
                pass
            if process.returncode == 0:
                publish_training_tasks(force=True)
            if process.returncode:
                state.update(stage_elapsed_s=round(time.perf_counter() - stage_started, 1),
                             stage_log_tail=log_tail(log, 80), child_return_code=process.returncode)
                save()
                log_event("study.stage_failed", level="ERROR", stage=stage,
                          return_code=process.returncode, log=str(log),
                          last_output=state["stage_log_tail"][-12:])
                sync_results(include_logs=True)
                raise subprocess.CalledProcessError(process.returncode, command)
        state.update(stage_elapsed_s=round(time.perf_counter() - stage_started, 1),
                     stage_log_tail=log_tail(log, 8))
        save()
        log_event("study.stage_complete", stage=stage, elapsed_s=state["stage_elapsed_s"],
                  log=str(log), dataset=job.get("dataset"), experiment=job.get("experiment"),
                  training_seed=job.get("training_seed"))
        sync_results(include_logs=True)

    try:
        save()
        cached: set[str] = set()
        for job in jobs:
            state["current_policy"] = {k: job[k] for k in
                                       ("dataset", "experiment", "training_seed", "run_signature")}
            save()
            cache = Path(job["cache"])
            if not args.online_encoder and str(cache) not in cached:
                cache.parent.mkdir(parents=True, exist_ok=True)
                state.update(stage="cache_validation", cache=str(cache))
                save()
                cache_started = time.perf_counter()
                log_event("study.cache_validation_started", cache=str(cache),
                          dataset=job["dataset"], backbone=job["backbone"])
                try:
                    report, reused = ensure_cache(cache, job["cache_config"], job["protocol"],
                                                  job["cache_identity"]["source_assets"],
                        lambda partial, j=job: execute("precompute", [sys.executable, "-u",
                            str(ROOT / "scripts/precompute.py"), "--dataset", j["dataset"],
                            "--exp", j["experiment"], "--split-manifest",
                            str(ROOT / plan["datasets"][j["dataset"]]["split_manifest"]),
                            "--output", str(partial)], j))
                except BaseException:
                    state.update(stage="cache_validation",
                                 stage_elapsed_s=round(time.perf_counter() - cache_started, 1))
                    save()
                    raise
                state.update(stage="cache_validation",
                             stage_elapsed_s=round(time.perf_counter() - cache_started, 1))
                save()
                write_json(cache.parent / "cache_validation.json", {**report, "reused": reused})
                log_event("study.cache_validation_complete", cache=str(cache), reused=reused,
                          elapsed_s=round(time.perf_counter() - cache_started, 3),
                          samples=report.get("frames"), status=report.get("status"))
                cached.add(str(cache))
            run = Path(job["run"])
            manifest_path = run / "run_manifest.json"
            if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "complete":
                execute("training", job["train"], job)
            verify_policy(job, plan["optimizer_updates"])
            if args.train_only:
                for step, final, checkpoint_path in checkpoint_candidates(job, plan["optimizer_updates"]):
                    publish_checkpoint_task(job, step=step, final=final, source=checkpoint_path,
                                            durable_root=durable_root, study_id=study_id,
                                            plan_id=contract["plan_id"],
                                            online_encoder=args.online_encoder)
            else:
                if not (run / "training_loss.png").is_file():
                    execute("plot", [sys.executable, str(ROOT / "scripts/plot_training.py"),
                                      str(run / "history.jsonl")], job)
                for step, final, result, command in evaluation_jobs(job, plan["optimizer_updates"]):
                    if not result.exists():
                        execute("evaluation", command, job)
            state["completed_policies"].append({"dataset": job["dataset"],
                                                "experiment": job["experiment"],
                                                "training_seed": job["training_seed"],
                                                "run_signature": job["run_signature"]})
            save()
            sync_results()

        if args.train_only:
            state.update(status="training_complete", stage="training_complete", completed_utc=stamp())
        elif len(args.datasets) == 1:
            state.update(status="worker_complete", stage="worker_complete", completed_utc=stamp())
        else:
            # Pair final rollout starts within each training seed, then summarize
            # training-seed variation separately from rollout uncertainty.
            pairs = []
            for dataset in ("aloha", "libero"):
                cfg_groot = get_config(dataset, plan["backbones"]["groot"])
                cfg_qwen = get_config(dataset, plan["backbones"]["qwen3vl"])
                for seed in plan["training_seeds"]:
                    groot = Path(run_root / dataset / Path(cfg_groot.output_dir).name
                                 / f"seed_{seed:03d}" / "evaluation_final.json")
                    qwen = Path(run_root / dataset / Path(cfg_qwen.output_dir).name
                                / f"seed_{seed:03d}" / "evaluation_final.json")
                    pairs.append({"label": dataset.upper(),
                                  "left": groot.relative_to(run_root).as_posix(),
                                  "right": qwen.relative_to(run_root).as_posix(),
                                  "vary": "backbone"})
            write_json(run_root / "paired_final_manifest.json", {
                "family_size": len(pairs), "pairs": pairs,
                "labels": {"ALOHA": {"left": "GR00T N1.7", "right": "Qwen3-VL-2B"},
                           "LIBERO": {"left": "GR00T N1.7", "right": "Qwen3-VL-2B"}}})
            state.update(stage="final_analysis", current_policy=None)
            save()
            execute("paired_analysis", [sys.executable, "-u",
                str(ROOT / "scripts/analysis/paired_results.py"), "--manifest",
                str(run_root / "paired_final_manifest.json"), "--output",
                str(run_root / "paired_final.json"), "--report-dir", str(run_root / "reports")], jobs[0])
            execute("loss_analysis", [sys.executable, "-u", str(ROOT / "scripts/plot_study_losses.py"),
                "--plan", str(args.plan.resolve()), "--run-root", str(run_root)], jobs[0])
            state.update(status="complete", stage="complete", completed_utc=stamp())
        # A single-dataset worker is ready to be merged by the finalizer;
        # the durable mount preserves all per-policy checkpoints and reports.
        sync_results(include_logs=True)
    except BaseException as exc:
        failed_stage = state.get("stage")
        state.update(status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}",
                     failed_stage=failed_stage, error_type=type(exc).__name__,
                     traceback=traceback.format_exc(),
                     failed_utc=stamp())
        log_event("study.failed", level="ERROR", error_type=type(exc).__name__, error=str(exc),
                  current_stage=failed_stage, log=state.get("log"),
                  last_output=state.get("stage_log_tail", [])[-12:])
        raise
    finally:
        save()
        sync_results(include_logs=True)


def torch_cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
