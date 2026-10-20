"""Evaluate complete checkpoints published by RunPod training-only workers.

The worker discovers only atomic ``*.task.json`` manifests on the shared volume.
It stages the exact model assets named by each task, verifies the checkpoint,
and dispatches ALOHA and LIBERO evaluation through their separate Python envs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from experiment_protocol import digest, file_digest
from scripts.run_log import log_event, write_json_atomic
from scripts.success_queue import validate_rollout_result

_VERIFIED_ASSETS: dict[tuple[str, int, int, str], bool] = {}
_SEEN_MANIFEST_ERRORS: set[str] = set()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative(root: Path, relative: str) -> Path:
    """Resolve a manifest path while rejecting absolute paths and traversal."""
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe relative path in evaluation task: {relative!r}")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Evaluation path escaped its root: {relative!r}")
    return resolved


def discover_tasks(durable_root: Path) -> list[tuple[Path, dict]]:
    task_root = durable_root / "evaluation_tasks"
    if not task_root.is_dir():
        return []
    tasks = []
    for path in sorted(task_root.rglob("*.task.json")):
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
            if task.get("schema") != 1 or task.get("status") != "ready":
                raise ValueError("unsupported schema or task not ready")
            if not task.get("task_id"):
                raise ValueError("task id is missing")
            tasks.append((path, task))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if path.as_posix() not in _SEEN_MANIFEST_ERRORS:
                log_event("eval.task_manifest_invalid", level="ERROR", path=str(path), error=str(exc))
                _SEEN_MANIFEST_ERRORS.add(path.as_posix())
    def task_order(entry):
        task = entry[1]
        purpose = task.get("purpose")
        priority = (0 if purpose == "development_selection" else
                    2 if purpose == "final_test" else int(task.get("queue_priority", 1)))
        return priority, str(task.get("created_utc", "")), task.get("task_id", "")

    return sorted(tasks, key=task_order)


def validate_task_checkpoint(task: dict, durable_root: Path) -> tuple[Path, dict]:
    if task.get("dataset") not in {"aloha", "libero"}:
        raise ValueError(f"Unsupported task dataset: {task.get('dataset')!r}")
    expected_keys = ("dataset", "experiment", "backbone", "training_seed", "protocol_id",
                     "run_signature", "feature_mode", "checkpoint_tag", "optimizer_step")
    missing = [key for key in expected_keys if key not in task]
    if missing:
        raise ValueError(f"Task metadata lacks fields: {missing}")
    if task["feature_mode"] not in {"cached", "online"}:
        raise ValueError(f"Invalid feature mode: {task['feature_mode']!r}")
    checkpoint_meta = task.get("checkpoint") or {}
    checkpoint_path = safe_relative(durable_root, checkpoint_meta.get("path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Published checkpoint is missing: {checkpoint_path}")
    if checkpoint_path.stat().st_size != checkpoint_meta.get("size_bytes"):
        raise ValueError("Published checkpoint size differs from task metadata")
    actual_sha = file_digest(checkpoint_path)
    if actual_sha != checkpoint_meta.get("sha256"):
        raise ValueError("Published checkpoint digest differs from task metadata")
    checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    expected = {"dataset": task["dataset"], "exp_id": task["experiment"],
                "training_seed": task["training_seed"], "protocol_id": task["protocol_id"],
                "run_signature": task["run_signature"]}
    actual_step = checkpoint.get("optimizer_step", checkpoint.get("global_step"))
    mismatches = {key: (checkpoint.get(key), value) for key, value in expected.items()
                  if checkpoint.get(key) != value}
    if actual_step != task["optimizer_step"] or actual_step != checkpoint_meta.get("optimizer_step"):
        mismatches["optimizer_step"] = (actual_step, task["optimizer_step"])
    if checkpoint.get("global_step", actual_step) != task["optimizer_step"]:
        mismatches["global_step"] = (checkpoint.get("global_step"), task["optimizer_step"])
    cfg = checkpoint.get("cfg")
    if cfg is None or getattr(cfg, "vlm_backbone", None) != task["backbone"]:
        mismatches["backbone"] = (getattr(cfg, "vlm_backbone", None), task["backbone"])
    if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
        mismatches["state_dict"] = ("missing or empty", "nonempty mapping")
    if checkpoint.get("feature_mode") != task["feature_mode"]:
        mismatches["feature_mode"] = (checkpoint.get("feature_mode"), task["feature_mode"])
    validation_metrics = checkpoint.get("validation_metrics")
    metrics_sha = digest(validation_metrics) if isinstance(validation_metrics, dict) else None
    if "validation_metrics" not in task and "validation_metrics_sha256" not in task:
        task["validation_metrics"] = validation_metrics
        task["validation_metrics_sha256"] = metrics_sha
    if validation_metrics != task.get("validation_metrics"):
        mismatches["validation_metrics"] = ("checkpoint/task values differ", "identical values")
    if metrics_sha != task.get("validation_metrics_sha256"):
        mismatches["validation_metrics_sha256"] = (metrics_sha, task.get("validation_metrics_sha256"))
    if validation_metrics is not None and (
            validation_metrics.get("protocol_id") != task["protocol_id"]
            or validation_metrics.get("split") != "validation_demonstrations"):
        mismatches["validation_metrics_scope"] = (
            validation_metrics.get("protocol_id"), validation_metrics.get("split"))
    protocol = checkpoint.get("protocol")
    if (not isinstance(protocol, dict)
            or digest({key: value for key, value in protocol.items() if key != "protocol_id"})
            != protocol.get("protocol_id")
            or protocol.get("protocol_id") != task["protocol_id"]):
        mismatches["protocol"] = (protocol.get("protocol_id") if isinstance(protocol, dict) else None,
                                   task["protocol_id"])
    provenance = checkpoint.get("encoder_provenance") or {}
    if provenance.get("source_assets") != task.get("source_assets"):
        mismatches["source_assets"] = ("checkpoint provenance differs", "task provenance")
    if mismatches:
        raise ValueError(f"Checkpoint/task identity mismatch: {mismatches}")
    if task["feature_mode"] == "online":
        if not provenance.get("online_encoder_code_sha256"):
            raise ValueError("Online checkpoint is missing online encoder provenance")
        if provenance.get("source_assets") != task.get("source_assets"):
            raise ValueError("Online checkpoint source assets differ from the evaluation task")
    return checkpoint_path, checkpoint


def stage_model_assets(task: dict, durable_root: Path) -> None:
    assets = task.get("source_assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Task has no locked model asset manifest")
    copied = 0
    for key, expected_sha in assets.items():
        source_key = PurePosixPath(key)
        if source_key.parts[:2] != ("asset", "models"):
            # Evaluation never needs the training dataset or embedding cache.
            raise ValueError(f"Task references a non-model runtime asset: {key}")
        source_rel = PurePosixPath(*source_key.parts[1:]).as_posix()
        source = safe_relative(durable_root / "assets", source_rel)
        if not source.is_file():
            raise ValueError(f"Shared model asset is missing or changed: {source}")
        source_stat = source.stat()
        source_key = (str(source), source_stat.st_size, source_stat.st_mtime_ns, expected_sha)
        if source_key not in _VERIFIED_ASSETS:
            if file_digest(source) != expected_sha:
                raise ValueError(f"Shared model asset is missing or changed: {source}")
            _VERIFIED_ASSETS[source_key] = True
        destination = safe_relative(ROOT, key)
        if destination.is_file():
            dest_stat = destination.stat()
            dest_key = (str(destination), dest_stat.st_size, dest_stat.st_mtime_ns, expected_sha)
            if dest_key in _VERIFIED_ASSETS or file_digest(destination) == expected_sha:
                _VERIFIED_ASSETS[dest_key] = True
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if file_digest(temporary) != expected_sha:
                raise IOError(f"Model asset copy failed verification: {key}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        dest_stat = destination.stat()
        _VERIFIED_ASSETS[(str(destination), dest_stat.st_size, dest_stat.st_mtime_ns, expected_sha)] = True
        copied += 1
    log_event("eval.model_assets_ready", task_id=task["task_id"], files_copied=copied,
              files_verified=len(assets))


def verified_file_matches(path: Path, expected_sha256: str) -> bool:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, expected_sha256)
    if key in _VERIFIED_ASSETS:
        return True
    if file_digest(path) != expected_sha256:
        return False
    _VERIFIED_ASSETS[key] = True
    return True


def local_result_matches(result_path: Path, task: dict, *, published: bool = False) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = {"dataset": task["dataset"], "exp_id": task["experiment"],
                "training_seed": task["training_seed"], "protocol_id": task["protocol_id"],
                "run_signature": task["run_signature"],
                "optimizer_step": task["optimizer_step"],
                "checkpoint_sha256": task["checkpoint"]["sha256"]}
    if any(result.get(key) != value for key, value in expected.items()):
        return False
    configuration = result.get("configuration") or {}
    if configuration.get("vlm_backbone") != task["backbone"]:
        return False
    if published and any(result.get(key) != value for key, value in {
            "task_id": task["task_id"], "study_id": task["study_id"],
            "experiment": task["experiment"], "purpose": task.get("purpose"),
            "is_final": task.get("is_final", False),
            "backbone": task["backbone"], "checkpoint_tag": task["checkpoint_tag"],
            "selection_epoch": task.get("selection_epoch"),
            "selected_checkpoint_sha256": task.get("selected_checkpoint_sha256"),
            "queue_priority": task.get("queue_priority"),
            "feature_mode": task["feature_mode"], "plan_id": task["plan_id"],
            "validation_metrics": task.get("validation_metrics"),
            "validation_metrics_sha256": task.get("validation_metrics_sha256")}.items()):
        return False
    evaluation = task["evaluation"]
    if task["dataset"] == "aloha":
        return (len(result.get("episodes", [])) == evaluation["episodes"]
                and result.get("seed_offset") == evaluation["seed_offset"]
                and result.get("render_samples") == evaluation["render_samples"]
                and [row.get("initial_state_id") for row in result.get("episodes", [])]
                == list(range(evaluation["seed_offset"],
                              evaluation["seed_offset"] + evaluation["episodes"])))
    rows = result.get("results", [])
    observed_tasks = {row.get("task") for row in rows}
    observed_conditions = {row.get("condition") for row in rows}
    if not (result.get("episodes") == evaluation["episodes_per_task"]
            and set(evaluation["tasks"]) == observed_tasks
            and set(evaluation["conditions"]) == observed_conditions
            and len(rows) == len(evaluation["tasks"]) * len(evaluation["conditions"])
            * evaluation["episodes_per_task"]):
        return False
    if "initial_state_ids" in evaluation:
        expected_ids = set(evaluation["initial_state_ids"])
        return all({row.get("initial_state_id") for row in rows
                    if row.get("task") == task_id and row.get("condition") == "canonical"} == expected_ids
                   for task_id in evaluation["tasks"])
    return True


def published_result_matches(durable_root: Path, task: dict) -> bool:
    result_path = durable_result_path(durable_root, task)
    if not local_result_matches(result_path, task, published=True):
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if task.get("trajectory_path"):
            pointer = result.get("trajectory") or {}
            trajectory_path = safe_relative(durable_root, task["trajectory_path"])
            if (pointer.get("path") != task["trajectory_path"] or not trajectory_path.is_file()
                    or trajectory_path.stat().st_size != pointer.get("size_bytes")
                    or not verified_file_matches(trajectory_path, pointer.get("sha256"))):
                return False
        elif result.get("trajectory") is not None:
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def validate_trajectory(path: Path, task: dict, result: dict) -> None:
    expected_count = task["evaluation"]["episodes"]
    with np.load(path, allow_pickle=False) as archive:
        count = int(archive["n_episodes"][0])
        if count != expected_count:
            raise ValueError(f"Trajectory episode count {count} does not match {expected_count}")
        episode_ids = [int(value) for value in archive["initial_state_ids"].tolist()]
        expected_ids = [int(row["initial_state_id"]) for row in result["episodes"]]
        if episode_ids != expected_ids:
            raise ValueError("Trajectory initial-state IDs differ from the evaluation result")
        for index in range(count):
            prefix = f"ep{index:04d}_"
            if not all(prefix + key in archive for key in ("state", "next_state", "action", "reward")):
                raise ValueError(f"Trajectory episode {index} is missing state/action/reward arrays")
        if any(key.endswith("attn_image_mass") for key in archive.files):
            raise ValueError("Trajectory-only capture unexpectedly contains attention weights")


def copy_file_atomic(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        size, sha256 = temporary.stat().st_size, file_digest(temporary)
        if size != source.stat().st_size or sha256 != file_digest(source):
            raise IOError(f"Durable copy verification failed: {source} -> {destination}")
        os.replace(temporary, destination)
        stat = destination.stat()
        _VERIFIED_ASSETS[(str(destination), stat.st_size, stat.st_mtime_ns, sha256)] = True
        return size, sha256
    finally:
        if temporary.exists():
            temporary.unlink()


def task_status_path(durable_root: Path, task: dict) -> Path:
    task_hash = hashlib.sha256(task["task_id"].encode("utf-8")).hexdigest()
    return durable_root / "evaluation_status" / f"{task_hash}.json"


def durable_result_path(durable_root: Path, task: dict) -> Path:
    return safe_relative(durable_root, task["result_path"])


def libero_initial_state_offset(evaluation: dict) -> int | None:
    """Resolve old offset-only and newer explicit-cohort task specifications."""
    if "initial_state_offset" not in evaluation and "initial_state_ids" not in evaluation:
        return None
    offset = evaluation.get("initial_state_offset")
    if offset is None:
        initial_ids = evaluation.get("initial_state_ids")
        if not isinstance(initial_ids, list) or not initial_ids:
            raise ValueError("LIBERO task has neither initial_state_offset nor initial_state_ids")
        offset = initial_ids[0]
    if type(offset) is not int or offset < 0:
        raise ValueError("LIBERO initial state offset must be a nonnegative integer")
    return offset


def retry_delay(status_path: Path, base_seconds: float, max_seconds: float) -> tuple[float, int]:
    """Return remaining bounded exponential backoff for a failed task."""
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "failed":
            return 0.0, 0
        attempt = max(1, int(status.get("attempt", 1)))
        failed = datetime.fromisoformat(status["updated_utc"].replace("Z", "+00:00"))
        delay = min(base_seconds * (2 ** min(attempt - 1, 20)), max_seconds)
        age = max(0.0, (datetime.now(timezone.utc) - failed).total_seconds())
        return max(0.0, delay - age), attempt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return 0.0, 0


def evaluate_one(task_file: Path, task: dict, durable_root: Path, work_root: Path,
                 aloha_python: str, libero_python: str, aloha_gl: str = "glfw") -> bool:
    status_path = task_status_path(durable_root, task)
    if published_result_matches(durable_root, task):
        write_json_atomic(status_path, {"task_id": task["task_id"], "status": "complete",
                                        "updated_utc": stamp(), "result": task["result_path"],
                                        "skipped": True})
        log_event("eval.task_skipped", task_id=task["task_id"], reason="matching result already published")
        return True

    started = time.perf_counter()
    try:
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))
        attempt = int(previous_status.get("attempt", 0)) + 1
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        attempt = 1
    write_json_atomic(status_path, {"task_id": task["task_id"], "status": "validating",
                                    "updated_utc": stamp(), "attempt": attempt,
                                    "task_file": str(task_file)})
    log_event("eval.task_started", task_id=task["task_id"], dataset=task["dataset"],
              experiment=task["experiment"], backbone=task["backbone"],
              training_seed=task["training_seed"], checkpoint_tag=task["checkpoint_tag"],
              optimizer_step=task["optimizer_step"])
    try:
        checkpoint_path, checkpoint = validate_task_checkpoint(task, durable_root)
        stage_model_assets(task, durable_root)
        local_root = work_root / hashlib.sha256(task["task_id"].encode("utf-8")).hexdigest()
        run_dir = local_root / "run"
        local_checkpoint = run_dir / "checkpoints" / checkpoint_path.name
        run_dir.mkdir(parents=True, exist_ok=True)
        local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not local_checkpoint.is_file() or file_digest(local_checkpoint) != task["checkpoint"]["sha256"]:
            shutil.copyfile(checkpoint_path, local_checkpoint)
        if file_digest(local_checkpoint) != task["checkpoint"]["sha256"]:
            raise IOError("Local evaluation checkpoint failed digest verification")
        local_result = local_root / "evaluation.json"
        local_trajectory = local_root / "trajectory.npz"
        if local_result.exists():
            local_result.unlink()
        if local_trajectory.exists():
            local_trajectory.unlink()
        if task["dataset"] == "aloha":
            evaluation = task["evaluation"]
            command = [aloha_python, "-u", str(ROOT / "scripts/evaluate.py"),
                       "--dataset", "aloha", "--exp", task["experiment"],
                       "--checkpoint", str(local_checkpoint),
                       "--episodes", str(evaluation["episodes"]),
                       "--seed-offset", str(evaluation["seed_offset"]),
                       "--render-samples", str(evaluation["render_samples"]),
                       "--no-video", "--output", str(local_result)]
            if evaluation.get("capture_trajectory"):
                command.extend(["--trajectory", str(local_trajectory)])
            gl = aloha_gl
        else:
            evaluation = task["evaluation"]
            command = [libero_python, "-u", str(ROOT / "scripts/eval_libero.py"),
                       "--exp", task["experiment"], "--checkpoint", local_checkpoint.name,
                       "--episodes", str(evaluation["episodes_per_task"]),
                       "--conditions", *map(str, evaluation["conditions"]),
                       "--tasks", *map(str, evaluation["tasks"]),
                       "--videos", str(evaluation["videos"]),
                       "--seed", str(evaluation["environment_seed"]),
                       "--output-dir", str(run_dir), "--output", str(local_result)]
            initial_offset = libero_initial_state_offset(evaluation)
            if initial_offset is not None:
                command.extend(["--initial-state-offset", str(initial_offset)])
            gl = "egl"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", MUJOCO_GL=gl)
        if (task["dataset"] == "aloha" and gl == "glfw"
                and sys.platform.startswith("linux") and not env.get("DISPLAY")):
            xvfb = shutil.which("xvfb-run")
            if xvfb is None:
                raise RuntimeError("Headless ALOHA needs xvfb-run; run scripts/setup_runpod_eval.sh")
            command = [xvfb, "-a", *command]
        log_path = local_root / "evaluation.log"
        write_json_atomic(status_path, {"task_id": task["task_id"], "status": "evaluating",
                                        "updated_utc": stamp(), "log": str(log_path),
                                        "command": command})
        log_event("eval.subprocess_started", task_id=task["task_id"], python=command[0],
                  simulator=task["dataset"], log=str(log_path))
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if not local_result_matches(local_result, task):
            raise ValueError("Evaluator output failed checkpoint identity or settings validation")
        result = json.loads(local_result.read_text(encoding="utf-8"))
        trajectory_pointer = None
        if task.get("trajectory_path"):
            if not local_trajectory.is_file():
                raise FileNotFoundError("Final ALOHA evaluator did not create its requested trajectory")
            validate_trajectory(local_trajectory, task, result)
            trajectory_dst = safe_relative(durable_root, task["trajectory_path"])
            trajectory_size, trajectory_sha = copy_file_atomic(local_trajectory, trajectory_dst)
            trajectory_pointer = {"path": task["trajectory_path"],
                                  "sha256": trajectory_sha, "size_bytes": trajectory_size}
        result.update({"task_id": task["task_id"], "study_id": task["study_id"],
                       "experiment": task["experiment"], "purpose": task.get("purpose"),
                       "is_final": task.get("is_final", False),
                       "backbone": task["backbone"], "checkpoint_tag": task["checkpoint_tag"],
                       "feature_mode": task["feature_mode"], "plan_id": task["plan_id"],
                       "selection_epoch": task.get("selection_epoch"),
                       "selected_checkpoint_sha256": task.get("selected_checkpoint_sha256"),
                       "queue_priority": task.get("queue_priority"),
                       "encoder_provenance": checkpoint.get("encoder_provenance"),
                       "validation_metrics": task.get("validation_metrics"),
                       "validation_metrics_sha256": task.get("validation_metrics_sha256"),
                       "trajectory": trajectory_pointer})
        if task.get("purpose") in {"development_selection", "final_test"}:
            if task["dataset"] == "aloha":
                outcomes = [row["is_success"] for row in result["episodes"]]
                result["success_rate"] = sum(outcomes) / len(outcomes)
            else:
                per_task = []
                for task_id in task["evaluation"]["tasks"]:
                    outcomes = [row["success"] for row in result["results"]
                                if row["task"] == task_id and row["condition"] == "canonical"]
                    per_task.append(sum(outcomes) / len(outcomes))
                result["success_rate"] = sum(per_task) / len(per_task)
            result = validate_rollout_result(task, result)
        result_rel = task["result_path"]
        result_dst = durable_result_path(durable_root, task)
        write_json_atomic(local_result, result)
        result_dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_dst.with_name(result_dst.name + f".{os.getpid()}.tmp")
        try:
            shutil.copyfile(local_result, temporary)
            os.replace(temporary, result_dst)
        finally:
            if temporary.exists():
                temporary.unlink()
        if not local_result_matches(result_dst, task, published=True):
            raise IOError("Durable evaluation result failed post-copy identity validation")
        if task.get("trajectory_path") and not published_result_matches(durable_root, task):
            raise IOError("Durable final trajectory/result pointer failed post-copy validation")
        write_json_atomic(status_path, {"task_id": task["task_id"], "status": "complete",
                                        "updated_utc": stamp(), "result": result_rel,
                                        "elapsed_s": round(time.perf_counter() - started, 3),
                                        "log": str(log_path)})
        log_event("eval.task_complete", task_id=task["task_id"], result=str(result_dst),
                  elapsed_s=round(time.perf_counter() - started, 3))
        return True
    except Exception as exc:
        write_json_atomic(status_path, {"task_id": task["task_id"], "status": "failed",
                                        "updated_utc": stamp(), "attempt": attempt,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc), "traceback": traceback.format_exc()})
        log_event("eval.task_failed", level="ERROR", task_id=task["task_id"],
                  attempt=attempt, error_type=type(exc).__name__, error=str(exc),
                  elapsed_s=round(time.perf_counter() - started, 3))
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable-root", type=Path,
                        default=Path(os.environ.get("ACTIONOMNI_DURABLE_ROOT", "/workspace-global/action-omni")))
    parser.add_argument("--work-root", type=Path, default=ROOT / "asset/eval_worker")
    parser.add_argument("--aloha-python", default=sys.executable,
                        help="Python executable with gym-aloha simulator dependencies")
    parser.add_argument("--libero-python", default=sys.executable,
                        help="Python executable with LIBERO simulator dependencies")
    parser.add_argument("--aloha-gl", choices=("egl", "glfw"), default="glfw",
                        help="MuJoCo rendering backend for ALOHA; EGL is headless, GLFW may use xvfb-run")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--retry-base-seconds", type=float, default=300,
                        help="Delay before retrying failed tasks; doubles per attempt")
    parser.add_argument("--retry-max-seconds", type=float, default=3600,
                        help="Maximum failed-task retry delay")
    parser.add_argument("--once", action="store_true", help="Process one task-discovery sweep and exit")
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or args.retry_base_seconds < 1 \
            or args.retry_max_seconds < args.retry_base_seconds:
        parser.error("Polling/retry delays must be positive and retry maximum >= retry base")
    durable_root = args.durable_root.resolve()
    work_root = args.work_root.resolve()
    if not durable_root.is_dir():
        parser.error(f"Durable root is not mounted or does not exist: {durable_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    log_event("eval.worker_started", durable_root=str(durable_root), work_root=str(work_root),
              aloha_python=args.aloha_python, libero_python=args.libero_python,
              aloha_gl=args.aloha_gl,
              once=args.once, poll_seconds=args.poll_seconds,
              retry_base_seconds=args.retry_base_seconds,
              retry_max_seconds=args.retry_max_seconds)
    seen_manifest_errors: set[str] = set()
    deferred_attempts: dict[str, int] = {}
    completed_task_ids: set[str] = set()
    total_ok = total_failed = 0
    idle_logged = False
    while True:
        task_entries = discover_tasks(durable_root)
        dispatched = False
        for task_file, task in task_entries:
            if task_file.as_posix() in seen_manifest_errors:
                continue
            if task["task_id"] in completed_task_ids:
                continue
            status_path = task_status_path(durable_root, task)
            try:
                previous_status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                previous_status = {}
            if previous_status.get("status") == "complete":
                if published_result_matches(durable_root, task):
                    completed_task_ids.add(task["task_id"])
                    continue
                log_event("eval.completed_artifact_invalid", level="WARNING",
                          task_id=task["task_id"], reason="published result/artifact no longer verifies")
            delay, attempt = retry_delay(task_status_path(durable_root, task),
                                         args.retry_base_seconds, args.retry_max_seconds)
            if delay > 0:
                if deferred_attempts.get(task["task_id"]) != attempt:
                    log_event("eval.task_retry_deferred", task_id=task["task_id"],
                              failed_attempt=attempt, retry_in_s=round(delay, 1))
                    deferred_attempts[task["task_id"]] = attempt
                continue
            try:
                ok = evaluate_one(task_file, task, durable_root, work_root,
                                  args.aloha_python, args.libero_python, args.aloha_gl)
            except Exception as exc:
                seen_manifest_errors.add(task_file.as_posix())
                log_event("eval.task_dispatch_failed", level="ERROR", path=str(task_file),
                          task_id=task.get("task_id"), error_type=type(exc).__name__, error=str(exc))
                ok = False
            if ok:
                total_ok += 1
                if published_result_matches(durable_root, task):
                    completed_task_ids.add(task["task_id"])
            else:
                total_failed += 1
            dispatched = True
            # Re-discover after each evaluation. A newly published development
            # rollout then preempts queued final tasks at the next dispatch point.
            break
        if args.once:
            if not dispatched:
                log_event("eval.worker_sweep_complete", completed=total_ok, failed=total_failed)
                return 1 if total_failed else 0
        if dispatched:
            if idle_logged:
                log_event("eval.queue_resumed", pending=len(task_entries))
            idle_logged = False
            continue
        if not idle_logged:
            log_event("eval.queue_idle", poll_seconds=args.poll_seconds,
                      discovered_tasks=len(task_entries), completed_tasks=len(completed_task_ids))
            idle_logged = True
        if args.once:
            log_event("eval.worker_sweep_complete", completed=total_ok, failed=total_failed)
            return 1 if total_failed else 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
