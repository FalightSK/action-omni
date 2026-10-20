"""Durable, hash-verified rollout queue shared by training and evaluator Pods."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from typing import Mapping

from experiment_protocol import digest, file_digest, sampling_seed
from scripts.run_log import log_event, write_json_atomic


def _safe_component(value: object) -> str:
    text = str(value)
    safe = "".join(char if char.isalnum() or char in "-_ ." else "_" for char in text).replace(" ", "_")
    if safe in {"", ".", ".."} or "/" in safe or "\\" in safe:
        raise ValueError(f"Unsafe rollout task path component: {text!r}")
    return safe


def _relative_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe durable rollout path: {value!r}")
    resolved = (root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Rollout path escaped durable root: {value!r}")
    return resolved


def _checkpoint_dict(path: Path) -> dict:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Selection checkpoint does not exist: {path}")
    value = torch.load(path, map_location="meta", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("Selection checkpoint must be a dictionary")
    return value


def _identity(cfg, metadata: Mapping, checkpoint: dict) -> dict:
    checkpoint_cfg = checkpoint.get("cfg")
    cfg_backbone = getattr(cfg, "vlm_backbone", None)
    checkpoint_backbone = getattr(checkpoint_cfg, "vlm_backbone", None)
    identity = {
        "dataset": metadata.get("dataset", checkpoint.get("dataset")),
        "experiment": metadata.get("experiment", metadata.get("exp_id", checkpoint.get("exp_id"))),
        "backbone": metadata.get("backbone", cfg_backbone or checkpoint_backbone),
        "training_seed": metadata.get("training_seed", checkpoint.get("training_seed")),
        "protocol_id": metadata.get("protocol_id", checkpoint.get("protocol_id")),
        "run_signature": metadata.get("run_signature", checkpoint.get("run_signature")),
        "feature_mode": metadata.get("feature_mode", checkpoint.get("feature_mode", "cached")),
        "optimizer_step": metadata.get("optimizer_step",
                                         checkpoint.get("optimizer_step", checkpoint.get("global_step"))),
    }
    if not identity["experiment"]:
        identity["experiment"] = checkpoint.get("experiment")
    if not identity["backbone"]:
        identity["backbone"] = checkpoint_backbone
    for key, ckpt_key in (("dataset", "dataset"), ("training_seed", "training_seed"),
                          ("protocol_id", "protocol_id"), ("run_signature", "run_signature")):
        actual = checkpoint.get(ckpt_key)
        if identity[key] != actual:
            raise ValueError(f"Selection checkpoint {key} mismatch: {actual!r} != {identity[key]!r}")
    actual_experiment = checkpoint.get("exp_id", checkpoint.get("experiment"))
    if identity["experiment"] != actual_experiment:
        raise ValueError(f"Selection checkpoint experiment mismatch: {actual_experiment!r} != {identity['experiment']!r}")
    if not identity["backbone"] or checkpoint_backbone != identity["backbone"] \
            or (cfg_backbone and cfg_backbone != identity["backbone"]):
        raise ValueError("Selection checkpoint/config backbone identity mismatch")
    actual_step = checkpoint.get("optimizer_step", checkpoint.get("global_step"))
    if actual_step != identity["optimizer_step"]:
        raise ValueError(f"Selection checkpoint optimizer step mismatch: {actual_step!r} != {identity['optimizer_step']!r}")
    if identity["feature_mode"] not in {"cached", "online"} \
            or checkpoint.get("feature_mode", "cached") != identity["feature_mode"]:
        raise ValueError("Selection checkpoint feature mode differs from metadata")
    if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
        raise ValueError("Selection checkpoint has no complete model state_dict")
    protocol = checkpoint.get("protocol")
    if (not isinstance(protocol, dict)
            or protocol.get("protocol_id") != identity["protocol_id"]
            or digest({key: value for key, value in protocol.items() if key != "protocol_id"})
            != protocol.get("protocol_id")):
        raise ValueError("Selection checkpoint protocol payload is missing or does not verify")
    provenance = checkpoint.get("encoder_provenance") or {}
    source_assets = metadata.get("source_assets", provenance.get("source_assets"))
    if not isinstance(source_assets, dict) or not source_assets:
        raise ValueError("Rollout task requires a nonempty locked source_assets manifest")
    if provenance.get("source_assets") != source_assets:
        raise ValueError("Selection checkpoint source_assets differ from rollout metadata")
    return {**identity, "source_assets": source_assets}


def _checkpoint_tag(metadata: Mapping, purpose: str) -> str:
    if purpose == "final_test":
        return "final_test"
    if purpose != "development_selection":
        raise ValueError(f"Unsupported rollout purpose: {purpose!r}")
    tag = metadata.get("checkpoint_tag")
    if tag is None and metadata.get("selection_epoch") is not None:
        tag = f"selection_epoch_{int(metadata['selection_epoch']):06d}"
    if not isinstance(tag, str) or not re.fullmatch(r"selection_epoch_\d{6}", tag):
        raise ValueError("Development task needs checkpoint_tag='selection_epoch_NNNNNN'")
    return tag


def _validate_evaluation(evaluation: Mapping, dataset: str, purpose: str) -> dict:
    spec = dict(evaluation)
    if dataset == "aloha":
        episodes = spec.get("episodes")
        if type(episodes) is not int or episodes < 1 or type(spec.get("seed_offset")) is not int:
            raise ValueError("ALOHA rollout spec requires positive episodes and integer seed_offset")
        if purpose == "final_test" and spec.get("capture_trajectory") is not True:
            raise ValueError("Final ALOHA evaluation must capture a trajectory")
    elif dataset == "libero":
        tasks = spec.get("tasks")
        initial_ids = spec.get("initial_state_ids")
        episodes = spec.get("episodes_per_task")
        if not isinstance(tasks, list) or not tasks or type(episodes) is not int or episodes < 1:
            raise ValueError("LIBERO rollout spec requires tasks and episodes_per_task")
        if initial_ids is None:
            offset = spec.get("initial_state_offset")
            if type(offset) is not int or offset < 0:
                raise ValueError("LIBERO rollout spec requires initial_state_ids or nonnegative initial_state_offset")
            initial_ids = list(range(offset, offset + episodes))
            spec["initial_state_ids"] = initial_ids
        if not isinstance(initial_ids, list) or not initial_ids:
            raise ValueError("LIBERO rollout initial_state_ids must be a nonempty list")
        if any(type(value) is not int or value < 0 for value in initial_ids):
            raise ValueError("LIBERO rollout initial_state_ids must contain nonnegative integers")
        if len(initial_ids) != episodes:
            raise ValueError("LIBERO episodes_per_task must equal the explicit initial-state cohort size")
        if len(set(initial_ids)) != len(initial_ids):
            raise ValueError("LIBERO rollout initial_state_ids must be unique")
        if purpose == "development_selection" and any(not 0 <= value <= 9 for value in initial_ids):
            raise ValueError("LIBERO development rollouts must use initial-state IDs 0-9")
        if purpose == "final_test" and any(not 10 <= value <= 49 for value in initial_ids):
            raise ValueError("LIBERO final tests must use disjoint initial-state IDs 10-49")
    else:
        raise ValueError(f"Unsupported rollout dataset: {dataset!r}")
    return spec


def _publish_checkpoint(source: Path, destination: Path, expected_sha256: str) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{time.time_ns()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source.stat().st_size or file_digest(temporary) != expected_sha256:
            raise IOError(f"Durable selection checkpoint copy failed verification: {source}")
        os.replace(temporary, destination)
        return destination.stat().st_size
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_rollout_task(cfg, metadata: Mapping, checkpoint_path: str | Path,
                         evaluation: Mapping, purpose: str, plan_id: str,
                         study_id: str, durable_root: str | Path) -> dict:
    """Verify and atomically publish one dev-selection or selected-final rollout task.

    Re-publishing an identical task is idempotent. Reusing a task identity for a
    different checkpoint or settings is rejected to protect resume and provenance.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping of trainer/run provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", str(plan_id)):
        raise ValueError("plan_id must be the full lowercase SHA-256 digest of the locked plan")
    study = _safe_component(study_id)
    checkpoint_path = Path(checkpoint_path).resolve()
    durable_root = Path(durable_root).resolve()
    durable_root.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_dict(checkpoint_path)
    identity = _identity(cfg, metadata, checkpoint)
    tag = _checkpoint_tag(metadata, purpose)
    evaluation_spec = _validate_evaluation(evaluation, identity["dataset"], purpose)
    checkpoint_sha256 = file_digest(checkpoint_path)
    source_stat = checkpoint_path.stat()
    dataset = _safe_component(identity["dataset"])
    experiment = _safe_component(identity["experiment"])
    plan_namespace = plan_id[:12]
    seed = int(identity["training_seed"])
    task_id = f"{study}/{plan_namespace}/{dataset}/{experiment}/seed_{seed:03d}/{tag}"
    checkpoint_rel = (Path("evaluation_checkpoints") / study / plan_namespace / dataset / experiment
                      / f"seed_{seed:03d}" / f"{tag}.pt").as_posix()
    result_rel = (Path("evaluation_results") / study / plan_namespace / dataset / experiment
                  / f"seed_{seed:03d}" / f"{tag}.json").as_posix()
    trajectory_rel = ((Path("evaluation_trajectories") / study / plan_namespace / dataset / experiment
                       / f"seed_{seed:03d}" / f"{tag}.npz").as_posix()
                      if identity["dataset"] == "aloha" and purpose == "final_test" else None)
    task_rel = (Path("evaluation_tasks") / study / plan_namespace / dataset / experiment
                / f"seed_{seed:03d}" / f"{tag}.task.json").as_posix()
    task_path = _relative_path(durable_root, task_rel)
    checkpoint_dst = _relative_path(durable_root, checkpoint_rel)
    validation_metrics = checkpoint.get("validation_metrics")
    metrics_sha256 = digest(validation_metrics) if isinstance(validation_metrics, dict) else None
    task = {
        "schema": 1, "status": "ready", "task_id": task_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": study, "plan_id": plan_id,
        "dataset": identity["dataset"], "experiment": identity["experiment"],
        "backbone": identity["backbone"], "training_seed": seed,
        "checkpoint_tag": tag, "purpose": purpose,
        "is_final": purpose == "final_test",
        "queue_priority": 0 if purpose == "development_selection" else 1,
        "optimizer_step": identity["optimizer_step"], "protocol_id": identity["protocol_id"],
        "run_signature": identity["run_signature"], "feature_mode": identity["feature_mode"],
        "checkpoint": {"path": checkpoint_rel, "size_bytes": source_stat.st_size,
                       "sha256": checkpoint_sha256, "optimizer_step": identity["optimizer_step"]},
        "validation_metrics": validation_metrics,
        "validation_metrics_sha256": metrics_sha256,
        "source_assets": identity["source_assets"],
        "evaluation": evaluation_spec, "trajectory_path": trajectory_rel,
        "result_path": result_rel,
        "selection_epoch": metadata.get("selection_epoch"),
        "selected_checkpoint_sha256": metadata.get("selected_checkpoint_sha256"),
    }
    if purpose == "final_test" and task["selected_checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Final test requires the controller-selected checkpoint SHA to match the published file")
    if purpose == "final_test" and task["selection_epoch"] is None:
        raise ValueError("Final test task must record the selected development epoch")

    if task_path.is_file():
        old = json.loads(task_path.read_text(encoding="utf-8"))
        identity_fields = ("task_id", "plan_id", "dataset", "experiment", "backbone", "training_seed",
                           "checkpoint_tag", "purpose", "is_final", "optimizer_step", "protocol_id",
                           "run_signature", "feature_mode", "checkpoint", "validation_metrics_sha256",
                           "source_assets", "evaluation", "trajectory_path", "result_path",
                           "selection_epoch", "selected_checkpoint_sha256")
        if any(old.get(key) != task.get(key) for key in identity_fields):
            raise ValueError(f"Rollout task identity already exists with different checkpoint/settings: {task_id}")
        if (checkpoint_dst.is_file() and checkpoint_dst.stat().st_size == source_stat.st_size
                and file_digest(checkpoint_dst) == checkpoint_sha256):
            return old
        raise IOError(f"Existing rollout task has missing/corrupt durable checkpoint: {task_id}")

    _publish_checkpoint(checkpoint_path, checkpoint_dst, checkpoint_sha256)
    write_json_atomic(task_path, task)
    log_event("success_rollout.task_published", task_id=task_id, purpose=purpose,
              dataset=identity["dataset"], experiment=identity["experiment"],
              training_seed=seed, optimizer_step=identity["optimizer_step"],
              checkpoint_sha256=checkpoint_sha256, queue_priority=task["queue_priority"])
    return task


def _result_path(task: Mapping, durable_root: Path) -> Path:
    return _relative_path(durable_root, task["result_path"])


def validate_rollout_result(task: Mapping, result: Mapping) -> dict:
    """Require complete result identity and exact rollout denominator before unblocking train."""
    expected = {
        "task_id": task["task_id"], "study_id": task["study_id"], "plan_id": task["plan_id"],
        "dataset": task["dataset"], "exp_id": task["experiment"], "experiment": task["experiment"],
        "backbone": task["backbone"], "training_seed": task["training_seed"],
        "checkpoint_tag": task["checkpoint_tag"], "purpose": task["purpose"],
        "is_final": task["is_final"], "optimizer_step": task["optimizer_step"],
        "protocol_id": task["protocol_id"], "run_signature": task["run_signature"],
        "feature_mode": task["feature_mode"],
        "checkpoint_sha256": task["checkpoint"]["sha256"],
        "validation_metrics_sha256": task.get("validation_metrics_sha256"),
        "selection_epoch": task.get("selection_epoch"),
        "selected_checkpoint_sha256": task.get("selected_checkpoint_sha256"),
    }
    mismatches = {key: (result.get(key), value) for key, value in expected.items()
                  if result.get(key) != value}
    if mismatches:
        raise ValueError(f"Rollout result provenance mismatch: {mismatches}")
    if result.get("validation_metrics") != task.get("validation_metrics"):
        raise ValueError("Rollout validation metrics differ from checkpoint task metadata")
    provenance = result.get("encoder_provenance") or {}
    if task["feature_mode"] == "online":
        if (not provenance.get("encoder_code_id") or not provenance.get("online_encoder_code_sha256")
                or provenance.get("source_assets") != task.get("source_assets")):
            raise ValueError("Online rollout lacks matching frozen-encoder/source-assets provenance")
    spec = task["evaluation"]
    if task["dataset"] == "aloha":
        records = result.get("episodes")
        expected_n = spec["episodes"]
        outcomes = [row.get("is_success") for row in records] if isinstance(records, list) else []
        if len(outcomes) != expected_n or any(type(value) is not bool for value in outcomes):
            raise ValueError("ALOHA rollout result has incomplete/nonboolean outcomes")
        if result.get("seed_offset") != spec["seed_offset"]:
            raise ValueError("ALOHA result seed offset differs from task")
        if [row.get("initial_state_id") for row in records] != list(
                range(spec["seed_offset"], spec["seed_offset"] + expected_n)):
            raise ValueError("ALOHA initial-state IDs differ from the locked scene cohort")
        if any(row.get("initial_state_kind") != "gym_aloha_mujoco_state_v1"
               or not _valid_sha256(row.get("initial_state_sha256"))
               or row.get("environment_seed") != row["initial_state_id"] * 137 + 42
               or row.get("policy_sampling_seed") != sampling_seed(
                   "aloha", task["training_seed"], 0, row["initial_state_id"])
               for row in records):
            raise ValueError("ALOHA result lacks valid real simulator-state fingerprints")
        computed_rate = sum(outcomes) / expected_n
    else:
        records = result.get("results")
        rows = records if isinstance(records, list) else []
        episodes = spec["episodes_per_task"]
        tasks = spec["tasks"]
        conditions = spec.get("conditions", ["canonical"])
        if len(rows) != len(tasks) * episodes or any(
                row.get("condition") != "canonical" for row in rows):
            raise ValueError("LIBERO result has extra/missing records or noncanonical conditions")
        counts = {int(index): [] for index in tasks}
        for row in rows:
            if row.get("condition") == "canonical" and row.get("task") in counts \
                    and type(row.get("success")) is bool:
                counts[row["task"]].append(row["success"])
        if any(len(counts[index]) != episodes for index in tasks):
            raise ValueError("LIBERO rollout result has incomplete canonical task/start outcomes")
        if set(conditions) != {"canonical"}:
            raise ValueError("Success-selection result must contain the canonical condition only")
        initial_ids = spec["initial_state_ids"]
        expected_ids = set(initial_ids)
        for index in tasks:
            task_rows = [row for row in rows
                         if row.get("task") == index and row.get("condition") == "canonical"]
            observed = {row.get("initial_state_id") for row in task_rows}
            if observed != expected_ids:
                raise ValueError(f"LIBERO initial-state IDs differ for task {index}")
            if any(row.get("initial_state_kind") != "libero_mujoco_state_after_init_and_settle_v1"
                   or not _valid_sha256(row.get("initial_state_sha256"))
                   or row.get("environment_seed") != spec["environment_seed"] + row["initial_state_id"]
                   or row.get("policy_sampling_seed") != sampling_seed(
                       "libero", task["training_seed"], index, row["initial_state_id"])
                   for row in task_rows):
                raise ValueError(f"LIBERO task {index} lacks valid real simulator-state fingerprints")
        computed_rate = sum(sum(values) / len(values) for values in counts.values()) / len(counts)
    if not isinstance(result.get("success_rate"), (int, float)) \
            or abs(float(result["success_rate"]) - computed_rate) > 1e-12:
        raise ValueError("Rollout result success_rate does not match its complete episode records")
    result_dict = dict(result)
    result_dict["success_rate"] = float(computed_rate)
    return result_dict


def _valid_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value.lower()))


def wait_for_rollout(task: Mapping, durable_root: str | Path,
                     timeout_s: float, poll_s: float) -> dict:
    """Block training until the evaluator publishes this exact complete rollout."""
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("timeout_s and poll_s must be positive")
    durable_root = Path(durable_root).resolve()
    result_path = _result_path(task, durable_root)
    status_hash = hashlib.sha256(task["task_id"].encode("utf-8")).hexdigest()
    status_path = durable_root / "evaluation_status" / f"{status_hash}.json"
    started = time.monotonic()
    last_status = None
    while time.monotonic() - started < timeout_s:
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                validated = validate_rollout_result(task, result)
                if task.get("trajectory_path"):
                    pointer = result.get("trajectory") or {}
                    trace = _relative_path(durable_root, task["trajectory_path"])
                    if (pointer.get("path") != task["trajectory_path"] or not trace.is_file()
                            or trace.stat().st_size != pointer.get("size_bytes")
                            or file_digest(trace) != pointer.get("sha256")):
                        raise ValueError("Final ALOHA rollout trajectory pointer/file failed verification")
                log_event("success_rollout.result_ready", task_id=task["task_id"],
                          purpose=task["purpose"], success_rate=validated["success_rate"])
                return validated
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                # The atomic result may be absent or not yet complete while the worker is publishing it.
                last_status = f"result_pending_or_invalid: {type(exc).__name__}: {exc}"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                last_status = f"{status.get('status')}: {status.get('error', '')}"
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(min(poll_s, max(0.0, timeout_s - (time.monotonic() - started))))
    raise TimeoutError(f"Timed out waiting for rollout {task['task_id']}; latest status: {last_status}")
