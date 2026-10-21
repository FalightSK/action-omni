#!/usr/bin/env python3
"""One-off v4 online -> v6 cached-feature continuation; no new training run.

The v4 checkpoint/history/lock remain read-only. Model, optimizer, scheduler,
RNG and loader-generator state are copied exactly; only documented run/cache
identity fields change. Cache and parent/controller provenance are audited.
"""
from __future__ import annotations
import argparse
import ast
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
from datetime import datetime
import torch

OLD_ARCHIVE_SHA256 = "1dfa9ac490150f5ac8240b354fccae122bcfce5e44ac1a9031bc5e9403b9b8c1"
PLAN_ID = "fcd23d9252310bc0586f8ba069f4858a0d6b43675fdbd5776dc98a78a5a144b3"
NEW_PLAN_ID = "a34bb34e8023bd4d532c62789da3471ae7f9f2d68dfcc1ee22221d80e8ba7907"
OLD_STUDY_PREFIX = "runpod_success_stopped_v4_"
NEW_STUDY_PREFIX = "runpod_success_stopped_v6_"
REVIEWED_CACHED_SOURCE_BATCHES = {
    "3bebc75e3fe90c5848024518ce19b19e4f2e1f2a65b9310eff995174399b6819": 48,
    "37974c552993fe6fae68ebe412a2fa30857ab7fc0bbfc4964983787a3bb3a508": 96,
}
ALLOWED_SOURCE_CHANGES = {
    "configs/runpod_cached_success_study.json",
    "scripts/prepare_cached_study.py",
    "scripts/runpod_success_study.py", "scripts/train.py",
    "scripts/runpod_bootstrap.py", "tests/test_runpod_success_study.py",
    # Filled with the focused cache-preparation test once its reviewed filename lands.
    "tests/test_prepare_cached_study.py",
    "tests/test_runpod_bootstrap.py",
    "configs/aloha/exp05_groot_transfer.py",
    "configs/libero/exp05_groot_2view.py",
}
EMBED_BATCH_CONFIG_CHANGES = {
    "aloha": {"path": "configs/aloha/exp05_groot_transfer.py", "old": 32, "new": 64},
    "libero": {"path": "configs/libero/exp05_groot_2view.py", "old": 24, "new": 48},
}
RUN_SIGNATURE_FILES = ("scripts/train.py", "experiment_protocol.py", "models/vla_train.py",
                       "models/flow_matching.py")
ENCODER_FILES = ("models/vla.py", "models/backbones.py", "models/vision_patch_projection.py",
                 "data/aloha/dataset.py", "data/libero/dataset.py", "data/libero/frames.py",
                 "data/hdf5_embeddings.py", "scripts/precompute.py")
REWRITTEN_KEYS = {"rollout_study_id", "rollout_plan_id", "run_signature",
                  "feature_mode", "cache_id", "encoder_provenance"}
IGNORED_CFG = {"output_dir", "embeddings_cache", "device", "embed_batch_size"}


def validate_training_source_delta(relative: str, old_bytes: bytes, new_bytes: bytes) -> dict:
    old_hash = hashlib.sha256(old_bytes).hexdigest()
    new_hash = hashlib.sha256(new_bytes).hexdigest()
    # Cached-mode entrypoint changes were reviewed; encoder/model files stay byte-identical.
    if relative == "scripts/train.py":
        if (old_hash, new_hash) != (
                "68e74ec905b7c77c6998fa8d3bc5842aad5c881fc450255f1c8e7858848622a4",
                "3eb9b31b4054b387c32d5e28c0aaea97afdbadc4475c355eefa760b1c93f0586"):
            raise ValueError(f"Unreviewed cached trainer transition: {relative}")
    elif old_hash != new_hash:
        raise ValueError(f"Training/encoder implementation changed: {relative}")
    return {"old_sha256": old_hash, "new_sha256": new_hash}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def archive_hashes(tf: tarfile.TarFile) -> dict[str, str]:
    result = {}
    for item in tf.getmembers():
        if not item.isfile():
            continue
        name = PurePosixPath(item.name).as_posix()
        if name in result:
            raise ValueError(f"Duplicate archive path: {name}")
        stream = tf.extractfile(item)
        if stream is None:
            raise ValueError(f"Cannot read archive member {name}")
        result[name] = hashlib.sha256(stream.read()).hexdigest()
    return result


def member(tf: tarfile.TarFile, relative: str) -> bytes:
    matches = [m for m in tf.getmembers() if m.isfile()
               and PurePosixPath(m.name).as_posix().endswith("/" + relative)]
    if not matches:
        matches = [m for m in tf.getmembers() if m.isfile()
                   and PurePosixPath(m.name).as_posix() == relative]
    if len(matches) != 1:
        raise ValueError(f"Archive must contain exactly one {relative}; found {len(matches)}")
    stream = tf.extractfile(matches[0])
    if stream is None:
        raise ValueError(f"Cannot read archive member {relative}")
    return stream.read()


def canon(cfg):
    if dataclasses.is_dataclass(cfg):
        value = dataclasses.asdict(cfg)
    elif isinstance(cfg, dict):
        value = dict(cfg)
    else:
        raise TypeError(f"Unsupported checkpoint config {type(cfg).__name__}")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def cfg_identity(cfg):
    value = canon(cfg)
    for key in ("output_dir", "embeddings_cache", "embed_batch_size"):
        value.pop(key, None)
    return value


def cfg_except_cache_identity(cfg):
    value = canon(cfg)
    for key in ("output_dir", "embeddings_cache", "embed_batch_size"):
        value.pop(key, None)
    return value


def validate_embed_batch_config_delta(old_bytes: bytes, new_bytes: bytes,
                                      relative_path: str, old_value: int,
                                      new_value: int) -> dict:
    """Permit exactly one numeric edit to the declared dataclass field line."""
    field = re.compile(
        rb"(?m)^([ \t]*embed_batch_size:[ \t]*int[ \t]*=[ \t]*)"
        + str(old_value).encode("ascii")
        + rb"(?=[ \t]*(?:\#[^\r\n]*)?\r?$)"
    )
    transformed, count = field.subn(lambda m: m.group(1) + str(new_value).encode("ascii"),
                                    old_bytes)
    if count != 1 or transformed != new_bytes:
        raise ValueError(
            f"{relative_path} must differ only by embed_batch_size {old_value}->{new_value}"
        )
    return {"path": relative_path, "old_embed_batch_size": old_value,
            "new_embed_batch_size": new_value,
            "old_config_sha256": hashlib.sha256(old_bytes).hexdigest(),
            "new_config_sha256": hashlib.sha256(new_bytes).hexdigest(),
            "raw_bytes_change": "single embed_batch_size numeric substitution; all other bytes identical"}


def equal(a, b, path="root"):
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())):
            raise ValueError(f"Checkpoint payload changed at {path}")
    elif isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()):
            raise ValueError(f"Checkpoint keys changed at {path}")
        for k in a:
            equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError(f"Checkpoint sequence changed at {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f"{path}[{i}]")
    else:
        try:
            same = a == b
            if hasattr(same, "all"):
                same = bool(same.all())
        except Exception:
            same = False
        if same is not True:
            raise ValueError(f"Checkpoint payload changed at {path}")


def require_unselected(state):
    if (not isinstance(state, dict) or state.get("schema") != 1
            or state.get("last_evaluated_epoch") != 0 or state.get("pending") is not None):
        raise ValueError("Only checkpoints before any development rollout can be continued")
    for key in ("best_success_rate", "best_epoch", "best_checkpoint_path",
                "best_checkpoint_sha256", "last_improvement_epoch", "last_success_rate"):
        if state.get(key) is not None:
            raise ValueError(f"Selection is not empty: {key}")


def exact_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if sha(src) != sha(dst):
        raise IOError(f"Copy hash mismatch: {src} -> {dst}")


def select_verified_checkpoint(last_path: Path):
    """Prefer the newest verified immutable recovery; fall back if mutable last.pt is partial."""
    from scripts.checkpoint_io import latest_verified_recovery
    immutable = latest_verified_recovery(last_path.parent)
    imm_ck = torch.load(immutable, map_location="cpu", weights_only=False) if immutable else None
    try:
        last_ck = torch.load(last_path, map_location="cpu", weights_only=False)
    except Exception:
        if immutable is None:
            raise
        return immutable, imm_ck
    if immutable is None:
        return last_path, last_ck
    def order(ck):
        return (int(ck.get("epoch", -1)), int(ck.get("global_step", -1)),
                int(ck.get("recovery_revision", -1)))
    if order(imm_ck) >= order(last_ck):
        # Same-order divergence means mutable last.pt was interrupted or changed;
        # checkpoint_io has already verified the immutable filename hash.
        return immutable, imm_ck
    return last_path, last_ck


def ensure_local_v6_lock(durable_lock: Path, local_lock: Path,
                         expected_contract: dict) -> None:
    """Restore only a missing local lock from the exact verified durable contract."""
    durable_lock, local_lock = Path(durable_lock), Path(local_lock)
    message = "Existing v6 namespace lacks its exact current lock; refusing partial restart"
    if not durable_lock.is_file():
        raise FileExistsError(message)
    durable_bytes = durable_lock.read_bytes()
    if json.loads(durable_bytes) != expected_contract:
        raise FileExistsError(message)
    if local_lock.exists():
        if json.loads(local_lock.read_bytes()) != expected_contract:
            raise FileExistsError(message)
        return

    local_lock.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_lock.with_name(local_lock.name + ".restore.tmp")
    try:
        temporary.write_bytes(durable_bytes)
        temporary.replace(local_lock)
    finally:
        if temporary.exists():
            temporary.unlink()
    if json.loads(local_lock.read_bytes()) != expected_contract:
        raise IOError("Restored local v6 study lock failed read-back verification")


def cache_manifest_entries(path: Path, dataset: str, study_id: str,
                           plan_id: str, protocol_id: str,
                           experiments: set[str]) -> dict[str, dict]:
    """Load one reviewed sidecar per experiment from the cache root or a sidecar path."""
    root = path if path.is_dir() else path.parent
    sidecars = sorted(root.rglob("*.manifest.json")) if root.is_dir() else []
    if path.is_file() and path.name.endswith(".manifest.json") and path not in sidecars:
        sidecars.append(path)
    indexed = {}
    for sidecar in sidecars:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != 1:
            continue
        exp = manifest.get("experiment")
        if exp not in experiments or manifest.get("dataset") != dataset:
            continue
        for key, expected in (("dataset", dataset), ("study_id", study_id),
                              ("plan_id", plan_id), ("protocol_id", protocol_id)):
            if manifest.get(key) != expected:
                raise ValueError(f"Cache sidecar {sidecar} {key} mismatch")
        if exp in indexed:
            raise ValueError(f"Duplicate cache sidecar for {exp}")
        if (not isinstance(manifest.get("sha256"), str)
                or len(manifest["sha256"]) != 64
                or not isinstance(manifest.get("cache_id"), str)
                or not isinstance(manifest.get("durable_path"), str)
                or not isinstance(manifest.get("local_path"), str)):
            raise ValueError(f"Cache sidecar lacks path, cache ID, or SHA256: {sidecar}")
        manifest["_manifest_path"] = str(sidecar.resolve())
        manifest["_manifest_sha256"] = sha(sidecar)
        indexed[exp] = manifest
    if set(indexed) != experiments:
        raise FileNotFoundError(f"Cache sidecars differ from the two policy arms: {sorted(set(indexed) ^ experiments)}")
    return indexed


def cache_manifest_digest(entries: dict[str, dict]) -> str:
    return digest({key: value["_manifest_sha256"] for key, value in sorted(entries.items())})


def cache_prepare_report_identity(report: dict) -> str:
    """Hash immutable cache evidence, excluding report timing/status fields.

    ``completed_utc``, ``elapsed_s`` and each cache's ``reused`` flag can
    legitimately change when bootstrap reruns preparation against the same
    published cache. Cache identity, paths, byte size and SHA may not change.
    """
    expected_top = {"schema", "study_id", "plan_id", "dataset", "feature_mode",
                    "caches", "completed_utc", "elapsed_s"}
    if not isinstance(report, dict) or set(report) != expected_top:
        raise ValueError("Cache preparation report has unexpected fields")
    if not isinstance(report.get("completed_utc"), str):
        raise ValueError("Cache preparation report lacks a completion timestamp")
    try:
        datetime.fromisoformat(report["completed_utc"])
    except ValueError as exc:
        raise ValueError("Cache preparation report timestamp is invalid") from exc
    elapsed = report.get("elapsed_s")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Cache preparation report duration is invalid")
    if not isinstance(report.get("caches"), list):
        raise ValueError("Cache preparation report caches must be a list")
    normalized = []
    expected_item = {"cache_id", "local_path", "durable_path",
                     "sha256", "size_bytes", "reused"}
    for item in report["caches"]:
        if not isinstance(item, dict) or set(item) != expected_item:
            raise ValueError("Cache preparation report entry has unexpected fields")
        if not isinstance(item.get("reused"), bool):
            raise ValueError("Cache preparation report reused status must be boolean")
        # Reused is an operational outcome, not provenance of the cache bytes.
        normalized.append({key: item[key] for key in sorted(expected_item - {"reused"})})
    normalized.sort(key=lambda item: item["durable_path"])
    return digest({key: report[key] for key in
                   ("schema", "study_id", "plan_id", "dataset", "feature_mode")} |
                  {"caches": normalized})


def validate_cache_prepare_report(path: Path, *, dataset: str, study_id: str,
                                  plan_id: str, experiments: set[str],
                                  cache_evidence: dict[str, dict],
                                  manifest_entries: dict[str, dict]) -> str:
    """Validate report identity and immutable artifact evidence before migration/restart."""
    report = json.loads(path.read_text(encoding="utf-8"))
    identity_sha = cache_prepare_report_identity(report)
    if (report.get("schema") != 1 or report.get("study_id") != study_id
            or report.get("plan_id") != plan_id or report.get("dataset") != dataset
            or report.get("feature_mode") != "cached"):
        raise ValueError("Cached preparation report differs from v6 plan identity")
    report_by_exp = {}
    for item in report["caches"]:
        matches = [exp for exp, manifest in manifest_entries.items()
                   if Path(item["durable_path"]).resolve()
                   == Path(manifest["durable_path"]).resolve()]
        if len(matches) != 1:
            raise ValueError("Cached preparation report path does not identify one policy arm")
        exp = matches[0]
        if exp in report_by_exp:
            raise ValueError("Cached preparation report has duplicate experiment entries")
        report_by_exp[exp] = item
    if set(report_by_exp) != experiments:
        raise ValueError("Cached preparation report does not list both policy arms")
    for exp, evidence in cache_evidence.items():
        item = report_by_exp[exp]
        manifest = manifest_entries[exp]
        cache_path = Path(evidence["path"])
        if (item.get("cache_id") != evidence["cache_id"]
                or item.get("sha256") != evidence["sha256"]
                or int(item.get("size_bytes", -1)) != cache_path.stat().st_size
                or Path(item.get("local_path", "")).resolve()
                   != Path(manifest["local_path"]).resolve()
                or Path(item.get("durable_path", "")).resolve()
                   != Path(manifest["durable_path"]).resolve()):
            raise ValueError(f"Cached preparation report disagrees with verified cache: {exp}")
    return identity_sha


def self_check_cache_prepare_report_identity() -> None:
    """CPU-only regression check: time changes pass; a cache SHA change does not."""
    report = {
        "schema": 1, "study_id": "study", "plan_id": "plan",
        "dataset": "aloha", "feature_mode": "cached",
        "completed_utc": "2026-10-20T12:00:00+00:00", "elapsed_s": 12.5,
        "caches": [{"cache_id": "id",
                    "local_path": "/local/cache.h5", "durable_path": "/cloud/cache.h5",
                    "sha256": "a" * 64, "size_bytes": 123, "reused": False}],
    }
    initial = cache_prepare_report_identity(report)
    rerun = copy.deepcopy(report)
    rerun["completed_utc"] = "2026-10-20T13:00:00+00:00"
    rerun["elapsed_s"] = 1.0
    rerun["caches"][0]["reused"] = True
    if cache_prepare_report_identity(rerun) != initial:
        raise AssertionError("Timing/status-only report changes altered immutable identity")
    altered = copy.deepcopy(rerun)
    altered["caches"][0]["sha256"] = "b" * 64
    if cache_prepare_report_identity(altered) == initial:
        raise AssertionError("Cache SHA change was not reflected in report identity")


def validate_cached_feature(cache_path: Path, entry: dict, job: dict,
                            protocol: dict, source_assets_value: dict,
                            encoder_id: str, cache_config_fn) -> dict:
    """Validate full-cache bytes and semantic provenance without encoding data."""
    if not cache_path.is_file() or sha(cache_path) != entry["sha256"]:
        raise ValueError(f"Cache file missing or SHA256 mismatch: {cache_path}")
    import h5py
    import numpy as np
    from types import SimpleNamespace
    config = SimpleNamespace(**job["config"])
    expected_cfg = cache_config_fn(config)
    with h5py.File(cache_path, "r") as f:
        def jattr(name):
            value = f.attrs.get(name)
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return json.loads(value) if isinstance(value, str) and name in {
                "cache_config", "source_assets"} else value
        attrs = {name: jattr(name) for name in f.attrs}
        if not bool(attrs.get("complete", True)):
            raise ValueError(f"Incomplete embedding cache: {cache_path}")
        if attrs.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError(f"Cache protocol mismatch: {cache_path}")
        if attrs.get("cache_id") != entry.get("cache_id"):
            raise ValueError(f"Cache ID differs from manifest: {cache_path}")
        if attrs.get("encoder_code_id") != encoder_id:
            raise ValueError(f"Cached encoder code ID mismatch: {cache_path}")
        if attrs.get("source_assets") != source_assets_value:
            raise ValueError(f"Cached model/source asset hashes mismatch: {cache_path}")
        if attrs.get("cache_config") != expected_cfg:
            raise ValueError(f"Cached input/normalization/encoder config mismatch: {cache_path}")
        if attrs.get("dataset") != job["dataset"] or attrs.get("exp_id") != job["experiment"]:
            raise ValueError(f"Cache dataset/experiment mismatch: {cache_path}")
        if entry.get("backbone") != job["backbone"]:
            raise ValueError(f"Cache sidecar backbone mismatch: {cache_path}")
        if not bool(attrs.get("full_dataset")) or not bool(attrs.get("embed_is_bf16_bits")):
            raise ValueError(f"Cache must be complete full-dataset bf16-bit features: {cache_path}")
        layers = tuple(int(x) for x in np.atleast_1d(attrs.get("vlm_extract_layers", [])))
        expected_layers = tuple(int(x) for x in config.vlm_extract_layers)
        if layers != expected_layers:
            raise ValueError(f"Cache layer mismatch: {cache_path}")
        for name in ("embeddings", "img_masks", "states", "actions", "episode_ids", "validation_mask"):
            if name not in f:
                raise ValueError(f"Cache dataset missing {name}: {cache_path}")
        n = int(attrs.get("n_samples", f["embeddings"].shape[0]))
        if (n != int(protocol["n_samples"]) or f["states"].shape[0] != n
                or f["actions"].shape[0] != n
                or f["episode_ids"].shape[0] != n
                or f["validation_mask"].shape[0] != n):
            raise ValueError(f"Cache row counts do not match full protocol: {cache_path}")
        layers_count = len(expected_layers)
        expected_embeddings_shape = ((n, layers_count, int(config.img_seq_len),
                                      int(config.vlm_hidden_size)) if layers_count > 1 else
                                     (n, int(config.img_seq_len), int(config.vlm_hidden_size)))
        expected_state_dim = 2 if job["dataset"] == "pusht" else int(config.state_dim)
        if (tuple(f["embeddings"].shape) != expected_embeddings_shape
                or tuple(f["img_masks"].shape) != (n, int(config.img_seq_len))
                or tuple(f["states"].shape) != (n, expected_state_dim)
                or tuple(f["actions"].shape) != (n, int(config.action_horizon), int(config.action_dim))):
            raise ValueError(f"Cache tensor dimensions differ from locked config: {cache_path}")
        ids = np.asarray(f["episode_ids"][:])
        val = np.asarray(f["validation_mask"][:]).astype(bool)
        from experiment_protocol import protocol_arrays
        expected_ids, expected_val = protocol_arrays(protocol)
        if not np.array_equal(ids, expected_ids) or not np.array_equal(val, expected_val):
            raise ValueError(f"Cache sample order or split mask mismatch: {cache_path}")
        cache_id = attrs["cache_id"]
    for key, value in (("protocol_id", protocol["protocol_id"]),
                       ("cache_id", cache_id), ("encoder_code_id", encoder_id),
                       ("source_assets", source_assets_value)):
        if entry.get(key) != value:
            raise ValueError(f"Cache manifest {key} differs from HDF5: {cache_path}")
    return {"path": str(cache_path), "sha256": entry["sha256"], "cache_id": cache_id,
            "encoder_code_id": encoder_id, "source_assets": source_assets_value,
            "cache_config": expected_cfg,
            "manifest_path": entry["_manifest_path"],
            "manifest_sha256": entry["_manifest_sha256"]}


def resolve_cache_path(entry: dict, job: dict, manifest_path: Path,
                       repo: Path) -> Path:
    configured = Path(str(job["config"].get("embeddings_cache", "")))
    if not str(configured):
        raise ValueError(f"Cached job has no embeddings_cache path: {job['experiment']}")
    expected = configured.resolve() if configured.is_absolute() else (repo / configured).resolve()
    candidates = [expected]
    sidecar_parent = Path(entry.get("_manifest_path", manifest_path)).parent
    for key in ("local_path", "durable_path"):
        raw = Path(entry[key])
        candidates.append(raw.resolve() if raw.is_absolute() else (sidecar_parent / raw).resolve())
    found = next((p for p in candidates if p.is_file() and sha(p) == entry["sha256"]), None)
    if found is None or found != expected:
        raise ValueError(f"Cache path differs from cached job config: {found} != {expected}")
    if entry.get("size_bytes") is not None and found.stat().st_size != int(entry["size_bytes"]):
        raise ValueError(f"Cache size differs from sidecar: {found}")
    return found


def old_signature_from_archive(cfg, protocol: dict, seed: int,
                               old_archive: Path | tarfile.TarFile,
                               old_encoder_id: str) -> str:
    fields = canon(cfg)
    for key in IGNORED_CFG:
        fields.pop(key, None)
    if isinstance(old_archive, Path):
        with tarfile.open(old_archive, "r:*") as tf:
            training_code = {p: hashlib.sha256(member(tf, p)).hexdigest()
                             for p in RUN_SIGNATURE_FILES}
    else:
        training_code = {p: hashlib.sha256(member(old_archive, p)).hexdigest()
                         for p in RUN_SIGNATURE_FILES}
    return digest({"cfg": fields, "protocol_id": protocol["protocol_id"],
                   "seed": seed, "training_code": training_code,
                   "encoder_code_id": old_encoder_id})


def runner_workflow_paths(runner_path: Path) -> list[str]:
    """Read the current runner's code_paths literal so the migration lock matches it."""
    tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "code_paths"
                for target in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                return value
    raise ValueError("Could not safely read code_paths from current success-study runner")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check-cache-report", action="store_true",
                    help="Run the CPU-only report identity regression check and exit")
    ap.add_argument("--dataset", required=True, choices=("aloha", "libero"))
    ap.add_argument("--durable-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--old-source-archive", type=Path,
                    default=Path("/workspace-global/action-omni/study_v1/source/runpod-source-online-v9.tar"))
    ap.add_argument("--new-source-archive", type=Path, required=True)
    ap.add_argument("--new-study-id", required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=Path("configs/runpod_cached_success_study.json"))
    raw_args = sys.argv[1:]
    if raw_args == ["--self-check-cache-report"]:
        self_check_cache_prepare_report_identity()
        print(json.dumps({"status": "self_check_passed", "check": "cache_prepare_report_identity"}))
        return 0
    args = ap.parse_args(raw_args)
    repo, durable = args.repo_root.resolve(), args.durable_root.resolve()
    if Path.cwd().resolve() != repo:
        raise ValueError("Run from --repo-root")
    if not args.new_study_id.startswith(NEW_STUDY_PREFIX):
        raise ValueError(f"--new-study-id must use {NEW_STUDY_PREFIX}<role>")
    old_archive, new_archive = args.old_source_archive.resolve(), args.new_source_archive.resolve()
    if sha(old_archive) != OLD_ARCHIVE_SHA256:
        raise ValueError("Old source archive does not match frozen v9")
    new_archive_sha = sha(new_archive)
    if new_archive_sha not in REVIEWED_CACHED_SOURCE_BATCHES:
        raise ValueError("New source archive is not a reviewed cached runtime")
    EMBED_BATCH_CONFIG_CHANGES["libero"]["new"] = REVIEWED_CACHED_SOURCE_BATCHES[new_archive_sha]
    old_study = OLD_STUDY_PREFIX + args.dataset
    old_root = durable / "success_studies" / old_study
    new_root = durable / "success_studies" / args.new_study_id
    old_lock_path = old_root / "locked_success_study.json"
    if not old_lock_path.is_file():
        raise FileNotFoundError(old_lock_path)
    old_contract = json.loads(old_lock_path.read_text(encoding="utf-8"))
    if old_contract.get("dataset_role") != args.dataset:
        raise ValueError("v4 lock dataset role mismatch")
    old_plan = old_contract.get("plan")
    old_plan_id = digest(old_plan)
    if old_plan_id != PLAN_ID or old_contract.get("plan_id") != old_plan_id:
        raise ValueError("v4 contract is not the frozen online scientific plan")
    plan_path = (args.plan if args.plan.is_absolute() else repo / args.plan).resolve()
    current_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_cached_plan = copy.deepcopy(old_plan)
    expected_cached_plan["feature_mode"] = "cached"
    if current_plan != expected_cached_plan or digest(current_plan) != NEW_PLAN_ID:
        raise ValueError("Cached plan changes more than feature_mode=online -> cached")
    plan = current_plan
    plan_id = digest(plan)
    if plan_id != NEW_PLAN_ID:
        raise ValueError("Cached v6 plan is not the reviewer-approved plan")
    with tarfile.open(old_archive, "r:*") as old_tf, tarfile.open(new_archive, "r:*") as new_tf:
        if json.loads(member(old_tf, "configs/runpod_success_stopped_study.json")) != old_plan:
            raise ValueError("v9 archive plan differs from v4 lock")
        if json.loads(member(new_tf, "configs/runpod_success_stopped_study.json")) != old_plan:
            raise ValueError("New archive changed the frozen v4 online plan")
        if json.loads(member(new_tf, "configs/runpod_cached_success_study.json")) != plan:
            raise ValueError("New archive cached plan differs from requested v6 plan")
        old_files, new_files = archive_hashes(old_tf), archive_hashes(new_tf)
        archive_changes = {p for p in set(old_files) | set(new_files)
                           if old_files.get(p) != new_files.get(p)}
        if archive_changes - ALLOWED_SOURCE_CHANGES:
            raise ValueError(f"Only runner/bootstrap/preflight test files may change: {sorted(archive_changes)}")
        batch_config_evidence = {}
        for dataset_name, change in EMBED_BATCH_CONFIG_CHANGES.items():
            relative = change["path"]
            if relative not in archive_changes:
                raise ValueError(f"Expected batch-size config change is absent: {relative}")
            batch_config_evidence[dataset_name] = validate_embed_batch_config_delta(
                member(old_tf, relative), member(new_tf, relative), relative,
                change["old"], change["new"])
        required_cached_files = {"configs/runpod_cached_success_study.json",
                                 "scripts/prepare_cached_study.py",
                                 "scripts/runpod_success_study.py", "scripts/train.py",
                                 "scripts/runpod_bootstrap.py"}
        if required_cached_files - set(new_files):
            raise ValueError(f"New archive lacks required cached-study files: {sorted(required_cached_files - set(new_files))}")
        for rel in archive_changes:
            if new_files.get(rel) is not None and new_files[rel] != sha(repo / rel):
                raise ValueError(f"Reviewed cached source differs from current checkout: {rel}")
        workflow = old_contract.get("workflow_code")
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("v4 lock has no workflow code manifest")
        changed = []
        for rel, expected_hash in workflow.items():
            old_hash = hashlib.sha256(member(old_tf, rel)).hexdigest()
            new_hash = hashlib.sha256(member(new_tf, rel)).hexdigest()
            if old_hash != expected_hash:
                raise ValueError(f"v4 archive/lock code hash mismatch: {rel}")
            if new_hash != sha(repo / rel):
                raise ValueError(f"Current checkout differs from new archive: {rel}")
            if old_hash != new_hash:
                changed.append(rel)
        if set(changed) - ALLOWED_SOURCE_CHANGES:
            raise ValueError(f"Unreviewed workflow source changed: {changed}")
        if not changed:
            raise ValueError("Expected reviewed cached-feature workflow changes are absent")
        training_source_evidence = {
            rel: validate_training_source_delta(rel, member(old_tf, rel), member(new_tf, rel))
            for rel in (*RUN_SIGNATURE_FILES, *ENCODER_FILES)}
    sys.path.insert(0, str(repo))
    from experiment_protocol import cache_config, encoder_code_id, load_protocol
    from scripts.runpod_success_study import build_jobs
    setting = plan["datasets"][args.dataset]
    protocol_path = repo / setting["split_manifest"]
    protocol = load_protocol(protocol_path, args.dataset)
    run_root = repo / "asset/runs" / args.new_study_id
    jobs = build_jobs(plan, args.dataset, run_root, 0, plan_path, durable)
    selected = [j for j in jobs if j["training_seed"] == 3]
    if len(selected) != 2:
        raise ValueError(f"Expected exactly the two seed-3 policy arms, got {len(selected)}")
    cache_manifest_path = args.cache_manifest.resolve()
    expected_experiments = {j["experiment"] for j in jobs if j["training_seed"] == 3}
    manifest_entries = cache_manifest_entries(cache_manifest_path, args.dataset,
                                               args.new_study_id, plan_id,
                                               protocol["protocol_id"], expected_experiments)
    cache_manifest_sha = cache_manifest_digest(manifest_entries)
    current_encoder_id = encoder_code_id()
    cache_evidence = {}
    for job in jobs:
        if job["experiment"] in cache_evidence:
            continue
        cache_entry = manifest_entries[job["experiment"]]
        cache_path = resolve_cache_path(cache_entry, job, cache_manifest_path, repo)
        cache_evidence[job["experiment"]] = validate_cached_feature(
            cache_path, cache_entry, job, protocol, job["source_assets"],
            current_encoder_id, cache_config)
        if (cache_evidence[job["experiment"]]["cache_id"] != job["cache_id"]
                or cache_evidence[job["experiment"]]["sha256"] != job["cache_sha256"]):
            raise ValueError(f"Cache sidecar/HDF5 differs from locked new job: {job['experiment']}")
    workflow_paths = runner_workflow_paths(repo / "scripts/runpod_success_study.py")
    with tarfile.open(new_archive, "r:*") as wf_tf:
        workflow_hashes = {}
        for rel in workflow_paths:
            expected_hash = hashlib.sha256(member(wf_tf, rel)).hexdigest()
            if expected_hash != sha(repo / rel):
                raise ValueError(f"Runner workflow source differs from reviewed archive: {rel}")
            workflow_hashes[rel] = expected_hash
    expected_contract = {"plan_id": plan_id, "plan": plan, "dataset_role": args.dataset,
                         "jobs": [{k: j[k] for k in ("dataset", "experiment", "backbone",
                                                      "training_seed", "protocol_id", "run_signature",
                                                      "feature_mode", "cache_id", "cache_sha256")}
                                  for j in jobs],
                         "workflow_code": workflow_hashes}
    new_lock_exists = (new_root / "locked_success_study.json").exists()
    local_lock_exists = (run_root / "locked_success_study.json").exists()
    cache_report_path = new_root / f"cache_prepare_{args.dataset}.json"
    if not cache_report_path.is_file():
        raise FileNotFoundError(f"Cached preparation completion report missing: {cache_report_path}")
    cache_report_identity_sha = validate_cache_prepare_report(
        cache_report_path, dataset=args.dataset, study_id=args.new_study_id,
        plan_id=plan_id, experiments=expected_experiments,
        cache_evidence=cache_evidence, manifest_entries=manifest_entries)
    if new_lock_exists or local_lock_exists:
        durable_lock = new_root / "locked_success_study.json"
        local_lock = run_root / "locked_success_study.json"
        ensure_local_v6_lock(durable_lock, local_lock, expected_contract)
        for job in selected:
            relative = job["run"].relative_to(run_root)
            old_run = old_root / args.dataset / relative
            old_ckpt = old_run / "checkpoints/last.pt"
            old_history = old_run / "history.jsonl"
            target_run = job["durable_run"]
            target_ckpt, target_history = target_run / "checkpoints/last.pt", target_run / "history.jsonl"
            audit_path = target_run / "migration_audit.json"
            parent_copy = target_run / "parent_controller_mirror.json"
            controller = durable / "controller_state" / plan_id[:12] / args.dataset / job["experiment"] / "seed_003.json"
            old_controller = durable / "controller_state" / old_plan_id[:12] / args.dataset / job["experiment"] / "seed_003.json"
            if not all(p.is_file() for p in (old_history, target_ckpt, target_history,
                                               audit_path, parent_copy, controller)):
                raise FileNotFoundError("v6 restart found incomplete continuation evidence")
            parent_path, parent = select_verified_checkpoint(old_ckpt)
            current_path, current = select_verified_checkpoint(target_ckpt)
            old_job = next((x for x in old_contract["jobs"]
                            if x.get("dataset") == args.dataset
                            and x.get("experiment") == job["experiment"]
                            and x.get("training_seed") == 3), None)
            require_unselected(parent.get("success_stopping_state"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            ctrl_bytes = parent_copy.read_bytes()
            ctrl = json.loads(controller.read_text(encoding="utf-8"))
            cache_meta = cache_evidence[job["experiment"]]
            old_sig = old_signature_from_archive(parent["cfg"], protocol, 3, old_archive,
                                                 parent.get("encoder_provenance", {}).get("encoder_code_id", ""))
            if (audit.get("parent_checkpoint_sha256") != sha(parent_path)
                    or audit.get("old_source_archive_sha256") != OLD_ARCHIVE_SHA256
                    or audit.get("new_source_archive_sha256") != new_archive_sha
                    or audit.get("parent_plan_id") != old_plan_id
                    or audit.get("new_plan_id") != plan_id
                    or audit.get("parent_study_id") != old_study
                    or audit.get("new_study_id") != args.new_study_id
                    or audit.get("cache_manifest_sha256") != cache_manifest_sha
                    or not isinstance(audit.get("cache_prepare_report_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", audit["cache_prepare_report_sha256"]) is None
                    or (audit.get("cache_prepare_report_identity_sha256") is not None
                        and audit.get("cache_prepare_report_identity_sha256")
                            != cache_report_identity_sha)
                    or audit.get("archive_paths_changed") != sorted(archive_changes)
                    or audit.get("embed_batch_size_config_changes") != batch_config_evidence
                    or audit.get("checkpoint_embed_batch_size_change") != {
                        "old": EMBED_BATCH_CONFIG_CHANGES[args.dataset]["old"],
                        "new": EMBED_BATCH_CONFIG_CHANGES[args.dataset]["new"],
                        "source_config": EMBED_BATCH_CONFIG_CHANGES[args.dataset]["path"],
                        "cache_config_and_run_signature_exclude_field": True}
                    or audit.get("cache_sha256") != cache_meta["sha256"]
                    or audit.get("cache_id") != cache_meta["cache_id"]
                    or audit.get("new_run_signature") != job["run_signature"]
                    or audit.get("experiment") != job["experiment"]
                    or audit.get("training_seed") != 3
                    or audit.get("parent_epoch") != parent.get("epoch")
                    or audit.get("continued_checkpoint_sha256", "") == ""
                    or old_job is None or old_job.get("run_signature") != parent.get("run_signature")
                    or old_sig != parent.get("run_signature")
                    or parent.get("rollout_study_id") != old_study
                    or parent.get("rollout_plan_id") != old_plan_id
                    or parent.get("feature_mode") != "online"
                    or current.get("dataset") != args.dataset
                    or current.get("exp_id") != job["experiment"]
                    or current.get("training_seed") != 3
                    or audit.get("parent_history_sha256") != sha(old_history)
                    or audit.get("parent_controller_sha256") != hashlib.sha256(ctrl_bytes).hexdigest()
                    or sha(parent_copy) != hashlib.sha256(ctrl_bytes).hexdigest()
                    or not old_controller.is_file()
                    or old_controller.read_bytes() != ctrl_bytes
                    or not target_history.read_bytes().startswith(old_history.read_bytes())
                    or current.get("run_signature") != job["run_signature"]
                    or current.get("rollout_study_id") != args.new_study_id
                    or current.get("rollout_plan_id") != plan_id
                    or current.get("protocol_id") != job["protocol_id"]
                    or current.get("epoch", -1) < parent.get("epoch", 0)
                    or current.get("global_step", -1) < parent.get("global_step", 0)
                    or cfg_identity(current.get("cfg")) != cfg_identity(job["config"])
                    or getattr(current.get("cfg"), "embed_batch_size", None)
                       != job["config"].get("embed_batch_size")
                    or Path(current.get("cfg").output_dir).resolve() != job["run"].resolve()
                    or current.get("protocol") != job["protocol"]
                    or current.get("feature_mode") != "cached"
                    or current.get("cache_id") != cache_meta["cache_id"]
                    or current.get("encoder_provenance") != {
                        "source_assets": job["source_assets"],
                        "encoder_code_id": current_encoder_id}
                    or str(current.get("cfg").embeddings_cache) != str(job["config"].get("embeddings_cache"))
                    or ctrl.get("study_id") != args.new_study_id
                    or ctrl.get("plan_id") != plan_id
                    or ctrl.get("run_signature") != job["run_signature"]):
                raise ValueError("v6 cached continuation failed parent/audit/checkpoint/controller verification")
            order_ctrl = (int(ctrl.get("last_committed_epoch", -1)),
                          int(ctrl.get("last_committed_recovery_revision", -1)))
            order_ck = (int(current.get("epoch", -1)), int(current.get("recovery_revision", -1)))
            if order_ctrl > order_ck:
                raise ValueError("v6 controller mirror is ahead of the durable checkpoint")
            if order_ctrl == order_ck:
                equal(ctrl.get("success_stopping_state"), current.get("success_stopping_state"),
                      "v6 controller continuation")
            selected_sha256 = sha(current_path)
            print(json.dumps({"event": "v6_restart_checkpoint_selected",
                              "dataset": args.dataset,
                              "experiment": job["experiment"],
                              "path": str(current_path),
                              "sha256": selected_sha256,
                              "epoch": int(current["epoch"]),
                              "global_step": int(current["global_step"]),
                              "recovery_revision": int(current.get("recovery_revision", -1))},
                             sort_keys=True))
        print(json.dumps({"status": "already_continued", "dataset": args.dataset,
                          "new_study_id": args.new_study_id, "plan_id": plan_id}))
        return 0
    if new_root.exists():
        allowed_children = {"feature_caches", f"cache_prepare_{args.dataset}.json"}
        manifest_rel = cache_manifest_path.resolve().relative_to(new_root.resolve()) \
            if cache_manifest_path.resolve().is_relative_to(new_root.resolve()) else None
        if manifest_rel is not None:
            allowed_children.add(manifest_rel.parts[0])
        unexpected = {p.name for p in new_root.iterdir()} - allowed_children
        if unexpected:
            raise FileExistsError(f"Uninitialized v6 root has unexpected contents: {sorted(unexpected)}")
        if not (new_root / "feature_caches").is_dir():
            raise FileNotFoundError("Prepared v6 cache root lacks feature_caches/")
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("Uninitialized local v6 run root contains unexpected files; refusing migration")
    prepared = []
    for job in selected:
        exp = job["experiment"]
        relative = job["run"].relative_to(run_root)
        old_run = old_root / args.dataset / relative
        old_ckpt = old_run / "checkpoints/last.pt"
        old_history = old_run / "history.jsonl"
        # Frozen v4 controller stays read-only; the v6 mirror uses a new plan namespace.
        controller = durable / "controller_state" / old_plan_id[:12] / args.dataset / exp / "seed_003.json"
        if not all(p.is_file() for p in (old_history, controller)):
            raise FileNotFoundError(f"Incomplete frozen v4 continuation inputs for {exp}")
        parent_path, parent = select_verified_checkpoint(old_ckpt)
        ckcfg = parent.get("cfg")
        if not dataclasses.is_dataclass(ckcfg):
            raise ValueError("Expected dataclass checkpoint config")
        if Path(ckcfg.output_dir).resolve() != (repo / "asset/runs" / old_study / relative).resolve():
            raise ValueError("Parent checkpoint output path differs from v4 run identity")
        require_unselected(parent.get("success_stopping_state"))
        if not 1 <= int(parent.get("epoch", 0)) < plan["stopping"]["minimum_epochs"]:
            raise ValueError("Parent checkpoint must precede the first development selection epoch")
        if parent.get("dataset") != args.dataset or parent.get("exp_id") != exp or parent.get("training_seed") != 3:
            raise ValueError("Parent dataset/experiment/seed identity mismatch")
        old_job = next((x for x in old_contract["jobs"]
                        if x.get("dataset") == args.dataset
                        and x.get("experiment") == exp and x.get("training_seed") == 3), None)
        if (old_job is None or old_job.get("protocol_id") != parent.get("protocol_id")
                or old_job.get("run_signature") != parent.get("run_signature")):
            raise ValueError("v4 locked job identity differs from the parent checkpoint")
        if parent.get("rollout_study_id") != old_study or parent.get("rollout_plan_id") != old_plan_id:
            raise ValueError("Parent study/plan identity mismatch")
        if parent.get("protocol") != protocol or parent.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("Parent protocol differs from staged frozen manifest")
        if parent.get("feature_mode") != "online":
            raise ValueError("Parent feature mode differs from frozen online protocol")
        provenance = parent.get("encoder_provenance", {})
        if provenance.get("source_assets") != job["source_assets"]:
            raise ValueError("Parent frozen-backbone assets differ from cached job")
        if provenance.get("encoder_code_id") != encoder_code_id():
            raise ValueError("Parent encoder provenance differs from byte-identical v4/v6 encoder")
        if (parent.get("rollout_stopping_config") != plan["stopping"]
                or parent.get("rollout_evaluation") != setting["development_evaluation"]):
            raise ValueError("Parent stopping/evaluation settings differ from frozen plan")
        if cfg_identity(ckcfg) != cfg_identity(job["config"]):
            raise ValueError("Parent resolved config differs from cached v6 job beyond output/cache/runtime fields")
        batch_change = EMBED_BATCH_CONFIG_CHANGES[args.dataset]
        if (getattr(ckcfg, "embed_batch_size", None) != batch_change["old"]
                or job["config"].get("embed_batch_size") != batch_change["new"]):
            raise ValueError("Parent/new effective embed_batch_size differs from the two reviewed config edits")
        old_signature = old_signature_from_archive(ckcfg, protocol, 3, old_archive,
                                                   provenance["encoder_code_id"])
        if (parent.get("run_signature") != old_signature
                or old_job.get("run_signature") != old_signature):
            raise ValueError("Parent/lock signature does not match frozen v4 source/config")
        if job.get("feature_mode", "cached") != "cached":
            raise ValueError("New job is not using cached features")
        for key in ("optim", "scheduler", "rng_state", "train_generator", "val_generator",
                    "state_dict", "global_step", "success_stopping_state"):
            if key not in parent or parent[key] is None:
                raise ValueError(f"Parent checkpoint lacks exact-continuation state: {key}")
        if (parent.get("global_step") != int(parent["epoch"]) * job["settings"]["steps_per_epoch"]
                or parent.get("steps_per_epoch") != job["settings"]["steps_per_epoch"]
                or parent.get("lr_schedule_steps") != plan["reference_optimizer_updates"]
                or parent.get("lr_decay_epochs") != job["settings"]["reference_epochs"]
                or parent.get("batch_size") != plan["effective_batch_size"]):
            raise ValueError("Parent optimizer schedule geometry differs from frozen plan")
        parent_sha = sha(parent_path)
        controller_bytes = controller.read_bytes()
        ctrl = json.loads(controller_bytes)
        require_unselected(ctrl.get("success_stopping_state"))
        if (ctrl.get("study_id") != old_study or ctrl.get("plan_id") != old_plan_id
                or ctrl.get("dataset") != args.dataset or ctrl.get("experiment") != exp
                or ctrl.get("training_seed") != 3 or ctrl.get("protocol_id") != protocol["protocol_id"]
                or ctrl.get("run_signature") != parent["run_signature"]
                or ctrl.get("last_committed_epoch") != parent["epoch"]
                or ctrl.get("last_committed_recovery_revision") != parent.get("recovery_revision")):
            raise ValueError("Frozen shared controller mirror differs from v4 checkpoint")
        equal(ctrl["success_stopping_state"], parent["success_stopping_state"], "controller mirror")
        cache_meta = cache_evidence[exp]
        new_controller = durable / "controller_state" / plan_id[:12] / args.dataset / exp / "seed_003.json"
        if new_controller.exists():
            raise FileExistsError(f"Unexpected v6 controller state already exists: {new_controller}")
        target_run = job["durable_run"]
        target_ckpt = target_run / "checkpoints/last.pt"
        target_history = target_run / "history.jsonl"
        audit_path = target_run / "migration_audit.json"
        parent_controller_copy = target_run / "parent_controller_mirror.json"
        for path in (target_ckpt, target_history, audit_path, parent_controller_copy):
            if path.exists():
                raise FileExistsError(f"Unexpected v6 migration destination already exists: {path}")
        migrated = copy.deepcopy(parent)
        migrated["rollout_study_id"] = args.new_study_id
        migrated["rollout_plan_id"] = plan_id
        migrated["run_signature"] = job["run_signature"]
        migrated["feature_mode"] = "cached"
        migrated["cache_id"] = cache_meta["cache_id"]
        migrated["encoder_provenance"] = {
            "encoder_code_id": current_encoder_id,
            "source_assets": job["source_assets"]}
        migrated["cfg"].output_dir = str(job["run"])
        migrated["cfg"].embeddings_cache = str(job["config"]["embeddings_cache"])
        migrated["cfg"].embed_batch_size = batch_change["new"]
        for key, value in parent.items():
            if key == "cfg":
                equal(cfg_except_cache_identity(value), cfg_except_cache_identity(migrated[key]), key)
            elif key not in REWRITTEN_KEYS:
                equal(value, migrated[key], key)
        prepared.append((job, parent_path, old_history, controller, controller_bytes, ctrl,
                         parent, parent_sha, migrated, target_run, target_ckpt,
                         target_history, audit_path, parent_controller_copy,
                         new_controller, cache_meta))
    # Preflight every destination and controller before writing any new namespace.
    for item in prepared:
        (job, old_ckpt, old_history, controller, controller_bytes, ctrl, parent,
         parent_sha, migrated, target_run, target_ckpt, target_history, audit_path,
         parent_copy, new_controller, cache_meta) = item
        target_run.mkdir(parents=True, exist_ok=True)
        target_ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = target_ckpt.with_name(target_ckpt.name + ".migration.tmp")
        if tmp.exists():
            raise FileExistsError(tmp)
        torch.save(migrated, tmp)
        reloaded = torch.load(tmp, map_location="cpu", weights_only=False)
        if (reloaded.get("run_signature") != job["run_signature"]
                or reloaded.get("feature_mode") != "cached"
                or reloaded.get("cache_id") != cache_meta["cache_id"]):
            raise ValueError("Migrated checkpoint cached identity failed read-back")
        for key, value in parent.items():
            if key == "cfg":
                equal(cfg_except_cache_identity(value), cfg_except_cache_identity(reloaded[key]), key)
            elif key not in REWRITTEN_KEYS:
                equal(value, reloaded[key], key)
        staged_sha = sha(tmp)
        exact_copy(tmp, target_ckpt)
        tmp.unlink()
        exact_copy(old_history, target_history)
        exact_copy(controller, parent_copy)
        if parent_copy.read_bytes() != controller_bytes:
            raise IOError("Saved parent controller bytes differ from preflight snapshot")
        audit = {"schema": 1, "migration": "v4_online_to_v6_cached_preselection_continuation",
                 "dataset": args.dataset, "experiment": job["experiment"], "training_seed": 3,
                 "parent_study_id": old_study, "new_study_id": args.new_study_id,
                 "parent_plan_id": old_plan_id, "new_plan_id": plan_id,
                 "parent_run_signature": parent["run_signature"],
                 "new_run_signature": job["run_signature"],
                 "parent_epoch": parent["epoch"],
                 "parent_global_step": parent["global_step"],
                 "parent_checkpoint_path": str(parent_path), "parent_checkpoint_sha256": parent_sha,
                 "continued_checkpoint_path": str(target_ckpt),
                 "continued_checkpoint_sha256": staged_sha,
                 "parent_history_sha256": sha(old_history),
                 "parent_controller_backup": str(parent_copy),
                 "parent_controller_sha256": hashlib.sha256(controller_bytes).hexdigest(),
                 "new_controller_path": str(new_controller),
                 "parent_controller_path_read_only": str(controller),
                 "old_source_archive_sha256": OLD_ARCHIVE_SHA256,
                 "new_source_archive_sha256": new_archive_sha,
                 "cache_manifest_path": str(cache_manifest_path),
                 "cache_manifest_sha256": cache_manifest_sha,
                 "cache_prepare_report_path": str(cache_report_path),
                 "cache_prepare_report_sha256": sha(cache_report_path),
                  "cache_prepare_report_identity_sha256": cache_report_identity_sha,
                 "cache_sidecars": {e: {"path": cache_evidence[e]["manifest_path"],
                                        "sha256": cache_evidence[e]["manifest_sha256"]}
                                    for e in sorted(cache_evidence)},
                 "cache_path": cache_meta["path"],
                 "cache_sha256": cache_meta["sha256"],
                 "cache_id": cache_meta["cache_id"],
                 "cache_config": cache_meta["cache_config"],
                 "workflow_code_changed": changed,
                 "archive_paths_changed": sorted(archive_changes),
                 "embed_batch_size_config_changes": batch_config_evidence,
                 "checkpoint_embed_batch_size_change": {
                     "old": batch_change["old"], "new": batch_change["new"],
                     "source_config": batch_change["path"],
                     "cache_config_and_run_signature_exclude_field": True},
                 "training_source_transition": training_source_evidence,
                 "training_and_encoder_architecture_code_unchanged": True,
                 "protocol_id": protocol["protocol_id"],
                 "source_assets": parent.get("encoder_provenance", {}).get("source_assets"),
                 "preserved_checkpoint_keys": sorted(k for k in parent if k not in REWRITTEN_KEYS and k != "cfg"),
                 "rewritten_identity_fields": ["rollout_study_id", "rollout_plan_id", "run_signature",
                                                "feature_mode", "cache_id", "encoder_provenance",
                                                "cfg.output_dir", "cfg.embeddings_cache",
                                                "cfg.embed_batch_size"],
                 "cached_features_use_same_encoder_contract": True,
                 "bitwise_online_feature_equivalence_claimed": False,
                 "floating_point_note": "Cached features use the same frozen encoder/source assets, preprocessing and protocol; floating-point rounding can differ across hardware.",
                 "controller_preimage_sha256": hashlib.sha256(controller_bytes).hexdigest()}
        audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if sha(target_ckpt) != staged_sha or sha(target_history) != sha(old_history):
            raise IOError("Copied v6 continuation checkpoint/history failed verification")
        new_ctrl = {**ctrl, "study_id": args.new_study_id, "plan_id": plan_id,
                    "run_signature": job["run_signature"]}
        new_controller.parent.mkdir(parents=True, exist_ok=True)
        new_controller.write_text(json.dumps(new_ctrl, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if json.loads(new_controller.read_text(encoding="utf-8")) != new_ctrl:
            raise IOError("Updated controller mirror failed read-back verification")
        print(json.dumps({"status": "migrated", "dataset": args.dataset,
                          "experiment": job["experiment"], "epoch": parent["epoch"],
                          "parent_sha256": parent_sha, "continued_sha256": staged_sha,
                          "controller_backup_sha256": hashlib.sha256(controller_bytes).hexdigest()}))
    durable_lock = new_root / "locked_success_study.json"
    local_lock = run_root / "locked_success_study.json"
    if durable_lock.exists() or local_lock.exists():
        raise FileExistsError("Refusing to overwrite a v6 lock during migration commit")
    lock_bytes = (json.dumps(expected_contract, sort_keys=True, indent=2) + "\n").encode("utf-8")
    local_lock.parent.mkdir(parents=True, exist_ok=True)
    local_lock.write_bytes(lock_bytes)
    if local_lock.read_bytes() != lock_bytes:
        raise IOError("Local v6 study lock failed read-back")
    durable_lock.parent.mkdir(parents=True, exist_ok=True)
    durable_lock.write_bytes(lock_bytes)
    if durable_lock.read_bytes() != lock_bytes:
        raise IOError("Durable v6 study lock failed read-back")
    print(json.dumps({"status": "complete", "dataset": args.dataset,
                      "migrated_seed3_arms": len(prepared), "plan_id": plan_id,
                      "new_study_id": args.new_study_id, "new_source_archive_sha256": new_archive_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
