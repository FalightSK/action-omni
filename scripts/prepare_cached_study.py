"""Build or stage the four protocol-locked frozen-VLM feature caches.

The cache is shared by all training seeds.  This command either computes a
missing cache once (on the designated cache-preparation worker) or, with
``--reuse-only``, waits for and stages an already published cache without
loading a VLM.  A manifest is written only after the HDF5 file is closed and
verified; readers must validate both the manifest and the file digest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from experiment_protocol import (apply_normalization, cache_config, digest,
                                 encoder_code_id, file_digest, load_protocol,
                                 protocol_arrays, source_assets, write_json)
from scripts.run_log import log_event


def cache_paths(durable_root: Path, study_id: str, dataset: str,
                experiment: str) -> tuple[Path, Path]:
    """Return the agreed local and Global Volume paths for one cache."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", study_id) or study_id in {".", ".."}:
        raise ValueError(f"Unsafe study ID: {study_id!r}")
    if dataset not in {"aloha", "libero"} or experiment not in {"exp05", "exp06"}:
        raise ValueError(f"Unsupported cached study arm: {dataset}/{experiment}")
    local = ROOT / "asset/cache/cached_success_study" / study_id / dataset / experiment / "vlm_embeddings.h5"
    durable = (Path(durable_root) / "success_studies" / study_id / "feature_caches"
               / dataset / experiment / "vlm_embeddings.h5")
    return local.resolve(), durable.resolve()


def _sha256_progress(path: Path, *, event: str, role: str) -> str:
    hasher = hashlib.sha256()
    size = path.stat().st_size
    started = time.perf_counter()
    last_log = 0
    read = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            hasher.update(block)
            read += len(block)
            if read - last_log >= 2 * 1024**3 or read == size:
                log_event(event, role=role, path=str(path), bytes_read=read,
                          total_bytes=size, elapsed_s=round(time.perf_counter() - started, 1))
                last_log = read
    return hasher.hexdigest()


def _copy_with_progress(source: Path, target: Path, *, role: str) -> str:
    """Copy through a unique staging name, hashing bytes and checking the result."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.staging-{uuid.uuid4().hex}.h5")
    started = time.perf_counter()
    hasher = hashlib.sha256()
    copied = 0
    total = source.stat().st_size
    log_event("cache.copy_started", role=role, source=str(source), target=str(target),
              total_bytes=total)
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            while True:
                block = src.read(16 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
                hasher.update(block)
                copied += len(block)
                if copied == total or copied % (2 * 1024**3) < len(block):
                    log_event("cache.copy_progress", role=role, bytes_copied=copied,
                              total_bytes=total,
                              elapsed_s=round(time.perf_counter() - started, 1))
            dst.flush()
            os.fsync(dst.fileno())
        if copied != total or temporary.stat().st_size != total:
            raise IOError(f"Short cache copy: {source} -> {temporary} ({copied}/{total})")
        copy_hash = hasher.hexdigest()
        temporary.replace(target)
        # Global Volume is eventually consistent; read back the published path
        # and verify it before allowing a sidecar to advertise completion.
        published_hash = _sha256_progress(target, event="cache.copy_verify_progress", role=role)
        if published_hash != copy_hash:
            raise IOError(f"Published cache checksum mismatch: {target}")
        log_event("cache.copy_complete", role=role, target=str(target), bytes=copied,
                  sha256=published_hash, elapsed_s=round(time.perf_counter() - started, 1))
        return published_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def _asset_manifest_check(plan: dict, dataset: str, cfg, durable_root: Path,
                          protocol: dict, assets: dict) -> None:
    """Check pinned assets/data against the shared asset manifest before encoding."""
    manifest_path = durable_root / "assets_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Shared asset manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_revisions") != plan.get("source_revisions"):
        raise ValueError("Shared model/dataset revisions differ from the locked plan")
    if dataset not in manifest.get("datasets_ready", []):
        raise ValueError(f"Shared asset manifest does not mark {dataset} ready")
    files = manifest.get("files", {})
    # Verify the exact locally staged model and processor files against the
    # revision-locked Global Volume manifest. The precompute stage hashes these
    # same assets into the HDF5 provenance too.
    for relative, actual_sha in assets.items():
        if not relative.startswith("asset/"):
            raise ValueError(f"Unexpected source asset path: {relative}")
        # `asset_manifest()` stores paths relative to its `asset_root` (the
        # contents of the Global Volume's `assets/` directory), so keys omit
        # the literal `assets/` prefix.
        cloud_key = relative[len("asset/"):]
        item = files.get(cloud_key)
        if not item or item.get("sha256") != actual_sha:
            raise ValueError(f"Local encoder asset differs from shared manifest: {cloud_key}")
    data_root = Path(cfg.dataset_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    data_root = data_root.resolve()
    try:
        data_relative = data_root.relative_to((ROOT / "asset").resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Dataset root is outside the staged asset tree: {data_root}") from exc
    for relative, expected_sha in protocol["source_files"].items():
        cloud_key = f"{data_relative}/{relative}"
        item = files.get(cloud_key)
        if not item or item.get("sha256") != expected_sha:
            raise ValueError(f"Local protocol data differs from shared manifest: {cloud_key}")


def _validate_hdf5(path: Path, *, dataset: str, experiment: str, protocol: dict,
                   cfg, assets: dict) -> str:
    """Validate complete HDF5 structure and provenance without reading features."""
    import h5py
    ids, validation = protocol_arrays(protocol)
    n = int(protocol["n_samples"])
    expected_layers = len(cfg.vlm_extract_layers)
    expected_embed_shape = ((n, expected_layers, cfg.img_seq_len, cfg.vlm_hidden_size)
                            if expected_layers > 1 else
                            (n, cfg.img_seq_len, cfg.vlm_hidden_size))
    expected_state_dim = 2 if dataset == "pusht" else cfg.state_dim
    with h5py.File(path, "r") as f:
        attrs = f.attrs
        required = {"embeddings", "img_masks", "states", "actions", "episode_ids",
                    "validation_mask"}
        missing = required.difference(f.keys())
        if missing:
            raise ValueError(f"Cache lacks required datasets: {sorted(missing)}")
        checks = {
            "complete": bool(attrs.get("complete", False)) is True,
            "full_dataset": bool(attrs.get("full_dataset", False)) is True,
            "dataset": attrs.get("dataset") == dataset,
            "exp_id": attrs.get("exp_id") == experiment,
            "protocol_id": attrs.get("protocol_id") == protocol["protocol_id"],
            "encoder_code_id": attrs.get("encoder_code_id") == encoder_code_id(),
            "source_assets": json.loads(attrs.get("source_assets", "null")) == assets,
            "cache_config": json.loads(attrs.get("cache_config", "null")) == cache_config(cfg),
            "n_samples": int(attrs.get("n_samples", -1)) == n,
            "layers": tuple(int(x) for x in attrs.get("vlm_extract_layers", ()))
                      == tuple(int(x) for x in cfg.vlm_extract_layers),
            "embeddings": tuple(f["embeddings"].shape) == expected_embed_shape,
            "img_masks": tuple(f["img_masks"].shape) == (n, cfg.img_seq_len),
            "states": tuple(f["states"].shape) == (n, expected_state_dim),
            "actions": tuple(f["actions"].shape) ==
                       (n, cfg.action_horizon, cfg.action_dim),
            "episode_ids": (f["episode_ids"].shape == (n,)
                            and (f["episode_ids"][:] == ids).all()),
            "validation_mask": (f["validation_mask"].shape == (n,)
                                and (f["validation_mask"][:].astype(bool) == validation).all()),
        }
        failed = [name for name, okay in checks.items() if not okay]
        if failed:
            raise ValueError(f"Cache HDF5 validation failed for {path}: {failed}")
        cache_id = attrs.get("cache_id")
        if isinstance(cache_id, bytes):
            cache_id = cache_id.decode("utf-8")
        if not cache_id:
            raise ValueError("Cache lacks cache_id")
        parity = json.loads(attrs.get("batch_one_parity", "{}"))
        if not parity.get("rows") or parity.get("masks_equal") is False:
            raise ValueError("Cache lacks successful batch-one parity evidence")
    return str(cache_id)


def _manifest_payload(*, cache_id: str, dataset: str, experiment: str,
                      backbone: str, study_id: str, plan_id: str,
                      protocol_id: str, encoder_id: str, assets: dict,
                      size_bytes: int, sha256: str, local_path: Path,
                      durable_path: Path) -> dict:
    return {
        "schema": 1, "dataset": dataset, "experiment": experiment,
        "backbone": backbone, "study_id": study_id, "plan_id": plan_id,
        "protocol_id": protocol_id, "cache_id": cache_id,
        "encoder_code_id": encoder_id, "source_assets": assets,
        "size_bytes": int(size_bytes), "sha256": sha256,
        "local_path": str(local_path.resolve()),
        "durable_path": str(durable_path.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _read_published(path: Path, expected: dict, *, role: str) -> tuple[dict, str] | None:
    sidecar = path.with_name(path.name + ".manifest.json")
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            log_event("cache.published_not_ready", level="WARNING", role=role,
                      reason="manifest_not_object", path=str(path))
            return None
        size = path.stat().st_size
        if size != manifest.get("size_bytes") or (
                "size_bytes" in expected and size != expected["size_bytes"]):
            log_event("cache.published_not_ready", level="WARNING", role=role,
                      reason="size_mismatch", path=str(path), actual=size,
                      manifest_size=manifest.get("size_bytes"))
            return None
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Published cache {role} manifest identity mismatch: {key}")
        actual_sha = _sha256_progress(path, event="cache.hash_progress", role=role)
        if manifest.get("sha256") != actual_sha:
            log_event("cache.published_not_ready", level="WARNING", role=role,
                      reason="sha_mismatch", path=str(path), expected_sha=manifest.get("sha256"),
                      actual_sha=actual_sha)
            return None
        return manifest, actual_sha
    except (OSError, json.JSONDecodeError) as exc:
        log_event("cache.published_not_ready", level="WARNING", role=role,
                  reason="manifest_not_visible_or_parseable", path=str(path), error=str(exc))
        return None


def _stage_local_from_durable(durable: Path, local: Path, durable_manifest: dict,
                              *, role: str) -> dict:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        raise FileExistsError(f"Local cache exists but is not verified; refusing overwrite: {local}")
    sha = _copy_with_progress(durable, local, role=role + ".durable_to_local")
    if sha != durable_manifest["sha256"]:
        raise IOError(f"Staged cache checksum differs from Global Volume: {local}")
    # The sidecar is rewritten for this Pod's local path while preserving the
    # immutable cache identity, SHA, and Global Volume path.
    local_manifest = dict(durable_manifest)
    local_manifest["local_path"] = str(local.resolve())
    local_manifest["durable_path"] = str(durable.resolve())
    write_json(local.with_name(local.name + ".manifest.json"), local_manifest)
    return local_manifest


def _publish_durable(local: Path, durable: Path, manifest_base: dict, *, role: str) -> dict:
    if durable.exists():
        raise FileExistsError(f"Global Volume cache already exists without a validated manifest: {durable}")
    sha = _copy_with_progress(local, durable, role=role + ".local_to_durable")
    payload = dict(manifest_base, size_bytes=durable.stat().st_size, sha256=sha,
                   local_path=str(local.resolve()), durable_path=str(durable.resolve()))
    # Publish the sidecar only after the final Global Volume object is read back
    # and its size/SHA have been verified.
    write_json(durable.with_name(durable.name + ".manifest.json"), payload)
    log_event("cache.published", role=role, cache=str(durable),
              manifest=str(durable.with_name(durable.name + ".manifest.json")),
              size_bytes=payload["size_bytes"], sha256=sha, cache_id=payload["cache_id"])
    return payload


def _prepare_one(args, plan: dict, plan_id: str, dataset: str,
                 experiment: str) -> dict:
    role = f"{dataset}/{experiment}"
    cfg = get_config(dataset, experiment)
    setting = plan["datasets"][dataset]
    protocol_path = ROOT / setting["split_manifest"]
    protocol = load_protocol(protocol_path, dataset)
    if protocol["protocol_id"] != Path(setting["split_manifest"]).stem:
        raise ValueError(f"Protocol file does not match locked plan for {dataset}")
    apply_normalization(cfg, protocol)
    backbone = "groot" if experiment == plan["backbones"]["groot"] else "qwen3vl"
    if cfg.vlm_backbone != backbone:
        raise ValueError(f"Configured encoder differs from locked arm {dataset}/{experiment}")
    local, durable = cache_paths(args.durable_root, args.study_id, dataset, experiment)
    provenance_started = time.perf_counter()
    log_event("cache.provenance_hash_started", role=role, backbone=cfg.vlm_backbone)
    assets = source_assets(cfg)
    log_event("cache.provenance_hash_complete", role=role, files=len(assets),
              elapsed_s=round(time.perf_counter() - provenance_started, 1))
    _asset_manifest_check(plan, dataset, cfg, args.durable_root, protocol, assets)
    encoder_id = encoder_code_id()
    expected_identity = {
        "schema": 1, "dataset": dataset, "experiment": experiment,
        "backbone": backbone, "study_id": args.study_id, "plan_id": plan_id,
        "protocol_id": protocol["protocol_id"], "encoder_code_id": encoder_id,
        "source_assets": assets,
    }
    expected_local_identity = dict(
        expected_identity, local_path=str(local.resolve()), durable_path=str(durable.resolve()))
    expected_durable_identity = dict(expected_identity, durable_path=str(durable.resolve()))

    # A fully validated local artifact is reusable without hashing/re-encoding.
    if local.is_file() and local.with_name(local.name + ".manifest.json").is_file():
        try:
            local_manifest, local_sha = _read_published(local, expected_local_identity,
                                                        role=role + ".local")
        except ValueError:
            raise
        if local_manifest is not None:
            cache_id = _validate_hdf5(local, dataset=dataset, experiment=experiment,
                                      protocol=protocol, cfg=cfg, assets=assets)
            if local_manifest.get("cache_id") != cache_id:
                raise ValueError(f"Local sidecar/HDF5 cache ID differs: {local}")
            log_event("cache.reused_local", role=role, path=str(local), sha256=local_sha,
                      cache_id=cache_id)
            # If remote manifest is present and valid, ensure both mirrors point
            # to identical bytes; otherwise publish/recover the durable copy.
            durable_expected = dict(expected_durable_identity,
                                    size_bytes=local.stat().st_size, cache_id=cache_id)
            cloud = _read_published(durable, durable_expected, role=role + ".durable")
            if cloud is None:
                if args.reuse_only or durable.exists():
                    return _wait_reuse(args, plan, role, local, durable, expected_durable_identity,
                                       dataset, experiment, protocol, cfg, assets)
                _publish_durable(local, durable,
                                 dict(expected_identity, cache_id=cache_id), role=role)
            elif cloud[1] != local_sha:
                raise ValueError(f"Local and durable caches differ for {role}")
            return {"cache_id": cache_id, "local_path": str(local),
                    "durable_path": str(durable), "sha256": local_sha,
                    "size_bytes": local.stat().st_size, "reused": True}

    # Reuse a complete Global Volume artifact on any worker, including a
    # restarted preparation job, before considering a new VLM pass.
    cloud = _read_published(durable, expected_durable_identity, role=role + ".durable")
    if cloud is not None:
        manifest, sha = cloud
        cache_id = _validate_hdf5(durable, dataset=dataset, experiment=experiment,
                                  protocol=protocol, cfg=cfg, assets=assets)
        if cache_id != manifest.get("cache_id"):
            raise ValueError(f"Global Volume manifest/HDF5 cache ID mismatch for {role}")
        _stage_local_from_durable(durable, local, manifest, role=role)
        local_cache_id = _validate_hdf5(local, dataset=dataset, experiment=experiment,
                                        protocol=protocol, cfg=cfg, assets=assets)
        if local_cache_id != cache_id:
            raise ValueError(f"Staged cache identity differs from Global Volume: {local}")
        return {"cache_id": cache_id, "local_path": str(local),
                "durable_path": str(durable), "sha256": sha,
                "size_bytes": local.stat().st_size, "reused": True}

    if args.reuse_only or durable.exists():
        return _wait_reuse(args, plan, role, local, durable, expected_durable_identity,
                           dataset, experiment, protocol, cfg, assets)

    if local.exists():
        raise FileExistsError(f"Unverified local cache exists; refusing overwrite: {local}")
    local.parent.mkdir(parents=True, exist_ok=True)
    partial = local.with_name(f"{local.stem}.building-{uuid.uuid4().hex}.h5")
    command = [sys.executable, "-u", str(ROOT / "scripts/precompute.py"),
               "--dataset", dataset, "--exp", experiment,
               "--split-manifest", str(protocol_path), "--output", str(partial),
               "--log-every-batches", str(args.log_every_batches)]
    started = time.perf_counter()
    log_event("cache.precompute_started", role=role, output=str(partial), command=command)
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        if not partial.is_file():
            raise FileNotFoundError(f"Precompute returned success but cache is missing: {partial}")
        cache_id = _validate_hdf5(partial, dataset=dataset, experiment=experiment,
                                  protocol=protocol, cfg=cfg, assets=assets)
        sha = _sha256_progress(partial, event="cache.local_hash_progress", role=role)
        size = partial.stat().st_size
        partial.replace(local)
        local_manifest = _manifest_payload(
            cache_id=cache_id, dataset=dataset, experiment=experiment, backbone=backbone,
            study_id=args.study_id, plan_id=plan_id, protocol_id=protocol["protocol_id"],
            encoder_id=encoder_id, assets=assets, size_bytes=size, sha256=sha,
            local_path=local, durable_path=durable)
        write_json(local.with_name(local.name + ".manifest.json"), local_manifest)
        log_event("cache.local_complete", role=role, path=str(local), size_bytes=size,
                  sha256=sha, cache_id=cache_id,
                  elapsed_s=round(time.perf_counter() - started, 1))
        _publish_durable(local, durable, local_manifest, role=role)
        return {"cache_id": cache_id, "local_path": str(local),
                "durable_path": str(durable), "sha256": sha,
                "size_bytes": size, "reused": False}
    finally:
        if partial.exists():
            partial.unlink()


def _wait_reuse(args, plan, role, local, durable, identity, dataset,
                experiment, protocol, cfg, assets) -> dict:
    started = time.perf_counter()
    deadline = started + args.wait_seconds
    expected = dict(identity)
    while time.perf_counter() <= deadline:
        entry = _read_published(durable, expected, role=role + ".durable")
        if entry is not None:
            manifest, sha = entry
            cache_id = _validate_hdf5(durable, dataset=dataset, experiment=experiment,
                                      protocol=protocol, cfg=cfg, assets=assets)
            if cache_id != manifest.get("cache_id"):
                raise ValueError(f"Global Volume manifest/HDF5 cache ID mismatch for {role}")
            if local.exists():
                # Local data can only be reused after proving its full SHA and
                # structural identity. A mismatch is a hard error, not a rewrite.
                local_expected = dict(expected, local_path=str(local.resolve()))
                local_entry = _read_published(local, local_expected, role=role + ".local")
                if local_entry is not None:
                    if local_entry[1] != sha:
                        raise ValueError(f"Existing local mirror differs from durable cache: {local}")
                    local_cache_id = _validate_hdf5(
                        local, dataset=dataset, experiment=experiment,
                        protocol=protocol, cfg=cfg, assets=assets)
                    if local_cache_id != cache_id:
                        raise ValueError(f"Local cache identity differs from durable cache: {local}")
                    return {"cache_id": cache_id, "local_path": str(local),
                            "durable_path": str(durable), "sha256": sha,
                            "size_bytes": local.stat().st_size, "reused": True}
                raise FileExistsError(f"Local cache exists but is not verified: {local}")
            _stage_local_from_durable(durable, local, manifest, role=role)
            # Verify the staged HDF5 structure before declaring it ready.
            staged_id = _validate_hdf5(local, dataset=dataset, experiment=experiment,
                                       protocol=protocol, cfg=cfg, assets=assets)
            if staged_id != cache_id:
                raise ValueError(f"Staged cache identity differs from Global Volume: {local}")
            return {"cache_id": cache_id, "local_path": str(local),
                    "durable_path": str(durable), "sha256": sha,
                    "size_bytes": local.stat().st_size, "reused": True}
        log_event("cache.waiting_for_durable", role=role,
                  elapsed_s=round(time.perf_counter() - started, 1),
                  timeout_s=args.wait_seconds)
        time.sleep(args.poll_seconds)
    raise TimeoutError(f"No complete, verified Global Volume cache became visible for {role} "
                       f"within {args.wait_seconds}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aloha", "libero"), required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_cached_success_study.json")
    parser.add_argument("--reuse-only", action="store_true",
                        help="Stage an already published cache; never invoke precompute/VLM")
    parser.add_argument("--wait-seconds", type=float, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--log-every-batches", type=int, default=10)
    args = parser.parse_args(argv)
    if args.wait_seconds < 0 or args.poll_seconds <= 0 or args.log_every_batches < 1:
        parser.error("wait must be nonnegative; poll and log interval must be positive")

    started = time.perf_counter()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != 2 or plan.get("feature_mode") != "cached":
        raise ValueError("Cache preparation requires a schema-2 cached-feature plan")
    if args.dataset not in plan.get("datasets", {}):
        raise ValueError(f"Dataset absent from plan: {args.dataset}")
    plan_id = digest(plan)
    log_event("cached_study_prepare.started", dataset=args.dataset, study_id=args.study_id,
              plan_id=plan_id, durable_root=str(args.durable_root.resolve()),
              reuse_only=args.reuse_only)
    results = []
    for experiment in (plan["backbones"]["groot"], plan["backbones"]["qwen3vl"]):
        results.append(_prepare_one(args, plan, plan_id, args.dataset, experiment))
    report = {"schema": 1, "study_id": args.study_id, "plan_id": plan_id,
              "dataset": args.dataset, "feature_mode": "cached", "caches": results,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_s": round(time.perf_counter() - started, 1)}
    report_path = (Path(args.durable_root) / "success_studies" / args.study_id
                   / f"cache_prepare_{args.dataset}.json")
    write_json(report_path, report)
    log_event("cached_study_prepare.complete", dataset=args.dataset, study_id=args.study_id,
              report=str(report_path), caches=len(results), elapsed_s=report["elapsed_s"])
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        log_event("cached_study_prepare.failed", level="ERROR",
                  error_type=type(exc).__name__, error=str(exc))
        raise
    raise SystemExit(exit_code)
