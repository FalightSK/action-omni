"""Download the revision-locked study assets and keep a copy on RunPod Global Volume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_protocol import file_digest, write_json
from huggingface_hub import snapshot_download
from scripts.run_log import log_event


MODEL_DIRS = ("groot_n17_3b", "qwen3_vl_2b", "cosmos_reason2_2b")


def copy_tree(source: Path, target: Path) -> None:
    started = time.perf_counter()
    files = 0
    bytes_copied = 0
    last_report_bytes = 0
    log_event("asset.copy_started", source=str(source), destination=str(target))
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if ".cache" in path.parts:
            continue
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            files += 1
            bytes_copied += path.stat().st_size
            if files % 100 == 0 or bytes_copied - last_report_bytes >= 1024**3:
                elapsed = time.perf_counter() - started
                log_event("asset.copy_progress", source=str(source), destination=str(target),
                          files_copied=files, bytes_copied=bytes_copied,
                          files_per_s=round(files / max(elapsed, 1e-9), 2),
                          elapsed_s=round(elapsed, 1))
                last_report_bytes = bytes_copied
    log_event("asset.copy_complete", source=str(source), destination=str(target),
              files=files, bytes=bytes_copied,
              elapsed_s=round(time.perf_counter() - started, 3))


def download_snapshot(stage: str, repo_id: str, **kwargs):
    started = time.perf_counter()
    stop_heartbeat = threading.Event()
    log_event("asset.download_started", stage=stage, repo_id=repo_id,
              revision=kwargs.get("revision"), destination=kwargs.get("local_dir"))
    def heartbeat() -> None:
        while not stop_heartbeat.wait(30):
            log_event("asset.download_in_progress", stage=stage, repo_id=repo_id,
                      elapsed_s=round(time.perf_counter() - started, 1))

    heartbeat_thread = threading.Thread(target=heartbeat, name=f"download-heartbeat-{stage}", daemon=True)
    heartbeat_thread.start()
    try:
        result = snapshot_download(repo_id, **kwargs)
    except BaseException as exc:
        log_event("asset.download_failed", level="ERROR", stage=stage, repo_id=repo_id,
                  error_type=type(exc).__name__, error=str(exc),
                  elapsed_s=round(time.perf_counter() - started, 1))
        raise
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)
    log_event("asset.download_complete", stage=stage, repo_id=repo_id,
              destination=str(result), elapsed_s=round(time.perf_counter() - started, 3))
    return result


def copy_from_global(global_root: Path, asset_root: Path, manifest: dict) -> None:
    expected = manifest.get("source_revisions")
    if expected is None:
        raise RuntimeError("Global asset manifest is missing pinned source revisions")
    for name in MODEL_DIRS:
        src, dst = global_root / "assets/models" / name, asset_root / "models" / name
        if not src.is_dir():
            raise FileNotFoundError(f"Global model assets are incomplete: {src}")
        copy_tree(src, dst)


def download_models(plan: dict, asset_root: Path) -> None:
    rev = plan["source_revisions"]
    download_snapshot("model_groot", rev["groot"]["repo_id"], repo_type="model",
                      revision=rev["groot"]["revision"], local_dir=str(asset_root / "models/groot_n17_3b"),
                      allow_patterns=["config.json", "model*.safetensors", "model.safetensors.index.json",
                                      "processor_config.json", "embodiment_id.json",
                                      "experiment_cfg/final_model_config.json",
                                      "experiment_cfg/final_processor_config.json"], max_workers=2)
    download_snapshot("model_groot_tokenizer", rev["groot_tokenizer"]["repo_id"], repo_type="model",
                      revision=rev["groot_tokenizer"]["revision"],
                      local_dir=str(asset_root / "models/cosmos_reason2_2b"),
                      allow_patterns=["*.json", "*.txt", "tokenizer*", "merges.txt", "vocab.json"],
                      max_workers=2)
    download_snapshot("model_qwen3vl", rev["qwen3vl"]["repo_id"], repo_type="model",
                      revision=rev["qwen3vl"]["revision"], local_dir=str(asset_root / "models/qwen3_vl_2b"),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
                      max_workers=2)


def download_dataset(plan: dict, dataset: str, asset_root: Path) -> None:
    source = plan["source_revisions"][f"{dataset}_dataset"]
    local_dir = asset_root / "data" / (
        "aloha_sim_transfer_cube_human" if dataset == "aloha" else "libero")
    patterns = ["libero_goal/*.hdf5"] if dataset == "libero" else None
    download_snapshot(f"dataset_{dataset}", source["repo_id"], repo_type="dataset", revision=source["revision"],
                      local_dir=str(local_dir), allow_patterns=patterns, max_workers=4)


def asset_manifest(plan: dict, datasets: list[str], asset_root: Path) -> dict:
    started = time.perf_counter()
    files = {}
    hashed_files = 0
    hashed_bytes = 0

    def hash_files(paths: list[Path], dataset_name: str) -> None:
        nonlocal hashed_files, hashed_bytes
        for path in paths:
            files[path.relative_to(asset_root).as_posix()] = {
                "size": path.stat().st_size, "sha256": file_digest(path)}
            hashed_files += 1
            hashed_bytes += path.stat().st_size
            if hashed_files % 25 == 0 or hashed_files == total_files:
                log_event("asset.hash_progress", dataset=dataset_name, files_hashed=hashed_files,
                          files_total=total_files, bytes_hashed=hashed_bytes,
                          elapsed_s=round(time.perf_counter() - started, 1))

    paths_by_group = []
    for name in MODEL_DIRS:
        directory = asset_root / "models" / name
        paths_by_group.append((name, [path for path in sorted(directory.rglob("*"))
                                      if path.is_file() and path.suffix in {".json", ".safetensors"}]))
    for dataset in datasets:
        name = "aloha_sim_transfer_cube_human" if dataset == "aloha" else "libero/libero_goal"
        paths_by_group.append((f"dataset_{dataset}", [path for path in sorted((asset_root / "data" / name).rglob("*"))
                                                       if path.is_file() and (dataset == "aloha" or path.suffix == ".hdf5")]))
    total_files = sum(len(paths) for _, paths in paths_by_group)
    for group, paths in paths_by_group:
        log_event("asset.hash_group_started", dataset=group, files=len(paths),
                  bytes=sum(path.stat().st_size for path in paths))
        hash_files(paths, group)
    result = {"schema": 1, "source_revisions": plan["source_revisions"],
            "datasets_ready": sorted(datasets), "files": files,
            "updated_utc": datetime.now(timezone.utc).isoformat()}
    log_event("asset.manifest_built", datasets=sorted(datasets), files=len(files),
              bytes=sum(item["size"] for item in files.values()),
              elapsed_s=round(time.perf_counter() - started, 3))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("aloha", "libero"), required=True)
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_update_matched_study.json")
    ap.add_argument("--durable-root", type=Path,
                    default=Path("/workspace-global/action-omni"))
    ap.add_argument("--models-from-global", action="store_true",
                    help="Reuse the revision-locked models already staged by the ALOHA worker")
    args = ap.parse_args()
    run_started = time.perf_counter()
    log_event("asset_setup.started", dataset=args.dataset, plan=str(args.plan),
              durable_root=str(args.durable_root), models_from_global=args.models_from_global)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    asset_root = ROOT / "asset"
    global_root = args.durable_root.resolve()
    global_root.mkdir(parents=True, exist_ok=True)
    manifest_path = global_root / "assets_manifest.json"

    if args.models_from_global:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Shared model manifest is not ready: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_revisions") != plan["source_revisions"]:
            raise ValueError("Global model revisions differ from this locked study")
        copy_from_global(global_root, asset_root, manifest)
    else:
        download_models(plan, asset_root)

    download_dataset(plan, args.dataset, asset_root)
    for name in MODEL_DIRS:
        copy_tree(asset_root / "models" / name, global_root / "assets/models" / name)
    dataset_name = "aloha_sim_transfer_cube_human" if args.dataset == "aloha" else "libero"
    copy_tree(asset_root / "data" / dataset_name, global_root / "assets/data" / dataset_name)

    ready_datasets = {args.dataset}
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("source_revisions") != plan["source_revisions"]:
            raise ValueError("Refusing to combine assets from different source revisions")
        ready_datasets.update(old.get("datasets_ready", []))
    write_json(manifest_path, asset_manifest(plan, sorted(ready_datasets), global_root / "assets"))
    print(f"RunPod assets ready: dataset={args.dataset}; manifest={manifest_path}", flush=True)
    log_event("asset_setup.complete", dataset=args.dataset, manifest=str(manifest_path),
              elapsed_s=round(time.perf_counter() - run_started, 3))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_event("asset_setup.failed", level="ERROR", error_type=type(exc).__name__, error=str(exc))
        raise
