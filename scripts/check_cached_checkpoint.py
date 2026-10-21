"""One real cached head update, followed by a Cloud checkpoint round trip."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.checkpoint_io import (atomic_copy_checkpoint, file_sha256,
                                   publish_recovery_revision)
from scripts.run_log import log_event


def validate_one_step_checkpoint(checkpoint: dict, protocol_id: str, cache_id: str,
                                 dataset: str = "aloha") -> None:
    expected = {"dataset": dataset, "exp_id": "exp05", "training_seed": 123,
                "feature_mode": "cached", "protocol_id": protocol_id, "cache_id": cache_id,
                "epoch": 1, "global_step": 1, "batch_size": 128,
                "steps_per_epoch": 1, "max_training_steps": 1}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Diagnostic checkpoint identity or one-update budget differs")
    model = checkpoint.get("state_dict")
    if (not isinstance(model, dict) or not model
            or not any(key.startswith("adapter.") for key in model)
            or not any(key.startswith("decoder.") for key in model)
            or not all(torch.is_tensor(value) for value in model.values())):
        raise ValueError("Diagnostic checkpoint lacks adapter/decoder model state")
    optimizer = checkpoint.get("optim", {})
    states = optimizer.get("state", {})
    if not states or len(optimizer.get("param_groups", [])) != 2:
        raise ValueError("Diagnostic checkpoint lacks both optimizer parameter groups")
    for state in states.values():
        step = state.get("step", -1)
        if hasattr(step, "item"):
            step = step.item()
        if step != 1 or not all(torch.is_tensor(state.get(key)) for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Diagnostic optimizer state does not contain exactly one AdamW update")
    if checkpoint.get("scheduler", {}).get("last_epoch") != 1:
        raise ValueError("Diagnostic scheduler did not advance exactly once")
    rng = checkpoint.get("rng_state")
    if (not isinstance(rng, dict) or not {"python", "numpy", "torch", "cuda"}.issubset(rng)
            or any(rng[key] is None for key in ("python", "numpy"))
            or not torch.is_tensor(rng["torch"]) or not rng["cuda"]
            or not all(torch.is_tensor(state) for state in rng["cuda"])
            or not all(torch.is_tensor(checkpoint.get(key))
                       for key in ("train_generator", "val_generator"))):
        raise ValueError("Diagnostic checkpoint lacks complete RNG and loader state")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aloha", "libero"), required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("The real checkpoint check requires CUDA")
    plan = json.loads((ROOT / "configs/runpod_cached_success_study.json").read_text())
    protocol = ROOT / plan["datasets"][args.dataset]["split_manifest"]
    protocol_id = json.loads(protocol.read_text())["protocol_id"]
    cache = (args.durable_root.resolve() / "success_studies"
             / ("runpod_success_stopped_v6_" + args.dataset)
             / "feature_caches" / args.dataset / "exp05/vlm_embeddings.h5")
    manifest_path = cache.with_name(cache.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {"schema": 1, "study_id": "runpod_success_stopped_v6_" + args.dataset,
                         "dataset": args.dataset, "durable_path": str(cache),
                         "experiment": "exp05", "backbone": "groot", "protocol_id": protocol_id,
                         "plan_id": "a34bb34e8023bd4d532c62789da3471ae7f9f2d68dfcc1ee22221d80e8ba7907"}
    if (not cache.is_file() or any(manifest.get(key) != value for key, value in expected_manifest.items())
            or not isinstance(manifest.get("cache_id"), str) or not manifest["cache_id"]
            or re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256", ""))) is None
            or manifest.get("size_bytes") != cache.stat().st_size):
        raise ValueError("Diagnostic cache is not the published v6 exp05 cache")
    cache_stat = cache.stat()
    cache_evidence = {"cache_path": str(cache), "published_cache_sha256": manifest["sha256"],
                      "cache_manifest_sha256": file_sha256(manifest_path),
                      "cache_size_bytes": cache_stat.st_size, "cache_mtime_ns": cache_stat.st_mtime_ns}
    check_id = (args.dataset + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                + uuid.uuid4().hex[:12])
    output = ROOT / "asset/runs/checkpoint_checks" / check_id
    durable = args.durable_root.resolve() / "checkpoint_checks" / check_id
    if output.exists() or durable.exists():
        raise FileExistsError("Diagnostic namespace already exists")
    command = [sys.executable, "-u", str(ROOT / "scripts/train.py"),
               "--dataset", args.dataset, "--exp", "exp05", "--cache", str(cache),
               "--split-manifest", str(protocol), "--output-dir", str(output),
               "--epochs", "1", "--min-epochs", "0", "--steps-per-epoch", "1",
               "--max-steps", "1", "--warmup-steps", "0", "--lr-decay-steps", "1",
               "--batch-size", "128", "--workers", "1", "--cpu-threads", "2",
               "--seed", "123", "--fresh"]
    log_event("checkpoint_check.started", check_id=check_id, command=command,
              dataset=args.dataset, production_policy=False, output_dir=str(output),
              durable_dir=str(durable), **cache_evidence)
    subprocess.run(command, cwd=ROOT, check=True)
    local = output / "checkpoints/last.pt"
    checkpoint = torch.load(local, map_location="cpu", weights_only=False)
    validate_one_step_checkpoint(checkpoint, protocol_id, manifest["cache_id"], args.dataset)
    revision = int(checkpoint["recovery_revision"])
    del checkpoint
    checksum = file_sha256(local)
    published = publish_recovery_revision(local, durable / "checkpoints", 1, revision)
    if file_sha256(published) != checksum:
        raise IOError("Fresh Cloud read differs from local one-step checkpoint")
    restored = output / "roundtrip/last.pt"
    atomic_copy_checkpoint(published, restored, checksum)
    checkpoint = torch.load(restored, map_location="cpu", weights_only=False)
    validate_one_step_checkpoint(checkpoint, protocol_id, manifest["cache_id"], args.dataset)
    log_event("checkpoint_check.passed", check_id=check_id, optimizer_steps=1, epoch=1,
              checkpoint_path=str(published), sha256=checksum, size_bytes=published.stat().st_size,
              restored_path=str(restored), protocol_id=protocol_id, cache_id=manifest["cache_id"],
              training_seed=123, dataset=args.dataset, production_policy=False, **cache_evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
