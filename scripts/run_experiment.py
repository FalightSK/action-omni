"""Run one episode-holdout experiment: reusable cache -> checks -> seeded training.

Run from any directory with the research Python environment. Repeating the same
command reuses the cache and resumes last.pt; completed runs are left intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from data.hdf5_embeddings import HDF5EmbeddingDataset
from experiment_protocol import (apply_normalization, cache_config, digest,
                                 encoder_code_id, file_digest, load_protocol,
                                 run_signature, source_assets, write_json)
from scripts.run_log import log_event


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


@contextmanager
def exclusive_run(path):
    """OS releases this lock on process exit, including forced termination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.write(b"\0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def validate_cache(path, cfg, protocol, assets):
    """Validate provenance, all finite values, geometry and live-batch parity."""
    import h5py
    import numpy as np

    started = time.perf_counter()
    log_event("cache_validation.started", cache=str(path), dataset=protocol["dataset"],
              expected_samples=protocol["n_samples"])
    ds = HDF5EmbeddingDataset(path, expect_layers=cfg.vlm_extract_layers,
                              protocol=protocol, config=cfg)
    if len(ds) != protocol["n_samples"]:
        raise ValueError("Cache length differs from protocol")
    if not np.isfinite(ds.states.numpy()).all() or not np.isfinite(ds.actions.numpy()).all():
        raise ValueError("Nonfinite cached states/actions")
    n, s, h = len(ds), cfg.img_seq_len, cfg.vlm_hidden_size
    shape = (n, s, h) if cfg.n_vlm_layers == 1 else (n, cfg.n_vlm_layers, s, h)
    with h5py.File(path, "r") as f:
        if (not bool(f.attrs.get("complete", False))
                or json.loads(f.attrs.get("source_assets", "null")) != assets):
            raise ValueError("Incomplete cache or source weights/processor mismatch")
        if (f["embeddings"].shape != shape or f["img_masks"].shape != (n, s)
                or ds.states.shape != (n, cfg.state_dim)
                or ds.actions.shape != (n, cfg.action_horizon, cfg.action_dim)):
            raise ValueError("Cache tensor geometry mismatch")
        parity = json.loads(f.attrs.get("batch_one_parity", "null"))
        batch = int(f.attrs.get("encoding_batch_size", 0))
        if batch < 1:
            raise ValueError("Cache lacks its encoding batch size")
        last = ((n - 1) // batch) * batch
        expected = sorted({0, min(batch, n) - 1, last, n - 1})
        if (not parity or sorted(set(parity["rows"])) != expected
                or parity["max_absolute_difference"] != 0 or not parity["masks_equal"]):
            raise ValueError("Cache lacks passing first/last batch-one parity checks")
        blocks = (n + 255) // 256
        for block_index, start in enumerate(range(0, n, 256), start=1):
            raw = f["embeddings"][start:start + 256]
            # BF16 exponent bits are all ones for both infinities and NaNs.
            if np.any((raw & 0x7F80) == 0x7F80):
                raise ValueError(f"Nonfinite cached embeddings at block {start}")
            masks = f["img_masks"][start:start + 256]
            if not np.isin(masks, [0, 1]).all() or not (masks.sum(axis=1) > 0).all():
                raise ValueError("Invalid/empty cached image masks")
            log_event("cache_validation.embedding_block", block=block_index, blocks=blocks,
                      rows_checked=min(start + 256, n), total_rows=n)
    # ALOHA normalization/chunk boundaries can be checked without decoding video.
    boundaries = 0
    if protocol["dataset"] == "aloha":
        import pandas as pd
        files = sorted(Path(cfg.dataset_root).glob("data/chunk-000/file-*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        df = df.sort_values("index").reset_index(drop=True)
        actions = np.asarray(df["action"].tolist(), dtype=np.float32)
        states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
        for ep in protocol["episodes"]:
            for row in sorted({ep["start"], max(ep["start"], ep["stop"] - cfg.action_horizon), ep["stop"] - 1}):
                indices = np.minimum(np.arange(row, row + cfg.action_horizon), ep["stop"] - 1)
                target = (actions[indices] - np.array(cfg.action_mean, dtype=np.float32)) / (np.array(cfg.action_std, dtype=np.float32) + 1e-8)
                state = (states[row] - np.array(cfg.state_mean, dtype=np.float32)) / (np.array(cfg.state_std, dtype=np.float32) + 1e-8)
                if (not np.allclose(ds.actions[row].numpy(), target, atol=1e-6, rtol=1e-6)
                        or not np.allclose(ds.states[row].numpy(), state, atol=1e-6, rtol=1e-6)):
                    raise ValueError(f"Raw demonstration/cache boundary mismatch at row {row}")
                boundaries += 1
    report = {"status": "passed", "cache": str(path), "cache_id": ds.cache_id,
            "protocol_id": protocol["protocol_id"], "frames": n,
            "validation_frames": int(ds.validation_mask.sum()),
            "all_values_finite": True, "batch_one_parity": parity,
            "boundary_rows_checked": boundaries}
    log_event("cache_validation.complete", cache=str(path), samples=n,
              boundary_rows_checked=boundaries,
              elapsed_s=round(time.perf_counter() - started, 3))
    return report


def ensure_cache(cache, cfg, protocol, assets, encode):
    """Publish only fully validated files; never overwrite a finished cache."""
    started = time.perf_counter()
    log_event("cache.ensure_started", cache=str(cache), exists=cache.exists())
    cache.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_run(cache.parent / ".encoding.lock"):
        if cache.exists():
            report = validate_cache(cache, cfg, protocol, assets)
            log_event("cache.reused", cache=str(cache), elapsed_s=round(time.perf_counter() - started, 3))
            return report, True
        partial = cache.with_suffix(".partial.h5")
        if partial.exists():
            interrupted = cache.with_name(f"interrupted_{stamp()}.h5")
            partial.rename(interrupted)
            log_event("cache.interrupted_partial_preserved", old_path=str(partial),
                      preserved_path=str(interrupted))
        log_event("cache.encode_started", partial=str(partial))
        encode(partial)
        log_event("cache.encode_complete", partial=str(partial), bytes=partial.stat().st_size)
        report = validate_cache(partial, cfg, protocol, assets)
        partial.rename(cache)
        report["cache"] = str(cache)
        log_event("cache.published", cache=str(cache), bytes=cache.stat().st_size,
                  elapsed_s=round(time.perf_counter() - started, 3))
        return report, False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("aloha", "libero"))
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "asset/runs/episode_holdout")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, help="Override maximum training epochs")
    parser.add_argument("--min-epochs", type=int, help="Override the early-stopping epoch floor")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.workers < 0:
        parser.error("seed/workers must be nonnegative")
    os.chdir(ROOT)
    cfg = get_config(args.dataset, args.exp)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.min_epochs is not None:
        cfg.min_epochs = args.min_epochs
    if cfg.num_epochs < 1 or not 0 <= cfg.min_epochs <= cfg.num_epochs:
        parser.error("Require epochs > 0 and 0 <= min-epochs <= epochs")
    protocol = load_protocol(args.split_manifest, args.dataset)
    apply_normalization(cfg, protocol)
    experiment = Path(cfg.output_dir).name
    run = args.run_root.resolve() / args.dataset / experiment / f"seed_{args.seed:03d}"
    cfg.output_dir = str(run)
    with exclusive_run(run / ".run.lock"):
        attempt = run / "attempts" / stamp()
        attempt.mkdir(parents=True)
        status = run / "pipeline.json"
        state = {"status": "preflight", "launcher_pid": os.getpid(),
                 "dataset": args.dataset, "experiment": args.exp, "training_seed": args.seed,
                 "started_utc": stamp(), "attempt": str(attempt), "stages": [],
                 "evaluation_status": "not run"}

        def save():
            state["updated_utc"] = stamp()
            write_json(attempt / "pipeline.json", state)
            write_json(status, state)

        def execute(stage, command):
            state["status"] = stage
            item = {"stage": stage, "command": command, "log": str(attempt / f"{stage}.log")}
            state["stages"].append(item)
            save()
            with Path(item["log"]).open("w", encoding="utf-8") as log:
                env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                item["pid"] = process.pid
                save()
                item["exit_code"] = process.wait()
            save()
            if item["exit_code"]:
                raise RuntimeError(f"{stage} failed; see {item['log']}")

        save()
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; full experiment was not started")
            write_json(attempt / "environment.json", {"python": sys.version, "executable": sys.executable,
                       "torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0), "config": asdict(cfg)})
            for relative, expected in protocol["source_files"].items():
                if file_digest(Path(cfg.dataset_root) / relative) != expected:
                    raise ValueError(f"Source data changed: {relative}")
            assets = source_assets(cfg)
            identity = {"protocol_id": protocol["protocol_id"], "config": cache_config(cfg),
                        "encoder_code_id": encoder_code_id(), "source_assets": assets}
            cache = ROOT / "asset/cache" / args.dataset / cfg.vlm_backbone / digest(identity) / "vlm_embeddings.h5"
            state["cache"] = str(cache)
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_json(cache.parent / "identity.json", identity)
            with zipfile.ZipFile(attempt / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for folder in ("models", "configs", "data", "envs", "scripts", "tests"):
                    for path in (ROOT / folder).rglob("*.py"):
                        archive.write(path, path.relative_to(ROOT))
                for name in ("experiment_protocol.py", "requirements.txt", "plan.md"):
                    archive.write(ROOT / name, name)
            report, reused = ensure_cache(cache, cfg, protocol, assets, lambda partial: execute(
                "precompute", [sys.executable, "-u", str(ROOT / "scripts/precompute.py"),
                               "--dataset", args.dataset, "--exp", args.exp,
                               "--split-manifest", str(args.split_manifest.resolve()), "--output", str(partial)]))
            write_json(attempt / "cache_validation.json", report)
            state.update(cache_reused=reused, status="cache_validated")
            save()
            manifest = run / "run_manifest.json"
            previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
            if previous.get("status") == "complete":
                if (previous.get("run_signature") != run_signature(cfg, protocol, args.seed)
                        or previous.get("cache_id") != report["cache_id"]
                        or not all((run / "checkpoints" / f"{name}.pt").is_file() for name in ("best", "last", "final"))):
                    raise ValueError("Completed run does not match this request; use another run root")
            else:
                execute("training", [sys.executable, "-u", str(ROOT / "scripts/train.py"),
                        "--dataset", args.dataset, "--exp", args.exp, "--seed", str(args.seed),
                        "--split-manifest", str(args.split_manifest.resolve()), "--cache", str(cache),
                        "--output-dir", str(run), "--workers", str(args.workers),
                        "--epochs", str(cfg.num_epochs), "--min-epochs", str(cfg.min_epochs)])
            result = json.loads(manifest.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                raise RuntimeError("Trainer did not report successful completion")
            state.update(status="training_complete", completed_epoch=result["completed_epoch"],
                         optimizer_steps=result["optimizer_steps"], best_validation_loss=result["best_validation_loss"])
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (attempt / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            save()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_event("experiment.failed", level="ERROR", error_type=type(exc).__name__,
                  error=str(exc))
        raise
