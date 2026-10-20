"""Prepare episode holdout and train-only statistics; never load/train a VLM."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from experiment_protocol import (digest, file_digest, protocol_arrays,
                                 split_episodes, write_json)
from scripts.run_log import log_event


def prepare(dataset, root, seed=42):
    started = time.perf_counter()
    log_event("protocol.data_scan_started", dataset=dataset, root=str(root), split_seed=seed)
    episodes, files = [], []
    if dataset == "aloha":
        import pandas as pd
        files = sorted(root.glob("data/chunk-000/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No ALOHA shards in {root}")
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        df = df.sort_values("index").reset_index(drop=True)
        log_event("protocol.parquet_loaded", shards=len(files), frames=len(df),
                  columns=len(df.columns))
        if df["index"].duplicated().any():
            raise ValueError("Duplicate source frame indices")
        ids = df["episode_index"].to_numpy()
        for ep in sorted(np.unique(ids)):
            rows = np.flatnonzero(ids == ep)
            if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
                raise ValueError("Source episodes must be contiguous")
            episodes.append({"id": int(ep), "task": "transfer_cube", "name": str(ep),
                             "start": int(rows[0]), "stop": int(rows[-1] + 1)})
        n = len(df)
    else:
        import h5py
        files = sorted(root.glob("*_demo.hdf5"))
        n = 0
        for path in files:
            file_started = time.perf_counter()
            with h5py.File(path, "r") as f:
                num_episodes = len(f["data"])
                for name in sorted(f["data"], key=lambda x: int(x.split("_")[1])):
                    count = len(f["data"][name]["actions"])
                    episodes.append({"id": len(episodes), "task": path.name,
                                     "name": name, "start": n, "stop": n + count})
                    n += count
            log_event("protocol.hdf5_file_scanned", file=path.name, episodes=num_episodes,
                      frames=n, elapsed_s=round(time.perf_counter() - file_started, 3))
        if not episodes:
            raise FileNotFoundError(f"No LIBERO demonstrations in {root}")
    log_event("protocol.episode_index_complete", episodes=len(episodes), frames=n)
    source_files = files + sorted(root.glob("videos/**/*.mp4")) if dataset == "aloha" else files
    hashes = {}
    for file_index, path in enumerate(source_files, start=1):
        hash_started = time.perf_counter()
        hashes[path.relative_to(root).as_posix()] = file_digest(path)
        log_event("protocol.source_hash_progress", file=file_index, files=len(source_files),
                  name=path.name, bytes=path.stat().st_size,
                  elapsed_s=round(time.perf_counter() - hash_started, 3))
    p = {"schema": 1, "dataset": dataset, "split_seed": seed, "n_samples": n,
         "episodes": split_episodes(episodes, seed), "normalization": {},
         "source_files": hashes}
    _, val = protocol_arrays(p)
    if dataset == "aloha":
        for column, prefix in (("observation.state", "state"), ("action", "action")):
            values = np.asarray(df[column].tolist(), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite source values: {column}")
            p["normalization"][prefix + "_mean"] = values[~val].mean(0).tolist()
            p["normalization"][prefix + "_std"] = values[~val].std(0).tolist()
        grip = np.asarray(df["observation.state"].tolist(), dtype=np.float64)[~val, 6]
        lo, hi = np.percentile(grip, [5, 95])
        p["receiver_gripper"] = {"dimension": 6, "low": float(lo), "high": float(hi),
                                  "threshold": float(lo + 0.6 * (hi - lo))}
    p["protocol_id"] = digest(p)
    log_event("protocol.data_scan_complete", dataset=dataset, frames=n,
              episodes=len(episodes), protocol_id=p["protocol_id"],
              elapsed_s=round(time.perf_counter() - started, 3))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["aloha", "libero"])
    ap.add_argument("--dataset-root", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    started = time.perf_counter()
    cfg = get_config(args.dataset, "exp05")
    p = prepare(args.dataset, args.dataset_root or Path(cfg.dataset_root), args.seed)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing protocol: {args.output}")
    write_json(args.output, p)
    _, val = protocol_arrays(p)
    print(f"{args.dataset}: {len(p['episodes'])} episodes; train={int((~val).sum())} "
          f"validation={int(val.sum())} frames; protocol={p['protocol_id']}")
    log_event("protocol.prepare_complete", dataset=args.dataset, output=str(args.output),
              train_frames=int((~val).sum()), validation_frames=int(val.sum()),
              elapsed_s=round(time.perf_counter() - started, 3))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_event("protocol.prepare_failed", level="ERROR",
                  error_type=type(exc).__name__, error=str(exc))
        raise
