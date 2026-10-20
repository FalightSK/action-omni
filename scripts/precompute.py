"""
scripts/precompute.py
──────────────────────
Step 1 of 3: Pre-compute and cache VLM embeddings for a dataset.

Run ONCE before training. The frozen VLM is run across the full dataset and
token sequences are saved to disk. Training then reads from this cache — much
faster than re-running the VLM every epoch.

Usage
─────
  # Full dataset (run overnight):
  python scripts/precompute.py --dataset pusht --exp exp02a

  # Quick subset for testing:
  python scripts/precompute.py --dataset pusht --exp exp02a --max-samples 2000

  # Re-cache actions only (no VLM re-run, e.g. when action_horizon changes):
  python scripts/precompute.py --dataset pusht --exp exp02a --recompute-actions

  # Override output path:
  python scripts/precompute.py --dataset pusht --exp exp02a --output /path/to/cache.pt

Supported datasets / experiments:
  pusht / exp01       — Exp01 MLP baseline
  pusht / exp02a      — Exp02a DiT (BEST, re-uses exp01 cache)
  pusht / exp03       — Exp03 multi-scale (~12.8 GB cache)
  pusht / exp04       — Exp04 ViT ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from models.vla import VLAModel
from data.hdf5_embeddings import is_hdf5_path, HDF5EmbeddingWriter
from scripts.run_log import log_event


# ── Dataset loader factory ─────────────────────────────────────────────────────

def _get_raw_dataset(dataset_name: str, cfg):
    if dataset_name == "pusht":
        from data.pusht import PushTDataset
        return PushTDataset(cfg.dataset_path, cfg, cache_frames=True)
    if dataset_name == "aloha":
        from data.aloha import AlohaDataset
        return AlohaDataset(cfg)
    if dataset_name == "language_table":
        from data.language_table import LanguageTableDataset
        return LanguageTableDataset(cfg)
    if dataset_name == "libero":
        from data.libero import LiberoDataset
        return LiberoDataset(cfg)
    raise NotImplementedError(f"Dataset {dataset_name!r} not yet implemented. "
                              f"Add a loader in data/{dataset_name}/dataset.py.")


def _custom_collate(batch):
    return {
        "images":     [item["image"]     for item in batch],
        "states":     torch.stack([item["state"]   for item in batch]),
        "actions":    torch.stack([item["actions"] for item in batch]),
        "task_texts": [item["task_text"] for item in batch],
        "idxs":       torch.tensor([item["idx"]    for item in batch]),
    }


# ── Episode ID helpers (PushT-specific, used for 6D state building) ────────────

def _load_episode_ids(cfg, N: int) -> np.ndarray:
    data_file = Path(cfg.dataset_path) / "data/chunk-000/file-000.parquet"
    df = pd.read_parquet(data_file)
    return df["episode_index"].to_numpy(dtype=np.int64)[:N]


def _build_6d_states(states_2d: torch.Tensor, actions: torch.Tensor,
                     episode_ids: np.ndarray) -> torch.Tensor:
    N = len(states_2d)
    states_6d = torch.zeros(N, 6, dtype=torch.float32)
    states_6d[:, :2] = states_2d
    for i in range(N):
        if i > 0 and episode_ids[i - 1] == episode_ids[i]:
            states_6d[i, 2:4] = actions[i - 1, 0]
        if i > 1 and episode_ids[i - 2] == episode_ids[i]:
            states_6d[i, 4:6] = actions[i - 2, 0]
    return states_6d


# ── Recompute helpers ──────────────────────────────────────────────────────────

def _recompute_actions_only(cfg, out_path: Path) -> None:
    if not out_path.exists():
        print(f"[ERROR] Cache not found at {out_path}.")
        sys.exit(1)

    print(f"[--recompute-actions] Loading existing cache from {out_path} …")
    cache = torch.load(out_path, weights_only=True, map_location="cpu")
    N = cache["n_samples"]
    print(f"  {N:,} samples | new action_horizon = {cfg.action_horizon}")

    data_file = Path(cfg.dataset_path) / "data/chunk-000/file-000.parquet"
    df = pd.read_parquet(data_file)
    episode_ids = df["episode_index"].to_numpy(dtype=np.int64)

    action_mean = np.array(cfg.action_mean, dtype=np.float32)
    action_std  = np.array(cfg.action_std,  dtype=np.float32)
    use_rel = getattr(cfg, "use_relative_actions", False)

    def norm(a):
        return (a - action_mean) / (action_std + 1e-8)

    H, D = cfg.action_horizon, cfg.action_dim
    new_actions = torch.zeros(N, H, D, dtype=torch.float32)

    print("  Re-building action chunks …")
    started = time.perf_counter()
    report_every = max(1, N // 20)
    for idx in range(N):
        row = df.iloc[idx]
        ep  = episode_ids[idx]
        state_now = np.array(row["observation.state"], dtype=np.float32)
        last_a    = np.array(row["action"], dtype=np.float32)
        for k in range(H):
            fi = idx + k
            if fi < len(df) and episode_ids[fi] == ep:
                a = np.array(df.iloc[fi]["action"], dtype=np.float32)
                last_a = a
                if use_rel:
                    s = np.array(df.iloc[fi]["observation.state"], dtype=np.float32)
                    a = a - s
            else:
                a = last_a
                if use_rel:
                    a = last_a - state_now
            new_actions[idx, k] = torch.from_numpy(norm(a))
        if (idx + 1) % report_every == 0 or idx + 1 == N:
            elapsed = time.perf_counter() - started
            log_event("precompute.action_chunk_progress", completed=idx + 1, total=N,
                      elapsed_s=round(elapsed, 3),
                      samples_per_s=round((idx + 1) / max(elapsed, 1e-9), 2),
                      eta_s=round((N - idx - 1) * elapsed / (idx + 1), 1))

    cache["actions"] = new_actions
    if cfg.state_dim > 2:
        print("  Re-building 6D state history from updated actions …")
        episode_ids_n = episode_ids[:N]
        states_2d     = cache["states"][:, :2]
        cache["states"] = _build_6d_states(states_2d, new_actions, episode_ids_n)

    cache["n_samples"] = N
    torch.save(cache, out_path)
    print(f"  Saved updated cache → {out_path}")
    log_event("precompute.action_chunks_complete", output=str(out_path), samples=N,
              elapsed_s=round(time.perf_counter() - started, 3),
              bytes=out_path.stat().st_size)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pre-compute VLM embeddings")
    parser.add_argument("--dataset",          type=str, default="pusht",
                        help="Dataset name: pusht, aloha, language_table, libero, ood")
    parser.add_argument("--exp",              type=str, default="exp02a",
                        help="Experiment ID: exp01, exp02a, exp03, exp04, …")
    parser.add_argument("--max-samples",      type=int, default=None,
                        help="Encode only this many samples (quick test)")
    parser.add_argument("--output",           type=str, default=None,
                        help="Override output cache path")
    parser.add_argument("--recompute-actions", action="store_true",
                        help="Rebuild action chunks without re-running VLM")
    parser.add_argument("--split-manifest", type=Path,
                        help="Opt in to the prepared episode-holdout protocol (full HDF5 only)")
    parser.add_argument("--log-every-batches", type=int, default=10,
                        help="Emit VLM encoding progress every N batches")
    args = parser.parse_args(argv)
    if args.log_every_batches < 1:
        parser.error("--log-every-batches must be positive")

    cfg = get_config(args.dataset, args.exp)
    run_started = time.perf_counter()
    print(f"Dataset    : {args.dataset}  |  Exp: {args.exp}")
    print(f"Output dir : {cfg.output_dir}")
    device   = cfg.get_device()
    out_path = Path(args.output or cfg.embeddings_cache)
    log_event("precompute.started", dataset=args.dataset, experiment=args.exp,
              device=str(device), output=str(out_path), max_samples=args.max_samples,
              split_manifest=str(args.split_manifest) if args.split_manifest else None)
    protocol = None
    if args.split_manifest:
        from experiment_protocol import (apply_normalization, load_protocol,
                                         file_digest)
        protocol_started = time.perf_counter()
        log_event("precompute.protocol_validation_started", manifest=str(args.split_manifest))
        protocol = load_protocol(args.split_manifest, args.dataset)
        if args.max_samples is not None or args.recompute_actions or not is_hdf5_path(out_path):
            raise ValueError("Protocol encoding requires full HDF5; no action-only rewrites/subsets")
        if out_path.exists():
            raise FileExistsError(f"Refusing to overwrite protocol cache: {out_path}")
        for relative, expected in protocol["source_files"].items():
            if file_digest(Path(cfg.dataset_root) / relative) != expected:
                raise ValueError(f"Source data changed: {relative}")
        apply_normalization(cfg, protocol)
        log_event("precompute.protocol_validation_complete", protocol_id=protocol["protocol_id"],
                  source_files=len(protocol["source_files"]),
                  elapsed_s=round(time.perf_counter() - protocol_started, 3))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.recompute_actions:
        _recompute_actions_only(cfg, out_path)
        log_event("precompute.complete", mode="actions_only",
                  elapsed_s=round(time.perf_counter() - run_started, 3))
        return

    n_layers = cfg.n_vlm_layers
    print(f"Device     : {device}")
    print(f"VLM layers : {cfg.vlm_extract_layers}  ({n_layers} scales)")
    print(f"State dim  : {cfg.state_dim}")

    # ── Build dataset ──────────────────────────────────────────────────────
    print("\n[1/3] Loading dataset …")
    data_started = time.perf_counter()
    log_event("data_collection.started", dataset=args.dataset, root=str(cfg.dataset_root))
    full_dataset = _get_raw_dataset(args.dataset, cfg)
    if protocol:
        from experiment_protocol import protocol_arrays
        episode_ids, validation_mask = protocol_arrays(protocol)
        if (len(full_dataset) != protocol["n_samples"]
                or not np.array_equal(full_dataset.episode_ids, episode_ids)):
            raise ValueError("Raw dataset order differs from the prepared protocol")

    if args.max_samples and args.max_samples < len(full_dataset):
        dataset = Subset(full_dataset, list(range(args.max_samples)))
        print(f"   Quick mode: {args.max_samples:,} of {len(full_dataset):,} samples.")
        is_full = False
    else:
        dataset = full_dataset
        is_full = True
    log_event("data_collection.complete", dataset=args.dataset, samples=len(dataset),
              full_dataset=is_full,
              episodes=int(len(np.unique(full_dataset.episode_ids)))
              if hasattr(full_dataset, "episode_ids") else None,
              elapsed_s=round(time.perf_counter() - data_started, 3))

    loader = DataLoader(dataset, batch_size=cfg.embed_batch_size,
                        shuffle=False, num_workers=0, collate_fn=_custom_collate)
    log_event("data_loader.ready", batches=len(loader), batch_size=cfg.embed_batch_size, workers=0)

    # ── Load VLM ───────────────────────────────────────────────────────────
    print("\n[2/3] Loading VLM …")
    model_started = time.perf_counter()
    log_event("backbone_load.started", backbone=cfg.vlm_backbone)
    model = VLAModel(cfg)
    model.vlm.to(device).eval()
    log_event("backbone_load.complete", backbone=cfg.vlm_backbone,
              elapsed_s=round(time.perf_counter() - model_started, 3),
              parameters=sum(p.numel() for p in model.vlm.parameters()))

    N        = len(dataset)
    H_size   = cfg.vlm_hidden_size
    S        = cfg.img_seq_len
    horizon  = cfg.action_horizon
    action_d = cfg.action_dim
    # PushT yields 2D position (optionally extended to 6D below). Other datasets
    # (e.g. ALOHA) return their full state vector directly from the loader.
    state_d  = 2 if args.dataset == "pusht" else cfg.state_dim

    use_hdf5 = is_hdf5_path(out_path)
    est_min  = N / 4.5 / 60
    gb = (N * (n_layers if n_layers > 1 else 1) * S * H_size * 2) / 1e9
    scale_str = (f"multi-scale (N, {n_layers}, {S}, {H_size})" if n_layers > 1
                 else f"single layer (N, {S}, {H_size})")
    sink = "HDF5 stream (low RAM)" if use_hdf5 else "in-RAM .pt"
    print(f"\n[3/3] Encoding {N:,} frames → {scale_str}")
    print(f"      Est. time: {est_min:.0f} min  |  Cache size: ~{gb:.1f} GB  |  sink: {sink}")

    fmt = f"v3_multi_scale_{n_layers}layers" if n_layers > 1 else "v2_full_sequence"

    # HDF5 streaming is incompatible with the PushT post-hoc 6D state build (which
    # needs the full state/action tensors in RAM). LT/ALOHA never use 6D, so guard.
    if use_hdf5 and args.dataset == "pusht" and cfg.state_dim > 2:
        print("[ERROR] HDF5 streaming does not support PushT 6D post-hoc state build. "
              "Use a .pt cache for that experiment.")
        sys.exit(1)

    # ── Allocate sink (preallocated tensors for .pt, streaming writer for HDF5) ──
    writer = None
    if use_hdf5:
        attrs = {"format": fmt, "dataset": args.dataset, "exp_id": args.exp,
                 "vlm_extract_layers": list(cfg.vlm_extract_layers),
                 "encoding_batch_size": cfg.embed_batch_size,
                 "state_dim": cfg.state_dim, "full_dataset": bool(is_full)}
        if protocol:
            from experiment_protocol import cache_config, digest, encoder_code_id, source_assets
            attrs.update(protocol_id=protocol["protocol_id"],
                         cache_config=json.dumps(cache_config(cfg), sort_keys=True),
                         encoder_code_id=encoder_code_id(),
                         source_assets=json.dumps(source_assets(cfg), sort_keys=True))
            attrs["cache_id"] = digest(attrs)
        writer = HDF5EmbeddingWriter(
            out_path, N=N, S=S, H=H_size, state_dim=state_d,
            horizon=horizon, action_dim=action_d, n_layers=n_layers,
            attrs=attrs)
        if protocol:
            writer.f.create_dataset("episode_ids", data=episode_ids)
            writer.f.create_dataset("validation_mask", data=validation_mask.astype(np.uint8))
    else:
        if n_layers > 1:
            all_embeds = torch.zeros(N, n_layers, S, H_size, dtype=torch.bfloat16)
        else:
            all_embeds = torch.zeros(N, S, H_size, dtype=torch.bfloat16)
        all_masks   = torch.zeros(N, S,               dtype=torch.bool)
        all_states  = torch.zeros(N, state_d,          dtype=torch.float32)
        all_actions = torch.zeros(N, horizon, action_d, dtype=torch.float32)

    ptr = 0
    batch_one_parity = {"rows": [], "max_absolute_difference": 0.0, "masks_equal": True}
    t0 = time.perf_counter()
    last_batch_end = t0
    total_batches = len(loader)
    log_event("feature_encoding.started", samples=N, batches=total_batches,
              batch_size=cfg.embed_batch_size, sink=sink, device=str(device))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch_started = time.perf_counter()
            data_wait_s = batch_started - last_batch_end
            B = len(batch["images"])
            encode_started = time.perf_counter()
            vlm_inputs = model.build_vlm_inputs(
                images=batch["images"], task_texts=batch["task_texts"], device=device)
            tokens, img_mask = model.encode_vlm(vlm_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            encode_s = time.perf_counter() - encode_started

            if protocol and (ptr == 0 or ptr + B == N):
                # Compare actual batched features with the live batch-one path.
                # Do this before allowing the cache to be marked complete.
                for j in sorted({0, B - 1}):
                    single_inputs = model.build_vlm_inputs(
                        [batch["images"][j]], [batch["task_texts"][j]], device)
                    single, single_mask = model.encode_vlm(single_inputs)
                    delta = (tokens[j].to(torch.bfloat16).float()
                             - single[0].to(torch.bfloat16).float()).abs().max().item()
                    if delta != 0 or not torch.equal(img_mask[j], single_mask[0]):
                        raise ValueError(f"Cache/live batch-one feature mismatch at row {ptr+j}: {delta}")
                    batch_one_parity["rows"].append(ptr + j)
                print(f"   Cache/live batch-one parity passed at rows {batch_one_parity['rows']}")

            if tokens.ndim == 4:
                actual_S = tokens.shape[2]
                sl = min(actual_S, S)
            else:
                actual_S = tokens.shape[1]
                sl = min(actual_S, S)

            write_started = time.perf_counter()
            if use_hdf5:
                # Build a (B, [L,] S, H) bf16 buffer padded/truncated to S, plus mask.
                if n_layers > 1:
                    emb_buf = torch.zeros(B, n_layers, S, H_size, dtype=torch.bfloat16)
                    emb_buf[:, :, :sl] = tokens[:, :, :sl].to("cpu", dtype=torch.bfloat16)
                else:
                    emb_buf = torch.zeros(B, S, H_size, dtype=torch.bfloat16)
                    emb_buf[:, :sl] = tokens[:, :sl].to("cpu", dtype=torch.bfloat16)
                mask_buf = torch.zeros(B, S, dtype=torch.bool)
                mask_buf[:, :sl] = img_mask[:, :sl].to("cpu")
                writer.write(ptr, emb_buf, mask_buf, batch["states"], batch["actions"])
            else:
                if n_layers > 1:
                    all_embeds[ptr:ptr+B, :, :sl] = tokens[:, :, :sl].to("cpu", dtype=torch.bfloat16)
                else:
                    all_embeds[ptr:ptr+B, :sl] = tokens[:, :sl].to("cpu", dtype=torch.bfloat16)
                all_masks [ptr:ptr+B, :sl] = img_mask[:, :sl].to("cpu")
                all_states [ptr:ptr+B]     = batch["states"]
                all_actions[ptr:ptr+B]     = batch["actions"]
            ptr += B
            batch_end = time.perf_counter()
            elapsed = batch_end - t0
            if (batch_index == 1 or batch_index % args.log_every_batches == 0
                    or batch_index == total_batches):
                log_event("feature_encoding.batch_progress", batch=batch_index, batches=total_batches,
                          completed_samples=ptr, total_samples=N,
                          data_wait_s=round(data_wait_s, 3), encoder_s=round(encode_s, 3),
                          cache_write_s=round(batch_end - write_started, 3),
                          elapsed_s=round(elapsed, 3), samples_per_s=round(ptr / max(elapsed, 1e-9), 2),
                          eta_s=round((N - ptr) * elapsed / max(ptr, 1), 1))
            last_batch_end = time.perf_counter()

    elapsed = time.perf_counter() - t0
    print(f"\n   Finished in {elapsed/60:.1f} min  ({N/elapsed:.1f} samples/sec)")

    # ── Save ───────────────────────────────────────────────────────────────
    if use_hdf5:
        if protocol:
            writer.f.attrs["batch_one_parity"] = json.dumps(batch_one_parity)
        writer.close()
        mb = out_path.stat().st_size / 1e6
        print(f"   Saved → {out_path}  ({mb:.0f} MB, HDF5 streamed)")
    else:
        if args.dataset == "pusht" and cfg.state_dim > 2:
            print(f"\n   Building {cfg.state_dim}D state with delta history (PushT covariate-shift expts) …")
            episode_ids = _load_episode_ids(cfg, N)
            all_states  = _build_6d_states(all_states, all_actions, episode_ids)
            print(f"   State tensor: {tuple(all_states.shape)}")

        payload = {
            "embeddings":         all_embeds,
            "img_masks":          all_masks,
            "states":             all_states,
            "actions":            all_actions,
            "n_samples":          N,
            "full_dataset":       is_full,
            "format":             fmt,
            "dataset":            args.dataset,
            "exp_id":             args.exp,
            "vlm_extract_layers": list(cfg.vlm_extract_layers),
            "state_dim":          cfg.state_dim,
        }
        torch.save(payload, out_path)
        mb = out_path.stat().st_size / 1e6
        print(f"   Saved → {out_path}  ({mb:.0f} MB)")

    log_event("precompute.complete", dataset=args.dataset, experiment=args.exp,
              samples=N, cache=str(out_path), cache_mb=round(mb, 2),
              elapsed_s=round(time.perf_counter() - run_started, 3),
              samples_per_s=round(N / max(elapsed, 1e-9), 2), full_dataset=is_full)

    if is_full:
        print(f"\n   Full dataset encoded. Run:")
        print(f"   python scripts/train.py --dataset {args.dataset} --exp {args.exp}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_event("precompute.failed", level="ERROR", error_type=type(exc).__name__, error=str(exc))
        raise
