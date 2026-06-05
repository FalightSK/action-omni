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
import sys
import time
from pathlib import Path
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from models.vla import VLAModel


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

    cache["actions"] = new_actions
    if cfg.state_dim > 2:
        print("  Re-building 6D state history from updated actions …")
        episode_ids_n = episode_ids[:N]
        states_2d     = cache["states"][:, :2]
        cache["states"] = _build_6d_states(states_2d, new_actions, episode_ids_n)

    cache["n_samples"] = N
    torch.save(cache, out_path)
    print(f"  Saved updated cache → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute VLM embeddings")
    parser.add_argument("--dataset",          type=str, default="pusht",
                        help="Dataset name: pusht, aloha, language_table, ood")
    parser.add_argument("--exp",              type=str, default="exp02a",
                        help="Experiment ID: exp01, exp02a, exp03, exp04, …")
    parser.add_argument("--max-samples",      type=int, default=None,
                        help="Encode only this many samples (quick test)")
    parser.add_argument("--output",           type=str, default=None,
                        help="Override output cache path")
    parser.add_argument("--recompute-actions", action="store_true",
                        help="Rebuild action chunks without re-running VLM")
    args = parser.parse_args()

    cfg = get_config(args.dataset, args.exp)
    print(f"Dataset    : {args.dataset}  |  Exp: {args.exp}")
    print(f"Output dir : {cfg.output_dir}")
    device   = cfg.get_device()
    out_path = Path(args.output or cfg.embeddings_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.recompute_actions:
        _recompute_actions_only(cfg, out_path)
        return

    n_layers = cfg.n_vlm_layers
    print(f"Device     : {device}")
    print(f"VLM layers : {cfg.vlm_extract_layers}  ({n_layers} scales)")
    print(f"State dim  : {cfg.state_dim}")

    # ── Build dataset ──────────────────────────────────────────────────────
    print("\n[1/3] Loading dataset …")
    full_dataset = _get_raw_dataset(args.dataset, cfg)

    if args.max_samples and args.max_samples < len(full_dataset):
        dataset = Subset(full_dataset, list(range(args.max_samples)))
        print(f"   Quick mode: {args.max_samples:,} of {len(full_dataset):,} samples.")
        is_full = False
    else:
        dataset = full_dataset
        is_full = True

    loader = DataLoader(dataset, batch_size=cfg.embed_batch_size,
                        shuffle=False, num_workers=0, collate_fn=_custom_collate)

    # ── Load VLM ───────────────────────────────────────────────────────────
    print("\n[2/3] Loading VLM …")
    model = VLAModel(cfg)
    model.vlm.to(device).eval()

    N        = len(dataset)
    H_size   = cfg.vlm_hidden_size
    S        = cfg.img_seq_len
    horizon  = cfg.action_horizon
    action_d = cfg.action_dim
    # PushT yields 2D position (optionally extended to 6D below). Other datasets
    # (e.g. ALOHA) return their full state vector directly from the loader.
    state_d  = 2 if args.dataset == "pusht" else cfg.state_dim

    est_min = N / 4.5 / 60
    if n_layers > 1:
        gb = N * n_layers * S * H_size * 2 / 1e9
        print(f"\n[3/3] Encoding {N:,} frames → multi-scale (N, {n_layers}, {S}, {H_size})")
        print(f"      Est. time: {est_min:.0f} min  |  Cache size: ~{gb:.1f} GB")
        all_embeds = torch.zeros(N, n_layers, S, H_size, dtype=torch.bfloat16)
    else:
        gb = N * S * H_size * 2 / 1e9
        print(f"\n[3/3] Encoding {N:,} frames → single layer (N, {S}, {H_size})")
        print(f"      Est. time: {est_min:.0f} min  |  Cache size: ~{gb:.1f} GB")
        all_embeds = torch.zeros(N, S, H_size, dtype=torch.bfloat16)

    all_masks   = torch.zeros(N, S,              dtype=torch.bool)
    all_states  = torch.zeros(N, state_d,         dtype=torch.float32)
    all_actions = torch.zeros(N, horizon, action_d, dtype=torch.float32)

    ptr = 0
    t0  = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="VLM encoding", unit="batch"):
            B = len(batch["images"])
            vlm_inputs = model.build_vlm_inputs(
                images=batch["images"], task_texts=batch["task_texts"], device=device)
            tokens, img_mask = model.encode_vlm(vlm_inputs)

            if tokens.ndim == 4:
                actual_S = tokens.shape[2]
                sl = min(actual_S, S)
                all_embeds[ptr:ptr+B, :, :sl] = tokens[:, :, :sl].to("cpu", dtype=torch.bfloat16)
            else:
                actual_S = tokens.shape[1]
                sl = min(actual_S, S)
                all_embeds[ptr:ptr+B, :sl] = tokens[:, :sl].to("cpu", dtype=torch.bfloat16)

            all_masks [ptr:ptr+B, :sl] = img_mask[:, :sl].to("cpu")
            all_states [ptr:ptr+B]     = batch["states"]
            all_actions[ptr:ptr+B]     = batch["actions"]
            ptr += B

    elapsed = time.time() - t0
    print(f"\n   Finished in {elapsed/60:.1f} min  ({N/elapsed:.1f} samples/sec)")

    if args.dataset == "pusht" and cfg.state_dim > 2:
        print(f"\n   Building {cfg.state_dim}D state with delta history (PushT covariate-shift expts) …")
        episode_ids = _load_episode_ids(cfg, N)
        all_states  = _build_6d_states(all_states, all_actions, episode_ids)
        print(f"   State tensor: {tuple(all_states.shape)}")

    # ── Save ───────────────────────────────────────────────────────────────
    fmt = f"v3_multi_scale_{n_layers}layers" if n_layers > 1 else "v2_full_sequence"
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
    if is_full:
        print(f"\n   Full dataset encoded. Run:")
        print(f"   python scripts/train.py --dataset {args.dataset} --exp {args.exp}")


if __name__ == "__main__":
    main()
