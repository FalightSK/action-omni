"""
precompute_embeddings.py
════════════════════════
Run **once** before training.

Experiment 2 (B+C) cache format — v3:
    embeddings : (N, n_layers, seq_len, 1024)  bfloat16   ← multi-scale (3 layers)
    img_masks  : (N, seq_len)                  bool
    states     : (N, 6)                        float32    ← extended w/ delta history
    actions    : (N, H, 2)                     float32

Performance note
----------------
  M1 Mac with output_hidden_states=True: ~4.5 samples/sec (slightly slower than v2).
  Full dataset (25,650 frames): ~95 minutes.
  3-layer cache at bfloat16: ~12.8 GB.

Usage
-----
  # Full dataset (run overnight):
  python precompute_embeddings.py

  # Quick subset:
  python precompute_embeddings.py --max-samples 2000

  # Re-cache actions only (no VLM re-run, e.g. when action_horizon changes):
  python precompute_embeddings.py --recompute-actions

  # Re-cache states only (no VLM re-run, e.g. to update delta history):
  python precompute_embeddings.py --recompute-states

  # Explicit output path:
  python precompute_embeddings.py --output asset/result_exp2/vlm_embeddings.pt
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config_loader import get_config
from data.pusht_dataset import PushTDataset
from models.vla import VLAModel


# ──────────────────────────────────────────────────────────────────────────────

def custom_collate(batch):
    return {
        "images":    [item["image"]     for item in batch],
        "states":    torch.stack([item["state"]   for item in batch]),
        "actions":   torch.stack([item["actions"] for item in batch]),
        "task_texts":[item["task_text"] for item in batch],
        "idxs":      torch.tensor([item["idx"]    for item in batch]),
    }


def _load_episode_ids(cfg: VLAConfig, N: int) -> np.ndarray:
    data_file = Path(cfg.dataset_path) / "data/chunk-000/file-000.parquet"
    df = pd.read_parquet(data_file)
    return df["episode_index"].to_numpy(dtype=np.int64)[:N]


def _build_6d_states(states_2d: torch.Tensor, actions: torch.Tensor,
                     episode_ids: np.ndarray) -> torch.Tensor:
    """
    Extend (N, 2) normalised position states to (N, 6) by prepending the
    previous 2 executed normalised deltas.

    state_6d[i] = [pos_norm_x, pos_norm_y,
                   prev_delta_1_norm_x, prev_delta_1_norm_y,  (1 step ago)
                   prev_delta_2_norm_x, prev_delta_2_norm_y]  (2 steps ago)

    The first action in each horizon (actions[i, 0]) is the delta executed at
    step i (from state[i] to state[i+1]) — already normalised.  Zeros are used
    at episode boundaries.
    """
    N = len(states_2d)
    states_6d = torch.zeros(N, 6, dtype=torch.float32)
    states_6d[:, :2] = states_2d

    for i in range(N):
        # prev delta 1 — 1 step ago (same episode)
        if i > 0 and episode_ids[i - 1] == episode_ids[i]:
            states_6d[i, 2:4] = actions[i - 1, 0]
        # prev delta 2 — 2 steps ago (same episode)
        if i > 1 and episode_ids[i - 2] == episode_ids[i]:
            states_6d[i, 4:6] = actions[i - 2, 0]

    return states_6d


def _recompute_actions_only(cfg: VLAConfig, out_path: Path) -> None:
    if not out_path.exists():
        print(f"[ERROR] Cache not found at {out_path}.")
        sys.exit(1)

    print(f"[--recompute-actions] Loading existing cache from {out_path} …")
    cache = torch.load(out_path, weights_only=True, map_location="cpu")
    N = cache["n_samples"]
    print(f"  {N:,} samples | new action_horizon = {cfg.action_horizon}")

    data_file = Path(cfg.dataset_path) / "data/chunk-000/file-000.parquet"
    df        = pd.read_parquet(data_file)
    episode_ids = df["episode_index"].to_numpy(dtype=np.int64)

    action_mean = np.array(cfg.action_mean, dtype=np.float32)
    action_std  = np.array(cfg.action_std,  dtype=np.float32)
    use_rel = getattr(cfg, "use_relative_actions", False)

    def norm(a):
        return (a - action_mean) / (action_std + 1e-8)

    H = cfg.action_horizon
    D = cfg.action_dim
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

    # also rebuild 6D states from updated actions
    if cfg.state_dim > 2:
        print("  Re-building 6D state history from updated actions …")
        episode_ids_n = episode_ids[:N]
        states_2d     = cache["states"][:, :2]
        cache["states"] = _build_6d_states(states_2d, new_actions, episode_ids_n)

    cache["n_samples"] = N
    torch.save(cache, out_path)
    print(f"  Saved updated cache → {out_path}")


def _recompute_states_only(cfg: VLAConfig, out_path: Path) -> None:
    """Rebuild the state tensor (6D) without re-running the VLM."""
    if not out_path.exists():
        print(f"[ERROR] Cache not found at {out_path}.")
        sys.exit(1)

    print(f"[--recompute-states] Loading existing cache from {out_path} …")
    cache = torch.load(out_path, weights_only=True, map_location="cpu")
    N     = cache["n_samples"]
    actions = cache["actions"]           # (N, H, 2) already normalised

    print(f"  {N:,} samples | target state_dim = {cfg.state_dim}")
    episode_ids = _load_episode_ids(cfg, N)

    states_2d = cache["states"][:, :2]  # keep only position dims as base
    if cfg.state_dim > 2:
        print("  Building 6D state with delta history …")
        new_states = _build_6d_states(states_2d, actions, episode_ids)
    else:
        new_states = states_2d

    cache["states"] = new_states
    torch.save(cache, out_path)
    print(f"  Saved → {out_path}  (states shape: {tuple(new_states.shape)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",               type=int,  default=2, choices=[1, 2],
                        help="Experiment config: 1=Exp1 (baseline), 2=Exp2 B+C (default)")
    parser.add_argument("--max-samples",       type=int,  default=None)
    parser.add_argument("--output",            type=str,  default=None)
    parser.add_argument("--recompute-actions", action="store_true")
    parser.add_argument("--recompute-states",  action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.exp)
    print(f"Experiment : {args.exp}  |  output: {cfg.output_dir}")
    device   = cfg.get_device()
    out_path = Path(args.output or cfg.embeddings_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.recompute_actions:
        _recompute_actions_only(cfg, out_path)
        return
    if args.recompute_states:
        _recompute_states_only(cfg, out_path)
        return

    n_layers = cfg.n_vlm_layers
    print(f"Device      : {device}")
    print(f"VLM layers  : {cfg.vlm_extract_layers}  ({n_layers} scales)")
    print(f"State dim   : {cfg.state_dim}  (includes delta history: {cfg.state_dim > 2})")

    # ── Build dataset ─────────────────────────────────────────────────────
    print("\n[1/3] Loading PushT dataset …")
    full_dataset = PushTDataset(cfg.dataset_path, cfg, cache_frames=True)

    if args.max_samples and args.max_samples < len(full_dataset):
        dataset = Subset(full_dataset, list(range(args.max_samples)))
        print(f"   Quick mode: {args.max_samples:,} of {len(full_dataset):,} samples.")
        is_full = False
    else:
        dataset = full_dataset
        is_full = True

    loader = DataLoader(dataset, batch_size=cfg.embed_batch_size,
                        shuffle=False, num_workers=0, collate_fn=custom_collate)

    # ── Load VLM ──────────────────────────────────────────────────────────
    print("\n[2/3] Loading VLM …")
    model = VLAModel(cfg)
    model.vlm.to(device).eval()

    N        = len(dataset)
    H_size   = cfg.vlm_hidden_size   # 1024
    S        = cfg.img_seq_len       # 82
    horizon  = cfg.action_horizon
    action_d = cfg.action_dim        # 2
    state_d  = 2                     # raw 2D; extended to cfg.state_dim after loop

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
    all_states  = torch.zeros(N, state_d,         dtype=torch.float32)  # 2D first
    all_actions = torch.zeros(N, horizon, action_d, dtype=torch.float32)

    ptr = 0
    t0  = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="VLM encoding", unit="batch"):
            B = len(batch["images"])
            vlm_inputs = model.build_vlm_inputs(
                images=batch["images"], task_texts=batch["task_texts"], device=device)

            tokens, img_mask = model.encode_vlm(vlm_inputs)
            # tokens: (B, n_layers, S, H) or (B, S, H)

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

    # ── Extend states to 6D using action delta history ────────────────────
    if cfg.state_dim > 2:
        print(f"\n   Building {cfg.state_dim}D state with delta history …")
        episode_ids = _load_episode_ids(cfg, N)
        all_states  = _build_6d_states(all_states, all_actions, episode_ids)
        print(f"   State tensor: {tuple(all_states.shape)}")

    # ── Save ───────────────────────────────────────────────────────────────
    fmt = f"v3_multi_scale_{n_layers}layers" if n_layers > 1 else "v2_full_sequence"
    payload = {
        "embeddings":        all_embeds,
        "img_masks":         all_masks,
        "states":            all_states,
        "actions":           all_actions,
        "n_samples":         N,
        "full_dataset":      is_full,
        "format":            fmt,
        "vlm_extract_layers": list(cfg.vlm_extract_layers),
        "state_dim":         cfg.state_dim,
    }

    torch.save(payload, out_path)
    mb = out_path.stat().st_size / 1e6
    print(f"   Saved → {out_path}  ({mb:.0f} MB)")
    if is_full:
        print(f"\n   Full dataset encoded. Run:  python train.py")


if __name__ == "__main__":
    main()
