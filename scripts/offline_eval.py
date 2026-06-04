"""
scripts/offline_eval.py — Offline evaluation of the VLA model (adapter + decoder).

Computes offline metrics on the cached embedding validation set:
  MSE, MAE, Action Error Norm (px), Directional Accuracy,
  per-horizon breakdown, step-count sensitivity, qualitative check.

Note: This is OFFLINE evaluation (on the cached val set).
For SIMULATION evaluation (actual robot episodes), use scripts/evaluate.py.

Usage
-----
  python scripts/offline_eval.py --dataset pusht --exp exp02a
  python scripts/offline_eval.py --dataset pusht --exp exp01
  python scripts/offline_eval.py --exp 2 --flow-steps 3   # legacy integer interface
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).parents[1]   # scripts/ → project root
sys.path.insert(0, str(ROOT))

from configs.registry import get_config_legacy, get_config as _registry_get
from data.pusht import PushTEmbeddingDataset
from models.vla_train import VLATrainModel

def get_config(exp: int): return get_config_legacy(exp, dataset="pusht")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def denorm(a: np.ndarray, cfg: VLAConfig) -> np.ndarray:
    return a * np.array(cfg.action_std) + np.array(cfg.action_mean)


def load_model(ckpt_path: Path, cfg: VLAConfig, device: torch.device) -> VLATrainModel:
    m = VLATrainModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    ep  = ckpt.get("epoch", "?")
    vl  = ckpt.get("val_loss", float("nan"))
    print(f"  Loaded: {ckpt_path.name}  (epoch={ep}, val_loss={vl:.4f})")
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, cfg, device, num_steps=3):
    all_pred, all_target = [], []
    for batch in val_loader:
        embed    = batch["embedding"].to(device)
        state    = batch["state"].to(device)
        actions  = batch["actions"]
        img_mask = batch.get("img_mask")
        if img_mask is not None:
            img_mask = img_mask.to(device)
        B = actions.shape[0]

        flat = model.sample(embed, state, num_steps=num_steps, img_mask=img_mask)
        pred = flat.view(B, cfg.action_horizon, cfg.action_dim)

        all_pred.append(pred.cpu().float().numpy())
        all_target.append(actions.float().numpy())

    pred_arr   = np.concatenate(all_pred)    # (N, H, 2)
    target_arr = np.concatenate(all_target)
    N, H, D    = pred_arr.shape

    mse = float(np.mean((pred_arr - target_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - target_arr)))

    pred_px   = denorm(pred_arr.reshape(-1, D),   cfg).reshape(N, H, D)
    target_px = denorm(target_arr.reshape(-1, D), cfg).reshape(N, H, D)
    l2        = np.linalg.norm(pred_px - target_px, axis=-1)  # (N, H)
    l2_mean   = float(l2.mean())

    pf  = pred_arr.reshape(-1, D)
    tf  = target_arr.reshape(-1, D)
    pn  = pf / (np.linalg.norm(pf, axis=-1, keepdims=True) + 1e-8)
    tn  = tf / (np.linalg.norm(tf, axis=-1, keepdims=True) + 1e-8)
    dir_acc = float((np.sum(pn * tn, axis=-1) > 0).mean())

    return {
        "mse": mse, "mae": mae,
        "action_error_norm": l2_mean,
        "directional_acc":   dir_acc,
        "per_horizon_mse":   [float(np.mean((pred_arr[:,h]-target_arr[:,h])**2)) for h in range(H)],
        "per_horizon_l2":    [float(l2[:,h].mean()) for h in range(H)],
        "n_samples": N,
    }


def print_metrics(m: dict, label: str = "") -> None:
    sep = "─" * 54
    print(f"\n{sep}")
    if label: print(f"  {label}")
    print(sep)
    print(f"  Samples          : {m['n_samples']:,}")
    print(f"  MSE (normalised) : {m['mse']:.4f}")
    print(f"  MAE (normalised) : {m['mae']:.4f}")
    print(f"  L2 error (px)    : {m['action_error_norm']:.2f}")
    print(f"  Directional acc  : {m['directional_acc']*100:.1f}%")
    print(f"\n  Per-step L2 error (px):")
    for i, v in enumerate(m["per_horizon_l2"]):
        print(f"    t+{i+1:2d}: {v:6.2f}  {'█'*int(v/3)}")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# All-checkpoint sweep
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all_checkpoints(cfg, val_loader, device):
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpts    = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    print(f"\nFound {len(ckpts)} checkpoints.\n")
    rows = []
    for p in ckpts:
        m = load_model(p, cfg, device)
        r = evaluate(m, val_loader, cfg, device)
        rows.append((p.name, r["mse"], r["mae"], r["action_error_norm"], r["directional_acc"]))

    print(f"\n{'Checkpoint':<30} {'MSE':>8} {'MAE':>8} {'L2(px)':>8} {'Dir%':>7}")
    print("─" * 64)
    best = min(r[1] for r in rows)
    for name, mse, mae, l2, da in rows:
        mark = "  ← best" if abs(mse - best) < 1e-8 else ""
        print(f"  {name:<28} {mse:>8.4f} {mae:>8.4f} {l2:>8.2f} {da*100:>6.1f}%{mark}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Qualitative check
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def qualitative_check(model, val_loader, cfg, device, n_show=5):
    action_std  = np.array(cfg.action_std)
    action_mean = np.array(cfg.action_mean)
    state_std   = np.array(cfg.state_std[:2])   # display uses agent pos only
    state_mean  = np.array(cfg.state_mean[:2])

    print(f"\n{'─'*62}")
    print(f"  Qualitative check — {n_show} samples")
    print(f"  Actions shown as relative DELTA (px) then absolute (px)")
    print(f"{'─'*62}")

    count = 0
    for batch in val_loader:
        embed    = batch["embedding"].to(device)
        state    = batch["state"].to(device)
        acts     = batch["actions"]
        img_mask = batch.get("img_mask")
        if img_mask is not None:
            img_mask = img_mask.to(device)
        B = acts.shape[0]

        flat = model.sample(embed, state, num_steps=3, img_mask=img_mask)
        pred = flat.view(B, cfg.action_horizon, cfg.action_dim).cpu().float().numpy()
        gt   = acts.numpy()

        pred_px = pred * action_std + action_mean
        gt_px   = gt   * action_std + action_mean
        st_raw  = state.cpu().numpy()[:, :2]   # always use agent pos dims
        st_px   = st_raw * state_std + state_mean

        for i in range(min(B, n_show - count)):
            print(f"\n  Sample {count+1}  |  agent @ ({st_px[i,0]:.1f}, {st_px[i,1]:.1f})")
            print(f"  {'Step':<5} {'Pred Δ(x,y) px':>22}  {'GT Δ(x,y) px':>22}  {'L2':>7}")
            for h in range(min(8, cfg.action_horizon)):   # show first 8
                px,py = pred_px[i,h]; gx,gy = gt_px[i,h]
                l2    = float(np.linalg.norm([px-gx, py-gy]))
                mark  = " ◀ exec" if h < cfg.inference_horizon else ""
                print(f"  t+{h+1:<3}  ({px:+7.1f}, {py:+7.1f})    "
                      f"({gx:+7.1f}, {gy:+7.1f})   {l2:>6.2f}{mark}")
            count += 1
            if count >= n_show: break
        if count >= n_show: break
    print(f"{'─'*62}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",            type=int, default=2, choices=[1, 2],
                        help="Experiment config: 1=Exp1 (baseline), 2=Exp2 B+C (default)")
    parser.add_argument("--checkpoint",     type=str, default=None)
    parser.add_argument("--cache",          type=str, default=None)
    parser.add_argument("--all",            action="store_true")
    parser.add_argument("--flow-steps",     type=int, default=3)
    parser.add_argument("--no-qualitative", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.exp)
    print(f"Experiment : {args.exp}  |  output: {cfg.output_dir}")
    device = cfg.get_device()
    print(f"Device: {device}\n")

    cache_path = Path(args.cache or cfg.embeddings_cache)
    if not cache_path.exists():
        print(f"[ERROR] Cache not found: {cache_path}")
        sys.exit(1)

    print("[1/3] Loading dataset …")
    full_ds = PushTEmbeddingDataset(str(cache_path))
    val_len = max(1, int(len(full_ds) * 0.10))
    _, val_ds = random_split(full_ds, [len(full_ds)-val_len, val_len],
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"   Val samples: {len(val_ds):,}")

    if args.all:
        print("\n[2/3] Sweeping all checkpoints …")
        evaluate_all_checkpoints(cfg, val_loader, device)
        return

    ckpt_path = Path(args.checkpoint or (Path(cfg.output_dir)/"checkpoints"/"best.pt"))
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n[2/3] Loading checkpoint …")
    model = load_model(ckpt_path, cfg, device)

    print(f"\n[3/3] Evaluating (flow_steps={args.flow_steps}) …")
    m = evaluate(model, val_loader, cfg, device, num_steps=args.flow_steps)
    print_metrics(m, label=f"Results — {ckpt_path.name}")

    if not args.no_qualitative:
        qualitative_check(model, val_loader, cfg, device)

    print("\nStep-count sensitivity:")
    print(f"  {'Steps':>6}  {'MSE':>8}  {'L2(px)':>8}")
    for n in [1, 3, 5, 10, 20]:
        r = evaluate(model, val_loader, cfg, device, num_steps=n)
        print(f"  {n:>6}  {r['mse']:>8.4f}  {r['action_error_norm']:>8.2f}")


if __name__ == "__main__":
    main()
