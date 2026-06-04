"""
scripts/evaluate.py
────────────────────
Step 3 of 3: Run simulation evaluation with receding-horizon control.

Loads the trained checkpoint, runs N simulation episodes, and saves results
to asset/runs/<dataset>/<exp>/sim_results.json.

Usage
─────
  # Standard eval (n=50 required for statistical conclusions):
  python scripts/evaluate.py --dataset pusht --exp exp02a --episodes 50

  # With videos:
  python scripts/evaluate.py --dataset pusht --exp exp02a --episodes 20

  # Specific checkpoint:
  python scripts/evaluate.py --dataset pusht --exp exp02a --checkpoint path/to/epoch_0200.pt

  # Ablation overrides (for inference_horizon / max_steps experiments):
  python scripts/evaluate.py --dataset pusht --exp exp02a --inference-horizon 8 \\
      --output asset/runs/pusht/exp02a_dit/sim_results_ih8.json

HARD CONSTRAINT: n_eval >= 50 before drawing SR conclusions (MPS non-determinism).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel


def _run_pusht(cfg, vlm_model, train_model, device, n_ep, save_video, output_path):
    from envs.pusht_env import PushTAgent, run_episode
    import gymnasium as gym
    import gym_pusht  # noqa

    video_dir = Path(cfg.output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    agent = PushTAgent(vlm_model, train_model, cfg, device)
    print(f"\n   Receding-horizon: predict {cfg.action_horizon} steps, "
          f"execute {cfg.inference_horizon} before re-plan")
    print(f"   Actions: {'relative (delta)' if cfg.use_relative_actions else 'absolute'}")

    print("\n[3/4] Starting gym_pusht …")
    env = gym.make("gym_pusht/PushT-v0",
                   obs_type="pixels_agent_pos", render_mode="rgb_array")

    print(f"\n[4/4] Running {n_ep} episodes …\n")
    results = []
    for ep in range(n_ep):
        results.append(run_episode(env, agent, cfg, ep, n_ep,
                                   save_video, video_dir))
    env.close()
    return results


def _run_aloha(cfg, vlm_model, train_model, device, n_ep, save_video, output_path):
    from envs.aloha_env import AlohaAgent, run_episode
    import gymnasium as gym
    import gym_aloha  # noqa

    video_dir = Path(cfg.output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    agent = AlohaAgent(vlm_model, train_model, cfg, device)
    print(f"\n   Receding-horizon: predict {cfg.action_horizon} steps, "
          f"execute {cfg.inference_horizon} before re-plan")
    print("   Actions: ABSOLUTE 14-DOF joint targets")

    print("\n[3/4] Starting gym_aloha ...")
    env = gym.make(cfg.env_id, obs_type="pixels_agent_pos",
                   max_episode_steps=cfg.sim_max_steps, disable_env_checker=True)

    print(f"\n[4/4] Running {n_ep} episodes ...\n")
    results = []
    for ep in range(n_ep):
        results.append(run_episode(env, agent, cfg, ep, n_ep, save_video, video_dir))
    env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulation evaluation")
    parser.add_argument("--dataset",          type=str, default="pusht",
                        help="Dataset name: pusht, aloha, language_table, ood")
    parser.add_argument("--exp",              type=str, default="exp02a",
                        help="Experiment ID: exp01, exp02a, exp03, exp04, …")
    parser.add_argument("--checkpoint",       type=str, default=None,
                        help="Override checkpoint path (default: <output_dir>/checkpoints/best.pt)")
    parser.add_argument("--episodes",         type=int, default=None,
                        help="Number of eval episodes (default: cfg.sim_episodes, recommend 50+)")
    parser.add_argument("--no-video",         action="store_true",
                        help="Skip video recording (faster)")
    parser.add_argument("--max-steps",        type=int, default=None,
                        help="Override sim_max_steps")
    parser.add_argument("--inference-horizon", type=int, default=None,
                        help="Override inference_horizon (WARNING: must match training!)")
    parser.add_argument("--output",           type=str, default=None,
                        help="Override results JSON path")
    args = parser.parse_args()

    cfg = get_config(args.dataset, args.exp)
    print(f"Dataset    : {args.dataset}  |  Exp: {args.exp}")
    print(f"Output dir : {cfg.output_dir}")
    device = cfg.get_device()

    if args.max_steps:
        cfg.sim_max_steps = args.max_steps
    if args.inference_horizon:
        print(f"[WARN] Overriding inference_horizon={cfg.inference_horizon} → "
              f"{args.inference_horizon}. This will degrade SR if it doesn't match training.")
        cfg.inference_horizon = args.inference_horizon

    n_ep = args.episodes or cfg.sim_episodes
    if n_ep < 50:
        print(f"[WARN] n_ep={n_ep} < 50. SR conclusions require n≥50 (MPS non-determinism).")
    print(f"Device     : {device}\n")

    # ── Load VLM ──────────────────────────────────────────────────────────
    print("[1/4] Loading VLM (frozen) …")
    vlm_model = VLAModel(cfg)
    vlm_model.vlm.to(device).eval()

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint or (Path(cfg.output_dir) / "checkpoints" / "best.pt"))
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print(f"        Run: python scripts/train.py --dataset {args.dataset} --exp {args.exp}")
        sys.exit(1)

    print(f"\n[2/4] Loading adapter+decoder from {ckpt_path} …")
    train_model = VLATrainModel(cfg).to(device)
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_model.load_state_dict(ckpt["state_dict"])
    train_model.eval()
    print(f"   Epoch {ckpt.get('epoch','?')}  |  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # ── Run environment ────────────────────────────────────────────────────
    if args.dataset == "pusht":
        results = _run_pusht(cfg, vlm_model, train_model, device,
                             n_ep, not args.no_video,
                             Path(args.output) if args.output else None)
    elif args.dataset == "aloha":
        results = _run_aloha(cfg, vlm_model, train_model, device,
                             n_ep, not args.no_video,
                             Path(args.output) if args.output else None)
    else:
        raise NotImplementedError(f"Eval for {args.dataset!r} not yet implemented. "
                                  f"Add a runner in envs/{args.dataset}_env.py.")

    # ── Summary ────────────────────────────────────────────────────────────
    succs = [r["is_success"]   for r in results]
    covs  = [r["max_coverage"] for r in results]
    steps = [r["steps"]        for r in results]
    sr    = float(np.mean(succs))
    print(f"\n{'═'*52}")
    print(f"  Results  ({n_ep} episodes)   [{args.dataset}/{args.exp}]")
    print(f"{'═'*52}")
    print(f"  Success rate      : {sr*100:.1f}%  ({sum(succs)}/{n_ep})")
    print(f"  Mean max coverage : {np.mean(covs)*100:.1f}%")
    print(f"  Mean steps        : {np.mean(steps):.1f}")
    print(f"  Horizon train/exec: {cfg.action_horizon}/{cfg.inference_horizon}")
    if n_ep >= 50:
        from scipy.stats import binom
        ci_lo = binom.ppf(0.025, n_ep, sr) / n_ep
        ci_hi = binom.ppf(0.975, n_ep, sr) / n_ep
        print(f"  Wilson 95% CI     : [{ci_lo*100:.0f}%, {ci_hi*100:.0f}%]")
    print(f"{'═'*52}\n")

    summary = {
        "dataset":    args.dataset,
        "exp_id":     args.exp,
        "checkpoint": str(ckpt_path),
        "n_episodes": n_ep,
        "success_rate": round(sr, 4),
        "mean_max_coverage": round(float(np.mean(covs)), 4),
        "mean_steps": round(float(np.mean(steps)), 1),
        "action_horizon": cfg.action_horizon,
        "inference_horizon": cfg.inference_horizon,
        "use_relative_actions": cfg.use_relative_actions,
        "num_flow_steps": cfg.num_flow_steps,
        "success_threshold": cfg.success_threshold,
        "episodes": results,
    }
    out = Path(args.output) if args.output else Path(cfg.output_dir) / "sim_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"  Results saved → {out}")


if __name__ == "__main__":
    main()
