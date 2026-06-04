"""
inference.py  [BACKWARD-COMPAT SHIM]
──────────────────────────────────────
Preserved so `python inference.py --exp 2 --episodes 50` and
`from inference import PushTAgent, run_episode` continue to work.

New canonical entry point:
  python scripts/evaluate.py --dataset pusht --exp exp02a --episodes 50

PushTAgent and run_episode are now in envs/pusht_env.py.
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Re-export so `from inference import PushTAgent, run_episode` still works
from envs.pusht_env import PushTAgent, run_episode, annotate_frame  # noqa: F401

from config_loader import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",               type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--checkpoint",        type=str, default=None)
    parser.add_argument("--episodes",          type=int, default=None)
    parser.add_argument("--no-video",          action="store_true")
    parser.add_argument("--max-steps",         type=int, default=None)
    parser.add_argument("--inference-horizon", type=int, default=None)
    parser.add_argument("--output",            type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args.exp)
    print(f"[inference.py shim] Exp {args.exp} | output: {cfg.output_dir}")
    print(f"[inference.py shim] New command: python scripts/evaluate.py --dataset pusht --exp exp{args.exp:02d}{'a' if args.exp==2 else ''} --episodes {args.episodes or cfg.sim_episodes}")
    device = cfg.get_device()
    if args.max_steps:         cfg.sim_max_steps    = args.max_steps
    if args.inference_horizon: cfg.inference_horizon = args.inference_horizon
    n_ep = args.episodes or cfg.sim_episodes

    print("[1/4] Loading VLM …")
    vlm_model = VLAModel(cfg)
    vlm_model.vlm.to(device).eval()

    ckpt_path = Path(args.checkpoint or (Path(cfg.output_dir) / "checkpoints" / "best.pt"))
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"[2/4] Loading adapter+decoder from {ckpt_path.name} …")
    train_model = VLATrainModel(cfg).to(device)
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_model.load_state_dict(ckpt["state_dict"])
    train_model.eval()

    agent     = PushTAgent(vlm_model, train_model, cfg, device)
    video_dir = Path(cfg.output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print("[3/4] Starting gym_pusht …")
    import gymnasium as gym
    import gym_pusht  # noqa
    env = gym.make("gym_pusht/PushT-v0",
                   obs_type="pixels_agent_pos", render_mode="rgb_array")

    print(f"\n[4/4] Running {n_ep} episodes …\n")
    results = []
    for ep in range(n_ep):
        results.append(run_episode(env, agent, cfg, ep, n_ep,
                                   not args.no_video, video_dir))
    env.close()

    succs = [r["is_success"]   for r in results]
    covs  = [r["max_coverage"] for r in results]
    steps = [r["steps"]        for r in results]
    sr    = float(np.mean(succs))
    print(f"\n  Success rate: {sr*100:.1f}%  ({sum(succs)}/{n_ep})")
    print(f"  Mean coverage: {np.mean(covs)*100:.1f}%")

    summary = {"checkpoint": str(ckpt_path), "n_episodes": n_ep,
               "success_rate": round(sr, 4),
               "mean_max_coverage": round(float(np.mean(covs)), 4),
               "mean_steps": round(float(np.mean(steps)), 1),
               "action_horizon": cfg.action_horizon,
               "inference_horizon": cfg.inference_horizon,
               "episodes": results}
    out = Path(args.output) if args.output else Path(cfg.output_dir) / "sim_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"  Results → {out}")


if __name__ == "__main__":
    main()
