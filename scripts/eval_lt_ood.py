"""
scripts/eval_lt_ood.py
──────────────────────
Phase 4 — Language Table in-distribution vs NEW-COMMAND / OOD generalization.

Loads the trained VLA once, then rolls out the live Language Table sim under several
command types (reward factories), each of which GENERATES its own instructions:

  in-distribution (trained-on verbs):
    block2block      "push the red moon to the blue cube"
    block2absolute   "move the green cube to the top left corner"
  OOD (verbs HELD OUT of training by scripts/convert_language_table.py):
    separate         "separate the green star from the red moon"
    point            "point your arm at the blue cube"

For each command type we run N episodes and record success (sparse reward > 0).
Result → asset/runs/language_table/exp01_baseline/ood_results.json  (feeds plot_lt_ood.py)

Usage
─────
  python scripts/eval_lt_ood.py --episodes 50
  python scripts/eval_lt_ood.py --episodes 50 --rewards block2block,separate,point
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

IN_DIST = {"block2block", "block2absolute"}
OOD     = {"separate", "point"}
# OOD rewards emit a MIX of phrasings; restrict eval to instructions that actually
# use the held-out verb (absent from training) for a clean new-command test.
REQUIRE = {"separate": "separate", "point": "point"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",   type=int, default=50)
    ap.add_argument("--rewards",    type=str,
                    default="block2block,block2absolute,separate,point")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--max-steps",  type=int, default=None)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--output",     type=str, default=None)
    ap.add_argument("--save-video", action="store_true",
                    help="render a video per episode (slow; for a few-episode demo run)")
    args = ap.parse_args()

    cfg = get_config("language_table", "exp01")
    if args.max_steps:
        cfg.sim_max_steps = args.max_steps
    device = cfg.get_device()
    n_ep = args.episodes
    rewards = [r.strip() for r in args.rewards.split(",") if r.strip()]
    print(f"Language Table OOD eval  |  rewards={rewards}  |  n={n_ep}  device={device}")

    # ── Load frozen VLM + trained adapter/decoder (once) ───────────────────────
    print("\n[1/2] Loading VLM (frozen) …")
    vlm_model = VLAModel(cfg)
    vlm_model.vlm.to(device).eval()

    ckpt_path = Path(args.checkpoint or (Path(cfg.output_dir) / "checkpoints" / "best.pt"))
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}\n"
              f"        Run: python scripts/train.py --dataset language_table --exp exp01")
        sys.exit(1)
    print(f"[2/2] Loading adapter+decoder from {ckpt_path} …")
    train_model = VLATrainModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_model.load_state_dict(ckpt["state_dict"])
    train_model.eval()
    print(f"   epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # env wrapper imported here (pulls in the tf stub + language_table)
    from envs.language_table_env import LanguageTableAgent, make_lt_env, run_episode
    agent = LanguageTableAgent(vlm_model, train_model, cfg, device)
    video_dir = Path(cfg.output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # ── Roll out each command type ─────────────────────────────────────────────
    by_reward = {}
    for reward in rewards:
        kind = "in_dist" if reward in IN_DIST else ("ood" if reward in OOD else "unknown")
        require = REQUIRE.get(reward)
        print(f"\n{'═'*60}\n  Command type: {reward}   [{kind}]"
              f"{'   require=' + require if require else ''}\n{'═'*60}")
        env = make_lt_env(reward, seed=args.seed)
        eps = [run_episode(env, agent, cfg, e, n_ep, args.save_video, video_dir,
                           require_substring=require)
               for e in range(n_ep)]
        try:
            env.close()
        except Exception:
            pass
        sr = float(np.mean([r["is_success"] for r in eps]))
        steps = float(np.mean([r["steps"] for r in eps]))
        n_match = sum(1 for r in eps if not require or require in r["instruction"].lower())
        by_reward[reward] = {"reward": reward, "kind": kind, "n": n_ep,
                             "require": require, "n_matched_verb": n_match,
                             "sr": round(sr, 4), "mean_steps": round(steps, 1),
                             "episodes": eps}
        print(f"  → {reward}: SR={sr*100:.1f}%  ({sum(r['is_success'] for r in eps)}/{n_ep})  "
              f"mean_steps={steps:.0f}  verb_matched={n_match}/{n_ep}")

    # ── Aggregate in-dist vs OOD ───────────────────────────────────────────────
    def agg(kind):
        vals = [v["sr"] for v in by_reward.values() if v["kind"] == kind]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "dataset": "language_table", "exp_id": "exp01",
        "checkpoint": str(ckpt_path),
        "n_episodes_per_reward": n_ep,
        "inference_horizon": cfg.inference_horizon,
        "action_horizon": cfg.action_horizon,
        "held_out_ood_commands": sorted(OOD),
        "mean_sr_in_dist": agg("in_dist"),
        "mean_sr_ood": agg("ood"),
        "by_reward": by_reward,
    }
    out = Path(args.output) if args.output else Path(cfg.output_dir) / "ood_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'═'*60}\n  SUMMARY  [language_table/exp01]\n{'═'*60}")
    for reward, v in by_reward.items():
        print(f"  {reward:16s} [{v['kind']:8s}]  SR={v['sr']*100:5.1f}%  ({v['n']} eps)")
    print(f"  {'-'*40}")
    print(f"  mean in-distribution SR : "
          f"{(summary['mean_sr_in_dist'] or 0)*100:.1f}%")
    print(f"  mean OOD (new command) SR: "
          f"{(summary['mean_sr_ood'] or 0)*100:.1f}%")
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
