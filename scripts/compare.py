"""
scripts/compare.py — Fair head-to-head comparison of two experiments.

Both models run on IDENTICAL environment resets (same seed per episode).
The environment is reset with the same seed for Exp1 and Exp2a so every
episode starts from exactly the same initial block and agent position.

Usage:
  python3 scripts/compare.py
  python3 scripts/compare.py --episodes 20
  python3 scripts/compare.py --no-video
"""

from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parents[1]   # scripts/ → project root
sys.path.insert(0, str(ROOT))

from configs.registry import get_config_legacy
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from envs.pusht_env import PushTAgent, run_episode

def get_config(exp: int): return get_config_legacy(exp, dataset="pusht")


def load_agent(exp: int, device: torch.device) -> PushTAgent:
    cfg = get_config(exp)
    vlm = VLAModel(cfg)
    vlm.vlm.to(device).eval()

    ckpt_path = Path(cfg.output_dir) / "checkpoints" / "best.pt"
    tm = VLATrainModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"])
    tm.eval()
    print(f"  Exp{exp}: epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}  "
          f"({Path(cfg.output_dir).name})")
    return PushTAgent(vlm, tm, cfg, device), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    device = get_config(1).get_device()
    print(f"Device: {device}\n")

    print("Loading models …")
    agent1, cfg1 = load_agent(1, device)
    agent2, cfg2 = load_agent(2, device)
    print()

    # Shared seeds — both experiments see exactly the same episodes
    seeds = [ep * 137 + 42 for ep in range(args.episodes)]

    import gymnasium as gym
    import gym_pusht  # noqa

    env1 = gym.make("gym_pusht/PushT-v0",
                    obs_type="pixels_agent_pos", render_mode="rgb_array")
    env2 = gym.make("gym_pusht/PushT-v0",
                    obs_type="pixels_agent_pos", render_mode="rgb_array")

    vid1 = Path(cfg1.output_dir) / "videos_compare"
    vid2 = Path(cfg2.output_dir) / "videos_compare"
    if not args.no_video:
        vid1.mkdir(parents=True, exist_ok=True)
        vid2.mkdir(parents=True, exist_ok=True)

    results1, results2 = [], []

    W = 78
    print("=" * W)
    print(f"  {'Ep':>3}  {'Seed':>6}  │  {'Exp1 (MLP)':^20}  │  {'Exp2a (DiT)':^20}  │  {'Δ cov':>7}")
    print(f"  {'':>3}  {'':>6}  │  {'cov':>7}  {'steps':>5}  {'':>4}  │  {'cov':>7}  {'steps':>5}  {'':>4}  │  {'':>7}")
    print("-" * W)

    for i, seed in enumerate(seeds):
        ep_idx = i

        r1 = run_episode(env1, agent1, cfg1, ep_idx=ep_idx,
                         total_eps=args.episodes,
                         save_video=not args.no_video,
                         video_dir=vid1)
        # Override seed used — run_episode uses ep_idx*137+42 internally,
        # which matches our seeds list exactly.

        r2 = run_episode(env2, agent2, cfg2, ep_idx=ep_idx,
                         total_eps=args.episodes,
                         save_video=not args.no_video,
                         video_dir=vid2)

        results1.append(r1)
        results2.append(r2)

        c1 = r1["max_coverage"] * 100
        c2 = r2["max_coverage"] * 100
        ok1 = "✅" if r1["is_success"] else "❌"
        ok2 = "✅" if r2["is_success"] else "❌"
        dc  = c2 - c1
        arrow = ("↑" if dc > 1 else ("↓" if dc < -1 else "→"))
        print(f"  {i+1:>3}  {seed:>6}  │  {c1:6.1f}%  {r1['steps']:>5}  {ok1}  │  "
              f"{c2:6.1f}%  {r2['steps']:>5}  {ok2}  │  {arrow}{dc:+6.1f}%")

    env1.close()
    env2.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    c1_all = [r["max_coverage"] for r in results1]
    c2_all = [r["max_coverage"] for r in results2]
    s1_all = [r["is_success"]   for r in results1]
    s2_all = [r["is_success"]   for r in results2]
    st1_ok = [r["steps"] for r in results1 if r["is_success"]]
    st2_ok = [r["steps"] for r in results2 if r["is_success"]]

    print("=" * W)
    print(f"\n  {'Metric':<30}  {'Exp1 (MLP)':>15}  {'Exp2a (DiT)':>15}")
    print("-" * W)

    def sr(s):  return "%d/%d  (%.0f%%)" % (sum(s), len(s), np.mean(s)*100)
    def cv(c):  return "%.1f%%" % (np.mean(c)*100)
    def med(c): return "%.1f%%" % (np.median(c)*100)
    def mn(c):  return "%.1f%%" % (min(c)*100)
    def ss(s):  return ("%.0f steps" % np.mean(s)) if s else "—"

    print(f"  {'Success rate':<30}  {sr(s1_all):>15}  {sr(s2_all):>15}")
    print(f"  {'Mean max coverage':<30}  {cv(c1_all):>15}  {cv(c2_all):>15}")
    print(f"  {'Median max coverage':<30}  {med(c1_all):>15}  {med(c2_all):>15}")
    print(f"  {'Min coverage':<30}  {mn(c1_all):>15}  {mn(c2_all):>15}")
    print(f"  {'Mean steps (successes)':<30}  {ss(st1_ok):>15}  {ss(st2_ok):>15}")

    buckets = [(0, .50,"< 50%"), (.50,.80,"50–80%"),
               (.80,.90,"80–90%"), (.90,.95,"90–95%"), (.95,1.1,"≥ 95%  ✅")]
    print(f"\n  {'Coverage bucket':<30}  {'Exp1':>15}  {'Exp2a':>15}")
    for lo, hi, label in buckets:
        n1 = sum(1 for c in c1_all if lo <= c < hi)
        n2 = sum(1 for c in c2_all if lo <= c < hi)
        print(f"  {label:<30}  {'%2d  %s' % (n1,'█'*n1):>15}  {'%2d  %s' % (n2,'█'*n2):>15}")

    # Episodes where one wins clearly
    print(f"\n  Episodes where Exp2a coverage > Exp1 by >10%: "
          f"{sum(1 for a,b in zip(c1_all,c2_all) if b-a > 0.10)}/20")
    print(f"  Episodes where Exp1 coverage > Exp2a by >10%: "
          f"{sum(1 for a,b in zip(c1_all,c2_all) if a-b > 0.10)}/20")
    print(f"  Episodes within 5% of each other:              "
          f"{sum(1 for a,b in zip(c1_all,c2_all) if abs(a-b) <= 0.05)}/20")
    print("=" * W)

    # Save
    out = ROOT / "asset" / "comparison_results.json"
    payload = {
        "seeds": seeds,
        "n_episodes": args.episodes,
        "exp1": {"results": results1,
                 "sr": float(np.mean(s1_all)),
                 "mean_coverage": float(np.mean(c1_all))},
        "exp2a": {"results": results2,
                  "sr": float(np.mean(s2_all)),
                  "mean_coverage": float(np.mean(c2_all))},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()
