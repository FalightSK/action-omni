"""
scripts/eval_lt_control.py
──────────────────────────
Difficulty-matched control for the Language Table new-command study.

The headline eval (eval_lt_ood.py) found held-out-verb commands (separate/point) get
HIGHER success than in-distribution block2block/absolute — but those are different
TASKS with looser success conditions (reach / push-apart vs precise placement), so
raw SR conflates verb-generalization with task-difficulty.

This control fixes the task and varies ONLY the verb, using the SAME reward factory:
  separate task:  "separate X from Y"  (held-out verb)  vs  "move X away from Y"  (trained verb)
  point/reach:    "point ... X"         (held-out verb)  vs  "move your arm ... X" (trained verb)

If SR(held-out verb) ≈ SR(trained verb) for the same task, the frozen-VLM VLA
follows a genuinely unseen word at no cost → clean verb-generalization evidence,
independent of task difficulty.

Output → asset/runs/language_table/exp01_baseline/control_results.json
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

# (factory, require_substring, label, verb_kind)  — fixed task, varied verb
CONDITIONS = [
    ("separate", "separate", "separate / held-out verb", "held_out"),
    ("separate", "away",     "separate / trained verb (move away)", "trained"),
    ("point",    "point",    "reach / held-out verb (point)", "held_out"),
    ("point",    "move",     "reach / trained verb (move arm)", "trained"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",   type=int, default=50)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--output",     type=str, default=None)
    args = ap.parse_args()

    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    n_ep = args.episodes
    print(f"Language Table verb-generalization CONTROL  |  n={n_ep}  device={device}")

    print("\n[1/2] Loading VLM (frozen) …")
    vlm_model = VLAModel(cfg)
    vlm_model.vlm.to(device).eval()

    ckpt_path = Path(args.checkpoint or (Path(cfg.output_dir) / "checkpoints" / "best.pt"))
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    print(f"[2/2] Loading adapter+decoder from {ckpt_path} …")
    train_model = VLATrainModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_model.load_state_dict(ckpt["state_dict"])
    train_model.eval()
    print(f"   epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    from envs.language_table_env import LanguageTableAgent, make_lt_env, run_episode
    agent = LanguageTableAgent(vlm_model, train_model, cfg, device)
    video_dir = Path(cfg.output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    conditions = []
    for factory, require, label, verb_kind in CONDITIONS:
        print(f"\n{'═'*60}\n  {label}   [factory={factory} require='{require}']\n{'═'*60}")
        env = make_lt_env(factory, seed=args.seed)
        eps = [run_episode(env, agent, cfg, e, n_ep, False, video_dir, require_substring=require)
               for e in range(n_ep)]
        try:
            env.close()
        except Exception:
            pass
        sr = float(np.mean([r["is_success"] for r in eps]))
        steps = float(np.mean([r["steps"] for r in eps]))
        matched = sum(1 for r in eps if require in r["instruction"].lower())
        conditions.append({"factory": factory, "require": require, "label": label,
                           "verb_kind": verb_kind, "n": n_ep, "n_matched": matched,
                           "sr": round(sr, 4), "mean_steps": round(steps, 1),
                           "episodes": eps})
        print(f"  → {label}: SR={sr*100:.1f}%  ({sum(r['is_success'] for r in eps)}/{n_ep})  "
              f"matched={matched}/{n_ep}  mean_steps={steps:.0f}")

    summary = {"dataset": "language_table", "exp_id": "exp01", "checkpoint": str(ckpt_path),
               "n_episodes": n_ep, "conditions": conditions}
    out = Path(args.output) if args.output else Path(cfg.output_dir) / "control_results.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'═'*60}\n  VERB-GENERALIZATION CONTROL  (same task, varied verb)\n{'═'*60}")
    for c in conditions:
        print(f"  {c['label']:42s}  SR={c['sr']*100:5.1f}%  (matched {c['n_matched']}/{c['n']})")
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
