"""
scripts/eval_lt_taskgen.py
──────────────────────────
Plan A seen/unseen TASK-generalization eval.

The balanced model trained ONLY on push-to-location tasks (block2block, block2absolute,
block2relative). This evals it on those SEEN tasks vs UNSEEN task families it never
trained on (separate, point, between, corner) — testing whether it can perform an
UNTRAINED task purely from the language instruction.

For separate/point the same factory also emits trained-phrasing variants ("move away",
"move arm" → classified block2block in training), so those are reset-filtered to the
held-out verb. between is cleanly held out. corner overlaps block2absolute ("corner" is
an absolute-location word) so it is reported but flagged as leaky.

Output → asset/runs/language_table/exp02_balanced/taskgen_results.json
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

#   (factory, kind, require-verb, note)
TASKS = [
    ("block2block",   "seen",   None, "push X to Y"),
    ("block2absolute", "seen",  None, "push X to a region"),
    ("block2relative", "seen",  None, "push X left/right/above/below of Y"),
    ("separate",      "unseen", "separate", "push apart (held-out verb)"),
    ("point",         "unseen", "point",    "reach (held-out verb)"),
    ("between",       "unseen", None, "push X between Y and Z (held-out task)"),
    ("corner",        "unseen", None, "push X to a corner (overlaps absolute — leaky)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "asset/data/language_table_sim_balanced"))
    ap.add_argument("--checkpoint", default=str(ROOT / "asset/runs/language_table/exp02_balanced/checkpoints/best.pt"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(ROOT / "asset/runs/language_table/exp02_balanced/taskgen_results.json"))
    args = ap.parse_args()

    os.environ["LT_DATASET_ROOT"] = str(Path(args.data_root).resolve())
    from configs.registry import get_config
    from models.vla import VLAModel
    from models.vla_train import VLATrainModel
    from envs.language_table_env import make_lt_env, run_episode, LanguageTableAgent

    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    n = args.episodes
    print(f"Plan A task-generalization eval | ckpt={args.checkpoint} | n={n}/task")

    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ck["state_dict"]); tm.eval()
    print(f"  loaded step={ck.get('step','?')} val={ck.get('val_loss', float('nan')):.4f}")
    agent = LanguageTableAgent(vlm, tm, cfg, device)
    vdir = Path(cfg.output_dir) / "videos"; vdir.mkdir(parents=True, exist_ok=True)

    results = []
    for factory, kind, require, note in TASKS:
        if factory not in __import__("envs.language_table_env", fromlist=["REWARD_FACTORIES"]).REWARD_FACTORIES:
            print(f"  [skip] {factory} not available"); continue
        print(f"\n{'='*56}\n  {factory}  [{kind}]  {note}\n{'='*56}")
        env = make_lt_env(factory, seed=args.seed)
        eps = [run_episode(env, agent, cfg, e, n, False, vdir, require_substring=require) for e in range(n)]
        try: env.close()
        except Exception: pass
        sr = float(np.mean([r["is_success"] for r in eps]))
        steps = float(np.mean([r["steps"] for r in eps]))
        matched = sum(1 for r in eps if not require or require in r["instruction"].lower())
        results.append({"factory": factory, "kind": kind, "require": require, "note": note,
                        "n": n, "sr": round(sr, 4), "mean_steps": round(steps, 1),
                        "verb_matched": matched, "episodes": eps})
        print(f"  → {factory}: SR={sr*100:.1f}%  ({sum(r['is_success'] for r in eps)}/{n})  matched={matched}/{n}")

    seen = [r for r in results if r["kind"] == "seen"]
    unseen = [r for r in results if r["kind"] == "unseen"]
    seen_sr = float(np.mean([r["sr"] for r in seen])) if seen else 0.0
    unseen_sr = float(np.mean([r["sr"] for r in unseen])) if unseen else 0.0
    out = {"dataset": "language_table", "exp_id": "exp02_balanced", "checkpoint": args.checkpoint,
           "n_episodes_per_task": n, "by_task": results,
           "mean_seen_sr": round(seen_sr, 4), "mean_unseen_sr": round(unseen_sr, 4)}
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"\n{'='*56}\n  SUMMARY  [exp02_balanced]\n{'='*56}")
    for r in results:
        print(f"  {r['factory']:15s} [{r['kind']:6s}]  SR={r['sr']*100:5.1f}%")
    print(f"  {'-'*40}\n  mean SEEN   SR: {seen_sr*100:.1f}%\n  mean UNSEEN SR: {unseen_sr*100:.1f}%")
    print(f"  saved → {args.output}")


if __name__ == "__main__":
    main()
