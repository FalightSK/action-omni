"""
scripts/_lt_eval_indist.py
──────────────────────────
In-distribution evaluation for the STABLE 5-instruction Language Table subset
(exp02_5instr). The block2block sim generates instructions from the FULL template
space, but this model only trained on 5 specific instructions — so a plain eval
would measure OOD generalization, not what the model actually learned.

This script resets the env until the live instruction EXACTLY matches one of the 5
trained instructions, then runs a normal receding-horizon rollout. It reports SR
overall + per instruction, so we can tell whether dense per-instruction coverage
lets the frozen-VLM VLA actually solve block2block (data-thinness hypothesis) or
whether it still fails (architecture/control-precision hypothesis).

Usage:
    python scripts/_lt_eval_indist.py --episodes 50
"""
from __future__ import annotations
import argparse, os, sys, time, json, math
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from envs.language_table_env import make_lt_env, get_instruction, LanguageTableAgent, resize_frame

# The 5 instructions the subset was trained on (must match _lt_make_5instr_subset.py).
TRAINED_5 = [
    "push the green star into the green cube",
    "push the yellow pentagon into the yellow star",
    "push the green cube into the green star",
    "push the yellow star into the yellow pentagon",
    "push the blue cube into the blue crescent",
]

# The live block2block sim paraphrases the task ("push/move/slide X {close to|next to|
# towards|into} Y"), so exact-string matching to the trained instructions NEVER fires.
# Instead we match on the ordered BLOCK PAIR (source → target): same task content as a
# trained instruction, tested under the sim's paraphrase (a fair in-distribution-by-
# content probe that also checks preposition/verb paraphrase robustness).
TRAINED_PAIRS = [
    ("green star", "green cube"),
    ("yellow pentagon", "yellow star"),
    ("green cube", "green star"),
    ("yellow star", "yellow pentagon"),
    ("blue cube", "blue crescent"),
]


def match_pair(ins: str):
    """Return the canonical 'src -> tgt' label if `ins` names a trained pair in order."""
    low = ins.lower()
    for src, tgt in TRAINED_PAIRS:
        i, j = low.find(f"the {src}"), low.find(f"the {tgt}")
        if i != -1 and j != -1 and i < j:
            return f"{src} -> {tgt}"
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-m)/d, (c+m)/d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-reset-tries", type=int, default=600)
    args = ap.parse_args()

    cfg = get_config("language_table", "exp02")
    device = cfg.get_device()
    keep = set(TRAINED_5)

    print("[1/3] Loading VLM (frozen) …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt",
                      map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    print(f"   checkpoint epoch={ckpt.get('epoch','?')} val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    agent = LanguageTableAgent(vlm, tm, cfg, device)
    env = make_lt_env("block2block", seed=42)

    print(f"[2/3] Running {args.episodes} IN-DISTRIBUTION episodes "
          f"(reset-filtered to the 5 trained BLOCK PAIRS; sim paraphrases the verb) …\n")
    results = []
    per_instr = defaultdict(lambda: [0, 0])   # pair label -> [successes, total]
    total_resets = 0

    for ep in range(args.episodes):
        # reset until the generated instruction names one of the 5 trained block pairs
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs); pair = match_pair(instr); tries = 0
        while pair is None and tries < args.max_reset_tries:
            out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
            instr = get_instruction(env, obs); pair = match_pair(instr); tries += 1
        total_resets += tries
        if pair is None:
            print(f"  Ep {ep+1:2d}: no trained pair in {args.max_reset_tries} resets — skipping")
            continue

        agent.reset(); agent.instruction = instr
        success, max_reward, reward, step = False, 0.0, 0.0, 0
        t0 = time.time()
        for step in range(cfg.sim_max_steps):
            image = resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h)
            state = np.asarray(obs["effector_translation"], dtype=np.float32)
            action = np.clip(agent.act(image, state), -0.1, 0.1).astype(np.float32)
            sout = env.step(action)
            obs, reward, done, info = (sout if len(sout) == 4 else
                                       (sout[0], sout[1], sout[2] or sout[3], sout[4]))
            reward = float(reward); max_reward = max(max_reward, reward)
            if reward > 0:
                success = True
            if done or success:
                break
        per_instr[pair][1] += 1
        per_instr[pair][0] += int(success)
        results.append({"episode": ep+1, "instruction": instr, "pair": pair,
                        "is_success": success, "max_reward": round(max_reward, 4),
                        "steps": step+1, "resets_to_match": tries,
                        "elapsed_sec": round(time.time()-t0, 1)})
        tag = "SUCCESS" if success else "failed "
        print(f"  Ep {ep+1:2d}/{args.episodes} | {tag} | reward={max_reward:.2f} | "
              f"steps={step+1} | resets={tries:3d} | {instr}")

    env.close()

    n = len(results); k = sum(r["is_success"] for r in results)
    lo, hi = wilson(k, n)
    print("\n" + "═"*60)
    print(f"  IN-DISTRIBUTION Results  [language_table/exp02_5instr]")
    print("═"*60)
    print(f"  Success rate      : {100*k/max(n,1):.1f}%  ({k}/{n})")
    print(f"  Wilson 95% CI     : [{100*lo:.0f}%, {100*hi:.0f}%]")
    print(f"  Avg resets/episode: {total_resets/max(args.episodes,1):.1f}  "
          f"(rarity of the 5 trained instructions in the sim)")
    print(f"\n  Per-block-pair success:")
    for src, tgt in TRAINED_PAIRS:
        label = f"{src} -> {tgt}"
        succ, tot = per_instr[label]
        rate = f"{100*succ/tot:.0f}%" if tot else "  n/a"
        print(f"    {succ}/{tot:<2d} ({rate:>4}) | {label}")
    print("═"*60)

    out_path = Path(cfg.output_dir) / "sim_results_indist.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": "language_table", "exp_id": "exp02_5instr", "mode": "in_distribution_by_pair",
        "n_episodes": n, "successes": k, "success_rate": k/max(n,1),
        "wilson_ci": [lo, hi], "avg_resets_per_ep": total_resets/max(args.episodes,1),
        "per_pair": {f"{s} -> {t}": per_instr[f"{s} -> {t}"] for s, t in TRAINED_PAIRS},
        "episodes": results,
    }, indent=2))
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
