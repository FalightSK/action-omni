"""
scripts/diagnose_lt_success.py
──────────────────────────────
Validity check for the Language Table success metric (is reward>0 actually earned?).

Part 1 (instant): from the saved eval JSON, the distribution of steps-to-success per
   command — a pile of successes at step 1-3 would mean trivial/pre-satisfied goals.
Part 2 (sim): a NO-OP baseline — run a do-nothing policy (zero action) under the EXACT
   eval setup. If a no-op succeeds, the success was not earned by the policy.  Also
   reports how often the goal is already satisfied at reset (env.succeeded at step 0).

Run in vla_lt.  Output: console table.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "asset/runs/language_table/exp01_baseline"
REQUIRE = {"separate": "separate", "point": "point"}
CMDS = ["block2block", "block2absolute", "separate", "point"]
N_NOOP = 25
MAX_STEPS = 200


def part1_steps_to_success():
    print("\n" + "=" * 64)
    print("PART 1 — steps-to-success distribution (trained policy, from eval JSON)")
    print("=" * 64)
    ood = json.loads((RUN / "ood_results.json").read_text())["by_reward"]
    for r in CMDS:
        if r not in ood:
            continue
        succ = [e["steps"] for e in ood[r]["episodes"] if e["is_success"]]
        if not succ:
            print(f"  {r:15s} no successes")
            continue
        succ = np.array(succ)
        le3 = int((succ <= 3).sum()); le5 = int((succ <= 5).sum())
        print(f"  {r:15s} n_succ={len(succ):2d}  steps min={succ.min():3d} "
              f"p25={np.percentile(succ,25):.0f} median={np.median(succ):.0f} "
              f"max={succ.max():3d}  | ≤3 steps: {le3}, ≤5 steps: {le5}")


def part2_noop():
    print("\n" + "=" * 64)
    print(f"PART 2 — NO-OP baseline (zero action) + goal-at-reset, n={N_NOOP}/command")
    print("=" * 64)
    from envs.language_table_env import make_lt_env, get_instruction
    rng_seed = 123
    rows = []
    for r in CMDS:
        env = make_lt_env(r, seed=rng_seed)
        require = REQUIRE.get(r)
        pre_sat, noop_succ, first_steps = 0, 0, []
        for _ in range(N_NOOP):
            out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
            instr = get_instruction(env, obs)
            if require:
                t = 0
                while require not in instr.lower() and t < 80:
                    out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
                    instr = get_instruction(env, obs); t += 1
            try:
                if bool(env.succeeded):
                    pre_sat += 1
            except Exception:
                pass
            success = False
            for step in range(MAX_STEPS):
                obs, reward, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
                if reward > 0:
                    success = True; first_steps.append(step + 1)
                if done or success:
                    break
            if success:
                noop_succ += 1
        try:
            env.close()
        except Exception:
            pass
        fs = f"(first-success steps {sorted(first_steps)})" if first_steps else ""
        rows.append((r, pre_sat, noop_succ))
        print(f"  {r:15s} goal-already-met-at-reset: {pre_sat:2d}/{N_NOOP}   "
              f"NO-OP success: {noop_succ:2d}/{N_NOOP}  {fs}")
    print("\n  interpretation:")
    print("   - NO-OP SR ≈ 0  → successes are earned by the policy (metric valid)")
    print("   - NO-OP SR high → goals trivially satisfied (metric too easy / inflated)")
    return rows


if __name__ == "__main__":
    part1_steps_to_success()
    part2_noop()
    print("\nDIAGNOSE OK")
