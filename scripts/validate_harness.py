"""
scripts/validate_harness.py
───────────────────────────
Does our closed-loop harness succeed when it is fed PERFECT actions?

Why this exists
───────────────
Pi-0.5's own report puts LIBERO-Goal near-ceiling; our Pi-0.5-backbone arm
reaches 33.0%. Most of that gap is explained by what we removed — the action
expert is discarded, the VLM is frozen and never sees LIBERO, and a 19.2M head
reads one hidden state instead of per-layer KV. But "explained" is not
"verified", and there is a failure mode that would look identical: every offline
number in this study is computed from the precomputed cache, while rollouts
re-encode live. Any mismatch between those two paths — image convention, state
layout, action scaling, control mode, horizon — degrades ONLY closed-loop
performance, which is exactly the signature we observe.

This script removes the policy from the loop entirely. It sets each task to a
demonstration's own recorded initial state and replays that demonstration's own
recorded actions through the same env construction, the same success detector
and the same step budget the evaluation uses.

Interpretation
──────────────
  ~100% success  the harness is correct. 33% is the cost of the ablation, and
                 can be reported as such.
  low success    the harness, not the policy, is the problem, and every SR in
                 Chapter 2 is suspect. Find it before publishing anything.

Two conditions are run because they fail differently:
  replay_full      all recorded actions, no step cap - tests env + success check
  replay_capped    same, truncated to cfg.sim_max_steps - tests the 400-step
                   budget the policy is held to

Run under vla_libero (the simulator stack is not installed in vla):
    MUJOCO_GL=wgl .../vla_libero/python.exe scripts/validate_harness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_DEMOS = 5          # per task; 10 tasks x 5 = 50 replays


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from configs.registry import get_config
    from envs.libero_env import make_env
    from libero.libero import benchmark, get_libero_path

    cfg = get_config("libero", "exp01")
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    files = sorted(Path(cfg.dataset_root).glob("*_demo.hdf5"))

    # The HDF5 files are sorted by filename; the benchmark indexes tasks its own
    # way. Match on the bddl file name recorded in each HDF5 rather than
    # assuming the two orders agree — if they do not, every replay would run the
    # right actions against the wrong task and fail for a bogus reason.
    bddl_to_task = {}
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        bddl_to_task[Path(t.bddl_file).name] = i

    print(f"replaying {N_DEMOS} demos x {len(files)} tasks "
          f"| max_steps={cfg.sim_max_steps}\n")
    rows = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            bddl = Path(f["data"].attrs["bddl_file_name"]).name
            demos = sorted(f["data"], key=lambda s: int(s.split("_")[1]))[:N_DEMOS]
            payload = [(d, np.asarray(f["data"][d]["actions"]),
                        np.asarray(f["data"][d].attrs["init_state"]
                                   if "init_state" in f["data"][d].attrs
                                   else f["data"][d]["states"][0]))
                       for d in demos]
        ti = bddl_to_task.get(bddl)
        if ti is None:
            print(f"[!] no benchmark task for {bddl}; skipping")
            continue
        task = suite.get_task(ti)
        # make_env takes a bddl PATH, not a task object — same construction the
        # evaluation uses, so this validates the env the policy actually meets.
        bddl_path = (Path(get_libero_path("bddl_files")) / task.problem_folder
                     / task.bddl_file)
        env = make_env(bddl_path, cfg)
        for dname, acts, init in payload:
            # settle=5 mirrors the policy rollout, which needs the scene to come
            # to rest after set_init_state. A REPLAY did not have those steps
            # when it was recorded, so they desynchronise it from the very first
            # action — the replay is chasing a state the demo never passed
            # through. Both are run so the two effects can be told apart:
            # settle=0 isolates harness correctness, settle=5 measures what the
            # settling itself costs.
            for settle in (0, 5):
                env.reset()
                env.set_init_state(init)
                for _ in range(settle):
                    env.step([0.0] * 6 + [-1.0])
                seq = acts[:cfg.sim_max_steps]
                ok = False
                for a in seq:
                    _, _, done, _ = env.step(a.tolist())
                    if done:
                        ok = True
                        break
                rows.append({"task": ti, "demo": dname, "settle": settle,
                             "success": bool(ok), "n_actions": int(len(seq))})
        env.close()
        cells = []
        for settle in (0, 5):
            sel = [r for r in rows if r["task"] == ti and r["settle"] == settle]
            cells.append(f"settle{settle} {100*np.mean([r['success'] for r in sel]):5.1f}%")
        print(f"  task {ti:>2}  " + "  ".join(cells) + f"   {task.language[:40]}")

    out = ROOT / "asset" / "runs" / "libero" / "harness_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n" + "=" * 60)
    for settle in (0, 5):
        sel = [r for r in rows if r["settle"] == settle]
        tag = "no settling steps  " if settle == 0 else "5 settling steps   "
        print(f"  ground-truth replay, {tag}"
              f"{100*np.mean([r['success'] for r in sel]):5.1f}%  "
              f"({sum(r['success'] for r in sel)}/{len(sel)})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
