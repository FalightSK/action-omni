"""
scripts/analysis/policy_agreement.py
────────────────────────────────────
Why is the backbone comparison a null on LIBERO but not on ALOHA?

The aggregate numbers say the two arms are 2.5 points apart on LIBERO and 9.5 on
ALOHA. That is compatible with two very different stories:

  (a) on LIBERO the two arms converged to behaviourally equivalent policies, or
  (b) they behave differently but their differences cancel in the aggregate.

Both testbeds are paired episode-for-episode -- LIBERO by its 50 FIXED initial
states, ALOHA by seed -- so the arms can be compared per episode rather than only
in aggregate, which distinguishes (a) from (b).

Two cautions are built into the reporting rather than left to the reader:

  * Cohen's kappa is unstable when the marginals are extreme. With both LIBERO
    arms near 90%, "both fail" is a handful of episodes and kappa swings on them.
    The raw 2x2 counts are printed and are what any claim should rest on.
  * Per-task success at n=20 has a standard error near 9.5 points on a
    difference, so a +/-20 point per-task gap is roughly 2 SE and several are
    expected across 10 tasks by chance alone. The script prints that SE so the
    per-task table cannot be read as if the differences were resolved.

    python scripts/analysis/policy_agreement.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "asset" / "runs"
OUT = ROOT / "asset" / "analysis" / "policy_agreement"


def libero(exp_dir):
    """Canonical-condition successes, ordered by (task, episode) so arms align."""
    j = json.loads((RUNS / "libero" / exp_dir / "eval_libero.json").read_text())
    rs = [x for x in j["results"] if x["condition"] == "canonical"]
    rs.sort(key=lambda x: (x["task"], x["episode"]))
    return (np.array([bool(x["success"]) for x in rs]),
            np.array([x["task"] for x in rs]))


def aloha(exp_dir):
    """Run 1 only (seed offset 0) -- the run both arms share episode-for-episode."""
    j = json.loads((RUNS / "aloha" / exp_dir / "sim_results.json").read_text())
    e = j["episodes"] if isinstance(j, dict) and "episodes" in j else j
    e = sorted(e, key=lambda x: x["episode"])
    return np.array([bool(x["is_success"]) for x in e])


def agreement(a, b, label):
    n = len(a)
    both = int((a & b).sum())
    neither = int((~a & ~b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    obs = (both + neither) / n
    pa, pb = a.mean(), b.mean()
    chance = pa * pb + (1 - pa) * (1 - pb)
    kappa = (obs - chance) / (1 - chance) if chance < 1 else float("nan")

    print(f"\n=== {label}   n={n}")
    print(f"  success rate:  pretrained {100*pa:.1f}%   stock {100*pb:.1f}%")
    print(f"  both succeed {both:4d} | both fail {neither:4d} | "
          f"only pretrained {only_a:3d} | only stock {only_b:3d}")
    print(f"  agreement {100*obs:.1f}%  (chance {100*chance:.1f}%)   "
          f"kappa {kappa:+.3f}")
    if min(pa, pb) > 0.85 or max(pa, pb) < 0.15:
        print("  [!] marginals are extreme -- kappa is unstable here; "
              "rely on the raw counts")
    return {"n": n, "both_succeed": both, "both_fail": neither,
            "only_pretrained": only_a, "only_stock": only_b,
            "agreement": float(obs), "chance": float(chance),
            "kappa": float(kappa)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    lg, tasks = libero("exp05_groot_2view")
    lq, _ = libero("exp06_qwen3vl_2view")
    res["libero_2view"] = agreement(lg, lq, "LIBERO-Goal 2-view (GR00T vs Qwen3-VL)")

    ag, aq = aloha("exp05_groot_transfer"), aloha("exp06_qwen3vl_transfer")
    m = min(len(ag), len(aq))
    res["aloha"] = agreement(ag[:m], aq[:m], "ALOHA transfer-cube (GR00T vs Qwen3-VL)")

    n_tasks = int(tasks.max()) + 1
    sg = np.array([lg[tasks == t].mean() for t in range(n_tasks)])
    sq = np.array([lq[tasks == t].mean() for t in range(n_tasks)])
    per_task_n = int((tasks == 0).sum())
    # SE on a per-task DIFFERENCE, at the observed base rate.
    p = float((sg.mean() + sq.mean()) / 2)
    se = math.sqrt(2 * p * (1 - p) / per_task_n)

    print(f"\n=== PER-TASK SUCCESS, LIBERO ({n_tasks} tasks, n={per_task_n} each)")
    for t in range(n_tasks):
        print(f"  task {t}:  pretrained {100*sg[t]:5.1f}%   stock {100*sq[t]:5.1f}%"
              f"   diff {100*(sg[t]-sq[t]):+6.1f}")
    print(f"  correlation r = {np.corrcoef(sg, sq)[0,1]:.3f}   "
          f"mean |diff| = {100*np.abs(sg-sq).mean():.1f} pts")
    print(f"  SE on a per-task difference = {100*se:.1f} pts  -> a +/-{100*2*se:.0f} "
          f"pt gap is about 2 SE, and several are expected across {n_tasks} tasks "
          f"by chance")

    res["libero_per_task"] = {
        "n_per_task": per_task_n,
        "pretrained": [float(x) for x in sg],
        "stock": [float(x) for x in sq],
        "correlation": float(np.corrcoef(sg, sq)[0, 1]),
        "mean_abs_diff": float(np.abs(sg - sq).mean()),
        "se_of_difference": float(se),
    }

    (OUT / "policy_agreement.json").write_text(json.dumps(res, indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {OUT / 'policy_agreement.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
