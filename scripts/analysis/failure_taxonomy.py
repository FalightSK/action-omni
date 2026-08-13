"""
scripts/analysis/failure_taxonomy.py
────────────────────────────────────
Why does a 15.5% closed-loop gap leave open-loop action error unchanged?

The stage decomposition localises the whole ALOHA gap to P(handover | lift):
71.6% vs 59.7%, p = 0.0009. The head diagnostics then find the two policies
indistinguishable offline -- velocity loss +0.1%, per-dimension nMAE -0.4%. The
obvious reviewer question is what is physically different, and the aggregate
result dict cannot answer it: it records max_reward, which says an episode failed
but not HOW.

This reads the per-step traces written by `evaluate.py --trace` and classifies
the failures of episodes that LIFTED the cube but never completed the handover --
the exact population the stage decomposition isolates.

Arm identity is measured, not assumed. The 14-DOF vector's block order is a
convention this project has not verified, and an earlier analysis was wrong
because a 7-DOF layout was assumed for a 14-DOF robot. Here the picking arm is
identified as the block that moves first, from the data.

Gripper thresholds are percentiles of each run's own observed range rather than
absolute values, so nothing depends on the simulator's unit convention.

    python scripts/analysis/failure_taxonomy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "asset" / "analysis" / "failure_taxonomy"
ARMS = {
    "GR00T":    "asset/runs/aloha/exp05_groot_transfer",
    "Qwen3-VL": "asset/runs/aloha/exp06_qwen3vl_transfer",
}


def load_traces(run_dir: str):
    p = ROOT / run_dir / "trace" / "rollout_trace.npz"
    if not p.exists():
        raise FileNotFoundError(f"no trace at {p} — run evaluate.py --trace first")
    d = np.load(p)
    n = int(d["n_episodes"][0])
    return [{
        "state":  d[f"ep{i:04d}_state"],
        "action": d[f"ep{i:04d}_action"],
        "reward": d[f"ep{i:04d}_reward"],
        "attn":   d[f"ep{i:04d}_attn_image_mass"],
    } for i in range(n)]


def identify_arms(eps) -> tuple[slice, slice, int, int]:
    """Return (picker_block, receiver_block, picker_grip_idx, receiver_grip_idx).

    The picking arm is whichever 7-DOF block accumulates more motion in the FIRST
    third of successful episodes: the task is 'pick with one arm, transfer to the
    other', so the picker necessarily moves first. Measured rather than assumed.
    """
    a_early = b_early = 0.0
    for e in eps:
        if e["reward"].max() < 4:
            continue
        T = len(e["state"])
        if T < 30:
            continue
        seg = e["state"][: T // 3]
        d = np.abs(np.diff(seg, axis=0)).sum(axis=0)
        a_early += d[0:7].sum()
        b_early += d[7:14].sum()
    if a_early >= b_early:
        return slice(0, 7), slice(7, 14), 6, 13
    return slice(7, 14), slice(0, 7), 13, 6


def classify(eps, pick, recv, pg, rg) -> dict:
    """Classify episodes that reached lift (reward>=2) but never reached 4.

    GRIPPER SEMANTICS, verified from the trajectories rather than assumed. The
    receiving gripper (dim 6) rests near 0.19 and rises to ~0.96; the picking
    gripper (dim 13) opens to ~0.97 to approach and closes to ~0.65 to hold the
    cube. HIGH = OPEN. An earlier version of this function had it backwards and
    classified "closed for most of the post-lift window", which is the gripper's
    DEFAULT state (closed 84% of all timesteps) rather than a failure -- so it
    fired for nearly every episode of both arms and manufactured a null.

    The receive sequence is OPEN -> cube arrives -> CLOSE on it. The three ways
    it can fail are therefore: never open, open but never close, or close on
    nothing.
    """
    allg = np.concatenate([e["state"][:, rg] for e in eps])
    lo, hi = np.percentile(allg, 5), np.percentile(allg, 95)
    open_thr = lo + 0.6 * (hi - lo)

    counts = {"success": 0, "never_lifted": 0, "receiver_never_opened": 0,
              "opened_never_closed": 0, "closed_on_nothing": 0}
    open_latency = []

    for e in eps:
        r = e["reward"].astype(int)
        if r.max() >= 4:
            counts["success"] += 1
            continue
        if r.max() < 2:
            counts["never_lifted"] += 1
            continue
        t_lift = int(np.argmax(r >= 2))
        g = e["state"][t_lift:, rg]
        opened = np.flatnonzero(g > open_thr)
        if len(opened) == 0:
            counts["receiver_never_opened"] += 1
            continue
        open_latency.append(int(opened[0]))
        if (g[opened[0]:] < open_thr).any():
            counts["closed_on_nothing"] += 1
        else:
            counts["opened_never_closed"] += 1

    return {"counts": counts, "open_thr": float(open_thr),
            "open_latency_median": float(np.median(open_latency))
            if open_latency else float("nan")}


def attention_by_phase(eps, horizon: int = 8) -> dict:
    """Image-attention mass before vs after the lift, and in the handover window.

    One attention value per replan; replan k covers steps [k*horizon, (k+1)*horizon).
    """
    pre, post, hand = [], [], []
    for e in eps:
        a = e["attn"]
        if len(a) == 0:
            continue
        r = e["reward"].astype(int)
        if r.max() < 2:
            continue
        t_lift = int(np.argmax(r >= 2))
        k_lift = min(t_lift // horizon, len(a) - 1)
        pre.extend(a[:k_lift].tolist())
        post.extend(a[k_lift:].tolist())
        # handover window: the 5 replans following the lift
        hand.extend(a[k_lift:k_lift + 5].tolist())

    def s(x):
        return {"mean": float(np.mean(x)) if x else float("nan"),
                "sd": float(np.std(x)) if x else float("nan"), "n": len(x)}
    return {"pre_lift": s(pre), "post_lift": s(post), "handover_window": s(hand)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    for name, run_dir in ARMS.items():
        eps = load_traces(run_dir)
        pick, recv, pg, rg = identify_arms(eps)
        cls = classify(eps, pick, recv, pg, rg)
        att = attention_by_phase(eps)
        res[name] = {"n_episodes": len(eps),
                     "picker_dims": [pick.start, pick.stop],
                     "receiver_dims": [recv.start, recv.stop],
                     "receiver_gripper_dim": rg,
                     **cls, "attention": att}

        c = cls["counts"]
        tot = sum(c.values())
        lifted_fail = tot - c["success"] - c["never_lifted"]
        print(f"\n===== {name}   n={tot}")
        print(f"  picker dims {pick.start}-{pick.stop-1} | "
              f"receiver dims {recv.start}-{recv.stop-1} | "
              f"receiver gripper dim {rg}")
        print(f"  success {c['success']}  |  never lifted {c['never_lifted']}  |  "
              f"lifted-but-failed {lifted_fail}")
        for k in ("receiver_never_opened", "opened_never_closed",
                  "closed_on_nothing"):
            pct = 100.0 * c[k] / max(lifted_fail, 1)
            print(f"    {k:26s} {c[k]:4d}   ({pct:5.1f}% of lifted-but-failed)")
        a = att
        print(f"  image attention  pre-lift {a['pre_lift']['mean']:.4f}  "
              f"post-lift {a['post_lift']['mean']:.4f}  "
              f"handover window {a['handover_window']['mean']:.4f}")

    (OUT / "failure_taxonomy.json").write_text(json.dumps(res, indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {OUT / 'failure_taxonomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
