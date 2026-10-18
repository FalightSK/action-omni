"""
scripts/analysis/drift_analysis.py
──────────────────────────────────
Does the stock policy reach the handover already outside the demonstrated
region?

Everything measurable on the TRAINING distribution says the two arms are the
same: five offline metrics within 2%, open-loop action correlation 0.979-0.997,
and a probe built specifically for the receive cue (cue_probe_aloha.py) that
returns a null because the cue is fully determined by the picker arm's pose.
The 9.5-point closed-loop gap therefore lives somewhere the demonstrations do
not reach. This script tests the obvious candidate: compounding error.

    drift(t) = distance from the rollout state at t to the nearest demo state

ALOHA episodes run 400 steps and the receive fires around step 230, so there is
a long stretch in which the policy drives its own state before the transition
that decides the episode. If the stock arm is further off-manifold when it gets
there, the failure is a distribution-shift failure and not a perception one.

The circularity this must avoid
───────────────────────────────
Failed episodes drift more by definition -- a policy that never completes the
task ends up somewhere the demos never go, whatever the cause. Comparing raw
drift between arms would therefore rediscover the success-rate difference and
call it a mechanism. Three conditions prevent that:

  * only episodes that REACHED LIFT are compared, so both arms are measured on
    the same accomplished sub-task;
  * drift is measured in the window from lift onward but BEFORE the receive
    resolves, so it is predictive rather than descriptive;
  * a successes-only comparison is reported alongside, which removes the outcome
    confound entirely -- if the arms differ there, it cannot be an artifact of
    failing.

Scales
──────
The demo manifold is read from the parquet `observation.state`, not from the
embedding cache: the cache stores NORMALISED states while the rollout traces
record raw env values, and mixing them would measure the normalisation. Verified
here rather than assumed. Distances are standardised per dimension by the demo
std so that no joint dominates by having a larger range.

    python scripts/analysis/drift_analysis.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "asset" / "analysis" / "drift"

ARMS = {
    "GR00T N1.7 (pretrained)": "asset/runs/aloha/exp05_groot_transfer",
    "Qwen3-VL-2B (stock)":     "asset/runs/aloha/exp06_qwen3vl_transfer",
}

RECEIVER_GRIPPER_DIM = 6
LIFT_REWARD = 2          # environment ladder: contact 1, lift 2, handover 4
SUCCESS_REWARD = 4
WINDOW = 40              # frames after lift = the five post-lift replans
CHUNK = 2000             # rollout states per distance block
PRE_W, POST_W = 120, 60  # lift-aligned curve extent, for the figure


def demo_manifold() -> np.ndarray:
    """Raw-scale demonstration states, (20000, 14)."""
    from configs.registry import get_config

    cfg = get_config("aloha", "exp05")
    shards = sorted(glob.glob(str(
        Path(cfg.dataset_root) / "data" / "chunk-000" / "file-*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["index", "observation.state"])
                    for f in shards], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    return np.stack(df["observation.state"].to_numpy()).astype(np.float64)


def load_episodes(run_dir: Path) -> list[dict]:
    tr = np.load(run_dir / "trace" / "rollout_trace.npz")
    n = int(tr["n_episodes"])
    return [{"state": tr[f"ep{i:04d}_state"].astype(np.float64),
             "reward": tr[f"ep{i:04d}_reward"]} for i in range(n)]


def nn_distance(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of X to the nearest row of M.

    Blocked so the (len(X), len(M)) matrix never materialises in full.
    """
    m2 = (M ** 2).sum(1)
    out = np.empty(len(X))
    for s in range(0, len(X), CHUNK):
        x = X[s:s + CHUNK]
        d2 = (x ** 2).sum(1)[:, None] - 2.0 * x @ M.T + m2[None, :]
        out[s:s + CHUNK] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)

    M = demo_manifold()
    mu, sd = M.mean(0), M.std(0)
    sd[sd < 1e-8] = 1.0
    Mz = (M - mu) / sd
    print(f"demo manifold: {M.shape[0]} states, {M.shape[1]} dims\n")

    res: dict = {}
    per_arm: dict = {}
    curve_store: dict = {}

    for name, run_dir in ARMS.items():
        eps = load_episodes(ROOT / run_dir)

        # rollout scale must match the demos, or the distance is meaningless
        allst = np.concatenate([e["state"] for e in eps])
        shift = np.abs(np.median(allst, 0) - np.median(M, 0)) / sd
        assert shift.max() < 1.0, f"scale mismatch vs demos: {shift.round(2)}"

        g_all = allst[:, RECEIVER_GRIPPER_DIM]
        lo, hi = np.percentile(g_all, 5), np.percentile(g_all, 95)
        open_thr = lo + 0.6 * (hi - lo)

        rows = []
        offsets = np.arange(-PRE_W, POST_W)
        curves = []
        for e in eps:
            r = e["reward"]
            lifted = np.flatnonzero(r >= LIFT_REWARD)
            if len(lifted) == 0:
                continue
            t_lift = int(lifted[0])
            S = e["state"]
            d = nn_distance((S - mu) / sd, Mz)

            # lift-aligned curve, NaN where the episode does not extend that far
            c = np.full(len(offsets), np.nan)
            idx = t_lift + offsets
            ok = (idx >= 0) & (idx < len(d))
            c[ok] = d[idx[ok]]
            curves.append(c)

            g = S[t_lift:, RECEIVER_GRIPPER_DIM]
            opened = bool((g > open_thr).any())
            win = d[t_lift:t_lift + WINDOW]
            rows.append({
                "drift_pre_lift": float(d[:t_lift].mean()) if t_lift else np.nan,
                "drift_at_lift": float(d[t_lift]),
                "drift_post_lift": float(win.mean()) if len(win) else np.nan,
                "opened": opened,
                "success": bool((r >= SUCCESS_REWARD).any()),
            })

        df = pd.DataFrame(rows)
        per_arm[name] = df
        C = np.array(curves)
        n_ok = np.sum(~np.isnan(C), 0)
        curve_store[name] = {
            "offsets": offsets,
            "mean": np.nanmean(C, 0),
            "se": np.nanstd(C, 0) / np.sqrt(np.maximum(n_ok, 1)),
            "n": n_ok,
        }
        res[name] = {
            "n_lifted": int(len(df)),
            "n_success": int(df["success"].sum()),
            "n_never_opened": int((~df["opened"]).sum()),
            "drift_pre_lift": float(df["drift_pre_lift"].mean()),
            "drift_at_lift": float(df["drift_at_lift"].mean()),
            "drift_post_lift": float(df["drift_post_lift"].mean()),
            "drift_post_lift_success_only":
                float(df.loc[df["success"], "drift_post_lift"].mean()),
        }
        print(f"{name}")
        print(f"  episodes reaching lift : {len(df)}   successes {int(df['success'].sum())}"
              f"   never-opened {int((~df['opened']).sum())}")
        print(f"  mean drift pre-lift    : {res[name]['drift_pre_lift']:.4f}")
        print(f"  mean drift at lift     : {res[name]['drift_at_lift']:.4f}")
        print(f"  mean drift post-lift   : {res[name]['drift_post_lift']:.4f}")
        print(f"  post-lift, successes   : {res[name]['drift_post_lift_success_only']:.4f}\n")

    a, b = list(ARMS)
    A, B = per_arm[a], per_arm[b]

    print("BETWEEN ARMS  (only episodes that reached lift)")
    for key, label in [("drift_pre_lift", "pre-lift"),
                       ("drift_at_lift", "at lift"),
                       ("drift_post_lift", "post-lift window")]:
        t, p = welch(A[key].dropna().values, B[key].dropna().values)
        print(f"  {label:18s} pretrained {A[key].mean():.4f}  stock {B[key].mean():.4f}"
              f"   Welch t={t:+.2f}  p={p:.4f}")
        res.setdefault("between_arms", {})[key] = {"t": t, "p": p}

    print("\n  successes only (outcome confound removed)")
    t, p = welch(A.loc[A["success"], "drift_post_lift"].dropna().values,
                 B.loc[B["success"], "drift_post_lift"].dropna().values)
    print(f"  {'post-lift window':18s} pretrained "
          f"{A.loc[A['success'],'drift_post_lift'].mean():.4f}  stock "
          f"{B.loc[B['success'],'drift_post_lift'].mean():.4f}"
          f"   Welch t={t:+.2f}  p={p:.4f}")
    res["between_arms_success_only"] = {"t": t, "p": p}

    print("\nWITHIN ARM  (does post-lift drift predict the receive failing?)")
    for name, df in per_arm.items():
        o = df.loc[df["opened"], "drift_post_lift"].dropna().values
        n = df.loc[~df["opened"], "drift_post_lift"].dropna().values
        if len(n) < 5:
            print(f"  {name:26s} too few never-opened episodes ({len(n)})")
            continue
        t, p = welch(n, o)
        print(f"  {name:26s} never-opened {n.mean():.4f} (n={len(n)})  "
              f"opened {o.mean():.4f} (n={len(o)})   Welch t={t:+.2f}  p={p:.4f}")
        res.setdefault("within_arm", {})[name] = {
            "never_opened_mean": float(n.mean()), "opened_mean": float(o.mean()),
            "n_never_opened": int(len(n)), "n_opened": int(len(o)),
            "t": t, "p": p}

    # per-episode values the figure needs, so no number is retyped there
    flat = {}
    for i, (name, df) in enumerate(per_arm.items()):
        c = curve_store[name]
        flat[f"arm{i}_name"] = np.array([name])
        for k, v in c.items():
            flat[f"arm{i}_curve_{k}"] = v
        for col in ("drift_post_lift", "opened", "success"):
            flat[f"arm{i}_{col}"] = df[col].to_numpy()
    np.savez(OUT / "drift_curves.npz", **flat)

    (OUT / "drift.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'drift.json'} and drift_curves.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
