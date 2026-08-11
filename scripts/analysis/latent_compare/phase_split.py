"""Does a backbone know the goal before it becomes visible?

LIBERO-Goal's ten tasks share one scene and object set (initial object positions
jitter slightly, but independently of task). So at the START of an episode the
frame carries essentially no task information — the robot has not acted yet, and
nothing in the image distinguishes "open the drawer" from "turn on the stove".

That makes episode phase a natural control. Splitting instruction eta^2 by phase
separates two very different abilities:

    high eta^2 EARLY   the goal is known before it is visible -> it can only
                       have come from the instruction
    rising eta^2       the goal is being inferred from the unfolding trajectory,
                       which is ordinary visual discrimination

The pooled number this study reported first (0.16-0.24 for every arm except
Pi-0.5) mixed both regimes and hid the distinction: pooling across phase puts
episode progress into the denominator, which flattens every arm toward a similar
value. Controlling for phase separates them cleanly.

The null is permuted at EPISODE level, not frame level: frames within an episode
share a task, so shuffling per frame would leave task information intact through
the episode structure and understate the floor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import _pcs, eta_squared  # noqa: E402
from backbones import ARMS  # noqa: E402

DIR = HERE.parents[2] / "asset" / "analysis" / "latent_compare"
KEY = "libero_goal"
POOL = "image"          # the question is about VISUAL representation
N_PC = 50
N_PERM = 40

BINS = [("early", lambda p: p < 0.10),
        ("mid", lambda p: (p >= 0.45) & (p <= 0.55)),
        ("late", lambda p: p > 0.90)]


def main() -> int:
    with h5py.File(DIR / f"probe_{KEY}.h5", "r") as f:
        phase, task, episode = f["phase"][:], f["instr_id"][:], f["episode"][:]

    rng = np.random.default_rng(0)
    out: dict = {}
    print(f"{'arm':<11}{'early':>8}{'mid':>8}{'late':>8}{'null p95':>10}{'excess':>9}")
    for arm in ARMS:
        p = DIR / f"latents_{arm}_{KEY}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            X = f[f"doc_{POOL}"][:]

        row = {}
        for name, sel in BINS:
            m = sel(phase)
            row[name] = float(eta_squared(_pcs(X[m], N_PC), task[m]))

        # episode-level permutation null on the early bin
        m = BINS[0][1](phase)
        Z = _pcs(X[m], N_PC)
        eps = np.unique(episode[m])
        ep_task = {int(e): int(task[m][episode[m] == e][0]) for e in eps}
        nulls = []
        for _ in range(N_PERM):
            lut = dict(zip(ep_task.keys(), rng.permutation(list(ep_task.values()))))
            nulls.append(eta_squared(Z, np.array([lut[int(e)] for e in episode[m]])))
        row["null_p95"] = float(np.percentile(nulls, 95))
        row["excess_early"] = row["early"] - row["null_p95"]
        row["n_early"] = int(m.sum())
        out[arm] = row
        print(f"{arm:<11}{row['early']:>8.3f}{row['mid']:>8.3f}{row['late']:>8.3f}"
              f"{row['null_p95']:>10.3f}{row['excess_early']:>+9.3f}")

    dst = DIR / "phase_split_libero.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
