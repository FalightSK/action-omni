"""
scripts/validate_harness_aloha.py
─────────────────────────────────
Does our gym-aloha wrapper drive the arms the way the demonstrations did?

Why this is not the same check as the LIBERO one
────────────────────────────────────────────────
For LIBERO we replayed each demo's own actions from its own recorded initial
state and asked whether the task still succeeded (~90%). That is impossible
here: `AlohaEnv.reset` samples the cube pose from `sample_box_pose(seed)` into a
module-level BOX_POSE global, and this dataset records only images, the 14-D arm
state and the 14-D action — no cube pose, no seed. So a demo's initial scene
cannot be reconstructed, and replaying its actions would drive the arms at a
cube that is somewhere else.

What CAN be validated is the part that silently broke things before: the action
plumbing. ALOHA actions are absolute target joint positions, and the arms track
them independently of where the cube is (until contact). So feeding a demo's
recorded actions and comparing the environment's resulting `agent_pos` against
that demo's recorded `observation.state` tests action ordering, scaling, control
mode and state extraction all at once — every failure mode that would make a
working policy look broken.

It deliberately does NOT validate success detection. That comes from gym-aloha's
own published reward, which we do not modify, and the external anchor is ACT at
87.6% on this dataset via LeRobot.

Interpretation
──────────────
  tracking error ~= the demo's own action->state lag   plumbing is correct
  large or growing error                               action space is wrong;
                                                       fix before training

Run under vla (gym_aloha is installed there, not in vla_libero):
    MUJOCO_GL=wgl .../vla/python.exe scripts/validate_harness_aloha.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "asset" / "data" / "aloha_sim_transfer_cube_human"
N_EPISODES = 5
N_STEPS = 120          # well before the cube is contacted in most demos


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import gymnasium as gym
    import gym_aloha  # noqa: F401  (registers the envs)

    files = sorted(glob.glob(str(DATA / "data" / "**" / "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    eps = sorted(df["episode_index"].unique())[:N_EPISODES]
    print(f"{len(files)} parquet files | {len(df):,} rows | using {len(eps)} episodes\n")

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    rows = []
    for ep in eps:
        d = df[df["episode_index"] == ep].sort_values("frame_index")
        act = np.stack(d["action"].to_numpy())[:N_STEPS]
        ref = np.stack(d["observation.state"].to_numpy())[:N_STEPS]

        obs, _ = env.reset(seed=int(ep))
        got = [np.asarray(obs["agent_pos"], dtype=np.float64)]
        for a in act[:-1]:
            obs, _, _, _, _ = env.step(np.asarray(a, dtype=np.float64))
            got.append(np.asarray(obs["agent_pos"], dtype=np.float64))
        got = np.stack(got)

        # The demo's OWN action->state lag is the floor: a perfect wrapper still
        # shows this, because a position target is not reached within one step.
        demo_lag = np.abs(act - ref).mean()
        err = np.abs(got - ref).mean()
        drift = np.abs(got[-10:] - ref[-10:]).mean()
        rows.append((int(ep), err, demo_lag, drift))
        print(f"  ep {ep:2d}  tracking err {err:.4f} rad   demo action-state lag "
              f"{demo_lag:.4f}   last-10-step err {drift:.4f}")

    env.close()
    e = np.array([r[1] for r in rows]); lag = np.array([r[2] for r in rows])
    dr = np.array([r[3] for r in rows])
    print(f"\n  mean tracking error {e.mean():.4f} rad | mean demo lag {lag.mean():.4f} rad"
          f" | ratio {e.mean()/lag.mean():.2f}x")
    print(f"  mean end-of-window error {dr.mean():.4f} rad (growing => wrong action space)")
    ok = e.mean() < 5 * lag.mean() and dr.mean() < 0.15
    print("\n  VERDICT:", "action plumbing OK" if ok else
          "SUSPECT — investigate action ordering/scaling before training")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
