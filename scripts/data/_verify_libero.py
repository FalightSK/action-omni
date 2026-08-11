"""Verify LIBERO actually runs closed-loop on this machine.

Run with the vla_libero interpreter:
    C:/Users/SK/miniconda3/envs/vla_libero/python.exe scripts/data/_verify_libero.py

This is the go/no-go spike for the anatomy study's language axis. Offline probes
only establish that information is *available* in a representation; a claim that
a component is *needed* requires closed-loop rollouts, and rollouts require the
simulator to work. LIBERO is Linux-oriented (its own __init__ prompts on stdin),
so this must be demonstrated rather than assumed before any pipeline is built on
top of it.

Checks, in increasing order of what they'd cost to discover later:
  1. benchmark registry loads and libero_goal lists its 10 tasks
  2. the bddl problem file for task 0 resolves on disk
  3. OffScreenRenderEnv constructs (this is where MuJoCo/EGL usually fails)
  4. reset() returns an observation with camera images
  5. step() advances physics and returns a well-formed observation
  6. the rendered frame is not a blank/constant image (a silent renderer failure
     returns valid-shaped all-zero or all-grey arrays, which passes 3-5)
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Must precede any robosuite import: it validates MUJOCO_GL at module scope and
# raises on values it does not recognise. This machine has MUJOCO_GL=egl set
# globally (EGL is Linux-only — it comes from an unrelated Isaac/Linux setup),
# which makes robosuite raise before rendering is ever attempted. On Windows the
# working backend is WGL.
if sys.platform == "win32" and os.environ.get("MUJOCO_GL", "").lower() != "wgl":
    os.environ["MUJOCO_GL"] = "wgl"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "asset" / "analysis" / "libero_setup"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = True

    print("[1] benchmark registry")
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    print(f"    libero_goal: {suite.n_tasks} tasks")
    for i in range(suite.n_tasks):
        print(f"      {i}: {suite.get_task(i).language}")

    print("\n[2] bddl problem file")
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print(f"    {bddl.name} exists={bddl.exists()}")
    if not bddl.exists():
        return 1

    print("\n[3] constructing OffScreenRenderEnv (MuJoCo render path)")
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=128,
        camera_widths=128,
    )
    print("    constructed")

    try:
        print("\n[4] reset()")
        env.seed(0)
        obs = env.reset()
        keys = sorted(k for k in obs if "image" in k or "state" in k)
        print(f"    obs keys (image/state): {keys}")
        img_key = next((k for k in obs if k.endswith("_image")), None)
        if img_key is None:
            print("    FAIL: no camera image in observation")
            return 1
        img = np.asarray(obs[img_key])
        print(f"    {img_key}: shape={img.shape} dtype={img.dtype} "
              f"min={img.min()} max={img.max()}")

        print("\n[5] step()")
        dim = env.env.action_dim
        print(f"    action_dim={dim}")
        rng = np.random.default_rng(0)
        for i in range(5):
            obs, reward, done, info = env.step(rng.uniform(-0.05, 0.05, dim))
        img2 = np.asarray(obs[img_key])
        print(f"    stepped 5x -> reward={reward} done={done}")

        print("\n[6] renderer sanity (frames must not be constant)")
        span = int(img.max()) - int(img.min())
        moved = float(np.abs(img2.astype(np.int32) - img.astype(np.int32)).mean())
        print(f"    frame value span={span}  mean |Δ| after stepping={moved:.3f}")
        if span < 10:
            print("    FAIL: frame is near-constant — renderer produced a blank image")
            ok = False
        else:
            print("    frame has real image content")

        OUT.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
            # LIBERO/robosuite return camera frames bottom-up; flip so the saved
            # PNG is human-checkable rather than upside down.
            imageio.imwrite(OUT / "libero_goal_task0_reset.png", img[::-1])
            print(f"    wrote {OUT / 'libero_goal_task0_reset.png'}")
        except Exception as e:
            print(f"    (could not save PNG: {e})")
    finally:
        env.close()

    print("\nRESULT:", "LIBERO_OK" if ok else "LIBERO_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\nRESULT: LIBERO_FAILED")
        sys.exit(1)
