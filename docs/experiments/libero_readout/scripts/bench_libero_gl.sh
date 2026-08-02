#!/bin/bash
# Real LIBERO scene throughput per GL backend (the trivial-sphere benchmark is
# not representative - the kitchen scene is ~20x heavier).
for B in osmesa glfw; do
  echo "===== MUJOCO_GL=$B ====="
  MUJOCO_GL=$B PYOPENGL_PLATFORM=$B /opt/libero/venv/bin/python - 2>&1 <<'PY' | grep -viE "FutureWarning|hasattr|robosuite WARNING|^\s*$|^\[info\]|Downloading|Fetching|Assets"
import os, time
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
try:
    bd = benchmark.get_benchmark_dict()["libero_goal"]()
    t = bd.get_task(8)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
        camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    o = env.set_init_state(bd.get_task_init_states(8)[0])
    for _ in range(10):
        o, _, _, _ = env.step([0,0,0,0,0,0,-1])
    t0 = time.time()
    for _ in range(60):
        o, _, _, _ = env.step([0,0,0,0,0,0,-1])
    ms = (time.time() - t0) / 60 * 1000
    img = o["agentview_image"]
    print(f"  OK  env.step+render = {ms:.1f} ms   img_mean={img.mean():.1f}")
    print(f"  -> 400-step episode = {ms*400/1000:.1f} s")
    env.close()
except Exception as e:
    print("  FAIL", type(e).__name__, str(e)[:120])
PY
done
echo BENCH_DONE
