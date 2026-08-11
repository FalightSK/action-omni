#!/bin/bash
# Which MuJoCo GL backend actually renders under WSL2, and how fast?
ROOT=/opt/libero
cd $ROOT
for BACKEND in egl osmesa; do
  echo "===== MUJOCO_GL=$BACKEND ====="
  MUJOCO_GL=$BACKEND PYOPENGL_PLATFORM=$BACKEND $ROOT/venv/bin/python - 2>&1 <<'PY' | grep -viE "FutureWarning|hasattr|^\s*$"
import os, time
import mujoco
try:
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><light pos='0 0 3'/>"
        "<geom type='sphere' size='.2' rgba='1 0 0 1'/></worldbody></mujoco>")
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 256, 256)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    img = r.render()
    t0 = time.time()
    for _ in range(30):
        r.update_scene(d); r.render()
    dt = (time.time() - t0) / 30
    print(f"  OK  shape={img.shape} nonzero={int((img>0).sum())} {dt*1000:.1f} ms/frame")
except Exception as e:
    print("  FAIL", type(e).__name__, str(e)[:130])
PY
done
echo "===== LIBERO env, best backend ====="
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa $ROOT/venv/bin/python - 2>&1 <<'PY' | grep -viE "FutureWarning|hasattr|robosuite WARNING|^\s*$|^\[info\]"
import os, time, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bd = benchmark.get_benchmark_dict()["libero_goal"]()
t = bd.get_task(8)
env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
    camera_heights=256, camera_widths=256)
env.seed(0); env.reset()
o = env.set_init_state(bd.get_task_init_states(8)[0])
for _ in range(10):
    o, _, _, _ = env.step([0,0,0,0,0,0,-1])
img = o["agentview_image"]
print("  render OK", img.shape, "mean", round(float(img.mean()),1))
t0 = time.time()
for _ in range(100):
    o, _, _, _ = env.step([0,0,0,0,0,0,-1])
print(f"  env.step+render {(time.time()-t0)/100*1000:.1f} ms")
env.close()
PY
echo RENDER_TEST_DONE
