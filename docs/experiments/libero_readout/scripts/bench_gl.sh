#!/bin/bash
# Which MuJoCo GL backend works under WSL2, and at what speed?
for B in egl osmesa glfw; do
  echo "== MUJOCO_GL=$B =="
  MUJOCO_GL=$B PYOPENGL_PLATFORM=$B /opt/libero/venv/bin/python - 2>&1 <<'PY' | grep -viE "FutureWarning|hasattr|^\s*$"
import time, mujoco
XML = "<mujoco><worldbody><light pos='0 0 3'/><geom type='sphere' size='.2'/></worldbody></mujoco>"
try:
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 256, 256)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    img = r.render()
    t0 = time.time()
    for _ in range(30):
        r.update_scene(d)
        r.render()
    print("  OK", img.shape, round((time.time() - t0) / 30 * 1000, 1), "ms/frame")
except Exception as e:
    print("  FAIL", type(e).__name__, str(e)[:110])
PY
done
echo "--- cores / mem ---"
nproc
free -g | head -2
