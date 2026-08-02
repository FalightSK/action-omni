"""Does LIBERO render on NATIVE Windows via MUJOCO_GL=wgl?

Per grootplayvla/README.md: egl does not exist on Windows, osmesa is rejected,
wgl and glfw both work.  Verify with the pixel std (a black buffer still
"succeeds" but has std ~0).
"""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.pop("PYOPENGL_PLATFORM", None)

import numpy as np
import torch

# LIBERO's init_files are numpy pickles; torch>=2.6 defaults weights_only=True
# and refuses them. These ship with LIBERO itself, so loading them is safe.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict()["libero_goal"]()
print("n_tasks", bd.n_tasks)
t = bd.get_task(8)
print("task", t.language)

env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
    camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
o = env.set_init_state(bd.get_task_init_states(8)[0])
for _ in range(10):
    o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

img = o["agentview_image"]
print(f"image shape={img.shape} dtype={img.dtype} mean={img.mean():.1f} std={img.std():.1f}")
print("  -> REAL FRAME" if img.std() > 20 else "  -> BLACK/FLAT BUFFER, backend not working")

t0 = time.time()
N = 60
for _ in range(N):
    o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
ms = (time.time() - t0) / N * 1000
print(f"env.step+render = {ms:.1f} ms  -> 400-step episode = {ms*400/1000:.1f} s")

# state convention check (must match the training data)
def quat2aa(q):
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0)
    ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-12))
    return np.zeros(3) if s < 1e-6 else np.array([x, y, z]) / s * ang

st = np.concatenate([o["robot0_eef_pos"], quat2aa(o["robot0_eef_quat"]),
                     o["robot0_gripper_qpos"]])
print("state(8) =", np.round(st, 3).tolist())
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                     "win_render_probe.npy"), img)
env.close()
print("WIN_RENDER_OK")
