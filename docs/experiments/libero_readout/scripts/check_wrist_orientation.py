"""SmolVLA consumes TWO cameras. We only ever validated agentview's orientation.

Determine the correct transform for BOTH cameras by differencing rendered frames
against the training dataset, exactly as was done for agentview (where the
conventional [::-1] turned out to give a MIRRORED scene and rot180 was correct).
"""
import io, os, pickle, sys
os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.pop("PYOPENGL_PLATFORM", None)

import numpy as np
import pandas as pd
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

paths = pickle.load(open(os.path.join(RES, "paths.pkl"), "rb"))
cols = ["observation.images.image", "observation.images.image2",
        "frame_index", "task_index"]
d = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)

bd = benchmark.get_benchmark_dict()["libero_goal"]()
SIM_TID, DS_TI = 8, 10          # "put the bowl on the plate"
t = bd.get_task(SIM_TID)
env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                t.problem_folder, t.bddl_file),
    camera_heights=256, camera_widths=256,
    camera_names=["agentview", "robot0_eye_in_hand"])
env.seed(0); env.reset()

sub = d[(d.task_index == DS_TI) & (d.frame_index == 0)]
def dec(col):
    return np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))
                     for b in sub[col]]).astype(np.float32)
ds_main, ds_wrist = dec("observation.images.image"), dec("observation.images.image2")
print(f"dataset first-frames: {len(ds_main)}")

TR = {"raw": lambda a: a, "vflip": lambda a: a[::-1],
      "hflip": lambda a: a[:, ::-1], "rot180": lambda a: a[::-1, ::-1]}
best = {("main", k): 1e18 for k in TR}
best.update({("wrist", k): 1e18 for k in TR})

for ii in range(10):
    o = env.set_init_state(bd.get_task_init_states(SIM_TID)[ii])
    for _ in range(10):
        o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    sim_main = o["agentview_image"].astype(np.float32)
    sim_wrist = o.get("robot0_eye_in_hand_image")
    if sim_wrist is None:
        print("!! wrist camera not rendered - check camera_names"); break
    sim_wrist = sim_wrist.astype(np.float32)
    for k, f in TR.items():
        best[("main", k)] = min(best[("main", k)],
                                ((ds_main - f(sim_main)[None]) ** 2).mean(axis=(1, 2, 3)).min())
        best[("wrist", k)] = min(best[("wrist", k)],
                                 ((ds_wrist - f(sim_wrist)[None]) ** 2).mean(axis=(1, 2, 3)).min())
env.close()

for cam in ("main", "wrist"):
    print(f"\n{cam} camera - best MSE per transform (lower = correct):")
    ranked = sorted(TR, key=lambda k: best[(cam, k)])
    for k in ranked:
        print(f"   {k:7s} {best[(cam,k)]:10.1f}")
    print(f"   -> USE {ranked[0].upper()}  "
          f"({best[(cam,ranked[0])]:.0f} vs next {best[(cam,ranked[1])]:.0f})")
