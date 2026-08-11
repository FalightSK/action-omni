"""
OOD test in the 'cube-position over the camera frame' style (like the diversity fig).

Left : where the 50 TRAINING cubes actually started (image space).
Right: the 5x5 OOD test cube positions on the same frame, colored by success rate
       -> green sits exactly on the training blob; everything beyond is red.

Renders each grid cube in gym_aloha to get its true image position; detects the
red cube; fits a world->image affine for the trained-region outline.

Output -> docs/experiments/aloha/qwen_ood_overlay.png
"""
from __future__ import annotations
import os, sys, glob, json
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from data.aloha.dataset import _SequentialFrameReader
import gymnasium as gym
import gym_aloha           # noqa
import gym_aloha.env as gae

cfg = get_config("aloha", "exp01")
root = Path(cfg.dataset_root)
OUT = ROOT / "docs/experiments/aloha/qwen_ood_overlay_combined.png"


def red_centroid(frame):
    hsv = mcolors.rgb_to_hsv(frame.astype(np.float32) / 255.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h < 0.045) | (h > 0.955)) & (s > 0.45) & (v > 0.25)
    ys, xs = np.where(mask)
    if len(xs) < 25:
        return np.nan, np.nan
    return float(np.median(xs)), float(np.median(ys))


# ── OOD success rates ────────────────────────────────────────────────────────────
ood = json.loads((ROOT / "asset/runs/aloha/exp01_baseline/ood_cube_results.json").read_text())
sr_of = {(round(e["x"], 2), round(e["y"], 2)): e["sr"] for e in ood["grid"]}
mr_of = {(round(e["x"], 2), round(e["y"], 2)): e.get("max_rewards", []) for e in ood["grid"]}
XS, YS = ood["xs"], ood["ys"]

# ── render each grid cube -> image position ─────────────────────────────────────
_POSE = [(0.1, 0.5)]
gae.sample_box_pose = lambda seed=None: np.array([_POSE[0][0], _POSE[0][1], 0.05, 1, 0, 0, 0], dtype=float)
env = gym.make(cfg.env_id, obs_type="pixels_agent_pos", max_episode_steps=cfg.sim_max_steps, disable_env_checker=True)

grid = []          # (x, y, u, v, sr, indist)
ref_frame = None
for y in YS:
    for x in XS:
        _POSE[0] = (x, y)
        obs, _ = env.reset(seed=0)
        fr = np.asarray(obs["pixels"]["top"])
        indist = (0.0 <= x <= 0.2) and (0.4 <= y <= 0.6)
        if indist and (x, y) == (0.1, 0.5):
            ref_frame = fr.copy()
        u, v = red_centroid(fr)
        grid.append((x, y, u, v, sr_of[(round(x, 2), round(y, 2))], indist))
env.close()
if ref_frame is None:
    ref_frame = np.asarray(obs["pixels"]["top"])
H, W = ref_frame.shape[:2]

# world->image affine from detected cells (for the trained-region outline + fallbacks)
det = [(x, y, u, v) for (x, y, u, v, _, _) in grid if not np.isnan(u)]
Wm = np.array([[d[0], d[1], 1] for d in det])
au = np.linalg.lstsq(Wm, np.array([d[2] for d in det]), rcond=None)[0]
av = np.linalg.lstsq(Wm, np.array([d[3] for d in det]), rcond=None)[0]
def proj(x, y):
    return float(au @ [x, y, 1]), float(av @ [x, y, 1])

# trained region corners x[0,0.2] y[0.4,0.6] -> image
box = np.array([proj(0.0, 0.4), proj(0.2, 0.4), proj(0.2, 0.6), proj(0.0, 0.6)])

# ── training cube positions (dataset, 50 episodes) ──────────────────────────────
shards = sorted(glob.glob(str(root / "data/chunk-000/file-*.parquet")))
import pandas as pd
df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True).sort_values("index").reset_index(drop=True)
ep_ids = df["episode_index"].to_numpy()
firsts = [int(df.index[ep_ids == e][0]) for e in sorted(df["episode_index"].unique())]
vid = sorted(glob.glob(str(root / "videos" / cfg.camera_key / "chunk-000" / "*.mp4")))[0]
rdr = _SequentialFrameReader(vid)
train_uv = []
for i in firsts:
    u, v = red_centroid(rdr.get(i))
    train_uv.append((u, v))
rdr.close()
train_uv = np.array(train_uv)

# ── figure ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.imshow(ref_frame, alpha=0.55)

# OOD: one scatter point per EPISODE -- blue = success, red = fail (jittered at the cube pos)
rng = np.random.RandomState(0)
sx, sy, fx, fy = [], [], [], []
for (x, y, u, v, sr, indist) in grid:
    if np.isnan(u):
        u, v = proj(x, y)
    for mr in mr_of[(round(x, 2), round(y, 2))]:
        jx, jy = u + rng.normal(0, 6), v + rng.normal(0, 6)
        if mr == 4:
            sx.append(jx); sy.append(jy)
        else:
            fx.append(jx); fy.append(jy)
ax.scatter(fx, fy, s=55, color="red", edgecolor="black", lw=0.4, alpha=0.85, zorder=4,
           label=f"OOD episode: FAIL ({len(fx)})")
ax.scatter(sx, sy, s=55, color="blue", edgecolor="black", lw=0.4, alpha=0.9, zorder=5,
           label=f"OOD episode: SUCCESS ({len(sx)})")

# training cube positions (gold)
ax.scatter(train_uv[:, 0], train_uv[:, 1], s=42, color="gold", edgecolor="black",
           lw=0.4, zorder=6, label="training cube starts (n=50)")
ax.add_patch(Polygon(box, closed=True, fill=False, ec="black", lw=2.5, ls="--",
                     zorder=7, label="trained region"))

ax.set_xlim(0, W); ax.set_ylim(H, 0)
ax.set_xlabel("image x (px)"); ax.set_ylabel("image y (px)")
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
ax.set_title("Qwen-DiT ALOHA OOD test: each dot = one episode (blue = success, red = fail)\n"
             "gold = where the 50 training cubes started;  black dashed = trained region\n"
             "blue (success) only on the training cluster; red (fail) everywhere beyond",
             fontsize=12, fontweight="bold")
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close()
print(f"detected {len(det)}/25 grid cubes; saved -> {OUT}")
