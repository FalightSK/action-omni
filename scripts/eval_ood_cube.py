"""
scripts/eval_ood_cube.py
─────────────────────────
Out-of-distribution generalization test for the Qwen-DiT ALOHA policy.

The model trained only on cubes sampled from x in [0.0,0.2], y in [0.4,0.6]
(gym_aloha sample_box_pose).  Here we FORCE the cube to a 5x5 grid of fixed
positions -- the centre 3x3 is inside the training square, the outer ring is
~0.1 m OUTSIDE it -- and measure success at each, to see whether the policy
generalises spatially or only interpolates within the trained region.

Mechanism: monkeypatch gym_aloha.env.sample_box_pose to return a fixed pose.

Output:
  asset/runs/aloha/<exp>/ood_cube_results.json
  docs/experiments/aloha/qwen_ood_generalization.png
"""

from __future__ import annotations
import os, sys, json, argparse
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from envs.aloha_env import AlohaAgent, run_episode
import gymnasium as gym
import gym_aloha            # noqa: F401  (registers the env)
import gym_aloha.env as gae

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="exp01")
ap.add_argument("--n", type=int, default=8, help="episodes per grid cell")
args = ap.parse_args()

cfg = get_config("aloha", args.exp)
device = cfg.get_device()
torch.manual_seed(0)
print(f"OOD cube test | exp={args.exp} | device={device} | n/cell={args.n}")
print(f"training cube region: x in [0.0,0.2], y in [0.4,0.6]\n")

# ── model ───────────────────────────────────────────────────────────────────────
vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
tm = VLATrainModel(cfg).to(device)
ck = torch.load(ROOT / cfg.output_dir / "checkpoints" / "best.pt",
                map_location=device, weights_only=False)
tm.load_state_dict(ck["state_dict"]); tm.eval()
agent = AlohaAgent(vlm, tm, cfg, device)

env = gym.make(cfg.env_id, obs_type="pixels_agent_pos",
               max_episode_steps=cfg.sim_max_steps, disable_env_checker=True)

# ── force the cube to a chosen (x,y) ────────────────────────────────────────────
_POSE = [(0.1, 0.5)]
def _fixed(seed=None):
    x, y = _POSE[0]
    return np.array([x, y, 0.05, 1, 0, 0, 0], dtype=float)
gae.sample_box_pose = _fixed                       # monkeypatch

XS = [-0.10, 0.00, 0.10, 0.20, 0.30]               # training x in [0,0.2]
YS = [0.30, 0.40, 0.50, 0.60, 0.70]                # training y in [0.4,0.6]
N = args.n

VIDDIR = ROOT / cfg.output_dir / "videos_ood"
VIDDIR.mkdir(parents=True, exist_ok=True)

results = []
sr_grid = np.zeros((len(YS), len(XS)))
cov_grid = np.zeros((len(YS), len(XS)))
counter = 0
total = len(XS) * len(YS) * N
for iy, y in enumerate(YS):
    for ix, x in enumerate(XS):
        _POSE[0] = (x, y)
        indist = (0.0 <= x <= 0.2) and (0.4 <= y <= 0.6)
        succ, cov = [], []
        for k in range(N):
            save = (k == 0)                         # one annotated video per cube position
            ep_idx = counter
            r = run_episode(env, agent, cfg, ep_idx, total, save, VIDDIR)
            counter += 1
            succ.append(r["is_success"]); cov.append(r["max_coverage"])
            if save:
                src = VIDDIR / f"episode_{ep_idx + 1:02d}.mp4"
                tag = "IN" if indist else "OOD"
                dst = VIDDIR / f"cube_x{x:+.2f}_y{y:.2f}_{tag}.mp4"
                if src.exists():
                    src.replace(dst)
        sr = float(np.mean(succ)); mc = float(np.mean(cov))
        sr_grid[iy, ix] = sr; cov_grid[iy, ix] = mc
        results.append({"x": x, "y": y, "n": N, "sr": sr, "cov": mc, "in_dist": indist})
        print(f">>> cube=({x:+.2f},{y:.2f}) {'IN ' if indist else 'OOD'} "
              f"SR={sr*100:3.0f}%  cov={mc*100:3.0f}%  ({counter}/{total} eps)\n", flush=True)
env.close()

# ── summary ──────────────────────────────────────────────────────────────────────
ind = [r for r in results if r["in_dist"]]
ood = [r for r in results if not r["in_dist"]]
summary = {
    "exp": args.exp, "n_per_cell": N, "xs": XS, "ys": YS,
    "train_region": {"x": [0.0, 0.2], "y": [0.4, 0.6]},
    "mean_sr_in_dist": round(float(np.mean([r["sr"] for r in ind])), 3),
    "mean_sr_ood": round(float(np.mean([r["sr"] for r in ood])), 3),
    "grid": results,
}
outj = ROOT / cfg.output_dir / "ood_cube_results.json"
outj.write_text(json.dumps(summary, indent=2))
print(f"\nIN-DIST mean SR = {summary['mean_sr_in_dist']*100:.0f}%   "
      f"OOD mean SR = {summary['mean_sr_ood']*100:.0f}%")
print(f"saved -> {outj}")

# ── heatmaps ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, grid, name in [(axes[0], sr_grid, "Success rate"), (axes[1], cov_grid, "Mean task progress (reward/4)")]:
    im = ax.imshow(grid * 100, origin="lower", cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(XS))); ax.set_xticklabels([f"{v:+.2f}" for v in XS])
    ax.set_yticks(range(len(YS))); ax.set_yticklabels([f"{v:.2f}" for v in YS])
    ax.set_xlabel("cube x  (m)"); ax.set_ylabel("cube y  (m)")
    for iy in range(len(YS)):
        for ix in range(len(XS)):
            ax.text(ix, iy, f"{grid[iy, ix]*100:.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
    # training square = x in[0,0.2] (cols 1..3), y in[0.4,0.6] (rows 1..3)
    ax.add_patch(Rectangle((0.5, 0.5), 3, 3, fill=False, ec="blue", lw=2.5, ls="--"))
    ax.text(2, 3.62, "TRAINED REGION", ha="center", color="blue", fontsize=9, fontweight="bold")
    ax.set_title(f"{name}  (n={N}/cell)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label=name + " (%)")
fig.suptitle(
    f"Qwen-DiT ({args.exp}) spatial generalization: success vs cube position\n"
    f"dashed = trained region (x[0,0.2] y[0.4,0.6]); outer ring = ~0.1 m OUT of distribution   "
    f"|  in-dist {summary['mean_sr_in_dist']*100:.0f}%  vs  OOD {summary['mean_sr_ood']*100:.0f}%",
    fontsize=12.5, fontweight="bold", y=1.02)
outp = ROOT / "docs" / "experiments" / "aloha" / "qwen_ood_generalization.png"
outp.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outp, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved -> {outp}")
