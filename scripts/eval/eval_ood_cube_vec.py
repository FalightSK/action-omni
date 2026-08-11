"""
scripts/eval_ood_cube_vec.py
─────────────────────────────
VECTORIZED out-of-distribution cube-position test for the Qwen-DiT policy.

Same experiment as eval_ood_cube.py, but K parallel envs run per cube cell and
their receding-horizon replans are BATCHED into one VLM+DiT forward (the GPU
bottleneck).  K parallel envs at a fixed cube pose = the N episodes for that cell
done in ~one episode's worth of GPU calls -> several x faster.

Single process + SyncVector-style manual stepping (mujoco step is cheap on CPU;
the win is batching the inference).  The cube pose is a module global
(gym_aloha BOX_POSE) so all K envs in the cell reset to the same forced position.

Output:
  asset/runs/aloha/<exp>/ood_cube_results.json
  asset/runs/aloha/<exp>/videos_ood/cube_x.._y.._IN|OOD.mp4   (env-0 per cell)
  docs/experiments/aloha/qwen_ood_generalization.png
"""

from __future__ import annotations
import os, sys, json, argparse, time
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from envs.aloha_env import annotate_frame
from data.aloha import resize_frame
import gymnasium as gym
import gym_aloha            # noqa: F401
import gym_aloha.env as gae

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="exp01")
ap.add_argument("--k", type=int, default=8, help="parallel envs per cube cell (= episodes/cell)")
ap.add_argument("--ih", type=int, default=None, help="override inference_horizon (<= action_horizon)")
ap.add_argument("--smoke", action="store_true", help="tiny 2-cell grid to verify")
args = ap.parse_args()

cfg = get_config("aloha", args.exp)
if args.ih is not None:
    assert args.ih <= cfg.action_horizon, f"ih={args.ih} > action_horizon={cfg.action_horizon}"
    cfg.inference_horizon = args.ih
IH_TAG = f"_ih{cfg.inference_horizon}" if args.ih is not None else ""
device = cfg.get_device()
torch.manual_seed(0)
K = args.k
print(f"VEC OOD test | exp={args.exp} | ih={cfg.inference_horizon} (predict {cfg.action_horizon}) | "
      f"device={device} | K={K} parallel envs/cell", flush=True)

amean = np.array(cfg.action_mean, np.float32); astd = np.array(cfg.action_std, np.float32)
smean = np.array(cfg.state_mean, np.float32);  sstd = np.array(cfg.state_std, np.float32)

# ── model ───────────────────────────────────────────────────────────────────────
vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
tm = VLATrainModel(cfg).to(device)
ck = torch.load(ROOT / cfg.output_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
tm.load_state_dict(ck["state_dict"]); tm.eval()
print("model loaded", flush=True)

# ── K parallel envs ──────────────────────────────────────────────────────────────
envs = [gym.make(cfg.env_id, obs_type="pixels_agent_pos",
                 max_episode_steps=cfg.sim_max_steps, disable_env_checker=True) for _ in range(K)]

_POSE = [(0.1, 0.5)]
def _fixed(seed=None):
    x, y = _POSE[0]
    return np.array([x, y, 0.05, 1, 0, 0, 0], dtype=float)
gae.sample_box_pose = _fixed


@torch.no_grad()
def batched_replan(images: list, states: np.ndarray) -> list[list[np.ndarray]]:
    """images: list[PIL] len M ; states: (M,14) raw -> list of M buffers (inference_horizon x (14,))."""
    st = torch.from_numpy(((states - smean) / (sstd + 1e-8)).astype(np.float32)).to(device)
    inp = vlm.build_vlm_inputs(images, [cfg.task_text] * len(images), device)
    tokens, img_mask = vlm.encode_vlm(inp)
    flat = tm.sample(tokens, st, num_steps=cfg.num_flow_steps, img_mask=img_mask)
    chunks = flat.view(len(images), cfg.action_horizon, cfg.action_dim).cpu().float().numpy()
    out = []
    for ch in chunks:
        acts = ch[: cfg.inference_horizon] * astd + amean
        out.append([a.astype(np.float32) for a in acts])
    return out


def run_cell(x, y, save_video, vid_path):
    """Run K parallel episodes at fixed cube (x,y). Returns (sr, mean_cov)."""
    _POSE[0] = (x, y)
    obs = [e.reset(seed=i * 7 + 1)[0] for i, e in enumerate(envs)]
    bufs = [[] for _ in range(K)]
    done = [False] * K
    maxr = [0] * K
    succ = [False] * K
    frames = []
    for step in range(cfg.sim_max_steps):
        need = [i for i in range(K) if not done[i] and not bufs[i]]
        if need:
            imgs = [resize_frame(np.asarray(obs[i]["pixels"]["top"]), cfg.aloha_img_w, cfg.aloha_img_h) for i in need]
            sts = np.stack([np.asarray(obs[i]["agent_pos"], np.float32) for i in need])
            newb = batched_replan(imgs, sts)
            for j, i in enumerate(need):
                bufs[i] = newb[j]
        for i in range(K):
            if done[i]:
                continue
            a = bufs[i].pop(0)
            obs[i], r, term, trunc, info = envs[i].step(a)
            r = int(r); maxr[i] = max(maxr[i], r)
            succ[i] = bool(info.get("is_success", r >= 4))
            if i == 0 and save_video:
                frames.append(annotate_frame(envs[0].render(), 0, K, step + 1,
                                              cfg.sim_max_steps, r, maxr[0], succ[0]))
            if term or trunc or succ[i]:
                done[i] = True
        if all(done):
            break
    if save_video and frames:
        try:
            import imageio
            imageio.mimsave(str(vid_path), frames, fps=cfg.video_fps)
        except Exception as e:
            print(f"  [warn] video: {e}", flush=True)
    sr = float(np.mean(succ)); cov = float(np.mean([m / 4.0 for m in maxr]))
    return sr, cov, maxr


# ── grid ──────────────────────────────────────────────────────────────────────────
if args.smoke:
    XS, YS = [0.10, 0.30], [0.50]
else:
    XS, YS = [-0.10, 0.00, 0.10, 0.20, 0.30], [0.30, 0.40, 0.50, 0.60, 0.70]

VIDDIR = ROOT / cfg.output_dir / "videos_ood"
VIDDIR.mkdir(parents=True, exist_ok=True)
results = []
sr_grid = np.zeros((len(YS), len(XS))); cov_grid = np.zeros((len(YS), len(XS)))
t0 = time.time()
for iy, y in enumerate(YS):
    for ix, x in enumerate(XS):
        indist = (0.0 <= x <= 0.2) and (0.4 <= y <= 0.6)
        tag = "IN" if indist else "OOD"
        vp = VIDDIR / f"cube_x{x:+.2f}_y{y:.2f}_{tag}.mp4"
        sr, cov, maxr = run_cell(x, y, save_video=True, vid_path=vp)
        sr_grid[iy, ix] = sr; cov_grid[iy, ix] = cov
        results.append({"x": x, "y": y, "n": K, "sr": sr, "cov": cov, "in_dist": indist,
                        "max_rewards": maxr})
        print(f">>> cube=({x:+.2f},{y:.2f}) {tag:3s} SR={sr*100:3.0f}%  cov={cov*100:3.0f}%  "
              f"rewards={maxr}  [{time.time()-t0:.0f}s]", flush=True)
for e in envs:
    e.close()

ind = [r for r in results if r["in_dist"]]; ood = [r for r in results if not r["in_dist"]]
summary = {"exp": args.exp, "k": K, "xs": XS, "ys": YS,
           "train_region": {"x": [0.0, 0.2], "y": [0.4, 0.6]},
           "mean_sr_in_dist": round(float(np.mean([r["sr"] for r in ind])), 3) if ind else None,
           "mean_sr_ood": round(float(np.mean([r["sr"] for r in ood])), 3) if ood else None,
           "grid": results}
outj = ROOT / cfg.output_dir / ("ood_cube_results_smoke.json" if args.smoke else f"ood_cube_results{IH_TAG}.json")
outj.write_text(json.dumps(summary, indent=2))
print(f"\nIN-DIST mean SR={summary['mean_sr_in_dist']}  OOD mean SR={summary['mean_sr_ood']}", flush=True)
print(f"total wall: {time.time()-t0:.0f}s  saved -> {outj}", flush=True)

if args.smoke:
    sys.exit(0)

# ── heatmaps ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, grid, name in [(axes[0], sr_grid, "Success rate"), (axes[1], cov_grid, "Mean task progress (reward/4)")]:
    im = ax.imshow(grid * 100, origin="lower", cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(XS))); ax.set_xticklabels([f"{v:+.2f}" for v in XS])
    ax.set_yticks(range(len(YS))); ax.set_yticklabels([f"{v:.2f}" for v in YS])
    ax.set_xlabel("cube x (m)"); ax.set_ylabel("cube y (m)")
    for iy in range(len(YS)):
        for ix in range(len(XS)):
            ax.text(ix, iy, f"{grid[iy, ix]*100:.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
    ax.add_patch(Rectangle((0.5, 0.5), 3, 3, fill=False, ec="blue", lw=2.5, ls="--"))
    ax.text(2, 3.62, "TRAINED REGION", ha="center", color="blue", fontsize=9, fontweight="bold")
    ax.set_title(f"{name}  (n={K}/cell)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label=name + " (%)")
fig.suptitle(
    f"Qwen-DiT ({args.exp}) spatial generalization: success vs cube position\n"
    f"dashed = trained region x[0,0.2] y[0.4,0.6]; outer ring ~0.1 m OUT of distribution   "
    f"|  in-dist {summary['mean_sr_in_dist']*100:.0f}% vs OOD {summary['mean_sr_ood']*100:.0f}%",
    fontsize=12.5, fontweight="bold", y=1.02)
outp = ROOT / "docs" / "experiments" / "aloha" / f"qwen_ood_generalization_{args.exp}{IH_TAG}.png"
outp.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outp, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved -> {outp}", flush=True)
