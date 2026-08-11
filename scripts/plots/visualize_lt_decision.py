"""
scripts/visualize_lt_decision.py
────────────────────────────────
Visualise the model's DECISION: the predicted action chunk integrated into a planned
effector trajectory, drawn on the scene.

Row 1 (3 scenes): the model's planned path (16 predicted ee-deltas integrated from the
   effector), colored start→end, with the named/goal block circled — "where does it
   decide to go?"
Row 2:
   D. SAME frame, two instructions ("point to A" vs "point to B") → two planned paths:
      does the decision follow the language?
   E. Decision consistency: 15 stochastic samples of the plan for one scene (flow
      matching is stochastic) → how repeatable is the decision?
   F. Decision forming: the planned path at each flow-matching denoising step
      (noise → committed plan).

Output → docs/experiments/language_table/lt_decision.png
"""
from __future__ import annotations
import os, sys, textwrap
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_decision.png"


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def cam(env):
    vm, pm, *_ = env.calc_camera_params(env._image_size)
    return (np.array(vm, np.float64).reshape((4, 4), order="F"),
            np.array(pm, np.float64).reshape((4, 4), order="F"),
            env._image_size[1], env._image_size[0])


def project(view, proj, W, H, xy, z=0.02):
    P = np.array([xy[0], xy[1], z, 1.0]); c = proj @ (view @ P); n = c[:3] / c[3]
    return np.array([(n[0] * 0.5 + 0.5) * W, (1.0 - (n[1] * 0.5 + 0.5)) * H])


def blk(env, b):
    p, _ = env._pybullet_client.getBasePositionAndOrientation(env._block_to_pybullet_id[b]); return np.array(p[:2])


def draw_path(ax, view, proj, W, H, start_xy, deltas, cmap="cool", lw=2.6, alpha=1.0):
    pts = [start_xy.copy()]
    for d in deltas:
        pts.append(pts[-1] + d)
    px = np.array([project(view, proj, W, H, p) for p in pts])           # (H+1, 2)
    segs = np.stack([px[:-1], px[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, array=np.linspace(0, 1, len(segs)), linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    ax.scatter([px[0, 0]], [px[0, 1]], s=45, c="white", edgecolor="black", zorder=5)   # start (effector)
    ax.annotate("", xy=px[-1], xytext=px[-2], arrowprops=dict(arrowstyle="-|>", lw=2.2, color="red"))
    return px


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)
    a_mean = np.array(cfg.action_mean, np.float32); a_std = np.array(cfg.action_std, np.float32)
    print("[1/3] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    tm = VLATrainModel(cfg).to(device)
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm.load_state_dict(ckpt["state_dict"]); tm.eval()

    from envs.language_table_env import make_lt_env, get_instruction

    @torch.no_grad()
    def cond_tokens(image, instruction, state_norm):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        tok, im = vlm.encode_vlm(inp)
        st = torch.from_numpy(state_norm.astype(np.float32)).unsqueeze(0).to(device)
        cond, vtok = tm._build_cond(tok, st, im)
        return cond, vtok

    @torch.no_grad()
    def plan(image, instruction, state_norm, seed=0):
        cond, vtok = cond_tokens(image, instruction, state_norm)
        seed_all(seed)
        x = tm.decoder.sample(cond, cfg.num_flow_steps, vtok)
        return x[0].cpu().numpy() * a_std + a_mean

    @torch.no_grad()
    def plan_avg(image, instruction, state_norm, J=6):
        return np.mean([plan(image, instruction, state_norm, seed=s) for s in range(J)], axis=0)

    @torch.no_grad()
    def denoise_steps(image, instruction, state_norm, seed=0):
        cond, vtok = cond_tokens(image, instruction, state_norm)
        seed_all(seed)
        H, D = cfg.action_horizon, cfg.action_dim
        x = torch.randn(1, H, D, device=device)
        outs = []
        for i in range(cfg.num_flow_steps):
            t = torch.full((1,), i / cfg.num_flow_steps, device=device)
            x = x + tm.decoder.forward(x, t, cond, vtok) * (1.0 / cfg.num_flow_steps)
            outs.append(x[0].cpu().numpy() * a_std + a_mean)
        return outs

    env = make_lt_env("point", seed=6)
    view, proj, W, H = cam(env)

    def grab():
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs); rc = env._reward_calculator
        rgb = np.asarray(obs["rgb"])
        et = np.asarray(obs["effector_target_translation"], np.float32)
        st = (np.asarray(obs["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8)
        return dict(rgb=rgb, instr=instr, goal=rc._block, et=et, st=st, blocks=list(env._blocks_on_table))

    print("[2/3] Collecting scenes + plans …")
    scenes = [grab() for _ in range(3)]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))

    # Row 1 — planned path per scene (goal circled)
    for k, s in enumerate(scenes):
        ax = axes[0, k]; ax.imshow(s["rgb"]); ax.set_xticks([]); ax.set_yticks([])
        img = resize_frame(s["rgb"], cfg.lt_img_w, cfg.lt_img_h)
        deltas = plan_avg(img, s["instr"], s["st"])
        draw_path(ax, view, proj, W, H, s["et"], deltas)
        gp = project(view, proj, W, H, blk(env, s["goal"]))
        ax.add_patch(plt.Circle(gp, 12, fill=False, ec="lime", lw=2.4))
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        wrapped = "\n".join(textwrap.wrap(f'"{s["instr"]}"', width=34))
        ax.set_title(f"{wrapped}\nplanned path (white=now→red=end), goal circled", fontsize=8.5)

    # D — same frame, two instructions → two plans
    s0 = scenes[0]
    alt = next((b for b in s0["blocks"] if b != s0["goal"]), s0["goal"])
    ax = axes[1, 0]; ax.imshow(s0["rgb"]); ax.set_xticks([]); ax.set_yticks([])
    img0 = resize_frame(s0["rgb"], cfg.lt_img_w, cfg.lt_img_h)
    for b, cmap, ec in [(s0["goal"], "winter", "lime"), (alt, "autumn", "orange")]:
        d = plan_avg(img0, f"point to the {b.replace('_',' ')}", s0["st"])
        draw_path(ax, view, proj, W, H, s0["et"], d, cmap=cmap, lw=2.6)
        ax.add_patch(plt.Circle(project(view, proj, W, H, blk(env, b)), 12, fill=False, ec=ec, lw=2.4))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(f"D. SAME frame, swap instruction\nblue→{s0['goal'].replace('_',' ')}  vs  orange→{alt.replace('_',' ')}", fontsize=9)

    # E — decision consistency: stochastic samples
    ax = axes[1, 1]; ax.imshow(s0["rgb"]); ax.set_xticks([]); ax.set_yticks([])
    for sd in range(15):
        d = plan(img0, s0["instr"], s0["st"], seed=sd)
        draw_path(ax, view, proj, W, H, s0["et"], d, cmap="cool", lw=1.2, alpha=0.5)
    ax.add_patch(plt.Circle(project(view, proj, W, H, blk(env, s0["goal"])), 12, fill=False, ec="lime", lw=2.4))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title("E. Decision consistency\n15 stochastic samples of the plan", fontsize=9.5, fontweight="bold")

    # F — decision forming across denoising steps
    ax = axes[1, 2]; ax.imshow(s0["rgb"]); ax.set_xticks([]); ax.set_yticks([])
    steps = denoise_steps(img0, s0["instr"], s0["st"])
    cols = plt.cm.viridis(np.linspace(0.2, 0.95, len(steps)))
    for i, d in enumerate(steps):
        pts = [s0["et"].copy()]
        for dd in d:
            pts.append(pts[-1] + dd)
        px = np.array([project(view, proj, W, H, p) for p in pts])
        ax.plot(px[:, 0], px[:, 1], color=cols[i], lw=2.0, alpha=0.9, label=f"step {i+1}/{len(steps)}")
    ax.add_patch(plt.Circle(project(view, proj, W, H, blk(env, s0["goal"])), 12, fill=False, ec="lime", lw=2.4))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("F. Decision forming\nplanned path at each flow-matching step", fontsize=9.5, fontweight="bold")

    env.close()
    fig.suptitle(
        "Understanding the model's DECISION — its predicted action chunk integrated into a planned effector path\n"
        "Row 1: where it plans to go per command (goal circled).  D: language changes the plan.  E: how repeatable.  F: plan emerging from noise.",
        fontsize=12, fontweight="bold", y=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[3/3] saved → {OUT}")


if __name__ == "__main__":
    main()
