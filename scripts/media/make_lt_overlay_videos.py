"""
scripts/make_lt_overlay_videos.py
─────────────────────────────────
Re-render Language Table rollouts with the GOAL made visible, to show that a
"success" is really the task being completed.

On every frame we draw:
  • the goal region (green circle = success threshold) at the task target,
  • the tracked object (the block being pushed, or the effector for `point`),
  • a line between them + the live distance-to-goal and the threshold,
  • SUCCESS when distance < threshold.

The goal/target world positions are camera-projected onto the image using the env's
own view/projection matrices, so the drawn circle sits exactly where the goal is.

Output → docs/experiments/language_table/videos/overlay_*.mp4
"""
from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel

VID = ROOT / "docs/experiments/language_table/videos"
SEP_THR = 0.025
BLK_THR = 0.05

# (factory, require, tag, want_success, max_attempts)
SPECS = [
    ("block2block", None,       "overlay_block2block_success", True,  30),
    ("block2block", None,       "overlay_block2block_fail",    False, 6),
    ("separate",    "separate", "overlay_separate_success",    True,  16),
    ("point",       "point",    "overlay_point_success",       True,  12),
]


def cam_matrices(env):
    viewm, projm, *_ = env.calc_camera_params(env._image_size)
    view = np.array(viewm, dtype=np.float64).reshape((4, 4), order="F")
    proj = np.array(projm, dtype=np.float64).reshape((4, 4), order="F")
    H, W = env._image_size
    return view, proj, W, H


def project(view, proj, W, H, xy, z=0.02):
    P = np.array([xy[0], xy[1], z, 1.0])
    clip = proj @ (view @ P)
    if abs(clip[3]) < 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] * 0.5 + 0.5) * W, (1.0 - (ndc[1] * 0.5 + 0.5)) * H)


def block_xy(env, b):
    p, _ = env._pybullet_client.getBasePositionAndOrientation(env._block_to_pybullet_id[b])
    return np.array(p[:2])


def geom(env, factory, obs):
    """Return (goal_xy, goal_label, tracked_xy, threshold)."""
    rc = env._reward_calculator
    if factory == "block2block":
        return block_xy(env, rc._target_block), rc._target_block, block_xy(env, rc._start_block), BLK_THR
    if factory == "separate":
        return np.array(rc._target_translation), "target", block_xy(env, rc._block), SEP_THR
    if factory == "point":
        return block_xy(env, rc._block), rc._block, np.array(obs["effector_target_translation"]), BLK_THR
    raise ValueError(factory)


def draw(rgb, view, proj, W, H, goal_xy, tracked_xy, dist, thr, instr, success, scale=2):
    img = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB").resize((W * scale, H * scale))
    d = ImageDraw.Draw(img)
    bar = Image.new("RGBA", (img.size[0], 44), (0, 0, 0, 175)); img.paste(bar, (0, 0), bar)
    d = ImageDraw.Draw(img)
    gp = project(view, proj, W, H, goal_xy)
    tp = project(view, proj, W, H, tracked_xy)
    rp = project(view, proj, W, H, goal_xy + np.array([thr, 0.0]))
    if gp and rp:
        gx, gy = gp[0] * scale, gp[1] * scale
        rad = abs(rp[0] - gp[0]) * scale
        col = (60, 230, 60) if dist < thr else (60, 200, 255)
        d.ellipse([gx - rad, gy - rad, gx + rad, gy + rad], outline=col, width=3)
        d.line([gx - 5, gy, gx + 5, gy], fill=col, width=1); d.line([gx, gy - 5, gx, gy + 5], fill=col, width=1)
    if tp:
        tx, ty = tp[0] * scale, tp[1] * scale
        d.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], fill=(255, 220, 0), outline=(0, 0, 0))
        if gp:
            d.line([tx, ty, gx, gy], fill=(255, 255, 255), width=1)
    dcol = (60, 230, 60) if dist < thr else (255, 255, 255)
    d.text((6, 4), (instr or "")[:54], fill=(220, 220, 120))
    d.text((6, 24), f"dist to goal: {dist:.3f} m   (success < {thr:.3f} m)", fill=dcol)
    if success:
        d.text((img.size[0] // 2 - 45, img.size[1] - 28), "SUCCESS", fill=(60, 230, 60))
    return np.array(img)


def rollout(env, agent, cfg, factory, require, want_success, max_att):
    from envs.language_table_env import get_instruction
    from data.language_table import resize_frame
    view, proj, W, H = cam_matrices(env)
    best = None
    for _ in range(max_att):
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs)
        if require:
            t = 0
            while require.lower() not in instr.lower() and t < 80:
                out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
                instr = get_instruction(env, obs); t += 1
        agent.reset(); agent.instruction = instr
        frames, success = [], False
        for step in range(cfg.sim_max_steps):
            g_xy, _, t_xy, thr = geom(env, factory, obs)
            dist = float(np.linalg.norm(np.array(t_xy) - np.array(g_xy)))
            frames.append(draw(np.asarray(obs["rgb"]), view, proj, W, H, g_xy, t_xy, dist, thr, instr, success))
            a = np.clip(agent.act(resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h),
                                  np.asarray(obs["effector_translation"], np.float32)), -0.1, 0.1).astype(np.float32)
            sout = env.step(a)
            obs, reward, done, info = (sout if len(sout) == 4 else (sout[0], sout[1], sout[2] or sout[3], sout[4]))
            if reward > 0:
                success = True
            if done or success:
                g_xy, _, t_xy, thr = geom(env, factory, obs)
                dist = float(np.linalg.norm(np.array(t_xy) - np.array(g_xy)))
                frames.append(draw(np.asarray(obs["rgb"]), view, proj, W, H, g_xy, t_xy, dist, thr, instr, success))
                break
        cand = {"instr": instr, "frames": frames, "success": success}
        if success == want_success:
            return cand
        best = cand
    return best


def main():
    VID.mkdir(parents=True, exist_ok=True)
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    print("[1/2] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    from envs.language_table_env import LanguageTableAgent, make_lt_env
    agent = LanguageTableAgent(vlm, tm, cfg, device)

    print("[2/2] Rolling out with goal overlay …")
    for factory, require, tag, want, max_att in SPECS:
        env = make_lt_env(factory, seed=5)
        r = rollout(env, agent, cfg, factory, require, want, max_att)
        try:
            env.close()
        except Exception:
            pass
        p = VID / f"{tag}.mp4"
        imageio.mimsave(str(p), r["frames"], fps=cfg.video_fps)
        Image.fromarray(r["frames"][-1]).save(VID / f"{tag}_last.png")   # for visual verification
        print(f"   saved {p.name}  (success={r['success']}, {len(r['frames'])}f) | {r['instr'][:46]!r}")


if __name__ == "__main__":
    main()
