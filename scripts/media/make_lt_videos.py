"""
scripts/make_lt_videos.py
─────────────────────────
Render annotated MP4 rollouts for the eval set (in-distribution) and the held-out
OOD commands, so the behaviour can be watched directly.

Saves, under docs/experiments/language_table/videos/:
  indist_block2block_success_1.mp4, indist_block2block_fail_1.mp4   (eval set)
  ood_separate_success_1.mp4, ood_separate_fail_1.mp4              (held-out verb)
  ood_point_success_1.mp4, ood_point_success_2.mp4                 (held-out verb)

Each frame carries the instruction + step + reward HUD (envs.language_table_env.annotate_frame).
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

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel

VID = ROOT / "docs/experiments/language_table/videos"

# (factory, require-verb, tag, n_success, n_fail, max_attempts)
SPECS = [
    ("block2block", None,       "indist_block2block", 1, 1, 25),
    ("separate",    "separate", "ood_separate",       1, 1, 14),
    ("point",       "point",    "ood_point",          2, 0, 10),
]


def rollout(env, agent, cfg, require, ep_idx):
    from envs.language_table_env import get_instruction, annotate_frame
    from data.language_table import resize_frame
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    instr = get_instruction(env, obs)
    if require:
        tries = 0
        while require.lower() not in instr.lower() and tries < 80:
            out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
            instr = get_instruction(env, obs); tries += 1
    agent.reset(); agent.instruction = instr
    frames, success, reward = [], False, 0.0
    for step in range(cfg.sim_max_steps):
        rgb = np.asarray(obs["rgb"])
        frames.append(annotate_frame(rgb, ep_idx, 1, step + 1, cfg.sim_max_steps, reward, success, instr))
        img = resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h)
        state = np.asarray(obs["effector_translation"], dtype=np.float32)
        a = np.clip(agent.act(img, state), -0.1, 0.1).astype(np.float32)
        sout = env.step(a)
        obs, reward, done, info = (sout if len(sout) == 4 else
                                   (sout[0], sout[1], sout[2] or sout[3], sout[4]))
        if reward > 0:
            success = True
        if done or success:
            frames.append(annotate_frame(np.asarray(obs["rgb"]), ep_idx, 1, step + 2,
                                         cfg.sim_max_steps, reward, success, instr))
            break
    return instr, frames, success


def main():
    VID.mkdir(parents=True, exist_ok=True)
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    print("[1/2] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt",
                      map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    from envs.language_table_env import LanguageTableAgent, make_lt_env
    agent = LanguageTableAgent(vlm, tm, cfg, device)

    print("[2/2] Rolling out + saving videos …")
    manifest = []
    for factory, require, tag, n_succ, n_fail, max_att in SPECS:
        env = make_lt_env(factory, seed=11)
        got_s, got_f, ep = 0, 0, 0
        while (got_s < n_succ or got_f < n_fail) and ep < max_att:
            instr, frames, success = rollout(env, agent, cfg, require, ep)
            ep += 1
            if success and got_s < n_succ:
                got_s += 1
                p = VID / f"{tag}_success_{got_s}.mp4"
            elif not success and got_f < n_fail:
                got_f += 1
                p = VID / f"{tag}_fail_{got_f}.mp4"
            else:
                continue
            imageio.mimsave(str(p), frames, fps=cfg.video_fps)
            manifest.append((p.name, success, len(frames), instr))
            print(f"   saved {p.name}  ({'SUCCESS' if success else 'fail'}, {len(frames)} frames) | {instr[:48]!r}")
        try:
            env.close()
        except Exception:
            pass

    print(f"\n  {len(manifest)} videos → {VID}")
    for name, succ, nf, instr in manifest:
        print(f"   {name:34s} {'OK ' if succ else 'X  '} {nf:3d}f  {instr[:44]!r}")


if __name__ == "__main__":
    main()
