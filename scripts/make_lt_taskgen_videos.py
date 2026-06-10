"""
scripts/make_lt_taskgen_videos.py
─────────────────────────────────
Annotated rollout videos of the Plan A BALANCED model (exp02_balanced) across the
seen/unseen task eval — to watch the headline:
  • untrained EASY tasks succeed (point 86%, separate 70%)
  • PRECISE placement fails whether seen (block2block/relative) or unseen (between)

Saves to docs/experiments/language_table/videos/balanced/.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch
import imageio

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
VID = ROOT / "docs/experiments/language_table/videos/balanced"

#   (factory, require, tag, want_success, max_attempts)
SPECS = [
    ("point",         "point",    "unseen_point_success",       True,  10),
    ("separate",      "separate", "unseen_separate_success",    True,  14),
    ("block2block",   None,       "seen_block2block_attempt",   True,  30),
    ("block2block",   None,       "seen_block2block_fail",      False,  4),
    ("block2relative", None,      "seen_relative_attempt",      True,  25),
    ("between",       None,       "unseen_between_fail",        False,  4),
]


def main():
    VID.mkdir(parents=True, exist_ok=True)
    os.environ["LT_DATASET_ROOT"] = str((ROOT / "asset/data/language_table_sim_balanced").resolve())
    from configs.registry import get_config
    from models.vla import VLAModel
    from models.vla_train import VLATrainModel
    from data.language_table import resize_frame
    from envs.language_table_env import make_lt_env, get_instruction, annotate_frame, LanguageTableAgent

    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    print("[1/2] Loading balanced model …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ck = ROOT / "asset/runs/language_table/exp02_balanced/checkpoints/best.pt"
    ckpt = torch.load(ck, map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    agent = LanguageTableAgent(vlm, tm, cfg, device)

    def rollout(env, require):
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs)
        if require:
            t = 0
            while require.lower() not in instr.lower() and t < 80:
                out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
                instr = get_instruction(env, obs); t += 1
        agent.reset(); agent.instruction = instr
        frames, success, reward = [], False, 0.0
        for step in range(cfg.sim_max_steps):
            rgb = np.asarray(obs["rgb"])
            frames.append(annotate_frame(rgb, 0, 1, step + 1, cfg.sim_max_steps, reward, success, instr))
            a = np.clip(agent.act(resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h),
                                  np.asarray(obs["effector_translation"], np.float32)), -0.1, 0.1).astype(np.float32)
            sout = env.step(a)
            obs, reward, done, info = (sout if len(sout) == 4 else (sout[0], sout[1], sout[2] or sout[3], sout[4]))
            if reward > 0:
                success = True
            if done or success:
                frames.append(annotate_frame(np.asarray(obs["rgb"]), 0, 1, step + 2, cfg.sim_max_steps, reward, success, instr))
                break
        return instr, frames, success

    print("[2/2] Rolling out + saving …")
    for factory, require, tag, want, max_att in SPECS:
        env = make_lt_env(factory, seed=11)
        best = None
        for _ in range(max_att):
            instr, frames, success = rollout(env, require)
            if success == want:
                best = (instr, frames, success); break
            if best is None or len(frames) < len(best[1]):
                best = (instr, frames, success)
        try: env.close()
        except Exception: pass
        instr, frames, success = best
        p = VID / f"{tag}.mp4"
        imageio.mimsave(str(p), frames, fps=cfg.video_fps)
        print(f"   {p.name}  success={success}  {len(frames)}f  | {instr[:46]!r}")
    print(f"\nvideos → {VID}")


if __name__ == "__main__":
    main()
