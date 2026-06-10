"""
scripts/make_lt_taskgen_pairs.py
────────────────────────────────
For EACH of the 7 eval tasks, render one SUCCESS and one FAILURE rollout of the Plan A
balanced model (exp02_balanced) — 14 annotated videos.

Rare-success tasks (block2block/between ~2%) need many attempts for a success; high-SR
tasks (point/separate) need a few for a failure. Per-task attempt cap is sized to the
measured SR. If a slot can't be filled in the cap, the best-available rollout is saved
with a note.

Output → docs/experiments/language_table/videos/balanced_pairs/
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
VID = ROOT / "docs/experiments/language_table/videos/balanced_pairs"

#   (factory, require, max_attempts)  — cap sized to the measured SR
TASKS = [
    ("point",          "point",    20),
    ("separate",       "separate", 20),
    ("corner",         None,       25),
    ("block2relative", None,       40),
    ("block2absolute", None,       50),
    ("block2block",    None,       70),
    ("between",        None,       70),
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
    ckpt = torch.load(ROOT / "asset/runs/language_table/exp02_balanced/checkpoints/best.pt",
                      map_location=device, weights_only=False)
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
            frames.append(annotate_frame(np.asarray(obs["rgb"]), 0, 1, step + 1, cfg.sim_max_steps, reward, success, instr))
            a = np.clip(agent.act(resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h),
                                  np.asarray(obs["effector_translation"], np.float32)), -0.1, 0.1).astype(np.float32)
            sout = env.step(a)
            obs, reward, done, info = (sout if len(sout) == 4 else (sout[0], sout[1], sout[2] or sout[3], sout[4]))
            if reward > 0:
                success = True
            if done or success:
                frames.append(annotate_frame(np.asarray(obs["rgb"]), 0, 1, step + 2, cfg.sim_max_steps, reward, success, instr))
                break
        return instr, frames, success

    print("[2/2] Capturing success + failure per task …")
    for factory, require, max_att in TASKS:
        env = make_lt_env(factory, seed=7)
        succ = fail = None; shortest = None; n = 0
        while (succ is None or fail is None) and n < max_att:
            instr, frames, ok = rollout(env, require); n += 1
            if ok and succ is None:
                succ = (instr, frames)
            elif not ok and fail is None:
                fail = (instr, frames)
            if shortest is None or len(frames) < len(shortest[1]):
                shortest = (instr, frames, ok)
        try: env.close()
        except Exception: pass
        for slot, data in [("success", succ), ("fail", fail)]:
            if data is None:
                note = "_NOTFOUND" if slot == "success" else "_NONE"
                data = (shortest[0], shortest[1]); tag = f"{factory}_{slot}{note}"
            else:
                tag = f"{factory}_{slot}"
            imageio.mimsave(str(VID / f"{tag}.mp4"), data[1], fps=cfg.video_fps)
            print(f"   {tag}.mp4  ({len(data[1])}f, {n} attempts)  | {data[0][:42]!r}", flush=True)
    print(f"\n14 videos → {VID}")


if __name__ == "__main__":
    main()
