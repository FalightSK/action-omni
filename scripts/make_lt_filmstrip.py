"""
scripts/make_lt_filmstrip.py
────────────────────────────
Visual proof that the VLA executes the command — a frame-by-frame filmstrip of one
SUCCESSFUL rollout per command type (mirrors the ALOHA OOD filmstrip).

Rows:
  block2block  (in-distribution)
  separate     (HELD-OUT verb)
  point        (HELD-OUT verb)

For each, rolls out episodes (reset-filtered to the right verb) until one succeeds,
captures evenly-spaced frames, and lays them in a strip with the instruction + step.

Output → docs/experiments/language_table/lt_rollout_filmstrip.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel

OUT = ROOT / "docs/experiments/language_table/lt_rollout_filmstrip.png"
N_FRAMES = 7
# (reward factory, require-verb, row label, max attempts to find a success)
# NB: this run trained on ALL command types (no OOD holdout), so every row below is
# in-distribution. The rows differ in MANIPULATION TYPE, not train/test split.
ROWS = [
    ("block2block", None,       "block2block",  30),
    ("separate",    "separate", "separate",     20),
    ("point",       "point",    "point",        15),
]


def rollout_until_success(env, agent, cfg, require, max_attempts):
    from envs.language_table_env import get_instruction
    best = None
    for _ in range(max_attempts):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs)
        if require and require.lower() not in instr.lower():
            # quick re-roll for the right verb without a full rollout
            tries = 0
            while require.lower() not in instr.lower() and tries < 80:
                out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
                instr = get_instruction(env, obs); tries += 1
        agent.reset(); agent.instruction = instr
        frames, success, reward = [], False, 0.0
        for step in range(cfg.sim_max_steps):
            rgb = np.asarray(obs["rgb"]).copy()
            frames.append(rgb)
            from data.language_table import resize_frame
            img = resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h)
            state = np.asarray(obs["effector_translation"], dtype=np.float32)
            a = np.clip(agent.act(img, state), -0.1, 0.1).astype(np.float32)
            sout = env.step(a)
            obs, reward, done, info = (sout if len(sout) == 4 else
                                       (sout[0], sout[1], sout[2] or sout[3], sout[4]))
            if reward > 0:
                success = True
            if done or success:
                frames.append(np.asarray(obs["rgb"]).copy())
                break
        cand = {"instr": instr, "frames": frames, "success": success, "steps": len(frames)}
        if success:
            return cand
        if best is None or len(frames) < best["steps"]:   # prefer a shorter (less flailing) failure
            best = cand
    return best


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    print("[1/2] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt",
                      map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ckpt["state_dict"]); tm.eval()

    from envs.language_table_env import LanguageTableAgent, make_lt_env
    agent = LanguageTableAgent(vlm, tm, cfg, device)

    print("[2/2] Rolling out one success per command …")
    strips = []
    for factory, require, label, max_att in ROWS:
        env = make_lt_env(factory, seed=7)
        r = rollout_until_success(env, agent, cfg, require, max_att)
        try:
            env.close()
        except Exception:
            pass
        idx = np.linspace(0, len(r["frames"]) - 1, N_FRAMES).round().astype(int)
        strips.append({"label": label, "instr": r["instr"], "success": r["success"],
                       "frames": [r["frames"][i] for i in idx], "fidx": idx,
                       "total": len(r["frames"])})
        print(f"   {factory}: success={r['success']}  steps={r['steps']}  | {r['instr'][:50]!r}")

    nrow = len(strips)
    fig, axes = plt.subplots(nrow, N_FRAMES, figsize=(2.0 * N_FRAMES, 2.25 * nrow + 1.0))
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.07, right=0.99, hspace=0.42, wspace=0.06)
    for ri, s in enumerate(strips):
        for ci in range(N_FRAMES):
            ax = axes[ri, ci]
            ax.imshow(s["frames"][ci]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"step {s['fidx'][ci]}", fontsize=8)
            if ci == N_FRAMES - 1 and s["success"]:
                ax.text(0.5, 0.08, "SUCCESS", transform=ax.transAxes, ha="center",
                        fontsize=11, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round", facecolor="#2ca02c", edgecolor="black"))
        # short rotated row label on the left (command type + outcome)
        tag = "solved ✓" if s["success"] else "best try"
        axes[ri, 0].set_ylabel(f"{s['label']}\n({tag})", fontsize=11, fontweight="bold")

    # instruction caption centred BELOW each row, positioned from the row's axes box
    fig.canvas.draw()
    for ri, s in enumerate(strips):
        pos0 = axes[ri, 0].get_position()
        pos_last = axes[ri, -1].get_position()
        x_mid = (pos0.x0 + pos_last.x1) / 2
        y_cap = pos0.y0 - 0.018
        fig.text(x_mid, y_cap, f"“{s['instr']}”", ha="center", va="top",
                 fontsize=11, style="italic", color="#222")

    fig.suptitle("Qwen-DiT VLA on Language Table — one successful rollout per command type (frame-by-frame)\n"
                 "All three command types were in the training mix (no held-out split); the rows differ in manipulation type, not train/test.",
                 fontsize=12.5, fontweight="bold", y=0.96)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
