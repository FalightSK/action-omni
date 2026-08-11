"""
scripts/measure_lt_language_effect.py
─────────────────────────────────────
How much does the INSTRUCTION causally change the model's decision?

(1) ACTION DIVERGENCE (open-loop, noise-controlled): for a fixed scene+state and a
    fixed flow-matching noise seed, swap or null the instruction and measure how much
    the predicted action chunk moves — compared to the sampling-noise floor (same
    inputs, different noise).  Ratio swap/noise >> 1 means language drives the action.

(2) DIRECTION FOLLOWING (open-loop, causal): with "point to block X", measure the
    cosine alignment of the predicted effector delta with the direction to X, vs the
    direction to OTHER blocks.  High named-alignment = the action follows the command.

(3) CLOSED-LOOP: point-task rollouts with the CORRECT instruction vs a WRONG one
    (naming a different block).  SR drop = how much behavior depends on the language.

Output → docs/experiments/language_table/lt_language_effect.png
"""
from __future__ import annotations
import os, sys, random
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame

OUT = ROOT / "docs/experiments/language_table/lt_language_effect.png"


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()
    s_mean = np.array(cfg.state_mean, np.float32); s_std = np.array(cfg.state_std, np.float32)
    a_mean = np.array(cfg.action_mean, np.float32); a_std = np.array(cfg.action_std, np.float32)
    print("[1/4] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ckpt = torch.load(Path(cfg.output_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device); tm.load_state_dict(ckpt["state_dict"]); tm.eval()

    from envs.language_table_env import make_lt_env, get_instruction, LanguageTableAgent

    def action_chunk(image, instruction, state_norm, seed):
        inp = vlm.build_vlm_inputs([image], [instruction], device)
        tok, im = vlm.encode_vlm(inp)
        st = torch.from_numpy(state_norm.astype(np.float32)).unsqueeze(0).to(device)
        seed_all(seed)
        flat = tm.sample(tok, st, num_steps=cfg.num_flow_steps, img_mask=im)
        return flat.view(cfg.action_horizon, cfg.action_dim).float().cpu().numpy()

    def blk(env, b):
        p, _ = env._pybullet_client.getBasePositionAndOrientation(env._block_to_pybullet_id[b]); return np.array(p[:2])

    # ── (1) action divergence + (2) direction following, on the point env ──────────
    print("[2/4] Open-loop action divergence + direction following …")
    env = make_lt_env("point", seed=7)
    rng = random.Random(0)
    INSTR_POOL = ["push the red moon to the blue cube", "separate the green star from the red moon",
                  "move the yellow pentagon to the top left", "slide the blue cube to the green star",
                  "point to the red pentagon", "move your arm to the green cube"]
    noise_d, swap_d, null_d, rms_a = [], [], [], []
    align_named, align_other = [], []
    M = 16
    for _ in range(M):
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        true_instr = get_instruction(env, obs)
        image = resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h)
        st = (np.asarray(obs["effector_translation"], np.float32) - s_mean) / (s_std + 1e-8)
        et = np.asarray(obs["effector_target_translation"], np.float32)

        a_true = action_chunk(image, true_instr, st, seed=1234)
        a_true2 = action_chunk(image, true_instr, st, seed=9999)          # different noise → floor
        a_swap = action_chunk(image, rng.choice(INSTR_POOL), st, seed=1234)
        a_null = action_chunk(image, "", st, seed=1234)
        noise_d.append(np.sqrt(np.mean((a_true - a_true2) ** 2)))
        swap_d.append(np.sqrt(np.mean((a_true - a_swap) ** 2)))
        null_d.append(np.sqrt(np.mean((a_true - a_null) ** 2)))
        rms_a.append(np.sqrt(np.mean(a_true ** 2)))

        # direction following: for each block, "point to <block>", denorm 1st delta, cosine to dir
        blocks = list(env._blocks_on_table)
        for b in blocks:
            instr_b = f"point to the {b.replace('_', ' ')}"
            d = action_chunk(image, instr_b, st, seed=1234)[0] * a_std + a_mean   # first delta (denorm)
            if np.linalg.norm(d) < 1e-6:
                continue
            d = d / np.linalg.norm(d)
            dir_named = blk(env, b) - et
            if np.linalg.norm(dir_named) > 1e-6:
                align_named.append(float(d @ (dir_named / np.linalg.norm(dir_named))))
            for ob in blocks:
                if ob == b:
                    continue
                dn = blk(env, ob) - et
                if np.linalg.norm(dn) > 1e-6:
                    align_other.append(float(d @ (dn / np.linalg.norm(dn))))
    env.close()

    # ── (3) closed-loop: correct vs wrong instruction on the point task ────────────
    print("[3/4] Closed-loop: correct vs wrong instruction (point task) …")
    env = make_lt_env("point", seed=21)
    agent = LanguageTableAgent(vlm, tm, cfg, device)
    N_CL = 25

    def point_roll(use_correct):
        out = env.reset(); obs = out[0] if isinstance(out, tuple) else out
        instr = get_instruction(env, obs); rc = env._reward_calculator
        if use_correct:
            agent.instruction = instr
        else:
            others = [b for b in env._blocks_on_table if b != rc._block]
            agent.instruction = f"point to the {rng.choice(others).replace('_', ' ')}"
        agent.reset()
        for _ in range(cfg.sim_max_steps):
            img = resize_frame(np.asarray(obs["rgb"]), cfg.lt_img_w, cfg.lt_img_h)
            a = np.clip(agent.act(img, np.asarray(obs["effector_translation"], np.float32)), -0.1, 0.1).astype(np.float32)
            sout = env.step(a); obs, reward, done, info = (sout if len(sout) == 4 else (sout[0], sout[1], sout[2] or sout[3], sout[4]))
            if reward > 0:
                return True
            if done:
                break
        return False

    sr_correct = np.mean([point_roll(True) for _ in range(N_CL)])
    sr_wrong = np.mean([point_roll(False) for _ in range(N_CL)])
    env.close()

    # ── numbers ────────────────────────────────────────────────────────────────────
    noise = float(np.mean(noise_d)); swap = float(np.mean(swap_d)); null = float(np.mean(null_d))
    an = float(np.mean(align_named)); ao = float(np.mean(align_other))
    print(f"   noise floor={noise:.3f}  swap={swap:.3f} ({swap/noise:.1f}x)  null={null:.3f} ({null/noise:.1f}x)")
    print(f"   direction cosine: named={an:.2f}  other={ao:.2f}")
    print(f"   closed-loop point SR: correct={sr_correct*100:.0f}%  wrong={sr_wrong*100:.0f}%")

    # ── figure ──────────────────────────────────────────────────────────────────────
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 6.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, wspace=0.26)
    axA.bar([0, 1, 2], [noise, swap, null], color=["#BBBBBB", "#4C72B0", "#DD8452"], edgecolor="black")
    axA.set_ylim(0, max(noise, swap, null) * 1.25)
    for i, v, lab in [(0, noise, "noise floor"), (1, swap, f"{swap/noise:.1f}x floor"), (2, null, f"{null/noise:.1f}x floor")]:
        axA.annotate(f"{v:.3f}\n{lab if i else ''}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9.5, fontweight="bold")
    axA.set_xticks([0, 1, 2]); axA.set_xticklabels(["same instr\n(noise floor)", "swapped\ninstruction", "null\ninstruction"], fontsize=9)
    axA.set_ylabel("RMS action change (norm. units)")
    axA.set_title("A. Does changing the instruction move the action?\n(fixed scene + fixed noise seed)", fontsize=10.5, fontweight="bold")
    axA.grid(axis="y", alpha=0.3)

    axB.bar([0, 1], [an, ao], color=["#55A868", "#C44E52"], edgecolor="black")
    for i, v in [(0, an), (1, ao)]:
        axB.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11, fontweight="bold")
    axB.axhline(0, color="black", lw=0.8)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["toward\nNAMED block", "toward\nother blocks"], fontsize=9.5, fontweight="bold")
    axB.set_ylabel("cosine(action, direction-to-block)"); axB.set_ylim(min(-0.1, ao - 0.1), 1.0)
    axB.set_title("B. 'point to X' — does the action point at X?\n(first-step effector delta direction)", fontsize=10.5, fontweight="bold")
    axB.grid(axis="y", alpha=0.3)

    axC.bar([0, 1], [sr_correct * 100, sr_wrong * 100], color=["#55A868", "#C44E52"], edgecolor="black")
    for i, v in [(0, sr_correct * 100), (1, sr_wrong * 100)]:
        axC.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=12, fontweight="bold")
    axC.set_xticks([0, 1]); axC.set_xticklabels(["CORRECT\ninstruction", "WRONG\ninstruction"], fontsize=9.5, fontweight="bold")
    axC.set_ylabel("point success rate (%)"); axC.set_ylim(0, 100)
    axC.set_title(f"C. Closed-loop point task (n={N_CL})\ngiving a wrong instruction breaks it", fontsize=10.5, fontweight="bold")
    axC.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "How much does the instruction drive the model's decision?\n"
        f"Swap moves the action {swap/noise:.1f}x the noise floor  ·  the action aims at the NAMED block "
        f"(cos {an:.2f} vs {ao:.2f})  ·  a WRONG instruction drops point SR {sr_correct*100:.0f}%→{sr_wrong*100:.0f}%\n"
        "Language has a strong, causal effect on WHAT the model does; the weakness is long-range pushing, not language use.",
        fontsize=12, fontweight="bold", y=0.99)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[4/4] saved -> {OUT}")


if __name__ == "__main__":
    main()
