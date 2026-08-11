"""
scripts/_lt_anti_random_collect.py
────────────────────────────────────
Paired-rollout data collection: random policy vs trained model on the SAME
episode initializations (same env seed → identical block layouts per episode).

Collected per episode:
  d0, d_min, toward, success, cat      — same as failure_modes.py
  ee_traj      [[x,y], ...]            — effector position every step
  sb_traj      [[x,y], ...]            — source block position every step
  actions      [[dx,dy], ...]          — raw (pre-clip) action at every step
  all_pos_0    [[x,y], ...]            — all 8 block positions at episode start
  sb_idx, tb_idx                       — which index in all_pos_0 is source/target
  first_contact_correct  bool|null     — was the first block the effector reached the correct one?
  first_contact_step     int           — step at which first contact occurred (-1 if never)

Output:
  asset/runs/language_table/exp01_baseline/anti_random_data.json
"""

from __future__ import annotations
import os, sys, json
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT))
from configs.registry import get_config
from models.vla import VLAModel
from models.vla_train import VLATrainModel
from data.language_table import resize_frame
from envs.language_table_env import make_lt_env, get_instruction, LanguageTableAgent

THRESH = 0.05
CONTACT_R = 0.08           # "effector is near a block" radius
OUT = ROOT / "asset/runs/language_table/exp01_baseline/anti_random_data.json"


class RandomAgent:
    """Pure random 2D ee-delta in the same range the model typically produces."""
    instruction = ""
    _buffer: list = []
    replan_count = 0

    def reset(self):
        self._buffer = []
        self.replan_count = 0

    def act(self, image, state) -> np.ndarray:
        return np.random.uniform(-0.05, 0.05, 2).astype(np.float32)


def _blk(env, b):
    p, _ = env._pybullet_client.getBasePositionAndOrientation(
        env._block_to_pybullet_id[b])
    return np.array(p[:2], dtype=np.float64)


def _reward_calc(env):
    for attr in ("_reward_calculator", "_reward", "reward_calculator"):
        rc = getattr(env, attr, None)
        if rc is not None:
            return rc
    raise RuntimeError("reward calculator not found")


def lag1_autocorr(seq):
    a = np.array(seq)
    if len(a) < 3:
        return 0.0
    c = np.corrcoef(a[:-1], a[1:])[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def rollout(env, agent, cfg, ep_idx, label):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    instr = get_instruction(env, obs)
    agent.reset()
    if hasattr(agent, "instruction"):
        agent.instruction = instr

    rc = _reward_calc(env)
    sb = getattr(rc, "_start_block", None)
    tb = getattr(rc, "_target_block", None)
    if sb is None or tb is None:
        return None

    all_blocks = list(env._block_to_pybullet_id.keys())
    try:
        sb_idx = all_blocks.index(sb)
        tb_idx = all_blocks.index(tb)
    except ValueError:
        return None

    all_pos_0 = [_blk(env, b).tolist() for b in all_blocks]
    sb0 = np.array(all_pos_0[sb_idx])
    tb0 = np.array(all_pos_0[tb_idx])
    d0 = float(np.linalg.norm(sb0 - tb0))
    if d0 < 1e-6:
        return None
    goal_dir = (tb0 - sb0) / d0

    ee_traj, sb_traj, actions_taken = [], [], []
    dists = [d0]
    success = False
    first_contact_correct = None
    first_contact_step = -1

    for step in range(cfg.sim_max_steps):
        rgb = np.asarray(obs["rgb"])
        image = resize_frame(rgb, cfg.lt_img_w, cfg.lt_img_h)
        state = np.asarray(obs["effector_translation"], dtype=np.float32)
        ee_xy = state[:2].tolist()
        ee_traj.append(ee_xy)

        cur_sb = _blk(env, sb).tolist()
        sb_traj.append(cur_sb)

        action = agent.act(image, state)
        actions_taken.append(action[:2].tolist())
        action_clipped = np.clip(action, -0.1, 0.1).astype(np.float32)

        sout = env.step(action_clipped)
        if len(sout) == 5:
            obs, reward, term, trunc, info = sout
            done = bool(term or trunc)
        else:
            obs, reward, done, info = sout

        cur_sb_arr = np.array(_blk(env, sb))
        dists.append(float(np.linalg.norm(cur_sb_arr - _blk(env, tb))))
        if float(reward) > 0:
            success = True

        # First-contact check: which block does the effector first come within CONTACT_R of?
        if first_contact_correct is None and step > 0:
            ee = np.array(ee_xy)
            dists_to_blocks = [np.linalg.norm(ee - np.array(p)) for p in all_pos_0]
            nearest_idx = int(np.argmin(dists_to_blocks))
            if dists_to_blocks[nearest_idx] < CONTACT_R:
                first_contact_correct = bool(nearest_idx == sb_idx)
                first_contact_step = step

        if done or success:
            break

    sbf = np.array(_blk(env, sb))
    moved = float(np.linalg.norm(sbf - sb0))
    toward = float((sbf - sb0) @ goal_dir)
    d_min = float(np.min(dists))
    d_final = float(dists[-1])
    prog = (d0 - d_min) / (d0 + 1e-9)

    if success:              cat = "success"
    elif d_min < 2 * THRESH: cat = "near_miss"
    elif prog > 0.33:        cat = "partial"
    else:                    cat = "wrong/idle"

    ac_x = lag1_autocorr([a[0] for a in actions_taken])
    ac_y = lag1_autocorr([a[1] for a in actions_taken])

    tag = "OK " if success else "x  "
    print(f"  [{label}] ep {ep_idx+1:2d} {tag} "
          f"d0={d0:.3f} d_min={d_min:.3f} toward={toward:+.3f} "
          f"fcc={first_contact_correct} ac={ac_x:.2f}/{ac_y:.2f} "
          f"[{cat}] {instr[:40]!r}")

    return dict(
        ep=ep_idx, instr=instr, label=label,
        d0=d0, d_min=d_min, d_final=d_final,
        moved=moved, toward=toward, prog=prog,
        success=success, cat=cat,
        ee_traj=ee_traj, sb_traj=sb_traj,
        actions=actions_taken,
        all_pos_0=all_pos_0, sb_idx=sb_idx, tb_idx=tb_idx,
        first_contact_correct=first_contact_correct,
        first_contact_step=first_contact_step,
        ac_x=ac_x, ac_y=ac_y,
        n_steps=len(actions_taken),
    )


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cfg = get_config("language_table", "exp01")
    device = cfg.get_device()

    print("[1/4] Loading VLM + checkpoint …")
    vlm = VLAModel(cfg); vlm.vlm.to(device).eval()
    ck = torch.load(
        Path(cfg.output_dir) / "checkpoints" / "best.pt",
        map_location=device, weights_only=False)
    tm = VLATrainModel(cfg).to(device)
    tm.load_state_dict(ck["state_dict"]); tm.eval()
    print(f"   epoch {ck.get('epoch','?')}  val_loss={ck.get('val_loss', float('nan')):.4f}")

    model_agent = LanguageTableAgent(vlm, tm, cfg, device)

    # ── Phase 1: random baseline (same seed = same episode configs) ──────────
    print(f"\n[2/4] Random policy — {N} episodes (seed=42) …")
    env_r = make_lt_env("block2block", seed=42)
    rand_agent = RandomAgent()
    random_rows = []
    for i in range(N):
        r = rollout(env_r, rand_agent, cfg, i, "RAND")
        if r is not None:
            random_rows.append(r)
    try: env_r.close()
    except Exception: pass

    # ── Phase 2: model rollouts (same seed) ─────────────────────────────────
    print(f"\n[3/4] Model policy — {N} episodes (seed=42) …")
    env_m = make_lt_env("block2block", seed=42)
    model_rows = []
    for i in range(N):
        r = rollout(env_m, model_agent, cfg, i, "MODEL")
        if r is not None:
            model_rows.append(r)
    try: env_m.close()
    except Exception: pass

    # ── Summary ──────────────────────────────────────────────────────────────
    def summarize(rows, name):
        sr = np.mean([r["success"] for r in rows])
        toward_pos = np.mean([r["toward"] > 0 for r in rows])
        fcc = [r["first_contact_correct"] for r in rows if r["first_contact_correct"] is not None]
        fcc_rate = np.mean(fcc) if fcc else float("nan")
        ac_x = np.mean([r["ac_x"] for r in rows])
        ac_y = np.mean([r["ac_y"] for r in rows])
        print(f"\n  {name}: n={len(rows)}  SR={sr*100:.0f}%  "
              f"toward_pos={toward_pos*100:.0f}%  "
              f"first_contact_correct={fcc_rate*100:.0f}%  "
              f"lag1_autocorr=({ac_x:.2f}, {ac_y:.2f})")
    summarize(random_rows, "RANDOM")
    summarize(model_rows, "MODEL")

    # ── Save ─────────────────────────────────────────────────────────────────
    data = {"random": random_rows, "model": model_rows,
            "N": N, "seed": 42, "contact_radius": CONTACT_R}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print(f"\n[4/4] saved -> {OUT}")


if __name__ == "__main__":
    main()
