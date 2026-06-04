"""
envs/pusht_env.py
──────────────────
PushT simulation environment wrapper.

Provides:
  PushTAgent     — receding-horizon VLA controller
  run_episode()  — runs one simulation episode and returns a result dict
  annotate_frame() — draws HUD overlay onto a video frame

Extracted from inference.py so that scripts/evaluate.py can import just
the env logic without pulling in the full training machinery.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class PushTAgent:
    """
    Receding-horizon VLA agent for PushT.
    Re-plans every `inference_horizon` steps using VLM + adapter + flow decoder.
    Converts normalised relative-delta predictions to absolute env actions.

    HARD CONSTRAINTS:
    - state = agent position only (never block/goal position)
    - inference_horizon at inference must equal inference_horizon at training
    """

    def __init__(self, vlm, train_model, cfg, device) -> None:
        self.vlm         = vlm
        self.train_model = train_model
        self.cfg         = cfg
        self.device      = device

        self.action_mean = np.array(cfg.action_mean, dtype=np.float32)
        self.action_std  = np.array(cfg.action_std,  dtype=np.float32)
        self.state_mean  = np.array(cfg.state_mean,  dtype=np.float32)
        self.state_std   = np.array(cfg.state_std,   dtype=np.float32)

        self._buffer: list[np.ndarray] = []
        self.replan_count = 0
        self._prev_deltas_raw: list[np.ndarray] = [
            np.zeros(cfg.action_dim, dtype=np.float32),
            np.zeros(cfg.action_dim, dtype=np.float32),
        ]

    def reset(self) -> None:
        self._buffer = []
        self.replan_count = 0
        self._prev_deltas_raw = [
            np.zeros(self.cfg.action_dim, dtype=np.float32),
            np.zeros(self.cfg.action_dim, dtype=np.float32),
        ]

    def _norm_state(self, s: np.ndarray) -> np.ndarray:
        if self.cfg.state_dim > 2:
            d1, d2 = self._prev_deltas_raw
            raw = np.concatenate([s, d1, d2])
        else:
            raw = s
        return (raw - self.state_mean) / (self.state_std + 1e-8)

    def _denorm_delta(self, d: np.ndarray) -> np.ndarray:
        return d * self.action_std + self.action_mean

    import torch

    @torch.no_grad()
    def _replan(self, image: Image.Image, state: np.ndarray) -> None:
        import torch
        state_t = torch.from_numpy(
            self._norm_state(state).astype(np.float32)
        ).unsqueeze(0).to(self.device)

        inputs           = self.vlm.build_vlm_inputs([image], [self.cfg.task_text], self.device)
        tokens, img_mask = self.vlm.encode_vlm(inputs)

        acts_flat = self.train_model.sample(
            tokens, state_t, num_steps=self.cfg.num_flow_steps, img_mask=img_mask
        )
        acts = acts_flat.view(
            self.cfg.action_horizon, self.cfg.action_dim
        ).cpu().float().numpy()

        acts = acts[: self.cfg.inference_horizon]

        abs_actions = []
        base = state.copy()
        for i in range(len(acts)):
            delta_px = self._denorm_delta(acts[i])
            abs_pos  = np.clip(base + delta_px, 0.0, 511.0).astype(np.float64)
            abs_actions.append(abs_pos)
            base = abs_pos

        self._buffer       = abs_actions
        self.replan_count += 1

    def act(self, image: Image.Image, state: np.ndarray) -> np.ndarray:
        if not self._buffer:
            self._replan(image, state)
        action = self._buffer.pop(0)
        raw_delta = (action - state).astype(np.float32)
        self._prev_deltas_raw = [raw_delta, self._prev_deltas_raw[0]]
        return action


def annotate_frame(frame, ep, total_eps, step, max_steps, coverage, success):
    """Draw HUD overlay onto a video frame."""
    img  = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    W, H = img.size
    overlay = Image.new("RGBA", (W, 72), (0, 0, 0, 160))
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)
    cov_c = (50,220,50) if coverage >= 0.9 else ((255,200,50) if coverage >= 0.5 else (255,255,255))
    draw.text((10,  6), f"Episode  {ep+1}/{total_eps}",   fill=(255,255,255))
    draw.text((10, 26), f"Step     {step}/{max_steps}",   fill=(255,255,255))
    draw.text((10, 46), f"Coverage {coverage*100:5.1f}%", fill=cov_c)
    if success:
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            draw.text((W//2-80+dx, H//2-20+dy), "SUCCESS!", fill=(0,0,0))
        draw.text((W//2-80, H//2-20), "SUCCESS!", fill=(50,220,50))
    return np.array(img.convert("RGB"))


def run_episode(env, agent, cfg, ep_idx, total_eps, save_video, video_dir):
    """
    Run one simulation episode.

    Returns dict with keys:
        episode, steps, max_coverage, final_coverage, is_success,
        replan_count, elapsed_sec
    """
    obs, _ = env.reset(seed=ep_idx * 137 + 42)
    agent.reset()
    frames, max_cov, success, coverage = [], 0.0, False, 0.0
    t0 = time.time()

    for step in range(cfg.sim_max_steps):
        image  = Image.fromarray(obs["pixels"])
        state  = obs["agent_pos"].astype(np.float32)
        action = agent.act(image, state)

        obs, reward, terminated, truncated, info = env.step(action)
        coverage = float(info.get("coverage", reward))
        success  = bool(info.get("is_success", coverage >= cfg.success_threshold))
        max_cov  = max(max_cov, coverage)

        if save_video:
            frames.append(annotate_frame(env.render(), ep_idx, total_eps,
                                         step+1, cfg.sim_max_steps, coverage, success))
        if terminated or truncated or success:
            break

    if save_video and frames:
        try:
            import imageio
            imageio.mimsave(str(video_dir / f"episode_{ep_idx+1:02d}.mp4"),
                            frames, fps=cfg.video_fps)
        except Exception as e:
            print(f"    [warn] video save: {e}")

    elapsed = time.time() - t0
    tag = "✅ SUCCESS" if success else "❌ failed "
    print(f"  Ep {ep_idx+1:2d}/{total_eps} | {tag} | cov={max_cov:.1%} | "
          f"steps={step+1} | replans={agent.replan_count} | {elapsed:.0f}s")
    return {
        "episode": ep_idx + 1,
        "steps": step + 1,
        "max_coverage": round(max_cov, 4),
        "final_coverage": round(coverage, 4),
        "is_success": success,
        "replan_count": agent.replan_count,
        "elapsed_sec": round(elapsed, 1),
    }
