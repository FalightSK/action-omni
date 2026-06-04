"""
envs/aloha_env.py
─────────────────
ALOHA (bimanual transfer-cube) simulation wrapper for gym-aloha.

Provides:
  AlohaAgent     — receding-horizon VLA controller (ABSOLUTE 14-DOF joint targets)
  run_episode()  — runs one gym_aloha episode, returns a result dict
  annotate_frame() — HUD overlay for videos

Key difference from PushTAgent:
  PushT predicts relative pixel deltas and the agent integrates them from the
  current position. ALOHA predicts ABSOLUTE joint targets (same space as the
  dataset action / env.step), so the agent denormalises and commands them
  directly — no integration, no drift. This is why inference_horizon can safely
  exceed PushT's 4 (executing more of the chunk open-loop is ACT-standard).

Success: gym-aloha TransferCube gives reward in {0,1,2,3,4}; reward==4 (cube
transferred to the left gripper and lifted) is success, and the env terminates.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.aloha import resize_frame


class AlohaAgent:
    """Receding-horizon VLA agent for ALOHA. Re-plans every inference_horizon steps.

    HARD CONSTRAINT: state = joint positions (env observation) only.
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

    def reset(self) -> None:
        self._buffer = []
        self.replan_count = 0

    def _norm_state(self, s: np.ndarray) -> np.ndarray:
        return (s - self.state_mean) / (self.state_std + 1e-8)

    def _denorm_action(self, a: np.ndarray) -> np.ndarray:
        return a * self.action_std + self.action_mean

    @torch.no_grad()
    def _replan(self, image: Image.Image, state: np.ndarray) -> None:
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
        # Absolute joint targets — denormalise, command directly (no integration).
        self._buffer       = [self._denorm_action(a).astype(np.float32) for a in acts]
        self.replan_count += 1

    def act(self, image: Image.Image, state: np.ndarray) -> np.ndarray:
        if not self._buffer:
            self._replan(image, state)
        return self._buffer.pop(0)


def annotate_frame(frame, ep, total_eps, step, max_steps, reward, max_reward, success):
    """Draw HUD overlay onto a video frame (ASCII only for cp1252 consoles)."""
    img  = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    W, H = img.size
    overlay = Image.new("RGBA", (W, 92), (0, 0, 0, 160))
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)
    rcol = (50, 220, 50) if reward >= 4 else ((255, 200, 50) if reward >= 2 else (255, 255, 255))
    draw.text((10,  6), f"Episode  {ep+1}/{total_eps}",            fill=(255, 255, 255))
    draw.text((10, 26), f"Step     {step}/{max_steps}",            fill=(255, 255, 255))
    draw.text((10, 46), f"Reward   {reward}/4  (max {max_reward})", fill=rcol)
    draw.text((10, 66), f"Replans  {step}",                        fill=(180, 180, 180))
    if success:
        draw.text((W // 2 - 80, H // 2 - 20), "SUCCESS!", fill=(50, 220, 50))
    return np.array(img.convert("RGB"))


def run_episode(env, agent, cfg, ep_idx, total_eps, save_video, video_dir):
    """Run one gym_aloha episode. Returns a result dict.

    max_coverage is reported as max_reward/4 so the generic evaluate.py summary
    (which averages 'max_coverage') reads as task-progress fraction for ALOHA.
    """
    obs, _ = env.reset(seed=ep_idx * 137 + 42)
    agent.reset()
    frames, max_reward, success, reward = [], 0, False, 0
    t0 = time.time()
    step = 0

    for step in range(cfg.sim_max_steps):
        rgb    = np.asarray(obs["pixels"]["top"])
        image  = resize_frame(rgb, cfg.aloha_img_w, cfg.aloha_img_h)
        state  = np.asarray(obs["agent_pos"], dtype=np.float32)
        action = agent.act(image, state)

        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        reward     = int(reward)
        max_reward = max(max_reward, reward)
        success    = bool(info.get("is_success", reward >= 4))

        if save_video:
            frames.append(annotate_frame(env.render(), ep_idx, total_eps,
                                         step + 1, cfg.sim_max_steps,
                                         reward, max_reward, success))
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
    tag = "SUCCESS" if success else "failed "
    print(f"  Ep {ep_idx+1:2d}/{total_eps} | {tag} | max_reward={max_reward}/4 | "
          f"steps={step+1} | replans={agent.replan_count} | {elapsed:.0f}s")
    return {
        "episode": ep_idx + 1,
        "steps": step + 1,
        "max_reward": max_reward,
        "final_reward": reward,
        "max_coverage": round(max_reward / 4.0, 4),
        "is_success": success,
        "replan_count": agent.replan_count,
        "elapsed_sec": round(elapsed, 1),
    }
