"""
envs/libero_env.py
──────────────────
Closed-loop LIBERO evaluation for the frozen-VLM VLA. Mirrors LanguageTableAgent
(per-episode language conditioning, receding horizon) with LIBERO sim specifics:

- OffScreenRenderEnv @ 256x256, benchmark init states, 5-step settle.
- The live agentview render is rotated 180 degrees (dataset/OpenVLA convention) so eval
  frames match the training distribution (data/libero/dataset.py trains on the stored,
  already-rotated frames).
- State = [eef_pos(3), quat->axis-angle(3), gripper_qpos(2)] = 8-dim.
- 7-dim delta-EEF action commanded directly (use_relative_actions=False), clipped [-1,1].
- Success = the env done flag; max_coverage mirrors success for evaluate.py's summary.

Requires the hf-libero package (HF's LIBERO fork) — the raw LIBERO repo renders wrong
textures and scores ~0%.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from data.libero.dataset import resize_frame


def get_libero_task(suite_name: str, task_index: int):
    """(task, bddl_path, init_states, n_tasks) for one task of a LIBERO suite."""
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_index)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    init_path = f"{get_libero_path('init_states')}/{task.problem_folder}/{task.init_states_file}"
    init_states = np.asarray(torch.load(init_path, weights_only=False))
    return task, bddl, init_states, suite.n_tasks


def make_libero_env(bddl_file: str, image_size: int = 256):
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=image_size, camera_widths=image_size)


def _live_frame(obs) -> np.ndarray:
    """agentview render rotated 180deg to match the stored dataset frames."""
    return obs["agentview_image"][::-1, ::-1].copy()


def _obs_state(obs) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    return np.concatenate([
        obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
    ]).astype(np.float32)


class LiberoAgent:
    """Receding-horizon VLA agent; conditions the frozen VLM on the episode instruction."""

    def __init__(self, vlm, train_model, cfg, device):
        self.vlm = vlm
        self.train_model = train_model
        self.cfg = cfg
        self.device = device
        self.action_mean = np.array(cfg.action_mean, dtype=np.float32)
        self.action_std = np.array(cfg.action_std, dtype=np.float32)
        self.state_mean = np.array(cfg.state_mean, dtype=np.float32)
        self.state_std = np.array(cfg.state_std, dtype=np.float32)
        self.instruction = cfg.task_text     # set per-episode by run_episode
        self._buffer: list[np.ndarray] = []
        self.replan_count = 0

    def reset(self):
        self._buffer = []
        self.replan_count = 0

    def _norm_state(self, s):
        return (s - self.state_mean) / (self.state_std + 1e-8)

    def _denorm_action(self, a):
        return a * self.action_std + self.action_mean

    @torch.no_grad()
    def _replan(self, image: Image.Image, state: np.ndarray):
        state_t = torch.from_numpy(self._norm_state(state).astype(np.float32)).unsqueeze(0).to(self.device)
        inputs = self.vlm.build_vlm_inputs([image], [self.instruction], self.device)
        tokens, img_mask = self.vlm.encode_vlm(inputs)
        acts_flat = self.train_model.sample(tokens, state_t, num_steps=self.cfg.num_flow_steps, img_mask=img_mask)
        acts = acts_flat.view(self.cfg.action_horizon, self.cfg.action_dim).cpu().float().numpy()
        acts = acts[: self.cfg.inference_horizon]
        self._buffer = [self._denorm_action(a).astype(np.float32) for a in acts]
        self.replan_count += 1

    def act(self, image: Image.Image, state: np.ndarray) -> np.ndarray:
        if not self._buffer:
            self._replan(image, state)
        return self._buffer.pop(0)


def run_episode(env, agent, cfg, ep_idx, total_eps, init_state, instruction,
                save_video=False, video_dir=None):
    """One LIBERO rollout from a benchmark init state. Returns evaluate.py's result dict."""
    t0 = time.time()
    agent.instruction = instruction
    agent.reset()

    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):                                  # settle physics (LIBERO metric convention)
        obs, _, _, _ = env.step(np.zeros(cfg.action_dim, dtype=np.float32))

    frames = []
    done, steps, success = False, 0, False
    while not done and steps < cfg.sim_max_steps:
        image = resize_frame(_live_frame(obs), cfg.image_size, cfg.image_size)
        action = np.clip(agent.act(image, _obs_state(obs)), -1.0, 1.0).astype(np.float32)
        obs, _, done, _ = env.step(action)
        steps += 1
        success = bool(done)
        if save_video and ep_idx == 0:
            frames.append(_live_frame(obs))

    result = {
        "episode": ep_idx,
        "instruction": instruction,
        "steps": steps,
        "is_success": success,
        "max_coverage": 1.0 if success else 0.0,     # summary averages this key
        "final_coverage": 1.0 if success else 0.0,
        "replan_count": agent.replan_count,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if save_video and frames and video_dir is not None:
        try:
            import os

            import imageio
            os.makedirs(video_dir, exist_ok=True)
            imageio.mimsave(f"{video_dir}/episode_{ep_idx:02d}.mp4", frames, fps=cfg.video_fps)
        except Exception as e:  # video is best-effort
            print(f"[libero] video save failed: {e}")
    return result
