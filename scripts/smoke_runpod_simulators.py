"""Reset, step, and render the real ALOHA and LIBERO evaluation simulators."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def aloha_smoke() -> None:
    import gymnasium as gym
    import gym_aloha  # noqa: F401

    os.environ.setdefault("MUJOCO_GL", "glfw")
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos",
                   max_episode_steps=3, disable_env_checker=True)
    try:
        obs, info = env.reset(seed=123)
        assert isinstance(obs, dict), f"unexpected ALOHA observation {type(obs)}"
        assert "pixels" in obs and isinstance(obs["pixels"], dict), \
            f"ALOHA returned no camera images: {list(obs)}"
        image_key = "top" if "top" in obs["pixels"] else next(iter(obs["pixels"]))
        frame = np.asarray(obs["pixels"][image_key])
        assert frame.ndim == 3 and (frame.shape[-1] == 3 or frame.shape[0] == 3), \
            f"bad ALOHA frame: {frame.shape}"
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        frame2 = np.asarray(obs2["pixels"][image_key])
        assert frame2.shape == frame.shape
        print(f"ALOHA OK: reset/step/render, obs_image={image_key} {frame.shape}, "
              f"reward={reward}, terminated={terminated}, truncated={truncated}", flush=True)
    finally:
        env.close()


def libero_smoke() -> None:
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    assert bddl.is_file(), f"missing LIBERO task definition: {bddl}"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128,
                             camera_widths=128)
    try:
        env.seed(123)
        obs = env.reset()
        key = "agentview_image"
        assert key in obs, f"missing LIBERO render; observation keys: {list(obs)}"
        frame = np.asarray(obs[key])
        assert frame.ndim == 3 and frame.shape[-1] == 3, frame.shape
        action = np.zeros(7, dtype=np.float32)
        obs2, reward, done, info = env.step(action)
        frame2 = np.asarray(obs2[key])
        assert frame2.shape == frame.shape
        print(f"LIBERO OK: reset/step/egl-render, task0={task.language!r}, "
              f"image={frame.shape}, reward={reward}, done={done}", flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=("aloha", "libero"))
    args = parser.parse_args()
    if args.environment == "aloha":
        aloha_smoke()
    else:
        libero_smoke()


if __name__ == "__main__":
    main()
