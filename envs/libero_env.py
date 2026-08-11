"""Closed-loop LIBERO-Goal rollouts, with the instruction as an experimental variable.

Run with the vla_libero interpreter — it is the only env holding both the
simulator (robosuite 1.4 / mujoco 2.3 / numpy 1.26) and transformers.

Why the instruction is a variable
─────────────────────────────────
Chapter 1 established, offline, that Pi-0.5's image tokens carry goal identity
before the goal is visible (eta^2 = 0.873 at frame 0, flat across the episode)
while six of eight backbones cannot see the instruction at all. That is
AVAILABILITY. Whether a policy USES it is a different question, and this project
has already been wrong about exactly that once: an ALOHA ablation found image
tokens carrying the control signal while offline loss overstated the text
pathway.

So every rollout is run under several instruction conditions, and the headline
number is not success rate — it is how success rate MOVES when the instruction
is corrupted:

    canonical  the string the policy trained on
    swapped    another task's instruction, same scene, same register
    empty      no text at all

If success does not move, the policy is solving LIBERO-Goal from vision and
proprioception, the language axis is untestable in this setup, and no amount of
further backbone comparison will say anything about language.

Distribution matching — five things that must agree with training or a working
policy will look broken
───────────────────────────────────────────────────────────────────────────────
1. image flip      the demos were recorded with macros_image_convention="opengl"
                   (bottom-up) and flipped once at load; the live env returns
                   frames the same way, so it must be flipped identically.
2. resize          224x224 bilinear, through the same helper the dataset uses.
3. sequence length padded to a FIXED 272 tokens, not to the batch longest. A
                   paraphrase tokenises longer than a canonical instruction, so
                   per-batch padding would silently change the sequence the head
                   sees between training and eval.
4. chunking        predict action_horizon (16), execute inference_horizon (8),
                   then re-plan. Executing a different number than training used
                   degrades success for reasons unrelated to the hypothesis.
5. state           the 9-D robot_states vector, matching state_dim.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must precede any robosuite import: robosuite validates MUJOCO_GL at module
# scope and raises on values it does not recognise. This machine carries a
# global MUJOCO_GL=egl from an unrelated Linux/Isaac setup; on Windows the
# working backend is WGL.
if sys.platform == "win32" and os.environ.get("MUJOCO_GL", "").lower() != "wgl":
    os.environ["MUJOCO_GL"] = "wgl"

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Loaded by path, not as `data.libero.frames`, because importing anything under
# the `data` package runs data/__init__.py, which eagerly imports the PushT
# dataset and therefore `av` — a codec this environment does not have and this
# path does not need. Loading the file directly gets the SAME function the
# training dataset uses without the package chain behind it.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_libero_frames", ROOT / "data" / "libero" / "frames.py")
_frames = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_frames)
resize_frame = _frames.resize_frame

CAMERA = "agentview_image"          # live env key; the HDF5 calls it agentview_rgb
# Wrist view. Verified against a live OffScreenRenderEnv reset rather than
# assumed: the simulator returns robot0_eye_in_hand_image, while the same camera
# is stored as eye_in_hand_rgb in the demo HDF5. Guessing this wrong would feed
# the policy the exterior view twice and look like a modelling failure.
CAMERA_WRIST = "robot0_eye_in_hand_image"


def build_state(obs: dict) -> np.ndarray:
    """The 9-D state vector, reconstructed from the live observation.

    The demos store this as a single `robot_states` array, which the live env
    does not expose — it has to be rebuilt from parts, and getting the
    composition wrong would feed the policy a scrambled state while every shape
    check still passed.

    Verified against the demo arrays rather than assumed. HDF5 frame 0 reads
        [ 0.0362 -0.0362 | -0.1996 -0.0025  1.1886 |  0.9997 -0.0087 -0.0231 -0.0035]
    which decomposes exactly as
        gripper_states(2) | ee_pos(3) | eef_quat(4)
    and the live env returns the same three quantities, with the quaternion in
    the same order (leading component ~1, not trailing).

    Note this is NOT what data/libero/dataset.py originally documented ("7 joint
    positions + 2 gripper"): the trailing four values are a unit quaternion
    (norm 1.00002), and `joint_states` holds entirely different numbers.
    """
    return np.concatenate([
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
    ]).astype(np.float32)


# ── instruction conditions ───────────────────────────────────────────────────

def build_conditions(languages: list[str], task_idx: int, rng) -> dict[str, str]:
    """The instruction fed for one task, per condition.

    `swapped` uses a fixed offset rather than a random draw so the mapping is
    reproducible and every task receives exactly one wrong instruction — a
    random draw would let a task occasionally keep its own.
    """
    n = len(languages)
    return {
        "canonical": languages[task_idx],
        "swapped": languages[(task_idx + n // 2) % n],
        "empty": "",
    }


class LiberoAgent:
    """Receding-horizon VLA agent for LIBERO-Goal.

    The instruction is passed per-episode rather than read from the config, so
    the same trained policy can be evaluated under several text conditions
    without reloading anything.
    """

    def __init__(self, vlm, train_model, cfg, device) -> None:
        self.vlm = vlm
        self.train_model = train_model
        self.cfg = cfg
        self.device = device
        self.instruction = ""
        self._buffer: list[np.ndarray] = []
        self.replan_count = 0

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self._buffer = []
        self.replan_count = 0

    @torch.no_grad()
    def _replan(self, image, state: np.ndarray) -> None:
        # `image` is a single PIL image, or a list of views when cfg.n_views > 1.
        # build_vlm_inputs takes one entry PER FRAME, so a multi-view frame is
        # passed as a nested list — [[exterior, wrist]] — not as two frames.
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        inputs = self.vlm.build_vlm_inputs([image], [self.instruction], self.device)
        tokens, img_mask = self.vlm.encode_vlm(inputs)
        acts_flat = self.train_model.sample(
            tokens, state_t, num_steps=self.cfg.num_flow_steps, img_mask=img_mask
        )
        acts = acts_flat.view(self.cfg.action_horizon, self.cfg.action_dim)
        acts = acts.cpu().float().numpy()
        # LIBERO actions are bounded OSC deltas already — no de-normalisation.
        # They ARE the env's step() space and pass straight through.
        self._buffer = list(acts[: self.cfg.inference_horizon])
        self.replan_count += 1

    def act(self, obs: dict) -> np.ndarray:
        if not self._buffer:
            img = np.asarray(obs[CAMERA])[::-1]            # opengl -> upright
            pil = resize_frame(img, self.cfg.lt_img_w, self.cfg.lt_img_h)
            if getattr(self.cfg, "n_views", 1) > 1:
                # Same order as LiberoDataset builds it — exterior then wrist.
                # Reversing them here would train and evaluate on different view
                # assignments, which the positional encoding now distinguishes.
                wrist = np.asarray(obs[CAMERA_WRIST])[::-1]
                pil = [pil, resize_frame(wrist, self.cfg.lt_img_w, self.cfg.lt_img_h)]
            self._replan(pil, build_state(obs))
        return np.clip(self._buffer.pop(0), -1.0, 1.0)


def load_init_states(task) -> np.ndarray:
    """LIBERO's 50 fixed initial states for one task.

    The benchmark's own `get_task_init_states` raises under torch >= 2.6: it
    calls torch.load without weights_only=False, and the default flipped to
    True. The file is a plain pickled array, so it is loaded directly here.

    Using these rather than a random `env.reset()` is not a nicety — it is the
    evaluation protocol. A random reset samples object placements from the
    task's full initialisation distribution, which can put the scene in
    configurations no demonstration ever showed; the policy is then judged
    out-of-distribution, and the resulting zero success says nothing about the
    hypothesis under test.
    """
    import torch as _t
    from libero.libero import get_libero_path
    p = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return np.asarray(_t.load(str(p), weights_only=False))


def make_env(bddl_path: str, cfg):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=cfg.lt_img_h,
        camera_widths=cfg.lt_img_w,
    )


def run_episode(env, agent, cfg, instruction: str, seed: int,
                record: bool = False, init_state: np.ndarray | None = None) -> dict:
    """One episode. Returns the outcome plus enough detail to diagnose failures.

    Success is the environment's own `done`, which LIBERO sets from the bddl
    goal predicate — not a heuristic on the final frame.
    """
    agent.reset(instruction)
    env.seed(seed)
    obs = env.reset()

    # Set one of LIBERO's 50 fixed initial states. Without this, reset() samples
    # object placements from the task's full initialisation distribution, which
    # can produce arrangements no demonstration ever showed — the policy is then
    # judged out of distribution and scores zero for reasons unrelated to the
    # hypothesis. The demos themselves start from these states, so this is what
    # makes eval and training comparable.
    if init_state is not None:
        obs = env.set_init_state(init_state)
        # Physics needs a few steps to settle after a state is forced in;
        # querying the policy before then feeds it a frame mid-jitter. The
        # gripper is held open (-1) during settling, matching how the demos
        # begin.
        settle = np.zeros(cfg.action_dim, dtype=np.float32)
        settle[-1] = -1.0
        for _ in range(5):
            obs, _r, _d, _i = env.step(settle)

    frames: list[np.ndarray] = []
    success, steps = False, 0
    gripper_changes, max_disp = 0, 0.0
    last_grip = None
    start_state = build_state(obs)

    for steps in range(1, cfg.sim_max_steps + 1):
        action = agent.act(obs)
        obs, _reward, done, _info = env.step(action)

        # cheap failure-mode signal: did the gripper ever actuate, and did the
        # arm move at all? "did nothing" and "did the wrong thing" are different
        # failures and a binary success flag cannot tell them apart.
        grip = float(action[-1] > 0)
        if last_grip is not None and grip != last_grip:
            gripper_changes += 1
        last_grip = grip
        cur = build_state(obs)
        max_disp = max(max_disp, float(np.abs(cur - start_state).max()))

        if record and steps % 2 == 0:
            frames.append(np.asarray(obs[CAMERA])[::-1].copy())
        if done:
            success = True
            break

    return {
        "success": success,
        "steps": steps,
        "replans": agent.replan_count,
        "gripper_changes": gripper_changes,
        "max_state_displacement": max_disp,
        "frames": frames,
    }
