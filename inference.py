"""
inference.py  –  Simulation inference with receding-horizon control.

Receding-horizon:
  Every inference_horizon steps (default 4):
    1. Capture (image, state) from env
    2. Qwen3.5 VLM encode  →  embed (1024)
    3. VLMFeatureAdapter   →  adapted (512)
    4. FlowMatchingDecoder →  predicted relative deltas (horizon=16)
    5. De-normalise + add current state  →  absolute actions
    6. Buffer first inference_horizon absolute actions

Usage
-----
  python inference.py                                  # 20 episodes, best.pt
  python inference.py --episodes 10 --no-video
  python inference.py --checkpoint path/to/epoch.pt
"""

from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config_loader import get_config
from models.vla import VLAModel
from models.flow_matching import FlowMatchingDecoder
from train import VLATrainModel


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class PushTAgent:
    """
    Receding-horizon VLA agent.
    Re-plans every `inference_horizon` steps using VLM + adapter + flow decoder.
    Converts normalised relative-delta predictions to absolute env actions.

    Experiment B: tracks the last 2 executed deltas and includes them in the
    6D state vector so the model can detect stalls and adapt direction.
    """

    def __init__(
        self,
        vlm:          VLAModel,
        train_model:  VLATrainModel,
        cfg:          VLAConfig,
        device:       torch.device,
    ) -> None:
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
        # Raw (pixel-space) previous executed deltas — zeros at episode start
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
        """
        Build and normalise the full state vector.

        state_dim == 2: [pos_x, pos_y]
        state_dim == 6: [pos_x, pos_y, dΔx₁, dΔy₁, dΔx₂, dΔy₂]

        The delta history is normalised with the same stats as actions
        (action_mean / action_std), which are stored in state_mean/std dims 2-5.
        """
        if self.cfg.state_dim > 2:
            d1, d2 = self._prev_deltas_raw
            raw = np.concatenate([s, d1, d2])
        else:
            raw = s
        return (raw - self.state_mean) / (self.state_std + 1e-8)

    def _denorm_delta(self, d: np.ndarray) -> np.ndarray:
        return d * self.action_std + self.action_mean

    @torch.no_grad()
    def _replan(self, image: Image.Image, state: np.ndarray) -> None:
        state_t = torch.from_numpy(
            self._norm_state(state).astype(np.float32)
        ).unsqueeze(0).to(self.device)

        # VLM encode (bottleneck, ~1.7 s on M1)
        inputs           = self.vlm.build_vlm_inputs([image], [self.cfg.task_text], self.device)
        tokens, img_mask = self.vlm.encode_vlm(inputs)   # multi-scale or single

        acts_flat = self.train_model.sample(
            tokens, state_t, num_steps=self.cfg.num_flow_steps, img_mask=img_mask
        )   # (1, horizon*2)
        acts = acts_flat.view(
            self.cfg.action_horizon, self.cfg.action_dim
        ).cpu().float().numpy()   # (16, 2)  normalised relative deltas

        acts = acts[: self.cfg.inference_horizon]   # (4, 2)

        # De-normalise and convert to absolute positions
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
        # Record executed delta in pixel space for next replan's state history
        raw_delta = (action - state).astype(np.float32)
        self._prev_deltas_raw = [raw_delta, self._prev_deltas_raw[0]]
        return action


# ──────────────────────────────────────────────────────────────────────────────
# Video annotation
# ──────────────────────────────────────────────────────────────────────────────

def _annotate(frame, ep, total_eps, step, max_steps, coverage, success):
    img  = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    W, H = img.size
    overlay = Image.new("RGBA", (W, 72), (0, 0, 0, 160))
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)
    cov_c = (50,220,50) if coverage >= 0.9 else ((255,200,50) if coverage >= 0.5 else (255,255,255))
    draw.text((10,  6), f"Episode  {ep+1}/{total_eps}",       fill=(255,255,255))
    draw.text((10, 26), f"Step     {step}/{max_steps}",       fill=(255,255,255))
    draw.text((10, 46), f"Coverage {coverage*100:5.1f}%",     fill=cov_c)
    if success:
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            draw.text((W//2-80+dx, H//2-20+dy), "SUCCESS!", fill=(0,0,0))
        draw.text((W//2-80, H//2-20), "SUCCESS!", fill=(50,220,50))
    return np.array(img.convert("RGB"))


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(env, agent, cfg, ep_idx, total_eps, save_video, video_dir):
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
            frames.append(_annotate(env.render(), ep_idx, total_eps,
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
    return {"episode": ep_idx+1, "steps": step+1,
            "max_coverage": round(max_cov,4), "final_coverage": round(coverage,4),
            "is_success": success, "replan_count": agent.replan_count,
            "elapsed_sec": round(elapsed,1)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",        type=int, default=2, choices=[1, 2],
                        help="Experiment config: 1=Exp1 (baseline), 2=Exp2 B+C (default)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes",   type=int, default=None)
    parser.add_argument("--no-video",   action="store_true")
    parser.add_argument("--max-steps",  type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.exp)
    print(f"Experiment : {args.exp}  |  output: {cfg.output_dir}")
    device = cfg.get_device()
    if args.max_steps: cfg.sim_max_steps = args.max_steps
    n_ep = args.episodes or cfg.sim_episodes
    print(f"Device : {device}\n")

    video_dir = Path(cfg.output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # ── Load VLM (frozen) ──────────────────────────────────────────────────
    print("[1/4] Loading Qwen3.5-0.8B VLM …")
    vlm_model = VLAModel(cfg)
    vlm_model.vlm.to(device).eval()

    # ── Load checkpoint (adapter + decoder) ───────────────────────────────
    ckpt_path = Path(args.checkpoint or (Path(cfg.output_dir)/"checkpoints"/"best.pt"))
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n[2/4] Loading adapter+decoder from {ckpt_path.name} …")
    train_model = VLATrainModel(cfg).to(device)
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_model.load_state_dict(ckpt["state_dict"])
    train_model.eval()
    print(f"   Epoch {ckpt.get('epoch','?')}  |  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # ── Agent ──────────────────────────────────────────────────────────────
    agent = PushTAgent(vlm_model, train_model, cfg, device)
    print(f"\n   Receding-horizon: predict {cfg.action_horizon} steps, "
          f"execute {cfg.inference_horizon} before re-plan")
    print(f"   Actions: {'relative (delta)' if cfg.use_relative_actions else 'absolute'}")

    # ── Gym ────────────────────────────────────────────────────────────────
    print("\n[3/4] Starting gym_pusht …")
    import gymnasium as gym
    import gym_pusht  # noqa

    env = gym.make("gym_pusht/PushT-v0",
                   obs_type="pixels_agent_pos", render_mode="rgb_array")

    # ── Run episodes ───────────────────────────────────────────────────────
    print(f"\n[4/4] Running {n_ep} episodes …\n")
    results = []
    for ep in range(n_ep):
        results.append(run_episode(env, agent, cfg, ep, n_ep,
                                   not args.no_video, video_dir))
    env.close()

    # ── Summary ────────────────────────────────────────────────────────────
    succs  = [r["is_success"]   for r in results]
    covs   = [r["max_coverage"] for r in results]
    steps  = [r["steps"]        for r in results]
    sr     = float(np.mean(succs))
    print(f"\n{'═'*52}")
    print(f"  Results  ({n_ep} episodes)")
    print(f"{'═'*52}")
    print(f"  Success rate      : {sr*100:.1f}%  ({sum(succs)}/{n_ep})")
    print(f"  Mean max coverage : {np.mean(covs)*100:.1f}%")
    print(f"  Mean steps        : {np.mean(steps):.1f}")
    print(f"  Horizon train/exec: {cfg.action_horizon}/{cfg.inference_horizon}")
    print(f"  Actions           : {'relative' if cfg.use_relative_actions else 'absolute'}")
    if not args.no_video:
        print(f"  Videos → {video_dir}/")
    print(f"{'═'*52}\n")

    summary = {"checkpoint": str(ckpt_path), "n_episodes": n_ep,
               "success_rate": round(sr,4),
               "mean_max_coverage": round(float(np.mean(covs)),4),
               "mean_steps": round(float(np.mean(steps)),1),
               "action_horizon": cfg.action_horizon,
               "inference_horizon": cfg.inference_horizon,
               "use_relative_actions": cfg.use_relative_actions,
               "num_flow_steps": cfg.num_flow_steps,
               "success_threshold": cfg.success_threshold,
               "episodes": results}
    out = Path(cfg.output_dir) / "sim_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  Results saved → {out}")


if __name__ == "__main__":
    main()
