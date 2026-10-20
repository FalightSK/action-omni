"""Closed-loop LIBERO-Goal evaluation, sweeping instruction conditions.

    C:/Users/SK/miniconda3/envs/vla_libero/python.exe scripts/eval_libero.py \
        --exp exp01 --episodes 10 --conditions canonical swapped empty

This is step 1 of CHAPTER2_PLAN.md, and it is a GATE rather than a result.

The question is not "what is the success rate" — it is whether success MOVES
when the instruction is corrupted. If a swapped or empty instruction leaves
success unchanged, the policy is solving LIBERO-Goal from vision and
proprioception, the language axis is untestable with this stack, and the two
backbone arms planned after this step would measure nothing about language.
That is worth one rollout sweep to find out rather than ~14 GPU-hours.

Everything about the observation path is taken from the training code rather
than reimplemented: the image flip and resize come from data/libero/dataset.py,
the token budget and chunk sizes from the config. A mismatch in any of them
makes a working policy look broken, which is indistinguishable from the
hypothesis being false.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

if sys.platform == "win32" and os.environ.get("MUJOCO_GL", "").lower() != "wgl":
    os.environ["MUJOCO_GL"] = "wgl"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_policy(cfg, ckpt_path: Path, device: str):
    """Rebuild the VLM + trained head from a checkpoint."""
    from models.vla import VLAModel
    from models.vla_train import VLATrainModel

    print(f"[1/4] Loading VLM backbone ({cfg.vlm_backbone}) ...", flush=True)
    vlm = VLAModel(cfg).to(device).eval()

    print(f"[2/4] Loading head from {ckpt_path.name} ...", flush=True)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = VLATrainModel(cfg).to(device).eval()
    state = ck.get("model", ck.get("state_dict", ck))
    missing, unexpected = head.load_state_dict(state, strict=False)
    hard = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    loss = ck.get("val_loss")
    loss_label = f"{loss:.4f}" if loss is not None else "unavailable (see history.jsonl)"
    print(f"      epoch={ck.get('epoch')} val_loss={loss_label} "
          f"| missing={len(hard)} unexpected={len(unexpected)}")
    if hard[:3]:
        print(f"      e.g. missing: {hard[:3]}")
    if hard or unexpected:
        raise ValueError("Checkpoint has unexplained missing/unexpected head tensors")
    return vlm, head


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp01")
    ap.add_argument("--episodes", type=int, default=10, help="per task per condition")
    ap.add_argument("--conditions", nargs="+",
                    default=["canonical", "swapped", "empty"])
    ap.add_argument("--tasks", nargs="+", type=int, default=None,
                    help="task indices; default all 10")
    ap.add_argument("--checkpoint", default="best.pt")
    # Without this every run writes out_dir/eval_libero.json, so a checkpoint
    # ladder would overwrite the headline result once per snapshot.
    ap.add_argument("--output", default=None,
                    help="Override results path (default: <output_dir>/eval_libero.json)")
    ap.add_argument("--videos", type=int, default=2,
                    help="record this many episodes per (task, condition)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path,
                    help="Seed-specific run directory (or supply an absolute --checkpoint)")
    a = ap.parse_args()
    from scripts.evaluate import deterministic_inference
    deterministic_inference()

    # Resolve through the registry, not a hard-coded import: --exp was being
    # parsed and then ignored, so `--exp exp02` silently rolled out exp01's
    # checkpoint AND exp01's backbone, i.e. it would have reported Pi-0.5's
    # numbers as PaliGemma's.
    from configs.registry import get_config
    cfg = get_config("libero", a.exp)
    if a.output_dir:
        cfg.output_dir = str(a.output_dir)
    out_dir = ROOT / cfg.output_dir
    ckpt = out_dir / "checkpoints" / a.checkpoint
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt}")
        return 1
    from experiment_protocol import (checkpoint_config, file_digest, sampling_seed,
                                     seed_everything, write_json)
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = checkpoint_config(cfg, checkpoint, "libero", verify_live_encoder=True)
    protocol = checkpoint.get("protocol")
    training_seed = checkpoint.get("training_seed")
    dst = Path(a.output) if a.output else ckpt.parent.parent / "eval_libero.json"
    if a.episodes < 1:
        raise ValueError("Episode count must be positive")
    if protocol and dst.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {dst}")

    from envs.libero_env import (LiberoAgent, build_conditions, load_init_states,
                                 make_env, run_episode)
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    languages = [suite.get_task(i).language for i in range(suite.n_tasks)]
    tasks = a.tasks if a.tasks is not None else list(range(suite.n_tasks))

    print(f"\nLIBERO-Goal closed-loop evaluation")
    print(f"  checkpoint : {ckpt}")
    print(f"  tasks      : {len(tasks)}  episodes/task/condition: {a.episodes}")
    print(f"  conditions : {', '.join(a.conditions)}")
    print(f"  chunking   : predict {cfg.action_horizon}, execute "
          f"{cfg.inference_horizon}, then re-plan")
    print(f"  max steps  : {cfg.sim_max_steps}\n")

    vlm, head = load_policy(cfg, ckpt, a.device)
    agent = LiberoAgent(vlm, head, cfg, a.device)

    print("[3/4] Instruction conditions per task (task 0 shown):")
    demo = build_conditions(languages, 0, None)
    for k, v in demo.items():
        print(f"      {k:<10} {v!r}")

    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t0 = time.time()

    print(f"\n[4/4] Rolling out ...\n")
    for ti in tasks:
        task = suite.get_task(ti)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        conds = build_conditions(languages, ti, None)
        # LIBERO's 50 fixed evaluation initial states for this task. Episode i
        # uses state i, so every condition is compared on identical starts —
        # otherwise a swapped-instruction difference could be scene variation.
        inits = load_init_states(task)
        if a.episodes > len(inits):
            raise ValueError("Requested more unique starts than available; refusing duplicate initial states")
        env = make_env(bddl, cfg)
        try:
            for cond in a.conditions:
                instr = conds[cond]
                wins = 0
                for ep in range(a.episodes):
                    record = ep < a.videos
                    policy_seed = None
                    if protocol:
                        policy_seed = sampling_seed("libero", training_seed, ti, ep)
                        seed_everything(policy_seed)
                    r = run_episode(env, agent, cfg, instr,
                                    seed=a.seed + ep, record=record,
                                    init_state=inits[ep % len(inits)])
                    wins += int(r["success"])
                    frames = r.pop("frames")
                    r.update(task=ti, task_language=task.language, condition=cond,
                             instruction=instr, episode=ep, initial_state_id=ep,
                             environment_seed=a.seed + ep, policy_sampling_seed=policy_seed,
                             termination_reason="success" if r["success"] else "horizon")
                    results.append(r)
                    if record and frames:
                        try:
                            import imageio.v2 as imageio
                            imageio.mimwrite(
                                video_dir / f"t{ti:02d}_{cond}_ep{ep}.mp4",
                                frames, fps=cfg.video_fps)
                        except Exception as e:            # noqa: BLE001
                            print(f"      (video failed: {e})")
                sr = wins / a.episodes
                print(f"  task {ti:2d} {cond:<10} SR {sr:5.1%}  ({wins}/{a.episodes})"
                      f"   {task.language[:40]}", flush=True)
        finally:
            env.close()

    # ── summary: the comparison IS the result ────────────────────────────────
    print(f"\n{'='*66}\nSummary  ({time.time()-t0:.0f}s)\n{'='*66}")
    by_cond: dict[str, list[dict]] = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)

    base = None
    print(f"{'condition':<12}{'SR':>8}{'vs canonical':>14}{'moved':>8}"
          f"{'grip changes':>14}")
    for cond in a.conditions:
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        sr = float(np.mean([r["success"] for r in rs]))
        moved = float(np.mean([r["max_state_displacement"] > 0.05 for r in rs]))
        grip = float(np.mean([r["gripper_changes"] for r in rs]))
        if base is None:
            base = sr
        delta = "--" if cond == a.conditions[0] else f"{sr - base:+.1%}"
        print(f"{cond:<12}{sr:>8.1%}{delta:>14}{moved:>8.0%}{grip:>14.1f}")

    write_json(dst, {"dataset": "libero", "exp_id": a.exp,
                    "checkpoint": str(ckpt), "episodes": a.episodes,
                    "conditions": a.conditions, "results": results,
                    "training_seed": training_seed,
                    "protocol_id": protocol["protocol_id"] if protocol else None,
                    "run_signature": checkpoint.get("run_signature"),
                    "cache_id": checkpoint.get("cache_id"),
                    "checkpoint_sha256": file_digest(ckpt),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "optimizer_step": checkpoint.get("optimizer_step", checkpoint.get("global_step")),
                    "torch_version": torch.__version__,
                    "deterministic_inference": torch.are_deterministic_algorithms_enabled(),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "configuration": asdict(cfg)})
    print(f"\nwrote {dst}")

    if base is not None and base < 0.05:
        print("\n[!] Canonical SR is at floor — the instruction comparison below it "
              "is uninterpretable. Fix the policy or the observation path first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
