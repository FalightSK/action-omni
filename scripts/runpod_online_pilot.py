"""Check each configured backbone on one seed before a full three-Pod study.

Pilot budgets and rollout counts are deliberately tiny and published in a
separate namespace. The production study configuration is never modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_protocol import digest, write_json
from scripts.run_log import log_event
from scripts.runpod_update_matched_study import (build_jobs, publish_checkpoint_task,
                                               sync_durable, verify_policy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("aloha", "libero"), required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_update_matched_study.json")
    parser.add_argument("--durable-root", type=Path, default=Path("/workspace-global/action-omni/study_v1"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--name", default="runpod_online_pilot_v2")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.name.startswith("runpod_online_pilot_") or not args.name.replace("_", "").isalnum():
        parser.error("Pilot namespace must start with runpod_online_pilot_ and use letters/digits/underscores")
    original = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.seed not in original["training_seeds"] or args.steps < 1:
        parser.error("Use a seed from the locked study and a positive update count")
    if not Path("/workspace-global").is_dir():
        raise RuntimeError("RunPod cloud storage is not mounted")
    plan = deepcopy(original)
    plan.update(name=args.name, purpose="Pipeline smoke only; not experiment results",
                optimizer_updates=args.steps, warmup_updates=0,
                checkpoint_every_updates=args.steps, pilot_seed=args.seed)
    for dataset, settings in plan["datasets"].items():
        settings.update(epochs=1, steps_per_epoch=args.steps, probe_updates=[], probe_epochs=[])
        if dataset == "aloha":
            settings["final_episodes"] = 1
        else:
            settings.update(final_episodes_per_task=1, tasks=[0])
    run_root = ROOT / "asset/runs" / f"{plan['name']}_{args.dataset}"
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "pilot_plan.json", plan)
    jobs = [job for job in build_jobs(plan, run_root, [args.dataset], args.workers)
            if job["training_seed"] == args.seed]
    started = time.perf_counter()
    for job in jobs:
        run = Path(job["run"])
        command = job["train"] + ["--online-encoder", "--log-every-steps", "1"]
        log_event("pilot.training_started", dataset=args.dataset, backbone=job["backbone"],
                  seed=args.seed, updates=args.steps, output=str(run))
        subprocess.run(command, cwd=ROOT,
                       env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1"),
                       check=True)
        verify_policy(job, args.steps)
        publish_checkpoint_task(job, step=args.steps, final=True,
                                source=run / "checkpoints/final.pt", durable_root=args.durable_root,
                                study_id=plan["name"], plan_id=digest(plan), online_encoder=True)
        sync_durable(run_root, args.durable_root / "runs" / run_root.name)
        log_event("pilot.policy_complete", dataset=args.dataset, backbone=job["backbone"], seed=args.seed)
    write_json(run_root / "pilot_complete.json", {"dataset": args.dataset, "seed": args.seed,
               "updates_per_policy": args.steps, "policies": len(jobs),
               "elapsed_s": round(time.perf_counter() - started, 1)})
    sync_durable(run_root, args.durable_root / "runs" / run_root.name)
    log_event("pilot.training_complete", dataset=args.dataset, policies=len(jobs),
              elapsed_s=round(time.perf_counter() - started, 1))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log_event("pilot.failed", level="ERROR", error_type=type(exc).__name__, error=str(exc))
        raise
