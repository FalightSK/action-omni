"""Run one or both dataset workers of the seed-locked camera-ready study.

RunPod Global Volume keeps durable outputs; active feature caches and training
I/O stay on each Pod's local disk. Re-runs reuse validated caches and resume.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from experiment_protocol import (apply_normalization, cache_config, digest,
                                 encoder_code_id, file_digest, load_protocol,
                                 run_signature, source_assets, write_json)
from scripts.run_experiment import ensure_cache


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def build_jobs(plan: dict, run_root: Path, datasets: list[str], workers: int) -> list[dict]:
    if plan.get("schema") != 1 or plan.get("checkpoint_rule") != "fixed_update_endpoint":
        raise ValueError("Unsupported RunPod study schema or checkpoint rule")
    seeds = plan["training_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Study requires three distinct, recorded nonnegative training seeds")
    if plan["effective_batch_size"] != 128:
        raise ValueError("This study is locked to effective batch size 128")
    jobs = []
    for dataset in datasets:
        setting = plan["datasets"][dataset]
        protocol_path = ROOT / setting["split_manifest"]
        protocol = load_protocol(protocol_path, dataset)
        for exp in (plan["backbones"]["groot"], plan["backbones"]["qwen3vl"]):
            cfg = get_config(dataset, exp)
            apply_normalization(cfg, protocol)
            cfg.batch_size = plan["effective_batch_size"]
            cfg.num_epochs = setting["epochs"]
            cfg.min_epochs = setting["epochs"]  # fixed horizon; validation never shortens it
            cfg.warmup_steps = plan["warmup_updates"]
            cfg.lr_decay_epochs = None
            if getattr(cfg, "n_views", 1) != len(setting.get("camera_views", ["top"])):
                raise ValueError(f"{dataset}/{exp} camera count differs from the locked plan")
            if cfg.num_epochs * setting["steps_per_epoch"] != plan["optimizer_updates"]:
                raise ValueError(f"{dataset} epoch geometry does not equal the fixed update budget")
            if setting["probe_epochs"] != [step // setting["steps_per_epoch"]
                                            for step in setting["probe_updates"]]:
                raise ValueError(f"{dataset} probe updates do not align with completed epochs")
            assets = source_assets(cfg)
            identity = {"protocol_id": protocol["protocol_id"], "config": cache_config(cfg),
                        "encoder_code_id": encoder_code_id(), "source_assets": assets}
            cache = ROOT / "asset/cache" / dataset / cfg.vlm_backbone / digest(identity) / "vlm_embeddings.h5"
            for seed in seeds:
                run = run_root / dataset / Path(cfg.output_dir).name / f"seed_{seed:03d}"
                run_cfg = json.loads(json.dumps({**asdict(cfg), "output_dir": str(run)}))
                train = [sys.executable, "-u", str(ROOT / "scripts/train.py"),
                         "--dataset", dataset, "--exp", exp, "--seed", str(seed),
                         "--split-manifest", str(protocol_path), "--cache", str(cache),
                         "--output-dir", str(run), "--workers", str(workers),
                         "--epochs", str(setting["epochs"]), "--min-epochs", str(setting["epochs"]),
                         "--batch-size", "128", "--steps-per-epoch", str(setting["steps_per_epoch"]),
                         "--max-steps", str(plan["optimizer_updates"]),
                         "--warmup-steps", str(plan["warmup_updates"]),
                         "--lr-decay-steps", str(plan["optimizer_updates"]),
                         "--checkpoint-every-steps", str(plan["checkpoint_every_updates"]),
                         "--probe-epochs", *map(str, setting["probe_epochs"])]
                jobs.append({"dataset": dataset, "experiment": exp, "backbone": cfg.vlm_backbone,
                             "training_seed": seed, "protocol_id": protocol["protocol_id"],
                             "run_signature": run_signature(cfg, protocol, seed),
                             "config": run_cfg, "cache": str(cache), "cache_identity": identity,
                             "cache_config": cfg, "protocol": protocol, "run": str(run),
                             "train": train, "settings": setting})
    return jobs


def result_path(run: Path, step: int, final: bool) -> Path:
    label = "final" if final else f"step_{step:08d}"
    return run / f"evaluation_{label}.json"


def verify_policy(job: dict, total_steps: int) -> None:
    run = Path(job["run"])
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    final = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    if (manifest.get("status") != "complete" or manifest.get("training_seed") != job["training_seed"]
            or manifest.get("protocol_id") != job["protocol_id"]
            or manifest.get("run_signature") != job["run_signature"]
            or manifest.get("optimizer_steps") != total_steps
            or final.get("training_seed") != job["training_seed"]
            or final.get("global_step") != total_steps
            or final.get("run_signature") != job["run_signature"]):
        raise ValueError(f"Policy did not reach its exact locked endpoint: {run}")
    for step in job["settings"]["probe_updates"]:
        if not (run / "checkpoints" / f"probe_step_{step:08d}.pt").is_file():
            raise FileNotFoundError(f"Missing probe checkpoint at update {step}: {run}")


def evaluation_jobs(job: dict, total_steps: int) -> list[tuple[int, bool, Path, list[str]]]:
    run = Path(job["run"])
    settings = job["settings"]
    specs = [(step, False) for step in settings["probe_updates"]] + [(total_steps, True)]
    output = []
    for step, final in specs:
        checkpoint = run / "checkpoints" / ("final.pt" if final else f"probe_step_{step:08d}.pt")
        result = result_path(run, step, final)
        if job["dataset"] == "aloha":
            count = settings["final_episodes"] if final else settings["probe_episodes"]
            seed_offset = settings["final_seed_offset"] if final else settings["probe_seed_offset"]
            command = [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"),
                       "--dataset", "aloha", "--exp", job["experiment"],
                       "--checkpoint", str(checkpoint), "--episodes", str(count),
                       "--seed-offset", str(seed_offset), "--render-samples", "0",
                       "--no-video", "--output", str(result)]
        else:
            count = settings["final_episodes_per_task"] if final else settings["probe_episodes_per_task"]
            command = [sys.executable, "-u", str(ROOT / "scripts/eval_libero.py"),
                       "--exp", job["experiment"], "--checkpoint", checkpoint.name,
                       "--episodes", str(count), "--conditions", "canonical", "--videos", "0",
                       "--seed", "0", "--output-dir", str(run), "--output", str(result)]
        output.append((step, final, result, command))
    return output


def sync_file(src: Path, dst: Path) -> bool:
    """Copy one changed file and check that its complete size reached storage."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (dst.is_file() and dst.stat().st_size == src.stat().st_size
                and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns):
            return False
    except OSError:
        pass
    # RunPod Global Volume is object-backed and rejects timestamp preservation.
    shutil.copyfile(src, dst)
    if dst.stat().st_size != src.stat().st_size:
        raise IOError(f"Durable copy size mismatch: {src} -> {dst}")
    return True


def sync_durable(source: Path, destination: Path) -> int:
    """Copy changed files to durable storage; active HDF5 caches stay local."""
    if not source.exists():
        return 0
    return sum(sync_file(src, destination / src.relative_to(source))
               for src in source.rglob("*") if src.is_file())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/runpod_update_matched_study.json")
    ap.add_argument("--run-root", type=Path, default=ROOT / "asset/runs/runpod_update_matched_v1")
    ap.add_argument("--datasets", nargs="+", choices=("aloha", "libero"), default=["aloha", "libero"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--durable-root", type=Path,
                    default=Path(os.environ.get("ACTIONOMNI_DURABLE_ROOT", "/workspace-global/action-omni")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.workers < 0 or len(set(args.datasets)) != len(args.datasets):
        ap.error("Workers must be nonnegative and datasets must be distinct")
    os.chdir(ROOT)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    run_root = args.run_root.resolve()
    durable_root = args.durable_root.resolve()
    jobs = build_jobs(plan, run_root, args.datasets, args.workers)
    if args.dry_run:
        print(json.dumps({"mode": "dry-run; no files written", "jobs": [
            {k: v for k, v in job.items() if k not in {"protocol", "cache_config", "cache_identity"}}
            for job in jobs]}, indent=2))
        return
    if not torch_cuda_available():
        raise RuntimeError("CUDA is unavailable; no RunPod training stage was started")
    global_mount = Path("/workspace-global")
    if not global_mount.is_dir():
        raise RuntimeError(f"RunPod Global Volume is not mounted at {global_mount}")
    durable_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    code_paths = ["scripts/runpod_update_matched_study.py", "scripts/prepare_runpod_study.py",
                  "scripts/train.py",
                  "scripts/evaluate.py", "scripts/eval_libero.py", "scripts/precompute.py",
                  "scripts/analysis/paired_results.py", "scripts/plot_training.py",
                  "scripts/plot_study_losses.py",
                  "scripts/run_experiment.py", "experiment_protocol.py", "models/vla_train.py",
                  "models/flow_matching.py", "models/vla.py", "models/backbones.py",
                  "data/aloha/dataset.py", "data/libero/dataset.py"]
    contract = {"plan": plan, "plan_id": digest(plan),
                "training_seeds": plan["training_seeds"],
                "jobs": [{k: job[k] for k in ("dataset", "experiment", "backbone", "training_seed",
                                               "protocol_id", "run_signature")}
                         for job in build_jobs(plan, run_root, ["aloha", "libero"], args.workers)],
                "workflow_code": {p: file_digest(ROOT / p) for p in code_paths}}
    lock = run_root / "locked_study.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != contract:
        raise ValueError("Locked RunPod plan/code changed; use a new run root to avoid mixing results")
    if not lock.exists():
        write_json(lock, contract)

    status_path = run_root / "study_status.json"
    state = {"plan_id": contract["plan_id"], "status": "running", "started_utc": stamp(),
             "training_seeds": plan["training_seeds"], "datasets": args.datasets,
             "completed_policies": [], "total_policies": len(jobs)}

    def save() -> None:
        state["updated_utc"] = stamp()
        write_json(status_path, state)

    def sync_results() -> None:
        if durable_root.exists():
            sync_durable(run_root, durable_root / "runs" / run_root.name)
            sync_file(args.plan.resolve(), durable_root / "configs" / args.plan.name)

    def execute(stage: str, command: list[str], job: dict) -> None:
        state.update(stage=stage, current_policy={k: job[k] for k in
                     ("dataset", "experiment", "training_seed", "run_signature")})
        logs = run_root / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{stamp()}_{stage}_{job['dataset']}_{job['experiment']}_seed{job['training_seed']:03d}.log"
        state["log"] = str(log)
        save()
        sync_results()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   MUJOCO_GL="glfw" if job["dataset"] == "aloha" else "egl")
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(60)
                sync_results()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        sync_results()

    try:
        save()
        cached: set[str] = set()
        for job in jobs:
            state["current_policy"] = {k: job[k] for k in
                                       ("dataset", "experiment", "training_seed", "run_signature")}
            save()
            cache = Path(job["cache"])
            if str(cache) not in cached:
                cache.parent.mkdir(parents=True, exist_ok=True)
                state.update(stage="cache_validation", cache=str(cache))
                save()
                report, reused = ensure_cache(cache, job["cache_config"], job["protocol"],
                                              job["cache_identity"]["source_assets"],
                    lambda partial, j=job: execute("precompute", [sys.executable, "-u",
                        str(ROOT / "scripts/precompute.py"), "--dataset", j["dataset"],
                        "--exp", j["experiment"], "--split-manifest",
                        str(ROOT / plan["datasets"][j["dataset"]]["split_manifest"]),
                        "--output", str(partial)], j))
                write_json(cache.parent / "cache_validation.json", {**report, "reused": reused})
                cached.add(str(cache))
            run = Path(job["run"])
            manifest_path = run / "run_manifest.json"
            if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "complete":
                execute("training", job["train"], job)
            verify_policy(job, plan["optimizer_updates"])
            if not (run / "training_loss.png").is_file():
                execute("plot", [sys.executable, str(ROOT / "scripts/plot_training.py"),
                                  str(run / "history.jsonl")], job)
            for step, final, result, command in evaluation_jobs(job, plan["optimizer_updates"]):
                if not result.exists():
                    execute("evaluation", command, job)
            state["completed_policies"].append({"dataset": job["dataset"],
                                                "experiment": job["experiment"],
                                                "training_seed": job["training_seed"],
                                                "run_signature": job["run_signature"]})
            save()
            sync_results()

        if len(args.datasets) == 1:
            state.update(status="worker_complete", stage="worker_complete", completed_utc=stamp())
        else:
            # Pair final rollout starts within each training seed, then summarize
            # training-seed variation separately from rollout uncertainty.
            pairs = []
            for dataset in ("aloha", "libero"):
                cfg_groot = get_config(dataset, plan["backbones"]["groot"])
                cfg_qwen = get_config(dataset, plan["backbones"]["qwen3vl"])
                for seed in plan["training_seeds"]:
                    groot = Path(run_root / dataset / Path(cfg_groot.output_dir).name
                                 / f"seed_{seed:03d}" / "evaluation_final.json")
                    qwen = Path(run_root / dataset / Path(cfg_qwen.output_dir).name
                                / f"seed_{seed:03d}" / "evaluation_final.json")
                    pairs.append({"label": dataset.upper(),
                                  "left": groot.relative_to(run_root).as_posix(),
                                  "right": qwen.relative_to(run_root).as_posix(),
                                  "vary": "backbone"})
            write_json(run_root / "paired_final_manifest.json", {
                "family_size": len(pairs), "pairs": pairs,
                "labels": {"ALOHA": {"left": "GR00T N1.7", "right": "Qwen3-VL-2B"},
                           "LIBERO": {"left": "GR00T N1.7", "right": "Qwen3-VL-2B"}}})
            state.update(stage="final_analysis", current_policy=None)
            save()
            execute("paired_analysis", [sys.executable, "-u",
                str(ROOT / "scripts/analysis/paired_results.py"), "--manifest",
                str(run_root / "paired_final_manifest.json"), "--output",
                str(run_root / "paired_final.json"), "--report-dir", str(run_root / "reports")], jobs[0])
            execute("loss_analysis", [sys.executable, "-u", str(ROOT / "scripts/plot_study_losses.py"),
                "--plan", str(args.plan.resolve()), "--run-root", str(run_root)], jobs[0])
            state.update(status="complete", stage="complete", completed_utc=stamp())
        # A single-dataset worker is ready to be merged by the finalizer;
        # the durable mount preserves all per-policy checkpoints and reports.
        sync_results()
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
        sync_results()


def torch_cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
