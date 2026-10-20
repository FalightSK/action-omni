"""Run the locked matched study sequentially, reusing the existing cache runner.

Dry-run prints all commands without encoding, training or writing files. Real
runs lock the plan/config/code before training. Restart the same command to
resume; existing test records are verified, never overwritten or reselected.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.registry import get_config
from experiment_protocol import (apply_normalization, digest, file_digest,
                                 load_protocol, run_signature, sampling_seed, write_json)
from scripts.analysis.paired_results import outcome_map
from scripts.run_experiment import exclusive_run


def battery_percent():
    """Return the Windows battery charge percentage, or None when unavailable."""
    if os.name != "nt":
        return None

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ac_line_status", ctypes.c_byte), ("battery_flag", ctypes.c_byte),
                    ("battery_life_percent", ctypes.c_byte), ("system_status_flag", ctypes.c_byte),
                    ("battery_life_time", ctypes.c_ulong), ("battery_full_life_time", ctypes.c_ulong)]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return None if status.battery_life_percent == 255 else int(status.battery_life_percent)


def wait_for_battery(stage, state, save):
    """Hold a new heavy stage below 35% until battery charge reaches 70%."""
    percent = battery_percent()
    state["battery_percent"] = percent
    if percent is None or percent >= 35:
        return
    state.update(status="battery_wait", waiting_for=stage)
    save()
    print(f"Battery is {percent}%; waiting to start {stage} until charge reaches 70%.", flush=True)
    while percent is None or percent < 70:
        time.sleep(60)
        percent = battery_percent()
        state["battery_percent"] = percent
        save()
    state.update(status="running")
    state.pop("waiting_for", None)
    save()
    print(f"Battery reached {percent}%; resuming {stage}.", flush=True)


def study_jobs(plan, run_root, datasets, libero_python=None, workers=6):
    if (plan.get("schema") != 1 or plan.get("checkpoint_rule") != "fixed_endpoint_final"
            or set(plan["datasets"]) != {"aloha", "libero"}
            or plan.get("family_size") != 6):
        raise ValueError("Unsupported study schema/selection rule/comparison family")
    seeds = plan["training_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Need three distinct nonnegative independent training seeds")
    jobs = []
    for dataset in datasets:
        settings = plan["datasets"][dataset]
        if settings["experiments"] != ["exp05", "exp06"] or settings["epochs"] < 1:
            raise ValueError("Study requires the matched exp05/exp06 arms and positive epochs")
        protocol_path = ROOT / settings["split_manifest"]
        protocol = load_protocol(protocol_path, dataset)
        configs = []
        for exp in settings["experiments"]:
            cfg = get_config(dataset, exp)
            apply_normalization(cfg, protocol)
            cfg.num_epochs = cfg.min_epochs = settings["epochs"]
            configs.append(cfg)
        allowed = {"vlm_backbone", "output_dir", "embeddings_cache", "model_path", "embed_batch_size"}
        controlled = [{k: v for k, v in asdict(c).items() if k not in allowed} for c in configs]
        if controlled[0] != controlled[1] or [c.vlm_backbone for c in configs] != ["groot", "qwen3vl"]:
            raise ValueError("Arms differ beyond backbone/source/cache settings")
        for index, seed in enumerate(seeds):
            arms = list(zip(settings["experiments"], configs))
            for exp, cfg in (arms if index % 2 == 0 else reversed(arms)):
                run = run_root / dataset / Path(cfg.output_dir).name / f"seed_{seed:03d}"
                checkpoint = run / "checkpoints/final.pt"
                output = run / "evaluation_test.json"
                train = [sys.executable, "-u", str(ROOT / "scripts/run_experiment.py"),
                         "--dataset", dataset, "--exp", exp, "--seed", str(seed),
                         "--split-manifest", str(protocol_path), "--run-root", str(run_root),
                         "--workers", str(workers), "--epochs", str(cfg.num_epochs),
                         "--min-epochs", str(cfg.min_epochs)]
                if dataset == "aloha":
                    if (settings["evaluation_episodes"] < 50 or settings["evaluation_seed_offset"] < 0
                            or settings.get("render_samples") != 0):
                        raise ValueError("ALOHA needs 50+ starts, nonnegative offset, and deterministic camera samples=0")
                    evaluate = [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"),
                                "--dataset", dataset, "--exp", exp, "--checkpoint", str(checkpoint),
                                "--episodes", str(settings["evaluation_episodes"]), "--seed-offset",
                                str(settings["evaluation_seed_offset"]), "--render-samples", "0",
                                "--no-video", "--output", str(output)]
                else:
                    if (settings["evaluation_episodes"] != 50
                            or settings["evaluation_tasks"] != list(range(10))):
                        raise ValueError("LIBERO requires all 50 standard starts for all 10 tasks")
                    evaluate = [str(libero_python or "__LIBERO_PYTHON__"), "-u",
                                str(ROOT / "scripts/eval_libero.py"), "--exp", exp,
                                "--checkpoint", str(checkpoint), "--episodes", "50",
                                "--conditions", "canonical", "--videos", "0", "--seed", "0",
                                "--output", str(output)]
                jobs.append({"dataset": dataset, "experiment": exp, "training_seed": seed,
                             "epochs": cfg.num_epochs, "protocol_id": protocol["protocol_id"],
                             "run_signature": run_signature(cfg, protocol, seed),
                             "configuration": json.loads(json.dumps({**asdict(cfg), "output_dir": str(run)})),
                             "run": str(run), "checkpoint": str(checkpoint), "output": str(output),
                             "train": train, "evaluate": evaluate})
    return jobs


def verify_result(job, settings):
    result = json.loads(Path(job["output"]).read_text(encoding="utf-8"))
    manifest = json.loads((Path(job["run"]) / "run_manifest.json").read_text(encoding="utf-8"))
    selected = json.loads((Path(job["run"]) / "selected_policy.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("completed_epoch") != job["epochs"]
            or result.get("dataset") != job["dataset"] or result.get("exp_id") != job["experiment"]
            or result.get("training_seed") != job["training_seed"]
            or result.get("protocol_id") != job["protocol_id"]
            or result.get("run_signature") != manifest.get("run_signature")
            or result.get("run_signature") != job["run_signature"]
            or result.get("cache_id") != manifest.get("cache_id")
            or not result.get("run_signature") or not result.get("cache_id")
            or result.get("checkpoint_epoch") != job["epochs"]
            or result.get("deterministic_inference") is not True
            or result.get("cublas_workspace_config") != ":4096:8"
            or selected.get("rule") != "fixed_endpoint_final"
            or selected.get("checkpoint_sha256") != result.get("checkpoint_sha256")
            or selected.get("checkpoint_epoch") != job["epochs"]
            or selected.get("training_seed") != job["training_seed"]
            or result.get("checkpoint_sha256") != file_digest(job["checkpoint"])
            or result.get("trace") is not None or result.get("attention_instrumented", False)):
        raise ValueError(f"Test record differs from locked checkpoint/protocol: {job['output']}")
    cfg = result["configuration"]
    expected_cfg = job["configuration"]
    if expected_cfg.get("vlm_backbone", "qwen") != "qwen":
        # VLAModel resolves this placeholder from the loaded backbone at inference.
        cfg = dict(cfg)
        expected_cfg = dict(expected_cfg)
        token_id = cfg.pop("image_token_id", None)
        expected_cfg.pop("image_token_id", None)
        if type(token_id) is not int or token_id < 0:
            raise ValueError("Resolved backbone image token ID is missing or invalid")
    if cfg != expected_cfg:
        raise ValueError("Test configuration differs from the locked arm/training budget")
    records = outcome_map(result)
    if job["dataset"] == "aloha":
        start = settings["evaluation_seed_offset"]
        expected = {(0, i) for i in range(start, start + settings["evaluation_episodes"])}
        if (result.get("n_episodes") != len(expected)
                or result.get("render_samples") != settings["render_samples"]
                or result.get("action_horizon") != cfg["action_horizon"]
                or result.get("inference_horizon") != cfg["inference_horizon"]):
            raise ValueError("ALOHA count/inference settings differ from locked configuration")
    else:
        expected = {(t, i) for t in settings["evaluation_tasks"] for i in range(settings["evaluation_episodes"])}
        if result.get("conditions") != ["canonical"]:
            raise ValueError("Primary LIBERO test must use only canonical instructions")
    if records.keys() != expected:
        raise ValueError("Test starts differ from the predeclared cohort")
    for (task, start), (_, env_seed, policy_seed) in records.items():
        expected_env = start * 137 + 42 if job["dataset"] == "aloha" else start
        if (env_seed != expected_env or policy_seed != sampling_seed(job["dataset"], job["training_seed"], task, start)):
            raise ValueError("Environment or policy-sampling seed differs from plan")
    wins = sum(r[0] for r in records.values())
    if job["dataset"] == "aloha" and result["success_rate"] != round(wins / len(records), 4):
        raise ValueError("Recorded success rate differs from episode outcomes")
    return {"dataset": job["dataset"], "experiment": job["experiment"],
            "training_seed": job["training_seed"], "successes": wins, "episodes": len(records),
            "success_rate": wins / len(records), "output": job["output"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/camera_ready_study.json")
    ap.add_argument("--run-root", type=Path, default=ROOT / "asset/runs/camera_ready_fixed_endpoint_v1")
    ap.add_argument("--datasets", nargs="+", choices=("aloha", "libero"), default=["aloha", "libero"])
    ap.add_argument("--libero-python", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.workers < 0 or len(set(args.datasets)) != len(args.datasets):
        ap.error("Workers must be nonnegative; datasets must be distinct")
    os.chdir(ROOT)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    run_root = args.run_root.resolve()
    jobs = study_jobs(plan, run_root, args.datasets, args.libero_python, args.workers)
    if args.dry_run:
        print(json.dumps({"mode": "dry_run; no files written", "jobs": jobs}, indent=2))
        return
    if "libero" in args.datasets and (args.libero_python is None or not args.libero_python.is_file()):
        ap.error("Supply the LIBERO simulator environment with --libero-python")
    code_paths = ["scripts/run_study.py", "scripts/run_log.py", "scripts/prepare_protocol.py",
                  "scripts/run_experiment.py", "scripts/evaluate.py",
                  "scripts/eval_libero.py", "scripts/analysis/paired_results.py", "scripts/plot_training.py", "scripts/train.py",
                  "experiment_protocol.py", "envs/aloha_env.py", "envs/libero_env.py"]
    contract = {"plan": plan, "plan_id": digest(plan),
                "training_signatures": [{k: j[k] for k in ("dataset", "experiment", "training_seed", "run_signature")}
                                        for j in study_jobs(plan, run_root, ["aloha", "libero"], args.libero_python, args.workers)],
                "workflow_code": {p: file_digest(ROOT / p) for p in code_paths}}
    with exclusive_run(run_root / ".study.lock"):
        lock = run_root / "locked_study.json"
        if lock.exists():
            if json.loads(lock.read_text(encoding="utf-8")) != contract:
                raise ValueError("Locked plan/code changed; restore it or use a new run root")
        else:
            write_json(lock, contract)
        state = {"plan_id": contract["plan_id"], "status": "running", "pid": os.getpid(),
                 "datasets": args.datasets, "completed_jobs": [], "total_jobs": len(jobs)}

        def save():
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(run_root / "study_status.json", state)

        def execute(stage, command, dataset):
            if stage in {"training", "evaluation"}:
                wait_for_battery(stage, state, save)
            state.update(stage=stage, command=command)
            save()
            print(f"Starting {stage}: {subprocess.list2cmdline(command)}", flush=True)
            logs = run_root / "logs"
            logs.mkdir(exist_ok=True)
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            log = logs / f"{name}_{stage}.log"
            state["log"] = str(log)
            save()
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                       MUJOCO_GL="wgl" if dataset == "libero" and stage == "evaluation" else "glfw")
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

        try:
            save()
            for job in jobs:
                state["current_job"] = job
                save()
                if not Path(job["output"]).exists():
                    execute("training", job["train"], job["dataset"])
                    manifest = json.loads((Path(job["run"]) / "run_manifest.json").read_text())
                    if manifest["completed_epoch"] != job["epochs"]:
                        raise ValueError("Training did not reach its fixed endpoint")
                    # Selection is recorded before any test outcomes are available.
                    selection = {"rule": plan["checkpoint_rule"], "plan_id": contract["plan_id"],
                                 "checkpoint": job["checkpoint"], "checkpoint_sha256": file_digest(job["checkpoint"]),
                                 "checkpoint_epoch": job["epochs"], "training_seed": job["training_seed"]}
                    selected = Path(job["run"]) / "selected_policy.json"
                    if selected.exists() and json.loads(selected.read_text()) != selection:
                        raise ValueError("Selected checkpoint changed after it was locked")
                    write_json(selected, selection)
                    execute("plot", [sys.executable, str(ROOT / "scripts/plot_training.py"),
                                     str(Path(job["run"]) / "history.jsonl")], job["dataset"])
                    execute("evaluation", job["evaluate"], job["dataset"])
                state["completed_jobs"].append(verify_result(job, plan["datasets"][job["dataset"]]))
                save()
            pairs = []
            for dataset in args.datasets:
                for seed in plan["training_seeds"]:
                    group = sorted((j for j in jobs if j["dataset"] == dataset and j["training_seed"] == seed),
                                   key=lambda j: j["experiment"])
                    pairs.append({"label": dataset, "left": group[0]["output"], "right": group[1]["output"]})
            paired_manifest = run_root / ("paired_" + "_".join(args.datasets) + ".json")
            write_json(paired_manifest, {"family_size": plan["family_size"], "pairs": pairs,
                                        "labels": {d: {"left": "GR00T", "right": "Qwen3-VL"} for d in args.datasets}})
            execute("analysis", [sys.executable, str(ROOT / "scripts/analysis/paired_results.py"),
                                 "--manifest", str(paired_manifest), "--output", str(paired_manifest.with_name(paired_manifest.stem + "_results.json")),
                                 "--report-dir", str(run_root / ("report_" + "_".join(args.datasets)))], args.datasets[0])
            state.update(status="complete", stage="complete")
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
