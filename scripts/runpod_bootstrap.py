"""Run the existing CLI pipeline from a Pod command configured through MCP.

No RunPod API credentials or SSH are used. Source overlays are embedded in the
reviewed startup payload and verified by SHA-256 before any setup/work begins.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import zlib


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    # Raw tar is deterministic across zlib versions on the caller/Pod hosts.
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def materialize_source(payload: dict) -> tuple[Path, str]:
    base = Path(payload["base_archive"])
    data = base.read_bytes()
    if sha(data) != payload["base_sha256"]:
        raise ValueError("Cloud base source archive SHA-256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            if member.isfile():
                name = member.name.removeprefix("./")
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError("Unsafe source archive member")
                files[name] = tar.extractfile(member).read()
    for name, item in payload.get("overlays", {}).items():
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("Unsafe overlay path")
        if "edits" in item:
            content = files[name]
            if sha(content) != item["base_sha256"]:
                raise ValueError("Source delta base SHA-256 mismatch: " + name)
            for edit in reversed(item["edits"]):
                content = (content[:edit["start"]] + base64.b64decode(edit["base64"])
                           + content[edit["stop"]:])
        else:
            content = base64.b64decode(item["base64"])
        if sha(content) != item["sha256"]:
            raise ValueError("Source overlay SHA-256 mismatch: " + name)
        files[name] = content
    archive_data = canonical_archive(files)
    digest = sha(archive_data)
    if digest != payload["source_sha256"]:
        raise ValueError("Materialized source SHA-256 differs from reviewed payload")
    destination = Path(payload["source_archive"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha(destination.read_bytes()) != digest:
        raise ValueError("Refusing to replace a different immutable source archive")
    if not destination.exists():
        temporary = destination.with_name(destination.name + "." + payload["role"] + ".tmp")
        temporary.write_bytes(archive_data)
        temporary.replace(destination)
    root = Path("/workspace/action-omni")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:*") as tar:
        tar.extractall(root, filter="data")
    return root, digest


def main(payload: dict) -> None:
    role = payload["role"]
    if role not in {"evaluation", "aloha", "libero"}:
        raise ValueError("Unknown bootstrap role")
    durable = Path("/workspace-global/action-omni/study_v1")
    if not durable.is_dir():
        raise RuntimeError("Expected Global Volume study root is unavailable")
    status_dir = durable / "workers" / role
    status_dir.mkdir(parents=True, exist_ok=True)
    log_path = status_dir / "mcp_bootstrap.log"
    status_path = status_dir / "mcp_bootstrap_status.json"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               ACTIONOMNI_DURABLE_ROOT=str(durable), OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PIP_CACHE_DIR=str(durable / "cache/pip"),
               UV_CACHE_DIR="/tmp/action-omni-uv-cache",
               HF_HOME=str(durable / "cache/huggingface"),
               LIBERO_CONFIG_PATH=str(durable / "config/libero"),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", MUJOCO_GL="egl")
    state = {"role": role, "status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
             "launch_channel": "RunPod MCP container command", "source_sha256": payload["source_sha256"]}

    def event(stage: str, **fields) -> None:
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_path)
        line = json.dumps({"event": "bootstrap." + stage, **state}, sort_keys=True)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")

    def run(stage: str, command: list[str], capture: Path | None = None) -> None:
        event(stage, command=command)
        with log_path.open("a", encoding="utf-8") as log:
            output = capture.open("w", encoding="utf-8") if capture else None
            try:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if output:
                        output.write(line)
                        output.flush()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if output:
                    output.close()
        event(stage + "_complete", command=None)

    try:
        event("source_verification")
        root, digest = materialize_source(payload)
        event("source_ready", source_archive=payload["source_archive"], source_sha256=digest)
        if role == "evaluation":
            run("runtime_setup", ["bash", "scripts/setup_runpod_eval.sh"])
            aloha_python = str(root / ".venv-eval-aloha/bin/python")
            libero_python = str(root / ".venv-eval-libero/bin/python")
            run("patch_unit_checks", [aloha_python, "-m", "unittest",
                "tests.test_vision_patch_projection", "-v"])
            for dataset, interpreter in (("aloha", aloha_python), ("libero", libero_python)):
                run("renderer_check_" + dataset, [interpreter, "-u", "scripts/check_simulation_renderer.py",
                    "--dataset", dataset, "--mujoco-gl", "egl", "--require-hardware-renderer"],
                    status_dir / (dataset + "_renderer.jsonl"))
            run("evaluation_worker", [aloha_python, "-u", "scripts/runpod_eval_worker.py",
                "--durable-root", str(durable), "--aloha-python", aloha_python,
                "--libero-python", libero_python, "--aloha-gl", "egl", "--poll-seconds", "15"])
        else:
            run("runtime_setup", ["bash", "scripts/setup_runpod_train.sh", role, str(durable)])
            python = str(root / ".venv/bin/python")
            run("unit_checks", [python, "-m", "unittest", "scripts.success_stopping",
                "tests.test_success_queue", "tests.test_eval_worker_queue",
                "tests.test_vision_patch_projection", "tests.test_online_encoder_stage",
                "tests.test_runpod_success_study", "-v"])
            if payload.get("direct_training"):
                # The first real batch checks old/new VLM features on the same
                # loaded model, then immediately performs an optimizer update.
                run("success_study", [python, "-u", "scripts/runpod_success_study.py",
                    "--dataset", role, "--durable-root", str(durable),
                    "--run-root", str(root / "asset/runs" / ("runpod_success_stopped_v2_" + role)),
                    "--workers", "1", "--parallel-policies", "2"])
                event("complete", status="complete")
                while True:
                    time.sleep(30)
            for backbone in ("groot", "qwen3vl"):
                run("patch_projection_check_" + backbone,
                    [python, "-u", "scripts/verify_vision_patch_projection.py",
                     "--dataset", role, "--backbone", backbone,
                     "--output", str(status_dir / ("patch_parity_" + backbone + ".json"))])
            run("throughput_check", [python, "-u", "scripts/profile_online_training.py",
                "--dataset", role, "--backbone", "groot", "--repeats", "2", "--workers", "2",
                "--output-json", str(status_dir / "profile_optimized.json")])
            profile = json.loads((status_dir / "profile_optimized.json").read_text())
            slowest = max(row["optimizer_step_wall_s"] for row in profile["updates"])
            if slowest > payload.get("maximum_pilot_step_seconds", 10):
                raise RuntimeError("Diagnostic throughput gate failed before long training: " + str(slowest))
            if payload.get("train_after_checks"):
                run("pipeline_pilot", [python, "-u", "scripts/runpod_online_pilot.py",
                    "--dataset", role, "--durable-root", str(durable)])
                run("pilot_rollout_gate", [python, "-u", "scripts/wait_for_pilot_rollouts.py",
                    "--dataset", role, "--durable-root", str(durable)])
                run("success_study", [python, "-u", "scripts/runpod_success_study.py",
                    "--dataset", role, "--durable-root", str(durable),
                    "--workers", "1", "--parallel-policies", "2"])
        event("complete", status="complete")
    except BaseException as exc:
        event("failed", status="failed", error_type=type(exc).__name__, error=str(exc),
              traceback=traceback.format_exc())
        # Preserve the evidence on the durable volume and in MCP container logs.
        # Power control is performed only through RunPod MCP by the caller.
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main(json.loads(zlib.decompress(base64.b64decode("".join(sys.argv[1:])))))
