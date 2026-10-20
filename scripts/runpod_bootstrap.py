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
import threading
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


def sample_nvidia_smi_until_stopped(stop_event, sample, interval_s: float = 30.0) -> None:
    """Sample while a bounded cache step runs; caller always signals and joins."""
    while not stop_event.is_set():
        sample()
        if stop_event.wait(interval_s):
            return


def log_nvidia_smi_sample(label: str) -> None:
    """Write one complete, unparsed nvidia-smi snapshot to container logs."""
    record = {
        "event": "bootstrap.nvidia_smi_sample",
        "label": label,
        "sampled_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["nvidia-smi"],
    }
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                timeout=10, check=False)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except Exception as exc:
        record.update(returncode=None, stdout="", stderr=str(exc), error_type=type(exc).__name__)
    print(json.dumps(record, sort_keys=True), flush=True)


def materialize_source(payload: dict) -> tuple[Path, str]:
    base = Path(payload["base_archive"])
    data = base.read_bytes()
    if sha(data) != payload["base_sha256"]:
        raise ValueError("Cloud base source archive SHA-256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
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
            study_id = payload["study_id"]
            if not study_id.startswith("runpod_success_stopped_") or not study_id.endswith("_" + role):
                raise ValueError("Study namespace does not match the training role")
            run("runtime_setup", ["bash", "scripts/setup_runpod_train.sh", role, str(durable)])
            python = str(root / ".venv/bin/python")
            run("unit_checks", [python, "-m", "unittest", "scripts.success_stopping",
                "tests.test_success_queue", "tests.test_eval_worker_queue",
                "tests.test_vision_patch_projection", "tests.test_online_encoder_stage",
                "tests.test_checkpoint_io", "tests.test_runpod_success_study",
                "tests.test_plot_study_losses", "tests.test_prepare_cached_study",
                "tests.test_runpod_bootstrap", "-v"])
            plan_path = root / "configs/runpod_cached_success_study.json"
            run("gpu_layout_check", [python, "-c",
                "import torch,json; assert torch.cuda.device_count()==1; "
                "name=torch.cuda.get_device_name(0); "
                "assert 'B300' in name; "
                "print(json.dumps({'event':'bootstrap.gpu_layout_verified','name':name,"
                "'memory_bytes':torch.cuda.get_device_properties(0).total_memory}))"])
            cache_command = [python, "-u", "scripts/prepare_cached_study.py",
                "--dataset", role, "--durable-root", str(durable), "--study-id", study_id,
                "--plan", str(plan_path)]
            event("nvidia_smi_before_cacheprep", command=["nvidia-smi"])
            log_nvidia_smi_sample("before_cacheprep")
            sampler_stop = threading.Event()
            sampler = threading.Thread(
                target=sample_nvidia_smi_until_stopped,
                args=(sampler_stop, lambda: log_nvidia_smi_sample("prepare_feature_caches")),
                name="cache-nvidia-smi-sampler", daemon=True)
            event("gpu_sampling_started", interval_s=30, command=["nvidia-smi"])
            sampler.start()
            try:
                run("prepare_feature_caches", cache_command)
            finally:
                sampler_stop.set()
                sampler.join(timeout=12)
                event("gpu_sampling_stopped", sampler_thread_alive=sampler.is_alive())
            migration = payload.get("preselection_migration")
            if not migration:
                raise ValueError("Cached continuation requires its reviewed migration payload")
            if migration:
                migration_code = base64.b64decode(migration["base64"])
                if sha(migration_code) != migration["sha256"]:
                    raise ValueError("Preselection continuation script SHA-256 mismatch")
                migration_path = status_dir / ("preselection_continuation_" + migration["sha256"][:12] + ".py")
                migration_path.write_bytes(migration_code)
                run("preselection_continuation", [python, "-u", str(migration_path),
                    "--dataset", role, "--durable-root", str(durable),
                    "--new-source-archive", payload["source_archive"],
                    "--cache-manifest", str(durable / "success_studies" / study_id / "feature_caches"),
                    "--new-study-id", study_id])
                run("loss_curve_backfill", [python, "-c",
                    "from pathlib import Path; from scripts.plot_study_losses import plot_run_losses; "
                    "from experiment_protocol import file_digest; import json; "
                    "root=Path(" + repr(str(durable / "success_studies" / study_id)) + "); "
                    "curves=[plot_run_losses(h.parent) for h in root.rglob('history.jsonl')]; "
                    "print(json.dumps({'event':'training.loss_curves_backfilled', 'curves':"
                    "[{'path':str(p), 'bytes':p.stat().st_size, 'sha256':file_digest(p)} for p in curves if p]}))"])
            command = [python, "-u", "scripts/runpod_success_study.py",
                "--dataset", role, "--durable-root", str(durable),
                "--plan", str(plan_path),
                "--run-root", str(root / "asset/runs" / study_id),
                "--workers", "1", "--parallel-policies", "2"]
            run("success_study", command)
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
