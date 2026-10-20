"""Concurrency checks for the RunPod success-stopped study runner."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch

from scripts import runpod_success_study as study


class MockProcess:
    next_pid = 2000

    def __init__(self, *_args, **_kwargs):
        self.pid = MockProcess.next_pid
        MockProcess.next_pid += 1
        self.returncode = None
        self.started = time.monotonic()

    def poll(self):
        if time.monotonic() - self.started > 0.05 and self.returncode is None:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def send_signal(self, _signal):
        self.returncode = -2

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def make_job(root: Path, index: int) -> dict:
    seed = 3 + index // 2
    experiment = "exp05" if index % 2 == 0 else "exp06"
    run = root / "runs" / f"{experiment}_seed{seed}"
    durable = root / "durable" / f"{experiment}_seed{seed}"
    return {"dataset": "aloha", "experiment": experiment,
            "backbone": "groot" if experiment == "exp05" else "qwen3vl",
            "training_seed": seed, "run": run, "durable_run": durable,
            "command": ["python", "scripts/train.py", "--rollout-durable-root", str(root / "durable")],
            "protocol_id": "protocol", "run_signature": f"signature-{index}",
            "source_assets": {"fixture": "sha256"},
            "settings": {"final_evaluation": {}}}


class RunpodSuccessStudyTests(unittest.TestCase):
    def test_two_training_processes_have_isolated_runner_status(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            jobs = [make_job(root, 0), make_job(root, 1)]
            registry, lock = {}, threading.Lock()
            with patch.object(study.subprocess, "Popen", MockProcess):
                threads = [threading.Thread(target=study.run_training_job,
                                            kwargs={"job": job, "durable_root": root / "durable",
                                                    "poll_seconds": 0.5, "process_registry": registry,
                                                    "registry_lock": lock}) for job in jobs]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)
                self.assertTrue(all(not thread.is_alive() for thread in threads))
            local_statuses = [job["run"] / "progress_runner.json" for job in jobs]
            durable_statuses = [job["durable_run"] / "progress_runner.json" for job in jobs]
            self.assertEqual(len(set(local_statuses)), 2)
            self.assertEqual(len(set(durable_statuses)), 2)
            for path in local_statuses + durable_statuses:
                state = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(state["stage"], "training_complete")
                self.assertEqual(state["return_code"], 0)
            self.assertFalse(registry)

    def test_main_runs_all_six_policies_once_with_two_slots(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps({"name": "test"}), encoding="utf-8")
            run_root = root / "local-study"
            durable_root = root / "cloud"
            jobs = [make_job(root, i) for i in range(6)]
            active = 0
            max_active = 0
            lock = threading.Lock()
            calls = []

            def fake_training(job, **_kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    calls.append(study.policy_key(job))
                time.sleep(0.03)
                with lock:
                    active -= 1
                return {"elapsed_s": 0.03}

            def fake_publish(job, **_kwargs):
                return {"task_id": f"task-{study.policy_key(job)}",
                        "selection_epoch": 100, "selected_checkpoint_sha256": "abc"}

            real_is_dir = Path.is_dir

            def is_dir(path):
                if path.name == "workspace-global":
                    return True
                return real_is_dir(path)

            argv = ["--dataset", "aloha", "--plan", str(plan_path),
                    "--run-root", str(run_root), "--durable-root", str(durable_root),
                    "--workers", "1", "--parallel-policies", "2"]
            with patch.object(study, "build_jobs", return_value=jobs), \
                 patch.object(study.torch.cuda, "is_available", return_value=True), \
                 patch.object(study, "file_digest", return_value="fixed-hash"), \
                 patch.object(study, "publish_code_snapshot"), \
                 patch.object(study, "run_training_job", side_effect=fake_training), \
                 patch.object(study, "publish_final", side_effect=fake_publish), \
                 patch.object(Path, "is_dir", autospec=True, side_effect=is_dir):
                self.assertEqual(study.main(argv), 0)

            self.assertEqual(len(calls), 6)
            self.assertEqual(len(set(calls)), 6)
            self.assertEqual(max_active, 2)
            self.assertEqual(calls[:2], [study.policy_key(jobs[0]), study.policy_key(jobs[1])])
            final = json.loads((run_root / "study_status.json").read_text(encoding="utf-8"))
            self.assertEqual(len(final["completed_policies"]), 6)
            self.assertEqual(len(final["pending_final_tests"]), 6)


if __name__ == "__main__":
    unittest.main()
