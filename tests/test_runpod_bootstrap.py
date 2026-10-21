"""Focused tests for bounded bootstrap GPU-memory sampling."""
from contextlib import redirect_stdout
import io
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.runpod_bootstrap import (cache_source_for_plan, log_nvidia_smi_sample,
                                     sample_nvidia_smi_until_stopped)


class StopAfterSample:
    def __init__(self):
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def wait(self, interval_s):
        self.waits.append(interval_s)
        self.stopped = True
        return True


class RunpodBootstrapTests(unittest.TestCase):
    def test_fresh_plan_reuses_only_pinned_cache_science_and_namespace(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / "configs/runpod_fresh_cached_success_study.json").read_text())
        for role in ("aloha", "libero"):
            with self.subTest(role=role):
                source_study, source_id, source_path = cache_source_for_plan(
                    plan, role, "runpod_success_stopped_fresh_cached_v1_" + role, root)
                self.assertEqual(source_study, "runpod_success_stopped_v6_" + role)
                self.assertEqual(source_id, plan["cache_source"]["plan_id"])
                self.assertEqual(source_path, root / "configs/runpod_cached_success_study.json")
        altered = copy.deepcopy(plan)
        altered["stopping"]["patience_epochs"] = 10
        with self.assertRaises(ValueError):
            cache_source_for_plan(altered, "aloha", "runpod_success_stopped_fresh_cached_v1_aloha", root)
        with self.assertRaises(ValueError):
            cache_source_for_plan(plan, "aloha", "runpod_success_stopped_v6_aloha", root)
        altered = copy.deepcopy(plan)
        altered.pop("initialization")
        with self.assertRaises(ValueError):
            cache_source_for_plan(altered, "aloha", "runpod_success_stopped_fresh_cached_v1_aloha", root)

    def test_nvidia_smi_sampler_logs_one_snapshot_then_stops(self):
        stop = StopAfterSample()
        stdout = io.StringIO()
        result = SimpleNamespace(returncode=0, stdout="GPU memory snapshot", stderr="")
        with patch("scripts.runpod_bootstrap.subprocess.run", return_value=result) as run_command:
            with redirect_stdout(stdout):
                sample_nvidia_smi_until_stopped(
                    stop, lambda: log_nvidia_smi_sample("prepare_feature_caches"), interval_s=30)

        run_command.assert_called_once()
        self.assertEqual(run_command.call_args.args[0], ["nvidia-smi"])
        self.assertEqual(stop.waits, [30])
        self.assertIn('"stdout": "GPU memory snapshot"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
