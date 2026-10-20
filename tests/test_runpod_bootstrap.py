"""Focused tests for bounded bootstrap GPU-memory sampling."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.runpod_bootstrap import log_nvidia_smi_sample, sample_nvidia_smi_until_stopped


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
