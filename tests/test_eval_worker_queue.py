"""Worker scheduling checks that avoid simulator/GPU startup."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from scripts import runpod_eval_worker as worker


class EvalWorkerQueueTests(unittest.TestCase):
    def test_libero_offset_only_legacy_task_does_not_require_ids(self):
        self.assertEqual(worker.libero_initial_state_offset({"initial_state_offset": 10}), 10)
        self.assertEqual(worker.libero_initial_state_offset({"initial_state_ids": [10, 11]}), 10)
        self.assertIsNone(worker.libero_initial_state_offset({"tasks": [0, 1]}))

    def test_priority_fifo_and_re_discovery_after_each_evaluation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            durable = root / "durable"
            durable.mkdir()
            task_dir = durable / "evaluation_tasks"
            task_dir.mkdir()
            manifests = [
                {"schema": 1, "status": "ready", "task_id": "final-old",
                 "purpose": "final_test", "created_utc": "2026-10-20T00:00:00+00:00"},
                {"schema": 1, "status": "ready", "task_id": "dev-new",
                 "purpose": "development_selection", "created_utc": "2026-10-20T00:01:00+00:00"},
                {"schema": 1, "status": "ready", "task_id": "dev-old",
                 "purpose": "development_selection", "created_utc": "2026-10-20T00:00:30+00:00"},
            ]
            for index, manifest in enumerate(manifests):
                (task_dir / f"task-{index}.task.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual([task["task_id"] for _, task in worker.discover_tasks(durable)],
                             ["dev-old", "dev-new", "final-old"])

            final_first = (root / "final-first.task.json", {
                "task_id": "final-first", "purpose": "final_test", "dataset": "aloha"})
            final_second = (root / "final-second.task.json", {
                "task_id": "final-second", "purpose": "final_test", "dataset": "aloha"})
            dev_arrives = (root / "dev-arrives.task.json", {
                "task_id": "dev-arrives", "purpose": "development_selection", "dataset": "aloha"})
            arrivals = iter([
                [final_first, final_second],
                [dev_arrives, final_first, final_second],
                [dev_arrives, final_first, final_second],
                [dev_arrives, final_first, final_second],
            ])
            processed = []
            with patch.object(sys, "argv", ["runpod_eval_worker.py", "--durable-root", str(durable),
                                             "--work-root", str(root / "work"), "--once",
                                             "--retry-base-seconds", "1", "--retry-max-seconds", "1",
                                             "--aloha-gl", "egl"]), \
                 patch.object(worker, "discover_tasks", side_effect=lambda _root: next(arrivals)), \
                 patch.object(worker, "task_status_path", return_value=root / "status.json"), \
                 patch.object(worker, "retry_delay", return_value=(0.0, 0)), \
                 patch.object(worker, "published_result_matches", return_value=True), \
                 patch.object(worker, "evaluate_one", side_effect=lambda _path, task, *_args: processed.append((task["task_id"], _args[-1])) or True):
                self.assertEqual(worker.main(), 0)
            self.assertEqual(processed, [("final-first", "egl"), ("dev-arrives", "egl"),
                                         ("final-second", "egl")])


if __name__ == "__main__":
    unittest.main()
