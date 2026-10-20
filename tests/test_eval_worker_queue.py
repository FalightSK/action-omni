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

    def test_partial_task_manifest_is_retried_after_it_becomes_readable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            durable = Path(tmp) / "durable"
            task_dir = durable / "evaluation_tasks"
            task_dir.mkdir(parents=True)
            path = task_dir / "task.task.json"
            path.write_text('{"schema":', encoding="utf-8")
            self.assertEqual(worker.discover_tasks(durable), [])
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(worker.discover_tasks(durable), [])
            path.write_text(json.dumps({"schema": 1, "status": "ready", "task_id": "retry-me",
                                        "purpose": "development_selection",
                                        "created_utc": "2026-10-20T00:00:00+00:00"}), encoding="utf-8")
            self.assertEqual([task["task_id"] for _, task in worker.discover_tasks(durable)], ["retry-me"])

    def test_visible_task_with_checkpoint_not_yet_available_is_pending(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            durable = Path(tmp) / "durable"
            durable.mkdir()
            task = {"dataset": "aloha", "experiment": "exp05", "backbone": "test",
                    "training_seed": 3, "protocol_id": "p", "run_signature": "s",
                    "feature_mode": "online", "checkpoint_tag": "selection_epoch_000100",
                    "optimizer_step": 14000,
                    "checkpoint": {"path": "checkpoints/selection.pt", "size_bytes": 12,
                                   "sha256": "a" * 64, "optimizer_step": 14000}}
            with self.assertRaises(worker.SharedArtifactPending):
                worker.validate_task_checkpoint(task, durable)

    def test_shared_artifact_copy_verifies_the_published_destination(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            source = root / "local-result.json"
            destination = root / "durable" / "result.json"
            source.write_text('{"success_rate": 1.0}\n', encoding="utf-8")
            size, sha256 = worker.copy_file_atomic(source, destination)
            self.assertEqual(destination.stat().st_size, size)
            self.assertEqual(worker.file_digest(destination), sha256)

    def test_dispatch_exception_does_not_permanently_suppress_task(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            durable = root / "durable"
            durable.mkdir()
            entry = (root / "task.task.json", {"task_id": "retry-task", "purpose": "development_selection",
                                                 "dataset": "aloha", "created_utc": "2026-10-20T00:00:00+00:00"})
            arrivals = iter([[entry], [entry], []])
            calls = []

            def evaluate(*_args):
                calls.append(1)
                if len(calls) == 1:
                    raise worker.SharedArtifactPending("temporarily unavailable shared checkpoint")
                return True

            with patch.object(sys, "argv", ["runpod_eval_worker.py", "--durable-root", str(durable),
                                             "--work-root", str(root / "work"), "--once",
                                             "--retry-base-seconds", "1", "--retry-max-seconds", "1",
                                             "--aloha-gl", "egl"]), \
                 patch.object(worker, "discover_tasks", side_effect=lambda _root: next(arrivals)), \
                 patch.object(worker, "retry_delay", return_value=(0.0, 0)), \
                 patch.object(worker, "published_result_matches", return_value=True), \
                 patch.object(worker, "evaluate_one", side_effect=evaluate):
                self.assertEqual(worker.main(), 0)
            self.assertEqual(len(calls), 2)

    def test_completed_but_not_yet_visible_result_is_pending_not_re_evaluated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            durable = root / "durable"
            durable.mkdir()
            entry = (root / "task.task.json", {"task_id": "completed-task", "purpose": "final_test",
                                                 "dataset": "aloha", "created_utc": "2026-10-20T00:00:00+00:00"})
            status = root / "status.json"
            status.write_text(json.dumps({"task_id": "completed-task", "status": "complete"}),
                              encoding="utf-8")
            arrivals = iter([[entry], []])
            with patch.object(sys, "argv", ["runpod_eval_worker.py", "--durable-root", str(durable),
                                             "--work-root", str(root / "work"), "--once",
                                             "--retry-base-seconds", "1", "--retry-max-seconds", "1",
                                             "--aloha-gl", "egl"]), \
                 patch.object(worker, "discover_tasks", side_effect=lambda _root: next(arrivals)), \
                 patch.object(worker, "task_status_path", return_value=status), \
                 patch.object(worker, "published_result_matches", return_value=False), \
                 patch.object(worker, "evaluate_one") as evaluate:
                self.assertEqual(worker.main(), 0)
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
