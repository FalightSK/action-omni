"""Focused contract checks for the shared rollout queue."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import types

# Queue contract checks replace the checkpoint loader and do not use tensor
# operations. Keep these focused tests runnable on lightweight CPU-only hosts.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from experiment_protocol import digest, file_digest, sampling_seed, write_json
from scripts.success_queue import publish_rollout_task, read_rollout_result, validate_rollout_result, wait_for_rollout


class SuccessQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)
        self.durable = self.root / "durable"
        self.durable.mkdir()
        self.source = self.root / "selection.pt"
        self.source.write_bytes(b"stable selection checkpoint")
        self.assets = {"asset/models/test/config.json": "1" * 64}
        self.provenance = {"encoder_code_id": "online-v1",
                           "online_encoder_code_sha256": "2" * 64,
                           "source_assets": self.assets}

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, dataset="aloha", seed=3):
        protocol_payload = {"version": 1, "dataset": dataset}
        protocol = {**protocol_payload, "protocol_id": digest(protocol_payload)}
        cfg = SimpleNamespace(vlm_backbone="test-backbone")
        return {
            "cfg": cfg, "dataset": dataset, "exp_id": "exp05",
            "training_seed": seed, "protocol_id": protocol["protocol_id"],
            "run_signature": f"signature-{dataset}-{seed}", "protocol": protocol,
            "feature_mode": "online", "optimizer_step": 14000,
            "state_dict": {"weight": object()}, "encoder_provenance": self.provenance,
            "validation_metrics": {"protocol_id": protocol["protocol_id"], "split": "validation_demonstrations"},
        }

    def publish(self, dataset="aloha", evaluation=None, purpose="development_selection",
                seed=3, extra=None):
        checkpoint = self.checkpoint(dataset, seed)
        metadata = {"dataset": dataset, "experiment": "exp05", "backbone": "test-backbone",
                    "training_seed": seed, "protocol_id": checkpoint["protocol_id"],
                    "run_signature": checkpoint["run_signature"], "feature_mode": "online",
                    "source_assets": self.assets, "optimizer_step": 14000,
                    "checkpoint_tag": "selection_epoch_000100", "selection_epoch": 100}
        if purpose == "final_test":
            metadata.update(checkpoint_tag="final_test", selected_checkpoint_sha256=file_digest(self.source))
        metadata.update(extra or {})
        spec = evaluation or ({"episodes": 2, "seed_offset": 20000, "capture_trajectory": False}
                              if dataset == "aloha" else
                              {"episodes_per_task": 2, "initial_state_offset": 0,
                               "tasks": [0, 1], "conditions": ["canonical"], "environment_seed": 0})
        with patch("scripts.success_queue._checkpoint_dict", return_value=checkpoint):
            return publish_rollout_task(SimpleNamespace(vlm_backbone="test-backbone"), metadata,
                                        self.source, spec, purpose, "a" * 64, "study-unit", self.durable)

    def encoder_provenance(self):
        return self.provenance

    def aloha_result(self, task, *, with_state=True):
        rows = []
        for state_id in range(task["evaluation"]["seed_offset"],
                              task["evaluation"]["seed_offset"] + task["evaluation"]["episodes"]):
            row = {"initial_state_id": state_id, "initial_state_kind": "gym_aloha_mujoco_state_v1",
                   "environment_seed": state_id * 137 + 42,
                   "policy_sampling_seed": sampling_seed("aloha", task["training_seed"], 0, state_id),
                   "is_success": state_id % 2 == 0}
            if with_state:
                row["initial_state_sha256"] = "a" * 64
            rows.append(row)
        result = self.identity_result(task)
        result.update(episodes=rows, seed_offset=task["evaluation"]["seed_offset"],
                      success_rate=sum(row["is_success"] for row in rows) / len(rows))
        return result

    def identity_result(self, task):
        result = {
            "task_id": task["task_id"], "study_id": task["study_id"], "plan_id": task["plan_id"],
            "dataset": task["dataset"], "exp_id": task["experiment"], "experiment": task["experiment"],
            "backbone": task["backbone"], "training_seed": task["training_seed"],
            "checkpoint_tag": task["checkpoint_tag"], "purpose": task["purpose"],
            "is_final": task["is_final"], "optimizer_step": task["optimizer_step"],
            "protocol_id": task["protocol_id"], "run_signature": task["run_signature"],
            "feature_mode": task["feature_mode"], "checkpoint_sha256": task["checkpoint"]["sha256"],
            "validation_metrics_sha256": task.get("validation_metrics_sha256"),
            "selection_epoch": task.get("selection_epoch"),
            "selected_checkpoint_sha256": task.get("selected_checkpoint_sha256"),
            "validation_metrics": task.get("validation_metrics"),
            "encoder_provenance": self.encoder_provenance(),
        }
        return result

    def test_publication_is_idempotent_and_rejects_changed_settings(self):
        task = self.publish()
        task_path = self.durable / "evaluation_tasks" / "study-unit" / ("a" * 12) / "aloha" / "exp05" / "seed_003" / "selection_epoch_000100.task.json"
        self.assertTrue(task_path.is_file())
        self.assertEqual(task["queue_priority"], 0)
        self.assertEqual(self.publish()["task_id"], task["task_id"])
        with self.assertRaisesRegex(ValueError, "different checkpoint/settings"):
            self.publish(evaluation={"episodes": 3, "seed_offset": 20000,
                                     "capture_trajectory": False})

    def test_aloha_result_requires_real_initial_state_fingerprint(self):
        task = self.publish()
        with self.assertRaisesRegex(ValueError, "real simulator-state fingerprints"):
            validate_rollout_result(task, self.aloha_result(task, with_state=False))
        validated = validate_rollout_result(task, self.aloha_result(task))
        self.assertEqual(validated["success_rate"], 0.5)

    def test_libero_expands_state_offset_and_validates_balanced_task_rate(self):
        task = self.publish(dataset="libero")
        self.assertEqual(task["evaluation"]["initial_state_ids"], [0, 1])
        result = self.identity_result(task)
        rows = []
        for task_id in (0, 1):
            for state_id in (0, 1):
                rows.append({"task": task_id, "condition": "canonical", "initial_state_id": state_id,
                             "success": task_id == state_id,
                             "initial_state_kind": "libero_mujoco_state_after_init_and_settle_v1",
                             "initial_state_sha256": "b" * 64,
                             "environment_seed": state_id,
                             "policy_sampling_seed": sampling_seed("libero", task["training_seed"],
                                                                   task_id, state_id)})
        result.update(results=rows, success_rate=0.5)
        self.assertEqual(validate_rollout_result(task, result)["success_rate"], 0.5)

    def test_wait_reads_only_the_published_exact_result(self):
        task = self.publish()
        result_path = self.durable / task["result_path"]
        write_json(result_path, self.aloha_result(task))
        result = wait_for_rollout(task, self.durable, timeout_s=1, poll_s=0.01)
        self.assertEqual(result["task_id"], task["task_id"])
        bad = self.aloha_result(task)
        bad["checkpoint_sha256"] = "f" * 64
        write_json(result_path, bad)
        with self.assertRaises(TimeoutError):
            wait_for_rollout(task, self.durable, timeout_s=0.02, poll_s=0.005)

    def test_nonblocking_result_read_distinguishes_pending_from_invalid(self):
        task = self.publish()
        self.assertIsNone(read_rollout_result(task, self.durable))
        result = self.aloha_result(task)
        result_path = self.durable / task["result_path"]
        write_json(result_path, result)
        self.assertEqual(read_rollout_result(task, self.durable)["success_rate"], 0.5)
        result["checkpoint_sha256"] = "f" * 64
        write_json(result_path, result)
        with self.assertRaises(ValueError):
            read_rollout_result(task, self.durable)


if __name__ == "__main__":
    unittest.main()
