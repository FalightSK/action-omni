"""Focused synthetic checks for RunPod paper-report cohort and seed handling."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_protocol import digest
from scripts.analysis import runpod_paper_results as paper


class SeedSummaryTests(unittest.TestCase):
    def test_fewer_than_three_seeds_has_missing_ids_and_no_sample_sd(self):
        summary = paper.summarize_independent_seeds([0.4, 0.6], [3, 4, 5], [3, 4])
        self.assertEqual(summary["n_training_seeds"], 2)
        self.assertEqual(summary["missing_training_seeds"], [5])
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["sample_sd"])
        self.assertIn("fewer than 3", summary["sample_sd_status"])

    def test_all_three_seeds_reports_sample_sd(self):
        summary = paper.summarize_independent_seeds([0.4, 0.5, 0.6], [3, 4, 5], [3, 4, 5])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["missing_training_seeds"], [])
        self.assertAlmostEqual(summary["sample_sd"], 0.1)

    def test_seed_values_must_map_one_to_one_to_planned_seeds(self):
        with self.assertRaises(ValueError):
            paper.summarize_independent_seeds([0.5, 0.6], [3, 4, 5], [3, 3])


class FinalReportCohortTests(unittest.TestCase):
    def test_repeated_development_checkpoints_are_not_rq2_policy_rows(self):
        plan = {
            "schema": 2,
            "checkpoint_rule": "highest_development_success_rate_earliest_tie",
            "training_seeds": [3, 4, 5],
            "backbones": {"groot": "exp05", "qwen3vl": "exp06"},
            "datasets": {"aloha": {}, "libero": {}},
            "scene_split": {"definition": "disjoint reset cohorts on same tasks"},
        }
        indexed, development = {}, []
        for dataset in ("aloha", "libero"):
            evaluation = ({"episodes": 2, "seed_offset": 10000} if dataset == "aloha" else
                          {"episodes_per_task": 2, "tasks": [0, 1]})
            for experiment in ("exp05", "exp06"):
                for seed in plan["training_seeds"]:
                    checkpoint_sha = f"{len(indexed) + 1:064x}"
                    validation_metrics = {"flow_matching_loss": 0.5, "action_mse": 0.25,
                                          "action_mae": 0.4, "per_dimension_action_mse": [0.25],
                                          "per_dimension_action_mae": [0.4], "n_samples": 16,
                                          "protocol_id": "protocol", "split": "validation_demonstrations",
                                          "validation_rng_seed": 4242, "action_sampling_seed_base": 9}
                    metrics_sha = digest(validation_metrics)
                    task = {"dataset": dataset, "experiment": experiment, "training_seed": seed,
                            "checkpoint_tag": "final_test", "purpose": "final_test", "is_final": True,
                            "selection_epoch": 100, "selected_checkpoint_sha256": checkpoint_sha,
                            "checkpoint": {"sha256": checkpoint_sha}, "evaluation": evaluation,
                            "protocol_id": "protocol", "validation_metrics": validation_metrics,
                            "validation_metrics_sha256": metrics_sha}
                    success = experiment == "exp05"
                    if dataset == "aloha":
                        result = {"episodes": [{"is_success": success}, {"is_success": False}]}
                    else:
                        result = {"results": [
                            {"task": task_id, "condition": "canonical", "success": success}
                            for task_id in evaluation["tasks"] for _ in range(2)]}
                    result.update({"success_rate": 0.5 if success else 0.0,
                                   "checkpoint_sha256": checkpoint_sha,
                                   "validation_metrics": validation_metrics,
                                   "validation_metrics_sha256": metrics_sha})
                    entry = {"dataset": dataset, "experiment": experiment, "backbone": experiment,
                             "training_seed": seed, "checkpoint_tag": "final_test", "purpose": "final_test",
                             "status": "complete", "task_id": f"final-{dataset}-{experiment}-{seed}",
                             "checkpoint_sha256": checkpoint_sha, "validation_metrics_sha256": metrics_sha,
                             "protocol_id": "protocol", "task": task, "result": result}
                    indexed[(dataset, experiment, seed, "final_test")] = entry
                    for epoch in (95, 100):
                        dev_sha = checkpoint_sha if epoch == 100 else f"{int(checkpoint_sha, 16) + 1000:064x}"
                        development.append({"dataset": dataset, "experiment": experiment,
                                            "training_seed": seed, "purpose": "development_selection",
                                            "selection_epoch": epoch, "status": "complete",
                                            "checkpoint_sha256": dev_sha,
                                            "success_rate": 0.9 if epoch == 100 else 0.7,
                                            "task_id": f"dev-{dataset}-{experiment}-{seed}-{epoch}"})

        fake_stats = {"left_success_rate": 0.5, "right_success_rate": 0.25,
                      "difference_pp": 25.0, "difference_bootstrap95_pp": [0.0, 50.0],
                      "mcnemar_exact_p": 0.5, "only_left": 1, "only_right": 0}
        metric_values = {"flow_matching_loss": 0.5, "action_mse": 0.25, "action_mae": 0.4,
                         "per_dimension_action_mse": [0.25], "per_dimension_action_mae": [0.4],
                         "n_samples": 16, "protocol_id": "protocol",
                         "split": "validation_demonstrations", "validation_rng_seed": 4242,
                         "action_sampling_seed_base": 9}
        # Use the repository's writable workspace; sandbox temp directories may
        # be read-only on managed Windows hosts.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with (patch.object(paper, "read_task_results", return_value=(indexed, development, [])),
                  patch.object(paper, "validate_rollout_result", side_effect=lambda task, result: dict(result)),
                  patch.object(paper, "valid_metrics", return_value=(metric_values, None)),
                  patch.object(paper, "compare_records", return_value=fake_stats)):
                report = paper.build_report(root, plan_path, root / "out")

        rows = report["rq2_offline_metrics_vs_success"]
        self.assertEqual(report["development_selection_task_count"], 24)
        self.assertEqual(report["rq2_final_policy_row_count"], 12)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["checkpoint_tag"] == "final_test" for row in rows))
        self.assertTrue(all(row["evaluation_cohort"] == "final_evaluation" for row in rows))
        self.assertTrue(all(row["selection_cohort_status"] == "complete" for row in rows))
        self.assertEqual(rows[0]["selection_cohort_success_rate"], 0.9)
        self.assertNotIn("pearson", json.dumps(report["rq2_offline_metrics_vs_success"]).lower())
        self.assertIn("not independent trained runs", report["interpretation"]["development_snapshots"])


if __name__ == "__main__":
    unittest.main()
