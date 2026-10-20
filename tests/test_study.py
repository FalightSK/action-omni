"""Exercise study matching, result provenance and uninstrumented trajectories."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiment_protocol import file_digest, sampling_seed, write_json
from scripts.run_study import study_jobs, verify_result, wait_for_battery


class StudyTests(unittest.TestCase):
    def test_battery_gate_waits_for_70_only_after_charge_falls_below_35(self):
        state = {"status": "running"}
        saves = []
        with patch("scripts.run_study.battery_percent", side_effect=[34, 69, 70]), \
                patch("scripts.run_study.time.sleep") as sleep:
            wait_for_battery("evaluation", state, lambda: saves.append(dict(state)))
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(state["battery_percent"], 70)
        self.assertEqual(state["status"], "running")
        self.assertNotIn("waiting_for", state)
        self.assertIn("battery_wait", [row.get("status") for row in saves])

        with patch("scripts.run_study.battery_percent", return_value=35), \
                patch("scripts.run_study.time.sleep") as sleep:
            wait_for_battery("training", state, lambda: None)
        sleep.assert_not_called()

    def jobs(self, directory):
        plan = json.loads(Path("configs/camera_ready_study.json").read_text())
        def protocol(_path, dataset):
            return {"dataset": dataset, "protocol_id": "synthetic", "normalization": {}}
        with patch("scripts.run_study.load_protocol", side_effect=protocol):
            jobs = study_jobs(plan, Path(directory), ["aloha", "libero"], "libero-python")
        return plan, jobs

    def test_matrix_is_matched_fixed_budget_with_independent_seeds(self):
        plan, jobs = self.jobs("study")
        self.assertEqual(len(jobs), 12)
        self.assertEqual({j["training_seed"] for j in jobs}, {3, 4, 5})
        for j in jobs:
            self.assertEqual(j["configuration"]["num_epochs"], j["epochs"])
            self.assertEqual(j["configuration"]["min_epochs"], j["epochs"])
            self.assertTrue(j["checkpoint"].endswith("final.pt"))
            self.assertNotIn("--trace", j["evaluate"])
        changed = deepcopy(plan)
        changed["training_seeds"] = [3, 3, 4]
        with self.assertRaises(ValueError):
            study_jobs(changed, Path("study"), ["aloha"])
        changed = deepcopy(plan)
        changed["checkpoint_rule"] = "choose_best_test_score"
        with self.assertRaises(ValueError):
            study_jobs(changed, Path("study"), ["aloha"])

    def test_test_records_reject_changed_cohort_seed_configuration_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, jobs = self.jobs(directory)
            j = jobs[0]
            run = Path(j["run"])
            checkpoint = Path(j["checkpoint"])
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"locked checkpoint")
            sha = file_digest(checkpoint)
            write_json(run / "selected_policy.json", {"rule": "fixed_endpoint_final", "checkpoint_sha256": sha,
                       "checkpoint_epoch": 100, "training_seed": 3})
            write_json(run / "run_manifest.json", {"status": "complete", "completed_epoch": 100,
                       "run_signature": j["run_signature"], "cache_id": "cache"})
            result = {"dataset": "aloha", "exp_id": "exp05", "training_seed": 3,
                      "protocol_id": j["protocol_id"], "run_signature": j["run_signature"], "cache_id": "cache",
                      "checkpoint_epoch": 100, "checkpoint_sha256": sha, "configuration": j["configuration"],
                      "deterministic_inference": True, "cublas_workspace_config": ":4096:8",
                      "render_samples": 0,
                      "n_episodes": 200, "success_rate": .5, "action_horizon": 16, "inference_horizon": 8,
                      "episodes": [{"initial_state_id": i, "environment_seed": i*137+42,
                                    "policy_sampling_seed": sampling_seed("aloha", 3, 0, i),
                                    "is_success": i % 2 == 0} for i in range(10000, 10200)]}
            write_json(j["output"], result)
            self.assertEqual(verify_result(j, plan["datasets"]["aloha"])["successes"], 100)
            resolved = deepcopy(result)
            resolved["configuration"]["image_token_id"] = 151655
            write_json(j["output"], resolved)
            self.assertEqual(verify_result(j, plan["datasets"]["aloha"])["successes"], 100)
            for bad in ("cohort", "sampling", "configuration", "checkpoint", "renderer"):
                with self.subTest(corruption=bad):
                    r = deepcopy(result)
                    if bad == "cohort":
                        r["episodes"][0]["initial_state_id"] = 0
                    elif bad == "sampling":
                        r["episodes"][0]["policy_sampling_seed"] += 1
                    elif bad == "configuration":
                        r["configuration"]["min_epochs"] = 0
                    elif bad == "checkpoint":
                        r["checkpoint_sha256"] = "another checkpoint"
                    else:
                        r["render_samples"] = 4
                    write_json(j["output"], r)
                    with self.assertRaises(ValueError):
                        verify_result(j, plan["datasets"]["aloha"])

    def test_trajectory_capture_does_not_request_attention(self):
        from scripts.evaluate import _run_aloha
        cfg = SimpleNamespace(output_dir="unused", action_horizon=16, inference_horizon=8,
                              env_id="test", sim_max_steps=400)
        record = {"initial_state_id": 4000, "is_success": False,
                  "_trace": {"reward": np.array([1, 2]), "attn_image_mass": np.array([])}}
        with tempfile.TemporaryDirectory() as directory:
            cfg.output_dir = directory
            trace = Path(directory) / "trajectory.npz"
            with patch("envs.aloha_env.AlohaAgent") as agent, patch("envs.aloha_env.run_episode", return_value=record):
                with patch("gymnasium.make") as make:
                    rows = _run_aloha(cfg, None, None, "cpu", 1, False, Path(directory)/"diagnostic.json",
                                      seed_offset=4000, trace_path=trace, training_seed=0, record_attention=False)
            self.assertFalse(agent.call_args.kwargs["record_attn"])
            make.return_value.close.assert_called_once()
            self.assertNotIn("_trace", rows[0])
            with np.load(trace, allow_pickle=False) as saved:
                self.assertEqual(saved["initial_state_ids"].tolist(), [4000])
                self.assertEqual(saved["ep0000_reward"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
