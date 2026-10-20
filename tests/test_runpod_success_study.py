"""Concurrency checks for the RunPod success-stopped study runner."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
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
    sys.modules["torch"] = torch
if not hasattr(torch, "cuda"):
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

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
            "feature_mode": "online", "cache_id": None, "cache_sha256": None,
            "source_assets": {"fixture": "sha256"},
            "settings": {"final_evaluation": {}}}


class RunpodSuccessStudyTests(unittest.TestCase):
    def test_sync_tree_skips_durable_recovery_backups(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            source = root / "durable-run"
            destination = root / "local-run"
            source.mkdir()
            (source / "last.pt").write_bytes(b"mutable checkpoint")
            (source / "history.jsonl").write_text('{"epoch": 4}\n', encoding="utf-8")
            backup = source / ("recovery_e000004_r00000002_" + "a" * 64 + ".pt")
            backup.write_bytes(b"large immutable recovery backup")

            copied = study.sync_tree(source, destination)

            self.assertEqual(copied, 2)
            self.assertTrue((destination / "last.pt").is_file())
            self.assertTrue((destination / "history.jsonl").is_file())
            self.assertFalse((destination / backup.name).exists())

    def test_build_jobs_signs_resolved_backbone_image_token(self):
        @dataclass
        class Config:
            vlm_backbone: str
            output_dir: str
            image_token_id: int = 248056
            n_views: int = 1
            batch_size: int = 1
            num_epochs: int = 1
            min_epochs: int = 0
            warmup_steps: int = 0
            min_lr_ratio: float = 0.0
            lr_decay_epochs: int | None = 1
            embeddings_cache: str = ""

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(path):
                token_id = 151655 if Path(path).name == "cosmos_reason2_2b" else 151656
                return types.SimpleNamespace(image_token_id=token_id)

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(path, **_kwargs):
                return types.SimpleNamespace(source=path)

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            model_root = root / "models"
            (model_root / "cosmos_reason2_2b").mkdir(parents=True)
            (model_root / "qwen3_vl_2b").mkdir()
            fake_backbones = types.ModuleType("models.backbones")
            fake_backbones.MODELS = model_root
            fake_backbones._img_token_id = lambda config, _processor: config.image_token_id
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoConfig = FakeAutoConfig
            fake_transformers.AutoProcessor = FakeAutoProcessor

            plan = {
                "schema": 2,
                "checkpoint_rule": "highest_development_success_rate_earliest_tie",
                "feature_mode": "online",
                "training_seeds": [3, 4, 5],
                "datasets": {"aloha": {"reference_epochs": 2, "steps_per_epoch": 140,
                                         "split_manifest": "configs/aloha/split.json",
                                         "camera_views": ["top"]}},
                "stopping": {"maximum_epochs": None, "minimum_epochs": 1},
                "backbones": {"groot": "exp05", "qwen3vl": "exp06"},
                "effective_batch_size": 128,
                "warmup_updates": 100,
                "minimum_learning_rate_ratio": 0.02,
                "reference_optimizer_updates": 280,
            }
            protocol = {"protocol_id": "protocol"}

            def get_fake_config(_dataset, experiment):
                backbone = "groot" if experiment == "exp05" else "qwen3vl"
                return Config(backbone, f"asset/runs/aloha/{experiment}")

            with patch.dict(sys.modules, {"models.backbones": fake_backbones,
                                          "transformers": fake_transformers}), \
                 patch.object(study, "get_config", side_effect=get_fake_config), \
                 patch.object(study, "apply_normalization"), \
                 patch.object(study, "load_protocol", return_value=protocol), \
                 patch.object(study, "source_assets", return_value={}), \
                 patch.object(study, "run_signature",
                              side_effect=lambda cfg, *_args: str(cfg.image_token_id)):
                jobs = study.build_jobs(plan, "aloha", root / "runs", 1,
                                        root / "plan.json", root / "durable")

            self.assertEqual(len(jobs), 6)
            for job in jobs:
                expected_id = 151655 if job["backbone"] == "groot" else 151656
                self.assertEqual(job["run_signature"], str(expected_id))
                self.assertEqual(job["config"]["image_token_id"], expected_id)

    def test_cached_jobs_use_locked_cache_without_online_encoder(self):
        @dataclass
        class Config:
            vlm_backbone: str
            output_dir: str
            image_token_id: int = 248056
            n_views: int = 1
            batch_size: int = 1
            num_epochs: int = 1
            min_epochs: int = 0
            warmup_steps: int = 0
            min_lr_ratio: float = 0.0
            lr_decay_epochs: int | None = 1
            embeddings_cache: str = ""
            vlm_extract_layers: tuple = (16,)

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(path):
                token_id = 151655 if Path(path).name == "cosmos_reason2_2b" else 151656
                return types.SimpleNamespace(image_token_id=token_id)

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(path, **_kwargs):
                return types.SimpleNamespace(source=path)

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            model_root = root / "models"
            (model_root / "cosmos_reason2_2b").mkdir(parents=True)
            (model_root / "qwen3_vl_2b").mkdir()
            fake_backbones = types.ModuleType("models.backbones")
            fake_backbones.MODELS = model_root
            fake_backbones._img_token_id = lambda config, _processor: config.image_token_id
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoConfig = FakeAutoConfig
            fake_transformers.AutoProcessor = FakeAutoProcessor
            plan = {
                "schema": 2,
                "checkpoint_rule": "highest_development_success_rate_earliest_tie",
                "feature_mode": "cached", "training_seeds": [3, 4, 5],
                "datasets": {"aloha": {"reference_epochs": 2, "steps_per_epoch": 140,
                                         "split_manifest": "configs/aloha/split.json",
                                         "camera_views": ["top"]}},
                "stopping": {"maximum_epochs": None, "minimum_epochs": 1},
                "backbones": {"groot": "exp05", "qwen3vl": "exp06"},
                "effective_batch_size": 128, "warmup_updates": 100,
                "minimum_learning_rate_ratio": 0.02, "reference_optimizer_updates": 280,
            }
            protocol = {"protocol_id": "protocol"}
            get_fake_config = lambda _dataset, experiment: Config(
                "groot" if experiment == "exp05" else "qwen3vl",
                f"asset/runs/aloha/{experiment}")
            with patch.dict(sys.modules, {"models.backbones": fake_backbones,
                                          "transformers": fake_transformers}), \
                 patch.object(study, "get_config", side_effect=get_fake_config), \
                 patch.object(study, "apply_normalization"), \
                 patch.object(study, "load_protocol", return_value=protocol), \
                 patch.object(study, "source_assets", return_value={"weights": "sha"}), \
                 patch.object(study, "validate_cached_cache",
                              return_value={"cache_id": "cache-id", "sha256": "cache-sha"}) as validate, \
                 patch.object(study, "run_signature",
                              side_effect=lambda cfg, *_args: str(cfg.image_token_id)):
                jobs = study.build_jobs(plan, "aloha", root / "study-cache", 1,
                                        root / "plan.json", root / "durable")

            self.assertEqual(len(validate.call_args_list), 2)
            self.assertEqual(len(jobs), 6)
            for job in jobs:
                self.assertEqual(job["feature_mode"], "cached")
                self.assertEqual(job["cache_id"], "cache-id")
                self.assertEqual(job["cache_sha256"], "cache-sha")
                self.assertIn("--cache", job["command"])
                self.assertNotIn("--online-encoder", job["command"])
                cache_index = job["command"].index("--cache")
                self.assertEqual(job["command"][cache_index + 1], job["cache"])

    def test_cached_cache_manifest_and_hdf5_provenance_are_checked(self):
        import numpy as np

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            cache_path = root / "vlm_embeddings.h5"
            durable_path = root / "cloud" / "vlm_embeddings.h5"
            cache_path.write_bytes(b"fixture-cache")
            source_assets = {"asset/models/mock.safetensors": "weights-sha"}
            cache_cfg = {"vlm_backbone": "groot"}
            protocol = {"protocol_id": "protocol-sha", "n_samples": 3}
            cfg = types.SimpleNamespace(vlm_extract_layers=(16,), img_seq_len=3,
                                        vlm_hidden_size=2, state_dim=14,
                                        action_horizon=2, action_dim=14)
            manifest = {
                "schema": 1, "dataset": "aloha", "experiment": "exp05",
                "backbone": "groot", "study_id": "study-v6", "plan_id": "plan-sha",
                "protocol_id": "protocol-sha", "cache_id": "cache-id",
                "encoder_code_id": "encoder-id", "source_assets": source_assets,
                "size_bytes": cache_path.stat().st_size, "sha256": "cache-sha",
                "local_path": str(cache_path.resolve()),
                "durable_path": str(durable_path.resolve()),
            }
            cache_path.with_name(cache_path.name + ".manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            attrs = {
                "complete": True, "dataset": "aloha", "exp_id": "exp05",
                "protocol_id": "protocol-sha", "cache_id": "cache-id",
                "encoder_code_id": "encoder-id",
                "source_assets": json.dumps(source_assets),
                "cache_config": json.dumps(cache_cfg), "full_dataset": True,
                "vlm_extract_layers": [16], "n_samples": 3,
                "batch_one_parity": json.dumps({"rows": [0, 2], "masks_equal": True}),
            }
            datasets = {
                "embeddings": np.zeros((3, 3, 2)), "img_masks": np.zeros((3, 3)),
                "states": np.zeros((3, 14)), "actions": np.zeros((3, 2, 14)),
                "episode_ids": np.array([10, 11, 12]),
                "validation_mask": np.array([1, 0, 0], dtype=np.uint8),
            }

            fake_h5py = types.ModuleType("h5py")
            class FakeContext:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def keys(self):
                    return datasets.keys()

                def __getitem__(self, key):
                    return datasets[key]

                @property
                def attrs(self):
                    return attrs

            fake_h5py.File = lambda *_args, **_kwargs: FakeContext()
            with patch.dict(sys.modules, {"h5py": fake_h5py}), \
                 patch.object(study, "encoder_code_id", return_value="encoder-id"), \
                 patch.object(study, "cache_config", return_value=cache_cfg), \
                 patch.object(study, "protocol_arrays", return_value=(
                     np.array([10, 11, 12]), np.array([True, False, False]))), \
                 patch.object(study, "file_digest", return_value="cache-sha"):
                result = study.validate_cached_cache(
                    cache_path, durable_path, dataset="aloha", experiment="exp05",
                    backbone="groot", study_id="study-v6", plan_id="plan-sha",
                    protocol=protocol, cfg=cfg, locked_source_assets=source_assets)
            self.assertEqual(result["cache_id"], "cache-id")
            self.assertEqual(result["source_assets"], source_assets)

            manifest["plan_id"] = "wrong-plan"
            cache_path.with_name(cache_path.name + ".manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with patch.object(study, "encoder_code_id", return_value="encoder-id"), \
                 patch.object(study, "file_digest", return_value="cache-sha"):
                with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    study.validate_cached_cache(
                        cache_path, durable_path, dataset="aloha", experiment="exp05",
                        backbone="groot", study_id="study-v6", plan_id="plan-sha",
                        protocol=protocol, cfg=cfg, locked_source_assets=source_assets)

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

    def test_cli_defaults_to_two_concurrent_policies(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps({"name": "test"}), encoding="utf-8")
            jobs = [make_job(root, 0)]
            argv = ["--dataset", "aloha", "--plan", str(plan_path), "--run-root",
                    str(root / "run"), "--durable-root", str(root / "durable"), "--dry-run"]
            with patch.object(study, "build_jobs", return_value=jobs):
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    self.assertEqual(study.main(argv), 0)
            self.assertIn('"parallel_policies": 2', output.getvalue())


if __name__ == "__main__":
    unittest.main()
