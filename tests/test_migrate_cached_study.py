"""Migration guard checks without loading GPU models or checkpoint files."""
import sys
import types
import unittest
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from scripts.migrate_cached_study import (
    cache_prepare_report_identity, self_check_cache_prepare_report_identity,
    ensure_local_v6_lock, select_verified_checkpoint,
    validate_cache_prepare_report, validate_training_source_delta,
)


class CachedMigrationGuardTests(unittest.TestCase):
    def test_missing_local_lock_is_restored_from_exact_durable_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            durable = root / "cloud" / "locked_success_study.json"
            local = root / "local" / "locked_success_study.json"
            contract = {"plan_id": "plan", "dataset_role": "aloha",
                        "jobs": [{"experiment": "exp05", "seed": 3}]}
            durable_bytes = (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()
            durable.parent.mkdir(parents=True)
            durable.write_bytes(durable_bytes)

            ensure_local_v6_lock(durable, local, contract)

            self.assertEqual(local.read_bytes(), durable_bytes)

    def test_mismatched_local_or_durable_lock_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            durable = root / "cloud" / "locked_success_study.json"
            local = root / "local" / "locked_success_study.json"
            contract = {"plan_id": "expected"}
            other = {"plan_id": "different"}
            durable.parent.mkdir(parents=True)
            durable.write_text(json.dumps(contract), encoding="utf-8")
            local.parent.mkdir(parents=True)
            local.write_text(json.dumps(other), encoding="utf-8")
            original_local = local.read_bytes()

            with self.assertRaises(FileExistsError):
                ensure_local_v6_lock(durable, local, contract)
            self.assertEqual(local.read_bytes(), original_local)

            local.unlink()
            durable.write_text(json.dumps(other), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_local_v6_lock(durable, local, contract)
            self.assertFalse(local.exists())

    def test_checkpoint_selector_uses_immutable_when_mutable_last_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / "last.pt"
            last.write_bytes(b"partial mutable checkpoint")
            immutable_payload = b"verified immutable checkpoint"
            immutable_sha = hashlib.sha256(immutable_payload).hexdigest()
            immutable = root / f"recovery_e000080_r00000012_{immutable_sha}.pt"
            immutable.write_bytes(immutable_payload)
            checkpoint = {"epoch": 80, "global_step": 11200,
                          "recovery_revision": 12}

            def load(path, **kwargs):
                if Path(path) == immutable:
                    return checkpoint
                raise OSError("simulated partial last.pt")

            with patch("scripts.migrate_cached_study.torch.load", side_effect=load,
                       create=True):
                selected_path, selected_checkpoint = select_verified_checkpoint(last)

            self.assertEqual(selected_path, immutable)
            self.assertEqual(selected_checkpoint, checkpoint)

    def test_exact_reviewed_trainer_transition_is_allowed(self):
        hashes = [
            "68e74ec905b7c77c6998fa8d3bc5842aad5c881fc450255f1c8e7858848622a4",
            "3eb9b31b4054b387c32d5e28c0aaea97afdbadc4475c355eefa760b1c93f0586",
        ]
        with patch("scripts.migrate_cached_study.hashlib.sha256",
                   side_effect=[Mock(hexdigest=lambda: hashes[0]),
                                Mock(hexdigest=lambda: hashes[1])]):
            result = validate_training_source_delta("scripts/train.py", b"old", b"new")
        self.assertEqual(result, dict(zip(("old_sha256", "new_sha256"), hashes)))

    def test_unreviewed_trainer_and_changed_encoder_are_rejected(self):
        for path in ("scripts/train.py", "models/vla.py", "models/vla_train.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_training_source_delta(path, b"old", b"unreviewed")
        validate_training_source_delta("models/vla.py", b"same", b"same")

    def test_cache_report_timing_changes_preserve_immutable_identity(self):
        self_check_cache_prepare_report_identity()

    def test_report_matches_actual_preparation_return_schema_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            caches, evidence, manifests = [], {}, {}
            for exp in ("exp05", "exp06"):
                cache = root / exp / "cache.h5"
                cache.parent.mkdir()
                cache.write_bytes(exp.encode())
                sha = hashlib.sha256(cache.read_bytes()).hexdigest()
                item = {"cache_id": exp, "local_path": str(cache),
                        "durable_path": str(cache), "sha256": sha,
                        "size_bytes": cache.stat().st_size, "reused": True}
                caches.append(item)
                evidence[exp] = dict(item, path=str(cache))
                manifests[exp] = dict(item)
            source = Path(__file__).resolve().parents[1] / "scripts/prepare_cached_study.py"
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for function in tree.body:
                if isinstance(function, ast.FunctionDef) and function.name in {
                        "_prepare_one", "_wait_reuse"}:
                    returns = [node.value for node in ast.walk(function)
                               if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
                    self.assertTrue(returns)
                    for value in returns:
                        self.assertEqual({ast.literal_eval(key) for key in value.keys}, set(caches[0]))
            report = {"schema": 1, "study_id": "study", "plan_id": "plan",
                      "dataset": "aloha", "feature_mode": "cached", "caches": caches,
                      "completed_utc": "2026-10-20T11:43:52+00:00", "elapsed_s": 1.0}
            path = root / "report.json"
            path.write_text(json.dumps(report))
            kwargs = dict(dataset="aloha", study_id="study", plan_id="plan",
                          experiments=set(evidence), cache_evidence=evidence,
                          manifest_entries=manifests)
            self.assertEqual(validate_cache_prepare_report(path, **kwargs),
                             cache_prepare_report_identity(report))
            report["caches"][0]["durable_path"] = str(root / "unknown")
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                validate_cache_prepare_report(path, **kwargs)


if __name__ == "__main__":
    unittest.main()
