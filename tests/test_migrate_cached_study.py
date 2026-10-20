"""Migration guard checks without loading GPU models or checkpoint files."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from scripts.migrate_cached_study import (
    self_check_cache_prepare_report_identity, validate_training_source_delta,
)


class CachedMigrationGuardTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
