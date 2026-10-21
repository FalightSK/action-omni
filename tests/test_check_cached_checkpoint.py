"""CPU guard tests; these do not claim real training or Cloud verification."""
import copy
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from scripts.check_cached_checkpoint import validate_one_step_checkpoint


class TensorFixture:
    pass


class CheckCachedCheckpointTests(unittest.TestCase):
    def setUp(self):
        checker = patch("scripts.check_cached_checkpoint.torch.is_tensor", create=True,
                        side_effect=lambda value: isinstance(value, TensorFixture))
        checker.start()
        self.addCleanup(checker.stop)

    def fixture(self):
        return {"dataset": "aloha", "exp_id": "exp05", "training_seed": 123,
                "feature_mode": "cached", "protocol_id": "protocol", "cache_id": "cache",
                "epoch": 1, "global_step": 1, "batch_size": 128,
                "steps_per_epoch": 1, "max_training_steps": 1,
                "state_dict": {"adapter.weight": TensorFixture(), "decoder.weight": TensorFixture()},
                "optim": {"param_groups": [{}, {}], "state": {
                    0: {"step": 1, "exp_avg": TensorFixture(), "exp_avg_sq": TensorFixture()}}},
                "scheduler": {"last_epoch": 1},
                "rng_state": {"python": [1], "numpy": [1], "torch": TensorFixture(),
                              "cuda": [TensorFixture()]},
                "train_generator": TensorFixture(), "val_generator": TensorFixture()}

    def test_full_one_update_state_is_required(self):
        valid = self.fixture()
        validate_one_step_checkpoint(valid, "protocol", "cache")
        for key in ("state_dict", "optim", "scheduler", "rng_state", "train_generator", "val_generator"):
            changed = copy.deepcopy(valid)
            changed.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_one_step_checkpoint(changed, "protocol", "cache")

    def test_libero_checkpoint_uses_its_own_dataset_identity(self):
        checkpoint = self.fixture()
        checkpoint["dataset"] = "libero"
        validate_one_step_checkpoint(checkpoint, "protocol", "cache", "libero")
        with self.assertRaises(ValueError):
            validate_one_step_checkpoint(checkpoint, "protocol", "cache", "aloha")

    def test_wrong_identity_or_multiple_optimizer_steps_are_rejected(self):
        for key, value in (("global_step", 2), ("epoch", 2), ("training_seed", 3),
                           ("cache_id", "other"), ("protocol_id", "other")):
            changed = self.fixture()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_one_step_checkpoint(changed, "protocol", "cache")
        changed = self.fixture()
        changed["optim"]["state"][0]["step"] = 2
        with self.assertRaises(ValueError):
            validate_one_step_checkpoint(changed, "protocol", "cache")


if __name__ == "__main__":
    unittest.main()
