"""Run with python -m unittest discover -s tests -v; no GPU/data required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from configs.base_config import BaseVLAConfig
from data.hdf5_embeddings import HDF5EmbeddingWriter
from experiment_protocol import cache_config, encoder_code_id, protocol_arrays
from scripts.run_experiment import ensure_cache, exclusive_run, validate_cache


class RunnerTests(unittest.TestCase):
    def test_cache_publication_reuse_and_interrupted_file_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "vlm_embeddings.h5"
            partial = cache.with_suffix(".partial.h5")
            partial.write_bytes(b"interrupted")
            calls = []

            def encode(path):
                calls.append(path)
                path.write_bytes(b"complete")

            with patch("scripts.run_experiment.validate_cache", return_value={"status": "passed"}):
                report, reused = ensure_cache(cache, None, None, None, encode)
                self.assertFalse(reused)
                self.assertEqual(report["cache"], str(cache))
                _, reused = ensure_cache(cache, None, None, None, encode)
                self.assertTrue(reused)
                self.assertEqual(len(calls), 1)
                self.assertEqual(next(cache.parent.glob("interrupted_*.h5")).read_bytes(), b"interrupted")
            with patch("scripts.run_experiment.validate_cache", side_effect=ValueError("bad cache")):
                with self.assertRaises(ValueError):
                    ensure_cache(cache, None, None, None, encode)
            self.assertEqual(cache.read_bytes(), b"complete")
            self.assertEqual(len(calls), 1)

    def test_failed_validation_never_publishes_cache_and_lock_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.h5"
            with patch("scripts.run_experiment.validate_cache", side_effect=ValueError("parity failed")):
                with self.assertRaises(ValueError):
                    ensure_cache(cache, None, None, None, lambda path: path.write_bytes(b"bad"))
            self.assertFalse(cache.exists())
            self.assertTrue(cache.with_suffix(".partial.h5").exists())
            with exclusive_run(cache.parent / ".encoding.lock"):
                with self.assertRaises(OSError):
                    with exclusive_run(cache.parent / ".encoding.lock"):
                        self.fail("second encoder acquired the lock")

    def test_real_cache_validation_rejects_missing_parity_bad_assets_and_nan(self):
        cfg = BaseVLAConfig(state_dim=2, action_dim=2, action_horizon=2,
                            img_seq_len=2, vlm_hidden_size=3, vlm_extract_layers=(28,))
        cfg.embed_batch_size = 2
        protocol = {"dataset": "libero", "n_samples": 4, "protocol_id": "test",
                    "episodes": [{"id": i, "start": i*2, "stop": i*2+2,
                                  "split": "train" if i == 0 else "validation"} for i in range(2)]}
        ids, val = protocol_arrays(protocol)
        assets = {"weights": "hash"}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.h5"
            writer = HDF5EmbeddingWriter(cache, 4, 2, 3, 2, 2, 2, attrs={
                "protocol_id": "test", "cache_id": "test-cache", "full_dataset": True,
                "cache_config": json.dumps(cache_config(cfg)), "encoder_code_id": encoder_code_id(),
                "source_assets": json.dumps(assets), "vlm_extract_layers": [28], "encoding_batch_size": 2})
            writer.f.create_dataset("episode_ids", data=ids)
            writer.f.create_dataset("validation_mask", data=val)
            writer.write(0, torch.zeros(4, 2, 3, dtype=torch.bfloat16), torch.ones(4, 2, dtype=torch.bool),
                         torch.zeros(4, 2), torch.zeros(4, 2, 2))
            writer.close()
            with self.assertRaises(ValueError):
                validate_cache(cache, cfg, protocol, assets)
            with h5py.File(cache, "r+") as f:
                f.attrs["batch_one_parity"] = json.dumps({"rows": [0, 1, 2, 3],
                    "max_absolute_difference": 0, "masks_equal": True})
            self.assertEqual(validate_cache(cache, cfg, protocol, assets)["status"], "passed")
            cfg.embed_batch_size = 32  # Training/cache reuse does not depend on encoding batch size.
            self.assertEqual(validate_cache(cache, cfg, protocol, assets)["status"], "passed")
            with self.assertRaises(ValueError):
                validate_cache(cache, cfg, protocol, {"weights": "different"})
            with h5py.File(cache, "r+") as f:
                f["embeddings"][0, 0, 0] = np.uint16(0x7FC0)
            with self.assertRaises(ValueError):
                validate_cache(cache, cfg, protocol, assets)


if __name__ == "__main__":
    unittest.main()
