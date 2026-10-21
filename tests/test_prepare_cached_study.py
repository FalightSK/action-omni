"""Fast cache-staging integrity checks; no VLM or dataset is loaded."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
if not hasattr(torch, "cuda"):
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

from scripts.prepare_cached_study import _copy_with_progress, _read_published, cache_paths


class PrepareCachedStudyTests(unittest.TestCase):
    def test_cache_paths_match_success_runner_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            durable_root = Path(tmp)
            local, durable = cache_paths(durable_root, "study_1", "aloha", "exp05")
            self.assertTrue(local.as_posix().endswith(
                "/asset/cache/cached_success_study/study_1/aloha/exp05/vlm_embeddings.h5"))
            self.assertEqual(
                durable,
                (durable_root / "success_studies/study_1/feature_caches/aloha/exp05"
                 / "vlm_embeddings.h5").resolve())
            with self.assertRaises(ValueError):
                cache_paths(durable_root, "../other", "aloha", "exp05")

    def test_published_cache_reader_checks_identity_size_and_sha(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            cache = Path(tmp) / "vlm_embeddings.h5"
            cache.write_bytes(b"complete cache payload")
            sha = hashlib.sha256(cache.read_bytes()).hexdigest()
            expected = {"schema": 1, "study_id": "study_1",
                        "size_bytes": cache.stat().st_size}
            manifest = dict(expected, sha256=sha)
            cache.with_name(cache.name + ".manifest.json").write_text(json.dumps(manifest))

            result = _read_published(cache, expected, role="test")
            self.assertEqual(result, (manifest, sha))

            cache.write_bytes(b"partial")
            self.assertIsNone(_read_published(cache, expected, role="test"))

    def test_copy_helper_verifies_published_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            source = root / "source.h5"
            target = root / "nested" / "cache.h5"
            source.write_bytes(b"cache contents" * 100)
            expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(_copy_with_progress(source, target, role="test"), expected_sha)
            self.assertEqual(target.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
