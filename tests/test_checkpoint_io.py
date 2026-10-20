"""Checkpoint copies must not depend on shared-volume metadata operations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import checkpoint_io


class CheckpointIOTests(unittest.TestCase):
    def test_recovery_ignores_partial_newer_revision(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            source = root / "last.pt"
            source.write_bytes(b"complete optimizer state")
            committed = checkpoint_io.publish_recovery_revision(source, root / "cloud", 100, 1)
            (root / "cloud" / ("recovery_e000101_r00000002_" + "a" * 64 + ".pt")).write_bytes(b"partial")
            self.assertEqual(checkpoint_io.latest_verified_recovery(root / "cloud"), committed)
            self.assertEqual(checkpoint_io.publish_recovery_revision(source, root / "cloud", 100, 1), committed)

    def test_atomic_copy_succeeds_when_copystat_is_forbidden(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            source = root / "last.pt"
            destination = root / "durable" / "last.pt"
            source.write_bytes(b"checkpoint bytes")
            expected_sha = checkpoint_io.file_sha256(source)

            with patch.object(checkpoint_io.shutil, "copystat",
                              side_effect=PermissionError("utime is unsupported")):
                actual_sha = checkpoint_io.atomic_copy_checkpoint(
                    source, destination, expected_sha)

            self.assertEqual(actual_sha, expected_sha)
            self.assertEqual(checkpoint_io.file_sha256(destination), expected_sha)
            self.assertEqual(list(destination.parent.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
