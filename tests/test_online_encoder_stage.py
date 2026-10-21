"""Atomic publication checks for the shared ALOHA frame stage."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.no_grad = lambda: (lambda function: function)
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.__path__ = []
    torch_data = types.ModuleType("torch.utils.data")
    torch_data.Dataset = object
    torch.utils = torch_utils
    torch_utils.data = torch_data
    sys.modules.update({"torch": torch, "torch.utils": torch_utils,
                        "torch.utils.data": torch_data})

import numpy as np

from scripts.online_encoder import _array_records, _publish_aloha_stage


class AlohaFrameStageTests(unittest.TestCase):
    def test_losing_concurrent_publisher_reuses_valid_winner(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            final_dir = root / "winner"
            final_dir.mkdir()
            frames = np.lib.format.open_memmap(final_dir / "frames.npy", mode="w+",
                                               dtype=np.uint8, shape=(2, 2, 2, 3))
            frames[:] = 17
            frames.flush()
            del frames
            identity_id = "locked-identity"
            identity = {"dataset": "aloha", "fixture": 1}
            arrays = _array_records(final_dir)
            manifest = {"schema": 1, "identity_id": identity_id,
                        "identity": identity, "samples": 2, "arrays": arrays}
            (final_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            temp_dir = root / "losing-temp"
            temp_dir.mkdir()
            (temp_dir / "frames.npy").write_bytes(b"private losing build")

            with patch("scripts.online_encoder.os.replace", side_effect=FileExistsError("winner exists")):
                published, reused_manifest, reused_arrays = _publish_aloha_stage(
                    temp_dir, final_dir, identity_id, identity, samples=2)

            self.assertFalse(published)
            self.assertEqual(reused_manifest, manifest)
            self.assertEqual(reused_arrays, arrays)
            self.assertEqual((temp_dir / "frames.npy").read_bytes(), b"private losing build")
            self.assertEqual(_array_records(final_dir), arrays)


if __name__ == "__main__":
    unittest.main()
