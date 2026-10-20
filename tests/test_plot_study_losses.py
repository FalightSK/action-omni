"""Focused checks for success-stopped per-policy loss plots."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plot_study_losses as plots


class PlotRunLossesTests(unittest.TestCase):
    def test_plot_skips_rollout_rows_and_uses_latest_duplicate_epoch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            run_dir = Path(tmp)
            rows = [
                {"epoch": 1, "global_step": 10, "train_loss": 0.8, "validation_loss": 1.2},
                {"epoch": 1, "development_success_rate": 0.4},
                {"epoch": 1, "global_step": 10, "train_loss": 0.7, "validation_loss": 1.1},
                {"epoch": 2, "global_step": 20, "train_loss": 0.6, "validation_loss": 0.9},
            ]
            (run_dir / "history.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            captured = []
            original_plot = plots.plt.Axes.plot

            def capture_plot(axis, x, y, *args, **kwargs):
                captured.append((list(x), list(y)))
                return original_plot(axis, x, y, *args, **kwargs)

            with patch.object(plots.plt.Axes, "plot", new=capture_plot):
                output = plots.plot_run_losses(run_dir)

            self.assertEqual(output, run_dir / "training_curve.png")
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(captured, [([1, 2], [0.7, 0.6]), ([1, 2], [1.1, 0.9])])

            output.unlink()
            (run_dir / "history.jsonl").write_text(
                json.dumps({"epoch": 5, "development_success_rate": 0.6}) + "\n",
                encoding="utf-8")
            self.assertIsNone(plots.plot_run_losses(run_dir))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
