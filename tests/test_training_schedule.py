"""Exercise the real trainer's LR schedule, patience and resume on tiny CPU data."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from configs.base_config import BaseVLAConfig
from scripts import train as trainer


class TinyHead(nn.Module):
    def __init__(self, _cfg):
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer("epochs", torch.tensor(0))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.epochs += 1
        return self

    def forward(self, embedding, state, actions, _mask):
        if not self.training:
            # A late improvement must reset patience after it has begun.
            return actions.new_tensor(0.5 if self.epochs >= 4 else 1.0)
        prediction = self.decoder(self.dropout(self.adapter(embedding) + state))
        return (prediction - actions).square().mean()


class TrainingScheduleTests(unittest.TestCase):
    def test_patience_after_warmup_and_exact_resume(self):
        cfg = BaseVLAConfig(batch_size=4, num_epochs=8, warmup_steps=10,
                            early_stop_patience=2, learning_rate=0.1,
                            min_lr_ratio=0.01, save_every=1, device="cpu")
        data = [{"embedding": torch.tensor([i / 20, 1.0]),
                 "state": torch.tensor([0.2, 0.3]),
                 "actions": torch.tensor([0.4, 0.5])} for i in range(20)]
        adam_step, replace = torch.optim.AdamW.step, Path.replace
        rates = []

        def checked_step(optim, *args, **kwargs):
            rates.append([group["lr"] for group in optim.param_groups])
            return adam_step(optim, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.pt"
            cache.touch()

            def run(output, interrupt_epoch=None):
                def commit(path, target):
                    result = replace(path, target)
                    if Path(target).name == "last.pt":
                        saved = torch.load(target, map_location="cpu", weights_only=False)
                        if saved["epoch"] == interrupt_epoch:
                            raise InterruptedError("Simulated interruption after checkpoint")
                    return result

                with patch.object(trainer, "get_config", side_effect=lambda *_: deepcopy(cfg)), \
                     patch.object(trainer, "_get_embedding_dataset", return_value=data), \
                     patch.object(trainer, "VLATrainModel", TinyHead), \
                     patch.object(torch.optim.AdamW, "step", checked_step), \
                     patch.object(Path, "replace", commit), redirect_stdout(io.StringIO()):
                    trainer.main(["--dataset", "aloha", "--exp", "exp05", "--seed", "7",
                                  "--cache", str(cache), "--output-dir", str(output), "--workers", "0"])
                return torch.load(output / "checkpoints/last.pt", map_location="cpu", weights_only=False)

            baseline_dir = root / "baseline"
            baseline = run(baseline_dir)
            history = [json.loads(line) for line in (baseline_dir / "history.jsonl").read_text().splitlines()]
            self.assertEqual(baseline["epoch"], 6)  # Stops while cosine decay is still running.
            self.assertEqual(baseline["global_step"], 24)
            self.assertEqual(baseline["scheduler"]["last_epoch"], 24)
            self.assertEqual(baseline["early_stop_start_step"], 10)
            self.assertEqual(baseline["early_stop_start_epoch"], 3)
            self.assertEqual([row["no_improve"] for row in history], [0, 0, 1, 0, 1, 2])
            self.assertEqual([row["phase"] for row in history], ["warmup"] * 2 + ["cosine_decay"] * 4)
            best = torch.load(baseline_dir / "checkpoints/best.pt", weights_only=False)
            self.assertEqual(best["epoch"], 4)
            decoder_rates = [row[1] for row in rates]
            self.assertEqual(decoder_rates[0], 0.0)
            self.assertAlmostEqual(decoder_rates[1], 0.01)
            self.assertAlmostEqual(decoder_rates[10], 0.1)
            self.assertTrue(all(a <= b for a, b in zip(decoder_rates[:10], decoder_rates[1:11])))
            self.assertTrue(all(a >= b for a, b in zip(decoder_rates[10:], decoder_rates[11:])))
            for adapter, decoder in rates:
                self.assertAlmostEqual(adapter, decoder * 0.5)
            rates.clear()
            self.assertEqual(run(baseline_dir)["global_step"], 24)
            self.assertEqual(rates, [])  # A completed patience window does not restart.

            for epoch in (2, 3, 5):
                with self.subTest(interruption_epoch=epoch):
                    output = root / f"resume_{epoch}"
                    with self.assertRaises(InterruptedError):
                        run(output, interrupt_epoch=epoch)
                    resumed = run(output)
                    self.assertEqual(resumed["epoch"], baseline["epoch"])
                    self.assertEqual(resumed["global_step"], baseline["global_step"])
                    self.assertEqual(resumed["no_improve"], baseline["no_improve"])
                    self.assertEqual(resumed["scheduler"], baseline["scheduler"])
                    self.assertEqual(resumed["optim"]["param_groups"], baseline["optim"]["param_groups"])
                    for key in baseline["state_dict"]:
                        torch.testing.assert_close(resumed["state_dict"][key], baseline["state_dict"][key], rtol=0, atol=0)
                    for param, state in baseline["optim"]["state"].items():
                        for key, value in state.items():
                            torch.testing.assert_close(resumed["optim"]["state"][param][key], value, rtol=0, atol=0)

            incompatible = deepcopy(baseline)
            incompatible["early_stopping"] = "after_lr_schedule"
            torch.save(incompatible, baseline_dir / "checkpoints/last.pt")
            with self.assertRaisesRegex(ValueError, "Resume stopping rule mismatch"):
                run(baseline_dir)
            torch.save(baseline, baseline_dir / "checkpoints/last.pt")
            cfg.min_lr_ratio = 0.02
            with self.assertRaisesRegex(ValueError, "Resume LR schedule mismatch"):
                run(baseline_dir)
            cfg.min_lr_ratio = 0.01
            cfg.num_epochs = 5
            capped = run(root / "max_epochs")
            self.assertEqual(capped["epoch"], 5)
            self.assertEqual(capped["no_improve"], 1)  # Hard cap applies before patience expires.
            self.assertAlmostEqual(capped["optim"]["param_groups"][1]["lr"], 0.001)
            self.assertAlmostEqual(capped["optim"]["param_groups"][0]["lr"], 0.0005)
            cfg.num_epochs = 8
            for warmup in (0, 8):
                with self.subTest(warmup_steps=warmup):
                    cfg.warmup_steps = warmup
                    result = run(root / f"warmup_{warmup}")
                    self.assertEqual(result["epoch"], 3)
                    self.assertEqual(result["no_improve"], 2)
                    self.assertEqual(result["early_stop_start_epoch"], 1 if warmup == 0 else 2)

            cfg.warmup_steps, cfg.lr_decay_epochs = 10, 4
            rates.clear()
            fast = run(root / "faster_decay")
            self.assertEqual(fast["epoch"], 6)
            self.assertEqual(fast["lr_schedule_steps"], 16)
            self.assertEqual(fast["max_training_steps"], 32)
            self.assertEqual(fast["scheduler"]["last_epoch"], 16)
            for adapter, decoder in rates[16:]:
                self.assertAlmostEqual(adapter, 0.0005)
                self.assertAlmostEqual(decoder, 0.001)  # No cosine rebound after the endpoint.
            with self.assertRaises(InterruptedError):
                run(root / "resume_floor", interrupt_epoch=5)
            resumed_floor = run(root / "resume_floor")
            self.assertEqual(resumed_floor["scheduler"], fast["scheduler"])
            self.assertEqual(resumed_floor["no_improve"], fast["no_improve"])
            for key in fast["state_dict"]:
                torch.testing.assert_close(resumed_floor["state_dict"][key], fast["state_dict"][key], rtol=0, atol=0)
            cfg.lr_decay_epochs = 5
            with self.assertRaisesRegex(ValueError, "Resume LR schedule mismatch"):
                run(root / "faster_decay")
            cfg.min_epochs = 7
            minimum = run(root / "minimum_epochs")
            self.assertEqual(minimum["epoch"], 7)
            self.assertEqual(minimum["no_improve"], 3)  # Counted after warmup, stopping deferred.
            with self.assertRaises(InterruptedError):
                run(root / "resume_minimum", interrupt_epoch=6)
            resumed_minimum = run(root / "resume_minimum")
            self.assertEqual(resumed_minimum["epoch"], minimum["epoch"])
            self.assertEqual(resumed_minimum["global_step"], minimum["global_step"])
            for key in minimum["state_dict"]:
                torch.testing.assert_close(resumed_minimum["state_dict"][key], minimum["state_dict"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
