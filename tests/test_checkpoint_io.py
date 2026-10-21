"""Checkpoint copies must not depend on shared-volume metadata operations."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import checkpoint_io


def trainer_function(name, namespace):
    """Exercise the trainer's recovery functions without GPU/model imports."""
    source = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    factory = ("def bind():\n    recovery_revision = 0\n"
               + textwrap.indent(ast.unparse(function), "    ")
               + f"\n    return {name}\n")
    exec(compile(factory, str(source), "exec"), namespace)
    return namespace["bind"]()


class CheckpointIOTests(unittest.TestCase):
    def test_trainer_accepts_one_step_cosine_horizon_without_warmup(self):
        source = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        guard = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Name)
                     and node.test.left.id == "decay_steps"
                     and isinstance(node.test.ops[0], ast.LtE))
        expression = compile(ast.Expression(guard.test), str(source), "eval")
        for horizon, warmup, rejected in ((1, 0, False), (1, 1, True),
                                           (0, 0, True), (53760, 500, False)):
            with self.subTest(horizon=horizon, warmup=warmup):
                self.assertEqual(eval(expression, {"decay_steps": horizon,
                                                  "cfg": SimpleNamespace(warmup_steps=warmup)}),
                                 rejected)

    def test_trainer_restores_immutable_recovery_without_last_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pt"
            source.write_bytes(b"complete model optimizer scheduler and RNG state")
            durable_alias = root / "cloud" / "last.pt"
            revision = checkpoint_io.publish_recovery_revision(source, durable_alias.parent, 12, 3)
            local_last = root / "local" / "last.pt"
            checkpoint = {"epoch": 12, "global_step": 60, "recovery_revision": 3}
            validate = Mock()
            log = Mock()
            restore = trainer_function("restore_durable_recovery_if_newer", {
                "rollout_controller": object(), "args": SimpleNamespace(fresh=False),
                "durable_last_path": durable_alias, "last_path": local_last,
                "latest_verified_recovery": checkpoint_io.latest_verified_recovery,
                "atomic_copy_checkpoint": checkpoint_io.atomic_copy_checkpoint,
                "torch": SimpleNamespace(load=Mock(return_value=checkpoint)),
                "validate_recovery_identity": validate, "log_event": log,
            })
            with patch.dict("sys.modules", {"experiment_protocol": SimpleNamespace(
                    file_digest=checkpoint_io.file_sha256)}):
                restore()

            self.assertFalse(durable_alias.exists())
            self.assertEqual(local_last.read_bytes(), source.read_bytes())
            validate.assert_called_once_with(checkpoint, revision)
            self.assertEqual(log.call_args.kwargs["durable_path"], str(revision))

    def test_trainer_commits_only_verified_immutable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_dir = root / "local" / "checkpoints"
            local_dir.mkdir(parents=True)
            durable_alias = root / "cloud" / "checkpoints" / "last.pt"
            log, commit = Mock(), Mock()
            state = SimpleNamespace(state_dict=lambda: {"state": "complete"})
            namespace = {
                "metadata": {}, "model": state, "optim": state, "scheduler": state,
                "validation_metrics": {}, "best_val_loss": 0.2, "global_step": 60,
                "next_recovery_step": 80, "no_improve": 0,
                "cfg": SimpleNamespace(batch_size=2, output_dir=str(root / "local")),
                "rng_state": lambda: {"rng": "complete"},
                "train_generator": None, "val_generator": None,
                "rollout_controller": SimpleNamespace(pending=None, state_dict=lambda: {"pending": None}),
                "ckpt_dir": local_dir, "last_path": local_dir / "last.pt",
                "torch": SimpleNamespace(save=lambda payload, path: path.write_bytes(repr(payload).encode())),
                "publish_recovery_revision": checkpoint_io.publish_recovery_revision,
                "file_sha256": checkpoint_io.file_sha256,
                "atomic_copy_checkpoint": checkpoint_io.atomic_copy_checkpoint,
                "durable_last_path": durable_alias, "Path": Path,
                "log_event": log, "write_json_atomic": commit,
                "rollout_state_path": root / "controller.json",
                "args": SimpleNamespace(rollout_study_id="study", rollout_plan_id="plan",
                                        dataset="aloha", exp="exp05", seed=0),
                "protocol": {"protocol_id": "protocol"}, "signature": "signature",
                "datetime": datetime, "timezone": timezone,
            }
            save = trainer_function("save_recovery_checkpoint", namespace)

            def verify_before_commit(*_):
                revision = checkpoint_io.latest_verified_recovery(durable_alias.parent)
                self.assertIsNotNone(revision)
                self.assertEqual(checkpoint_io.file_sha256(revision), revision.stem.rsplit("_", 1)[1])
                self.assertEqual(revision.read_bytes(), namespace["last_path"].read_bytes())

            commit.side_effect = verify_before_commit
            save(12, 0.2)
            commit.assert_called_once()
            self.assertFalse(durable_alias.exists())
            event = next(call for call in log.call_args_list
                         if call.args[0] == "training.durable_last_committed")
            revision = Path(event.kwargs["path"])
            self.assertEqual(event.kwargs["sha256"], checkpoint_io.file_sha256(revision))
            self.assertEqual(event.kwargs["immutable_recovery_path"], str(revision))

            namespace["publish_recovery_revision"] = Mock(side_effect=OSError("unverified durable bytes"))
            with self.assertRaisesRegex(OSError, "unverified durable bytes"):
                save(13, 0.1)
            commit.assert_called_once()
            self.assertEqual(sum(call.args[0] == "training.durable_last_committed"
                                 for call in log.call_args_list), 1)

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
