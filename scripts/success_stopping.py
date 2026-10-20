"""Pure development-success-rate stopping controller for rollout studies.

The controller deliberately has no simulator, torch, or filesystem dependency.
Its state is JSON/checkpoint friendly so a trainer can commit the pending
evaluation together with its epoch-boundary recovery checkpoint.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Mapping


class SuccessStoppingController:
    """Track periodic success evaluations and strict-improvement patience."""

    STATE_SCHEMA = 1

    def __init__(self, *, minimum_epochs: int = 100,
                 evaluation_every_epochs: int = 5,
                 patience_epochs: int = 30,
                 minimum_improvement: float = 0.0,
                 state: Mapping[str, Any] | None = None):
        if minimum_epochs < 1 or evaluation_every_epochs < 1 or patience_epochs < 1:
            raise ValueError("minimum_epochs, evaluation_every_epochs, and patience_epochs must be positive")
        if not math.isfinite(float(minimum_improvement)) or minimum_improvement < 0:
            raise ValueError("minimum_improvement must be a finite nonnegative value")
        self.minimum_epochs = int(minimum_epochs)
        self.evaluation_every_epochs = int(evaluation_every_epochs)
        self.patience_epochs = int(patience_epochs)
        self.minimum_improvement = float(minimum_improvement)
        self.state = {
            "schema": self.STATE_SCHEMA,
            "best_success_rate": None,
            "best_epoch": None,
            "best_checkpoint_path": None,
            "best_checkpoint_sha256": None,
            "last_improvement_epoch": None,
            "last_evaluated_epoch": 0,
            "last_success_rate": None,
            "pending": None,
        }
        if state is not None:
            self.load_state(state)

    def load_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.STATE_SCHEMA:
            raise ValueError(f"Unsupported success-stopping state schema: {state.get('schema')!r}")
        for key in self.state:
            if key in state:
                self.state[key] = state[key]
        if self.state["pending"] is not None:
            pending_epoch = int(self.state["pending"]["epoch"])
            if not self.is_evaluation_epoch(pending_epoch):
                raise ValueError("Recovered pending rollout is not on the configured evaluation cadence")

    def state_dict(self) -> dict[str, Any]:
        # Return a detached mapping so callers cannot mutate controller state
        # accidentally while serializing last.pt.
        return {**self.state, "pending": (dict(self.state["pending"])
                                           if self.state["pending"] is not None else None)}

    def is_evaluation_epoch(self, epoch: int) -> bool:
        return (epoch >= self.minimum_epochs
                and (epoch - self.minimum_epochs) % self.evaluation_every_epochs == 0)

    def evaluation_due(self, epoch: int) -> bool:
        return (self.state["pending"] is None
                and self.is_evaluation_epoch(epoch)
                and epoch > int(self.state["last_evaluated_epoch"] or 0))

    @property
    def pending(self) -> dict[str, Any] | None:
        value = self.state["pending"]
        return dict(value) if value is not None else None

    def begin_evaluation(self, *, epoch: int, checkpoint_path: str,
                         optimizer_step: int) -> dict[str, Any]:
        if not self.evaluation_due(epoch):
            raise ValueError(f"Epoch {epoch} is not an available success-evaluation point")
        pending = {
            "epoch": int(epoch),
            "checkpoint_tag": f"selection_epoch_{int(epoch):06d}",
            "checkpoint_path": str(checkpoint_path),
            "optimizer_step": int(optimizer_step),
            "task": None,
        }
        self.state["pending"] = pending
        return dict(pending)

    def set_pending_task(self, task: Mapping[str, Any]) -> None:
        if self.state["pending"] is None:
            raise ValueError("Cannot attach a queue task when no evaluation is pending")
        self.state["pending"]["task"] = dict(task)

    def observe_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        pending = self.state["pending"]
        if pending is None:
            raise ValueError("There is no pending rollout evaluation to observe")
        raw_success_rate = result["success_rate"]
        if isinstance(raw_success_rate, bool):
            raise ValueError("Invalid rollout success_rate: boolean is not a numeric rate")
        success_rate = float(raw_success_rate)
        if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
            raise ValueError(f"Invalid rollout success_rate: {success_rate!r}")
        epoch = int(pending["epoch"])
        if int(self.state["last_evaluated_epoch"] or 0) >= epoch:
            raise ValueError(f"Epoch {epoch} has already been evaluated")
        previous_best = self.state["best_success_rate"]
        improved = (previous_best is None
                    or success_rate > float(previous_best) + self.minimum_improvement)
        if improved:
            self.state["best_success_rate"] = success_rate
            self.state["best_epoch"] = epoch
            self.state["best_checkpoint_path"] = pending["checkpoint_path"]
            self.state["best_checkpoint_sha256"] = pending.get("checkpoint_sha256")
            self.state["last_improvement_epoch"] = epoch
        self.state["last_evaluated_epoch"] = epoch
        self.state["last_success_rate"] = success_rate
        self.state["pending"] = None
        return {
            "epoch": epoch,
            "success_rate": success_rate,
            "improved": improved,
            "best_success_rate": self.state["best_success_rate"],
            "best_epoch": self.state["best_epoch"],
            "last_improvement_epoch": self.state["last_improvement_epoch"],
            "stop_due": self.should_stop(epoch),
        }

    def should_stop(self, epoch: int) -> bool:
        last_improvement = self.state["last_improvement_epoch"]
        return (self.state["pending"] is None
                and last_improvement is not None
                and epoch >= self.minimum_epochs + self.patience_epochs
                and epoch - int(last_improvement) >= self.patience_epochs)


class SuccessStoppingControllerTests(unittest.TestCase):
    """Small deterministic checks for the state machine's protocol edges."""

    def test_minimum_epoch_and_cadence(self):
        c = SuccessStoppingController()
        self.assertFalse(c.evaluation_due(99))
        self.assertTrue(c.evaluation_due(100))
        c.begin_evaluation(epoch=100, checkpoint_path="cp100", optimizer_step=1)
        self.assertFalse(c.evaluation_due(105))
        c.observe_result({"success_rate": 0.2})
        self.assertTrue(c.evaluation_due(105))
        self.assertFalse(c.evaluation_due(106))

    def test_tie_keeps_earliest_and_does_not_reset_patience(self):
        c = SuccessStoppingController()
        c.begin_evaluation(epoch=100, checkpoint_path="cp100", optimizer_step=1)
        c.state["pending"]["checkpoint_sha256"] = "a" * 64
        c.observe_result({"success_rate": 0.2})
        c.begin_evaluation(epoch=105, checkpoint_path="cp105", optimizer_step=2)
        result = c.observe_result({"success_rate": 0.2})
        self.assertFalse(result["improved"])
        self.assertEqual(c.state["best_epoch"], 100)
        self.assertEqual(c.state["best_checkpoint_path"], "cp100")
        self.assertEqual(c.state["best_checkpoint_sha256"], "a" * 64)
        self.assertEqual(c.state["last_improvement_epoch"], 100)
        self.assertFalse(c.should_stop(129))
        self.assertTrue(c.should_stop(130))

    def test_strict_improvement_resets_elapsed_epoch_patience(self):
        c = SuccessStoppingController()
        c.begin_evaluation(epoch=100, checkpoint_path="cp100", optimizer_step=1)
        c.observe_result({"success_rate": 0.2})
        c.begin_evaluation(epoch=130, checkpoint_path="cp130", optimizer_step=2)
        result = c.observe_result({"success_rate": 0.3})
        self.assertTrue(result["improved"])
        self.assertFalse(c.should_stop(130))
        self.assertFalse(c.should_stop(159))
        self.assertTrue(c.should_stop(160))

    def test_pending_evaluation_round_trips_for_resume(self):
        c = SuccessStoppingController()
        c.begin_evaluation(epoch=100, checkpoint_path="cp100", optimizer_step=14000)
        c.set_pending_task({"task_id": "study/selection_epoch_000100"})
        resumed = SuccessStoppingController(state=c.state_dict())
        self.assertEqual(resumed.pending["epoch"], 100)
        self.assertEqual(resumed.pending["task"]["task_id"], "study/selection_epoch_000100")
        resumed.observe_result({"success_rate": 0.0})
        self.assertEqual(resumed.state["best_epoch"], 100)
        self.assertFalse(resumed.should_stop(100))


if __name__ == "__main__":
    unittest.main()
