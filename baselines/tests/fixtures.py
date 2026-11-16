"""Data-free fixtures for the tests: a task registry that needs no pinned data, a learner that is interrupted, and what
the tests of a run stopped and resumed share."""

from __future__ import annotations

import json
from pathlib import Path

from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

CLOCK = ("wall_seconds",)
"""What a record reads from the clock, which no two plays share."""


class Stopped(Exception):
    """The stop a test makes partway through a run, as a preemption or a crash stops one."""


def rows(path: Path) -> list[dict]:
    """The JSON lines of ``path``, less what a record reads from the clock (:data:`CLOCK`)."""
    out = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "resources" in row:
            row["resources"] = {k: v for k, v in row["resources"].items() if k not in CLOCK}
        out.append(row)
    return out


def fake_registry() -> TaskRegistry:
    """A registry serving the synthetic ``FakeSource`` tasks (no pinned data needed)."""
    source = FakeSource(n_tasks=12)
    return TaskRegistry({source.name: source})


class InterruptedLearner:
    """``ShiftMemoryLearner`` that interrupts its own process, as the verifier stops a run, in the middle of its ``n``-th
    instance: the run should end once that instance has."""

    def __init__(self, n: int = 3):
        from continual_arc.testing import ShiftMemoryLearner

        self.inner = ShiftMemoryLearner()
        self.n = int(n)
        self.acts = 0
        self.instances = 0

    def act(self, obs):
        import os
        import signal
        import time

        self.acts += 1
        if self.acts == 1:
            self.instances += 1
            if self.instances == self.n:
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.5)  # the interrupt's handler runs in the main thread, while this one plays in its own
        return self.inner.act(obs)

    def observe(self, obs, action, feedback) -> None:
        self.inner.observe(obs, action, feedback)
        if getattr(feedback, "correct", False) or getattr(feedback, "failed", False) or type(feedback).__name__ == "GiveUpFeedback":
            self.acts = 0

    def reset(self) -> None:
        self.inner.reset()

    def begin_round(self, n_instances: int) -> None:
        self.inner.begin_round(n_instances)

    def end_round(self) -> None:
        self.inner.end_round()

    def resources(self):
        return self.inner.resources()
