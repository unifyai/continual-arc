"""Data-free fixtures for the tests: a task registry that needs no pinned data, and learners that are interrupted."""

from __future__ import annotations

from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource


def fake_registry() -> TaskRegistry:
    """A registry serving the synthetic ``FakeSource`` tasks (no pinned data needed)."""
    source = FakeSource(n_tasks=12)
    return TaskRegistry({source.name: source})


class CrashAfterLearner:
    """``ShiftMemoryLearner`` that dies after ``n`` solved instances: an interruption for the ``--resume`` tests."""

    def __init__(self, n: int = 3):
        from continual_arc.testing import ShiftMemoryLearner

        self.inner = ShiftMemoryLearner()
        self.n = int(n)
        self.finished = 0

    def act(self, obs):
        if self.finished >= self.n:
            raise RuntimeError("interrupted")
        return self.inner.act(obs)

    def observe(self, obs, action, feedback) -> None:
        self.inner.observe(obs, action, feedback)
        if getattr(feedback, "correct", False):
            self.finished += 1

    def reset(self) -> None:
        self.inner.reset()

    def begin_round(self, n_instances: int) -> None:
        self.inner.begin_round(n_instances)

    def end_round(self) -> None:
        self.inner.end_round()

    def resources(self):
        return self.inner.resources()


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
