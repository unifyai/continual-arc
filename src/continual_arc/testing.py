"""Test utilities: a data-free task source and trivial learners.

Nothing here is a baseline. These exist so the environment, runner and metrics can
be exercised without the Re-ARC data and without any model.
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter

from continual_arc.env import (
    Action,
    DemosFeedback,
    Feedback,
    GiveUp,
    Observation,
    RequestDemos,
    Submit,
)
from continual_arc.grids import Grid
from continual_arc.learner import BaseLearner
from continual_arc.seeds import derive_seed
from continual_arc.tasks import Pair


class FakeSource:
    """Deterministic synthetic tasks: output = (input + shift) mod 10.

    Each task id carries its own shift derived from its name, so different tasks
    have different rules and a learner can meaningfully "remember" them.
    """

    def __init__(self, name: str = "fake", n_tasks: int = 10, size: tuple[int, int] = (3, 3)):
        self.name = name
        self._ids = [f"fake{i:03d}" for i in range(n_tasks)]
        self.size = size

    def task_ids(self) -> list[str]:
        return list(self._ids)

    def shift(self, task_id: str) -> int:
        return 1 + derive_seed("shift", task_id) % 9

    def _pair(self, task_id: str, seed: int) -> Pair:
        rng = random.Random(seed)
        h, w = self.size
        inp: Grid = [[rng.randrange(10) for _ in range(w)] for _ in range(h)]
        s = self.shift(task_id)
        out: Grid = [[(c + s) % 10 for c in row] for row in inp]
        return Pair(inp, out)

    def test_instance(self, task_id: str, instance_seed: int) -> Pair:
        return self._pair(task_id, derive_seed("fake-test", task_id, instance_seed))

    def demos(self, task_id, instance_seed, request_index, n, exclude) -> list[Pair]:
        return [
            self._pair(task_id, derive_seed("fake-demo", task_id, instance_seed, request_index, k))
            for k in range(n)
        ]


class GiveUpLearner(BaseLearner):
    def act(self, obs: Observation) -> Action:
        return GiveUp()


class EchoLearner(BaseLearner):
    """Submits the input grid unchanged, forever. Always hits the attempt cap."""

    def act(self, obs: Observation) -> Action:
        return Submit(obs.input)


class DemoHungryLearner(BaseLearner):
    """Requests demos until refused, then gives up. Exercises the demo cap."""

    def act(self, obs: Observation) -> Action:
        if obs.demo_requests_used < obs.demo_cap:
            return RequestDemos()
        return GiveUp()


class ShiftMemoryLearner(BaseLearner):
    """Solves FakeSource tasks and remembers the shift per task id.

    On a task it has not seen, it requests demos once, infers the shift from the
    first pair, stores it, and submits. On a seen task it submits straight away.
    ``reset()`` wipes the memory. This is the canonical 'learning works' fixture.
    """

    def __init__(self) -> None:
        self.memory: dict[str, int] = {}
        self._pending_demos: list[Pair] = []
        self.calls = 0

    def reset(self) -> None:
        self.memory.clear()

    def act(self, obs: Observation) -> Action:
        self.calls += 1
        shift = self.memory.get(obs.task_id)
        if shift is None:
            if self._pending_demos:
                d = self._pending_demos[0]
                self.memory[obs.task_id] = (d.output[0][0] - d.input[0][0]) % 10
                self._pending_demos = []
                shift = self.memory[obs.task_id]
            else:
                return RequestDemos()
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs: Observation, action: Action, feedback: Feedback) -> None:
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            self._pending_demos = list(feedback.pairs)

    def resources(self) -> dict[str, float]:
        return {"calls": float(self.calls), "state_bytes": float(len(self.memory) * 16)}


class RoundMemoryLearner(ShiftMemoryLearner):
    """A ShiftMemoryLearner that honours the round contract: shifts inferred during a
    round are staged and enter the shared memory only in ``end_round``, so every
    instance of a round is solved with the memory the round began with. Working
    state per instance (the demos it was served) is keyed by slot."""

    def __init__(self) -> None:
        super().__init__()
        self._demos_by_step: dict[int, list[Pair]] = {}
        self._staged: dict[str, int] = {}
        self.rounds_begun = 0
        self.rounds_ended = 0

    def begin_round(self, n_instances: int) -> None:
        self.rounds_begun += 1

    def end_round(self) -> None:
        self.memory.update(self._staged)
        self._staged.clear()
        self._demos_by_step.clear()
        self.rounds_ended += 1

    def act(self, obs: Observation) -> Action:
        self.calls += 1
        shift = self.memory.get(obs.task_id)
        if shift is None:
            demos = self._demos_by_step.get(obs.step)
            if not demos:
                return RequestDemos()
            d = demos[0]
            shift = (d.output[0][0] - d.input[0][0]) % 10
            self._staged[obs.task_id] = shift
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs: Observation, action: Action, feedback: Feedback) -> None:
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            self._demos_by_step[obs.step] = list(feedback.pairs)


class ConcurrentRoundMemoryLearner(RoundMemoryLearner):
    """A RoundMemoryLearner that serves a round's users at the same time
    (``concurrent_users``): the runner calls it from one thread per user. Its state is
    guarded by a lock, its counters are kept per user (``user_resources``), each action
    waits ``delay_s`` (longer for some users than others, so users overtake one another),
    and it notes how many users were mid-action at once (``max_inside``)."""

    concurrent_users = True

    def __init__(self, delay_s: float = 0.0) -> None:
        super().__init__()
        self.delay_s = delay_s
        self._lock = threading.Lock()
        self._calls_by_user: Counter[int] = Counter()
        self._inside = 0
        self.max_inside = 0

    def act(self, obs: Observation) -> Action:
        with self._lock:
            self._inside += 1
            self.max_inside = max(self.max_inside, self._inside)
            self._calls_by_user[obs.user] += 1
        try:
            if self.delay_s:
                time.sleep(self.delay_s * (1 + obs.user % 3))
            with self._lock:
                return super().act(obs)
        finally:
            with self._lock:
                self._inside -= 1

    def observe(self, obs: Observation, action: Action, feedback: Feedback) -> None:
        with self._lock:
            super().observe(obs, action, feedback)

    def user_resources(self, user: int) -> dict[str, float]:
        with self._lock:
            return {"calls": float(self._calls_by_user[user])}


class ScriptedLearner(BaseLearner):
    """Replays a fixed list of actions; useful for asserting exact env behaviour."""

    def __init__(self, actions: list[Action]):
        self._actions = list(actions)
        self.seen: list[tuple[Observation, Action, Feedback]] = []

    def act(self, obs: Observation) -> Action:
        if not self._actions:
            return GiveUp()
        return self._actions.pop(0)

    def observe(self, obs, action, feedback) -> None:
        self.seen.append((obs, action, feedback))
