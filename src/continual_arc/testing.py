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


# ---------------------------------------------------------------------------
# ARC-AGI-3 fakes: an in-memory game so Arc3Env runs without the toolkit
# ---------------------------------------------------------------------------


class FakeArc3Game:
    """A deterministic stand-in for :class:`continual_arc.arc3.games.GameHandle`.

    ``n_levels`` levels; ACTION1 completes the current level after ``baseline[level]``
    presses, ACTION2 ends the game (``GAME_OVER``) until a level reset, ACTION3 is
    a valid no-op. Frames are 64x64 zeros with the level index at ``[0][0]`` and
    the press count at ``[0][1]``. After ``GAME_OVER`` or ``WIN`` every action is
    ignored (empty frame list), like the real engine.
    """

    available_actions_default = (1, 2, 3)

    def __init__(
        self,
        game_id: str = "fake-game",
        baseline: tuple[int, ...] = (4, 6, 8),
        seed: int = 0,
        accepts_seed: bool = False,
        available_actions: tuple[int, ...] | None = None,
    ):
        self.game_id = game_id
        self.baseline_actions = list(baseline)
        self.n_levels = len(baseline)
        self.seed = seed
        self.accepts_seed = accepts_seed
        self.available_actions = list(available_actions or self.available_actions_default)
        self.level = 0
        self.presses = 0
        self.state = "NOT_PLAYED"
        self.levels_completed = 0
        self.starts: list[int] = []
        self.closed = False

    def _frame(self) -> list[list[int]]:
        frame = [[0] * 64 for _ in range(64)]
        frame[0][0] = self.level
        frame[0][1] = self.presses
        return frame

    def _outcome(self, frames, ignored: bool = False):
        from continual_arc.arc3.games import StepOutcome

        return StepOutcome(
            frames=frames,
            state=self.state,
            levels_completed=self.levels_completed,
            level_completed=self.levels_completed > self.level or self.state == "WIN",
            available_actions=list(self.available_actions),
            ignored=ignored,
        )

    def start_level(self, level_index: int) -> list[list[int]]:
        if not 0 <= level_index < self.n_levels:
            raise IndexError(level_index)
        self.level = level_index
        self.presses = 0
        self.state = "NOT_FINISHED"
        self.levels_completed = level_index
        self.starts.append(level_index)
        return self._frame()

    def act(self, action_id: int, x: int | None = None, y: int | None = None):
        if self.state in ("GAME_OVER", "WIN"):
            return self._outcome([], ignored=True)
        if action_id == 1:
            self.presses += 1
            if self.presses >= self.baseline_actions[self.level]:
                self.levels_completed = self.level + 1
                if self.level == self.n_levels - 1:
                    self.state = "WIN"
        elif action_id == 2:
            self.state = "GAME_OVER"
        return self._outcome([self._frame()])

    def reset_level(self):
        self.presses = 0
        self.state = "NOT_FINISHED"
        self.levels_completed = self.level
        return self._outcome([self._frame()])

    def close(self) -> None:
        self.closed = True


class FakeArc3Source:
    """A pool of fake games and an ``open_game`` factory for :class:`Arc3Env`."""

    def __init__(self, n_games: int = 8, baseline: tuple[int, ...] = (4, 6, 8), accepts_seed: bool = False):
        self._ids = [f"fk{i:02d}" for i in range(n_games)]
        self.baseline = baseline
        self.accepts_seed = accepts_seed
        self.opened: list[FakeArc3Game] = []

    def task_ids(self) -> list[str]:
        return list(self._ids)

    def open_game(self, game_id: str, seed: int = 0) -> FakeArc3Game:
        if game_id not in self._ids:
            raise KeyError(game_id)
        game = FakeArc3Game(game_id, self.baseline, seed=seed, accepts_seed=self.accepts_seed)
        self.opened.append(game)
        return game


class Arc3RandomLearner(BaseLearner):
    """Uniform random actions from the game's available set; resets after ``GAME_OVER``."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, obs):
        from continual_arc.arc3.env import Act, ResetLevel

        if obs.state == "GAME_OVER":
            return ResetLevel()
        choices = obs.available_actions or [1, 2, 3, 4, 5, 6, 7]
        action = self._rng.choice(choices)
        if action == 6:
            return Act(6, x=self._rng.randrange(64), y=self._rng.randrange(64))
        return Act(action)


class Arc3GiveUpLearner(BaseLearner):
    def act(self, obs):
        return GiveUp()
