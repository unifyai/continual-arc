"""The Continual-ARC environment over ARC-AGI-3 games.

A task is a game; an instance is one level of it. The level served at a slot is
``exposure_index % n_levels``, so successive exposures to a game climb through
its levels and wrap around. Levels are fixed (there is no fresh input), so a
recurrence tests whether the learner kept the skill, not whether it cached an
output: a cached action sequence for level 0 is no help on level 1.

The learner may :class:`Act` (one engine action), :class:`ResetLevel` (reload
the level, needed after ``GAME_OVER``) or :class:`GiveUp`. Every action and
every reset counts one toward the instance's action cap
``min(max_actions, ceil(cap_multiplier * baseline))`` where ``baseline`` is the
human action count for that level. An instance ends solved when the level is
completed, failed when the cap is reached, or given up.

Cost of an instance, in units of the human baseline for its level::

    solved:              (actions_used + resets_used) / baseline   (at most cap_multiplier)
    failed or given up:  cap_multiplier

so ``max_cost`` is ``cap_multiplier`` and the score is
``100 * (1 - mean_cost / cap_multiplier)``. Invalid actions (id outside 1..7,
ACTION6 without coordinates in 0..63, an action the game does not offer, or an
action the engine ignored because the state was ``GAME_OVER`` or ``WIN``) are
charged like any other action and counted separately.

Records are :class:`continual_arc.env.InstanceRecord` with these fields reused:

* ``wrong_attempts``      = ``actions_used`` (non-reset actions on the instance)
* ``demo_requests``       = ``resets_used``
* ``invalid_submissions`` = invalid actions (see above)
* ``solved``, ``gave_up``, ``cost`` as defined above
* ``step``, ``task_id`` (game id), ``phase``, ``source`` (:data:`SOURCE_ARC3`),
  ``exposure_index``, ``gap``, ``instance_seed``, ``reset_before`` as usual.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

from continual_arc.arc3.games import (
    ACTION_IDS,
    COMPLEX_ACTION,
    STATE_NOT_FINISHED,
    Frame,
    StepOutcome,
)
from continual_arc.env import GiveUp, GiveUpFeedback, InstanceRecord, StepResult
from continual_arc.schedule import Schedule, SlotInfo

SOURCE_ARC3 = "arc3_game"
"""Source name of every ARC-AGI-3 game in a schedule (stream and epilogue)."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arc3Config:
    cap_multiplier: float = 5.0
    """Actions allowed per instance, as a multiple of the human baseline for the level."""
    max_actions: int = 1000
    """Hard ceiling on the per-instance action cap."""

    @property
    def max_cost(self) -> float:
        return self.cap_multiplier

    def validate(self) -> None:
        if self.cap_multiplier <= 0:
            raise ValueError("cap_multiplier must be > 0")
        if self.max_actions < 1:
            raise ValueError("max_actions must be >= 1")

    def action_cap(self, baseline: int) -> int:
        return max(1, min(self.max_actions, math.ceil(self.cap_multiplier * baseline)))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Arc3Config:
        return cls(**d)


# ---------------------------------------------------------------------------
# observation / actions / feedback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arc3Observation:
    step: int
    n_steps: int
    task_id: str
    """Game id, for example ``ls20``."""
    phase: str
    level: int
    n_levels: int
    exposure_index: int
    frames: list[Frame]
    """Frames from the last engine call that produced any; the single initial
    frame on a fresh instance. The last frame is the current board."""
    state: str
    available_actions: list[int]
    actions_used: int
    resets_used: int
    action_cap: int
    baseline_actions: int
    reset_before: bool
    episode_type: str = "benchmark"
    """Set by the runner's plugin layer; the benchmark stream is the only kind here."""
    round: int = 0
    user: int = 0
    """Games are served to one user; the field is here so every observation carries it."""
    user_exposure_index: int = 0

    @property
    def is_fresh(self) -> bool:
        return self.actions_used == 0 and self.resets_used == 0

    @property
    def actions_left(self) -> int:
        return self.action_cap - self.actions_used - self.resets_used

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Act:
    action: int
    x: int | None = None
    y: int | None = None


@dataclass(frozen=True)
class ResetLevel:
    pass


Arc3Action = Act | ResetLevel | GiveUp


@dataclass(frozen=True)
class ActFeedback:
    frames: list[Frame]
    state: str
    level_completed: bool
    actions_used: int
    valid: bool
    ignored: bool
    failed: bool = False  # action cap reached on this action


@dataclass(frozen=True)
class ResetFeedback:
    frames: list[Frame]
    state: str
    actions_used: int
    resets_used: int
    failed: bool = False  # action cap reached on this reset


Arc3Feedback = ActFeedback | ResetFeedback | GiveUpFeedback


# ---------------------------------------------------------------------------
# game handles
# ---------------------------------------------------------------------------


class GameHandleLike(Protocol):
    """What :class:`Arc3Env` needs from a game; :class:`continual_arc.arc3.games.GameHandle`
    and :class:`continual_arc.testing.FakeArc3Game` both satisfy it."""

    game_id: str
    n_levels: int
    baseline_actions: list[int]
    seed: int
    accepts_seed: bool

    def start_level(self, level_index: int) -> Frame: ...

    def act(self, action_id: int, x: int | None = None, y: int | None = None) -> StepOutcome: ...

    def reset_level(self) -> StepOutcome: ...

    def close(self) -> None: ...


OpenGame = Callable[[str, int], GameHandleLike]


def _default_open_game(game_id: str, seed: int) -> GameHandleLike:
    from continual_arc.arc3.games import open_game

    return open_game(game_id, seed)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class _Current:
    __slots__ = (
        "index", "info", "handle", "level", "baseline", "cap", "frames", "state",
        "available_actions", "actions", "resets", "invalid", "accrued",
    )

    def __init__(
        self,
        index: int,
        info: SlotInfo,
        handle: GameHandleLike,
        level: int,
        baseline: int,
        cap: int,
        frame: Frame,
    ):
        self.index = index
        self.info = info
        self.handle = handle
        self.level = level
        self.baseline = baseline
        self.cap = cap
        self.frames: list[Frame] = [frame]
        self.state = getattr(handle, "state", STATE_NOT_FINISHED)
        self.available_actions = list(getattr(handle, "available_actions", []))
        self.actions = 0
        self.resets = 0
        self.invalid = 0
        self.accrued = 0.0  # cost charged so far on this instance

    @property
    def used(self) -> int:
        return self.actions + self.resets


class Arc3Env:
    """Serve a :class:`Schedule` of ARC-AGI-3 games one level at a time.

    Same surface as :class:`continual_arc.env.ContinualArcEnv` (``reset``,
    ``observation``, ``step``, ``open_indices``, ``records``, ``total_cost``,
    ``done``, ``n_steps``) and the same :class:`continual_arc.env.StepResult`, so
    the runner and the metrics are shared. Games are served in the individual
    regime only: the schedule must have one user. ``open_game(game_id, seed)`` is the handle factory; the
    default opens the pinned games through the toolkit. Handles are cached per
    game id and reopened only when a game takes a seed and the slot's differs.
    """

    def __init__(
        self,
        schedule: Schedule,
        config: Arc3Config | None = None,
        open_game: OpenGame | None = None,
    ):
        self.config = config or Arc3Config()
        self.config.validate()
        if schedule.users != 1:
            raise ValueError("ARC-AGI-3 games are served one level at a time: the schedule must have one user")
        self.schedule = schedule
        self._open_game: OpenGame = open_game or _default_open_game
        self._handles: dict[str, GameHandleLike] = {}
        self._infos = schedule.annotate()
        self._records: list[InstanceRecord] = []
        self._cur: _Current | None = None
        self._total_cost = 0.0
        self._started = False

    # -- public state --------------------------------------------------------

    @property
    def records(self) -> list[InstanceRecord]:
        return list(self._records)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def done(self) -> bool:
        return self._started and self._cur is None

    @property
    def n_steps(self) -> int:
        return len(self.schedule)

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> Arc3Observation | None:
        """Start (or restart) the stream. Returns the first observation."""
        self._records = []
        self._total_cost = 0.0
        self._started = True
        self._cur = None
        return self._advance(0)

    def restore(self, records: list[InstanceRecord]) -> Arc3Observation | None:
        """Continue a stream after an interruption (see
        :meth:`continual_arc.env.ContinualArcEnv.restore`): accept the finished
        ``records`` and start the next level."""
        if not self._started:
            raise RuntimeError("call reset() before restore()")
        if self._records:
            raise RuntimeError("restore() must follow reset() directly")
        for i, record in enumerate(records):
            if i >= len(self.schedule):
                raise ValueError(f"record {i} is past the end of the schedule")
            slot = self.schedule.slots[i]
            if record.step != i or record.task_id != slot.task_id or record.instance_seed != slot.instance_seed:
                raise ValueError(f"record {i} ({record.task_id}) does not match the schedule slot ({slot.task_id})")
        self._records = list(records)
        self._total_cost = float(sum(record.cost for record in records))
        return self._advance(len(records))

    def observation(self, index: int | None = None) -> Arc3Observation | None:
        if self._cur is None or (index is not None and index != self._cur.index):
            return None
        cur = self._cur
        slot = self.schedule.slots[cur.index]
        return Arc3Observation(
            step=cur.index,
            n_steps=self.n_steps,
            task_id=slot.task_id,
            phase=slot.phase,
            level=cur.level,
            n_levels=cur.handle.n_levels,
            exposure_index=cur.info.exposure_index,
            frames=[[list(row) for row in f] for f in cur.frames],
            state=cur.state,
            available_actions=list(cur.available_actions),
            actions_used=cur.actions,
            resets_used=cur.resets,
            action_cap=cur.cap,
            baseline_actions=cur.baseline,
            reset_before=slot.reset_before,
            round=cur.info.round,
            user_exposure_index=cur.info.user_exposure_index,
        )

    def open_indices(self) -> list[int]:
        """Games are served one level at a time: the one open instance, if any."""
        return [] if self._cur is None else [self._cur.index]

    def is_open(self, index: int) -> bool:
        return self._cur is not None and self._cur.index == index

    def step(self, action: Arc3Action, index: int | None = None) -> StepResult:
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self._cur is None:
            raise RuntimeError("stream is finished")
        if index is not None and index != self._cur.index:
            raise RuntimeError(f"instance {index} is not open")
        if isinstance(action, Act):
            result = self._act(action)
        elif isinstance(action, ResetLevel):
            result = self._reset_level()
        elif isinstance(action, GiveUp):
            result = self._give_up()
        else:
            raise TypeError(f"unknown action {action!r}")
        if result.instance_done:
            result.round_records = [result.record]  # every instance is a round of one
        return result

    def close(self) -> None:
        """Close every cached game handle."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    # -- internals -----------------------------------------------------------

    def _handle(self, game_id: str, instance_seed: int) -> GameHandleLike:
        seed = instance_seed % 2**31
        handle = self._handles.get(game_id)
        if handle is not None and (not handle.accepts_seed or handle.seed == seed):
            return handle
        if handle is not None:
            handle.close()
        handle = self._open_game(game_id, seed)
        self._handles[game_id] = handle
        return handle

    def _advance(self, index: int) -> Arc3Observation | None:
        if index >= len(self.schedule):
            self._cur = None
            return None
        slot = self.schedule.slots[index]
        info = self._infos[index]
        handle = self._handle(slot.task_id, slot.instance_seed)
        level = info.exposure_index % handle.n_levels
        baseline = int(handle.baseline_actions[level])
        cap = self.config.action_cap(baseline)
        frame = handle.start_level(level)
        self._cur = _Current(index, info, handle, level, baseline, cap, frame)
        return self.observation()

    def _charge(self, amount: float) -> float:
        """Charge up to ``amount`` on the current instance, never past ``max_cost``."""
        cur = self._cur
        assert cur is not None
        delta = max(0.0, min(amount, self.config.max_cost - cur.accrued))
        cur.accrued += delta
        self._total_cost += delta
        return delta

    def _finish(self, solved: bool, gave_up: bool) -> tuple[InstanceRecord, Arc3Observation | None]:
        cur = self._cur
        assert cur is not None
        slot = self.schedule.slots[cur.index]
        if solved:
            cost = min(cur.used / cur.baseline, self.config.max_cost)
        else:
            cost = self.config.max_cost
        record = InstanceRecord(
            step=cur.index,
            task_id=slot.task_id,
            phase=slot.phase,
            source=self.schedule.sources.get(slot.task_id, SOURCE_ARC3),
            exposure_index=cur.info.exposure_index,
            gap=cur.info.gap,
            instance_seed=slot.instance_seed,
            reset_before=slot.reset_before,
            solved=solved,
            wrong_attempts=cur.actions,
            invalid_submissions=cur.invalid,
            demo_requests=cur.resets,
            gave_up=gave_up,
            cost=cost,
            round=cur.info.round,
            user_exposure_index=cur.info.user_exposure_index,
            user_gap=cur.info.user_gap,
        )
        self._records.append(record)
        nxt = self._advance(cur.index + 1)
        return record, nxt

    def _apply(self, cur: _Current, outcome: StepOutcome) -> None:
        if outcome.frames:
            cur.frames = outcome.frames
        cur.state = outcome.state
        cur.available_actions = list(outcome.available_actions)

    @staticmethod
    def _validate(cur: _Current, action: Act) -> bool:
        if action.action not in ACTION_IDS:
            return False
        if cur.available_actions and action.action not in cur.available_actions:
            return False
        if action.action == COMPLEX_ACTION:
            if action.x is None or action.y is None:
                return False
            if not (0 <= int(action.x) <= 63 and 0 <= int(action.y) <= 63):
                return False
        return True

    def _act(self, action: Act) -> StepResult:
        cur = self._cur
        assert cur is not None
        valid = self._validate(cur, action)
        ignored = False
        level_completed = False
        frames: list[Frame] = []
        if valid:
            outcome = cur.handle.act(action.action, action.x, action.y)
            self._apply(cur, outcome)
            frames = outcome.frames
            ignored = outcome.ignored
            level_completed = outcome.level_completed
            if ignored:
                valid = False
        cur.actions += 1
        if not valid:
            cur.invalid += 1
        delta = self._charge(1.0 / cur.baseline)
        if level_completed:
            fb = ActFeedback(frames, cur.state, True, cur.actions, valid, ignored, failed=False)
            record, nxt = self._finish(solved=True, gave_up=False)
            return StepResult(fb, delta, True, nxt is None, record, nxt)
        failed = cur.used >= cur.cap
        fb = ActFeedback(frames, cur.state, False, cur.actions, valid, ignored, failed=failed)
        if failed:
            delta += self._charge(self.config.max_cost)
            record, nxt = self._finish(solved=False, gave_up=False)
            return StepResult(fb, delta, True, nxt is None, record, nxt)
        return StepResult(fb, delta, False, False, None, self.observation())

    def _reset_level(self) -> StepResult:
        cur = self._cur
        assert cur is not None
        outcome = cur.handle.reset_level()
        self._apply(cur, outcome)
        cur.resets += 1
        delta = self._charge(1.0 / cur.baseline)
        failed = cur.used >= cur.cap
        fb = ResetFeedback(outcome.frames, cur.state, cur.actions, cur.resets, failed=failed)
        if failed:
            delta += self._charge(self.config.max_cost)
            record, nxt = self._finish(solved=False, gave_up=False)
            return StepResult(fb, delta, True, nxt is None, record, nxt)
        return StepResult(fb, delta, False, False, None, self.observation())

    def _give_up(self) -> StepResult:
        cur = self._cur
        assert cur is not None
        charged = self._charge(self.config.max_cost)
        fb = GiveUpFeedback(charged=charged)
        record, nxt = self._finish(solved=False, gave_up=True)
        return StepResult(fb, charged, True, nxt is None, record, nxt)
