"""The Continual-ARC environment over ARC-AGI-3 games, the third track.

A task is one game's rules at its first level (level 0); an instance is a fresh level of those rules, drawn from the
slot's instance seed by the game's generator (:mod:`continual_arc.arc3.instances`), and played in the real engine.
The public levels are never served.

The learner may :class:`Act` (one engine action), :class:`ResetLevel` (reload the level, needed after a game over) or
:class:`~continual_arc.env.RequestDemos`: one more demonstration, the solver's play of another fresh level of the
game, at any time while the instance is open, at most ``demo_cap`` an instance, which leaves the level being played
as it was. Every action and every reset counts toward the instance's action cap, ``ceil(cap_multiplier x baseline)``
and at most ``max_actions``, where the baseline is level 0's human action count scaled by the instance's solution
length over level 0's (:class:`continual_arc.arc3.instances.Drawn`). There is no give-up in the learner's protocol, so
quitting is never a choice (20 Sep 2026: models gave up at once when they could); :class:`~continual_arc.env.GiveUp` is
the runner's, for a runaway learner, and is charged as a failure.

Cost of an instance::

    solved:                 min(cap_multiplier, (actions + resets) / baseline) + demo_cost x demonstrations
    failed (cap reached):   max_cost = cap_multiplier + demo_cost x demo_cap

so that failing is never cheaper than trying, as on the grid tracks. Invalid actions (an id outside 1..7, a click
without coordinates in 0..63, an action the game does not offer, or one the engine ignored because the level is over)
are charged like any other and counted.

Instances are served in rounds, one instance per user of the schedule, as :class:`continual_arc.env.ContinualArcEnv`
serves them: every instance of a round is open at once, each in a game of its own, and the round's records are
committed together when its last instance ends, so a frozen test plays as one round of users. Records are
:class:`continual_arc.env.InstanceRecord`, with ``wrong_attempts`` the instance's actions (resets excluded),
``demo_requests`` its demonstrations, ``invalid_submissions`` its invalid actions, and ``metadata`` its resets,
baseline, action cap and solution length.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from continual_arc.arc3 import levels
from continual_arc.arc3.instances import (
    COMPLEX_ACTION,
    Act,
    Demonstration,
    Drawn,
    Frame,
    LevelSource,
)
from continual_arc.env import GiveUp, GiveUpFeedback, InstanceRecord, RequestDemos, StepResult
from continual_arc.schedule import Schedule, SlotInfo

ACTION_IDS = (1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class Arc3Config:
    cap_multiplier: float = 5.0
    """Actions allowed an instance, as a multiple of its baseline; the most a solved instance's play costs."""
    max_actions: int = 1000
    """Ceiling on any instance's action cap."""
    demo_cap: int = 3
    """Demonstrations an instance may buy; further requests are refused at no cost."""
    demo_cost: float = 1.0
    """Cost of one demonstration, in baselines' worth of play."""

    @property
    def max_cost(self) -> float:
        return self.cap_multiplier + self.demo_cost * self.demo_cap

    def action_cap(self, baseline: float) -> int:
        return max(1, min(self.max_actions, math.ceil(self.cap_multiplier * baseline)))

    def validate(self) -> None:
        if self.cap_multiplier <= 0 or self.max_actions < 1:
            raise ValueError("cap_multiplier must be > 0 and max_actions >= 1")
        if self.demo_cap < 0 or self.demo_cost < 0:
            raise ValueError("demo_cap and demo_cost must be >= 0")

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
    """The environment's whole view of an open instance, which the runner keeps for itself and its plug-ins."""

    step: int
    n_steps: int
    task_id: str
    phase: str
    frames: list[Frame]
    """The frames of the last engine call that rendered any (several when an action animates): the reset's one
    frame on a fresh instance. The last is the board now."""
    state: str
    available_actions: list[int]
    actions_used: int
    resets_used: int
    action_cap: int
    baseline_actions: float
    demo_requests_used: int
    demo_cap: int
    demo_cost: float
    max_cost: float
    exposure_index: int
    reset_before: bool
    episode_type: str = "benchmark"
    plugin: str | None = None
    metadata: dict = field(default_factory=dict)
    round: int = 0
    user: int = 0
    user_exposure_index: int = 0

    @property
    def is_fresh(self) -> bool:
        return self.actions_used == 0 and self.resets_used == 0 and self.demo_requests_used == 0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Arc3LearnerObservation:
    """What the runner shows a learner of a level: the game under the run's alias, the frames, the game's state and
    the actions it offers, the learner's actions, resets and demonstrations against their caps, the instance's
    baseline and the budget's figures, and its place in its own work. Nothing about the schedule (see
    :class:`continual_arc.env.LearnerObservation`)."""

    step: int
    task_id: str
    frames: list[Frame]
    state: str
    available_actions: list[int]
    actions_used: int
    resets_used: int
    action_cap: int
    baseline_actions: float
    demo_requests_used: int
    demo_cap: int
    demo_cost: float
    max_cost: float
    episode_type: str = "benchmark"
    plugin: str | None = None
    metadata: dict = field(default_factory=dict)
    round: int = 0
    user: int = 0

    @classmethod
    def of(cls, obs: Arc3Observation, task_id: str) -> Arc3LearnerObservation:
        return cls(step=obs.step, task_id=task_id, frames=obs.frames, state=obs.state,
                   available_actions=list(obs.available_actions), actions_used=obs.actions_used,
                   resets_used=obs.resets_used, action_cap=obs.action_cap, baseline_actions=obs.baseline_actions,
                   demo_requests_used=obs.demo_requests_used, demo_cap=obs.demo_cap, demo_cost=obs.demo_cost,
                   max_cost=obs.max_cost, episode_type=obs.episode_type, plugin=obs.plugin,
                   metadata=dict(obs.metadata), round=obs.round, user=obs.user)

    @property
    def is_fresh(self) -> bool:
        return self.actions_used == 0 and self.resets_used == 0 and self.demo_requests_used == 0

    @property
    def actions_left(self) -> int:
        return self.action_cap - self.actions_used - self.resets_used

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResetLevel:
    pass


Arc3Action = Act | ResetLevel | RequestDemos | GiveUp


@dataclass(frozen=True)
class ActFeedback:
    """The verdict on one action. The board it left is the next observation's, as a grid track's feedback is the
    verdict alone, so that the run's event log does not hold every frame twice."""

    state: str
    level_completed: bool
    actions_used: int
    valid: bool
    ignored: bool
    failed: bool = False
    """The action cap is reached on this action."""
    rendered: int = 0
    """Frames the action rendered: several when it animates, none when it was not valid or the engine ignored it."""


@dataclass(frozen=True)
class ResetFeedback:
    state: str
    actions_used: int
    resets_used: int
    failed: bool = False


@dataclass(frozen=True)
class DemonstrationFeedback:
    demonstration: Demonstration | None
    """None when the request was refused (the instance's demo cap reached)."""
    refused: bool
    demo_requests_used: int


Arc3Feedback = ActFeedback | ResetFeedback | DemonstrationFeedback | GiveUpFeedback


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class _Current:
    __slots__ = ("index", "info", "drawn", "game", "cap", "frames", "state", "available", "actions", "resets",
                 "invalid", "demos", "charged")

    def __init__(self, index: int, info: SlotInfo, drawn: Drawn, cap: int):
        self.index = index
        self.info = info
        self.drawn = drawn
        self.game, frame = levels.new_game(drawn.game, drawn.level)
        self.cap = cap
        self.frames: list[Frame] = list(frame.frame)
        self.state = frame.state.value
        self.available = [int(a) for a in frame.available_actions]
        self.actions = 0
        self.resets = 0
        self.invalid = 0
        self.demos = 0
        self.charged = 0.0

    @property
    def used(self) -> int:
        return self.actions + self.resets


class Arc3Env:
    """Serve a :class:`Schedule` of ARC-AGI-3 games in rounds, one fresh level per instance, with the surface of
    :class:`continual_arc.env.ContinualArcEnv` (``reset``, ``restore``, ``observation``, ``step``, ``open_indices``,
    ``is_open``, ``records``, ``total_cost``, ``done``, ``n_steps``) and its :class:`continual_arc.env.StepResult`, so
    that the runner and the metrics are shared. ``source`` draws the levels and demonstrations (a
    :class:`LevelSource` over the track's generators by default). Add-on episodes are not served."""

    def __init__(self, schedule: Schedule, config: Arc3Config | None = None, source: LevelSource | None = None):
        self.config = config or Arc3Config()
        self.config.validate()
        schedule.check_users()
        self.schedule = schedule
        self.source = source or LevelSource()
        self._infos = schedule.annotate()
        self._starts = schedule.round_starts()
        self._records: list[InstanceRecord] = []
        self._open: dict[int, _Current] = {}
        self._pending: dict[int, InstanceRecord] = {}
        self._round_end = 0
        self._focus: int | None = None
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
        return self._started and not self._open

    @property
    def n_steps(self) -> int:
        return len(self.schedule)

    def open_indices(self) -> list[int]:
        return sorted(self._open)

    def is_open(self, index: int) -> bool:
        return index in self._open

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> Arc3Observation | None:
        """Start (or restart) the stream. Returns the first observation."""
        self._records = []
        self._total_cost = 0.0
        self._started = True
        return self._open_round(0)

    def restore(self, records: list[InstanceRecord]) -> Arc3Observation | None:
        """Continue a stream after an interruption from the whole rounds an earlier run finished (see
        :meth:`continual_arc.env.ContinualArcEnv.restore`)."""
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
        n = len(records)
        if 0 < n < len(self.schedule) and not self._starts[n]:
            raise ValueError(f"{n} records end inside a round; restore whole rounds")
        self._records = list(records)
        self._total_cost = float(sum(record.cost for record in records))
        return self._open_round(n)

    def observation(self, index: int | None = None) -> Arc3Observation | None:
        """The observation of instance ``index``, or of the focused instance."""
        if index is None:
            index = self._focus
        if index is None or index not in self._open:
            return None
        cur = self._open[index]
        slot = self.schedule.slots[index]
        return Arc3Observation(
            step=index, n_steps=self.n_steps, task_id=slot.task_id, phase=slot.phase,
            frames=[[list(row) for row in f] for f in cur.frames], state=cur.state,
            available_actions=list(cur.available), actions_used=cur.actions, resets_used=cur.resets,
            action_cap=cur.cap, baseline_actions=cur.drawn.baseline, demo_requests_used=cur.demos,
            demo_cap=self.config.demo_cap, demo_cost=self.config.demo_cost, max_cost=self.config.max_cost,
            exposure_index=cur.info.exposure_index, reset_before=slot.reset_before, round=cur.info.round,
            user=slot.user, user_exposure_index=cur.info.user_exposure_index,
        )

    def step(self, action: Arc3Action, index: int | None = None) -> StepResult:
        """Act on instance ``index`` (the focused one when omitted)."""
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if not self._open:
            raise RuntimeError("stream is finished")
        if index is None:
            index = self._focus
        if index not in self._open:
            raise RuntimeError(f"instance {index} is not open")
        cur = self._open[index]
        self._focus = index
        if isinstance(action, Act):
            return self._act(cur, action)
        if isinstance(action, ResetLevel):
            return self._reset_level(cur)
        if isinstance(action, RequestDemos):
            return self._request_demo(cur)
        if isinstance(action, GiveUp):
            return self._give_up(cur)
        raise TypeError(f"unknown action {action!r}")

    # -- internals -----------------------------------------------------------

    def _open_round(self, index: int) -> Arc3Observation | None:
        """Open every instance of the round that starts at ``index``, each on the level its seed draws."""
        self._open = {}
        self._pending = {}
        if index >= len(self.schedule):
            self._focus = None
            self._round_end = len(self.schedule)
            return None
        end = index + 1
        while end < len(self.schedule) and not self._starts[end]:
            end += 1
        for i in range(index, end):
            slot = self.schedule.slots[i]
            drawn = self.source.instance(slot.task_id, slot.instance_seed)
            self._open[i] = _Current(i, self._infos[i], drawn, self.config.action_cap(drawn.baseline))
        self._round_end = end
        self._focus = index
        return self.observation(index)

    def _spent(self, cur: _Current) -> float:
        """What ``cur`` costs if it is solved now: its play in baselines, at most ``cap_multiplier``, and its
        demonstrations."""
        play = min(self.config.cap_multiplier, cur.used / cur.drawn.baseline)
        return play + self.config.demo_cost * cur.demos

    def _charge(self, cur: _Current, cost: float) -> float:
        """Bring ``cur``'s charge to ``cost``; the difference."""
        delta = cost - cur.charged
        cur.charged = cost
        self._total_cost += delta
        return delta

    def _result(self, cur: _Current, feedback: Any, *, solved: bool = False, failed: bool = False,
                gave_up: bool = False) -> StepResult:
        """The step's result: the instance's charge brought up to date, and the instance ended when it is solved,
        failed or given up."""
        if not (solved or failed or gave_up):
            delta = self._charge(cur, self._spent(cur))
            return StepResult(feedback, delta, False, False, None, self.observation(cur.index))
        delta = self._charge(cur, self._spent(cur) if solved else self.config.max_cost)
        record, nxt, committed = self._finish(cur, solved=solved, gave_up=gave_up)
        return StepResult(feedback, delta, True, nxt is None, record, nxt, committed)

    def _finish(self, cur: _Current, solved: bool, gave_up: bool):
        slot = self.schedule.slots[cur.index]
        record = InstanceRecord(
            step=cur.index, task_id=slot.task_id, phase=slot.phase, source=self.schedule.sources[slot.task_id],
            exposure_index=cur.info.exposure_index, gap=cur.info.gap, instance_seed=slot.instance_seed,
            reset_before=slot.reset_before, solved=solved, wrong_attempts=cur.actions,
            invalid_submissions=cur.invalid, demo_requests=cur.demos, gave_up=gave_up, cost=cur.charged,
            metadata={"resets": cur.resets, "baseline": cur.drawn.baseline, "action_cap": cur.cap,
                      "solution_length": len(cur.drawn.solution)},
            round=cur.info.round, user=slot.user, user_exposure_index=cur.info.user_exposure_index,
            user_gap=cur.info.user_gap,
        )
        del self._open[cur.index]
        self._pending[cur.index] = record
        if self._open:
            self._focus = min(self._open)
            return record, self.observation(), None
        committed = [self._pending[i] for i in sorted(self._pending)]
        self._records.extend(committed)
        return record, self._open_round(self._round_end), committed

    def _valid(self, cur: _Current, action: Act) -> bool:
        if action.action not in ACTION_IDS or action.action not in cur.available:
            return False
        if action.action == COMPLEX_ACTION:
            return action.x is not None and action.y is not None and 0 <= action.x <= 63 and 0 <= action.y <= 63
        return True

    def _act(self, cur: _Current, action: Act) -> StepResult:
        valid = self._valid(cur, action)
        frames: list[Frame] = []
        ignored = completed = False
        if valid:
            out = cur.game.perform_action(action.engine())
            frames = list(out.frame)
            ignored = not frames   # the level is over (game over) and the engine did nothing
            if frames:
                cur.frames = frames   # else the board stays as it was
            cur.state = out.state.value
            cur.available = [int(a) for a in out.available_actions]
            completed = levels.won(out)
        cur.actions += 1
        if not valid or ignored:
            cur.invalid += 1
        failed = not completed and cur.used >= cur.cap
        feedback = ActFeedback(cur.state, completed, cur.actions, valid and not ignored, ignored, failed, len(frames))
        return self._result(cur, feedback, solved=completed, failed=failed)

    def _reset_level(self, cur: _Current) -> StepResult:
        from arcengine import ActionInput, GameAction

        out = cur.game.perform_action(ActionInput(id=GameAction.RESET))
        cur.frames = list(out.frame)
        cur.state = out.state.value
        cur.available = [int(a) for a in out.available_actions]
        cur.resets += 1
        failed = cur.used >= cur.cap
        feedback = ResetFeedback(cur.state, cur.actions, cur.resets, failed)
        return self._result(cur, feedback, failed=failed)

    def _request_demo(self, cur: _Current) -> StepResult:
        if cur.demos >= self.config.demo_cap:
            return self._result(cur, DemonstrationFeedback(None, True, cur.demos))
        slot = self.schedule.slots[cur.index]
        demonstration = self.source.demonstration(slot.task_id, slot.instance_seed, cur.demos, cur.drawn.layout)
        cur.demos += 1
        return self._result(cur, DemonstrationFeedback(demonstration, False, cur.demos))

    def _give_up(self, cur: _Current) -> StepResult:
        charged = self.config.max_cost - cur.charged
        return self._result(cur, GiveUpFeedback(charged=charged), gave_up=True)
