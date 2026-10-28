"""The Continual-ARC environment.

The environment serves a task id and a test input per instance. The learner may
:class:`Submit` a grid or :class:`RequestDemos`. Submissions get correct/incorrect
feedback only. An instance ends when a submission is correct, when the
wrong-attempt cap is reached, or when the learner gives up.

Instances are served in rounds, one instance per user of the schedule: every
instance of a round is open at once, :meth:`ContinualArcEnv.step` names the instance
it acts on, and the round's records are committed together, in slot order, when its
last instance ends. With one user a round is one instance and the environment is serial.

A solved instance costs ``wrong_attempts + demo_cost * demo_requests``. An instance
that ends unsolved (cap reached or give-up) costs the whole budget,
:attr:`EnvConfig.max_cost`, whatever it spent: failing is never cheaper than trying,
and a demonstration is paid for only by an instance it helped to solve. Demo requests beyond the instance's demo
cap are refused at no cost: the configuration's cap, or fewer when the task's source
has fewer demonstrations to serve (a held-out task, which has no generator, serves only
its own pairs). The cost bound is the configuration's either way. Nothing else is
scored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from continual_arc.grids import Grid, InvalidGrid, normalize_grid
from continual_arc.schedule import Schedule, Slot, SlotInfo
from continual_arc.tasks import Pair, TaskRegistry

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    attempt_cap: int = 8
    """Wrong submissions allowed before the instance is marked failed."""
    demo_cap: int = 8
    """Demo requests allowed per instance; further requests are refused at no cost. Each request
    buys one more pair (``demos_per_request``): a fresh pair from the track's generators on the stream (Re-ARC's
    on arc1, ARC-GEN's on arc2); in the epilogue, whose
    tasks have no generators, the task's own pairs and nothing after them, so there an instance's cap is the
    number of pairs its task has left. With eight attempts and eight requests the cost bound is 16 (26 Sep;
    one request buying the whole set, bound 9, from 23 Sep; own pairs only in the epilogue from 27 Sep)."""
    demo_cost: float = 1.0
    """Cost of one demo request, in units of one wrong attempt."""
    demos_per_request: int = 1
    """Pairs returned per request."""
    diff_lb: float = 0.0
    diff_ub: float = 1.0
    """Re-ARC difficulty bounds for generated instances and demos."""

    @property
    def max_cost(self) -> float:
        return self.attempt_cap + self.demo_cost * self.demo_cap

    def validate(self) -> None:
        if self.attempt_cap < 1:
            raise ValueError("attempt_cap must be >= 1")
        if self.demo_cap < 0 or self.demo_cost < 0:
            raise ValueError("demo_cap and demo_cost must be >= 0")
        if self.demos_per_request < 1:
            raise ValueError("demos_per_request must be >= 1")
        if not 0 <= self.diff_lb <= self.diff_ub <= 1:
            raise ValueError("need 0 <= diff_lb <= diff_ub <= 1")

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> EnvConfig:
        return cls(**d)


# ---------------------------------------------------------------------------
# observation / actions / feedback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    step: int
    n_steps: int
    task_id: str
    phase: str
    input: Grid
    exposure_index: int
    attempts_used: int
    demo_requests_used: int
    attempt_cap: int
    demo_cap: int
    reset_before: bool
    episode_type: str = "benchmark"
    plugin: str | None = None
    metadata: dict = field(default_factory=dict)
    round: int = 0
    user: int = 0
    """Whose work this instance is (always 0 with one user)."""
    user_exposure_index: int = 0
    """Earlier rounds in which this user met the task (``exposure_index`` counts every user)."""

    @property
    def is_fresh(self) -> bool:
        return self.attempts_used == 0 and self.demo_requests_used == 0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LearnerObservation:
    """What the runner shows a learner of the instance it is working on: the task under the run's alias
    (:class:`continual_arc.aliases.TaskAliases`), the input, its own attempts and demonstration requests against
    their caps, and its place in its own work (``step``, ``round``, whose work it is), which it could count itself.
    Nothing about the schedule: not the phase, how often the task has come before, whether its state was wiped,
    or how long the stream is. :class:`Observation` is the environment's whole view of the slot, which the runner
    keeps for itself and its plug-ins."""

    step: int
    task_id: str
    input: Grid
    attempts_used: int
    demo_requests_used: int
    attempt_cap: int
    demo_cap: int
    episode_type: str = "benchmark"
    plugin: str | None = None
    metadata: dict = field(default_factory=dict)
    round: int = 0
    user: int = 0

    @classmethod
    def of(cls, obs: Observation, task_id: str) -> LearnerObservation:
        return cls(step=obs.step, task_id=task_id, input=obs.input, attempts_used=obs.attempts_used,
                   demo_requests_used=obs.demo_requests_used, attempt_cap=obs.attempt_cap, demo_cap=obs.demo_cap,
                   episode_type=obs.episode_type, plugin=obs.plugin, metadata=dict(obs.metadata), round=obs.round,
                   user=obs.user)

    @property
    def is_fresh(self) -> bool:
        return self.attempts_used == 0 and self.demo_requests_used == 0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Submit:
    grid: Grid


@dataclass(frozen=True)
class RequestDemos:
    pass


@dataclass(frozen=True)
class GiveUp:
    pass


Action = Submit | RequestDemos | GiveUp


@dataclass(frozen=True)
class SubmitFeedback:
    correct: bool
    valid: bool
    attempts_used: int
    failed: bool  # attempt cap reached on this submission


@dataclass(frozen=True)
class DemosFeedback:
    pairs: list[Pair]
    refused: bool
    demo_requests_used: int


@dataclass(frozen=True)
class GiveUpFeedback:
    charged: float


Feedback = SubmitFeedback | DemosFeedback | GiveUpFeedback


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass
class InstanceRecord:
    step: int
    task_id: str
    phase: str
    source: str
    exposure_index: int
    gap: int | None
    instance_seed: int
    reset_before: bool
    solved: bool
    wrong_attempts: int
    invalid_submissions: int
    demo_requests: int
    gave_up: bool
    cost: float
    resources: dict[str, float] = field(default_factory=dict)
    episode_type: str = "benchmark"
    addon: str | None = None
    metadata: dict = field(default_factory=dict)
    plugin: str | None = None
    round: int = 0
    user: int = 0
    user_exposure_index: int = 0
    user_gap: int | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> InstanceRecord:
        d = dict(d)
        if "user_exposure_index" not in d:  # records written before 24 Sep 2026: one user, so the counts agree
            d["user_exposure_index"], d["user_gap"] = d.get("exposure_index", 0), d.get("gap")
        return cls(**d)


@dataclass
class StepResult:
    feedback: Feedback
    cost_delta: float
    instance_done: bool
    stream_done: bool
    record: InstanceRecord | None
    observation: Observation | None
    """The acted-on instance's next observation while it is open; when it has
    ended, the observation of the next open instance, or of the first instance of
    the next round, or None when the stream is finished."""
    round_records: list[InstanceRecord] | None = None
    """The round's records in slot order, when this step ended the round."""


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class _Current:
    __slots__ = (
        "index", "pair", "attempts", "invalid", "demo_requests", "info", "done", "demo_cap",
    )

    def __init__(self, index: int, pair: Pair, info: SlotInfo, demo_cap: int):
        self.index = index
        self.pair = pair
        self.attempts = 0
        self.invalid = 0
        self.demo_requests = 0
        self.info = info
        self.done = False
        self.demo_cap = demo_cap


class ContinualArcEnv:
    """Serve a :class:`Schedule` in rounds, one instance per user."""

    def __init__(
        self,
        schedule: Schedule,
        registry: TaskRegistry | None = None,
        config: EnvConfig | None = None,
    ):
        self.config = config or EnvConfig()
        self.config.validate()
        schedule.check_users()
        self.schedule = schedule
        if registry is None:
            registry = TaskRegistry.default(self.config.diff_lb, self.config.diff_ub)
        registry.assign_many(schedule.sources)
        self.registry = registry
        # The original schedule remains immutable from the add-on's point of
        # view. Auxiliary episodes are kept in parallel arrays so their
        # insertion cannot change exposure/gap metadata for benchmark slots.
        self._base_slots = list(schedule.slots)
        self._base_infos = schedule.annotate()
        self._base_starts = schedule.round_starts()
        self._slots: list[Slot] = []
        self._sources: list[str] = []
        self._kinds: list[str] = []
        self._addons: list[str | None] = []
        self._plugins: list[str | None] = []
        self._metadata: list[dict] = []
        self._infos: list[SlotInfo] = []
        self._starts: list[bool] = []
        self._records: list[InstanceRecord] = []
        self._open: dict[int, _Current] = {}  # the round's instances still running
        self._pending: dict[int, InstanceRecord] = {}  # the round's finished instances
        self._round_end = 0  # index one past the open round
        self._focus: int | None = None
        self._total_cost = 0.0
        self._started = False

    # -- public state --------------------------------------------------------

    @property
    def records(self) -> list[InstanceRecord]:
        """Records of every completed round, in slot order."""
        return list(self._records)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def done(self) -> bool:
        return self._started and not self._open

    @property
    def n_steps(self) -> int:
        return len(self._slots)

    @property
    def benchmark_steps(self) -> int:
        """Number of original benchmark slots, excluding add-on episodes."""
        return len(self.schedule)

    def open_indices(self) -> list[int]:
        """The open instances of the current round, in slot order."""
        return sorted(self._open)

    def is_open(self, index: int) -> bool:
        return index in self._open

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> Observation | None:
        """Start (or restart) the stream. Returns the first observation."""
        self._slots = list(self._base_slots)
        self._sources = [self.schedule.sources[slot.task_id] for slot in self._slots]
        self._kinds = ["benchmark"] * len(self._slots)
        self._addons = [None] * len(self._slots)
        self._plugins = [None] * len(self._slots)
        self._metadata = [{} for _ in self._slots]
        self._infos = list(self._base_infos)
        self._starts = list(self._base_starts)
        self._records = []
        self._total_cost = 0.0
        self._started = True
        return self._open_round(0)

    def restore(self, records: list[InstanceRecord]) -> Observation | None:
        """Continue a stream after an interruption: accept the ``records`` an earlier
        run of this schedule finished and serve the next round. Only whole rounds
        are restored (a half-played round is replayed from scratch); the learner's
        own persistent state is the learner's business. Requires :meth:`reset`
        first and refuses add-on episodes."""
        if not self._started:
            raise RuntimeError("call reset() before restore()")
        if self._records:
            raise RuntimeError("restore() must follow reset() directly")
        for i, record in enumerate(records):
            if record.episode_type != "benchmark":
                raise ValueError("cannot restore a run with add-on episodes")
            if i >= len(self._base_slots):
                raise ValueError(f"record {i} is past the end of the schedule")
            slot = self._base_slots[i]
            if record.step != i or record.task_id != slot.task_id or record.instance_seed != slot.instance_seed:
                raise ValueError(f"record {i} ({record.task_id}) does not match the schedule slot ({slot.task_id})")
        n = len(records)
        if 0 < n < len(self._base_slots) and not self._base_starts[n]:
            raise ValueError(f"{n} records end inside a round; restore whole rounds")
        self._records = list(records)
        self._total_cost = float(sum(record.cost for record in records))
        return self._open_round(n)

    def observation(self, index: int | None = None) -> Observation | None:
        """The observation of instance ``index``, or of the focused instance: the
        one most recently acted on while open, else the first open one."""
        if index is None:
            index = self._focus
        if index is None or index not in self._open:
            return None
        cur = self._open[index]
        slot = self._slots[index]
        return Observation(
            step=index,
            n_steps=self.n_steps,
            task_id=slot.task_id,
            phase=slot.phase,
            input=cur.pair.input,
            exposure_index=cur.info.exposure_index,
            attempts_used=cur.attempts,
            demo_requests_used=cur.demo_requests,
            attempt_cap=self.config.attempt_cap,
            demo_cap=cur.demo_cap,
            reset_before=slot.reset_before,
            episode_type=self._kinds[index],
            plugin=self._plugins[index],
            metadata=dict(self._metadata[index]),
            round=cur.info.round,
            user=slot.user,
            user_exposure_index=cur.info.user_exposure_index,
        )

    def step(self, action: Action, index: int | None = None) -> StepResult:
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
        if isinstance(action, Submit):
            return self._submit(cur, action.grid)
        if isinstance(action, RequestDemos):
            return self._request_demos(cur)
        if isinstance(action, GiveUp):
            return self._give_up(cur)
        raise TypeError(f"unknown action {action!r}")

    # -- internals -----------------------------------------------------------

    def _open_round(self, index: int) -> Observation | None:
        """Open every instance of the round that starts at ``index``."""
        self._open = {}
        self._pending = {}
        if index >= len(self._slots):
            self._focus = None
            self._round_end = len(self._slots)
            return None
        end = index + 1
        while end < len(self._slots) and not self._starts[end]:
            end += 1
        for i in range(index, end):
            slot = self._slots[i]
            pair = self.registry.test_instance(slot.task_id, slot.instance_seed)
            self._open[i] = _Current(i, pair, self._infos[i], self._demo_cap(slot, pair))
        self._round_end = end
        self._focus = index
        return self.observation(index)

    def insert_next(self, requests) -> Observation | None:
        """Insert auxiliary episodes immediately before the next episode.

        This is the only mutation available to an add-on. It may be called
        between episodes, after :meth:`step` returns an instance-complete
        result, and only with one user, where the next episode is the one open,
        unstarted instance. The returned observation is the first inserted
        episode (or the existing next episode when ``requests`` is empty).
        """
        if not self._started:
            raise RuntimeError("call reset() before insert_next()")
        if self.schedule.users != 1:
            raise RuntimeError("add-on episodes are served one at a time: they need the individual regime (one user)")
        requests = list(requests)
        if not requests:
            return self.observation()
        index = self._focus if self._open else len(self._slots)

        slots = []
        sources = []
        kinds = []
        addons = []
        plugins = []
        metadata = []
        for request in requests:
            self.registry.assign(request.task_id, request.source)
            slots.append(Slot(
                task_id=request.task_id,
                instance_seed=int(request.instance_seed),
                phase=request.phase,
                reset_before=bool(request.reset_before),
            ))
            sources.append(request.source)
            kinds.append(request.episode_type)
            addons.append(request.addon)
            plugins.append(request.plugin)
            metadata.append(dict(request.metadata))

        self._slots[index:index] = slots
        self._sources[index:index] = sources
        self._kinds[index:index] = kinds
        self._addons[index:index] = addons
        self._plugins[index:index] = plugins
        self._metadata[index:index] = metadata
        self._infos[index:index] = [SlotInfo(exposure_index=-1, gap=None, round=-1, user_exposure_index=-1)] * len(slots)
        self._starts[index:index] = [True] * len(slots)  # each inserted episode is a round of one
        # The open instance was the old next episode; it shifted right and is
        # reached after these requests complete. Start at the first inserted one.
        return self._open_round(index)

    def _record_metadata(self, cur: _Current) -> dict:
        """The slot's metadata, plus the instance's demo cap where the source made it smaller than the
        configuration's (records of tracks whose sources never run out are unchanged)."""
        meta = dict(self._metadata[cur.index])
        if cur.demo_cap != self.config.demo_cap:
            meta["demo_cap"] = cur.demo_cap
        return meta

    def _finish(self, cur: _Current, solved: bool, gave_up: bool) -> tuple[InstanceRecord, Observation | None, list[InstanceRecord] | None]:
        """End ``cur``: its record joins the round's pending records, and when it was
        the round's last open instance the round is committed in slot order and the
        next round opened. Returns the record, the next observation and, on a
        completed round, its records."""
        slot = self._slots[cur.index]
        spent = self._spent(cur)
        cost = spent if solved else self.config.max_cost
        self._total_cost += cost - spent
        record = InstanceRecord(
            step=cur.index,
            task_id=slot.task_id,
            phase=slot.phase,
            source=self._sources[cur.index],
            exposure_index=cur.info.exposure_index,
            gap=cur.info.gap,
            instance_seed=slot.instance_seed,
            reset_before=slot.reset_before,
            solved=solved,
            wrong_attempts=cur.attempts,
            invalid_submissions=cur.invalid,
            demo_requests=cur.demo_requests,
            gave_up=gave_up,
            cost=cost,
            episode_type=self._kinds[cur.index],
            addon=self._addons[cur.index],
            metadata=self._record_metadata(cur),
            plugin=self._plugins[cur.index],
            round=cur.info.round,
            user=slot.user,
            user_exposure_index=cur.info.user_exposure_index,
            user_gap=cur.info.user_gap,
        )
        cur.done = True
        del self._open[cur.index]
        self._pending[cur.index] = record
        if self._open:
            self._focus = min(self._open)
            return record, self.observation(), None
        committed = [self._pending[i] for i in sorted(self._pending)]
        self._records.extend(committed)
        nxt = self._open_round(self._round_end)
        return record, nxt, committed

    def _submit(self, cur: _Current, grid: Grid) -> StepResult:
        try:
            candidate = normalize_grid(grid)
            valid = True
        except InvalidGrid:
            candidate = None
            valid = False
            cur.invalid += 1
        correct = valid and candidate == cur.pair.output
        if correct:
            fb = SubmitFeedback(correct=True, valid=True, attempts_used=cur.attempts, failed=False)
            record, nxt, committed = self._finish(cur, solved=True, gave_up=False)
            return StepResult(fb, 0.0, True, nxt is None, record, nxt, committed)
        cur.attempts += 1
        self._total_cost += 1.0
        failed = cur.attempts >= self.config.attempt_cap
        fb = SubmitFeedback(correct=False, valid=valid, attempts_used=cur.attempts, failed=failed)
        if failed:
            rest = self.config.max_cost - self._spent(cur)   # the unsolved instance's charge up to the whole budget
            record, nxt, committed = self._finish(cur, solved=False, gave_up=False)
            return StepResult(fb, 1.0 + rest, True, nxt is None, record, nxt, committed)
        self._focus = cur.index
        return StepResult(fb, 1.0, False, False, None, self.observation(cur.index))

    def _demo_cap(self, slot: Slot, pair: Pair) -> int:
        """The instance's demo cap: the configuration's, or the requests that the source's supply of
        demonstrations for the instance fills, when that is fewer."""
        supply = self.registry.demo_supply(slot.task_id, slot.instance_seed, pair)
        if supply is None:
            return self.config.demo_cap
        return min(self.config.demo_cap, -(-supply // self.config.demos_per_request))

    def _request_demos(self, cur: _Current) -> StepResult:
        self._focus = cur.index
        if cur.demo_requests >= cur.demo_cap:
            fb = DemosFeedback(pairs=[], refused=True, demo_requests_used=cur.demo_requests)
            return StepResult(fb, 0.0, False, False, None, self.observation(cur.index))
        slot = self._slots[cur.index]
        pairs = self.registry.demos(
            slot.task_id,
            slot.instance_seed,
            cur.demo_requests,
            self.config.demos_per_request,
            cur.pair,
        )
        cur.demo_requests += 1
        self._total_cost += self.config.demo_cost
        fb = DemosFeedback(pairs=pairs, refused=False, demo_requests_used=cur.demo_requests)
        return StepResult(fb, self.config.demo_cost, False, False, None, self.observation(cur.index))

    def _give_up(self, cur: _Current) -> StepResult:
        charged = self.config.max_cost - self._spent(cur)   # the rest of the whole budget
        fb = GiveUpFeedback(charged=charged)
        record, nxt, committed = self._finish(cur, solved=False, gave_up=True)
        return StepResult(fb, charged, True, nxt is None, record, nxt, committed)

    def _spent(self, cur: _Current) -> float:
        """What ``cur`` has been charged so far: one per wrong attempt and ``demo_cost`` per demo request."""
        return cur.attempts + self.config.demo_cost * cur.demo_requests
