"""Schedules: the sequence of task instances a learner faces.

The benchmark's stream is just a schedule. A schedule is a list of :class:`Slot`
values, each naming a task, an instance seed (which fully determines the test
input and any demos served for that slot), a phase label, whether the learner's
persistent state must be wiped before the slot, and the user whose work it is.

A schedule serves one user (the individual regime) or many at once (the collective
regime, :attr:`Schedule.users`). With many users every user walks a stream of their
own, and the schedule holds them round by round: a round is one instance per user,
in user order, all served at once (:func:`interleave_users`).

One process generates every stream (:func:`stream_schedule`): a set of active tasks,
each worked in a spell, resting between spells, with new tasks arriving over time.
Its parameters (:class:`StreamParams`) place the stream on an axis from blocked to
structureless, and its orders are named points on it:

* ``workstreams`` — the benchmark's: a few tasks active at once, exponential bursts, log-normal dormancy, Poisson
  arrivals.
* ``sequential``  — one task active, one spell of every instance, tasks never return (blocked order).
* ``random``      — every task active from the first step, spells never end (uniform i.i.d. order).

The last two serve the stream's visits only to check that the order does not change a finding (the paper's
Appendix I). A configuration gives every task the same *supply* of instances (its stream length over its number of
tasks), and the process serves exactly that supply in every order, so the three are the same multiset of instances
in three orders. A supply of ``None`` is the process without that constraint.

:func:`epilogue_schedule` generates the held-out tasks, each seen once.

Schedules serialise to JSON and carry their generation parameters as provenance.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from continual_arc.seeds import derive_seed

PHASE_STREAM = "stream"
PHASE_EPILOGUE = "epilogue"
# The two frozen tests after the stream (continual_arc.configs.test_schedule), each a phase of its own
PHASE_HELD_OUT_TEST = "held-out-test"
PHASE_SEEN_TEST = "seen-test"
PHASES = (PHASE_STREAM, PHASE_EPILOGUE, PHASE_HELD_OUT_TEST, PHASE_SEEN_TEST)
TEST_HELD_OUT = "held-out"
TEST_SEEN = "seen"
TESTS = (TEST_HELD_OUT, TEST_SEEN)
# each test's phase; the held-out test's references may also be fresh runs, whose held-out instances are an epilogue
# with the learner wiped before each, frozen by construction
TEST_PHASE = {TEST_HELD_OUT: PHASE_HELD_OUT_TEST, TEST_SEEN: PHASE_SEEN_TEST}
TEST_REFERENCE_PHASES = {TEST_HELD_OUT: (PHASE_HELD_OUT_TEST, PHASE_EPILOGUE), TEST_SEEN: (PHASE_SEEN_TEST,)}


@dataclass(frozen=True)
class Slot:
    task_id: str
    instance_seed: int
    phase: str
    reset_before: bool = False
    user: int = 0
    """Whose work the instance is: 0 in the individual regime, 0 to ``users - 1`` in the collective."""

    def to_json(self) -> dict:
        d = asdict(self)
        if not self.user:
            del d["user"]  # an individual schedule's file is unchanged by the user field
        return d

    @classmethod
    def from_json(cls, d: dict) -> Slot:
        return cls(
            task_id=d["task_id"],
            instance_seed=int(d["instance_seed"]),
            phase=d["phase"],
            reset_before=bool(d.get("reset_before", False)),
            user=int(d.get("user", 0)),
        )


@dataclass(frozen=True)
class SlotInfo:
    """Derived per-slot metadata, counted in rounds: the round the slot is served in,
    how many earlier rounds held the task (for any user), and the rounds since the last of
    them; and the same two counts for the slot's own user alone. With one user the two
    pairs are equal."""

    exposure_index: int
    gap: int | None  # rounds since the task's previous round; None on first exposure
    round: int = 0
    user_exposure_index: int = 0  # earlier rounds in which this slot's user met the task
    user_gap: int | None = None   # rounds since this user last met the task; None on the user's first


@dataclass
class Schedule:
    """A list of slots and the number of users they serve.

    ``users`` is the regime. One user is the individual regime: one person's stream,
    one instance at a time, every slot a round of its own. Many users is the collective
    regime: one system serving a team at once. Every user walks a stream of their own
    over the same tasks, and a round is one instance per user, served concurrently and
    acted on with the state the system had when the round began. The slots are stored
    round by round, a round's ``users`` slots consecutive and in user order, and a round
    never spans a phase, so the epilogue starts on a round boundary.
    """

    name: str
    seed: int
    slots: list[Slot]
    sources: dict[str, str]  # task_id -> source name (see continual_arc.tasks)
    meta: dict = field(default_factory=dict)
    users: int = 1

    # -- basics ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.slots)

    def __iter__(self):
        return iter(self.slots)

    def task_ids(self) -> list[str]:
        return sorted({s.task_id for s in self.slots})

    def phases(self) -> list[str]:
        seen: list[str] = []
        for s in self.slots:
            if s.phase not in seen:
                seen.append(s.phase)
        return seen

    def rounds(self) -> list[int]:
        """The round of every slot: ``users`` consecutive slots share a round, and a
        new round starts at every phase change."""
        out: list[int] = []
        r = -1
        in_round = 0
        phase = None
        for s in self.slots:
            if s.phase != phase or in_round == self.users:
                r += 1
                in_round = 0
                phase = s.phase
            out.append(r)
            in_round += 1
        return out

    def check_users(self) -> None:
        """Raise unless every round's slots name their users in order (0, 1, 2, ...), as
        :func:`interleave_users` lays them out; a round cut short by a phase change holds the
        first users only. With one user every slot is user 0."""
        rounds = self.rounds()
        position = 0
        for i, (slot, r) in enumerate(zip(self.slots, rounds, strict=True)):
            position = 0 if i == 0 or r != rounds[i - 1] else position + 1
            if slot.user != position:
                raise ValueError(f"slot {i} (round {r}, position {position}) is user {slot.user}: a round's slots "
                                 "must be its users in order (build collective schedules with interleave_users)")

    def round_starts(self) -> list[bool]:
        rounds = self.rounds()
        return [i == 0 or rounds[i] != rounds[i - 1] for i in range(len(rounds))]

    def first_rounds(self, r: int) -> Schedule:
        """The first ``r`` rounds, a literal prefix of this schedule's own slots: the same
        tasks, instance seeds and users as rounds 0 to ``r - 1`` here, because it is sliced
        from them rather than rebuilt. Rebuilding would not do: a stream under ``even_visits``
        gives every task a supply fixed by the configuration's full length, so a shorter stream
        built from the same seed is a different schedule, not this one's prefix
        (:mod:`continual_arc.configs`). ``r`` may stop mid-phase, in which case the slice never
        reaches the epilogue. Used to stop a collective stream short and later resume it under
        the full schedule (:func:`continual_arc.runner.run`'s ``resume`` checks the prefix
        matches slot by slot, which a true slice always does)."""
        rounds = self.rounds()
        n_rounds = (rounds[-1] + 1) if rounds else 0
        if not 0 <= r <= n_rounds:
            raise ValueError(f"r must be between 0 and this schedule's {n_rounds} rounds, got {r}")
        cut = sum(1 for x in rounds if x < r)
        return Schedule(
            name=f"{self.name}-r{r}",
            seed=self.seed,
            slots=list(self.slots[:cut]),
            sources=dict(self.sources),
            meta={**self.meta, "cut_to_round": r, "full_rounds": n_rounds},
            users=self.users,
        )

    def annotate(self) -> list[SlotInfo]:
        """Exposure index and gap for every slot, counted in rounds over the whole
        schedule: the exposure index is the number of earlier rounds that held the
        task, for any user, and the gap the rounds since the last of them, so two slots
        of one task in the same round share both. The user's own exposure index and gap
        count only the rounds in which the slot's user met the task. With one user the
        two agree: the times the task was seen before and the steps since it was last seen."""
        rounds = self.rounds()
        last_round: dict[str, int] = {}
        count: Counter[str] = Counter()
        user_last: dict[tuple[int, str], int] = {}
        user_count: Counter[tuple[int, str]] = Counter()
        infos: list[SlotInfo] = []
        seen_this_round: set[str] = set()
        current = -1
        for s, r in zip(self.slots, rounds, strict=True):
            if r != current:
                for tid in seen_this_round:
                    count[tid] += 1
                    last_round[tid] = current
                seen_this_round = set()
                current = r
            prev = last_round.get(s.task_id)
            mine = (s.user, s.task_id)
            user_prev = user_last.get(mine)
            infos.append(SlotInfo(exposure_index=count[s.task_id], gap=None if prev is None else r - prev, round=r,
                                  user_exposure_index=user_count[mine],
                                  user_gap=None if user_prev is None else r - user_prev))
            seen_this_round.add(s.task_id)
            user_count[mine] += 1  # a user has one slot per round, so their own count moves at once
            user_last[mine] = r
        return infos

    def reset_each(self) -> Schedule:
        """The same slots with the learner's state wiped before every one: the learner reset before every
        instance of its own stream, the share learned's reference (:func:`continual_arc.metrics.share_learned`).
        In the collective regime a wipe belongs to the round, so every round starts from nothing."""
        return Schedule(
            name=f"{self.name}-reset",
            seed=self.seed,
            slots=[replace(s, reset_before=True) for s in self.slots],
            sources=dict(self.sources),
            meta={**self.meta, "reset_each": True},
            users=self.users,
        )

    def concat(self, other: Schedule, name: str | None = None) -> Schedule:
        sources = dict(self.sources)
        for tid, src in other.sources.items():
            if tid in sources and sources[tid] != src:
                raise ValueError(f"task {tid} assigned to both {sources[tid]} and {src}")
            sources[tid] = src
        return Schedule(
            name=name or f"{self.name}+{other.name}",
            seed=self.seed,
            slots=self.slots + other.slots,
            sources=sources,
            meta={"parts": [self.meta, other.meta]},
            users=self.users,
        )

    def summary(self) -> dict:
        infos = self.annotate()
        per_phase = Counter(s.phase for s in self.slots)
        exposures = Counter(s.task_id for s in self.slots)
        gaps = [i.gap for i in infos if i.gap is not None]
        return {
            "name": self.name,
            "seed": self.seed,
            "n_steps": len(self.slots),
            "users": self.users,
            "n_rounds": (self.rounds()[-1] + 1) if self.slots else 0,
            "n_tasks": len(exposures),
            "phases": dict(per_phase),
            "exposures_per_task": {
                "min": min(exposures.values()) if exposures else 0,
                "max": max(exposures.values()) if exposures else 0,
                "mean": (sum(exposures.values()) / len(exposures)) if exposures else 0.0,
            },
            "gaps": {
                "n": len(gaps),
                "median": _median(gaps),
                "p90": _quantile(gaps, 0.9),
                "max": max(gaps) if gaps else None,
            },
            "n_resets": sum(1 for s in self.slots if s.reset_before),
        }

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "format": "continual-arc/schedule/v1",
            "name": self.name,
            "seed": self.seed,
            "users": self.users,
            "sources": self.sources,
            "meta": self.meta,
            "slots": [s.to_json() for s in self.slots],
        }

    @classmethod
    def from_json(cls, d: dict) -> Schedule:
        if d.get("format") != "continual-arc/schedule/v1":
            raise ValueError(f"unsupported schedule format {d.get('format')!r}")
        return cls(
            name=d["name"],
            seed=int(d["seed"]),
            slots=[Slot.from_json(s) for s in d["slots"]],
            sources=dict(d["sources"]),
            meta=dict(d.get("meta", {})),
            # "width" is the name files written before 24 Sep 2026 give the same number
            users=int(d.get("users", d.get("width", 1))),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Schedule:
        return cls.from_json(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _slot_seed(schedule_seed: int, phase: str, step: int) -> int:
    return derive_seed(schedule_seed, phase, step)


@dataclass(frozen=True)
class StreamParams:
    """Parameters of the one process behind every stream.

    concurrency: tasks active at once; ``None`` means every task is active.
    burst_mean: mean length of an active spell in instances; ``None`` means a spell never ends.
    burst_fixed: spells are exactly ``burst_mean`` long instead of exponential draws.
    dormancy_median / dormancy_sigma: log-normal gap (in steps) between a task's spells;
        a ``None`` median means a task whose spell has ended never returns.
    arrival_interval: mean steps between the introduction of never-seen tasks; ``0``
        means the next task is due the moment a slot is free.

    Workstreams is the defaults. Sequential is ``concurrency=1``, a fixed burst of the
    task's whole supply, no return and immediate arrivals; random is every task active
    with spells that never end. See :func:`sequential_params` and :data:`RANDOM`.
    """

    concurrency: int | None = 4
    burst_mean: float | None = 6.0
    burst_fixed: bool = False
    dormancy_median: float | None = 80.0
    dormancy_sigma: float = 1.0
    arrival_interval: float = 30.0

    def validate(self) -> None:
        if self.concurrency is not None and self.concurrency < 1:
            raise ValueError("concurrency must be >= 1 or None for every task")
        if self.burst_mean is not None and self.burst_mean < 1:
            raise ValueError("burst_mean must be >= 1 or None for spells that never end")
        if self.burst_fixed and (self.burst_mean is None or self.burst_mean != int(self.burst_mean)):
            raise ValueError("a fixed burst needs a whole burst_mean")
        if (self.dormancy_median is not None and self.dormancy_median <= 0) or self.dormancy_sigma < 0:
            raise ValueError("dormancy parameters must be positive")
        if self.arrival_interval < 0:
            raise ValueError("arrival_interval must be >= 0")


def sequential_params(instances_per_task: int) -> StreamParams:
    """Blocked order: one task at a time, all of its instances, then the next."""
    return StreamParams(concurrency=1, burst_mean=instances_per_task, burst_fixed=True,
                        dormancy_median=None, arrival_interval=0)


RANDOM = StreamParams(concurrency=None, burst_mean=None, dormancy_median=None, arrival_interval=0)
"""Structureless order: every task active from the first step, sampled uniformly."""


def stream_schedule(
    task_ids: Iterable[str],
    length: int,
    seed: int,
    source: str,
    params: StreamParams | None = None,
    order: str = "workstreams",
    name: str | None = None,
    instances_per_task: int | None = None,
) -> Schedule:
    """The stream: ``length`` instances over ``task_ids`` under one process.

    ``instances_per_task`` is each task's *supply*: when given, every task is served exactly
    that many times over the stream, in every order, so ``length`` must be the number of tasks
    times the supply. A spell never exceeds what the task has left, and a task whose supply is
    spent is finished (it neither rests nor returns). Sequential is unchanged by it (one spell of
    the whole supply); random becomes a uniform shuffle of the balanced multiset; workstreams keeps
    its bursts, rests and arrivals and ends with every task served equally. ``None`` leaves the
    supply unbounded, the process as first defined, under which a longer stream is a superset of
    a shorter one.

    Up to ``concurrency`` tasks are *active*. Each step samples one active task
    uniformly. An active task runs for a spell (exponential with mean ``burst_mean``,
    or exactly that long when fixed, or forever when ``None``), then goes *dormant*
    for a log-normal number of steps, after which it is *ready* to be re-activated;
    with no dormancy median it is finished instead. Never-seen tasks arrive as a
    Poisson process with mean interval ``arrival_interval`` (at once when ``0``) and
    take priority for a free slot when due. When no task is ready and no arrival is
    due, the soonest-waking dormant task is woken early so the stream never idles;
    unseen tasks are only pulled ahead of schedule when nothing else exists.

    ``order`` names the point on the axis (it seeds the generator and labels the
    schedule, under the key ``mode`` that saved schedules and run records carry), so the
    same seed gives different streams in different orders and the same stream under the
    same parameters.
    """
    params = params or StreamParams()
    params.validate()
    tasks = sorted(set(task_ids))
    if not tasks:
        raise ValueError("stream_schedule needs at least one task")
    rng = random.Random(derive_seed(seed, order))
    if instances_per_task is not None:
        if instances_per_task < 1:
            raise ValueError("instances_per_task must be >= 1")
        if length != len(tasks) * instances_per_task:
            raise ValueError(
                f"length {length} is not {len(tasks)} tasks x {instances_per_task} instances: with a supply "
                "every task is served exactly instances_per_task times")
    supply: dict[str, int] = {tid: (math.inf if instances_per_task is None else instances_per_task) for tid in tasks}

    unseen = list(tasks)
    rng.shuffle(unseen)
    concurrency = len(tasks) if params.concurrency is None else params.concurrency
    active: dict[str, float] = {}  # task -> remaining spell
    spell_start: dict[str, int] = {}  # task -> step at which its current spell began
    spells: list[list] = []  # [task, first step, one past the last step] of every spell, in order of start
    dormant: dict[str, int] = {}  # task -> step at which it becomes ready
    ready: list[str] = []
    next_arrival = 0.0  # absolute step at which the next new task is due
    log_median = None if params.dormancy_median is None else math.log(params.dormancy_median)

    def sample_burst(tid: str) -> float:
        if params.burst_mean is None:
            burst = math.inf
        elif params.burst_fixed:
            burst = params.burst_mean
        else:
            burst = max(1, int(rng.expovariate(1.0 / params.burst_mean) + 0.5))
        return min(burst, supply[tid])  # a spell never exceeds what the task has left

    def sample_dormancy() -> int:
        return max(1, int(rng.lognormvariate(log_median, params.dormancy_sigma) + 0.5))

    def introduce(t: int) -> str:
        nonlocal next_arrival
        tid = unseen.pop()
        next_arrival = t if params.arrival_interval == 0 else t + rng.expovariate(1.0 / params.arrival_interval)
        return tid

    slots: list[Slot] = []
    for t in range(length):
        for tid in sorted(k for k, wake in dormant.items() if wake <= t):
            del dormant[tid]
            ready.append(tid)
        while len(active) < concurrency:
            if unseen and t >= next_arrival:
                tid = introduce(t)  # a new task is due: it takes priority
            elif ready:
                tid = ready.pop(rng.randrange(len(ready)))
            elif dormant:
                tid = min(sorted(dormant), key=dormant.__getitem__)  # wake the soonest early
                del dormant[tid]
            elif unseen:
                tid = introduce(t)  # nothing else exists yet (only at the very start)
            else:
                break
            active[tid] = sample_burst(tid)
            spell_start[tid] = t
        if not active:
            raise RuntimeError(f"stream stalled at step {t}: every task is finished")
        tid = rng.choice(sorted(active))
        slots.append(Slot(tid, _slot_seed(seed, PHASE_STREAM, t), PHASE_STREAM))
        active[tid] -= 1
        supply[tid] -= 1
        if active[tid] <= 0:
            del active[tid]
            spells.append([tid, spell_start.pop(tid), t + 1])
            if log_median is not None and supply[tid] > 0:
                dormant[tid] = t + 1 + sample_dormancy()
    for tid in sorted(active):  # spells still open when the stream ends
        spells.append([tid, spell_start[tid], length])
    spells.sort(key=lambda x: (x[1], x[0]))

    return Schedule(
        name=name or order,
        seed=seed,
        slots=slots,
        sources={tid: source for tid in tasks},
        meta={
            "generator": "stream",
            "mode": order,
            "length": length,
            "n_tasks": len(tasks),
            "params": asdict(params),
            "instances_per_task": instances_per_task,
            "n_tasks_introduced": len(tasks) - len(unseen),
            "spells": spells,  # when each task was active: [task, first step, one past the last step]
        },
    )


def epilogue_schedule(
    task_ids: Iterable[str],
    seed: int,
    source: str,
    instances_per_task: int = 1,
    reset_each: bool = False,
    name: str = "epilogue",
) -> Schedule:
    """Held-out tasks, each seen ``instances_per_task`` times, in shuffled order.

    With ``reset_each`` the learner's state is wiped before every task, which is
    the reference condition for measuring forward transfer.
    """
    tasks = sorted(set(task_ids))
    rng = random.Random(derive_seed(seed, "epilogue"))
    order = list(tasks)
    rng.shuffle(order)
    slots: list[Slot] = []
    for tid in order:
        for k in range(instances_per_task):
            slots.append(
                Slot(tid, _slot_seed(seed, PHASE_EPILOGUE, len(slots)), PHASE_EPILOGUE, reset_each and k == 0)
            )
    return Schedule(
        name=name,
        seed=seed,
        slots=slots,
        sources={tid: source for tid in tasks},
        meta={"generator": "epilogue", "n_tasks": len(tasks), "instances_per_task": instances_per_task, "reset_each": reset_each},
    )


def interleave_users(per_user: list[Schedule], seed: int, name: str) -> Schedule:
    """Many users' own schedules as one collective schedule, round by round.

    ``per_user[u]`` is user ``u``'s schedule (a stream, an epilogue, or one then the other).
    Every user's must have the same phases in the same lengths, so that every round holds
    one instance per user and never spans a phase. Round ``r`` is the ``r``-th slot of every
    user, in user order. A wipe belongs to a round, not to a user, since the users of a
    round share the state it began with: a round is marked for a wipe when any user's slot
    is, on its first slot only.
    """
    if not per_user:
        raise ValueError("interleave_users needs at least one user")
    phases = [x.phase for x in per_user[0].slots]
    for u, sched in enumerate(per_user):
        if [x.phase for x in sched.slots] != phases:
            raise ValueError(f"user {u}'s schedule has a different shape of phases from user 0's: "
                             "every user needs the same phases in the same lengths")
    sources: dict[str, str] = {}
    for sched in per_user:
        for tid, src in sched.sources.items():
            if sources.setdefault(tid, src) != src:
                raise ValueError(f"task {tid} assigned to both {sources[tid]} and {src}")
    slots: list[Slot] = []
    for r in range(len(phases)):
        wipe = any(sched.slots[r].reset_before for sched in per_user)
        for u, sched in enumerate(per_user):
            x = sched.slots[r]
            slots.append(Slot(x.task_id, x.instance_seed, x.phase, reset_before=wipe and u == 0, user=u))
    return Schedule(name=name, seed=seed, slots=slots, sources=sources,
                    meta={"generator": "users", "users": len(per_user)}, users=len(per_user))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _median(xs: list[int]) -> float | None:
    return _quantile(xs, 0.5)


def _quantile(xs: list[int], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
