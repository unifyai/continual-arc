"""Named benchmark configurations, the tracks they belong to, and the schedules built
from them.

Continual-ARC has a track for each ARC-AGI edition (:data:`TRACKS`): ARC-AGI-1 (``arc1``), its stream fresh
instances of the ARC-AGI-1 training tasks from Re-ARC's generators; ARC-AGI-2 (``arc2``), its stream fresh instances
of the ARC-AGI-2 training tasks new to the second edition from ARC-GEN's generators (``docs/arc2.md``); and ARC-AGI-3
(``arc3``), its stream fresh levels of its stream games' first-level rules from generators of our own
(:mod:`continual_arc.arc3`, ``docs/arc3.md``). Every configuration name carries its track (``arc1-small``), so a bare
size such as ``small`` names nothing. Each
track has one canonical protocol for each number of users served at once: its ``small`` configuration's stream and
then its two frozen tests, at the default environment settings, at seeds 0, 1 and 2, at 1, 2, 4, 8 or 16 users
(:data:`CANONICAL_USERS`, :func:`canonical_protocol`); learners are compared only on the same track at the same
number of users. Every other configuration, order, seed, number of users or setting is an ablation and is labeled
as such in run metadata and reports (:func:`protocol_of`).

A configuration fixes the number of tasks, the stream length, the stream's parameters, the size of the held-out
epilogue and the environment settings. From one configuration and one seed, :func:`build_schedule` produces its
stream. One process generates every stream (:func:`continual_arc.schedule.stream_schedule`), and the benchmark's is
one setting of its parameters, ``workstreams``: a few tasks active at once in bursts, dormancy between, new tasks
arriving. Two other settings serve the same visits over the *same* task set in other orders, only to check that the
order does not change a finding (the paper's Appendix I), and a run in either is an ablation:

* ``sequential`` — one task at a time, every instance of a task before the next (blocked order).
* ``random``     — every task active from the first step, sampled uniformly (no temporal structure).

An order is an order of presentation and nothing else; the environment, the interface and every metric are the same
under each.

The stream may be followed by the epilogue of never-seen evaluation tasks, met while the learner still learns, which
no measure reads since the frozen tests replaced it. ``fresh`` is not an order but the epilogue tasks alone, met by a
learner whose state is wiped before each, which serves as the held-out test's reset reference.

After the stream the system is measured by two frozen tests (:func:`test_schedule`; Dan, 21 Oct 2026): the
held-out tasks, and new instances of the stream's own tasks, every instance played from the system's state at the
stream's end with nothing learnt between them, so that what a system learnt is measured apart from its learning. The
held-out test measures what carries to tasks it never met, the seen test what it kept of the tasks it did; both are
read against the same two references as transfer (:func:`continual_arc.metrics.transfer`).

A configuration is served to a number of users (:func:`with_users`): how many people the
system serves at once, and so how much experience arrives between two updates. One user is
the individual regime, one system walking one person's stream one instance at a time. More
is the collective regime: one system serving a team, every user walking a stream of their
own over the same tasks, and every round one instance per user, served concurrently and
acted on with the state the system had when the round began, with updates between rounds.
The canonical numbers of users are the powers of two from 1 to 16 (Dan, 18 Oct 2026): a
low-data regime by design, every update seeing at most 16 new instances, so that a learner
that trains its weights between rounds finishes a whole stream on one machine in days, not
weeks.

A configuration's ``stream_length`` is an observation window, not a ceiling on
what a learner may learn from: the generators produce fresh instances forever.
Every task gets the same supply of instances, the stream length over the number
of tasks, in every order (:meth:`BenchmarkConfig.instances_per_task`), so the three orders serve one multiset of
instances and a configuration's stream depends on its length: a
longer configuration is a different schedule, not a superset. ``checkpoints`` name
the horizons within a run at which the score is also reported; a checkpoint read is
a prefix of that run and never changes when the run is carried further.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace

from continual_arc.arc3.env import Arc3Config
from continual_arc.arc3.games import HELD_OUT_GAMES, SOURCE_ARC3, STREAM_GAMES
from continual_arc.data import DATASET_ARC1, DATASET_ARC2, DATASET_ARC3
from continual_arc.env import EnvConfig
from continual_arc.schedule import (
    PHASE_SEEN_TEST,
    RANDOM,
    TEST_HELD_OUT,
    TEST_PHASE,
    TEST_SEEN,
    TESTS,
    Schedule,
    Slot,
    StreamParams,
    epilogue_schedule,
    interleave_users,
    sequential_params,
    stream_schedule,
)
from continual_arc.seeds import derive_seed
from continual_arc.tasks import SOURCE_ARC2_EVAL, SOURCE_ARC_EVAL, SOURCE_ARCGEN, SOURCE_REARC

TRACKS = (DATASET_ARC1, DATASET_ARC2, DATASET_ARC3)
# each track's sources: its stream's (and seen test's), and its held-out tasks'
_STREAM_SOURCE = {DATASET_ARC1: SOURCE_REARC, DATASET_ARC2: SOURCE_ARCGEN, DATASET_ARC3: SOURCE_ARC3}
_HELD_OUT_SOURCE = {DATASET_ARC1: SOURCE_ARC_EVAL, DATASET_ARC2: SOURCE_ARC2_EVAL, DATASET_ARC3: SOURCE_ARC3}
ORDER_WORKSTREAMS = "workstreams"  # the benchmark's stream
ORDER_SEQUENTIAL = "sequential"
ORDER_RANDOM = "random"
ORDERS = (ORDER_WORKSTREAMS, ORDER_SEQUENTIAL, ORDER_RANDOM)
FRESH = "fresh"  # the transfer reference: epilogue only, state wiped before each task
REGIME_INDIVIDUAL = "individual"  # one user: one system, one person's stream, one instance at a time
REGIME_COLLECTIVE = "collective"  # many users: one system serving a team, one instance per user per round


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    n_tasks: int
    stream_length: int
    """Instances in each user's stream (the whole stream in the individual regime)."""
    n_epilogue_tasks: int
    workstreams: StreamParams = field(default_factory=StreamParams)
    """The stream's point on the stream axis. The check's orders, sequential and random, are fixed
    functions of ``n_tasks`` and ``stream_length`` (:meth:`stream_params`)."""
    users: int = 1
    """People served at once (:func:`with_users`): 1 is the individual regime; more is the collective, where every
    user walks a stream of their own over the configuration's tasks (``stream_length``
    instances each, every task ``instances_per_task`` times), then meets the held-out tasks
    once each in an order of their own, and a round is one instance per user
    (:attr:`continual_arc.schedule.Schedule.users`). User 0 at seed ``s`` walks exactly the
    stream and epilogue one user alone would at seed ``s``."""
    checkpoints: tuple[int, ...] = ()
    """Stream horizons, in instances over all users, at which the score is also reported:
    prefix reads of the run, the same whatever horizon the run was carried to."""
    env: EnvConfig | Arc3Config = field(default_factory=EnvConfig)
    """:class:`EnvConfig` for ``arc1`` and ``arc2``, :class:`Arc3Config` for ``arc3``."""
    dataset: str = DATASET_ARC1
    """The track: ``arc1``, whose stream draws on Re-ARC generators over the ARC-AGI-1
    training tasks and whose held-out tasks are the ARC-AGI-1 evaluation tasks; ``arc2``, whose stream draws on
    ARC-GEN's generators over the ARC-AGI-2 training tasks new to the second edition that passed the screen and whose
    held-out tasks are the ARC-AGI-2 evaluation tasks new to it; or ``arc3``, whose stream draws on the track's
    stream games and whose held-out tasks are its held-out games, every instance a fresh level of a game's first level
    (:mod:`continual_arc.arc3.games`)."""
    held_out_per_task: int = 1
    """Instances of each held-out task in the epilogue and the held-out test: one of each ARC-AGI-1 or ARC-AGI-2
    evaluation task, which has its own pairs alone, and several fresh levels of each held-out ARC-AGI-3 game, which
    are few."""

    def __post_init__(self) -> None:
        if self.dataset not in TRACKS:
            raise ValueError(f"dataset must be one of {TRACKS}, got {self.dataset!r}")
        want = Arc3Config if self.dataset == DATASET_ARC3 else EnvConfig
        if not isinstance(self.env, want):
            raise ValueError(f"track {self.dataset!r} needs env of type {want.__name__}")
        if self.stream_length % self.n_tasks:
            raise ValueError("stream_length must be a multiple of n_tasks so the sequential order gives every task the "
                             "same count")
        if self.users < 1:
            raise ValueError("users must be >= 1")
        total = self.stream_length * self.users
        if list(self.checkpoints) != sorted(self.checkpoints) or any(k < 1 or k > total for k in self.checkpoints):
            raise ValueError("checkpoints must be increasing instance counts within the stream")

    @property
    def regime(self) -> str:
        return REGIME_INDIVIDUAL if self.users == 1 else REGIME_COLLECTIVE

    @property
    def instances_per_task(self) -> int:
        """Each task's supply in one user's stream: the stream length shared equally, served
        in every order and in one block per task in the sequential order.
        Over all users a task is met ``users`` times as often."""
        return self.stream_length // self.n_tasks

    def stream_params(self, order: str) -> StreamParams:
        if order == ORDER_WORKSTREAMS:
            return self.workstreams
        if order == ORDER_SEQUENTIAL:
            return sequential_params(self.instances_per_task)
        if order == ORDER_RANDOM:
            return RANDOM
        raise ValueError(f"{order!r} is not an order of the stream; choose from {ORDERS}")

    @property
    def track(self) -> str:
        """The ARC-AGI edition this configuration belongs to (same as ``dataset``)."""
        return self.dataset

    @property
    def stream_source(self) -> str:
        return _STREAM_SOURCE[self.dataset]

    @property
    def epilogue_source(self) -> str:
        return _HELD_OUT_SOURCE[self.dataset]

    def to_json(self) -> dict:
        d = asdict(self)
        if self.held_out_per_task == 1:
            del d["held_out_per_task"]   # a grid track's schedule files are unchanged by the field
        return d


CONFIGS: dict[str, BenchmarkConfig] = {
    "arc1-smoke": BenchmarkConfig(
        name="arc1-smoke",
        n_tasks=8,
        stream_length=120,
        n_epilogue_tasks=8,
        workstreams=StreamParams(
            concurrency=3, burst_mean=5.0, dormancy_median=20.0, dormancy_sigma=0.8, arrival_interval=10.0
        ),
    ),
    "arc1-small": BenchmarkConfig(
        name="arc1-small",
        n_tasks=25,
        stream_length=200,
        n_epilogue_tasks=100,
        workstreams=StreamParams(
            concurrency=4, burst_mean=3.0, dormancy_median=30.0, dormancy_sigma=1.0, arrival_interval=2.0
        ),
    ),
    "arc1-medium": BenchmarkConfig(
        name="arc1-medium",
        n_tasks=100,
        stream_length=3000,
        n_epilogue_tasks=100,
        workstreams=StreamParams(
            concurrency=5, burst_mean=6.0, dormancy_median=120.0, dormancy_sigma=1.0, arrival_interval=20.0
        ),
    ),
    "arc1-full": BenchmarkConfig(
        name="arc1-full",
        n_tasks=400,
        stream_length=12000,
        n_epilogue_tasks=400,
        workstreams=StreamParams(
            concurrency=6, burst_mean=6.0, dormancy_median=300.0, dormancy_sigma=1.0, arrival_interval=20.0
        ),
    ),
    # ARC-AGI-2: the first track's smoke and small shapes, the stream from the screened ARC-GEN pool and the held-out
    # tasks from the 114 evaluation tasks new to the second edition
    "arc2-smoke": BenchmarkConfig(
        name="arc2-smoke",
        n_tasks=8,
        stream_length=120,
        n_epilogue_tasks=8,
        workstreams=StreamParams(
            concurrency=3, burst_mean=5.0, dormancy_median=20.0, dormancy_sigma=0.8, arrival_interval=10.0
        ),
        dataset=DATASET_ARC2,
    ),
    "arc2-small": BenchmarkConfig(
        name="arc2-small",
        n_tasks=25,
        stream_length=200,
        n_epilogue_tasks=100,
        workstreams=StreamParams(
            concurrency=4, burst_mean=3.0, dormancy_median=30.0, dormancy_sigma=1.0, arrival_interval=2.0
        ),
        dataset=DATASET_ARC2,
    ),
    # ARC-AGI-3: the track's stream games, eight fresh levels each, as arc1-small serves its tasks, the stream's
    # parameters scaled to its length; four fresh levels of each held-out game.
    "arc3-smoke": BenchmarkConfig(
        name="arc3-smoke",
        n_tasks=2,
        stream_length=4,
        n_epilogue_tasks=1,
        held_out_per_task=2,
        workstreams=StreamParams(
            concurrency=1, burst_mean=2.0, dormancy_median=2.0, dormancy_sigma=0.5, arrival_interval=1.0
        ),
        env=Arc3Config(),
        dataset=DATASET_ARC3,
    ),
    "arc3-small": BenchmarkConfig(
        name="arc3-small",
        n_tasks=len(STREAM_GAMES),
        stream_length=8 * len(STREAM_GAMES),
        n_epilogue_tasks=len(HELD_OUT_GAMES),
        held_out_per_task=4,
        workstreams=StreamParams(
            concurrency=3, burst_mean=3.0, dormancy_median=16.0, dormancy_sigma=1.0, arrival_interval=2.0
        ),
        env=Arc3Config(),
        dataset=DATASET_ARC3,
    ),
}


CANONICAL_SEEDS = (0, 1, 2)
CANONICAL_SIZE = "small"
# The numbers of users a canonical score is read at, each a number of its own: powers of two up to 16, a low-data
# regime by design that one machine finishes in days (Dan, 18 Oct 2026)
CANONICAL_USERS = (1, 2, 4, 8, 16)


def canonical_config_name(track: str) -> str:
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    return f"{track}-{CANONICAL_SIZE}"


def canonical_protocol(track: str) -> dict:
    """The protocol whose score is *the* number for a track, one number for each canonical number of users."""
    return {
        "track": track,
        "config": canonical_config_name(track),
        "mode": ORDER_WORKSTREAMS,
        "users": list(CANONICAL_USERS),
        "tests": list(TESTS),
        "seeds": list(CANONICAL_SEEDS),
        "env": get_config(canonical_config_name(track)).env.to_json(),
    }


# the seen test's instances of each stream task: arc1-small's 25 tasks give 100, as many as the held-out test's
SEEN_TEST_PER_TASK = 4


def test_schedule(
    config: str | BenchmarkConfig,
    seed: int,
    test: str,
    width: int | None = None,
    rearc_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
) -> Schedule:
    """One of the two frozen tests after ``config``'s stream at ``seed`` (the module docstring).

    ``held-out``: the held-out tasks, each ``held_out_per_task`` times, at the instances user 0 meets in the epilogue
    (:func:`epilogue_for_users`), so the transfer references serve it as they are. ``seen``:
    :data:`SEEN_TEST_PER_TASK` new instances of each of the stream's tasks, at instance seeds of their
    own (``derive_seed(seed, "seen-test", i)``), which no stream slot draws, in a seeded order.

    Every instance is played from the system's state at the stream's end and nothing passes from one
    to another. ``width`` users play at once, every instance by default: one round, each user's own
    state the end state. With fewer, each user plays several and restores the end state before each
    (``reset_before``), which is what a frozen test's learner does when it resets. The test is the
    same at any width: only how many instances play at once differs."""
    cfg = get_config(config)
    if test == TEST_HELD_OUT:
        epilogue = epilogue_for_users(cfg, seed, 1, rearc_task_ids, eval_task_ids)
        drawn = [(s.task_id, s.instance_seed) for s in epilogue.slots]
        sources = dict(epilogue.sources)
    elif test == TEST_SEEN:
        tasks = sorted(config_tasks(cfg, rearc_task_ids if rearc_task_ids is not None else stream_task_pool(cfg),
                                    seed))
        order = [tid for tid in tasks for _ in range(SEEN_TEST_PER_TASK)]
        random.Random(derive_seed(seed, PHASE_SEEN_TEST)).shuffle(order)
        drawn = [(tid, derive_seed(seed, PHASE_SEEN_TEST, i)) for i, tid in enumerate(order)]
        sources = {tid: cfg.stream_source for tid in tasks}
    else:
        raise ValueError(f"test must be {TEST_HELD_OUT!r} or {TEST_SEEN!r}, got {test!r}")
    n = len(drawn)
    width = n if width is None else width
    if not 1 <= width <= n or n % width:
        raise ValueError(f"width must divide the test's {n} instances, got {width}")
    slots = [Slot(tid, instance_seed, TEST_PHASE[test], reset_before=width < n, user=i % width)
             for i, (tid, instance_seed) in enumerate(drawn)]
    return Schedule(name=f"{cfg.name}-{test}-test", seed=seed, slots=slots, sources=sources, users=width,
                    meta={"config": cfg.to_json(), "seed": seed, "dataset": cfg.dataset, "track": cfg.track,
                          "test": test, "width": width})


def protocol_of(schedule, env_config=None) -> dict:
    """Describe a schedule's protocol and say whether it is the canonical one.

    Returns ``{"track", "config", "mode", "regime", "users", "seed", "epilogue",
    "canonical", "deviations"}``, ``mode`` naming the order under the key run records carry.
    ``deviations`` lists, in words, every way the run departs
    from the track's canonical protocol; it is empty when ``canonical`` is true. A schedule cut
    short (:meth:`continual_arc.schedule.Schedule.first_rounds`) or built as a K-user epilogue
    alone (:func:`epilogue_for_users`) is always a deviation, named by round or by user count.
    ``env_config`` is the environment configuration the run actually used; when
    given it is compared with the configuration's own.
    """
    meta = schedule.meta or {}
    cfg_meta = meta.get("config") or {}
    track = meta.get("dataset") or cfg_meta.get("dataset") or DATASET_ARC1
    name = cfg_meta.get("name") or schedule.name
    order = meta.get("mode")
    seed = meta.get("seed", schedule.seed)
    if meta.get("test"):   # a frozen test is canonical when its configuration and seed are
        deviations = ([f"configuration {name} is not {canonical_config_name(track)}"]
                      if name != canonical_config_name(track) else [])
        deviations += [f"seed {seed} is not one of the canonical seeds {CANONICAL_SEEDS}"] if seed not in CANONICAL_SEEDS else []
        return {"track": track, "config": name, "test": meta["test"], "width": meta.get("width", schedule.users),
                "seed": seed, "canonical": not deviations, "deviations": deviations}
    epilogue = bool(meta.get("epilogue", False))
    epilogue_only = meta.get("epilogue_only")
    cut_to_round = meta.get("cut_to_round")
    # an epilogue-alone schedule's own width (K), not the number of users the configuration is served to
    users = (int(epilogue_only["users"]) if epilogue_only
            else int(cfg_meta.get("users", cfg_meta.get("width", schedule.users))))  # "width" before 24 Sep 2026
    deviations: list[str] = []
    if name != canonical_config_name(track):
        deviations.append(f"configuration {name} is not {canonical_config_name(track)}")
    if users not in CANONICAL_USERS:
        deviations.append(f"{users} users is not one of the canonical numbers of users {CANONICAL_USERS}")
    if not epilogue_only and order != ORDER_WORKSTREAMS:
        deviations.append(f"order {order} is not the stream's own, {ORDER_WORKSTREAMS}")
    if seed not in CANONICAL_SEEDS:
        deviations.append(f"seed {seed} is not one of the canonical seeds {CANONICAL_SEEDS}")
    if cut_to_round is not None:
        deviations.append(f"stream cut at round {cut_to_round}")
    if epilogue_only:
        deviations.append(f"epilogue alone, users 0–{epilogue_only['users'] - 1} of {epilogue_only['of']}")
    if env_config is not None and cfg_meta.get("env") is not None:
        actual = env_config.to_json()
        for key, value in cfg_meta["env"].items():
            if key in actual and actual[key] != value:
                deviations.append(f"{key} = {actual[key]} instead of {value}")
    return {
        "track": track, "config": name, "mode": order,
        "regime": REGIME_INDIVIDUAL if users == 1 else REGIME_COLLECTIVE, "users": users,
        "seed": seed, "epilogue": epilogue, "canonical": not deviations, "deviations": deviations,
    }


def get_config(name_or_config: str | BenchmarkConfig) -> BenchmarkConfig:
    if isinstance(name_or_config, BenchmarkConfig):
        return name_or_config
    try:
        return CONFIGS[name_or_config]
    except KeyError:
        pass
    sizes = sorted({n.split("-", 1)[1] for n in CONFIGS})
    if name_or_config in sizes:
        raise KeyError(
            f"config {name_or_config!r} does not name its track; "
            f"choose {', '.join(f'{t}-{name_or_config}' for t in TRACKS if f'{t}-{name_or_config}' in CONFIGS)}"
        )
    raise KeyError(f"unknown config {name_or_config!r}; choose from {sorted(CONFIGS)}")


def with_users(config: str | BenchmarkConfig, users: int) -> BenchmarkConfig:
    """``config`` served to ``users`` people at once: every user walks a stream of their own over the
    configuration's tasks, and user ``u`` walks the same stream whatever the number of users (user 0 at
    seed ``s`` exactly the stream one user alone walks at ``s``), so a run at fewer users is a run at more
    cut to its first users. Its checkpoints, horizons in instances over all users, scale with them."""
    cfg = get_config(config)
    return replace(cfg, users=users, checkpoints=tuple(k // cfg.users * users for k in cfg.checkpoints))


def select_tasks(pool: list[str], n: int, seed: int, label: str) -> list[str]:
    """Seeded sample of ``n`` task ids from ``pool`` (sorted first, for determinism)."""
    pool = sorted(set(pool))
    if n > len(pool):
        raise ValueError(f"asked for {n} tasks but pool has {len(pool)}")
    rng = random.Random(derive_seed(seed, label))
    return sorted(rng.sample(pool, n))


def _size_chain(cfg: BenchmarkConfig, attr: str) -> list[int]:
    """The nested sizes of ``cfg``'s track, ascending: the ``attr`` of every configuration of
    the same dataset but the smoke ones, whose draws stand alone (they are never in a paper)."""
    sizes = {getattr(c, attr) for c in CONFIGS.values() if c.dataset == cfg.dataset and not c.name.endswith("-smoke")}
    return sorted(sizes)


def select_nested(pool: list[str], n: int, seed: int, label: str, chain: list[int]) -> list[str]:
    """``n`` task ids from ``pool`` such that each size in ``chain`` contains every smaller one.

    The smallest size is the plain seeded sample :func:`select_tasks` would draw, so fixing
    the sets this way changes nothing for the small configurations already run; each larger
    size adds a seeded sample from what the smaller sizes left. A size below the chain (a smoke
    configuration) is drawn on its own.
    """
    chain = [s for s in chain if s <= n]
    if not chain:
        return select_tasks(pool, n, seed, label)
    chosen: list[str] = []
    remaining = sorted(set(pool))
    for size in chain:
        k = size - len(chosen)
        chosen += select_tasks(remaining, k, seed, label if not chosen else f"{label}-{size}")
        taken = set(chosen)
        remaining = [t for t in remaining if t not in taken]
    if len(chosen) < n:
        chosen += select_tasks(remaining, n - len(chosen), seed, f"{label}-{n}")
    return sorted(chosen)


def config_tasks(cfg: BenchmarkConfig, pool: list[str], seed: int) -> list[str]:
    """The stream task set of ``cfg`` at ``seed`` from ``pool``: nested across the sizes of its track."""
    return select_nested(pool, cfg.n_tasks, seed, "tasks", _size_chain(cfg, "n_tasks"))


def stream_task_pool(cfg: BenchmarkConfig) -> list[str]:
    """Every task id the stream of ``cfg`` may draw on: on ``arc1`` the ARC-AGI-1 training tasks, through their
    Re-ARC generators; on ``arc2`` the screened pool of ARC-AGI-2 training tasks, through their ARC-GEN generators;
    on ``arc3`` the track's stream games."""
    if cfg.dataset == DATASET_ARC2:
        from continual_arc.arcgen import pool

        return pool()
    if cfg.dataset == DATASET_ARC3:
        return list(STREAM_GAMES)
    from continual_arc.rearc import ReArc

    return ReArc().task_ids


def epilogue_task_pool(cfg: BenchmarkConfig) -> list[str]:
    """Every task id the held-out tasks of ``cfg`` may be drawn from: the ARC-AGI-1 evaluation tasks, the ARC-AGI-2
    evaluation tasks new to the second edition, or the ARC-AGI-3 track's held-out games."""
    if cfg.dataset == DATASET_ARC3:
        return list(HELD_OUT_GAMES)
    from continual_arc.data import arc2_new_task_ids, arc_task_ids

    if cfg.dataset == DATASET_ARC2:
        return arc2_new_task_ids("evaluation")
    return arc_task_ids("evaluation", dataset=cfg.dataset)


def _held_out_tasks(cfg: BenchmarkConfig, seed: int, eval_task_ids: list[str]) -> list[str]:
    """The configuration's held-out tasks at ``seed``, drawn from ``eval_task_ids``."""
    return select_nested(eval_task_ids, cfg.n_epilogue_tasks, seed, "epilogue", _size_chain(cfg, "n_epilogue_tasks"))


def _user_seeds(cfg: BenchmarkConfig, seed: int) -> list[int]:
    """One seed per user a configuration is served to: user 0 keeps ``seed`` itself, so it walks
    exactly the stream and epilogue one user alone would at that seed; every other user draws the
    same process under a seed of its own."""
    return [seed] + [derive_seed(seed, "user", u) for u in range(1, cfg.users)]


def build_schedule(
    config: str | BenchmarkConfig,
    seed: int,
    *,
    order: str = ORDER_WORKSTREAMS,
    rearc_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
    epilogue: bool = True,
) -> Schedule:
    """The schedule of ``config`` at ``seed``: the benchmark's stream, then the held-out epilogue unless
    ``epilogue`` is false.

    ``order`` serves the stream's visits in one of the check's orders instead (:data:`ORDERS`), or is
    :data:`FRESH`, the held-out tasks alone with the learner wiped before each. ``rearc_task_ids`` (the
    stream pool) and ``eval_task_ids`` default to the full pinned pools and are only overridable for
    testing. The same ``config``/``seed`` pair selects the same task set in every order, so a check's
    run meets the same visits as the stream's.
    """
    cfg = get_config(config)
    if order not in (*ORDERS, FRESH):
        raise ValueError(f"order must be one of {ORDERS} or {FRESH!r}, got {order!r}")
    if rearc_task_ids is None:
        rearc_task_ids = stream_task_pool(cfg)
    tasks = config_tasks(cfg, rearc_task_ids, seed)
    source = cfg.stream_source
    if eval_task_ids is None:
        eval_task_ids = epilogue_task_pool(cfg)

    def one_user(user_seed: int) -> Schedule:
        """One user's own schedule: the stream process and the epilogue order at ``user_seed``,
        over the tasks and held-out tasks drawn at the configuration's ``seed``."""
        if order == FRESH:
            # the transfer reference: the held-out tasks alone, a fresh learner before each of their instances
            return epilogue_schedule(_held_out_tasks(cfg, seed, eval_task_ids), user_seed, cfg.epilogue_source,
                                     cfg.held_out_per_task, reset_each=True, name=f"{cfg.name}-fresh").reset_each()
        one = stream_schedule(
            tasks, cfg.stream_length, user_seed, source, cfg.stream_params(order), order, name=f"{cfg.name}-{order}",
            instances_per_task=cfg.instances_per_task,
        )
        if epilogue and cfg.n_epilogue_tasks > 0:
            epi = epilogue_schedule(_held_out_tasks(cfg, seed, eval_task_ids), user_seed, cfg.epilogue_source,
                                    cfg.held_out_per_task, reset_each=False, name="epilogue")
            one = one.concat(epi, name=one.name)
        return one

    user_meta: dict = {}
    if cfg.users == 1:
        sched = one_user(seed)
        parts = sched.meta.get("parts", [sched.meta])
    else:
        # User 0 keeps the configuration's own seed, so it walks exactly the stream one user alone
        # would; every other user draws the same process with a seed of their own.
        user_seeds = _user_seeds(cfg, seed)
        per_user = [one_user(s) for s in user_seeds]
        sched = interleave_users(per_user, seed, name=f"{cfg.name}-{order}")
        parts = []  # each user's own parts (spells included) are under "users"
        user_meta = {"users": {"n": cfg.users, "seeds": user_seeds,
                               "parts": [x.meta.get("parts", [x.meta]) for x in per_user]}}
    sched.meta = {
        "config": cfg.to_json(),
        "mode": order,
        "seed": seed,
        "dataset": cfg.dataset,
        "track": cfg.track,
        "epilogue": bool(epilogue and cfg.n_epilogue_tasks > 0),
        "pools": {"stream": len(rearc_task_ids), "epilogue": len(eval_task_ids)},
        "parts": parts,
        **user_meta,
    }
    return sched


def epilogue_for_users(
    config: str | BenchmarkConfig,
    seed: int,
    users: int,
    rearc_task_ids: list[str] | None = None,
    eval_task_ids: list[str] | None = None,
) -> Schedule:
    """The held-out tasks alone, served to the first ``users`` of ``config``'s users, as a
    ``users``-wide collective schedule: user ``u`` (0 to ``users - 1``) meets them in the exact
    order and at the exact instance seeds it would in the full schedule's own epilogue
    (:func:`build_schedule`), because both draw the same per-user seed (:func:`_user_seeds`) and
    the same held-out task selection (:func:`_held_out_tasks`) from ``config`` and ``seed`` alone.
    The learner is not reset between tasks (``reset_each=False``), as in the full schedule's own
    epilogue: nothing is frozen for it.

    Lets transfer be read from a minority of a collective run's users without playing
    out the whole stream to every one of them, which is when the full schedule's own epilogue is
    due (:meth:`continual_arc.schedule.Schedule.first_rounds` stops the stream itself short).
    """
    cfg = get_config(config)
    if cfg.n_epilogue_tasks <= 0:
        raise ValueError(f"{cfg.name} has no epilogue")
    if not 1 <= users <= cfg.users:
        raise ValueError(f"users must be between 1 and {cfg.name}'s {cfg.users} users, got {users}")
    if rearc_task_ids is None:
        rearc_task_ids = stream_task_pool(cfg)
    if eval_task_ids is None:
        eval_task_ids = epilogue_task_pool(cfg)
    held_out = _held_out_tasks(cfg, seed, eval_task_ids)
    user_seeds = _user_seeds(cfg, seed)[:users]
    per_user = [epilogue_schedule(held_out, s, cfg.epilogue_source, cfg.held_out_per_task, reset_each=False,
                                  name="epilogue") for s in user_seeds]
    sched = interleave_users(per_user, seed, name=f"{cfg.name}-epilogue-{users}")
    sched.meta = {
        "config": cfg.to_json(),
        "mode": None,
        "seed": seed,
        "dataset": cfg.dataset,
        "track": cfg.track,
        "epilogue": True,
        "epilogue_only": {"users": users, "of": cfg.users},
        "pools": {"stream": len(rearc_task_ids), "epilogue": len(eval_task_ids)},
    }
    return sched
