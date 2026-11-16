"""Run a learner through a schedule and write a run directory.

A run directory contains::

    schedule.json   the exact schedule served
    config.json     the EnvConfig used
    meta.json       the learner's identity, the run's own settings, timestamps, package version, every resume
    records.jsonl   one InstanceRecord per instance, in order, by whole rounds
    learner-state.pkl  the learner's state at the last round boundary, for a learner that gives one
    summary.json    metrics computed by continual_arc.metrics.summarize
    round.jsonl     a frozen test's instances that ended in the round still open
    events.jsonl    (optional) every action and feedback, including grids
    plugin-events.jsonl  (optional) events emitted by plug-ins

The runner is the only place that calls ``learner.reset()``; it does so exactly
when the schedule slot says so. It also guards against runaway learners with a
per-instance action limit, after which the instance is force-abandoned.

A learner is shown the instance alone (:class:`continual_arc.env.LearnerObservation`,
:class:`continual_arc.arc3.env.Arc3LearnerObservation`): the task under the run's alias,
salted in the run directory's ``alias-salt``, and nothing about the schedule. Plug-ins, events and records see the slot
(:class:`continual_arc.env.Observation`), real task id, phase and exposure included.

Instances are served in rounds, one instance per user of the schedule. The runner
calls ``learner.begin_round`` before the round's first action and ``learner.end_round``
after its last instance ends, and writes the round's records together, in slot order,
so ``records.jsonl`` only ever holds whole rounds. With one user every instance is a
round of one and this is the serial loop.

Every run resumes as though it had never stopped. A round's end saves it: the runner writes the round's records (the
events of every action before them), the learner ends the round, and the runner keeps the learner's state
(``learner.state()``, :mod:`continual_arc.learner`) beside them, whole or not at all, as it does once as the run
starts. ``run(..., resume=True)`` in the run's own directory goes on from its saved point (:func:`saved_point`): its
whole rounds of records back to the last round boundary its learner's state was kept at, that state given back to the
learner (``restore``), with the run's own settings, any other refused (:func:`refuse_other_settings`). Its records are
cut back on disk to that point, whole or not at all, a last line a stop cut off and the round it stopped in with them,
and added to, so that no row comes twice; the events of every instance played again go too, and each resume is
recorded in ``meta.json``'s ``resumed_at``. A schedule that serves every slot of the run's as the run's did may go on
past it, so that a driver can play a cut of a collective stream a round longer at each call. A frozen test
(``schedule.meta["test"]``) keeps more: every one of its instances starts from the learner as its stream left it and
nothing one does reaches another, so each instance is kept as it ends (``round.jsonl``) and a stopped test plays only
the instances it had not ended. A fresh run never writes over a directory that holds a run.

With many users (the collective regime) the users of a round work at the same time. A
learner that can serve them at once sets ``concurrent_users = True``: every user's
instance is then served on a thread of its own, from its first action to its verdict, at
its own pace, with the environment behind a lock, and the round ends when every user's
has. Such a learner reports its counters per user (``user_resources(user)``), and an
instance's resources are that user's counter deltas over its actions. Any other learner
is served one action at a time: one action on every open instance before a second on
any, an instance's resources the learner's counter deltas over its own actions.

The same loop drives :class:`continual_arc.env.ContinualArcEnv` and
:class:`continual_arc.arc3.env.Arc3Env`; :func:`build_env` picks one by the schedule's track,
or pass a ready ``env``.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from continual_arc._version import __version__
from continual_arc.addons import Addon, Plugin, PluginManager, PluginRuntime
from continual_arc.aliases import TaskAliases
from continual_arc.arc3.env import Arc3Config, Arc3Env, Arc3LearnerObservation, Arc3Observation
from continual_arc.arc3.instances import LevelSource
from continual_arc.configs import protocol_of
from continual_arc.data import DATASET_ARC3
from continual_arc.env import (
    ContinualArcEnv,
    EnvConfig,
    GiveUp,
    InstanceRecord,
    LearnerObservation,
    Observation,
)
from continual_arc.learner import Learner
from continual_arc.metrics import summarize
from continual_arc.schedule import TESTS, Schedule
from continual_arc.tasks import TaskRegistry

ProgressFn = Callable[[InstanceRecord, int], None]

STATE = "learner-state.pkl"
"""The learner's state at the run's last round boundary and how many records it follows (:func:`keep_state`)."""
OPEN_ROUND = "round.jsonl"
"""A frozen test's instances that ended in the round still open, one record per line."""
OWN = ("learner", "alias_style", "max_actions_per_instance", "log_events", "settings")
"""What of a run's ``meta.json`` a resume keeps as it was, beside its configuration and its schedule
(:func:`own_settings`)."""


def holds_run(out_dir: str | Path) -> bool:
    """Whether ``out_dir`` holds a run, which only a resume goes on with."""
    return (Path(out_dir) / "meta.json").is_file()


def write_whole(path: Path, data: bytes) -> None:
    """``data`` at ``path`` whole or not at all: written beside it, synced, then renamed over it."""
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("wb") as fp:
        fp.write(data)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(pending, path)


def keep_state(out_dir: str | Path, records: int, state: object) -> None:
    """``state``, a learner's after the run's first ``records`` records, a round boundary, kept in ``out_dir``
    (:data:`STATE`) for a resume to give back."""
    write_whole(Path(out_dir) / STATE, pickle.dumps({"records": records, "state": state}))


def own_settings(config: EnvConfig | Arc3Config, learner_name: str, alias_style: str = "hex",
                 log_events: bool = False, settings: dict | None = None,
                 max_actions_per_instance: int | None = None) -> dict:
    """The settings of a run that its ``meta.json`` keeps and a resume must keep as they were (:data:`OWN`): the
    learner's name, the style of the task names it is shown, the action limit (the configuration's default when None),
    whether every action is logged, and ``settings``, whatever else the driver says a resume must keep, such as the
    learner's own arguments."""
    return {"learner": learner_name, "alias_style": alias_style,
            "max_actions_per_instance": max_actions_per_instance or _default_max_actions(config),
            "log_events": log_events, "settings": settings}


def refuse_other_settings(out_dir: str | Path, schedule: Schedule, config: EnvConfig | Arc3Config, own: dict) -> None:
    """Refuse (ValueError) to go on with the run in ``out_dir`` under any settings but its own: ``schedule`` must serve
    every slot both it and the run's schedule hold exactly as the run's does, to as many users and from the same
    sources, ``config`` must be the run's configuration, and ``own`` (:func:`own_settings`) what the run's ``meta.json``
    recorded of it."""
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text())
    now = json.loads(json.dumps(own))
    differ = [key for key in OWN if key in meta and meta[key] != now[key]]
    if json.loads((out / "config.json").read_text()) != json.loads(json.dumps(config.to_json())):
        differ.append("config")
    played = Schedule.load(out / "schedule.json")
    both = min(len(played.slots), len(schedule.slots))
    if played.users != schedule.users or played.slots[:both] != schedule.slots[:both] or any(
            played.sources[slot.task_id] != schedule.sources[slot.task_id] for slot in schedule.slots[:both]):
        differ.append("schedule")
    if differ:
        raise ValueError(f"{out} goes on only with its own settings, and these differ from them: {', '.join(differ)}")


@dataclass
class SavedPoint:
    """Where the run in a directory goes on from (:func:`saved_point`): its records' lines and records that it keeps,
    whole rounds; a frozen test's instances of the next round that it ended; and its learner's state there, None when
    it kept none."""

    lines: list[str]
    records: list[InstanceRecord]
    finished_lines: list[str]
    finished: list[InstanceRecord]
    state: object


def whole_lines(path: Path) -> list[str]:
    """The lines of the JSON Lines file at ``path`` (none when there is no file), less a last line that a stop cut off
    before its newline."""
    if not path.is_file():
        return []
    return [line for line in path.read_text().split("\n")[:-1] if line.strip()]


def _parsed(path: Path) -> list[tuple[str, InstanceRecord]]:
    """The records of ``path`` that parse, each with its line, up to the first that does not."""
    parsed = []
    for line in whole_lines(path):
        try:
            parsed.append((line, InstanceRecord.from_json(json.loads(line))))
        except (ValueError, KeyError, TypeError):
            break
    return parsed


def saved_point(out_dir: str | Path, schedule: Schedule) -> SavedPoint:
    """What the run in ``out_dir`` saved, to go on from under ``schedule``: its records' whole rounds, back to the round
    boundary its learner's state was last kept at (:data:`STATE`), with that state; and on a frozen test the instances
    of the next round it ended (:data:`OPEN_ROUND`), all but the last of them to end when they are all of it, so that
    the round ends as every round does, its records written by the runner and ended by the learner."""
    out = Path(out_dir)
    parsed = _parsed(out / "records.jsonl")
    n = len(whole_rounds([record for _, record in parsed], schedule))
    state = None
    if (out / STATE).is_file():
        kept = pickle.loads((out / STATE).read_bytes())
        if kept["records"] > n:
            raise ValueError(f"{out}'s learner state was kept after {kept['records']} records, and its records.jsonl "
                             f"holds {n} in whole rounds")
        n, state = kept["records"], kept["state"]
    finished: list[tuple[str, InstanceRecord]] = []
    if schedule.meta.get("test") in TESTS:
        starts = schedule.round_starts()
        end = next((i for i in range(n + 1, len(starts)) if starts[i]), len(starts))
        for line in whole_lines(out / OPEN_ROUND):
            record = InstanceRecord.from_json(json.loads(line))
            if n <= record.step < end and all(record.step != other.step for _, other in finished):
                finished.append((line, record))
        if finished and len(finished) == end - n:
            finished.pop()
    return SavedPoint([line for line, _ in parsed[:n]], [record for _, record in parsed[:n]],
                      [line for line, _ in finished], [record for _, record in finished], state)


def _go_back(out: Path, point: SavedPoint) -> dict:
    """The run's files in ``out`` cut back to ``point``, each whole or not at all: its records, its events but those of
    the instances to be played again, and a frozen test's open round. Returns the resume's entry for ``meta.json``."""
    on_disk = len(whole_lines(out / "records.jsonl"))
    write_whole(out / "records.jsonl", "".join(line + "\n" for line in point.lines).encode())
    kept = set(range(len(point.records))) | {record.step for record in point.finished}
    events = out / "events.jsonl"
    if events.is_file():
        write_whole(events, "".join(line + "\n" for line in whole_lines(events) if json.loads(line)["step"] in kept).encode())
    if (out / OPEN_ROUND).is_file():
        write_whole(out / OPEN_ROUND, "".join(line + "\n" for line in point.finished_lines).encode())
    return {"at": datetime.now(timezone.utc).isoformat(), "records": len(point.records),
            "finished": len(point.finished), "dropped": on_disk - len(point.records),
            "continual_arc_version": __version__}


class _Files:
    """A run directory's records as the run plays (none without a directory): every action's event, a frozen test's
    instances as each ends (``per_instance``), and every round's records as it ends, each written after the events
    of the actions before it, so that whatever stops the run, a record on disk has its events on disk."""

    def __init__(self, out: Path | None, mode: str = "w", events: bool = False, per_instance: bool = False):
        self.records = (out / "records.jsonl").open(mode) if out is not None else None
        self.events = (out / "events.jsonl").open(mode) if out is not None and events else None
        self.open_round = (out / OPEN_ROUND).open(mode) if out is not None and per_instance else None

    def event(self, line: str) -> None:
        if self.events is not None:
            self.events.write(line)

    def instance_ended(self, record: InstanceRecord) -> None:
        if self.open_round is not None:
            if self.events is not None:
                self.events.flush()
            self.open_round.write(json.dumps(record.to_json()) + "\n")
            self.open_round.flush()

    def round_ended(self, records: list[InstanceRecord]) -> None:
        if self.records is None:
            return
        if self.events is not None:
            self.events.flush()
        for record in records:
            self.records.write(json.dumps(record.to_json()) + "\n")
        self.records.flush()
        if self.open_round is not None:
            self.open_round.truncate(0)

    def close(self) -> None:
        for fp in (self.records, self.events, self.open_round):
            if fp is not None:
                fp.close()


@dataclass
class RunResult:
    out_dir: Path | None
    records: list[InstanceRecord]
    summary: dict


class _InstanceState:
    """An open instance's observation and the resources its actions have consumed."""

    __slots__ = ("obs", "actions", "forced", "resources", "wall")

    def __init__(self, obs: Observation):
        self.obs = obs
        self.actions = 0
        self.forced = False
        self.resources: dict[str, float] = {}
        self.wall = 0.0

    def add_resources(self, delta: dict[str, float], seconds: float) -> None:
        for key, val in delta.items():
            self.resources[key] = self.resources.get(key, 0.0) + val
        self.wall += seconds

    def finish(self) -> dict[str, float]:
        out = dict(self.resources)
        out["wall_seconds"] = self.wall
        out["actions"] = float(self.actions)
        if self.forced:
            out["forced_give_up"] = 1.0
        return out


def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _diff_resources(before: dict[str, float | Decimal], after: dict[str, float | Decimal]) -> dict[str, float]:
    """Each counter's difference over an instance, as a float, leaving out what is not a number. A counter the learner
    keeps as a Decimal, a running total of charges, is differenced as a Decimal, so that an instance shows exactly what
    it was charged whatever the process spent before it, a resumed process and one never stopped alike; a float total's
    difference carries the total's rounding."""
    out: dict[str, float] = {}
    for key, val in after.items():
        prior = before.get(key, 0)
        try:
            out[key] = float(val - Decimal(prior) if isinstance(val, Decimal) else float(val) - float(prior))
        except (TypeError, ValueError, ArithmeticError):
            continue
    return out


def is_arc3(schedule: Schedule) -> bool:
    """Whether ``schedule`` serves the ARC-AGI-3 track's games."""
    return (schedule.meta or {}).get("dataset") == DATASET_ARC3


def default_config(schedule: Schedule) -> EnvConfig | Arc3Config:
    """The default environment settings of ``schedule``'s track."""
    return Arc3Config() if is_arc3(schedule) else EnvConfig()


def load_config(schedule: Schedule, d: dict) -> EnvConfig | Arc3Config:
    """Environment settings saved as JSON (a run's ``config.json``), read as ``schedule``'s track's."""
    return Arc3Config.from_json(d) if is_arc3(schedule) else EnvConfig.from_json(d)


def build_env(
    schedule: Schedule,
    config: EnvConfig | Arc3Config | None = None,
    registry: TaskRegistry | LevelSource | None = None,
) -> ContinualArcEnv | Arc3Env:
    """The environment for ``schedule``, at ``config`` or its track's default settings: the games of ARC-AGI-3, their
    levels drawn by ``registry``, a :class:`~continual_arc.arc3.instances.LevelSource` (the pinned games' generators
    when None), or the grids of a task registry."""
    config = config or default_config(schedule)
    if is_arc3(schedule):
        if not isinstance(config, Arc3Config):
            raise TypeError("an ARC-AGI-3 schedule is played under an Arc3Config")
        if registry is not None and not isinstance(registry, LevelSource):
            raise TypeError("an ARC-AGI-3 schedule's levels come from a LevelSource, not a task registry")
        return Arc3Env(schedule, config=config, source=registry)
    if not isinstance(config, EnvConfig):
        raise TypeError("a grid schedule is played under an EnvConfig")
    return ContinualArcEnv(schedule, registry=registry, config=config)


def learner_view(obs: Observation | Arc3Observation, aliases: TaskAliases) -> LearnerObservation | Arc3LearnerObservation:
    """The instance alone, as a learner is shown it: the task under the run's alias."""
    view = Arc3LearnerObservation if isinstance(obs, Arc3Observation) else LearnerObservation
    return view.of(obs, aliases.alias(obs.task_id))


def _default_max_actions(config: EnvConfig | Arc3Config) -> int:
    """Runaway guard: an instance can take at most its caps' actions and demonstrations plus one (a game's instance
    ends at its own action cap, at most ``max_actions``)."""
    if isinstance(config, Arc3Config):
        return config.max_actions + config.demo_cap + 2
    return config.attempt_cap + config.demo_cap + 2


def run(
    learner: Learner,
    schedule: Schedule,
    config: EnvConfig | Arc3Config | None = None,
    out_dir: str | Path | None = None,
    registry: TaskRegistry | LevelSource | None = None,
    max_actions_per_instance: int | None = None,
    log_events: bool = False,
    progress: ProgressFn | None = None,
    learner_name: str | None = None,
    addon: Addon | None = None,
    plugins: list[Plugin] | tuple[Plugin, ...] | None = None,
    env: ContinualArcEnv | Arc3Env | None = None,
    resume: bool = False,
    alias_style: str = "hex",
    max_concurrent: int | None = None,
    settings: dict | None = None,
) -> RunResult:
    """Run ``learner`` over ``schedule`` and return records plus summary.

    An optional add-on observes completed records and may enqueue auxiliary
    episodes. Existing callers are unchanged when ``addon`` is omitted. ``env``
    is an already-built environment; when None one is built from the schedule
    and ``config`` by :func:`build_env`. With ``resume`` and an ``out_dir`` that holds a run, the run goes on from
    its saved point, as the module's docstring says, refusing any settings but its own (``settings`` among them, what
    else the driver says the run must keep, such as its learner's own arguments); with ``resume`` and an ``out_dir``
    that holds none, the run starts. ``alias_style`` is how the learner's view
    names tasks (:data:`continual_arc.aliases.ALIAS_STYLES`). ``max_concurrent`` caps how many of
    a round's users a concurrent learner serves at once (all of them by default), for a round wider
    than the machine: every user still acts on the state the round began with.
    """
    if env is None:
        env = build_env(schedule, config, registry)
    config = env.config
    if addon is not None and plugins:
        raise ValueError("pass either addon or plugins, not both")
    plugin_list = list(plugins or ())
    if addon is not None:
        plugin_list.append(addon)
    if plugin_list and schedule.users != 1:
        raise ValueError("add-ons and plugins are served one episode at a time: they need one user")
    name = learner_name or type(learner).__name__
    run_id = f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    own = own_settings(config, name, alias_style, log_events, settings, max_actions_per_instance)
    max_actions_per_instance = own["max_actions_per_instance"]
    keeps_state = callable(getattr(learner, "state", None))
    concurrent = bool(getattr(learner, "concurrent_users", False)) and schedule.users > 1
    if concurrent and not callable(getattr(learner, "user_resources", None)):
        raise TypeError("a learner that serves users concurrently reports its counters per user: "
                        "it needs user_resources(user)")

    out_path: Path | None = None
    point: SavedPoint | None = None
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if holds_run(out_path):
            if not resume:
                raise ValueError(f"{out_path} holds a run already: go on with it (resume), or give this run a "
                                 "directory of its own")
            if plugin_list:
                raise ValueError("cannot resume a run with add-ons or plugins")
            refuse_other_settings(out_path, schedule, config, own)
            point = saved_point(out_path, schedule)
            meta = json.loads((out_path / "meta.json").read_text())
            meta.setdefault("resumed_at", []).append(_go_back(out_path, point))
        else:
            meta = {
                **own,
                "protocol": protocol_of(schedule, config),
                "learner_class": f"{type(learner).__module__}.{type(learner).__qualname__}",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "continual_arc_version": __version__,
                "addon": getattr(addon, "name", None) if addon is not None else None,
                "plugins": [plugin.name for plugin in plugin_list],
            }
            write_whole(out_path / "config.json", json.dumps(config.to_json(), indent=1).encode())
        write_whole(out_path / "meta.json", json.dumps(meta, indent=1).encode())
        write_whole(out_path / "schedule.json", json.dumps(schedule.to_json(), indent=1).encode())
    files = _Files(out_path, "w" if point is None else "a", log_events, schedule.meta.get("test") in TESTS)

    def keep(records: int) -> None:
        """The learner's state at a round boundary, after ``records`` records, kept beside them."""
        if out_path is not None and keeps_state:
            state = learner.state()
            if state is not None:
                keep_state(out_path, records, state)

    aliases = TaskAliases.for_run(out_path, alias_style)

    def shown(obs):
        return learner_view(obs, aliases)

    runtime = PluginRuntime(out_path / "plugin-events.jsonl" if out_path and plugin_list else None)
    manager = PluginManager(plugin_list, schedule, config, run_id, runtime)
    manager.on_run_start()

    try:
        env.reset()
        if point is None:
            keep(0)
        else:
            env.restore(point.records, point.finished)  # type: ignore[arg-type]
            if point.state is not None:
                learner.restore(point.state)
        states: dict[int, _InstanceState] = {}
        round_open = False
        while env.open_indices():
            indices = env.open_indices()
            if not round_open:
                learner.begin_round(len(indices))
                round_open = True
            if concurrent:
                committed = _serve_round_concurrently(learner, env, manager, indices, max_actions_per_instance,
                                                      files, progress, shown, max_concurrent)
                files.round_ended(committed)
                learner.end_round()
                keep(len(env.records))
                round_open = False
                if manager.stop_requested:
                    break
                continue
            for index in indices:
                if not env.is_open(index):
                    continue
                state = states.get(index)
                if state is None:
                    obs = manager.on_episode_start(env.observation(index))
                    if obs.reset_before and obs.is_fresh:
                        learner.reset()
                    state = states[index] = _InstanceState(obs)
                obs = state.obs
                res_before = dict(learner.resources() or {})
                t0 = time.perf_counter()
                if state.actions >= max_actions_per_instance:
                    action = GiveUp()
                    state.forced = True
                else:
                    action = learner.act(shown(obs))
                learner_action = action
                action = manager.on_action(obs, action)
                state.actions += 1
                result = env.step(action, index)
                feedback = manager.on_feedback(obs, action, result.feedback)
                learner.observe(shown(obs), action, feedback)
                state.add_resources(_diff_resources(res_before, dict(learner.resources() or {})), time.perf_counter() - t0)
                files.event(_event_line(obs, learner_action, action, result, feedback))
                if not result.instance_done:
                    state.obs = manager.on_observation(env.observation(index))
                    continue
                record = result.record
                assert record is not None
                record.resources = state.finish()
                del states[index]
                manager.on_episode_end(record)
                files.instance_ended(record)
                if progress is not None:
                    progress(record, env.n_steps)
                if result.round_records is not None:
                    files.round_ended(result.round_records)
                    learner.end_round()
                    keep(len(env.records))
                    round_open = False
                    if manager.stop_requested:
                        break
                    requests = manager.take_episode_requests()
                    if requests:
                        env.insert_next(requests)
                    break  # the next round's instances are read fresh
            if manager.stop_requested:
                break
    finally:
        files.close()
        close = getattr(env, "close", None)
        if close is not None:
            close()

    records = env.records
    manager.on_run_end(records)
    summary = summarize(records, config, checkpoints=(schedule.meta.get("config") or {}).get("checkpoints", ()))
    summary["protocol"] = protocol_of(schedule, config)
    if plugin_list:
        summary["plugins"] = manager.summary()
        if addon is not None:
            summary["addon"] = {"name": addon.name, **(addon.summary() or {})}
    if out_path is not None:
        (out_path / "summary.json").write_text(json.dumps(summary, indent=1))
    return RunResult(out_dir=out_path, records=records, summary=summary)


def _event_line(obs, learner_action, action, result, feedback) -> str:
    return json.dumps(
        {
            "step": obs.step,
            "round": obs.round,
            "user": getattr(obs, "user", 0),
            "task_id": obs.task_id,
            "action": {"type": type(learner_action).__name__, **_jsonable(learner_action)},
            "env_action": {"type": type(action).__name__, **_jsonable(action)},
            "feedback": {"type": type(result.feedback).__name__, **_jsonable(result.feedback)},
            "delivered_feedback": {"type": type(feedback).__name__, **_jsonable(feedback)},
            "cost_delta": result.cost_delta,
        }
    ) + "\n"


def _serve_round_concurrently(learner, env, manager, indices, max_actions, files, progress, shown,
                              max_concurrent=None) -> list[InstanceRecord]:
    """Serve one round of a collective schedule with every user at their own pace: one thread per
    instance, acting, stepping and observing until the instance ends, the environment (and the
    run's files) behind one lock. A wipe the round is marked for happens once, before any user
    acts. If any user's thread fails, the others stop at their next action and the error is
    raised; the round's records are then never written, and a resumed run plays the round again, all
    of it but a frozen test's instances that ended. Returns the round's records in slot order."""
    lock = threading.Lock()
    failed = threading.Event()
    committed: list[InstanceRecord] = []
    first = {index: manager.on_episode_start(env.observation(index)) for index in indices}
    if any(obs.reset_before and obs.is_fresh for obs in first.values()):
        learner.reset()

    def serve(index: int) -> None:
        state = _InstanceState(first[index])
        try:
            while not failed.is_set():
                obs = state.obs
                before = dict(learner.user_resources(obs.user) or {})
                t0 = time.perf_counter()
                if state.actions >= max_actions:
                    action = GiveUp()
                    state.forced = True
                else:
                    action = learner.act(shown(obs))
                learner_action = action
                with lock:
                    action = manager.on_action(obs, action)
                    state.actions += 1
                    result = env.step(action, index)
                    feedback = manager.on_feedback(obs, action, result.feedback)
                learner.observe(shown(obs), action, feedback)
                state.add_resources(_diff_resources(before, dict(learner.user_resources(obs.user) or {})),
                                    time.perf_counter() - t0)
                with lock:
                    files.event(_event_line(obs, learner_action, action, result, feedback))
                    if not result.instance_done:
                        state.obs = manager.on_observation(env.observation(index))
                        continue
                    record = result.record
                    assert record is not None
                    record.resources = state.finish()
                    manager.on_episode_end(record)
                    files.instance_ended(record)
                    if progress is not None:
                        progress(record, env.n_steps)
                    if result.round_records is not None:
                        committed.extend(result.round_records)
                return
        except BaseException:
            failed.set()
            raise

    with ThreadPoolExecutor(max_workers=min(len(indices), max_concurrent or len(indices)), thread_name_prefix="user") as pool:
        futures = [pool.submit(serve, index) for index in indices]
    for future in futures:
        future.result()  # the first failure, if any
    if not committed:
        raise RuntimeError("a concurrent round ended without committing its records")
    return committed


def prior_records(out_dir: str | Path) -> list[InstanceRecord]:
    """The records a run left in ``out_dir``, in order, up to the first line that does not parse and less a last line
    that a stop cut off; empty when there are none."""
    return [record for _, record in _parsed(Path(out_dir) / "records.jsonl")]


def whole_rounds(records: list[InstanceRecord], schedule: Schedule) -> list[InstanceRecord]:
    """``records`` cut back to the last round boundary of ``schedule``, since a round
    is replayed from its start after an interruption."""
    starts = schedule.round_starts()
    n = len(records)
    while 0 < n < len(starts) and not starts[n]:
        n -= 1
    return records[:n]


def load_run(out_dir: str | Path) -> tuple[Schedule, EnvConfig | Arc3Config, list[InstanceRecord]]:
    """Load the schedule, config and records of a finished run directory."""
    out_path = Path(out_dir)
    schedule = Schedule.load(out_path / "schedule.json")
    config = load_config(schedule, json.loads((out_path / "config.json").read_text()))
    records = [
        InstanceRecord.from_json(json.loads(line))
        for line in (out_path / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return schedule, config, records


def rescore(out_dir: str | Path) -> int:
    """Charge a run directory written before 20 Oct 2026 under the rule since (:mod:`continual_arc.env`): an unsolved
    instance costs the whole budget, ``max_cost``, where it was charged the attempt cap plus its demo requests, and a
    solved one costs what it was charged, the same under both. ``records.jsonl`` and the metrics in ``summary.json`` are
    rewritten and ``meta.json`` gains a ``rescored`` entry saying when, how many records changed and the rule the run
    was played under; the play itself (events, transcripts, the rules text the learner read) is left as it was. Returns
    the number of records whose cost changed, none on a second pass."""
    out_path = Path(out_dir)
    schedule, config, records = load_run(out_path)
    changed = 0
    for record in records:
        if not record.solved and record.cost != config.max_cost:
            record.cost = config.max_cost
            changed += 1
    if not changed:
        return 0
    (out_path / "records.jsonl").write_text("".join(json.dumps(r.to_json()) + "\n" for r in records))
    summary_path = out_path / "summary.json"
    played = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary = summarize(records, config, checkpoints=(schedule.meta.get("config") or {}).get("checkpoints", ()))
    summary.update({k: played[k] for k in ("protocol", "plugins", "addon") if k in played})
    summary_path.write_text(json.dumps(summary, indent=1))
    meta_path = out_path / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["rescored"] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "records_changed": changed,
        "rule": "an unsolved instance costs max_cost",
        "played_under": "an unsolved instance charged attempt_cap plus demo_cost per demo request, and so told",
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    return changed
