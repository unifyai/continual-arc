"""Run a learner through a schedule and write a run directory.

A run directory contains::

    schedule.json   the exact schedule served
    config.json     the EnvConfig used
    meta.json       learner identity, timestamps, package version
    records.jsonl   one InstanceRecord per instance, in order
    summary.json    metrics computed by continual_arc.metrics.summarize
    events.jsonl    (optional) every action and feedback, including grids
    plugin-events.jsonl  (optional) events emitted by plug-ins

The runner is the only place that calls ``learner.reset()``; it does so exactly
when the schedule slot says so. It also guards against runaway learners with a
per-instance action limit, after which the instance is force-abandoned.

A learner is shown the instance alone (:class:`continual_arc.env.LearnerObservation`):
the task under the run's alias, salted in the run directory's ``alias-salt``, and nothing
about the schedule. Plug-ins, events and records see the slot
(:class:`continual_arc.env.Observation`), real task id, phase and exposure included.

Instances are served in rounds, one instance per user of the schedule. The runner
calls ``learner.begin_round`` before the round's first action and ``learner.end_round``
after its last instance ends, and writes the round's records together, in slot order,
so ``records.jsonl`` only ever holds whole rounds and a resumed run replays the
interrupted round from its start. With one user every instance is a round of one and
this is the serial loop.

With many users (the collective regime) the users of a round work at the same time. A
learner that can serve them at once sets ``concurrent_users = True``: every user's
instance is then served on a thread of its own, from its first action to its verdict, at
its own pace, with the environment behind a lock, and the round ends when every user's
has. Such a learner reports its counters per user (``user_resources(user)``), and an
instance's resources are that user's counter deltas over its actions. Any other learner
is served one action at a time: one action on every open instance before a second on
any, an instance's resources the learner's counter deltas over its own actions.

The loop drives a :class:`continual_arc.env.ContinualArcEnv`, built from the schedule by
:func:`build_env` or passed ready as ``env``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from continual_arc._version import __version__
from continual_arc.addons import Addon, Plugin, PluginManager, PluginRuntime
from continual_arc.aliases import TaskAliases
from continual_arc.configs import protocol_of
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
from continual_arc.schedule import Schedule
from continual_arc.tasks import TaskRegistry

ProgressFn = Callable[[InstanceRecord, int], None]


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


def _diff_resources(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in after.items():
        try:
            out[key] = float(val) - float(before.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return out


def build_env(
    schedule: Schedule,
    config: EnvConfig | None = None,
    registry: TaskRegistry | None = None,
) -> ContinualArcEnv:
    """The environment for ``schedule``, at ``config`` or the default settings."""
    return ContinualArcEnv(schedule, registry=registry, config=config or EnvConfig())


def learner_view(obs: Observation, aliases: TaskAliases) -> LearnerObservation:
    """The instance alone, as a learner is shown it: the task under the run's alias."""
    return LearnerObservation.of(obs, aliases.alias(obs.task_id))


def _default_max_actions(config: EnvConfig) -> int:
    """Runaway guard: an instance can take at most the attempt and demo caps plus one."""
    return config.attempt_cap + config.demo_cap + 2


def run(
    learner: Learner,
    schedule: Schedule,
    config: EnvConfig | None = None,
    out_dir: str | Path | None = None,
    registry: TaskRegistry | None = None,
    max_actions_per_instance: int | None = None,
    log_events: bool = False,
    progress: ProgressFn | None = None,
    learner_name: str | None = None,
    addon: Addon | None = None,
    plugins: list[Plugin] | tuple[Plugin, ...] | None = None,
    env: ContinualArcEnv | None = None,
    resume: bool = False,
    alias_style: str = "hex",
    max_concurrent: int | None = None,
) -> RunResult:
    """Run ``learner`` over ``schedule`` and return records plus summary.

    An optional add-on observes completed records and may enqueue auxiliary
    episodes. Existing callers are unchanged when ``addon`` is omitted. ``env``
    is an already-built environment; when None one is built from the schedule
    and ``config`` by :func:`build_env`. With ``resume`` and an ``out_dir`` holding a ``records.jsonl``
    from an interrupted run of the same schedule, the finished instances are
    restored and the stream continues from the next one; records and events are
    appended and ``meta.json`` gains a ``resumed_at`` entry. The learner's own
    persistent state is the learner's business. ``alias_style`` is how the learner's view
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
    run_id = f"{learner_name or type(learner).__name__}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if max_actions_per_instance is None:
        max_actions_per_instance = _default_max_actions(config)

    out_path: Path | None = None
    records_fp = events_fp = None
    prior: list[InstanceRecord] = []
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if resume:
            prior = whole_rounds(prior_records(out_path), schedule)
            if prior and plugin_list:
                raise ValueError("cannot resume a run with add-ons or plugins")
        mode = "a" if prior else "w"
        schedule.save(out_path / "schedule.json")
        (out_path / "config.json").write_text(json.dumps(config.to_json(), indent=1))
        meta = {
            "learner": learner_name or type(learner).__name__,
            "protocol": protocol_of(schedule, config),
            "learner_class": f"{type(learner).__module__}.{type(learner).__qualname__}",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "continual_arc_version": __version__,
            "max_actions_per_instance": max_actions_per_instance,
            "addon": getattr(addon, "name", None) if addon is not None else None,
            "plugins": [plugin.name for plugin in plugin_list],
            "alias_style": alias_style,
        }
        if prior:
            try:
                meta = json.loads((out_path / "meta.json").read_text())
            except (OSError, ValueError):
                pass
            meta.setdefault("resumed_at", []).append(
                {"at": datetime.now(timezone.utc).isoformat(), "records": len(prior)}
            )
        (out_path / "meta.json").write_text(json.dumps(meta, indent=1))
        records_fp = (out_path / "records.jsonl").open(mode)
        if log_events:
            events_fp = (out_path / "events.jsonl").open(mode)

    if plugin_list and schedule.users != 1:
        raise ValueError("add-ons and plugins are served one episode at a time: they need one user")
    concurrent = bool(getattr(learner, "concurrent_users", False)) and schedule.users > 1
    if concurrent and not callable(getattr(learner, "user_resources", None)):
        raise TypeError("a learner that serves users concurrently reports its counters per user: "
                        "it needs user_resources(user)")
    aliases = TaskAliases.for_run(out_path, alias_style)

    def shown(obs):
        return learner_view(obs, aliases)

    runtime = PluginRuntime(out_path / "plugin-events.jsonl" if out_path and plugin_list else None)
    manager = PluginManager(plugin_list, schedule, config, run_id, runtime)
    manager.on_run_start()

    try:
        env.reset()
        if prior:
            env.restore(prior)  # type: ignore[arg-type]
        states: dict[int, _InstanceState] = {}
        round_open = False
        while env.open_indices():
            indices = env.open_indices()
            if not round_open:
                learner.begin_round(len(indices))
                round_open = True
            if concurrent:
                committed = _serve_round_concurrently(learner, env, manager, indices, max_actions_per_instance,
                                                      events_fp, progress, shown, max_concurrent)
                if records_fp is not None:
                    for finished in committed:
                        records_fp.write(json.dumps(finished.to_json()) + "\n")
                    records_fp.flush()
                learner.end_round()
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
                if events_fp is not None:
                    events_fp.write(_event_line(obs, learner_action, action, result, feedback))
                if not result.instance_done:
                    state.obs = manager.on_observation(env.observation(index))
                    continue
                record = result.record
                assert record is not None
                record.resources = state.finish()
                del states[index]
                manager.on_episode_end(record)
                if progress is not None:
                    progress(record, env.n_steps)
                if result.round_records is not None:
                    if records_fp is not None:
                        for finished in result.round_records:
                            records_fp.write(json.dumps(finished.to_json()) + "\n")
                        records_fp.flush()
                    learner.end_round()
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
        if records_fp is not None:
            records_fp.close()
        if events_fp is not None:
            events_fp.close()
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


def _serve_round_concurrently(learner, env, manager, indices, max_actions, events_fp, progress, shown,
                              max_concurrent=None) -> list[InstanceRecord]:
    """Serve one round of a collective schedule with every user at their own pace: one thread per
    instance, acting, stepping and observing until the instance ends, the environment (and the
    run's files) behind one lock. A wipe the round is marked for happens once, before any user
    acts. If any user's thread fails, the others stop at their next action and the error is
    raised; the round's records are then never written, and a resumed run replays the round.
    Returns the round's records in slot order."""
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
                    if events_fp is not None:
                        events_fp.write(_event_line(obs, learner_action, action, result, feedback))
                    if not result.instance_done:
                        state.obs = manager.on_observation(env.observation(index))
                        continue
                    record = result.record
                    assert record is not None
                    record.resources = state.finish()
                    manager.on_episode_end(record)
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
    """The whole records an interrupted run left in ``out_dir`` (a trailing partial
    line is dropped), in order; empty when there is nothing to resume from."""
    path = Path(out_dir) / "records.jsonl"
    if not path.is_file():
        return []
    records: list[InstanceRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(InstanceRecord.from_json(json.loads(line)))
        except (ValueError, KeyError, TypeError):
            break
    return records


def whole_rounds(records: list[InstanceRecord], schedule: Schedule) -> list[InstanceRecord]:
    """``records`` cut back to the last round boundary of ``schedule``, since a round
    is replayed from its start after an interruption."""
    starts = schedule.round_starts()
    n = len(records)
    while 0 < n < len(starts) and not starts[n]:
        n -= 1
    return records[:n]


def load_run(out_dir: str | Path) -> tuple[Schedule, EnvConfig, list[InstanceRecord]]:
    """Load the schedule, config and records of a finished run directory."""
    out_path = Path(out_dir)
    schedule = Schedule.load(out_path / "schedule.json")
    config = EnvConfig.from_json(json.loads((out_path / "config.json").read_text()))
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
