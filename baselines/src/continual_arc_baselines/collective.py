"""A team served at once: the collective regime with an off-the-shelf harness per user.

In the collective regime (``continual_arc``: a configuration's ``users``) every user walks a stream
of their own and a round serves one instance per user, concurrently. Here every user runs a harness
of their own (one hermes-agent, openclaw or prime-agent with its own private state: memories,
skills, sessions, workspace), as each person on a team runs their own copy, and every harness calls
the same model through the same server. What the users share is the model, and only the model: its
weights may change between rounds (the trained rows), nothing else passes from one user to another.
The benchmark's contract therefore holds by construction: a user's harness changes its own state
only while that user works, and each user works on one instance per round.

Each user's harness talks to a tracking proxy of its own (:class:`LLMProxy`), so the usage the
runner charges to an instance is that user's (:meth:`CollectiveHarnessLearner.user_resources`), and
every model call can be captured whole (the request as the server received it, the reply, and the
server's own token ids when the proxy asks for them) and tagged with the instance and the protocol
turn it belongs to: the material a trained row learns from. After every round the ``round_hook``
receives the round (every user's instance: its outcome, its turns, its model calls, and where the
user's harness state was before and is after) and may return new fields for every proxy to force,
such as the name of the adapter it has just trained and loaded. So that a trained row can do its
update's work on an instance while the rest of the round still plays, ``begin_hook`` is told when a
round begins (before any user acts) and ``instance_hook`` receives each instance as soon as it has
ended, its closing message and storage step done and every model call made for it reported by the
proxy (from that user's thread: it must not block). ``call_hook`` receives each of an instance's
model calls as it ends, so that a driver can keep every call on disk as it is made and resume a
round cut short from its last call rather than from its start.

Every user's harness adapter is attached with its state logged (``SystemLearner.attach``'s ``log_state``), so at
the end of every round each user's harness state is committed to that user's own history,
``users/uNNN/system/state.git`` and its ``state-log.jsonl`` (``SystemLearner.end_round``,
:mod:`continual_arc_baselines.statelog`): ``git diff HEAD~1 HEAD`` there is what the user's harness
wrote in the round, and a run's frozen tests fork every user's commit at the end of the stream
(:mod:`continual_arc_baselines.frozen_collective`).

The users' harnesses are started lazily, on each user's first action, and serve concurrently
(``concurrent_users``): the runner calls :meth:`act` and :meth:`observe` for different users from
different threads. Every user sees the same task names, the runner's for the run (its ``alias-salt`` in the run
directory, kept for a resume), as tickets have one id for everyone in a team.

The harnesses see only the instance, as every learner does. The traces the hooks read also carry the slot's own
bookkeeping, which a trained row's records and teachers need (the real task id, the phase, the user's earlier visits to
the task): the learner reads them from the ``schedule`` its driver runs, by slot, and never passes them to a harness.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from continual_arc.env import (
    Action,
    DemosFeedback,
    Feedback,
    GiveUpFeedback,
    LearnerObservation,
    SubmitFeedback,
)
from continual_arc.learner import BaseLearner
from continual_arc.schedule import Schedule

from continual_arc_baselines.proxy import LLMProxy
from continual_arc_baselines.systems.base import SystemLearner
from continual_arc_baselines.tracking import LLMCall, Tracker

log = logging.getLogger(__name__)

# What a snapshot of a user's harness state leaves out: runtimes and caches the harness installs for
# itself (prime-agent copies a whole Python environment into its state), not what it has learned, and the
# files of the process that was running (its temporary files, sockets and locks).
RUNTIMES = ("kernel-venv", "node_modules", ".cache", "__pycache__", ".npm", "*.pyc")
SNAPSHOT_EXCLUDE = (*RUNTIMES, "tmp", "*.sock", "*.lock")
SNAPSHOT_MAX_FILE_BYTES = 2_000_000


@dataclass
class CapturedCall:
    """One model call a user's harness made, as that user's proxy saw it. ``phase`` is ``turn``
    (while the harness answered an observation), ``after`` (the instance's closing message and
    the harness's storage step, once the verdict was in) or ``idle`` (between instances: a
    harness's own background work). ``request`` is the body the server received and ``response``
    the reply (assembled from the stream when streamed), both present only when captured."""

    call_id: str
    user: int
    slot: int | None
    turn: int | None
    phase: str
    started: float
    status: str
    latency_ms: float | None
    tokens: dict
    finish_reason: str | None
    request: dict | None = None
    response: dict | None = None

    def meta(self) -> dict[str, Any]:
        """Everything but the bodies (the per-user call log)."""
        out = {k: v for k, v in dataclasses.asdict(self).items() if k not in ("request", "response")}
        if self.response is not None:
            out["prompt_token_ids"] = len(self.response.get("prompt_token_ids") or [])
            out["reply_token_ids"] = len(((self.response.get("choices") or [{}])[0]).get("token_ids") or [])
        return out


@dataclass
class InstanceTrace:
    """One user's instance, as the round hook reads it: the slot (its real ``task_id``, ``phase`` and
    ``user_exposure_index``, from the schedule) and ``task_label``, the name the harness saw it under, what the
    harness did at the protocol level (``turns``: the action and the environment's answer, per turn), every
    model call it made for the instance, the outcome, and where the user's harness state was
    when the round began (``state_before``, a filtered copy, or None) and is now (``state_dir``)."""

    user: int
    slot: int
    round: int
    task_id: str
    task_label: str
    phase: str
    user_exposure_index: int
    turns: list[dict] = field(default_factory=list)
    calls: list[CapturedCall] = field(default_factory=list)
    solved: bool = False
    ended: bool = False
    accepted_grid: list | None = None
    state_before: Path | None = None
    state_dir: Path | None = None
    started: float = field(default_factory=time.time)
    seconds: float = 0.0


@dataclass
class RoundData:
    index: int
    instances: list[InstanceTrace]


RoundHook = Callable[[RoundData], "dict[str, Any] | None"]
InstanceHook = Callable[[InstanceTrace], None]
CallHook = Callable[[InstanceTrace, CapturedCall], None]


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"type": type(obj).__name__, **{k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _ended(feedback: Feedback) -> bool:
    return isinstance(feedback, GiveUpFeedback) or (
        isinstance(feedback, SubmitFeedback) and (feedback.correct or feedback.failed))


def snapshot_state(src: Path, dst: Path, *, exclude: tuple[str, ...] = SNAPSHOT_EXCLUDE,
                   max_file_bytes: int | None = SNAPSHOT_MAX_FILE_BYTES) -> dict[str, int]:
    """Copy a harness's state directory to ``dst``, without its runtimes and caches, its links and its files over
    ``max_file_bytes``: a copy to read, with a directory only for the files in it, and none at all when nothing is
    copied. With ``max_file_bytes`` None it is a copy to restore from (:func:`restore_state`): every file and every
    directory, empty ones too. A state directory that does not exist leaves no copy. Returns what was copied and
    skipped. The copy is made beside ``dst`` and replaces it only once whole, so a copy cut short leaves the snapshot
    that was there."""
    stats = {"files": 0, "bytes": 0, "skipped": 0}
    complete = max_file_bytes is None
    building = dst.with_name(dst.name + ".partial")
    shutil.rmtree(building, ignore_errors=True)
    if src.is_dir():
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pat) for pat in exclude)]
            rel = Path(root).relative_to(src)
            if complete:
                (building / rel).mkdir(parents=True, exist_ok=True)
            for name in files:
                path = Path(root) / name
                if any(fnmatch.fnmatch(name, pat) for pat in exclude) or path.is_symlink():
                    stats["skipped"] += 1
                    continue
                try:
                    size = path.stat().st_size
                    if not complete and size > max_file_bytes:
                        stats["skipped"] += 1
                        continue
                    (building / rel).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, building / rel / name)
                except OSError:
                    stats["skipped"] += 1  # the harness removed or locked it meanwhile
                    continue
                stats["files"] += 1
                stats["bytes"] += size
    shutil.rmtree(dst, ignore_errors=True)
    if building.is_dir():
        building.rename(dst)
    return stats


def restore_state(snapshot: Path, dst: Path, *, keep: tuple[str, ...] = RUNTIMES) -> None:
    """Put a harness's state directory ``dst`` back as ``snapshot`` holds it (a :func:`snapshot_state` copy with no
    size limit; none, of a state that did not exist): everything under ``dst`` is deleted but its runtimes and caches
    (``keep``, which a snapshot leaves out and the harness would only install again), and the snapshot is copied in."""
    def kept(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in keep)

    dst.mkdir(parents=True, exist_ok=True)
    emptied = []
    for root, dirs, files in os.walk(dst):
        for name in files:
            if not kept(name):
                (Path(root) / name).unlink()
        for name in list(dirs):
            path = Path(root) / name
            if path.is_symlink() or kept(name):
                dirs.remove(name)   # a kept tree is left whole; a link is never followed
                if path.is_symlink() and not kept(name):
                    path.unlink()
        emptied.append(Path(root))
    for path in reversed(emptied[1:]):   # deepest first, once their files are gone
        if not any(path.iterdir()):
            path.rmdir()
    if snapshot.is_dir():
        shutil.copytree(snapshot, dst, dirs_exist_ok=True)


class _User:
    """One user: their harness, their proxy, what they are working on, and the calls it made."""

    def __init__(self, index: int, workdir: Path, system: Any, proxy: LLMProxy, tracker: Tracker,
                 capture: bool, call_hook: CallHook | None) -> None:
        self.index, self.workdir, self.system, self.proxy, self.tracker = index, workdir, system, proxy, tracker
        self.capture, self.call_hook = capture, call_hook
        self.lock = threading.Lock()
        self.reported = threading.Condition(self.lock)  # notified as each call's end is reported
        self.context: tuple[int | None, int | None, str] = (None, None, "idle")  # slot, turn, phase
        self.trace: InstanceTrace | None = None
        self._starts: dict[str, tuple[int | None, int | None, str, float]] = {}  # the calls not yet ended
        self._settled: set[int] = set()  # the instances (slots) whose calls are final: what the hooks get
        self._log = (workdir / "calls.jsonl").open("a")
        tracker.subscribe(self.on_call)

    def on_call(self, call: LLMCall) -> None:
        """The proxy's report of a call (from its handler threads): tag it with what the user was
        doing when it started, log it, and keep it with the instance it belongs to, unless that
        instance has settled (:meth:`settle`) before the report came. The call hook gets each call
        the instance keeps, and the call stays open until the hook has returned, so the instance
        never settles before its hook calls are done."""
        if call.status == "started":
            with self.lock:
                self._starts[call.call_id] = (*self.context, time.time())
            return
        late, kept = False, None
        with self.lock:
            slot, turn, phase, t0 = self._starts.get(call.call_id, (None, None, "idle", time.time()))
            captured = CapturedCall(
                call_id=call.call_id, user=self.index, slot=slot, turn=turn, phase=phase, started=t0,
                status=call.status, latency_ms=call.latency_ms, tokens=dict(call.tokens),
                finish_reason=call.finish_reason,
                request=call.request_body if self.capture else None,
                response=call.response_body if self.capture else None)
            if phase != "idle":
                if self.trace is not None and slot == self.trace.slot and slot not in self._settled:
                    self.trace.calls.append(captured)
                    kept = self.trace
                else:
                    late = True
            self._log.write(json.dumps(captured.meta()) + "\n")
            self._log.flush()
        try:
            if kept is not None and self.call_hook is not None:
                self.call_hook(kept, captured)
        finally:
            with self.lock:
                self._starts.pop(call.call_id, None)
                self.reported.notify_all()
        if late:
            log.warning("user %d, slot %s: model call %s (%s) ended %.1f s after it started, too late for its "
                        "instance: it is in calls.jsonl only", self.index, slot, call.call_id[:8], phase,
                        time.time() - t0)

    def _open(self, slot: int) -> list[tuple[str, str, float]]:
        """The calls started for the instance at ``slot`` whose end is not yet reported (under the lock)."""
        return [(cid, phase, t0) for cid, (s, _, phase, t0) in self._starts.items() if s == slot and phase != "idle"]

    def settle(self, trace: InstanceTrace, deadline: float) -> None:
        """Wait, until ``deadline`` (``time.monotonic()``) at the latest, for the proxy to report the end of every
        call the harness started for ``trace``, then fix its calls: the hooks get these, and a call reported later
        is only logged. The proxy reports a call once it has written the reply, so the harness can have taken the
        reply and moved on first (CI, 25 Sep: the round hook saw an instance without its last call)."""
        with self.lock:
            if trace.slot in self._settled:
                return
            started = time.monotonic()
            self.reported.wait_for(lambda: not self._open(trace.slot), timeout=max(0.0, deadline - started))
            still = self._open(trace.slot)
            self._settled.add(trace.slot)
        if still:
            log.error("user %d, slot %d: %d model call(s) started for the instance still open after waiting %.1f s "
                      "for them (%s); the hooks get the instance without them", self.index, trace.slot, len(still),
                      time.monotonic() - started,
                      ", ".join(f"{cid[:8]}: {phase}, open {time.time() - t0:.1f} s" for cid, phase, t0 in still))

    def resources(self) -> dict[str, float]:
        return {**self.tracker.usage.as_resources(), **(self.system.resources() or {})}

    def close(self) -> None:
        try:
            if hasattr(self.system, "close"):
                self.system.close()
        finally:
            self.proxy.stop()
            self._log.close()


class CollectiveHarnessLearner(BaseLearner):
    """One harness per user, every user's instance of a round served concurrently.

    ``make_system(user)`` builds a user's harness adapter (a
    :class:`continual_arc_baselines.systems.base.SystemLearner`), not yet attached. ``schedule`` is the one the
    runner serves, read for the traces' slot bookkeeping alone. Each user's
    proxy forwards to ``upstream`` (with ``api_key``), sets ``body_defaults`` where the harness
    left a field out, forces ``body_overrides``, removes ``body_drop``, renames message roles by
    ``role_map``, gives a history's assistant turn without reasoning an empty ``thinking_field`` and
    keeps the reasoning of the past turns ``keep_reasoning`` names (``proxy.KEEP_REASONING``), the others' cut
    to their last ``reasoning_tail`` characters or replaced by a summary of at most ``reasoning_summary`` the
    served model writes (asked with ``summary_overrides``; each logged to the user's ``summaries.jsonl``);
    with ``capture`` every
    call's request and reply are kept for the round hook (a trained row), otherwise only usage.
    ``snapshot`` copies every user's harness state at the start of each round, for a writer to
    see what the user's memory held when their instance began. ``begin_hook(n_instances)`` runs when
    a round begins, after the snapshots, ``call_hook(trace, call)`` as each of an instance's calls
    ends (from the proxy's thread: it must not block), and ``instance_hook(trace)`` as each instance
    ends. As the round ends, the ``end_round`` of every system that has one (a harness adapter's
    commits its state) runs for each user who played it, in user order, before ``round_hook``. An
    instance reaches the instance and round hooks once the proxy has reported the end of
    every call its harness started for it, and the call hook has had each, waiting at most
    ``calls_wait_s``: a call still open then is left out of the instance's calls (for both hooks
    alike) and logged as an error. Each user's proxy is a ``proxy`` (an :class:`LLMProxy` or a
    subclass that serves some calls itself), listening on the user's port in ``ports``, or on any
    free one when none is given or another process holds it: a driver that plays a round again
    gives each user the port they had, since a model can write its proxy's address into its code.
    """

    concurrent_users = True

    def __init__(self, make_system: Callable[[int], Any], run_dir: Path, *, schedule: Schedule, upstream: str,
                 api_key: str | None = None, body_defaults: dict[str, Any] | None = None,
                 body_overrides: dict[str, Any] | None = None, body_drop: tuple[str, ...] = (),
                 role_map: dict[str, str] | None = None, thinking_field: str | None = None,
                 keep_reasoning: str = "all", reasoning_tail: int = 0, reasoning_summary: int = 0,
                 summary_overrides: dict[str, Any] | None = None, shortened_key: str | None = None,
                 shortened_prefix: str = "", router: Any = None, declared_context_length: int | None = None, capture: bool = False,
                 snapshot: bool = False, round_hook: RoundHook | None = None,
                 reset_hook: Callable[[], None] | None = None,
                 begin_hook: Callable[[int], None] | None = None, call_hook: CallHook | None = None,
                 instance_hook: InstanceHook | None = None, calls_wait_s: float = 600.0, serve: Any = None,
                 proxy: type[LLMProxy] = LLMProxy, ports: dict[int, int] | None = None) -> None:
        self.make_system = make_system
        self.serve = serve   # every user's proxy answers chat completions with it when set (LLMProxy's serve)
        self.proxy, self.ports = proxy, dict(ports or {})
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.schedule = schedule
        self._infos = schedule.annotate()
        self.upstream, self.api_key = upstream, api_key
        self.body_defaults = dict(body_defaults or {})
        self.body_overrides = dict(body_overrides or {})
        self.body_drop = tuple(body_drop)
        self.role_map = dict(role_map or {})
        self.thinking_field = thinking_field
        self.keep_reasoning, self.reasoning_tail = keep_reasoning, reasoning_tail
        self.reasoning_summary, self.summary_overrides = reasoning_summary, dict(summary_overrides or {})
        self.shortened_key, self.shortened_prefix = shortened_key, shortened_prefix
        self.router = router  # routing.ReplicaRouter shared by every user's proxy, or None (the server routes)
        self.declared_context_length = declared_context_length
        self.capture, self.snapshot = capture, snapshot
        self.round_hook, self.reset_hook = round_hook, reset_hook
        self.begin_hook, self.call_hook, self.instance_hook = begin_hook, call_hook, instance_hook
        # a call normally ends within milliseconds of the harness taking its reply; one the harness started and
        # left running (a compaction, a housekeeping call) can take minutes, and one it abandoned ends when the
        # proxy next writes to it, or when the upstream finishes
        self.calls_wait_s = calls_wait_s
        self._users: dict[int, _User] = {}
        self._users_lock = threading.Lock()
        self._round: list[InstanceTrace] = []
        self._round_index: int | None = None
        self._round_lock = threading.Lock()

    # -- users ------------------------------------------------------------------------------------

    def user_dir(self, user: int) -> Path:
        return self.run_dir / "users" / f"u{user:03d}"

    def _user(self, user: int) -> _User:
        with self._users_lock:
            found = self._users.get(user)
            if found is not None:
                return found
            workdir = self.user_dir(user)
            workdir.mkdir(parents=True, exist_ok=True)
            tracker = Tracker()
            settings = dict(api_key=self.api_key, tracker=tracker, body_defaults=self.body_defaults,
                            body_overrides=self.body_overrides, body_drop=self.body_drop, role_map=self.role_map,
                            thinking_field=self.thinking_field, keep_reasoning=self.keep_reasoning,
                            reasoning_tail=self.reasoning_tail, reasoning_summary=self.reasoning_summary,
                            summary_overrides=self.summary_overrides, summary_log=workdir / "summaries.jsonl",
                            shortened_key=self.shortened_key, shortened_prefix=self.shortened_prefix,
                            router=self.router, route_key=f"u{user}", capture_content=self.capture, serve=self.serve,
                            declared_context_length=self.declared_context_length)
            port = self.ports.get(user, 0)
            proxy = self.proxy(self.upstream, port=port, **settings)
            try:
                proxy.start()
            except OSError as error:   # another process holds the port the user was given since it was chosen
                if not port:
                    raise
                proxy.stop()
                log.warning("user %d: port %d is taken (%s); the user's proxy listens on another", user, port, error)
                proxy = self.proxy(self.upstream, **settings).start()
            system = self.make_system(user)
            if isinstance(system, SystemLearner):
                system.attach(workdir / "system", proxy.environ(), tracker=tracker, log_state=True)
            else:   # a system of its own kind, which keeps its state as it does
                system.attach(workdir / "system", proxy.environ(), tracker=tracker)
            found = self._users[user] = _User(user, workdir, system, proxy, tracker, self.capture, self.call_hook)
            return found

    def state_dir(self, user: int) -> Path:
        """Where a user's harness keeps everything it persists."""
        return self.user_dir(user) / "system" / "state"

    def set_overrides(self, **fields: Any) -> None:
        """Force ``fields`` in every user's later calls (and in users not yet started)."""
        self.body_overrides.update(fields)
        with self._users_lock:
            users = list(self._users.values())
        for u in users:
            u.proxy.set_overrides(**fields)

    # -- the benchmark's hooks ---------------------------------------------------------------------

    def _known_users(self) -> list[int]:
        """Every user started in this process or with a directory of their own from an earlier launch: a resumed run
        starts a user's harness only at their first action, and their state is on disk from before."""
        with self._users_lock:
            users = set(self._users)
        for d in (self.run_dir / "users").glob("u*"):
            if d.is_dir() and d.name[1:].isdigit():
                users.add(int(d.name[1:]))
        return sorted(users)

    def begin_round(self, n_instances: int) -> None:
        with self._round_lock:
            self._round = []
            self._round_index = None
        if self.snapshot:
            # every user with state, started here or not: a resumed run's first round would otherwise keep the
            # snapshot an earlier launch took of an earlier round
            users = self._known_users()
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(lambda u: snapshot_state(self.state_dir(u), self.user_dir(u) / "state-before"), users))
        if self.begin_hook is not None:
            self.begin_hook(n_instances)

    def act(self, obs: LearnerObservation) -> Action:
        u = self._user(obs.user)
        if obs.is_fresh and (u.trace is None or u.trace.slot != obs.step):
            before = self.user_dir(obs.user) / "state-before"
            slot = self.schedule.slots[obs.step]
            trace = InstanceTrace(user=obs.user, slot=obs.step, round=obs.round, task_id=slot.task_id,
                                  task_label=obs.task_id, phase=slot.phase,
                                  user_exposure_index=self._infos[obs.step].user_exposure_index,
                                  state_before=before if self.snapshot and before.is_dir() else None,
                                  state_dir=self.state_dir(obs.user))
            with u.lock:
                u.trace = trace
            with self._round_lock:
                self._round.append(trace)
                self._round_index = obs.round
        trace = u.trace
        with u.lock:
            u.context = (obs.step, len(trace.turns), "turn")
        action = u.system.act(obs)
        with u.lock:
            trace.turns.append({"action": _jsonable(action)})
        return action

    def observe(self, obs: LearnerObservation, action: Action, feedback: Feedback) -> None:
        u = self._user(obs.user)
        trace = u.trace
        ended = _ended(feedback)
        with u.lock:
            if trace.turns:
                trace.turns[-1]["feedback"] = _jsonable(
                    dataclasses.replace(feedback, pairs=[]) if isinstance(feedback, DemosFeedback) else feedback)
                if isinstance(feedback, DemosFeedback):
                    trace.turns[-1]["feedback"]["pairs_served"] = len(feedback.pairs)
            if isinstance(feedback, SubmitFeedback) and feedback.correct:
                trace.solved = True
                trace.accepted_grid = getattr(action, "grid", None)
            if ended:
                trace.ended = True
                u.context = (obs.step, len(trace.turns) - 1, "after")  # the closing message and the storage step
        u.system.observe(obs, action, feedback)
        if ended:
            if self.router is not None:  # the closing message and storage step are done: no longer active
                self.router.release(f"u{obs.user}")
            with u.lock:
                trace.seconds = time.time() - trace.started
                u.context = (obs.step, None, "idle")
            # before the hook, and before the runner reads the user's counters for the instance
            u.settle(trace, time.monotonic() + self.calls_wait_s)
            if self.instance_hook is not None:
                self.instance_hook(trace)

    def end_round(self) -> None:
        with self._round_lock:
            instances = sorted(self._round, key=lambda t: t.slot)
            index = self._round_index
        deadline = time.monotonic() + self.calls_wait_s
        for t in instances:  # each settled as it ended; one that never ended in observe settles now
            self._user(t.user).settle(t, deadline)
        # every user's state as the round left it, the round's commit in their own history (SystemLearner.end_round),
        # before the round hook: a hook may train for hours, or fail, once the round's records are written, and a round
        # left without its commits would number every later one a round early (a state log counts its own rounds). One
        # user at a time, since a commit holds every other process of the run stopped (statelog.StateLog.commit).
        for t in instances:
            system = self._user(t.user).system
            if hasattr(system, "end_round"):
                system.end_round()
        if self.round_hook is not None and instances:
            overrides = self.round_hook(RoundData(index=index if index is not None else -1, instances=instances))
            if overrides:
                self.set_overrides(**overrides)
        for t in instances:  # the bodies were for the hook; keep the metadata in the call logs only
            t.calls = []

    def reset(self) -> None:
        """Wipe every user's harness state (the fresh reference) and, through ``reset_hook``, the model's."""
        with self._users_lock:
            users = list(self._users.values())
        with ThreadPoolExecutor(max_workers=max(1, min(32, len(users)))) as pool:
            list(pool.map(lambda u: u.system.reset(), users))
        if self.reset_hook is not None:
            self.reset_hook()

    # -- resources -----------------------------------------------------------------------------------

    def user_resources(self, user: int) -> dict[str, float]:
        with self._users_lock:
            u = self._users.get(user)
        return u.resources() if u is not None else {}

    def resources(self) -> dict[str, float]:
        with self._users_lock:
            users = list(self._users.values())
        total: dict[str, float] = {}
        for u in users:
            for k, v in u.resources().items():
                total[k] = total.get(k, 0.0) + float(v)
        return total

    def describe(self) -> dict[str, Any]:
        with self._users_lock:
            first = next(iter(self._users.values()), None)
        system = first.system.describe() if first is not None and hasattr(first.system, "describe") else {}
        return {"learner": "collective-harness", "system": system, "users_started": len(self._users),
                "capture": self.capture, "snapshot": self.snapshot, "body_overrides": self.body_overrides,
                "body_drop": list(self.body_drop), "keep_reasoning": self.keep_reasoning,
                "reasoning_tail": self.reasoning_tail, "reasoning_summary": self.reasoning_summary,
                "shortened_key": self.shortened_key, "shortened_prefix": self.shortened_prefix,
                "router": self.router.describe() if self.router is not None else None}

    def close(self) -> None:
        with self._users_lock:
            users = list(self._users.values())
            self._users = {}
        with ThreadPoolExecutor(max_workers=max(1, min(32, len(users)))) as pool:
            list(pool.map(lambda u: u.close(), users))
