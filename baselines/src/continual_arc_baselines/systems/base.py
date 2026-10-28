"""The adapter base: one conversation at a time with one long-running system.

Subclasses implement how to start, prompt, reset and stop the system. The base handles the benchmark side: rendering
observations, delivering feedback, parsing actions, re-prompting on protocol violations, restarting a system that stops
answering, and wiping state on ``reset``. It counts nothing; usage comes from the proxy.

A system sees what the runner shows every learner: the instance alone, its task under the run's name (``task-<hex>``,
:mod:`continual_arc.aliases`), which the adapter passes on as it is. A grid is rendered by the grid protocol
(:mod:`continual_arc_baselines.systems.protocol`); an ARC-AGI-3 game by the instance's
:class:`~continual_arc_baselines.systems.arc3_protocol.GameView`, which shows the cells changed since the board it last
showed, and a sequence of game actions in one reply is queued here and played without another message.

One policy lives here because it applies to every system: **a system failure costs one action, not the run.** If the
system cannot be brought back after ``max_restarts`` restarts, the action becomes a give-up, counted in
``system_forced_give_ups``, and the run continues.

Every instance begins as every other does, whichever process plays it (:meth:`SystemLearner._begin_instance`): a
stateless system is wiped before it, and any other has a conversation opened for it before its first message, after
whatever start or restart that message needs. A start alone does not give one, since openclaw's ACP bridge takes up at
every start the one conversation its state holds, and a learner is often made anew on a state an earlier process left:
a run resumed, a frozen test played from the stream's state, a test user forked from the stream's last commit.

Every turn the adapter logs (``turns.jsonl``) carries how it ended when the adapter knows it (``outcome``: the
harness's stop reason and its own error message).

Everything a system keeps between instances lives under its state directory (``state_dir``). A frozen test plays every
instance from the state the stream left there: :meth:`SystemLearner.freeze` makes each reset put a copy of that state
in its place, at the same path, instead of wiping it (:mod:`continual_arc_baselines.frozen`). A system attached with
``log_state`` commits its whole state directory at the end of every round (:mod:`continual_arc_baselines.statelog`), so
that a run can be taken up again from any round (:mod:`continual_arc_baselines.fork`) and its frozen tests played as one
round of users, each forked from the commit of the stream's last round.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from continual_arc.arc3.env import Arc3LearnerObservation
from continual_arc.env import Action, Feedback, GiveUp, LearnerObservation
from continual_arc.learner import BaseLearner

from continual_arc_baselines.attempt import ROOT
from continual_arc_baselines.frozen import put_back
from continual_arc_baselines.statelog import StateLog
from continual_arc_baselines.systems import arc3_protocol, protocol
from continual_arc_baselines.systems.protocol import Transcript
from continual_arc_baselines.tracking import instance_ended

log = logging.getLogger(__name__)


END_MODES = ("closing", "storage", "finish")
"""How a harness row's instance ends (SystemLearner ``end``)."""

EMPTY_REPLY_MODES = ("restart", "reprompt")
"""What an empty reply to an observation or a re-prompt does (SystemLearner ``empty_reply``)."""

STARTS_AT_ONCE = 8
"""How many systems one process starts at a time (SystemLearner._ensure_started). A collective round starts every
user's system together, and a harness's start can cost in proportion to what it has kept: openclaw's gateway copies its
whole agent database as it boots and compares the copy with the original before its first line of log, so a hundred
users' gateways started at once on databases of 76 to 803 MB never came up within the launcher's wait. A few at a time,
each starts in about the time it takes alone."""
_starting = threading.BoundedSemaphore(STARTS_AT_ONCE)


class SystemUnavailable(RuntimeError):
    """The system process died or stopped answering."""


def remove(path: Path) -> None:
    """A file, a link or a directory tree, as a wipe deletes the entries of a state directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


class SystemLearner(BaseLearner):
    """A learner that forwards the benchmark conversation to an external system.

    Lifecycle: :meth:`attach` (where to keep state, what environment children
    inherit, the run's tracker) is called once by the CLI before the run;
    :meth:`start` lazily on the first action; :meth:`reset` stops, wipes (or, once
    :meth:`freeze` is called, puts a saved state back) and lazily restarts;
    :meth:`end_round` commits the state when it is logged; :meth:`stop` at the end. ``demo_cost`` and ``max_cost`` (the
    configuration's whole budget, what an unsolved instance costs) are only for the
    wording of the rules.
    """

    name = "system"
    # What the harness keeps between instances, in its own terms, for the storage turn's message (storage_turn): the
    # facts its agent needs to save anything that lasts, and cannot know otherwise.
    storage_note = ""
    # Whether everything the system keeps between instances is in files under state_dir, which a frozen test puts back
    # before every instance; a learner that keeps it in memory sets False, and has no frozen tests.
    state_in_files = True

    def __init__(self, *, max_reprompts: int = 2, refine_every: int = 0, demo_cost: float = 1.0,
                 max_cost: float = 16.0, max_restarts: int = 2, session_per_instance: bool = True, instance_gap_s: float = 0.0,
                 stateless: bool = False, storage_turn: bool = False, end: str = "", empty_reply: str = "restart",
                 archive_wipes: bool = False, ask_to_store: bool = False, program_library: bool = False,
                 rules_page: str = ""):
        self.max_reprompts = max_reprompts
        # every refine_every instances (0: never), the system's own refinement after the instance (refine): prime-agent's
        # global refinement, openclaw's /learn; never for a stateless row
        self.refine_every = refine_every
        # How an instance ends once its verdict is in (``end``; storage_turn picks "storage" when it is not given):
        # - "closing": a closing turn with the verdict and "no action is needed";
        # - "storage": the verdict and a turn of the agent's own in which to save what will help later, with whatever
        #   tools its harness has (protocol.render_storage): a harness stores at will, and the turn that submits ends
        #   before the verdict comes, so without it nothing could be stored knowing the outcome. Never for a stateless
        #   row, whose state could not survive it;
        # - "finish" (every harness row of the paper's grid): the verdict alone, with no instruction, and the harness
        #   ends the instance itself with the finish action, which is refused until the instance is over, so that the
        #   harness is free to do what it wants. The rules state facts only (protocol.RULES_FINISH). Rows H and H0 are
        #   treated alike: nothing is said to one that is not said to the other.
        self.end = end or ("storage" if storage_turn else "closing")
        if self.end not in END_MODES:
            raise ValueError(f"end must be one of {END_MODES}, got {end!r}")
        self.storage_turn = self.end == "storage"
        # The ablation row HA: the finish protocol with one change, an ask to save liberally after the verdict
        # (protocol.ASK_TO_STORE), so that the row shows the most that asking a harness to store can give. Never for a
        # stateless row, whose state could not survive it.
        if ask_to_store and self.end != "finish":
            raise ValueError("ask_to_store is an addition to the finish protocol (end='finish')")
        self.ask_to_store = ask_to_store
        # Expert guidance, the rows H-E: the finish protocol's rules gain the program library's whole method as a
        # procedure (protocol.PROGRAM_LIBRARY_RULES: each task's pairs and program kept as two files of a folder of the
        # workspace, the program checked on the kept pairs and run first when the task returns, every program
        # checked on every pair before a submission, and a memory naming the task's files), and the verdict is followed
        # by the ask to save them (library_ask). A stateless row gets the same messages, so that H-E0 differs from H-E
        # only in what survives.
        if program_library and (self.end != "finish" or ask_to_store):
            raise ValueError("program_library is an addition to the finish protocol, and not beside ask_to_store")
        self.program_library = program_library
        # A page of playing rules the system is told, such as the trained rows' labeller's c1 for the rows given expert
        # guidance: shown after the storage note and before the procedure at the start of every conversation, and in
        # none of the asks that quote the storage note, so that a row is told it once an instance.
        self.rules_page = rules_page
        # An empty reply to an observation or a re-prompt. "restart": the system is taken for dead and restarted, state
        # intact, the rule the paper's earliest cells ran under (scripts/paper-grid.txt marks them). "reprompt": it is an
        # unusable reply like any other, re-prompted in the same conversation within the same max_reprompts; the system
        # is restarted only when the adapter finds its process dead or silent (SystemUnavailable), or when it has
        # answered nothing at all for longer than one action's re-prompts (max_reprompts + 2 empty replies in a row),
        # which is silence too. openclaw's empty replies are a turn whose work has gone to subagents, a model call that
        # failed, turns of tool calls alone, or text that never reached the adapter: a restart kills what the harness
        # has in flight and sends the observation again, which a re-prompt does not.
        if empty_reply not in EMPTY_REPLY_MODES:
            raise ValueError(f"empty_reply must be one of {EMPTY_REPLY_MODES}, got {empty_reply!r}")
        self.empty_reply = empty_reply
        self._empty_in_a_row = 0
        # Before a stateless row's wipe (and any other reset), copy the system's own transcripts (transcript_paths) to
        # workdir/wipe-archive/, outside the system's reach, so that its turns can be read afterwards as a persistent
        # row's can: the wipe deletes them with the rest of its state.
        self.archive_wipes = archive_wipes
        # Row H0: the same harness with its own persistence switched off. Before every instance the
        # system is stopped, everything it wrote is deleted (the fresh mode's wipe: home, memory,
        # skills, sessions, workspace, tmp; only the adapter's provider configuration survives) and
        # it is started again. Its end-of-instance storage step and idle gap are skipped, since
        # nothing could survive them. Paired against row H on the same schedule, the difference is
        # the value of persistence with the harness's within-instance skill held equal.
        self.stateless = stateless
        if stateless:
            instance_gap_s = 0.0
        # Each instance is a new piece of work: the system gets a fresh conversation for it and
        # keeps only what it persisted itself (files, memory, skills); False keeps one conversation for the whole
        # stream. The gap is idle wall-clock after an instance, for systems whose housekeeping waits for quiet
        # (openclaw).
        self.session_per_instance = session_per_instance
        self.instance_gap_s = instance_gap_s
        # Set as an instance ends and taken by the next one's first observation (_begin_instance), and set from the
        # start, so that a learner's first instance begins as every later one does: a learner made anew (a run resumed,
        # a frozen test, a test user forked from the stream's last commit) finds in its state whatever the instance
        # before it left there, a stateless system's files and a harness's last conversation alike.
        self._session_pending = True
        # A stateful system's instance has begun, and its first message waits for a conversation opened for it
        # (new_session, in _ask_resilient), whether the system is running then or has to be started or restarted: a
        # start need not open one, since openclaw's ACP bridge takes up the one conversation its state holds.
        self._conversation_pending = False
        self.demo_cost = demo_cost
        self.max_cost = max_cost
        self.max_restarts = max_restarts
        self.workdir: Path = Path(ROOT) / "system"   # until attach names the attempt's
        self._state_dir: Path | None = None  # a state directory attach named outside the workdir
        self.frozen: Path | None = None      # the saved state every reset puts back (freeze)
        self.state_log: StateLog | None = None   # the round-by-round commits of the state (attach's log_state)
        self.environ: dict[str, str] = dict(os.environ)
        self.tracker: Any = None
        self.transcript = Transcript()
        self._started = False
        self._fresh_session = True
        self.last_turn_cut = False  # set by an adapter that aborted the last turn for budget
        # How the last turn ended, as the adapter knows it (stop reason, the harness's error message, ...): logged with
        # the turn and cleared (log_turn). Adapters set it in ask().
        self.last_turn_outcome: dict[str, Any] = {}
        self._instances_done = 0
        self.stats = {"reprompts": 0, "parse_failures": 0, "restarts": 0, "resets": 0,
                      "refines": 0, "forced_give_ups": 0, "budget_hits": 0, "stalls": 0,
                      "idle_timeouts": 0, "provider_retries": 0,
                      "sessions": 0, "wipes": 0, "closings": 0, "storage_turns": 0,
                      "finishes": 0, "finish_refusals": 0, "finish_reprompts": 0, "finish_forced": 0,
                      "finish_budget_hits": 0, "empty_replies": 0, "wipe_archives": 0, "store_asks": 0,
                      "library_asks": 0, "batched_actions": 0}
        self._turns = None      # turns.jsonl, opened on first use
        self._queue: list = []  # the game actions of a sequence still to play, with no message between them
        self._view = arc3_protocol.GameView()   # the board a game's conversation last showed

    # -- to implement --------------------------------------------------------

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def ask(self, message: str, *, reprompt: bool = False) -> str:
        """Send one message, return the system's final reply text.

        ``reprompt`` marks the short follow-up after an unusable reply; systems
        with per-action budgets give it a smaller one.
        """
        raise NotImplementedError

    def instance_preface(self, obs: LearnerObservation) -> str:
        """Text put before the instance in the first message of its conversation (protocol.render_observation's
        ``preface``): nothing for a harness row; the demo-cache floor's cached pairs (systems/bare.py)."""
        return ""

    def after_instance(self) -> None:
        """Hook at the end of an instance, before its conversation is closed: a system's own
        explicit storage step goes here (prime-agent's global refine)."""

    def new_session(self) -> None:
        """Start a fresh conversation without touching persistent state."""
        raise NotImplementedError

    def wipe_state(self) -> None:
        """Delete everything the system persisted. Called while stopped."""
        raise NotImplementedError

    def save_state(self) -> None:
        """Write into :attr:`state_dir` whatever the learner keeps in memory rather than in files, before the round's
        commit (:meth:`end_round`). Nothing for a harness, whose state is its files already."""

    def load_state(self) -> None:
        """Read back what :meth:`save_state` wrote, when the learner is attached to a state directory that holds it (a
        run forked from a round, a collective test's user forked from the stream's end)."""

    def transcript_paths(self) -> list[Path]:
        """The system's own transcripts and logs (not its caches, environments or skills), copied out before a wipe
        when ``archive_wipes`` is on. Nothing by default; a harness adapter names its own."""
        return []

    def refine(self) -> None:
        """Optional self-improvement step between instances."""

    def describe(self) -> dict[str, Any]:
        """Identity for the run record: system name, version, model, settings."""
        return {"system": self.name, "max_restarts": self.max_restarts,
                "session_per_instance": self.session_per_instance, "instance_gap_s": self.instance_gap_s, "stateless": self.stateless, "storage_turn": self.storage_turn,
                "end": self.end, "empty_reply": self.empty_reply, "archive_wipes": self.archive_wipes,
                "ask_to_store": self.ask_to_store, "program_library": self.program_library,
                "rules_page": bool(self.rules_page)}

    def system_usage(self) -> dict[str, float]:
        """What the system itself reports about its usage (secondary to the proxy)."""
        return {}

    # -- lifecycle -----------------------------------------------------------

    def attach(self, workdir: Path, environ: dict[str, str], tracker: Any = None,
               state_dir: Path | None = None, log_state: bool = False) -> None:
        """``state_dir`` is ``workdir/state`` unless named: a frozen test's system keeps its logs in a workdir of its
        own and plays from the state its stream left, where the stream left it. ``log_state`` commits the state
        directory at the end of every round into ``workdir/state.git``, each round's commit listed in
        ``workdir/state-log.jsonl`` (:meth:`end_round`)."""
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._state_dir = None if state_dir is None else Path(state_dir)
        self.environ = {**os.environ, **environ}
        self.tracker = tracker
        if log_state:
            self.state_log = StateLog(self.state_dir, self.workdir / "state.git", self.workdir / "state-log.jsonl")
        self.load_state()

    @property
    def state_dir(self) -> Path:
        """Everything the system persists between instances lives under here, and a system may create siblings of its
        own folders in it (prime-agent's ``session-artifacts``). The adapter's own logs stay outside it, in the
        workdir, unreadable by the system."""
        return self._state_dir or self.workdir / "state"

    def freeze(self, state: Path) -> None:
        """Play every instance from ``state``, a state directory saved whole, from now on: a frozen test's, the state the
        stream left (:mod:`continual_arc_baselines.frozen`). It is put in :attr:`state_dir`'s place now and by every
        :meth:`reset`, in place of a wipe; a stateless row's reset wipes it after, as its stream's did."""
        self.frozen = Path(state)
        put_back(self.frozen, self.state_dir)

    def end_round(self) -> None:
        """The round is over: with ``log_state`` (:meth:`attach`) the learner's whole state becomes the round's commit,
        from which the run can be taken up again at this point."""
        if self.state_log is not None:
            self.save_state()
            self.state_log.commit()

    def _ensure_started(self) -> None:
        if not self._started:
            with _starting:   # STARTS_AT_ONCE
                self.start()
            self._started = True
            self._fresh_session = True

    def reset(self) -> None:
        if self._started:
            self.stop()
            self._started = False
        if self.frozen is None:
            self._archive_before_wipe()
        else:   # what the instance wrote goes, its transcripts with it
            put_back(self.frozen, self.state_dir)
        if self.frozen is None or self.stateless:
            self.wipe_state()
        self.transcript = Transcript()
        self.stats["resets"] += 1

    def _archive_before_wipe(self) -> None:
        """The last instance's transcripts, into ``workdir/wipe-archive/instance-NNNN.tar.gz`` (the system is stopped)."""
        if not self.archive_wipes or self._instances_done == 0:
            return
        paths = [p for p in self.transcript_paths() if p.exists()]
        if not paths:
            return
        dest = self.workdir / "wipe-archive" / f"instance-{self._instances_done - 1:04d}.tar.gz"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(dest, "w:gz") as tar:
                for path in paths:
                    tar.add(path, arcname=str(path.relative_to(self.workdir)) if path.is_relative_to(self.workdir)
                            else path.name)
            self.stats["wipe_archives"] += 1
        except OSError as error:
            self.note(f"archive before the wipe failed: {error}")

    def close(self) -> None:
        if self._started:
            self.stop()
            self._started = False
        if self._turns is not None:
            self._turns.close()
            self._turns = None

    # -- the benchmark loop --------------------------------------------------

    def note(self, message: str) -> None:
        """Diagnostics hook; systems may override to keep a log with the evidence."""
        log.warning(message)

    def log_turn(self, kind: str, message: str, reply: str, **meta: Any) -> None:
        """The protocol conversation, one line per exchange, in ``turns.jsonl`` beside the
        run: what the adapter sent, what came back, when, and which instance it belonged to.
        The harnesses' own event streams and stores are partial records by their design
        (prime-agent's daemon injects its digest only when the store changed and drops
        events under load; the ACP harnesses return one text per turn and keep the rest
        in their own databases), so the adapter, which sees every exchange for every
        harness, is the one place a complete record can be written."""
        try:
            if self._turns is None:
                self._turns = (self.workdir / "turns.jsonl").open("a")
            outcome = {"outcome": self.last_turn_outcome} if self.last_turn_outcome else {}
            self._turns.write(json.dumps({"t": time.time(), "kind": kind, "instance": self._instances_done,
                                          "session": self.stats.get("sessions", 0), **meta, **outcome,
                                          "message": message, "reply": reply}, default=str) + "\n")
            self._turns.flush()
        except OSError as error:
            log.warning("turn log: %s", error)
        self.last_turn_outcome = {}

    def _ask_resilient(self, render: Callable[[bool], str], *, reprompt: bool = False, kind: str = "turn",
                       **meta: Any) -> str:
        """Ask, restarting the system up to ``max_restarts`` times if it fails.

        ``render(fresh)`` produces the message; ``fresh`` is True when the system
        has just (re)started and needs the rules again.
        """
        for attempt in range(self.max_restarts + 1):
            try:
                self._ensure_started()
                if self._conversation_pending:
                    self.new_session()
                    self._conversation_pending = False
                    self.stats["sessions"] += 1
                fresh, self._fresh_session = self._fresh_session, False
                message = render(fresh)
                started = time.monotonic()
                self.last_turn_outcome = {}
                reply = self.ask(message, reprompt=reprompt)
                self.log_turn("reprompt" if reprompt else kind, message, reply, seconds=round(time.monotonic() - started, 2),
                              fresh=fresh, **meta)
                if reply.strip() or self.last_turn_cut:
                    # a turn the adapter cut short for budget was the system working, and its follow-up carries the
                    # budget notice
                    self._empty_in_a_row = 0
                    return reply
                self.stats["empty_replies"] += 1
                self._empty_in_a_row += 1
                if self.empty_reply == "restart":
                    # a system that answers with nothing is taken to have stopped functioning as surely as one that
                    # crashed (its context exhausted, or its model call failed inside it), and is restarted, state
                    # intact
                    raise SystemUnavailable("empty reply")
                if self._empty_in_a_row > self.max_reprompts + 1:
                    # nothing at all for longer than one action's re-prompts: silence, as a dead process is
                    self._empty_in_a_row = 0
                    raise SystemUnavailable(f"no text in {self.max_reprompts + 2} replies in a row")
                # an unusable reply like any other: the caller's parse refuses it and re-prompts
                return reply
            except SystemUnavailable as error:
                self.close()
                if attempt == self.max_restarts:
                    raise
                self.stats["restarts"] += 1
                self.note(f"system unavailable ({error}); restart {attempt + 1}/{self.max_restarts}")
        raise SystemUnavailable("unreachable")

    def _begin_instance(self, obs: LearnerObservation) -> None:
        """At an instance's first observation, once (``_session_pending``): a stateless system is stopped, everything it
        wrote is deleted and it starts again on the next ask; any other gets a fresh conversation before the instance's
        first message (``_conversation_pending``), so that a system made anew on a state an earlier process left never
        plays on in the conversation that state holds."""
        if not (obs.is_fresh and self._session_pending):
            return
        self._session_pending = False
        if self.stateless:
            self.reset()
            self.stats["wipes"] += 1
            self.stats["sessions"] += 1
        else:
            self._conversation_pending = True

    def protocol_for(self, obs: Any) -> Any:
        """What renders ``obs`` and parses the replies to it: the grid protocol module, or for a game the instance's
        GameView, made anew at its first observation."""
        if not isinstance(obs, Arc3LearnerObservation):
            return protocol
        if obs.is_fresh:
            self._view = arc3_protocol.GameView()
        return self._view

    def act(self, obs: LearnerObservation | Arc3LearnerObservation) -> Action:
        if self._queue:
            # the rest of the sequence the system sent in its last reply, emptied when the instance ended or a game
            # action stopped it (observe)
            self.stats["batched_actions"] += 1
            return self._queue.pop(0)
        self._begin_instance(obs)
        proto = self.protocol_for(obs)
        if self.program_library and proto is not protocol:
            raise ValueError("expert guidance (program_library) is the grid tracks' procedure")
        pending = self.transcript.take()
        finish = self.end == "finish"
        procedure = protocol.PROGRAM_LIBRARY_RULES.format(programs=self.programs_dir()) if self.program_library else ""
        note = "\n\n".join(x for x in (self.storage_note, self.rules_page, procedure) if x)
        extra = {"finish": True, "rules_note": note} if finish else {}

        def observation(fresh: bool) -> str:
            preface = self.instance_preface(obs) if fresh else ""
            return proto.render_observation(obs, pending=pending, demo_cost=self.demo_cost, max_cost=self.max_cost,
                                            first_in_session=fresh,
                                            **({**extra, "preface": preface} if preface else extra))

        try:
            reply = self._ask_resilient(observation, kind="observation", step=obs.step, task=obs.task_id)
            for attempt in range(self.max_reprompts + 1):
                try:
                    parsed = proto.parse_action(reply, finish=finish)
                    if isinstance(parsed, protocol.Finish):
                        # finish while the instance can still be played: refused, like any action the protocol lacks
                        self.stats["finish_refusals"] += 1
                        raise protocol.ParseError("finish is refused until the instance is over")
                    if isinstance(parsed, list):   # a sequence of game actions
                        first, self._queue = parsed[0], parsed[1:]
                        return first
                    return parsed
                except protocol.ParseError as error:
                    self.stats["parse_failures"] += 1
                    self.note(f"step {obs.step}: unusable reply ({error}); tail: {reply[-300:]!r}")
                    if attempt == self.max_reprompts:
                        self.note(f"step {obs.step}: giving up after {attempt + 1} unusable replies")
                        return GiveUp()
                    self.stats["reprompts"] += 1
                    hint = (proto.REPROMPT_FINISH if finish else proto.REPROMPT).format(reason=error)

                    # A system restarted mid-reprompt has lost the instance: resend it.
                    def follow_up(fresh: bool, hint: str = hint) -> str:
                        return observation(True) + "\n\n" + hint if fresh else hint

                    reply = self._ask_resilient(follow_up, reprompt=True, step=obs.step, task=obs.task_id)
        except SystemUnavailable as error:
            self.stats["forced_give_ups"] += 1
            self.note(f"step {obs.step}: system unavailable after {self.max_restarts} restart(s) "
                      f"({error}); this action is a give-up")
        return GiveUp()

    def observe(self, obs: LearnerObservation | Arc3LearnerObservation, action: Action, feedback: Feedback) -> None:
        ended = instance_ended(feedback)  # the rule by which TrackedLearner waits for the instance's calls
        game = isinstance(obs, Arc3LearnerObservation)
        if self._queue and (ended or not feedback.valid or feedback.state == "GAME_OVER"):
            # a sequence stops where the system must look again: the level over, or an action refused or ignored
            self._queue = []
        self.transcript.add_text((arc3_protocol if game else protocol).render_feedback(action, feedback))
        if ended:
            self._instances_done += 1
            self._close_instance(obs)
            if self._refine_due():
                self.stats["refines"] += 1
                self.refine()
            if not self.stateless:
                try:
                    self.after_instance()
                except SystemUnavailable as error:
                    self.note(f"step {obs.step}: after-instance step failed ({error})")
            if self.instance_gap_s > 0 and self.frozen is None:
                # frozen, the next instance starts from the saved state, so housekeeping in the gap would be lost
                time.sleep(self.instance_gap_s)
            if self.session_per_instance:
                self._session_pending = True

    def library_files(self, task_id: str) -> dict[str, str]:
        """The expert guidance's two files for ``task_id`` (protocol.PROGRAM_LIBRARY_RULES), its program and its kept
        pairs, with the task's id as the agent sees it, which names the task's memory."""
        return {"program": f"{self.programs_dir()}/{task_id}.py", "pairs": f"{self.programs_dir()}/{task_id}.json",
                "task": task_id}

    def library_ask(self, task_id: str) -> str:
        """The ask after each verdict for a row given expert guidance (protocol.PROGRAM_LIBRARY_ASK): save this task's
        pairs, its program and its memory."""
        return protocol.PROGRAM_LIBRARY_ASK.format(note=f" {self.storage_note}" if self.storage_note else "",
                                                   **self.library_files(task_id))

    def teacher_lesson(self, task_id: str) -> str:
        """The teacher's page for ``task_id`` (protocol.TEACHER_LIBRARY_LESSON), for a teacher keeping this harness's
        notes."""
        return protocol.TEACHER_LIBRARY_LESSON.format(note=f" {self.storage_note}" if self.storage_note else "",
                                                      programs=self.programs_dir(), **self.library_files(task_id))

    def programs_dir(self) -> str:
        """Row HPL's program library: a folder of the harness's workspace (every harness adapter has one, kept between
        instances except in a stateless row), named by its absolute path so that no harness has to guess its cwd."""
        workspace = getattr(self, "workspace", None)
        return str(Path(workspace) / "programs") if workspace else "programs"

    def _refine_due(self) -> bool:
        """Whether the instance that has just ended is followed by the system's refinement (every ``refine_every``
        instances; never for a stateless row, whose state could not survive it)."""
        return bool(self.refine_every) and self._instances_done % self.refine_every == 0 and not self.stateless

    def _close_instance(self, obs: LearnerObservation) -> None:
        """Deliver the verdict on the instance's final submission to the system, in the same
        conversation, before any storage step runs and before the conversation is closed. Without
        it the verdict would wait in the transcript until the next instance's first message, in a new
        conversation about another task, and a harness's storage step would review a trajectory whose
        outcome it did not know. One turn per instance; the reply is not read. With
        ``storage_turn`` the turn is the agent's own to save what will help later
        (:func:`protocol.render_storage`), logged as kind ``storage``; else a short closing."""
        pending = self.transcript.take()
        if not self._started:
            return
        if self.end == "finish":
            self._finish_instance(obs, pending)
            return
        storage = self.storage_turn and not self.stateless
        try:
            if storage:
                message = protocol.render_storage(pending, conversation_ends=self.session_per_instance,
                                                  note=self.storage_note)
            else:
                message = protocol.render_closing(pending)
            started = time.monotonic()
            self.last_turn_outcome = {}
            reply = self.ask(message)   # the reply is not read
            self.log_turn("storage" if storage else "closing", message, reply,
                          seconds=round(time.monotonic() - started, 2), step=obs.step, task=obs.task_id)
            self.stats["storage_turns" if storage else "closings"] += 1
        except SystemUnavailable as error:
            # a system that cannot take the closing is restarted for the next instance; the
            # verdict is lost to it
            self.note(f"step {obs.step}: closing message failed ({error}); restarting the system")
            self.close()
            self.stats["restarts"] += 1

    def _finish_instance(self, obs: LearnerObservation, pending: list[str]) -> None:
        """The end of an instance under the finish protocol: the verdict alone, in the same conversation, and the
        harness does what it wants (store, or nothing) until it replies with finish. Any other action is refused with
        the fact that the instance is over (a submission cannot change the scored record); after ``max_reprompts``
        such replies, or when the harness spends its turn's budget, the instance ends here as if it had finished."""
        message = "\n".join(pending) if pending else "The instance is over."
        parse = arc3_protocol.parse_action if isinstance(obs, Arc3LearnerObservation) else protocol.parse_action
        if self.ask_to_store and not self.stateless:
            message += "\n\n" + protocol.ASK_TO_STORE.format(note=f" {self.storage_note}" if self.storage_note else "")
            self.stats["store_asks"] += 1
        if self.program_library:
            message += "\n\n" + self.library_ask(obs.task_id)
            self.stats["library_asks"] += 1
        try:
            for attempt in range(self.max_reprompts + 1):
                started = time.monotonic()
                self.last_turn_outcome = {}
                reply = self.ask(message, reprompt=attempt > 0)
                self.log_turn("verdict" if attempt == 0 else "finish_reprompt", message, reply,
                              seconds=round(time.monotonic() - started, 2), step=obs.step, task=obs.task_id)
                if self.last_turn_cut:
                    # the harness used its whole turn after the verdict: the instance ends here, with no notice
                    self.stats["finish_budget_hits"] += 1
                    if hasattr(self, "_budget_pending"):
                        self._budget_pending = False
                    return
                try:
                    parsed = parse(reply, finish=True)
                    if isinstance(parsed, protocol.Finish):
                        self.stats["finishes"] += 1
                        return
                    name = {"Submit": "a submission", "RequestDemos": "a demo request"}.get(type(parsed).__name__,
                                                                                           "a game action")
                    reason = f"{name} is refused"
                except protocol.ParseError as error:
                    reason = str(error)
                if attempt == self.max_reprompts:
                    self.stats["finish_forced"] += 1
                    self.note(f"step {obs.step}: no finish after {attempt + 1} replies; the instance ends here")
                    return
                self.stats["finish_reprompts"] += 1
                message = protocol.OVER_REPROMPT.format(reason=reason)
        except SystemUnavailable as error:
            # a system that cannot take the verdict is restarted for the next instance
            self.note(f"step {obs.step}: verdict turn failed ({error}); restarting the system")
            self.close()
            self.stats["restarts"] += 1

    def resources(self) -> dict[str, float]:
        out = {f"system_{k}": float(v) for k, v in self.stats.items()}
        out.update(self.system_usage())
        return out
