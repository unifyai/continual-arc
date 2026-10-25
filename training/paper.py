"""The paper driver: plays `arc1-small` served to its users one round per call, for the block's rows.

The paper rows are driven from here, because this folder's loop.py already has what a paper row needs: a runner that resumes under a longer schedule
(`continual_arc.runner.run(..., resume=True)`), baselines' `CollectiveHarnessLearner` with prime-agent, every call
captured with token ids, and Qwen3.5's adapter-serving fix (`loop.served_adapter`).

Rows:
  h0    frozen, harness wiped before every instance (the control; on the held-out tasks, transfer's reset reference)
  pg    trained (train.pg_items: plain policy gradient, REINFORCE on the won games alone, a win's reward 1 and a
        failure's 0, no baseline), harness memory kept
  opsd  trained with dagger's fixed one-round lag (realtime.py): dagger's phase with on-policy self-distillation
        in SDFT's form in place of imitation (train.SelfDistillation: the frozen base model, reading the dagger row's
        label at a call as a demonstration, grades a fresh reply of the student's at that call), harness memory kept
        (train.py's ``--objective opd``, on-policy distillation from a separately scored teacher, is not a paper row)
  dagger  trained with a fixed one-round lag (realtime.py, which plays its rounds through :func:`play_round`, as it
        does PG's and OPSD's for the paper): the oracle,
        GLM-5.3-flash, writes its own reply to every player call, and a resident trainer fine-tunes on every reply
        so far, round r played by the model trained on rounds up to r-2; harness memory kept
  h     frozen, harness memory kept (the "should" row, prime-agent's own continual harness)
  bare  frozen, the bare model with no harness (baselines' ``BareModelLearner`` at the demo-cache floors' settings,
        :data:`FLOOR`), which keeps nothing from one instance to the next: on the held-out tasks, transfer's floor
        reference
Only h0 is wiped. The trained rows keep memory because the story under test is that training teaches the model to
store a solved puzzle's program and fetch it when the puzzle returns, so a returning puzzle gets cheaper, which a
wiped row could never show (the oracle's replies, which opsd and dagger both learn from, coach that habit:
continual_arc_baselines.systems.protocol.PROGRAM_LIBRARY_RULES, oracle_hint.library_rules). "Wiped" is baselines' own stateless mode
(`continual_arc_baselines.systems.base.SystemLearner.stateless`): before every instance the system is stopped,
everything it wrote is deleted, and it restarts; rows pg, opsd, dagger and h leave it off.

The mechanism, one round r = 0, 1, 2, ...: paper.sh serves the adapter round r needs (the base model at r = 0, or
round r-1's saved adapter), calls this file's ``play`` for round r (below), and, for a trained row, labels round r
(opsd only: the dagger row's labeller) and trains on it (train.py; opsd's fresh replies are drawn from the very same
server play just used), saving the new adapter for round r+1: the synchronous protocol. The paper's trained rows' rounds
are realtime.py's instead, with a fixed one-round lag.

``play`` plays the schedule (:func:`schedule_cut`: ``Schedule.first_rounds(r + 1)``, a literal
prefix) with ``continual_arc.runner.run(resume=True)``: every whole round already in the run directory's
``records.jsonl`` is restored and the run continues from the next one, so calling ``play`` again for the same
round is a cheap no-op and calling it for round r plays exactly that one new round. The run directory is one
continuous benchmark run (``records.jsonl``, ``schedule.json``, ``meta.json``, ``summary.json``: `continual-arc
report`/`compare`'s own format), while a round trains on its own games alone, and every user plays again every round
under the same user id. :func:`save_instance_hook` therefore shards captured calls under ``rounds/rNNNN/`` by the
round they were played in (``round r``'s own `train.load_calls` call reads only that round, and a flat ``captured/``
would have a later round overwrite an earlier one's files); the benchmark's own ``records.jsonl`` at the run
directory's root is unaffected and still holds every round in order, which is where ``train.load_calls`` finds
each game's cost.

The epilogue is the schedule's last 100 rounds (:func:`full_schedule`: the stream's 200, then every user meets the 100
held-out tasks once each, in an order of their own). The paper's runs end with the stream's 200 rounds and are then
tested frozen (frozen.py); a run given more rounds plays the epilogue on as in the stream, training on and its users'
memory carried over, though no measure reads it. The held-out test's instances are those user 0 meets in the epilogue
(``continual_arc.configs.test_schedule``), so the epilogue alone serves transfer's references: h0, the learner reset,
and bare, the bare model, each played over the epilogue alone (:func:`epilogue_cut`,
``continual_arc.configs.epilogue_for_users``) in a run directory of its own (paper.sh's EPILOGUE=1). Neither carries
anything from one instance to the next, and a user's held-out instances are the same at any number of users, so one
play of each over users 0-3 serves every trained run at 1, 2 or 4 users, its instances dealt wider than four to a round
(:func:`wide_epilogue_cut`). paper.sh's synchronous trained rows can still play the epilogue alone from round R's
adapter (``--init-adapter``), with an empty memory.

Subcommands (on the GPU node, from prime-rl's checkout, as loop.py's:
``uv run --no-sync python ~/continual-arc/training/paper.py ...``):
  play          play round ``--round`` of ``--row``'s schedule (the whole, or ``--epilogue-users``' epilogue alone)
                under the served model ``--served``, into ``--run-dir``; resumable, see above. ``--users`` serves the
                schedule to that many users (:func:`served`; the paper rows' 100 by default): users 0..N-1 are exactly
                any larger number's own first N, so a smaller run's rows are a true prefix of a larger one's, never a
                different draw
  check-served  the served-adapter check: the fixed prompt's reply through ``--served`` differs from the base
                model's (vLLM silently serves the base model when an adapter's layer names are wrong)
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import logging
import shutil
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import loop  # noqa: E402  (this folder's loop.py: MODEL, TEMPLATE_KWARGS, HARNESS, harness_of, HARNESS_BUDGETS)

STREAM_CONFIG = loop.STREAM_CONFIG  # "arc1-small"
# The users the full schedule is built for by default. A run sets its own (--users): the benchmark's canonical numbers
# are 1, 2, 4, 8 and 16 (continual_arc.configs.CANONICAL_USERS), so 100 is not one of them, and the paper's runs serve
# the first 1, 2 or 4 of these users.
PAPER_USERS = 100
SEED = 0  # workstreams, seed 0: the schedule every row plays

ROWS = ("h0", "pg", "opsd", "dagger", "h", "bare")
# Wiped before every instance (baselines' stateless mode): h0 alone, the control. pg, opsd and dagger keep
# prime-agent's own memory (the harness's files), so a trained row can show a returning puzzle getting cheaper through
# what it saved; h is the frozen row that keeps memory, the trained rows' reference. bare has no harness and keeps
# nothing between instances by construction.
STATELESS_ROWS = ("h0",)
# Transfer's two references on the held-out tasks, the learner reset and the bare model: neither keeps anything from one
# instance to the next, so their instances may be dealt wider than the run's users (wide_epilogue_cut)
TRANSFER_REFERENCES = ("h0", "bare")
# The bare model's settings, every paper floor's (the baselines' floors, reference.py's demo-cache floors): the
# harness rows' rules (end "finish"), an empty reply asked again, a wipe archived first
FLOOR = {"end": "finish", "empty_reply": "reprompt", "archive_wipes": True}

# The shared settings every row plays under: temperature 1, top_p 1, no top_k forced on every call, with
# return_token_ids for the harness rows, whose calls train (bare's streamed calls never do). loop.HARNESS_BUDGETS
# ["block"] already codifies the call cap and its time budget, and loop.HARNESS the compaction point -- read from there,
# never hard-coded here.
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
# every row's context window and reply cap
CONTEXT_WINDOW, MAX_TOKENS = 131_072, 32_768

DEFAULT_EPILOGUE_ROUND = 20
DEFAULT_EPILOGUE_USERS = 10

# The served-adapter check's fixed prompt: vLLM silently serves the base model under an adapter's name when its
# layer names are wrong, so one chat turn, nothing from the benchmark itself, catches it before a whole round plays
# on an adapter that never loaded.
SERVED_ADAPTER_PROMPT = [{"role": "user",
                         "content": "In one short sentence, what should an agent do when it is unsure whether its "
                                    "last answer was correct?"}]
# The check reads the replies' tokens and, at each, the log-probabilities of the top SERVED_TOP_LOGPROBS candidates,
# never the reply's text: Qwen3.5 opens with its reasoning, which vLLM's reasoning parser moves out of ``content``,
# so 16 tokens leave ``content`` empty for any weights. When both replies are the same tokens, the largest gap between
# a shared candidate's two log-probabilities must reach SERVED_LOGPROB_GAP. Measured on a node: the base model
# asked twice gave a largest gap of exactly 0.0 (one request at a time, the same kernels); OPD's adapter after 8 steps
# at 2e-5 gave the same 16 tokens with a largest gap of 0.375 (mean 0.108), while the chosen tokens' own
# log-probabilities, all near 0, moved by under 1e-3.
SERVED_TOP_LOGPROBS = 5
SERVED_LOGPROB_GAP = 1e-2

# Whether a round cut short is taken up from its last model call (:class:`Resume`) instead of played again whole by the
# model. Off: the paper's runs played without it, a cut round played again whole.
RESUME_CALLS = False


# -- schedules -------------------------------------------------------------------------------------------

def served(users: int = PAPER_USERS) -> Any:
    """`arc1-small` served to ``users`` people at once (``continual_arc.configs.with_users``). User u's stream depends
    only on the seed and u (``continual_arc.configs._user_seeds``), and task selection only on the dataset and
    ``n_tasks`` (``config_tasks``), neither on how many users there are, so the first N users of any larger number are
    exactly N users' own, round by round (this file's own test)."""
    from continual_arc.configs import with_users
    return with_users(STREAM_CONFIG, users)


def full_schedule(seed: int = SEED, users: int = PAPER_USERS) -> Any:
    """The whole schedule, built once: 300 rounds of ``users`` instances (:func:`served`), the workstreams stream's
    200 and then the epilogue's 100, in which every user meets the 100 held-out tasks once each. :func:`schedule_cut`
    slices it rather than rebuilding a shorter one, which would not be this schedule's own prefix
    (``Schedule.first_rounds``)."""
    from continual_arc.configs import build_schedule
    return build_schedule(served(users), "workstreams", seed)


def schedule_cut(round_: int, seed: int = SEED, users: int = PAPER_USERS) -> Any:
    """The schedule cut at ``round_ + 1`` rounds: a literal prefix of :func:`full_schedule`, so resuming under it after
    round ``round_ - 1`` was played plays exactly round ``round_``, a stream round up to 199 and an epilogue round
    from 200."""
    return full_schedule(seed, users).first_rounds(round_ + 1)


def epilogue_cut(round_: int, epilogue_users: int = DEFAULT_EPILOGUE_USERS, seed: int = SEED,
                 users: int = PAPER_USERS) -> Any:
    """The K-user epilogue alone, cut the same way as :func:`schedule_cut`: round ``round_`` of the held-out tasks
    served to the first ``epilogue_users`` of ``users``, in the exact order and at the exact instance seeds the full
    schedule's own epilogue would give them (``continual_arc.configs.epilogue_for_users``)."""
    from continual_arc.configs import epilogue_for_users
    return epilogue_for_users(served(users), seed, epilogue_users).first_rounds(round_ + 1)


def wide_epilogue_cut(round_: int, width: int, epilogue_users: int, seed: int = SEED,
                      users: int = PAPER_USERS) -> Any:
    """The K-user epilogue alone (:func:`epilogue_cut`'s instances, each the same task at the same instance seed) dealt
    ``width`` at a time, in its own order, into rounds of their own, each instance a user of its own, cut at
    ``round_ + 1`` rounds: transfer's references' schedule. A row that keeps nothing from one instance to the next (h0,
    bare) meets each instance as it would in the K-user epilogue, so dealing them wider changes only how many play at
    once and how few rounds wait on their slowest game: users 0-3's 400 instances are 20 rounds 20 wide, not 100
    rounds 4 wide. The records' user is the instance's place in its round; its task and instance seed, which transfer
    matches on, are the epilogue's."""
    from continual_arc.configs import epilogue_for_users
    from continual_arc.schedule import Schedule, Slot
    epilogue = epilogue_for_users(served(users), seed, epilogue_users)
    if len(epilogue.slots) % width:
        raise ValueError(f"width {width} does not divide the epilogue's {len(epilogue.slots)} instances into whole "
                         "rounds")
    slots = [Slot(s.task_id, s.instance_seed, s.phase, user=i % width) for i, s in enumerate(epilogue.slots)]
    wide = Schedule(name=f"{epilogue.name}-wide{width}", seed=seed, slots=slots, sources=epilogue.sources,
                    meta={**epilogue.meta, "dealt": {"width": width, "from_users": epilogue_users}}, users=width)
    return wide.first_rounds(round_ + 1)


# -- capture, sharded by round ---------------------------------------------------------------------------

def round_dir(run_dir: Path, round_: int) -> Path:
    """Where round ``round_``'s own captured calls (and, once scored or trained, its teacher scores and adapter)
    live under ``run_dir``: a directory train.py's ``--train-runs`` can be pointed at alone, so a round trains on its
    own games, never another round's."""
    return run_dir / "rounds" / f"r{round_:04d}"


CALL_FIELDS = ("call_id", "turn", "phase", "status", "finish_reason", "request", "response")


def captured(call: Any) -> dict[str, Any]:
    """A call as the run keeps it when the model made it: every field the learner captured (baselines'
    ``CapturedCall``)."""
    return dict(vars(call))


def played(trace: Any) -> list[dict[str, Any]]:
    """An instance's calls as the run keeps them when the model made them all (:func:`captured`), in the order the
    learner keeps them: the order they ended."""
    return [captured(c) for c in trace.calls]


def save_instance_hook(run_dir: Path, calls: Callable[[Any], list[dict[str, Any]]]):
    """An instance_hook (``CollectiveHarnessLearner``) that saves each instance exactly as ``loop.py``'s own
    ``save_instance`` does, under :func:`round_dir` instead of one flat ``captured/``: the played run directory is
    continuous (every round, for the whole stream), but a round's training must read only its own games. An instance's
    calls are saved as ``calls`` gives them: :func:`played`, or a resumed round's :meth:`Resume.calls`."""

    def hook(t: Any) -> None:
        captured_dir = round_dir(run_dir, t.round) / "captured"
        captured_dir.mkdir(parents=True, exist_ok=True)
        doc = {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
               "phase": t.phase, "solved": getattr(t, "solved", None), "turns": t.turns,
               "calls": [{k: c[k] for k in CALL_FIELDS} for c in calls(t)]}
        tmp = captured_dir / f"u{t.user:03d}.json.gz.tmp"
        try:
            # default=str, never fatal: as loop.py's save_instance (a harness's turn record can hold values JSON has
            # no form for; one instance's save failing must not cost the round the rest of its instances)
            with gzip.open(tmp, "wt") as fp:
                json.dump(doc, fp, default=str)
            tmp.rename(captured_dir / f"u{t.user:03d}.json.gz")
        except Exception:  # noqa: BLE001
            logging.exception("could not save instance of user %s, round %s", t.user, t.round)

    return hook


# -- a round cut short, taken up from its last model call ------------------------------------------------

def user_dir(run_dir: Path, user: int) -> Path:
    """A user's folder in a run (``CollectiveHarnessLearner.user_dir``): their harness's state under ``system/state``,
    its turn log ``system/turns.jsonl``, and, while a round is played with :data:`RESUME_CALLS`, ``round-start``, the
    state the round began with."""
    return Path(run_dir) / "users" / f"u{user:03d}"


def journal_dir(run_dir: Path, round_: int) -> Path:
    """Where a round's model calls are kept as they end while it is played (:class:`Resume`), until it is whole."""
    return round_dir(run_dir, round_) / "journal"


def round_recorded(run_dir: Path, schedule: Any) -> bool:
    """Whether ``run_dir``'s records already hold every slot of ``schedule`` (a cut of the stream), so that playing it
    changes nothing, its users' memory least of all."""
    from continual_arc.runner import prior_records, whole_rounds
    return len(whole_rounds(prior_records(run_dir), schedule)) >= len(schedule.slots)


def free_port() -> int:
    """A port on the loopback no process listens on now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _message(response: dict | None) -> dict:
    return (((response or {}).get("choices") or [{}])[0]).get("message") or {}


class Resume:
    """One attempt at playing round ``round_`` of ``run_dir`` with every model call kept as it ends, so that when the
    attempt is cut short (a node preempted, a crash) the next one takes the round up from its last call instead of
    asking the model for the whole round again (:data:`RESUME_CALLS`).

    The runner's contract stays as it is: ``records.jsonl`` holds whole rounds only, and a resumed run plays the cut
    round again from its start. What changes is what the model is asked. Every call an instance keeps is written to
    the round's journal as it ends (:meth:`journal`, the learner's ``call_hook``), request and reply whole, token ids
    and all, with its seconds. The first attempt snapshots every user's harness state before any game begins
    (``user_dir/round-start``, baselines' ``snapshot_state`` with no size limit, ``round-start.done`` written once all
    are whole), and a later one puts that state back (``restore_state``): a trained row keeps prime-agent's memory,
    which the cut attempt's games wrote to. Then each user's game is played through ``loop.replay_proxy`` from its
    latest recording (:meth:`recording`): a finished game is replayed whole, a cut one to its last call and then played
    on by the model; the environment answers as it did, its instances being seeded. A replayed call takes no
    generation: the recorded reply is served at once and its seconds are counted on its turn's clock. A replayed call is
    kept as recorded (:meth:`kept`), in the journal and the captured game alike, its id, request and reply the
    recording's, so the labeller's label for it still holds and its training item is the one the model played; the
    captured game keeps its calls in the order they began (:meth:`calls`), as a recording is read. A stateless row is
    wiped before every attempt anyway, and keeps no snapshot.

    Each attempt's start is logged in the journal's ``attempts.jsonl``. Once the round is whole (:meth:`finish`), its
    folder's ``resume.json`` says what each attempt started from and how the last one went, by user (calls replayed and
    played live, a game handed over to the model at the end of its recording or drifted from it), ``replay.jsonl``
    holds each hand-over and drift, and the journal and the snapshots are deleted."""

    def __init__(self, run_dir: Path, round_: int, users: int, stateless: bool) -> None:
        import loop
        from continual_arc_baselines.collective import restore_state, snapshot_state
        from continual_arc_baselines.proxy import LLMProxy
        self.run_dir, self.round, self.users = Path(run_dir), round_, users
        self.dir = journal_dir(run_dir, round_)
        for old in self.run_dir.glob("rounds/r*/journal"):   # a whole round's, left by an attempt that died after it
            if old != self.dir:
                shutil.rmtree(old)
        self.dir.mkdir(parents=True, exist_ok=True)
        log = self.dir / "attempts.jsonl"
        self.attempts = [json.loads(line) for line in open(log)] if log.exists() else []
        self.attempt = len(self.attempts) + 1
        # each user's proxy listens on one port for every attempt at the round: a model can write its proxy's address
        # into its code (a replayed reply that called the model through the recording's port was refused at the
        # replay's, and the game drifted)
        ports = self.dir / "ports.json"
        if not ports.exists():
            ports.write_text(json.dumps({u: free_port() for u in range(users)}))
        self.ports = {int(u): p for u, p in json.loads(ports.read_text()).items()}
        if not stateless:
            if (self.dir / "round-start.done").exists():
                for u in range(users):
                    restore_state(user_dir(run_dir, u) / "round-start", user_dir(run_dir, u) / "system" / "state")
            else:
                for u in range(users):
                    snapshot_state(user_dir(run_dir, u) / "system" / "state", user_dir(run_dir, u) / "round-start",
                                   max_file_bytes=None)
                (self.dir / "round-start.done").write_text(json.dumps({"round": round_, "at": time.time()}))
        plan = []
        for u in range(users):
            calls, ended = self.recording(u)
            if calls:
                source = self.dir / f"sources-a{self.attempt}" / f"u{u:03d}.json.gz"
                source.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(source, "wt", compresslevel=1) as fp:
                    json.dump(calls, fp)
                plan.append({"user": u, "source": str(source), "branch": None, "reply": None, "cut": not ended,
                             "calls": len(calls)})
        plan_file = self.dir / f"plan-a{self.attempt}.jsonl"
        plan_file.write_text("".join(json.dumps(p) + "\n" for p in plan))
        self.proxy = loop.replay_proxy(LLMProxy, plan_file, round_dir(run_dir, round_) / "replay.jsonl")
        self.at = time.time()   # before any call: a call journaled with an earlier start was replayed from a recording
        self.started = {"attempt": self.attempt, "at": self.at, "replaying": {p["user"]: p["calls"] for p in plan}}
        with open(log, "a") as fp:
            fp.write(json.dumps(self.started) + "\n")

    def recording(self, user: int) -> tuple[list[dict[str, Any]], bool]:
        """User ``user``'s game as the attempts so far left it, its calls in the order they were made, and whether the
        game ended in it: the journal of the latest attempt that made calls of its own, of an attempt that replayed
        the game only once its replay was handed over or drifted (``replay.jsonl``); one that only replayed part of
        the recording before it was cut adds nothing to it. Each cancelled call made in that attempt gets
        ``partial``, the reply the harness had read when it cut the call's turn at its budget: the first of the user's
        turn rows logged during the attempt after the call began, if it was cut at the budget
        (``loop.recorded_calls`` reads a captured game alike)."""
        best: list[dict[str, Any]] = []
        ended = False
        replays = round_dir(self.run_dir, self.round) / "replay.jsonl"
        events = [e for e in map(json.loads, open(replays)) if e.get("user") == user] if replays.exists() else []
        for k, attempt in enumerate(self.attempts, 1):
            folder = self.dir / f"a{k}" / f"u{user:03d}"
            calls = []
            for f in folder.glob("*.json.gz") if folder.is_dir() else []:
                with gzip.open(f, "rt") as fp:
                    calls.append(json.load(fp))
            calls.sort(key=lambda c: c["started"])
            own = [c for c in calls if c["started"] >= attempt["at"]]
            until = self.attempts[k]["at"] if k < len(self.attempts) else time.time()
            # an attempt that replayed this game makes it anew only once it went past its recording or drifted from
            # it: a call of its own before then was the model's code calling on the side, and the recording stands
            went_on = any(attempt["at"] <= e["t"] < until and ("handed_off_at" in e or "drifted_at" in e)
                          for e in events)
            if not own or (str(user) in attempt["replaying"] and not went_on):
                continue
            turns = user_dir(self.run_dir, user) / "system" / "turns.jsonl"
            rows = [r for r in map(json.loads, open(turns)) if attempt["at"] <= r.get("t", 0.0) < until] \
                if turns.exists() else []
            ends = [r["t"] for r in rows]
            for c in own:
                if not _message(c.get("response")):
                    i = bisect.bisect_left(ends, c["started"] - 0.5)
                    cut = i < len(rows) and (rows[i].get("outcome") or {}).get("stop_reason") == "budget"
                    c["partial"] = (rows[i].get("reply") or "") if cut else ""
            best, ended = calls, (folder / "ended").exists()
        return best, ended

    def kept(self, call: Any) -> dict[str, Any]:
        """``call`` as the run keeps it: a replayed call as it was recorded, any other as the model made it. A replayed
        call is known by its own id (the proxy's ``replays``), not by its response: one held at a cut ends cancelled,
        with none, and kept as made it would join the next recording at this attempt's time, out of its place (moved
        after the calls that followed it, it shifts them, and the next resume drifts at the cut)."""
        replayed = self.proxy.replays.get(call.call_id)
        return self.proxy.recorded[replayed] if replayed else captured(call)

    def calls(self, trace: Any) -> list[dict[str, Any]]:
        """``trace``'s calls as the run keeps them (:meth:`kept`), in the order they began, as :meth:`recording` reads
        them: the learner keeps them in the order they ended, and a call cancelled at a cut ends after the next has
        begun, later in a replay than in the play it replays, so only in this order is a game replayed whole saved as
        it was played."""
        return sorted((self.kept(c) for c in trace.calls), key=lambda c: c["started"])

    def ended(self, trace: Any) -> None:
        """``trace``'s game ended in this attempt (once it is saved): its recording is a whole game, not one cut short,
        so a later replay that runs past it has drifted."""
        folder = self.dir / f"a{self.attempt}" / f"u{trace.user:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ended").touch()

    def journal(self, trace: Any, call: Any) -> None:
        """The learner's call hook: ``call``, as kept, written to this attempt's journal as it ends."""
        entry = self.kept(call)
        folder = self.dir / f"a{self.attempt}" / f"u{call.user:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{entry['started']:.6f}-{entry['call_id']}.json.gz"
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", compresslevel=1) as fp:
            json.dump(entry, fp, default=str)
        tmp.rename(path)

    def finish(self) -> dict[str, Any]:
        """The round is whole: the round folder's ``resume.json`` (every attempt's start, and how this one went, by
        user), then the journal and the snapshots deleted. Returns what ``resume.json`` holds."""
        out = {"round": self.round, "attempts": [*self.attempts, self.started],
               "counts": {u: dict(c) for u, c in sorted(self.proxy.counts.items())}}
        (round_dir(self.run_dir, self.round) / "resume.json").write_text(json.dumps(out, indent=1))
        shutil.rmtree(self.dir)
        for u in range(self.users):
            shutil.rmtree(user_dir(self.run_dir, u) / "round-start", ignore_errors=True)
        return out


# -- play --------------------------------------------------------------------------------------------------

def _wipe_stateless_users(learner: Any, schedule: Any) -> None:
    """Baselines' own wipe (``SystemLearner.reset``: stop if started, archive, ``wipe_state()``, a fresh
    ``Transcript``) for every user of ``schedule``'s collective regime, called once per round: a ``play`` call plays
    exactly one instance per user, so the harness's own in-process wipe (``SystemLearner.act``, gated on
    ``_session_pending``, which is set only at the *end* of the instance this very call plays) never fires across
    separate processes -- it is one call too late, every call. Calling ``reset()`` directly reproduces "wiped before
    every instance" at the only point this driver can: here, before the round's one instance per user. Raises if any
    user's own ``stats["resets"]`` (``reset()``'s own counter; the automatic in-process path above bumps
    ``stats["wipes"]`` instead, which never applies here) did not advance by exactly one, since a system that
    silently failed to wipe would otherwise look identical to one that did."""
    for u in range(schedule.users):
        system = learner._user(u).system
        before = system.stats["resets"]
        system.reset()
        after = system.stats["resets"]
        if after != before + 1:
            raise RuntimeError(f"user {u}'s system was not wiped exactly once before the round (resets {before} "
                               f"-> {after})")


def _refresh_meta_protocol(run_dir: Path, protocol: dict[str, Any]) -> None:
    """``continual_arc.runner.run`` reloads ``meta.json`` wholesale on a resumed call rather than recomputing its
    ``protocol`` block, so it freezes at whatever cut the run's first call saw; ``summary.json``'s own protocol is
    recomputed every call and is always right. Since this driver resumes every round, rewrite ``meta.json``'s
    ``protocol`` from the just-recomputed one (``RunResult.summary["protocol"]``) after every call, so a reader of
    ``meta.json`` alone is not misled."""
    path = run_dir / "meta.json"
    meta = json.loads(path.read_text())
    meta["protocol"] = protocol
    path.write_text(json.dumps(meta, indent=1))


def play_round(row: str, run_dir: Path, schedule: Any, *, base_url: str, served_name: str,
              registry: Any = None) -> Any:
    """Play ``schedule`` into ``run_dir`` under ``row``'s harness memory setting, serving ``served_name`` (the base
    model's own id, or the adapter name paper.sh already loaded into the server): ``continual_arc.runner.run``'s own
    ``resume=True`` restores every whole round already recorded and plays only the rest, so calling this with a cut
    schedule one round longer than last time plays exactly that new round, and calling it again under the same cut
    does nothing. With :data:`RESUME_CALLS`, a round an earlier call was cut short in is taken up from its last model
    call (:class:`Resume`). ``registry`` defaults to the real benchmark's (``TaskRegistry.default()``); tests pass their own.
    Returns the run's ``RunResult`` (``.records``, ``.summary``)."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.proxy import LLMProxy
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    from continual_arc_baselines.proxy import THINKING_FIELD, VLLM_DROP, VLLM_ROLES

    if row not in ROWS:
        raise ValueError(f"row must be one of {ROWS}, got {row!r}")
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = {"model": served_name, "chat_template_kwargs": dict(loop.TEMPLATE_KWARGS), **SAMPLING}
    if row == "bare":
        from continual_arc_baselines.systems.bare import BareModelLearner

        def system(user: int) -> Any:
            return BareModelLearner(loop.MODEL, max_tokens=MAX_TOKENS, **FLOOR)
        setting = {"floor": FLOOR, "max_tokens": MAX_TOKENS}
    else:
        # the fixed harness settings every harness row plays under: loop.HARNESS's own compaction point (49,152
        # tokens, keeping 8,000) and loop.HARNESS_BUDGETS["block"]'s call cap, never overridden per round
        budget = loop.HARNESS_BUDGETS["block"]
        kwargs = {"model": loop.MODEL, "context_window": CONTEXT_WINDOW, "max_tokens": MAX_TOKENS,
                  "stateless": row in STATELESS_ROWS, **loop.HARNESS, **budget}

        def system(user: int) -> Any:
            return PrimeAgentLearner(**kwargs)
        overrides["return_token_ids"] = True
        setting = {"harness": dict(loop.HARNESS), "harness_budget": {"name": "block", **budget}}
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": row, "served": served_name, "stateless": row in STATELESS_ROWS, "model": loop.MODEL, **setting,
         "sampling": SAMPLING}, indent=1))
    resume = None
    if RESUME_CALLS and not round_recorded(run_dir, schedule):
        resume = Resume(run_dir, schedule.rounds()[-1], schedule.users, row in STATELESS_ROWS)
    save = save_instance_hook(run_dir, resume.calls if resume else played)

    def ended(t: Any) -> None:
        save(t)
        if resume is not None:
            resume.ended(t)

    learner = CollectiveHarnessLearner(system, run_dir, schedule=schedule,
                                       upstream=base_url, body_overrides=overrides, body_drop=VLLM_DROP, role_map=VLLM_ROLES,
                                       thinking_field=THINKING_FIELD, declared_context_length=CONTEXT_WINDOW, capture=True,
                                       snapshot=False, call_hook=resume.journal if resume else None,
                                       instance_hook=ended, reset_hook=lambda: None,
                                       proxy=resume.proxy if resume else LLMProxy,
                                       ports=resume.ports if resume else None)
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        if row in STATELESS_ROWS:
            _wipe_stateless_users(learner, schedule)
        result = run_stream(learner, schedule, EnvConfig(), out_dir=run_dir, env=env, resume=True,
                            learner_name=f"paper-{row}")
        _refresh_meta_protocol(run_dir, result.summary["protocol"])
        if row != "bare":   # the bare model has no harness, so no memory to count
            _report_round_memory(learner, schedule, run_dir)
    finally:
        learner.close()
    if resume is not None:
        logging.info("round %s, attempt %s: %s", resume.round, resume.attempt, json.dumps(resume.finish()["counts"]))
    return result


def _report_round_memory(learner: Any, schedule: Any, run_dir: Path) -> None:
    """Every row's memory-use counters for the round just played (:mod:`memory`'s ``round_report``), the round
    being ``schedule.rounds()[-1]`` -- ``Schedule.rounds()`` gives every slot's own round, O(len(slots)), and
    ``[-1]`` is the cut's last slot's, which is correct here only because the cut's last slot belongs to the cut's
    last round (a resumed call plays exactly that one, see :func:`play_round`'s docstring). Row-agnostic -- H, H0,
    PG and OPSD alike. A wiped row's own wipe runs at the *next* call's start (:func:`_wipe_stateless_users`),
    not before this snapshot, so its snapshot still holds this round's own instance; what the wipe does guarantee
    is that no file survives *unchanged* into the next round (:mod:`memory`'s own docstring). A failure here (a
    user's harness state unreadable, say) is logged and never costs the round its own result."""
    import memory
    round_ = schedule.rounds()[-1]
    try:
        report = memory.round_report(run_dir, round_, schedule.users, lambda u: learner._user(u).system.state_dir,
                                      round_dir)
    except Exception:  # noqa: BLE001
        logging.exception("could not compute round %s's memory-use counters", round_)
        return
    logging.info(memory.summary_line(report))


# -- the served-adapter check -------------------------------------------------------------------------------

def adapter_is_untrained(adapter_dir: Path) -> bool:
    """Whether every ``lora_B`` matrix of the PEFT adapter in ``adapter_dir`` is exactly zero: PEFT starts B at zero,
    so an adapter no optimizer step has touched (a round whose data gave no item, e.g. PG's first, with no earlier game) adds
    nothing and is the base model's own weights, which the served-adapter check cannot tell from a base model served
    under the adapter's name."""
    from safetensors import safe_open
    with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), "pt") as st:
        return all(not st.get_tensor(k).any() for k in st.keys() if "lora_B" in k)


def check_served_adapter(server: Any, base_model: str, served_name: str, prompt: list[dict[str, str]],
                         max_tokens: int = 16) -> dict[str, Any]:
    """``prompt`` (a fixed chat prompt) sent at temperature 0 to ``base_model`` and to ``served_name``: ``differs`` is
    false exactly when vLLM answered both from the same weights, which is how it silently serves the base model when
    an adapter's layer names are wrong (``loop.served_adapter``) instead of refusing to load it. Reads each reply's
    tokens and their top :data:`SERVED_TOP_LOGPROBS` candidates' log-probabilities, never the reply's text (see
    :data:`SERVED_LOGPROB_GAP`'s note): different tokens differ outright; the same tokens differ when a candidate
    both replies list moved by at least :data:`SERVED_LOGPROB_GAP` (``logprob_gap_max``, the largest such move).
    ``answered`` is false when either reply came back with no tokens (a failed request), which says nothing about
    the weights. ``server`` is an ``ask_sd.server.Server`` (or, in tests, anything with its ``chat`` method)."""

    def reply(model: str) -> tuple[list[str], dict[tuple[int, str], float]]:
        out = server.chat({"model": model, "messages": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                           "logprobs": True, "top_logprobs": SERVED_TOP_LOGPROBS})
        choice = ((out or {}).get("choices") or [{}])[0]
        content = ((choice.get("logprobs") or {}).get("content")) or []
        tops = {(i, c["token"]): float(c["logprob"]) for i, t in enumerate(content) for c in t.get("top_logprobs") or []}
        return [t["token"] for t in content], tops

    (base_tokens, base_tops), (served_tokens, served_tops) = reply(base_model), reply(served_name)
    out: dict[str, Any] = {"base": "".join(base_tokens), "served": "".join(served_tokens),
                           "answered": bool(base_tokens) and bool(served_tokens)}
    if not out["answered"]:
        return {**out, "differs": False}
    if base_tokens != served_tokens:
        return {**out, "differs": True}
    gaps = [abs(base_tops[k] - served_tops[k]) for k in base_tops.keys() & served_tops.keys()]
    out["logprob_gap_max"] = max(gaps) if gaps else None
    return {**out, "differs": bool(gaps) and max(gaps) >= SERVED_LOGPROB_GAP}


# -- CLI -----------------------------------------------------------------------------------------------------

def cmd_play(a: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(a.run_dir).expanduser()
    if a.width is not None:
        if a.row not in TRANSFER_REFERENCES or a.epilogue_users is None:
            raise SystemExit(f"--width deals the epilogue alone's instances wider: only for --epilogue-users and a row "
                             f"that keeps nothing between instances, {TRANSFER_REFERENCES}, not {a.row}")
        schedule = wide_epilogue_cut(a.round, a.width, a.epilogue_users, a.seed, a.users)
    elif a.epilogue_users is not None:
        schedule = epilogue_cut(a.round, a.epilogue_users, a.seed, a.users)
    else:
        schedule = schedule_cut(a.round, a.seed, a.users)
    result = play_round(a.row, run_dir, schedule, base_url=a.base_url, served_name=a.served)
    print(json.dumps({"out": str(run_dir), "round": a.round, "records": len(result.records),
                      "summary": result.summary}, default=str))


def cmd_check_served(a: argparse.Namespace) -> None:
    if a.adapter and adapter_is_untrained(Path(a.adapter).expanduser()):
        print(json.dumps({"skipped": f"{a.adapter} has taken no optimizer step (every lora_B is zero): it is the "
                                     "base model's own weights, so there is nothing to tell apart"}))
        return
    from ask_sd.server import Server
    out = check_served_adapter(Server(a.vllm_url), a.base_model, a.served, SERVED_ADAPTER_PROMPT,
                               max_tokens=a.max_tokens)
    print(json.dumps(out))
    if not out["answered"]:
        raise SystemExit(f"no tokens came back from {a.base_model!r} or {a.served!r} at {a.vllm_url}: the check could "
                         "not compare them (is the server up, and is the adapter loaded under that name?)")
    if not out["differs"]:
        raise SystemExit(f"{a.served!r} answered exactly as the base model {a.base_model!r}: vLLM may be serving the "
                         "base weights under the adapter's name (check the adapter's layer names, loop.served_adapter)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    play = sub.add_parser("play",
                          help="play one round of a row's schedule (or of the K-user epilogue alone), resumable")
    play.add_argument("--row", required=True, choices=ROWS)
    play.add_argument("--run-dir", required=True, dest="run_dir")
    play.add_argument("--base-url", required=True, dest="base_url", help="the served model's OpenAI-compatible base URL")
    play.add_argument("--served", required=True,
                      help="the vLLM-served model name for this round: the base model's own id at round 0, or the "
                           "adapter name serve.sh already loaded the previous round's adapter under")
    play.add_argument("--round", type=int, required=True, help="0-indexed: the round this call plays")
    play.add_argument("--epilogue-users", type=int, default=None, dest="epilogue_users",
                      help="play the K-user epilogue alone (continual_arc.configs.epilogue_for_users) instead of the "
                           "whole schedule: transfer's references")
    play.add_argument("--width", type=int, default=None,
                      help="with --epilogue-users, for h0 and bare: the epilogue alone's instances dealt this many to a "
                           "round (wide_epilogue_cut)")
    play.add_argument("--seed", type=int, default=SEED)
    play.add_argument("--users", type=int, default=PAPER_USERS,
                      help="how many users the schedule is served to (served): users 0..N-1 are any larger number's "
                           "own first N")
    play.set_defaults(fn=cmd_play)

    check = sub.add_parser("check-served",
                           help="the served-adapter check: the fixed prompt's reply through --served differs from "
                                "the base model's")
    check.add_argument("--vllm-url", required=True, dest="vllm_url")
    check.add_argument("--base-model", default=loop.MODEL, dest="base_model")
    check.add_argument("--served", required=True)
    check.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    check.add_argument("--adapter", default=None,
                       help="the served adapter's own directory: an adapter no step has touched is skipped "
                            "(adapter_is_untrained), since it is the base model's weights by construction")
    check.set_defaults(fn=cmd_check_served)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
