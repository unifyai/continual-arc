"""The collective regime with a harness per user: concurrency, capture, the round hook, overrides, every user's state
committed at the end of every round."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import pytest
from continual_arc.aliases import TaskAliases
from continual_arc.runner import run
from continual_arc.schedule import interleave_users, stream_schedule
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

from continual_arc_baselines.collective import (
    CollectiveHarnessLearner,
    restore_state,
    snapshot_state,
)
from continual_arc_baselines.statelog import StateLog

from .fake_harness import FakeHarness, StoringHarness, late_reports
from .fake_upstream import FakeUpstream


def test_every_user_runs_a_harness_of_their_own_and_the_hook_sees_the_round(tmp_path):
    source = FakeSource(n_tasks=2)
    registry = TaskRegistry({source.name: source})
    ids = source.task_ids()
    users = 3
    sched = interleave_users([stream_schedule(ids, 6, seed=u, source=source.name) for u in range(users)],
                             seed=0, name="team")
    server = FakeUpstream().start()
    rounds: list = []
    events: list = []
    events_lock = threading.Lock()

    def hook(data):
        # the calls are the hook's to use during the round only (the learner drops the bodies after it)
        rounds.append((data, {t.slot: list(t.calls) for t in data.instances}))
        events.append(("round", data.index))
        return {"model": f"adapter-{data.index + 1}"}

    def ended(t):
        # an instance, as soon as it has ended: its storage step done (the rule is in its user's memory)
        memory = tmp_path / "run" / "users" / f"u{t.user:03d}" / "system" / "state" / "memory.json"
        with events_lock:
            events.append(("ended", t.slot, t.ended and t.solved and t.seconds > 0 and bool(t.calls)
                           and t.task_label in json.loads(memory.read_text())))

    learner = CollectiveHarnessLearner(FakeHarness, tmp_path / "run", schedule=sched, upstream=server.base_url,
                                       capture=True, snapshot=True, body_overrides={"model": "base", "return_token_ids": True},
                                       body_drop=("reasoning_effort",), round_hook=hook,
                                       begin_hook=lambda n: events.append(("begin", n)), instance_hook=ended)
    try:
        result = run(learner, sched, out_dir=tmp_path / "run", registry=registry)
    finally:
        learner.close()
        server.stop()
    recs = result.records
    assert len(recs) == 18 and [r.user for r in recs] == [0, 1, 2] * 6
    assert FakeHarness.max_inside > 1  # users worked at the same time
    # a user's own memory: the first meeting with a task costs a demonstration, later ones nothing
    for r in recs:
        assert r.cost == (1.0 if r.user_exposure_index == 0 else 0.0), r
    # the hook saw every round, every user's instance with its calls, tagged and captured whole
    assert [d.index for d, _ in rounds] == list(range(6))
    for d, calls in rounds:
        assert [t.user for t in d.instances] == [0, 1, 2] and all(t.ended and t.solved for t in d.instances)
        for t in d.instances:
            mine = calls[t.slot]
            assert mine and all(c.slot == t.slot and c.phase == "turn" for c in mine)
            assert [c.turn for c in mine] == list(range(len(t.turns)))
            assert mine[0].response["choices"][0]["token_ids"] == [7, 8]
            assert t.accepted_grid is not None and t.turns[-1]["feedback"]["correct"] is True
    # one name per task for the whole team, the runner's; the traces keep each slot's real id, phase and visits
    aliases = TaskAliases.for_run(tmp_path / "run")
    traces = [t for d, _ in rounds for t in d.instances]
    assert all(t.task_label == aliases.alias(t.task_id) and t.task_id in ids for t in traces)
    by_slot = {r.step: r for r in recs}
    assert all((t.task_id, t.phase, t.user_exposure_index) == (by_slot[t.slot].task_id, by_slot[t.slot].phase,
                                                                by_slot[t.slot].user_exposure_index) for t in traces)
    # the model each round was served: what the previous round's hook returned; the refused field never left
    sent = [r["body"] for r in server.requests if r["body"]]
    assert {b["model"] for b in sent} == {"base"} | {f"adapter-{k}" for k in range(1, 6)}
    assert all("reasoning_effort" not in b and b["return_token_ids"] for b in sent)
    # a snapshot of each user's state when the round began: the memory written in earlier rounds
    last = rounds[-1][0].instances[0]
    assert last.state_before is not None and json.loads((last.state_before / "memory.json").read_text())
    # each instance is charged its own user's calls
    assert all(r.resources["llm_requests"] == r.resources["harness_calls"] >= 1 for r in recs)
    assert sum(r.resources["llm_requests"] for r in recs) == len(sent)
    log = (tmp_path / "run" / "users" / "u000" / "calls.jsonl").read_text().splitlines()
    assert json.loads(log[0])["reply_token_ids"] == 2 and "request" not in json.loads(log[0])
    # every round: the begin hook before any instance, each instance once as it ended, then the round hook
    for r in range(6):
        begin, *mine, closing = events[5 * r: 5 * (r + 1)]
        assert begin == ("begin", 3) and closing == ("round", r)
        assert sorted(e[1] for e in mine) == [3 * r, 3 * r + 1, 3 * r + 2] and all(e[0] == "ended" and e[2] for e in mine)
    assert len(events) == 30


class CommittingHarness(FakeHarness):
    """A FakeHarness whose state is committed at the end of every round, as a harness adapter's is
    (``SystemLearner.end_round``), which notes every instance it solves and keeps, as a harness's Python environment
    does, a script that names its own folder."""

    committing = 0
    most_committing = 0

    def attach(self, workdir: Path, environ: dict[str, str], tracker=None) -> None:
        super().attach(workdir, environ, tracker)
        tool = self.workdir / "state" / "bin" / "tool"
        if not tool.exists():
            tool.parent.mkdir(parents=True)
            tool.write_text(f"#!{self.workdir}/state/venv/bin/python\n")
        self.state_log = StateLog(self.workdir / "state", self.workdir / "state.git", self.workdir / "state-log.jsonl")

    def observe(self, obs, action, feedback) -> None:
        super().observe(obs, action, feedback)
        if getattr(feedback, "correct", False):
            with (self.workdir / "state" / "notes.md").open("a") as fp:
                fp.write(f"user {self.user} solved {obs.task_id} in round {obs.round}\n")

    def end_round(self) -> None:
        with FakeHarness.lock:
            CommittingHarness.committing += 1
            CommittingHarness.most_committing = max(CommittingHarness.most_committing, CommittingHarness.committing)
        try:
            time.sleep(0.02)   # room for another user's commit to overlap this one, were they not one at a time
            self.state_log.commit()
        finally:
            with FakeHarness.lock:
                CommittingHarness.committing -= 1


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", f"--git-dir={repo}", *args], check=True, capture_output=True, text=True).stdout


def run_team(tmp_path, users: int, per_user: int, **learner_args) -> tuple[dict, dict, dict]:
    """A team of StoringHarness users, ``per_user`` instances each. Returns every instance's calls, as (id, phase)
    by slot: as the instance hook got them, as the round hook got them, and as the users' call logs have them once
    every call's end has been reported."""
    source = FakeSource(n_tasks=2)
    sched = interleave_users([stream_schedule(source.task_ids(), per_user, seed=u, source=source.name)
                              for u in range(users)], seed=0, name="team")
    server = FakeUpstream().start()
    ended: dict = {}
    closed: dict = {}

    def calls(t):
        return [(c.call_id, c.phase) for c in t.calls]

    learner = CollectiveHarnessLearner(StoringHarness, tmp_path / "run", schedule=sched, upstream=server.base_url,
                                       capture=True,
                                       instance_hook=lambda t: ended.update({t.slot: calls(t)}),
                                       round_hook=lambda d: closed.update({t.slot: calls(t) for t in d.instances}),
                                       **learner_args)
    logs = [tmp_path / "run" / "users" / f"u{u:03d}" / "calls.jsonl" for u in range(users)]
    try:
        run(learner, sched, out_dir=tmp_path / "run", registry=TaskRegistry({source.name: source}))
        deadline = time.monotonic() + 10
        while sum(p.read_text().count("\n") for p in logs) < len(server.requests) and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        learner.close()
        server.stop()
    logged: dict = {}
    for p in logs:
        for line in p.read_text().splitlines():
            c = json.loads(line)
            logged.setdefault(c["slot"], []).append((c["call_id"], c["phase"]))
    return ended, closed, logged


def test_the_hooks_get_every_call_of_an_instance_though_the_proxy_reports_it_after_the_reply(tmp_path, monkeypatch):
    # the proxy reports a call's end once it has written the reply, so the harness can act on the reply first
    # (CI, 25 Sep: the round hook got an instance without its last call); here every report comes 0.2 s late
    late_reports(monkeypatch, 0.2)
    ended, closed, logged = run_team(tmp_path, users=2, per_user=2)
    assert len(logged) == 4 and all(any(phase == "after" for _, phase in calls) for calls in logged.values())
    assert ended == closed == logged


def test_a_call_still_open_after_the_wait_is_left_out_for_both_hooks_and_logged(tmp_path, monkeypatch, caplog):
    # every storage step's call ends 1.5 s after its reply, and the learner waits 0.5 s for it
    late_reports(monkeypatch, 1.5, lambda call: "storage" in json.dumps(call.request_body))
    with caplog.at_level(logging.WARNING, logger="continual_arc_baselines.collective"):
        ended, closed, logged = run_team(tmp_path, users=2, per_user=1, calls_wait_s=0.5)
    assert len(logged) == 2 and all([phase for _, phase in calls].count("after") == 1 for calls in logged.values())
    assert ended == closed == {slot: [c for c in calls if c[1] != "after"] for slot, calls in logged.items()}
    mine = [r for r in caplog.records if r.name == "continual_arc_baselines.collective"]
    assert sorted(r.levelname for r in mine) == ["ERROR", "ERROR", "WARNING", "WARNING"]
    assert all("1 model call(s) started for the instance still open" in r.getMessage()
               for r in mine if r.levelname == "ERROR")
    assert all("too late for its instance" in r.getMessage() for r in mine if r.levelname == "WARNING")


def test_a_resumed_run_snapshots_every_users_state_when_its_first_round_begins(tmp_path):
    """A relaunched driver starts a user's harness only at their first action, so the snapshot of every user's state when
    its first round begins (what the plan writer reads as the memory before) is taken from disk too: the state as the
    last launch left it, not the snapshot that launch took when an earlier round began."""
    source = FakeSource(n_tasks=2)
    registry = TaskRegistry({source.name: source})
    sched = interleave_users([stream_schedule(source.task_ids(), 4, seed=u, source=source.name) for u in range(2)],
                             seed=0, name="team")
    run_dir = tmp_path / "run"
    server = FakeUpstream().start()

    def dies(data):
        if data.index == 1:
            raise RuntimeError("the driver dies in round 1's update")

    first = CollectiveHarnessLearner(FakeHarness, run_dir, schedule=sched, upstream=server.base_url, snapshot=True,
                                     round_hook=dies)
    try:
        with pytest.raises(RuntimeError, match="round 1's update"):
            run(first, sched, out_dir=run_dir, registry=registry, resume=True)
    finally:
        first.close()
    for u in range(2):   # what each harness kept after round 1 began: the snapshot taken then lacks it
        (run_dir / "users" / f"u{u:03d}" / "system" / "state" / "kept-in-round-1.md").write_text("a note")
    before: list = []

    def began(n):
        before.append(sorted(sorted(p.name for p in (run_dir / "users" / f"u{u:03d}" / "state-before").iterdir())
                             for u in range(2)))

    second = CollectiveHarnessLearner(FakeHarness, run_dir, schedule=sched, upstream=server.base_url, snapshot=True,
                                      begin_hook=began)
    try:
        result = run(second, sched, out_dir=run_dir, registry=registry, resume=True)
    finally:
        second.close()
        server.stop()
    assert len(result.records) == 8 and len(before) == 2   # rounds 2 and 3, played by the second launch
    assert all("kept-in-round-1.md" in names and "memory.json" in names for names in before[0])


def test_every_users_state_is_committed_at_the_end_of_every_round_one_user_at_a_time(tmp_path):
    """Three users over four rounds: each user's harness state is committed to their own history as every round ends,
    one user at a time and before the round hook, so a user's log has a commit per round and each commit's diff is
    what that user wrote in its round. An update that fails once the round is recorded (round 1's here) leaves the
    round's commits, and the relaunch numbers its rounds on from them."""
    source = FakeSource(n_tasks=2)
    registry = TaskRegistry({source.name: source})
    users, rounds = 3, 4
    sched = interleave_users([stream_schedule(source.task_ids(), rounds, seed=u, source=source.name)
                              for u in range(users)], seed=0, name="team")
    run_dir = tmp_path / "run"
    systems = [run_dir / "users" / f"u{u:03d}" / "system" for u in range(users)]
    logs = [s / "state-log.jsonl" for s in systems]
    hooked: list = []

    def update(data):
        hooked.append((data.index, [log.read_text().count("\n") if log.exists() else 0 for log in logs]))
        if data.index == 1 and len(hooked) == 2:
            raise RuntimeError("round 1's update fails")

    CommittingHarness.most_committing = 0
    server = FakeUpstream().start()
    first = CollectiveHarnessLearner(CommittingHarness, run_dir, schedule=sched, upstream=server.base_url,
                                     round_hook=update)
    try:
        with pytest.raises(RuntimeError, match="round 1's update fails"):
            run(first, sched, out_dir=run_dir, registry=registry, resume=True)
    finally:
        first.close()
    second = CollectiveHarnessLearner(CommittingHarness, run_dir, schedule=sched, upstream=server.base_url,
                                      round_hook=update)
    try:
        result = run(second, sched, out_dir=run_dir, registry=registry, resume=True)
    finally:
        second.close()
        server.stop()
    assert len(result.records) == users * rounds
    # every round's commits were in when its hook ran, the failed update's round's too, and none overlapped another
    assert hooked == [(r, [r + 1] * users) for r in range(rounds)]
    assert CommittingHarness.most_committing == 1
    aliases = TaskAliases.for_run(run_dir)
    played = {(r.round, r.user): r for r in result.records}
    for user, system in enumerate(systems):
        repo = system / "state.git"
        log = [json.loads(line) for line in logs[user].read_text().splitlines()]
        assert [entry["round"] for entry in log] == list(range(rounds))
        assert git(repo, "log", "--format=%s").splitlines() == [f"round {r}" for r in reversed(range(rounds))]
        assert git(repo, "rev-parse", "HEAD").strip() == log[-1]["commit"]
        for r, entry in enumerate(log):
            mine = played[(r, user)]
            # the round's diff: its note, the rule of a task met for the first time, the harness's script in round 0
            wrote = {"notes.md"} | ({"memory.json"} if mine.user_exposure_index == 0 else set()) | (
                {"bin/tool"} if r == 0 else set())
            assert set(git(repo, "show", "--name-only", "--format=", entry["commit"]).split()) == wrote
        last = played[(rounds - 1, user)]
        diff = git(repo, "diff", "HEAD~1", "HEAD")
        assert f"+user {user} solved {aliases.alias(last.task_id)} in round {rounds - 1}\n" in diff


def test_snapshot_leaves_out_runtimes_and_large_files(tmp_path):
    src = tmp_path / "state"
    (src / "memories").mkdir(parents=True)
    (src / "memories" / "MEMORY.md").write_text("rule for task-1: shift by 3")
    (src / "agent" / "kernel-venv" / "lib").mkdir(parents=True)
    (src / "agent" / "kernel-venv" / "lib" / "big.so").write_bytes(b"x" * 10)
    (src / "sessions").mkdir()
    (src / "sessions" / "huge.jsonl").write_bytes(b"x" * 3_000_000)
    stats = snapshot_state(src, tmp_path / "copy")
    assert (tmp_path / "copy" / "memories" / "MEMORY.md").read_text().startswith("rule")
    assert not (tmp_path / "copy" / "agent" / "kernel-venv").exists()
    assert not (tmp_path / "copy" / "sessions" / "huge.jsonl").exists()
    assert stats["files"] == 1 and stats["skipped"] == 1


def test_the_call_hook_gets_each_kept_call_before_its_instance_settles(tmp_path, monkeypatch):
    """A driver that keeps every call on disk as it is made (to resume a round from its last call) gets each call the
    instance keeps, with the instance's trace, before the instance hook can run;
    every report comes 0.2 s late, and the hook takes a while, and still the instance waits for it."""
    late_reports(monkeypatch, 0.2)
    hooked: dict = {}
    lock = threading.Lock()

    def journal(t, call):
        time.sleep(0.1)
        with lock:
            hooked.setdefault(t.slot, []).append((call.call_id, call.phase, t.round, call.request is not None))

    ended, closed, logged = run_team(tmp_path, users=2, per_user=2, call_hook=journal)
    assert ended == closed == logged
    # the hooks of one instance's calls run in the proxy's threads at once, so they may finish in any order
    assert {slot: sorted((cid, phase) for cid, phase, _, _ in calls) for slot, calls in hooked.items()} == {
        slot: sorted(calls) for slot, calls in ended.items()}
    assert all(captured for calls in hooked.values() for *_, captured in calls)   # the bodies, when captured


def test_each_users_proxy_is_the_learners_proxy_class(tmp_path):
    """A driver that serves some calls itself (a replay, a thought pipeline) gives the learner its proxy class, which
    every user's proxy is made from."""
    from continual_arc_baselines.proxy import LLMProxy

    made: list = []

    class Counting(LLMProxy):
        def __init__(self, *args, **kwargs):
            made.append(kwargs["route_key"])
            super().__init__(*args, **kwargs)

    run_team(tmp_path, users=2, per_user=1, proxy=Counting)
    assert sorted(made) == ["u0", "u1"]


def test_a_complete_snapshot_restores_the_state_as_it_was(tmp_path):
    """A round cut short is played again from the state its users had when it began: the snapshot taken then, with no
    size limit, is put back, so what the cut attempt wrote is gone and what it changed is as it was, while the
    harness's runtimes (its Python environment) stay as they are and the dead process's locks go."""
    state = tmp_path / "state"
    (state / "memories").mkdir(parents=True)
    (state / "memories" / "MEMORY.md").write_text("rule for task-1: shift by 3")
    (state / "sessions").mkdir()
    (state / "sessions" / "big.jsonl").write_bytes(b"x" * 3_000_000)
    (state / "agent" / "kernel-venv" / "lib").mkdir(parents=True)
    (state / "agent" / "kernel-venv" / "lib" / "big.so").write_bytes(b"y" * 10)
    stats = snapshot_state(state, tmp_path / "round-start", max_file_bytes=None)
    assert stats["files"] == 2 and (tmp_path / "round-start" / "sessions" / "big.jsonl").stat().st_size == 3_000_000
    # the cut attempt: a note added, the memory rewritten, a session removed, a package installed, a lock left
    (state / "memories" / "MEMORY.md").write_text("half-written")
    (state / "memories" / "new.md").write_text("from the cut attempt")
    (state / "notes").mkdir()
    (state / "notes" / "a.md").write_text("from the cut attempt")
    (state / "sessions" / "big.jsonl").unlink()
    (state / "agent" / "kernel-venv" / "lib" / "added.so").write_bytes(b"z")
    (state / "agent" / "session.lock").write_text("pid 1")
    restore_state(tmp_path / "round-start", state)
    files = sorted(str(p.relative_to(state)) for p in state.rglob("*") if p.is_file())
    assert files == ["agent/kernel-venv/lib/added.so", "agent/kernel-venv/lib/big.so", "memories/MEMORY.md",
                     "sessions/big.jsonl"]
    assert (state / "memories" / "MEMORY.md").read_text().startswith("rule") and not (state / "notes").exists()


def test_a_snapshot_cut_short_leaves_the_one_before(tmp_path, monkeypatch):
    """A snapshot is made beside the old one and replaces it only once whole: a process killed while it copies leaves
    the snapshot that was there, which a resumed round restores, and the next snapshot starts afresh."""
    import shutil

    state = tmp_path / "state"
    state.mkdir()
    (state / "a.md").write_text("round 3")
    snapshot_state(state, tmp_path / "round-start", max_file_bytes=None)
    (state / "a.md").write_text("round 4")
    (state / "b.md").write_text("round 4")
    copy = shutil.copy2

    def killed(src, dst, **kwargs):
        if Path(src).name == "b.md":
            raise KeyboardInterrupt("the node stops")
        return copy(src, dst, **kwargs)

    monkeypatch.setattr(shutil, "copy2", killed)
    with pytest.raises(KeyboardInterrupt):
        snapshot_state(state, tmp_path / "round-start", max_file_bytes=None)
    assert [p.name for p in (tmp_path / "round-start").iterdir()] == ["a.md"]
    assert (tmp_path / "round-start" / "a.md").read_text() == "round 3"
    monkeypatch.setattr(shutil, "copy2", copy)
    snapshot_state(state, tmp_path / "round-start", max_file_bytes=None)
    assert sorted(p.name for p in (tmp_path / "round-start").iterdir()) == ["a.md", "b.md"]
    assert not (tmp_path / "round-start.partial").exists()


def test_a_user_is_given_the_port_asked_for_and_any_other_when_it_is_taken(tmp_path):
    """A replay serves recorded replies whose code can name the proxy's address, so a driver gives each user the port
    they had; a port another process holds since is no reason to fail the round."""
    import socket

    from continual_arc_baselines.proxy import LLMProxy

    def free() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    wanted, taken = free(), socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    held = taken.getsockname()[1]
    seen: dict = {}

    class Port(LLMProxy):
        def start(self):
            out = super().start()
            seen[self.route_key] = self.port
            return out

    try:
        run_team(tmp_path, users=2, per_user=1, proxy=Port, ports={0: wanted, 1: held})
    finally:
        taken.close()
    assert seen["u0"] == wanted and seen["u1"] not in (0, held)


def test_a_copy_to_read_is_as_it_was_and_a_copy_to_restore_from_has_every_directory(tmp_path):
    """A snapshot to read (the begin-round snapshot, the team rows' memory after) keeps a directory only for the files
    in it and leaves nothing for a state that does not exist; a copy to restore from keeps every directory, an empty
    one the model made among them, and restoring from no copy leaves the state empty but for its runtimes."""
    state = tmp_path / "state"
    (state / "programs").mkdir(parents=True)                  # made by the model, still empty
    (state / "notes").mkdir()
    (state / "notes" / "a.md").write_text("a")
    snapshot_state(state, tmp_path / "read")
    snapshot_state(state, tmp_path / "whole", max_file_bytes=None)
    assert sorted(p.name for p in (tmp_path / "read").iterdir()) == ["notes"]
    assert sorted(p.name for p in (tmp_path / "whole").iterdir()) == ["notes", "programs"]
    (state / "agent" / "kernel-venv").mkdir(parents=True)
    (state / "agent" / "kernel-venv" / "lib.so").write_bytes(b"x")
    snapshot_state(tmp_path / "missing", tmp_path / "none", max_file_bytes=None)
    assert not (tmp_path / "none").exists()
    restore_state(tmp_path / "none", state)
    assert sorted(str(p.relative_to(state)) for p in state.rglob("*")) == ["agent", "agent/kernel-venv",
                                                                          "agent/kernel-venv/lib.so"]
