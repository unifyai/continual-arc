"""One state for every user of a collective run (shared.py): every user's instance of a round starts from a copy of the
one state named as their own folder, and what every user wrote is merged into it in user order as the round ends, each
rule of the merge on its own, a SQLite database row by row; then the same through CollectiveHarnessLearner, whose run
of one user stays as it was."""

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest
from continual_arc.runner import run
from continual_arc.schedule import Schedule, Slot, interleave_users, stream_schedule
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

from continual_arc_baselines.collective import CollectiveHarnessLearner
from continual_arc_baselines.shared import (
    JOURNAL,
    MERGES,
    Entry,
    SharedState,
    merge_database,
    merge_entries,
    merge_json,
)

from .fake_upstream import FakeUpstream
from .test_collective import CommittingHarness


def team(tmp_path: Path, users: int) -> tuple[SharedState, list[Path]]:
    systems = [tmp_path / "run" / "users" / f"u{user:03d}" / "system" for user in range(users)]
    for system in systems:
        system.mkdir(parents=True)
    return SharedState(lambda user: systems[user], users), systems


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", f"--git-dir={repo}", *args], check=True, capture_output=True, text=True).stdout


def query(path: Path, sql: str) -> list:
    """The rows ``sql`` reads from the database at ``path``, closed again so that it leaves no side files."""
    db = sqlite3.connect(path)
    try:
        return db.execute(sql).fetchall()
    finally:
        db.close()


def sessions_db(path: Path, cwd: str) -> None:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT)")
    db.execute("INSERT INTO sessions VALUES ('s0', ?)", (cwd,))
    db.commit()
    db.close()


def test_every_user_starts_the_round_from_a_copy_of_the_one_state_named_as_their_own(tmp_path):
    shared, systems = team(tmp_path, 3)
    one = systems[0] / "state"
    (one / "bin").mkdir(parents=True)
    (one / "bin" / "tool").write_text(f"#!{systems[0]}/kernel-venv/bin/python\n")
    (one / "notes.md").write_text("task-1: shift by 3\n")
    (one / "here").symlink_to(f"{systems[0]}/state/bin")
    sessions_db(one / "agent.sqlite", f"{systems[0]}/state/workspace")
    shared.begin_round()
    for system in systems:
        mine = system / "state"
        assert (mine / "bin" / "tool").read_text() == f"#!{system}/kernel-venv/bin/python\n"
        assert (mine / "notes.md").read_text() == "task-1: shift by 3\n"
        assert os.readlink(mine / "here") == f"{system}/state/bin"
        assert query(mine / "agent.sqlite", "SELECT cwd FROM sessions") == [(f"{system}/state/workspace",)]
    # the base is the one state as the round began
    assert set(shared.base) == {"bin/tool", "notes.md", "here", "agent.sqlite"}


def test_what_every_user_writes_is_merged_into_the_one_state_and_every_user_starts_the_next_round_from_it(tmp_path):
    shared, systems = team(tmp_path, 3)
    one = systems[0] / "state"
    one.mkdir()
    (one / "log.txt").write_text("start\n")
    (one / "memory.json").write_text(json.dumps({"task-1": 3}))
    (one / "stale.md").write_text("a note nobody needs\n")
    sessions_db(one / "agent.sqlite", f"{systems[0]}/state/workspace")
    shared.begin_round()
    for user, system in enumerate(systems):
        mine = system / "state"
        (mine / f"own-{user}.md").write_text(f"named for {system}\n")              # a file of its own
        with (mine / "log.txt").open("a") as fp:                                   # an append-only log
            fp.write(f"user {user} was here\n")
        memory = json.loads((mine / "memory.json").read_text())                    # a JSON store
        memory[f"task-{user + 2}"] = user
        (mine / "memory.json").write_text(json.dumps(memory))
        db = sqlite3.connect(mine / "agent.sqlite")                                # a database
        db.execute("INSERT INTO sessions VALUES (?, ?)", (f"s{user + 1}", f"{system}/state/workspace"))
        db.commit()
        db.close()
    (systems[2] / "state" / "stale.md").unlink()                                   # a removal
    record = shared.end_round(0)
    assert sorted(p.name for p in one.iterdir()) == ["agent.sqlite", "log.txt", "memory.json", "own-0.md", "own-1.md",
                                                     "own-2.md"]
    assert [(one / f"own-{user}.md").read_text() for user in range(3)] == [f"named for {systems[0]}\n"] * 3
    assert (one / "log.txt").read_text() == "start\nuser 0 was here\nuser 1 was here\nuser 2 was here\n"
    assert json.loads((one / "memory.json").read_text()) == {"task-1": 3, "task-2": 0, "task-3": 1, "task-4": 2}
    assert query(one / "agent.sqlite", "SELECT id, cwd FROM sessions ORDER BY id") == [(f"s{k}", f"{systems[0]}/state/workspace") for k in range(4)]
    # the round's record: what each user wrote against the base, and how each path several users changed was merged
    assert record["users"][2] == {"user": 2, "added": ["own-2.md"], "removed": ["stale.md"],
                                  "changed": ["agent.sqlite", "log.txt", "memory.json"]}
    assert [(c["path"], c["users"], c["rule"]) for c in record["conflicts"]] == [
        ("agent.sqlite", [0, 1], "merged row by row"), ("log.txt", [0, 1], "merged line by line"),
        ("memory.json", [0, 1], "merged as JSON"), ("agent.sqlite", [0, 1, 2], "merged row by row"),
        ("log.txt", [0, 1, 2], "merged line by line"), ("memory.json", [0, 1, 2], "merged as JSON")]
    assert json.loads((systems[0] / MERGES).read_text()) == record
    shared.close_round()
    assert not (systems[0] / JOURNAL).exists()
    # the next round: every user's copy is the merged state, named as their own folder
    shared.begin_round()
    for system in systems:
        mine = system / "state"
        assert (mine / "log.txt").read_text() == (one / "log.txt").read_text()
        assert [(mine / f"own-{user}.md").read_text() for user in range(3)] == [f"named for {system}\n"] * 3
        assert not (mine / "stale.md").exists()
    assert set(shared.base) == {"agent.sqlite", "log.txt", "memory.json", "own-0.md", "own-1.md", "own-2.md"}


def test_a_path_several_users_changed_keeps_what_each_wrote_where_it_can_and_the_later_users_where_it_cannot(tmp_path):
    shared, systems = team(tmp_path, 2)
    one, outside = systems[0] / "state", tmp_path / "outside"
    one.mkdir()
    outside.mkdir()
    (one / "rule.md").write_text("first\n")
    (one / "blob.bin").write_bytes(b"\0base")
    (one / "kept.md").write_text("as it was\n")
    (one / "folder").mkdir()
    shared.begin_round()
    zero, first = one, systems[1] / "state"
    (zero / "rule.md").write_text("first, by user 0\n")                   # both changed the same line
    (first / "rule.md").write_text("first, by user 1\n")
    (zero / "blob.bin").write_bytes(b"\0zero")                            # a binary file both changed
    (first / "blob.bin").write_bytes(b"\0one")
    (zero / "kept.md").write_text("changed by user 0\n")                  # one removed what the other changed
    (first / "kept.md").unlink()
    (zero / "way").symlink_to(outside)                                    # a link on the way to another's file
    (first / "way").mkdir()
    (first / "way" / "escaped.md").write_text("never outside the state\n")
    (zero / "folder" / "inside.md").write_text("a folder's file\n")       # a file where another made a folder
    (first / "folder").rmdir()
    (first / "folder").write_text("a file\n")
    record = shared.end_round(0)
    assert (zero / "rule.md").read_text() == "first, by user 0\nfirst, by user 1\n"
    assert (zero / "blob.bin").read_bytes() == b"\0one"
    assert (zero / "kept.md").read_text() == "changed by user 0\n"
    assert os.readlink(zero / "way") == str(outside) and not any(outside.iterdir())
    assert (zero / "folder" / "inside.md").read_text() == "a folder's file\n"
    rules = {c["path"]: c["rule"] for c in record["conflicts"]}
    assert rules == {"rule.md": "merged line by line", "blob.bin": "the later user's",
                     "kept.md": "a removal yields to a change",
                     "way/escaped.md": "left out: a link or a file stands on its way, or a folder holds it",
                     "folder": "left out: a link or a file stands on its way, or a folder holds it"}


def test_json_is_merged_key_by_key_an_array_keeping_both_sides_items_and_a_value_both_changed_the_later_users(
        tmp_path):
    base = {"a": 1, "b": {"x": 1}, "gone": True, "list": [1, 2, 3], "same": 0}
    ours = {"a": 2, "b": {"x": 1, "y": 2}, "list": [1, 2, 3, 4], "same": 0, "mine": "m"}
    theirs = {"a": 3, "b": {"x": 5}, "gone": True, "list": [2, 3, 5], "same": 0, "theirs": "t"}
    assert merge_json(base, ours, theirs) == {"a": 3, "b": {"x": 5, "y": 2}, "list": [2, 3, 4, 5], "same": 0,
                                              "mine": "m", "theirs": "t"}
    # a .jsonl log is text: both users' records are kept, one a line
    merged, rule = merge_entries("calls.jsonl", Entry(b'{"n": 0}\n'), Entry(b'{"n": 0}\n{"n": 1}\n'),
                                 Entry(b'{"n": 0}\n{"n": 2}\n'), tmp_path)
    assert (merged.data, rule) == (b'{"n": 0}\n{"n": 1}\n{"n": 2}\n', "merged line by line")


SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT);
CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
                       content TEXT);
CREATE TABLE marks (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER REFERENCES messages(id), note TEXT);
CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE VIRTUAL TABLE messages_fts USING fts5(content, content=messages, content_rowid=id);
CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content); END;
CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content); END;
CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content); END;
CREATE VIEW messages_src AS SELECT id, content FROM messages WHERE content NOT LIKE 'base%';
CREATE VIRTUAL TABLE messages_tri USING fts5(content, content='messages_src', content_rowid='id', tokenize='trigram');
CREATE TRIGGER messages_tri_insert AFTER INSERT ON messages WHEN new.content NOT LIKE 'base%' BEGIN
    INSERT INTO messages_tri(rowid, content) VALUES (new.id, new.content); END;
CREATE TRIGGER messages_tri_delete AFTER DELETE ON messages WHEN old.content NOT LIKE 'base%' BEGIN
    INSERT INTO messages_tri(messages_tri, rowid, content) VALUES ('delete', old.id, old.content); END;
"""


def database(path: Path, *statements: str, schema: str = SCHEMA) -> bytes:
    db = sqlite3.connect(path)
    db.executescript(schema)
    for statement in statements:
        db.execute(statement)
    db.commit()
    db.close()
    return path.read_bytes()


def test_a_database_both_users_changed_is_merged_row_by_row_keeping_both_users_sessions_and_messages(tmp_path):
    """A hermes-like store: sessions by id, messages and marks by ids the database allocates, a full-text index its
    triggers keep. Both users' new sessions and messages are kept, the later user's messages numbered after the earlier
    user's and its mark following its message; a row both changed takes the later user's, a row the later user removed
    and nobody else changed goes, and the index finds every message the merged database holds."""
    base = ["INSERT INTO sessions VALUES ('s0', 'base')", "INSERT INTO messages (session_id, content) VALUES "
            "('s0', 'base hello'), ('s0', 'base again')", "INSERT INTO state_meta VALUES ('theme', 'light')",
            "INSERT INTO schema_version VALUES (1)"]
    before = database(tmp_path / "base", *base)
    ours = database(tmp_path / "ours", *base, "INSERT INTO sessions VALUES ('s1', 'zero')",
                    "INSERT INTO messages (session_id, content) VALUES ('s1', 'zero first'), ('s1', 'zero second')",
                    "INSERT INTO marks (message_id, note) VALUES (4, 'zero marked')",
                    "UPDATE state_meta SET value = 'dark'")
    theirs = database(tmp_path / "theirs", *base, "INSERT INTO sessions VALUES ('s2', 'one')",
                      "INSERT INTO messages (session_id, content) VALUES ('s2', 'one a'), ('s2', 'one b'), ('s2', 'one c')",
                      "INSERT INTO marks (message_id, note) VALUES (5, 'one marked')", "DELETE FROM messages WHERE id = 1",
                      "UPDATE state_meta SET value = 'blue'", "UPDATE sessions SET title = 'renamed' WHERE id = 's0'")
    (tmp_path / "merged").write_bytes(merge_database(before, ours, theirs, tmp_path))
    db = sqlite3.connect(tmp_path / "merged")
    assert db.execute("SELECT id, title FROM sessions ORDER BY id").fetchall() == [
        ("s0", "renamed"), ("s1", "zero"), ("s2", "one")]
    assert db.execute("SELECT id, session_id, content FROM messages ORDER BY id").fetchall() == [
        (2, "s0", "base again"), (3, "s1", "zero first"), (4, "s1", "zero second"),
        (5, "s2", "one a"), (6, "s2", "one b"), (7, "s2", "one c")]
    assert db.execute("SELECT id, message_id, note FROM marks ORDER BY id").fetchall() == [
        (1, 4, "zero marked"), (2, 7, "one marked")]
    assert db.execute("SELECT value FROM state_meta").fetchall() == [("blue",)]
    assert db.execute("SELECT version FROM schema_version").fetchall() == [(1,)]
    found = {word: [r[0] for r in db.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rowid",
                                             (word,))] for word in ("zero", "one", "hello")}
    assert found == {"zero": [3, 4], "one": [5, 6, 7], "hello": []}
    assert [r[0] for r in db.execute("SELECT rowid FROM messages_tri WHERE messages_tri MATCH 'one' ORDER BY rowid")] == [
        5, 6, 7]   # the index a view gives its content, kept by triggers on the view's table
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db.execute("INSERT INTO messages (session_id, content) VALUES ('s2', 'next')").lastrowid == 8
    # databases whose schemas differ take the later user's whole
    other = database(tmp_path / "other", schema=SCHEMA.replace("title TEXT", "title TEXT, pinned INTEGER"))
    merged, rule = merge_entries("agent.sqlite", Entry(before), Entry(ours), Entry(other), tmp_path)
    assert (merged.data, rule) == (other, "the later user's: their schemas differ")


class Viewing(CommittingHarness):
    """A CommittingHarness (a memory of solved tasks, a note of every solve, a script naming its own folder, its state
    committed when it ends a round) that writes a file of its own as it solves, and records, as every instance begins,
    what its state holds, its own folder written as {system}."""

    views: dict = {}
    closes = 0

    def act(self, obs):
        if obs.is_fresh:
            state = self.workdir / "state"
            Viewing.views[(obs.round, self.user)] = {
                str(p.relative_to(state)): p.read_text().replace(str(self.workdir), "{system}")
                for p in sorted(state.rglob("*")) if p.is_file()}
        return super().act(obs)

    def observe(self, obs, action, feedback) -> None:
        super().observe(obs, action, feedback)
        if getattr(feedback, "correct", False):
            (self.workdir / "state" / f"own-{self.user}-{obs.round}.md").write_text(f"{self.workdir}\n")

    def close(self) -> None:
        Viewing.closes += 1


def play_team(tmp_path: Path, users: int, rounds: int) -> tuple:
    """``users`` Viewing users over ``rounds`` rounds, each user on two tasks of their own."""
    source = FakeSource(n_tasks=2 * users)
    ids = source.task_ids()
    sched = interleave_users([stream_schedule(ids[2 * u: 2 * u + 2], rounds, seed=u, source=source.name)
                              for u in range(users)], seed=0, name="team")
    Viewing.views, Viewing.closes = {}, 0
    server = FakeUpstream().start()
    learner = CollectiveHarnessLearner(Viewing, tmp_path / "run", schedule=sched, upstream=server.base_url)
    try:
        result = run(learner, sched, out_dir=tmp_path / "run", registry=TaskRegistry({source.name: source}))
        closes = Viewing.closes
    finally:
        learner.close()
        server.stop()
    return learner, result, closes


def test_a_teams_instances_of_a_round_start_from_the_same_store_and_what_each_wrote_survives_into_the_next(tmp_path):
    users, rounds = 3, 3
    learner, result, closes = play_team(tmp_path, users, rounds)
    systems = [tmp_path / "run" / "users" / f"u{user:03d}" / "system" for user in range(users)]
    for r in range(rounds):   # every user's instance of a round starts from the same store
        assert all(Viewing.views[(r, user)] == Viewing.views[(r, 0)] for user in range(users)), r
    for r in range(1, rounds):   # and it holds what every user wrote in the round before
        view = Viewing.views[(r, 0)]
        assert {f"own-{user}-{r - 1}.md" for user in range(users)} <= set(view)
        assert all(view[f"own-{user}-{r - 1}.md"] == "{system}\n" for user in range(users))
        assert [line.split(" in round ")[1] for line in view["notes.md"].splitlines()].count(f"{r - 1}") == users
    assert view["bin/tool"] == "#!{system}/state/venv/bin/python\n"
    # the team's one memory: a task costs a demonstration only the first round anyone meets it
    assert all(r.cost == (1.0 if r.exposure_index == 0 else 0.0) for r in result.records)
    # one history, user 0's: a commit a round, each the merged state; no other user commits
    log = [json.loads(line) for line in (systems[0] / "state-log.jsonl").read_text().splitlines()]
    assert [entry["round"] for entry in log] == list(range(rounds))
    merges = [json.loads(line) for line in (systems[0] / MERGES).read_text().splitlines()]
    assert [m["round"] for m in merges] == list(range(rounds))
    for entry in log:
        assert {f"own-{user}-{entry['round']}.md" for user in range(users)} <= set(
            git(systems[0] / "state.git", "show", "--name-only", "--format=", entry["commit"]).split())
    assert not any((system / "state-log.jsonl").exists() for system in systems[1:])
    assert closes == users * rounds   # every user's system stopped as every round ended
    assert learner.describe()["shared_state"] is True


def test_a_run_of_one_user_keeps_its_state_in_place_as_it_always_has(tmp_path):
    """One user: no copy, no merge and no record of one; the harness is never stopped between rounds, and the state's
    every round is a commit of what the user wrote, in the user's own history."""
    learner, result, closes = play_team(tmp_path, 1, 3)
    system = tmp_path / "run" / "users" / "u000" / "system"
    assert learner.shared is None and learner.describe()["shared_state"] is False
    assert closes == 0
    assert not (system / MERGES).exists() and not (system / "state-merges.index").exists()
    log = [json.loads(line) for line in (system / "state-log.jsonl").read_text().splitlines()]
    assert [entry["round"] for entry in log] == [0, 1, 2]
    for entry in log:
        wrote = set(git(system / "state.git", "show", "--name-only", "--format=", entry["commit"]).split())
        assert f"own-0-{entry['round']}.md" in wrote and "notes.md" in wrote
    assert [r.cost for r in result.records] == [1.0 if r.user_exposure_index == 0 else 0.0 for r in result.records]


def test_a_frozen_tests_users_are_never_merged(tmp_path):
    """A frozen test's round is played from forks of the stream's end that nothing after it reads: its users keep
    their writes apart."""
    ids = FakeSource(n_tasks=3).task_ids()
    test = Schedule(name="seen", seed=0, users=3, sources={t: "fake" for t in ids},
                    slots=[Slot(task, 500 + u, "seen-test", user=u) for u, task in enumerate(ids)],
                    meta={"test": "seen", "width": 3})
    stream = Schedule(name="stream", seed=0, users=3, sources={t: "fake" for t in ids},
                      slots=[Slot(task, 500 + u, "stream", user=u) for u, task in enumerate(ids)])
    assert CollectiveHarnessLearner(Viewing, tmp_path / "test", schedule=test, upstream="http://unused").shared is None
    assert CollectiveHarnessLearner(Viewing, tmp_path / "run", schedule=stream, upstream="http://unused").shared


def test_a_round_cut_short_while_it_played_is_undone_and_the_one_state_put_back_as_it_began(tmp_path):
    """The runner plays a round cut short again from its start: the process that plays it again finds the round open
    and puts the one state back as the round began, so that the round's every user starts from that state again."""
    shared, systems = team(tmp_path, 2)
    one = systems[0] / "state"
    one.mkdir()
    (one / "notes.md").write_text("as the round began\n")
    shared.begin_round()
    (one / "notes.md").write_text("written by the cut attempt\n")
    (one / "partial.md").write_text("written by the cut attempt\n")
    (systems[1] / "state" / "theirs.md").write_text("written by the cut attempt\n")
    again = SharedState(lambda user: systems[user], 2)
    assert again.recover(0) is None
    assert sorted(p.name for p in one.iterdir()) == ["notes.md"] and (one / "notes.md").read_text() == "as the round began\n"
    assert not (systems[0] / JOURNAL).exists()
    again.begin_round()
    assert sorted(p.name for p in (systems[1] / "state").iterdir()) == ["notes.md"]


def test_a_merge_stopped_after_the_runner_recorded_the_round_is_finished_and_committed_before_the_next(tmp_path,
                                                                                                      monkeypatch):
    """A process stopped in a round's merge, after the runner wrote the round's records: the relaunch merges that round
    first, from what every user wrote in it, user 0's writes as user 0 ended the round and not as the cut merge left
    them, and commits it as its own round before the next begins, so the one state's log runs on without a gap."""
    users, rounds = 3, 3
    source = FakeSource(n_tasks=2 * users)
    ids = source.task_ids()
    sched = interleave_users([stream_schedule(ids[2 * u: 2 * u + 2], rounds, seed=u, source=source.name)
                              for u in range(users)], seed=0, name="team")
    registry = TaskRegistry({source.name: source})
    run_dir = tmp_path / "run"
    writes = SharedState.writes

    def stopped(self, user, scratch):
        if user == 2:
            raise KeyboardInterrupt("stopped in the merge")
        return writes(self, user, scratch)

    Viewing.views = {}
    server = FakeUpstream().start()
    try:
        monkeypatch.setattr(SharedState, "writes", stopped)
        first = CollectiveHarnessLearner(Viewing, run_dir, schedule=sched, upstream=server.base_url)
        try:
            run(first, sched, out_dir=run_dir, registry=registry, resume=True)
        except KeyboardInterrupt:
            pass
        finally:
            first.close()
        system = run_dir / "users" / "u000" / "system"
        assert (run_dir / "records.jsonl").read_text().count("\n") == users and not (system / "state-log.jsonl").exists()
        assert (system / "state" / "own-1-0.md").exists()   # the cut merge had placed user 1's writes
        monkeypatch.setattr(SharedState, "writes", writes)
        second = CollectiveHarnessLearner(Viewing, run_dir, schedule=sched, upstream=server.base_url)
        try:
            result = run(second, sched, out_dir=run_dir, registry=registry, resume=True)
        finally:
            second.close()
    finally:
        server.stop()
    assert len(result.records) == users * rounds
    log = [json.loads(line) for line in (system / "state-log.jsonl").read_text().splitlines()]
    assert [entry["round"] for entry in log] == list(range(rounds))
    first_round = set(git(system / "state.git", "show", "--name-only", "--format=", log[0]["commit"]).split())
    assert {f"own-{user}-0.md" for user in range(users)} <= first_round
    assert all(Viewing.views[(1, user)] == Viewing.views[(1, 0)] for user in range(users))
    assert {f"own-{user}-0.md" for user in range(users)} <= set(Viewing.views[(1, 0)])
    merges = [json.loads(line)["round"] for line in (system / MERGES).read_text().splitlines()]
    assert merges == list(range(rounds)) and not (system / JOURNAL).exists()


def test_a_stop_after_the_rounds_commit_leaves_nothing_to_merge_again(tmp_path):
    shared, systems = team(tmp_path, 2)
    (systems[0] / "state").mkdir()
    shared.begin_round()
    (systems[1] / "state" / "theirs.md").write_text("user 1\n")
    shared.ending(0)
    shared.end_round(0)
    (systems[0] / "state-log.jsonl").write_text(json.dumps({"round": 0, "commit": "c0"}) + "\n")   # the commit made
    assert SharedState(lambda user: systems[user], 2).recover(1) is None
    assert (systems[0] / MERGES).read_text().count("\n") == 1 and not (systems[0] / JOURNAL).exists()


def test_a_git_repository_in_the_state_is_merged_file_by_file_its_own_files_the_later_users(tmp_path):
    """A harness's workspace can hold a git repository of its own, which git keeps out of a tree: its work tree's files
    merge as any others, a file one user changed taking that change alone, and git's own files, a branch's tip among
    them, take the later user's; a repository with no commit yet is no obstacle."""
    shared, systems = team(tmp_path, 2)
    repo = systems[0] / "state" / "workspace" / "repo"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".git" / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    (repo / "rule.md").write_text("rule: rotate\n")
    (systems[0] / "state" / "workspace" / "fresh" / ".git").mkdir(parents=True)
    (systems[0] / "state" / "workspace" / "fresh" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    shared.begin_round()
    theirs = systems[1] / "state" / "workspace" / "repo"
    (theirs / "rule.md").write_text("rule: rotate, then mirror\n")
    (theirs / ".git" / "refs" / "heads" / "main").write_text("b" * 40 + "\n")
    (repo / ".git" / "refs" / "heads" / "main").write_text("c" * 40 + "\n")
    record = shared.end_round(0)
    assert (repo / "rule.md").read_text() == "rule: rotate, then mirror\n"
    assert (repo / ".git" / "refs" / "heads" / "main").read_text() == "b" * 40 + "\n"
    assert [(c["path"], c["rule"]) for c in record["conflicts"]] == [
        ("workspace/repo/.git/refs/heads/main", "the later user's")]


def test_a_database_whose_rows_the_merge_cannot_read_whole_takes_the_later_users(tmp_path):
    """A database whose own code keeps a full-text index, or whose trigger rewrites a table's rows (openclaw's agent
    database has both), takes the later user's whole, its reason in the record's rule."""
    counted = ("CREATE TABLE sessions (id TEXT PRIMARY KEY); CREATE TABLE counts (n INTEGER);"
               "CREATE TRIGGER counted AFTER INSERT ON sessions BEGIN UPDATE counts SET n = n + 1; END;")
    indexed = "CREATE TABLE sessions (id TEXT PRIMARY KEY); CREATE VIRTUAL TABLE found USING fts5(body);"
    vocabulary = ("CREATE TABLE sessions (id TEXT PRIMARY KEY); CREATE VIRTUAL TABLE found USING fts5(body);"
                  "CREATE VIRTUAL TABLE words USING fts5vocab(found, 'row');"
                  "CREATE TRIGGER kept AFTER INSERT ON sessions BEGIN INSERT INTO found(body) VALUES (new.id); END;")
    unkept = ("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);"
              "CREATE VIRTUAL TABLE found USING fts5(body, content=notes, content_rowid=id);")
    for k, (schema, why) in enumerate(((counted, "its trigger counted changes the rows of counts"),
                                       (indexed, "no trigger of its own keeps its full-text index found"),
                                       (vocabulary, "its virtual table words is no full-text index"),
                                       (unkept, "no trigger of its own keeps its full-text index found"))):
        folder = tmp_path / f"case-{k}"
        folder.mkdir()
        before = database(folder / "base", schema=schema)
        ours = database(folder / "ours", "INSERT INTO sessions VALUES ('s1')" if "sessions" in schema
                        else "INSERT INTO notes VALUES (1, 'zero')", schema=schema)
        theirs = database(folder / "theirs", "INSERT INTO sessions VALUES ('s2')" if "sessions" in schema
                          else "INSERT INTO notes VALUES (1, 'one')", schema=schema)
        merged, rule = merge_entries("agent.sqlite", Entry(before), Entry(ours), Entry(theirs), folder)
        assert (merged.data, rule) == (theirs, f"the later user's: {why}")


def test_new_rows_whose_integer_ids_clash_are_numbered_after_the_earlier_users_and_a_row_both_added_is_kept_once(
        tmp_path):
    schema = ("CREATE TABLE sources (id INTEGER PRIMARY KEY, path TEXT UNIQUE);"
              "CREATE TABLE chunks (id INTEGER PRIMARY KEY, source INTEGER REFERENCES sources(id), body TEXT);")
    before = database(tmp_path / "base", "INSERT INTO sources VALUES (1, 'shared.md')", schema=schema)
    ours = database(tmp_path / "ours", "INSERT INTO sources VALUES (1, 'shared.md'), (2, 'zero.md')",
                    "INSERT INTO chunks VALUES (1, 2, 'zero')", schema=schema)
    theirs = database(tmp_path / "theirs", "INSERT INTO sources VALUES (1, 'shared.md'), (2, 'one.md')",
                      "INSERT INTO chunks VALUES (1, 2, 'one')", schema=schema)
    (tmp_path / "merged").write_bytes(merge_database(before, ours, theirs, tmp_path))
    assert query(tmp_path / "merged", "SELECT * FROM sources ORDER BY id") == [
        (1, "shared.md"), (2, "zero.md"), (3, "one.md")]
    assert query(tmp_path / "merged", "SELECT * FROM chunks ORDER BY id") == [(1, 2, "zero"), (2, 3, "one")]


def test_a_write_ahead_log_a_stopped_harness_left_is_taken_into_the_base_before_any_copy(tmp_path):
    """A harness killed with its database's write-ahead log unchecked leaves rows the database file alone lacks: they
    are taken in before the round's base is read and its copies made, so no later user's merge adds them again."""
    shared, systems = team(tmp_path, 2)
    one = systems[0] / "state"
    one.mkdir()
    live = tmp_path / "live" / "agent.sqlite"
    live.parent.mkdir()
    db = sqlite3.connect(live)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA wal_autocheckpoint=0")
    db.executescript(SCHEMA)
    db.execute("INSERT INTO sessions VALUES ('s0', 'base')")
    db.execute("INSERT INTO messages (session_id, content) VALUES ('s0', 'base hello'), ('s0', 'base again')")
    db.commit()
    for name in ("agent.sqlite", "agent.sqlite-wal"):   # the files as the killed process left them
        (one / name).write_bytes((live.parent / name).read_bytes())
    db.close()
    shared.begin_round()
    assert not (one / "agent.sqlite-wal").exists()
    db = sqlite3.connect(systems[1] / "state" / "agent.sqlite")
    db.execute("INSERT INTO messages (session_id, content) VALUES ('s0', 'one more')")
    db.commit()
    db.close()
    shared.end_round(0)
    assert query(one / "agent.sqlite", "SELECT id, content FROM messages ORDER BY id") == [
        (1, "base hello"), (2, "base again"), (3, "one more")]


def test_a_run_folder_named_relative_to_where_it_plays_is_one_state_as_any(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    systems = [Path("run") / "users" / f"u{user:03d}" / "system" for user in range(2)]
    for system in systems:
        (system / "state").mkdir(parents=True)
    shared = SharedState(lambda user: systems[user], 2)
    shared.begin_round()
    (systems[1] / "state" / "theirs.md").write_text("user 1\n")
    shared.end_round(0)
    assert (systems[0] / "state" / "theirs.md").read_text() == "user 1\n"


def test_a_run_whose_users_kept_stores_of_their_own_cannot_go_on_with_one_state(tmp_path):
    ids = FakeSource(n_tasks=2).task_ids()
    stream = Schedule(name="stream", seed=0, users=2, sources={t: "fake" for t in ids},
                      slots=[Slot(task, 500 + u, "stream", user=u) for u, task in enumerate(ids)])
    log = tmp_path / "run" / "users" / "u001" / "system" / "state-log.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"round": 0, "commit": "c0"}) + "\n")
    with pytest.raises(SystemExit, match="kept stores of their own"):
        CollectiveHarnessLearner(Viewing, tmp_path / "run", schedule=stream, upstream="http://unused")


def test_a_round_recorded_in_the_moment_before_it_was_marked_is_merged_not_undone(tmp_path):
    """The runner writes a round's records just before the learner marks the round recorded (SharedState.ending): a
    process stopped between the two leaves an open round that the records show was played, which is merged."""
    shared, systems = team(tmp_path, 2)
    (systems[0] / "state").mkdir()
    shared.begin_round()
    (systems[0] / "state" / "mine.md").write_text("user 0\n")
    (systems[1] / "state" / "theirs.md").write_text("user 1\n")
    again = SharedState(lambda user: systems[user], 2)
    assert again.recover(1) == 0
    assert sorted(p.name for p in (systems[0] / "state").iterdir()) == ["mine.md", "theirs.md"]
    again.close_round()
    assert again.recover(1) is None


def test_a_stop_before_user_0s_end_of_the_round_was_kept_takes_in_its_write_ahead_log_first(tmp_path):
    """Stopped after the round was marked recorded and before user 0's own end of it was kept, with user 0's harness
    killed and its database's write-ahead log left: the merge takes the log in before it reads user 0's state, so user
    0's rows and user 1's are both kept."""
    shared, systems = team(tmp_path, 2)
    one = systems[0] / "state"
    one.mkdir()
    database(one / "agent.sqlite", "INSERT INTO sessions VALUES ('s0', 'base')")
    shared.begin_round()
    db = sqlite3.connect(systems[1] / "state" / "agent.sqlite")
    db.execute("INSERT INTO sessions VALUES ('s2', 'one')")
    db.commit()
    db.close()
    live = sqlite3.connect(one / "agent.sqlite")
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("PRAGMA wal_autocheckpoint=0")
    live.execute("INSERT INTO sessions VALUES ('s1', 'zero')")
    live.commit()
    killed = {name: (one / name).read_bytes() for name in ("agent.sqlite", "agent.sqlite-wal")}
    live.close()
    for name, data in killed.items():   # the files as the killed process left them
        (one / name).write_bytes(data)
    (one / "agent.sqlite-shm").unlink(missing_ok=True)
    shared.ending(0)
    assert SharedState(lambda user: systems[user], 2).recover(1) == 0
    assert query(one / "agent.sqlite", "SELECT id FROM sessions ORDER BY id") == [("s0",), ("s1",), ("s2",)]


def test_undoing_a_round_goes_through_folders_its_attempt_made_read_only(tmp_path):
    shared, systems = team(tmp_path, 2)
    (systems[0] / "state").mkdir()
    shared.begin_round()
    cache = systems[0] / "state" / "go" / "pkg" / "mod"
    cache.mkdir(parents=True)
    (cache / "module.go").write_text("package module\n")
    cache.chmod(0o555)
    assert SharedState(lambda user: systems[user], 2).recover(0) is None
    assert not (cache / "module.go").exists()
