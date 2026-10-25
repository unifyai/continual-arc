"""frozen.py off any GPU: the end-of-stream state each test user starts from (stream user i mod U, forked from its
commit of the stream's last round when the run committed its users' states, otherwise copied with its folder's absolute
paths renamed to the test user's in text files, links and a SQLite database's rows; a bare program library's kept
programs and the team's demonstration cache), the run's last phase refused unless it trained on every round, a
trained run's test played on its server and a reference row's on OpenRouter by the row's own play function, the
summary, and the two schedules against the benchmark's own ``test_schedule``. Run with ``python3 -m pytest test_frozen.py -q``;
the schedules' test needs a benchmark with the frozen tests (``continual_arc.configs.test_schedule``) importable, and
the forks' test baselines that commit their users' states (``continual_arc_baselines.frozen_collective`` with ``user_system``,
from ``../baselines``), each skipped without it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

import frozen

HERE = Path(__file__).resolve().parent


def _stream(tmp_path, users=2, rounds=3, latest_rounds=3):
    """A finished real-time run's folder: its schedule, alias-salt and driver.json, and each user's harness state naming
    its own folder in a script, a session, a link and a compiled module."""
    run = tmp_path / "runs" / "rt" / "pg-u2-lag1"
    play = run / "play"
    play.mkdir(parents=True)
    (play / "schedule.json").write_text(json.dumps({"users": users, "slots": [{}] * (users * rounds)}))
    (play / "alias-salt").write_text("salt\n")
    (play / "driver.json").write_text(json.dumps({"row": "pg"}))
    for u in range(users):
        home = play / "users" / f"u{u:03d}"
        state = home / "system" / "state"
        (state / "workspace" / ".venv" / "bin").mkdir(parents=True)
        (state / "sessions").mkdir()
        (state / "workspace" / ".venv" / "bin" / "pip").write_text(f"#!{state}/workspace/.venv/bin/python\n")
        (state / "workspace" / ".venv" / "bin" / "pip").chmod(0o755)
        (state / "sessions" / "s.jsonl").write_text(json.dumps({"cwd": f"{state}/workspace", "user": u}) + "\n"
                                                   + json.dumps({"other": f"{home}0/x", "note": str(home)}) + "\n")
        (state / "workspace" / "lib").symlink_to(f"{state}/workspace/.venv")
        (state / "workspace" / "py").symlink_to("/usr/bin/python3")
        (state / "workspace" / "m.pyc").write_bytes(b"\0compiled " + str(state).encode())
    (run / "train").mkdir()
    (run / "train" / "latest.json").write_text(json.dumps({"phase": latest_rounds - 1, "rounds": latest_rounds,
                                                           "adapter": str(run / "train" / "p0002" / "adapter")}))
    return run


def _commit(run, rounds=3):
    """``run`` (:func:`_stream`) as a stream on baselines that commit their users' states leaves it: every user's state
    committed at the end of each round as CollectiveHarnessLearner commits it (``statelog.StateLog``), each round adding
    a line to the user's notes, and the benchmark's schedule and a record of every instance, which baselines' seeding reads."""
    from continual_arc_baselines.statelog import StateLog
    play, users = run / "play", frozen.stream_users(run)
    (play / "schedule.json").write_text(json.dumps({
        "format": "continual-arc/schedule/v1", "name": "stream", "seed": 0, "users": users, "sources": {}, "meta": {},
        "slots": [{"task_id": f"t{r}", "instance_seed": r, "phase": "stream", "user": u}
                  for r in range(rounds) for u in range(users)]}))
    (play / "records.jsonl").write_text("".join(json.dumps({"phase": "stream", "round": r, "user": u}) + "\n"
                                                for r in range(rounds) for u in range(users)))
    for r in range(rounds):
        for u in range(users):
            system = play / "users" / f"u{u:03d}" / "system"
            with (system / "state" / "notes.md").open("a") as notes:
                notes.write(f"round {r}\n")
            StateLog(system / "state", system / "state.git", system / "state-log.jsonl").commit()
    return run


def _git(system, *args):
    return subprocess.run(["git", f"--git-dir={system / 'state.git'}", *args], capture_output=True, text=True,
                          check=True).stdout


def test_each_test_user_starts_from_its_stream_users_end_state_in_a_folder_of_its_own(tmp_path):
    run = _stream(tmp_path)
    test_dir = run / "frozen" / "held-out" / "play"
    memory = frozen.seed(run, test_dir, 4)
    assert (test_dir / "alias-salt").read_text() == "salt\n"
    assert memory == {"memory": "test user i from stream user i mod stream_users, relocated", "stream_users": 2,
                      "stream_rounds": 3,   # each user's script, session and link renamed
                      "relocated": {u: {"files": 2, "links": 1, "databases": 0} for u in ("u000", "u001")}}
    for user in range(4):
        home, src = test_dir / "users" / f"u{user:03d}", run / "play" / "users" / f"u{user % 2:03d}"
        state = home / "system" / "state"
        pip = state / "workspace" / ".venv" / "bin" / "pip"
        assert pip.read_text() == f"#!{state}/workspace/.venv/bin/python\n"
        assert os.access(pip, os.X_OK)
        cwd, other = [json.loads(line) for line in (state / "sessions" / "s.jsonl").read_text().splitlines()]
        assert cwd == {"cwd": f"{state}/workspace", "user": user % 2}
        # a path that only begins with the folder's name is another folder's, and kept
        assert other == {"other": f"{src}0/x", "note": str(home)}
        assert os.readlink(state / "workspace" / "lib") == f"{state}/workspace/.venv"
        assert os.readlink(state / "workspace" / "py") == "/usr/bin/python3"
        assert (state / "workspace" / "m.pyc").read_bytes() == b"\0compiled " + str(src / "system" / "state").encode()
    # the stream users' own states are as they were
    src = run / "play" / "users" / "u001" / "system" / "state"
    assert json.loads((src / "sessions" / "s.jsonl").read_text().splitlines()[0])["cwd"] == f"{src}/workspace"


def test_a_run_with_state_logs_forks_its_users_commits_and_one_without_copies_their_folders(tmp_path):
    forks = pytest.importorskip("continual_arc_baselines.frozen_collective")
    if not hasattr(forks, "user_system"):
        pytest.skip("the installed baselines predates collective runs' commits")
    from continual_arc_baselines.statelog import StateLog
    # no state logs: every user's folder copied and relocated, its history none of the stream's
    copied = _stream(tmp_path / "copied")
    assert not frozen.committed(copied)
    doc = frozen.seed(copied, copied / "frozen" / "held-out" / "play", 4)
    assert "from" not in doc and set(doc["relocated"]) == {"u000", "u001"}
    assert [p.name for p in (copied / "frozen" / "held-out" / "play" / "users" / "u003" / "system").iterdir()] == [
        "state"]
    # a state log for each user: test user i a fork of stream user i mod 2's commit of round 2, the stream's last
    run = _commit(_stream(tmp_path / "committed"))
    play, test_dir = run / "play", run / "frozen" / "held-out" / "play"
    (play / "library").mkdir()
    for u in range(2):
        (play / "library" / f"r0002-u{u:03d}.json").write_text(json.dumps({"user": u}))
    (play / "demo-cache").mkdir()
    (play / "demo-cache" / "r0002.json").write_text("[]")
    assert frozen.committed(run)
    commits = [json.loads((play / "users" / f"u{u:03d}" / "system" / "state-log.jsonl").read_text().splitlines()[-1])[
        "commit"] for u in range(2)]
    assert frozen.seed(run, test_dir, 4) == {
        "memory": "test user i a fork of stream user i mod stream_users at its commit of the stream's last round, "
                  "relocated", "stream_users": 2, "stream_rounds": 3,
        "from": {"round": 2, "systems": [{"system": str(play / "users" / f"u{u:03d}" / "system"), "commit": commits[u],
                                          "relocated": 3, "databases": 0} for u in range(2)]}}   # script, session, link
    assert (test_dir / "alias-salt").read_text() == "salt\n"
    for user in range(4):
        system, src = test_dir / "users" / f"u{user:03d}" / "system", play / "users" / f"u{user % 2:03d}" / "system"
        state = system / "state"
        assert (state / "workspace" / ".venv" / "bin" / "pip").read_text() == f"#!{state}/workspace/.venv/bin/python\n"
        assert os.access(state / "workspace" / ".venv" / "bin" / "pip", os.X_OK)
        assert json.loads((state / "sessions" / "s.jsonl").read_text().splitlines()[0]) == {
            "cwd": f"{state}/workspace", "user": user % 2}
        assert os.readlink(state / "workspace" / "lib") == f"{state}/workspace/.venv"
        assert (state / "workspace" / "m.pyc").read_bytes() == b"\0compiled " + str(src / "state").encode()
        assert (state / "notes.md").read_text() == "round 0\nround 1\nround 2\n"
        # the stream user's history, then the renaming, committed before the test begins
        assert _git(system, "log", "--format=%s").splitlines() == [f"relocated from {src}", "round 2", "round 1",
                                                                    "round 0"]
        assert _git(system, "rev-parse", "HEAD~1").strip() == commits[user % 2]
        assert _git(system, "diff", "--name-only", "HEAD~1", "HEAD").split() == [
            "sessions/s.jsonl", "workspace/.venv/bin/pip", "workspace/lib"]
        assert (system / "state-log.jsonl").read_text() == (src / "state-log.jsonl").read_text()
    # the test round's commit, as the learner's end_round makes it, holds what the user wrote in it and nothing else
    system = test_dir / "users" / "u003" / "system"
    (system / "state" / "workspace" / "answer.py").write_text("print(1)\n")
    StateLog(system / "state", system / "state.git", system / "state-log.jsonl").commit()
    assert _git(system, "diff", "--name-only", "HEAD~1", "HEAD").split() == ["workspace/answer.py"]
    assert json.loads((system / "state-log.jsonl").read_text().splitlines()[-1])["round"] == 3
    # a bare row's memory and the team's cache are copied as before, and the stream users' states are as they were
    assert [json.loads((test_dir / "library" / f"r0002-u{u:03d}.json").read_text()) for u in range(4)] == [
        {"user": u % 2} for u in range(4)]
    assert [p.name for p in (test_dir / "demo-cache").iterdir()] == ["r0002.json"]
    src = play / "users" / "u001" / "system"
    assert _git(src, "rev-parse", "HEAD").strip() == commits[1]
    assert json.loads((src / "state" / "sessions" / "s.jsonl").read_text().splitlines()[0])["cwd"] == \
        f"{src}/state/workspace"


def test_a_database_names_the_test_users_folder_in_its_rows_its_full_text_index_left_alone(tmp_path):
    run = _stream(tmp_path, users=1)
    state = run / "play" / "users" / "u000" / "system" / "state"
    home = state.parent.parent
    db = sqlite3.connect(state / "agent.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT)")
    db.execute("CREATE VIRTUAL TABLE notes USING fts5(body)")
    db.execute("INSERT INTO sessions VALUES ('s1', ?), ('s2', ?)", (f"{state}/workspace", f"{home}0/elsewhere"))
    db.execute("INSERT INTO notes VALUES (?)", (f"ran in {state}",))
    db.commit()   # left open: the rows are in the write-ahead log the copy takes with the file
    frozen.seed(run, run / "frozen" / "seen" / "play", 2)
    for user in range(2):
        mine = run / "frozen" / "seen" / "play" / "users" / f"u{user:03d}"
        copy = sqlite3.connect(mine / "system" / "state" / "agent.sqlite")
        assert copy.execute("SELECT cwd FROM sessions ORDER BY id").fetchall() == [
            (f"{mine}/system/state/workspace",), (f"{home}0/elsewhere",)]
        assert copy.execute("SELECT body FROM notes").fetchall() == [(f"ran in {state}",)]
    assert db.execute("SELECT cwd FROM sessions WHERE id = 's1'").fetchone() == (f"{state}/workspace",)


def test_a_bare_rows_memory_is_its_users_last_library_and_the_teams_cache(tmp_path):
    run = _stream(tmp_path)   # two users, three rounds
    play = run / "play"
    for u in range(2):
        (play / "users" / f"u{u:03d}" / "system" / "state").rename(play / "users" / f"u{u:03d}" / "system" / "gone")
    (play / "library").mkdir()
    for r in range(3):
        for u in range(2):
            (play / "library" / f"r{r:04d}-u{u:03d}.json").write_text(json.dumps({"round": r, "user": u}))
    (play / "demo-cache").mkdir()
    for r in range(3):
        (play / "demo-cache" / f"r{r:04d}.json").write_text(json.dumps([{"round": r}]))
    test_dir = run / "frozen" / "held-out" / "play"
    assert frozen.seed(run, test_dir, 4)["relocated"] == {}
    assert [json.loads((test_dir / "library" / f"r0002-u{u:03d}.json").read_text()) for u in range(4)] == [
        {"round": 2, "user": u % 2} for u in range(4)]
    assert sorted(p.name for p in (test_dir / "library").iterdir()) == [f"r0002-u{u:03d}.json" for u in range(4)]
    assert sorted(p.name for p in (test_dir / "demo-cache").iterdir()) == ["r0000.json", "r0001.json", "r0002.json"]
    assert not (test_dir / "users").exists()


def _play_with(monkeypatch, *argv):
    """frozen.py play of a held-out test with stand-ins for the run's own code, which record each play."""
    played = []

    class Result:
        records = [types.SimpleNamespace(solved=True)]

    def play(*args, **kwargs):
        played.append((args, kwargs))
        return Result()

    monkeypatch.setitem(sys.modules, "paper", types.SimpleNamespace(round_recorded=lambda d, s: False, play_round=play))
    monkeypatch.setitem(sys.modules, "reference", types.SimpleNamespace(BARE_ROWS={"m": ()}, play_round=play,
                                                                       play_bare_round=play))
    monkeypatch.setattr(frozen, "check_served", lambda *a: played.append(("checked", a)))
    monkeypatch.setenv("OPENROUTER_API_KEY", "a key")
    monkeypatch.setattr(sys, "argv", ["frozen.py", "play", "--test", "held-out", "--schedule",
                                      str(HERE / "schedules" / "arc1-small-held-out-test-seed0.json"), *argv])
    frozen.main()
    return played


def test_a_trained_run_plays_on_its_server_and_a_reference_row_through_its_own_play(tmp_path, monkeypatch):
    pytest.importorskip("continual_arc.schedule")
    run = _stream(tmp_path)
    played = _play_with(monkeypatch, "--run", str(run), "--base-url", "http://localhost:8120/v1", "--served",
                        "pg-u2-lag1-p0002", "--adapter", "/a")
    assert played[0][0] == "checked" and played[1][0][0] == "pg" and played[1][1] == {
        "base_url": "http://localhost:8120/v1", "served_name": "pg-u2-lag1-p0002"}
    (run / "play" / "driver.json").write_text(json.dumps({"row": "hg", "player": "qwen"}))
    played = _play_with(monkeypatch, "--run", str(run), "--player", "qwen")
    assert [args[:2] for args, _ in played] == [("qwen", "hg")] and played[0][1] == {"api_key": "a key"}
    (run / "play" / "driver.json").write_text(json.dumps({"row": "m", "player": "glm"}))
    played = _play_with(monkeypatch, "--run", str(run), "--player", "glm")
    assert [args[:2] for args, _ in played] == [("glm", "m")] and played[0][1] == {"api_key": "a key", "rounds_done": 3}
    doc = json.loads((run / "frozen" / "held-out" / "play" / "frozen.json").read_text())
    assert (doc["player"], doc["row"], doc["stream_rounds"]) == ("glm", "m", 3)
    # the same row from an empty state: nothing seeded, no rounds done, into held-out-reference
    played = _play_with(monkeypatch, "--run", str(run), "--player", "glm", "--reference")
    assert [args[:3] for args, _ in played] == [("glm", "m", run / "frozen" / "held-out-reference" / "play")]
    assert played[0][1] == {"api_key": "a key", "rounds_done": 0}
    assert sorted(p.name for p in (run / "frozen" / "held-out-reference" / "play").iterdir()) == ["frozen.json"]
    # another player from the row's memory, under a name of its own
    played = _play_with(monkeypatch, "--run", str(run), "--player", "qwen", "--name", "held-out-qwen")
    assert [args[:3] for args, _ in played] == [("qwen", "m", run / "frozen" / "held-out-qwen" / "play")]
    with pytest.raises(SystemExit, match="give --run and --player"):
        _play_with(monkeypatch, "--run", str(run), "--player", "glm", "--base-url", "x", "--served", "y", "--adapter", "z")


def test_a_reference_from_nothing_plays_the_base_model_with_no_memory(tmp_path, monkeypatch):
    pytest.importorskip("continual_arc.schedule")
    out = tmp_path / "qwen-f" / "frozen"
    played = _play_with(monkeypatch, "--row", "bare", "--out", str(out), "--base-url", "http://localhost:8126/v1",
                        "--served", "Qwen/Qwen3.5-9B")
    play = out / "held-out-reference" / "play"
    assert [args[0] for args, _ in played] == ["bare"] and played[0][0][1] == play
    assert played[0][1] == {"base_url": "http://localhost:8126/v1", "served_name": "Qwen/Qwen3.5-9B"}
    assert sorted(p.name for p in play.iterdir()) == ["frozen.json"]   # no memory, no task names
    assert json.loads((play / "frozen.json").read_text())["memory"] == "none: played from an empty state"
    with pytest.raises(SystemExit, match="takes --row, --out, --base-url and --served alone"):
        _play_with(monkeypatch, "--row", "bare", "--run", str(tmp_path), "--out", str(out), "--base-url", "x",
                   "--served", "y")


def test_a_trained_runs_weights_alone_play_from_an_empty_state_under_its_own_task_names(tmp_path, monkeypatch):
    pytest.importorskip("continual_arc.schedule")
    run = _stream(tmp_path)
    argv = ("--run", str(run), "--base-url", "http://localhost:8120/v1", "--served", "pg-u2-lag1-p0002", "--adapter", "/a")
    played = _play_with(monkeypatch, *argv, "--no-memory")
    play = run / "frozen" / "held-out-weights" / "play"
    assert played[0][0] == "checked" and played[1][0][:2] == ("pg", play)
    assert played[1][1] == {"base_url": "http://localhost:8120/v1", "served_name": "pg-u2-lag1-p0002"}
    # the run's task names and nothing its harness kept
    assert sorted(p.name for p in play.iterdir()) == ["alias-salt", "frozen.json"]
    assert (play / "alias-salt").read_text() == "salt\n"
    doc = json.loads((play / "frozen.json").read_text())
    assert (doc["row"], doc["adapter"], doc["memory"]) == (
        "pg", "/a", "none: the run's weights alone, from an empty state under the run's task names")
    with pytest.raises(SystemExit, match="give --run and --player"):
        _play_with(monkeypatch, "--run", str(run), "--player", "glm", "--no-memory")
    with pytest.raises(SystemExit, match="takes --row, --out, --base-url and --served alone"):
        _play_with(monkeypatch, "--row", "h0", "--out", str(tmp_path / "o"), "--base-url", "x", "--served", "y",
                   "--no-memory")


def test_seeding_again_starts_from_nothing(tmp_path):
    run = _stream(tmp_path)
    test_dir = run / "frozen" / "seen" / "play"
    frozen.seed(run, test_dir, 2)
    (test_dir / "records.jsonl").write_text("{}\n")
    (test_dir / "users" / "u000" / "system" / "state" / "workspace" / "written-in-the-test").write_text("x")
    frozen.seed(run, test_dir, 2)
    assert not (test_dir / "records.jsonl").exists()
    assert not (test_dir / "users" / "u000" / "system" / "state" / "workspace" / "written-in-the-test").exists()


def test_the_tests_play_the_last_phase_only_when_it_trained_on_every_round(tmp_path):
    assert frozen.final_phase(_stream(tmp_path / "a"))["phase"] == 2
    with pytest.raises(SystemExit, match="trained on 2 rounds, not the 3"):
        frozen.final_phase(_stream(tmp_path / "b", latest_rounds=2))


def test_the_summary_counts_each_tests_solves_and_cost(tmp_path):
    run = _stream(tmp_path)
    for test, games in (("held-out", [(True, 2), (False, 16)]), ("seen", [(True, 1)])):
        d = run / "frozen" / test / "play"
        d.mkdir(parents=True)
        (d / "records.jsonl").write_text("".join(json.dumps({"solved": s, "cost": c}) + "\n" for s, c in games))
        (d / "frozen.json").write_text(json.dumps({"test": test, "served": "pg-u2-lag1-p0002"}))
    frozen.cmd_summary(type("O", (), {"run": str(run), "out": None})())
    doc = json.loads((run / "frozen" / "summary.json").read_text())
    assert doc["tests"] == {
        "held-out": {"test": "held-out", "served": "pg-u2-lag1-p0002", "instances": 2, "solved": 1, "mean_cost": 9.0},
        "seen": {"test": "seen", "served": "pg-u2-lag1-p0002", "instances": 1, "solved": 1, "mean_cost": 1.0}}


@pytest.mark.parametrize("test", frozen.TESTS)
def test_the_schedules_are_the_benchmarks_own(test):
    configs = pytest.importorskip("continual_arc.configs")
    if not hasattr(configs, "test_schedule"):
        pytest.skip("the installed benchmark predates the frozen tests")
    from continual_arc.schedule import Schedule
    kept = Schedule.load(HERE / "schedules" / f"arc1-small-{test}-test-seed0.json")
    made = configs.test_schedule(configs.get_config("arc1-small"), 0, test)
    assert kept.to_json() == json.loads(json.dumps(made.to_json()))   # as saved: a tuple is a list in JSON
    assert (kept.users, len(kept.slots), set(kept.rounds())) == (100, 100, {0})
