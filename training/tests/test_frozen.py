"""The frozen tests (continual_arc_training.frozen and scripts/frozen.sh) off any GPU: the end-of-stream state each test
user starts from (stream user i mod U, forked from its commit of the stream's last round when the run committed its
users' states, otherwise copied with its folder's absolute paths renamed to the test user's in text files, links and a
SQLite database's rows; a bare program library's kept programs and the team's demonstration cache), the run's last
phase refused unless it trained on every round, each kind of play dispatched to its own play function, the summary,
the two schedules against the benchmark's own, and frozen.sh end to end with every node tool stubbed."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from continual_arc_training import frozen, play, references

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _stream(tmp_path, users=2, rounds=3, latest_rounds=3):
    """A finished real-time run's folder: its schedule, alias-salt and driver.json, and each user's harness state naming
    its own folder in a script, a session, a link and a compiled module."""
    run = tmp_path / "runs" / "rt" / "pg-u2-lag1"
    played = run / "play"
    played.mkdir(parents=True)
    (played / "schedule.json").write_text(json.dumps({"users": users, "slots": [{}] * (users * rounds)}))
    (played / "alias-salt").write_text("salt\n")
    (played / "driver.json").write_text(json.dumps({"row": "pg"}))
    for u in range(users):
        home = played / "users" / f"u{u:03d}"
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
        assert pip.read_text() == f"#!{state}/workspace/.venv/bin/python\n" and os.access(pip, os.X_OK)
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


def _forks(monkeypatch):
    """The baselines' seeding of a test from a run's commits (``continual_arc_baselines.frozen.seed`` and
    ``user_system``), where the installed baselines has it."""
    import importlib

    import continual_arc_baselines
    for name in ("frozen", "frozen_collective"):
        try:
            module = importlib.import_module(f"continual_arc_baselines.{name}")
        except ImportError:
            continue
        if hasattr(module, "seed") and hasattr(module, "user_system"):
            monkeypatch.setattr(continual_arc_baselines, "frozen", module, raising=False)
            return module
    pytest.skip("the installed baselines has no seeding from a run's commits")


def _commit(run, rounds=3):
    """``run`` (:func:`_stream`) as a stream whose learner committed its users' states leaves it: every user's state
    committed at the end of each round (the baselines' ``statelog.StateLog``), each round adding a line to the user's
    notes, and the benchmark's schedule and a record of every instance, which the baselines' seeding reads."""
    from continual_arc_baselines.statelog import StateLog
    played, users = run / "play", frozen.stream_users(run)
    (played / "schedule.json").write_text(json.dumps({
        "format": "continual-arc/schedule/v1", "name": "stream", "seed": 0, "users": users, "sources": {}, "meta": {},
        "slots": [{"task_id": f"t{r}", "instance_seed": r, "phase": "stream", "user": u}
                  for r in range(rounds) for u in range(users)]}))
    (played / "records.jsonl").write_text("".join(json.dumps({"phase": "stream", "round": r, "user": u}) + "\n"
                                                  for r in range(rounds) for u in range(users)))
    for r in range(rounds):
        for u in range(users):
            system = played / "users" / f"u{u:03d}" / "system"
            with (system / "state" / "notes.md").open("a") as notes:
                notes.write(f"round {r}\n")
            StateLog(system / "state", system / "state.git", system / "state-log.jsonl").commit()
    return run


def _git(system, *args):
    return subprocess.run(["git", f"--git-dir={system / 'state.git'}", *args], capture_output=True, text=True,
                          check=True).stdout


def test_a_run_with_state_logs_forks_its_users_commits_and_one_without_copies_their_folders(tmp_path, monkeypatch):
    _forks(monkeypatch)
    from continual_arc_baselines.statelog import StateLog
    copied = _stream(tmp_path / "copied")
    assert not frozen.committed(copied)
    doc = frozen.seed(copied, copied / "frozen" / "held-out" / "play", 4)
    assert "from" not in doc and set(doc["relocated"]) == {"u000", "u001"}
    run = _commit(_stream(tmp_path / "committed"))
    played, test_dir = run / "play", run / "frozen" / "held-out" / "play"
    (played / "library").mkdir()
    for u in range(2):
        (played / "library" / f"r0002-u{u:03d}.json").write_text(json.dumps({"user": u}))
    (played / "demo-cache").mkdir()
    (played / "demo-cache" / "r0002.json").write_text("[]")
    assert frozen.committed(run)
    commits = [json.loads((played / "users" / f"u{u:03d}" / "system" / "state-log.jsonl").read_text()
                          .splitlines()[-1])["commit"] for u in range(2)]
    assert frozen.seed(run, test_dir, 4) == {
        "memory": "test user i a fork of stream user i mod stream_users at its commit of the stream's last round, "
                  "relocated", "stream_users": 2, "stream_rounds": 3,
        "from": {"round": 2, "systems": [{"system": str(played / "users" / f"u{u:03d}" / "system"),
                                          "commit": commits[u], "relocated": 3, "databases": 0} for u in range(2)]}}
    assert (test_dir / "alias-salt").read_text() == "salt\n"
    for user in range(4):
        system, src = test_dir / "users" / f"u{user:03d}" / "system", played / "users" / f"u{user % 2:03d}" / "system"
        state = system / "state"
        assert (state / "workspace" / ".venv" / "bin" / "pip").read_text() == \
            f"#!{state}/workspace/.venv/bin/python\n"
        assert json.loads((state / "sessions" / "s.jsonl").read_text().splitlines()[0]) == {
            "cwd": f"{state}/workspace", "user": user % 2}
        assert (state / "notes.md").read_text() == "round 0\nround 1\nround 2\n"
        # the stream user's history, then the renaming, committed before the test begins
        assert _git(system, "log", "--format=%s").splitlines() == [f"relocated from {src}", "round 2", "round 1",
                                                                    "round 0"]
        assert _git(system, "rev-parse", "HEAD~1").strip() == commits[user % 2]
    # the test round's commit, as the learner makes it, holds what the user wrote in it and nothing else
    system = test_dir / "users" / "u003" / "system"
    (system / "state" / "workspace" / "answer.py").write_text("print(1)\n")
    StateLog(system / "state", system / "state.git", system / "state-log.jsonl").commit()
    assert _git(system, "diff", "--name-only", "HEAD~1", "HEAD").split() == ["workspace/answer.py"]
    # a bare row's memory and the team's cache are copied as before
    assert [json.loads((test_dir / "library" / f"r0002-u{u:03d}.json").read_text()) for u in range(4)] == [
        {"user": u % 2} for u in range(4)]
    assert [p.name for p in (test_dir / "demo-cache").iterdir()] == ["r0002.json"]


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
    played = run / "play"
    for u in range(2):
        (played / "users" / f"u{u:03d}" / "system" / "state").rename(played / "users" / f"u{u:03d}" / "system" / "gone")
    (played / "library").mkdir()
    (played / "demo-cache").mkdir()
    for r in range(3):
        for u in range(2):
            (played / "library" / f"r{r:04d}-u{u:03d}.json").write_text(json.dumps({"round": r, "user": u}))
        (played / "demo-cache" / f"r{r:04d}.json").write_text(json.dumps([{"round": r}]))
    test_dir = run / "frozen" / "held-out" / "play"
    assert frozen.seed(run, test_dir, 4)["relocated"] == {}
    assert [json.loads((test_dir / "library" / f"r0002-u{u:03d}.json").read_text()) for u in range(4)] == [
        {"round": 2, "user": u % 2} for u in range(4)]
    assert sorted(p.name for p in (test_dir / "library").iterdir()) == [f"r0002-u{u:03d}.json" for u in range(4)]
    assert sorted(p.name for p in (test_dir / "demo-cache").iterdir()) == ["r0000.json", "r0001.json", "r0002.json"]
    assert not (test_dir / "users").exists()


def test_seeding_again_starts_from_nothing(tmp_path):
    run = _stream(tmp_path)
    test_dir = run / "frozen" / "seen" / "play"
    frozen.seed(run, test_dir, 2)
    (test_dir / "records.jsonl").write_text("{}\n")
    (test_dir / "users" / "u000" / "system" / "state" / "workspace" / "written-in-the-test").write_text("x")
    frozen.seed(run, test_dir, 2)
    assert not (test_dir / "records.jsonl").exists()
    assert not (test_dir / "users" / "u000" / "system" / "state" / "workspace" / "written-in-the-test").exists()


def _play_with(monkeypatch, *argv):
    """The ``play`` command of a held-out test with stand-ins for the plays, which record each one."""
    played = []

    class Result:
        records = [types.SimpleNamespace(solved=True)]

    def stand_in(*args, **kwargs):
        played.append((args, kwargs))
        return Result()

    monkeypatch.setattr(play, "round_recorded", lambda d, s: False)
    monkeypatch.setattr(play, "play_round", stand_in)
    monkeypatch.setattr(references, "BARE_ROWS", {"m": ()})
    monkeypatch.setattr(references, "play_round", stand_in)
    monkeypatch.setattr(references, "play_bare_round", stand_in)
    monkeypatch.setattr(frozen, "check_served", lambda *a: played.append(("checked", a)))
    monkeypatch.setenv("OPENROUTER_API_KEY", "a key")
    monkeypatch.setattr(sys, "argv", ["frozen", "play", "--test", "held-out", *argv])
    frozen.main()
    return played


def test_a_trained_run_plays_on_its_server_and_a_reference_row_through_its_own_play(tmp_path, monkeypatch):
    run = _stream(tmp_path)
    played = _play_with(monkeypatch, "--run", str(run), "--base-url", "http://127.0.0.1:8120/v1", "--served",
                        "pg-u2-lag1-p0002", "--adapter", "/a")
    assert played[0][0] == "checked" and played[1][0][0] == "pg" and played[1][1] == {
        "base_url": "http://127.0.0.1:8120/v1", "served_name": "pg-u2-lag1-p0002"}
    assert played[1][0][2].users == 100 and played[1][0][2].meta["test"] == "held-out"   # the package's schedule
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
        _play_with(monkeypatch, "--run", str(run), "--player", "glm", "--base-url", "x", "--served", "y", "--adapter",
                   "z")


def test_a_reference_from_nothing_plays_the_base_model_with_no_memory(tmp_path, monkeypatch):
    out = tmp_path / "qwen-f" / "frozen"
    played = _play_with(monkeypatch, "--row", "bare", "--out", str(out), "--base-url", "http://127.0.0.1:8126/v1",
                        "--served", "Qwen/Qwen3.5-9B")
    test_play = out / "held-out-reference" / "play"
    assert [args[0] for args, _ in played] == ["bare"] and played[0][0][1] == test_play
    assert played[0][1] == {"base_url": "http://127.0.0.1:8126/v1", "served_name": "Qwen/Qwen3.5-9B"}
    assert sorted(p.name for p in test_play.iterdir()) == ["frozen.json"]   # no memory, no task names
    assert json.loads((test_play / "frozen.json").read_text())["memory"] == "none: played from an empty state"
    with pytest.raises(SystemExit, match="takes --row, --out, --base-url and --served alone"):
        _play_with(monkeypatch, "--row", "bare", "--run", str(tmp_path), "--out", str(out), "--base-url", "x",
                   "--served", "y")


def test_a_trained_runs_weights_alone_play_from_an_empty_state_under_its_own_task_names(tmp_path, monkeypatch):
    run = _stream(tmp_path)
    argv = ("--run", str(run), "--base-url", "http://127.0.0.1:8120/v1", "--served", "pg-u2-lag1-p0002", "--adapter",
            "/a")
    played = _play_with(monkeypatch, *argv, "--no-memory")
    test_play = run / "frozen" / "held-out-weights" / "play"
    assert played[0][0] == "checked" and played[1][0][:2] == ("pg", test_play)
    assert sorted(p.name for p in test_play.iterdir()) == ["alias-salt", "frozen.json"]
    doc = json.loads((test_play / "frozen.json").read_text())
    assert (doc["row"], doc["adapter"], doc["memory"]) == (
        "pg", "/a", "none: the run's weights alone, from an empty state under the run's task names")
    with pytest.raises(SystemExit, match="give --run and --player"):
        _play_with(monkeypatch, "--run", str(run), "--player", "glm", "--no-memory")
    with pytest.raises(SystemExit, match="takes --row, --out, --base-url and --served alone"):
        _play_with(monkeypatch, "--row", "h0", "--out", str(tmp_path / "o"), "--base-url", "x", "--served", "y",
                   "--no-memory")


def test_the_tests_play_the_last_phase_only_when_it_trained_on_every_round(tmp_path, monkeypatch, capsys):
    run = _stream(tmp_path / "a")
    assert frozen.final_phase(run)["phase"] == 2
    monkeypatch.setattr(sys, "argv", ["frozen", "model", "--run", str(run)])
    frozen.main()
    assert capsys.readouterr().out == f"pg-u2-lag1-p0002 {run / 'train' / 'p0002' / 'adapter'}\n"
    with pytest.raises(SystemExit, match="trained on 2 rounds, not the 3"):
        frozen.final_phase(_stream(tmp_path / "b", latest_rounds=2))


def test_the_summary_counts_each_tests_solves_and_cost(tmp_path):
    run = _stream(tmp_path)
    for test, games in (("held-out", [(True, 2), (False, 16)]), ("seen", [(True, 1)])):
        d = run / "frozen" / test / "play"
        d.mkdir(parents=True)
        (d / "records.jsonl").write_text("".join(json.dumps({"solved": s, "cost": c}) + "\n" for s, c in games))
        (d / "frozen.json").write_text(json.dumps({"test": test, "served": "pg-u2-lag1-p0002"}))
    frozen.cmd_summary(types.SimpleNamespace(run=str(run), out=None))
    doc = json.loads((run / "frozen" / "summary.json").read_text())
    assert doc["tests"] == {
        "held-out": {"test": "held-out", "served": "pg-u2-lag1-p0002", "instances": 2, "solved": 1, "mean_cost": 9.0},
        "seen": {"test": "seen", "served": "pg-u2-lag1-p0002", "instances": 1, "solved": 1, "mean_cost": 1.0}}


@pytest.mark.parametrize("test", frozen.TESTS)
def test_the_schedules_are_the_benchmarks_own(test):
    from continual_arc import configs
    from continual_arc.schedule import Schedule
    if not hasattr(configs, "test_schedule"):
        pytest.skip("the installed benchmark predates the frozen tests")
    kept = Schedule.load(frozen.schedule_path(test))
    made = configs.test_schedule(configs.get_config("arc1-small"), 0, test)
    assert kept.to_json() == json.loads(json.dumps(made.to_json()))   # as saved: a tuple is a list in JSON
    assert (kept.users, len(kept.slots), set(kept.rounds())) == (100, 100, {0})


# -- frozen.sh end to end ----------------------------------------------------------------------------------------

def _stub(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _node(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """A fake home with every node tool frozen.sh calls stubbed on its PATH (nvidia-smi, curl, sudo, setsid, uv,
    vLLM's python, and vllm-serve.sh beside the scripts): a server answers once vllm-serve.sh has started it and serves
    the adapters loaded into it; a play writes its test's records after a moment, the held-out test's first play
    stopping short without them; the model and the summary are the package's own."""
    home = tmp_path / "home"
    tools = home / ".local" / "bin"
    tools.mkdir(parents=True)
    scripts = tmp_path / "training" / "scripts"
    shutil.copytree(SCRIPTS, scripts)
    (home / "prime-rl").mkdir()
    (home / "runs").mkdir()
    schedules = tmp_path / "schedules"
    schedules.mkdir()
    for t in frozen.TESTS:   # each test cut to three instances
        d = json.loads(frozen.schedule_path(t).read_text())
        d["slots"], d["users"] = d["slots"][:3], 3
        (schedules / f"arc1-small-{t}-test-seed0.json").write_text(json.dumps(d))
    _stub(tools, "nvidia-smi", "echo 0")
    _stub(tools, "sudo", "exit 0")
    _stub(tools, "setsid", 'exec "$@"')
    _stub(tools, "curl", '''url=""; data=""; while [ $# -gt 0 ]; do case $1 in -d) data=$2; shift ;; localhost:*) url=$1 ;; esac; shift; done
port=${url#localhost:}; port=${port%%/*}
case $url in
    */health) [ -f "$HOME/up-$port" ] ;;
    */v1/models) echo "{\\"data\\": [$(cat "$HOME/loaded-$port" 2>/dev/null)]}" ;;
    */v1/load_lora_adapter) name=$(echo "$data" | sed "s/.*lora_name\\": \\"\\([^\\"]*\\)\\".*/\\1/"); echo "{\\"id\\": \\"$name\\"}" >> "$HOME/loaded-$port"
        echo "$data" >> "$HOME/loads" ;;
esac''')
    (scripts / "vllm-serve.sh").write_text('echo "$@" >> "$HOME/served"; touch "$HOME/up-$3"; '
                                           'echo 1 > "$HOME/runs/vllm-$3.pid"\n')
    (home / "vllm-0.30.0" / "bin").mkdir(parents=True)
    _stub(home / "vllm-0.30.0" / "bin", "python", 'echo "$HOME/adapter.vllm"')
    _stub(tools, "uv", f'''shift 5; cmd=$1
case $cmd in
    model) echo "pg-u1-lag1-p0002 $HOME/runs/rt/pg-u1-lag1/train/p0002/adapter" ;;
    summary) exec {sys.executable} -m continual_arc_training.frozen "$@" ;;
    play) echo "$@" >> "$HOME/plays"
        ref=""; while [ $# -gt 0 ]; do case $1 in --test) t=$2 ;; --out) out=$2 ;; --schedule) s=$2 ;; --row) ref=-reference ;; --no-memory) ref=-weights ;; esac; shift; done
        sleep 2
        if [ "$t" = held-out ] && [ ! -f "$out/held-out-once" ]; then touch "$out/held-out-once"; exit 1; fi
        p=$out/$t$ref/play
        mkdir -p "$p/users/u000/system/state/agent" "$p/users/u000/system/state/sessions"
        echo "{{\\"test\\": \\"$t\\"}}" > "$p/frozen.json"
        n=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['slots']))" "$s")
        for i in $(seq "$n"); do echo "{{\\"solved\\": $([ "$i" = 1 ] && echo true || echo false), \\"cost\\": $i}}"; done > "$p/records.jsonl"
        echo "{{\\"test\\": \\"$t\\", \\"records\\": $n}}" ;;
esac''')
    env = {"PATH": f"{tools}:{os.environ['PATH']}", "HOME": str(home), "POLL": "1", "SCHEDULES": str(schedules)}
    return home, scripts, env


def _frozen_sh(scripts, env, *args, background=False, **extra):
    argv = ["bash", str(scripts / "frozen.sh"), *args]
    if background:
        return subprocess.Popen(argv, env={**env, **extra}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(argv, env={**env, **extra}, capture_output=True, text=True, timeout=300)


def _wait_for(condition, what, seconds=120):
    for _ in range(seconds * 4):
        if condition():
            return
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {what}")


def test_frozen_sh_waits_for_the_stream_then_serves_the_last_phase_and_plays_both_tests(tmp_path):
    """It waits while the run's stream goes on; once realtime.sh has said "done", it serves the run's last phase on two
    servers of its own, one on each half of GPUS at 8120 plus its first GPU, loads the adapter into each, plays both
    tests, plays the held-out test again after its first play stops short, writes the summary, deletes the test users'
    copies of the memory, and leaves done; run again, it does nothing."""
    home, scripts, env = _node(tmp_path)
    run = home / "runs" / "rt" / "pg-u1-lag1"
    (run / "train" / "p0002" / "adapter").mkdir(parents=True)
    (run / "progress").write_text("10-21 01:46 realtime.sh: started: pg, 1 users, 3 rounds\n")
    out = run / "frozen"
    proc = _frozen_sh(scripts, env, "pg-u1-lag1", "6,7", background=True)
    try:
        _wait_for(lambda: "waiting for pg-u1-lag1" in ((out / "progress").read_text() if (out / "progress").exists()
                                                       else ""), "the wait for the stream")
        time.sleep(3)
        assert not (home / "served").exists()   # nothing is served while the stream goes on
        (run / "player.done").touch()
        time.sleep(2)
        assert not (home / "served").exists()   # loops done, but realtime.sh has not said done
        (run / "trainer.done").touch()
        with open(run / "progress", "a") as fp:
            fp.write("10-21 20:00 realtime.sh: done\n")
        assert proc.wait(timeout=180) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    served = (home / "served").read_text().splitlines()
    assert any(line.startswith("Qwen/Qwen3.5-9B 6 8126 ") for line in served)
    assert any(line.startswith("Qwen/Qwen3.5-9B 7 8127 ") for line in served)
    loads = (home / "loads").read_text()
    assert loads.count(f'"lora_name": "pg-u1-lag1-p0002", "lora_path": "{home}/adapter.vllm"') == 2
    progress = (out / "progress").read_text()
    assert "playing held-out, attempt 2" in progress and "playing seen, attempt 2" not in progress
    assert "serving pg-u1-lag1-p0002 on GPUs 6 (:8126, held-out) and 7 (:8127, seen)" in progress
    assert (out / "held-out.done").exists() and (out / "seen.done").exists() and (out / "done").exists()
    tests = json.loads((out / "summary.json").read_text())["tests"]
    assert tests["held-out"]["instances"] == 3 and tests["seen"]["solved"] == 1 and tests["seen"]["mean_cost"] == 2.0
    assert not (out / "held-out" / "play" / "users" / "u000" / "system" / "state").exists()
    assert (out / "held-out" / "play" / "frozen.json").exists()
    plays = (home / "plays").read_text().splitlines()
    assert all(line.startswith(f"play --run {run} --test ") and line.endswith(
        f"--served pg-u1-lag1-p0002 --adapter {run}/train/p0002/adapter") for line in plays) and len(plays) == 3
    before = progress
    assert _frozen_sh(scripts, env, "pg-u1-lag1", "6,7").returncode == 0
    assert (out / "progress").read_text() == before   # a second run once the tests are done does nothing


def test_frozen_sh_plays_a_reference_from_nothing_and_a_runs_weights_alone(tmp_path):
    """A reference from nothing serves the base model and loads nothing; the weights alone (MEMORY=none) play another
    adapter with --no-memory into TEST-weights, which a reference refuses."""
    home, scripts, env = _node(tmp_path)
    assert _frozen_sh(scripts, env, "qwen-f", "6,7", ROW="bare").returncode == 0
    out = home / "runs" / "paper" / "qwen-f" / "frozen"
    assert all((out / name).exists() for name in ("done", "held-out-reference.done", "seen-reference.done",
                                                  "seen-reference/play/records.jsonl"))
    progress = (out / "progress").read_text()
    assert "row bare from nothing" in progress and "serving Qwen/Qwen3.5-9B on GPUs 6" in progress
    assert not (home / "loads").exists() and len((home / "served").read_text().splitlines()) == 2
    doc = json.loads((out / "summary.json").read_text())
    assert doc["run"] is None and doc["tests"]["seen-reference"]["instances"] == 3

    for f in [home / "served", home / "plays", *home.glob("up-*"), *home.glob("loaded-*")]:
        f.unlink(missing_ok=True)
    run = home / "runs" / "rt" / "pg-u1-lag1"
    (run / "play").mkdir(parents=True)
    weights = run / "scratch" / "weights"
    adapter = home / "continual" / "adapter"
    assert _frozen_sh(scripts, env, "pg-u1-lag1", "6,7", MEMORY="none", ADAPTER=str(adapter),
                      SERVED="scratch-pg-u1-lag1-continual", OUT=str(weights)).returncode == 0
    assert all((weights / name).exists() for name in ("done", "held-out-weights.done", "seen-weights.done",
                                                      "seen-weights/play/records.jsonl"))
    progress = (weights / "progress").read_text()
    assert f"its weights alone from an empty memory, into {weights}" in progress
    assert "serving scratch-pg-u1-lag1-continual on GPUs 6 (:8126, held-out-weights) and 7 (:8127, seen-weights)" in \
        progress
    assert (home / "loads").read_text().count('"lora_name": "scratch-pg-u1-lag1-continual"') == 2
    plays = (home / "plays").read_text().splitlines()
    assert len(plays) == 3 and all(line.endswith(f"--adapter {adapter} --no-memory") for line in plays)
    refused = _frozen_sh(scripts, env, "qwen-h0", "6,7", ROW="h0", MEMORY="none")
    assert refused.returncode == 1
    assert "a reference from nothing has no memory to leave out" in \
        (home / "runs" / "paper" / "qwen-h0" / "weights" / "progress").read_text()
