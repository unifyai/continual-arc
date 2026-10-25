"""A run's frozen tests (frozen_collective.py): every test user a fork of the commit at the stream's end, its folder's paths
renamed to the user's (in text files, links and a SQLite database's rows, its write-ahead log taken in) and committed
apart from the test, every user's objects in one store reading the run's; a collective run's test users forking its
users' own commits, user i mod U; a run
that keeps nothing forking its empty end; a run with no stream starting every user from nothing; and a test played end to end from the run's own launch, the users
solving from the memory the stream left without asking for a demonstration, their states committed as the next round
and their working trees gone, but a stateless system's users, wiped before their instance, playing it from nothing."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import httpx
import pytest
from continual_arc.aliases import TaskAliases
from continual_arc.env import DemosFeedback, RequestDemos, Submit
from continual_arc.runner import run as play
from continual_arc.schedule import Schedule, Slot, interleave_users, stream_schedule
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

from continual_arc_baselines import frozen_collective as frozen
from continual_arc_baselines.collective import CollectiveHarnessLearner
from continual_arc_baselines.proxy import PLACEHOLDER_KEY
from continual_arc_baselines.statelog import StateLog, commit_of
from continual_arc_baselines.systems.base import SystemLearner

from .fake_upstream import FakeUpstream
from .test_collective import CommittingHarness

SOURCE = FakeSource(n_tasks=3)
SALT = "salt-of-the-run"


def registry() -> TaskRegistry:
    return TaskRegistry({SOURCE.name: SOURCE})


class Remembering:
    """A system that keeps, in its state, each solved task's rule under the name it was shown, solves a task it has
    a rule for without a demonstration, and makes one model call an action through its proxy."""

    def __init__(self, **kwargs) -> None:
        self.kwargs, self.demos = kwargs, []

    def attach(self, workdir: Path, environ: dict[str, str], tracker=None) -> None:
        self.workdir, self.environ = Path(workdir), dict(environ)
        self.state_log = StateLog(self.workdir / "state", self.workdir / "state.git", self.workdir / "state-log.jsonl")

    @property
    def memory_file(self) -> Path:
        return self.workdir / "state" / "memory.json"

    def memory(self) -> dict:
        return json.loads(self.memory_file.read_text()) if self.memory_file.exists() else {}

    def act(self, obs):
        httpx.post(self.environ["ARC_LLM_BASE_URL"] + "/chat/completions", timeout=10,
                   headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"},
                   json={"model": "m", "messages": [{"role": "user", "content": obs.task_id}]})
        shift = self.memory().get(obs.task_id)
        if shift is None and self.demos:
            shift = (self.demos[0].output[0][0] - self.demos[0].input[0][0]) % 10
        if shift is None:
            return RequestDemos()
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs, action, feedback) -> None:
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            self.demos = list(feedback.pairs)
        if getattr(feedback, "correct", False) and self.demos:
            mem = self.memory()
            mem[obs.task_id] = (self.demos[0].output[0][0] - self.demos[0].input[0][0]) % 10
            self.memory_file.write_text(json.dumps(mem))
            self.demos = []

    def end_round(self) -> None:
        self.state_log.commit()

    def reset(self) -> None:
        pass

    def resources(self) -> dict[str, float]:
        return {}

    def close(self) -> None:
        pass


class Stateless(SystemLearner):
    """A system of rows H0, HPL0 and M0 (``stateless=True``) behind the harness protocol: it solves a task from the rule
    its state's ``memory.json`` holds for it, else from a demonstration it asks for, and each time it starts notes what
    its state held (``found.json``)."""

    name = "stateless"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.obs, self.demos = None, []

    def start(self) -> None:
        held = sorted(str(path.relative_to(self.state_dir)) for path in self.state_dir.rglob("*"))
        (self.state_dir / "found.json").write_text(json.dumps(held))

    def stop(self) -> None:
        pass

    def new_session(self) -> None:
        pass

    def wipe_state(self) -> None:
        for path in self.state_dir.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()

    def act(self, obs):
        self.obs = obs
        return super().act(obs)

    def observe(self, obs, action, feedback) -> None:
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            self.demos = list(feedback.pairs)
        super().observe(obs, action, feedback)

    def ask(self, message: str, *, reprompt: bool = False) -> str:
        memory = self.state_dir / "memory.json"
        shift = json.loads(memory.read_text()).get(self.obs.task_id) if memory.exists() else None
        if shift is None and self.demos:
            shift = (self.demos[0].output[0][0] - self.demos[0].input[0][0]) % 10
        if shift is None:
            return '{"action": "request_demos"}'
        return json.dumps({"action": "submit", "grid": [[(c + shift) % 10 for c in row] for row in self.obs.input]})


def git(repo: Path, *args: str, tree: Path | None = None) -> str:
    work = [f"--work-tree={tree}"] if tree is not None else []
    return subprocess.run(["git", f"--git-dir={repo}", *work, *args], check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def stream_run(tmp_path):
    """A one-user run of three stream rounds and one epilogue round, its system's state committed after each: the
    rules of two tasks by the run's names for them, a script and a link that name the system's own folder, and a
    database in write-ahead mode whose rows name it, beside a full-text index of them."""
    system = tmp_path / ".workbench" / "runs" / "adhoc-remembering-1" / "attempts" / "a1" / "system"
    log = StateLog(system / "state", system / "state.git", system / "state-log.jsonl")
    aliases = TaskAliases(salt=SALT)
    ids = SOURCE.task_ids()
    for k in range(4):
        rules = {aliases.alias(task): SOURCE.shift(task) for task in ids[:2]} if k >= 2 else {}
        (system / "state" / "memory.json").write_text(json.dumps(rules))
        (system / "state" / "bin").mkdir(exist_ok=True)
        (system / "state" / "bin" / "tool").write_text(f"#!{system}/state/venv/bin/python\nround {k}\n")
        if not (system / "state" / "here").is_symlink():
            (system / "state" / "here").symlink_to(f"{system}/state/bin")
            db = sqlite3.connect(system / "state" / "agent.sqlite")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, turns INTEGER)")
            db.execute("CREATE VIRTUAL TABLE notes USING fts5(body)")
            db.execute("INSERT INTO sessions VALUES (?, ?, 3), ('s2', ?, 1)",
                       ("s1", f"{system}/state/workspace", f"{system}2/elsewhere"))
            db.execute("INSERT INTO notes VALUES (?)", (f"ran in {system}/state/workspace",))
            db.commit()   # left open: its rows stay in the write-ahead log, which the commit takes with the file
        log.commit()
    return run_folder(tmp_path, system), system, log


def run_folder(tmp_path: Path, system: Path) -> Path:
    """The folder of a one-user run played by the system in ``system``: three stream rounds and one epilogue round."""
    ids = SOURCE.task_ids()
    run = tmp_path / "run"
    run.mkdir()
    slots = [Slot(task, 100 + k, "stream" if k < 3 else "epilogue") for k, task in enumerate(ids[:2] + ids[:2])]
    Schedule(name="tiny", seed=0, slots=slots, sources={t: SOURCE.name for t in ids}).save(run / "schedule.json")
    (run / "records.jsonl").write_text("".join(json.dumps({"phase": s.phase}) + "\n" for s in slots))
    (run / "alias-salt").write_text(SALT)
    (run / "attempt.json").write_text(json.dumps({"attempt_dir": str(system.parent)}))
    return run


def test_every_test_user_is_a_fork_of_the_streams_last_commit_in_a_folder_of_its_own(stream_run, tmp_path):
    run, system, log = stream_run
    out = tmp_path / "test"
    started = frozen.seed(run, out, 3)
    assert started["from"] == {"round": 2, "systems": [{"system": str(system), "commit": log.commit_of(2),
                                                       "relocated": 2, "databases": 1}]}
    assert (out / "alias-salt").read_text() == SALT
    for user in range(3):
        mine = out / "users" / f"u{user:03d}" / "system"
        assert (mine / "state" / "bin" / "tool").read_text() == f"#!{mine}/state/venv/bin/python\nround 2\n"
        assert os.readlink(mine / "state" / "here") == f"{mine}/state/bin"
        db = sqlite3.connect(mine / "state" / "agent.sqlite")
        assert db.execute("SELECT id, cwd FROM sessions ORDER BY id").fetchall() == [
            ("s1", f"{mine}/state/workspace"), ("s2", f"{system}2/elsewhere")]
        assert db.execute("SELECT body FROM notes").fetchall() == [(f"ran in {system}/state/workspace",)]
        db.close()
        # the relocation is a commit of its own on the stream's last, and the log runs to the stream's end
        assert git(mine / "state.git", "log", "--format=%s").splitlines() == [
            f"relocated from {system}", "round 2", "round 1", "round 0"]
        assert git(mine / "state.git", "rev-parse", "HEAD~1").strip() == log.commit_of(2)
        assert [json.loads(line)["round"] for line in (mine / "state-log.jsonl").read_text().splitlines()] == [0, 1, 2]
        assert os.readlink(mine / "state.git" / "objects") == "../../../../objects.git/objects"   # relative, as the store's
        assert (mine / "state.git" / "objects").resolve() == (out / "objects.git" / "objects").resolve()
        assert not git(mine / "state.git", "status", "--porcelain", tree=mine / "state")
    alternate = (out / "objects.git" / "objects" / "info" / "alternates").read_text().strip()
    assert not alternate.startswith("/") and (out / "objects.git" / "objects" / alternate).resolve() == (
        system / "state.git" / "objects").resolve()
    # the run's own state is as the stream's last round left it, and its repository has nothing new
    assert (system / "state" / "bin" / "tool").read_text() == f"#!{system}/state/venv/bin/python\nround 3\n"
    assert git(system / "state.git", "rev-parse", "HEAD").strip() == log.commit_of(3)


def test_a_collective_runs_test_user_i_forks_its_user_i_mod_u_and_commits_its_instance_on_top(tmp_path):
    """A run of two users (CollectiveHarnessLearner, which commits every user's state as each round ends): test user i
    starts from the run's user i mod 2 at the stream's end, that user's commit relocated to the test user's folder,
    its log that user's rounds, and solves its task from that user's memory; the test round's commit is what the test
    user wrote in its instance, and the run's users' histories are as the stream left them."""
    ids = SOURCE.task_ids()
    stream = interleave_users([stream_schedule(ids, 3, seed=u, source=SOURCE.name) for u in range(2)], seed=0,
                              name="team")
    run = tmp_path / "run"
    server = FakeUpstream().start()
    try:
        learner = CollectiveHarnessLearner(CommittingHarness, run, schedule=stream, upstream=server.base_url)
        try:
            play(learner, stream, out_dir=run, registry=registry())
        finally:
            learner.close()
        systems = [run / "users" / f"u{u:03d}" / "system" for u in range(2)]
        ends = [commit_of(system / "state-log.jsonl", 2) for system in systems]
        out = tmp_path / "test"
        started = frozen.seed(run, out, 3)
        tasks = [next(s.task_id for s in stream.slots if s.user == user % 2) for user in range(3)]
        test = Schedule(name="team-seen-test", seed=0, users=3, sources={t: SOURCE.name for t in ids},
                        slots=[Slot(task, 500 + user, "seen-test", user=user) for user, task in enumerate(tasks)],
                        meta={"test": "seen", "width": 3})
        learner = CollectiveHarnessLearner(CommittingHarness, out, schedule=test, upstream=server.base_url)
        try:
            result = play(learner, test, out_dir=out, registry=registry())
        finally:
            learner.close()
    finally:
        server.stop()
    assert started["from"] == {"round": 2, "systems": [{"system": str(system), "commit": end, "relocated": 1,
                                                       "databases": 0} for system, end in zip(systems, ends, strict=True)]}
    assert [(r.user, r.solved, r.cost) for r in result.records] == [(user, True, 0.0) for user in range(3)]
    aliases = TaskAliases.for_run(run)
    for user, task in enumerate(tasks):
        mine, source = out / "users" / f"u{user:03d}" / "system", systems[user % 2]
        assert git(mine / "state.git", "log", "--format=%s").splitlines() == [
            "round 3", f"relocated from {source}", "round 2", "round 1", "round 0"]
        assert git(mine / "state.git", "rev-parse", "HEAD~2").strip() == ends[user % 2]
        assert [json.loads(line)["round"] for line in (mine / "state-log.jsonl").read_text().splitlines()] == [0, 1, 2, 3]
        assert (mine / "state" / "bin" / "tool").read_text() == f"#!{mine}/state/venv/bin/python\n"
        assert git(mine / "state.git", "diff", "--name-only", "HEAD~1", "HEAD").split() == ["notes.md"]
        assert f"+user {user} solved {aliases.alias(task)} in round 0\n" in git(mine / "state.git", "diff", "HEAD~1",
                                                                                "HEAD")
    alternates = (out / "objects.git" / "objects" / "info" / "alternates").read_text().splitlines()
    assert [(out / "objects.git" / "objects" / line).resolve() for line in alternates] == [
        (system / "state.git" / "objects").resolve() for system in systems]
    assert [git(system / "state.git", "rev-parse", "HEAD").strip() for system in systems] == ends


def test_a_run_that_keeps_nothing_forks_its_empty_end(tmp_path):
    """A floor's every round commits the empty tree, which its test users fork as any run's users fork its last."""
    system = tmp_path / "attempt" / "system"
    log = StateLog(system / "state", system / "state.git", system / "state-log.jsonl")
    for _ in range(4):
        log.commit()
    out = tmp_path / "test"
    frozen.seed(run_folder(tmp_path, system), out, 2)
    for user in range(2):
        mine = out / "users" / f"u{user:03d}" / "system"
        assert (mine / "state").is_dir() and not any((mine / "state").iterdir())
        assert git(mine / "state.git", "log", "--format=%s").splitlines() == [
            f"relocated from {system}", "round 2", "round 1", "round 0"]


def test_a_run_with_no_stream_starts_every_user_from_nothing(stream_run, tmp_path):
    run, _, _ = stream_run
    epilogue = Schedule.load(run / "schedule.json")
    epilogue.slots = epilogue.slots[3:]
    epilogue.save(run / "schedule.json")
    out = tmp_path / "test"
    assert frozen.seed(run, out, 2) == {"from": None}
    for user in range(2):
        mine = out / "users" / f"u{user:03d}" / "system"
        assert not (mine / "state").exists() and not (mine / "state-log.jsonl").exists()
        assert (mine / "state.git" / "objects").is_symlink()


def test_a_stream_still_being_played_is_not_tested(stream_run, tmp_path):
    run, _, _ = stream_run
    lines = (run / "records.jsonl").read_text().splitlines()
    (run / "records.jsonl").write_text("\n".join(lines[:2]) + "\n")
    with pytest.raises(SystemExit, match="2 of its stream's 3"):
        frozen.seed(run, tmp_path / "test", 2)


def test_a_test_is_played_from_the_runs_own_launch_every_user_from_the_streams_memory(stream_run, tmp_path):
    run, system, log = stream_run
    server = FakeUpstream().start()
    argv = ["run", "--learner", "tests.test_frozen_collective:Remembering", "--learner-kwargs", '{"model": "m"}',
            "--llm-defaults", '{"reasoning_effort": "low"}', "--config", "arc1-small", "--out", str(run),
            "--upstream", server.base_url, "--upstream-key-env", "NO_SUCH_KEY", "--registry",
            "tests.test_frozen_collective:registry", "--quiet"]
    (run / "launches.jsonl").write_text(json.dumps({"argv": argv}) + "\n")
    ids = SOURCE.task_ids()
    test = Schedule(name="tiny-seen-test", seed=0, users=4,
                    slots=[Slot(task, 500 + k, "seen-test", user=k) for k, task in enumerate(ids[:2] * 2)],
                    sources={t: SOURCE.name for t in ids}, meta={"test": "seen", "width": 4})
    test.save(tmp_path / "seen.json")
    try:
        out = frozen.test(run, tmp_path / "seen.json")
    finally:
        server.stop()
    assert out == run / "frozen" / "seen" / "play"
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    # every user remembered its task from the stream: solved at once, no demonstration bought
    assert [(r["user"], r["phase"], r["solved"], r["cost"]) for r in records] == [(u, "seen-test", True, 0.0)
                                                                                for u in range(4)]
    bodies = [r["body"] for r in server.requests if r["body"]]
    assert bodies and all(b["reasoning_effort"] == "low" and b["session_id"] == "run-seen" for b in bodies)
    for user in range(4):
        mine = out / "users" / f"u{user:03d}" / "system"
        assert not (mine / "state").exists()   # its working tree gone, its state in its repository
        assert git(mine / "state.git", "log", "--format=%s").splitlines()[:2] == ["round 3", f"relocated from {system}"]
        assert json.loads((mine / "state-log.jsonl").read_text().splitlines()[-1])["round"] == 3
    assert json.loads((out / "frozen.json").read_text())["from"]["systems"][0]["commit"] == log.commit_of(2)
    # a second call finds the round recorded and plays nothing
    assert frozen.test(run, tmp_path / "seen.json") == out


def test_a_stateless_runs_test_users_are_wiped_before_their_instance_and_play_it_from_nothing(stream_run, tmp_path):
    """Rows H0, HPL0 and M0: a stateless system is wiped before every instance, the first a learner plays included, so
    a test user forked from the stream's last commit, which holds what the stream's last instance wrote, plays its test
    instance from nothing, as each of the stream's instances was played: none solves from the rules the commit holds,
    each buys a demonstration, and its test commit holds what its instance wrote and nothing of the stream's."""
    run, _, _ = stream_run
    server = FakeUpstream().start()
    argv = ["run", "--learner", "tests.test_frozen_collective:Stateless", "--learner-kwargs", '{"stateless": true}', "--config",
            "arc1-small", "--out", str(run), "--upstream", server.base_url, "--upstream-key-env", "NO_SUCH_KEY",
            "--registry", "tests.test_frozen_collective:registry", "--quiet"]
    (run / "launches.jsonl").write_text(json.dumps({"argv": argv}) + "\n")
    ids = SOURCE.task_ids()
    test = Schedule(name="tiny-seen-test", seed=0, users=2,
                    slots=[Slot(task, 500 + k, "seen-test", user=k) for k, task in enumerate(ids[:2])],
                    sources={t: SOURCE.name for t in ids}, meta={"test": "seen", "width": 2})
    test.save(tmp_path / "seen.json")
    try:
        out = frozen.test(run, tmp_path / "seen.json")
    finally:
        server.stop()
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert [(r["user"], r["solved"], r["cost"], r["resources"]["system_wipes"]) for r in records] == [
        (u, True, 1.0, 1.0) for u in range(2)]
    for user in range(2):
        repo = out / "users" / f"u{user:03d}" / "system" / "state.git"
        assert json.loads(git(repo, "show", "HEAD:found.json")) == []
        assert git(repo, "ls-tree", "-r", "--name-only", "HEAD").split() == ["found.json"]


def test_a_logged_runs_test_is_played_as_a_round_of_users_from_the_command_line(tmp_path, monkeypatch):
    """arc-baselines run --log-state commits a harness's state at the end of every round of its stream, and
    arc-baselines test-collective forks every test user from the commit of the stream's last round; the stream's own
    state is left as it was."""
    from continual_arc_baselines.cli import main

    ids = SOURCE.task_ids()
    stream = stream_schedule(ids[:2], 4, seed=0, source=SOURCE.name)
    stream.save(tmp_path / "stream.json")
    test = Schedule(name="tiny-seen-test", seed=0, users=2,
                    slots=[Slot(task, 500 + k, "seen-test", user=k) for k, task in enumerate(ids[:2])],
                    sources={t: SOURCE.name for t in ids}, meta={"test": "seen", "width": 2})
    test.save(tmp_path / "seen.json")
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    assert main(["run", "--learner", "tests.test_frozen_collective:Stateless", "--schedule", str(tmp_path / "stream.json"),
                 "--no-proxy", "--registry", "tests.test_frozen_collective:registry", "--quiet", "--log-state",
                 "--out", str(run)]) == 0
    system = Path(json.loads((run / "attempt.json").read_text())["attempt_dir"]) / "system"
    log = [json.loads(line) for line in (system / "state-log.jsonl").read_text().splitlines()]
    assert [entry["round"] for entry in log] == [0, 1, 2, 3]
    stream_end = git(system / "state.git", "rev-parse", "HEAD").strip()
    assert main(["test-collective", str(run), "--schedule", str(tmp_path / "seen.json")]) == 0
    out = run / "frozen" / "seen" / "play"
    assert len((out / "records.jsonl").read_text().splitlines()) == 2
    assert json.loads((out / "frozen.json").read_text())["from"]["systems"][0]["commit"] == log[-1]["commit"]
    for user in range(2):
        repo = out / "users" / f"u{user:03d}" / "system" / "state.git"
        assert git(repo, "log", "--format=%s").splitlines()[:2] == ["round 4", f"relocated from {system}"]
    assert git(system / "state.git", "rev-parse", "HEAD").strip() == stream_end

