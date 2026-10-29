"""A run's frozen tests after its stream (arc-baselines test, frozen.py).

One instance at a time: every instance from a copy of the state the stream left, at the stream's own path, the
stream's own state set aside meanwhile and given back, played over arc1-smoke's real schedules with FakeSource tasks,
so that none of the pinned ARC data is needed.

As one round of users (--collective): every test user a fork of the commit at the stream's end, its folder's paths
renamed to the user's (in text files, links and a SQLite database's rows, its write-ahead log taken in) and committed
apart from the test, every user's objects in one store reading the run's; a collective run's test users forking its
users' own commits, user i mod U; a run that keeps nothing forking its empty end; a run with no stream starting every
user from nothing; and a test played end to end from the run's own launch, the users solving from the memory the
stream left without asking for a demonstration, their states committed as the next round and their working trees gone,
but a stateless system's users, wiped before their instance, playing it from nothing."""

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
from dataclasses import replace
from functools import partial
from pathlib import Path

import httpx
import pytest
from continual_arc.aliases import TaskAliases
from continual_arc.configs import build_schedule, get_config
from continual_arc.configs import test_schedule as make_test  # not collected as a test
from continual_arc.env import DemosFeedback, RequestDemos, Submit
from continual_arc.runner import run as run_schedule
from continual_arc.schedule import Schedule, Slot, interleave_users, stream_schedule
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

from continual_arc_baselines import frozen
from continual_arc_baselines.cli import main
from continual_arc_baselines.collective import CollectiveHarnessLearner
from continual_arc_baselines.frozen import hold_aside, holding, put_back
from continual_arc_baselines.proxy import PLACEHOLDER_KEY
from continual_arc_baselines.statelog import StateLog, commit_of
from continual_arc_baselines.systems.base import SystemLearner
from continual_arc_baselines.tracking import instance_ended

from .fake_upstream import FakeUpstream
from .test_acp import FakeSystem, obs
from .test_collective import CommittingHarness

CONFIG = "arc1-smoke"
IDS = FakeSource(n_tasks=24).task_ids()
STREAM_POOL, HELD_OUT_POOL = IDS[:12], IDS[12:]
INSTANCES = {"held-out": 8, "seen": 32}   # arc1-smoke's 8 held-out tasks once, and 4 of each of its 8 stream tasks


def fake_arc1() -> TaskRegistry:
    """arc1-smoke's two sources, each serving FakeSource tasks."""
    cfg = get_config(CONFIG)
    return TaskRegistry({s: FakeSource(name=s, n_tasks=len(IDS)) for s in (cfg.stream_source, cfg.epilogue_source)})


def listing(root: Path) -> dict:
    """Every entry under ``root``: a file's mode, modification time and content, a link's target, a directory's mode;
    a socket or a pipe by its kind alone."""
    out = {}
    for folder, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(folder) / name
            st = path.lstat()
            mode, kind = stat.S_IMODE(st.st_mode), stat.S_IFMT(st.st_mode)
            if kind == stat.S_IFLNK:
                out[str(path.relative_to(root))] = ("link", os.readlink(path))
            elif kind == stat.S_IFDIR:
                out[str(path.relative_to(root))] = ("dir", mode)
            elif kind == stat.S_IFREG:
                out[str(path.relative_to(root))] = ("file", mode, st.st_mtime_ns, path.read_bytes())
            else:
                out[str(path.relative_to(root))] = ("special",)
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(repr(sorted(listing(path).items())).encode()).hexdigest()


class FileMemoryHarness(SystemLearner):
    """ShiftMemoryLearner kept as a harness keeps what it learns, in files under its state directory: each task's
    shift, written down once a demonstration shows it, and a note of every instance it played, named by its step, so
    that a test instance's note would change what the next one finds. As an instance begins it logs, outside its state,
    the hash of the state it finds (``starts.jsonl`` in its workdir)."""

    name = "file-memory"

    def start(self):
        pass

    def stop(self):
        pass

    def new_session(self):
        pass

    def wipe_state(self):
        (self.state_dir / "memory.json").unlink(missing_ok=True)
        for note in self.state_dir.glob("notes/*"):
            note.unlink()

    def memory(self) -> dict:
        path = self.state_dir / "memory.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def act(self, obs):
        if obs.is_fresh:
            with (self.workdir / "starts.jsonl").open("a") as log:
                log.write(json.dumps({"step": obs.step, "state": digest(self.state_dir)}) + "\n")
        shift = self.memory().get(obs.task_id)
        if shift is None:
            return RequestDemos()
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs, action, feedback):
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            pair = feedback.pairs[0]
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / "memory.json").write_text(
                json.dumps({**self.memory(), obs.task_id: (pair.output[0][0] - pair.input[0][0]) % 10}))
        if instance_ended(feedback):
            (self.state_dir / "notes").mkdir(parents=True, exist_ok=True)
            (self.state_dir / "notes" / f"{obs.step:04d}").write_text(obs.task_id)


def run_stream(out: Path, *extra: str, learner: str = "tests.test_frozen:FileMemoryHarness") -> Path:
    assert main(["run", "--learner", learner, "--no-proxy", "--registry", "tests.test_frozen:fake_arc1", "--quiet",
                 "--out", str(out), *extra]) == 0
    return out


def system_of(run: Path) -> Path:
    return Path(json.loads((run / "attempt.json").read_text())["attempt_dir"]) / "system"


def starts(play: Path) -> list[dict]:
    attempt = Path(json.loads((play / "attempt.json").read_text())["attempt_dir"])
    return [json.loads(line) for line in (attempt / "system" / "starts.jsonl").read_text().splitlines()]


def records(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]


@pytest.fixture
def frozen_tests(tmp_path, monkeypatch) -> dict[str, Path]:
    """The two tests of arc1-smoke at seed 0, one instance at a time, with build_schedule drawing on the fake pools."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("continual_arc.configs.build_schedule",
                        partial(build_schedule, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL))
    paths = {}
    for test in INSTANCES:
        paths[test] = tmp_path / f"{test}-1.json"
        make_test(CONFIG, 0, test, width=1, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL).save(
            paths[test])
    return paths


def test_both_tests_play_every_instance_from_the_state_the_stream_left(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "runs" / "harness-ws0", "--config", CONFIG, "--no-epilogue")
    assert {r["phase"] for r in records(run)} == {"stream"} and len(records(run)) == 120
    system = system_of(run)
    end = digest(system / "state")
    memory = json.loads((system / "state" / "memory.json").read_text())
    assert len(memory) == 8   # every stream task, written down under the name the run gave it
    for test, n in INSTANCES.items():
        assert main(["test", str(run), "--schedule", str(frozen_tests[test]), "--quiet"]) == 0
        play = run / "frozen" / test / "play"
        assert [r["phase"] for r in records(play)] == [f"{test}-test"] * n
        assert (play / "alias-salt").read_text() == (run / "alias-salt").read_text()
        assert json.loads((play / "frozen.json").read_text()) == {"stream": str(run.resolve()), "instances": 120}
        # every instance found the state the stream left, though each wrote to it, and the stream's own is back
        assert [s["state"] for s in starts(play)] == [end] * n
        assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    # the seen test is solved from the stream's memory under the stream's names, at no cost; the held-out tasks, which
    # the stream never served, each need a demonstration
    assert all(r["solved"] and r["cost"] == 0 for r in records(run / "frozen" / "seen" / "play"))
    assert all(r["solved"] and r["cost"] == 1 for r in records(run / "frozen" / "held-out" / "play"))


def test_a_test_cut_short_goes_on_and_one_played_whole_is_not_played_again(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    end = digest(system_of(run) / "state")
    command = ["test", str(run), "--schedule", str(frozen_tests["held-out"]), "--quiet"]
    assert main(command) == 0
    play = run / "frozen" / "held-out" / "play"
    lines = (play / "records.jsonl").read_text().splitlines()
    (play / "records.jsonl").write_text("\n".join(lines[:3]) + "\n")
    assert main(command) == 0
    assert (play / "records.jsonl").read_text().splitlines()[:3] == lines[:3]
    assert [(r["step"], r["solved"], r["cost"]) for r in records(play)] == [
        (r["step"], r["solved"], r["cost"]) for r in map(json.loads, lines)]
    assert json.loads((play / "meta.json").read_text())["resumed_at"][0]["records"] == 3
    assert [s["state"] for s in starts(play)] == [end] * (8 + 5)   # one attempt, the 5 left played again
    assert main(command) == 0
    assert len(starts(play)) == 13


def test_a_test_that_fails_gives_the_streams_state_back_and_goes_on_when_launched_again(frozen_tests, tmp_path,
                                                                                       monkeypatch):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    system = system_of(run)
    end = digest(system / "state")
    act = FileMemoryHarness.act

    def dies_at_the_fourth(self, obs):
        if obs.is_fresh and obs.step == 3:
            raise RuntimeError("the harness died")
        return act(self, obs)

    monkeypatch.setattr(FileMemoryHarness, "act", dies_at_the_fourth)
    command = ["test", str(run), "--schedule", str(frozen_tests["seen"]), "--quiet"]
    with pytest.raises(RuntimeError, match="the harness died"):
        main(command)
    assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    monkeypatch.undo()
    assert main(command) == 0
    play = run / "frozen" / "seen" / "play"
    assert len(records(play)) == 32 and [s["state"] for s in starts(play)] == [end] * (3 + 29)


def test_a_test_stopped_hard_leaves_the_streams_state_aside_and_the_next_play_takes_it_back(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    system = system_of(run)
    end = digest(system / "state")

    def killed_mid_test():
        """What a test killed during an instance leaves: the stream's state aside, a copy written to in its place."""
        put_back(hold_aside(system, 120), system / "state")
        (system / "state" / "memory.json").write_text("{}")

    killed_mid_test()
    assert main(["test", str(run), "--schedule", str(frozen_tests["held-out"]), "--quiet"]) == 0
    assert [s["state"] for s in starts(run / "frozen" / "held-out" / "play")] == [end] * 8
    assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    killed_mid_test()
    run_stream(run, "--config", CONFIG, "--no-epilogue", "--resume")   # the stream taken up again, with nothing to play
    assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    hold_aside(system, 60)   # set aside at another point of the stream than where it stands
    with pytest.raises(SystemExit, match="has played 120 instances since"):
        run_stream(run, "--config", CONFIG, "--no-epilogue", "--resume")


def test_a_test_of_a_stream_that_went_on_is_refused_and_the_stream_plays_on_from_its_own_state(frozen_tests, tmp_path):
    full = build_schedule(CONFIG, 0, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL,
                          epilogue=False)
    full.save(tmp_path / "full.json")
    full.first_rounds(60).save(tmp_path / "cut.json")
    run = run_stream(tmp_path / "run", "--schedule", str(tmp_path / "cut.json"))
    state = system_of(run) / "state"
    at_60 = digest(state)
    command = ["test", str(run), "--schedule", str(frozen_tests["seen"]), "--quiet"]
    assert main(command) == 0
    assert digest(state) == at_60
    run_stream(run, "--schedule", str(tmp_path / "full.json"), "--resume")
    assert len(records(run)) == 120
    with pytest.raises(SystemExit, match="another stream end"):
        main(command)
    at_120 = digest(state)
    assert main([*command, "--out", str(tmp_path / "seen-at-120")]) == 0
    assert [s["state"] for s in starts(tmp_path / "seen-at-120")] == [at_120] * 32


def test_a_stateless_row_plays_every_test_instance_wiped_and_leaves_the_stream_as_it_was(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue", "--learner-kwargs", '{"stateless": true}')
    state = system_of(run) / "state"
    end = digest(state)
    assert main(["test", str(run), "--schedule", str(frozen_tests["seen"]), "--quiet"]) == 0
    play = run / "frozen" / "seen" / "play"
    assert len({s["state"] for s in starts(play)}) == 1 and starts(play)[0]["state"] != end
    assert all(r["solved"] and r["cost"] == 1 for r in records(play))   # nothing of the stream's memory
    assert digest(state) == end


def test_what_a_harness_cannot_play_is_refused_before_anything_plays(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    wide = tmp_path / "held-out-8.json"
    make_test(CONFIG, 0, "held-out", rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL).save(wide)
    unreset = Schedule.load(frozen_tests["held-out"])
    slots = list(unreset.slots)
    slots[3] = replace(slots[3], reset_before=False)
    replace(unreset, slots=slots).save(tmp_path / "held-out-unreset.json")
    other_seed = tmp_path / "held-out-seed1.json"
    make_test(CONFIG, 1, "held-out", width=1, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL).save(
        other_seed)
    for schedule, refusal in ((wide, "one instance at a time"), (tmp_path / "held-out-unreset.json",
                              "one instance at a time"), (run / "schedule.json", "not a frozen test"),
                              (other_seed, "at seed 1")):
        with pytest.raises(SystemExit, match=refusal):
            main(["test", str(run), "--schedule", str(schedule)])
    launches = (run / "launches.jsonl").read_text()
    with holding(run):   # another process playing the run: its stream, or a test of it
        with pytest.raises(SystemExit, match="another process is playing"):
            main(["test", str(run), "--schedule", str(frozen_tests["seen"])])
        with pytest.raises(SystemExit, match="another process is playing"):
            run_stream(run, "--config", CONFIG, "--no-epilogue", "--resume")
    assert not (run / "frozen").exists() and not list(system_of(run).glob("stream-end*"))
    assert (run / "launches.jsonl").read_text() == launches

    with_epilogue = run_stream(tmp_path / "with-epilogue", "--config", CONFIG)
    held = run_stream(tmp_path / "held", "--config", CONFIG, "--no-epilogue", "--stop-after", "50")
    in_memory = run_stream(tmp_path / "in-memory", "--config", CONFIG, "--no-epilogue",
                           learner="continual_arc.testing:ShiftMemoryLearner")
    for other, refusal in ((with_epilogue, "and then epilogue"), (held, "has played 50 of its stream's 120"),
                           (in_memory, "keeps no state as files")):
        with pytest.raises(SystemExit, match=refusal):
            main(["test", str(other), "--schedule", str(frozen_tests["held-out"])])
        assert not (other / "frozen").exists()
    with pytest.raises(SystemExit, match="--no-epilogue builds the schedule from --config"):
        run_stream(tmp_path / "x", "--schedule", str(frozen_tests["seen"]), "--no-epilogue")


def test_a_frozen_reset_puts_the_saved_state_back_and_a_stateless_one_still_wipes(tmp_path):
    saved = tmp_path / "saved"
    (saved / "notes").mkdir(parents=True)
    (saved / "memory.json").write_text('{"task-1": 3}')
    for stateless, memory in ((False, {"task-1": 3}), (True, {})):
        harness = FileMemoryHarness(stateless=stateless)
        harness.attach(tmp_path / f"work-{stateless}", {}, state_dir=tmp_path / f"state-{stateless}")
        harness.freeze(saved)
        (harness.state_dir / "memory.json").write_text('{"task-2": 5}')
        harness.reset()
        assert harness.memory() == memory
        assert (harness.state_dir / "notes").is_dir()
        assert not (harness.workdir / "state").exists()


def test_an_adapter_frozen_plays_from_the_saved_state_where_it_lies_and_keeps_its_logs_apart(tmp_path):
    saved, state, work = tmp_path / "stream" / "stream-end-5", tmp_path / "stream" / "state", tmp_path / "test"
    (saved / "agent").mkdir(parents=True)
    (saved / "config.json").write_text("{}")
    (saved / "agent" / "kept.txt").write_text("what the stream left")
    system = FakeSystem(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60)
    system.attach(work, {}, state_dir=state)
    try:
        system.freeze(saved)
        assert system.child_env()["HOME"] == str(state / "home") and state in system.sandbox_spec().write_paths
        system.act(obs())
        assert (state / "agent" / "prompt-1.txt").exists() and (state / "agent" / "kept.txt").exists()
        system.reset()
        assert listing(state) == listing(saved)
        assert (work / "fake.log").exists() and not (work / "state").exists()
    finally:
        system.close()


def test_put_back_makes_a_whole_copy_whatever_an_instance_changed(tmp_path):
    saved, state = tmp_path / "saved", tmp_path / "state"
    (saved / "Skills").mkdir(parents=True)
    (saved / "Skills" / "a.py").write_text("print(1)")
    (saved / "Notes.md").write_text("AAAA")
    (saved / "run.sh").write_text("#!/bin/sh\n")
    (saved / "run.sh").chmod(0o755)
    (saved / "link").symlink_to("Skills")
    (saved / "go" / "mod").mkdir(parents=True)
    (saved / "go" / "mod" / "x.go").write_text("package x")
    (saved / "go" / "mod").chmod(0o555)   # a Go module cache's directories are read-only
    os.mkfifo(saved / "pipe")
    copy = {k: v for k, v in listing(saved).items() if v != ("special",)}
    put_back(saved, state)
    assert listing(state) == copy
    (state / "Notes.md").rename(state / "notes.md")   # a name changed in case alone
    (state / "Skills").rename(state / "skills")
    times = (state / "run.sh").stat()
    (state / "run.sh").write_text("#!/bin/X\n")       # the same size, its times put back as tar or cp -p would
    os.utime(state / "run.sh", ns=(times.st_atime_ns, times.st_mtime_ns))
    (state / "run.sh").chmod(0o700)
    (state / "go" / "mod").chmod(0o755)
    (state / "go" / "mod" / "y.go").write_text("package y")
    (state / "go" / "mod").chmod(0o555)
    (state / "link").unlink()
    (state / "link").mkdir()
    os.mkfifo(state / "fifo")
    (state / "hidden").mkdir()
    (state / "hidden" / "notes.md").write_text("kept from the listing")
    (state / "hidden").chmod(0o300)                   # a directory that cannot be listed
    put_back(saved, state)
    assert listing(state) == copy
    for root in (saved, state):   # so that the test's folder can be removed
        (root / "go" / "mod").chmod(0o755)


# -- as one round of users, each forked from the stream's last commit ------------------------------

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
    """A system of rows H0, HPL0 and PL0 (``stateless=True``) behind the harness protocol: it solves a task from the rule
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
            run_schedule(learner, stream, out_dir=run, registry=registry())
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
            result = run_schedule(learner, test, out_dir=out, registry=registry())
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
    argv = ["run", "--learner", "tests.test_frozen:Remembering", "--learner-kwargs", '{"model": "m"}',
            "--llm-defaults", '{"reasoning_effort": "low"}', "--config", "arc1-small", "--out", str(run),
            "--upstream", server.base_url, "--upstream-key-env", "NO_SUCH_KEY", "--registry",
            "tests.test_frozen:registry", "--quiet"]
    (run / "launches.jsonl").write_text(json.dumps({"argv": argv}) + "\n")
    ids = SOURCE.task_ids()
    test = Schedule(name="tiny-seen-test", seed=0, users=4,
                    slots=[Slot(task, 500 + k, "seen-test", user=k) for k, task in enumerate(ids[:2] * 2)],
                    sources={t: SOURCE.name for t in ids}, meta={"test": "seen", "width": 4})
    test.save(tmp_path / "seen.json")
    try:
        out = frozen.play_forks(run, tmp_path / "seen.json")
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
    assert frozen.play_forks(run, tmp_path / "seen.json") == out


def test_a_stateless_runs_test_users_are_wiped_before_their_instance_and_play_it_from_nothing(stream_run, tmp_path):
    """Rows H0, HPL0 and PL0: a stateless system is wiped before every instance, the first a learner plays included, so
    a test user forked from the stream's last commit, which holds what the stream's last instance wrote, plays its test
    instance from nothing, as each of the stream's instances was played: none solves from the rules the commit holds,
    each buys a demonstration, and its test commit holds what its instance wrote and nothing of the stream's."""
    run, _, _ = stream_run
    server = FakeUpstream().start()
    argv = ["run", "--learner", "tests.test_frozen:Stateless", "--learner-kwargs", '{"stateless": true}', "--config",
            "arc1-small", "--out", str(run), "--upstream", server.base_url, "--upstream-key-env", "NO_SUCH_KEY",
            "--registry", "tests.test_frozen:registry", "--quiet"]
    (run / "launches.jsonl").write_text(json.dumps({"argv": argv}) + "\n")
    ids = SOURCE.task_ids()
    test = Schedule(name="tiny-seen-test", seed=0, users=2,
                    slots=[Slot(task, 500 + k, "seen-test", user=k) for k, task in enumerate(ids[:2])],
                    sources={t: SOURCE.name for t in ids}, meta={"test": "seen", "width": 2})
    test.save(tmp_path / "seen.json")
    try:
        out = frozen.play_forks(run, tmp_path / "seen.json")
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
    arc-baselines test --collective forks every test user from the commit of the stream's last round; the stream's own
    state is left as it was. A test's layout says how it is played: one round with --collective, one instance at a time
    without, each refusing the other's."""
    ids = SOURCE.task_ids()
    stream = stream_schedule(ids[:2], 4, seed=0, source=SOURCE.name)
    stream.save(tmp_path / "stream.json")
    test = Schedule(name="tiny-seen-test", seed=0, users=2,
                    slots=[Slot(task, 500 + k, "seen-test", user=k) for k, task in enumerate(ids[:2])],
                    sources={t: SOURCE.name for t in ids}, meta={"test": "seen", "width": 2})
    test.save(tmp_path / "seen.json")
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    assert main(["run", "--learner", "tests.test_frozen:Stateless", "--schedule", str(tmp_path / "stream.json"),
                 "--no-proxy", "--registry", "tests.test_frozen:registry", "--quiet", "--log-state",
                 "--out", str(run)]) == 0
    system = Path(json.loads((run / "attempt.json").read_text())["attempt_dir"]) / "system"
    log = [json.loads(line) for line in (system / "state-log.jsonl").read_text().splitlines()]
    assert [entry["round"] for entry in log] == [0, 1, 2, 3]
    stream_end = git(system / "state.git", "rev-parse", "HEAD").strip()
    frozen.one_at_a_time(test).save(tmp_path / "seen-1.json")
    with pytest.raises(SystemExit, match="is not one round"):
        main(["test", str(run), "--schedule", str(tmp_path / "seen-1.json"), "--collective"])
    with pytest.raises(SystemExit, match="as one round of users with --collective"):
        main(["test", str(run), "--schedule", str(tmp_path / "seen.json")])
    replace(test, meta={**test.meta, "stream_users": 2}).save(tmp_path / "seen-after-two.json")
    with pytest.raises(SystemExit, match="the seen test after 2 users' stream"):   # each user's tasks are their own
        main(["test", str(run), "--schedule", str(tmp_path / "seen-after-two.json"), "--collective"])
    assert not (run / "frozen").exists()
    assert main(["test", str(run), "--schedule", str(tmp_path / "seen.json"), "--collective"]) == 0
    out = run / "frozen" / "seen" / "play"
    assert len((out / "records.jsonl").read_text().splitlines()) == 2
    assert json.loads((out / "frozen.json").read_text())["from"]["systems"][0]["commit"] == log[-1]["commit"]
    for user in range(2):
        repo = out / "users" / f"u{user:03d}" / "system" / "state.git"
        assert git(repo, "log", "--format=%s").splitlines()[:2] == ["round 4", f"relocated from {system}"]
    assert git(system / "state.git", "rev-parse", "HEAD").strip() == stream_end

