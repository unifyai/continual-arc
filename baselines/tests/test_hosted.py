"""A submitted learner run whole inside the sandbox (systems/hosted.py, systems/learner_host.py): the protocol end to
end through the CLI and the proxy, the learner's prints kept off the channel, its exceptions raised in the run, its
reach confined, and its install run under the sandbox too."""

import json
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import continual_arc
import pytest
from continual_arc.schedule import stream_schedule
from continual_arc.testing import FakeSource
from continual_workbench.context import create_context

from continual_arc_baselines.cli import main
from continual_arc_baselines.proxy import PLACEHOLDER_KEY
from continual_arc_baselines.systems.hosted import (
    HostedLearner,
    HostedLearnerError,
    brings_benchmark,
    install_submission,
)

REPO = Path(__file__).resolve().parents[1]
SECRET = "sk-upstream-secret"
BACKENDS = [
    "none",
    pytest.param("seatbelt", marks=pytest.mark.skipif(platform.system() != "Darwin", reason="seatbelt is macOS only")),
    pytest.param("bwrap", marks=pytest.mark.skipif(platform.system() != "Linux" or shutil.which("bwrap") is None,
                                                   reason="bubblewrap on Linux only")),
]

# The toy submission: FakeSource's shift remembered per task, noisy on every channel a learner has, and a probe of
# what it can reach (probe.json beside it, written by each test).
TOY = '''
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

from continual_arc.env import DemosFeedback, RequestDemos, Submit
from continual_arc.learner import BaseLearner

PROBE = json.loads((Path(__file__).resolve().parent / "probe.json").read_text())


def attempt(fn):
    try:
        fn()
        return "ok"
    except OSError as error:
        return f"failed: {error}"


class Learner(BaseLearner):
    def __init__(self):
        print("noise from __init__")
        if PROBE.get("fail_on_init"):
            raise RuntimeError("cannot start")
        self.memory, self.pending, self.calls = {}, None, 0

    def probe(self):
        out = {"env": dict(os.environ), "cwd": os.getcwd()}
        if PROBE.get("read"):
            out["read"] = attempt(lambda: Path(PROBE["read"]).read_text())
        if PROBE.get("connect"):
            out["connect"] = attempt(lambda: socket.create_connection(("127.0.0.1", PROBE["connect"]), timeout=5).close())
        body = json.dumps({"model": "vendor/model", "messages": [{"role": "user", "content": "hi"}]}).encode()
        request = urllib.request.Request(os.environ["OPENAI_BASE_URL"] + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
        with urllib.request.urlopen(request, timeout=30) as reply:
            out["call"] = reply.status
        Path("probe-result.json").write_text(json.dumps(out))  # the working directory: the state directory

    def act(self, obs):
        self.calls += 1
        print(f"act on {obs.task_id}")
        os.write(1, b"a raw write to file descriptor 1\\n")
        subprocess.run([sys.executable, "-c", "print('a child process prints too')"], check=True)
        if PROBE.get("fail_on_act"):
            raise ValueError("boom from act")
        if PROBE.get("hang_on_act"):
            import time
            time.sleep(600)
        if self.calls == 1:
            self.probe()
        shift = self.memory.get(obs.task_id)
        if shift is None:
            if self.pending is None:
                return RequestDemos()
            pair = self.pending[0]
            shift = self.memory[obs.task_id] = (pair.output[0][0] - pair.input[0][0]) % 10
            self.pending = None
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs, action, feedback):
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            self.pending = list(feedback.pairs)

    def resources(self):
        return {"calls": float(self.calls), "label": "not a number"}
'''


@pytest.fixture
def upstream():
    from .fake_upstream import FakeUpstream

    server = FakeUpstream().start()
    yield server
    server.stop()


def toy_package(root: Path, probe: dict) -> Path:
    package = root / "toy_submission"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(TOY)
    (package / "probe.json").write_text(json.dumps(probe))
    return root


def run_hosted(tmp_path, monkeypatch, upstream, kwargs: dict, n_slots: int = 6):
    source = FakeSource(n_tasks=3)
    tmp_path.mkdir(parents=True, exist_ok=True)
    sched = tmp_path / "schedule.json"
    stream_schedule(source.task_ids(), n_slots, seed=1, source=source.name).save(sched)
    context = create_context(tmp_path / "wb", "hosted", {"experiment_id": "t", "project_key": "t"})
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.setenv("ARC_TEST_SECRET", SECRET)  # in this process's environment, never in the learner's
    code = main(["run", "--learner", "continual_arc_baselines.systems.hosted:HostedLearner",
                 "--learner-kwargs", json.dumps(kwargs), "--schedule", str(sched),
                 "--registry", "continual_arc_baselines.testing:fake_registry",
                 "--upstream", upstream.base_url, "--quiet"])
    assert code == 0
    system = context.directory / "system"
    records = [json.loads(line) for line in (context.directory / "continual-arc" / "records.jsonl").read_text().splitlines()]
    return system, records


def assert_learned(records):
    assert all(r["solved"] for r in records)
    first = [r for r in records if r["exposure_index"] == 0]
    later = [r for r in records if r["exposure_index"] > 0]
    assert first and all(r["cost"] == 1.0 for r in first)  # one demo request
    assert later and all(r["cost"] == 0.0 for r in later)   # remembered in the host, across instances


def assert_confined(system: Path, upstream, records) -> dict:
    probe = json.loads((system / "state" / "probe-result.json").read_text())
    assert Path(probe["cwd"]).resolve() == (system / "state").resolve()
    env = probe["env"]
    assert SECRET not in json.dumps(env) and "ARC_TEST_SECRET" not in env
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ARC_LLM_API_KEY"):
        assert env[name] == PLACEHOLDER_KEY
    assert env["OPENAI_BASE_URL"] == env["OPENROUTER_BASE_URL"] == env["ARC_LLM_BASE_URL"]
    assert env["OPENAI_BASE_URL"].startswith("http://127.0.0.1:")
    # the call went through the proxy, which put the real key on it, and the run measured it
    assert probe["call"] == 200
    assert [r["authorization"] for r in upstream.requests] == [f"Bearer {SECRET}"]
    assert records[0]["resources"]["llm_requests"] == 1.0
    assert sum(r["resources"]["calls"] for r in records) == sum(r["resources"]["actions"] for r in records)
    assert "label" not in records[0]["resources"]  # a counter that is not a number is dropped
    return probe


def test_a_hosted_learner_learns_and_its_prints_stay_off_the_channel(tmp_path, monkeypatch, upstream):
    code = toy_package(tmp_path / "code", {})
    system, records = run_hosted(tmp_path, monkeypatch, upstream, {
        "python": sys.executable, "learner": "toy_submission:Learner", "code": str(code), "sandbox": "none"})
    assert_learned(records)
    assert_confined(system, upstream, records)
    log = (system / "learner.log").read_text()
    for noise in ("noise from __init__", "act on task-", "a raw write to file descriptor 1", "a child process prints too"):
        assert noise in log
    assert "[hosted] started toy_submission.Learner" in log


def test_the_learners_exception_fails_the_run(tmp_path, monkeypatch, upstream):
    code = toy_package(tmp_path / "code", {"fail_on_act": True})
    with pytest.raises(HostedLearnerError, match="boom from act") as raised:
        run_hosted(tmp_path, monkeypatch, upstream, {
            "python": sys.executable, "learner": "toy_submission:Learner", "code": str(code), "sandbox": "none"})
    assert "Traceback" in str(raised.value) and "ValueError" in str(raised.value)


def test_a_learner_that_does_not_answer_fails_the_run(tmp_path, monkeypatch, upstream):
    code = toy_package(tmp_path / "code", {"hang_on_act": True})
    with pytest.raises(HostedLearnerError, match="did not answer act within 2s"):
        run_hosted(tmp_path, monkeypatch, upstream, {
            "python": sys.executable, "learner": "toy_submission:Learner", "code": str(code), "sandbox": "none",
            "timeout_s": 2})


def test_a_learner_that_cannot_start_fails_before_the_run(tmp_path, monkeypatch, upstream):
    code = toy_package(tmp_path / "code", {"fail_on_init": True})
    with pytest.raises(HostedLearnerError, match="cannot start"):
        run_hosted(tmp_path, monkeypatch, upstream, {
            "python": sys.executable, "learner": "toy_submission:Learner", "code": str(code), "sandbox": "none"})
    with pytest.raises(HostedLearnerError, match="has no attribute"):
        run_hosted(tmp_path / "again", monkeypatch, upstream, {
            "python": sys.executable, "learner": "toy_submission:Missing", "code": str(code), "sandbox": "none"})


def test_a_hosted_learner_plays_its_test_from_the_memory_its_stream_left(tmp_path, monkeypatch, upstream):
    """The memory a hosted learner keeps in its own process has no snapshot, so its frozen test follows the stream in
    the same launch (``--test``): every test instance, one round of a user each, is solved from what the stream
    taught it, under the stream's names, with no demonstration bought."""
    from continual_arc.schedule import Schedule, Slot

    code = toy_package(tmp_path / "code", {})
    source = FakeSource(n_tasks=3)
    stream_schedule(source.task_ids(), 6, seed=1, source=source.name).save(tmp_path / "schedule.json")
    Schedule(name="toy-seen-test", seed=1, users=3, sources={t: source.name for t in source.task_ids()},
             slots=[Slot(t, 700 + u, "seen-test", user=u) for u, t in enumerate(source.task_ids())],
             meta={"test": "seen", "width": 3}).save(tmp_path / "test.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    out = tmp_path / "run"
    assert main(["run", "--learner", "continual_arc_baselines.systems.hosted:HostedLearner",
                 "--learner-kwargs", json.dumps({"python": sys.executable, "learner": "toy_submission:Learner",
                                                 "code": str(code), "sandbox": "none"}),
                 "--schedule", str(tmp_path / "schedule.json"), "--test", str(tmp_path / "test.json"),
                 "--registry", "continual_arc_baselines.testing:fake_registry", "--upstream", upstream.base_url,
                 "--out", str(out), "--quiet"]) == 0
    stream = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert_learned(stream)
    tested = [json.loads(line) for line in (out / "frozen" / "seen" / "play" / "records.jsonl").read_text().splitlines()]
    assert [(r["user"], r["phase"], r["solved"], r["cost"]) for r in tested] == [(u, "seen-test", True, 0.0)
                                                                                for u in range(3)]


# A learner that keeps what it learns as a harness does, in files: FakeSource's shift in its working directory, read
# afresh on every action, a note of every instance it plays, and a count of instances in its HOME. As an instance
# begins it prints what it finds there, and as it ends what it leaves.
FILE_TOY = '''
import hashlib
import json
import os
from pathlib import Path

from continual_arc.env import DemosFeedback, RequestDemos, Submit
from continual_arc.learner import BaseLearner


def digest(root):
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode() + b"\\0" + path.read_bytes() + b"\\0")
    return h.hexdigest()


class FileLearner(BaseLearner):
    state_in_files = True

    def __init__(self):
        self.workdir, self.pending, self.seen, self.round = None, None, 0, None

    def begin_round(self, n):
        self.round = n

    def end_round(self):
        self.round = None

    def attach(self, workdir, environ):
        self.workdir = workdir
        (workdir / "attached.json").write_text(json.dumps({"workdir": str(workdir), "cwd": os.getcwd(),
                                                           "base_url": environ.get("OPENAI_BASE_URL")}))

    def memory(self):
        path = self.workdir / "memory.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def act(self, obs):
        if obs.is_fresh:
            count = Path.home() / "count"
            self.seen = int(count.read_text()) if count.exists() else 0
            count.write_text(str(self.seen + 1))
            print("START " + json.dumps({"home": self.seen, "state": digest(self.workdir), "round": self.round}),
                  flush=True)
        shift = self.memory().get(obs.task_id)
        if shift is None:
            if self.pending is None:
                return RequestDemos()
            pair = self.pending[0]
            shift = (pair.output[0][0] - pair.input[0][0]) % 10
            (self.workdir / "memory.json").write_text(json.dumps({**self.memory(), obs.task_id: shift}))
            self.pending = None
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs, action, feedback):
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            self.pending = list(feedback.pairs)
        if getattr(feedback, "correct", False):
            (self.workdir / "notes").mkdir(exist_ok=True)
            (self.workdir / "notes" / f"{self.seen:04d}").write_text(obs.task_id)
            print("END " + json.dumps({"state": digest(self.workdir)}), flush=True)

    def reset(self):
        (self.workdir / "memory.json").unlink(missing_ok=True)
'''


def file_toy_package(root: Path) -> Path:
    (root / "file_toy").mkdir(parents=True)
    (root / "file_toy" / "__init__.py").write_text(FILE_TOY)
    return root


def printed(log: Path, kind: str) -> list[dict]:
    return [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines() if line.startswith(kind + " ")]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_learner_that_keeps_its_state_in_files_plays_its_test_one_instance_at_a_time_from_its_streams_end(
        tmp_path, monkeypatch, upstream, backend):
    """A hosted learner that says it keeps everything it learns as files, as a harness does, is frozen at the end of
    its stream (``--test``): it plays the test's instances one at a time, and before every instance its host is ended
    and its working directory and HOME are put back as the stream left them. Every test instance finds the stream's
    files, though each writes to them, and solves from the stream's memory under the stream's names; at the end the
    stream's own files are back in their place."""
    from continual_arc.schedule import Schedule, Slot

    code = file_toy_package(tmp_path / "code")
    source = FakeSource(n_tasks=3)
    stream_schedule(source.task_ids(), 6, seed=1, source=source.name).save(tmp_path / "schedule.json")
    test = Schedule(name="toy-seen-test", seed=1, users=3, sources={t: source.name for t in source.task_ids()},
                    slots=[Slot(t, 700 + u, "seen-test", user=u) for u, t in enumerate(source.task_ids())],
                    meta={"test": "seen", "width": 3})
    test.save(tmp_path / "test.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    python = sys.executable if backend == "none" else submission_python(tmp_path)
    out = tmp_path / "run"
    assert main(["run", "--learner", "continual_arc_baselines.systems.hosted:HostedLearner",
                 "--learner-kwargs", json.dumps({"python": python, "learner": "file_toy:FileLearner",
                                                 "code": str(code), "sandbox": backend}),
                 "--schedule", str(tmp_path / "schedule.json"), "--test", str(tmp_path / "test.json"),
                 "--registry", "continual_arc_baselines.testing:fake_registry", "--upstream", upstream.base_url,
                 "--out", str(out), "--quiet"]) == 0
    assert_learned([json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()])
    play = out / "frozen" / "seen" / "play"
    tested = [json.loads(line) for line in (play / "records.jsonl").read_text().splitlines()]
    # the test's instances in its order, one at a time, each solved from the stream's memory with nothing bought
    assert [(r["task_id"], r["instance_seed"]) for r in tested] == [(s.task_id, s.instance_seed) for s in test.slots]
    assert [(r["user"], r["reset_before"], r["solved"], r["cost"]) for r in tested] == [(0, True, True, 0.0)] * 3
    assert Schedule.load(play / "schedule.json").users == 1

    system = Path(json.loads((out / "attempt.json").read_text())["attempt_dir"]) / "system"
    starts, ends = printed(system / "learner.log", "START"), printed(system / "learner.log", "END")
    assert len(starts) == len(ends) == 6 + 3
    assert len({s["state"] for s in starts[:6]}) == 6                    # the stream learns
    assert {s["state"] for s in starts[6:]} == {ends[5]["state"]}         # every test instance finds the stream's end
    assert all(e["state"] != ends[5]["state"] for e in ends[6:])          # though each writes to it
    assert [s["home"] for s in starts] == [0, 1, 2, 3, 4, 5, 6, 6, 6]     # and to its HOME
    assert [s["round"] for s in starts] == [1] * 9                        # each new learner joins the round begun
    assert "frozen: every instance from here on" in (system / "learner.log").read_text()
    # the stream's own files are back in their place, without anything the test wrote
    assert not (system / "stream-end").exists()
    assert sorted(p.name for p in (system / "state" / "notes").iterdir()) == [f"{n:04d}" for n in range(6)]
    assert (system / "home" / "count").read_text() == "6"
    # attached as arc-baselines run attaches a learner: its working directory, by an absolute path, and the proxy
    attached = json.loads((system / "state" / "attached.json").read_text())
    assert Path(attached["workdir"]).is_absolute()
    assert Path(attached["workdir"]).resolve() == Path(attached["cwd"]).resolve() == (system / "state").resolve()
    assert attached["base_url"].startswith("http://127.0.0.1:")


def test_a_learner_that_keeps_its_state_in_its_process_cannot_be_frozen(tmp_path):
    learner = HostedLearner(sys.executable, "toy_submission:Learner", code=str(toy_package(tmp_path / "code", {})),
                            sandbox="none")
    learner.attach(tmp_path / "system", {})
    try:
        assert not learner.state_in_files
        with pytest.raises(HostedLearnerError, match="state_in_files"):
            learner.freeze()
    finally:
        learner.close()


@pytest.mark.parametrize("home_aside", [True, False])
def test_the_files_a_frozen_test_stopped_hard_left_aside_get_their_place_back(tmp_path, home_aside):
    """A run taken up again after its frozen test was stopped hard goes on from the stream's files, not the copy the
    stopped instance wrote to; stopped between setting aside the state directory and HOME, it keeps its own HOME."""
    workdir = tmp_path / "system"
    folders = {"stream-end/state": '{"task-a": 3}', "state": '{"task-a": 3, "task-b": 4}'}
    folders.update({"stream-end/home": "6", "home": "7"} if home_aside else {"home": "6"})
    for name, text in folders.items():
        (workdir / name).mkdir(parents=True)
        (workdir / name / ("count" if name.endswith("home") else "memory.json")).write_text(text)
    learner = HostedLearner(sys.executable, "file_toy:FileLearner", code=str(file_toy_package(tmp_path / "code")),
                            sandbox="none")
    learner.attach(workdir, {})
    try:
        assert (workdir / "state" / "memory.json").read_text() == '{"task-a": 3}'
        assert (workdir / "home" / "count").read_text() == "6"
        assert not (workdir / "stream-end").exists() and learner.frozen is None
    finally:
        learner.close()


def submission_python(root: Path) -> str:
    """An environment of the submission's own, holding a copy of the benchmark as an install from the index would: the
    test's own environment may install the benchmark editable from a checkout the sandbox rightly hides."""
    venv = root / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    site = next(venv.glob("lib/python*/site-packages"))
    shutil.copytree(Path(continual_arc.__file__).parent, site / "continual_arc",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return str(venv / "bin" / "python")


@pytest.mark.parametrize("backend", BACKENDS[1:])
def test_the_sandbox_confines_the_learner(tmp_path, monkeypatch, upstream, backend):
    """Under the sandbox the learner cannot read the repository that launched it, nor reach the upstream past the
    proxy, and still learns and calls its model through the proxy."""
    assert (REPO / "pyproject.toml").is_file()
    code = toy_package(tmp_path / "code", {"read": str(REPO / "pyproject.toml"),
                                           "connect": upstream._server.server_address[1]})
    system, records = run_hosted(tmp_path, monkeypatch, upstream, {
        "python": submission_python(tmp_path), "learner": "toy_submission:Learner", "code": str(code),
        "sandbox": backend})
    assert_learned(records)
    probe = assert_confined(system, upstream, records)
    assert probe["read"].startswith("failed") and probe["connect"].startswith("failed")


# -- installing a submission ----------------------------------------------------------------------------------------

# An in-tree build backend with no requirements, so an install needs no index: it packs the project's packages into
# a wheel, after trying to read the file the test names (its build code runs under the install's sandbox).
BACKEND = '''
import base64, hashlib, json, os, zipfile
from pathlib import Path

NAME, VERSION, PROBE = {name!r}, "0.0.1", {probe!r}


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    dist = NAME.replace("-", "_") + "-" + VERSION
    files = {{}}
    for package in sorted(p for p in Path(".").iterdir() if (p / "__init__.py").is_file()):
        for path in sorted(package.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.as_posix()] = path.read_bytes()
        if PROBE:
            try:
                Path(PROBE).read_bytes()
                result = "ok"
            except OSError as error:
                result = f"failed: {{error}}"
            files[package.name + "/build-probe.json"] = json.dumps({{"read": result}}).encode()
    files[dist + ".dist-info/METADATA"] = f"Metadata-Version: 2.1\\nName: {{NAME}}\\nVersion: {{VERSION}}\\n".encode()
    files[dist + ".dist-info/WHEEL"] = b"Wheel-Version: 1.0\\nGenerator: toy\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n"
    record = [f"{{n}},sha256={{base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode()}},{{len(d)}}"
              for n, d in files.items()]
    files[dist + ".dist-info/RECORD"] = ("\\n".join(record + [dist + ".dist-info/RECORD,,"]) + "\\n").encode()
    wheel = dist + "-py3-none-any.whl"
    with zipfile.ZipFile(os.path.join(wheel_directory, wheel), "w") as archive:
        for n, d in files.items():
            archive.writestr(n, d)
    return wheel
'''


def project(root: Path, name: str, probe: str | None = None) -> Path:
    (root / "_build").mkdir(parents=True)
    (root / "_build" / "backend.py").write_text(BACKEND.format(name=name, probe=probe))
    (root / "pyproject.toml").write_text(textwrap.dedent(f"""\
        [build-system]
        requires = []
        build-backend = "backend"
        backend-path = ["_build"]

        [project]
        name = "{name}"
        version = "0.0.1"
        """))
    return root


def git_repo(root: Path) -> str:
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", "submission"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.skipif(shutil.which("uv") is None or shutil.which("git") is None, reason="needs uv and git")
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_submission_is_installed_under_the_sandbox_and_runs_from_its_environment(tmp_path, monkeypatch, upstream,
                                                                                    backend):
    repo = project(toy_package(tmp_path / "repo", {}), "toy-submission", probe=str(REPO / "pyproject.toml"))
    commit = git_repo(repo)
    # a stand-in for the public benchmark that needs no network: the benchmark this suite runs on, packed by the
    # same backend under the name continual-arc
    bench = project(tmp_path / "bench", "continual-arc")
    shutil.copytree(Path(continual_arc.__file__).parent, bench / "continual_arc",
                    ignore=shutil.ignore_patterns("__pycache__"))
    venv, checkout = tmp_path / "sub" / "venv", tmp_path / "sub" / "code"
    installed = install_submission(f"file://{repo}", commit, venv, checkout=checkout, sandbox=backend,
                                   python=f"{sys.version_info.major}.{sys.version_info.minor}",
                                   benchmark=str(bench), read_paths=[repo, bench])
    assert installed["python"] == str(venv / "bin" / "python") and installed["code"] == str(checkout)
    assert installed["benchmark"] == str(bench) and brings_benchmark(venv)
    record = json.loads((venv / "arc-submission.json").read_text())
    assert record["commit"] == commit and record["sandbox"] == backend
    assert subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip() == commit
    (site,) = venv.glob("lib/python*/site-packages")
    build_probe = json.loads((site / "toy_submission" / "build-probe.json").read_text())
    if backend == "none":
        assert build_probe["read"] == "ok"
    else:
        assert build_probe["read"].startswith("failed")  # the build code could not read outside the sandbox
    assert HostedLearner(installed["python"], "toy_submission:Learner").describe()["submission"]["commit"] == commit
    with pytest.raises(FileExistsError):  # an install goes into a new directory, never over another
        install_submission(f"file://{repo}", commit, venv, sandbox=backend, read_paths=[repo, bench],
                           python=f"{sys.version_info.major}.{sys.version_info.minor}")

    system, records = run_hosted(tmp_path / "run", monkeypatch, upstream, {
        "python": installed["python"], "learner": "toy_submission:Learner", "code": installed["code"],
        "sandbox": backend})
    assert_learned(records)
    assert_confined(system, upstream, records)
