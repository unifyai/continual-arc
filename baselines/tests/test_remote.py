"""A learner served from a machine of its own (systems/remote.py), with ssh stood in for by a command that runs the same
thing here: the protocol over the connection, a frozen test played there, what the learner printed and kept brought back
as bytes, and a far side that misbehaves."""

import json
import os
import platform
import shutil
import stat
import sys
from pathlib import Path

import pytest
from continual_arc.schedule import Schedule, Slot, stream_schedule
from continual_arc.testing import FakeSource

from continual_arc_baselines.attempt import Attempt
from continual_arc_baselines.cli import main
from continual_arc_baselines.systems.hosted import HostedLearnerError
from continual_arc_baselines.systems.remote import RemoteLearner, installed, ssh_command
from continual_arc_baselines.systems.sandbox import python_runtime_root

from .test_hosted import SECRET, file_toy_package, submission_python

FAKE_SSH = r"""#!/bin/sh
# Stands in for ssh: its options are dropped and its command run here, as the arena would.
while [ "$#" -gt 2 ]; do shift; done
exec /bin/sh -c "$2"
"""
LINUX = pytest.mark.skipif(platform.system() != "Linux" or shutil.which("bwrap") is None,
                           reason="the arena's side runs under bubblewrap, on Linux")


@pytest.fixture
def upstream():
    from .fake_upstream import FakeUpstream

    server = FakeUpstream().start()
    yield server
    server.stop()


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    folder = tmp_path / "fake-bin"
    folder.mkdir()
    script = folder / "ssh"
    script.write_text(FAKE_SSH)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")


def kwargs_for(tmp_path, code) -> dict:
    python = submission_python(tmp_path)
    return {"arena": "arena@test", "key": str(tmp_path / "key"), "known_hosts": str(tmp_path / "known"),
            "python": python, "learner": "file_toy:FileLearner", "code": str(code),
            "runtime": [str(python_runtime_root(Path(python)))], "arena_workdir": str(tmp_path / "arena" / "cell"),
            "arena_python": sys.executable, "timeout_s": 120.0, "startup_timeout_s": 120.0, "margin_s": 30.0}


def test_what_a_virtual_environment_holds_is_read_from_the_names_of_its_folders_alone(tmp_path):
    site = tmp_path / "lib" / "python3.12" / "site-packages"
    for name in ("numpy-2.1.0.dist-info", "my_pkg-1.0+local.dist-info", "@evil[x](y)-1.dist-info", "numpy", "x-1.dist-info.txt"):
        (site / name).mkdir(parents=True)
    assert installed(tmp_path) == ["my_pkg==1.0+local", "numpy==2.1.0"]


def test_a_cells_folder_goes_whatever_its_learner_did_to_its_permissions(tmp_path):
    from continual_arc_baselines.systems.remote import main as remote_main

    cell = tmp_path / "cell"
    (cell / "locked" / "deep").mkdir(parents=True)
    (cell / "locked" / "deep" / "file").write_text("x")
    (cell / "locked").chmod(0)
    assert remote_main(["forget", "--workdir", str(cell)]) == 0 and not cell.exists()


def test_the_ssh_command_turns_off_what_a_far_side_could_use():
    command = ssh_command("arena@host", "/k", "/known", "serve", forward=(7, 9), max_bytes=1024)
    assert "ssh" in command and command[-2:] == ["arena@host", "serve"]
    if shutil.which("prlimit"):
        assert command[:2] == ["prlimit", "--fsize=1024"]
    assert command[command.index("-R") + 1] == "127.0.0.1:7:127.0.0.1:9"
    for option in ("ForwardAgent=no", "ForwardX11=no", "PermitLocalCommand=no", "UpdateHostKeys=no", "BatchMode=yes",
                   "UserKnownHostsFile=/known", "ExitOnForwardFailure=yes"):
        assert option in command
    assert command[command.index("-F") + 1] == "/dev/null" and "-R" not in ssh_command("a@b", "/k", "/known", "x")


@LINUX
def test_a_learner_plays_its_stream_and_frozen_test_from_the_arena(tmp_path, monkeypatch, upstream, fake_ssh):
    source = FakeSource(n_tasks=3)
    stream_schedule(source.task_ids(), 6, seed=1, source=source.name).save(tmp_path / "schedule.json")
    test = Schedule(name="toy-seen-test", seed=1, users=3, sources={t: source.name for t in source.task_ids()},
                    slots=[Slot(t, 700 + u, "seen-test", user=u) for u, t in enumerate(source.task_ids())],
                    meta={"test": "seen", "width": 3})
    test.save(tmp_path / "test.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    reboots = []  # never a real one: ssh here is this machine
    monkeypatch.setattr(RemoteLearner, "_reboot", lambda self: reboots.append(self.arena))
    context = Attempt.create("remote", root=tmp_path / "wb")
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))
    out = tmp_path / "run"
    kwargs = kwargs_for(tmp_path, file_toy_package(tmp_path / "code"))
    assert main(["run", "--learner", "continual_arc_baselines.systems.remote:RemoteLearner",
                 "--learner-kwargs", json.dumps(kwargs), "--schedule", str(tmp_path / "schedule.json"),
                 "--test", str(tmp_path / "test.json"), "--test-out", str(tmp_path / "tested"),
                 "--registry", "tests.fixtures:fake_registry", "--upstream", upstream.base_url, "--quiet",
                 "--out", str(out)]) == 0
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    tested = [json.loads(line) for line in (tmp_path / "tested" / "records.jsonl").read_text().splitlines()]
    assert len(records) == 6 and len(tested) == 3 and all(r["user"] == 0 and r["reset_before"] for r in tested)
    assert reboots == ["arena@test"] * 3  # before every instance of the test, and none in the stream
    system = context.directory / "system"
    # what each instance printed left the arena before the next began, since the disk's free space would show it
    assert (system / "learner.log").read_text().count("[hosted] started") == 1 + 1 + 3
    # what the learner printed and kept came back as bytes, and the arena's own folder holds the rest
    assert "[hosted] frozen" in (system / "learner.log").read_text()
    assert (system / "kept.tar").stat().st_size > 0
    assert (tmp_path / "arena" / "cell" / "state").is_dir()
    assert not (system / "state").exists()
    params = json.loads((context.directory / "parameters.json").read_text()) if (
        context.directory / "parameters.json").exists() else None
    if params:
        assert params["system"]["system"] == "remote"


class FarSide:
    """A far side written as a script in place of the arena's interpreter, for what a machine that is not to be trusted
    might send."""

    def __init__(self, tmp_path, script: str):
        self.path = tmp_path / "far-side"
        self.path.write_text(f"#!{sys.executable}\n{script}\n")
        self.path.chmod(self.path.stat().st_mode | stat.S_IXUSR)

    def learner(self) -> RemoteLearner:
        return RemoteLearner("arena@test", "/k", "/known", "/nowhere/python", "x:Y", "/nowhere/cell", str(self.path),
                             timeout_s=2.0, startup_timeout_s=2.0, close_timeout_s=1.0, margin_s=1.0)


def attach(learner: RemoteLearner, tmp_path) -> None:
    learner.attach(tmp_path / "system", {"ARC_LLM_BASE_URL": "http://127.0.0.1:7000/v1"})


@pytest.mark.parametrize("script,message", [
    ("print('x' * 20_000_000, flush=True)", "characters or more"),
    ("import time; time.sleep(30)", "did not answer"),
    ("pass", "exited"),
    ("for i in range(100000):\n    print('noise', flush=True)", "broke the protocol"),
])
def test_a_far_side_that_misbehaves_at_the_start_fails_the_cell(tmp_path, fake_ssh, script, message):
    far = FarSide(tmp_path, "import sys\n" + script)
    with pytest.raises(HostedLearnerError, match=message):
        attach(far.learner(), tmp_path)


@LINUX
def test_what_a_far_side_sends_is_cut_at_its_cap_whatever_it_says_it_will_send(tmp_path):
    """The arena's own ``head -c`` is the arena's word. ssh's file stops at the limit this side sets."""
    import subprocess

    target = tmp_path / "out"
    with target.open("wb") as out:
        done = subprocess.run(["prlimit", "--fsize=1000", "sh", "-c", "head -c 5000 /dev/zero"], stdout=out, check=False)
    assert target.stat().st_size == 1000 and done.returncode != 0


def test_a_far_side_that_misbehaves_in_play_fails_the_cell(tmp_path, fake_ssh):
    far = FarSide(tmp_path, """
import json, sys
print(json.dumps({"ok": True, "result": {"ready": True}}), flush=True)
for line in sys.stdin:
    print("{not json", flush=True)
""")
    learner = far.learner()
    attach(learner, tmp_path)
    with pytest.raises(HostedLearnerError, match="broke the protocol"):
        learner.begin_round(1)
    learner.close()
