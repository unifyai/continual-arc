"""Every round's end state as a git commit (statelog.py): a restored round is the state as it was, files a system
ignores, its symlinks and its executable bits included, and an empty state an empty directory; a log taken up again goes
on counting; the bare learners'
memory reaches a logged state's commit and comes back when a run is taken up again, a stateless one's wiped before its
first instance; and a git command that fails says why."""

import json
import os
import shutil
import subprocess
import sys

import pytest
from continual_arc.env import DemosFeedback, LearnerObservation, RequestDemos, Submit
from continual_arc.grids import Grid
from continual_arc.tasks import Pair

from continual_arc_baselines import shared
from continual_arc_baselines.statelog import StateLog, _descendants, _git, restore
from continual_arc_baselines.systems.bare import BareProgramLibraryLearner

from .test_bare_library import FLIP, Scripted, obs


def files(root):
    """Every file under ``root``: its bytes and whether it is executable, or where it links."""
    out = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if path.is_symlink():
            out[name] = ("link", os.readlink(path))
        elif path.is_file():
            out[name] = ("file", path.read_bytes(), bool(path.stat().st_mode & 0o100))
    return out


def test_every_round_restores_as_it_was(tmp_path):
    state = tmp_path / "state"
    log = StateLog(state, tmp_path / "state.git", tmp_path / "state-log.jsonl")
    (state / "notes").mkdir()
    (state / "notes" / "a.md").write_text("first")
    (state / ".gitignore").write_text("*.cache\n")    # a system's own ignore file must not thin the commit
    (state / "index.cache").write_bytes(os.urandom(1000))
    first = log.commit()
    rounds = [files(state)]
    (state / "notes" / "a.md").write_text("second")
    (state / "run.sh").write_text("#!/bin/sh\n")
    (state / "run.sh").chmod(0o755)
    (state / "latest").symlink_to("notes/a.md")
    (state / "index.cache").unlink()
    second = log.commit()
    rounds.append(files(state))
    assert first != second and log.rounds == 2
    for index, expected in enumerate(rounds):
        into = tmp_path / f"round-{index}"
        restore(log.repo, log.commit_of(index), into)
        assert files(into) == expected
    with pytest.raises(KeyError):
        log.commit_of(2)


def test_an_empty_state_restores_as_an_empty_directory(tmp_path):
    log = StateLog(tmp_path / "state", tmp_path / "state.git", tmp_path / "state-log.jsonl")
    log.commit()
    restore(log.repo, log.commit_of(0), tmp_path / "round-0")
    assert (tmp_path / "round-0").is_dir() and not any((tmp_path / "round-0").iterdir())


def test_a_log_taken_up_again_goes_on_counting(tmp_path):
    paths = (tmp_path / "state", tmp_path / "state.git", tmp_path / "state-log.jsonl")
    StateLog(*paths).commit()
    again = StateLog(*paths)
    assert again.rounds == 1
    again.commit()
    assert [json.loads(line)["round"] for line in paths[2].read_text().splitlines()] == [0, 1]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the process table is read from /proc")
def test_the_cells_other_processes_are_the_ones_held_still():
    child = subprocess.Popen(["sleep", "30"])
    try:
        assert child.pid in _descendants(os.getpid())
    finally:
        child.kill()
        child.wait()


def test_the_program_librarys_memory_reaches_the_commit_and_comes_back(tmp_path):
    library = BareProgramLibraryLearner("vendor/model")
    library.attach(tmp_path / "system", {}, log_state=True)
    pair = Pair(Grid([[1]]), Grid([[2]]))
    seen = LearnerObservation(step=0, task_id="t1", input=[[0]], attempts_used=0, demo_requests_used=0,
                              attempt_cap=8, demo_cap=1)
    library.observe(seen, RequestDemos(), DemosFeedback(pairs=[pair], refused=False, demo_requests_used=1))
    library.library["t1"] = "def transform(grid):\n    return grid\n"
    library.end_round()
    assert library.state_log.rounds == 1
    taken_up = BareProgramLibraryLearner("vendor/model")
    taken_up.attach(tmp_path / "system", {}, log_state=True)
    assert taken_up.library == library.library
    assert [p.to_json() for p in taken_up.cache["t1"]] == [pair.to_json()]
    taken_up.wipe_state()
    assert not (tmp_path / "system" / "state" / "library.json").exists()
    assert not (tmp_path / "system" / "state" / "cache.json").exists()


def test_a_stateless_librarys_memory_reaches_the_commit_and_is_wiped_before_a_new_learners_first_instance(tmp_path):
    """Row PL0, the program library wiped before every instance: what it kept in a round is in the round's commit, and a
    learner made anew on its folder (a run forked from a round, a collective test's user, a driver that plays each
    round in a learner of its own) wipes it before its first instance as before every other, so the model answers the
    task and the program kept for it is never run."""
    library = BareProgramLibraryLearner("vendor/model", stateless=True)
    library.attach(tmp_path / "system", {}, log_state=True)
    library.library["t1"] = "def transform(grid):\n    return grid\n"
    library.end_round()
    restore(library.state_log.repo, library.state_log.commit_of(0), tmp_path / "round-0")
    assert json.loads((tmp_path / "round-0" / "library.json").read_text()) == library.library
    taken_up = Scripted([FLIP], stateless=True)
    taken_up.attach(tmp_path / "system", {}, log_state=True)
    assert taken_up.act(obs(0)) == Submit([[2, 1]])   # the model's program: the kept one would submit [[1, 2]]
    assert taken_up.stats["library_runs"] == 0 and taken_up.stats["wipes"] == 1 and len(taken_up.sent) == 1


def test_a_state_goes_back_to_where_it_was_saved_and_its_history_goes_on_from_there(tmp_path):
    """A run that resumes puts its system's state back as it was committed before the round it goes on from: the
    directory that commit holds, whatever a stopped instance wrote since and whatever it committed after; the log's
    later rounds forgotten, so the next commit follows the saved one; and, saved before any commit, nothing at all."""
    state = tmp_path / "state"
    log = StateLog(state, tmp_path / "state.git", tmp_path / "state-log.jsonl")
    nothing = log.saved()
    assert nothing == {"rounds": 0, "commit": None}
    (state / "memory.md").write_text("round 0")
    log.commit()
    saved = log.saved()
    assert saved == {"rounds": 1, "commit": log.commit_of(0)}
    (state / "memory.md").write_text("round 1")
    (state / "x.cache").write_text("ignored, but committed")
    log.commit()
    (state / "memory.md").write_text("half of round 2")   # what a stop in round 2 leaves
    (state / "partial").mkdir()
    (state / "partial" / "note").write_text("cut off")
    log.go_back(saved)
    assert files(state) == {"memory.md": ("file", b"round 0", False)}
    assert [json.loads(line)["round"] for line in (tmp_path / "state-log.jsonl").read_text().splitlines()] == [0]
    assert log.rounds == 1 and log.head() == saved["commit"] and not log.changed()
    log.go_back(saved)   # where it is already: nothing to write
    (state / "memory.md").write_text("round 1, again")
    again = log.commit()
    git = ["git", f"--git-dir={tmp_path / 'state.git'}"]
    assert subprocess.run([*git, "rev-parse", f"{again}~1"], capture_output=True, text=True).stdout.strip() == saved["commit"]
    assert json.loads((tmp_path / "state-log.jsonl").read_text().splitlines()[-1]) | {"at": 0} == {
        "round": 1, "commit": again, "at": 0}
    log.go_back(nothing)
    assert files(state) == {} and log.head() is None and log.rounds == 0
    (state / "memory.md").write_text("round 0, from nothing")
    first = log.commit()
    assert subprocess.run([*git, "rev-list", "--count", first], capture_output=True, text=True).stdout.strip() == "1"
    shutil.rmtree(state)   # a system's wipe that takes its state directory with it
    log.go_back({"rounds": 1, "commit": first})
    assert files(state) == {"memory.md": ("file", b"round 0, from nothing", False)}


@pytest.mark.parametrize("run", [_git, shared._git])
def test_a_failed_git_command_says_why(tmp_path, run):
    log = StateLog(tmp_path / "state", tmp_path / "state.git", tmp_path / "state-log.jsonl")
    with pytest.raises(RuntimeError, match=r"git rev-parse no-such-commit exited 128 on .*: fatal: "):
        run(log.repo, "rev-parse", "no-such-commit")
