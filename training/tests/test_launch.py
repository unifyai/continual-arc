"""A run's launch (continual_arc_training.launch, and lib.sh's begin that every launcher calls): every launch runs
from a checkout of the run's own commit and changes, made once, the script started again there; a new run keeps its
settings, code and record; RESUME=1 goes on with it only with its own settings, refusing other ones, a folder that
holds no run and a run that is going already; and a launch without it into a folder that holds one is refused."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from continual_arc_training import launch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PURPOSE = "a test of the launch record, made by the training package's own tests"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A clone of the repository's layout: the package's scripts and source as files of its own, so that a commit
    changes the code a launch runs, committed."""
    root = tmp_path / "repo"
    shutil.copytree(SCRIPTS, root / "training" / "scripts")
    shutil.copytree(SCRIPTS.parent / "src", root / "training" / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "schedules"))
    for name in ("src", "baselines/src"):
        (root / name).mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'continual-arc'\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "first")
    return root


def begin(folder, root, resume=False, settings=None, by="realtime.sh", **kw):
    """A launch as a script makes it: from ``root``, and once more from the checkout of the run's code it returns,
    as lib.sh starts the script again there; the checkout, or None for one that ran from it already."""
    kw = {"purpose": PURPOSE, "home": folder.parents[2] / "code", **kw}
    args = dict(run="rt/x", by=by, pid=os.getpid(), resume=resume,
                settings=settings or {"ROW": "dagger", "TRAIN_GPUS": "4,5"}, free={"TRAIN_GPUS"})
    at = launch.begin(folder, root=root, **args, **kw)
    if at is not None and not kw.get("moved"):
        assert launch.begin(folder, root=at, **args, **{**kw, "moved": True}) is None
    return at


def test_a_new_run_keeps_its_settings_code_and_launch_and_a_resume_goes_on_only_as_itself(tmp_path, repo):
    run = tmp_path / "runs" / "rt" / "x"
    at = begin(run, repo)
    assert at == tmp_path / "code" / git(repo, "rev-parse", "HEAD")   # the run's code, which no pull changes
    record = json.loads((run / "settings.json").read_text())
    assert record == {"settings": {"ROW": "dagger"}, "code": {"commit": git(repo, "rev-parse", "HEAD"),
                                                              "diff_sha256": None}}
    manifest = json.loads((run / "manifest.json").read_text())
    assert (manifest["repo"], manifest["run"], manifest["purpose"]) == ("continual-arc", "rt/x", PURPOSE)
    [first] = manifest["launches"]
    assert first["settings"] == {"TRAIN_GPUS": "4,5"} and first["status"] == "running" and not first["resume"]
    launch.end(run, os.getpid(), 0)
    assert json.loads((run / "manifest.json").read_text())["launches"][0]["status"] == "done"
    (run / "play").mkdir()   # the run's own files
    with pytest.raises(launch.Refused, match="already holds a run"):
        begin(run, repo)
    with pytest.raises(launch.Refused, match="ROW 'dagger', not 'pg'"):
        begin(run, repo, resume=True, settings={"ROW": "pg", "TRAIN_GPUS": "4,5"})
    # its GPUs may change; the launch's process is this test's, which runs python
    assert begin(run, repo, resume=True, settings={"ROW": "dagger", "TRAIN_GPUS": "0,1"}, by="python") == at
    manifest = json.loads((run / "manifest.json").read_text())
    assert [(x["id"], x["resume"], x["settings"]["TRAIN_GPUS"], x["code"]) for x in manifest["launches"]] == [
        (1, False, "4,5", str(at)), (2, True, "0,1", str(at))]
    assert [json.loads(line)["launch"] for line in (run / "resumes.jsonl").read_text().splitlines()] == [2]
    with pytest.raises(launch.Refused, match="goes on already"):   # launch 2, this process, has not ended
        begin(run, at, resume=True)
    launch.end(run, os.getpid(), 0)
    assert begin(run, repo, resume=True) == at


def test_a_resume_needs_a_run_and_a_new_run_a_purpose_or_the_one_of_the_run_that_holds_it(tmp_path, repo):
    runs = tmp_path / "runs"
    with pytest.raises(launch.Refused, match="holds none"):
        begin(runs / "rt" / "never", repo, resume=True)
    with pytest.raises(launch.Refused, match="says why"):
        begin(runs / "rt" / "y", repo, purpose="a test")
    begin(runs / "rt" / "y", repo)
    begin(runs / "rt" / "y" / "frozen", repo, purpose="", runs=runs, home=tmp_path / "code")
    held = json.loads((runs / "rt" / "y" / "frozen" / "manifest.json").read_text())["purpose"]
    assert held == f"part of rt/x (frozen): {PURPOSE}"


def test_a_checkout_with_changes_keeps_them_and_a_resume_from_other_code_goes_on_from_a_checkout_of_its_own(tmp_path,
                                                                                                         repo):
    """The run began on the first commit with a change to the package and a new file: a resume from the next commit
    returns a checkout of the first with both, made once beside the repository, where the code is the run's own."""
    first = git(repo, "rev-parse", "HEAD")
    module = repo / "training" / "src" / "continual_arc_training" / "launch.py"
    module.write_text(module.read_text() + "\n# a change the run began with\n")
    (repo / "training" / "src" / "continual_arc_training" / "extra.py").write_text("EXTRA = 1\n")
    run, home = tmp_path / "runs" / "rt" / "x", tmp_path / "code"
    moved = begin(run, repo)
    record = json.loads((run / "settings.json").read_text())["code"]
    assert record["commit"] == first and record["diff_sha256"] and "extra.py" in (run / "code.diff").read_text()
    assert moved.parent == home and moved.name.startswith(first + "-")
    assert (moved / "training" / "src" / "continual_arc_training" / "extra.py").read_text() == "EXTRA = 1\n"
    assert not (moved / "runs").exists() and (moved / "pyproject.toml").exists()   # the code alone
    launch.end(run, os.getpid(), 1)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "the next commit")
    assert begin(run, repo, resume=True) == moved   # the run's own code, made once
    with pytest.raises(launch.Refused, match="does not hold it"):
        begin(run, repo, resume=True, moved=True)
    with pytest.raises(launch.Refused, match="checkout of its own"):
        begin(run, repo, resume=True, stay=True)
    launches = json.loads((run / "manifest.json").read_text())["launches"]
    assert [x["code"] for x in launches] == [str(moved)] * 2


def test_a_launch_from_no_git_checkout_is_refused(tmp_path):
    with pytest.raises(launch.Refused, match="no git checkout"):
        begin(tmp_path / "run", tmp_path)


def _launcher(root: Path) -> Path:
    """A launcher as the package's scripts are written: lib.sh sourced, its settings, then begin; it prints the
    checkout it runs from and where its stdout goes, then fails, so that its launch is recorded as failed."""
    path = root / "training" / "scripts" / "try.sh"
    path.write_text('set -uo pipefail\nsource "$(dirname "${BASH_SOURCE[0]}")/lib.sh"\nROW=${ROW:-dagger}\n'
                    'L=$RUNS/rt/x\nmkdir -p "$L"\nbegin "$L" rt/x "TRAIN_GPUS" ROW TRAIN_GPUS || exit 1\n'
                    'echo "running from $ROOT"\nexit 3\n')
    return path


def test_begin_records_a_launch_keeps_the_console_and_starts_a_resume_again_from_the_runs_own_code(tmp_path, repo):
    script = _launcher(repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "a launcher")
    runs, home = tmp_path / "runs", tmp_path / "code"
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "RUNS": str(runs), "CODE_HOME": str(home),
           "TRAIN_GPUS": "4,5"}

    def launch_it(**extra):
        return subprocess.run(["bash", str(script)], env={**env, **extra}, capture_output=True, text=True)

    assert launch_it(PURPOSE=PURPOSE).returncode == 3
    run, at = runs / "rt" / "x", home / git(repo, "rev-parse", "HEAD")
    assert (run / "console.log").read_text() == f"running from {at}\n"   # started again from the run's own code
    [first] = json.loads((run / "manifest.json").read_text())["launches"]
    assert (first["status"], first["exit"], first["by"], first["code"]) == ("failed", 3, "try.sh", str(at))
    refused = launch_it(PURPOSE=PURPOSE)
    assert refused.returncode == 1 and "already holds a run" in refused.stderr
    assert "try.sh: refused: " in (run / "progress").read_text()
    assert launch_it(RESUME="1", ROW="pg").returncode == 1   # other settings
    (repo / "training" / "scripts" / "lib.sh").write_text((repo / "training" / "scripts" / "lib.sh").read_text()
                                                         + "\n# the next commit\n")
    git(repo, "commit", "-qam", "the next commit")
    assert launch_it(RESUME="1", TRAIN_GPUS="0,1").returncode == 3   # still the run's own code, its GPUs another's
    assert (run / "console.log").read_text().splitlines() == [f"running from {at}"] * 2 and len(list(home.iterdir())) == 1
    launches = json.loads((run / "manifest.json").read_text())["launches"]
    assert [x["settings"] for x in launches] == [{"TRAIN_GPUS": "4,5"}, {"TRAIN_GPUS": "0,1"}]


def test_the_launch_module_runs_on_the_machines_own_python_with_its_standard_library_alone():
    """lib.sh runs it with the node's python3 and the package's source on PYTHONPATH, nothing installed."""
    code = "import sys; import continual_arc_training.launch; print(sorted(m for m in sys.modules if '.' not in m))"
    out = subprocess.run([sys.executable, "-I", "-S", "-c", f"import sys; sys.path.insert(0, {str(SCRIPTS.parent / 'src')!r}); {code}"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
