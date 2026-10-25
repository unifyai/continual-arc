"""The node scripts (scripts/) end to end in a fake home, with every tool they call on a node stubbed on its PATH: the
frozen rows' round-by-round play (paper.sh), a reference row kept going (reference.sh), its frozen tests
(ref-tests.sh), the references from nothing after a run's tests (refs.sh), and training once at the end (scratch.sh).
frozen.sh's own are in test_frozen.py, realtime.sh's settings in test_realtime.py."""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from continual_arc_training import memory

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
def test_every_script_parses(script):
    assert subprocess.run(["bash", "-n", str(SCRIPTS / script)], capture_output=True).returncode == 0


def _stub(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def node(tmp_path):
    """A fake home: the scripts copied beside the package's source (so that lib.sh finds it), ``~/.local/bin`` first on
    the PATH with setsid, sudo and python3 (this interpreter, as the node's own) stubbed, an empty prime-rl checkout and
    runs folder."""
    home = tmp_path / "home"
    tools = home / ".local" / "bin"
    tools.mkdir(parents=True)
    training = tmp_path / "training"
    shutil.copytree(SCRIPTS, training / "scripts")
    (training / "src").symlink_to(SCRIPTS.parent / "src")
    for folder in ("prime-rl", "runs", "baselines"):
        (home / folder).mkdir()
    _stub(tools, "setsid", 'exec "$@"')
    _stub(tools, "sudo", "exit 0")
    _stub(tools, "python3", f'exec {sys.executable} "$@"')
    env = {"PATH": f"{tools}:{os.environ['PATH']}", "HOME": str(home), "BASELINES": str(home / "baselines")}
    return home, training / "scripts", tools, env


def _run(scripts, env, script, *args, timeout=120, **extra):
    return subprocess.run(["bash", str(scripts / script), *args], env={**env, **extra}, capture_output=True, text=True,
                          timeout=timeout)


# -- paper.sh ----------------------------------------------------------------------------------------------------

def _paper_stubs(home, scripts, tools):
    """serve.sh records its calls; a play (uv run ... continual_arc_training.play) records its arguments and writes its
    round as the runner does, the cut's schedule.json first, then its records and the round's memory counters; a round
    listed in ``fail-once`` dies its first time after its schedule.json, with no record."""
    (scripts / "serve.sh").write_text('echo "$@" >> "$HOME/served"\n')
    report = {"round": 0, "users": {}, "total": {"memory": {a: {"added": 1, "changed": 0, "removed": 0}
                                                            for a in memory.AREAS},
                                                 "reads": {"programs": 1}, "reused_program": 1,
                                                 "solved_without_demo": 0}}
    (home / "report.json").write_text(json.dumps(report))
    _stub(tools, "uv", '''shift 5; echo "$@" >> "$HOME/plays"
while [ $# -gt 0 ]; do case $1 in --run-dir) d=$2 ;; --round) r=$2 ;; --users) u=$2 ;; --epilogue-users) u=$2 ;; --width) w=$2 ;; esac; shift; done
u=${w:-$u}
mkdir -p "$d/rounds/$(printf r%04d "$r")"
python3 -c "import json, sys; json.dump({'slots': [{}] * (int(sys.argv[2]) * (int(sys.argv[3]) + 1))}, open(sys.argv[1], 'w'))" "$d/schedule.json" "$u" "$r"
if grep -qx "$r" "$HOME/fail-once" 2>/dev/null && [ ! -f "$HOME/failed-$r" ]; then touch "$HOME/failed-$r"; exit 1; fi
for i in $(seq "$u"); do echo "{}" >> "$d/records.jsonl"; done
cp "$HOME/report.json" "$d/rounds/$(printf r%04d "$r")/memory.json"''')
    return report


def test_paper_sh_plays_each_round_on_the_base_model_retries_a_crashed_play_and_resumes(node):
    home, scripts, tools, env = node
    report = _paper_stubs(home, scripts, tools)
    (home / "fail-once").write_text("1\n")
    out = _run(scripts, env, "paper.sh", ROW="h0", USERS="2", ROUNDS="2", GRACE="1")
    assert out.returncode == 0, out.stdout + out.stderr
    run = home / "runs" / "paper" / "h0-u2"
    play_dir = run / "play-u2"
    assert (play_dir / "paper-0-1.done").exists() and (play_dir / "round-play1.done").exists()
    plays = (home / "plays").read_text().splitlines()
    assert plays == [f"--row h0 --run-dir {play_dir} --base-url http://localhost:8101/v1 --served Qwen/Qwen3.5-9B "
                     f"--round {r} --users 2" for r in (0, 1, 1)]
    assert (home / "served").read_text().splitlines() == ["0,1,2,3,4,5,6,7 8"] * 2
    progress = re.sub(r"\( +", "(", (run / "progress").read_text())   # BSD's wc pads its count
    assert "play round 1 short (2/4) after attempt 1" in progress and "play round 1 done (4/4)" in progress
    assert progress.count(memory.summary_line(report)) == 2
    again = _run(scripts, env, "paper.sh", ROW="h0", USERS="2", ROUNDS="2", GRACE="0")
    assert again.returncode == 0 and len((home / "plays").read_text().splitlines()) == 3   # done: nothing played
    assert (run / "progress").read_text().endswith("already done\n")


def test_paper_sh_plays_the_epilogue_alone_wide_into_a_folder_of_its_own_and_refuses_what_it_cannot_play(node):
    home, scripts, tools, env = node
    _paper_stubs(home, scripts, tools)
    (home / "report.json").unlink()   # no memory counters: the round is not failed for them
    out = _run(scripts, env, "paper.sh", ROW="bare", USERS="4", ROUNDS="1", EPILOGUE="1", EPILOGUE_USERS="4",
               WIDTH="4", GRACE="0")
    assert out.returncode == 0, out.stdout + out.stderr
    play_dir = home / "runs" / "paper" / "bare-u4" / "epilogue-u4-w4"
    assert (play_dir / "paper-0-0.done").exists()
    assert (home / "plays").read_text().endswith("--round 0 --epilogue-users 4 --users 4 --width 4\n")
    refused = _run(scripts, env, "paper.sh", ROW="h0", USERS="4", ROUNDS="1", EPILOGUE="1", EPILOGUE_USERS="10")
    assert refused.returncode == 1 and "EPILOGUE_USERS (10) must not be more than USERS (4)" in refused.stdout
    refused = _run(scripts, env, "paper.sh", ROW="dagger", ROUNDS="1")
    assert refused.returncode == 1 and "the trained rows run through realtime.sh" in refused.stdout


# -- reference.sh and ref-tests.sh --------------------------------------------------------------------------------

def test_reference_sh_plays_a_row_from_the_baselines_environment_with_its_key(node):
    home, scripts, tools, env = node
    _stub(tools, "uv", 'echo "$(pwd) $*" >> "$HOME/driver"')
    assert _run(scripts, env, "reference.sh", PLAYER="glm", ROW="hg", USERS="2").returncode == 0
    assert _run(scripts, env, "reference.sh", PLAYER="qwen", ROW="dc", MAX_TOKENS="128000").returncode == 0
    base = home / "baselines"
    assert (home / "driver").read_text().splitlines() == [
        f"{base} run --no-sync --env-file .env python -m continual_arc_training.references play --player glm --row hg "
        f"--run-dir {home}/runs/paper/glm-hg-u2/play --users 2 --rounds 200",
        f"{base} run --no-sync --env-file .env python -m continual_arc_training.references bare --player qwen --row dc "
        f"--run-dir {home}/runs/paper/qwen-dc-u1/play --users 1 --rounds 200 --max-tokens 128000"]
    assert (home / "runs" / "paper" / "glm-hg-u2" / "play.done").exists()


def _ref_tests(scripts, env, then):
    """ref-tests.sh's functions lifted out (its loop keeps a table bash 3 has no form for), then ``then``."""
    script = (scripts / "ref-tests.sh").read_text()
    head = script[script.index("POLL=${POLL"):script.index("\n{   # read whole")]
    body = f'source {scripts}/lib.sh\n{head}\n{then}'
    return subprocess.run(["bash", "-c", body], env=env, capture_output=True, text=True, timeout=120)


def _reference_row(home, name, player, row, users):
    played = home / "runs" / "paper" / name / "play"
    played.mkdir(parents=True)
    (played / "driver.json").write_text(json.dumps({"player": player, "row": row}))
    (played / "schedule.json").write_text(json.dumps({"users": users}))
    (played.parent / "play.done").touch()


def test_ref_tests_sh_plays_each_finished_rows_tests_with_its_extra_plays_then_its_summary(node):
    home, scripts, tools, env = node
    _stub(tools, "uv", '''echo "$*" >> "$HOME/frozen"
case $* in *" summary "*) exit 0 ;; esac
while [ $# -gt 0 ]; do case $1 in --run) run=$2 ;; --name) name=$2 ;; esac; shift; done
mkdir -p "$run/frozen/$name/play/users/u000/system/state"; echo "{}" > "$run/frozen/$name/play/records.jsonl"''')
    for name, player, row, users in (("glm-h-u1", "glm", "h", 1), ("qwen-h0-u1", "qwen", "h0", 1),
                                     ("qwen-hg-u2", "qwen", "hg", 2)):
        _reference_row(home, name, player, row, users)
    (home / "runs" / "paper" / "qwen-h-u1.old" / "play").mkdir(parents=True)   # set aside: no row
    paper = home / "runs" / "paper"
    listed = _ref_tests(scripts, env, "finished")
    assert sorted(listed.stdout.split()) == ["glm-h-u1", "qwen-h0-u1", "qwen-hg-u2"]
    for row in ("glm-h-u1", "qwen-h0-u1", "qwen-hg-u2"):
        assert _ref_tests(scripts, env, f"take {row}").returncode == 0
        assert (paper / row / "frozen" / "done").exists()
        assert not list((paper / row / "frozen").glob("*/play/users/*/system/state"))
    plays = [line for line in (home / "frozen").read_text().splitlines() if " play " in line]

    def names(row):
        return [line.split("--name ")[1] for line in plays if f"--run {paper}/{row} " in line]

    assert names("glm-h-u1") == ["held-out --player glm", "held-out-qwen --player qwen", "seen --player glm",
                                 "seen-qwen --player qwen"]
    assert names("qwen-h0-u1") == ["held-out --player qwen", "held-out-reference --player qwen --reference",
                                   "seen --player qwen", "seen-reference --player qwen --reference"]
    assert names("qwen-hg-u2") == ["held-out --player qwen", "seen --player qwen"]
    assert sum(" summary --run " in line for line in (home / "frozen").read_text().splitlines()) == 3
    assert _ref_tests(scripts, env, "finished").stdout.split() == []   # every row's tests done


def test_ref_tests_sh_counts_a_row_whose_play_keeps_failing(node):
    home, scripts, tools, env = node
    _stub(tools, "uv", 'case $* in *" summary "*) exit 0 ;; esac; exit 1')
    _reference_row(home, "glm-hpl-u1", "glm", "hpl", 1)
    assert _ref_tests(scripts, env, "take glm-hpl-u1").returncode == 0
    paper = home / "runs" / "paper"
    assert (paper / ".ref-tests-failed" / "glm-hpl-u1").read_text() == "1\n"
    assert not (paper / "glm-hpl-u1" / "frozen" / "done").exists()
    log = (paper / "ref-tests.log").read_text()
    assert "glm-hpl-u1: held-out stopped short, attempt 3" in log and "not done (1 of 2 passes)" in log


# -- refs.sh and scratch.sh ---------------------------------------------------------------------------------------

def test_refs_sh_plays_both_references_from_nothing_once_the_runs_tests_are_done(node):
    home, scripts, tools, env = node
    (scripts / "frozen.sh").write_text('echo "$ROW $*" >> "$HOME/refs"\n')
    (home / "runs" / "rt" / "pg-u2-lag1" / "frozen").mkdir(parents=True)
    (home / "runs" / "rt" / "pg-u2-lag1" / "frozen" / "done").touch()
    assert _run(scripts, env, "refs.sh", "pg-u2-lag1", "0,1,2,3").returncode == 0
    assert (home / "refs").read_text().splitlines() == ["h0 qwen-h0 0,1,2,3", "bare qwen-f 0,1,2,3"]


def test_scratch_sh_tests_the_weights_alone_trains_once_at_the_end_tests_the_fresh_adapter_and_summarises(node):
    home, scripts, tools, env = node
    run = home / "runs" / "rt" / "dagger-u1-lag1"
    refused = _run(scripts, env, "scratch.sh", "dagger-u1-lag1", "2")
    assert refused.returncode == 1 and "frozen tests are not done" in (run / "scratch" / "progress").read_text()
    (run / "frozen").mkdir(parents=True)
    (run / "frozen" / "done").touch()
    (run / "play").mkdir()
    (scripts / "frozen.sh").write_text('echo "${PYTORCH_CUDA_ALLOC_CONF:-none} $ADAPTER $SERVED ${MEMORY:-} $OUT $*" '
                                       '>> "$HOME/tests"; mkdir -p "$OUT"; touch "$OUT/done"\n')
    _stub(tools, "uv", 'echo "${PYTORCH_CUDA_ALLOC_CONF:-none} $*" >> "$HOME/scratch"; echo "{\\"steps\\": 3}"')
    refused = _run(scripts, env, "scratch.sh", "dagger-u1-lag1", "2")
    assert refused.returncode == 1 and "phase 2's was pruned" in (run / "scratch" / "progress").read_text()
    (run / "train" / "p0002" / "adapter").mkdir(parents=True)
    (run / "train" / "p0002" / "DONE").write_text("{}")
    out = _run(scripts, env, "scratch.sh", "dagger-u1-lag1", "2")
    assert out.returncode == 0, out.stdout + out.stderr
    o = run / "scratch"
    # expandable segments for the training step alone: a vLLM server started under them fails at graph capture
    assert (home / "tests").read_text().splitlines() == [
        f"none {o}/continual/adapter scratch-dagger-u1-lag1-continual none {o}/weights dagger-u1-lag1 0,1,2,3,4,5,6,7",
        f"none {o}/model/adapter scratch-dagger-u1-lag1  {o}/frozen dagger-u1-lag1 0,1,2,3,4,5,6,7"]
    assert (home / "scratch").read_text().splitlines() == [
        f"expandable_segments:True run --no-sync python -m continual_arc_training.scratch train --run {o}/view "
        f"--phase 2 --out {o}/model --gpus 8", f"none run --no-sync python -m continual_arc_training.scratch summary {o}"]
    assert os.readlink(o / "continual") == str(run / "train" / "p0002")
    assert os.readlink(o / "view" / "play") == str(run / "play")
    assert all((o / f"{stage}.done").exists() for stage in ("weights", "train", "test", "summary"))
    assert (o / "progress").read_text().endswith("done\n")
