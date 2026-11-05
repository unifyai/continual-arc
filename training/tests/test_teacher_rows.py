"""The teacher's rows (continual_arc_training.teacher_rows and scripts/teacher-rows.sh) off any network: each stream instance
played from the teacher's commit of the round before, in a fork of its own whose folder paths are the user's, round 0 from
nothing and the tests from the last commit; the stream's parts as rounds of users who share no state; the student playing
the row each teacher's row maps to; and RUN/play, the parts read back as the stream."""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from continual_arc.schedule import Schedule, Slot

from continual_arc_training import frozen, play, references, teacher_rows

ROUNDS = 6
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _stream(rounds=ROUNDS) -> Schedule:
    return Schedule(name="stream", seed=0, users=1, sources={f"t{r % 2}": "rearc" for r in range(rounds)},
                    slots=[Slot(task_id=f"t{r % 2}", instance_seed=r, phase="stream", user=0) for r in range(rounds)])


def _teacher(root: Path, rounds=ROUNDS) -> Path:
    """A teacher's finished one-user run: its schedule, alias salt and one state committed at the end of every round, each
    round adding a line to its notes and every commit naming the folder the state lives in."""
    from continual_arc_baselines.statelog import StateLog
    teacher = root / "glm-h-u1"
    played = teacher / "play"
    system = played / "users" / "u000" / "system"
    state = system / "state"
    state.mkdir(parents=True)
    _stream(rounds).save(played / "schedule.json")
    (played / "alias-salt").write_text("salt\n")
    log = StateLog(state, system / "state.git", system / "state-log.jsonl")
    (state / "where").write_text(f"{system}/state/notes.md\n")
    for r in range(rounds):
        with (state / "notes.md").open("a") as fp:
            fp.write(f"round {r}\n")
        log.commit()
    return teacher


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(teacher_rows, "PART", 3)
    monkeypatch.setattr(teacher_rows, "ROUNDS", ROUNDS)


def test_a_part_is_one_round_of_users_serving_the_streams_slots_as_forks(small):
    part = teacher_rows.part_schedule(_stream(), 3)
    assert part.users == 3 and part.meta["forked"] is True and part.meta["first_round"] == 3
    assert [(s.user, s.task_id, s.instance_seed, s.reset_before) for s in part.slots] == [
        (0, "t1", 3, False), (1, "t0", 4, False), (2, "t1", 5, False)]
    with pytest.raises(SystemExit):
        teacher_rows.part_schedule(_stream(), 4)


def test_each_user_forks_the_state_the_teacher_held_before_that_instance(tmp_path):
    teacher = _teacher(tmp_path)
    play_dir = tmp_path / "qwen-hg-u1" / "frozen" / "stream-000" / "play"
    forked = teacher_rows.seed(play_dir, teacher, [None, 0, 3])
    log = [json.loads(line) for line in (teacher_rows.teacher_system(teacher) / "state-log.jsonl").read_text().splitlines()]
    assert (play_dir / "alias-salt").read_text() == "salt\n" and not (play_dir / "naming").exists()
    assert forked[0] == {"user": 0, "round": None, "commit": None}
    assert (forked[1]["round"], forked[1]["commit"]) == (0, log[0]["commit"])
    assert (forked[2]["round"], forked[2]["commit"]) == (3, log[3]["commit"])
    users = [play_dir / "users" / f"u{u:03d}" / "system" for u in range(3)]
    assert not (users[0] / "state" / "notes.md").exists()
    assert (users[1] / "state" / "notes.md").read_text() == "round 0\n"
    assert (users[2] / "state" / "notes.md").read_text() == "round 0\nround 1\nround 2\nround 3\n"
    # a user's own folder in place of the teacher's, and its log numbered up to the round it forked
    assert (users[2] / "state" / "where").read_text() == f"{users[2]}/state/notes.md\n"
    assert len((users[1] / "state-log.jsonl").read_text().splitlines()) == 1
    assert len((users[2] / "state-log.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("row,played_as", [("hg", "h"), ("hgpl", "h"), ("hplg", "hpl")])
def test_the_student_plays_the_row_each_teachers_row_maps_to_once(row, played_as, tmp_path, monkeypatch, small):
    teacher, run = _teacher(tmp_path), tmp_path / f"qwen-{row}-u1"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    calls = []

    def played(player, row_, play_dir, schedule, *, api_key):
        calls.append((player, row_, schedule.users, api_key))
        return SimpleNamespace(records=[SimpleNamespace(solved=True)] * schedule.users)

    monkeypatch.setattr(references, "play_round", played)
    args = SimpleNamespace(run=str(run), teacher=str(teacher), row=row, name="stream-003", first=3, test=None)
    teacher_rows.cmd_play(args)
    assert calls == [("qwen", played_as, 3, "k")]
    doc = json.loads((run / "frozen" / "stream-003" / "play" / "frozen.json").read_text())
    assert [u["round"] for u in doc["from_users"]] == [2, 3, 4] and doc["row"] == row and doc["player"] == "qwen"
    monkeypatch.setattr(play, "round_recorded", lambda *_: True)
    teacher_rows.cmd_play(args)
    assert len(calls) == 1


def test_a_round_0_part_starts_from_nothing_and_a_test_from_the_last_commit(tmp_path, monkeypatch, small):
    teacher, run = _teacher(tmp_path), tmp_path / "qwen-hg-u1"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(references, "play_round", lambda *a, **k: SimpleNamespace(records=[]))
    teacher_rows.cmd_play(SimpleNamespace(run=str(run), teacher=str(teacher), row="hg", name="stream-000", first=0, test=None))
    doc = json.loads((run / "frozen" / "stream-000" / "play" / "frozen.json").read_text())
    assert [u["round"] for u in doc["from_users"]] == [None, 0, 1]
    test = Schedule(name="seen", seed=0, users=2, sources={"t0": "rearc"}, meta={"test": "seen"},
                    slots=[Slot(task_id="t0", instance_seed=k, phase="seen-test", user=k) for k in range(2)])
    path = tmp_path / "seen.json"
    test.save(path)
    monkeypatch.setattr(frozen, "schedule_path", lambda name, users=1: path)
    teacher_rows.cmd_play(SimpleNamespace(run=str(run), teacher=str(teacher), row="hg", name="seen", first=None, test="seen"))
    doc = json.loads((run / "frozen" / "seen" / "play" / "frozen.json").read_text())
    assert [u["round"] for u in doc["from_users"]] == [ROUNDS - 1, ROUNDS - 1]


def _part(run: Path, first: int, seeds: list[int]) -> None:
    part = run / "frozen" / f"stream-{first:03d}" / "play"
    part.mkdir(parents=True)
    from continual_arc.env import EnvConfig
    (part / "config.json").write_text(json.dumps(EnvConfig().to_json()))
    (part / "driver.json").write_text(json.dumps({"row": "h", "player": "qwen"}))
    (part / "records.jsonl").write_text("".join(json.dumps(
        {"step": u, "task_id": f"t{seed % 2}", "phase": "stream", "source": "rearc", "exposure_index": 0, "gap": None,
         "instance_seed": seed, "reset_before": False, "solved": u == 0, "wrong_attempts": 0, "invalid_submissions": 0,
         "demo_requests": 1, "gave_up": False, "cost": 1.0, "resources": {}, "round": 0, "user": u}) + "\n"
        for u, seed in enumerate(seeds)))


def test_the_view_reads_the_parts_back_as_the_stream(tmp_path, monkeypatch, small):
    teacher, run = _teacher(tmp_path), tmp_path / "qwen-hg-u1"
    _part(run, 0, [0, 1, 2])
    _part(run, 3, [3, 4, 5])
    teacher_rows.cmd_view(SimpleNamespace(run=str(run), teacher=str(teacher)))
    records = [json.loads(line) for line in (run / "play" / "records.jsonl").read_text().splitlines()]
    assert [(r["step"], r["round"], r["user"], r["instance_seed"]) for r in records] == [(k, k, 0, k) for k in range(6)]
    assert [r["exposure_index"] for r in records] == [0, 0, 1, 1, 2, 2]
    assert json.loads((run / "play" / "driver.json").read_text())["teacher"] == str(teacher)
    assert (run / "play" / "alias-salt").read_text() == "salt\n" and (run / "play" / "summary.json").exists()
    wrong = tmp_path / "wrong"
    _part(wrong, 0, [0, 1, 2])
    _part(wrong, 3, [3, 4, 7])
    with pytest.raises(SystemExit):
        teacher_rows.cmd_view(SimpleNamespace(run=str(wrong), teacher=str(teacher)))


def test_teacher_rows_sh_plays_the_stream_then_the_tests_and_marks_them_done(tmp_path):
    home, tools = tmp_path / "home", tmp_path / "tools"
    tools.mkdir()
    (home / "runs" / "paper" / "glm-hpl-u1").mkdir(parents=True)
    (home / "baselines").mkdir()
    uv = tools / "uv"
    uv.write_text('#!/usr/bin/env bash\necho "$*" >> "$HOME/driver"\n')
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "HOME": str(home), "PATH": f"{tools}:{os.environ['PATH']}", "BASELINES": str(home / "baselines"),
           "ROW": "hplg", "GAP": "0", "PLAYS": "2", "RUNS": str(home / "runs")}
    script = SCRIPTS / "teacher-rows.sh"
    assert subprocess.run(["bash", str(script)], env=env, capture_output=True).returncode == 1   # no teacher yet
    (home / "runs" / "paper" / "glm-hpl-u1" / "play.done").touch()
    assert subprocess.run(["bash", str(script)], env=env, capture_output=True).returncode == 0
    lines = (home / "driver").read_text().splitlines()
    names = sorted(line.split("--name ")[1].split()[0] for line in lines if " play " in line)
    assert names == sorted(["stream-000", "stream-100", "held-out", "seen", "held-out-2", "seen-2"])
    assert all(f"--teacher {home}/runs/paper/glm-hpl-u1 --run {home}/runs/paper/qwen-hplg-u1" in line
               for line in lines if " play " in line)
    row = home / "runs" / "paper" / "qwen-hplg-u1"
    assert (row / "play.done").exists() and (row / "frozen" / "done").exists()
    assert subprocess.run(["bash", str(script)], env={**env, "ROW": "h"}, capture_output=True).returncode == 2
