"""A run taken up again from the end of a round (fork.py): the new run carries that round's records, the old run's
task aliases and the state of that round's commit, its repository goes on from that commit, a new schedule may serve
anything after the carried rounds but not change them, and the command line plays on with the old launch's arguments."""

import json
import pickle
import subprocess
from pathlib import Path

import pytest
from continual_arc.runner import STATE
from continual_arc.schedule import Schedule, Slot

from continual_arc_baselines.fork import fork
from continual_arc_baselines.statelog import StateLog


def schedule(*tasks, phase="stream"):
    slots = [Slot(task, 100 + k, phase if k < 3 else "epilogue") for k, task in enumerate(tasks)]
    return Schedule(name="tiny", seed=0, slots=slots, sources={t: "fake" for t in tasks})


@pytest.fixture
def old_run(tmp_path, monkeypatch):
    """A run of three stream rounds and one epilogue round, its learner's state committed at the end of each."""
    monkeypatch.chdir(tmp_path)
    system = tmp_path / ".workbench" / "runs" / "adhoc-primeagentlearner-1792480000" / "attempts" / "a1" / "system"
    log = StateLog(system / "state", system / "state.git", system / "state-log.jsonl")
    for k in range(4):
        (system / "state" / "notes.md").write_text(f"after round {k}\n")
        log.commit()
    run = tmp_path / "run"
    run.mkdir()
    schedule("t1", "t2", "t1", "e1").save(run / "schedule.json")
    (run / "records.jsonl").write_text("".join(json.dumps({"step": k}) + "\n" for k in range(4)))
    (run / "events.jsonl").write_text("".join(json.dumps({"step": k, "action": a}) + "\n"
                                              for k in range(4) for a in ("RequestDemos", "Submit")))
    (run / "config.json").write_text(json.dumps({"attempt_cap": 8}))
    (run / "meta.json").write_text(json.dumps({"learner": "Learner", "resumed_at": [{"records": 2}]}))
    (run / "alias-salt").write_text("salt-of-the-old-run")
    (run / "attempt.json").write_text(json.dumps({"attempt_dir": str(system.parent)}))
    argv = ["run", "--learner", "pkg:Learner", "--config", "arc1", "--order", "workstreams", "--seed", "0",
            "--out", str(run), "--stop-after", "25", "--quiet"]
    (run / "launches.jsonl").write_text(json.dumps({"argv": argv}) + "\n")
    return run, log


def test_a_fork_starts_from_the_rounds_commit_and_plays_on_under_a_new_epilogue(old_run, tmp_path):
    run, log = old_run
    other_epilogue = schedule("t1", "t2", "t1", "e9")
    command = fork(run, 2, tmp_path / "forked", other_epilogue)
    forked = tmp_path / "forked"
    assert [json.loads(line)["step"] for line in (forked / "records.jsonl").read_text().splitlines()] == [0, 1, 2]
    assert (forked / "alias-salt").read_text() == "salt-of-the-old-run"
    assert Schedule.load(forked / "schedule.json").slots[3].task_id == "e9"
    system = Path(json.loads((forked / "attempt.json").read_text())["attempt_dir"]) / "system"   # from the checkout
    assert (system / "state" / "notes.md").read_text() == "after round 2\n"
    head = subprocess.run(["git", f"--git-dir={system / 'state.git'}", "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == log.commit_of(2)
    assert [json.loads(line)["round"] for line in (system / "state-log.jsonl").read_text().splitlines()] == [0, 1, 2]
    taken_up = StateLog(system / "state", system / "state.git", system / "state-log.jsonl")
    (system / "state" / "notes.md").write_text("after round 3, on the new epilogue\n")
    assert taken_up.rounds == 3 and taken_up.commit() != log.commit_of(3)
    assert json.loads((forked / "fork.json").read_text())["commit"] == log.commit_of(2)
    assert command == ["arc-baselines", "run", "--learner", "pkg:Learner", "--quiet",
                       "--schedule", str(forked / "schedule.json"), "--out", str(forked), "--resume"]
    # the old run's own settings, which its resume keeps, and its events of the carried rounds
    assert json.loads((forked / "meta.json").read_text()) == {"learner": "Learner"}
    assert (forked / "config.json").read_text() == (run / "config.json").read_text()
    assert (forked / "launches.jsonl").read_text() == (run / "launches.jsonl").read_text()
    assert [json.loads(line)["step"] for line in (forked / "events.jsonl").read_text().splitlines()] == [0, 0, 1, 1, 2, 2]
    # the carried rounds' end kept as the learner's state, which the resume gives back: round 2's commit
    kept = pickle.loads((forked / STATE).read_bytes())
    assert kept == {"records": 3, "state": {"rounds": 3, "commit": log.commit_of(2)}}


def test_a_fork_refuses_a_schedule_that_changes_what_it_carries(old_run, tmp_path):
    run, _ = old_run
    with pytest.raises(ValueError, match="exactly as"):
        fork(run, 2, tmp_path / "forked", schedule("t1", "t9", "t1", "e1"))
    with pytest.raises(KeyError):
        fork(run, 7, tmp_path / "forked-late")


def test_a_forked_run_plays_on_from_its_rounds_commit_as_the_old_run_played_on(tmp_path, monkeypatch):
    """A harness's run forked at the end of a round and played on with the command the fork prints, its schedule the
    old run's: every record and every event after the round is the old run's, and so is every round's committed
    state, since the fork's learner went on from the round's commit."""
    from continual_arc.schedule import stream_schedule
    from continual_arc.testing import FakeSource

    from continual_arc_baselines.cli import main

    from .fixtures import rows

    source = FakeSource(n_tasks=6)
    stream_schedule(source.task_ids()[:3], 12, seed=2, source=source.name).save(tmp_path / "sched.json")
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    assert main(["run", "--learner", "tests.test_frozen:FileMemoryHarness", "--schedule", str(tmp_path / "sched.json"),
                 "--no-proxy", "--registry", "tests.fixtures:fake_registry", "--quiet", "--events",
                 "--out", str(run)]) == 0
    command = fork(run, 5, tmp_path / "forked")
    assert main(command[1:]) == 0
    forked = tmp_path / "forked"
    assert rows(forked / "records.jsonl") == rows(run / "records.jsonl")
    assert rows(forked / "events.jsonl") == rows(run / "events.jsonl")

    def trees(out: Path) -> list[str]:
        system = Path(json.loads((out / "attempt.json").read_text())["attempt_dir"]) / "system"
        log = [json.loads(line)["commit"] for line in (system / "state-log.jsonl").read_text().splitlines()]
        return [subprocess.run(["git", f"--git-dir={system / 'state.git'}", "rev-parse", f"{commit}^{{tree}}"],
                               check=True, capture_output=True, text=True).stdout.strip() for commit in log]

    assert trees(forked) == trees(run) and len(trees(run)) == 12
    assert [entry["records"] for entry in json.loads((forked / "meta.json").read_text())["resumed_at"]] == [6]
