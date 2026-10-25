import json
from pathlib import Path

from continual_arc.learner import BaseLearner
from continual_arc.schedule import stream_schedule
from continual_arc.testing import FakeSource
from continual_workbench.context import create_context

from continual_arc_baselines.cli import main


def test_run_inside_workbench_attempt(tmp_path, monkeypatch, capsys):
    source = FakeSource(n_tasks=6)
    sched_path = tmp_path / "sched.json"
    stream_schedule(source.task_ids(), 20, seed=2, source=source.name).save(sched_path)
    context = create_context(tmp_path / "wb", "cli-run", {"experiment_id": "t", "project_key": "t"})
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))

    code = main(["run", "--learner", "continual_arc.testing:ShiftMemoryLearner",
                 "--schedule", str(sched_path), "--no-proxy",
                 "--registry", "continual_arc_baselines.testing:fake_registry", "--quiet"])
    assert code == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["inside_workbench"] is True
    assert outcome["run_dir"] == str(context.directory / "continual-arc")
    assert (context.directory / "continual-arc" / "summary.json").is_file()
    assert outcome["overall_score"] > 0
    events = [json.loads(line) for line in (context.directory / "events.jsonl").read_text().splitlines()]
    assert {"parameters", "metric", "artifact", "continual_arc_summary"} <= {e["kind"] for e in events}
    params = next(e["payload"]["values"] for e in events if e["kind"] == "parameters")
    # Keys the workbench exports to MLflow must be present and scalar.
    assert params["method"] == "continual_arc.testing:ShiftMemoryLearner"
    assert params["protocol"] == "workstreams/workstreams" and params["seed"] == 2
    assert params["benchmark"] == "continual-arc" and params["model"] is None
    assert set(params["dependencies"]) == {"continual-arc"}
    assert params["protocol_version"].startswith("0.1.0+")
    assert params["proxy"] == {"enabled": False} and params["system"] == {}


def test_plain_learner_gets_proxy_counters_for_free(tmp_path, monkeypatch, capsys):
    """Any learner that builds a client from the environment is measured."""
    from .fake_upstream import FakeUpstream

    upstream = FakeUpstream().start()
    try:
        source = FakeSource(n_tasks=3)
        sched_path = tmp_path / "sched.json"
        stream_schedule(source.task_ids(), 6, seed=0, source=source.name).save(sched_path)
        monkeypatch.delenv("WORKBENCH_RUN_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        code = main(["run", "--learner", "tests.test_cli:EnvClientLearner",
                     "--schedule", str(sched_path), "--upstream", upstream.base_url,
                     "--registry", "continual_arc_baselines.testing:fake_registry", "--quiet"])
        assert code == 0
        outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert outcome["inside_workbench"] is False and outcome["llm"]["llm_requests"] == 6
        records = [json.loads(line) for line in Path(outcome["run_dir"]).joinpath("records.jsonl").read_text().splitlines()]
        assert all(r["resources"]["llm_requests"] == 1 and r["resources"]["llm_cost_usd"] > 0 for r in records)
    finally:
        upstream.stop()


class EnvClientLearner(BaseLearner):
    """Calls the model once per instance through OPENAI_BASE_URL, then gives up."""

    def act(self, obs):
        import os

        import httpx
        from continual_arc.env import GiveUp

        httpx.post(os.environ["OPENAI_BASE_URL"] + "/chat/completions", timeout=10,
                   json={"model": "m", "messages": [{"role": "user", "content": obs.task_id}]})
        return GiveUp()



def test_run_requires_exactly_one_schedule_source(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["run", "--learner", "continual_arc.testing:GiveUpLearner"])


def test_resume_continues_in_the_same_attempt(tmp_path, monkeypatch, capsys):
    import pytest

    source = FakeSource(n_tasks=6)
    sched_path = tmp_path / "sched.json"
    stream_schedule(source.task_ids(), 20, seed=2, source=source.name).save(sched_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "wb" / "stage1" / "cell"
    common = ["run", "--schedule", str(sched_path), "--no-proxy", "--registry",
              "continual_arc_baselines.testing:fake_registry", "--quiet", "--out", str(out), "--events"]
    with pytest.raises(RuntimeError, match="interrupted"):
        main(common + ["--learner", "continual_arc_baselines.testing:CrashAfterLearner", "--learner-kwargs", '{"n": 5}'])
    assert len((out / "records.jsonl").read_text().splitlines()) == 5
    attempt = json.loads((out / "attempt.json").read_text())["attempt_dir"]
    code = main(common + ["--learner", "continual_arc.testing:ShiftMemoryLearner", "--resume"])
    assert code == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["attempt_dir"] == attempt, "the resumed run keeps the attempt (and the system's state) of the first"
    assert len((out / "records.jsonl").read_text().splitlines()) == 20
    assert json.loads((out / "meta.json").read_text())["resumed_at"][0]["records"] == 5
    assert json.loads((out / "summary.json").read_text())["overall"]["n"] == 20
    # nothing to resume: a plain run in a fresh attempt
    code = main(common[:-3] + ["--out", str(tmp_path / "other"), "--learner", "continual_arc.testing:ShiftMemoryLearner", "--resume"])
    assert code == 0


def test_stream_only_plays_the_full_runs_stream_and_stops_before_its_epilogue():
    """arc1-small's real schedule in each mode, over fake task ids (25 stream, 100 held out),
    so the test needs none of the pinned ARC data, which CI does not fetch."""
    from continual_arc.configs import build_schedule

    from continual_arc_baselines.cli import stream_of

    pool = FakeSource(n_tasks=125).task_ids()
    for mode in ("workstreams", "sequential", "random"):
        full = build_schedule("arc1-small", mode, seed=0, rearc_task_ids=pool[:25], eval_task_ids=pool[25:])
        stream = stream_of(full)
        streamed = [s for s in full.slots if s.phase == "stream"]
        assert stream.slots == full.slots[:len(streamed)] and {s.phase for s in stream.slots} == {"stream"}
        assert stream_of(stream).slots == stream.slots


def _stream_and_test(tmp_path, source, users: int, rounds: int = 1):
    """A stream over the first three tasks with an epilogue over two others, and a seen test of the stream's tasks at
    instances of their own laid out as ``rounds`` rounds of ``users``."""
    from continual_arc.schedule import Schedule, Slot, epilogue_schedule

    ids = source.task_ids()
    stream = stream_schedule(ids[:3], 12, seed=2, source=source.name)
    full = stream.concat(epilogue_schedule(ids[3:5], 2, source.name), name=stream.name)
    full.save(tmp_path / "sched.json")
    slots = [Slot(ids[k % 3], 900 + k, "seen-test", user=k % users) for k in range(users * rounds)]
    test = Schedule(name="tiny-seen-test", seed=2, users=users, slots=slots, sources={t: source.name for t in ids[:3]},
                    meta={"test": "seen", "width": users})
    test.save(tmp_path / "test.json")
    return tmp_path / "sched.json", tmp_path / "test.json"


def test_a_test_after_the_stream_is_played_by_the_same_learner_from_the_state_its_stream_left(tmp_path, monkeypatch,
                                                                                              capsys):
    """The learner plays the stream, its epilogue cut off, then the test in a directory of its own under the run's,
    holding the run's task names: every test instance is solved from the memory the stream left, no demonstration
    bought, which a learner from nothing or one shown other names could not do."""
    sched, test = _stream_and_test(tmp_path, FakeSource(n_tasks=6), users=3)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    code = main(["run", "--learner", "continual_arc.testing:ShiftMemoryLearner", "--schedule", str(sched),
                 "--test", str(test), "--no-proxy", "--registry", "continual_arc_baselines.testing:fake_registry",
                 "--quiet", "--out", str(out)])
    assert code == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["test"] == {"run_dir": str(out / "frozen" / "seen" / "play"), "test": "seen", "score": 100.0}
    stream = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert len(stream) == 12 and {r["phase"] for r in stream} == {"stream"}
    tested = [json.loads(line) for line in (out / "frozen" / "seen" / "play" / "records.jsonl").read_text().splitlines()]
    assert [(r["user"], r["round"], r["phase"], r["solved"], r["cost"]) for r in tested] == [
        (u, 0, "seen-test", True, 0.0) for u in range(3)]
    assert (out / "frozen" / "seen" / "play" / "alias-salt").read_text() == (out / "alias-salt").read_text()


def test_a_test_that_is_not_one_round_is_refused_before_anything_plays(tmp_path, monkeypatch):
    import pytest

    sched, test = _stream_and_test(tmp_path, FakeSource(n_tasks=6), users=1, rounds=3)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    common = ["run", "--learner", "continual_arc.testing:ShiftMemoryLearner", "--schedule", str(sched), "--no-proxy",
              "--registry", "continual_arc_baselines.testing:fake_registry", "--quiet", "--out", str(out)]
    with pytest.raises(SystemExit, match="not one round"):
        main(common + ["--test", str(test)])
    with pytest.raises(SystemExit, match="names no test"):
        main(common + ["--test", str(sched)])
    _, one_round = _stream_and_test(tmp_path, FakeSource(n_tasks=6), users=3)
    with pytest.raises(SystemExit, match="schedule has none"):   # a test after a test: no stream to follow
        main([*common[:4], str(one_round), *common[5:], "--test", str(one_round)])
    assert not out.exists()


def test_log_state_is_for_a_system_learner_and_a_harness_plays_no_test_after_its_stream(tmp_path, monkeypatch):
    """--log-state commits a SystemLearner's state directory, so a learner of another kind is refused; a harness
    adapter, which holds one conversation, plays its tests with arc-baselines test rather than --test. Both are refused
    before any instance plays."""
    import pytest

    sched, test = _stream_and_test(tmp_path, FakeSource(n_tasks=6), users=3)
    monkeypatch.chdir(tmp_path)
    common = ["run", "--schedule", str(sched), "--no-proxy", "--registry",
              "continual_arc_baselines.testing:fake_registry", "--quiet"]
    with pytest.raises(SystemExit, match="no SystemLearner"):
        main(common + ["--learner", "continual_arc.testing:ShiftMemoryLearner", "--log-state",
                       "--out", str(tmp_path / "logged")])
    with pytest.raises(SystemExit, match="harness adapter"):
        main(common + ["--learner", "continual_arc_baselines.systems.bare:BareModelLearner",
                       "--learner-kwargs", '{"model": "m"}', "--test", str(test), "--out", str(tmp_path / "tested")])
    assert not any((tmp_path / out / "records.jsonl").exists() for out in ("logged", "tested"))

