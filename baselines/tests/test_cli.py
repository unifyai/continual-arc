import json
from pathlib import Path

from continual_arc import __version__ as benchmark_version
from continual_arc.learner import BaseLearner
from continual_arc.schedule import stream_schedule
from continual_arc.testing import FakeSource

from continual_arc_baselines.attempt import Attempt
from continual_arc_baselines.cli import main

from .fixtures import Stopped, rows


def test_run_inside_a_drivers_attempt(tmp_path, monkeypatch, capsys):
    source = FakeSource(n_tasks=6)
    sched_path = tmp_path / "sched.json"
    stream_schedule(source.task_ids(), 20, seed=2, source=source.name).save(sched_path)
    context = Attempt.create("cli run", root=tmp_path / "wb")
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))

    code = main(["run", "--learner", "continual_arc.testing:ShiftMemoryLearner",
                 "--schedule", str(sched_path), "--no-proxy",
                 "--registry", "tests.fixtures:fake_registry", "--quiet"])
    assert code == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["inside_workbench"] is True
    assert outcome["run_dir"] == str(context.directory / "continual-arc")
    assert (context.directory / "continual-arc" / "summary.json").is_file()
    assert outcome["overall_score"] > 0
    events = [json.loads(line) for line in (context.directory / "events.jsonl").read_text().splitlines()]
    assert {"parameters", "metric", "artifact", "continual_arc_summary"} <= {e["kind"] for e in events}
    params = next(e["payload"]["values"] for e in events if e["kind"] == "parameters")
    # the keys by which runs are compared, present and scalar
    assert params["method"] == "continual_arc.testing:ShiftMemoryLearner"
    assert params["protocol"] == "workstreams/workstreams" and params["seed"] == 2
    assert params["benchmark"] == "continual-arc" and params["model"] is None
    assert set(params["dependencies"]) == {"continual-arc"}
    assert params["protocol_version"].startswith(f"{benchmark_version}+")
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
                     "--registry", "tests.fixtures:fake_registry", "--quiet"])
        assert code == 0
        outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert outcome["inside_workbench"] is False and outcome["llm"]["llm_requests"] == 6
        records = [json.loads(line) for line in Path(outcome["run_dir"]).joinpath("records.jsonl").read_text().splitlines()]
        assert all(r["resources"]["llm_requests"] == 1 and r["resources"]["llm_cost_usd"] > 0 for r in records)
    finally:
        upstream.stop()


def test_every_chat_completion_goes_to_the_learners_own_model(tmp_path, monkeypatch, capsys):
    """A learner named by its model has each chat completion sent to that model, whatever model the call names, and
    its run's parameters say so; a raw completion never leaves the proxy."""
    from .fake_upstream import FakeUpstream

    upstream = FakeUpstream().start()
    try:
        source = FakeSource(n_tasks=3)
        sched_path = tmp_path / "sched.json"
        stream_schedule(source.task_ids(), 4, seed=0, source=source.name).save(sched_path)
        monkeypatch.delenv("WORKBENCH_RUN_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        code = main(["run", "--learner", "tests.test_cli:StrongerModelLearner", "--learner-kwargs",
                     '{"model": "vendor/row-model"}', "--schedule", str(sched_path), "--upstream", upstream.base_url,
                     "--registry", "tests.fixtures:fake_registry", "--quiet"])
        assert code == 0
        outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert [(r["path"], r["body"]["model"]) for r in upstream.requests] == [
            ("/v1/chat/completions", "vendor/row-model")] * 4
        assert outcome["llm"]["llm_requests"] == 8 and outcome["llm"]["llm_failed_requests"] == 4
        log = Path(outcome["attempt_dir"]) / "events.jsonl"
        events = [json.loads(line) for line in log.read_text().splitlines()]
        params = next(e["payload"]["values"] for e in events if e["kind"] == "parameters")
        assert params["proxy"]["body_overrides"] == {"model": "vendor/row-model"}
    finally:
        upstream.stop()


class StrongerModelLearner(BaseLearner):
    """Names another model than its own in a chat completion each instance, then asks for a raw completion."""

    def __init__(self, model: str) -> None:
        self.model = model

    def act(self, obs):
        import os

        import httpx
        from continual_arc.env import GiveUp

        for path, body in (("/chat/completions", {"messages": [{"role": "user", "content": obs.task_id}]}),
                           ("/completions", {"prompt": obs.task_id})):
            httpx.post(os.environ["OPENAI_BASE_URL"] + path, timeout=10, json={"model": "vendor/stronger", **body})
        return GiveUp()


class EnvClientLearner(BaseLearner):
    """Calls the model once per instance through OPENAI_BASE_URL, then gives up."""

    def act(self, obs):
        import os

        import httpx
        from continual_arc.env import GiveUp

        httpx.post(os.environ["OPENAI_BASE_URL"] + "/chat/completions", timeout=10,
                   json={"model": "m", "messages": [{"role": "user", "content": obs.task_id}]})
        return GiveUp()



def test_a_resumed_runs_records_charge_every_instance_exactly_what_it_cost(tmp_path, monkeypatch):
    """Every record's model-call cost is the Decimal sum of what the provider charged for its instance, whatever the
    process spent before it: the same in a resumed process, which counts from nothing again, as in one never stopped,
    where the difference of a float running total would differ in its last digits."""
    import pytest

    from .fake_upstream import FakeUpstream

    upstream = FakeUpstream(cost=0.0007).start()
    try:
        source = FakeSource(n_tasks=3)
        sched_path = tmp_path / "sched.json"
        stream_schedule(source.task_ids(), 12, seed=0, source=source.name).save(sched_path)
        monkeypatch.delenv("WORKBENCH_RUN_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        common = ["run", "--learner", "tests.test_cli:EnvClientLearner", "--schedule", str(sched_path), "--upstream",
                  upstream.base_url, "--registry", "tests.fixtures:fake_registry", "--quiet"]
        whole, out = tmp_path / "whole", tmp_path / "run"
        assert main([*common, "--out", str(whole)]) == 0
        act, calls = EnvClientLearner.act, iter(range(1, 100))

        def stopped(self, obs):
            if next(calls) == 6:
                raise Stopped
            return act(self, obs)

        with monkeypatch.context() as patched:
            patched.setattr(EnvClientLearner, "act", stopped)
            with pytest.raises(Stopped):
                main([*common, "--out", str(out)])
        assert main([*common, "--out", str(out), "--resume"]) == 0
    finally:
        upstream.stop()
    assert rows(out / "records.jsonl") == rows(whole / "records.jsonl")
    assert {r["resources"]["llm_cost_usd"] for r in rows(whole / "records.jsonl")} == {0.0007}


def test_run_requires_exactly_one_schedule_source(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["run", "--learner", "continual_arc.testing:GiveUpLearner"])


def git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(["git", f"--git-dir={repo}", *args], check=True, capture_output=True, text=True).stdout


def test_a_harness_run_stopped_partway_goes_on_exactly_as_one_never_stopped(tmp_path, monkeypatch):
    """A harness stopped in the middle of an instance has written to its state what the instance taught it so far.
    ``--resume`` goes on in the run's own attempt, its state put back as the last round committed it and the instance
    played again: every record, every event and every round's committed state is the run's never stopped, and the
    stopped instance buys its demonstration again, which a harness that kept what the stop left would not."""
    import shutil

    import pytest
    from continual_arc.schedule import Schedule

    from tests.test_frozen import FileMemoryHarness

    source = FakeSource(n_tasks=6)
    sched_path = tmp_path / "sched.json"
    stream_schedule(source.task_ids()[:3], 12, seed=2, source=source.name).save(sched_path)
    infos = Schedule.load(sched_path).annotate()
    stop = [step for step, info in enumerate(infos) if info.exposure_index == 0][-1]   # the last task's first visit
    monkeypatch.chdir(tmp_path)
    common = ["run", "--learner", "tests.test_frozen:FileMemoryHarness", "--schedule", str(sched_path), "--no-proxy",
              "--registry", "tests.fixtures:fake_registry", "--quiet", "--events"]
    whole, out = tmp_path / "whole", tmp_path / "run"
    assert main([*common, "--out", str(whole)]) == 0
    out.mkdir()
    shutil.copy(whole / "alias-salt", out)
    act = FileMemoryHarness.act

    def stopped(self, obs):
        if obs.step == stop and not obs.is_fresh:   # its demonstration bought and written down, before its answer
            raise Stopped
        return act(self, obs)

    with monkeypatch.context() as patched:
        patched.setattr(FileMemoryHarness, "act", stopped)
        with pytest.raises(Stopped):
            main([*common, "--out", str(out)])
    system = Path(json.loads((out / "attempt.json").read_text())["attempt_dir"]) / "system"
    assert len(rows(out / "records.jsonl")) == stop and len(json.loads((system / "state" / "memory.json").read_text())) == 3
    launches = (out / "launches.jsonl").read_text()
    for argv, refusal in (([*common, "--out", str(out)], "holds a run already"),
                          ([*common, "--out", str(tmp_path / "none"), "--resume"], "holds none"),
                          ([*common, "--out", str(out), "--resume", "--learner-kwargs", '{"stateless": true}'],
                           "differ from them: settings"),
                          ([*common, "--out", str(out), "--resume", "--alias-style", "names"], "alias_style")):
        with pytest.raises(SystemExit, match=refusal):
            main(argv)
    assert (out / "launches.jsonl").read_text() == launches and not (tmp_path / "none").exists()
    assert main([*common, "--out", str(out), "--resume"]) == 0
    assert Path(json.loads((out / "attempt.json").read_text())["attempt_dir"]) / "system" == system
    assert rows(out / "records.jsonl") == rows(whole / "records.jsonl")
    assert rows(out / "events.jsonl") == rows(whole / "events.jsonl")
    assert rows(out / "records.jsonl")[stop]["demo_requests"] == 1
    unstopped = Path(json.loads((whole / "attempt.json").read_text())["attempt_dir"]) / "system"

    def trees(system: Path) -> list[str]:   # each round's committed state, by its content alone
        log = [json.loads(line)["commit"] for line in (system / "state-log.jsonl").read_text().splitlines()]
        return [git(system / "state.git", "rev-parse", f"{commit}^{{tree}}").strip() for commit in log]

    assert trees(system) == trees(unstopped) and len(trees(system)) == 12
    assert git(system / "state.git", "rev-list", "--count", "HEAD").strip() == "12"   # the stopped round left nothing
    launched = [json.loads(line) for line in (out / "launches.jsonl").read_text().splitlines()]
    assert [(entry["resume"], entry["prior_records"]) for entry in launched] == [(False, 0), (True, stop)]
    assert [entry["records"] for entry in json.loads((out / "meta.json").read_text())["resumed_at"]] == [stop]


def test_a_python_learners_run_resumes_with_the_state_it_kept(tmp_path, monkeypatch):
    """A learner that keeps its memory in its own process gives it to the runner at every round's end
    (``Learner.state``), and the resumed run's learner, made anew, takes it back: it plays on as the run never
    stopped, solving the tasks it met before the stop without a demonstration."""
    import shutil

    import pytest
    from continual_arc.testing import ShiftMemoryLearner

    source = FakeSource(n_tasks=6)
    sched_path = tmp_path / "sched.json"
    stream_schedule(source.task_ids(), 20, seed=2, source=source.name).save(sched_path)
    monkeypatch.chdir(tmp_path)
    common = ["run", "--learner", "continual_arc.testing:ShiftMemoryLearner", "--schedule", str(sched_path),
              "--no-proxy", "--registry", "tests.fixtures:fake_registry", "--quiet"]
    whole, out = tmp_path / "whole", tmp_path / "run"
    assert main([*common, "--out", str(whole)]) == 0
    out.mkdir()
    shutil.copy(whole / "alias-salt", out)
    act, calls = ShiftMemoryLearner.act, iter(range(1, 100))

    def stopped(self, obs):
        if next(calls) == 17:
            raise Stopped
        return act(self, obs)

    with monkeypatch.context() as patched:
        patched.setattr(ShiftMemoryLearner, "act", stopped)
        with pytest.raises(Stopped):
            main([*common, "--out", str(out)])
    kept = len(rows(out / "records.jsonl"))
    assert 0 < kept < 20
    assert main([*common, "--out", str(out), "--resume"]) == 0
    assert rows(out / "records.jsonl") == rows(whole / "records.jsonl")
    assert json.loads((out / "meta.json").read_text())["resumed_at"][0]["records"] == kept


def test_stream_only_plays_the_full_runs_stream_and_stops_before_its_epilogue():
    """arc1's real schedule in each order, over fake task ids (25 stream, 100 held out),
    so the test needs none of the pinned ARC data, which CI does not fetch."""
    from continual_arc.configs import build_schedule

    from continual_arc_baselines.cli import stream_of

    pool = FakeSource(n_tasks=125).task_ids()
    for order in ("workstreams", "sequential", "random"):
        full = build_schedule("arc1", 0, order=order, rearc_task_ids=pool[:25], eval_task_ids=pool[25:])
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
                 "--test", str(test), "--no-proxy", "--registry", "tests.fixtures:fake_registry",
                 "--quiet", "--out", str(out)])
    assert code == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    before = outcome["test"].pop("llm_before")  # the model calls as the stream ended, so that what fell outside both shows
    assert outcome["test"] == {"run_dir": str(out / "frozen" / "seen" / "play"), "test": "seen", "score": 100.0}
    assert before == outcome["llm"]  # no proxy, so no calls to count before the test or after it
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
              "--registry", "tests.fixtures:fake_registry", "--quiet", "--out", str(out)]
    with pytest.raises(SystemExit, match="not one round"):
        main(common + ["--test", str(test)])
    with pytest.raises(SystemExit, match="names no test"):
        main(common + ["--test", str(sched)])
    _, one_round = _stream_and_test(tmp_path, FakeSource(n_tasks=6), users=3)
    with pytest.raises(SystemExit, match="schedule has none"):   # a test after a test: no stream to follow
        main([*common[:4], str(one_round), *common[5:], "--test", str(one_round)])
    assert not out.exists()


def test_a_harness_plays_no_test_after_its_stream(tmp_path, monkeypatch):
    """A harness adapter, which holds one conversation, plays its tests with arc-baselines test rather than --test,
    refused before any instance plays."""
    import pytest

    sched, test = _stream_and_test(tmp_path, FakeSource(n_tasks=6), users=3)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="harness adapter"):
        main(["run", "--schedule", str(sched), "--no-proxy", "--registry", "tests.fixtures:fake_registry", "--quiet",
              "--learner", "continual_arc_baselines.systems.bare:BareModelLearner", "--learner-kwargs",
              '{"model": "m"}', "--test", str(test), "--out", str(tmp_path / "tested")])
    assert not (tmp_path / "tested" / "records.jsonl").exists()


def test_an_interrupt_stops_the_run_before_its_next_instance_with_its_records_written(tmp_path, monkeypatch, capsys):
    """The verifier stops a run that has spent its budget or passed its deadline with an interrupt. The runner plays its
    instances in threads of its own, so an exception in the main thread would leave them playing on to the stream's end."""
    import signal

    sched = tmp_path / "sched.json"
    stream_schedule(FakeSource(n_tasks=6).task_ids()[:3], 12, seed=2, source=FakeSource(n_tasks=6).name).save(sched)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run"
    code = main(["run", "--learner", "tests.fixtures:InterruptedLearner", "--learner-kwargs", '{"n": 4}', "--schedule",
                 str(sched), "--no-proxy", "--registry", "tests.fixtures:fake_registry", "--quiet", "--out", str(out)])
    assert code == 130 and "stopped after 4 instances, as asked" in capsys.readouterr().err
    assert len((out / "records.jsonl").read_text().splitlines()) == 4
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler   # the run's handler goes with it
