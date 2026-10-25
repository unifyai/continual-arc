"""The attempt directory a run writes (manifest, event log, metrics, artifacts) and its cost journal."""

import json
from decimal import Decimal

import pytest
from continual_arc.env import EnvConfig
from continual_arc.runner import run
from continual_arc.schedule import stream_schedule
from continual_arc.testing import FakeSource, GiveUpLearner, ShiftMemoryLearner

from continual_arc_baselines.attempt import Attempt
from continual_arc_baselines.cli import dependency_revisions
from continual_arc_baselines.costs import CostJournal, JournalCorruptionError

from .fixtures import fake_registry


def events(attempt):
    return [json.loads(line) for line in (attempt.directory / "events.jsonl").read_text().splitlines()]


@pytest.fixture
def attempt(tmp_path):
    return Attempt.create("test run", root=tmp_path / "wb")


def test_progress_and_finish_emit_metrics_and_artifacts(tmp_path):
    source = FakeSource(n_tasks=12)
    schedule = stream_schedule(source.task_ids(), 30, seed=1, source=source.name)
    config = EnvConfig()
    attempt = Attempt.create("test run", root=tmp_path / "wb", max_cost=config.max_cost)
    result = run(ShiftMemoryLearner(), schedule, config=config, out_dir=tmp_path / "run",
                 registry=fake_registry(), progress=attempt.progress)
    emitted = attempt.finish(result)

    metrics = [e["payload"] for e in events(attempt) if e["kind"] == "metric"]
    names = {m["name"] for m in metrics}
    assert {"instance/cost", "instance/solved", "running/mean_cost", "running/score",
            "instance/resource/calls", "summary/overall/score", "summary/stream/score"} <= names
    costs = [m for m in metrics if m["name"] == "instance/cost"]
    assert len(costs) == len(schedule) and [m["step"] for m in costs] == list(range(len(schedule)))
    # ShiftMemoryLearner solves everything after one demo request per task, so the
    # running score must climb and the final summary must match the run's own summary.
    running = [m["value"] for m in metrics if m["name"] == "running/score"]
    assert running[-1] > running[0]
    assert emitted["summary/overall/score"] == pytest.approx(result.summary["overall"]["score"])

    artifacts = [e["payload"] for e in events(attempt) if e["kind"] == "artifact"]
    assert {a["name"] for a in artifacts} == {"schedule.json", "config.json", "meta.json", "records.jsonl", "summary.json"}
    for a in artifacts:
        assert (attempt.directory / a["path"]).is_file()


def test_summary_metrics_skip_missing_phases(tmp_path, attempt):
    source = FakeSource(n_tasks=3)
    schedule = stream_schedule(source.task_ids(), 5, seed=0, source=source.name)  # no epilogue
    result = run(GiveUpLearner(), schedule, out_dir=tmp_path / "run", registry=fake_registry(),
                 progress=attempt.progress)
    emitted = attempt.finish(result)
    assert "summary/epilogue/score" not in emitted and "summary/stream/score" in emitted
    assert not any(m["name"] == "running/score" for m in
                   (e["payload"] for e in events(attempt) if e["kind"] == "metric"))


def test_a_run_makes_its_attempt_where_it_is_launched_unless_a_driver_names_one(tmp_path, monkeypatch, attempt):
    monkeypatch.delenv("WORKBENCH_RUN_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    made = Attempt.from_environment("Stateless GPT-5 / v2!")
    assert not made.given
    assert made.directory.parent.parent == tmp_path / ".workbench" / "runs" / made.run_id
    assert made.run_id.startswith("adhoc-stateless-gpt-5-v2-") and made.directory.name == made.attempt_id
    manifest = (made.directory / "manifest.json").read_bytes()
    assert b" " not in manifest and set(json.loads(manifest)) == {
        "experiment_id", "project_key", "recovery", "resume_performed", "run_id", "attempt_id", "created_at",
        "schema_version"}
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(attempt.directory))
    given = Attempt.from_environment("ignored")
    assert given.given and given.directory == attempt.directory and given.run_id == attempt.run_id


def test_events_are_canonical_lines_with_the_named_secrets_redacted(attempt, monkeypatch):
    monkeypatch.setenv("SOME_KEY", "sk-very-secret")
    monkeypatch.setenv("WORKBENCH_REDACT_NAMES", json.dumps(["SOME_KEY", "UNSET_KEY"]))
    attempt = Attempt(attempt.directory)
    attempt.emit("note", {"text": "key sk-very-secret in use", "sk-very-secret": ["sk-very-secret", 1]})
    line = (attempt.directory / "events.jsonl").read_text().splitlines()[-1]
    assert "sk-very-secret" not in line and line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))
    assert json.loads(line)["payload"] == {"text": "key [REDACTED] in use", "[REDACTED]": ["[REDACTED]", 1]}
    with (attempt.directory / "events.jsonl").open("a") as fh:
        fh.write('{"cut short')
    with pytest.raises(ValueError, match="cut short"):
        attempt.metric("x", 1.0)


def test_the_cost_journal_resolves_each_call_to_one_entity_and_refuses_a_cut_line(tmp_path):
    journal = CostJournal(tmp_path)
    for call, cost in (("a", "0.25"), ("b", None)):
        journal.record({"provider": "openrouter", "origin": "request_started", "request_attempt_id": call,
                        "currency": "USD", "unit": "openrouter_credits", "tokens": {}})
        journal.record({"provider": "openrouter", "origin": "response", "request_attempt_id": call,
                        "currency": "USD", "unit": "openrouter_credits", "account_charge": cost,
                        "tokens": {"input": 10, "output": 2}})
    written = [json.loads(line) for line in (tmp_path / "costs.jsonl").read_text().splitlines()]
    assert all(set(row["tokens"]) == {"input", "output", "cache_read", "cache_write", "reasoning"} for row in written)
    assert len({row["source_id"] for row in written}) == 4 and Decimal(written[1]["account_charge"]) == Decimal("0.25")
    summary = journal.summary()
    assert summary["records"] == 4 and summary["entities"] == 2
    assert summary["tokens"]["input"] == {"known_total": 20, "unknown_entities": 0}
    assert summary["charges"][0]["account_charge"] == {"known_total": "0.25", "unknown_entities": 1}
    with pytest.raises(ValueError, match="decimal"):
        journal.record({"provider": "openrouter", "origin": "response", "account_charge": "a lot"})
    with (tmp_path / "costs.jsonl").open("a") as fh:
        fh.write('{"cut')
    with pytest.raises(JournalCorruptionError):
        journal.record({"provider": "openrouter", "origin": "response"})


def test_dependency_revisions_name_the_installed_sources():
    revs = dependency_revisions()
    assert set(revs) == {"continual-arc"}
    for value in revs.values():
        assert value.startswith(("git+", "editable:")) or value[0].isdigit(), value
    assert dependency_revisions(("no-such-package",)) == {"no-such-package": "not-installed"}
