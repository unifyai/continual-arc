import copy
import itertools
import json
import shutil
from pathlib import Path

import pytest
from conftest import as_users

from continual_arc.aliases import TaskAliases
from continual_arc.env import EnvConfig, RequestDemos
from continual_arc.learner import BaseLearner
from continual_arc.runner import OPEN_ROUND, load_run, rescore, run
from continual_arc.schedule import (
    RANDOM,
    Schedule,
    Slot,
    epilogue_schedule,
    sequential_params,
    stream_schedule,
)
from continual_arc.testing import (
    ConcurrentRoundMemoryLearner,
    DemoHungryLearner,
    EchoLearner,
    RoundMemoryLearner,
    ShiftMemoryLearner,
)

CLOCK = ("wall_seconds",)  # what a record reads from the clock, which no two plays share


class Stopped(Exception):
    """The stop a test makes partway through a run, as a preemption or a crash stops one."""


def stopping(monkeypatch, learner_class, at: int) -> None:
    """``learner_class``'s ``act`` raising :class:`Stopped` at its ``at``-th call."""
    act, calls = learner_class.act, itertools.count(1)

    def stopped(self, obs):
        if next(calls) == at:
            raise Stopped
        return act(self, obs)

    monkeypatch.setattr(learner_class, "act", stopped)


def rows(path: Path) -> list[dict]:
    """The JSON lines of ``path``, less what a record reads from the clock."""
    out = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "resources" in row:
            row["resources"] = {k: v for k, v in row["resources"].items() if k not in CLOCK}
        out.append(row)
    return out


def cut_off_mid_write(run_dir: Path, whole: Path, users: int) -> int:
    """What a stop while the runner wrote the next round's records leaves in ``run_dir``: the first records of that
    round as ``whole`` (the same run never stopped) holds them, then a record and an event cut off before their newline.
    Returns how many records it added whole."""
    kept = len((run_dir / "records.jsonl").read_text().splitlines())
    added = (whole / "records.jsonl").read_text().splitlines()[kept:kept + max(1, users // 2)]
    with (run_dir / "records.jsonl").open("a") as fp:
        fp.write("".join(line + "\n" for line in added) + '{"step": 99, "task_id": "fak')
    with (run_dir / "events.jsonl").open("a") as fp:
        fp.write('{"step": 99, "rou')
    return len(added)


def test_memory_learner_pays_once_per_task_in_stream(fake_registry, fake_source, tmp_path):
    ids = fake_source.task_ids()
    sched = stream_schedule(ids, 300, seed=1, source=fake_source.name)
    learner = ShiftMemoryLearner()
    result = run(learner, sched, out_dir=tmp_path / "run", registry=fake_registry)
    recs = result.records
    assert len(recs) == 300
    assert all(r.solved for r in recs)
    firsts = [r for r in recs if r.exposure_index == 0]
    later = [r for r in recs if r.exposure_index > 0]
    assert len(firsts) == len(ids)
    assert all(r.cost == 1 and r.demo_requests == 1 for r in firsts)
    assert all(r.cost == 0 and r.demo_requests == 0 for r in later)
    s = result.summary
    assert s["acquisition"]["mean_cost"] == 1.0
    assert all(row["mean_cost"] in (None, 0.0) for row in s["retention_by_gap"])
    assert s["overall"]["score"] > 90
    # resources logged per instance
    assert recs[0].resources["calls"] == 2.0 and recs[-1].resources["calls"] == 1.0
    assert "wall_seconds" in recs[0].resources
    assert s["resources"]["calls"]["total"] == 300 + len(ids)
    # files written and reloadable
    sched2, cfg, recs2 = load_run(tmp_path / "run")
    assert len(sched2) == 300 and isinstance(cfg, EnvConfig) and len(recs2) == 300
    assert recs2[0].task_id == recs[0].task_id
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["overall"]["n"] == 300
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert meta["learner"] == "ShiftMemoryLearner"


def test_the_sequential_order_keeps_memory_across_tasks(fake_registry, fake_source):
    ids = fake_source.task_ids()[:4]
    sched = stream_schedule(ids, len(ids) * 3, seed=1, source=fake_source.name, params=sequential_params(3), order="sequential")
    learner = ShiftMemoryLearner()
    result = run(learner, sched, registry=fake_registry)
    assert [r.cost for r in result.records] == [1, 0, 0] * 4
    assert len(learner.memory) == 4  # nothing is wiped between tasks


def test_fresh_reference_resets_memory_before_each_task(fake_registry, fake_source):
    ids = fake_source.task_ids()[:4]
    sched = epilogue_schedule(ids, seed=1, source=fake_source.name, reset_each=True, instances_per_task=3)
    learner = ShiftMemoryLearner()
    result = run(learner, sched, registry=fake_registry)
    assert [r.cost for r in result.records] == [1, 0, 0] * 4
    assert len(learner.memory) == 1  # only the last task survives


def test_epilogue_with_reset_each(fake_registry, fake_source):
    ids = fake_source.task_ids()
    stream = stream_schedule(ids[:6], 60, seed=1, source=fake_source.name)
    epi = epilogue_schedule(ids[6:10], seed=1, source=fake_source.name, reset_each=True)
    learner = ShiftMemoryLearner()
    result = run(learner, stream.concat(epi), registry=fake_registry)
    epi_recs = [r for r in result.records if r.phase == "epilogue"]
    assert len(epi_recs) == 4 and all(r.cost == 1 for r in epi_recs)
    assert result.summary["epilogue"]["n"] == 4
    assert result.summary["stream"]["n"] == 60


def test_echo_learner_hits_cap(fake_registry, fake_source):
    sched = stream_schedule(fake_source.task_ids()[:2], 4, seed=0, source=fake_source.name, params=sequential_params(2), order="sequential")
    result = run(EchoLearner(), sched, config=EnvConfig(attempt_cap=3), registry=fake_registry)
    # never solved: every instance costs the whole budget, 3 attempts + 8 demos
    assert all(r.cost == EnvConfig(attempt_cap=3).max_cost == 11 and not r.solved for r in result.records)
    assert result.summary["overall"]["score"] == 0


def test_demo_hungry_learner_respects_cap(fake_registry, fake_source):
    sched = stream_schedule(fake_source.task_ids()[:1], 1, seed=0, source=fake_source.name, params=sequential_params(1), order="sequential")
    cfg = EnvConfig(attempt_cap=4, demo_cap=3)
    result = run(DemoHungryLearner(), sched, config=cfg, registry=fake_registry)
    rec = result.records[0]
    assert rec.demo_requests == 3 and rec.gave_up and rec.cost == 4 + 3


def test_runaway_learner_is_forced_to_give_up(fake_registry, fake_source):
    class Forever(BaseLearner):
        def act(self, obs):
            return RequestDemos()

    sched = stream_schedule(fake_source.task_ids()[:1], 2, seed=0, source=fake_source.name, params=sequential_params(2), order="sequential")
    cfg = EnvConfig(attempt_cap=2, demo_cap=2)
    result = run(Forever(), sched, config=cfg, registry=fake_registry)
    assert len(result.records) == 2
    for rec in result.records:
        assert rec.gave_up and rec.demo_requests == 2 and rec.cost == 4
        assert rec.resources["forced_give_up"] == 1.0
        assert rec.resources["actions"] == cfg.attempt_cap + cfg.demo_cap + 3


def test_events_log(fake_registry, fake_source, tmp_path):
    sched = stream_schedule(fake_source.task_ids()[:1], 1, seed=0, source=fake_source.name, params=sequential_params(1), order="sequential")
    run(ShiftMemoryLearner(), sched, out_dir=tmp_path / "r", registry=fake_registry, log_events=True)
    lines = (tmp_path / "r" / "events.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["action"]["type"] for line in lines]
    assert kinds == ["RequestDemos", "Submit"]
    assert json.loads(lines[1])["feedback"]["correct"] is True


@pytest.mark.parametrize("learner_class, users, at", [(ShiftMemoryLearner, 1, 23), (RoundMemoryLearner, 4, 13),
                                                     (ConcurrentRoundMemoryLearner, 4, 13)])
def test_a_run_stopped_partway_goes_on_exactly_as_one_never_stopped(fake_registry, fake_source, tmp_path, monkeypatch,
                                                                    learner_class, users, at):
    """A run stopped in the middle of a round, with the next round's records half written as a stop mid-write leaves
    them, goes on from its last whole round, its records cut back to it on disk and its learner's state as it was there:
    every record and every event is the run never stopped's, row for row, and so is the learner's state at the end."""
    sched = as_users(stream_schedule(fake_source.task_ids()[:6], 24, seed=4, source=fake_source.name), users)
    whole, broken = tmp_path / "whole", tmp_path / "broken"
    never = learner_class()
    run(never, sched, out_dir=whole, registry=fake_registry, log_events=True)
    broken.mkdir()
    shutil.copy(whole / "alias-salt", broken)   # the same names for the tasks, which the learners' memories hold
    stopping(monkeypatch, learner_class, at)
    with pytest.raises(Stopped):
        run(learner_class(), sched, out_dir=broken, registry=fake_registry, log_events=True)
    monkeypatch.undo()
    kept = len(rows(broken / "records.jsonl"))
    assert 0 < kept < 24 and kept % users == 0
    added = cut_off_mid_write(broken, whole, users)
    again = learner_class()
    result = run(again, sched, out_dir=broken, registry=fake_registry, log_events=True, resume=True)
    assert rows(broken / "records.jsonl") == rows(whole / "records.jsonl")
    played, unstopped = rows(broken / "events.jsonl"), rows(whole / "events.jsonl")
    assert (played if users == 1 else sorted(played, key=json.dumps)) == (
        unstopped if users == 1 else sorted(unstopped, key=json.dumps))   # users at once act in any order
    assert again.state() == never.state() and [r.step for r in result.records] == list(range(24))
    resumes = json.loads((broken / "meta.json").read_text())["resumed_at"]
    assert [(r["records"], r["dropped"]) for r in resumes] == [(kept, added)]


def test_a_resume_refuses_any_settings_but_the_runs_own_and_a_fresh_run_never_writes_over_one(fake_registry,
                                                                                              fake_source, tmp_path):
    ids = fake_source.task_ids()[:4]
    sched = stream_schedule(ids, 12, seed=4, source=fake_source.name)
    out = tmp_path / "run"
    run(ShiftMemoryLearner(), sched.first_rounds(6), out_dir=out, registry=fake_registry, settings={"memory": "shift"})
    with pytest.raises(ValueError, match="holds a run already"):
        run(ShiftMemoryLearner(), sched, out_dir=out, registry=fake_registry, settings={"memory": "shift"})
    for other, refused in (({"config": EnvConfig(attempt_cap=3)}, "config"), ({"alias_style": "names"}, "alias_style"),
                           ({"learner_name": "other"}, "learner"), ({"log_events": True}, "log_events"),
                           ({"max_actions_per_instance": 5}, "max_actions_per_instance"),
                           ({"settings": {"memory": "none"}}, "settings")):
        with pytest.raises(ValueError, match=f"differ from them: .*{refused}"):
            run(ShiftMemoryLearner(), sched, out_dir=out, registry=fake_registry, resume=True,
                **{"settings": {"memory": "shift"}, **other})
    with pytest.raises(ValueError, match="differ from them: schedule"):
        run(ShiftMemoryLearner(), stream_schedule(ids, 12, seed=5, source=fake_source.name), out_dir=out,
            registry=fake_registry, resume=True, settings={"memory": "shift"})
    assert json.loads((out / "meta.json").read_text()).get("resumed_at") is None
    # its own settings, under a schedule that goes on past its own: it plays on
    result = run(ShiftMemoryLearner(), sched, out_dir=out, registry=fake_registry, resume=True,
                 settings={"memory": "shift"})
    assert [r.step for r in result.records] == list(range(12))
    unstopped = run(ShiftMemoryLearner(), sched, registry=fake_registry)
    assert [(r.cost, r.solved) for r in result.records] == [(r.cost, r.solved) for r in unstopped.records]


def test_a_frozen_test_stopped_partway_plays_only_the_instances_it_had_not_ended(fake_registry, fake_source, tmp_path,
                                                                                monkeypatch):
    """A frozen test's instances all start from the learner as its stream left it, so a test stopped partway keeps
    every instance it had ended, and the relaunch, from a copy of the stream's learner, plays only the others: its
    records are the test never stopped's."""
    ids = fake_source.task_ids()
    stream = stream_schedule(ids[:3], 9, seed=1, source=fake_source.name)
    learner = RoundMemoryLearner()
    run(learner, stream, out_dir=tmp_path / "stream", registry=fake_registry)
    slots = [Slot(task, 700 + k, "seen-test", user=k) for k, task in enumerate(ids[:3] + ids[6:9])]
    test = Schedule(name="seen", seed=1, slots=slots, sources={t: fake_source.name for t in ids}, users=6,
                    meta={"test": "seen", "width": 6})
    out = tmp_path / "broken"
    for play in (tmp_path / "whole", out):   # each test under the names the stream gave its tasks
        play.mkdir()
        shutil.copy(tmp_path / "stream" / "alias-salt", play)
    never = run(copy.deepcopy(learner), test, out_dir=tmp_path / "whole", registry=fake_registry)
    # one action on every instance before a second on any: the stream's three tasks are solved at the first, the
    # others ask for a demonstration and answer at the second, so stopped at the fourth instance's answer the test has
    # ended the first three
    stopping(monkeypatch, RoundMemoryLearner, 6 + 1)
    with pytest.raises(Stopped):
        run(copy.deepcopy(learner), test, out_dir=out, registry=fake_registry)
    monkeypatch.undo()
    assert [json.loads(line)["step"] for line in (out / OPEN_ROUND).read_text().splitlines()] == [0, 1, 2]
    assert not (out / "records.jsonl").read_text()
    again = copy.deepcopy(learner)
    calls = again.calls
    result = run(again, test, out_dir=out, registry=fake_registry, resume=True)
    assert again.calls - calls == 3 * 2   # the three instances it had not ended, a demonstration and an answer each
    assert rows(out / "records.jsonl") == rows(tmp_path / "whole" / "records.jsonl")
    assert [(r.cost, r.solved) for r in result.records] == [(r.cost, r.solved) for r in never.records]
    assert not (out / OPEN_ROUND).read_text()
    assert json.loads((out / "meta.json").read_text())["resumed_at"][0]["finished"] == 3


def test_restore_rejects_a_mismatched_schedule(fake_registry, fake_source):
    from continual_arc.env import ContinualArcEnv

    ids = fake_source.task_ids()[:4]
    a = stream_schedule(ids, 10, seed=3, source=fake_source.name)
    b = stream_schedule(ids, 10, seed=4, source=fake_source.name)
    done = run(ShiftMemoryLearner(), a, registry=fake_registry).records[:5]
    env = ContinualArcEnv(b, registry=fake_registry)
    env.reset()
    with pytest.raises(ValueError):
        env.restore(done)


def test_collective_regime_runs_in_rounds(fake_registry, fake_source, tmp_path):
    from continual_arc.testing import RoundMemoryLearner

    ids = fake_source.task_ids()[:4]
    sched = stream_schedule(ids, 24, seed=2, source=fake_source.name, params=sequential_params(6), order="sequential")
    as_users(sched, 4)  # six rounds of four; each task fills a round and a half
    learner = RoundMemoryLearner()
    result = run(learner, sched, out_dir=tmp_path / "run", registry=fake_registry)
    recs = result.records
    assert [r.step for r in recs] == list(range(24)) and [r.round for r in recs] == [i // 4 for i in range(24)]
    assert learner.rounds_begun == learner.rounds_ended == 6
    # a task first met in a round costs a demo on every instance of that round, and nothing afterwards
    by_task_round: dict[tuple[str, int], list[float]] = {}
    for r in recs:
        by_task_round.setdefault((r.task_id, r.round), []).append(r.cost)
    for (tid, rnd), costs in by_task_round.items():
        first_round = min(rr for (t, rr) in by_task_round if t == tid)
        assert all(c == (1.0 if rnd == first_round else 0.0) for c in costs), (tid, rnd, costs)
    assert [r.exposure_index for r in recs if r.task_id == recs[0].task_id] == [0, 0, 0, 0, 1, 1]
    assert result.summary["acquisition"]["n"] == 12  # rounds of 4 over blocks of 6: tasks debut with 4, 2, 4, 2 instances
    assert all(r.resources["calls"] >= 1 for r in recs) and sum(r.resources["calls"] for r in recs) == learner.calls
    lines = (tmp_path / "run" / "records.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == list(range(24))


def test_records_are_written_by_whole_rounds_and_resume_replays_the_round(fake_registry, fake_source, tmp_path,
                                                                          monkeypatch):
    from continual_arc.testing import RoundMemoryLearner

    ids = fake_source.task_ids()[:3]
    sched = stream_schedule(ids, 12, seed=5, source=fake_source.name, params=RANDOM, order="random")
    as_users(sched, 4)
    act = RoundMemoryLearner.act

    def crash_in_round(self, obs):
        if obs.round == 1 and obs.step == 6:
            raise RuntimeError("interrupted")
        return act(self, obs)

    out = tmp_path / "run"
    monkeypatch.setattr(RoundMemoryLearner, "act", crash_in_round)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(RoundMemoryLearner(), sched, out_dir=out, registry=fake_registry)
    monkeypatch.undo()
    lines = (out / "records.jsonl").read_text().splitlines()
    assert len(lines) == 4  # only the finished round is on disk
    result = run(RoundMemoryLearner(), sched, out_dir=out, registry=fake_registry, resume=True)
    assert [r.step for r in result.records] == list(range(12))
    assert json.loads((out / "meta.json").read_text())["resumed_at"][0]["records"] == 4


def test_resume_plays_exactly_one_round_per_call_under_a_growing_cut(fake_registry, fake_source, tmp_path):
    """A small collective cut (2 users, 3 rounds, a stateless learner): calling run(resume=True) under
    the schedule cut at 1, then 2, then 3 rounds plays exactly one new round each time, and the result
    equals one uninterrupted 3-round run. This is how a driver in another repo plays a cut collective
    stream: run one round, train outside the runner, repeat under a schedule one round longer."""
    ids = fake_source.task_ids()[:2]
    full = as_users(
        stream_schedule(ids, 6, seed=7, source=fake_source.name, params=sequential_params(3), order="sequential"), 2
    )
    assert full.rounds()[-1] + 1 == 3

    out = tmp_path / "run"
    for r in (1, 2, 3):
        cut = full.first_rounds(r)
        result = run(EchoLearner(), cut, out_dir=out, registry=fake_registry, resume=True)
        assert len(result.records) == r * 2  # exactly r whole rounds of 2 users

    uninterrupted = run(EchoLearner(), full, registry=fake_registry)
    _, _, resumed_records = load_run(out)

    def key(records):
        return [(r.step, r.user, r.task_id, r.instance_seed, r.cost, r.solved) for r in records]

    assert key(resumed_records) == key(uninterrupted.records)
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["resumed_at"]) == 2  # the round-2 and round-3 calls each resumed; round-1 started fresh


def test_resume_works_under_a_schedule_cut_longer_than_the_last_run(fake_registry, fake_source, tmp_path):
    ids = fake_source.task_ids()[:3]
    full = as_users(
        stream_schedule(ids, 12, seed=5, source=fake_source.name, params=RANDOM, order="random"), 4
    )
    out = tmp_path / "run"
    run(ShiftMemoryLearner(), full.first_rounds(1), out_dir=out, registry=fake_registry)
    result = run(ShiftMemoryLearner(), full.first_rounds(3), out_dir=out, registry=fake_registry, resume=True)
    assert len(result.records) == 12 and [r.step for r in result.records] == list(range(12))


def test_concurrent_users_are_served_at_their_own_pace(fake_registry, fake_source, tmp_path):
    from continual_arc.testing import ConcurrentRoundMemoryLearner, RoundMemoryLearner

    ids = fake_source.task_ids()[:4]
    sched = as_users(stream_schedule(ids, 24, seed=2, source=fake_source.name, params=sequential_params(6),
                                     order="sequential"), 4)
    serial = run(RoundMemoryLearner(), sched, registry=fake_registry)
    learner = ConcurrentRoundMemoryLearner(delay_s=0.01)
    result = run(learner, sched, out_dir=tmp_path / "run", registry=fake_registry)
    recs = result.records
    # the round contract makes the outcome independent of who finished first
    assert [(r.step, r.cost, r.solved) for r in recs] == [(r.step, r.cost, r.solved) for r in serial.records]
    assert learner.max_inside > 1  # users were mid-action at the same time
    assert learner.rounds_begun == learner.rounds_ended == 6
    assert [r.user for r in recs] == [i % 4 for i in range(24)]
    # each instance is charged its own user's counters
    assert all(r.resources["calls"] >= 1 for r in recs) and sum(r.resources["calls"] for r in recs) == learner.calls
    assert all(r.resources["wall_seconds"] > 0 for r in recs)
    lines = (tmp_path / "run" / "records.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == list(range(24))
    assert result.summary["users"]["n"] == 4
    assert "users" not in run(RoundMemoryLearner(), stream_schedule(ids, 8, seed=2, source=fake_source.name),
                              registry=fake_registry).summary


def test_a_failing_user_stops_the_round_and_resume_replays_it(fake_registry, fake_source, tmp_path, monkeypatch):
    from continual_arc.testing import ConcurrentRoundMemoryLearner

    ids = fake_source.task_ids()[:3]
    sched = as_users(stream_schedule(ids, 12, seed=5, source=fake_source.name, params=RANDOM, order="random"), 4)
    act = ConcurrentRoundMemoryLearner.act

    def crash_in_round(self, obs):
        if obs.round == 1 and obs.user == 2:
            raise RuntimeError("interrupted")
        return act(self, obs)

    out = tmp_path / "run"
    monkeypatch.setattr(ConcurrentRoundMemoryLearner, "act", crash_in_round)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(ConcurrentRoundMemoryLearner(), sched, out_dir=out, registry=fake_registry)
    monkeypatch.undo()
    assert len((out / "records.jsonl").read_text().splitlines()) == 4  # only the finished round is on disk
    result = run(ConcurrentRoundMemoryLearner(), sched, out_dir=out, registry=fake_registry, resume=True)
    assert [r.step for r in result.records] == list(range(12))


def test_a_concurrent_learner_counts_per_user(fake_registry, fake_source):
    from continual_arc.testing import RoundMemoryLearner

    class NoCounters(RoundMemoryLearner):
        concurrent_users = True

    sched = as_users(stream_schedule(fake_source.task_ids()[:2], 4, seed=0, source=fake_source.name, params=RANDOM,
                                     order="random"), 2)
    with pytest.raises(TypeError, match="per user"):
        run(NoCounters(), sched, registry=fake_registry)
    single = stream_schedule(fake_source.task_ids()[:2], 4, seed=0, source=fake_source.name, params=RANDOM, order="random")
    assert len(run(NoCounters(), single, registry=fake_registry).records) == 4  # one user: the serial loop


def test_add_ons_need_the_individual_regime(fake_registry, fake_source):
    from continual_arc.addons import Addon

    class Quiet(Addon):
        name = "quiet"

    sched = stream_schedule(fake_source.task_ids()[:2], 4, seed=0, source=fake_source.name, params=RANDOM, order="random")
    as_users(sched, 2)
    with pytest.raises(ValueError, match="one user"):
        run(ShiftMemoryLearner(), sched, registry=fake_registry, addon=Quiet())


class _Recorder(ShiftMemoryLearner):
    """A ShiftMemoryLearner that keeps every observation it is shown."""

    def __init__(self) -> None:
        super().__init__()
        self.shown = []

    def act(self, obs):
        self.shown.append(obs)
        return super().act(obs)


def test_a_learner_is_shown_an_alias_and_nothing_about_the_stream(fake_registry, fake_source, tmp_path):
    ids = fake_source.task_ids()
    sched = stream_schedule(ids, 120, seed=1, source=fake_source.name)
    shown = _Recorder()
    a = run(shown, sched, out_dir=tmp_path / "run", registry=fake_registry)
    names = {o.task_id for o in shown.shown}
    assert all(n.startswith("task-") for n in names) and not names & set(ids)
    assert len(names) == len({r.task_id for r in a.records})   # one name per task, the same on every return
    for field in ("phase", "exposure_index", "user_exposure_index", "reset_before", "n_steps"):
        assert not hasattr(shown.shown[0], field)
    # the records keep the real ids, and the run's salt names each task as the learner saw it
    aliases = TaskAliases.for_run(tmp_path / "run")
    assert [aliases.alias(r.task_id) for r in a.records] == [o.task_id for o in shown.shown if o.is_fresh]
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert meta["alias_style"] == "hex" and "view" not in meta


def test_a_resumed_run_keeps_its_names(fake_registry, fake_source, tmp_path, monkeypatch):
    ids = fake_source.task_ids()
    sched = stream_schedule(ids, 60, seed=2, source=fake_source.name)
    whole = _Recorder()
    run(whole, sched, out_dir=tmp_path / "whole", registry=fake_registry)
    stopping(monkeypatch, _Recorder, 40)
    with pytest.raises(Stopped):
        run(_Recorder(), sched, out_dir=tmp_path / "run", registry=fake_registry)
    monkeypatch.undo()
    again = _Recorder()
    run(again, sched, out_dir=tmp_path / "run", registry=fake_registry, resume=True)
    salt = (tmp_path / "run" / "alias-salt").read_text()
    names = {(r["step"], TaskAliases(salt=salt).alias(r["task_id"])) for r in rows(tmp_path / "run" / "records.jsonl")}
    assert {(o.step, o.task_id) for o in again.shown} <= names and again.shown


def test_the_cli_passes_the_alias_style(tmp_path, monkeypatch):
    from continual_arc import cli

    seen = {}

    def fake_run(learner, sched, **kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr("continual_arc.runner.run", fake_run)
    sched = tmp_path / "s.json"
    stream_schedule(["fake000", "fake001"], 4, seed=0, source="fake").save(sched)
    with pytest.raises(SystemExit):
        cli.main(["run", "--learner", "continual_arc.testing:GiveUpLearner", "--schedule", str(sched), "--quiet",
                  "--alias-style", "names"])
    assert seen["alias_style"] == "names"


def test_continual_arc_run_resumes_a_run_with_its_own_settings_and_refuses_any_other(fake_registry, fake_source,
                                                                                     tmp_path, monkeypatch):
    """``continual-arc run --resume`` goes on with a run stopped partway as though it had never stopped, record for
    record and event for event; it refuses a directory that holds no run and any settings but the run's own, and a
    launch without it never writes over a run."""
    from continual_arc import cli

    monkeypatch.setattr("continual_arc.tasks.TaskRegistry.default", classmethod(lambda cls, *args: fake_registry))
    schedule = tmp_path / "schedule.json"
    stream_schedule(fake_source.task_ids()[:4], 20, seed=3, source=fake_source.name).save(schedule)
    common = ["run", "--learner", "continual_arc.testing:ShiftMemoryLearner", "--schedule", str(schedule), "--quiet",
              "--events"]
    whole, out = tmp_path / "whole", tmp_path / "run"
    assert cli.main([*common, "--out", str(whole)]) == 0
    out.mkdir()
    shutil.copy(whole / "alias-salt", out)
    with monkeypatch.context() as patched:
        stopping(patched, ShiftMemoryLearner, 17)
        with pytest.raises(Stopped):
            cli.main([*common, "--out", str(out)])
    stopped = (out / "records.jsonl").read_text()
    for argv, refusal in (([*common, "--out", str(out)], "holds a run already"),
                          ([*common, "--out", str(tmp_path / "none"), "--resume"], "holds none"),
                          ([*common, "--out", str(out), "--resume", "--attempt-cap", "3"], "config"),
                          ([*common, "--out", str(out), "--resume", "--alias-style", "names"], "alias_style"),
                          ([*common, "--out", str(out), "--resume", "--name", "other"], "learner"),
                          ([*common[:-1], "--out", str(out), "--resume"], "log_events")):
        with pytest.raises(SystemExit, match=refusal):
            cli.main(argv)
    assert (out / "records.jsonl").read_text() == stopped and "resumed_at" not in json.loads(
        (out / "meta.json").read_text())
    assert cli.main([*common, "--out", str(out), "--resume"]) == 0
    assert rows(out / "records.jsonl") == rows(whole / "records.jsonl")
    assert rows(out / "events.jsonl") == rows(whole / "events.jsonl")
    assert len(json.loads((out / "meta.json").read_text())["resumed_at"]) == 1


def test_rescore_charges_an_old_run_as_the_benchmark_charges_now(fake_registry, fake_source, tmp_path):
    """A run written under the earlier rule (an unsolved instance charged the attempt cap plus its demo requests) is
    recharged: every unsolved instance costs max_cost, the summary follows, meta.json says so, and a second pass
    changes nothing."""
    sched = stream_schedule(fake_source.task_ids()[:2], 4, seed=0, source=fake_source.name, params=sequential_params(2), order="sequential")
    out = tmp_path / "run"
    config = EnvConfig(attempt_cap=3)
    run(EchoLearner(), sched, config=config, out_dir=out, registry=fake_registry)
    lines = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    for d in lines:   # as the earlier rule charged them: the attempt cap, no demos asked
        d["cost"] = 3.0
    (out / "records.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines))
    assert rescore(out) == len(lines)
    _, _, records = load_run(out)
    assert all(r.cost == config.max_cost == 11 for r in records)
    assert json.loads((out / "summary.json").read_text())["overall"]["score"] == 0
    assert json.loads((out / "meta.json").read_text())["rescored"]["records_changed"] == len(lines)
    assert rescore(out) == 0


def test_report_shows_the_protocol_the_run_recorded(fake_registry, fake_source, tmp_path, capsys):
    """A run reads as the release that played it judged it: report prints the protocol block of its summary.json,
    under the configuration's name then, and judges only a run with no summary yet."""
    from continual_arc.cli import main

    sched = stream_schedule(fake_source.task_ids()[:2], 4, seed=0, source=fake_source.name)
    out = tmp_path / "run"
    run(EchoLearner(), sched, out_dir=out, registry=fake_registry)
    summary = json.loads((out / "summary.json").read_text())
    assert not summary["protocol"]["canonical"]   # a bare stream of fake tasks is no configuration of a track
    summary["protocol"] = {"track": "arc1", "config": "arc1-small", "mode": "workstreams", "seed": 0, "epilogue": True,
                           "canonical": True, "deviations": []}   # as a run played before v0.6.0 records it
    (out / "summary.json").write_text(json.dumps(summary))
    main(["report", str(out)])
    assert "protocol: arc1-small/workstreams seed 0   canonical: yes" in capsys.readouterr().out
    (out / "summary.json").unlink()
    main(["report", str(out)])
    assert "canonical: NO (configuration" in capsys.readouterr().out


def test_a_wide_round_can_be_served_a_few_users_at_a_time(fake_registry, fake_source):
    from continual_arc.testing import ConcurrentRoundMemoryLearner, RoundMemoryLearner

    ids = fake_source.task_ids()[:4]
    sched = as_users(stream_schedule(ids, 24, seed=2, source=fake_source.name, params=sequential_params(6),
                                     order="sequential"), 4)
    serial = run(RoundMemoryLearner(), sched, registry=fake_registry)
    learner = ConcurrentRoundMemoryLearner(delay_s=0.01)
    result = run(learner, sched, registry=fake_registry, max_concurrent=2)
    assert learner.max_inside <= 2   # never more than two users mid-action
    assert [(r.step, r.cost, r.solved) for r in result.records] == [(r.step, r.cost, r.solved) for r in serial.records]
