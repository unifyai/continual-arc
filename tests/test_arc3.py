"""The ARC-AGI-3 track: its games and split, configurations and frozen tests, the instance draw and its cache,
demonstrations and charges in the environment, and the measures read on its records.

Tests that play levels use the real engine on the pinned games, with the small generator of
``tests/arc3_generators/sp80.py`` in place of the track's own."""

import json
import math

import pytest
from conftest import needs_arc3

from continual_arc.arc3.env import (
    ActFeedback,
    Arc3Config,
    Arc3Env,
    Arc3LearnerObservation,
    DemonstrationFeedback,
    ResetFeedback,
    ResetLevel,
)
from continual_arc.arc3.games import (
    GAMES,
    HELD_OUT_GAMES,
    SOURCE_ARC3,
    STREAM_GAMES,
    TRACK_GAMES,
    split,
)
from continual_arc.arc3.instances import (
    Act,
    Demonstration,
    DemoStep,
    LevelSource,
    changes,
    precompute,
)
from continual_arc.configs import (
    FRESH,
    SEEN_TEST_PER_TASK,
    TRACKS,
    BenchmarkConfig,
    build_schedule,
    canonical_protocol,
    get_config,
)
from continual_arc.configs import test_schedule as frozen_test
from continual_arc.env import EnvConfig, GiveUp, GiveUpFeedback, RequestDemos
from continual_arc.learner import BaseLearner
from continual_arc.metrics import share_learned, transfer
from continual_arc.report import text_report
from continual_arc.runner import build_env, load_run, run
from continual_arc.schedule import Schedule, Slot
from continual_arc.seeds import derive_seed

GENERATORS = {"sp80": "arc3_generators.sp80"}
SP80_BASELINE = 39   # ARC Prize's human action count for sp80's level 0
SP80_LEVEL0 = 4      # levels.bfs's solution of it


def schedule(n: int, phase: str = "stream", users: int = 1, seed0: int = 100) -> Schedule:
    slots = [Slot("sp80", seed0 + i, phase, user=i % users) for i in range(n)]
    return Schedule(name="sp80", seed=0, slots=slots, sources={"sp80": SOURCE_ARC3}, users=users,
                    meta={"dataset": "arc3", "track": "arc3"})


@pytest.fixture
def source(tmp_path):
    return LevelSource(GENERATORS, cache=tmp_path / "solutions")


class Oracle(BaseLearner):
    """Plays each instance's own solution, read from the environment, after buying ``demos`` demonstrations."""

    def __init__(self, env: Arc3Env, demos: int = 0):
        self.env, self.demos = env, demos
        self.played: dict[int, int] = {}

    def act(self, obs):
        assert isinstance(obs, Arc3LearnerObservation) and not obs.task_id.startswith("sp80")
        if obs.demo_requests_used < self.demos:
            return RequestDemos()
        k = self.played.get(obs.step, 0)
        self.played[obs.step] = k + 1
        return self.env._open[obs.step].drawn.solution[k]


class Idle(BaseLearner):
    """Moves the bar back and forth, never pouring: reaches the cap."""

    def act(self, obs):
        return Act(3 if obs.actions_used % 2 else 4)


# -- configuration, games and the split ------------------------------------------------------------------------------


def test_config_charges_and_json():
    cfg = Arc3Config()
    assert cfg.max_cost == 5 + 3 == 8 and cfg.to_json() == {"cap_multiplier": 5.0, "max_actions": 1000,
                                                            "demo_cap": 3, "demo_cost": 1.0}
    assert Arc3Config.from_json(cfg.to_json()) == cfg
    assert cfg.action_cap(22) == 110 and cfg.action_cap(58.5) == 293 and cfg.action_cap(400) == 1000
    with pytest.raises(ValueError):
        Arc3Config(cap_multiplier=0).validate()
    with pytest.raises(ValueError):
        Arc3Config(demo_cap=-1).validate()


def test_the_track_s_games_split_once_into_stream_and_held_out_games():
    assert set(TRACK_GAMES) <= set(GAMES) and len(GAMES) == 25
    assert set(STREAM_GAMES) | set(HELD_OUT_GAMES) == set(TRACK_GAMES) and not set(STREAM_GAMES) & set(HELD_OUT_GAMES)
    assert len(STREAM_GAMES) == round(0.6 * len(TRACK_GAMES))
    assert split() == (STREAM_GAMES, HELD_OUT_GAMES) and split(TRACK_GAMES, seed=1) != split()
    assert "arc3" in TRACKS


def test_arc3_small_serves_every_stream_game_eight_times_and_tests_four_fresh_levels_of_each():
    cfg = get_config("arc3-small")
    assert cfg.track == "arc3" and isinstance(cfg.env, Arc3Config) and cfg.stream_source == SOURCE_ARC3
    stream = build_schedule(cfg, 0, epilogue=False)
    counts = {g: sum(s.task_id == g for s in stream.slots) for g in STREAM_GAMES}
    assert set(counts.values()) == {8} and len(stream) == 8 * len(STREAM_GAMES)
    assert stream.meta["dataset"] == "arc3" and stream.meta["mode"] == "workstreams"
    assert set(stream.sources.values()) == {SOURCE_ARC3}
    seen = frozen_test(cfg, 0, "seen")
    assert sorted(s.task_id for s in seen.slots) == sorted(g for g in STREAM_GAMES for _ in range(SEEN_TEST_PER_TASK))
    held = frozen_test(cfg, 0, "held-out")
    assert sorted(s.task_id for s in held.slots) == sorted(g for g in HELD_OUT_GAMES for _ in range(4))
    assert held.users == len(held.slots) and set(held.rounds()) == {0}   # one round, every instance a user
    stream_seeds = {s.instance_seed for s in stream.slots}
    assert not stream_seeds & {s.instance_seed for s in seen.slots}   # fresh levels, not the stream's
    # the held-out test is the epilogue's instances, and the fresh reference wipes the learner before every one
    epilogue = [s for s in build_schedule(cfg, 0).slots if s.phase == "epilogue"]
    assert {(s.task_id, s.instance_seed) for s in epilogue} == {(s.task_id, s.instance_seed) for s in held.slots}
    fresh = build_schedule(cfg, 0, order=FRESH)
    assert len(fresh) == 4 * len(HELD_OUT_GAMES) and all(s.reset_before for s in fresh.slots)
    assert canonical_protocol("arc3")["config"] == "arc3-small"
    assert frozen_test(cfg, 0, "seen", width=1).users == 1


def test_a_track_s_configuration_takes_its_own_environment_settings():
    with pytest.raises(ValueError, match="Arc3Config"):
        BenchmarkConfig(name="x", n_tasks=1, stream_length=1, n_epilogue_tasks=0, dataset="arc3")
    with pytest.raises(ValueError, match="EnvConfig"):
        BenchmarkConfig(name="x", n_tasks=1, stream_length=1, n_epilogue_tasks=0, env=Arc3Config())
    sched = build_schedule("arc3-smoke", 0)
    assert isinstance(build_env(sched).config, Arc3Config)
    with pytest.raises(TypeError):
        build_env(sched, EnvConfig())


def test_cli_reads_an_arc3_schedule_s_settings():
    from types import SimpleNamespace

    from continual_arc.cli import _env_config

    sched = build_schedule("arc3-smoke", 0)
    args = SimpleNamespace(cap_multiplier=None, max_actions=40, demo_cap=None, demo_cost=None, attempt_cap=3)
    assert _env_config(args, sched) == Arc3Config(max_actions=40)


def test_changes_and_demonstrations_round_trip():
    a = [[0, 0], [0, 0]]
    b = [[0, 5], [7, 0]]
    assert changes(a, b) == ((1, 0, 5), (0, 1, 7))
    demo = Demonstration(a, (DemoStep(Act(6, 1, 0), ((1, 0, 5),)),))
    assert demo.final == [[0, 5], [0, 0]] and demo.frames()[0] == a
    assert Demonstration.from_json(json.loads(json.dumps(demo.to_json()))) == demo
    assert Act.from_json(Act(6, 3, 4).to_json()) == Act(6, 3, 4) and Act(2).to_json() == [2]


# -- the pinned games ------------------------------------------------------------------------------------------------


@needs_arc3
def test_the_pinned_games_are_where_the_fetch_puts_them():
    from continual_arc.arc3 import levels
    from continual_arc.data import status

    st = status(("arc3",))
    assert len(st) == 25 and all(s["ok"] for s in st.values())
    assert levels.game_file("ls20").parent.name == GAMES["ls20"] == st["arc-agi-3/ls20"]["pinned"]
    assert levels.baseline("sp80") == SP80_BASELINE


# -- the instance draw and its cache ---------------------------------------------------------------------------------


@needs_arc3
def test_an_instance_seed_draws_one_level_solved_in_the_band_and_scales_the_baseline(source, tmp_path):
    from continual_arc.arc3 import levels

    drawn = source.instance("sp80", 7)
    again = LevelSource(GENERATORS, cache=tmp_path / "elsewhere").instance("sp80", 7)
    assert levels.layout(drawn.level) == levels.layout(again.level) and drawn.solution == again.solution
    assert source.level0_length("sp80") == SP80_LEVEL0
    assert SP80_LEVEL0 * 0.5 <= len(drawn.solution) <= SP80_LEVEL0 * 2
    assert drawn.baseline == pytest.approx(SP80_BASELINE * len(drawn.solution) / SP80_LEVEL0)
    assert levels.replay("sp80", drawn.level, [a.engine() for a in drawn.solution])
    assert drawn.layout != levels.layout(levels.level0("sp80"))
    layouts = {levels.layout(source.instance("sp80", s).level) for s in range(12)}
    assert len(layouts) > 6


@needs_arc3
def test_a_solution_is_solved_once_and_read_from_the_cache_after(source, monkeypatch):
    from continual_arc.arc3 import levels

    drawn = source.instance("sp80", 3)
    files = sorted(source.folder("sp80").glob("*.json"))
    assert {f.stem for f in files} >= {"level0"} and len(files) >= 2
    assert source.folder("sp80").parent.name == GAMES["sp80"]   # keyed by the game's version, then the generator

    def no_search(*args, **kwargs):
        raise AssertionError("solved again")

    monkeypatch.setattr(levels, "bfs", no_search)
    fresh = LevelSource(GENERATORS, cache=source.cache)
    assert fresh.instance("sp80", 3).solution == drawn.solution
    # a generator that draws another layout from the same seed is caught, not served a stale solution
    for f in files:
        if f.stem != "level0":
            kept = json.loads(f.read_text())
            kept["layout"] = "0" * 16
            f.write_text(json.dumps(kept))
    with pytest.raises(RuntimeError, match="not deterministic"):
        LevelSource(GENERATORS, cache=source.cache).instance("sp80", 3)


@needs_arc3
def test_precompute_solves_every_instance_and_demonstration_ahead(source, monkeypatch):
    from continual_arc.arc3 import levels

    instances = [("sp80", s) for s in range(4)]
    assert precompute(source, instances, demos=2, workers=2) == 4

    def no_search(*args, **kwargs):
        raise AssertionError("a run waited on the solver")

    monkeypatch.setattr(levels, "bfs", no_search)
    warm = LevelSource(GENERATORS, cache=source.cache)
    for _, seed in instances:
        drawn = warm.instance("sp80", seed)
        for k in range(2):
            assert warm.demonstration("sp80", seed, k, drawn.layout).steps


@needs_arc3
def test_a_demonstration_is_the_solver_s_play_of_another_fresh_level(source):
    from continual_arc.arc3 import levels

    drawn = source.instance("sp80", 11)
    demos = [source.demonstration("sp80", 11, k, drawn.layout) for k in range(3)]
    for demo in demos:
        frames = demo.frames()
        assert len(frames) == len(demo.steps) + 1 and len(frames[0]) == 64 and len(frames[0][0]) == 64
        assert all(step.changes for step in demo.steps)
    assert demos[0] == source.demonstration("sp80", 11, 0, drawn.layout)   # the same seed, the same demonstration
    assert len({json.dumps(d.to_json()) for d in demos}) > 1
    level = source.draw("sp80", derive_seed("arc3-demo", 11, 0), exclude=(drawn.layout,)).level
    assert levels.layout(level) != drawn.layout
    assert levels.replay("sp80", level, [step.action.engine() for step in demos[0].steps])
    assert demos[0].frames()[0] == levels.new_game("sp80", level)[1].frame[-1]


# -- charges -----------------------------------------------------------------------------------------------------------


@needs_arc3
def test_a_solved_instance_costs_its_play_in_baselines_plus_its_demonstrations(source):
    env = Arc3Env(schedule(2), source=source)
    obs = env.reset()
    assert obs.is_fresh and len(obs.frames) == 1 and obs.state == "NOT_FINISHED"
    assert obs.max_cost == 8 and obs.demo_cap == 3 and obs.action_cap == math.ceil(5 * obs.baseline_actions)
    solution = env._open[0].drawn.solution
    for act in solution:
        res = env.step(act)
        assert isinstance(res.feedback, ActFeedback) and res.feedback.valid and res.feedback.rendered
    assert res.instance_done and res.feedback.level_completed and res.record.solved
    rec = res.record
    assert rec.cost == pytest.approx(len(solution) / obs.baseline_actions) == pytest.approx(SP80_LEVEL0 / SP80_BASELINE)
    assert rec.wrong_attempts == len(solution) and rec.demo_requests == 0 and rec.metadata["resets"] == 0
    assert rec.metadata["solution_length"] == len(solution) and rec.source == SOURCE_ARC3
    # the next instance: two demonstrations, a reset and the solution
    obs = res.observation
    assert obs.step == 1 and obs.is_fresh
    first = env.step(RequestDemos())
    assert isinstance(first.feedback, DemonstrationFeedback) and first.cost_delta == pytest.approx(1.0)
    assert isinstance(first.feedback.demonstration, Demonstration) and first.observation.actions_used == 0
    env.step(RequestDemos())
    env.step(Act(4))
    reset = env.step(ResetLevel())
    assert isinstance(reset.feedback, ResetFeedback) and reset.feedback.resets_used == 1
    assert reset.observation.frames[-1] == obs.frames[-1]
    for act in env._open[1].drawn.solution:
        res = env.step(act)
    rec = res.record
    assert rec.solved and rec.demo_requests == 2 and rec.metadata["resets"] == 1
    assert rec.wrong_attempts == 1 + rec.metadata["solution_length"]
    assert rec.cost == pytest.approx(2 + (rec.wrong_attempts + 1) / rec.metadata["baseline"])
    assert env.done and env.total_cost == pytest.approx(sum(r.cost for r in env.records))


@needs_arc3
def test_an_instance_that_reaches_its_cap_costs_the_whole_budget(source):
    cfg = Arc3Config(cap_multiplier=0.25)
    env = Arc3Env(schedule(1), config=cfg, source=source)
    obs = env.reset()
    cap = obs.action_cap
    assert cap == math.ceil(0.25 * obs.baseline_actions)
    env.step(RequestDemos())
    res = None
    for i in range(cap):
        res = env.step(ResetLevel() if i == 0 else Act(3 if i % 2 else 4))
        assert res.instance_done == (i == cap - 1)
    assert res.feedback.failed and not res.record.solved and not res.record.gave_up
    assert res.record.cost == cfg.max_cost == 0.25 + 3 and env.total_cost == pytest.approx(cfg.max_cost)


@needs_arc3
def test_invalid_actions_are_charged_and_counted_and_the_demo_cap_refuses_at_no_cost(source):
    env = Arc3Env(schedule(1), config=Arc3Config(demo_cap=1), source=source)
    obs = env.reset()
    assert 7 not in obs.available_actions
    bad = [Act(0), Act(9), Act(7), Act(6), Act(6, x=64, y=0), Act(6, x=1, y=-1)]
    for i, act in enumerate(bad):
        res = env.step(act)
        assert not res.feedback.valid and not res.feedback.ignored and res.feedback.rendered == 0
        assert res.observation.actions_used == i + 1
    env.step(RequestDemos())
    refused = env.step(RequestDemos())
    assert refused.feedback.refused and refused.feedback.demonstration is None and refused.cost_delta == 0
    for act in env._open[0].drawn.solution:
        res = env.step(act)
    assert res.record.invalid_submissions == len(bad) and res.record.demo_requests == 1


@needs_arc3
def test_a_give_up_is_the_runner_s_and_costs_the_whole_budget(source):
    env = Arc3Env(schedule(1), source=source)
    env.reset()
    env.step(Act(4))
    res = env.step(GiveUp())
    assert isinstance(res.feedback, GiveUpFeedback) and res.record.gave_up and res.record.cost == 8
    assert res.feedback.charged == pytest.approx(8 - 1 / res.record.metadata["baseline"])


@needs_arc3
def test_a_game_over_is_ignored_by_the_engine_until_a_reset(source):
    env = Arc3Env(schedule(1), source=source)
    env.reset()
    for _ in range(5):   # sp80 is lost at its fifth pour
        res = env.step(Act(5))
        if res.feedback.state == "GAME_OVER":
            break
    assert res.feedback.state == "GAME_OVER" and not res.instance_done
    res = env.step(Act(4))
    assert res.feedback.ignored and not res.feedback.valid and not res.feedback.rendered and res.observation.frames
    res = env.step(ResetLevel())
    assert res.feedback.state == "NOT_FINISHED"
    for act in env._open[0].drawn.solution:
        res = env.step(act)
    assert res.record.solved and res.record.invalid_submissions == 1


# -- rounds, the runner and the measures -------------------------------------------------------------------------------


@needs_arc3
def test_a_frozen_test_s_round_opens_every_user_s_level_and_commits_together(source):
    env = Arc3Env(schedule(3, phase="seen-test", users=3), source=source)
    env.reset()
    assert env.open_indices() == [0, 1, 2]
    for index in (2, 0):
        for act in env._open[index].drawn.solution:
            res = env.step(act, index)
        assert res.instance_done and res.round_records is None and not env.records
    for act in env._open[1].drawn.solution:
        res = env.step(act, 1)
    assert [r.step for r in res.round_records] == [0, 1, 2] and res.stream_done and len(env.records) == 3


@needs_arc3
def test_the_runner_plays_an_arc3_schedule_and_every_measure_reads_its_records(source, tmp_path):
    sched = schedule(4)
    env = Arc3Env(sched, source=source)
    result = run(Oracle(env), sched, env=env, out_dir=tmp_path / "run", log_events=True)
    assert all(r.solved for r in result.records)
    s = result.summary
    assert s["max_cost"] == 8 and s["config"] == Arc3Config().to_json()
    assert s["stream"]["n"] == 4 and s["stream"]["solve_rate"] == 1
    assert s["stream"]["score"] == pytest.approx(100 * (1 - (SP80_LEVEL0 / SP80_BASELINE) / 8))
    assert "score" in text_report(s, title="sp80")
    loaded, config, records = load_run(tmp_path / "run")
    assert isinstance(config, Arc3Config) and len(records) == 4 and loaded.meta["dataset"] == "arc3"
    kinds = {json.loads(line)["action"]["type"] for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()}
    assert kinds == {"Act"}
    # the learner reset on the same instances, buying a demonstration first: a unit a game dearer
    reset_env = Arc3Env(sched, source=source)
    reset = run(Oracle(reset_env, demos=1), sched.reset_each(), env=reset_env)
    learned = share_learned(result.records, reset.records, config, reps=0)
    assert learned["lift"] == pytest.approx(100 / 8) and learned["learnable_budget"] == pytest.approx(75 / 8 * (1 + SP80_LEVEL0 / SP80_BASELINE))
    held = Schedule(name="t", seed=0, slots=[Slot("sp80", s.instance_seed, "held-out-test", reset_before=True)
                                             for s in sched.slots], sources=sched.sources, meta=sched.meta)
    test_env, ref_env = Arc3Env(held, source=source), Arc3Env(held, source=source)
    t = transfer(run(Oracle(test_env), held, env=test_env).records,
                 run(Oracle(ref_env, demos=2), held, env=ref_env).records, config, reps=50)
    assert t["n"] == 4 and t["transfer"] == pytest.approx(200 / 8) and t["solved"] == {"run": 1, "reset": 1}


@needs_arc3
def test_an_interrupted_arc3_run_resumes_from_its_last_whole_round(source, tmp_path):
    sched = schedule(4)

    class Dies(Oracle):
        def act(self, obs):
            if obs.step >= 2:
                raise RuntimeError("interrupted")
            return super().act(obs)

    out = tmp_path / "run"
    env = Arc3Env(sched, source=source)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(Dies(env), sched, out_dir=out, env=env)
    assert len((out / "records.jsonl").read_text().splitlines()) == 2
    env = Arc3Env(sched, source=source)
    result = run(Oracle(env), sched, out_dir=out, env=env, resume=True)
    assert [r.step for r in result.records] == [0, 1, 2, 3] and all(r.solved for r in result.records)


@needs_arc3
def test_the_runner_resets_the_learner_before_each_instance_of_the_fresh_reference(source):
    class Counting(Idle):
        resets = 0

        def reset(self):
            Counting.resets += 1

    sched = schedule(3).reset_each()
    run(Counting(), sched, env=Arc3Env(sched, config=Arc3Config(cap_multiplier=0.1), source=source))
    assert Counting.resets == 3
