import json
import math

import pytest
from conftest import needs_arc3

from continual_arc.arc3.env import (
    SOURCE_ARC3,
    Act,
    ActFeedback,
    Arc3Config,
    Arc3Env,
    ResetFeedback,
    ResetLevel,
)
from continual_arc.configs import MODES, build_schedule, get_config
from continual_arc.env import GiveUp, GiveUpFeedback
from continual_arc.learner import BaseLearner
from continual_arc.runner import build_env, load_run, run
from continual_arc.schedule import Schedule, Slot, epilogue_schedule
from continual_arc.testing import (
    Arc3GiveUpLearner,
    Arc3RandomLearner,
    FakeArc3Game,
    FakeArc3Source,
    ScriptedLearner,
)

BASELINE = (4, 6, 8)


def repeat_schedule(task_id: str, n: int, phase: str = "stream") -> Schedule:
    slots = [Slot(task_id, 1000 + i, phase) for i in range(n)]
    return Schedule(name="rep", seed=0, slots=slots, sources={task_id: SOURCE_ARC3}, meta={"dataset": "arc3"})


def make_env(sched: Schedule, config: Arc3Config | None = None, source: FakeArc3Source | None = None):
    source = source or FakeArc3Source(baseline=BASELINE)
    return Arc3Env(sched, config=config, open_game=source.open_game), source


def solve(env: Arc3Env, obs):
    """Press ACTION1 until the level completes; returns the final StepResult."""
    for _ in range(obs.baseline_actions):
        res = env.step(Act(1))
        if res.instance_done:
            return res
    raise AssertionError("did not solve")


# -- levels and cost ---------------------------------------------------------


def test_level_is_exposure_mod_n_levels_and_solved_cost_is_one():
    env, source = make_env(repeat_schedule("fk00", 5))
    obs = env.reset()
    levels = []
    while obs is not None:
        assert obs.is_fresh and obs.n_levels == 3 and len(obs.frames) == 1
        assert obs.frames[0][0][0] == obs.level  # the fake writes the level into the frame
        levels.append(obs.level)
        res = solve(env, obs)
        obs = res.observation
    assert levels == [0, 1, 2, 0, 1]
    recs = env.records
    assert [r.exposure_index for r in recs] == [0, 1, 2, 3, 4]
    assert all(r.solved and r.cost == 1.0 for r in recs)
    assert [r.wrong_attempts for r in recs] == [4, 6, 8, 4, 6]
    assert all(r.demo_requests == 0 and r.invalid_submissions == 0 for r in recs)
    assert all(r.source == SOURCE_ARC3 for r in recs)
    assert env.total_cost == pytest.approx(sum(r.cost for r in recs))
    # one handle per game id, and every instance started its level explicitly
    assert len(source.opened) == 1 and source.opened[0].starts == [0, 1, 2, 0, 1]
    assert env.done


def test_cap_ends_instance_as_failed_at_cap_multiplier():
    cfg = Arc3Config(cap_multiplier=2.5, max_actions=1000)
    env, _ = make_env(repeat_schedule("fk00", 1), cfg)
    obs = env.reset()
    assert obs.action_cap == math.ceil(2.5 * 4) == 10 and obs.baseline_actions == 4
    for i in range(9):
        res = env.step(Act(3))  # valid no-op
        assert not res.instance_done and res.feedback.valid and not res.feedback.failed
        assert res.observation.actions_used == i + 1
    res = env.step(Act(3))
    assert res.instance_done and res.feedback.failed and res.stream_done
    rec = res.record
    assert not rec.solved and not rec.gave_up and rec.cost == 2.5 and rec.wrong_attempts == 10
    assert env.total_cost == pytest.approx(2.5)


def test_max_actions_bounds_the_cap_and_cost_is_clamped():
    cfg = Arc3Config(cap_multiplier=5.0, max_actions=3)
    env, _ = make_env(repeat_schedule("fk00", 1), cfg)
    obs = env.reset()
    assert obs.action_cap == 3
    res = env.step(Act(1))
    res = env.step(Act(1))
    res = env.step(Act(1))
    assert res.instance_done and not res.record.solved and res.record.cost == 5.0


def test_give_up_charges_cap_multiplier():
    env, _ = make_env(repeat_schedule("fk00", 2))
    env.reset()
    env.step(Act(1))
    res = env.step(GiveUp())
    assert isinstance(res.feedback, GiveUpFeedback)
    assert res.feedback.charged == pytest.approx(5.0 - 1 / 4)
    assert res.record.gave_up and res.record.cost == 5.0 and res.record.wrong_attempts == 1
    assert res.observation is not None and res.observation.level == 1
    result = run(Arc3GiveUpLearner(), repeat_schedule("fk01", 3), env=make_env(repeat_schedule("fk01", 3))[0])
    assert all(r.gave_up and r.cost == 5.0 for r in result.records)
    assert result.summary["overall"]["score"] == 0.0


# -- resets and game over -----------------------------------------------------


def test_reset_after_game_over_keeps_level_and_counts():
    env, source = make_env(repeat_schedule("fk00", 2))
    obs = env.reset()
    res = solve(env, obs)  # move to exposure 1 -> level 1
    obs = res.observation
    assert obs.level == 1
    res = env.step(Act(1))
    res = env.step(Act(2))  # game over
    assert res.feedback.state == "GAME_OVER" and res.feedback.valid and not res.instance_done
    res = env.step(Act(1))  # ignored by the engine: invalid, still charged
    assert isinstance(res.feedback, ActFeedback)
    assert res.feedback.ignored and not res.feedback.valid and res.feedback.frames == []
    assert res.observation.actions_used == 3 and res.observation.state == "GAME_OVER"
    assert res.observation.frames  # the last board is still shown
    res = env.step(ResetLevel())
    fb = res.feedback
    assert isinstance(fb, ResetFeedback)
    assert fb.state == "NOT_FINISHED" and fb.resets_used == 1 and fb.actions_used == 3
    obs = res.observation
    assert obs.level == 1 and obs.resets_used == 1 and not obs.is_fresh
    assert source.opened[0].starts == [0, 1]  # no restart happened
    res = solve(env, obs)
    rec = res.record
    assert rec.solved and rec.demo_requests == 1 and rec.wrong_attempts == 3 + 6
    assert rec.invalid_submissions == 1
    assert rec.cost == pytest.approx((9 + 1) / 6)
    assert env.total_cost == pytest.approx(1.0 + 10 / 6)


def test_resets_count_toward_the_cap():
    cfg = Arc3Config(cap_multiplier=1.0)
    env, _ = make_env(repeat_schedule("fk00", 1), cfg)
    obs = env.reset()
    assert obs.action_cap == 4
    for _ in range(3):
        res = env.step(ResetLevel())
        assert not res.instance_done
    res = env.step(ResetLevel())
    assert res.instance_done and res.feedback.failed
    assert res.record.demo_requests == 4 and res.record.wrong_attempts == 0 and res.record.cost == 1.0


# -- invalid actions ----------------------------------------------------------


def test_invalid_actions_are_counted_and_charged():
    def open_game(game_id, seed):
        return FakeArc3Game(game_id, BASELINE, available_actions=(1, 2, 6))

    env = Arc3Env(repeat_schedule("fk00", 1), open_game=open_game)
    obs = env.reset()
    assert obs.available_actions == [1, 2, 6]
    bad = [Act(0), Act(9), Act(3), Act(6), Act(6, x=64, y=0), Act(6, x=1, y=-1)]
    for i, action in enumerate(bad):
        res = env.step(action)
        assert not res.feedback.valid and not res.feedback.ignored
        assert res.feedback.frames == [] and res.observation.actions_used == i + 1
    res = env.step(Act(6, x=3, y=3))
    assert res.feedback.valid
    res = solve(env, res.observation)
    rec = res.record
    assert rec.solved and rec.invalid_submissions == len(bad)
    assert rec.wrong_attempts == len(bad) + 1 + 4 and rec.cost == pytest.approx((len(bad) + 5) / 4)


# -- config -------------------------------------------------------------------


def test_config_validation_and_json():
    cfg = Arc3Config(cap_multiplier=3.0, max_actions=50)
    assert cfg.max_cost == 3.0 and cfg.to_json() == {"cap_multiplier": 3.0, "max_actions": 50}
    assert Arc3Config.from_json(cfg.to_json()) == cfg
    assert cfg.action_cap(10) == 30 and cfg.action_cap(100) == 50
    with pytest.raises(ValueError):
        Arc3Config(cap_multiplier=0).validate()
    with pytest.raises(ValueError):
        Arc3Config(max_actions=0).validate()


def test_seeded_games_are_reopened_per_seed_and_unseeded_are_cached():
    seeded = FakeArc3Source(baseline=BASELINE, accepts_seed=True)
    env, _ = make_env(repeat_schedule("fk00", 3), source=seeded)
    obs = env.reset()
    while obs is not None:
        obs = solve(env, obs).observation
    assert len(seeded.opened) == 3 and [g.seed for g in seeded.opened] == [1000, 1001, 1002]
    assert all(g.closed for g in seeded.opened[:-1])
    env.close()
    assert seeded.opened[-1].closed


# -- schedules ----------------------------------------------------------------


POOL8 = [f"fk{i:02d}" for i in range(8)]


@pytest.mark.parametrize("mode", MODES)
def test_arc3_schedule_epilogue_is_disjoint_from_stream(mode):
    cfg = get_config("arc3-smoke")
    assert cfg.dataset == "arc3" and isinstance(cfg.env, Arc3Config)
    assert cfg.stream_source == SOURCE_ARC3 and cfg.epilogue_source == SOURCE_ARC3
    sched = build_schedule(cfg, mode, seed=2, rearc_task_ids=POOL8)
    stream = {s.task_id for s in sched.slots if s.phase != "epilogue"}
    epi = {s.task_id for s in sched.slots if s.phase == "epilogue"}
    assert len(epi) == cfg.n_epilogue_tasks and epi <= set(POOL8)
    assert set(sched.sources.values()) == {SOURCE_ARC3}
    assert sched.meta["dataset"] == "arc3" and sched.meta["config"]["env"] == {"cap_multiplier": 5.0, "max_actions": 1000}
    if mode == "fresh":
        assert not stream and all(s.reset_before for s in sched.slots)  # the transfer reference
    else:
        assert len(stream) == cfg.n_tasks and not (stream & epi) and (stream | epi) <= set(POOL8)
        assert not any(s.reset_before for s in sched.slots)  # a mode is an order, never a reset


def test_arc3_pool_is_the_manifest_by_default():
    from continual_arc.arc3.games import GAMES
    from continual_arc.configs import epilogue_task_pool, stream_task_pool

    cfg = get_config("arc3-small")
    assert stream_task_pool(cfg) == sorted(GAMES) == epilogue_task_pool(cfg)
    assert all(g.n_levels == len(g.baseline_actions) and min(g.baseline_actions) > 0 for g in GAMES.values())
    assert len(GAMES) == 25 and GAMES["ls20"].version == "9607627b"


def test_env_requires_matching_config():
    from continual_arc.configs import BenchmarkConfig
    from continual_arc.env import EnvConfig

    with pytest.raises(ValueError):
        BenchmarkConfig(name="x", n_tasks=1, stream_length=1,
                        n_epilogue_tasks=0, dataset="arc3")
    with pytest.raises(ValueError):
        BenchmarkConfig(name="x", n_tasks=1, stream_length=1,
                        n_epilogue_tasks=0, env=Arc3Config())
    assert isinstance(build_env(build_schedule("arc3-smoke", "random", 0, rearc_task_ids=POOL8),
                                Arc3Config(), None), Arc3Env)
    with pytest.raises(TypeError):
        build_env(build_schedule("arc3-smoke", "random", 0, rearc_task_ids=POOL8), EnvConfig())


# -- runner -------------------------------------------------------------------


def test_runner_and_summary_with_random_learner(tmp_path):
    sched = build_schedule("arc3-smoke", "workstreams", seed=0, rearc_task_ids=POOL8)
    source = FakeArc3Source(baseline=BASELINE)
    env = Arc3Env(sched, config=Arc3Config(cap_multiplier=2.0), open_game=source.open_game)
    result = run(Arc3RandomLearner(seed=1), sched, out_dir=tmp_path / "run", env=env, log_events=True)
    assert len(result.records) == len(sched) == 9
    s = result.summary
    assert s["max_cost"] == 2.0 and s["config"] == {"cap_multiplier": 2.0, "max_actions": 1000}
    assert s["overall"]["n"] == 9 and 0.0 <= s["overall"]["score"] <= 100.0
    assert s["stream"]["n"] == 8 and s["epilogue"]["n"] == 1
    assert all(r.cost <= 2.0 for r in result.records)
    assert all(r.resources["actions"] == r.wrong_attempts + r.demo_requests for r in result.records if not r.gave_up)
    assert not any(r.resources.get("forced_give_up") for r in result.records)
    sched2, cfg, recs = load_run(tmp_path / "run")
    assert isinstance(cfg, Arc3Config) and cfg.cap_multiplier == 2.0 and len(recs) == 9
    assert json.loads((tmp_path / "run" / "config.json").read_text())["cap_multiplier"] == 2.0
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["overall"]["score"] is not None
    kinds = {json.loads(line)["action"]["type"] for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()}
    assert kinds <= {"Act", "ResetLevel", "GiveUp"} and "Act" in kinds


def test_runner_resets_learner_on_fresh_reference_slots():
    class Counting(BaseLearner):
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

        def act(self, obs):
            return Act(1)

    sched = epilogue_schedule(POOL8[:3], seed=0, source=SOURCE_ARC3, reset_each=True, instances_per_task=2)
    sched.meta["dataset"] = "arc3"
    source = FakeArc3Source(baseline=BASELINE)
    learner = Counting()
    result = run(learner, sched, env=Arc3Env(sched, open_game=source.open_game))
    assert learner.resets == 3
    assert all(r.solved and r.cost == 1.0 for r in result.records)


def test_scripted_learner_gives_up_when_exhausted():
    sched = repeat_schedule("fk00", 1)
    env, _ = make_env(sched)
    result = run(ScriptedLearner([Act(1), ResetLevel()]), sched, env=env)
    rec = result.records[0]
    assert rec.gave_up and rec.wrong_attempts == 1 and rec.demo_requests == 1 and rec.cost == 5.0


def test_cli_env_config_for_arc3_schedule():
    from types import SimpleNamespace

    from continual_arc.cli import _env_config

    sched = build_schedule("arc3-smoke", "random", seed=0, rearc_task_ids=POOL8)
    cfg = _env_config(SimpleNamespace(cap_multiplier=None, max_actions=40, attempt_cap=3), sched)
    assert cfg == Arc3Config(cap_multiplier=5.0, max_actions=40)


# -- the real toolkit ---------------------------------------------------------


@needs_arc3
def test_real_ls20_level_jump_reset_and_accounting():
    from continual_arc.arc3.games import open_game

    handle = open_game("ls20")
    assert handle.n_levels == 7 and handle.baseline_actions[2] == 73
    frame = handle.start_level(2)
    game = handle._game
    assert len(frame) == 64 and all(len(row) == 64 for row in frame)
    assert all(0 <= v <= 15 for row in frame for v in row)
    assert game.level_index == 2 and game._action_count == 0
    assert handle.levels_completed == 2 and handle.state == "NOT_FINISHED"
    out = handle.act(1)
    assert out.frames and not out.ignored and out.levels_completed == 2 and not out.level_completed
    assert game._action_count == 1
    out = handle.reset_level()
    assert game.level_index == 2 and out.levels_completed == 2 and out.state == "NOT_FINISHED"
    assert out.frames[-1] == frame
    # a reset straight after start_level must not fall back to level 0
    frame3 = handle.start_level(3)
    out = handle.reset_level()
    assert game.level_index == 3 and out.levels_completed == 3 and out.frames[-1] == frame3
    # through the environment: one slot, one action, one reset, give up
    sched = repeat_schedule("ls20", 2)
    env = Arc3Env(sched)
    obs = env.reset()
    assert obs.level == 0 and obs.baseline_actions == 22 and obs.action_cap == 110
    assert obs.available_actions == [1, 2, 3, 4]
    res = env.step(Act(1))
    assert res.feedback.valid and res.observation.actions_used == 1
    res = env.step(Act(6, x=1, y=1))  # ls20 does not offer ACTION6
    assert not res.feedback.valid and res.observation.actions_used == 2
    res = env.step(ResetLevel())
    assert res.observation.resets_used == 1 and res.observation.level == 0
    res = env.step(GiveUp())
    assert res.record.cost == 5.0 and res.observation.level == 1
    env.close()


def test_resume_continues_an_interrupted_arc3_run(tmp_path):
    sched = build_schedule("arc3-smoke", "workstreams", seed=0, rearc_task_ids=POOL8)
    source = FakeArc3Source(baseline=BASELINE)
    out = tmp_path / "run"

    class Dies(Arc3RandomLearner):
        def act(self, obs):
            if obs.step >= 4:
                raise RuntimeError("interrupted")
            return super().act(obs)

    import pytest
    with pytest.raises(RuntimeError, match="interrupted"):
        run(Dies(seed=1), sched, out_dir=out, env=Arc3Env(sched, open_game=source.open_game))
    assert len((out / "records.jsonl").read_text().splitlines()) == 4
    result = run(Arc3RandomLearner(seed=1), sched, out_dir=out, resume=True,
                 env=Arc3Env(sched, open_game=source.open_game))
    assert [r.step for r in result.records] == list(range(9))
    _, _, recs = load_run(out)
    assert len(recs) == 9 and json.loads((out / "meta.json").read_text())["resumed_at"][0]["records"] == 4
