import pytest
from conftest import as_users

from continual_arc.env import (
    ContinualArcEnv,
    DemosFeedback,
    EnvConfig,
    GiveUp,
    GiveUpFeedback,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.schedule import RANDOM, sequential_params, stream_schedule


def make_env(fake_registry, fake_source, n=6, **cfg):
    sched = stream_schedule(fake_source.task_ids()[:3], n, seed=0, source=fake_source.name, params=RANDOM, order="random")
    return ContinualArcEnv(sched, registry=fake_registry, config=EnvConfig(**cfg)), sched


def solve(env, obs):
    return [[(c + env.registry.source_for(obs.task_id).shift(obs.task_id)) % 10 for c in row] for row in obs.input]


def test_step_before_reset_raises(fake_registry, fake_source):
    env, _ = make_env(fake_registry, fake_source)
    with pytest.raises(RuntimeError):
        env.step(GiveUp())


def test_correct_first_try_costs_nothing(fake_registry, fake_source):
    env, sched = make_env(fake_registry, fake_source, n=2)
    obs = env.reset()
    assert obs.step == 0 and obs.n_steps == 2 and obs.exposure_index == 0
    res = env.step(Submit(solve(env, obs)))
    assert isinstance(res.feedback, SubmitFeedback) and res.feedback.correct
    assert res.instance_done and not res.stream_done
    assert res.record.solved and res.record.cost == 0 and res.record.wrong_attempts == 0
    assert res.observation.step == 1
    res2 = env.step(Submit(solve(env, res.observation)))
    assert res2.stream_done and res2.observation is None
    assert env.done and env.total_cost == 0
    assert [r.step for r in env.records] == [0, 1]


def test_wrong_attempts_accumulate_then_cap(fake_registry, fake_source):
    env, _ = make_env(fake_registry, fake_source, n=1, attempt_cap=3)
    obs = env.reset()
    wrong = [[9 for _ in row] for row in obs.input]
    r1 = env.step(Submit(wrong))
    assert not r1.feedback.correct and r1.feedback.attempts_used == 1 and not r1.instance_done
    assert r1.cost_delta == 1 and r1.observation.attempts_used == 1
    r2 = env.step(Submit(wrong))
    assert r2.feedback.attempts_used == 2 and not r2.instance_done
    r3 = env.step(Submit(wrong))
    assert r3.feedback.failed and r3.instance_done and r3.stream_done
    rec = r3.record
    # unsolved: the whole budget, 3 attempts + 8 demos at 1, the last step charging the rest of it
    assert not rec.solved and rec.wrong_attempts == 3 and rec.cost == 11 and not rec.gave_up
    assert r3.cost_delta == 9 and env.total_cost == 11


def test_invalid_submission_counts_as_wrong(fake_registry, fake_source):
    env, _ = make_env(fake_registry, fake_source, n=1, attempt_cap=2)
    env.reset()
    r = env.step(Submit([[1, 2], [3]]))
    assert not r.feedback.valid and not r.feedback.correct and r.feedback.attempts_used == 1
    r = env.step(Submit("garbage"))
    assert r.instance_done and r.record.invalid_submissions == 2 and r.record.cost == 10


def test_demo_requests_cost_and_cap(fake_registry, fake_source):
    env, _ = make_env(fake_registry, fake_source, n=1, demo_cap=2, demo_cost=0.5, demos_per_request=4)
    obs = env.reset()
    r1 = env.step(RequestDemos())
    assert isinstance(r1.feedback, DemosFeedback) and not r1.feedback.refused
    assert len(r1.feedback.pairs) == 4 and r1.cost_delta == 0.5
    assert all(p.input != obs.input for p in r1.feedback.pairs)
    r2 = env.step(RequestDemos())
    assert len(r2.feedback.pairs) == 4 and r2.feedback.demo_requests_used == 2
    assert r2.feedback.pairs[0] != r1.feedback.pairs[0]
    r3 = env.step(RequestDemos())
    assert r3.feedback.refused and r3.feedback.pairs == [] and r3.cost_delta == 0
    assert env.observation().demo_requests_used == 2
    r4 = env.step(Submit(solve(env, obs)))
    assert r4.record.solved and r4.record.demo_requests == 2 and r4.record.cost == pytest.approx(1.0)


def test_give_up_charges_the_rest_of_the_budget(fake_registry, fake_source):
    env, _ = make_env(fake_registry, fake_source, n=2, attempt_cap=5)
    obs = env.reset()
    wrong = [[9 for _ in row] for row in obs.input]
    env.step(Submit(wrong))
    env.step(RequestDemos())
    r = env.step(GiveUp())
    assert isinstance(r.feedback, GiveUpFeedback) and r.feedback.charged == 11  # 13 less the 2 spent
    assert r.record.gave_up and not r.record.solved
    assert r.record.wrong_attempts == 1 and r.record.demo_requests == 1 and r.record.cost == 13
    assert env.total_cost == 13
    assert r.observation.step == 1


def test_an_unsolved_instance_costs_the_whole_budget_however_it_failed(fake_registry, fake_source):
    """Failing is never cheaper than trying: guessing to the cap, asking for every demonstration
    first, and giving up at once all cost max_cost."""
    costs = []
    for demos, give_up in ((0, False), (8, False), (0, True)):
        env, _ = make_env(fake_registry, fake_source, n=1)
        obs = env.reset()
        wrong = [[9 for _ in row] for row in obs.input]
        for _ in range(demos):
            env.step(RequestDemos())
        if give_up:
            r = env.step(GiveUp())
        else:
            while not (r := env.step(Submit(wrong))).instance_done:
                pass
        costs.append((r.record.cost, env.total_cost))
    assert costs == [(16.0, 16.0)] * 3


def test_step_after_done_raises(fake_registry, fake_source):
    env, _ = make_env(fake_registry, fake_source, n=1)
    env.reset()
    env.step(GiveUp())
    with pytest.raises(RuntimeError):
        env.step(GiveUp())


def test_records_carry_schedule_metadata(fake_registry, fake_source):
    ids = fake_source.task_ids()[:2]
    sched = stream_schedule(ids, 6, seed=0, source=fake_source.name, params=sequential_params(3), order="sequential")
    env = ContinualArcEnv(sched, registry=fake_registry)
    obs = env.reset()
    assert not obs.reset_before and obs.phase == "stream"
    while obs is not None:
        obs = env.step(GiveUp()).observation
    recs = env.records
    assert [r.exposure_index for r in recs] == [0, 1, 2, 0, 1, 2]
    assert [r.gap for r in recs] == [None, 1, 1, None, 1, 1]
    assert not any(r.reset_before for r in recs)
    assert all(r.source == fake_source.name for r in recs)


def test_instances_deterministic_across_envs(fake_registry, fake_source):
    env1, sched = make_env(fake_registry, fake_source, n=3)
    env2 = ContinualArcEnv(sched, registry=fake_registry)
    assert env1.reset().input == env2.reset().input
    a = env1.step(RequestDemos()).feedback.pairs
    b = env2.step(RequestDemos()).feedback.pairs
    assert a == b


def test_config_validation():
    with pytest.raises(ValueError):
        EnvConfig(attempt_cap=0).validate()
    with pytest.raises(ValueError):
        EnvConfig(diff_lb=0.9, diff_ub=0.1).validate()
    assert EnvConfig(attempt_cap=8, demo_cap=8, demo_cost=1.0).max_cost == 16


def test_rounds_open_every_instance_at_once_and_commit_in_slot_order(fake_registry, fake_source):
    ids = fake_source.task_ids()[:1]
    sched = stream_schedule(ids, 6, seed=0, source=fake_source.name, params=sequential_params(6), order="sequential")
    as_users(sched, 3)
    env = ContinualArcEnv(sched, registry=fake_registry)
    obs = env.reset()
    assert obs.step == 0 and obs.round == 0 and env.open_indices() == [0, 1, 2]
    assert env.observation(2).step == 2 and env.observation(5) is None
    # act on the round in any order; nothing is committed until the round ends
    r = env.step(Submit(solve(env, env.observation(2))), 2)
    assert r.instance_done and r.round_records is None and r.observation.step == 0 and env.records == []
    env.step(GiveUp(), 0)
    assert env.open_indices() == [1] and env.records == []
    r = env.step(Submit(solve(env, env.observation(1))), 1)
    assert r.round_records is not None and [x.step for x in r.round_records] == [0, 1, 2]
    assert [x.step for x in env.records] == [0, 1, 2] and r.observation.step == 3 and r.observation.round == 1
    assert env.open_indices() == [3, 4, 5]
    # the same task three times in one round shares exposure and gap; the next round sees it once more
    assert [x.exposure_index for x in env.records] == [0, 0, 0]
    assert env.observation(3).exposure_index == 1
    # each slot of a round is one user's; each user met the task once before
    assert [x.user for x in env.records] == [0, 1, 2] and [env.observation(i).user for i in (3, 4, 5)] == [0, 1, 2]
    assert env.observation(4).user_exposure_index == 1 and env.records[1].user_gap is None
    with pytest.raises(RuntimeError, match="not open"):
        env.step(GiveUp(), 0)
    for i in (3, 4, 5):
        r = env.step(GiveUp(), i)
    assert r.stream_done and env.done and len(env.records) == 6
    assert [x.round for x in env.records] == [0, 0, 0, 1, 1, 1]


def test_restore_takes_whole_rounds_only(fake_registry, fake_source):
    ids = fake_source.task_ids()[:2]
    sched = stream_schedule(ids, 6, seed=0, source=fake_source.name, params=sequential_params(3), order="sequential")
    as_users(sched, 3)
    env = ContinualArcEnv(sched, registry=fake_registry)
    env.reset()
    for i in (0, 1, 2):
        env.step(GiveUp(), i)
    done = env.records
    env2 = ContinualArcEnv(sched, registry=fake_registry)
    env2.reset()
    with pytest.raises(ValueError, match="inside a round"):
        env2.restore(done[:2])
    obs = env2.restore(done)
    assert obs.step == 3 and env2.open_indices() == [3, 4, 5]
