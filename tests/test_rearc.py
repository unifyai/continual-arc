"""Tests that need the pinned Re-ARC / ARC-AGI data (skipped otherwise)."""

import pytest
from conftest import needs_data

from continual_arc.data import arc_task_ids, load_arc_tasks
from continual_arc.env import ContinualArcEnv, EnvConfig, RequestDemos, Submit
from continual_arc.grids import grid_hash
from continual_arc.rearc import ReArc
from continual_arc.schedule import RANDOM, epilogue_schedule, stream_schedule
from continual_arc.tasks import SOURCE_ARC_EVAL, SOURCE_REARC, TaskRegistry

pytestmark = [needs_data, pytest.mark.data]

SAMPLE = ["007bbfb7", "00d62c1b", "dbc1a6ce", "2281f1f4", "c1d99e64", "f8ff0b80", "a8d7556c"]


@pytest.fixture(scope="module")
def rearc():
    return ReArc()


def test_all_400_generators_and_verifiers(rearc):
    ids = rearc.task_ids
    assert len(ids) == 400
    assert set(ids) == set(arc_task_ids("training"))


def test_loading_does_not_pollute_sys_modules(rearc):
    import sys

    _ = rearc.task_ids
    for name in ("dsl", "utils", "generators", "verifiers"):
        mod = sys.modules.get(name)
        assert mod is None or not str(getattr(mod, "__file__", "")).endswith(f"re-arc/{name}.py")


@pytest.mark.parametrize("task_id", SAMPLE)
def test_generation_deterministic_and_verified(rearc, task_id):
    a = rearc.generate(task_id, seed=123)
    b = rearc.generate(task_id, seed=123)
    c = rearc.generate(task_id, seed=124)
    assert a == b
    assert a.input != c.input
    assert rearc.verify(task_id, a.input) == a.output


def test_generation_restores_global_rng(rearc):
    import random

    random.seed(7)
    expected = random.random()
    random.seed(7)
    rearc.generate("007bbfb7", seed=1)
    assert random.random() == expected


def test_exclusion(rearc):
    a = rearc.generate("00d62c1b", seed=5)
    b = rearc.generate("00d62c1b", seed=5, exclude_input_hashes={grid_hash(a.input)})
    assert b.input != a.input


def test_verifiers_agree_with_fixed_original_tasks(rearc):
    tasks = load_arc_tasks("training")
    for tid in SAMPLE:
        for pair in tasks[tid]["train"] + tasks[tid]["test"]:
            assert rearc.verify(tid, pair["input"]) == pair["output"], tid


def test_env_with_real_sources():
    ids = SAMPLE[:3]
    sched = stream_schedule(ids, 6, seed=0, source=SOURCE_REARC, params=RANDOM, mode="random").concat(
        epilogue_schedule(arc_task_ids("evaluation")[:2], seed=0, source=SOURCE_ARC_EVAL)
    )
    env = ContinualArcEnv(sched, config=EnvConfig(demos_per_request=2))
    obs = env.reset()
    seen_inputs = set()
    rearc = env.registry.source_for(ids[0])._rearc
    while obs is not None:
        h = grid_hash(obs.input)
        assert h not in seen_inputs
        seen_inputs.add(h)
        demos = env.step(RequestDemos()).feedback.pairs
        if obs.phase == "epilogue":
            assert len(demos) >= 2  # fixed original demos
            answer = env._open[obs.step].pair.output
        else:
            assert len(demos) == 2
            assert all(grid_hash(p.input) != h for p in demos)
            answer = rearc.verify(obs.task_id, obs.input)
        res = env.step(Submit(answer))
        assert res.record.solved and res.record.cost == 1
        obs = res.observation
    assert len(env.records) == 8


def test_registry_default_has_all_sources():
    reg = TaskRegistry.default()
    reg.assign("007bbfb7", SOURCE_REARC)
    reg.assign(arc_task_ids("evaluation")[0], SOURCE_ARC_EVAL)
    assert reg.test_instance("007bbfb7", 1).input
