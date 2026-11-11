"""capacity.py off any machine: the runs' task sets nest, each batch deals its run's tasks evenly at instance seeds the
seen test never draws, a game is one instance for a fresh user, GLM's games read in the student's view with each
call's own reply its label, and the stop waits for PATIENCE phases without a gain. Run with continual-arc-research's
venv, which has the benchmark: ``~/continual-arc-baselines/.venv/bin/python -m pytest search/capacity -q``."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the loop's modules (dagger, loop) at the top

import capacity  # noqa: E402

POOL = [f"{i:08x}" for i in range(400)]


def test_task_sets_nest():
    t25, t100, t400 = (capacity.tasks(run, POOL) for run in ("t25", "t100", "t400"))
    assert (len(t25), len(t100), len(t400)) == (25, 100, 400)
    assert set(t25) <= set(t100) <= set(t400) == set(POOL)


@pytest.mark.parametrize("run,per_task,batches", [("t25", 4, 1), ("t100", 1, 1), ("t400", 1, 4)])
def test_batches_deal_tasks_evenly(run, per_task, batches):
    ids = capacity.tasks(run, POOL)
    dealt = collections.Counter(t for k in range(batches) for t in capacity.order(run, ids, k))
    assert set(dealt) == set(ids) and set(dealt.values()) == {per_task}
    assert all(len(capacity.order(run, ids, k)) == capacity.BATCH for k in range(batches))
    assert capacity.order(run, ids, 0) != capacity.order(run, ids, batches)   # every round in an order of its own


def test_instances_are_fresh():
    from continual_arc.seeds import derive_seed
    seen = {derive_seed(capacity.SEED, capacity.SEEN_TEST, i) for i in range(capacity.SEEN_TEST_SIZE)}
    seeds = [s for run in capacity.RUNS for k in range(3)
             for _, s in capacity.instances(run, capacity.tasks(run, POOL), k)]
    assert len(set(seeds)) == len(seeds) and not seen & set(seeds)


def test_a_game_is_one_instance_for_a_fresh_user(tmp_path):
    ids = capacity.tasks("t25", POOL)
    folders = capacity.write_schedules("t25", 2, tmp_path, ids)
    assert [f.name for f in folders] == [f"g{j:03d}" for j in range(capacity.BATCH)]
    from continual_arc.schedule import Schedule
    s = Schedule.load(folders[7] / "schedule.json")
    task, seed = capacity.instances("t25", ids, 2)[7]
    assert s.users == 1 and len(s.slots) == 1 and s.slots[0].task_id == task and s.slots[0].instance_seed == seed
    assert s.slots[0].phase == "stream" and s.sources == {task: "rearc"} and s.meta["capacity"]["batch"] == 2


def test_the_student_view_and_own_labels():
    reply = {"role": "assistant", "content": "The rows mirror.\n```json\n{\"action\": \"submit\"}\n```",
             "reasoning": "check the grids", "reasoning_details": [{"x": 1}]}
    tool = {"role": "assistant", "content": None, "reasoning": "load them",
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "ipython", "arguments": "{}"}}]}
    doc = {"user": 0, "calls": [
        {"request": {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, tool,
                                  {"role": "tool", "tool_call_id": "1", "content": "out"}, reply,
                                  {"role": "user", "content": "wrong"}]},
         "response": {"choices": [{"message": dict(reply)}], "usage": {"cost": 0.002}}}]}
    view = capacity.viewed(doc)
    msgs = view["calls"][0]["request"]["messages"]
    assert view["calls"][0]["request"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert msgs[0] == {"role": "system", "content": "s"} and msgs[3]["role"] == "tool"
    assert msgs[4] == {"role": "assistant", "content": "```json\n{\"action\": \"submit\"}\n```",
                       "reasoning_content": "check the grids\n\nThe rows mirror."}
    assert msgs[2]["tool_calls"] == tool["tool_calls"] and msgs[2]["reasoning_content"] == "load them"
    assert view["calls"][0]["response"] == doc["calls"][0]["response"]   # the label is GLM's own reply
    label = capacity.own_label(view["calls"][0])
    assert label["message"] == {"role": "assistant", "content": reply["content"], "tool_calls": None}
    assert label["reasoning"] == "check the grids" and label["spent"] == 0.002


def test_the_stop_waits_for_patience_phases_without_a_gain():
    assert not capacity.plateaued([1.0, 0.8, 0.7])
    assert not capacity.plateaued([1.0, 0.8, 0.7, 0.6, 0.6, 0.6])   # each of the last three under 0.7
    assert capacity.plateaued([1.0, 0.8, 0.7, 0.6, 0.598, 0.61, 0.6])
    assert not capacity.plateaued([1.0, 0.8, 0.7, 0.6, 0.598, 0.61, 0.59])   # 1.7% under 0.6


def test_the_tests_are_cells_of_the_generalisation_search():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generalisation"))
    import cell
    for name, spec in capacity.cells("t100", 7).items():
        c = cell.KINDS[spec["kind"]](spec)
        assert c.name == name and c.bundle == "code-dagger-rows-turns" and c.check()
        lines = cell.inputs(c, ["play"], done=set())
        assert any("modal-capacity/cap-t100/phases/p0007/adapter/adapter_model.safetensors" in x for x in lines)
        if spec["kind"] == "frozen":
            assert any(x.startswith("results/runs/gpu-nodes/NODE-cap-gen/cap-t100/play/alias-salt ") for x in lines)
