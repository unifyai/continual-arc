"""Checks for loop.py and render.py: the benchmark's own schedules (the stream, the validation split, the held-out test
and a task set's), the keep rule's exact test, how a context is shown to the model, Qwen3.5's adapter served under the
checkpoint's names, and render.py's grid squeezing.

Run from this folder: ``python -m pytest test_loop.py -q``, with ``continual_arc`` importable; the adapter test needs
torch and safetensors, and the Hugging Face hub or its cache for Qwen3.5's config.
"""
import copy
import importlib.util
import json

import pytest

import loop

# the tests that train or read train.py's items need ask_sd, which train.py imports
needs_ask_sd = pytest.mark.skipif(importlib.util.find_spec("ask_sd") is None, reason="needs ask_sd")


def test_stream_rounds_are_fresh_instances_of_the_same_25_tasks():
    r0, r1 = loop.stream_slots(0), loop.stream_slots(1)
    tasks = {x["task_id"] for x in r0}
    assert len(r0) == 100 and len(tasks) == 25
    assert all(sum(x["task_id"] == t for x in r0) == 4 for t in tasks)
    assert {x["task_id"] for x in r1} == tasks
    seeds0 = {(x["task_id"], x["instance_seed"]) for x in r0}
    assert not seeds0 & {(x["task_id"], x["instance_seed"]) for x in r1}
    assert len(seeds0) == 100


def test_a_cycle_plays_one_fresh_user_of_the_same_25_tasks(tmp_path):
    """The whole-game cycles' stream games: round 0's play is users 0-3 (``stream --round 0``), so round R's window k
    plays user 4 + (R-1) x 6 + k - 1: 25 instances, one per task, of the same 25 tasks, met by no earlier play."""
    import argparse
    r0 = loop.stream_slots(0)
    tasks = {x["task_id"] for x in r0}
    assert sum((loop.stream_slots(0, user=u) for u in range(4)), []) == r0
    seen = {(x["task_id"], x["instance_seed"]) for x in r0}
    for r in (1, 2):
        for k in range(1, 7):
            games = loop.stream_slots(0, user=4 + (r - 1) * 6 + k - 1)
            assert len(games) == 25 and {x["task_id"] for x in games} == tasks
            fresh = {(x["task_id"], x["instance_seed"]) for x in games}
            assert len(fresh) == 25 and not fresh & seen
            seen |= fresh
    out = tmp_path / "sched-r1-c1.json"
    loop.cmd_schedule(argparse.Namespace(group="stream", round=0, user=4, out=str(out)))
    saved = json.loads(out.read_text())
    assert saved["name"] == "loop-stream-u4" and saved["meta"]["user"] == 4
    assert sorted((s["task_id"], s["instance_seed"]) for s in saved["slots"]) == sorted(
        (x["task_id"], x["instance_seed"]) for x in loop.stream_slots(0, user=4))


def test_the_validation_split_is_100_evaluation_tasks_no_epilogue_or_stream_plays(tmp_path):
    """The validation split: 100 ARC-1 evaluation tasks, one instance each, in no epilogue of arc1-small, arc1-medium or
    arc1-small served to any number of users at the canonical seeds (the held-out test is arc1-medium's at seed 0), never in the stream, which
    draws on the training tasks; on the benchmark the split was drawn on (v0.1.0), 174 such tasks exist. It is the same
    split on every call, and ``schedule validation`` writes exactly these instances. A later benchmark whose held-out
    tasks grew leaves fewer tasks outside its epilogues, so the draw there is another split, and the test skips."""
    import argparse

    from continual_arc.configs import CANONICAL_SEEDS, build_schedule, epilogue_task_pool, get_config
    epilogues = {s.task_id for name in ("arc1-small", "arc1-medium") for seed in CANONICAL_SEEDS
                 for s in build_schedule(name, "workstreams", seed).slots if s.phase == "epilogue"}
    free = set(epilogue_task_pool(get_config("arc1-medium"))) - epilogues
    if len(free) != 174:
        pytest.skip(f"this benchmark leaves {len(free)} evaluation tasks outside its epilogues, not the 174 of v0.1.0 "
                    "that the validation split was drawn from")
    slots = loop.validation_slots()
    tasks = {x["task_id"] for x in slots}
    assert len(slots) == len(tasks) == loop.VALIDATION_TASKS == 100
    assert tasks == set(loop.validation_tasks()) and {x["source"] for x in slots} == {"arc_eval"}
    assert not tasks & epilogues
    assert not tasks & {x["task_id"] for x in loop.heldout_slots()}
    assert not tasks & {x["task_id"] for x in loop.stream_slots(0)}
    pool = set(epilogue_task_pool(get_config("arc1-medium")))
    assert tasks <= pool and len(pool - epilogues) == 174
    assert loop.validation_slots() == slots
    out = tmp_path / "sched-val.json"
    loop.cmd_schedule(argparse.Namespace(group="validation", round=0, user=None, out=str(out)))
    saved = json.loads(out.read_text())
    assert saved["name"] == "loop-validation"
    assert sorted((s["task_id"], s["instance_seed"]) for s in saved["slots"]) == sorted(
        (x["task_id"], x["instance_seed"]) for x in slots)


def test_heldout_is_100_distinct_evaluation_tasks_containing_arc1_small_heldout():
    from continual_arc.configs import build_schedule
    held = loop.heldout_slots()
    assert len(held) == len({x["task_id"] for x in held}) == 100
    small = {s.task_id for s in build_schedule("arc1-small", "workstreams", 0).slots if s.phase == "epilogue"}
    assert small <= {x["task_id"] for x in held}
    assert not {x["task_id"] for x in held} & {x["task_id"] for x in loop.stream_slots(0)}


def test_a_task_set_schedule_gives_each_task_its_fresh_games_and_meets_no_validation_or_test_task(tmp_path):
    """A task set of the stream's own Re-ARC tasks: ``games`` fresh instances of each trained task and ``held_games`` of
    each held-out one, every seed its own and the same on every call, none a validation or held-out test task."""
    import argparse
    from continual_arc.rearc import ReArc
    stream = sorted({x["task_id"] for x in loop.stream_slots(0)})
    d = {"name": "small", "tasks": stream[:4], "held": stream[4:5], "games": 3, "held_games": 12}
    f = tmp_path / "small.json"
    f.write_text(json.dumps(d))
    slots = loop.task_set_slots(f)
    per = {t: sum(x["task_id"] == t for x in slots) for t in d["tasks"] + d["held"]}
    assert len(slots) == 4 * 3 + 12 and all(per[t] == 3 for t in d["tasks"]) and all(per[t] == 12 for t in d["held"])
    assert len({x["instance_seed"] for x in slots}) == len(slots) and {x["source"] for x in slots} == {"rearc"}
    assert slots == loop.task_set_slots(f)
    assert not set(per) & set(loop.validation_tasks()) and not set(per) & {x["task_id"] for x in loop.heldout_slots()}
    assert set(per) <= set(ReArc().task_ids)
    out = tmp_path / "s.json"
    loop.cmd_schedule(argparse.Namespace(group="tasks", round=0, user=None, file=str(f), out=str(out)))
    saved = json.loads(out.read_text())
    assert saved["name"] == "loop-tasks-small" and saved["meta"]["round"] is None
    assert sorted((s["task_id"], s["instance_seed"]) for s in saved["slots"]) == sorted(
        (x["task_id"], x["instance_seed"]) for x in slots)
    d["held"] = d["tasks"][:1]
    f.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="both trained on and held out"):
        loop.task_set_slots(f)


def test_a_student_is_clearly_worse_only_when_it_solves_clearly_fewer():
    """The keep rule's one-sided exact McNemar p that the round's student solves fewer than the best student: none of
    four discordant pairs won is 1/16, past WORSE_P, so clearly worse; none of three is 1/8, not enough; winning more
    than losing is never evidence of solving fewer. K2 + C1 on the first 50 held-out puzzles (1 won, 15 lost against
    plain K2) is clearly worse."""
    assert loop.mcnemar_p_fewer(0, 4) == 1 / 16 < loop.WORSE_P <= loop.mcnemar_p_fewer(0, 3) == 1 / 8
    assert loop.mcnemar_p_fewer(4, 0) == 1.0 and loop.mcnemar_p_fewer(0, 0) == 1.0
    assert loop.mcnemar_p_fewer(1, 15) < 0.001 < loop.WORSE_P


def _play(folder, solved):
    """A play's records.jsonl, one instance a task, solved as ``solved`` says."""
    folder.mkdir(parents=True)
    (folder / "records.jsonl").write_text("".join(
        json.dumps({"task_id": f"t{i}", "instance_seed": i, "solved": ok, "cost": 2.0 if ok else 16.0,
                    "demo_requests": 1, "wrong_attempts": 0 if ok else 8}) + "\n" for i, ok in enumerate(solved)))
    return folder


def test_a_round_is_kept_unless_clearly_worse_and_is_the_new_best_only_when_it_solves_more(tmp_path):
    """The keep rule (loop.decide) on two plays of the same 20 instances: one more solve is kept and the new best; one
    fewer is kept but not the new best, and so is a tie; five fewer, none won, is clearly worse (one-sided p 1/32), so
    rejected. The verdict is compare's plus new_best."""
    best = _play(tmp_path / "best", [True] * 10 + [False] * 10)
    better = _play(tmp_path / "better", [True] * 11 + [False] * 9)
    worse = _play(tmp_path / "worse", [True] * 5 + [False] * 15)
    for a, b, kept, new_best in ((best, better, True, True), (better, best, True, False), (best, best, True, False),
                                 (best, worse, False, False)):
        out = loop.decide(a, b)
        assert (out["not_worse"], out["new_best"]) == (kept, new_best), (a.name, b.name)
        assert out == {**loop.compare(a, b), "new_best": new_best}
    out = loop.compare(best, worse)
    assert out["paired"] == 20 and (out["solved_only_new"], out["solved_only_base"]) == (0, 5)
    assert out["mean_cost_diff"] == round(5 * 14.0 / 20, 3)
    lo, hi = out["mean_cost_diff_95"]
    assert lo <= out["mean_cost_diff"] <= hi
    assert loop.mcnemar_p(0, 13) < 0.001 and loop.mcnemar_p(3, 3) == 1.0


def _request():
    return {"model": "m", "messages": [{"role": "system", "content": "You play a grid game."},
                                       {"role": "user", "content": "Test input (2x2):\n1 2\n3 4"},
                                       {"role": "assistant", "content": '{"action": "request_demos"}'},
                                       {"role": "user", "content": [{"type": "text", "text": "Demo pair 1 ..."}]}]}


@needs_ask_sd
def test_context_is_appended_to_the_last_message_only():
    request = _request()
    before = copy.deepcopy(request)
    block = loop.context_block("Load demonstration grids from the session log.")
    out = loop.inject(request, block)
    assert out["messages"][:-1] == before["messages"][:-1]
    last, was = out["messages"][-1], before["messages"][-1]
    text = last["content"] if isinstance(last["content"], str) else "".join(p.get("text", "") for p in last["content"])
    assert text.endswith(block) and "written after" not in block
    assert len(json.dumps(last)) > len(json.dumps(was))


def test_context_as_a_thought_opens_the_reasoning_and_leaves_the_shared_settings_alone():
    """Placement thought: the request's template settings gain the context, stripped, then a blank line, as
    reasoning_prefill (the text the teacher reads in training); the messages are untouched, and the settings the proxy
    shares with every other request (body_overrides puts one dict on all of them) are copied, never changed."""
    pytest.importorskip("ask_sd")
    from ask_sd.calls import THOUGHT_SEP
    shared = {"enable_thinking": True}
    for context in ("  Load demonstration grids from the session log.\n", "Request a demo before submitting."):
        request = _request()
        request["chat_template_kwargs"] = shared
        messages = copy.deepcopy(request["messages"])
        out = loop.inject_thought(request, context)
        assert out["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_prefill": context.strip() + THOUGHT_SEP}
        assert out["chat_template_kwargs"]["reasoning_prefill"] == loop.thought_text(context)
        assert out["messages"] == messages
    assert shared == {"enable_thinking": True}


def test_a_qwen35_adapter_is_served_under_the_checkpoint_names(tmp_path):
    """vLLM reads Qwen3.5 through its multimodal class, so an adapter PEFT saved from the text-only model is served from
    a copy whose weights are renamed from model.layers to model.language_model.layers, one for one, the tensors and the
    config unchanged and the trainer's own files left as they were; another model's adapter is served as saved."""
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    src = tmp_path / "adapter"
    src.mkdir()
    # the names PEFT writes for Qwen3.5-9B's text model (as seen on a node), and the config beside them
    weights = {f"{loop.PEFT_LAYERS}{layer}.{module}.lora_{ab}.weight": torch.randn(4, 4)
               for layer, module in ((0, "mlp.down_proj"), (3, "self_attn.q_proj")) for ab in "AB"}
    st.save_file(weights, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({"r": 4, "target_modules": ["q_proj", "down_proj"]}))
    served = loop.served_adapter("Qwen/Qwen3.5-9B", src)
    assert served == tmp_path / "adapter.vllm"
    renamed = st.load_file(str(served / "adapter_model.safetensors"))
    assert sorted(renamed) == sorted(loop.SERVED_LAYERS + k[len(loop.PEFT_LAYERS):] for k in weights)
    assert all(torch.equal(renamed[loop.SERVED_LAYERS + k[len(loop.PEFT_LAYERS):]], v) for k, v in weights.items())
    assert (served / "adapter_config.json").read_text() == (src / "adapter_config.json").read_text()
    assert sorted(st.load_file(str(src / "adapter_model.safetensors"))) == sorted(weights)
    assert loop.served_adapter("Qwen/Qwen3.8-27B", src) == served   # the same architecture under another name
    assert loop.served_adapter("IFM/K2-Horizon-7B", src) == src


def test_render_squeezes_grids():
    import render
    text = "Test input (3x4):\n1 2 3 4\n5 6 7 8\n9 0 1 2\nWrong attempts used: 0/8."
    assert render.squeeze_grids(text) == "Test input (3x4):\n  [3 grid rows]\nWrong attempts used: 0/8."


def test_a_play_can_set_its_own_summary_point_and_zero_leaves_prime_agents_default():
    assert loop.harness_of(49_152, 8_000) == loop.HARNESS
    later = loop.harness_of(114_688, 20_000)
    assert later["compact_at"] == 114_688 and later["compact_keep"] == 20_000 and later["end"] == loop.HARNESS["end"]
    assert loop.harness_of(0, 0)["compact_at"] is None and loop.harness_of(0, 0)["compact_keep"] is None
