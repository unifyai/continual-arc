"""Tests for expert.py (the takeover's trigger and the expert's view) and dagger.py's helpers.
Run: python -m pytest -q test_dagger.py. The DAgger phase tests run ask-sd's tiny CPU trainer from ask-sd's own test
helpers (``test_ask_sd``), so they need torch, ask_sd and ask-sd's tests/ folder on the path, and skip without them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dagger  # noqa: E402
import expert  # noqa: E402

SYSTEM = {"role": "system", "content": "You play ARC puzzles."}
WRONG = {"role": "user", "content": "Feedback since your last action:\nYour submission was INCORRECT. Wrong attempts so far: 1."}
OVER = {"role": "user", "content": "Your submission was INCORRECT. Wrong attempts so far: 8. The attempt cap is reached; "
                                   "the instance is over."}
REPLY = {"role": "assistant", "content": "Let me check.", "reasoning_content": "private thoughts",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ipython", "arguments": json.dumps({"code": "print(1)"})}}]}
TOOL = {"role": "tool", "tool_call_id": "c1", "content": "1"}


def test_triggers_on_the_call_right_after_a_failure():
    assert expert.triggers([SYSTEM, {"role": "user", "content": "start"}, REPLY, TOOL, WRONG])


def test_no_trigger_once_the_agent_has_replied_since():
    assert not expert.triggers([SYSTEM, WRONG, REPLY, TOOL])


def test_no_trigger_without_a_failure_or_when_the_instance_is_over():
    assert not expert.triggers([SYSTEM, {"role": "user", "content": "Demos received (request 1):"}])
    assert not expert.triggers([SYSTEM, REPLY, OVER])


def test_refused_demo_request_triggers():
    assert expert.triggers([SYSTEM, REPLY, {"role": "user", "content": "Demo request REFUSED: the demo cap is reached."}])


def test_prime_agents_own_calls_never_trigger():
    refine = {"role": "system", "content": "You are Prime Agent's /refine pass."}
    assert not expert.triggers([refine, REPLY, WRONG])
    assert not expert.triggers([SYSTEM, REPLY, {"role": "user", "content": "<conversation>\n" + WRONG["content"]}])


def test_expert_sees_no_earlier_reasoning_and_messages_are_copies():
    msgs = [SYSTEM, REPLY, TOOL, WRONG]
    seen = expert.expert_messages(msgs)
    assert "reasoning_content" not in seen[1] and seen[1]["tool_calls"] == REPLY["tool_calls"]
    assert REPLY["reasoning_content"] == "private thoughts"   # the call itself untouched


def test_serve_leaves_other_calls_to_the_student(tmp_path):
    e = expert.Expert("http://127.0.0.1:9/v1", "m", log=tmp_path / "x.jsonl")
    assert e.serve({"messages": [SYSTEM, REPLY, TOOL]}, "id") is None
    assert e.counts["calls"] == 1 and e.counts["triggered"] == 0


def test_serve_returns_the_experts_reply_with_its_reasoning_kept_apart(tmp_path, monkeypatch):
    e = expert.Expert("http://127.0.0.1:9/v1", "m", log=tmp_path / "x.jsonl")
    answer = {"choices": [{"message": {"content": "Need more evidence.\n\n{\"action\": \"request_demos\"}",
                                       "reasoning": "the rule failed on the test"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}}
    monkeypatch.setattr(e, "ask", lambda parsed: answer)
    out = e.serve({"model": "b1-6", "messages": [SYSTEM, REPLY, WRONG]}, "id")
    msg = out["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": answer["choices"][0]["message"]["content"]}
    assert out["expert"]["reasoning"] == "the rule failed on the test" and out["_budget_credit_s"] >= 0
    assert out["choices"][0]["finish_reason"] == "stop" and out["model"] == "b1-6"
    assert json.loads((tmp_path / "x.jsonl").read_text())["outcome"] == "served"


def test_serve_forwards_a_refusal_or_an_empty_reply(tmp_path, monkeypatch):
    e = expert.Expert("http://127.0.0.1:9/v1", "m", log=tmp_path / "x.jsonl")
    for message, outcome in [({"content": None, "refusal": "no"}, "refused"), ({"content": "  "}, "empty")]:
        monkeypatch.setattr(e, "ask", lambda parsed, m=message: {"choices": [{"message": m}]})
        assert e.serve({"messages": [SYSTEM, REPLY, WRONG]}, "id") is None
        assert e.counts[outcome] == 1


def test_step_text_shows_text_and_code():
    text = dagger.step_text({"content": "Testing the idea.", "tool_calls": REPLY["tool_calls"]})
    assert text == "Testing the idea.\n[ipython call]\nprint(1)"


def test_target_message_keeps_the_experts_action():
    served = {"choices": [{"message": {"role": "assistant", "content": "x", "tool_calls": REPLY["tool_calls"]}}]}
    m = dagger.target_message(served, "my reasoning")
    assert m["reasoning_content"] == "my reasoning" and m["tool_calls"] == REPLY["tool_calls"] and m["content"] == "x"


def _sft_run(tmp_path, monkeypatch, lr: float, **args):
    """DAgger's phase (train.fit with train.Imitation) on ask-sd's tiny CPU trainer: the same four items to train on
    and to hold out; ``args`` over the defaults (one measurement per pass of two steps). DONE and the held-out
    measurements."""
    import argparse
    import dataclasses
    import random

    import pytest
    pytest.importorskip("torch")
    t = pytest.importorskip("test_ask_sd", reason="needs ask-sd's tests/ on the path")
    import train
    tr = t._perturbed(t._peaked(t._tiny_trainer()))
    tr.model.float()
    tr.set_lr(lr)
    items = [dataclasses.replace(it, teacher_ids=it.student_ids) for it in t._numbered()]
    monkeypatch.setattr(train, "sft_items", lambda tok, calls, msl: (items, {"calls": 4, "items": 4, "too_long": 0}))
    a = argparse.Namespace(**{**dict(eval_steps=0, epochs=5, batch=2, gpus=1, chain_slack=8, chain_calls=2, patience=2,
                                     max_minutes=60, max_steps=None, chains=False, divergence="forward_kl",
                                     max_seq_len=512), **args})
    train.fit(a, tr, train.Imitation(a, tr.tokenizer, [], ["held-out"]), tmp_path, None, random.Random(0))
    return json.loads((tmp_path / "DONE").read_text()), json.loads((tmp_path / "phase.json").read_text())["held_out"]


def test_sft_trains_to_fit_and_keeps_the_best_pass(tmp_path, monkeypatch):
    done, held = _sft_run(tmp_path, monkeypatch, lr=1e-2)
    dev = [e["loss"] for e in held]
    assert [e["step"] for e in held] == [0, 2, 4, 6, 8, 10] and dev[-1] < dev[0]   # before the first pass, after each
    assert done["best_step"] == held[min(range(6), key=lambda i: dev[i])]["step"] and done["steps"] == 10
    assert (tmp_path / "adapter").exists()


def test_sft_stops_once_passes_stop_improving(tmp_path, monkeypatch):
    done, held = _sft_run(tmp_path, monkeypatch, lr=0.0)   # nothing moves: no pass lowers the held-out loss
    assert [e["step"] for e in held] == [0, 2, 4] and done["best_step"] == 0 and done["steps"] == 4
    assert (tmp_path / "adapter").exists()                 # the starting adapter, saved as the best


def test_sft_measures_every_eval_steps_and_stops_on_measurements_not_passes(tmp_path, monkeypatch):
    """The paper's DAgger row measures the held-out loss every --eval-steps steps over a buffer that grows every
    round, --patience counting measurements: here every step (a pass is two), and nothing moves, so two measurements
    without a gain stop it after two steps, inside the first pass, the starting adapter kept as the best."""
    done, held = _sft_run(tmp_path, monkeypatch, lr=0.0, eval_steps=1)
    assert [e["step"] for e in held] == [0, 1, 2] and done["steps"] == 2 and done["best_step"] == 0
    assert (tmp_path / "adapter").exists()
    (tmp_path / "fit").mkdir()
    fitted, held = _sft_run(tmp_path / "fit", monkeypatch, lr=1e-2, eval_steps=3, patience=0, epochs=2)
    assert [e["step"] for e in held] == [0, 3, 4]   # every 3 steps, and the last step (4) once


def _check_dir(tmp_path, samples: dict[str, list[str]]):
    """A check folder with 20 states whose expert move is a demo request, and K replies per model as moves: a submit
    carries the right grid when its move is "right"."""
    (tmp_path / "labels").mkdir()
    (tmp_path / "states").mkdir()
    with open(tmp_path / "states.jsonl", "w") as fp:
        for i in range(20):
            fp.write(json.dumps({"id": f"s{i}", "task_id": "t", "source": "x", "instance_seed": 0, "exposure": 0}) + "\n")
            (tmp_path / "labels" / f"s{i}.json").write_text(json.dumps({"move": "request"}))
            (tmp_path / "states" / f"s{i}.json").write_text(json.dumps({"messages": [WRONG]}))
    for arm, moves in samples.items():
        (tmp_path / "samples" / arm).mkdir(parents=True)
        for i in range(20):
            for k, mv in enumerate(moves):
                content = {"request": '{"action": "request_demos"}', "right": '{"action": "submit", "grid": [[1]]}',
                           "wrong": '{"action": "submit", "grid": [[2]]}'}.get(mv, "")
                rec = {"move": "submit" if mv in ("right", "wrong") else mv, "tokens": 1000, "finish_reason": "stop",
                       "content": content, "tool_calls": None}
                (tmp_path / "samples" / arm / f"s{i}-{k}.json").write_text(json.dumps(rec))


def _score(tmp_path, monkeypatch, arms):
    import argparse
    monkeypatch.setattr(dagger, "true_outputs", lambda states: {s["id"]: [[1]] for s in states})
    dagger.cmd_check_score(argparse.Namespace(dir=str(tmp_path), arms=arms, baseline="b6", margin=0.10, length_floor=0.85))
    return json.loads((tmp_path / "check.json").read_text())["arms"]


def test_check_goes_when_the_expert_move_is_made_more_often(tmp_path, monkeypatch):
    _check_dir(tmp_path, {"b6": ["code", "code", "code", "request"], "arm": ["request", "request", "code", "right"]})
    out = _score(tmp_path, monkeypatch, ["b6", "arm"])
    assert out["b6"]["same_as_expert"] == 0.25 and out["arm"]["same_as_expert"] == 0.5
    assert out["arm"]["vs_baseline"][0] == 0.25 and out["arm"]["submissions_right"] == 1.0 and out["arm"]["go"]
    assert out["arm"]["requests_where_expert_requested"] == 0.5 and not out["b6"]["go"]


def test_check_stops_an_arm_whose_submissions_get_worse(tmp_path, monkeypatch):
    _check_dir(tmp_path, {"b6": ["code", "code", "right", "code"], "arm": ["request", "request", "wrong", "code"]})
    out = _score(tmp_path, monkeypatch, ["b6", "arm"])
    assert out["arm"]["vs_baseline"][0] == 0.5 and out["arm"]["submissions_right"] == 0.0 and not out["arm"]["go"]
    assert out["arm"]["wrong_submissions_per_100"] == 25.0


def test_grid_of_reads_the_last_submit():
    assert dagger.grid_of('x {"action": "submit", "grid": [[1, 2]]}\n{"action": "submit", "grid": [[3]]}') == [[3]]
    assert dagger.grid_of("no grid") is None


def test_decision_points_are_where_the_game_spoke_and_a_move_is_open():
    start = [SYSTEM, {"role": "user", "content": "New instance. Test input: ..."}]
    assert dagger.decision_kind(start) == "start"
    assert dagger.decision_kind([SYSTEM, REPLY, WRONG]) == "failure"
    assert dagger.decision_kind([SYSTEM, REPLY, {"role": "user", "content": "Demos received (request 2):\n..."}]) == "demos"
    budget = {"role": "user", "content": "Your previous turn was stopped because it exceeded the time or model-call budget"}
    assert dagger.decision_kind([SYSTEM, REPLY, budget]) == "budget"
    assert dagger.decision_kind([SYSTEM, REPLY, TOOL]) is None                       # mid-action: a tool's output
    note = {"role": "user", "content": "[python-state]\n\nYour Python kernel persisted through compaction"}
    assert dagger.decision_kind([SYSTEM, REPLY, TOOL, note]) is None                 # a harness note only
    solved = {"role": "user", "content": "Your submission was CORRECT. Instance solved with 1 wrong attempt(s)."}
    assert dagger.decision_kind([SYSTEM, REPLY, solved]) is None                     # the only move left is finish


def _call(msgs, content, tool_code=None):
    m = {"role": "assistant", "content": content}
    if tool_code:
        m["tool_calls"] = [{"id": "t", "type": "function", "function": {"name": "ipython", "arguments": json.dumps({"code": tool_code})}}]
    return {"request": {"messages": msgs}, "response": {"choices": [{"message": m, "token_ids": [1] * 10}]}}


def test_game_process_counts_a_games_submission_habits():
    start = {"role": "user", "content": "New instance. Test input: ..."}
    demos = {"role": "user", "content": "Demos received (request 1):\n\nDemo pair 1 input (1x1):\n1\nDemo pair 1 output (1x1):\n2"}
    code = "ok = all(f(G['Demo pair 1 input']) == G['Demo pair 1 output'])\nprint(json.dumps(f(G['Test input'])))"
    sub_a = '{"action": "submit", "grid": [[3]]}'
    printed = {"role": "tool", "tool_call_id": "t", "content": "True\n[[3]]"}
    h1 = [SYSTEM, start]
    h2 = h1 + [{"role": "assistant", "content": '{"action": "request_demos"}'}, demos]
    h3 = h2 + [{"role": "assistant", "content": "", "tool_calls": _call([], "", code)["response"]["choices"][0]["message"]["tool_calls"]}, printed]
    h4 = h3 + [{"role": "assistant", "content": sub_a}, WRONG]
    h5 = h4 + [{"role": "assistant", "content": sub_a}, WRONG]
    doc = {"task_id": "t", "user": 0, "solved": False, "calls": [
        _call(h1, '{"action": "request_demos"}'), _call(h2, "", code), _call(h3, sub_a), _call(h4, sub_a),
        _call(h5, '{"action": "submit", "grid": [[9]]}')]}
    g = dagger.game_process(doc)
    first, again, guess = g["subs"]
    assert first == {"right": False, "resend": False, "printed": True, "tested": True}
    assert again["resend"] and again["right"] is False
    assert not guess["printed"] and guess["right"] is None            # a grid no code printed; the game ended unscored
    assert [p["kind"] for p in g["points"]] == ["start", "demos", "failure", "failure"]
    assert [p["demos_seen"] for p in g["points"]] == [0, 1, 1, 1]


def test_cites_note():
    assert dagger.CITES_NOTE.search("As this note says, I should ask for a demo.")
    assert dagger.CITES_NOTE.search("The reply is already decided, so")
    assert not dagger.CITES_NOTE.search("The rule failed on the test input, so a new demo will show the ordering.")


def test_merge_renumbers_games_from_several_plays(tmp_path):
    import argparse
    import gzip
    for name, users in [("a", [0, 1]), ("b", [0])]:
        (tmp_path / name / "captured").mkdir(parents=True)
        for u in users:
            with gzip.open(tmp_path / name / "captured" / f"u{u:03d}.json.gz", "wt") as fp:
                json.dump({"user": u, "task_id": f"{name}{u}", "calls": []}, fp)
    dagger.cmd_merge(argparse.Namespace(plays=[str(tmp_path / "a"), str(tmp_path / "b")], out=str(tmp_path / "ab")))
    docs = [json.load(gzip.open(f, "rt")) for f in sorted((tmp_path / "ab" / "captured").glob("u*.json.gz"))]
    assert [(d["user"], d["task_id"]) for d in docs] == [(0, "a0"), (1, "a1"), (2, "b0")]


def test_grid_rows_reject_a_long_number_at_once():
    import time
    t = time.time()
    assert dagger.grids_in("x = 1234567890123456789012345678901234567890x\n1 2\n3 4\n") == [[[1, 2], [3, 4]]]
    assert time.time() - t < 1


def test_drop_wrong_leaves_out_submits_of_a_wrong_grid(tmp_path, monkeypatch):
    import argparse
    import gzip
    play = tmp_path / "play"
    play.mkdir()
    (play / "schedule.json").write_text(json.dumps({"sources": {"t1": "rearc"}, "slots": [
        {"task_id": "t1", "instance_seed": 1}, {"task_id": "t1", "instance_seed": 2, "user": 1}]}))
    monkeypatch.setattr(dagger, "true_outputs", lambda states: {s["id"]: [[s["instance_seed"]]] for s in states})

    def call(content, tool=False):
        m = {"role": "assistant", "content": content}
        if tool:
            m["tool_calls"] = [{"id": "c", "type": "function", "function": {"name": "ipython", "arguments": "{}"}}]
        return {"response": {"choices": [{"message": m}]}}
    submit = lambda g: call(json.dumps({"action": "submit", "grid": g}))  # noqa: E731
    (tmp_path / "set" / "captured").mkdir(parents=True)
    for n, (user, calls) in enumerate([(0, [submit([[1]]), call("", tool=True)]), (1, [submit([[1]])])]):
        with gzip.open(tmp_path / "set" / "captured" / f"u{n:03d}.json.gz", "wt") as fp:
            json.dump({"user": n, "source_play": str(play), "source_user": user, "calls": calls}, fp)
    dagger.cmd_drop_wrong(argparse.Namespace(set=str(tmp_path / "set"), out=str(tmp_path / "out")))
    docs = [json.load(gzip.open(f, "rt")) for f in sorted((tmp_path / "out" / "captured").glob("u*.json.gz"))]
    # user 0's submit is right (its truth is [[1]]) and kept with its code call; user 1's is wrong, so its game goes
    assert [(d["source_user"], len(d["calls"])) for d in docs] == [(0, 2)]
