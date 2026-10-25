"""DAgger's and OPSD's labels (continual_arc_training.labeller) on small hand-made games: which calls are points and of
what kind, the labels and their budget, the student's target as the label set renders it, and the split."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from fakes import TurnTokenizer

from continual_arc_training import labeller
from continual_arc_training.calls import REFINE, load_calls

SYSTEM = {"role": "system", "content": "You are a general purpose agent."}
WRONG = {"role": "user", "content": "Feedback since your last action:\nYour submission was INCORRECT. Wrong attempts "
                                    "so far: 1."}
REPLY = {"role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "ipython", "arguments": json.dumps({"code": "print(1)"})}}]}
TOOL = {"role": "tool", "tool_call_id": "c1", "content": "1"}


def ask(*msgs):
    return {"model": "m", "messages": [SYSTEM, *msgs]}


def reply(content="", tools=None):
    msg = {"role": "assistant", "content": content}
    if tools:
        msg["tool_calls"] = [{"id": t, "type": "function",
                              "function": {"name": "ipython", "arguments": json.dumps({"code": c})}} for t, c in tools]
    return {"choices": [{"index": 0, "message": msg}]}


def write_game(folder: Path, user: int, task: str, calls: list, solved=False, ids="c") -> None:
    (folder / "captured").mkdir(parents=True, exist_ok=True)
    doc = {"user": user, "task_id": task, "solved": solved,
           "calls": [{"call_id": f"{ids}{i}", "request": req, "response": resp} for i, (req, resp) in enumerate(calls)]}
    with gzip.open(folder / "captured" / f"u{user:03d}.json.gz", "wt") as fp:
        json.dump(doc, fp)


def test_decision_points_are_where_the_game_spoke_and_a_move_is_open():
    start = [SYSTEM, {"role": "user", "content": "New instance. Test input: ..."}]
    assert labeller.decision_kind(start) == "start"
    assert labeller.decision_kind([SYSTEM, REPLY, WRONG]) == "failure"
    assert labeller.decision_kind([SYSTEM, REPLY, {"role": "user", "content": "Demos received (request 2):\n..."}]) \
        == "demos"
    budget = {"role": "user", "content": "Your previous turn was stopped because it exceeded the time or model-call "
                                         "budget"}
    assert labeller.decision_kind([SYSTEM, REPLY, budget]) == "budget"
    assert labeller.decision_kind([SYSTEM, REPLY, TOOL]) is None                       # mid-action: a tool's output
    note = {"role": "user", "content": "[python-state]\n\nYour Python kernel persisted through compaction"}
    assert labeller.decision_kind([SYSTEM, REPLY, TOOL, note]) is None                 # a harness note only
    solved = {"role": "user", "content": "Your submission was CORRECT. Instance solved with 1 wrong attempt(s)."}
    assert labeller.decision_kind([SYSTEM, REPLY, solved]) is None                     # the only move left is finish
    assert labeller.kind([SYSTEM, REPLY, solved]) == "forced" and labeller.kind([SYSTEM, REPLY, TOOL]) == "loop"


def test_points_are_the_agents_own_answered_calls(tmp_path):
    play = tmp_path / "r0000"
    first = ask({"role": "user", "content": "Test input (2x2):\n1 2\n3 4\nReply with exactly one action JSON."})
    code = reply("", [("t1", "print(1)")])
    after_tool = ask(first["messages"][1], code["choices"][0]["message"],
                     {"role": "tool", "tool_call_id": "t1", "content": "1"})
    compaction = {"messages": [SYSTEM, {"role": "user", "content": "<conversation> summarise"}]}
    refine = {"messages": [{"role": "system", "content": REFINE + " pass"}, {"role": "user", "content": "x"}]}
    write_game(play, 0, "taskA", [(first, code), (after_tool, reply('{"action": "request_demos"}')),
                                  (compaction, reply("summary")), (refine, reply("r")), (after_tool, None)],
               solved=True)
    pts = labeller.points([play])
    assert [p["i"] for p in pts] == [0, 1] and [p["n"] for p in pts] == [1, 2] and all(p["len"] == 2 for p in pts)
    assert pts[0]["kind"] == "start" and pts[1]["kind"] == "loop"
    assert pts[0]["id"] == "r0000-u000-c0000-c0" and pts[0]["play"] == str(play) and pts[0]["solved"] is True


def test_a_replayed_game_never_inherits_the_discarded_games_labels(tmp_path):
    """A round replayed after one of its games was labelled reuses every call index. Each point's id names its call, so
    a label saved for the discarded game is never kept for the new game's call at the same index; a kept copy of the
    game is read again once the replay rewrote it, and a point read before the replay is refused rather than answered
    with the new game's call."""
    play = tmp_path / "r0008"
    first = ask({"role": "user", "content": "Test input (2x2):\n1 2\n3 4"})
    write_game(play, 1, "taskA", [(first, reply('{"action": "request_demos"}'))])
    [before] = labeller.points([play])
    assert labeller.point_call(before)["call_id"] == "c0"
    write_game(play, 1, "taskA", [(first, reply('{"action": "finish"}'))], ids="replayed")
    [after] = labeller.points([play])
    assert after["i"] == before["i"] and after["id"] != before["id"]
    assert labeller.point_call(after)["response"]["choices"][0]["message"]["content"] == '{"action": "finish"}'
    with pytest.raises(labeller.StalePoint):
        labeller.point_call(before)


def _points_file(tmp_path):
    play = tmp_path / "play"
    first = ask({"role": "user", "content": "Test input (2x2):\n1 2\n3 4"})
    code = reply("", [("t1", "print(1)")])
    after_tool = ask(first["messages"][1], code["choices"][0]["message"],
                     {"role": "tool", "tool_call_id": "t1", "content": "1"})
    write_game(play, 0, "taskA", [(first, code), (after_tool, reply('{"action": "request_demos"}'))])
    out = tmp_path / "r0000"
    out.mkdir()
    (out / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in labeller.points([play])))
    return out


def _answer(content, cost, reasoning="the corners change colour, so the fill must stop at the frame"):
    return {"choices": [{"message": {"content": content, "reasoning": reasoning}}], "usage": {"cost": cost}}


def test_a_round_fails_under_its_minimum_labelled_share_and_the_spend_counts_every_saved_label(tmp_path):
    """A round whose oracle labelled too few of its points (a dead relay or key, or an oracle that will not think)
    fails, counting labels kept from an earlier attempt; the run's spend is every saved label's cost across its
    rounds."""
    out = _points_file(tmp_path)
    answers = [_answer('{"action": "request_demos"}', 0.5), _answer("", 0.25)]   # empty: no label
    seen = []

    def oracle_ask(request):
        seen.append(request["messages"][-1]["content"])
        return answers.pop(0)

    with pytest.raises(SystemExit, match="only 1 of 2 points got a label"):
        labeller.label(out, oracle_ask, "1. Ask first.", budget=10.0, workers=1, need_thinking=0, min_labelled_pct=60)
    assert all("1. Ask first." in s for s in seen)   # the working rules at the end of the call's last message
    assert labeller.spent([out]) == 0.5   # the empty reply saved no label, so it is not counted on a resume
    answers.append(_answer('{"action": "submit", "grid": [[1]]}', 0.5))
    done = labeller.label(out, oracle_ask, "1. Ask first.", budget=10.0, workers=1, need_thinking=0,
                          min_labelled_pct=100)   # the first point's label kept, the second asked again
    assert done["outcomes"] == {"kept": 1, "labelled": 1}
    old = tmp_path / "r0001"
    (old / "labels").mkdir(parents=True)
    (old / "labels" / "x.json").write_text(json.dumps({"usage": {"cost": 0.25}}))   # saved without ``spent``
    assert labeller.spent([out, old]) == 1.25
    assert labeller.label_cost({"usage": {"cost": 0.01}, "spent": 0.03}) == 0.03 and labeller.label_cost({}) == 0.0


def test_no_new_point_is_asked_once_the_rounds_budget_is_spent(tmp_path):
    out = _points_file(tmp_path)
    answers = [_answer('{"action": "request_demos"}', 2.0)]
    with pytest.raises(SystemExit, match="budget"):
        labeller.label(out, lambda request: answers.pop(0), "R", budget=1.0, workers=1, need_thinking=0,
                       min_labelled_pct=0)
    assert len(list((out / "labels").glob("*.json"))) == 1


def test_the_students_target_takes_the_oracles_prose_as_its_reasoning_and_keeps_only_the_action_in_view():
    act = '{"action": "submit", "grid": [[1, 2]]}'
    label = {"message": {"content": "The fill stops at the frame.\n\n" + act}, "reasoning": "corners change"}
    assert labeller.student_message(label) == {"role": "assistant", "content": act,
                                               "reasoning_content": "corners change\n\nThe fill stops at the frame."}
    code = {"message": {"content": "Load first.", "tool_calls": [{"function": {"name": "ipython"}}]}, "reasoning": ""}
    assert labeller.student_message(code) == {"role": "assistant", "content": "", "reasoning_content": "Load first.",
                                              "tool_calls": [{"function": {"name": "ipython"}}]}
    assert labeller.move({"content": 'x {"action": "request_demos"}'}) == "request"
    assert labeller.move({"content": act}) == "submit" and labeller.move({"content": "", "tool_calls": [{}]}) == "code"


def test_a_label_renders_after_its_prompt_as_the_model_writes_it_through_its_end_of_turn():
    """The reply is tokenised on its own after the prompt's text, the prompt's ids are the call's rendered prompt, and
    the reply ends at its end of turn; a reply the template cannot render is None."""
    tok = TurnTokenizer()
    request = ask({"role": "user", "content": "Test input (1x1):\n0"})
    request["chat_template_kwargs"] = {"enable_thinking": True}
    message = {"role": "assistant", "content": '{"action": "request_demos"}', "reasoning_content": "no demo yet"}
    prompt, target = labeller.render_reply(tok, request, message)
    assert tok.decode(prompt).endswith("<user>\nTest input (1x1):\n0\n<assistant>\n<think>\n")
    assert tok.decode(target) == 'no demo yet\n</think>\n{"action": "request_demos"}\x03'
    broken = {"role": "assistant", "content": "x", "tool_calls": [{"type": "function"}]}   # no function to render
    assert labeller.render_reply(tok, request, broken) is None


def test_a_label_sets_game_holds_each_labelled_points_target_and_leaves_out_one_that_makes_no_move(tmp_path,
                                                                                                  monkeypatch):
    """A source game's labels as the set's game: each labelled point's request, prompt ids and target, the call's place
    in the source game recoverable from its call id; a label that only says what it will do is left out and counted.
    The set reads as any play's calls, with no round and no cost."""
    play = tmp_path / "r0002"
    first = ask({"role": "user", "content": "Test input (1x1):\n0"})
    later = ask({"role": "user", "content": "Test input (1x1):\n0"}, {"role": "assistant", "content": "{}"}, WRONG)
    write_game(play, 4, "taskA", [(first, reply("x")), (later, reply("y"))], solved=True)
    out = tmp_path / "labels"
    (out / "labels").mkdir(parents=True)
    pts = labeller.points([play])
    with open(out / "points.jsonl", "w") as fp:
        fp.writelines(json.dumps(p) + "\n" for p in pts)
    (out / "labels" / f"{pts[0]['id']}.json").write_text(json.dumps(
        {"message": {"role": "assistant", "content": 'Two demos first.\n{"action": "request_demos"}'},
         "reasoning": "the test input alone cannot show the rule"}))
    (out / "labels" / f"{pts[1]['id']}.json").write_text(json.dumps(
        {"message": {"role": "assistant", "content": "I'll load the grids."}, "reasoning": "check"}))
    monkeypatch.setattr(labeller, "_tokenizer", TurnTokenizer())
    stats = labeller._label_set_game((out, 0, pts))
    assert stats["examples"] == 1 and stats["no_move"] == 1 and stats["move:request"] == 1
    [call], _ = load_calls(out / "set")
    assert call.meta["task_id"] == "taskA" and call.meta["cost"] is None and call.meta["round"] is None
    assert labeller.source(call.meta["call_id"]) == (4, 0)
    assert TurnTokenizer().decode(call.reply_ids).startswith("the test input alone cannot show the rule\n\nTwo demos")


def test_held_game_takes_every_tenth_game_alone_and_a_rotating_user_of_ten():
    """Alone, every 10th round's game from round 9 is held out; with 10 users, one game a round, each round a
    different user's, so no user's own stream is held out for good."""
    assert [r for r in range(40) if labeller.held_game(r, 0, 10)] == [9, 19, 29, 39]
    held = [[u for u in range(10) if labeller.held_game(r, u, 10)] for r in range(10)]
    assert all(len(h) == 1 for h in held) and sorted(h[0] for h in held) == list(range(10))


def test_the_split_holds_out_held_games_under_each_rounds_own_names(tmp_path):
    """Each round's label set is split by held_game on its games' source users, into sides named for the round (so a
    phase never takes two rounds' games for one); a side with no game is empty."""
    s = tmp_path / "set" / "captured"
    s.mkdir(parents=True)
    for n in range(10):
        with gzip.open(s / f"u{n:03d}.json.gz", "wt") as fp:
            json.dump({"user": n, "source_play": "r0003", "source_user": n, "task_id": f"t{n}", "solved": True,
                       "calls": [{}] * (n % 3)}, fp)

    def sides(round_, out):
        labeller.split(tmp_path / "set", out, round_, 10, (f"train-r{round_:04d}", f"dev-r{round_:04d}"))
        dev = {labeller.load(f)["task_id"] for f in labeller.games(out / f"dev-r{round_:04d}")}
        train = {labeller.load(f)["task_id"] for f in labeller.games(out / f"train-r{round_:04d}")}
        return dev, train

    dev, train = sides(3, tmp_path / "a")
    assert dev == {"t6"} and len(train) == 9 and not dev & train   # (3 + 6) % 10 == 9
    stats = json.loads((tmp_path / "a" / "split.json").read_text())
    assert stats["dev-r0003"] == {"games": 1, "examples": 0} and stats["train-r0003"]["examples"] == 9
    labeller.split(tmp_path / "set", tmp_path / "b", 0, 100, ("train-r0000", "dev-r0000"))
    assert json.loads((tmp_path / "b" / "split.json").read_text())["dev-r0000"]["games"] == 0
