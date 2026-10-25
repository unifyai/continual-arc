"""Tests for forward.py and loop.py's stop hook, on small hand-made games."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import dagger
import expert
import forward
import loop

SYSTEM = {"role": "system", "content": "You are a general purpose agent."}
TRUTH = [[1, 2], [3, 4]]


def ask(*msgs):
    return {"model": "m", "messages": [SYSTEM, *msgs]}


def reply(content="", tools=None):
    msg = {"role": "assistant", "content": content}
    if tools:
        msg["tool_calls"] = [{"id": t, "type": "function", "function": {"name": "ipython", "arguments": json.dumps({"code": c})}}
                             for t, c in tools]
    return {"choices": [{"index": 0, "message": msg}]}


def write_game(folder: Path, user: int, task: str, calls: list, solved=False, ids="c") -> None:
    (folder / "captured").mkdir(parents=True, exist_ok=True)
    doc = {"user": user, "task_id": task, "solved": solved,
           "calls": [{"call_id": f"{ids}{i}", "request": req, "response": resp} for i, (req, resp) in enumerate(calls)]}
    with gzip.open(folder / "captured" / f"u{user:03d}.json.gz", "wt") as fp:
        json.dump(doc, fp)


class FakeProxy:
    def __init__(self, serve=None):
        self.serve = serve


def test_stop_hook_counts_only_the_students_calls_and_then_finishes():
    proxy = loop.stop_proxy(FakeProxy, 2)()
    student = ask({"role": "user", "content": "Test input (1x1):\n0"})
    internal = {"messages": [{"role": "system", "content": expert.REFINE + " pass"}, {"role": "user", "content": "x"}]}
    assert proxy.serve(student, "a") is None
    assert proxy.serve(internal, "b") is None          # prime-agent's own call: not counted
    assert proxy.serve(student, "c") is None
    out = proxy.serve(student, "d")
    assert out["stopped"] is True and out["choices"][0]["message"]["content"] == loop.STOP_REPLY
    assert proxy.serve(student, "e")["stopped"] is True
    assert proxy.serve(internal, "f") is None


def test_stop_hook_keeps_an_earlier_serve_for_the_first_calls():
    seen = []
    proxy = loop.stop_proxy(FakeProxy, 1)(serve=lambda parsed, cid: seen.append(cid) or {"x": 1})
    student = ask({"role": "user", "content": "hi"})
    assert proxy.serve(student, "a") == {"x": 1}
    assert proxy.serve(student, "b")["stopped"] is True
    assert seen == ["a"]


def test_expert_plays_every_call_only_when_asked(monkeypatch):
    answer = {"choices": [{"message": {"content": '{"action": "request_demos"}', "reasoning": "why"}}], "usage": {}}
    plain = ask({"role": "user", "content": "Test input (1x1):\n0"})
    internal = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "<conversation>x"}]}
    for every, want in ((False, [None, None]), (True, ['{"action": "request_demos"}'] * 2)):
        e = expert.Expert("http://127.0.0.1:1/v1", "m", every=every)
        monkeypatch.setattr(e, "ask", lambda parsed: answer)
        got = [e.serve(plain, "a"), e.serve(internal, "b")]
        assert [g and g["choices"][0]["message"]["content"] for g in got] == want
        e.close()


def test_points_leave_out_stopped_and_internal_calls(tmp_path):
    play = tmp_path / "play"
    first = ask({"role": "user", "content": "Test input (2x2):\n1 2\n3 4\nReply with exactly one action JSON."})
    code = reply("", [("t1", "print(1)")])
    after_tool = ask(first["messages"][1], code["choices"][0]["message"], {"role": "tool", "tool_call_id": "t1", "content": "1"})
    compaction = {"messages": [SYSTEM, {"role": "user", "content": expert.INTERNAL_OPENINGS[0] + " summarise"}]}
    stopped = reply(loop.STOP_REPLY)
    stopped["stopped"] = True
    write_game(play, 0, "taskA", [(first, code), (after_tool, reply('{"action": "request_demos"}')),
                                  (compaction, reply("summary")), (after_tool, stopped)])
    pts = forward.points([play])
    assert [p["i"] for p in pts] == [0, 1]
    assert [p["n"] for p in pts] == [1, 2] and all(p["len"] == 2 for p in pts)
    assert pts[0]["kind"] == "start" and pts[1]["kind"] == "loop"
    assert [p["i"] for p in forward.points([play], decision_only=True)] == [0]


def test_split_holds_out_whole_puzzles(tmp_path):
    s = tmp_path / "set"
    write_game(s, 0, "A", [(ask(), reply("x"))] * 2)
    write_game(s, 1, "B", [(ask(), reply("x"))] * 3)
    write_game(s, 2, "A", [(ask(), reply("x"))])
    forward.cmd_split(type("A", (), {"set": str(s), "held": "B", "sides": "train-all,dev-all", "out": str(tmp_path / "out")}))
    stats = json.loads((tmp_path / "out" / "split.json").read_text())
    assert stats["train-all"] == {"games": 2, "examples": 3} and stats["dev-all"] == {"games": 1, "examples": 3}
    train_tasks = {forward.load(f)["task_id"] for f in forward.games(tmp_path / "out" / "train-all")}
    assert train_tasks == {"A"}


def test_match_takes_whole_games_until_the_target(tmp_path):
    s = tmp_path / "set"
    for u, n in enumerate([3, 4, 5, 6]):
        write_game(s, u, f"T{u}", [(ask(), reply("x"))] * n)
    write_game(s, 9, "H", [(ask(), reply("x"))] * 7)
    a = type("A", (), {"set": str(s), "target": 8, "held": "H", "seed": 0, "tolerance": 0.02, "out": str(tmp_path / "m")})
    forward.cmd_match(a)
    stats = json.loads((tmp_path / "m" / "split.json").read_text())
    assert stats["train-all"]["examples"] >= 8 and stats["dev-all"] == {"games": 1, "examples": 7}
    a.target = 100
    with pytest.raises(SystemExit):
        forward.cmd_match(a)


def test_game_measures_on_a_small_game():
    first = ask({"role": "user", "content": "Test input (2x2):\n1 2\n3 4"})
    c1 = reply("", [("t1", "bad(")])
    m1 = first["messages"][1]
    after1 = ask(m1, c1["choices"][0]["message"], {"role": "tool", "tool_call_id": "t1",
                                                     "content": "  File \"<cell-1>\", line 1\nSyntaxError: invalid syntax"})
    c2 = reply("", [("t2", "print(G)")])
    after2 = ask(m1, c2["choices"][0]["message"], {"role": "tool", "tool_call_id": "t2", "content": "[[1, 2], [3, 4]]"})
    wrong = reply('{"action": "submit", "grid": [[0]]}')
    doc = {"task_id": "T", "solved": False,
           "calls": [{"request": first, "response": c1}, {"request": after1, "response": c2},
                     {"request": after2, "response": wrong}]}
    g = forward.game_measures(doc, TRUTH)
    assert g["m"]["code_errors"] == [1.0, 2.0]
    assert g["m"]["right_submissions"] == [0.0, 1.0]
    assert g["m"]["unsubmitted"] == [1.0, 1.0]       # the truth was printed, the game not solved
    assert "costly" not in g["m"]


def test_costly_mistakes():
    sub = lambda grid: {"content": json.dumps({"action": "submit", "grid": grid})}  # noqa: E731
    code = {"content": "", "tool_calls": [{"function": {"name": "ipython", "arguments": "{}"}}]}
    assert forward.costly(sub([[0]]), code, "failure", TRUTH) == "wrong guess"
    assert forward.costly(code, sub(TRUTH), "demos", TRUTH) == "missed answer"
    assert forward.costly(code, {"content": '{"action": "request_demos"}'}, "budget", TRUTH) == "codes at decide now"
    assert forward.costly(sub(TRUTH), code, "failure", TRUTH) is None
    assert forward.costly(code, sub([[0]]), "demos", TRUTH) is None


def test_paired_and_verdict():
    a = {f"t{i}": {"task_id": f"t{i}", "solved": i < 6, "m": {"code_errors": [1, 10], "right_submissions": [3, 4],
                                                               "costly": [1, 10], "unsubmitted": [0, 1]}} for i in range(20)}
    b = {f"t{i}": {"task_id": f"t{i}", "solved": i < 5, "m": {"code_errors": [5, 10], "right_submissions": [1, 4],
                                                               "costly": [5, 10], "unsubmitted": [1, 1]}} for i in range(20)}
    v = forward.verdict(a, b)
    assert v["measures_better"] == 4 and v["passes"] and not v["clearly_fewer_solves"]
    assert forward.verdict(b, a)["measures_better"] == 0


def test_working_rules_go_at_the_end_of_the_last_message_or_the_system_prompt():
    req = ask({"role": "user", "content": "Test input (1x1):\n0"})
    at_end = forward.with_guidelines(req, "1. Load the grids.", "message")
    assert at_end["messages"][-1]["content"].endswith(forward.GUIDE_TAIL)
    assert "1. Load the grids." in at_end["messages"][-1]["content"]
    assert req["messages"][-1]["content"] == "Test input (1x1):\n0"          # the student's call is left as it was
    in_system = forward.with_guidelines(req, "1. Load the grids.", "system")
    assert in_system["messages"][0]["content"].endswith(forward.GUIDE_TAIL)


def test_mentions_of_the_rules_are_caught_and_a_puzzles_rule_is_not():
    assert forward.mentions_rules({"content": ""}, "Rule 3: with no demo pair, request_demos first.")
    assert forward.mentions_rules({"content": "I follow the working rules and load the grids."}, "")
    assert not forward.mentions_rules({"content": "The rule looks like: fill each rectangle."}, "the rule is a mirror")


def test_habits():
    h = forward.habits(["G=load(); print(sorted(G))", "g=[[1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0]]",
                        "def f(g):\n    return g\nprint(all(f(d)==o for d,o in pairs))"])
    assert h["code_calls"] == 3 and h["loads_grids_pct"] == 33.3 and h["copies_by_hand_pct"] == 33.3
    assert h["writes_and_checks_pct"] == 33.3


def test_an_ask_after_a_failure_says_what_was_wrong_and_a_reply_that_mentions_the_note_is_asked_again():
    import forward
    seen = []

    def ask_with(*answers):
        it = iter(answers)

        def ask(request):
            seen.append(request["messages"][-1]["content"])
            content, reasoning = next(it)
            return {"choices": [{"message": {"content": content, "reasoning": reasoning}}], "usage": {"cost": 0.01}}
        return ask

    request = {"messages": [{"role": "user", "content": "Test input (1x1):\n0"}]}
    why = "the corners are the only cells that change, so the fill stops at the frame"
    go = '{"action": "request_demos"}'
    out, m, *_ , n, _ = forward.ask_label(ask_with((go, ""), (go, ""), (go, "per rule 3 I ask first, " + why),
                                                   (go, "as requested, " + why), (go, why)),
                                         request, "1. Ask first.", 3, need_thinking=4)
    assert (out, n) == ("labelled", 5)
    assert seen[0] == "Test input (1x1):\n0"                              # the first ask is the call as it was
    assert "answered this message once without any reasoning" in seen[1]
    assert "answered this message 2 times without any reasoning" in seen[2]
    assert '("rule 3")' in seen[3] and '("as requested")' in seen[4]        # a rule, then the note itself, named
    assert all(s.startswith("Test input (1x1):\n0") for s in seen)
    assert forward.mentions_rules({"content": "the test input as printed in the working-rules note"}, "")


def test_an_empty_reply_is_asked_again_with_a_note_and_the_replys_prose_counts_as_reasoning():
    import forward
    seen = []

    def ask_with(*answers):
        it = iter(answers)

        def ask(request):
            seen.append(request["messages"][-1]["content"])
            content, reasoning = next(it)
            return {"choices": [{"message": {"content": content, "reasoning": reasoning}}], "usage": {"cost": 0.01}}
        return ask

    call = {"messages": [{"role": "user", "content": "Test input (1x1):\n0"}]}
    explained = 'I have no example of this task yet, so I ask for one first.\n\n{"action": "request_demos"}'
    out, *_, n, _ = forward.ask_label(ask_with(("", ""), (explained, "")), call, None, 3, need_thinking=4)
    assert (out, n) == ("labelled", 2) and "your last reply was empty" in seen[1]   # the reply's prose is its reasoning
    out, *_, n, _ = forward.ask_label(ask_with(("", "")), call, None, 3, need_thinking=0)   # off: an empty reply ends it
    assert (out, n) == ("empty", 1)


def test_a_reply_that_makes_no_move_is_asked_again_with_a_note():
    """A reply of prose that runs no code and gives no action ("I'll load the grids") would teach the student to spend
    a turn on nothing: it is asked again with a note saying so, within the reasoning's asks, and ends unlabelled
    ("no move") once they are spent; a tool call or an action is a move."""
    import forward
    seen = []

    def ask_with(*answers):
        it = iter(answers)

        def ask(request):
            seen.append(request["messages"][-1]["content"])
            content, tool_calls = next(it)
            return {"choices": [{"message": {"content": content, "tool_calls": tool_calls,
                                             "reasoning": "the corners change colour and the fill stops at the frame"}}],
                    "usage": {"cost": 0.01}}
        return ask

    call = {"messages": [{"role": "user", "content": "Test input (1x1):\n0"}]}
    announce = "I'll load all the grids and diff each demo pair."
    code = [{"type": "function", "function": {"name": "ipython", "arguments": "{}"}}]
    out, *_, n, _ = forward.ask_label(ask_with((announce, None), ("", code)), call, None, 3, need_thinking=4)
    assert (out, n) == ("labelled", 2) and "your last reply made no move" in seen[1]
    out, *_, n, _ = forward.ask_label(ask_with((announce, None), ('Done.\n\n{"action": "request_demos"}', None)), call,
                                      None, 3, need_thinking=4)
    assert (out, n) == ("labelled", 2)
    out, *_, n, _ = forward.ask_label(ask_with(*[(announce, None)] * 3), call, None, 3, need_thinking=2)
    assert (out, n) == ("no move", 3)


def test_a_reply_makes_a_move_when_it_calls_a_tool_or_gives_an_action():
    """dagger.makes_move, which keeps a reply that does nothing in the game out of the student's targets and OPSD's
    demonstrations: a tool call or an action is a move, a reply that only says what it will do is none, whatever it
    reasons."""
    import dagger
    assert dagger.makes_move({"content": '{"action": "request_demos"}'})
    assert dagger.makes_move({"content": "", "tool_calls": [{"function": {"name": "ipython"}}]})
    assert not dagger.makes_move({"content": "", "reasoning_content": "Let me load the grids."})
    label = {"message": {"content": "I'll load the grids and diff each demo pair."}, "reasoning": "two demos so far"}
    assert not dagger.makes_move(dagger.student_message(label))


def test_the_students_target_takes_the_oracles_prose_as_its_reasoning_and_keeps_only_the_action_in_view():
    import dagger
    act = '{"action": "submit", "grid": [[1, 2]]}'
    assert dagger.split_reply("The fill stops at the frame.\n\n" + act) == ("The fill stops at the frame.", act)
    assert dagger.split_reply("Checked.\n```json\n" + act + "\n```") == ("Checked.", "```json\n" + act + "\n```")
    assert dagger.split_reply("Let me load the grids.") == ("Let me load the grids.", "")
    label = {"message": {"content": "The fill stops at the frame.\n\n" + act}, "reasoning": "corners change"}
    assert dagger.student_message(label) == {"role": "assistant", "content": act,
                                             "reasoning_content": "corners change\n\nThe fill stops at the frame."}
    code = {"message": {"content": "Load first.", "tool_calls": [{"function": {"name": "ipython"}}]}, "reasoning": ""}
    assert dagger.student_message(code) == {"role": "assistant", "content": "", "reasoning_content": "Load first.",
                                            "tool_calls": [{"function": {"name": "ipython"}}]}


def test_a_label_without_thinking_is_asked_again_and_dropped_when_it_never_thinks():
    import forward

    def replies(*thinking):
        it = iter(thinking)

        def ask(_request):
            return {"choices": [{"message": {"content": "", "tool_calls": [{"function": {"arguments": "{}"}}],
                                             "reasoning": next(it)}}], "usage": {"cost": 0.01}}
        return ask

    long = "the wrong cells are the corners, so the fill must stop at the frame " * 2
    call = {"messages": [{"role": "user", "content": "Test input (1x1):\n0"}]}
    out, _, reasoning, _, n, cost = forward.ask_label(replies("", "", long), call, None, 3, need_thinking=4)
    assert (out, reasoning, n, round(cost, 2)) == ("labelled", long, 3, 0.03)
    out, *_, n, _ = forward.ask_label(replies("", "", "", ""), call, None, 3, need_thinking=3)
    assert (out, n) == ("no thinking", 4)
    out, *_, n, _ = forward.ask_label(replies(""), call, None, 3, need_thinking=0)   # off: any reply is kept
    assert (out, n) == ("labelled", 1)
    # a resumed run counts every ask a saved label cost; a label saved without ``spent`` has only its last ask's
    assert forward.label_cost({"usage": {"cost": 0.01}, "spent": 0.03}) == 0.03
    assert forward.label_cost({"usage": {"cost": 0.01}}) == 0.01 and forward.label_cost({}) == 0.0


def test_thinking_on_sends_the_reasoning_switch_and_the_template_flag(monkeypatch):
    import expert
    import forward
    sent = {}

    class Fake:
        def post(self, url, json):
            sent.update(json)

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "x"}}]}
            return R()
    e = expert.Expert("http://127.0.0.1:1/v1", "m", reasoning=forward.THINKING_ON["reasoning"],
                      extra=forward.THINKING_ON["extra"])
    monkeypatch.setattr(e, "_client", Fake())
    e.ask({"messages": [{"role": "user", "content": "hi"}]})
    assert sent["reasoning"] == {"enabled": True, "effort": "high"}
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}


def _label_args(out, **kw):
    import argparse
    base = dict(out=str(out), expert="m", expert_url="http://127.0.0.1:1/v1", effort="medium", reasoning=None,
                budget=10.0, workers=1, guidelines=None, guidelines_at="message", rule_retries=0, thinking=False,
                need_thinking=0, limit=0, seed=0, min_labelled_pct=0)
    return argparse.Namespace(**{**base, **kw})


def test_label_fails_under_min_labelled_pct_and_spent_counts_every_saved_label(tmp_path, monkeypatch):
    """The paper's DAgger row fails a round whose expert labelled too few of its points (a dead relay or key, or an
    expert that will not think), counting labels kept from an earlier attempt; and its run budget is every saved
    label's cost across the run's rounds (forward.spent)."""
    play = tmp_path / "play"
    first = ask({"role": "user", "content": "Test input (2x2):\n1 2\n3 4"})
    code = reply("", [("t1", "print(1)")])
    after_tool = ask(first["messages"][1], code["choices"][0]["message"],
                     {"role": "tool", "tool_call_id": "t1", "content": "1"})
    write_game(play, 0, "taskA", [(first, code), (after_tool, reply('{"action": "request_demos"}'))])
    out = tmp_path / "r0000"
    out.mkdir()
    (out / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in forward.points([play])))
    answers = [{"choices": [{"message": {"content": '{"action": "request_demos"}'}}], "usage": {"cost": 0.5}},
               {"choices": [{"message": {"content": ""}}], "usage": {"cost": 0.25}}]   # empty: no label

    class FakeExpert:
        def __init__(self, *a, **k):
            pass

        def ask(self, request):
            return answers.pop(0)

    monkeypatch.setattr(expert, "Expert", FakeExpert)
    with pytest.raises(SystemExit, match="only 1 of 2 points got a label"):
        forward.cmd_label(_label_args(out, min_labelled_pct=60))
    assert forward.spent([out]) == 0.5   # the empty reply saved no label, so it is not counted on a resume
    answers.append({"choices": [{"message": {"content": '{"action": "submit", "grid": [[1]]}'}}],
                    "usage": {"cost": 0.5}})
    forward.cmd_label(_label_args(out, min_labelled_pct=100))   # the first point's label kept, the second asked again
    old = tmp_path / "r0001"
    (old / "labels").mkdir(parents=True)
    (old / "labels" / "x.json").write_text(json.dumps({"usage": {"cost": 0.25}}))   # saved without ``spent``
    assert forward.spent([out, old]) == 1.25


def test_a_replayed_game_never_inherits_the_discarded_games_labels(tmp_path):
    """A round replayed after one of its games was labelled reuses every call index. Each point's id names its call
    (dagger.point_id), so a label saved for the discarded game is never kept for the new game's call at the same index
    (3-4% of two real-time runs' labels when ids were positional); a kept copy of the game is read again once the replay rewrote
    it, and a point read before the replay is refused rather than answered with the new game's call."""
    play = tmp_path / "r0008"
    first = ask({"role": "user", "content": "Test input (2x2):\n1 2\n3 4"})
    write_game(play, 1, "taskA", [(first, reply('{"action": "request_demos"}'))])
    [before] = forward.points([play])
    assert dagger.point_call(before)["call_id"] == "c0"
    write_game(play, 1, "taskA", [(first, reply('{"action": "finish"}'))], ids="replayed")
    [after] = forward.points([play])
    assert after["i"] == before["i"] and after["id"] != before["id"]
    assert dagger.point_call(after)["response"]["choices"][0]["message"]["content"] == '{"action": "finish"}'
    with pytest.raises(dagger.StalePoint):
        dagger.point_call(before)


def test_held_game_takes_every_tenth_game_alone_and_a_rotating_user_of_ten():
    """Alone, every 10th round's game from round 9 is held out; with 10 users, one game a round, each round a
    different user's, so no user's own stream is held out for good."""
    assert [r for r in range(40) if forward.held_game(r, 0, 10)] == [9, 19, 29, 39]
    held = [[u for u in range(10) if forward.held_game(r, u, 10)] for r in range(10)]
    assert all(len(h) == 1 for h in held) and sorted(h[0] for h in held) == list(range(10))


def test_split_holds_out_held_games_under_each_rounds_own_names(tmp_path):
    """The real-time DAgger row splits each round's label set by held_game on its games' source users, into sides
    named for the round (so train.load_calls never takes two rounds' games for one); a side with no game is empty."""
    import argparse
    s = tmp_path / "set" / "captured"
    s.mkdir(parents=True)
    for n in range(10):
        with gzip.open(s / f"u{n:03d}.json.gz", "wt") as fp:
            json.dump({"user": n, "source_play": "r0003", "source_user": n, "task_id": f"t{n}", "solved": True,
                       "calls": []}, fp)

    def sides(round_, out):
        forward.cmd_split(argparse.Namespace(set=str(tmp_path / "set"), held=None, hold_every=10, round=round_,
                                             sides=f"train-r{round_:04d},dev-r{round_:04d}", out=str(out)))
        dev = {forward.load(f)["task_id"] for f in forward.games(out / f"dev-r{round_:04d}")}
        train = {forward.load(f)["task_id"] for f in forward.games(out / f"train-r{round_:04d}")}
        return dev, train

    dev, train = sides(3, tmp_path / "a")
    assert dev == {"t6"} and len(train) == 9 and not dev & train   # (3 + 6) % 10 == 9
    forward.cmd_split(argparse.Namespace(set=str(tmp_path / "set"), held=None, hold_every=100, round=0,
                                         sides="train-r0000,dev-r0000", out=str(tmp_path / "b")))
    assert json.loads((tmp_path / "b" / "split.json").read_text())["dev-r0000"]["games"] == 0
