"""The oracle (continual_arc_training.oracle): what it is sent, and the checks that ask a label again."""
from __future__ import annotations

import pytest

from continual_arc_training import oracle

CALL = {"messages": [{"role": "user", "content": "Test input (1x1):\n0"}]}
GO = '{"action": "request_demos"}'
WHY = "the corners are the only cells that change, so the fill stops at the frame"


def test_the_working_rules_and_a_note_go_at_the_end_of_the_last_message_of_a_copy():
    out = oracle.with_rules(CALL, "1. Load the grids.")
    assert out["messages"][-1]["content"] == "Test input (1x1):\n0" + oracle.GUIDE_HEAD + "1. Load the grids." + \
        oracle.GUIDE_TAIL
    assert CALL["messages"][-1]["content"] == "Test input (1x1):\n0"          # the student's call is left as it was
    parts = {"messages": [{"role": "tool", "content": [{"type": "text", "text": "1"}]}]}
    assert oracle.with_rules(parts, "R")["messages"][-1]["content"][-1]["text"].startswith(oracle.GUIDE_HEAD)
    assert oracle.with_note(out, "think first")["messages"][-1]["content"].endswith("\n\n[Note: think first]")


def test_the_oracle_sees_no_earlier_reasoning_and_its_thinking_is_switched_on(monkeypatch):
    messages = [{"role": "system", "content": "s"},
                {"role": "assistant", "content": None, "reasoning_content": "private", "reasoning": "private",
                 "tool_calls": [{"id": "t"}]},
                {"role": "tool", "content": "1"}]
    seen = oracle.oracle_messages(messages)
    assert seen[1] == {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]}
    assert messages[1]["reasoning_content"] == "private"   # copies
    sent = {}

    class Client:
        def post(self, url, json):
            sent.update(url=url, body=json)

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "x"}}]}

            return Response()

    o = oracle.Oracle("http://127.0.0.1:18230/v1/", "z-ai/glm-5.3-flash", "medium")
    monkeypatch.setattr(o, "_client", Client())
    o.ask({"messages": messages, "tools": [{"type": "function"}]})
    assert sent["url"] == "http://127.0.0.1:18230/v1/chat/completions"
    body = sent["body"]
    assert body["reasoning"] == {"enabled": True, "effort": "medium"}
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["model"] == "z-ai/glm-5.3-flash" and body["max_tokens"] == oracle.MAX_TOKENS and body["tools"]
    assert body["messages"] == seen


def test_mentions_of_the_rules_are_caught_and_a_puzzles_rule_is_not():
    assert oracle.MENTIONS_RULES.search("Rule 3: with no demo pair, request_demos first.")
    assert oracle.MENTIONS_RULES.search("I follow the working rules and load the grids.")
    assert oracle.MENTIONS_RULES.search("the test input as printed in the working-rules note")
    assert not oracle.MENTIONS_RULES.search("The rule looks like: fill each rectangle; the rule is a mirror")


def test_a_reply_makes_a_move_when_it_calls_a_tool_or_gives_an_action():
    act = '{"action": "submit", "grid": [[1, 2]]}'
    assert oracle.split_reply("The fill stops at the frame.\n\n" + act) == ("The fill stops at the frame.", act)
    assert oracle.split_reply("Checked.\n```json\n" + act + "\n```") == ("Checked.", "```json\n" + act + "\n```")
    assert oracle.split_reply("Let me load the grids.") == ("Let me load the grids.", "")
    assert oracle.makes_move({"content": GO})
    assert oracle.makes_move({"content": "", "tool_calls": [{"function": {"name": "ipython"}}]})
    assert not oracle.makes_move({"content": "", "reasoning_content": "Let me load the grids."})


def _asks(*answers, seen=None):
    it = iter(answers)

    def ask(request):
        if seen is not None:
            seen.append(request["messages"][-1]["content"])
        content, reasoning, tool_calls = next(it)
        return {"choices": [{"message": {"content": content, "reasoning": reasoning, "tool_calls": tool_calls}}],
                "usage": {"cost": 0.01}}
    return ask


def test_an_ask_after_a_failure_says_what_was_wrong_and_a_reply_that_mentions_the_note_is_asked_again():
    seen = []
    out, m, *_, n, spent = oracle.ask_label(_asks((GO, "", None), (GO, "", None), (GO, "per rule 3 I ask first, " + WHY,
                                                                                    None),
                                                  (GO, "as requested, " + WHY, None), (GO, WHY, None), seen=seen),
                                            CALL, "1. Ask first.", need_thinking=4)
    assert (out, n, round(spent, 2)) == ("labelled", 5, 0.05)
    assert seen[0] == "Test input (1x1):\n0"                              # the first ask is the call as it was
    assert "answered this message once without any reasoning" in seen[1]
    assert "answered this message 2 times without any reasoning" in seen[2]
    assert '("rule 3")' in seen[3] and '("as requested")' in seen[4]        # a rule, then the note itself, named
    assert all(s.startswith("Test input (1x1):\n0") for s in seen)


def test_an_empty_reply_is_asked_again_with_a_note_and_the_replys_prose_counts_as_reasoning():
    seen = []
    explained = 'I have no example of this task yet, so I ask for one first.\n\n' + GO
    out, *_, n, _ = oracle.ask_label(_asks(("", "", None), (explained, "", None), seen=seen), CALL, "", need_thinking=4)
    assert (out, n) == ("labelled", 2) and "your last reply was empty" in seen[1]
    out, *_, n, _ = oracle.ask_label(_asks(("", "", None)), CALL, "", need_thinking=0)   # off: an empty reply ends it
    assert (out, n) == ("empty", 1)


def test_a_reply_that_makes_no_move_is_asked_again_and_ends_unlabelled_once_the_asks_are_spent():
    seen = []
    announce = "I'll load all the grids and diff each demo pair."
    code = [{"type": "function", "function": {"name": "ipython", "arguments": "{}"}}]
    out, *_, n, _ = oracle.ask_label(_asks((announce, WHY, None), ("", WHY, code), seen=seen), CALL, "",
                                     need_thinking=4)
    assert (out, n) == ("labelled", 2) and "your last reply made no move" in seen[1]
    out, *_, n, _ = oracle.ask_label(_asks(*[(announce, WHY, None)] * 3), CALL, "", need_thinking=2)
    assert (out, n) == ("no move", 3)


def test_a_label_without_thinking_is_asked_again_and_dropped_when_it_never_thinks():
    code = [{"function": {"arguments": "{}"}}]
    long = WHY * 2
    out, _, reasoning, _, n, cost = oracle.ask_label(_asks(("", "", code), ("", "", code), ("", long, code)), CALL, "",
                                                     need_thinking=4)
    assert (out, reasoning, n, round(cost, 2)) == ("labelled", long, 3, 0.03)
    out, *_, n, _ = oracle.ask_label(_asks(*[("", "", code)] * 4), CALL, "", need_thinking=3)
    assert (out, n) == ("no thinking", 4)
    out, *_, n, _ = oracle.ask_label(_asks(("", "", code)), CALL, "", need_thinking=0)   # off: any reply is kept
    assert (out, n) == ("labelled", 1)


def test_a_reply_that_keeps_naming_the_rules_ends_unlabelled_and_a_failed_ask_ends_at_once():
    named = (GO, "rule 3 says so: " + WHY, None)
    out, *_, n, _ = oracle.ask_label(_asks(*[named] * (oracle.RULE_RETRIES + 1)), CALL, "", need_thinking=0)
    assert (out, n) == ("mentions the rules", oracle.RULE_RETRIES + 1)

    def down(request):
        raise RuntimeError("the relay is down")

    assert oracle.ask_label(down, CALL, "", need_thinking=4)[0] == "failed"


def test_the_relays_key_is_read_from_the_environment_or_the_baselines_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv(oracle.KEY, "from-the-environment")
    assert oracle.read_key() == "from-the-environment"
    monkeypatch.delenv(oracle.KEY)
    (tmp_path / ".env").write_text(f'OTHER=1\n{oracle.KEY} = "from-the-file"\n')
    monkeypatch.setenv("BASELINES", str(tmp_path))
    assert oracle.read_key() == "from-the-file"
    monkeypatch.setenv("BASELINES", str(tmp_path / "none"))
    with pytest.raises(SystemExit, match=oracle.KEY):
        oracle.read_key()
