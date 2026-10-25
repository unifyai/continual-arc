"""Checks for oracle_hint.py's hint checks, its refusal to hint prime-agent's own calls, the printed grid it fills in and
its poster's retries, with a fake oracle in place of GLM and the card of 63613498 from overfit_cards.json.

Run: ``python -m pytest test_oracle_hint.py -q`` (continual_arc_baselines importable, as oracle_hint imports it).
"""
import json
from pathlib import Path

import pytest

import oracle_hint as oh
import overfit

HERE = Path(__file__).resolve().parent


class _Aliases:
    def alias(self, task: str) -> str:
        return "a1b2c3d4"


CARD = overfit.card_text("63613498", overfit.load_cards(HERE / "overfit_cards.json"), _Aliases())
C1 = (HERE / "rules" / "c1.md").read_text()


def test_the_checks_refuse_unprinted_grids_long_hints_and_written_out_programs():
    messages = [{"role": "user", "content": "Test input (3x3):\n1 2 3\n4 5 6\n7 8 9\n"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "content": 'ANSWER [[1,1,1],[2,2,2]]'}]
    assert oh.check_hint('I submit it: {"action": "submit", "grid": [[1, 1, 1], [2, 2, 2]]}', messages) == []
    assert oh.check_hint("The input was [[1,2,3],[4,5,6],[7,8,9]].", messages) == []
    assert oh.check_hint('{"action": "submit", "grid": [[3, 3, 3], [2, 2, 2]]}', messages) == [
        "it holds a grid that the game has not printed: write PRINTED_GRID instead of copying a grid"]
    assert oh.check_hint(" ".join(["word"] * 130), messages)
    program = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(25)) + "\n```"
    assert oh.check_hint("I run this:\n" + program, messages)
    assert oh.check_hint("Your last reply called a tool named submit.", messages)
    assert oh.check_hint('The game said "Reply again, you must act". I request a demo.', messages) == []
    right, wrong = [[2, 2, 2], [1, 1, 1]], [[1, 1, 1], [2, 2, 2]]
    assert oh.check_hint('{"action": "submit", "grid": [[1, 1, 1], [2, 2, 2]]}', messages, right) == [
        "it holds a printed grid that is not the card transform's output for the test input"]
    assert oh.check_hint('{"action": "submit", "grid": [[1, 1, 1], [2, 2, 2]]}', messages, wrong) == []


def test_prime_agent_calls_get_no_hint():
    messages = [{"role": "user", "content": "<trigger>\ncompact; 12 assistant turns since last auto-refine review"}]
    record = oh.Oracle(lambda body: pytest.fail("the oracle was asked")).hint(CARD, C1, messages, oh.GameState())
    assert record["hint"] is None and record["skipped"]


def test_a_submit_hint_gets_the_grid_the_code_printed_filled_in_exactly():
    big = [[(r * 7 + c) % 10 for c in range(30)] for r in range(30)]
    shown = [{"role": "user", "content": "Test input (2x2):\n1 2\n3 4\n"},
             {"role": "tool", "content": "demo pairs [1] match [True]\nANSWER " + json.dumps(big, separators=(",", ":"))}]
    hint = 'My check passed. This reply ends with {"action": "submit", "grid": PRINTED_GRID}'
    filled, problems = oh.fill_printed(hint, shown, big)
    assert problems == [] and filled.endswith('"grid": ' + json.dumps(big) + "}")
    assert oh.check_hint(filled, shown, big) == []   # 900 cells are no words
    # nothing printed yet, or a printed grid that is not the card's answer: no grid is filled in
    assert oh.fill_printed(hint, shown[:1], big)[1]
    assert oh.fill_printed(hint, shown, [[0]])[1]
    assert oh.fill_printed("I request a demo.", shown, big) == ("I request a demo.", [])


def test_the_poster_retries_a_reset_connection(monkeypatch):
    import httpx
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ReadError("Connection reset by peer")
        return httpx.Response(200, json={"choices": [{"message": {"content": "<hint>ok</hint>"}}]})

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert oh.http_post("http://relay/v1", wait_s=0)({"messages": []})["choices"][0]["message"]["content"] == \
        "<hint>ok</hint>"
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ReadError("reset"))), **kw))
    with pytest.raises(httpx.ReadError):
        oh.http_post("http://relay/v1", wait_s=0)({"messages": []})
