"""A play's calls (continual_arc_training.calls): read back from its captured games, told apart from prime-agent's own
calls, and rendered locally as the server renders them."""
from __future__ import annotations

import json

from fakes import TOK, call, flat_game, read_game, trace, write_game, write_records

from continual_arc_training import play
from continual_arc_training.calls import (
    Call,
    after_failure,
    agent_call,
    by_game,
    is_compaction,
    load_calls,
    render_prompt_ids,
    resample,
    template_messages,
    template_tools,
    with_note,
)

SYSTEM = {"role": "system", "content": "You play ARC puzzles."}
WRONG = {"role": "user", "content": "Feedback since your last action:\nYour submission was INCORRECT. Wrong attempts "
                                    "so far: 1."}
OVER = {"role": "user", "content": "Your submission was INCORRECT. Wrong attempts so far: 8. The attempt cap is "
                                   "reached; the instance is over."}
REPLY = {"role": "assistant", "content": "Let me check.", "reasoning_content": "private thoughts",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "ipython", "arguments": json.dumps({"code": "print(1)"})}}]}
TOOL = {"role": "tool", "tool_call_id": "c1", "content": "1"}


def _request():
    """A harness's call as sent upstream: a tool call made (arguments as a JSON string, reasoning beside it) and its
    result as text parts, which the model now answers."""
    return {"model": "base", "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.7,
            "max_completion_tokens": 512, "chat_template_kwargs": {"enable_thinking": True},
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read a file.",
                                                          "parameters": {"type": "object"}}}],
            "messages": [{"role": "system", "content": "You are an agent with a memory."},
                         {"role": "user", "content": "Do the work."},
                         {"role": "assistant", "content": None, "reasoning": "The notes first.",
                          "tool_calls": [{"id": "c1", "type": "function",
                                          "function": {"name": "read", "arguments": '{"path": "notes.md"}'}}]},
                         {"role": "tool", "tool_call_id": "c1",
                          "content": [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]}]}


# -- which calls are the agent's -------------------------------------------------------------------------------

def test_a_failure_verdict_is_seen_only_in_the_messages_after_the_agents_last_reply():
    assert after_failure([SYSTEM, {"role": "user", "content": "start"}, REPLY, TOOL, WRONG])
    assert not after_failure([SYSTEM, WRONG, REPLY, TOOL])
    assert not after_failure([SYSTEM, {"role": "user", "content": "Demos received (request 1):"}])
    assert not after_failure([SYSTEM, REPLY, OVER])   # the instance is over
    assert after_failure([SYSTEM, REPLY, {"role": "user", "content": "Demo request REFUSED: the demo cap is reached."}])


def test_prime_agents_own_calls_are_not_the_agents():
    assert agent_call([SYSTEM, REPLY, WRONG])
    assert not agent_call([{"role": "system", "content": "You are Prime Agent's /refine pass."}, REPLY, WRONG])
    assert not agent_call([SYSTEM, REPLY, {"role": "user", "content": "<conversation>\n" + WRONG["content"]}])
    assert not agent_call([SYSTEM, {"role": "user", "content": [{"type": "text", "text": "<trigger> review"}]}])
    assert not agent_call([])


def test_a_compaction_call_is_known_by_its_last_message_and_an_odd_body_never_is():
    assert is_compaction({"request": {"messages": [SYSTEM, {"role": "user", "content": "<conversation> all of it"}]}})
    assert not is_compaction({"request": {"messages": [SYSTEM, WRONG]}})
    assert not is_compaction({"request": {"prompt": "def transform(g):"}})          # a raw completion
    assert not is_compaction({"request": {"messages": {"role": "user", "content": "<conversation>"}}})
    assert not is_compaction({"request": None})


# -- a call from its exchange ----------------------------------------------------------------------------------

def test_a_call_from_its_exchange_reads_the_servers_ids():
    request = _request()
    response = {"prompt_token_ids": [1, 2, 3],
                "choices": [{"message": {"content": ""}, "token_ids": [4, 5], "finish_reason": "tool_calls"}]}
    c = Call.from_exchange(request, response, meta={"run": "r", "i": 0})
    assert (c.prompt_ids, c.reply_ids, c.finish_reason) == ([1, 2, 3], [4, 5], "tool_calls")
    assert c.request is request and c.meta == {"run": "r", "i": 0}
    assert Call.from_exchange(request, {"choices": [{"token_ids": [4]}]}) is None          # ids not asked for
    assert Call.from_exchange(request, {"prompt_token_ids": [1], "choices": [{"token_ids": []}]}) is None
    assert Call.from_exchange(request, {"prompt_token_ids": [1], "choices": []}) is None
    assert Call.from_exchange(request, None) is None


def test_a_played_reply_ends_with_the_end_of_turn_only_where_the_model_ended_it():
    eos = TOK.eos_token_id
    cases = [("stop", [65], [65, eos]), ("tool_calls", [65], [65, eos]), ("stop", [65, eos], [65, eos]),
             ("length", [65], [65]), (None, [65], [65])]
    for finish, reply, want in cases:
        assert Call({}, [1], reply, finish).played_reply(eos) == want
    assert Call({}, [1], [65], "stop").played_reply(None) == [65]


# -- rendering a call locally ----------------------------------------------------------------------------------

def test_template_messages_normalise_the_way_vllm_does_under_the_string_content_format():
    msgs = _request()["messages"] + [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "x"}}]},
        {"role": "assistant", "content": "", "reasoning": "r", "reasoning_content": "kept",
         "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": ""}},
                        {"type": "function", "function": {"name": "g", "arguments": "{not json"}}]},
        {"role": "assistant", "content": "done", "tool_calls": []}]
    before = json.dumps(msgs)
    out = template_messages(msgs)
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"path": "notes.md"}   # parsed
    assert out[2]["reasoning_content"] == "The notes first." and out[2]["content"] == ""
    assert out[3]["content"] == "line 1\nline 2" and out[4]["content"] == "a\nb"    # text parts joined
    assert out[5]["content"] == msgs[5]["content"]                                    # an image kept as parts
    assert out[6]["reasoning_content"] == "kept"
    assert [c["function"]["arguments"] for c in out[6]["tool_calls"]] == [{}, "{not json"]
    assert "tool_calls" not in out[7]
    assert out[1]["content"] == "Do the work."
    assert json.dumps(msgs) == before                     # a copy: the call's own messages are untouched


def test_template_tools_are_the_schema_vllm_0_30_hands_the_template():
    """The parameters in the order they were sent, and the function's ``strict`` (and ``defer_loading``) after them
    where the request sets one."""
    params = {"type": "object", "required": ["city"], "properties": {"unit": {"type": "string"}, "city": {}}}
    sent = [{"type": "function", "function": {"name": "f", "parameters": params, "strict": False, "extra": 1}},
            {"type": "function", "function": {"description": "run", "name": "bash"}, "defer_loading": True}]
    out = template_tools(sent)
    assert out[0] == {"type": "function", "function": {"name": "f", "description": None, "parameters": params,
                                                       "strict": False}}
    assert list(out[0]["function"]["parameters"]) == ["type", "required", "properties"]
    assert list(out[0]["function"]["parameters"]["properties"]) == ["unit", "city"]
    assert out[1] == {"type": "function", "defer_loading": True,
                      "function": {"name": "bash", "description": "run", "parameters": None, "defer_loading": True}}
    assert [list(t["function"]) for t in out] == [["name", "description", "parameters", "strict"],
                                                  ["name", "description", "parameters", "defer_loading"]]
    assert template_tools(None) is None and template_tools([]) is None


def test_with_note_adds_to_the_last_message_only():
    msgs = [{"role": "user", "content": "Do the work."}, {"role": "tool", "content": "result"}]
    out = with_note(msgs, "NOTE")
    assert out[0] == msgs[0] and out[1]["content"] == "resultNOTE" and msgs[1]["content"] == "result"
    parts = [{"role": "tool", "content": [{"type": "text", "text": "a"}]}]
    assert with_note(parts, "NOTE")[0]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "NOTE"}]
    assert parts[0]["content"] == [{"type": "text", "text": "a"}]
    assert with_note([{"role": "user", "content": None}], "NOTE")[0]["content"] == "NOTE"
    assert with_note([{"role": "user"}], "NOTE")[0]["content"] == "NOTE"


def test_a_prompt_renders_with_the_requests_tools_switches_and_generation_prompt():
    request = _request()
    text = TOK.decode(render_prompt_ids(TOK, request))
    assert text.startswith("<system> You are an agent with a memory.\n<user> Do the work.\n<assistant> \n<think> "
                           "The notes first.\n<call> ")
    assert "<tool> line 1\nline 2\n<tools> " in text and '"description": "Read a file."' in text
    assert text.endswith('<kwargs> {"enable_thinking": true}\n<assistant>\n')
    noted = TOK.decode(render_prompt_ids(TOK, request, with_note(request["messages"], "\nNOTE")))
    assert noted == text.replace("line 2\n", "line 2\n\nNOTE\n", 1)


class ChatServer:
    """Answers every chat request with the same answer and keeps the payloads it was sent."""

    def __init__(self, answer):
        self.answer, self.payloads = answer, []

    def chat(self, payload):
        self.payloads.append(payload)
        return self.answer


def test_resample_draws_more_replies_at_the_calls_prompt():
    request = _request()
    c = Call(request, [1, 2], [3], "stop", {"i": 0})
    server = ChatServer({"prompt_token_ids": [1, 2], "choices": [
        {"index": 0, "message": {}, "token_ids": [4, 5], "finish_reason": "stop"},
        {"index": 1, "message": {}, "token_ids": [6], "finish_reason": "length"}]})
    more = resample(server, c, "run-p0003", 2, max_tokens=64)
    [sent] = server.payloads
    assert (sent["model"], sent["n"], sent["stream"], sent["return_token_ids"]) == ("run-p0003", 2, False, True)
    assert "stream_options" not in sent and "max_completion_tokens" not in sent and sent["max_tokens"] == 64
    assert sent["temperature"] == 0.7
    assert sent["messages"] == request["messages"] and sent["messages"] is not request["messages"]
    assert request["stream"] is True and request["model"] == "base" and "n" not in request   # the call's own, untouched
    assert [(x.prompt_ids, x.reply_ids, x.finish_reason) for x in more] == [([1, 2], [4, 5], "stop"),
                                                                            ([1, 2], [6], "length")]
    assert all(x.meta == {"i": 0, "resampled": True} for x in more) and c.meta == {"i": 0}
    resample(server, c, "run-p0003", 1)
    assert server.payloads[-1]["max_completion_tokens"] == 512 and "max_tokens" not in server.payloads[-1]
    assert resample(ChatServer(None), c, "run-p0003", 2) == []


# -- a play's calls read back ----------------------------------------------------------------------------------

def test_a_round_reads_its_own_games_with_their_costs_from_the_runs_records(tmp_path):
    """Every user plays every round under the same user id, so each round's games are saved apart, and a round's own
    folder gives that round's games alone, each with its instance's cost from the run's records two levels up."""
    hook = play.save_instance_hook(tmp_path)
    for user, solved in {0: True, 1: False}.items():
        hook(trace(user, 0, "A", solved, seed=user))
    for user, solved in {0: True, 1: True}.items():
        hook(trace(user, 1, "A", solved, seed=10 + user))
    write_records(tmp_path / "records.jsonl", [(0, 0, "A", True, 3.0), (1, 0, "A", False, 12.0),
                                               (0, 1, "A", True, 4.0), (1, 1, "A", True, 5.0),
                                               (5, 1, "B", False, 99.0)])
    dir0, dir1 = play.round_dir(tmp_path, 0), play.round_dir(tmp_path, 1)
    calls0, _ = load_calls(dir0)
    calls1, _ = load_calls(dir1)
    assert {c.meta["run"] for c in calls0} == {dir0.name} and {c.meta["round"] for c in calls0} == {0}
    assert {c.meta["user"]: c.meta["cost"] for c in calls0} == {0: 3.0, 1: 12.0}
    assert {c.meta["user"]: (c.meta["cost"], c.meta["solved"]) for c in calls1} == {0: (4.0, True), 1: (5.0, True)}
    assert calls1[0].reply_ids == [20] and calls1[0].prompt_ids == [1000, 1]   # round 1's own game


def test_a_play_reads_the_records_beside_its_captured_games_and_a_games_own_cost_first(tmp_path):
    t = trace(0, 0, "A", True, seed=1)
    write_game(tmp_path / "flat" / "captured" / "u000.json.gz", flat_game(t))
    write_records(tmp_path / "flat" / "records.jsonl", [(0, 0, "A", True, 5.0)])
    assert load_calls(tmp_path / "flat")[0][0].meta["cost"] == 5.0
    write_game(tmp_path / "own" / "captured" / "u000.json.gz", {**flat_game(t), "cost": 2.0})
    write_records(tmp_path / "own" / "records.jsonl", [(0, 0, "A", True, 999.0)])
    assert load_calls(tmp_path / "own")[0][0].meta["cost"] == 2.0


def test_a_label_sets_games_have_no_round_and_no_cost(tmp_path):
    doc = flat_game(trace(0, 0, "A", True, seed=1))
    del doc["round"]
    write_game(tmp_path / "captured" / "u000.json.gz", doc)
    [c], _ = load_calls(tmp_path)
    assert c.meta["cost"] is None and c.meta["round"] is None and c.meta["task_id"] == "A"


def test_compactions_raw_completions_and_odd_bodies_are_passed_over_and_the_rest_kept(tmp_path):
    """The model's own code can post any body through the proxy: a raw completion (a prompt, no messages) or messages
    that are no list. Neither is a move of the agent's, nor is a compaction's summary; every other call is kept."""
    hook = play.save_instance_hook(tmp_path)
    compaction = [SYSTEM, {"role": "user", "content": "<conversation> all of it"}]
    hook(trace(0, 1, "A", True, seed=1, calls=[call(0, [1, 2], [3]), call(1, [1, 2, 3], [4], messages=compaction),
                                               call(2, [5], [6])]))
    write_records(tmp_path / "records.jsonl", [(0, 1, "A", True, 3.0)])
    f = play.round_dir(tmp_path, 1) / "captured" / "u000.json.gz"
    doc = read_game(f)
    doc["calls"] += [{"call_id": "raw", "request": {"model": "m", "prompt": "def transform(g):"},
                      "response": {"choices": [{"text": " return g", "finish_reason": "length"}]}},
                     {"call_id": "odd", "request": {"messages": {"role": "user", "content": "grid?"}},
                      "response": {"choices": [{"message": {"content": "ok"}}]}},
                     {"call_id": "failed", "request": {"messages": [WRONG]}, "response": None}]
    write_game(f, doc)
    calls, compactions = load_calls(play.round_dir(tmp_path, 1))
    assert [c.meta["i"] for c in calls] == [0, 2] and compactions == 1
    assert [c.meta["call_id"] for c in calls] == [0, 2]


def test_calls_by_game_in_run_then_user_order():
    calls = [Call({}, [1], [2], meta={"run": run, "user": user, "i": i})
             for run, user, i in [("r1", 1, 0), ("r0", 2, 0), ("r1", 1, 1), ("r0", 0, 0)]]
    assert [[(c.meta["run"], c.meta["user"], c.meta["i"]) for c in g] for g in by_game(calls)] == [
        [("r0", 0, 0)], [("r0", 2, 0)], [("r1", 1, 0), ("r1", 1, 1)]]
