"""The model calls a harness makes, as a play keeps them and as training reads them.

prime-agent makes its own chat-completion calls (a system prompt, tools, tool calls and their results, reasoning)
through the benchmark's proxy, which asks the model server for the exact token ids (``return_token_ids``). A play
saves every call of a game in ``captured/uNNN.json.gz`` (:mod:`continual_arc_training.play`); :func:`load_calls`
reads a play's calls back as :class:`Call` objects: the request as sent, the prompt's ids as the server tokenised them
and the reply's ids as it sampled them.

Where a call's prompt is rendered locally (a label's target after its prompt, a teacher's prompt with a demonstration
in it), :func:`render_prompt_ids` renders it the way the server's vLLM 0.30 renders a chat completion: the messages
normalised as vLLM normalises them (:func:`template_messages`), the tools re-serialised as it re-serialises them
(:func:`template_tools`), then the model's chat template.

The agent's own calls are told apart from prime-agent's internal ones (its /refine pass, its compaction summarizer and
its reviews: :func:`agent_call`), and the environment's verdicts from the rest of a game's messages
(:func:`after_failure`)."""
from __future__ import annotations

import copy
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# prime-agent's own calls, not the agent's: its /refine pass (a system prompt of its own), its compaction summarizer
# (whose one message packs the conversation so far and opens with COMPACTION) and its trigger review
REFINE = "You are Prime Agent's /refine"
COMPACTION = "<conversation>"
INTERNAL_OPENINGS = (COMPACTION, "<trigger>")
# the environment's failure verdicts as the harness words them, and the verdict that ends the instance
FAILURES = ("Your submission was INCORRECT", "Demo request REFUSED")
OVER = "The attempt cap is reached"


def text_of(content: Any) -> str:
    """A message's content as text: a string as it is, a list of parts as their texts joined."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def agent_call(messages: list[dict[str, Any]]) -> bool:
    """Whether a call is the agent's own: neither prime-agent's /refine pass nor one of its internal calls."""
    if not messages:
        return False
    first, last = messages[0], messages[-1]
    if first.get("role") == "system" and text_of(first.get("content")).startswith(REFINE):
        return False
    return not text_of(last.get("content")).lstrip().startswith(INTERNAL_OPENINGS)


def is_compaction(call: dict[str, Any]) -> bool:
    """Whether a captured call is prime-agent's compaction summarizer's: its last message, a user's, packs the
    conversation so far. The model's own code can post any JSON body through the proxy, so a body whose messages are
    not a list of chat messages is never one."""
    messages = (call.get("request") or {}).get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return False
    last = messages[-1]
    return last.get("role") == "user" and text_of(last.get("content")).lstrip().startswith(COMPACTION)


def after_failure(messages: list[dict[str, Any]]) -> bool:
    """Whether the call's newest messages (those after the agent's last reply) carry a failure verdict that leaves the
    instance open."""
    last = max((j for j, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    texts = [text_of(m.get("content")) for m in messages[last + 1:] if m.get("role") != "assistant"]
    return any(f in t for t in texts for f in FAILURES) and not any(OVER in t for t in texts)


@dataclass
class Call:
    """One model call: the chat-completion body as sent upstream, the prompt as the server tokenised it, the reply it
    sampled and why it stopped; ``meta`` names the call (its game and place) and carries what training adds to it."""

    request: dict[str, Any]
    prompt_ids: list[int]
    reply_ids: list[int]
    finish_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exchange(cls, request: dict[str, Any], response: Any, meta: dict[str, Any] | None = None) -> Call | None:
        """A call from its request body and the server's answer: the prompt's ids (top-level ``prompt_token_ids``), the
        first choice's reply ids (``token_ids``) and finish reason. None when the answer carries no ids (the call
        failed, or the server was not asked for them)."""
        if not isinstance(response, dict):
            return None
        choices = response.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        prompt_ids, reply_ids = response.get("prompt_token_ids"), choice.get("token_ids")
        if not prompt_ids or not reply_ids:
            return None
        return cls(request, list(prompt_ids), list(reply_ids), choice.get("finish_reason"), dict(meta or {}))

    def played_reply(self, eos: int | None) -> list[int]:
        """The reply's ids as the model ended it: the server leaves out the end-of-turn token of a reply that stopped
        (``stop``) or called a tool (``tool_calls``), which the model did sample, so it goes back on."""
        reply = list(self.reply_ids)
        if self.finish_reason in ("stop", "tool_calls") and eos is not None and reply[-1] != eos:
            reply.append(eos)
        return reply


def instance_records(run_dir: Path) -> list[Any]:
    """Every instance the run has played so far (the benchmark's records: user, round, task, outcome and cost), from
    the run's ``records.jsonl``: at ``run_dir`` itself (a play whose ``captured/`` is beside it) or two levels up (a
    round's folder, ``play/rounds/rNNNN``, whose run's records are at ``play/``). Empty where neither holds one."""
    from continual_arc.runner import prior_records

    for candidate in (run_dir, run_dir.parent.parent):
        records = prior_records(candidate)
        if records:
            return records
    return []


def load_calls(run_dir: Path) -> tuple[list[Call], int]:
    """Every completed call of a play that the model made as the agent, each labelled with its game (``run``, the
    folder's name, and ``user``), its round, task and place (``i``, its index in the game's captured calls), whether its
    game solved its instance and that instance's cost; and how many compaction calls were left out. A compaction call's
    reply is a summary the environment never reads, and its one message packs the whole conversation. A body the
    model's own code sent through the proxy (a raw completion, or messages that are not a list) is no move of the
    agent's and is passed over, as is a call whose answer carries no token ids. A label set's game (no played
    instance) has no round and no cost."""
    calls, compactions = [], 0
    costs: dict[tuple[int, int, str], float] | None = None
    for f in sorted((run_dir / "captured").glob("u*.json.gz")):
        with gzip.open(f, "rt") as fp:
            doc = json.load(fp)
        for i, c in enumerate(doc["calls"]):
            if not (c.get("request") and c.get("response") and isinstance(c["request"].get("messages"), list)
                    and c["request"]["messages"]):
                continue
            if is_compaction(c):
                compactions += 1
                continue
            cost = doc.get("cost")
            if cost is None and "round" in doc:
                if costs is None:
                    costs = {(r.user, r.round, r.task_id): r.cost for r in instance_records(run_dir)}
                cost = costs.get((doc["user"], doc["round"], doc["task_id"]))
            meta = {"user": doc["user"], "round": doc.get("round"), "task_id": doc["task_id"], "i": i,
                    "call_id": c.get("call_id"), "run": run_dir.name, "solved": doc.get("solved"), "cost": cost}
            call = Call.from_exchange(c["request"], c["response"], meta=meta)
            if call is not None:
                calls.append(call)
    return calls, compactions


def by_game(calls: list[Call]) -> list[list[Call]]:
    """``calls`` (:func:`load_calls`') as their games' calls, a game a run's user's, in run then user order."""
    games: dict[tuple[str, int], list[Call]] = {}
    for call in calls:
        games.setdefault((call.meta["run"], call.meta["user"]), []).append(call)
    return [games[key] for key in sorted(games)]


# -- rendering a call locally, as the server's vLLM 0.30 renders it ----------------------------------------------------

def _joined(content: Any) -> Any:
    """A content made of text parts only (a missing one counting as no parts) as their texts joined by newlines, as
    vLLM joins them; any other content as it is."""
    parts = [] if content is None else content
    if isinstance(parts, list) and all(isinstance(p, dict) and p.get("type") == "text" for p in parts):
        return "\n".join(p.get("text") or "" for p in parts)
    return content


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A deep copy of ``messages`` normalised the way vLLM normalises a request's messages before it applies a Hugging
    Face chat template under ``--chat-template-content-format string`` (serve.sh's): an assistant message's tool call
    arguments, a JSON string, parsed into a dict (a missing or empty one becomes ``{}``; one that does not parse is left
    as it is, since vLLM refuses such a request), an empty list of tool calls dropped, ``reasoning`` copied to
    ``reasoning_content`` where that is absent, and every message's content of text parts joined into one string."""
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "assistant":
            if not m.get("tool_calls"):
                m.pop("tool_calls", None)
            for call in m.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if not args:
                    fn["arguments"] = {}
                elif isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except ValueError:
                        continue
                    fn["arguments"] = {} if parsed is None else parsed
            if m.get("reasoning") is not None and m.get("reasoning_content") is None:
                m["reasoning_content"] = m["reasoning"]
        m["content"] = _joined(m.get("content"))
    return out


def template_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """A request's tools as vLLM 0.30 hands them to the chat template (its ``FunctionDefinition``): ``type`` then
    ``function``, the function's ``name``, ``description`` and ``parameters`` in that order (a missing one None), the
    parameters as sent, then ``strict`` and ``defer_loading`` where the request sets them; a tool-level
    ``defer_loading`` stays on the tool and reaches the function when the function sets none. None for no tools."""
    if not tools:
        return None
    out = []
    for tool in tools:
        fn = (tool or {}).get("function") or {}
        function = {"name": fn.get("name"), "description": fn.get("description"), "parameters": fn.get("parameters")}
        entry = {"type": (tool or {}).get("type") or "function", "function": function}
        deferred = (tool or {}).get("defer_loading")
        flags = {"strict": fn.get("strict"),
                 "defer_loading": fn.get("defer_loading") if fn.get("defer_loading") is not None else deferred}
        function.update({k: v for k, v in flags.items() if v is not None})
        if deferred is not None:
            entry["defer_loading"] = deferred
        out.append(entry)
    return out


def with_note(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """A deep copy of ``messages`` with ``note`` added to the end of the last message's content, the one the reply
    answers (a user message or a tool result): after a string, as one more text part after a list of parts, or as the
    whole content when there is none. Every earlier message is left exactly as the model saw it."""
    out = copy.deepcopy(messages)
    last = out[-1]
    content = last.get("content")
    if content is None:
        last["content"] = note
    elif isinstance(content, list):
        last["content"] = content + [{"type": "text", "text": note}]
    else:
        last["content"] = content + note
    return out


def render_prompt_ids(tokenizer: Any, request: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> list[int]:
    """The prompt ids the local chat template gives ``request`` (its own messages, or ``messages`` in their place) on
    the server's path for a chat completion: the template applied to the normalised messages with the request's tools,
    its ``chat_template_kwargs`` and the generation prompt, then the text tokenised without added special tokens."""
    text = tokenizer.apply_chat_template(
        template_messages(messages if messages is not None else request["messages"]),
        tools=template_tools(request.get("tools")), add_generation_prompt=True, tokenize=False,
        **(request.get("chat_template_kwargs") or {}))
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def resample(server: Any, call: Call, model: str, n: int, *, max_tokens: int | None = None) -> list[Call]:
    """``n`` more replies drawn at ``call``'s prompt from ``model`` (an adapter the server serves): the call's request
    sent again (a copy) with the model, ``n``, no streaming and the token ids asked for, and ``max_tokens`` in place of
    the call's when given. One :class:`Call` per choice, with the prompt ids the server reports and the call's meta
    marked ``resampled``; [] when the server fails."""
    payload = copy.deepcopy(call.request)
    payload.pop("stream_options", None)   # vLLM refuses stream options on a request that does not stream
    payload.update({"model": model, "n": n, "stream": False, "return_token_ids": True})
    if max_tokens is not None:
        payload.pop("max_completion_tokens", None)   # vLLM reads it in preference to max_tokens
        payload["max_tokens"] = max_tokens
    out = server.chat(payload)
    if not isinstance(out, dict):
        return []
    meta = {**call.meta, "resampled": True}
    drawn = [Call.from_exchange(payload, {"prompt_token_ids": out.get("prompt_token_ids"), "choices": [choice]}, meta)
             for choice in out.get("choices") or []]
    return [c for c in drawn if c is not None]
