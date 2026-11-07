"""A local OpenAI-compatible reverse proxy: the one hook every system can use.

Point any agent system at :attr:`LLMProxy.base_url` with the placeholder :attr:`LLMProxy.api_key` and every physical
model request is observed, journaled and counted, without the system knowing. The real upstream credential lives only
in this process, so it never enters the system's configuration files or logs.

Only a chat completion (a POST to ``/chat/completions``) and a read of the model list (a GET of ``/models`` or of an
entry under it) go upstream. Anything else is answered here with 404 and counted as a failed call, so that a system's
own code, which can read the proxy's address and placeholder key from its environment as any program the system runs
can, reaches no other endpoint on the run's key: a completion, an embedding or any path it invents. When
:attr:`~LLMProxy.body_overrides` names a model, as every row's overrides do (:func:`own_model` gives a learner's), each
chat completion goes to that model, whatever model the call names.

A verified run's learner is untrusted and plays on a key its submitter has paid for, so its proxy is **confined**
(:attr:`~LLMProxy.confined_to`, ``arc-baselines run --confine-models``): only the models its submission names go
upstream, each at the reasoning effort it names, and only what cannot leave the model call. A request carries a plain chat
completion and nothing an upstream would act on apart from the conversation: no other model's name, no fallback list, no
routing, no plugin or server-side tool, no URL for a provider to fetch, no header or field of its own. Each call goes only
to the hosts of :data:`CONFINED_HOSTS` that keep no data, and the pace and size of what it sends and how much of it is
journaled are bounded, since this process runs on the verifier's machine. Where the run pins a model to one endpoint of
those hosts (:attr:`~LLMProxy.pinned`, ``arc-baselines run --pin-providers``), every call to it goes there alone, so that
each of a run's cells is served alike: OpenRouter otherwise spreads an open model's calls over every host serving it,
holding a session on one for an hour or two, and the hosts quantize a model and map a reasoning effort each their own way,
Kimi K3.1 Pro's medium played as its high on one host and as its low on others (7 November 2026).

A provider's prompt cache outlives a call, and answers whoever sends the same opening, on the provider's account and so
perhaps on anyone's: a prompt written by one instance and probed by another, or from outside the run through the same
provider, would carry what it says past every other wall, one reply's ``cached_tokens`` or its latency at a time. So a
confined call opens with a salt (:meth:`LLMProxy.salt`), secret to this process and new with each of the learner's resets,
at the head of everything a provider counts as a prompt's opening (:func:`_salted`). Within an instance, and over a stream
with no resets in it, the cache works as ever. A salt the model repeats to the learner is 256 bits, more than a run's
results can carry out (HELD-OUT.md), and it lasts one instance.

What a run may spend is enforced here, before each call and not only at the key (:class:`SpendLedger`): OpenRouter refuses
a key at its limit only as a call starts, so calls begun together would pass it by as much as they cost.

The proxy is transparent: no caching and no fallback models. Streaming responses are passed through byte for byte while
the server-sent events are parsed on the side for the usage block. A request is changed only as follows, in
chat completions alone:

* ``usage.include`` and ``stream_options.include_usage``, so that the provider reports tokens and charge;
* :attr:`~LLMProxy.body_defaults`, the run's request fields, set only where the system set none;
* for an Anthropic model, a cache breakpoint when the system set none, and prime-agent's per-session log path in its
  system prompt replaced by a constant, so that the prompt's prefix is cached across instances;
* for a run that serves its own model on vLLM: :attr:`~LLMProxy.body_overrides`, fields that replace the system's
  (the name of the adapter being served, ``return_token_ids``, the chat template's switches), which can change between
  calls (:meth:`LLMProxy.set_overrides`); :attr:`~LLMProxy.body_drop`, fields the server would refuse or ignore;
  :attr:`~LLMProxy.role_map`, message roles the server does not know renamed to ones it does; and
  :attr:`~LLMProxy.thinking_field`, the one key every past assistant turn carries its reasoning under, an empty one
  where it had none, for a chat template that refuses such a turn.

A throttled or failing upstream is retried with backoff before anything reaches the system (429, 5xx, and a 403 that
OpenRouter relays from the provider behind it), and an account out of credit (402) is waited out rather than failed.
A call refused only because its prompt and the reply budget it asks together pass the model's window is sent once more
without the budget, so that the server gives the reply the room the window has left: a harness asks the same budget
at every call (prime-agent min(maxTokens, 32,000) for a turn, 0.8 x its compaction reserve to compact), which a hosted
provider shortens to fit and vLLM refuses, so on vLLM every call of a long conversation was refused once it passed the
window less the budget. A prompt that fills the window alone is still refused, and the system's own overflow handling
(prime-agent compacts) takes it from there. ``serve`` lets a run answer some chat completions itself. The captured
request is what the upstream received, the budget left out where it was sent again.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from continual_arc_baselines.tracking import TOKEN_FIELDS, LLMCall, Tracker

OPENROUTER = "https://openrouter.ai/api/v1"
PLACEHOLDER_KEY = "arc-proxy"
PLACEHOLDERS = (PLACEHOLDER_KEY, "ARC_LLM_API_KEY")
RETRY_STATUSES = {429, 500, 502, 503, 529}
WAIT_STATUSES = {402}  # the account is out of credit: wait for a top-up rather than hand the system a failure
# A 403 that OpenRouter relays from the provider behind it, retried like a 5xx: a provider may refuse calls at random,
# one request at a time. OpenRouter's own 403s (moderation, key limits) carry no such marker and pass straight through.
PROVIDER_REFUSAL = b"Provider returned error"
KEY_LIMIT = b"openrouter_key_limit"  # what OpenRouter says of a key that has spent its limit, with a 403
# The fields a chat completion asks its reply budget in, and the mark of a refusal for the room it asks: vLLM's "This
# model's maximum context length is 131072 tokens. However, you requested 32000 output tokens and your prompt contains
# 99073 input tokens ..." (vllm/renderers/params.py), OpenRouter's "This endpoint's maximum context length is ..."
REPLY_BUDGETS = ("max_tokens", "max_completion_tokens")
WINDOW_REFUSAL = b"maximum context length"
# The keys chat templates read an assistant turn's reasoning from, in the order a template that reads them all does.
THINKING_KEYS = ("think", "think_fast", "think_faster", "reasoning_content", "reasoning")
# A proxy's settings for a model served by vLLM (body_drop, role_map, thinking_field), as the open-model rows set them.
# The fields a harness may send that vLLM would refuse or ignore: OpenRouter's reasoning settings (vLLM validates
# reasoning_effort against its own few levels, and thinking is the chat template's switch, set for every call), its
# usage flag, routing and caching fields.
VLLM_DROP = ("reasoning_effort", "reasoning", "usage", "provider", "session_id", "cache_control", "transforms",
             "models", "route")
# Message roles vLLM's router refuses, renamed: prime-agent sends its system prompt as "developer" (OpenAI's name for
# it), which the router answers with 422 ("unknown role: developer"), and the model's chat template knows as "system".
VLLM_ROLES = {"developer": "system"}
# The key a past assistant turn carries its thinking under for the open models' chat templates, which refuse a turn
# with none.
THINKING_FIELD = "reasoning_content"
# The endpoints the proxy forwards, as a request names them under its version prefix: a chat completion, POSTed, and
# the model list or one model's entry, read with a GET (hermes-agent reads the model's window there).
CHAT, MODELS = "/chat/completions", "/models"
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding",
})
MODEL_ENTRY = re.compile(r"/models(/[A-Za-z0-9_~:.-]+)*")
MAX_BODY = 32 * 2**20      # the largest request read: a window's worth of text is a few megabytes
MAX_CLIENTS = 1024         # the connections served at once; past them a new one is closed
IDLE_S = 900.0             # how long a client may sit on a blocked read or write
# A confined proxy's request: the fields of a plain chat completion. Anything else is dropped before it goes up.
CONFINED_FIELDS = frozenset({
    "model", "messages", "stream", "stream_options", "temperature", "top_p", "top_k", "min_p", "top_a", "max_tokens",
    "max_completion_tokens", "stop", "seed", "frequency_penalty", "presence_penalty", "repetition_penalty", "logit_bias",
    "logprobs", "top_logprobs", "response_format", "structured_outputs", "tools", "tool_choice", "parallel_tool_calls",
    "reasoning", "reasoning_effort", "include_reasoning", "verbosity", "usage", "cache_control",
})
# The hosts a confined proxy's calls may go to, by OpenRouter's slugs: companies whose names a submitter could not stand
# behind. OpenRouter lists anyone who serves a model, a submitter's own company among them, and a host reads every prompt.
CONFINED_HOSTS = ("alibaba", "amazon-bedrock", "anthropic", "azure", "baseten", "cerebras", "cloudflare", "cohere", "coreweave",
                  "crusoe", "deepinfra", "deepseek", "digitalocean", "fireworks", "google-ai-studio", "google-vertex", "groq",
                  "minimax", "mistral", "modal", "moonshotai", "nebius", "novita", "nvidia", "openai", "parasail", "sambanova",
                  "siliconflow", "together", "xai", "z-ai")
# Where a confined proxy's calls may go: those hosts' endpoints that keep nothing and train on nothing (OpenRouter's routing).
CONFINED_ROUTING = {"zdr": True, "data_collection": "deny", "only": list(CONFINED_HOSTS)}
# The most a confined call may ask for in reply, which is what the spend ledger holds back for it (128,000 is a bare model's).
MAX_REPLY_TOKENS = 128_000
CONFINED_RATE = 5.0        # the calls per second a confined proxy forwards or refuses; more wait their turn
# How many calls that cost nothing, refused ones and reads of the model list, a confined proxy journals: past them it only
# counts them, since a learner could otherwise fill the verifier's disk with its journal at no cost to itself.
CONFINED_FREE_JOURNAL = 20_000


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _endpoint(path: str) -> str:
    """The endpoint a request's path names: its version prefix (``/v1``, or OpenRouter's ``/api/v1``) and its query left
    out."""
    path = path.split("?", 1)[0]
    for prefix in ("/api/v1", "/v1"):
        if path == prefix or path.startswith(prefix + "/"):
            path = path[len(prefix):]
            break
    return path.rstrip("/")


def forwarded(method: str, path: str) -> bool:
    """Whether the proxy sends a request upstream: a chat completion, or a read of the model list or of a model's
    entry, named plainly: a ``..`` in a path would reach any endpoint, since a client resolves it before it sends."""
    endpoint = _endpoint(path)
    if method == "POST":
        return endpoint == CHAT
    return (method == "GET" and MODEL_ENTRY.fullmatch(endpoint) is not None
            and not {".", ".."} & set(endpoint.split("/")))


class Refused(Exception):
    """A request a confined proxy will not send upstream: the status and the reason its client is told."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _alive(pid: int) -> bool:
    """Whether the process ``pid`` is still running on this machine."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class SpendLedger:
    """What one run's proxies, a process for each of its cells, have spent and hold back for their calls in flight, in a
    file they share, against the run's budget. A call holds back the most it could cost before it goes (:meth:`hold`, which
    waits while calls in flight may yet free enough), and what it did cost replaces that when it ends (:meth:`settle`). Once
    nothing in flight can free enough for the next call, the run is over budget.

    What is held is kept under the holding process's id and goes when that process has gone, so that a cell that dies with
    calls in flight, killed at the run's deadline or by its own learner's bomb, leaves nothing for the others to wait on."""

    EMPTY = {"spent": 0.0, "held": {}, "over": False}
    WAIT_S = 1.0

    def __init__(self, path: str | Path, cap: float):
        self.path, self.cap = Path(path), float(cap)

    def _with(self, change: Any) -> Any:
        with self.path.open("a+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            state = json.loads(file.read() or json.dumps(self.EMPTY))
            state["held"] = {pid: amount for pid, amount in state["held"].items() if _alive(int(pid))}
            answer = change(state)
            file.seek(0)
            file.truncate()
            file.write(json.dumps(state))
            return answer

    def hold(self, amount: float) -> bool:
        """Hold ``amount`` back for a call; False if the run is over, or nothing in flight can free enough for it."""
        me = str(os.getpid())

        def take(state: dict) -> bool | None:
            if state["over"]:
                return False
            held = sum(state["held"].values())
            if state["spent"] + held + amount <= self.cap:
                state["held"][me] = state["held"].get(me, 0.0) + amount
                return True
            return None if held > 1e-9 else False

        while (answer := self._with(take)) is None:
            time.sleep(self.WAIT_S)
        return answer

    def settle(self, held: float, spent: float) -> None:
        me = str(os.getpid())

        def change(state: dict) -> None:
            left = state["held"].pop(me, 0.0) - held
            if left > 1e-9:
                state["held"][me] = left
            state["spent"] += spent

        self._with(change)

    def correct(self, spent: float) -> bool:
        """Put what has been spent right from the upstream's own figure, when that is less: a call whose client went away
        ends with no price and counts as the most it could have cost. Whether it was less."""
        def change(state: dict) -> bool:
            lower = spent < state["spent"]
            if lower:
                state["spent"] = spent
            return lower

        return self._with(change)

    def close(self) -> None:
        """The run is over budget: every call from here on is refused."""
        self._with(lambda state: state.update(over=True))

    @property
    def over(self) -> bool:
        return self._with(lambda state: state["over"])

    @property
    def spent(self) -> float:
        return self._with(lambda state: state["spent"])


def _confined_content(content: Any) -> None:
    """A message's content in a confined request: text, or a list of parts each text or an image carried in the request
    itself, since a provider fetches any URL it is given, query string and all."""
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        raise Refused(400, "a message's content is text or a list of parts")
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            continue
        url = part.get("image_url") if kind == "image_url" else None
        url = url.get("url") if isinstance(url, dict) else url
        if not (isinstance(url, str) and url.startswith("data:")):
            raise Refused(400, f"a message part may be text or an image in a data: URL, not {str(kind)[:40]}")


def _confined(parsed: Any, models: dict[str, str | None]) -> dict:
    """A confined chat completion's body: :data:`CONFINED_FIELDS` of it, once its model and what it carries are allowed,
    at the reasoning effort the submission names for its model whatever the call asks, since a model played at more than it
    played bare would not be the model read beside it, and asking for at most :data:`MAX_REPLY_TOKENS` in reply."""
    if not isinstance(parsed, dict):
        raise Refused(400, "a chat completion is a JSON object")
    if not isinstance(parsed.get("model"), str) or parsed["model"] not in models:
        raise Refused(403, "the model is not one this submission names")
    messages = parsed.get("messages")
    if not isinstance(messages, list):
        raise Refused(400, "a chat completion has a list of messages")
    for message in messages:
        if not isinstance(message, dict):
            raise Refused(400, "a message is a JSON object")
        _confined_content(message.get("content"))
    tools = parsed.get("tools") or []
    if not isinstance(tools, list) or not all(isinstance(t, dict) and t.get("type") == "function" for t in tools):
        raise Refused(400, "a tool is a function the learner itself runs")
    choice = parsed.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") != "function":
        raise Refused(400, "a tool is a function the learner itself runs")
    body = {key: value for key, value in parsed.items() if key in CONFINED_FIELDS}
    effort = models[parsed["model"]]
    if effort:
        body.pop("reasoning_effort", None)
        body["reasoning"] = {"effort": effort}
    asked = [body[key] for key in REPLY_BUDGETS if key in body]
    if not all(isinstance(budget, int) and not isinstance(budget, bool) for budget in asked):
        raise Refused(400, "a reply budget is a whole number")
    if not asked:
        body["max_tokens"] = MAX_REPLY_TOKENS
    for key in REPLY_BUDGETS:
        if key in body:
            body[key] = min(body[key], MAX_REPLY_TOKENS)
    return body


def confined_routing(model: str, pinned: dict[str, str]) -> dict:
    """A confined call's routing: the hosts of :data:`CONFINED_ROUTING`, and the one endpoint ``pinned`` names for
    ``model``, if it names one, with no fallback, so that a host down holds the call back rather than serving it elsewhere."""
    routing = dict(CONFINED_ROUTING)
    if model in pinned:
        routing.update(order=[pinned[model]], allow_fallbacks=False)
    return routing


def _salted(body: dict, salt: str) -> None:
    """``salt`` at the head of ``body``'s first message, of its first tool's description and of its response format's
    schema, the three things a provider may render first and read its cache by, so that no prefix of the prompt is shared
    with a call under another salt."""
    mark = f"[{salt}]\n"
    if body["messages"]:
        first = body["messages"][0]
        content = first.get("content")
        first["content"] = [{"type": "text", "text": mark}, *content] if isinstance(content, list) else mark + (content or "")
    tools = body.get("tools") or []
    if tools and isinstance(tools[0].get("function"), dict):
        tools[0]["function"]["description"] = mark + str(tools[0]["function"].get("description") or "")
    schema = body.get("response_format")
    schema = schema.get("json_schema") if isinstance(schema, dict) else None
    if isinstance(schema, dict):
        schema["description"] = mark + str(schema.get("description") or "")


def own_model(learner_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The body overrides that send every chat completion of a learner to its own model, the ``model`` argument every
    adapter takes, whatever model the call names; none for a learner that takes no model, such as a submission run
    whole (:mod:`continual_arc_baselines.systems.hosted`), whose models its submission names."""
    return {"model": learner_kwargs["model"]} if "model" in learner_kwargs else {}


def _usage_tokens(usage: dict) -> dict[str, int | None]:
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    return {
        "input": _int(usage.get("prompt_tokens")),
        "output": _int(usage.get("completion_tokens")),
        "cache_read": _int(prompt.get("cached_tokens")),
        "cache_write": _int(prompt.get("cache_write_tokens")),
        "reasoning": _int(completion.get("reasoning_tokens")),
    }


def _absorb(call: LLMCall, payload: dict) -> None:
    """Fold a response object or stream chunk into ``call``."""
    if not isinstance(payload, dict):
        return
    if payload.get("id"):
        call.generation_id = payload["id"]
    if payload.get("model"):
        call.model = payload["model"]
    usage = payload.get("usage")
    if isinstance(usage, dict):
        tokens = _usage_tokens(usage)
        for key in TOKEN_FIELDS:
            if tokens[key] is not None:
                call.tokens[key] = tokens[key]
        cost = usage.get("cost")
        if isinstance(cost, (int, float, str)) and not isinstance(cost, bool):
            try:
                call.cost = Decimal(str(cost))
            except ArithmeticError:
                pass
    for choice in payload.get("choices") or []:
        if isinstance(choice, dict) and choice.get("finish_reason"):
            call.finish_reason = choice["finish_reason"]


def _snapshot(call: LLMCall) -> LLMCall:
    """Subscribers get an immutable-by-convention copy, not the live record."""
    return dataclasses.replace(call, tokens=dict(call.tokens),
                               response_headers=dict(call.response_headers))


class _StreamAssembler:
    """Fold streamed chat-completion chunks back into one response object, so a
    streamed call is captured the same way a non-streamed one is: the assembled
    ``message`` (content, reasoning, tool calls), the finish reason and the usage.
    Only used when content capture is on; the bytes forwarded to the system are
    untouched."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.model: str | None = None
        self.usage: dict | None = None
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.role: str = "assistant"
        # a vLLM server asked for return_token_ids: the prompt's ids on the first chunk, the
        # generated ids chunk by chunk
        self.prompt_token_ids: list[int] | None = None
        self.token_ids: list[int] = []

    def feed(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        self.id = event.get("id") or self.id
        self.model = event.get("model") or self.model
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        if isinstance(event.get("prompt_token_ids"), list):
            self.prompt_token_ids = event["prompt_token_ids"]
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if isinstance(choice.get("token_ids"), list):
                self.token_ids.extend(choice["token_ids"])
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("role"):
                self.role = delta["role"]
            if isinstance(delta.get("content"), str):
                self.content.append(delta["content"])
            for key in ("reasoning", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    self.reasoning.append(delta[key])
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                slot = self.tool_calls.setdefault(tc.get("index", len(self.tool_calls)),
                                                  {"id": None, "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    slot["function"]["arguments"] += fn["arguments"]

    def body(self) -> dict:
        message: dict = {"role": self.role, "content": "".join(self.content)}
        if self.reasoning:
            message["reasoning"] = "".join(self.reasoning)
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        choice: dict = {"index": 0, "message": message, "finish_reason": self.finish_reason}
        out = {"id": self.id, "model": self.model, "streamed": True, "choices": [choice], "usage": self.usage}
        if self.prompt_token_ids is not None:
            out["prompt_token_ids"] = self.prompt_token_ids
            choice["token_ids"] = self.token_ids
        return out


class _SSEParser:
    """Incrementally parse ``data:`` events out of a byte stream."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[dict]:
        self._buffer += chunk
        events: list[dict] = []
        while b"\n\n" in self._buffer:
            block, self._buffer = self._buffer.split(b"\n\n", 1)
            for line in block.splitlines():
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data and data != b"[DONE]":
                        try:
                            events.append(json.loads(data))
                        except ValueError:
                            pass
        return events


class _Server(ThreadingHTTPServer):
    """One thread per connection, at most :data:`MAX_CLIENTS` at once, whose clients may be hostile."""

    daemon_threads = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(MAX_CLIENTS)

    def process_request(self, request: Any, client_address: Any) -> None:
        if self._slots.acquire(blocking=False):
            super().process_request(request, client_address)
        else:
            self.close_request(request)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class LLMProxy:
    """Serve ``http://127.0.0.1:<port>/v1`` and forward to ``upstream``."""

    def __init__(
        self,
        upstream: str = OPENROUTER,
        api_key: str | None = None,
        tracker: Tracker | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = 600.0,
        max_connections: int | None = 100,
        upstream_retries: int = 6,
        credit_wait_s: float = 60.0,
        credit_wait_attempts: int = 240,
        include_usage: bool = True,
        body_defaults: dict[str, Any] | None = None,
        capture_content: bool = False,
        declared_context_length: int | None = 262_144,
        body_overrides: dict[str, Any] | None = None,
        body_drop: tuple[str, ...] | list[str] | None = None,
        role_map: dict[str, str] | None = None,
        thinking_field: str | None = None,
        serve: Any = None,
        confined_to: dict[str, str | None] | None = None,
        pinned: dict[str, str] | None = None,
        ledger: SpendLedger | None = None,
        prices: dict[str, dict[str, float]] | None = None,
    ):
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.tracker = tracker or Tracker()
        self.host, self._port = host, port
        self.timeout = timeout
        # the upstream calls open at once (httpx's default, 100); a call past it waits up to 30 s for a free one, then
        # fails. A relay for many parallel callers raises it
        self.max_connections = max_connections
        self.upstream_retries = upstream_retries  # on 429 and 5xx, with backoff up to a minute each
        # on 402 the account has no credit: every call would fail at zero cost and a learner that
        # does not retry would race through its stream as failures, so wait instead (4 h by default)
        self.credit_wait_s, self.credit_wait_attempts = credit_wait_s, credit_wait_attempts
        self.include_usage = include_usage
        # What a system that asks the endpoint about the model is told its context length is.
        # The adapters declare the same figure in every harness's own provider entry; a harness
        # that trusts the endpoint over its config (hermes-agent probes GET /models) must hear
        # the same number, or it runs against the provider's advertised window while the others
        # run against ours. None passes the provider's figures through.
        self.declared_context_length = declared_context_length
        self.body_defaults = dict(body_defaults or {})
        self.body_overrides: dict[str, Any] = dict(body_overrides or {})
        self.body_drop: tuple[str, ...] = tuple(body_drop or ())
        self.role_map: dict[str, str] = dict(role_map or {})
        self.thinking_field = thinking_field
        # a chat completion answered here instead of by the upstream: serve(parsed body, call id) -> a response body, or
        # None to forward the call as it is; sent to the harness as JSON, or as one streamed message when the call asked
        # for a stream, and captured and counted as any other. A streamed call is opened and kept alive while serve
        # runs (_Handler._serve), so a None after its first KEEPALIVE_S is no longer forwarded but fails the call
        self.serve = serve
        self.capture_content = capture_content
        # the models an untrusted learner may call and each one's reasoning effort, for a proxy in the confined mode this
        # module's docstring describes
        self.confined_to = confined_to
        # for a confined proxy, the one endpoint of CONFINED_HOSTS each model it names is served by, with no fallback
        # (confined_routing), by OpenRouter's endpoint tag, such as moonshotai/mxfp4
        self.pinned: dict[str, str] = dict(pinned or {})
        # the run's budget and, for each model it names, what a call costs at most (worst_cost)
        self.ledger, self.prices = ledger, prices or {}
        self.free_calls = 0      # a confined proxy's calls that cost nothing, of which it journals the first few
        self._pace_lock = threading.Lock()
        self._next_turn = 0.0
        self._pace_epoch = 0
        self._salt_key = secrets.token_bytes(32)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._client: httpx.Client | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> LLMProxy:
        if self._server is not None:
            return self
        # ``timeout`` is the read timeout: the longest silence tolerated on an upstream
        # stream before the request fails (and the system sees a 502 to retry). A
        # reasoning model streams deltas continuously, so silence this long means a hang.
        self._client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=self.timeout,
                                                          write=60.0, pool=30.0),
                                    limits=httpx.Limits(max_connections=self.max_connections, max_keepalive_connections=20))
        proxy = self

        class Handler(_Handler):
            owner = proxy

        self._server = _Server((self.host, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name="llm-proxy")
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def set_overrides(self, **fields: Any) -> None:
        """Replace some of :attr:`body_overrides` for every later call (a new adapter each round)."""
        self.body_overrides = {**self.body_overrides, **fields}

    def __enter__(self) -> LLMProxy:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # -- addressing ----------------------------------------------------------

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("proxy is not running")
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def environ(self) -> dict[str, str]:
        """Environment for a child system: every common base-URL convention."""
        return {
            "OPENAI_BASE_URL": self.base_url,
            "OPENAI_API_BASE": self.base_url,
            "OPENAI_API_KEY": PLACEHOLDER_KEY,
            "OPENROUTER_BASE_URL": self.base_url,
            "OPENROUTER_API_KEY": PLACEHOLDER_KEY,
            "ARC_LLM_BASE_URL": self.base_url,
            "ARC_LLM_API_KEY": PLACEHOLDER_KEY,
        }

    # -- forwarding ----------------------------------------------------------

    def upstream_url(self, path: str) -> str:
        if self.confined_to is not None:
            return self.upstream + _endpoint(path)  # no query string: nothing of the client's goes up in a URL
        for prefix in ("/api/v1", "/v1"):
            if path == prefix or path.startswith(prefix + "/"):
                path = path[len(prefix):]
                break
        return self.upstream + path

    def emit(self, call: LLMCall, quiet: bool) -> None:
        """A call's start or end to the tracker, or to its counters alone when it is ``quiet``."""
        if quiet:
            self.tracker.count(call)
        else:
            self.tracker.emit(_snapshot(call))

    def pace(self) -> None:
        """A confined proxy's turn for the next call, at :data:`CONFINED_RATE` per second."""
        if self.confined_to is None:
            return
        with self._pace_lock:
            if self._pace_epoch != self.tracker.epoch:  # a queue left by the last instance is no message to the next
                self._pace_epoch, self._next_turn = self.tracker.epoch, 0.0
            now = time.monotonic()
            turn = max(now, self._next_turn)
            self._next_turn = turn + 1.0 / CONFINED_RATE
        if turn > now:
            time.sleep(turn - now)

    def salt(self) -> str:
        """What opens every call of this epoch: 256 bits, secret to this process, new with each of the learner's resets."""
        return hashlib.sha256(self._salt_key + str(self.tracker.epoch).encode()).hexdigest()

    def worst_cost(self, parsed: dict, size: int) -> float:
        """The most a call of ``size`` bytes can cost: a token takes at least a byte, the model's window bounds the prompt,
        and the reply budget, which :func:`_confined` always sets, the reply."""
        price = self.prices[parsed["model"]]
        reply = max(parsed[key] for key in REPLY_BUDGETS if key in parsed)
        return price["request"] + min(size, price["window"]) * price["input"] + reply * price["output"]

    def spent_at_upstream(self) -> float | None:
        """What the key has spent by the upstream's own count, or None if it does not answer."""
        try:
            response = self._client.get(f"{self.upstream}/key", headers={"Authorization": f"Bearer {self.api_key}"})
            return float(response.raise_for_status().json()["data"]["usage"])
        except httpx.HTTPError:
            return None

    def afford(self, amount: float) -> bool:
        """``amount`` held back for a call in the run's ledger. Before the run is called over budget, the upstream's own
        figure replaces the ledger's, which counts a call with no price as the most it could have cost."""
        if self.ledger.hold(amount):
            return True
        if self.ledger.over:
            return False
        spent = self.spent_at_upstream()
        if spent is not None and self.ledger.correct(spent) and self.ledger.hold(amount):
            return True
        self.ledger.close()
        return False

    def prepare(self, headers: dict[str, str], body: bytes, path: str) -> tuple[dict[str, str], bytes, dict | None]:
        if self.confined_to is not None:
            # nothing of the client's headers goes up, and the body only as the fields allowed
            kept = {k: v for k, v in headers.items() if k.lower() == "accept"}
            headers = {**kept, "Content-Type": "application/json"}
            if _endpoint(path) == CHAT:
                try:
                    parsed = _confined(json.loads(body), self.confined_to)
                except (ValueError, RecursionError):
                    raise Refused(400, "a chat completion is a JSON object") from None
                _salted(parsed, self.salt())
                parsed["provider"] = confined_routing(parsed["model"], self.pinned)
                body = json.dumps(parsed).encode()
            else:
                body = b""
        out = {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}
        incoming = out.get("Authorization") or out.get("authorization")
        for key in list(out):
            if key.lower() == "authorization":
                del out[key]
        if self.api_key:
            out["Authorization"] = f"Bearer {self.api_key}"
        elif incoming and not any(p in incoming for p in PLACEHOLDERS):
            out["Authorization"] = incoming  # a system carrying its own real key
        out["Accept-Encoding"] = "identity"
        parsed: dict | None = None
        if body and "json" in (out.get("Content-Type") or out.get("content-type") or ""):
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = None
        if isinstance(parsed, dict) and _endpoint(path) == CHAT:
            for key, value in self.body_defaults.items():
                parsed.setdefault(key, value)
            parsed.update(self.body_overrides)
            if self.role_map:
                for message in parsed.get("messages") or []:
                    if isinstance(message, dict) and message.get("role") in self.role_map:
                        message["role"] = self.role_map[message["role"]]
            if self.thinking_field:
                # every past reply's reasoning under the one key the chat template renders, and an empty one for a
                # reply with none, which the template requires
                for message in parsed.get("messages") or []:
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        text = next((message[k] for k in THINKING_KEYS if isinstance(message.get(k), str)), "")
                        for k in THINKING_KEYS:
                            message.pop(k, None)
                        message[self.thinking_field] = text
            # Anthropic models are cached only when the request asks for it: OpenRouter
            # honours a top-level ``cache_control`` by placing the breakpoint on the last
            # cacheable block and moving it forward as the conversation grows. Systems
            # speak plain OpenAI format and never set it, so without this no call would read
            # the cache. Other providers cache on their own. A system that sets its own
            # breakpoints is left alone.
            model = str(parsed.get("model") or "")
            if model.startswith("anthropic/"):
                if "cache_control" not in parsed and not _has_block_cache_control(parsed):
                    # One-hour TTL: an agent's action can take longer than the default
                    # five minutes, after which the whole prefix is rewritten (2x write
                    # cost instead of 1.25x; reads cost the same 0.1x).
                    parsed["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                _canonicalise_volatile_prefix(parsed)
            if self.include_usage:
                usage = parsed.get("usage")
                parsed["usage"] = {**(usage if isinstance(usage, dict) else {}), "include": True}
                if parsed.get("stream"):
                    opts = parsed.get("stream_options")
                    parsed["stream_options"] = {**(opts if isinstance(opts, dict) else {}),
                                                "include_usage": True}
            for key in self.body_drop:
                parsed.pop(key, None)
            body = json.dumps(parsed).encode()
        return out, body, parsed if isinstance(parsed, dict) else None


_SESSION_PATH = re.compile(r"(sessions/)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\.jsonl)")


def _canonicalise_volatile_prefix(payload: dict) -> None:
    """A cache hit needs an identical prefix. prime-agent names its per-session
    conversation log (a fresh UUID per instance) inside its developer prompt, a few
    thousand characters in, so every new instance would write the whole prompt to the
    cache again. The path is informational to the model; it is replaced by a constant in
    system and developer messages only, before the request leaves for the provider."""
    for message in payload.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer"):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _SESSION_PATH.sub(r"\1session\2", content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = _SESSION_PATH.sub(r"\1session\2", part["text"])


def _declare_context_length(payload: dict, length: int) -> None:
    """Rewrite every model entry's advertised context length to ``length``, in a model list
    (``{"data": [...]}``) or a single model object."""
    entries = payload.get("data") if isinstance(payload.get("data"), list) else [payload]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("context_length", "max_context_length", "context_window"):
            if key in entry:
                entry[key] = length
        top = entry.get("top_provider")
        if isinstance(top, dict) and "context_length" in top:
            top["context_length"] = length


def _has_block_cache_control(payload: dict) -> bool:
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(isinstance(p, dict) and "cache_control" in p for p in content):
            return True
    return False


KEEPALIVE_S = 30.0   # how often a call being served (LLMProxy.serve) touches the tracker and, streamed, sends a comment
# What a streamed call whose serve gave up once its stream was open ends on: a mid-stream error as OpenRouter sends one,
# which a system's client raises and retries as an upstream's failure.
SERVE_FAILED = {"object": "chat.completion.chunk",
                "error": {"code": "server_error", "message": "the call was not answered"},
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}


def served_events(served: dict) -> list[dict]:
    """A response body as the chunks of a streamed one: the message in one delta (its role, content, reasoning under both
    keys, tool calls with their index), then the finish reason with the usage."""
    choice = (served.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    delta: dict[str, Any] = {"role": message.get("role") or "assistant", "content": message.get("content") or ""}
    for key in ("reasoning", "reasoning_content"):
        if message.get(key):
            delta[key] = message[key]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]
    head = {k: served[k] for k in ("id", "created", "model") if k in served}
    head["object"] = "chat.completion.chunk"
    return [{**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}],
             "usage": served.get("usage")}]


class _Handler(BaseHTTPRequestHandler):
    owner: LLMProxy
    protocol_version = "HTTP/1.1"
    timeout = IDLE_S

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if os.environ.get("ARC_PROXY_DEBUG"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")

    def _read_body(self) -> bytes | None:
        """The request's body, or None when it says it is longer than :data:`MAX_BODY` or has no honest length, unread,
        so that the connection is closed once it is answered."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True
            return None
        return self.rfile.read(length) if length else b""

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _open_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _end_stream(self, events: list[dict]) -> None:
        """An open stream's server-sent events, then ``[DONE]`` and its end."""
        for event in events:
            self._chunk(b"data: " + json.dumps(event).encode() + b"\n\n")
        self._chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _serve(self, parsed: dict, call: LLMCall, stream: bool, started: float) -> bool:
        """Answer a chat completion with :attr:`LLMProxy.serve`: as JSON, or, when the call asked for a stream, as
        server-sent events with the whole message in one delta, then its finish reason and usage, then ``[DONE]`` (what
        a client assembles into the same message; :func:`served_events`). False for a call the hook declined (None)
        before its stream opened, which is forwarded.

        The proxy is busy on the call, not hung. Every KEEPALIVE_S while serve runs it touches the tracker, as bytes
        moving would, so that an adapter's idle watchdog does not take a long serve for a dead system; and a streamed
        call gets its stream's headers and then a comment, as an upstream's stream carries bytes before its reply, so
        that the system's client keeps waiting: prime-agent's drops a call that brings nothing back for about five
        minutes and asks again, and the reply it dropped is paid for and lost. A hook that declines or fails once the
        stream is open ends it on :data:`SERVE_FAILED`. A reply is counted before it is sent, so one whose system went
        away still has its tokens and cost."""
        owner = self.owner
        writing, done = threading.Lock(), threading.Event()
        opened = False

        def keepalive() -> None:
            nonlocal opened
            while not done.wait(KEEPALIVE_S):
                owner.tracker.touch()
                with writing:
                    if not stream or done.is_set():
                        continue
                    try:
                        if not opened:
                            self._open_stream()
                            opened = True
                        self._chunk(b": serving\n\n")
                    except OSError:   # the system went away: the reply is still counted when it comes
                        return

        threading.Thread(target=keepalive, daemon=True, name="serve-keepalive").start()
        try:
            served = owner.serve(parsed, call.call_id)
        except Exception:  # noqa: BLE001  (declined: forwarded, or failed once the stream is open)
            logging.getLogger(__name__).exception("call %s: serve failed", call.call_id[:8])
            served = None
        finally:
            with writing:
                done.set()
        if served is None and not opened:
            return False
        try:
            if served is None:
                call.status, call.error = "failed", "serve gave no reply after the stream opened"
                self._end_stream([SERVE_FAILED])
                return True
            # the seconds of this call that were not the reply's generation (the served body's _budget_credit_s): its
            # adapter leaves them out of the turn's budget (Tracker.credit), or, negative, counts a replayed reply's
            # recorded generation that the call did not take; the system never sees the field
            owner.tracker.credit(float(served.pop("_budget_credit_s", 0.0) or 0.0))
            call.status_code, call.response_headers = 200, {"x-proxy-served": "local"}
            _absorb(call, served)
            if owner.capture_content:
                call.response_body = served
            if not stream:
                self._send_json(200, served)
            else:
                if not opened:
                    self._open_stream()
                self._end_stream(served_events(served))
            call.status = "completed"
        except OSError:   # the system went away (a broken pipe, a reset, or macOS's EPROTOTYPE on a closing socket)
            if served is not None:
                call.status, call.error = "cancelled", "client disconnected"
        finally:
            call.latency_ms = (time.monotonic() - started) * 1000
            owner.tracker.emit(_snapshot(call))
        return True

    def _refuse(self, status: int, message: str, call: LLMCall, started: float, quiet: bool) -> None:
        """Answer a request the proxy does not forward, as an upstream answers one it does not serve, and count it as a
        failed call; it ends once, whether or not its system is still there to read the answer."""
        call.status, call.status_code, call.error = "failed", status, f"HTTP {status}: {message[:200]}"
        call.refused = True
        try:
            self._send_json(status, {"error": {"message": message, "type": "proxy_refused", "code": status}})
        except OSError:   # the system went away
            call.status, call.error = "cancelled", "client disconnected"
        finally:
            call.latency_ms = (time.monotonic() - started) * 1000
            self.owner.emit(call, quiet)

    def _forward(self, method: str) -> None:
        owner = self.owner
        assert owner._client is not None
        owner.pace()
        body = self._read_body()
        refusal: tuple[int, str] | None = None
        headers: dict[str, str] = {}
        parsed: dict | None = None
        if body is None:
            refusal = 413, f"the proxy reads requests of {MAX_BODY} bytes or fewer"
        else:
            try:
                headers, body, parsed = owner.prepare(dict(self.headers.items()), body, self.path)
                if not forwarded(method, self.path):
                    refusal = 404, f"the proxy forwards chat completions and the model list alone, not {method} {self.path[:200]}"
            except Refused as why:
                refusal = why.status, str(why)
        held = 0.0
        if refusal is None and owner.ledger is not None and parsed is not None and _endpoint(self.path) == CHAT:
            held = owner.worst_cost(parsed, len(body))
            if not owner.afford(held):
                refusal, held, parsed = (402, "this run's budget is spent"), 0.0, None
        stream = bool(parsed and parsed.get("stream"))
        call = LLMCall(
            call_id=uuid.uuid4().hex, status="started", path=self.path[:400], stream=stream,
            requested_model=(parsed or {}).get("model") if parsed else None,
            request_body=parsed if owner.capture_content else None,
        )
        owner.free_calls += bool(refusal) or method == "GET"
        quiet = (bool(refusal) or method == "GET") and owner.confined_to is not None and (
            owner.free_calls > CONFINED_FREE_JOURNAL)
        owner.emit(call, quiet)
        started = time.monotonic()
        if refusal:
            self._refuse(*refusal, call, started, quiet)
            return
        if (owner.serve is not None and parsed is not None and _endpoint(self.path) == CHAT
                and self._serve(parsed, call, stream, started)):
            return
        sent = False
        try:
            # A throttled or failing upstream is infrastructure, not the system's answer: retry
            # with backoff before anything reaches the system, the same for every harness.
            attempts = 0
            waits = 0
            overran: dict[str, Any] = {}   # the reply budget the window had no room for, left out of the call sent again
            while True:
                cm = owner._client.stream(method, owner.upstream_url(self.path), headers=headers, content=body)
                response = cm.__enter__()
                if response.status_code in WAIT_STATUSES and waits < owner.credit_wait_attempts:
                    cm.__exit__(None, None, None)
                    waits += 1
                    owner.tracker.touch()
                    time.sleep(owner.credit_wait_s)
                    continue
                if owner.ledger is not None and response.status_code == 403 and KEY_LIMIT in response.read():
                    owner.ledger.close()
                retry = response.status_code in RETRY_STATUSES or (
                    response.status_code == 403 and PROVIDER_REFUSAL in response.read())
                if retry and attempts < owner.upstream_retries:
                    retry_after = response.headers.get("retry-after")
                    cm.__exit__(None, None, None)
                    attempts += 1
                    try:
                        delay = float(retry_after)
                    except (TypeError, ValueError):
                        delay = 2.0 ** attempts
                    owner.tracker.touch()
                    time.sleep(min(max(delay, 1.0), 60.0))
                    continue
                if (not overran and response.status_code == 400 and parsed is not None
                        and any(key in parsed for key in REPLY_BUDGETS) and WINDOW_REFUSAL in response.read().lower()):
                    cm.__exit__(None, None, None)
                    overran = {key: parsed.pop(key) for key in REPLY_BUDGETS if key in parsed}
                    body = json.dumps(parsed).encode()
                    owner.tracker.touch()
                    continue
                break
            refusal = ""
            try:
                call.status_code = response.status_code
                call.response_headers = {k.lower(): v for k, v in response.headers.items()
                                         if k.lower().startswith("x-")}
                if attempts:
                    call.response_headers["x-proxy-upstream-retries"] = str(attempts)
                if waits:
                    call.response_headers["x-proxy-credit-waits"] = str(waits)
                if overran:
                    call.response_headers["x-proxy-reply-budget-dropped"] = json.dumps(overran)
                self.send_response(response.status_code)
                passthrough = {k: v for k, v in response.headers.items()
                               if k.lower() not in HOP_BY_HOP and k.lower() != "content-encoding"}
                is_sse = "text/event-stream" in (response.headers.get("content-type") or "")
                sent = True
                if is_sse:
                    for key, value in passthrough.items():
                        self.send_header(key, value)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    parser = _SSEParser()
                    assembler = _StreamAssembler() if owner.capture_content else None
                    for chunk in response.iter_bytes():
                        owner.tracker.touch()
                        for event in parser.feed(chunk):
                            if response.status_code < 400:
                                _absorb(call, event)
                                if assembler is not None:
                                    assembler.feed(event)
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    if assembler is not None and response.status_code < 400:
                        call.response_body = assembler.body()
                else:
                    data = response.read()
                    try:
                        payload = json.loads(data)
                    except ValueError:
                        payload = None
                    if response.status_code >= 400:
                        refusal = data[:400].decode("utf-8", "replace")
                    if (method == "GET" and response.status_code < 400 and isinstance(payload, dict)
                            and owner.declared_context_length):   # a read of the models, the one GET forwarded
                        _declare_context_length(payload, owner.declared_context_length)
                        data = json.dumps(payload).encode()
                    for key, value in passthrough.items():
                        if key.lower() != "content-length":
                            self.send_header(key, value)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    if isinstance(payload, dict) and response.status_code < 400:
                        _absorb(call, payload)
                        if owner.capture_content:
                            call.response_body = payload
                call.latency_ms = (time.monotonic() - started) * 1000
                call.status = "completed" if response.status_code < 400 else "failed"
                if call.status == "failed":
                    call.error = f"HTTP {response.status_code}" + (f" after {attempts} retries" if attempts else "")
                    if refusal:  # what the upstream said, for the calls log
                        call.error += f": {refusal}"
            finally:
                cm.__exit__(None, None, None)
        except (BrokenPipeError, ConnectionResetError):
            call.latency_ms = (time.monotonic() - started) * 1000
            call.status, call.error = "cancelled", "client disconnected"
        except httpx.HTTPError as error:
            call.latency_ms = (time.monotonic() - started) * 1000
            call.status, call.error = "failed", type(error).__name__
            if sent:
                # the upstream broke once its reply had begun: the client's connection is closed, as the upstream's
                # was, so that its read fails now instead of waiting on a kept-alive connection for the rest of a reply
                # that never comes (a bare learner's waited out its hour-long read timeout and restarted its system)
                self.close_connection = True
                return
            try:
                self._send_json(502, {"error": {"message": f"upstream {type(error).__name__}: {error}",
                                                "type": "proxy_upstream_error"}})
            except OSError:
                pass
        except Exception as error:  # anything else: the call still ends, once (a user's instance waits for its calls)
            if call.status == "started":
                call.latency_ms = (time.monotonic() - started) * 1000
                call.status, call.error = "failed", f"proxy {type(error).__name__}"
            raise
        finally:
            if held:
                owner.ledger.settle(held, float(call.cost) if call.cost is not None else held)
            owner.emit(call, quiet)
