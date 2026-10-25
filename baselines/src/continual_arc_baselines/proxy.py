"""A local OpenAI-compatible reverse proxy: the one hook every system can use.

Point any agent system at :attr:`LLMProxy.base_url` with the placeholder :attr:`LLMProxy.api_key` and every physical
model request is observed, journaled and counted, without the system knowing. The real upstream credential lives only
in this process, so it never enters the system's configuration files or logs.

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
``serve`` lets a run answer some chat completions itself. The captured request is what the upstream received.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
import time
import uuid
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding",
})


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


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
        # None to forward the call as it is; sent to the harness
        # as JSON, or as one streamed message when the call asked for a stream, and captured and counted as any other
        self.serve = serve
        self.capture_content = capture_content
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

        self._server = ThreadingHTTPServer((self.host, self._port), Handler)
        self._server.daemon_threads = True
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
        for prefix in ("/api/v1", "/v1"):
            if path == prefix or path.startswith(prefix + "/"):
                path = path[len(prefix):]
                break
        return self.upstream + path

    def prepare(self, headers: dict[str, str], body: bytes, path: str) -> tuple[dict[str, str], bytes, dict | None]:
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
        if isinstance(parsed, dict) and path.rstrip("/").endswith("/chat/completions"):
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


def _is_models_path(path: str) -> bool:
    p = path.split("?", 1)[0].rstrip("/")
    return p.endswith("/models") or "/models/" in p


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


KEEPALIVE_S = 30.0   # how often a call being served (LLMProxy.serve) touches the tracker


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

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if os.environ.get("ARC_PROXY_DEBUG"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_served(self, served: dict, stream: bool) -> None:
        """A response made here (:attr:`LLMProxy.serve`): as JSON, or, when the call asked for a stream, as server-sent
        events with the whole message in one delta, then its finish reason and usage, then ``[DONE]`` (what a client
        assembles into the same message; :func:`served_events`)."""
        if not stream:
            self._send_json(200, served)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for event in served_events(served) + ["[DONE]"]:
            data = b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _forward(self, method: str) -> None:
        owner = self.owner
        assert owner._client is not None
        body = self._read_body()
        headers, body, parsed = owner.prepare(dict(self.headers.items()), body, self.path)
        stream = bool(parsed and parsed.get("stream"))
        call = LLMCall(
            call_id=uuid.uuid4().hex, status="started", path=self.path, stream=stream,
            requested_model=(parsed or {}).get("model") if parsed else None,
            request_body=parsed if owner.capture_content else None,
        )
        owner.tracker.emit(_snapshot(call))
        started = time.monotonic()
        if owner.serve is not None and parsed is not None and self.path.rstrip("/").endswith("/chat/completions"):
            # the proxy is busy on the call, not hung: its tracker is touched every KEEPALIVE_S while serve runs, as bytes
            # moving would touch it, so that an adapter's idle watchdog does not take a long serve (minutes, near
            # prime-agent's idle timeout) for a dead system
            done = threading.Event()

            def keepalive() -> None:
                while not done.wait(KEEPALIVE_S):
                    owner.tracker.touch()

            threading.Thread(target=keepalive, daemon=True, name="serve-keepalive").start()
            try:
                served = owner.serve(parsed, call.call_id)
            except Exception:  # noqa: BLE001  (the call is forwarded instead)
                logging.getLogger(__name__).exception("call %s: serve failed; the call is forwarded", call.call_id[:8])
                served = None
            finally:
                done.set()
            if served is not None:
                # the seconds of this call that were not the reply's generation (the served body's _budget_credit_s):
                # its adapter leaves them out of the turn's budget (Tracker.credit), or, negative, counts a replayed
                # reply's recorded generation that the call did not take; the system never sees the field
                owner.tracker.credit(float(served.pop("_budget_credit_s", 0.0) or 0.0))
                try:
                    self._send_served(served, stream)
                    call.status_code = 200
                    call.response_headers = {"x-proxy-served": "local"}
                    _absorb(call, served)
                    if owner.capture_content:
                        call.response_body = served
                    call.latency_ms = (time.monotonic() - started) * 1000
                    call.status = "completed"
                except (BrokenPipeError, ConnectionResetError):
                    call.latency_ms = (time.monotonic() - started) * 1000
                    call.status, call.error = "cancelled", "client disconnected"
                finally:
                    owner.tracker.emit(_snapshot(call))
                return
        sent = False
        try:
            # A throttled or failing upstream is infrastructure, not the system's answer: retry
            # with backoff before anything reaches the system, the same for every harness.
            attempts = 0
            waits = 0
            while True:
                cm = owner._client.stream(method, owner.upstream_url(self.path), headers=headers, content=body)
                response = cm.__enter__()
                if response.status_code in WAIT_STATUSES and waits < owner.credit_wait_attempts:
                    cm.__exit__(None, None, None)
                    waits += 1
                    owner.tracker.touch()
                    time.sleep(owner.credit_wait_s)
                    continue
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
                            and owner.declared_context_length and _is_models_path(self.path)):
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
            owner.tracker.emit(_snapshot(call))
