import json
import time
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from continual_arc_baselines.attempt import Attempt
from continual_arc_baselines.costs import CostJournal
from continual_arc_baselines.proxy import PLACEHOLDER_KEY, LLMProxy
from continual_arc_baselines.tracking import JournalSink, Tracker

from .fake_upstream import FakeUpstream


@pytest.fixture
def upstream():
    server = FakeUpstream().start()
    yield server
    server.stop()


def chat(proxy: LLMProxy, **body):
    payload = {"model": "vendor/model", "messages": [{"role": "user", "content": "hi"}], **body}
    return httpx.post(proxy.base_url + "/chat/completions", json=payload,
                      headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"}, timeout=10)


def test_non_streaming_call_is_forwarded_and_counted(upstream):
    tracker = Tracker()
    seen = []
    tracker.subscribe(seen.append)
    with LLMProxy(upstream.base_url, api_key="sk-real", tracker=tracker,
                  body_defaults={"reasoning_effort": "xhigh"}) as proxy:
        response = chat(proxy)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "OK"
    sent = upstream.requests[0]
    assert sent["authorization"] == "Bearer sk-real"          # the system never held the key
    assert sent["body"]["usage"] == {"include": True}          # provider asked to report cost
    assert sent["body"]["reasoning_effort"] == "xhigh"         # run-level default applied
    assert [c.status for c in seen] == ["started", "completed"]
    done = seen[-1]
    assert done.generation_id == "gen-1" and done.model == "vendor/model"
    assert done.tokens["input"] == 12 and done.tokens["output"] == 3 and done.tokens["reasoning"] == 2
    assert done.cost == Decimal("0.00025") and done.finish_reason == "stop"
    usage = tracker.usage.as_resources()
    assert usage["llm_requests"] == 1 and usage["llm_failed_requests"] == 0
    assert usage["llm_input_tokens"] == 12 and usage["llm_cost_usd"] == pytest.approx(0.00025)


def test_anthropic_requests_get_a_cache_breakpoint(upstream):
    with LLMProxy(upstream.base_url) as proxy:
        chat(proxy, model="anthropic/claude-sonnet-5")
        chat(proxy, model="vendor/model")
        chat(proxy, model="anthropic/claude-sonnet-5", cache_control={"type": "ephemeral", "ttl": "1h"})
    sent = [r["body"] for r in upstream.requests]
    assert sent[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}   # added for Anthropic
    assert "cache_control" not in sent[1]                        # other providers cache on their own
    assert sent[2]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}  # the system's own choice stands


def test_anthropic_developer_prompt_session_path_is_canonicalised(upstream):
    dev = "Conversation log: /x/sessions/01a0be98-2e8d-70bf-8a2e-159bd10bfbce.jsonl\nRules follow."
    with LLMProxy(upstream.base_url) as proxy:
        chat(proxy, model="anthropic/claude-sonnet-5",
             messages=[{"role": "developer", "content": dev}, {"role": "user", "content": "hi"}])
        chat(proxy, model="vendor/model",
             messages=[{"role": "developer", "content": dev}, {"role": "user", "content": "hi"}])
    a, b = (r["body"]["messages"][0]["content"] for r in upstream.requests)
    assert a == "Conversation log: /x/sessions/session.jsonl\nRules follow."  # stable prefix for the cache
    assert b == dev                                                          # other providers untouched


def test_system_choice_beats_body_defaults(upstream):
    with LLMProxy(upstream.base_url, body_defaults={"reasoning_effort": "xhigh"}) as proxy:
        chat(proxy, reasoning_effort="low")
    assert upstream.requests[0]["body"]["reasoning_effort"] == "low"


def test_streaming_passthrough_and_usage(upstream):
    tracker = Tracker()
    with LLMProxy(upstream.base_url, tracker=tracker) as proxy:
        with httpx.stream("POST", proxy.base_url + "/chat/completions", timeout=10, json={
            "model": "vendor/model", "messages": [], "stream": True,
        }) as response:
            raw = b"".join(response.iter_bytes())
    events = [json.loads(line[5:]) for line in raw.decode().splitlines()
              if line.startswith("data:") and "[DONE]" not in line]
    assert events[0]["choices"][0]["delta"]["content"] == "OK"
    assert upstream.requests[0]["body"]["stream_options"] == {"include_usage": True}
    usage = tracker.usage.as_resources()
    assert usage["llm_requests"] == 1 and usage["llm_output_tokens"] == 3
    assert usage["llm_cost_usd"] == pytest.approx(0.00025)


def test_streamed_reply_is_captured_when_content_capture_is_on(upstream):
    seen = []
    tracker = Tracker()
    tracker.subscribe(seen.append)
    with LLMProxy(upstream.base_url, tracker=tracker, capture_content=True) as proxy:
        with httpx.stream("POST", proxy.base_url + "/chat/completions", timeout=10, json={
            "model": "vendor/model", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        }) as response:
            b"".join(response.iter_bytes())
    completed = [c for c in seen if c.status == "completed"]
    assert len(completed) == 1
    assert completed[0].request_body["messages"][0]["content"] == "hi"
    body = completed[0].response_body
    assert body["streamed"] is True
    assert body["choices"][0]["message"]["content"] == "OK"     # the deltas, reassembled
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3


def test_upstream_error_is_a_failed_request(upstream):
    upstream.status = 500
    tracker = Tracker()
    with LLMProxy(upstream.base_url, tracker=tracker, upstream_retries=0) as proxy:
        assert chat(proxy).status_code == 500
    assert tracker.usage.failed_requests == 1 and tracker.usage.cost_usd == 0


def test_unreachable_upstream_yields_502():
    tracker = Tracker()
    with LLMProxy("http://127.0.0.1:9", tracker=tracker) as proxy:
        response = chat(proxy)
    assert response.status_code == 502 and response.json()["error"]["type"] == "proxy_upstream_error"
    assert tracker.usage.failed_requests == 1


def test_a_call_the_proxy_itself_fails_on_still_ends_once(upstream, monkeypatch):
    # a user's instance waits for every call started for it to end, so none may stay started
    def broken(call, payload):
        raise ValueError("a bug in the proxy")

    monkeypatch.setattr("continual_arc_baselines.proxy._absorb", broken)
    tracker = Tracker()
    seen = []
    tracker.subscribe(seen.append)
    with LLMProxy(upstream.base_url, tracker=tracker) as proxy:
        assert chat(proxy).status_code == 200  # the reply was written before the proxy failed on it
        deadline = time.monotonic() + 5
        while len(seen) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert [c.status for c in seen] == ["started", "failed"] and seen[-1].error == "proxy ValueError"
    assert tracker.usage.requests == 1 and tracker.usage.failed_requests == 1


def test_openrouter_style_path_prefix_is_accepted(upstream):
    with LLMProxy(upstream.base_url) as proxy:
        url = proxy.base_url.replace("/v1", "/api/v1") + "/chat/completions"
        assert httpx.post(url, json={"model": "m", "messages": []}, timeout=10).status_code == 200
    assert upstream.requests[0]["path"] == "/v1/chat/completions"


def test_journal_sink_writes_the_cost_journal_and_the_event_log(upstream, tmp_path):
    context = Attempt.create("proxy run", root=tmp_path)
    tracker = Tracker()
    tracker.subscribe(JournalSink(context, provider="openrouter"))
    with LLMProxy(upstream.base_url, tracker=tracker) as proxy:
        chat(proxy)
        chat(proxy, stream=True)
    summary = CostJournal(context.directory).summary()
    assert summary["entities"] == 2
    charge = summary["charges"][0]
    assert charge["account_charge"] == {"known_total": "0.00050", "unknown_entities": 0}
    assert summary["tokens"]["output"] == {"known_total": 6, "unknown_entities": 0}
    kinds = [json.loads(line)["kind"] for line in (context.directory / "events.jsonl").read_text().splitlines()]
    assert kinds.count("llm_usage") == 2 and kinds.count("llm_request") == 2


# -- the network bridge (private network namespace on Linux) --------------------

def test_netbridge_relays_between_tcp_and_unix(tmp_path):
    """Host half (unix2tcp) plus sandbox half (tcp2unix) relay a request end to end."""
    import http.server
    import socket
    import subprocess
    import sys
    import threading
    import time
    import urllib.request

    from continual_arc_baselines.systems.sandbox import proxy_port, start_bridge

    class Upstream(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            data = b"hello from upstream"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    port = up.server_address[1]
    assert proxy_port(f"http://127.0.0.1:{port}/v1") == port and proxy_port("https://openrouter.ai/api/v1") is None
    import tempfile
    short = Path(tempfile.mkdtemp(prefix="nb-", dir="/tmp"))  # UNIX socket paths are length-limited
    sock = short / "p.sock"
    script = tmp_path / "netbridge.py"
    inner, host = start_bridge(["true"], port, sock, script)
    assert inner[:2] == ["/usr/bin/python3", str(script)] and inner[2] == "tcp2unix" and inner[-1] == "true"
    assert sock.exists() and host is not None
    # The sandbox half, on a different port here (same host, no namespace in the test).
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    inner_port = s.getsockname()[1]
    s.close()
    sandbox_half = subprocess.Popen([sys.executable, str(script), "tcp2unix", str(inner_port), str(sock)])
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{inner_port}/x", timeout=2).read()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("bridge never answered")
        assert body == b"hello from upstream"
    finally:
        sandbox_half.kill()
        host.kill()
        up.shutdown()


def test_rate_limited_upstream_is_retried_before_the_system_sees_it():
    server = FakeUpstream(status=[429, 429, 200]).start()
    try:
        tracker = Tracker()
        seen = []
        tracker.subscribe(seen.append)
        with LLMProxy(server.base_url, api_key="sk-real", tracker=tracker) as proxy:
            response = chat(proxy)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "OK"
        assert len(server.requests) == 3
        assert [c.status for c in seen] == ["started", "completed"]
        assert seen[-1].response_headers.get("x-proxy-upstream-retries") == "2"
        assert tracker.usage.as_resources()["llm_failed_requests"] == 0
    finally:
        server.stop()


def test_a_refusal_relayed_from_the_provider_is_retried_but_an_openrouter_refusal_is_not():
    relayed = {"error": {"message": "Provider returned error", "code": 403, "metadata": {
        "provider_name": "Alibaba", "raw": "Access to model denied. (AccessDenied.Unpurchased)"}}}
    server = FakeUpstream(status=[403, 200], error=relayed).start()
    try:
        tracker = Tracker()
        seen = []
        tracker.subscribe(seen.append)
        with LLMProxy(server.base_url, api_key="sk-real", tracker=tracker) as proxy:
            response = chat(proxy)
        assert response.status_code == 200 and len(server.requests) == 2
        assert seen[-1].response_headers.get("x-proxy-upstream-retries") == "1"
    finally:
        server.stop()
    moderated = {"error": {"message": "vendor/model requires moderation. Your input was flagged.", "code": 403}}
    server = FakeUpstream(status=403, error=moderated).start()
    try:
        tracker = Tracker()
        seen = []
        tracker.subscribe(seen.append)
        with LLMProxy(server.base_url, api_key="sk-real", tracker=tracker) as proxy:
            response = chat(proxy)
        assert response.status_code == 403 and response.json() == moderated and len(server.requests) == 1
        assert seen[-1].status == "failed" and "requires moderation" in seen[-1].error  # the reason, in the calls log
    finally:
        server.stop()


def test_every_past_reply_carries_its_reasoning_under_the_thinking_field_and_one_without_an_empty_one():
    server = FakeUpstream().start()
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a", "reasoning": "because"},
                {"role": "assistant", "content": "b", "reasoning_content": None},
                {"role": "assistant", "content": "c"},
                {"role": "user", "content": "go on"}]
    try:
        with LLMProxy(server.base_url, thinking_field="reasoning_content") as proxy:
            httpx.post(proxy.base_url + "/chat/completions", json={"model": "m", "messages": messages}, timeout=10)
        with LLMProxy(server.base_url) as proxy:
            httpx.post(proxy.base_url + "/chat/completions", json={"model": "m", "messages": messages}, timeout=10)
    finally:
        server.stop()
    fixed, untouched = (r["body"]["messages"] for r in server.requests)
    assert fixed[1] == {"role": "assistant", "content": "a", "reasoning_content": "because"}  # the one key
    assert fixed[2] == {"role": "assistant", "content": "b", "reasoning_content": ""}
    assert fixed[3] == {"role": "assistant", "content": "c", "reasoning_content": ""}
    assert fixed[0] == messages[0] and fixed[4] == messages[4] and untouched == messages


def test_out_of_credit_upstream_is_waited_out_not_failed():
    server = FakeUpstream(status=[402, 402, 200]).start()
    try:
        tracker = Tracker()
        seen = []
        tracker.subscribe(seen.append)
        with LLMProxy(server.base_url, api_key="sk-real", tracker=tracker, credit_wait_s=0.05) as proxy:
            response = chat(proxy)
        assert response.status_code == 200
        assert len(server.requests) == 3
        assert seen[-1].response_headers.get("x-proxy-credit-waits") == "2"
        assert tracker.usage.as_resources()["llm_failed_requests"] == 0
    finally:
        server.stop()


def test_out_of_credit_gives_up_only_after_the_wait_budget():
    server = FakeUpstream(status=402).start()
    try:
        with LLMProxy(server.base_url, api_key="sk-real", credit_wait_s=0.01, credit_wait_attempts=3) as proxy:
            response = chat(proxy)
        assert response.status_code == 402
        assert len(server.requests) == 4
    finally:
        server.stop()


def test_models_probe_hears_the_declared_context_length():
    """A harness that asks the endpoint what the model's window is (hermes-agent does) must hear
    the figure the adapters declare, not the provider's advertised one."""
    server = FakeUpstream().start()
    try:
        with LLMProxy(server.base_url, api_key="sk-real", declared_context_length=262_144) as proxy:
            listing = httpx.get(f"{proxy.base_url}/models", headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"}, timeout=10)
        entry = listing.json()["data"][0]
        assert entry["context_length"] == 262_144
        assert entry["top_provider"]["context_length"] == 262_144
        with LLMProxy(server.base_url, api_key="sk-real", declared_context_length=None) as proxy:
            raw = httpx.get(f"{proxy.base_url}/models", headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"}, timeout=10)
        assert raw.json()["data"][0]["context_length"] == 1_050_000
    finally:
        server.stop()


def test_body_defaults_reach_the_upstream_without_overriding_the_system():
    """The run's defaults (reasoning effort, OpenRouter session_id) are added when absent and
    left alone when the system set them."""
    server = FakeUpstream().start()
    try:
        with LLMProxy(server.base_url, api_key="sk-real",
                      body_defaults={"reasoning_effort": "low", "session_id": "attempt-1"}) as proxy:
            chat(proxy)
            httpx.post(f"{proxy.base_url}/chat/completions", headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"},
                       json={"model": "vendor/model", "messages": [{"role": "user", "content": "hi"}],
                             "session_id": "mine"}, timeout=10)
        assert server.requests[0]["body"]["session_id"] == "attempt-1"
        assert server.requests[0]["body"]["reasoning_effort"] == "low"
        assert server.requests[1]["body"]["session_id"] == "mine"
    finally:
        server.stop()


def test_overrides_replace_the_systems_fields_and_drop_removes_them():
    """A run serving its own model forces the adapter's name and asks for token ids, removes the
    fields its server would reject, and can move to a new adapter between calls; the captured
    request is what the server received."""
    server = FakeUpstream().start()
    seen = []
    tracker = Tracker()
    tracker.subscribe(seen.append)
    try:
        with LLMProxy(server.base_url, tracker=tracker, capture_content=True,
                      body_overrides={"model": "adapter-1", "return_token_ids": True},
                      body_drop=("reasoning_effort", "usage"), role_map={"developer": "system"}) as proxy:
            chat(proxy, reasoning_effort="xhigh", messages=[{"role": "developer", "content": "be brief"},
                                                            {"role": "user", "content": "hi"}])
            proxy.set_overrides(model="adapter-2")
            chat(proxy)
        sent = [r["body"] for r in server.requests]
        assert [b["model"] for b in sent] == ["adapter-1", "adapter-2"]
        assert all(b["return_token_ids"] is True and "reasoning_effort" not in b and "usage" not in b for b in sent)
        assert [m["role"] for m in sent[0]["messages"]] == ["system", "user"]
        done = [c for c in seen if c.status == "completed"]
        assert done[0].request_body == sent[0]
        assert done[0].response_body["prompt_token_ids"] == [2, 1, 2]   # two messages, then 1, 2
        assert done[0].response_body["choices"][0]["token_ids"] == [7, 8]
    finally:
        server.stop()


def test_streamed_token_ids_are_assembled(upstream):
    seen = []
    tracker = Tracker()
    tracker.subscribe(seen.append)
    with LLMProxy(upstream.base_url, tracker=tracker, capture_content=True,
                  body_overrides={"return_token_ids": True}) as proxy:
        with httpx.stream("POST", proxy.base_url + "/chat/completions", timeout=10, json={
            "model": "vendor/model", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        }) as response:
            b"".join(response.iter_bytes())
    body = [c for c in seen if c.status == "completed"][0].response_body
    assert body["prompt_token_ids"] == [1, 1, 2] and body["choices"][0]["token_ids"] == [7, 8]


def test_a_served_reply_answers_the_system_without_the_upstream_as_json_or_as_a_stream(upstream):
    """LLMProxy(serve=...): a chat completion answered here reaches the system as JSON,
    or as one streamed delta with the finish reason and usage when it asked for a stream; it is counted and captured
    like any other, the upstream never sees it, and a call the hook declines (None) is forwarded as before."""
    from continual_arc_baselines.proxy import served_events

    seen, asked = [], []
    tracker = Tracker()
    tracker.subscribe(seen.append)

    def serve(parsed, call_id):
        asked.append((parsed["messages"][-1]["content"], call_id))
        if parsed["messages"][-1]["content"] == "forward me":
            return None
        return {"id": "chatcmpl-thought-1", "object": "chat.completion", "model": parsed["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"action": "request_demos"}',
                                                     "reasoning": "Nothing here.", "reasoning_content": "Nothing here.",
                                                     "tool_calls": [{"id": "call_1", "type": "function",
                                                                     "function": {"name": "ipython", "arguments": "{\"code\": \"1\"}"}}]},
                             "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 7, "total_tokens": 47}, "thought": "Nothing here.",
                "_budget_credit_s": 2.5}

    with LLMProxy(upstream.base_url, tracker=tracker, capture_content=True, serve=serve) as proxy:
        response = chat(proxy)
        body = response.json()
        with httpx.stream("POST", proxy.base_url + "/chat/completions", timeout=10, json={
            "model": "vendor/model", "messages": [{"role": "user", "content": "hi"}], "stream": True}) as streamed:
            raw = b"".join(streamed.iter_bytes())
        forwarded = chat(proxy, messages=[{"role": "user", "content": "forward me"}])
    assert response.status_code == 200 and body["choices"][0]["message"]["content"] == '{"action": "request_demos"}'
    assert body["thought"] == "Nothing here." and body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ipython"
    # the credit goes to the tracker, never to the system: its adapter leaves it out of the turn's budget
    assert "_budget_credit_s" not in body and "_budget_credit_s" not in raw.decode() and tracker.credit_s == 5.0
    lines = raw.decode().splitlines()
    events = [json.loads(line[5:]) for line in lines if line.startswith("data:") and "[DONE]" not in line]
    assert lines[-2].startswith("data: [DONE]") or "data: [DONE]" in raw.decode()
    assert len(events) == 2 and events[0]["object"] == "chat.completion.chunk"
    delta = events[0]["choices"][0]["delta"]
    assert delta["content"] == '{"action": "request_demos"}' and delta["reasoning"] == "Nothing here."
    assert delta["tool_calls"] == [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "ipython", "arguments": "{\"code\": \"1\"}"}}]
    assert events[1]["choices"][0]["finish_reason"] == "tool_calls" and events[1]["usage"]["completion_tokens"] == 7
    assert served_events(body)[0]["choices"][0]["delta"]["role"] == "assistant"
    # the upstream saw only the forwarded call; the two served ones are counted and captured
    assert [r["body"]["messages"][-1]["content"] for r in upstream.requests] == ["forward me"]
    assert forwarded.json()["choices"][0]["message"]["content"] == "OK"
    done = [c for c in seen if c.status == "completed"]
    assert len(done) == 3 and [c.response_headers.get("x-proxy-served") for c in done] == ["local", "local", None]
    assert done[0].tokens["output"] == 7 and done[0].finish_reason == "tool_calls" and done[0].response_body["thought"] == "Nothing here."
    assert done[1].response_body["choices"][0]["message"]["content"] == '{"action": "request_demos"}'
    assert [c[1] for c in asked] == [c.call_id for c in done]
    usage = tracker.usage.as_resources()
    assert usage["llm_requests"] == 3 and usage["llm_output_tokens"] == 7 + 7 + 3


def test_a_long_serve_keeps_the_tracker_touched(upstream, monkeypatch):
    """A call the proxy answers itself can take minutes: its tracker is touched every KEEPALIVE_S
    meanwhile, so that an adapter's idle watchdog, which reads the tracker's last activity, does not take it for a hung
    system: a served call can come near prime-agent's 600-second idle timeout."""
    import time as _time

    import continual_arc_baselines.proxy as proxy_module

    monkeypatch.setattr(proxy_module, "KEEPALIVE_S", 0.05)
    tracker = Tracker()
    touches: list[float] = []

    def serve(parsed, call_id):
        start = tracker.last_activity
        for _ in range(6):
            _time.sleep(0.05)
            touches.append(tracker.last_activity - start)
        return {"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    with LLMProxy(upstream.base_url, tracker=tracker, serve=serve) as proxy:
        assert chat(proxy).json()["choices"][0]["message"]["content"] == "ok"
    assert touches[-1] > 0.15   # the tracker moved on while serve ran, without a byte through the proxy
    assert upstream.requests == []
