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


def test_an_upstream_that_breaks_mid_stream_breaks_the_clients_stream_at_once():
    """An upstream whose stream stops partway and closes its connection has the client's stream end the same way, at
    once, rather than leave the client reading a kept-alive connection until its own read timeout."""
    server = FakeUpstream(broken=True).start()
    try:
        tracker = Tracker()
        with LLMProxy(server.base_url, tracker=tracker) as proxy:
            began = time.monotonic()
            with pytest.raises(httpx.RemoteProtocolError):
                with _stream(proxy, read_s=20) as response:
                    b"".join(response.iter_bytes())
            assert time.monotonic() - began < 5
            assert chat(proxy).status_code == 200   # the proxy serves the next call as before
        assert tracker.usage.failed_requests == 1
    finally:
        server.stop()


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


def test_only_chat_completions_and_reads_of_the_models_reach_the_upstream(upstream):
    """A system's own code can read the proxy's address and key and send anything: a completion, an embedding, a path
    it invents. Each is answered here with 404 and counted as a failed call, its path and the model it named kept, and
    the upstream never hears of it; chat completions and reads of the model list go upstream as before."""
    tracker = Tracker()
    seen: list = []
    tracker.subscribe(seen.append)
    refused = [("POST", "/completions"), ("POST", "/embeddings"), ("POST", "/v1/arc/solve"), ("GET", "/key"),
               ("GET", "/chat/completions"), ("POST", "/models"), ("POST", "/chat/completions/x")]
    with LLMProxy(upstream.base_url, api_key="sk-real", tracker=tracker) as proxy:
        for method, path in refused:
            response = httpx.request(method, proxy.base_url + path, json={"model": "vendor/stronger", "prompt": "hi"},
                                     headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"}, timeout=10)
            assert response.status_code == 404 and response.json()["error"]["type"] == "proxy_refused"
        assert upstream.requests == []
        assert chat(proxy).status_code == 200
        assert httpx.get(proxy.base_url + "/models", timeout=10).status_code == 200
        assert httpx.get(proxy.base_url + "/models/vendor/model/endpoints", timeout=10).status_code == 200
    assert [r["path"] for r in upstream.requests] == ["/v1/chat/completions", "/v1/models",
                                                      "/v1/models/vendor/model/endpoints"]
    ended = [c for c in seen if c.status != "started"]
    assert [(c.status, c.status_code, c.path, c.requested_model) for c in ended[:len(refused)]] == [
        ("failed", 404, "/v1" + path, "vendor/stronger") for _, path in refused]
    assert tracker.usage.requests == len(refused) + 3 and tracker.usage.failed_requests == len(refused)


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


def test_a_call_whose_reply_budget_overruns_the_window_is_sent_again_for_the_room_left():
    """vLLM refuses a prompt and a reply budget that together pass its window, where a hosted provider shortens the
    reply: such a call, streamed or not, is sent once more without its budget, so that the server gives the reply the
    room the window has left. A prompt that passes the window alone still reaches the system refused, for its own
    overflow handling, and any other 400 reaches it as it came."""
    server = FakeUpstream(window=1_000).start()
    try:
        tracker = Tracker()
        seen = []
        tracker.subscribe(seen.append)
        with LLMProxy(server.base_url, tracker=tracker, capture_content=True) as proxy:
            fits = chat(proxy, messages=[{"role": "user", "content": "x" * 600}], max_completion_tokens=400)
            turn = chat(proxy, messages=[{"role": "user", "content": "x" * 700}], max_completion_tokens=400, stream=True)
            full = chat(proxy, messages=[{"role": "user", "content": "x" * 1_001}], max_tokens=400)
        assert (fits.status_code, turn.status_code, full.status_code) == (200, 200, 400)
        assert "maximum context length" in full.json()["error"]["message"]
        asked = [{k: r["body"][k] for k in ("max_tokens", "max_completion_tokens") if k in r["body"]}
                 for r in server.requests]
        assert asked == [{"max_completion_tokens": 400}, {"max_completion_tokens": 400}, {}, {"max_tokens": 400}, {}]
        done = [c for c in seen if c.status != "started"]
        assert [c.status for c in done] == ["completed", "completed", "failed"]
        assert "x-proxy-reply-budget-dropped" not in done[0].response_headers
        assert json.loads(done[1].response_headers["x-proxy-reply-budget-dropped"]) == {"max_completion_tokens": 400}
        assert "max_completion_tokens" not in done[1].request_body   # the request as the upstream answered it
        assert json.loads(done[2].response_headers["x-proxy-reply-budget-dropped"]) == {"max_tokens": 400}
        assert "maximum context length" in done[2].error
    finally:
        server.stop()
    other = {"error": {"message": "tools[0].function.parameters is not a valid schema", "code": 400}}
    server = FakeUpstream(status=400, error=other).start()
    try:
        with LLMProxy(server.base_url) as proxy:
            response = chat(proxy, max_tokens=400)
        assert response.status_code == 400 and response.json() == other and len(server.requests) == 1
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


def _served_reply(cost: float) -> dict:
    return {"id": "chatcmpl-served", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                  "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 7, "cost": cost}}


def _stream(proxy: LLMProxy, read_s: float):
    return httpx.stream("POST", proxy.base_url + "/chat/completions",
                        timeout=httpx.Timeout(10, read=read_s),
                        json={"model": "vendor/model", "messages": [{"role": "user", "content": "hi"}],
                              "stream": True})


def test_a_long_serve_keeps_a_streamed_call_open(upstream, monkeypatch):
    """A streamed call the proxy answers itself gets its stream's headers and a comment every KEEPALIVE_S while serve
    runs, so that a client that drops a call with nothing back for a while (prime-agent's, after about five minutes)
    waits for the reply, which then follows on the same stream."""
    import continual_arc_baselines.proxy as proxy_module

    monkeypatch.setattr(proxy_module, "KEEPALIVE_S", 0.05)
    tracker = Tracker()

    def serve(parsed, call_id):
        time.sleep(0.5)
        return _served_reply(0.25)

    with LLMProxy(upstream.base_url, tracker=tracker, serve=serve) as proxy:
        with _stream(proxy, read_s=0.3) as response:   # a read timeout shorter than the serve
            raw = b"".join(response.iter_bytes()).decode()
    assert raw.startswith(": serving\n\n")
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    assert events[0]["choices"][0]["delta"]["content"] == "ok" and raw.rstrip().endswith("data: [DONE]")
    usage = tracker.usage.as_resources()
    assert usage["llm_failed_requests"] == 0 and usage["llm_cost_usd"] == pytest.approx(0.25)
    assert upstream.requests == []


def test_a_served_reply_whose_system_went_away_is_still_counted(upstream, monkeypatch):
    """The provider charges for a reply the system went away from: it is counted, the call cancelled."""
    import threading

    import continual_arc_baselines.proxy as proxy_module

    monkeypatch.setattr(proxy_module, "KEEPALIVE_S", 0.05)
    tracker = Tracker()
    seen = []
    tracker.subscribe(seen.append)
    gone = threading.Event()

    def serve(parsed, call_id):
        gone.wait(5)
        time.sleep(0.3)   # a few comments meet the closed connection before the reply does
        return _served_reply(0.5)

    since = time.monotonic()
    with LLMProxy(upstream.base_url, tracker=tracker, serve=serve) as proxy:
        with _stream(proxy, read_s=5) as response:
            next(response.iter_bytes())   # the stream opened; then the system leaves
        gone.set()
        assert tracker.wait_ended(since, timeout=5) == {}
    assert seen[-1].status == "cancelled" and seen[-1].cost == Decimal("0.5")
    usage = tracker.usage.as_resources()
    assert usage["llm_failed_requests"] == 1 and usage["llm_cost_usd"] == pytest.approx(0.5)


def test_a_serve_that_gives_up_once_the_stream_is_open_fails_the_call(upstream, monkeypatch):
    """A hook that declines a streamed call after its stream opened cannot have it forwarded: the stream ends on a
    mid-stream error, which a client retries as an upstream's failure, and the upstream never sees the call."""
    import continual_arc_baselines.proxy as proxy_module

    monkeypatch.setattr(proxy_module, "KEEPALIVE_S", 0.05)
    tracker = Tracker()

    def serve(parsed, call_id):
        time.sleep(0.2)
        return None

    with LLMProxy(upstream.base_url, tracker=tracker, serve=serve) as proxy:
        with _stream(proxy, read_s=5) as response:
            raw = b"".join(response.iter_bytes()).decode()
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    assert events == [proxy_module.SERVE_FAILED] and events[0]["choices"][0]["finish_reason"] == "error"
    assert tracker.usage.as_resources()["llm_failed_requests"] == 1 and upstream.requests == []


# -- a confined proxy: an untrusted learner on its submitter's key ------------------------------------------------------


@pytest.fixture
def confined(upstream):
    tracker = Tracker()
    with LLMProxy(upstream.base_url, api_key="sk-run", tracker=tracker, include_usage=True,
                  confined_to={"vendor/named": "medium", "vendor/other": None}) as proxy:
        yield proxy


def post(proxy, body, **kwargs):
    return httpx.post(proxy.base_url + "/chat/completions", json=body, timeout=30, **kwargs)


def plain(model="vendor/named", **extra):
    return {"model": model, "messages": [{"role": "user", "content": "hello"}], **extra}


def test_a_confined_proxy_forwards_the_named_models_alone(confined, upstream):
    assert post(confined, plain()).status_code == 200
    assert post(confined, plain("vendor/other")).status_code == 200
    for model in ("vendor/unnamed", "vendor/named:online", None):
        response = post(confined, plain(model))
        assert response.status_code == 403 and response.json()["error"]["type"] == "proxy_refused"
    assert [r["body"]["model"] for r in upstream.requests] == ["vendor/named", "vendor/other"]
    usage = confined.tracker.usage
    assert usage.refused_requests == 3 and usage.failed_requests == 3 and usage.as_resources()["llm_refused_requests"] == 3


def test_a_call_that_names_no_effort_is_given_its_models(confined, upstream):
    assert post(confined, plain()).status_code == 200
    assert post(confined, plain(reasoning={"effort": "high"})).status_code == 200
    assert post(confined, plain("vendor/other")).status_code == 200
    bodies = [r["body"] for r in upstream.requests]
    assert bodies[0]["reasoning"] == {"effort": "medium"} and bodies[1]["reasoning"] == {"effort": "high"}
    assert "reasoning" not in bodies[2]


def test_a_confined_proxy_sends_a_plain_chat_completion_and_its_own_routing(confined, upstream):
    extra = {"models": ["vendor/unnamed"], "route": "fallback", "provider": {"order": ["mine"]}, "plugins": [{"id": "web"}],
             "user": "leak", "metadata": {"a": "leak"}, "session_id": "leak", "trace": {"t": "leak"}, "store": True,
             "web_search_options": {}, "mcp_servers": [{"url": "leak"}], "transforms": ["leak"], "temperature": 0.5,
             "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}], "max_tokens": 4}
    headers = {"X-Title": "leak", "HTTP-Referer": "leak", "X-OpenRouter-Title": "leak", "X-Evil": "leak", "Cookie": "leak",
               "User-Agent": "leak", "Anthropic-Beta": "leak", "Authorization": "Bearer leak"}
    assert post(confined, plain(**extra), headers=headers).status_code == 200
    seen = upstream.requests[0]
    assert "leak" not in json.dumps(seen["body"])
    assert seen["body"]["provider"] == {"zdr": True, "data_collection": "deny"}
    assert seen["body"]["temperature"] == 0.5 and seen["body"]["max_tokens"] == 4 and seen["body"]["tools"]
    assert seen["authorization"] == "Bearer sk-run"
    assert "leak" not in json.dumps(seen.get("headers", {}))


@pytest.mark.parametrize("message", [
    [{"type": "image_url", "image_url": {"url": "https://example.com/leak.png"}}],
    [{"type": "image_url", "image_url": "https://example.com/leak.png"}],
    [{"type": "file", "file": {"filename": "a.pdf", "file_data": "https://example.com/leak.pdf"}}],
    [{"type": "input_audio", "input_audio": {"data": "leak", "format": "wav"}}],
    [{"type": "video_url", "video_url": {"url": "https://example.com/leak.mp4"}}],
    {"not": "a part"},
])
def test_a_confined_proxy_refuses_what_a_provider_would_fetch(confined, upstream, message):
    response = post(confined, {"model": "vendor/named", "messages": [{"role": "user", "content": message}]})
    assert response.status_code == 400 and not upstream.requests


def test_a_confined_proxy_carries_text_and_an_image_in_the_request_itself(confined, upstream):
    content = [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert post(confined, {"model": "vendor/named", "messages": [{"role": "user", "content": content}]}).status_code == 200
    assert upstream.requests[0]["body"]["messages"][0]["content"] == content


@pytest.mark.parametrize("tools", [
    [{"type": "openrouter:web_search"}], [{"type": "web_search_20250305", "name": "web_search"}],
    [{"type": "mcp", "server_url": "https://example.com"}], "not a list", ["not a tool"],
])
def test_a_confined_proxy_refuses_a_tool_the_provider_would_run(confined, upstream, tools):
    assert post(confined, plain(tools=tools)).status_code == 400 and not upstream.requests


@pytest.mark.parametrize("body,content_type", [
    (json.dumps(plain("vendor/unnamed")), "text/plain"), (json.dumps(plain()), "text/plain"), ("not json", "application/json"),
    ("[1, 2]", "application/json"), ("", "application/json"),
])
def test_a_confined_proxy_reads_every_body_itself(confined, upstream, body, content_type):
    """A body the proxy does not read would go up past everything it enforces."""
    response = httpx.post(confined.base_url + "/chat/completions", content=body, timeout=30,
                          headers={"Content-Type": content_type})
    if content_type == "text/plain" and "unnamed" not in body:
        assert response.status_code == 200 and upstream.requests[0]["body"]["model"] == "vendor/named"
    elif "unnamed" in body:
        assert response.status_code == 403 and not upstream.requests
    else:
        assert response.status_code == 400 and not upstream.requests


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/models/../key"), ("GET", "/v1/models/../../v1/keys"), ("GET", "/v1/models/./x"), ("POST", "/v1/chat/completions/../key"),
    ("GET", "/v1/key"), ("GET", "/v1/credits"), ("GET", "/v1/generation?id=1"), ("POST", "/v1/completions"), ("POST", "/v1/embeddings"),
    ("GET", "/v1/models/%2e%2e/key"), ("GET", "http://example.com/v1/models"),
])
def test_only_the_endpoints_named_plainly_go_upstream(upstream, method, path):
    """A ``..`` in a path reaches any endpoint: httpx resolves it before it sends."""
    with LLMProxy(upstream.base_url, api_key="sk-run", tracker=Tracker()) as proxy:
        import http.client
        connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=30)
        connection.request(method, path, b"{}" if method == "POST" else None, {"Content-Type": "application/json"})
        assert connection.getresponse().status == 404
        connection.close()
    assert not upstream.requests


def test_a_confined_proxy_sends_no_query_string(confined, upstream):
    response = httpx.post(confined.base_url + "/chat/completions?leak=1", json=plain(), timeout=30)
    assert response.status_code == 200 and upstream.requests[0]["path"].endswith("/chat/completions")
    assert httpx.get(confined.base_url + "/models?leak=1", timeout=30).status_code == 200
    assert "leak" not in upstream.requests[-1]["path"]


def test_a_body_past_the_bound_is_refused_unread(confined, upstream):
    from continual_arc_baselines.proxy import MAX_BODY

    response = httpx.post(confined.base_url + "/chat/completions", content=b"x" * (MAX_BODY + 1), timeout=60,
                          headers={"Content-Type": "application/json"})
    assert response.status_code == 413 and not upstream.requests


def test_a_call_through_the_keys_own_provider_account_stops_those_after_it(confined, upstream):
    """OpenRouter says so on every reply (is_byok): the submitter's own provider account would show the prompts."""
    upstream.byok = True
    assert post(confined, plain()).status_code == 200
    response = post(confined, plain())
    assert response.status_code == 403 and "own provider account" in response.json()["error"]["message"]
    assert len(upstream.requests) == 1 and confined.tracker.usage.byok_requests == 1


def test_a_confined_proxy_paces_what_it_answers(confined, monkeypatch):
    from continual_arc_baselines import proxy as module

    monkeypatch.setattr(module, "CONFINED_RATE", 20.0)
    started = time.monotonic()
    for _ in range(11):
        assert post(confined, plain("vendor/unnamed")).status_code == 403
    assert time.monotonic() - started >= 0.45


def test_a_flood_of_free_calls_is_counted_and_journaled_only_so_far(upstream, tmp_path, monkeypatch):
    from continual_arc_baselines import proxy as module

    monkeypatch.setattr(module, "CONFINED_RATE", 1000.0)
    monkeypatch.setattr(module, "CONFINED_FREE_JOURNAL", 3)
    journaled = []
    tracker = Tracker()
    tracker.subscribe(lambda call: journaled.append(call.status))
    with LLMProxy(upstream.base_url, api_key="sk-run", tracker=tracker, confined_to={"vendor/named": None}) as proxy:
        for _ in range(10):
            assert post(proxy, plain("vendor/unnamed")).status_code == 403
        assert post(proxy, plain()).status_code == 200  # a call that goes up is always journaled
    assert tracker.usage.refused_requests == 10 and tracker.usage.requests == 11
    assert journaled.count("started") == 3 + 1 and journaled.count("completed") == 1
