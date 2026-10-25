"""An expert that plays the student's call right after each failure, inside the student's own game (DAgger's
takeover): the play's proxies (loop.py play ``--expert``) send every chat completion through :meth:`Expert.serve`,
which answers the calls whose newest messages carry the environment's failure verdict and leaves every other call to
the student's server.

The trigger is the environment's own verdict that the last action failed, as the harness shows it to the agent: a
submission judged INCORRECT (a wrong grid, or one that is no valid ARC grid; protocol.render_feedback) or a demo
request refused at the cap. No other rule: no grace call, no cap on takeovers, no length. One takeover is one model
call; the student plays the next call, and a later failure brings the expert back.

The expert sees what the student sees: the call's messages and tools as the server would get them, every earlier
reply's reasoning left out (the student's private text, not the game). Its reply goes to the harness as the student's
would, its visible text and tool calls only, so the game and the student's later calls go on from its move. The served
body keeps the expert's reasoning, model, seconds and usage under ``expert``, which the capture saves with the call
(dagger.py builds the training sets from it). The expert's seconds are credited back to the action's time budget
(``_budget_credit_s``): the student's clock stops while the expert thinks. A call the expert does not answer (an
error, a timeout, a refusal, an empty reply) goes to the student's server as it is, and is logged.

The expert is reached through an OpenAI-compatible endpoint (``base_url``): the relay (expert_relay.py), which holds
the key and pins every request to the one expert model and its hosts, on the same machine or behind an SSH tunnel.

With ``every``, the expert plays every call of the game (prime-agent's own calls too), so the play is the expert's own.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import render

# the environment's failure verdicts as the harness words them (continual_arc_baselines.systems.protocol)
FAILURES = ("Your submission was INCORRECT", "Demo request REFUSED")
# a verdict that ends the instance: no call after it can change the game
OVER = "The attempt cap is reached"
# prime-agent's own calls, not the agent's: its /refine pass (a system prompt of its own), its compaction summarizer
# (render.COMPACTION) and its trigger and conversation reviews (analysis/trend.py's internal)
REFINE = "You are Prime Agent's /refine"
INTERNAL_OPENINGS = (render.COMPACTION, "<trigger>")
THINKING_KEYS = ("reasoning_content", "reasoning", "thinking")
# the expert's reply: a reasoning model's budget covers its reasoning too (GLM-5.3 at medium effort reasoned 15,000
# tokens on one moment)
MAX_TOKENS = 32_768
# under prime-agent's idle timeout (600 s; the proxy's keepalive covers a long serve) with room for one retry
TIMEOUT_S = 420.0
RETRIES = 1


def agent_call(messages: list[dict[str, Any]]) -> bool:
    """Whether a call is the agent's own: neither prime-agent's /refine pass nor one of its internal calls."""
    if not messages:
        return False
    first, last = messages[0], messages[-1]
    if first.get("role") == "system" and render.text_of(first.get("content")).startswith(REFINE):
        return False
    return not render.text_of(last.get("content")).lstrip().startswith(INTERNAL_OPENINGS)


def after_failure(messages: list[dict[str, Any]]) -> bool:
    """Whether the call's newest messages (those after the agent's last reply) carry a failure verdict that leaves the
    instance open."""
    last = max((j for j, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    texts = [render.text_of(m.get("content")) for m in messages[last + 1:] if m.get("role") != "assistant"]
    return any(f in t for t in texts for f in FAILURES) and not any(OVER in t for t in texts)


def triggers(messages: list[dict[str, Any]]) -> bool:
    """Whether the expert plays this call: an agent call right after a failure."""
    return agent_call(messages) and after_failure(messages)


def expert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The call's messages as the expert gets them: copies, every earlier reply's reasoning left out."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            for k in THINKING_KEYS:
                m.pop(k, None)
            if m.get("content") is None:
                m["content"] = ""
        out.append(m)
    return out


class Expert:
    """The expert's side of a play (the module's docstring): ``serve`` is LLMProxy's hook. ``log`` gets one JSON line per
    triggered call (served, or why not)."""

    def __init__(self, base_url: str, model: str, *, providers: list[str] | None = None, effort: str | None = None,
                 log: Path | None = None, timeout_s: float = TIMEOUT_S, every: bool = False,
                 reasoning: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> None:
        import httpx
        self.base, self.model, self.effort, self.every = base_url.rstrip("/"), model, effort, every
        # the request's whole reasoning setting (OpenRouter's, e.g. {"max_tokens": 4000}); else {"effort": effort}
        self.reasoning = reasoning
        # more request fields, passed through to the host (e.g. {"chat_template_kwargs": {"enable_thinking": True}})
        self.extra = extra or {}
        self.routing = {"order": providers, "allow_fallbacks": False} if providers else None
        self.log_path = log
        self._lock = threading.Lock()
        # no cap on calls at once (httpx's default is 100, past which a call waits 30 s for a free one and then fails):
        # the caller's threads bound them (labelling runs with 320 workers)
        self._client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=60.0, pool=30.0),
                                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=20))
        self.counts = {"calls": 0, "triggered": 0, "served": 0, "failed": 0, "refused": 0, "empty": 0}

    def _log(self, record: dict[str, Any]) -> None:
        with self._lock:
            for key in ("triggered", record.get("outcome")):
                if key in self.counts:
                    self.counts[key] += 1
            if self.log_path is not None:
                with self.log_path.open("a") as fp:
                    fp.write(json.dumps({"t": round(time.time(), 1), **record}) + "\n")

    def ask(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """The expert's answer to the call (the endpoint's JSON), retried once on an error."""
        body: dict[str, Any] = {"model": self.model, "messages": expert_messages(parsed.get("messages") or []),
                                "max_tokens": MAX_TOKENS, "stream": False}
        if parsed.get("tools"):
            body["tools"] = parsed["tools"]
        if self.reasoning:
            body["reasoning"] = self.reasoning
        elif self.effort:
            body["reasoning"] = {"effort": self.effort}
        if self.routing:
            body["provider"] = self.routing
        body.update(self.extra)
        error: Exception | None = None
        for attempt in range(RETRIES + 1):
            try:
                resp = self._client.post(self.base + "/chat/completions", json=body)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001  (retried once, then the call goes to the student)
                error = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"the expert failed: {type(error).__name__}: {error}")

    def serve(self, parsed: dict[str, Any], call_id: str) -> dict[str, Any] | None:
        """LLMProxy's hook: the expert's reply to a call right after a failure (to every call, with ``every``), as the
        harness gets a reply; None (the student's server answers) for every other call and for a call the expert does
        not answer."""
        with self._lock:
            self.counts["calls"] += 1
        messages = parsed.get("messages") or []
        if not (self.every or triggers(messages)):
            return None
        t0 = time.time()
        record: dict[str, Any] = {"call_id": call_id, "messages": len(messages)}
        try:
            answer = self.ask(parsed)
        except Exception as e:  # noqa: BLE001
            self._log({**record, "outcome": "failed", "error": str(e)[:500], "seconds": round(time.time() - t0, 1)})
            return None
        choice = (answer.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content, tool_calls = message.get("content") or "", message.get("tool_calls") or []
        reasoning = next((message[k] for k in THINKING_KEYS if isinstance(message.get(k), str)), "")
        seconds = round(time.time() - t0, 1)
        record.update({"seconds": seconds, "finish_reason": choice.get("finish_reason"), "usage": answer.get("usage"),
                       "tool_calls": len(tool_calls), "content_chars": len(content), "reasoning_chars": len(reasoning)})
        if message.get("refusal"):
            self._log({**record, "outcome": "refused", "refusal": str(message["refusal"])[:500]})
            return None
        if not content.strip() and not tool_calls:
            self._log({**record, "outcome": "empty"})
            return None
        self._log({**record, "outcome": "served"})
        reply: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            reply["tool_calls"] = tool_calls
        usage = answer.get("usage") or {}
        return {"id": f"expert-{call_id}", "object": "chat.completion", "created": int(t0),
                "model": str(parsed.get("model") or ""),
                "choices": [{"index": 0, "message": reply, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                "expert": {"model": self.model, "provider": answer.get("provider"), "reasoning": reasoning,
                           "seconds": seconds, "usage": usage, "finish_reason": choice.get("finish_reason")},
                "_budget_credit_s": seconds}

    def close(self) -> None:
        self._client.close()
        logging.getLogger(__name__).info("expert: %s", json.dumps(self.counts))
