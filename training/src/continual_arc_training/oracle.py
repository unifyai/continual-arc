"""The oracle that labels DAgger's and OPSD's rounds: GLM-5.3 Flash at medium effort, reading each call the student
made exactly as the student read it, with its working rules at the end of the last message, and replying with its own
move there.

- **Its endpoint** (:func:`relay`): a proxy on the node (``python -m continual_arc_training.oracle relay``, which
  realtime.sh starts) that forwards to OpenRouter with the key it alone holds, every request pinned to the one model
  and its hosts, so the loops never hold the key and reach nothing but that model.
- **Its client** (:class:`Oracle`): a call's messages and tools as the student's server got them, every earlier
  reply's reasoning left out (the student's private text, not the game), its reasoning switched on.
- **Its working rules** (:func:`oracle_rules`): c1, the playing rules (``rules/c1.md``), and the program library's
  procedure (the baselines' ``PROGRAM_LIBRARY_RULES``, row HPL's text), which the student is never told, so the
  procedure reaches the student's weights through the labels alone; or c1 alone.
- **An ask's checks** (:func:`ask_label`): a reply that names the rules, has no reasoning or makes no move is asked
  again, each time with a note saying what was wrong.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

OPENROUTER = "https://openrouter.ai/api/v1"
KEY = "OPENROUTER_API_KEY"
RULES = Path(__file__).resolve().parent / "rules" / "c1.md"
THINKING_KEYS = ("reasoning_content", "reasoning", "thinking")
# a reasoning model's budget covers its reasoning too (GLM-5.3 at medium effort reasoned 15,000 tokens on one moment)
MAX_TOKENS = 32_768
# under prime-agent's idle timeout (600 s) with room for one retry
TIMEOUT_S = 420.0
RETRIES = 1
# what the oracle is told (realtime.sh's ORACLE_RULES)
ORACLE_RULES = ("library", "c1")

# The rules go at the end of the last message, and close by asking for reasoning in the oracle's own words with the
# rules unspoken (the student never sees them): on one round's 380 moments this labelled 89% against 79% for the rules
# alone, and finished in 153 s against 547. In the system prompt the oracle mostly stopped reasoning (45% labelled).
GUIDE_HEAD = "\n\n[Working rules for this harness:]\n"
GUIDE_TAIL = ("\n\nBefore every move, think it through in your own words: what the grids and outputs so far show, what you "
              "will do next and why. Reason about the puzzle itself: these habits are simply how you work, so never name, "
              "number or quote them, never call them rules or guidelines, and never mention a student."
              "\n[End of working rules. Follow them in every reply, and never mention them.]")
# a reply that refers to the working rules is asked again; a puzzle's own "rule" is not one
MENTIONS_RULES = re.compile(r"(?i)\b(working[- ]rules?|rules for this harness|harness rules|rule\s*#?[1-7]\b|"
                            r"(?:the|these|my|our) rules (?:say|says|tell|state|require|above|given)|per the rules|"
                            r"following the rules|as the rules|guidelines?|(?:the|this) student)")
# a reply that refers to the note an earlier ask carried is asked again like one that names the rules
NOTE_LEAK = re.compile(r"(?i)\b(this note|the note|as (?:requested|instructed)|this time|previous (?:reply|answer|attempt)|"
                       r"answered (?:this|it) (?:\d+|once|twice))")
# A label "thinks" when its reasoning and the prose of its visible text together hold at least this many characters,
# all of which becomes the student's reasoning. GLM-5.3 Flash sometimes answers with no reasoning at all, and a student
# trained on such labels stopped thinking; on an obvious move it reasons in 9 or 10 tokens and explains the move in its
# visible reply instead.
THINK_MIN_CHARS = 50
# how many more times a reply that names the rules or the note is asked again
RULE_RETRIES = 3
# where a reply's action starts: its last JSON action object, with the code fence that opens it if there is one
ACTION_START = re.compile(r'(?:```(?:json)?\s*)?\{\s*"action"\s*:')


def oracle_rules(c1: str, rules: str) -> str:
    """The working rules the oracle labels every call under (the module docstring): ``library``, c1 and the program
    library's procedure with the library in ``programs``; or ``c1``, c1 alone, so that the procedure is the one
    difference between the two oracles."""
    if rules == "c1":
        return c1.strip()
    from continual_arc_baselines.systems.protocol import PROGRAM_LIBRARY_RULES
    return c1.strip() + "\n\n" + PROGRAM_LIBRARY_RULES.format(programs="programs")


def split_reply(content: str | None) -> tuple[str, str]:
    """A reply's visible text as its prose and its action: the action from the last JSON action object (and the fence
    that opens it) to the end, the prose everything before; a reply with no action, the text beside a tool call, is all
    prose."""
    content = content or ""
    starts = [m.start() for m in ACTION_START.finditer(content)]
    if not starts:
        return content.strip(), ""
    return content[:starts[-1]].strip(), content[starts[-1]:].strip()


def makes_move(message: dict[str, Any]) -> bool:
    """Whether a reply does anything in the game: it calls a tool or gives an action. A reply that only says what it
    will do ("Let me load the grids.") makes none, and as a target it teaches the student to spend a turn on nothing:
    the harness answers such a reply with "no valid action" and, after a few, gives the instance up."""
    return bool(message.get("tool_calls")) or bool(split_reply(message.get("content"))[1])


def with_rules(request: dict[str, Any], rules: str) -> dict[str, Any]:
    """A copy of the call with the oracle's working rules at the end of its last message (the student's call itself is
    left as it was): the rules shape the oracle's moves, and its reply is still a reply to the student's conversation."""
    msgs = [dict(m) for m in request.get("messages") or []]
    block = GUIDE_HEAD + rules.strip() + GUIDE_TAIL
    c = msgs[-1].get("content")
    msgs[-1]["content"] = (c or "") + block if isinstance(c, str) or c is None else [*c, {"type": "text", "text": block}]
    return {**request, "messages": msgs}


def with_note(request: dict[str, Any], note: str) -> dict[str, Any]:
    """A copy of the call with ``note`` at the end of its last message, after the working rules."""
    msgs = [dict(m) for m in request.get("messages") or []]
    c = msgs[-1].get("content")
    block = "\n\n[Note: " + note + "]"
    msgs[-1]["content"] = (c or "") + block if isinstance(c, str) or c is None else [*c, {"type": "text", "text": block}]
    return {**request, "messages": msgs}


def oracle_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The call's messages as the oracle gets them: copies, every earlier reply's reasoning left out."""
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


class Oracle:
    """The oracle's client: a chat completion through ``base_url`` (the relay) for ``model`` at reasoning ``effort``,
    its thinking on as hard as the hosts allow (the request's reasoning enabled and the chat template's
    ``enable_thinking``), retried once on an error."""

    def __init__(self, base_url: str, model: str, effort: str) -> None:
        import httpx

        self.base, self.model, self.effort = base_url.rstrip("/"), model, effort
        # no cap on calls at once (httpx's default is 100, past which a call waits 30 s for a free one and fails): the
        # labeller's threads bound them
        self._client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=TIMEOUT_S, write=60.0, pool=30.0),
                                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=20))

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        """The oracle's answer to the call (the endpoint's JSON)."""
        body: dict[str, Any] = {"model": self.model, "messages": oracle_messages(request.get("messages") or []),
                                "max_tokens": MAX_TOKENS, "stream": False}
        if request.get("tools"):
            body["tools"] = request["tools"]
        body["reasoning"] = {"enabled": True, "effort": self.effort}
        body["chat_template_kwargs"] = {"enable_thinking": True}
        error: Exception | None = None
        for attempt in range(RETRIES + 1):
            try:
                resp = self._client.post(self.base + "/chat/completions", json=body)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001  (retried once, then the point stays unlabelled)
                error = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"the oracle failed: {type(error).__name__}: {error}")


def ask_label(ask: Callable[[dict[str, Any]], dict[str, Any]], request: dict[str, Any], rules: str,
              need_thinking: int) -> tuple[str, dict[str, Any], str, dict[str, Any], int, float]:
    """One point's label from ``ask`` (the oracle's call) for ``request`` (the call with the working ``rules`` in it),
    asked again when the reply names the rules or a note (up to :data:`RULE_RETRIES` more times) and, with
    ``need_thinking`` above 0, when it has no reasoning or makes no move (up to that many more times). A reply that makes
    no move, empty or prose that runs no code and gives no action, would teach the student to spend a turn on nothing.
    Each ask after the first carries a note saying what was wrong and how often, asking for a reply as if for the first
    time that never mentions it, since the reply becomes the student's target: told what was wrong, the oracle labelled
    96% of one round's 380 moments against 89% when merely asked again. Returns the outcome (labelled; empty; no move;
    failed; mentions the rules; no thinking), the reply, its reasoning, the last usage, the number of asks and the
    dollars spent on all of them."""
    rules_left, think_left, n, spent, unreasoned = RULE_RETRIES, need_thinking, 0, 0.0, 0
    asked = request
    while True:
        n += 1
        try:
            answer = ask(asked)
        except Exception:  # noqa: BLE001  (a point the oracle does not answer stays unlabelled)
            return "failed", {}, "", {}, n, spent
        m = (answer.get("choices") or [{}])[0].get("message") or {}
        reasoning = next((m[k] for k in ("reasoning", "reasoning_content") if isinstance(m.get(k), str)), "")
        usage = answer.get("usage") or {}
        spent += usage.get("cost") or 0.0
        if not m.get("tool_calls") and not split_reply(m.get("content"))[1]:
            empty = not (m.get("content") or "").strip()
            if need_thinking and think_left:
                think_left -= 1
                said = "was empty" if empty else "made no move: it ran no code and gave no action"
                asked = with_note(request, f"your last reply {said}. Think it through in your own words, then reply "
                                           "with your move as if for the first time, never mentioning this note.")
                continue
            return "empty" if empty else "no move", m, reasoning, usage, n, spent
        text = (reasoning or "") + "\n" + (m.get("content") or "")
        named = MENTIONS_RULES.search(text) or (NOTE_LEAK.search(text) if n > 1 else None)
        if named:
            if rules_left:
                rules_left -= 1
                asked = with_note(request, f'your last answer named your working rules or this note ("{named.group(0)}"). '
                                           "Answer again, reasoning about the puzzle itself in your own words: never "
                                           "name, number or quote the rules, and reply as if for the first time, never "
                                           "mentioning this note.")
                continue
            return "mentions the rules", m, reasoning, usage, n, spent
        if need_thinking and len(reasoning.strip()) + len(split_reply(m.get("content"))[0]) < THINK_MIN_CHARS:
            if think_left:
                think_left -= 1
                unreasoned += 1
                times = "once" if unreasoned == 1 else f"{unreasoned} times"
                asked = with_note(request, f"you have answered this message {times} without any reasoning. Think it "
                                           "through in your own words first: what the grids and outputs so far show, "
                                           "what you will do next and why. Then reply as if for the first time, never "
                                           "mentioning this note.")
                continue
            return "no thinking", m, reasoning, usage, n, spent
        return "labelled", m, reasoning, usage, n, spent


# -- the relay ---------------------------------------------------------------------------------------------------

def read_key() -> str:
    """The OpenRouter key from the environment, or from the baselines' .env file (``BASELINES/.env``, BASELINES the
    baselines folder of this repository by default)."""
    if os.environ.get(KEY):
        return os.environ[KEY]
    env_file = Path(os.environ.get("BASELINES") or Path(__file__).resolve().parents[3] / "baselines") / ".env"
    for line in (env_file.read_text().splitlines() if env_file.is_file() else []):
        name, _, value = line.partition("=")
        if name.strip() == KEY:
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"no {KEY} in the environment or {env_file}")


def relay(port: int, model: str, providers: list[str], max_connections: int) -> None:
    """The oracle's endpoint on 127.0.0.1:``port`` (the module docstring), until SIGTERM."""
    from continual_arc_baselines.proxy import LLMProxy

    pinned = {"model": model, "provider": {"order": providers, "allow_fallbacks": False}}
    proxy = LLMProxy(OPENROUTER, api_key=read_key(), port=port, body_overrides=pinned, declared_context_length=None,
                     timeout=600.0, max_connections=max_connections).start()
    print(f"oracle relay on {proxy.base_url}: {model} via {','.join(providers)}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proxy.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="the oracle's relay to OpenRouter")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relay")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--providers", required=True, help="the hosts, in order, comma-separated: no other is used")
    r.add_argument("--max-connections", type=int, default=1024, dest="max_connections",
                   help="the oracle's calls open at once (the labeller runs hundreds in parallel)")
    a = p.parse_args()
    relay(a.port, a.model, a.providers.split(","), a.max_connections)


if __name__ == "__main__":
    main()
