"""The plays every row and experiment in this folder shares, and their paired comparisons: schedules, prime-agent's
play of a served model with every call captured, the harness settings (:data:`HARNESS`, :data:`HARNESS_BUDGETS`), the
replay of recorded games (:func:`replay_proxy`), Qwen3.5's adapter-serving fix (:func:`served_adapter`), and the
paired comparison of two plays.

It began as the driver of context distillation round after round, the experiments before the paper's rows: a round's
student plays fresh stream instances (training data), then the validation split (100 ARC-1 evaluation tasks that
neither the stream nor the epilogue of arc1-small or arc1-medium at the canonical seeds uses); the round's student is
kept unless it solves clearly fewer validation instances than the best student so far (``decide``), since
distillation copies its teacher and a round can make the student worse; the held-out test (arc1-medium's epilogue) is
played once, at the end. Those subcommands stay, for reading the earlier experiments and for the validation plays.

Subcommands (on the GPU node, from prime-rl's checkout:
``uv run --no-sync python ~/continual-arc/training/loop.py ...``):
  schedule  write a schedule: ``stream --round R`` (arc1-small's 25 stream tasks x 4 fresh instances, users 4R..4R+3
            of the 100-user collective configuration),
            ``stream --user U`` (that one user's 25 instances, one per task: a cycle's games, users 4 and up),
            ``dev --round R`` / ``dev --user U`` (the same shape over the development set instead: DEV_TASKS training
            tasks off arc1-small's stream and every epilogue, so a recipe tried on it is never tuned on the paper's
            data),
            ``validation`` (the validation split, one instance per task, the same every round),
            ``heldout`` (arc1-medium's 100 ARC-1 evaluation tasks, one instance each: the test) or
            ``tasks --file F`` (a task set's own Re-ARC tasks, ``games`` fresh instances each and ``held_games`` of
            each held-out task)
  play      run the served model (``--model``: the base or an adapter) through prime-agent over a schedule, one fresh
            user per instance, under the finish protocol, with
            the team rows' harness settings (``--harness block`` for the trained rows' option: the 40-call cap, with
            the per-action and follow-up seconds raised to 1,740 s, just under the 1,800 s hard ceiling); every model
            call saved token-exact as each instance ends; with
            ``--context-file``, that text shown at every model call: appended to the last message, or with
            ``--context-placement thought`` as the opening of the model's own reasoning (the server's chat template
            must take ``reasoning_prefill``: qwen35_prefill.jinja, as serve.sh serves Qwen3.5); with ``--expert``,
            the call right after each failure played by that model instead (expert.py); with ``--oracle-card``, an
            oracle's hint for each move after c1 and the card in that opening (oracle_hint.py); with
            ``--stop-after N``, each game cut after the student's first N calls; with
            ``--replay PLAN``, each user's recorded game replayed up to a branch call, a planned reply served there,
            and the model playing on (the branch-and-compare test)
  compare   pair two runs' records by instance: mean cost difference with a paired bootstrap interval, solves with an
            exact McNemar test, behaviour counts, and whether the new run is clearly worse (``not_worse``)
  decide    the keep rule after a round: compare the best student's validation play with the round student's, and
            say whether the round's student is kept and whether it is the new best
  vllm-adapter  the directory vLLM should load a saved adapter from, written when the model needs one (Qwen3.5)
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any

# The loop's model, its chat-template settings at every call (the model card's thinking mode), and so the default of
# play and train.py; K2 Horizon runs pass --model IFM/K2-Horizon-7B and that model's own template settings
MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_KWARGS = {"enable_thinking": True}
STREAM_CONFIG, HELDOUT_CONFIG = "arc1-small", "arc1-medium"
DEV_USERS = 100   # the users the dev loop's schedule is served to (users 4R..4R+3 play round R), as it ran
INSTANCES_PER_TASK = 4
# The validation split: this many ARC-1 evaluation tasks outside every epilogue that the loop or the paper plays, drawn
# under its own label (174 such tasks exist at the canonical seeds)
VALIDATION_TASKS = 100
VALIDATION_LABEL = "loop-validation"
# The context as the model sees it, at every call during play and in the teacher's prompt during training: the same
# words in both places, and nothing that reads as written after the work was done. As a message block (placement
# "message"): wrapped in these, at the end of the last message. As a thought (placement "thought"): the text alone,
# then ask_sd.calls.THOUGHT_SEP, as the opening of the model's own reasoning (right after <think>)
CONTEXT_HEAD = "\n\n[Notes on working in this harness:]\n"
CONTEXT_TAIL = "\n[End of notes.]"
PLACEMENTS = ("message", "thought")
# The team rows' harness settings, with the finish protocol in place of their
# storage turn, compacting at 48k tokens keeping 8k where they compact at 32k (on the compaction probe untrained
# Qwen3.5-9B solved 16 of 48 games at 48k against 10 at 32k, for less a game); the action budget is
# the harness default (900 s, 40 model calls). An empty reply is re-prompted in the
# same conversation, as the paper's grid does (baselines' empty_reply): a restart loses the conversation, so a model
# whose reply ran into the token cap on reasoning alone forgot the instance's demos and attempts and, restarted twice,
# gave the action up (GLM-5.3-flash on row h: one instance of 76 minutes). The trained runs took it mid-run, each run's
# first round under it in docs/how-a-run-works.md; their earlier rounds restarted.
HARNESS = {"compact_at": 49_152, "compact_keep": 8_000, "end": "finish", "empty_reply": "reprompt"}


def harness_of(compact_at: int, compact_keep: int) -> dict:
    """:data:`HARNESS` with a play's own compaction point and kept tail (``play --compact-at/--compact-keep``); 0 leaves
    prime-agent's default (a 16,384-token reserve under the context window; a 20,000-token tail)."""
    return {**HARNESS, "compact_at": compact_at or None, "compact_keep": compact_keep or None}
BUDGET_STOP = "exceeded the time or model-call budget"
# play --harness: PrimeAgentLearner kwargs beyond HARNESS. "default" leaves prime-agent's own action budget (900 s or
# 40 model calls, a follow-up of 300 s or 6 calls). "block" is the trained rows' harness: the
# call cap (40) is meant to be the only thing that routinely stops a turn, so the per-action and follow-up seconds are
# raised close to timeout_s's 1,800 s hard ceiling (left at its default), not equal to it: at 1,800 exactly, the hard
# ceiling fires first (PrimeAgentLearner._turn checks it before the action budget) and aborts the whole turn with a
# restart (SystemUnavailable, eventually a forced give-up) instead of the budget's graceful "decide now" with 60 s to
# collect an answer. 1,740 s leaves that 60 s of room; the follow-up's 6 calls are kept
HARNESS_BUDGETS = {
    "default": {},
    "block": {"max_model_calls": 40, "action_budget_s": 1740.0, "reprompt_budget_s": 1740.0,
              "reprompt_max_model_calls": 6},
}
# vLLM serves the Qwen3.5 architecture (Qwen3.5-9B, Qwen3.8-27B, ...) as its multimodal class, which maps the
# checkpoint's names (model.language_model.layers...) onto its modules; PEFT on the text-only class the trainer loads
# writes model.layers..., which match no module, and vLLM then serves the adapter as the base model without a word
# (a test LoRA served raw read exactly as the base)
PEFT_LAYERS, SERVED_LAYERS = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
MULTIMODAL_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration",)
# A run is clearly worse than the run it is paired with when it solves fewer instances at this one-sided exact McNemar
# p: the keep rule's guard against large drops (K2 + C1: 13 against 28), not a selector among near-equal students
WORSE_P = 0.1


def context_block(context: str) -> str:
    """The context wrapped as every call and every teacher prompt carries it."""
    return CONTEXT_HEAD + context.strip() + CONTEXT_TAIL


def thought_text(context: str) -> str:
    """The context as the opening of the model's reasoning: what play sends as ``reasoning_prefill`` and what the
    teacher reads in training (``ask_sd.calls.thought_ids`` tokenises the same text)."""
    from ask_sd.calls import THOUGHT_SEP
    return context.strip() + THOUGHT_SEP


def inject_thought(parsed: dict, context: str) -> dict:
    """A chat request whose reply opens with the context as the model's own reasoning: the chat template's
    ``reasoning_prefill``, placed right after the generation prompt's ``<think>`` (qwen35_prefill.jinja). The
    template settings are copied, never shared with other requests."""
    parsed["chat_template_kwargs"] = {**(parsed.get("chat_template_kwargs") or {}), "reasoning_prefill": thought_text(context)}
    return parsed


def inject(parsed: dict, block: str) -> dict:
    """A chat request with ``block`` appended to its last message (ask_sd's ``with_note``: every earlier message
    left exactly as the model saw it)."""
    from ask_sd.calls import with_note
    if parsed.get("messages"):
        parsed["messages"] = with_note(parsed["messages"], block)
    return parsed


def seeded_proxy(base: type) -> type:
    """``base`` with a random seed of its own on every chat request that has none: the loop's vLLM serves unseeded
    requests from one random stream, so two plays of the same games would otherwise start alike."""
    class SeededProxy(base):
        def prepare(self, headers, body, path):
            out, body, parsed = super().prepare(headers, body, path)
            if parsed is not None and path.rstrip("/").endswith("/chat/completions") and "seed" not in parsed:
                parsed["seed"] = random.getrandbits(31)
                body = json.dumps(parsed).encode()
            return out, body, parsed
    return SeededProxy


# What the stop hook answers once a game has had its calls: the finish protocol refuses finish while the instance can
# still be played, and after a few refused replies the harness gives up, so the game ends and is saved as any other
STOP_REPLY = '{"action": "finish"}'


def stop_proxy(base: type, n: int) -> type:
    """``base`` that lets the student make the first ``n`` calls of its game (expert.agent_call: prime-agent's own calls
    are not counted) and answers every later one itself with :data:`STOP_REPLY`, so the game is cut at ``n`` calls. Each
    proxy serves one user, and each user plays one instance. An answered call's captured response carries
    ``stopped: true``, so labelling leaves it out; a ``serve`` set before (an expert's) still runs for the first ``n``."""
    import threading
    import time
    import expert

    class StopProxy(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            inner, lock, count = self.serve, threading.Lock(), [0]

            def serve(parsed, call_id):
                if expert.agent_call(parsed.get("messages") or []):
                    with lock:
                        count[0] += 1
                        over = count[0] > n
                    if over:
                        return {"id": f"stop-{call_id}", "object": "chat.completion", "created": int(time.time()),
                                "model": str(parsed.get("model") or ""),
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": STOP_REPLY},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                "stopped": True}
                return inner(parsed, call_id) if inner is not None else None

            self.serve = serve
    return StopProxy


REPLAY_HOLD_S = 1200   # longer than the harness's 900 s action budget: a held call outlasts the turn it was cut in
# while a held call waits for the harness to cut its turn, its stream sends an SSE comment this often: prime-agent gives
# up on a reply that sends nothing for about 270 s (a recorded generation of 315 s, waited out in silence, made
# prime-agent retry, and the replay fell a call behind)
REPLAY_KEEPALIVE_S = 15
# a user's folder in a play under runs/loop, whatever home or root holds it
_USER_DIR = re.compile(r"/[^\s\"']*/runs/loop/[^/\s\"']+/users/u\d+")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# a prime-agent sub-agent's own id, in its system prompt's session paths: a fresh one each time the agent spawns it
_SUBAGENT = re.compile(r"\bsub-[0-9a-f]{8}\b")
# how a sub-agent's task opens: the parent agent's instruction, which the sub-agent's call answers
SUBAGENT_TASK = "[task from parent]"
_IDS = re.compile(f"{_UUID.pattern}|{_SUBAGENT.pattern}")


def harness_note(message: dict) -> bool:
    """Whether a message is one of prime-agent's own bracketed notes to the model ("[harness-digest]",
    "[python-skills-unavailable]", ...): what it says depends on the harness's own timing (whether the kernel
    had imported prime-agent's skill modules yet), not on the game. A sub-agent's task from its parent
    (:data:`SUBAGENT_TASK`) is no note: it is what the sub-agent's call answers."""
    import render
    text = render.text_of(message.get("content")).lstrip()
    return message.get("role") == "user" and text.startswith("[") and not text.startswith(SUBAGENT_TASK)


def _normal(text: str) -> str:
    """``text`` with what differs from user to user and from one spawn to the next (user folders, session ids, a
    sub-agent's own id) and its spacing made alike."""
    return " ".join(_SUBAGENT.sub("sub-<id>", _UUID.sub("<id>", _USER_DIR.sub("<user>", text))).split())


def conversation_of(messages: list[dict]) -> str:
    """Which of a harness's conversations an agent call belongs to: its first message that is neither a system prompt
    nor a harness note (:func:`harness_note`), as :func:`_normal` makes it alike. The agent's own conversation opens
    with the instance (and after a compaction with its summary), each sub-agent's with its task from the parent, and a
    call the model's own code sends through the proxy with whatever it sends; prime-agent runs sub-agents beside the
    agent, so their calls interleave in no fixed order (11 of a game's 51 calls began before the call before
    them had ended)."""
    import render
    return next((_normal(render.text_of(m.get("content"))) for m in messages
                 if m.get("role") not in ("system", "developer") and not harness_note(m)), "")


def _swap_ids(message: dict, ids: dict[str, str]) -> dict:
    """``message`` (a reply) with each recorded session id and sub-agent id in its text and tool calls replaced by the
    replay's own: the model writes the paths it was shown, such as its session log's, into its code (a
    replayed reply read the recorded session's log, which the replay's session does not have, and the code failed
    where it had run)."""
    if not ids:
        return message
    pattern = re.compile("|".join(map(re.escape, ids)))
    swap = lambda text: pattern.sub(lambda m: ids[m.group(0)], text) if isinstance(text, str) else text  # noqa: E731
    out = {k: swap(v) for k, v in message.items() if k != "tool_calls"}
    if message.get("tool_calls"):
        out["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": swap(tc["function"].get("arguments"))}}
                             if isinstance(tc.get("function"), dict) else tc for tc in message["tool_calls"]]
    return out


def call_kind(messages: list[dict]) -> str:
    """"agent" for the agent's own call, else which of prime-agent's internal calls: its /refine pass, or the opening
    tag of its compaction summarizer (``<conversation>``) or review (``<trigger>``)."""
    import expert
    import render
    if expert.agent_call(messages):
        return "agent"
    if messages and render.text_of(messages[0].get("content")).startswith(expert.REFINE):
        return "refine"
    return (render.text_of(messages[-1].get("content")).lstrip().split(maxsplit=1) or ["?"])[0] if messages else "?"


def replay_mismatch(new: list[dict], old: list[dict]) -> str | None:
    """Why a replayed request does not match the recorded one (None when it does).

    An internal call of prime-agent's (:func:`call_kind`) matches a recorded one of the same kind: its text carries a
    window of the conversation whose edges move with the harness's token counts (a compaction began at the game's
    first message in one run and at its first demo in the other), so it is not compared, and the recorded summary or
    review is served. An agent call matches when the two things the game turns on are as recorded: the latest message
    (what the model answers: a tool result with the same error status, or the same environment text) and the latest
    environment message anywhere in the request (the verdicts, demos and budget notices), user folders, session ids and
    sub-agent ids aside, which differ from user to user and from one spawn to the next. Tool output text is not
    compared: rerun code prints in other formats (the code-loop pilot: 98% of cells keep their error status, 40% their
    exact text). Earlier history is not compared either: compaction keeps a tail measured in tokens, so which messages
    survive it moves too (after two compactions, a game's empty tool result was kept in the recording and not
    in the replay). Prime-agent's own notes (:func:`harness_note`) are left out; a sub-agent's task from its parent is
    compared as the environment's text is (left out as a note, the comparison fell to the sub-agent's system
    prompt, whose own id differs at every spawn, and a resumed game drifted at its first sub-agent)."""
    import render
    kn, ko = call_kind(new), call_kind(old)
    if kn != ko:
        return f"call kind: {kn} against {ko} recorded"
    if kn != "agent":
        return None
    norm = _normal
    new, old = [m for m in new if not harness_note(m)], [m for m in old if not harness_note(m)]
    if not new or not old:
        return f"{len(new)} messages against {len(old)} recorded"
    a, b = new[-1], old[-1]
    if a.get("role") != b.get("role"):
        return f"latest message: {a.get('role')} against {b.get('role')} recorded"
    ta, tb = render.text_of(a.get("content")), render.text_of(b.get("content"))
    if a.get("role") == "tool" and bool(render.ERROR.search(ta)) != bool(render.ERROR.search(tb)):
        return "latest message: a tool result's error status"
    if a.get("role") in ("user", "system") and norm(ta) != norm(tb):
        return f"latest message: {norm(ta)[:120]!r} against {norm(tb)[:120]!r}"
    env = lambda ms: next((norm(render.text_of(m.get("content"))) for m in reversed(ms) if m.get("role") == "user"), "")  # noqa: E731
    if env(new) != env(old):
        return f"latest environment message: {env(new)[:120]!r} against {env(old)[:120]!r}"
    return None


def recorded_calls(source: Path) -> list[dict]:
    """A recorded game's model calls in the order they were made, as :func:`replay_proxy` serves them: each with its
    ``call_id``, ``request`` and ``response``, its generation's ``latency_ms``, and ``partial``, the reply the harness
    had read of a call it cancelled when it cut its turn at the budget ("" for any other). Only chat calls are kept,
    and of those not one that failed (the upstream answered an error, and the harness asked again): the harness's
    next ask is the one recorded, and a call of the model's own code that failed fails again live.
    ``source`` (gzipped JSON) is either a list of these already (``paper.py``'s journal of a round cut short) or a
    captured game (``captured/uNNN.json.gz``), whose seconds come from its user's ``calls.jsonl`` and whose partial
    replies from the rows of ``system/turns.jsonl`` cut at their budget, in the play's ``users/uNNN``. A turn row (an
    observation or a reprompt) is logged at its end ("t"); a call belongs to the first row ending after it started. A
    call's own "turn" counts actions, and one action can span an observation and its reprompts, each cut on its own
    (matching by "turn" streamed one cut turn's partial reply into a later cut)."""
    with gzip.open(source, "rt") as fp:
        doc = json.load(fp)
    chat = lambda c: isinstance((c.get("request") or {}).get("messages"), list) and c.get("status") != "failed"  # noqa: E731
    if isinstance(doc, list):
        return [c for c in doc if chat(c)]
    sysdir = Path(source).parent.parent / "users" / f"u{doc['user']:03d}"
    rows = list(map(json.loads, open(sysdir / "system" / "turns.jsonl"))) \
        if (sysdir / "system" / "turns.jsonl").exists() else []
    ends = [r.get("t", 0.0) for r in rows]
    cut = {k: r.get("reply") or "" for k, r in enumerate(rows) if (r.get("outcome") or {}).get("stop_reason") == "budget"}
    timing = {r["call_id"]: r for r in map(json.loads, open(sysdir / "calls.jsonl"))} \
        if (sysdir / "calls.jsonl").exists() else {}
    out = []
    for c in doc["calls"]:
        if not chat(c):
            continue
        t = timing.get(c.get("call_id"), {})
        row = bisect.bisect_left(ends, t["started"] - 0.5) if t.get("started") is not None else None
        out.append({**c, "latency_ms": t.get("latency_ms"), "partial": cut.get(row, "")})
    return out


def replay_proxy(base: type, plan_file: Path, log: Path | None = None, hold_s: float = REPLAY_HOLD_S,
                 keepalive_s: float = REPLAY_KEEPALIVE_S) -> type:
    """``base`` that replays each user's recorded game (:func:`recorded_calls`) up to a branch call or to the end of the
    recording, then hands the game to the model: the branch-and-compare test, and a round taken up from its last call
    after it was cut short (``paper.py``). ``plan_file`` holds one JSON line per user of the play: ``{"user": U,
    "source": a recorded game, "branch": the index in its calls of the branch call (null: none), "reply": the assistant
    message served there}``; a user with no line plays live from the start. Each proxy serves one user (its
    ``route_key``), and each user plays one instance.

    Every model call before the branch gets a recorded call's reply once the request passes :func:`replay_mismatch`;
    the environment then answers as it did, since its instances are seeded. The agent's own calls are replayed in their
    recorded order; prime-agent's internal calls (:func:`call_kind`) are matched in a queue of their own per kind, the
    next recorded summary to a summary, the next review to a review, since the harness times them by its own counters
    and clocks (a review fired after 57 turns in the recording, later in the replay; two summaries in a row
    came one agent call earlier), and one that never comes is skipped. A reply is served at once, with its recorded
    token counts (prime-agent compacts the conversation by them), and its recorded generation's seconds are taken off
    its turn's budget (the body's ``_budget_credit_s``, negative), so that a turn cut at its budget is cut where it was,
    and live play after the recording has what was left of its turn's budget, without the replay waiting either out
    (served at once with no account of their time, a turn cut at its budget left its cancelled call held for
    most of 900 s and ended without its cut; waiting out each reply's recorded seconds instead makes a replay nearly as
    long as the game, since 36% of the paper runs' turns end at the call cap).
    A recorded call with no reply (the one the harness cancelled when it cut the turn) is held until the harness moves
    on, its stream sending first the partial reply the harness had read by the cut (a half-written submit there
    was judged, and the turn's action was that submission), then an SSE comment every ``keepalive_s``, since
    prime-agent gives up on a reply that sends nothing for about 270 s. The serve hook returns the hold as the body's
    ``_replay_wait``, which the proxy's request handler (swapped in by ``start``) runs and removes before sending.

    The branch call gets the plan's reply. Once every recorded call has been served, the recording has ended and the
    game is handed to the model. A request that does not match, or of a kind the recording has no call of left while
    it still has others, marks the run drifted, and it goes to the model. After a branch, a hand-over or a drift, every
    call goes to ``base``'s own serve or the model. A replayed response carries ``replayed``, the id of the call it
    replays, and the branch's ``branch``. The class's ``recorded`` holds every call replayed from, by its id;
    ``replays``, the recorded call's id by the id of the call that replayed it (a held call ends cancelled, with no
    response to carry the mark); and ``counts``, by user, how many calls were replayed and how many went to the model,
    and whether the game branched, was handed over or drifted. ``log`` gets one JSON line per branch, hand-over or
    drift."""
    import threading
    import time
    plan = {p["user"]: p for p in map(json.loads, open(plan_file))}
    log_lock = threading.Lock()

    def write(record: dict) -> None:
        if log is not None:
            with log_lock, open(log, "a") as fp:
                fp.write(json.dumps({"t": round(time.time(), 1), **record}) + "\n")

    def body(call_id: str, parsed: dict, message: dict, finish: str | None, usage: dict | None, mark: str,
             value: Any) -> dict:
        msg = {k: message[k] for k in ("role", "content", "reasoning", "reasoning_content", "tool_calls") if message.get(k) is not None}
        msg.setdefault("role", "assistant")
        msg.setdefault("content", "")
        # the recorded call's token counts: prime-agent compacts the conversation by them (with zeros it never
        # compacted, and its compaction calls fell out of the recorded order)
        return {"id": f"{mark}-{call_id}", "object": "chat.completion", "created": int(time.time()),
                "model": str(parsed.get("model") or ""),
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish or ("tool_calls" if msg.get("tool_calls") else "stop")}],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, mark: value}

    class ReplayProxy(base):
        recorded: dict[str, dict] = {}
        replays: dict[str, str] = {}
        counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            inner, lock = self.serve, threading.Lock()
            user = int(str(getattr(self, "route_key", "u-1")).lstrip("u"))
            entry = plan.get(user)
            counts = ReplayProxy.counts[user]
            state = {"n": 0, "live": entry is None, "calls": None, "unserved": {}, "key": {}, "handed": set(),
                     "seen": collections.Counter(), "ids": {}, "handed_over": False}

            def live(parsed, call_id):
                counts["live"] += 1
                return inner(parsed, call_id) if inner is not None else None

            def key_of(kind: str, messages: list[dict]) -> str:
                """An agent call's conversation (:func:`conversation_of`), an internal call's own text."""
                import render
                if kind == "agent":
                    return conversation_of(messages)
                return _normal(render.text_of(messages[-1].get("content"))) if messages else ""

            def recorded() -> list[dict]:
                if state["calls"] is None:
                    state["calls"] = recorded_calls(Path(entry["source"]))
                    for j, c in enumerate(state["calls"]):   # each kind of call in its recorded order
                        messages = c["request"].get("messages") or []
                        kind = call_kind(messages)
                        state["unserved"].setdefault(kind, []).append(j)
                        state["key"][j] = key_of(kind, messages)
                        ReplayProxy.recorded[c["call_id"]] = c
                return state["calls"]

            def hold(key: str, seen: int, seconds: float) -> None:
                """Up to ``seconds``, ended early once the harness has sent its conversation's next call (it cut the
                turn)."""
                t0 = time.time()
                while time.time() - t0 < seconds and state["seen"][key] <= seen:
                    time.sleep(min(1.0, max(0.0, seconds - (time.time() - t0))))

            def serve(parsed, call_id):
                if state["live"]:
                    return live(parsed, call_id)
                calls, branch = recorded(), entry.get("branch")
                messages = parsed.get("messages") or []
                kind = call_kind(messages)
                key = key_of(kind, messages)
                with lock:
                    i = state["n"]              # this request's place among all of this user's requests
                    state["n"] += 1
                    state["seen"][key] += 1
                    seen = state["seen"][key]
                    handed = kind == "agent" and key in state["handed"]
                    j = None
                    if not handed:
                        # an agent call takes the next recorded call of its own conversation; an internal call the
                        # recorded one of its kind with the same text, else the next of its kind
                        unserved = state["unserved"].get(kind, [])
                        j = next((x for x in unserved if state["key"][x] == key),
                                 unserved[0] if unserved and kind != "agent" else None)
                        if j is not None:
                            unserved.remove(j)
                            left = None
                        else:
                            left = sum(map(len, state["unserved"].values()))
                if handed:
                    return live(parsed, call_id)
                if j is None:
                    if entry.get("cut"):
                        # a recording cut short ends here for this conversation: it is handed to the model, as is any
                        # conversation the recording never reached (a call of the model's own code, say), and the
                        # others play their recorded calls on
                        if kind == "agent":
                            # the game is handed over at its first live call once every recorded call is served, or
                            # once its own conversation's recording has run out; a call of a conversation the recording
                            # never had, while it still has calls to serve, is the model's own code calling on the side
                            with lock:
                                first = not state["handed_over"] and (left == 0 or key in state["key"].values())
                                state["handed_over"] |= first
                                state["handed"].add(key)
                            if first:
                                counts["handed_off"] += 1
                                write({"user": user, "handed_off_at": i, "recorded": len(calls), "left": left,
                                       "conversation": key[:80]})
                        return live(parsed, call_id)
                    state["live"] = True
                    counts["drifted"] += 1
                    write({"user": user, "drifted_at": i, "branch": branch, "left": left,
                           "why": f"no recorded {kind} call {'of its conversation ' if kind == 'agent' else ''}left"})
                    return live(parsed, call_id)
                rec = calls[j]
                why = replay_mismatch(messages, rec["request"].get("messages") or [])
                if why:
                    state["live"] = True
                    counts["drifted"] += 1
                    import render
                    last = lambda ms: [(m.get("role"), render.text_of(m.get("content"))[:100]) for m in ms[-3:]]  # noqa: E731
                    write({"user": user, "drifted_at": i, "recorded_call": j, "branch": branch, "why": why,
                           "turn": rec.get("turn"), "new": last(messages),
                           "recorded": last(rec["request"].get("messages") or [])})
                    return live(parsed, call_id)
                usage = (rec.get("response") or {}).get("usage")
                if j == branch:
                    state["live"] = True
                    counts["branched"] += 1
                    write({"user": user, "branched_at": i, "recorded_call": j})
                    return body(call_id, parsed, entry["reply"], None, usage, "branch", True)
                counts["replayed"] += 1
                ReplayProxy.replays[call_id] = rec["call_id"]
                # the replay's own session and sub-agent ids for the recorded ones, read off the two system prompts,
                # which differ in nothing else
                import render
                system = lambda ms: render.text_of(ms[0].get("content")) if ms and ms[0].get("role") == "system" else ""  # noqa: E731
                ids_new, ids_old = _IDS.findall(system(messages)), _IDS.findall(system(rec["request"].get("messages") or []))
                if len(ids_new) == len(ids_old):
                    with lock:
                        state["ids"].update((o, n) for o, n in zip(ids_old, ids_new) if o != n)
                choice = ((rec.get("response") or {}).get("choices") or [{}])[0]
                out = body(call_id, parsed, _swap_ids(choice.get("message") or {}, dict(state["ids"])),
                           choice.get("finish_reason"), usage, "replayed", rec["call_id"])
                if not choice.get("message"):
                    # cancelled when its turn was cut: held until the harness has cut the turn again and sent its
                    # conversation's next call, streaming first the partial reply the harness had read by the cut
                    out["_replay_wait"] = lambda: hold(key, seen, hold_s)
                    out["_replay_partial"] = _swap_ids({"content": rec.get("partial") or ""}, dict(state["ids"]))["content"]
                else:
                    out["_budget_credit_s"] = -(rec.get("latency_ms") or 0.0) / 1000
                return out

            self.serve = serve

        def start(self):
            out = super().start()
            from continual_arc_baselines.proxy import served_events
            handler = self._server.RequestHandlerClass

            class KeepaliveHandler(handler):
                """The proxy's handler, which runs a held reply's ``_replay_wait`` while the reply streams: headers at
                once, the partial reply, an SSE comment every ``keepalive_s`` (and the tracker touched, as a long serve
                touches it), then the reply's events."""

                def _send_served(self, served, stream):
                    hold = served.pop("_replay_wait", None)
                    partial = served.pop("_replay_partial", "")
                    if hold is None:
                        return super()._send_served(served, stream)
                    done = threading.Event()
                    threading.Thread(target=lambda: (hold(), done.set()), daemon=True, name="replay-wait").start()
                    if not stream:
                        while not done.wait(keepalive_s):
                            self.owner.tracker.touch()
                        return super()._send_served(served, stream)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()

                    def chunk(data: bytes) -> None:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                        self.wfile.flush()
                    if partial:
                        head = {k: served[k] for k in ("id", "created", "model") if k in served}
                        delta = {"role": "assistant", "content": partial}
                        chunk(b"data: " + json.dumps({**head, "object": "chat.completion.chunk",
                                                       "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}).encode() + b"\n\n")
                    while not done.wait(keepalive_s):
                        self.owner.tracker.touch()
                        chunk(b": keepalive\n\n")
                    for event in served_events(served) + ["[DONE]"]:
                        chunk(b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()

            self._server.RequestHandlerClass = KeepaliveHandler
            return out
    return ReplayProxy


def served_as_multimodal(model: str) -> bool:
    """Whether vLLM serves ``model`` (a Hugging Face id) through its multimodal class: the checkpoint's architecture,
    read from its config (the local cache once downloaded), not its name."""
    from huggingface_hub import hf_hub_download
    config = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    return any(a in MULTIMODAL_ARCHITECTURES for a in config.get("architectures") or [])


def served_adapter(model: str, src: Path) -> Path:
    """The directory vLLM should load the adapter saved at ``src`` from: ``src`` itself, or, for a model vLLM serves as
    multimodal, a copy beside it (``src.vllm``) with every weight under the checkpoint's names. The trainer's own files
    are left as they are."""
    if not served_as_multimodal(model):
        return src
    from safetensors.torch import load_file, save_file
    dst = src.with_name(src.name + ".vllm")
    dst.mkdir(exist_ok=True)
    shutil.copy(src / "adapter_config.json", dst)
    weights = load_file(str(src / "adapter_model.safetensors"))
    if not all(k.startswith(PEFT_LAYERS) for k in weights):
        raise ValueError(f"{src}: an adapter weight outside the language model's layers")
    save_file({SERVED_LAYERS + k[len(PEFT_LAYERS):]: v for k, v in weights.items()}, str(dst / "adapter_model.safetensors"))
    return dst


# -- schedule ------------------------------------------------------------------------------------------

def _round_slots(full: Any, round_: int, user: int | None) -> list[dict]:
    """The first stream slot of every task a built schedule's users meet, for a round's four users (4R..4R+3) or just
    ``user``: fresh test inputs and demonstrations each time, over the schedule's fixed task set."""
    if user is not None:
        users = range(user, user + 1)
    else:
        users = range(INSTANCES_PER_TASK * round_, INSTANCES_PER_TASK * (round_ + 1))
    first: dict[tuple[int, str], dict] = {}
    for s in full.slots:
        if s.phase == "stream" and s.user in users and (s.user, s.task_id) not in first:
            first[(s.user, s.task_id)] = {"task_id": s.task_id, "instance_seed": s.instance_seed,
                                          "source": full.sources[s.task_id]}
    return [first[k] for k in sorted(first)]


def stream_slots(round_: int, user: int | None = None) -> list[dict]:
    """The first stream instance of every task for four of the stream's :data:`DEV_USERS` users (users 4R..4R+3), or
    with ``user`` for that user alone (25 instances, one per task: a cycle's games): fresh test inputs and
    demonstrations each time, over the same 25 tasks."""
    from continual_arc.configs import build_schedule, with_users
    full = build_schedule(with_users(STREAM_CONFIG, DEV_USERS), "workstreams", 0)
    return _round_slots(full, round_, user)


# The development set: this many ARC-1 training tasks (Re-ARC has 400), drawn under their own label
DEV_TASKS = 25
DEV_LABEL = "loop-dev"


def dev_task_pool() -> list[str]:
    """Every ARC-1 training task id the development draw may choose from: outside arc1-small's 25 stream tasks and
    outside every epilogue that the loop or the paper plays, at the canonical seeds."""
    from continual_arc.configs import CANONICAL_SEEDS, build_schedule, get_config, stream_task_pool
    stream_tasks = {x["task_id"] for x in stream_slots(0)}
    epilogues = {s.task_id for name in (STREAM_CONFIG, HELDOUT_CONFIG) for seed in CANONICAL_SEEDS
                 for s in build_schedule(name, "workstreams", seed).slots if s.phase == "epilogue"}
    pool = stream_task_pool(get_config(STREAM_CONFIG))
    return [t for t in pool if t not in stream_tasks and t not in epilogues]


def dev_tasks() -> list[str]:
    """The development set's tasks: :data:`DEV_TASKS` training tasks drawn from :func:`dev_task_pool`, off the paper's
    own stream and every epilogue it plays, so the loop's recipe is chosen without tuning on the paper's data. A seeded
    sample of its own: the same tasks every round."""
    from continual_arc.configs import select_tasks
    return select_tasks(dev_task_pool(), DEV_TASKS, 0, DEV_LABEL)


def dev_slots(round_: int, user: int | None = None) -> list[dict]:
    """:func:`stream_slots`'s shape (rounds of :data:`DEV_TASKS` tasks x four fresh users, or one user's own cycle) but
    over :func:`dev_tasks` instead of arc1-small's 25, built the same way (``build_schedule``'s own stream process,
    pointed at the development tasks in place of the pinned pool)."""
    from continual_arc.configs import build_schedule, with_users
    full = build_schedule(with_users(STREAM_CONFIG, DEV_USERS), "workstreams", 0, rearc_task_ids=dev_tasks())
    return _round_slots(full, round_, user)


def heldout_slots() -> list[dict]:
    """One instance of each of arc1-medium's 100 held-out ARC-1 evaluation tasks (a superset of arc1-small's 25)."""
    from continual_arc.configs import build_schedule
    full = build_schedule(HELDOUT_CONFIG, "workstreams", 0)
    seen: dict[str, dict] = {}
    for s in full.slots:
        if s.phase == "epilogue" and s.task_id not in seen:
            seen[s.task_id] = {"task_id": s.task_id, "instance_seed": s.instance_seed, "source": full.sources[s.task_id]}
    return list(seen.values())


def validation_tasks() -> list[str]:
    """The validation split's tasks: :data:`VALIDATION_TASKS` ARC-1 evaluation tasks in no arc1-small or arc1-medium
    epilogue at the canonical seeds (served to any number of users, the held-out tasks are arc1-small's), so neither
    the held-out test nor any epilogue
    the paper plays holds one, and the stream, which draws on the training tasks, never meets them. A seeded sample of
    its own: the same tasks every round."""
    from continual_arc.configs import CANONICAL_SEEDS, build_schedule, epilogue_task_pool, get_config, select_tasks
    played = {s.task_id for name in (STREAM_CONFIG, HELDOUT_CONFIG) for seed in CANONICAL_SEEDS
              for s in build_schedule(name, "workstreams", seed).slots if s.phase == "epilogue"}
    pool = [t for t in epilogue_task_pool(get_config(HELDOUT_CONFIG)) if t not in played]
    return select_tasks(pool, VALIDATION_TASKS, 0, VALIDATION_LABEL)


def validation_slots() -> list[dict]:
    """One instance of each validation task, built as an epilogue builds its instances (the same every round)."""
    from continual_arc.configs import get_config
    from continual_arc.schedule import epilogue_schedule
    source = get_config(HELDOUT_CONFIG).epilogue_source
    return [{"task_id": s.task_id, "instance_seed": s.instance_seed, "source": source}
            for s in epilogue_schedule(validation_tasks(), 0, source).slots]


def task_set_slots(path: str | Path) -> list[dict]:
    """A task set's instances (a JSON file: ``tasks`` and ``held`` lists of ARC-1 training task ids, ``games`` and
    ``held_games`` per task): fresh Re-ARC instances, each seed derived from the set's name, the task and the game's
    number, as the stream derives its own, in a seeded order so a play's cutoff spreads over the tasks."""
    from continual_arc.seeds import derive_seed
    d = json.loads(Path(path).expanduser().read_text())
    if set(d["tasks"]) & set(d["held"]):
        raise SystemExit(f"{path}: a task is both trained on and held out")
    want = [(t, d["games"]) for t in d["tasks"]] + [(t, d["held_games"]) for t in d["held"]]
    slots = [{"task_id": t, "instance_seed": derive_seed("loop-tasks", d["name"], t, k), "source": "rearc"}
             for t, n in want for k in range(n)]
    random.Random(derive_seed("loop-tasks", d["name"], "order")).shuffle(slots)
    return slots


ROUND_GROUPS = ("stream", "dev")  # schedule groups that take --round/--user, one round's instances at a time


def cmd_schedule(a: argparse.Namespace) -> None:
    from continual_arc.schedule import Schedule, Slot, interleave_users
    if a.group == "tasks" and not a.file:
        raise SystemExit("schedule tasks needs --file")
    picked = {"stream": lambda: stream_slots(a.round, a.user), "dev": lambda: dev_slots(a.round, a.user),
              "validation": validation_slots, "heldout": heldout_slots,
              "tasks": lambda: task_set_slots(a.file)}[a.group]()
    per_user = [Schedule(name=f"u{i}", seed=0, slots=[Slot(x["task_id"], x["instance_seed"], "probe")],
                         sources={x["task_id"]: x["source"]}) for i, x in enumerate(picked)]
    if a.group == "tasks":
        name = f"loop-tasks-{Path(a.file).stem}"
    elif a.group not in ROUND_GROUPS:
        name = f"loop-{a.group}"
    else:
        name = f"loop-{a.group}-u{a.user}" if a.user is not None else f"loop-{a.group}-r{a.round}"
    sched = interleave_users(per_user, 0, name=name)
    sched.meta = {"mode": "probe", "seed": 0, "group": a.group, "round": a.round if a.group in ROUND_GROUPS else None,
                  "user": a.user if a.group in ROUND_GROUPS else None}
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    sched.save(out)
    print(json.dumps({"out": str(out), "instances": len(picked), "tasks": len({x["task_id"] for x in picked})}))


# -- play ----------------------------------------------------------------------------------------------

def cmd_play(a: argparse.Namespace) -> None:
    import logging
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, run as run_stream
    from continual_arc.schedule import Schedule
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.proxy import LLMProxy
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    from continual_arc_baselines.proxy import THINKING_FIELD, VLLM_DROP, VLLM_ROLES

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(a.run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    captured = run_dir / "captured"
    captured.mkdir(exist_ok=True)
    if a.alias_from:
        # the same task aliases as the run this one is paired with, so both see identical prompts
        src = Path(a.alias_from).expanduser() / "alias-salt"
        if not (run_dir / "alias-salt").exists():
            shutil.copy(src, run_dir / "alias-salt")
    sched = Schedule.load(Path(a.schedule).expanduser())
    context = Path(a.context_file).expanduser().read_text() if a.context_file else None
    settings = {k: v for k, v in vars(a).items() if k != "fn"}
    harness = harness_of(a.compact_at, a.compact_keep)
    budget = HARNESS_BUDGETS[a.harness]
    (run_dir / "driver.json").write_text(json.dumps(
        {**settings, "harness": harness, "harness_budget": {"name": a.harness, **budget}, "context": context}, indent=1))

    # every user's proxy: the run's own, with each of the play's options below wrapped around it
    proxy = LLMProxy
    if a.oracle_card:
        # an oracle's hint before each of the player's moves (oracle_hint.py): c1 (the context), the task's card and the
        # hint for that moment as the opening of the model's reasoning (placement thought) or as notes at the end of the
        # last message (message), as train.py's teacher reads a call with the card and the hint as its thought or note
        if not context:
            raise SystemExit("--oracle-card needs --context-file (c1)")
        import oracle_hint
        oracle = oracle_hint.Oracle(oracle_hint.http_post(a.oracle_url), model=a.oracle_model,
                                    effort=a.oracle_effort or None)
        proxy = oracle_hint.hint_proxy(LLMProxy, context, Path(a.oracle_card).expanduser().read_text(),
                                       oracle, run_dir / "hints.jsonl", placement=a.context_placement)
    elif context:
        block = context_block(context)

        class ContextProxy(LLMProxy):
            """The run's proxy, with the context shown at every model call: appended to the last message, or as the
            opening of the model's reasoning (``--context-placement``)."""

            def prepare(self, headers, body, path):
                out, body, parsed = super().prepare(headers, body, path)
                if parsed is not None and path.rstrip("/").endswith("/chat/completions") and parsed.get("messages"):
                    shown = inject_thought(parsed, context) if a.context_placement == "thought" else inject(parsed, block)
                    body = json.dumps(shown).encode()
                return out, body, parsed

        proxy = ContextProxy

    if a.seed_per_call:
        proxy = seeded_proxy(proxy)
    if a.stop_after:
        proxy = stop_proxy(proxy, a.stop_after)
    if a.replay:
        proxy = replay_proxy(proxy, Path(a.replay).expanduser(), run_dir / "replay.jsonl")

    kwargs = {"model": MODEL, "context_window": 131072, "max_tokens": 32768, **harness, **budget}

    def save_instance(t) -> None:
        doc = {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
               "phase": t.phase, "solved": getattr(t, "solved", None), "turns": t.turns,
               "calls": [{"call_id": c.call_id, "turn": c.turn, "phase": c.phase, "status": c.status,
                          "finish_reason": c.finish_reason, "request": c.request, "response": c.response}
                         for c in t.calls]}
        tmp = captured / f"u{t.user:03d}.json.gz.tmp"
        try:
            # default=str: a harness's turn record can hold values JSON has no form for (an Ellipsis in one
            # feedback crashed a whole run at its last instance); a save that still fails is logged, never fatal
            with gzip.open(tmp, "wt") as fp:
                json.dump(doc, fp, default=str)
            tmp.rename(captured / f"u{t.user:03d}.json.gz")
        except Exception:  # noqa: BLE001
            logging.exception("could not save instance of user %s", t.user)

    template = json.loads(a.template_kwargs) if a.template_kwargs else dict(TEMPLATE_KWARGS)
    overrides = {"model": a.model, "chat_template_kwargs": template, "return_token_ids": True,
                 **json.loads(a.sampling or "{}")}
    expert = None
    if a.expert:
        # the expert plays the call right after each failure (expert.py), every user's proxy asking it first
        from expert import Expert
        expert = Expert(a.expert_url, a.expert, providers=a.expert_providers.split(",") if a.expert_providers else None,
                        effort=a.expert_effort or None, log=run_dir / "expert.jsonl", every=a.expert_every)
    learner = CollectiveHarnessLearner(lambda user: PrimeAgentLearner(**kwargs), run_dir, schedule=sched,
                                       upstream=a.base_url, body_overrides=overrides, body_drop=VLLM_DROP,
                                       role_map=VLLM_ROLES, thinking_field=THINKING_FIELD, declared_context_length=131072,
                                       capture=True, snapshot=False, instance_hook=save_instance, reset_hook=lambda: None,
                                       serve=expert.serve if expert else None, proxy=proxy)
    env = build_env(sched, EnvConfig(), TaskRegistry.default())
    try:
        run_stream(learner, sched, EnvConfig(), out_dir=run_dir, env=env, resume=True, learner_name=f"loop-{sched.name}")
    finally:
        learner.close()
        if expert:
            expert.close()


# -- compare -------------------------------------------------------------------------------------------

def records(run_dir: Path) -> dict[tuple[str, int], dict]:
    """A run's instance records, keyed by (task, instance seed)."""
    out = {}
    for line in open(run_dir / "records.jsonl"):
        r = json.loads(line)
        out[(r["task_id"], r["instance_seed"])] = r
    return out


def budget_stops(run_dir: Path) -> int:
    """The environment's re-prompts after a turn ran over the action budget, over the run's users."""
    n = 0
    for f in run_dir.glob("users/u*/system/turns.jsonl"):
        n += sum(BUDGET_STOP in (json.loads(line).get("message") or "") for line in open(f))
    return n


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    m, k = b + c, min(b, c)
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def mcnemar_p_fewer(wins: int, losses: int) -> float:
    """Exact one-sided McNemar p-value that the new run solves fewer, from its ``wins`` (solved by it alone) and
    ``losses`` (solved by the base alone): the chance of at most ``wins`` of the discordant pairs under no difference."""
    m = wins + losses
    if m == 0:
        return 1.0
    return sum(math.comb(m, i) for i in range(wins + 1)) / 2 ** m


def paired_bootstrap(diffs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """A 95% percentile interval for the mean of paired differences."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def compare(base_dir: Path, new_dir: Path, tasks: set[str] | None = None) -> dict[str, Any]:
    """The paired comparison of ``new_dir`` against ``base_dir`` over their shared instances (only ``tasks``' when
    given): ``not_worse`` unless the new run solves fewer at one-sided p < :data:`WORSE_P`."""
    a, b = records(base_dir), records(new_dir)
    keys = [k for k in sorted(set(a) & set(b)) if tasks is None or k[0] in tasks]
    diffs = [b[k]["cost"] - a[k]["cost"] for k in keys]
    only_new = sum(b[k]["solved"] and not a[k]["solved"] for k in keys)
    only_base = sum(a[k]["solved"] and not b[k]["solved"] for k in keys)

    def counts(recs: dict, run_dir: Path) -> dict[str, Any]:
        rs = [recs[k] for k in keys]
        return {"mean_cost": round(sum(r["cost"] for r in rs) / len(rs), 3) if rs else None,
                "solved": sum(r["solved"] for r in rs),
                "gave_up": sum(bool(r.get("gave_up")) for r in rs), "demo_requests": sum(r["demo_requests"] for r in rs),
                "wrong_attempts": sum(r["wrong_attempts"] for r in rs),
                "invalid_submissions": sum(r.get("invalid_submissions", 0) for r in rs),
                "model_calls": sum((r.get("resources") or {}).get("llm_requests") or 0 for r in rs),
                "parse_failures": sum((r.get("resources") or {}).get("system_parse_failures") or 0 for r in rs),
                "forced_give_ups": sum((r.get("resources") or {}).get("system_forced_give_ups") or 0 for r in rs),
                "budget_stops": budget_stops(run_dir)}

    lo, hi = paired_bootstrap(diffs) if diffs else (float("nan"), float("nan"))
    fewer_p = mcnemar_p_fewer(only_new, only_base)
    return {"paired": len(keys), "base": counts(a, base_dir), "new": counts(b, new_dir),
            "mean_cost_diff": round(sum(diffs) / len(diffs), 3) if diffs else None,
            "mean_cost_diff_95": [round(lo, 3), round(hi, 3)],
            "solved_only_new": only_new, "solved_only_base": only_base,
            "mcnemar_p": mcnemar_p(only_new, only_base), "mcnemar_p_fewer": fewer_p,
            "not_worse": bool(keys) and fewer_p >= WORSE_P}


def decide(best_dir: Path, new_dir: Path) -> dict[str, Any]:
    """The keep rule after a round, from the best student's validation play and the round's student's, paired by
    instance: the round's student is kept unless it is clearly worse (``not_worse``), and a kept student that solves
    more of the paired instances is the new best (``new_best``). A kept student that is not the new best still starts
    the next round."""
    out = compare(best_dir, new_dir)
    out["new_best"] = out["not_worse"] and out["new"]["solved"] > out["base"]["solved"]
    return out


def cmd_compare(a: argparse.Namespace) -> None:
    print(json.dumps(compare(Path(a.base).expanduser(), Path(a.new).expanduser()), indent=1))


def cmd_decide(a: argparse.Namespace) -> None:
    print(json.dumps(decide(Path(a.best).expanduser(), Path(a.new).expanduser()), indent=1))


def cmd_vllm_adapter(a: argparse.Namespace) -> None:
    print(served_adapter(a.model, Path(a.adapter).expanduser()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("schedule")
    s.add_argument("group", choices=["stream", "dev", "validation", "heldout", "tasks"])
    s.add_argument("--round", type=int, default=0)
    s.add_argument("--user", type=int, default=None,
                   help="stream/dev: this one user's 25 instances (a cycle's games) in place of a round's four users")
    s.add_argument("--file", default=None,
                   help="tasks: the task set, a JSON file of name, tasks, held, games and held_games")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_schedule)
    pl = sub.add_parser("play")
    pl.add_argument("--schedule", required=True)
    pl.add_argument("--run-dir", required=True)
    pl.add_argument("--base-url", default="http://localhost:8101/v1")
    pl.add_argument("--model", default=MODEL, help="the served name: the base model, or an adapter's")
    pl.add_argument("--context-file", default=None, help="text shown at every model call")
    pl.add_argument("--context-placement", choices=PLACEMENTS, default="message",
                    help="message: appended to the last message; thought: the opening of the model's own reasoning")
    pl.add_argument("--alias-from", default=None, help="the run whose task aliases to reuse (a paired run)")
    pl.add_argument("--template-kwargs", default=None,
                    help="JSON chat_template_kwargs for every call, for a model other than the loop's; "
                         "default TEMPLATE_KWARGS")
    pl.add_argument("--sampling", default=None,
                    help="JSON sampling fields set on every call, e.g. {\"temperature\":1.0,\"top_p\":1.0,\"top_k\":-1} "
                         "(the harness sends none, so the server's defaults apply without it)")
    pl.add_argument("--expert", default=None,
                    help="the expert model that plays the call right after each failure (expert.py), e.g. z-ai/glm-5.3")
    pl.add_argument("--expert-url", default="http://localhost:18200/v1",
                    help="the expert's OpenAI-compatible endpoint (expert_relay.py's, on this machine or tunnelled to it)")
    pl.add_argument("--expert-providers", default=None, help="comma-separated hosts the expert's calls may use")
    pl.add_argument("--expert-effort", default=None, help="the expert's reasoning effort")
    pl.add_argument("--replay", default=None,
                    help="a branch-and-compare plan (JSON lines, one per user): replay each user's recorded game up to "
                         "its branch call, serve the planned reply there, then the model plays on (replay_proxy)")
    pl.add_argument("--expert-every", action="store_true",
                    help="the expert plays every call, so the game is the expert's own (with --context-file, it reads it)")
    pl.add_argument("--oracle-card", default=None,
                    help="the task's card: before each of the player's moves an oracle writes a hint (oracle_hint.py), "
                         "and c1 (--context-file, placement thought), the card and the hint open the model's reasoning")
    pl.add_argument("--oracle-url", default="http://localhost:18200/v1",
                    help="the oracle's OpenAI-compatible endpoint (expert_relay.py's, on this machine or tunnelled to it)")
    pl.add_argument("--oracle-model", default="z-ai/glm-5.3")
    pl.add_argument("--oracle-effort", default="low", help="the oracle's reasoning effort")
    pl.add_argument("--seed-per-call", action="store_true",
                    help="a random seed on every model call (the server otherwise serves unseeded calls from one stream)")
    pl.add_argument("--compact-at", type=int, default=HARNESS["compact_at"],
                    help="the context size (tokens) at which prime-agent summarizes the conversation; 0: its default")
    pl.add_argument("--compact-keep", type=int, default=HARNESS["compact_keep"],
                    help="the tokens a summary keeps word for word; 0: prime-agent's default")
    pl.add_argument("--stop-after", type=int, default=0,
                    help="cut each game after the student's first N calls (stop_proxy; 0: whole games)")
    pl.add_argument("--harness", choices=sorted(HARNESS_BUDGETS), default="default",
                    help="default: prime-agent's own action budget (900 s or 40 model calls, follow-up 300 s or 6 "
                         "calls); block: the trained rows' harness (40 calls meant to stop a turn; the per-action and "
                         "follow-up seconds raised to 1,740 s, just under the 1,800 s hard ceiling; the follow-up's "
                         "6 calls kept)")
    pl.set_defaults(fn=cmd_play)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.set_defaults(fn=cmd_compare)
    d = sub.add_parser("decide")
    d.add_argument("best", help="the best student's validation play")
    d.add_argument("new", help="the round's student's validation play")
    d.set_defaults(fn=cmd_decide)
    v = sub.add_parser("vllm-adapter")
    v.add_argument("adapter")
    v.add_argument("--model", default=MODEL)
    v.set_defaults(fn=cmd_vllm_adapter)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
