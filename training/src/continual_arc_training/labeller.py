"""DAgger's and OPSD's labels: the oracle's own reply at every call the student made, made into a training set and
split into a training side and a held-out side. The real-time driver's labeller runs these round by round
(continual_arc_training.realtime), each writing into the round's ``labels/`` folder:

- :func:`points`: every call the student made as the agent, one point each, in ``points.jsonl``; a point's id names
  its call by play, user, index and the call's own id (:func:`point_id`), so a label is never read for another game's
  call at the same index (a round played again reuses every index);
- :func:`label`: the oracle's reply at every point (:func:`continual_arc_training.oracle.ask_label`), under its
  working rules, as ``labels/ID.json``; resumable, and no new ask once the round's budget is spent;
- :func:`label_set`: each label as the student writes it (:func:`student_message`: all of the oracle's prose as the
  student's reasoning, only the action and any tool calls in view), rendered after its prompt by the local chat
  template (:func:`render_reply`), a game of labels a ``set/captured/uNNN.json.gz``, the counts in ``set.json``; a label
  that makes no move, or that the template cannot render, is left out and counted;
- :func:`split`: the set's games into a training side and a held-out side, one game in ``hold_every``
  (:func:`held_game`), written last as ``split/split.json``, which marks the round labelled.
"""
from __future__ import annotations

import collections
import concurrent.futures
import gzip
import json
import re
import threading
from pathlib import Path
from typing import Any

from continual_arc_training import oracle
from continual_arc_training.calls import (
    after_failure,
    agent_call,
    render_prompt_ids,
    template_messages,
    template_tools,
    text_of,
)

# a verdict or notice after which the only move left is to finish: nothing to decide
FORCED = ("submission was CORRECT", "You gave up", "The instance is over", "The attempt cap is reached")
# prime-agent's own notes to the model mid-action (after a tool's output, a compaction, a refinement): no new game state
HARNESS_NOTE = "["


def games(play: Path) -> list[Path]:
    return sorted((play / "captured").glob("u*.json.gz"))


def load(f: Path) -> dict[str, Any]:
    with gzip.open(f, "rt") as fp:
        return json.load(fp)


def student_calls(doc: dict[str, Any]) -> list[int]:
    """The indexes of a game's answered calls the model made as the agent, in order."""
    return [i for i, c in enumerate(doc["calls"])
            if c.get("request") and (c.get("response") or {}).get("choices")
            and agent_call(c["request"].get("messages") or [])]


def decision_kind(messages: list[dict[str, Any]]) -> str | None:
    """Which decision point a call is: the agent's call at the start of the game ("start"), or right after the game
    said something new (a failure, demonstrations received, a turn stopped at its budget, a reply with no valid
    action, other feedback); None for a call in the middle of an action (after a tool's output or a harness note
    only), and for one where the only move left is to finish."""
    if not agent_call(messages):
        return None
    last = max((j for j, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    if last == -1:
        return "start"
    said = [text_of(m.get("content")).strip() for m in messages[last + 1:] if m.get("role") == "user"]
    said = [t for t in said if t and not t.startswith(HARNESS_NOTE)]
    if not said:
        return None
    text = " ".join(said)
    if any(f in text for f in FORCED):
        return None
    if after_failure(messages):
        return "failure"
    if "Demos received" in text:
        return "demos"
    if "exceeded the time or model-call budget" in text:
        return "budget"
    if "could not find a valid action" in text:
        return "invalid"
    return "other"


def kind(messages: list[dict[str, Any]]) -> str:
    """A point's kind: a decision point's (:func:`decision_kind`), "forced" when only finishing is left, else "loop"
    (in the middle of an action)."""
    k = decision_kind(messages)
    if k:
        return k
    last = max((j for j, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    said = " ".join(text_of(m.get("content")) for m in messages[last + 1:] if m.get("role") == "user")
    return "forced" if any(f in said for f in FORCED) else "loop"


def point_id(play: Path, doc: dict[str, Any], i: int) -> str:
    """A point's id, which names its label's file: the play, the user and the call's index, then the call's own id."""
    return f"{play.name}-u{doc['user']:03d}-c{i:04d}-{doc['calls'][i]['call_id'][:12]}"


def points(plays: list[Path]) -> list[dict[str, Any]]:
    """Every call the model made as the agent in ``plays``' games, as a point: its id, play, user, index, call id,
    place in the game (``n`` of ``len``), kind, task and the game's outcome."""
    out = []
    for play in plays:
        for f in games(play):
            doc = load(f)
            calls = student_calls(doc)
            for n, i in enumerate(calls):
                out.append({"id": point_id(play, doc, i), "play": str(play), "user": doc["user"], "i": i,
                            "call_id": doc["calls"][i]["call_id"], "n": n + 1, "len": len(calls),
                            "kind": kind(doc["calls"][i]["request"].get("messages") or []),
                            "task_id": doc["task_id"], "solved": doc.get("solved")})
    return out


class StalePoint(Exception):
    """A point whose game no longer has its call: the round was played again since the point was read."""


_games: dict[tuple[str, int], list] = {}
_games_lock = threading.Lock()


def point_call(p: dict[str, Any]) -> dict[str, Any]:
    """A point's call. Each game is read once and kept while its points are worked through (they come in game order),
    since a game's file is several megabytes. The call must be the one the point names (its call id): a kept game that
    has since been played again is read again, and a point the game on disk no longer has raises :class:`StalePoint`,
    never another call in its place."""
    key = (p["play"], p["user"])
    with _games_lock:
        for attempt in range(2):
            if attempt or key not in _games:
                _games.pop(key, None)
                if len(_games) >= 48:
                    _games.pop(next(iter(_games)))
                _games[key] = load(Path(p["play"]) / "captured" / f"u{p['user']:03d}.json.gz")["calls"]
            calls = _games[key]
            if p["i"] < len(calls) and calls[p["i"]]["call_id"] == p["call_id"]:
                return calls[p["i"]]
        raise StalePoint(f"{p['id']}: {p['play']} u{p['user']:03d} no longer has call {p['i']} ({p['call_id']})")


def label_cost(label: dict[str, Any]) -> float:
    """A saved label's dollars: every ask's (``spent``), else, for a label saved without that field, the last ask's."""
    return label["spent"] if "spent" in label else (label.get("usage") or {}).get("cost") or 0.0


def spent(label_dirs: list[Path]) -> float:
    """The dollars every label under ``label_dirs`` (each a round's ``labels/`` folder) cost."""
    return sum(label_cost(json.loads(f.read_text())) for d in label_dirs for f in (d / "labels").glob("*.json"))


def label(out: Path, ask: Any, rules: str, *, budget: float, workers: int, need_thinking: int,
          min_labelled_pct: float) -> dict[str, Any]:
    """The oracle's reply (``ask``, :meth:`continual_arc_training.oracle.Oracle.ask`) at every point of
    ``out/points.jsonl``, the call read exactly as the student read it with the working ``rules`` at the end of its last
    message, saved as ``out/labels/ID.json``; a point already labelled is kept. No new point is asked once the labels so
    far cost ``budget`` dollars (SystemExit after the pass), and the pass fails (SystemExit) when fewer than
    ``min_labelled_pct`` percent of the points have a label. Returns the outcomes' counts and the cost."""
    (out / "labels").mkdir(parents=True, exist_ok=True)
    pts = [json.loads(line) for line in open(out / "points.jsonl")]
    lock = threading.Lock()
    cost = [sum(label_cost(json.loads(f.read_text())) for f in (out / "labels").glob("*.json"))]

    def one(p: dict[str, Any]) -> str:
        path = out / "labels" / f"{p['id']}.json"
        if path.exists():
            return "kept"
        with lock:
            if cost[0] >= budget:
                return "over budget"
        try:
            request = point_call(p)["request"]
        except StalePoint:   # the game was played again during this pass: the next pass reads its points afresh
            return "stale"
        outcome, m, reasoning, usage, n, dollars = oracle.ask_label(ask, oracle.with_rules(request, rules), rules,
                                                                    need_thinking)
        with lock:
            cost[0] += dollars
        if outcome != "labelled":
            return outcome
        path.write_text(json.dumps({"message": {k: m.get(k) for k in ("role", "content", "tool_calls")},
                                    "reasoning": reasoning, "usage": usage, "from": "label", "asked": n,
                                    "spent": dollars}))
        return "labelled" if n == 1 else "labelled, asked again"

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        res = collections.Counter(pool.map(one, pts))
    done = {"points": len(pts), "outcomes": dict(res), "cost_usd": round(cost[0], 2)}
    print(json.dumps(done), flush=True)
    if res.get("over budget"):
        raise SystemExit(f"the budget (${budget}) ran out before every point was labelled")
    labelled = res["labelled"] + res["labelled, asked again"] + res["kept"]
    if labelled < min_labelled_pct / 100 * len(pts):
        raise SystemExit(f"only {labelled} of {len(pts)} points got a label, under the minimum of {min_labelled_pct}%: "
                         "a dead relay or key, or an oracle that will not think")
    return done


# -- the label set -------------------------------------------------------------------------------------------------

def student_message(label: dict[str, Any]) -> dict[str, Any]:
    """The student's target for an oracle's label: all of the oracle's prose, its reasoning and the prose of its visible
    text, as the student's reasoning, and only the action (with any tool calls) as its visible text, so that the student
    reasons in its thinking and acts in the text the harness reads (the oracle often explains an easy move in its reply
    rather than in its reasoning)."""
    prose, action = oracle.split_reply(label["message"].get("content"))
    reasoning = "\n\n".join(x for x in ((label.get("reasoning") or "").strip(), prose) if x)
    message = {"role": "assistant", "content": action, "reasoning_content": reasoning}
    if label["message"].get("tool_calls"):
        message["tool_calls"] = label["message"]["tool_calls"]
    return message


def move(message: dict[str, Any]) -> str:
    """A reply's move, for the set's counts: its last JSON action's (request, for a demonstration, or submit), else code
    for a tool call, else other."""
    acts = re.findall(r'"action"\s*:\s*"(\w+)"', message.get("content") or "")
    if acts:
        a = acts[-1]
        return "request" if a.startswith("request") or a == "demo" else ("submit" if a == "submit" else "other")
    return "code" if message.get("tool_calls") else "other"


def render_reply(tok: Any, request: dict[str, Any], message: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    """The prompt's ids and the reply's (through its end of turn) as the local chat template renders ``message`` after
    the call's messages, the reply tokenised on its own after the prompt's text, as the model writes it after the served
    prompt: rendering prompt and reply as one text merges tokens across the boundary when the reply's reasoning is empty
    ("<think>\\n" then "\\n</think>"). None when the template cannot render it, the prompt's text does not tokenise to
    :func:`continual_arc_training.calls.render_prompt_ids`' own ids, or the reply has no end of turn. Both renderings
    are local: the served ids are never compared, so a call the local template renders otherwise than the server is
    kept."""
    kwargs = request.get("chat_template_kwargs") or {}
    try:
        prompt = render_prompt_ids(tok, request)
        tools = template_tools(request.get("tools"))
        head = tok.apply_chat_template(template_messages(request["messages"]), tools=tools,
                                       add_generation_prompt=True, tokenize=False, **kwargs)
        full = tok.apply_chat_template(template_messages(request["messages"] + [message]), tools=tools,
                                       add_generation_prompt=False, tokenize=False, **kwargs)
    except Exception:  # noqa: BLE001  (a reply the template cannot render, e.g. tool arguments that are no JSON)
        return None
    if not full.startswith(head) or tok(head, add_special_tokens=False)["input_ids"] != prompt:
        return None
    reply = tok(full[len(head):], add_special_tokens=False)["input_ids"]
    if tok.eos_token_id not in reply:
        return None
    return prompt, reply[:reply.index(tok.eos_token_id) + 1]


def write_game(out: Path, n: int, calls: list[tuple[dict[str, Any], list[int], list[int], dict[str, Any]]]) -> None:
    """One game of a label set as its ``captured/u{n}.json.gz``: the calls of one source game, each its request, the
    prompt ids and the target, as a play's captured game holds a call."""
    first = calls[0][0]
    play, user = first["play"], first["user"]
    doc = {"user": n, "source_play": play, "source_user": user, "task_id": first["task_id"],
           "solved": first["solved"], "calls": []}
    for call, prompt, reply, message in calls:
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        doc["calls"].append({"call_id": f"{play}-u{user:03d}-c{call['i']:04d}", "request": call["request"],
                             "response": {"prompt_token_ids": prompt,
                                          "choices": [{"index": 0, "message": message, "finish_reason": finish,
                                                       "token_ids": reply}]}})
    (out / "captured").mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "captured" / f"u{n:03d}.json.gz", "wt") as fp:
        json.dump(doc, fp)


def source(call_id: str) -> tuple[int, int]:
    """A label set's call's place in the game it labels, from its call id (:func:`write_game`'s): the play's user and
    the call's index in that game."""
    head, i = call_id.rsplit("-c", 1)
    return int(head.rsplit("-u", 1)[1]), int(i)


_tokenizer: Any = None


def _load_tokenizer(model: str) -> None:
    global _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _label_set_game(job: tuple[Path, int, list[dict[str, Any]]]) -> collections.Counter:
    """One source game's labelled points as the set's game ``u{n}`` (:func:`write_game`), each rendered with
    :func:`render_reply` in a worker process of :func:`label_set`; its counts."""
    out, n, pts = job
    items, stats = [], collections.Counter()
    for p in pts:
        lab = json.loads((out / "labels" / f"{p['id']}.json").read_text())
        call = point_call(p)
        message = student_message(lab)
        if not oracle.makes_move(message):
            stats["no_move"] += 1
            continue
        ids = render_reply(_tokenizer, call["request"], message)
        if ids is None:
            stats["render_failed"] += 1
            stats[f"render_failed:{p['kind']}"] += 1
            continue
        items.append(({**p, "request": call["request"]}, *ids, message))
        stats[f"kind:{p['kind']}"] += 1
        stats[f"move:{move(lab['message'])}"] += 1
        stats["no_reasoning"] += not message["reasoning_content"]
    if items:
        write_game(out / "set", n, items)
    stats["examples"] = len(items)
    return stats


def label_set(out: Path, model: str, workers: int) -> dict[str, int]:
    """The labelled points of ``out/points.jsonl`` as a training set (``out/set``) and its counts (``out/set.json``),
    a point the template cannot render left out and counted. Rendering a point renders its whole conversation again, so
    each source game goes to one of ``workers`` processes; a game none of whose labels renders leaves its number
    unused."""
    by_source: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    stats: collections.Counter = collections.Counter()
    for p in map(json.loads, open(out / "points.jsonl")):
        if (out / "labels" / f"{p['id']}.json").exists():
            by_source[(p["play"], p["user"])].append(p)
        else:
            stats["unlabelled"] += 1
    jobs = [(out, n, pts) for n, (_, pts) in enumerate(sorted(by_source.items()))]
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_load_tokenizer, initargs=(model,)) as pool:
        for game_stats in pool.map(_label_set_game, jobs):
            stats.update(game_stats)
    (out / "set.json").write_text(json.dumps(stats, indent=1))
    return dict(stats)


# -- the split ---------------------------------------------------------------------------------------------------

def held_game(round_: int, user: int, every: int) -> bool:
    """Whether round ``round_``'s game of ``user`` is held out under one game in ``every``: when ``(round_ + user) %
    every == every - 1``. Alone (one user) that is every ``every``-th round's game from round ``every - 1``; with
    ``every`` users or more it is one in ``every`` of each round's games, a different user's each round, so no user's
    own stream is held out for good. A held-out game is played and scored like any other: only its labels go to the
    held-out side instead of training."""
    return (round_ + user) % every == every - 1


def write_games(docs: list[dict[str, Any]], out: Path) -> int:
    """Games as a play folder, renumbered; the number of calls (training examples) written."""
    (out / "captured").mkdir(parents=True, exist_ok=True)
    calls = 0
    for n, d in enumerate(docs):
        with gzip.open(out / "captured" / f"u{n:03d}.json.gz", "wt") as fp:
            json.dump({**d, "user": n}, fp)
        calls += len(d["calls"])
    return calls


def split(set_dir: Path, out: Path, round_: int, hold_every: int, sides: tuple[str, str]) -> dict[str, Any]:
    """A round's label set's games into a training side and a held-out side (folders named ``sides``, which training
    takes as each game's run, so a phase that trains on several rounds' sides at once names each round's apart): the
    games :func:`held_game` holds out go to the held-out side. A side with no game is written empty. ``split.json``,
    written last, says how many games and examples each side holds."""
    docs = [load(f) for f in games(set_dir)]
    held = {n for n, d in enumerate(docs) if held_game(round_, d["source_user"], hold_every)}
    stats: dict[str, Any] = {"hold_every": hold_every, "round": round_}
    train_side, held_side = sides
    for side, keep in [(train_side, lambda n: n not in held), (held_side, lambda n: n in held)]:
        picked = [d for n, d in enumerate(docs) if keep(n)]
        stats[side] = {"games": len(picked), "examples": write_games(picked, out / side)}
    (out / "split.json").write_text(json.dumps(stats, indent=1))
    return stats
