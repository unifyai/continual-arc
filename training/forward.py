"""Labelling every call of a play: an expert's reply at every moment the student reached, made into a training set
and split into a training and a held-out side. realtime.py's labeller runs ``points``, ``label``, ``label-set`` and
``split`` for the trained rows DAgger and OPSD, round by round. The other subcommands served the earlier forward-training
experiments (games cut at 15, 30 and 60 calls, then whole games, against a comparison of whole games matched on the
number of labels) and the validation plays (``game-measures``).

Subcommands (run where the plays are, with continual_arc for the true outputs and httpx for the relay):
  trim      a schedule's first N instances (the smoke test's games; on the node, with no extra package)
  held      the held-out puzzles: N of the stream's 25 tasks, drawn with a seed (the same in both runs)
  points    every call the student made in the plays (expert.agent_call), but those the stop hook answered, as
            ``OUT/points.jsonl`` in dagger.py's layout; ``--decision-only`` for the decision points alone
  label     the expert's reply at every point, through the relay, as ``OUT/labels/`` in dagger.py's layout
  spent     the dollars the labels of label folders cost (the paper's DAgger row's run budget)
  label-set the labels as a training set (dagger.py's, with a reply that has no reasoning rendered as the model writes
            it; transformers for the tokenizer)
  split     a label set's games into train-all (every puzzle but the held-out ones) and dev-all (the held-out ones), or
            one game in --hold-every (held_game) into the held-out side
  count     the training examples (calls) in play folders
  match     the comparison's games, drawn whole with a seed until their training examples reach a target
  check     a stage's early warnings: agreement with the expert, costly mistakes and code errors, by third of the game
  game-measures  one play's per-game measures as a small file (run where the play is)
  measures  the four measures and the pass rule, for validation plays or game-measures files
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import json
import os
import random
import re
import threading
from pathlib import Path
from typing import Any

import dagger
import expert
import render

THIRDS = ("early", "middle", "late")
# an error in a code call's recorded output: a traceback, or an exception's line (a syntax error has no traceback)
ERROR_LINE = re.compile(r"^\s*\w*(Error|Exception)\b", re.M)
# code habits (the expert's labels without working rules taught the student to copy grids by hand): a grid typed as
# a list literal of 40 numbers or more, or pasted as 4 lines or more of digits; loading the grids with code (c1's loader
# or a call of it); a function written and checked in the same call
GRID_LITERAL = re.compile(r"\[\s*\[[\d,\s\[\]]{80,}\]")
GRID_PASTE = re.compile(r"(?:^[ \t]*\d(?:[ ,]+\d){4,}[ \t]*$\n?){4,}", re.M)


def ipython_code(message: dict[str, Any]) -> list[str]:
    out = []
    for t in message.get("tool_calls") or []:
        if t.get("function", {}).get("name") == "ipython":
            try:
                out.append(json.loads(t["function"]["arguments"]).get("code") or "")
            except (ValueError, TypeError, AttributeError):
                out.append(str(t["function"].get("arguments") or ""))
    return out


def habits(codes: list[str]) -> dict[str, float]:
    """Shares of code calls, in percent: grids loaded with code, grids copied by hand, a function written and checked."""
    n = max(1, len(codes))
    loads = sum(("Demo pair" in c and ("re.finditer" in c or "glob" in c)) or bool(re.search(r"\bload\(\)", c)) for c in codes)
    copies = sum(bool(GRID_LITERAL.search(c) or GRID_PASTE.search(c)) for c in codes)
    checks = sum(bool(re.search(r"def \w+\(", c) and re.search(r"==|all\(", c)) for c in codes)
    return {"code_calls": len(codes), "loads_grids_pct": round(100 * loads / n, 1),
            "copies_by_hand_pct": round(100 * copies / n, 1), "writes_and_checks_pct": round(100 * checks / n, 1)}


# -- games --------------------------------------------------------------------------------------------------------------

def games(play: Path) -> list[Path]:
    return sorted((play / "captured").glob("u*.json.gz"))


def load(f: Path) -> dict[str, Any]:
    with gzip.open(f, "rt") as fp:
        return json.load(fp)


def stopped(call: dict[str, Any]) -> bool:
    """Whether the stop hook answered this call (loop.stop_proxy)."""
    return bool((call.get("response") or {}).get("stopped"))


def student_calls(doc: dict[str, Any]) -> list[int]:
    """The indexes of the student's own answered calls, in order: prime-agent's own calls and those the stop hook
    answered left out."""
    out = []
    for i, c in enumerate(doc["calls"]):
        resp = c.get("response") or {}
        if c.get("request") and resp.get("choices") and not stopped(c) \
                and expert.agent_call(c["request"].get("messages") or []):
            out.append(i)
    return out


def kind(messages: list[dict[str, Any]]) -> str:
    """A call's kind: a decision point's (dagger.decision_kind), "forced" when only finishing is left, else "loop" (in
    the middle of an action, after a tool's output or a harness note)."""
    k = dagger.decision_kind(messages)
    if k:
        return k
    last = max((j for j, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    said = " ".join(render.text_of(m.get("content")) for m in messages[last + 1:] if m.get("role") == "user")
    return "forced" if any(f in said for f in dagger.FORCED) else "loop"


def third(n: int, total: int) -> str:
    """Which third of a game the n-th (1-based) of its total calls falls in."""
    f = (n - 1) / max(1, total)
    return THIRDS[0] if f < 1 / 3 else (THIRDS[1] if f < 2 / 3 else THIRDS[2])


def code_steps(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The student's code calls with the output the game recorded for each (from the next call's tool messages): its
    place in the game and whether it shows an error. A call whose output never came back (the game ended) is left out."""
    calls = student_calls(doc)
    out = []
    for n, i in enumerate(calls):
        msg = doc["calls"][i]["response"]["choices"][0].get("message") or {}
        ids = [tc.get("id") for tc in msg.get("tool_calls") or [] if tc.get("function", {}).get("name") == "ipython"]
        if not ids or n + 1 >= len(calls):
            continue
        nxt = doc["calls"][calls[n + 1]]["request"].get("messages") or []
        for tid in ids:
            hit = [m for m in nxt if m.get("role") == "tool" and m.get("tool_call_id") == tid]
            if hit:
                text = render.text_of(hit[0].get("content"))
                out.append({"n": n + 1, "third": third(n + 1, len(calls)), "output": text,
                            "error": "Traceback (most recent call last)" in text or bool(ERROR_LINE.search(text))})
    return out


def truths(plays: list[Path]) -> dict[tuple[str, int], Any]:
    """Each game's true test output, by (play folder, user), from the play's schedule."""
    items = []
    for play in plays:
        for u, v in dagger.instances(play).items():
            items.append({**v, "id": (str(play), u)})
    return dagger.true_outputs(items)


# -- held, points, label, split, count, match ---------------------------------------------------------------------------

def cmd_trim(a: argparse.Namespace) -> None:
    """A schedule's first N instances as a schedule of its own (the smoke test's few games)."""
    s = json.loads(Path(a.schedule).expanduser().read_text())
    s["slots"] = s["slots"][:a.n]
    s["users"] = len(s["slots"])
    s["sources"] = {t: src for t, src in s["sources"].items() if t in {x["task_id"] for x in s["slots"]}}
    s["name"] = f"{s['name']}-first{a.n}"
    Path(a.out).expanduser().write_text(json.dumps(s, indent=1))
    print(json.dumps({"out": a.out, "instances": len(s["slots"])}))


def cmd_held(a: argparse.Namespace) -> None:
    import loop
    tasks = sorted({x["task_id"] for x in loop.stream_slots(a.round)})
    print(",".join(sorted(random.Random(a.seed).sample(tasks, a.n))))


def points(plays: list[Path], decision_only: bool = False) -> list[dict[str, Any]]:
    out = []
    for play in plays:
        for f in games(play):
            doc = load(f)
            calls = student_calls(doc)
            for n, i in enumerate(calls):
                msgs = doc["calls"][i]["request"].get("messages") or []
                k = kind(msgs)
                if decision_only and k in ("loop", "forced"):
                    continue
                out.append({"id": dagger.point_id(play, doc, i), "play": str(play), "user": doc["user"], "i": i,
                            "call_id": doc["calls"][i]["call_id"], "n": n + 1, "len": len(calls), "kind": k,
                            "task_id": doc["task_id"], "solved": doc.get("solved"), "played_by_expert": False})
    return out


def cmd_points(a: argparse.Namespace) -> None:
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ps = points([Path(p).expanduser().resolve() for p in a.plays], a.decision_only)
    with open(out / "points.jsonl", "w") as fp:
        for p in ps:
            fp.write(json.dumps(p) + "\n")
    print(json.dumps({"points": len(ps), "games": len({(p["play"], p["user"]) for p in ps}),
                      "kinds": dict(collections.Counter(p["kind"] for p in ps))}))


GUIDE_HEAD = "\n\n[Working rules for this harness:]\n"
# The rules close by asking for reasoning in the oracle's own words with the rules unspoken (the student never sees
# them): on one round's 380 moments this labelled 89% against 79% for "never mention them" alone, 43% of them on the
# first ask against 34%, and finished in 153 s against 547 (docs/method-search.md). The rules stay at the end
# of the last message: in the system prompt the oracle mostly stopped reasoning (45% labelled).
GUIDE_TAIL = ("\n\nBefore every move, think it through in your own words: what the grids and outputs so far show, what you "
              "will do next and why. Reason about the puzzle itself: these habits are simply how you work, so never name, "
              "number or quote them, never call them rules or guidelines, and never mention a student."
              "\n[End of working rules. Follow them in every reply, and never mention them.]")
# a reply that refers to the working rules (the student never sees them): asked again. A puzzle's own "rule" is not one.
MENTIONS_RULES = re.compile(r"(?i)\b(working[- ]rules?|rules for this harness|harness rules|rule\s*#?[1-7]\b|"
                            r"(?:the|these|my|our) rules (?:say|says|tell|state|require|above|given)|per the rules|"
                            r"following the rules|as the rules|guidelines?|(?:the|this) student)")


def mentions_rules(message: dict[str, Any], reasoning: str) -> bool:
    return bool(MENTIONS_RULES.search((reasoning or "") + "\n" + (message.get("content") or "")))


def with_guidelines(request: dict[str, Any], guidelines: str, at: str = "system") -> dict[str, Any]:
    """A copy of the call with the expert's working rules at the end of its system prompt, or with ``at="message"`` at
    the end of the last message (the student's call itself is left as it was): the rules shape the expert's moves, and
    its reply is still a reply to the student's conversation."""
    msgs = [dict(m) for m in request.get("messages") or []]
    block = GUIDE_HEAD + guidelines.strip() + GUIDE_TAIL
    if at == "message" and msgs:
        c = msgs[-1].get("content")
        msgs[-1]["content"] = (c or "") + block if isinstance(c, str) or c is None else [*c, {"type": "text", "text": block}]
    elif msgs and msgs[0].get("role") == "system":
        c = msgs[0].get("content")
        msgs[0]["content"] = (c + block) if isinstance(c, str) else [*(c or []), {"type": "text", "text": block}]
    else:
        msgs.insert(0, {"role": "system", "content": block.strip()})
    return {**request, "messages": msgs}


# a label "thinks" when its prose, its reasoning and the prose of its visible text together (dagger.split_reply), holds
# at least this many characters, all of which becomes the student's reasoning (dagger.student_message).
# GLM-5.3-flash, at any setting, sometimes answers with no reasoning at all, and a student trained on such labels stopped
# thinking, 99% of calls to 31%; on an obvious move, such as asking for a first example, it reasons in 9 or 10 tokens
# and explains the move in its visible reply instead
THINK_MIN_CHARS = 50
# --thinking: thinking switched on as hard as the hosts allow. On 20 moments where it had not thought: the template
# flag 13/20, GMICloud 15/20, a plain retry 10/20; no single setting was reliable, so --need-thinking re-asks
THINKING_ON = {"reasoning": {"enabled": True, "effort": "high"}, "extra": {"chat_template_kwargs": {"enable_thinking": True}}}


# a reply that refers to the note an earlier ask carried (with_note): asked again like one that names the rules
NOTE_LEAK = re.compile(r"(?i)\b(this note|the note|as (?:requested|instructed)|this time|previous (?:reply|answer|attempt)|"
                       r"answered (?:this|it) (?:\d+|once|twice))")


def with_note(request: dict[str, Any], note: str) -> dict[str, Any]:
    """A copy of the call with ``note`` at the end of its last message, after anything with_guidelines put there."""
    msgs = [dict(m) for m in request.get("messages") or []]
    c = msgs[-1].get("content")
    block = "\n\n[Note: " + note + "]"
    msgs[-1]["content"] = (c or "") + block if isinstance(c, str) or c is None else [*c, {"type": "text", "text": block}]
    return {**request, "messages": msgs}


def ask_label(ask: Any, request: dict[str, Any], guidelines: str | None, rule_retries: int,
              need_thinking: int) -> tuple[str, dict[str, Any], str, dict[str, Any], int, float]:
    """One point's label from ``ask`` (the expert's call), asked again when the reply names the working rules (up to
    ``rule_retries`` more times) and, with ``need_thinking`` > 0, when it has no reasoning (up to that many more times).
    A reply that makes no move, empty or prose that runs no code and gives no action ("Let me load the grids."), is
    asked again too, within the reasoning's asks: it would teach the student to spend a turn on nothing (1.5% of one
    run's training targets, 12% of one round's, against 0.3% to 0.4% in the runs labelled without the oracle's nudge
    to reason first). Each ask after the first carries a note saying what
    was wrong and how often (with_note), and asking for a reply as
    if for the first time that never mentions it, since the reply becomes the student's target; a reply that does
    mention the note is asked again like one that names the rules. Told what was wrong, the oracle labelled 96% of one
    round's 380 moments against 89% when merely asked again, 87% within three asks against 73%, for a third less spend
    (docs/method-search.md). Returns the outcome (labelled; empty; no move; failed; mentions the rules; no
    thinking), the
    reply, its reasoning, the last usage, the number of asks and the dollars spent on all of them."""
    rules_left, think_left, n, spent, unreasoned = (rule_retries if guidelines else 0), need_thinking, 0, 0.0, 0
    asked = request
    while True:
        n += 1
        try:
            answer = ask(asked)
        except Exception:  # noqa: BLE001  (a point the expert does not answer stays unlabelled)
            return "failed", {}, "", {}, n, spent
        m = (answer.get("choices") or [{}])[0].get("message") or {}
        reasoning = next((m[k] for k in ("reasoning", "reasoning_content") if isinstance(m.get(k), str)), "")
        usage = answer.get("usage") or {}
        spent += usage.get("cost") or 0.0
        if not m.get("tool_calls") and not dagger.split_reply(m.get("content"))[1]:
            empty = not (m.get("content") or "").strip()
            if need_thinking and think_left:
                think_left -= 1
                said = "was empty" if empty else "made no move: it ran no code and gave no action"
                asked = with_note(request, f"your last reply {said}. Think it through in your own words, then reply "
                                           "with your move as if for the first time, never mentioning this note.")
                continue
            return "empty" if empty else "no move", m, reasoning, usage, n, spent
        text = (reasoning or "") + "\n" + (m.get("content") or "")
        named = (MENTIONS_RULES.search(text) if guidelines else None) or (NOTE_LEAK.search(text) if n > 1 else None)
        if named:
            if rules_left:
                rules_left -= 1
                asked = with_note(request, f'your last answer named your working rules or this note ("{named.group(0)}"). '
                                           "Answer again, reasoning about the puzzle itself in your own words: never "
                                           "name, number or quote the rules, and reply as if for the first time, never "
                                           "mentioning this note.")
                continue
            return "mentions the rules", m, reasoning, usage, n, spent
        if need_thinking and len(reasoning.strip()) + len(dagger.split_reply(m.get("content"))[0]) < THINK_MIN_CHARS:
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


def label_cost(label: dict[str, Any]) -> float:
    """A saved label's dollars: every ask's (``spent``), else, for a label saved without that field, the last ask's."""
    return label["spent"] if "spent" in label else (label.get("usage") or {}).get("cost") or 0.0


def cmd_label(a: argparse.Namespace) -> None:
    """dagger.cmd_label for every point (not only decision points): the expert reads the call exactly as the student did
    (with ``--guidelines``, its working rules added to the system prompt), and its reply goes to OUT/labels/ID.json;
    resumable, and no new request once ``--budget`` dollars are spent."""
    out = Path(a.out).expanduser()
    (out / "labels").mkdir(parents=True, exist_ok=True)
    pts = [json.loads(line) for line in open(out / "points.jsonl")]
    if a.limit:
        pts = random.Random(a.seed).sample(pts, min(a.limit, len(pts)))
    guidelines = Path(a.guidelines).expanduser().read_text() if a.guidelines else None
    reasoning = json.loads(a.reasoning) if a.reasoning else (THINKING_ON["reasoning"] if a.thinking else None)
    e = expert.Expert(a.expert_url, a.expert, effort=a.effort or None, reasoning=reasoning,
                      extra=THINKING_ON["extra"] if a.thinking else None)
    lock = threading.Lock()
    spent = [sum(label_cost(json.loads(f.read_text())) for f in (out / "labels").glob("*.json"))]

    def one(p: dict[str, Any]) -> str:
        path = out / "labels" / f"{p['id']}.json"
        if path.exists():
            return "kept"
        with lock:
            if spent[0] >= a.budget:
                return "over budget"
        try:
            request = dagger.point_call(p)["request"]
        except dagger.StalePoint:   # the game was replayed during this pass: the next pass reads its points afresh
            return "stale"
        asked = with_guidelines(request, guidelines, a.guidelines_at) if guidelines else request
        outcome, m, reasoning, usage, n, cost = ask_label(e.ask, asked, guidelines, a.rule_retries, a.need_thinking)
        with lock:
            spent[0] += cost
        if outcome != "labelled":
            return outcome
        path.write_text(json.dumps({"message": {k: m.get(k) for k in ("role", "content", "tool_calls")},
                                    "reasoning": reasoning, "usage": usage, "from": "label", "asked": n,
                                    "spent": cost}))
        return "labelled" if n == 1 else "labelled, asked again"

    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        res = collections.Counter(pool.map(one, pts))
    print(json.dumps({"points": len(pts), "outcomes": dict(res), "cost_usd": round(spent[0], 2)}))
    if res.get("over budget"):
        raise SystemExit(f"the budget (${a.budget}) ran out before every point was labelled")
    labelled = res["labelled"] + res["labelled, asked again"] + res["kept"]
    if labelled < a.min_labelled_pct / 100 * len(pts):
        raise SystemExit(f"only {labelled} of {len(pts)} points got a label, under --min-labelled-pct "
                         f"{a.min_labelled_pct}: a dead relay or key, or an expert that will not think")


def spent(label_dirs: list[Path]) -> float:
    """The dollars every label under ``label_dirs`` (each a label command's OUT) cost (:func:`label_cost`)."""
    return sum(label_cost(json.loads(f.read_text())) for d in label_dirs for f in (d / "labels").glob("*.json"))


def cmd_spent(a: argparse.Namespace) -> None:
    print(round(spent([Path(d).expanduser() for d in a.outs]), 2))


def render_reply(tok: Any, request: dict[str, Any], message: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    """dagger.render_target with the reply tokenised on its own after the prompt's text, as the model writes it after the
    served prompt: rendering prompt and reply as one text merges tokens across the boundary when the reply's reasoning
    is empty ("<think>\\n" then "\\n</think>"; the expert wrote no reasoning in 86% of its replies inside a stretch of
    coding, all of which dagger's rendering dropped). None when the template cannot render it, the prompt's text does not tokenise to
    ask_sd.calls.render_prompt_ids' own ids, or the reply has no end of turn. Both prompt renderings are local: the served
    prompt ids are never compared, so a call the local template renders differently from the server is kept."""
    from ask_sd.calls import ToolSchema, render_prompt_ids, template_messages, template_tools
    schema = ToolSchema[dagger.TOOL_SCHEMA_NAME]
    kwargs = request.get("chat_template_kwargs") or {}
    try:
        prompt = render_prompt_ids(tok, request, join_text_parts=True, tool_schema=schema)
        tools = template_tools(request.get("tools"), schema)
        head = tok.apply_chat_template(template_messages(request["messages"], join_text_parts=True), tools=tools,
                                       add_generation_prompt=True, tokenize=False, **kwargs)
        full = tok.apply_chat_template(template_messages(request["messages"] + [message], join_text_parts=True),
                                       tools=tools, add_generation_prompt=False, tokenize=False, **kwargs)
    except Exception:  # noqa: BLE001  (a reply the template cannot render, e.g. tool arguments that are no JSON)
        return None
    if not full.startswith(head) or tok(head, add_special_tokens=False)["input_ids"] != prompt:
        return None
    reply = tok(full[len(head):], add_special_tokens=False)["input_ids"]
    if tok.eos_token_id not in reply:
        return None
    return prompt, reply[:reply.index(tok.eos_token_id) + 1]


_TOKENIZER: Any = None


def _load_tokenizer(model: str) -> None:
    global _TOKENIZER
    _TOKENIZER = dagger.tokenizer(model)


def _label_set_game(job: tuple[Path, int, list[dict[str, Any]]]) -> collections.Counter:
    """One source game's labelled points as the training set's file ``u{n}`` (:func:`dagger.write_game`), each
    rendered with :func:`render_reply` in a worker process of :func:`cmd_label_set`; its counts."""
    out, n, pts = job
    items, stats = [], collections.Counter()
    for p in pts:
        label = json.loads((out / "labels" / f"{p['id']}.json").read_text())
        call = dagger.point_call(p)
        message = dagger.student_message(label)
        if not dagger.makes_move(message):   # it would teach the student to spend a turn on nothing
            stats["no_move"] += 1
            continue
        ids = render_reply(_TOKENIZER, call["request"], message)
        if ids is None:
            stats["render_failed"] += 1
            stats[f"render_failed:{p['kind']}"] += 1
            continue
        items.append(({**p, "request": call["request"]}, *ids, message))
        stats[f"kind:{p['kind']}"] += 1
        stats[f"move:{dagger.coarse(label['message'])}"] += 1
        stats["no_reasoning"] += not message["reasoning_content"]
    if items:
        dagger.write_game(out / "set", n, items)
    stats["examples"] = len(items)
    return stats


def cmd_label_set(a: argparse.Namespace) -> None:
    """dagger.cmd_label_set with :func:`render_reply`: the labelled points as a training set (OUT/set, dagger.write_set's
    play folder) and its counts (OUT/set.json), a point the template cannot render left out and counted. Rendering a
    point re-renders its whole conversation, so a source game goes to each of ``--workers`` processes (29,000 points
    took an hour in one); a game none of whose labels renders leaves its number unused."""
    out = Path(a.out).expanduser()
    games: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    stats: collections.Counter = collections.Counter()
    for p in map(json.loads, open(out / "points.jsonl")):
        if (out / "labels" / f"{p['id']}.json").exists():
            games[(p["play"], p["user"])].append(p)
        else:
            stats["unlabelled"] += 1
    jobs = [(out, n, pts) for n, (_, pts) in enumerate(sorted(games.items()))]
    with concurrent.futures.ProcessPoolExecutor(a.workers, initializer=_load_tokenizer, initargs=(a.model,)) as pool:
        for game_stats in pool.map(_label_set_game, jobs):
            stats.update(game_stats)
    (out / "set.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(dict(stats)))


def write_games(docs: list[dict[str, Any]], out: Path) -> int:
    """Games as a play folder, renumbered; the number of calls (training examples) written."""
    (out / "captured").mkdir(parents=True, exist_ok=True)
    calls = 0
    for n, d in enumerate(docs):
        with gzip.open(out / "captured" / f"u{n:03d}.json.gz", "wt") as fp:
            json.dump({**d, "user": n}, fp)
        calls += len(d["calls"])
    return calls


def held_game(round_: int, user: int, every: int) -> bool:
    """Whether round ``round_``'s game of ``user`` is held out under one game in ``every``: when ``(round_ + user) %
    every == every - 1``. Alone (one user) that is every ``every``-th round's game from round ``every - 1``; with
    ``every`` users or more it is one in ``every`` of each round's games, a different user's each round, so no user's
    own stream is held out for good. A held-out game is played and scored like any other: only its labels go to the
    held-out side instead of training (realtime.py's labeller)."""
    return (round_ + user) % every == every - 1


def cmd_split(a: argparse.Namespace) -> None:
    """A label set's games into a training side and a held-out side (``--sides``' names): the games of ``--held``'s
    puzzles held out (the dev loop's early stopping, by puzzle), or, with ``--hold-every`` N and ``--round`` R, the
    games :func:`held_game` holds out (the real-time DAgger row, each round's label set split as it is made). The
    sides' names are the folders' own, which train.load_calls takes as each game's run, so a run that trains on
    several rounds' sides at once names each round's apart. A side with no game is written empty."""
    docs = [load(f) for f in games(Path(a.set).expanduser())]
    out = Path(a.out).expanduser()
    if a.held:
        held = set(a.held.split(","))
        dev = {n for n, d in enumerate(docs) if d["task_id"] in held}
        stats: dict[str, Any] = {"held": sorted(held)}
    else:
        dev = {n for n, d in enumerate(docs) if held_game(a.round, d["source_user"], a.hold_every)}
        stats = {"hold_every": a.hold_every, "round": a.round}
    train_side, dev_side = a.sides.split(",")
    for side, keep in [(train_side, lambda n: n not in dev), (dev_side, lambda n: n in dev)]:
        picked = [d for n, d in enumerate(docs) if keep(n)]
        stats[side] = {"games": len(picked), "examples": write_games(picked, out / side)}
    (out / "split.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


def count(folders: list[Path]) -> int:
    return sum(len(load(f)["calls"]) for folder in folders for f in games(folder))


def cmd_count(a: argparse.Namespace) -> None:
    print(count([Path(p).expanduser() for p in a.folders]))


def cmd_match(a: argparse.Namespace) -> None:
    """The comparison's training games: its label set's games outside the held-out puzzles, in a seeded random order,
    taken whole until their examples reach ``--target`` (the staged pool's); dev-all is every held-out game."""
    held = set(a.held.split(","))
    docs = [load(f) for f in games(Path(a.set).expanduser())]
    rest = [d for d in docs if d["task_id"] not in held]
    random.Random(a.seed).shuffle(rest)
    picked, n = [], 0
    for d in rest:
        if n >= a.target:
            break
        picked.append(d)
        n += len(d["calls"])
    if n < a.target * (1 - a.tolerance):
        raise SystemExit(f"only {n} examples for a target of {a.target}: play more comparison games")
    out = Path(a.out).expanduser()
    stats = {"target": a.target, "held": sorted(held),
             "train-all": {"games": len(picked), "examples": write_games(picked, out / "train-all")},
             "dev-all": {"games": sum(d["task_id"] in held for d in docs),
                         "examples": write_games([d for d in docs if d["task_id"] in held], out / "dev-all")},
             "unused_games": len(rest) - len(picked)}
    (out / "split.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


# -- the measures --------------------------------------------------------------------------------------------------------

def costly(own: dict[str, Any], exp: dict[str, Any], k: str, truth: Any) -> str | None:
    """Whether the student's move at a decision point is a costly mistake against the expert's (where-students-disagree):
    a wrong grid submitted where the expert would not submit, the right grid the expert would submit left unsent, or code
    run at a "decide now" moment."""
    o, e = dagger.coarse(own), dagger.coarse(exp)
    if o == "submit" and e != "submit" and dagger.grid_of(own.get("content")) != truth:
        return "wrong guess"
    if e == "submit" and o != "submit" and truth is not None and dagger.grid_of(exp.get("content")) == truth:
        return "missed answer"
    if k == "budget" and o == "code" and e != "code":
        return "codes at decide now"
    return None


def game_measures(doc: dict[str, Any], truth: Any, labels: dict[int, dict[str, Any]] | None = None,
                  kinds: dict[int, str] | None = None) -> dict[str, Any]:
    """One game's counts for the four measures, each as numerator and denominator, overall and by third (the costly one
    only where ``labels``, the expert's messages by call index, are given)."""
    calls = student_calls(doc)
    m: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    steps = code_steps(doc)
    for s in steps:
        for key in ("code_errors", f"code_errors:{s['third']}"):
            m[key][0] += s["error"]
            m[key][1] += 1
    for n, i in enumerate(calls):
        msg = doc["calls"][i]["response"]["choices"][0].get("message") or {}
        t = third(n + 1, len(calls))
        if dagger.coarse(msg) == "submit":
            right = truth is not None and dagger.grid_of(msg.get("content")) == truth
            for key in ("right_submissions", f"right_submissions:{t}"):
                m[key][0] += right
                m[key][1] += 1
        if labels is not None and i in labels:
            c = costly(msg, labels[i], (kinds or {}).get(i, ""), truth)
            for key in ("costly", f"costly:{t}"):
                m[key][0] += c is not None
                m[key][1] += 1
    printed = truth is not None and any(truth in dagger.grids_in(s["output"]) for s in steps)
    m["unsubmitted"] = [float(printed and not doc.get("solved")), 1.0]
    return {"task_id": doc["task_id"], "solved": bool(doc.get("solved")), "m": dict(m)}


def run_measures(play: Path, labels_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Every game of a play, by task: its measures (game_measures), with the expert's labels at its decision points from
    ``labels_dir`` (forward.py points --decision-only + label) when given."""
    t = truths([play])
    by_game: dict[int, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    kinds: dict[int, dict[int, str]] = collections.defaultdict(dict)
    if labels_dir is not None:
        for line in open(labels_dir / "points.jsonl"):
            p = json.loads(line)
            f = labels_dir / "labels" / f"{p['id']}.json"
            if f.exists():
                by_game[p["user"]][p["i"]] = json.loads(f.read_text())["message"]
                kinds[p["user"]][p["i"]] = p["kind"]
    out = {}
    for f in games(play):
        doc = load(f)
        out[doc["task_id"]] = game_measures(doc, t.get((str(play), doc["user"])),
                                            by_game.get(doc["user"], {}) if labels_dir is not None else None,
                                            kinds.get(doc["user"], {}))
    return out


# measure -> whether a higher value is better
MEASURES = {"code_errors": False, "right_submissions": True, "costly": False, "unsubmitted": False}


def ratio(games_: list[dict[str, Any]], key: str) -> float | None:
    num = sum(g["m"].get(key, [0, 0])[0] for g in games_)
    den = sum(g["m"].get(key, [0, 0])[1] for g in games_)
    return num / den if den else None


def paired(a: dict[str, dict], b: dict[str, dict], key: str, n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """A's value minus B's over the puzzles both played, with a 95% paired bootstrap interval (puzzles resampled)."""
    tasks = sorted(set(a) & set(b))
    va, vb = ratio([a[t] for t in tasks], key), ratio([b[t] for t in tasks], key)
    if va is None or vb is None:
        return {"a": va, "b": vb, "diff": None, "lo": None, "hi": None, "puzzles": len(tasks)}
    rng, diffs = random.Random(seed), []
    for _ in range(n):
        pick = [rng.choice(tasks) for _ in tasks]
        x, y = ratio([a[t] for t in pick], key), ratio([b[t] for t in pick], key)
        if x is not None and y is not None:
            diffs.append(x - y)
    diffs.sort()
    return {"a": round(va, 4), "b": round(vb, 4), "diff": round(va - vb, 4),
            "lo": round(diffs[int(0.025 * len(diffs))], 4), "hi": round(diffs[int(0.975 * len(diffs)) - 1], 4),
            "puzzles": len(tasks)}


def verdict(a: dict[str, dict], b: dict[str, dict]) -> dict[str, Any]:
    """The pass rule for A against B: A better on at least 3 of the 4 measures (each interval clear of zero on the good
    side), and A not solving clearly fewer puzzles (exact McNemar p < 0.05 against A)."""
    import loop
    res, better = {}, 0
    for key, higher in MEASURES.items():
        r = paired(a, b, key)
        r["better"] = r["diff"] is not None and ((r["lo"] > 0) if higher else (r["hi"] < 0))
        better += r["better"]
        res[key] = r
        res[key + ":thirds"] = {t: paired(a, b, f"{key}:{t}") for t in THIRDS} if key != "unsubmitted" else None
    tasks = sorted(set(a) & set(b))
    only_a = sum(a[t]["solved"] and not b[t]["solved"] for t in tasks)
    only_b = sum(b[t]["solved"] and not a[t]["solved"] for t in tasks)
    p = loop.mcnemar_p(only_a, only_b)
    clearly_fewer = only_b > only_a and p < 0.05
    return {"measures": res, "measures_better": better, "solved": [sum(a[t]["solved"] for t in tasks),
            sum(b[t]["solved"] for t in tasks)], "only_a": only_a, "only_b": only_b, "mcnemar_p": round(p, 4),
            "clearly_fewer_solves": clearly_fewer, "passes": better >= 3 and not clearly_fewer}


def cmd_game_measures(a: argparse.Namespace) -> None:
    """One play's per-game measures (run_measures) as a small JSON file: made where the play is (on its node), so that
    only this file has to travel for ``measures``."""
    res = run_measures(Path(a.play).expanduser().resolve(), Path(a.labels).expanduser() if a.labels else None)
    Path(a.out).expanduser().write_text(json.dumps(res))
    print(json.dumps({"games": len(res), "solved": sum(g["solved"] for g in res.values())}))


def cmd_measures(a: argparse.Namespace) -> None:
    runs = {}
    for spec in a.runs:
        name, rest = spec.split("=", 1)
        if rest.endswith(".json") and Path(rest).expanduser().is_file():   # game-measures' output
            runs[name] = json.loads(Path(rest).expanduser().read_text())
            continue
        play, _, labels = rest.partition("@")
        runs[name] = run_measures(Path(play).expanduser().resolve(), Path(labels).expanduser() if labels else None)
    out = {"runs": {n: {k: ratio(list(g.values()), k) for k in MEASURES} | {"solved": sum(x["solved"] for x in g.values()),
                        "games": len(g)} for n, g in runs.items()}, "pairs": {}}
    for pair in a.pairs:
        x, y = pair.split(":")
        out["pairs"][f"{x} vs {y}"] = verdict(runs[x], runs[y])
    Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
    lines = ["| Pair | " + " | ".join(MEASURES) + " | Solved | Measures better | Passes |", "|" + "---|" * (len(MEASURES) + 4)]
    for name, v in out["pairs"].items():
        cells = []
        for key in MEASURES:
            r = v["measures"][key]
            cells.append("—" if r["diff"] is None else f"{r['a']:.3f} vs {r['b']:.3f} ({r['lo']:+.3f}..{r['hi']:+.3f})"
                         + (" **better**" if r["better"] else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {v['solved'][0]} vs {v['solved'][1]} (p {v['mcnemar_p']}) | "
                     f"{v['measures_better']} of 4 | {'yes' if v['passes'] else 'no'} |")
    print("\n".join(lines))


def cmd_check(a: argparse.Namespace) -> None:
    """A stage's early warnings from its own games and labels: the share of points where the student's move matches the
    expert's, costly mistakes per labelled decision point and code errors per step, each by third of the game."""
    labels_dir = Path(a.labels).expanduser()
    plays = sorted({Path(json.loads(line)["play"]) for line in open(labels_dir / "points.jsonl")})
    t = truths(plays)
    agree, cost, errs = (collections.defaultdict(lambda: [0, 0]) for _ in range(3))
    docs: dict[tuple[str, int], dict[str, Any]] = {}
    for line in open(labels_dir / "points.jsonl"):
        p = json.loads(line)
        f = labels_dir / "labels" / f"{p['id']}.json"
        if not f.exists():
            continue
        key = (p["play"], p["user"])
        if key not in docs:
            docs[key] = load(Path(p["play"]) / "captured" / f"u{p['user']:03d}.json.gz")
        own = docs[key]["calls"][p["i"]]["response"]["choices"][0]["message"]
        exp = json.loads(f.read_text())["message"]
        th = third(p["n"], p["len"])
        for k in ("all", th):
            agree[k][0] += dagger.coarse(own) == dagger.coarse(exp)
            agree[k][1] += 1
            if p["kind"] not in ("loop", "forced"):
                cost[k][0] += costly(own, exp, p["kind"], t.get(key)) is not None
                cost[k][1] += 1
    student_codes, expert_codes = [], []
    for doc in docs.values():
        for s in code_steps(doc):
            for k in ("all", s["third"]):
                errs[k][0] += s["error"]
                errs[k][1] += 1
        for i in student_calls(doc):
            student_codes += ipython_code(doc["calls"][i]["response"]["choices"][0].get("message") or {})
    for f in (labels_dir / "labels").glob("*.json"):
        expert_codes += ipython_code(json.loads(f.read_text())["message"])
    pct = lambda d: {k: round(100 * v[0] / v[1], 1) for k, v in d.items() if v[1]}  # noqa: E731
    print(json.dumps({"games": len(docs), "same_move_pct": pct(agree), "costly_per_decision_pct": pct(cost),
                      "code_errors_pct": pct(errs), "student_habits": habits(student_codes),
                      "expert_habits": habits(expert_codes)}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("trim")
    s.add_argument("schedule")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_trim)
    s = sub.add_parser("held")
    s.add_argument("--round", type=int, default=22)
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_held)
    s = sub.add_parser("points")
    s.add_argument("plays", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--decision-only", action="store_true")
    s.set_defaults(fn=cmd_points)
    s = sub.add_parser("label")
    s.add_argument("--out", required=True)
    s.add_argument("--expert", default="z-ai/glm-5.3-flash")
    s.add_argument("--expert-url", default="http://127.0.0.1:18230/v1")
    s.add_argument("--effort", default="medium")
    s.add_argument("--reasoning", default=None,
                   help='the request\'s whole reasoning setting as JSON (e.g. \'{"max_tokens": 4000}\'), in place of --effort')
    s.add_argument("--budget", type=float, required=True)
    s.add_argument("--workers", type=int, default=48)
    s.add_argument("--guidelines", default=None, help="the expert's working rules, added to its prompt")
    s.add_argument("--guidelines-at", choices=["system", "message"], default="system",
                   help="where the rules go: the end of the system prompt, or of the last message")
    s.add_argument("--rule-retries", type=int, default=3, help="with --guidelines: asks again when a reply mentions them")
    s.add_argument("--thinking", action="store_true",
                   help="thinking switched on as hard as the hosts allow (reasoning enabled at high effort, and the "
                        "chat template's enable_thinking)")
    s.add_argument("--need-thinking", type=int, default=0,
                   help="asks again, up to this many more times, when a reply has no reasoning; a point that never "
                        "thinks gets no label ('no thinking'). 0: any reply is kept")
    s.add_argument("--limit", type=int, default=0, help="label only this many points, drawn with --seed (a test)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-labelled-pct", type=float, default=0, dest="min_labelled_pct",
                   help="fail when fewer points than this share got a label (kept from an earlier run included)")
    s.set_defaults(fn=cmd_label)
    s = sub.add_parser("spent", help="the dollars every label of these label folders cost")
    s.add_argument("outs", nargs="+", help="label commands' --out folders")
    s.set_defaults(fn=cmd_spent)
    s = sub.add_parser("label-set")
    s.add_argument("--out", required=True, help="the label command's folder")
    s.add_argument("--model", default="Qwen/Qwen3.5-9B")
    s.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1),
                   help="processes rendering games at once")
    s.set_defaults(fn=cmd_label_set)
    s = sub.add_parser("split")
    s.add_argument("set")
    held_or_frac = s.add_mutually_exclusive_group(required=True)
    held_or_frac.add_argument("--held", help="the held-out puzzles' task ids, comma-separated")
    held_or_frac.add_argument("--hold-every", type=int, dest="hold_every",
                              help="hold out one game in this many (held_game), with --round")
    s.add_argument("--round", type=int, default=0, help="--hold-every's round: the label set's own")
    s.add_argument("--sides", default="train-all,dev-all", help="the two sides' folder names: training,held-out")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_split)
    s = sub.add_parser("count")
    s.add_argument("folders", nargs="+")
    s.set_defaults(fn=cmd_count)
    s = sub.add_parser("match")
    s.add_argument("set")
    s.add_argument("--target", type=int, required=True)
    s.add_argument("--held", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tolerance", type=float, default=0.02)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_match)
    s = sub.add_parser("check")
    s.add_argument("--labels", required=True)
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("game-measures")
    s.add_argument("play")
    s.add_argument("--labels", default=None, help="the play's decision-point labels (points --decision-only + label)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_game_measures)
    s = sub.add_parser("measures")
    s.add_argument("--runs", nargs="+", required=True,
                   help="NAME=PLAY[@LABELS] (LABELS: decision-point labels), or NAME=FILE.json from game-measures")
    s.add_argument("--pairs", nargs="+", required=True, help="A:B, A's measures against B's")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_measures)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
