"""DAgger's training sets and report, from plays in which an expert played the student's call right after each
failure (loop.py play ``--expert``, expert.py). Every call the expert played is one training state, the same states
in every set:

- ``raw``: the expert's reply as the target, its reasoning as the student's reasoning, then its visible text and tool
  calls;
- ``rewrite``: the expert's action in the student's own words: the student, shown the expert's reply as a decided next
  step (a note on its last message), writes its reasoning toward it, and the target is that reasoning, then the
  expert's visible text and tool calls unchanged. The action is the expert's exactly, so the game after it is the same;
  a reasoning that cites the note is drawn again, and a state whose every draw cites it leaves every set;
- ``control``: the same states for on-policy distillation toward the notes teacher with no expert (train.py's sdft
  objective draws its own replies; the reply stored is the rewrite's target, which it does not read).

Each set is a play folder train.py reads (``captured/u*.json.gz``): the call's request as captured, the prompt's ids as
the local chat template renders it (ask_sd.calls.render_prompt_ids, never compared with the served ids: train.py's
build_items compares them for distillation items only, and its sft objective trains on these local ids as written) and
the target's ids, the target rendered as the chat template renders an assistant reply
after that prompt (this rendering gives 245 of 260 of b6's own served replies token for token, the rest the same text
split into other tokens).

The trained rows' labeller (realtime.py, through forward.py) uses this module's labelling pieces: a point's id and call
(:func:`point_id`, :func:`point_call`), a decision point's kind (:func:`decision_kind`), a label as the student's
target (:func:`student_message`, :func:`makes_move`), a training set's game (:func:`write_game`) and the replay dose
(:func:`replay_plan`). The three sets, the takeover plays' labels and the report served the earlier DAgger experiments.

Subcommands (run where the plays are, with the model server reachable at --server):
  sets PLAY... --out DIR   the three sets from the plays' expert calls
  report                   the arms' validation plays against b6's, and the expert's calls
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import json
import random
import re
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import loop  # noqa: E402
import render  # noqa: E402

TOOL_SCHEMA_NAME = "vllm_0_30"   # train.py's TOOL_SCHEMA: the loop's server is vLLM 0.30
REWRITE_MAX_TOKENS = 8192        # the student's reasoning toward the decided step
REWRITE_TRIES = 6
# the note is no message of the game's: worded as the next reply "already decided", 29% of the student's
# reasonings spoke of the user (16% of its own: the game's feedback comes as the user's), many of them the note as the
# user's suggestion ("the user is providing the next action")
NOTE_HEAD = ("\n\n[Writing exercise, not part of the game and from no one in it: the reply below is the one you will "
             "give next, exactly:\n<<<\n")
NOTE_TAIL = ("\n>>>\nThink it through as if you had reached this step on your own, from the game's messages only, in "
             "your own words. Your thinking must never mention this exercise, a chosen, suggested or given step or "
             "reply, or anyone telling you what to do.]")
# a reasoning that reads as written from the note, not toward the step: these words at once, the rest by the judge
CITES_NOTE = re.compile(r"\b(this note|the note|this exercise|already decided|decided reply|decided step|"
                        r"I(?:'m| am) told to|instructed to|the reply is given|given reply|suggested (step|reply|action))\b",
                        re.I)
JUDGE = "z-ai/glm-5.3"
JUDGE_PROVIDERS = {"order": ["fireworks", "parasail", "gmicloud"], "allow_fallbacks": False}
JUDGE_PROMPT = (
    "An agent in a puzzle game was shown, outside the game, the exact reply it must give next, and asked to write the "
    "thinking it would have written had it chosen that reply itself. Below are the game's last message to the agent "
    "and the thinking it wrote.\n\nDoes the thinking refer to having been shown or given its next step: a note, a "
    "hint, an exercise, a provided, suggested, chosen or decided step, reply, action or code, or someone (the user, "
    "an instruction) proposing or telling it what to do next? The game's own feedback (\"my submission was "
    "INCORRECT\", \"the user says it was wrong\", the demos, the game's standing rules such as one action JSON on the "
    "last line) does not count.\n\nThe game's last message:\n<<<\n{feedback}\n>>>\n\nThe thinking:\n<<<\n{thinking}\n"
    ">>>\n\nAnswer with one word: YES if it refers to the given step, NO if it does not.")


def tokenizer(model: str = loop.MODEL) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def expert_calls(play: Path) -> list[dict[str, Any]]:
    """Every call of a play that the expert played, with its game: the request as captured and the served reply."""
    out = []
    for f in sorted((play / "captured").glob("u*.json.gz")):
        with gzip.open(f, "rt") as fp:
            doc = json.load(fp)
        for i, c in enumerate(doc["calls"]):
            resp = c.get("response") or {}
            if c.get("request") and isinstance(resp.get("expert"), dict):
                out.append({"play": play.name, "user": doc["user"], "task_id": doc["task_id"], "solved": doc.get("solved"),
                            "i": i, "request": c["request"], "response": resp})
    return out


def step_text(message: dict[str, Any]) -> str:
    """A reply as the note shows it: its visible text, then each tool call's code (or arguments)."""
    parts = [message.get("content") or ""]
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = None
        code = args.get("code") if isinstance(args, dict) else None
        parts.append(f"[{fn.get('name')} call]\n{code}" if isinstance(code, str) else
                     f"[{fn.get('name')} call] {fn.get('arguments')}")
    return "\n".join(p for p in parts if p.strip())


def target_message(served: dict[str, Any], reasoning: str) -> dict[str, Any]:
    """The assistant reply to train on: ``reasoning``, then the served reply's visible text and tool calls."""
    m = served["choices"][0]["message"]
    out: dict[str, Any] = {"role": "assistant", "content": m.get("content") or "", "reasoning_content": reasoning}
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    return out


def render_target(tok: Any, request: dict[str, Any], message: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    """The prompt's ids and the reply's (through its end-of-turn token) as the chat template renders ``message`` after
    the call's messages; None when the template cannot render it or the rendering does not continue the prompt."""
    from ask_sd.calls import ToolSchema, render_prompt_ids, template_messages, template_tools
    schema = ToolSchema[TOOL_SCHEMA_NAME]
    try:
        prompt = render_prompt_ids(tok, request, join_text_parts=True, tool_schema=schema)
        full = tok.apply_chat_template(template_messages(request["messages"] + [message], join_text_parts=True),
                                       tools=template_tools(request.get("tools"), schema), add_generation_prompt=False,
                                       tokenize=False, **(request.get("chat_template_kwargs") or {}))
    except Exception:  # noqa: BLE001  (a reply the template cannot render, e.g. tool arguments that are no JSON)
        return None
    ids = tok(full, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    if ids[:len(prompt)] != prompt or eos not in ids[len(prompt):]:
        return None
    reply = ids[len(prompt):]
    return prompt, reply[:reply.index(eos) + 1]


_judge = None


def judge() -> Any:
    """The judge's OpenRouter endpoint (the key read as probe.py's oracle reads it)."""
    global _judge
    if _judge is None:
        from ask_sd.thoughts import Endpoint
        _judge = Endpoint("https://openrouter.ai/api/v1", JUDGE, "OPENROUTER_API_KEY", routing=JUDGE_PROVIDERS,
                          effort="low", retries=6)
    return _judge


def last_message(request: dict[str, Any]) -> str:
    """The text of the call's last message: the game's newest words to the agent (its feedback)."""
    content = (request.get("messages") or [{}])[-1].get("content")
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def cites_given_step(request: dict[str, Any], thinking: str) -> bool:
    """Whether ``thinking`` refers to the step it was given (the judge's YES; no clear answer counts as yes)."""
    prompt = JUDGE_PROMPT.format(feedback=last_message(request)[-3000:], thinking=thinking[:12000])
    answer = (judge().ask([{"role": "user", "content": prompt}], max_tokens=4000, temperature=0.0) or "").strip().upper()
    return not answer.endswith("NO") and not answer.startswith("NO")


def rewrite(server: Any, call: dict[str, Any], student: str, rng: random.Random) -> tuple[str | None, int, int]:
    """The student's reasoning toward the expert's reply (the module's docstring), how many draws it took, and how many
    the judge turned down for referring to the given step; None when no draw passed (each cited the note, ran out of
    room or came back empty)."""
    from ask_sd.calls import with_note
    request = call["request"]
    note = NOTE_HEAD + step_text(call["response"]["choices"][0]["message"]) + NOTE_TAIL
    flagged = 0
    for k in range(1, REWRITE_TRIES + 1):
        payload = {key: v for key, v in request.items() if key not in ("stream", "stream_options", "store",
                                                                        "max_completion_tokens", "return_token_ids")}
        # a seed per draw: the loop's server gives unseeded requests one stream (the same request, the same reply)
        payload.update({"messages": with_note(request["messages"], note), "model": student, "stream": False, "n": 1,
                        "max_tokens": REWRITE_MAX_TOKENS, "seed": rng.randrange(2 ** 31)})
        try:
            out = server.chat(payload)
        except Exception as error:  # noqa: BLE001  (a failed draw is drawn again)
            print(f"a rewrite draw failed: {type(error).__name__}: {error}", flush=True)
            continue
        choice = (out or {}).get("choices", [{}])[0]
        m = choice.get("message") or {}
        reasoning = (m.get("reasoning") or m.get("reasoning_content") or "").strip()
        if choice.get("finish_reason") == "length" or not reasoning or CITES_NOTE.search(reasoning):
            continue
        if cites_given_step(request, reasoning):
            flagged += 1
            continue
        return reasoning, k, flagged
    return None, REWRITE_TRIES, flagged


def write_set(out: Path, items: list[tuple[dict[str, Any], list[int], list[int], dict[str, Any]]]) -> None:
    """A training set as a play folder: one captured file per game, each call its request, prompt ids and target."""
    games: dict[tuple[str, int], list] = collections.defaultdict(list)
    for call, prompt, reply, message in items:
        games[(call["play"], call["user"])].append((call, prompt, reply, message))
    for n, (_, calls) in enumerate(sorted(games.items())):
        write_game(out, n, calls)


def write_game(out: Path, n: int, calls: list[tuple[dict[str, Any], list[int], list[int], dict[str, Any]]]) -> None:
    """One game of a training set (:func:`write_set`'s play folder) as its ``u{n}`` file: the calls of one source game,
    each its request, prompt ids and target."""
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
    """A training set's call's place in the game it labels, from its call_id (:func:`write_game`'s): the play's user and
    the call's index in that game."""
    head, i = call_id.rsplit("-c", 1)
    return int(head.rsplit("-u", 1)[1]), int(i)


def cmd_sets(a: argparse.Namespace) -> None:
    from ask_sd.server import Server
    tok = tokenizer(a.model)
    server = Server(a.server)
    calls = [c for p in a.plays for c in expert_calls(Path(p).expanduser())]
    rng = random.Random(a.seed)
    seeds = [rng.randrange(2 ** 31) for _ in calls]
    stats: dict[str, Any] = collections.Counter({"expert_calls": len(calls)})

    def one(k: int) -> dict[str, Any] | None:
        call = calls[k]
        raw = target_message(call["response"], call["response"]["expert"].get("reasoning") or "")
        raw_ids = render_target(tok, call["request"], raw)
        if raw_ids is None:
            return {"drop": "render_raw"}
        reasoning, draws, flagged = rewrite(server, call, a.student, random.Random(seeds[k]))
        if reasoning is None:
            return {"drop": "rewrite", "draws": draws, "flagged": flagged}
        new = target_message(call["response"], reasoning)
        new_ids = render_target(tok, call["request"], new)
        if new_ids is None:
            return {"drop": "render_rewrite"}
        return {"raw": (call, *raw_ids, raw), "rewrite": (call, *new_ids, new), "draws": draws, "flagged": flagged}

    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(one, range(len(calls))))
    kept = [r for r in results if r and "drop" not in r]
    for r in results:
        if r and "drop" in r:
            stats[f"dropped_{r['drop']}"] += 1
    stats["states"] = len(kept)
    stats["rewrite_draws"] = collections.Counter(r["draws"] for r in kept)
    stats["judge_turned_down"] = sum(r.get("flagged", 0) for r in results if r)
    for name in ("raw", "rewrite"):
        stats[f"{name}_reply_tokens_median"] = sorted(len(r[name][2]) for r in kept)[len(kept) // 2] if kept else None
    out = Path(a.out).expanduser()
    write_set(out / "train-raw", [r["raw"] for r in kept])
    write_set(out / "train-rewrite", [r["rewrite"] for r in kept])
    write_set(out / "train-control", [r["rewrite"] for r in kept])
    (out / "sets.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, default=str), flush=True)


def cmd_split(a: argparse.Namespace) -> None:
    """A training set's games split into those trained on and those held out (``--dev`` of them, drawn by game, so no
    game is on both sides), as ``all`` (every game) and ``solved`` (the games the student went on to solve): four play
    folders under ``--out`` (train-all, dev-all, train-solved, dev-solved) for train.py's ``--train-runs`` and
    ``--val-run``."""
    src = Path(a.set).expanduser()
    docs = [json.load(gzip.open(f, "rt")) for f in sorted((src / "captured").glob("u*.json.gz"))]
    order = list(range(len(docs)))
    random.Random(a.seed).shuffle(order)
    held = set(order[:round(a.dev * len(docs))])
    out = Path(a.out).expanduser()
    stats = {}
    for name, keep in [("all", lambda d: True), ("solved", lambda d: bool(d.get("solved")))]:
        for side, chosen in [("train", lambda i: i not in held), ("dev", lambda i: i in held)]:
            folder = out / f"{side}-{name}" / "captured"
            folder.mkdir(parents=True, exist_ok=True)
            picked = [d for i, d in enumerate(docs) if chosen(i) and keep(d)]
            for n, d in enumerate(picked):
                with gzip.open(folder / f"u{n:03d}.json.gz", "wt") as fp:
                    json.dump({**d, "user": n}, fp)
            stats[f"{side}-{name}"] = {"games": len(picked), "turns": sum(len(d["calls"]) for d in picked)}
    (out / "split.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


def instances(play: Path) -> dict[int, dict[str, Any]]:
    """Each user's instance in a play, from the play's schedule (loop.py schedule: user U plays slot U, the first slot
    carrying no user field): its task, instance seed and source, at exposure 0 as a probe slot plays it."""
    s = json.load(open(play / "schedule.json"))
    return {x.get("user", 0): {"task_id": x["task_id"], "instance_seed": x["instance_seed"],
                               "source": s["sources"][x["task_id"]], "exposure": 0} for x in s["slots"]}


def cmd_drop_wrong(a: argparse.Namespace) -> None:
    """A training set (a play folder of label targets: write_set's, a split's side) without the targets that submit a
    wrong grid, judged against the instance's true test output; a game left with no target is left out."""
    src, out = Path(a.set).expanduser(), Path(a.out).expanduser() / "captured"
    out.mkdir(parents=True, exist_ok=True)
    docs = [json.load(gzip.open(f, "rt")) for f in sorted((src / "captured").glob("u*.json.gz"))]
    plays = {d["source_play"]: instances(Path(d["source_play"])) for d in docs}
    truth = true_outputs([{**plays[d["source_play"]][d["source_user"]], "id": i} for i, d in enumerate(docs)])
    stats, n = collections.Counter(), 0
    for i, d in enumerate(docs):
        keep = []
        for c in d["calls"]:
            m = c["response"]["choices"][0]["message"]
            if coarse(m) == "submit":
                right = truth[i] is not None and grid_of(m.get("content")) == truth[i]
                stats["submits right" if right else "submits wrong, dropped"] += 1
                if not right:
                    continue
            keep.append(c)
        stats["targets kept"] += len(keep)
        if keep:
            with gzip.open(out / f"u{n:03d}.json.gz", "wt") as fp:
                json.dump({**d, "user": n, "calls": keep}, fp)
            n += 1
    stats["games"] = n
    print(json.dumps(dict(stats)))


def cmd_merge(a: argparse.Namespace) -> None:
    """Several play folders' games as one play folder (``--out``), renumbered in order: train.py takes one ``--val-run``,
    so the held-out games of the rounds pooled for DAgger's aggregate set go in together."""
    out = Path(a.out).expanduser() / "captured"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in a.plays:
        for f in sorted((Path(src).expanduser() / "captured").glob("u*.json.gz")):
            doc = json.load(gzip.open(f, "rt"))
            with gzip.open(out / f"u{n:03d}.json.gz", "wt") as fp:
                json.dump({**doc, "user": n}, fp)
            n += 1
    print(json.dumps({"games": n, "from": a.plays}))


def games(calls: list[Any]) -> list[list[Any]]:
    """Calls (``train.load_calls``) grouped by game, their ``run`` and ``user``, each game's calls in order, the games in
    a fixed order."""
    by: dict[tuple, list[Any]] = collections.defaultdict(list)
    for c in calls:
        by[(c.meta["run"], c.meta["user"])].append(c)
    return [sorted(by[k], key=lambda c: c.meta["i"]) for k in sorted(by, key=str)]


def replay_plan(new: list[Any], earlier: list[Any], trained_before: bool, ratio: float,
                rng: random.Random) -> list[Any]:
    """A replay dose's calls for a phase: every one of ``new`` and ``earlier`` for the first phase that trains; afterwards
    ``new`` and whole earlier games drawn at random until they hold at least ``ratio`` times ``len(new)`` labels (all of
    them when there are fewer). Whole games keep a game's calls together, so that a chain still reads its trunk once."""
    if not trained_before:
        return earlier + new
    want, picked, pool = ratio * len(new), [], games(earlier)
    rng.shuffle(pool)
    for game in pool:
        if len(picked) >= want:
            break
        picked += game
    return new + picked


# -- expert labels at a play's decision points (DAgger's labelling: the student played, the expert says what it would do) -

# a verdict or notice after which the only move left is to finish: nothing to decide
FORCED = ("submission was CORRECT", "You gave up", "The instance is over", "The attempt cap is reached")
# prime-agent's own notes to the model mid-action (after a tool's output, a compaction, a refinement): no new game state
HARNESS_NOTE = "["


def decision_kind(messages: list[dict[str, Any]]) -> str | None:
    """Whether a call is a decision point, and which: the agent's call at the start of the game, or right after the game
    said something new (a failure, demos received, a turn stopped at its budget, a reply with no valid action, other
    feedback); None for a call in the middle of an action (after a tool's output or a harness note only), and for one
    where the only move left is to finish."""
    import expert
    if not expert.agent_call(messages):
        return None
    last = max((j for j, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    if last == -1:
        return "start"
    said = [render.text_of(m.get("content")).strip() for m in messages[last + 1:] if m.get("role") == "user"]
    said = [t for t in said if t and not t.startswith(HARNESS_NOTE)]
    if not said:
        return None
    text = " ".join(said)
    if any(f in text for f in FORCED):
        return None
    if expert.after_failure(messages):
        return "failure"
    if "Demos received" in text:
        return "demos"
    if "exceeded the time or model-call budget" in text:
        return "budget"
    if "could not find a valid action" in text:
        return "invalid"
    return "other"


def decision_points(plays: list[Path]) -> list[dict[str, Any]]:
    """Every decision point of the plays (:func:`decision_kind`), as where to find its call (play, game, call index),
    with its game's task and outcome, and the expert's reply when the expert played that call (an assisted play)."""
    out = []
    for play in plays:
        for f in sorted((play / "captured").glob("u*.json.gz")):
            with gzip.open(f, "rt") as fp:
                doc = json.load(fp)
            for i, c in enumerate(doc["calls"]):
                resp = c.get("response") or {}
                if not (c.get("request") and resp.get("choices")):
                    continue
                kind = decision_kind(c["request"].get("messages") or [])
                if kind:
                    out.append({"id": point_id(play, doc, i), "play": str(play), "user": doc["user"], "i": i,
                                "call_id": c["call_id"], "kind": kind, "task_id": doc["task_id"],
                                "solved": doc.get("solved"), "played_by_expert": isinstance(resp.get("expert"), dict)})
    return out


def point_id(play: Path, doc: dict[str, Any], i: int) -> str:
    """A point's id, which names its label's file: the play, the user and the call's index, then the call's own id. A
    replayed round's games reuse every index, so without the call's id a label made for a game the replay discarded
    would be kept for the new game's call at the same index and trained on as its label (3-4% of two real-time runs'
    labels, from a round replayed after one of its games had been labelled, when ids were positional)."""
    return f"{play.name}-u{doc['user']:03d}-c{i:04d}-{doc['calls'][i]['call_id'][:12]}"


class StalePoint(Exception):
    """A point whose game no longer has its call: the round was replayed since the point was read."""


_games: dict[tuple[str, int], list] = {}
_games_lock = threading.Lock()


def point_call(p: dict[str, Any]) -> dict[str, Any]:
    """A point's call. Each game is read once and kept while its points are worked through (they come in game order);
    a game's file is several megabytes, and reading it for every point made labelling hours long. The call must be the
    one the point names (its call_id): a kept game a replay has since rewritten is read again, and a point the game
    on disk no longer has raises StalePoint, never another call in its place."""
    key = (p["play"], p["user"])
    with _games_lock:
        for attempt in range(2):
            if attempt or key not in _games:
                _games.pop(key, None)
                if len(_games) >= 48:
                    _games.pop(next(iter(_games)))
                with gzip.open(Path(p["play"]) / "captured" / f"u{p['user']:03d}.json.gz", "rt") as fp:
                    _games[key] = json.load(fp)["calls"]
            calls = _games[key]
            if p["i"] < len(calls) and calls[p["i"]]["call_id"] == p["call_id"]:
                return calls[p["i"]]
        raise StalePoint(f"{p['id']}: {p['play']} u{p['user']:03d} no longer has call {p['i']} ({p['call_id']})")


def cmd_label(a: argparse.Namespace) -> None:
    """The expert's reply at every decision point of the plays, saved under OUT/labels/ (a call the expert already played
    in an assisted play keeps that reply): the expert reads the call exactly as the student did (expert.Expert.ask,
    through the relay). Resumable; no new request once the replies so far cost ``--budget`` dollars."""
    import expert
    out = Path(a.out).expanduser()
    (out / "labels").mkdir(parents=True, exist_ok=True)
    points = decision_points([Path(p).expanduser() for p in a.plays])
    with open(out / "points.jsonl", "w") as fp:
        for p in points:
            fp.write(json.dumps(p) + "\n")
    print(json.dumps({"points": len(points), "kinds": dict(collections.Counter(p["kind"] for p in points))}), flush=True)
    e = expert.Expert(a.expert_url, a.expert, effort=a.effort or None)
    lock = threading.Lock()
    spent = [sum((json.loads(f.read_text()).get("usage") or {}).get("cost") or 0.0 for f in (out / "labels").glob("*.json"))]

    def one(p: dict[str, Any]) -> str:
        path = out / "labels" / f"{p['id']}.json"
        if path.exists():
            return "kept"
        call = point_call(p)
        if p["played_by_expert"]:
            served = call["response"]
            path.write_text(json.dumps({"message": served["choices"][0]["message"],
                                        "reasoning": served["expert"].get("reasoning") or "", "usage": None, "from": "takeover"}))
            return "takeover"
        with lock:
            if spent[0] >= a.budget:
                return "over budget"
        try:
            answer = e.ask(call["request"])
        except Exception:  # noqa: BLE001  (a point the expert does not answer stays unlabelled)
            return "failed"
        m = (answer.get("choices") or [{}])[0].get("message") or {}
        reasoning = next((m[k] for k in ("reasoning", "reasoning_content") if isinstance(m.get(k), str)), "")
        if not (m.get("content") or "").strip() and not m.get("tool_calls"):
            return "empty"
        usage = answer.get("usage") or {}
        with lock:
            spent[0] += usage.get("cost") or 0.0
        path.write_text(json.dumps({"message": {k: m.get(k) for k in ("role", "content", "tool_calls")}, "reasoning": reasoning,
                                    "usage": usage, "from": "label"}))
        return "labelled"

    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        res = collections.Counter(pool.map(one, points))
    print(json.dumps({"outcomes": dict(res), "cost_usd": round(spent[0], 2)}))


def cmd_label_set(a: argparse.Namespace) -> None:
    """The labelled decision points as a training set (OUT/set: :func:`write_set`'s play folder), each target the
    expert's reply as the chat template renders it after the call's prompt (its reasoning, then its text and tool calls);
    a point whose reply the template cannot render, or that makes no move (:func:`makes_move`), is left out and
    counted."""
    out = Path(a.out).expanduser()
    tok = tokenizer(a.model)
    items, stats = [], collections.Counter()
    for p in map(json.loads, open(out / "points.jsonl")):
        path = out / "labels" / f"{p['id']}.json"
        if not path.exists():
            stats["unlabelled"] += 1
            continue
        label = json.loads(path.read_text())
        call = point_call(p)
        message = student_message(label)
        if not makes_move(message):
            stats["no_move"] += 1
            continue
        ids = render_target(tok, call["request"], message)
        if ids is None:
            stats["render_failed"] += 1
            continue
        items.append(({**p, "request": call["request"]}, *ids, message))
        stats[f"kind:{p['kind']}"] += 1
        stats[f"move:{coarse(label['message'])}"] += 1
    write_set(out / "set", items)
    stats["examples"] = len(items)
    (out / "set.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(dict(stats)))


# -- process measures: the lessons' habits counted over whole plays ------------------------------------------------------

DEMO_PAIR = re.compile(r"Demo pair (\d+) (?:input|output)")
# code that tests a rule against the demos: it names them and compares
TESTS_DEMOS = re.compile(r"(?i)(demo|pairs?\b|G\[)")
COMPARES = re.compile(r"(==|!=|\bmatch|\bassert\b|\bverify|\bcheck|\bcompare|\ball\(|\bequal)")


# one printed row: digits apart by spaces or commas, in or out of brackets ("1 2 3", "[1, 2, 3],", " [1 2 3]]", "[[1 2 3]");
# the separator between numbers is required, so a long run of digits is read one way only (an optional one made a line
# like a 30-digit number take minutes to reject)
GRID_ROW_LINE = re.compile(r"\s*\[*\s*\d+(?:(?:\s*,\s*|\s+)\d+)*(?:\s*,)?\s*\]*\s*,?\s*")


def grids_in(text: str) -> list[Any]:
    """Every grid a text holds: Python or JSON list-of-lists literals on one line, and runs of printed rows, one per
    line, whatever their form (plain digits, a list per line, numpy's repr: L's printed answers were mostly the last
    two, which an earlier reading missed)."""
    out = []
    for lit in re.findall(r"\[\s*\[[\d,\s\[\]]+\]\s*\]", text):
        try:
            out.append(json.loads(lit))
        except ValueError:
            pass
    rows: list[list[int]] = []
    for line in text.splitlines() + [""]:
        if line.strip() and GRID_ROW_LINE.fullmatch(line) and re.search(r"\d", line):
            rows.append([int(x) for x in re.findall(r"\d+", line)])
        else:
            if rows:
                out.append(rows)
            rows = []
    return out


def game_process(doc: dict[str, Any]) -> dict[str, Any]:
    """One game's habits: its submissions (right, a resend of a grid already judged wrong, a grid no code of the game had
    printed, tested against the demos in the same action), its decision points (kind, move, demos seen), and its reply
    lengths."""
    import expert
    calls = [c for c in doc["calls"] if (c.get("response") or {}).get("choices")
             and expert.agent_call((c.get("request") or {}).get("messages") or [])]
    subs, points, lengths, fail_lengths = [], [], [], []
    rejected: list[Any] = []
    for i, c in enumerate(calls):
        msgs = c["request"].get("messages") or []
        choice = c["response"]["choices"][0]
        m = choice.get("message") or {}
        n = len(choice.get("token_ids") or [])
        lengths.append(n)
        if expert.after_failure(msgs):
            fail_lengths.append(n)
        history = " ".join(render.text_of(x.get("content")) for x in msgs)
        seen = max([int(k) for k in DEMO_PAIR.findall(history)] or [0])
        kind = decision_kind(msgs)
        if kind:
            points.append({"kind": kind, "move": coarse(m), "demos_seen": seen})
        if coarse(m) != "submit":
            continue
        grid = grid_of(m.get("content"))
        verdict = None
        if i + 1 < len(calls):
            nxt = calls[i + 1]["request"].get("messages") or []
            last = max((j for j, x in enumerate(nxt) if x.get("role") == "assistant"), default=-1)
            said = " ".join(render.text_of(x.get("content")) for x in nxt[last + 1:])
            verdict = True if "submission was CORRECT" in said else (False if "submission was INCORRECT" in said else None)
        elif doc.get("solved"):
            verdict = True
        printed = [g for x in msgs if x.get("role") == "tool" for g in grids_in(render.text_of(x.get("content")))]
        # the action so far: the agent's calls since the game last spoke (its last non-harness user message)
        start = max((j for j, x in enumerate(msgs) if x.get("role") == "user"
                     and not render.text_of(x.get("content")).lstrip().startswith(HARNESS_NOTE)), default=0)
        code = " ".join((tc.get("function") or {}).get("arguments") or "" for x in msgs[start:] if x.get("role") == "assistant"
                        for tc in x.get("tool_calls") or [])
        subs.append({"right": verdict, "resend": grid is not None and grid in rejected,
                     "printed": grid is not None and grid in printed, "tested": bool(TESTS_DEMOS.search(code) and COMPARES.search(code))})
        if verdict is False and grid is not None:
            rejected.append(grid)
    return {"task_id": doc["task_id"], "user": doc["user"], "solved": bool(doc.get("solved")), "subs": subs, "points": points,
            "lengths": lengths, "fail_lengths": fail_lengths}


def play_process(play: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Every game of a play (:func:`game_process`), keyed by (task, user): two plays of the same schedule give user U
    the same instance, so they pair game by game, and a play stopped before its last game ends (no records file) pairs
    as well as a whole one."""
    out = {}
    for f in sorted((play / "captured").glob("u*.json.gz")):
        with gzip.open(f, "rt") as fp:
            doc = json.load(fp)
        g = game_process(doc)
        out[(g["task_id"], doc["user"])] = g
    return out


def summarise(games: list[dict[str, Any]]) -> dict[str, Any]:
    """A play's measures from its games."""
    import statistics
    subs = [s for g in games for s in g["subs"]]
    decided = [s for s in subs if s["right"] is not None]
    pts = [p for g in games for p in g["points"]]
    asks = lambda ps: round(sum(p["move"] == "request" for p in ps) / len(ps), 3) if ps else None  # noqa: E731
    n = len(games)
    return {"games": n, "solved": sum(g["solved"] for g in games),
            "first_try": sum(g["solved"] and not any(s["right"] is False for s in g["subs"]) for g in games),
            "submissions_per_game": round(len(subs) / n, 2), "wrong_per_game": round(sum(s["right"] is False for s in subs) / n, 2),
            "submissions_right": round(sum(s["right"] is True for s in decided) / len(decided), 3) if decided else None,
            "resends": sum(s["resend"] for s in subs), "not_printed": round(sum(not s["printed"] for s in subs) / len(subs), 3) if subs else None,
            "tested_first": round(sum(s["tested"] for s in subs) / len(subs), 3) if subs else None,
            "decision_points": len(pts), "moves": {k: round(v / len(pts), 3) for k, v in collections.Counter(p["move"] for p in pts).items()} if pts else {},
            "asks_with_0_1_demos": asks([p for p in pts if p["demos_seen"] <= 1]),
            "asks_with_2plus_demos": asks([p for p in pts if p["demos_seen"] >= 2]),
            "asks_after_failure": asks([p for p in pts if p["kind"] == "failure"]),
            "reply_tokens_median": statistics.median([x for g in games for x in g["lengths"]] or [0]),
            "after_failure_tokens_median": statistics.median([x for g in games for x in g["fail_lengths"]] or [0])}


def cmd_process(a: argparse.Namespace) -> None:
    """The process measures of plays of the same schedule, each against the first (the baseline): the habits the lessons
    meant to teach (careful submissions, no guessing, testing before submitting, asking at the right time, thinking),
    then solves; per-game differences with paired 95% bootstrap intervals over the instances both played."""
    plays = [Path(p).expanduser() for p in a.plays]
    names = a.names or [p.name for p in plays]
    data = [play_process(p) for p in plays]
    rows = {name: summarise(list(d.values())) for name, d in zip(names, data)}
    per_game = {"wrong submissions": lambda g: sum(s["right"] is False for s in g["subs"]),
                "submissions not printed by code": lambda g: sum(not s["printed"] for s in g["subs"]),
                "resends": lambda g: sum(s["resend"] for s in g["subs"]),
                "solved": lambda g: float(g["solved"])}
    paired = {}
    for name, d in list(zip(names, data))[1:]:
        both = sorted(set(d) & set(data[0]))
        paired[name] = {k: [round(x, 3) for x in paired_ci([f(d[i]) - f(data[0][i]) for i in both])] + [len(both)]
                        for k, f in per_game.items()} if both else {}
        if both:
            only_new = sum(d[i]["solved"] and not data[0][i]["solved"] for i in both)
            only_base = sum(data[0][i]["solved"] and not d[i]["solved"] for i in both)
            paired[name]["solves: only it / only the first / McNemar p"] = [only_new, only_base,
                                                                             round(loop.mcnemar_p(only_new, only_base), 4)]
    out = {"plays": rows, "paired_vs_first": paired}
    Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
    keys = list(rows[names[0]].keys())
    print("| measure | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(json.dumps(rows[n][k]) for n in names) + " |")
    for name, p in paired.items():
        print(f"\n{name} minus {names[0]}, per game (mean, 95% interval, games): {json.dumps(p)}")


def cmd_agreement(a: argparse.Namespace) -> None:
    """How often a play's own move at its decision points was the expert's move there (the label command's folder for
    that play): overall, by kind of point, and the asks split by the expert's choice and by the demos seen."""
    out = Path(a.dir).expanduser()
    rows = []
    for p in map(json.loads, open(out / "points.jsonl")):
        path = out / "labels" / f"{p['id']}.json"
        if not path.exists():
            continue
        call = point_call(p)
        own = coarse(call["response"]["choices"][0].get("message") or {})
        glm = coarse(json.loads(path.read_text())["message"])
        msgs = call["request"].get("messages") or []
        seen = max([int(k) for k in DEMO_PAIR.findall(" ".join(render.text_of(x.get("content")) for x in msgs))] or [0])
        rows.append({"kind": p["kind"], "own": own, "glm": glm, "seen": seen})
    share = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    res = {"points": len(rows), "same_as_expert": share([r["own"] == r["glm"] for r in rows]),
           "by_kind": {k: share([r["own"] == r["glm"] for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})},
           "expert_moves": dict(collections.Counter(r["glm"] for r in rows)), "own_moves": dict(collections.Counter(r["own"] for r in rows)),
           "asks_where_expert_asked": share([r["own"] == "request" for r in rows if r["glm"] == "request"]),
           "asks_where_expert_did_not": share([r["own"] == "request" for r in rows if r["glm"] != "request"]),
           "submits_where_expert_submitted": share([r["own"] == "submit" for r in rows if r["glm"] == "submit"]),
           "submits_where_expert_did_not": share([r["own"] == "submit" for r in rows if r["glm"] != "submit"]),
           "asks_with_2plus_demos": share([r["own"] == "request" for r in rows if r["seen"] >= 2])}
    (out / "agreement.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# -- the decision check: at held-out states right after a failure, does a student make the expert's move? --------------

COARSE = {"code": "code", "request": "request", "submit": "submit"}   # any other action or plain text: "other"


def coarse(message: dict[str, Any]) -> str:
    """A reply's move as the check counts it: code, request (a demo), submit, or other."""
    return COARSE.get(reply_kind(message), "other")


def cmd_check_states(a: argparse.Namespace) -> None:
    """The check's states: up to ``--per-game`` calls right after a failure (expert.triggers) from each game of a play
    (the student's own, on tasks never trained on), ``--n`` of them drawn, each saved with its request and the instance it
    belongs to (for checking a submitted grid)."""
    import expert
    play, out = Path(a.play).expanduser(), Path(a.out).expanduser()
    records = {r["user"]: r for r in map(json.loads, open(play / "records.jsonl"))}
    rng = random.Random(a.seed)
    pool = []
    for f in sorted((play / "captured").glob("u*.json.gz")):
        with gzip.open(f, "rt") as fp:
            doc = json.load(fp)
        here = [i for i, c in enumerate(doc["calls"])
                if (c.get("response") or {}).get("choices") and expert.triggers((c.get("request") or {}).get("messages") or [])]
        rec = records.get(doc["user"], {})
        for i in rng.sample(here, min(a.per_game, len(here))):
            pool.append({"id": f"{play.name}-u{doc['user']:03d}-c{i:04d}", "request": doc["calls"][i]["request"],
                         "task_id": doc["task_id"], "source": rec.get("source"), "instance_seed": rec.get("instance_seed"),
                         "exposure": rec.get("exposure_index"), "played": coarse(doc["calls"][i]["response"]["choices"][0]["message"])})
    chosen = sorted(rng.sample(pool, min(a.n, len(pool))), key=lambda s: s["id"])
    (out / "states").mkdir(parents=True, exist_ok=True)
    with open(out / "states.jsonl", "w") as fp:
        for s in chosen:
            (out / "states" / f"{s['id']}.json").write_text(json.dumps(s["request"]))
            fp.write(json.dumps({k: v for k, v in s.items() if k != "request"}) + "\n")
    print(json.dumps({"candidates": len(pool), "states": len(chosen), "games": len({s["id"].rsplit("-c", 1)[0] for s in chosen})}))


def check_states(d: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(d / "states.jsonl")]


def cmd_check_labels(a: argparse.Namespace) -> None:
    """The expert's move at each state: one reply from the expert, with the student's exact view (expert.Expert.ask,
    through the relay), saved under labels/."""
    import expert
    d = Path(a.dir).expanduser()
    (d / "labels").mkdir(exist_ok=True)
    e = expert.Expert(a.expert_url, a.expert, effort=a.effort or None)
    todo = [s for s in check_states(d) if not (d / "labels" / f"{s['id']}.json").exists()]

    def one(s: dict[str, Any]) -> str:
        request = json.loads((d / "states" / f"{s['id']}.json").read_text())
        try:
            answer = e.ask(request)
        except Exception as error:  # noqa: BLE001  (a state the expert does not answer is left without a label)
            return f"failed: {error}"[:200]
        m = (answer.get("choices") or [{}])[0].get("message") or {}
        (d / "labels" / f"{s['id']}.json").write_text(json.dumps({"move": coarse(m), "message": m, "usage": answer.get("usage")}))
        return coarse(m)

    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        out = list(pool.map(one, todo))
    print(json.dumps({"labelled": len(out), "moves": dict(collections.Counter(out))}))


def cmd_check_sample(a: argparse.Namespace) -> None:
    """``--k`` replies of each served model (``--arms``) at each state, each with a seed of its own (the loop's server
    gives unseeded requests one stream), saved under samples/ARM/: the move, the reply's tokens, how it ended, the
    visible text and any tool calls."""
    from ask_sd.server import Server
    d = Path(a.dir).expanduser()
    server = Server(a.server)
    states = check_states(d)
    jobs = [(arm, s, k) for arm in a.arms for s in states for k in range(a.k)
            if not (d / "samples" / arm / f"{s['id']}-{k}.json").exists()]

    def one(job: tuple[str, dict[str, Any], int]) -> None:
        arm, s, k = job
        request = json.loads((d / "states" / f"{s['id']}.json").read_text())
        payload = {key: v for key, v in request.items() if key not in ("stream", "stream_options", "store",
                                                                        "max_completion_tokens")}
        payload.update({"model": arm, "stream": False, "n": 1, "max_tokens": a.max_tokens, "return_token_ids": True,
                        "seed": random.Random(f"{arm}|{s['id']}|{k}").randrange(2 ** 31)})
        out = server.chat(payload)
        choice = ((out or {}).get("choices") or [{}])[0]
        m = choice.get("message") or {}
        rec = {"move": coarse(m) if out else "failed", "tokens": len(choice.get("token_ids") or []),
               "finish_reason": choice.get("finish_reason"), "content": m.get("content"), "tool_calls": m.get("tool_calls")}
        (d / "samples" / arm).mkdir(parents=True, exist_ok=True)
        (d / "samples" / arm / f"{s['id']}-{k}.json").write_text(json.dumps(rec))

    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, jobs))
    print(json.dumps({"replies": len(jobs), "arms": a.arms}))


def true_outputs(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Each state's instance's correct test output (continual_arc's registry), by state id; None where it cannot be had."""
    from continual_arc.tasks import TaskRegistry
    reg, out = TaskRegistry.default(), {}
    for s in states:
        try:
            reg.assign(s["task_id"], s["source"])
            out[s["id"]] = [list(r) for r in reg.test_instance(s["task_id"], s["instance_seed"]).output]
        except Exception:  # noqa: BLE001
            out[s["id"]] = None
    return out


def pct(x: float | None) -> str:
    """A share as a whole percent, or a dash."""
    return "—" if x is None else f"{100 * x:.0f}%"


def paired_ci(diffs: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """The mean of paired differences and its 95% bootstrap interval."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def cmd_check_score(a: argparse.Namespace) -> None:
    """The check's measures per model, against the baseline model state by state (95% bootstrap intervals), and the go
    rule: the same move as the expert at least ``--margin`` more often than the baseline (interval above 0), submissions
    right no less often than the baseline's, and replies at least ``--length-floor`` of the baseline's length."""
    import statistics
    d = Path(a.dir).expanduser()
    states = [s for s in check_states(d) if (d / "labels" / f"{s['id']}.json").exists()]
    glm = {s["id"]: json.loads((d / "labels" / f"{s['id']}.json").read_text())["move"] for s in states}
    truth = true_outputs(states)
    arms = [a.baseline] + [x for x in a.arms if x != a.baseline]
    per: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm in arms:
        per[arm] = {s["id"]: [json.loads(f.read_text()) for f in sorted((d / "samples" / arm).glob(f"{s['id']}-*.json"))]
                    for s in states}

    def same(arm: str, sid: str) -> float | None:
        rs = per[arm][sid]
        return sum(r["move"] == glm[sid] for r in rs) / len(rs) if rs else None

    def grid_right(r: dict[str, Any], sid: str) -> bool:
        return truth.get(sid) is not None and grid_of(r.get("content")) == truth[sid]

    def demos_seen(sid: str) -> int:
        msgs = json.loads((d / "states" / f"{sid}.json").read_text()).get("messages") or []
        text = " ".join(render.text_of(m.get("content")) for m in msgs)
        return max([int(n) for n in re.findall(r"Demo pair (\d+) (?:input|output)", text)] or [0])

    seen = {s["id"]: demos_seen(s["id"]) for s in states}
    rows, out = [], {"states": len(states), "expert_moves": dict(collections.Counter(glm.values())), "arms": {}}
    base_len = statistics.median([r["tokens"] for rs in per[a.baseline].values() for r in rs]) or 1
    base_sub = [r for rs_id, rs in per[a.baseline].items() for r in rs if r["move"] == "submit"]
    base_acc = sum(grid_right(r, sid) for sid, rs in per[a.baseline].items() for r in rs if r["move"] == "submit") / max(1, len(base_sub))
    for arm in arms:
        flat = [(sid, r) for sid, rs in per[arm].items() for r in rs]
        if not flat:
            continue
        moves = collections.Counter(r["move"] for _, r in flat)
        subs = [(sid, r) for sid, r in flat if r["move"] == "submit"]
        acc = sum(grid_right(r, sid) for sid, r in subs) / len(subs) if subs else None
        req_where = [r["move"] == "request" for sid, r in flat if glm[sid] == "request"]
        req_not = [r["move"] == "request" for sid, r in flat if glm[sid] != "request"]
        req_2plus = [r["move"] == "request" for sid, r in flat if seen[sid] >= 2]
        length = statistics.median(r["tokens"] for _, r in flat)
        diffs = [same(arm, sid) - same(a.baseline, sid) for sid in per[arm]
                 if same(arm, sid) is not None and same(a.baseline, sid) is not None]
        m, lo, hi = paired_ci(diffs) if diffs and arm != a.baseline else (0.0, 0.0, 0.0)
        agree = statistics.mean(x for x in (same(arm, sid) for sid in per[arm]) if x is not None)
        go = (arm != a.baseline and m >= a.margin and lo > 0 and (acc is None or acc >= base_acc)
              and length >= a.length_floor * base_len)
        out["arms"][arm] = {"replies": len(flat), "same_as_expert": round(agree, 3), "vs_baseline": [round(m, 3), round(lo, 3), round(hi, 3)],
                            "moves": {k: round(v / len(flat), 3) for k, v in moves.items()},
                            "requests_where_expert_requested": round(sum(req_where) / len(req_where), 3) if req_where else None,
                            "requests_where_expert_did_not": round(sum(req_not) / len(req_not), 3) if req_not else None,
                            "requests_with_2plus_demos_seen": round(sum(req_2plus) / len(req_2plus), 3) if req_2plus else None,
                            "submissions": len(subs), "submissions_right": acc, "wrong_submissions_per_100": round(100 * sum(not grid_right(r, sid) for sid, r in subs) / len(flat), 1),
                            "reply_tokens_median": length, "cut_off": sum(r["finish_reason"] == "length" for _, r in flat), "go": go}
        rows.append(f"| {arm} | {100 * agree:.0f}% | {'—' if arm == a.baseline else f'{100 * m:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})'} | "
                    f"{100 * moves['code'] / len(flat):.0f} / {100 * moves['request'] / len(flat):.0f} / {100 * moves['submit'] / len(flat):.0f} | "
                    f"{pct(out['arms'][arm]['requests_where_expert_requested'])} | {pct(out['arms'][arm]['requests_where_expert_did_not'])} | "
                    f"{pct(out['arms'][arm]['requests_with_2plus_demos_seen'])} | {len(subs)} ({'—' if acc is None else f'{100 * acc:.0f}%'}) | "
                    f"{length:.0f} | {'**go**' if go else ('—' if arm == a.baseline else 'no')} |")
    em = out["expert_moves"]
    lines = [f"Decision check: {len(states)} states right after a failure (held-out games); the expert's moves {json.dumps(em)}.",
             f"Go rule: same move as the expert {100 * a.margin:.0f}+ points above {a.baseline} (interval above 0), "
             f"submissions right no less often, replies at least {100 * a.length_floor:.0f}% of {a.baseline}'s length.", "",
             "| Model | Same move as the expert | vs baseline, points (95%) | code / request / submit % | Asks for a demo where the expert did | ...where it did not | ...with 2+ demos seen | Submissions (right) | Reply tokens, median | Go |",
             "|---|---|---|---|---|---|---|---|---|---|"] + rows
    (d / "check.md").write_text("\n".join(lines) + "\n")
    (d / "check.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


def grid_of(content: str | None) -> Any:
    """The grid of a reply's last JSON submit action, or None."""
    found = re.findall(r'"grid"\s*:\s*(\[\s*\[.*?\]\s*\])', content or "", re.S)
    try:
        return json.loads(found[-1]) if found else None
    except ValueError:
        return None


ACTION = re.compile(r'"action"\s*:\s*"(\w+)"')
# where a reply's action starts: its last JSON action object, with the code fence that opens it if there is one
ACTION_START = re.compile(r'(?:```(?:json)?\s*)?\{\s*"action"\s*:')


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
    """Whether a reply does anything in the game: it calls a tool or gives an action (:func:`split_reply` finds one). A
    reply that only says what it will do ("Let me load the grids.") makes none, and as a target it teaches the student to
    spend a turn on nothing: since its prose goes to the reasoning (:func:`student_message`), it is an empty reply, which
    the harness answers with "no valid action" and, after a few, a given-up instance (20 such targets in one test run's
    first five rounds, whose trained students replied empty at a turn's start and gave up)."""
    return bool(message.get("tool_calls")) or bool(split_reply(message.get("content"))[1])


def student_message(label: dict[str, Any]) -> dict[str, Any]:
    """The student's target for an oracle's label: all of the oracle's prose, its reasoning and the prose of its visible
    text, as the student's reasoning, and only the action (with any tool calls) as its visible text, so that the student
    reasons in its thinking and acts in the text the harness reads (the oracle often explains an easy move in its reply
    rather than in its reasoning)."""
    prose, action = split_reply(label["message"].get("content"))
    reasoning = "\n\n".join(x for x in ((label.get("reasoning") or "").strip(), prose) if x)
    message = {"role": "assistant", "content": action, "reasoning_content": reasoning}
    if label["message"].get("tool_calls"):
        message["tool_calls"] = label["message"]["tool_calls"]
    return message


def reply_kind(message: dict[str, Any]) -> str:
    """What a reply does: asks for a demo, submits, gives up or finishes (its JSON action), runs code (a tool call), or
    only writes text."""
    acts = ACTION.findall(message.get("content") or "")
    if acts:
        a = acts[-1]
        return "request" if a.startswith("request") or a == "demo" else ("submit" if a == "submit" else a)
    return "code" if message.get("tool_calls") else "text"


def after_failure_kinds(play: Path) -> dict[str, Any]:
    """What a play's model did at the call right after each failure (expert.triggers: the calls the expert would
    play), as counts and shares."""
    import expert
    kinds: collections.Counter = collections.Counter()
    for f in sorted((play / "captured").glob("u*.json.gz")):
        with gzip.open(f, "rt") as fp:
            doc = json.load(fp)
        for c in doc["calls"]:
            req, resp = c.get("request") or {}, c.get("response") or {}
            if not resp.get("choices") or not expert.triggers(req.get("messages") or []):
                continue
            kinds[reply_kind(resp["choices"][0].get("message") or {})] += 1
    n = sum(kinds.values())
    return {"calls": n, **{k: f"{100 * v / n:.0f}%" for k, v in kinds.most_common()}} if n else {"calls": 0}


def expert_log(play: Path) -> dict[str, Any]:
    """A play's expert calls from its log (expert.jsonl): outcomes, seconds and cost."""
    rows = [json.loads(line) for line in open(play / "expert.jsonl")] if (play / "expert.jsonl").exists() else []
    served = [r for r in rows if r.get("outcome") == "served"]
    secs = sorted(r["seconds"] for r in served if r.get("seconds") is not None)
    return {"triggered": len(rows), "outcomes": dict(collections.Counter(r.get("outcome") for r in rows)),
            "seconds_median": secs[len(secs) // 2] if secs else None,
            "cost_usd": round(sum((r.get("usage") or {}).get("cost") or 0.0 for r in served), 2)}


def cmd_report(a: argparse.Namespace) -> None:
    d = Path(a.dir).expanduser()
    base = Path(a.baseline).expanduser()
    lines = [f"Baseline: b6's validation play ({base}).", "",
             "| Arm | Solved | vs b6: only arm / only b6 | McNemar p | Gave up | Demo requests | Wrong attempts | Calls | "
             "Parse failures |", "|---|---|---|---|---|---|---|---|---|"]
    out: dict[str, Any] = {}
    b = loop.compare(base, base)["base"]
    lines.append(f"| b6 | {b['solved']} | - | - | {b['gave_up']} | {b['demo_requests']} | {b['wrong_attempts']} | "
                 f"{b['model_calls']:.0f} | {b['parse_failures']:.0f} |")
    rows = [(arm, d / "runs" / f"val-{arm}") for arm in a.arms] + [(Path(p).name, Path(p).expanduser()) for p in a.also]
    for arm, run in rows:
        if not (run / "records.jsonl").exists():
            continue
        c = loop.compare(base, run)
        out[arm] = c
        n = c["new"]
        lines.append(f"| {arm} | {n['solved']} | {c['solved_only_new']} / {c['solved_only_base']} | "
                     f"{c['mcnemar_p']:.3f} | {n['gave_up']} | {n['demo_requests']} | {n['wrong_attempts']} | "
                     f"{n['model_calls']:.0f} | {n['parse_failures']:.0f} |")
    for x, y in [(p, q) for i, p in enumerate(a.arms) for q in a.arms[i + 1:]]:
        rx, ry = d / "runs" / f"val-{x}", d / "runs" / f"val-{y}"
        if (rx / "records.jsonl").exists() and (ry / "records.jsonl").exists():
            c = loop.compare(rx, ry)
            out[f"{y}_vs_{x}"] = c
            lines.append(f"\n{y} vs {x}: solved only by {y} {c['solved_only_new']}, only by {x} {c['solved_only_base']}, "
                         f"McNemar p {c['mcnemar_p']:.3f}")
    lines.append("\nRight after a failure, what the next call does (share of those calls):")
    for name, run in [("b6", base)] + rows:
        if (run / "captured").exists():
            lines.append(f"- {name}: {json.dumps(after_failure_kinds(run))}")
    plays = sorted((d / "runs").glob("assist-*"))
    if plays:
        lines.append("\nThe assisted plays (b6, the expert after each failure):")
        for p in plays:
            # the games as each saved itself on ending (a play stopped before its last game ends has no records file)
            games = [json.load(gzip.open(f, "rt")) for f in sorted((p / "captured").glob("u*.json.gz"))]
            lines.append(f"- {p.name}: solved {sum(bool(g.get('solved')) for g in games)}/{len(games)} games; "
                         f"expert {json.dumps(expert_log(p))}")
    if (d / "sets" / "sets.json").exists():
        lines.append(f"\nTraining sets: {(d / 'sets' / 'sets.json').read_text().strip()}")
    (d / "report.md").write_text("\n".join(lines) + "\n")
    (d / "report.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("plays", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--server", default="http://localhost:18101/v1")
    s.add_argument("--student", default="b1-6", help="the served name of the student that writes the rewrites")
    s.add_argument("--model", default=loop.MODEL)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sets)
    lb = sub.add_parser("label")
    lb.add_argument("plays", nargs="+", help="the plays whose decision points the expert labels")
    lb.add_argument("--out", required=True)
    lb.add_argument("--expert", default="z-ai/glm-5.3")
    lb.add_argument("--expert-url", default="http://127.0.0.1:18200/v1")
    lb.add_argument("--effort", default="medium")
    lb.add_argument("--workers", type=int, default=16)
    lb.add_argument("--budget", type=float, default=150.0, help="dollars: no new request once the labels cost this")
    lb.set_defaults(fn=cmd_label)
    ls = sub.add_parser("label-set")
    ls.add_argument("--out", required=True, help="the label command's folder")
    ls.add_argument("--model", default=loop.MODEL)
    ls.set_defaults(fn=cmd_label_set)
    pr = sub.add_parser("process")
    pr.add_argument("plays", nargs="+", help="plays of the same schedule; the first is the baseline")
    pr.add_argument("--names", nargs="*", default=None)
    pr.add_argument("--out", required=True, help="the measures as JSON")
    pr.set_defaults(fn=cmd_process)
    ag = sub.add_parser("agreement")
    ag.add_argument("dir", help="a label command's folder for one play")
    ag.set_defaults(fn=cmd_agreement)
    cs = sub.add_parser("check-states")
    cs.add_argument("play", help="the student's own play of held-out tasks (a validation play)")
    cs.add_argument("--out", required=True)
    cs.add_argument("--n", type=int, default=150)
    cs.add_argument("--per-game", type=int, default=2)
    cs.add_argument("--seed", type=int, default=0)
    cs.set_defaults(fn=cmd_check_states)
    cl = sub.add_parser("check-labels")
    cl.add_argument("dir")
    cl.add_argument("--expert", default="z-ai/glm-5.3")
    cl.add_argument("--expert-url", default="http://127.0.0.1:18200/v1")
    cl.add_argument("--effort", default="medium")
    cl.add_argument("--workers", type=int, default=8)
    cl.set_defaults(fn=cmd_check_labels)
    cm = sub.add_parser("check-sample")
    cm.add_argument("dir")
    cm.add_argument("--arms", nargs="+", required=True, help="the served names to sample")
    cm.add_argument("--server", default="http://localhost:18101/v1")
    cm.add_argument("--k", type=int, default=4)
    cm.add_argument("--max-tokens", type=int, default=16384)
    cm.add_argument("--workers", type=int, default=48)
    cm.set_defaults(fn=cmd_check_sample)
    cc = sub.add_parser("check-score")
    cc.add_argument("dir")
    cc.add_argument("--arms", nargs="+", required=True)
    cc.add_argument("--baseline", default="b1-6")
    cc.add_argument("--margin", type=float, default=0.10)
    cc.add_argument("--length-floor", type=float, default=0.85)
    cc.set_defaults(fn=cmd_check_score)
    sp = sub.add_parser("split")
    sp.add_argument("set", help="a training set's play folder (sets/train-raw)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--dev", type=float, default=0.15, help="the share of games held out")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_split)
    mg = sub.add_parser("merge")
    mg.add_argument("plays", nargs="+", help="play folders (a split's dev-all, ...)")
    mg.add_argument("--out", required=True)
    mg.set_defaults(fn=cmd_merge)
    dw = sub.add_parser("drop-wrong")
    dw.add_argument("set", help="a training set's play folder (a split's train-all, dev-all)")
    dw.add_argument("--out", required=True)
    dw.set_defaults(fn=cmd_drop_wrong)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True, help="the experiment's folder (runs/val-ARM, runs/assist-*, sets/)")
    r.add_argument("--baseline", required=True, help="b6's validation play")
    r.add_argument("--arms", nargs="+", default=["raw", "rewrite", "control"])
    r.add_argument("--also", nargs="*", default=[], help="other plays of the validation split shown against the baseline")
    r.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
