"""Same-state replay: what candidate teachers, and the student, do at the moments the student is stuck (an experiment
from before the paper's rows; they reach this module only through overfit.py's card check, :func:`check_program`).

A moment is a model call of the student's own play right after the evidence went against its current idea:

- ``incorrect``: its submission was just judged INCORRECT;
- ``check_false``: one of its own checks just printed a failed comparison (``Match: False``), with two or more demos in
  hand.

Each moment's call is sent again, word for word, to a model under a setup, and its next reply sampled several times:

- ``student``: the student as it played (no notes);
- ``notes``: the base model with the notes (the loop's teacher, and the teacher a takeover would hand the call to);
- ``hint``: the base model with the notes and a hint for this call, in the same notes block;
- ``thought``: the base model with the notes and the same hint as the model's own thought, opening its reasoning
  (``reasoning_prefill``, as play's ``--context-placement thought``), worded by the base model;
- ``thought_only``: the base model with that thought and no notes: the thought in place of the notes, not beside them;
- ``fair``: the base model with the notes and a fair note as its own thought: a fair oracle's next move at this call,
  worded by the base model.

The fair oracle (PROBE_FAIR) sees exactly what the student saw at the call (its system prompt and the conversation,
the notes block taken out) and nothing else: no rule card, no answer. It plays the next step; its reasoning and plan,
worded by the base model as a thought, are the fair note. A note built only from what the student sees carries no
information the student lacks, so a student can learn what it says (the per-call hints below cannot: stage B).

The hint and the thought come from an oracle that knows the task's rule (a rule card per task, written once from the
instance's demos, its test pair and, for a Re-ARC task, the reference program); both must point at evidence the agent
has already seen and never state the rule or a grid. Each thought's mean log-probability under the student at the bare
call is recorded next to the student's own reasoning's: how far the thought is from what the student would think there.

A judge (also knowing the rule card) reads every reply, the setups mixed and unlabelled, and labels it: ``same`` (keeps
its idea, or resends a rejected grid), ``new_wrong``, ``right_track`` (the key part of the actual rule), or
``no_progress``; and ``leap`` when it asserts the rule or answer with nothing visible behind it (the leak a per-call hint
can teach). Scripts add the notes citations and whether the reply carries an action or a tool call.

Steps (each writes one file per moment under DIR and skips what is there, so a rerun carries on):
  probe.py moments DIR PLAY... [--per-game 2] [--max 150]   the moments of the student's plays
  probe.py cards DIR                                        a rule card per task (oracle)
  probe.py subset SRC DIR [--with-replies]                    SRC's moments that are play calls, with their cards
  probe.py hints DIR --server URL                            a hint and a thought per moment (oracle, base model)
  probe.py fair DIR --server URL                             a fair note per moment (fair oracle, base model)
  probe.py replay DIR --server URL [--setups ...] [--n 8]    the replies
  probe.py judge DIR                                         the labels (oracle)
  probe.py report DIR                                        the table, to stdout and DIR/report.md
The server is the loop's vLLM (serve.sh) with the base model and the student loaded; from another machine through a
tunnel: ``ssh -N -L 18101:localhost:8101 NODE``. The writer and the judge are OpenRouter's (PROBE_WRITER, PROBE_JUDGE;
OPENROUTER_API_KEY from the environment).
Run with ask_sd and continual_arc importable: ``HF_HUB_OFFLINE=1 python training/probe.py ...``
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import inspect
import json
import os
import random
import re
import statistics
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

LOOP = Path(__file__).resolve().parent
sys.path[:0] = [str(LOOP), str(LOOP / "analysis")]
import loop  # noqa: E402
import render  # noqa: E402
from trend import CITES, internal, text_of  # noqa: E402

SETUPS = ("student", "notes", "hint", "thought", "thought_only", "fair")
KINDS = ("incorrect", "check_false")
STUDENT = "b1-6"
ORACLE_URL, ORACLE_KEY = "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
# who writes the rule cards and the hints (the text a thought teacher trains toward) and who judges the replies, on
# OpenRouter; PROBE_WRITER / PROBE_JUDGE set them, each "model" or "model|provider,provider|effort". Anthropic's
# filter refused Opus writing the thoughts (its terms on duplicating model outputs), so the writer is an open-weight
# model, served by hosts of its weights (GLM-5.3: zai-org's license, MIT-style); the judge, who only grades, is Opus
WRITER = os.environ.get("PROBE_WRITER", "z-ai/glm-5.3|fireworks,parasail,gmicloud|medium")
# Anthropic's filter refused Opus judging as well (most chunks of the fair setup's replies), so the judge is the
# writer's model; each judge keeps its labels in a folder of its own (Opus's in judged/), so that one table never mixes
# two judges
JUDGE = os.environ.get("PROBE_JUDGE", WRITER)
JUDGED = "judged" if JUDGE.startswith("anthropic/") else "judged-" + JUDGE.split("|")[0].split("/")[-1]
# the fair oracle, who plays the next step from the student's view: an open-weight model, as the writer
# (Anthropic's filter refused Opus here too, on the same terms, for 187 of 202 moments)
FAIR = os.environ.get("PROBE_FAIR", WRITER)
# prime-agent's /refine calls (the continual harness's editor, with the trajectory in its prompt) are not play calls;
# internal() does not catch them (11 of stage A's 106 judged moments were refine calls)
REFINE = "You are Prime Agent's /refine"
NOTES = LOOP / "rules" / "c1.md"   # the notes b6 was trained toward
INCORRECT = "Your submission was INCORRECT"
DEMOS_RECEIVED = re.compile(r"Demos received \(request (\d+)\)")
# a failed comparison the model's own code printed: "Match: False", "matches demo 2: False", "equal == False", ...
CHECK_FALSE = re.compile(r"(?im)^.*\b(match\w*|equal\w*|correct|same|ok|pass\w*)\b[^\n]{0,40}\bFalse\b")
ACTION = re.compile(r'"action"\s*:\s*"(\w+)"')
REPLAY_MAX_TOKENS = 4096   # replies run to a few thousand tokens (round 0: median 566, p90 2,068)


# -- the conversation, as the oracle and the judge read it ------------------------------------------------------------

def cut(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head] + f"\n… [{len(text) - head - tail:,} characters left out] …\n" + text[-tail:]


def strip_notes(text: str, block: str) -> str:
    return text.replace(block, "") if block else text


def render_context(messages: list[dict[str, Any]], *, block: str = "", total: int = 60_000, scale: int = 1) -> str:
    """The conversation without its system prompt: the environment's messages whole up to a cap (they carry the grids),
    tool results cut, the agent's reasoning cut, its code; the middle dropped first when it runs over ``total``.
    ``scale`` multiplies every cap (the fair oracle reads more of each message than the judge)."""
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "assistant":
            reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
            body = []
            if reasoning:
                body.append("(reasoning) " + cut(reasoning, 500 * scale, 900 * scale))
            if text_of(m.get("content")).strip():
                body.append("(reply) " + cut(text_of(m.get("content")), 800 * scale, 400 * scale))
            for tc in m.get("tool_calls") or []:
                args = (tc.get("function") or {}).get("arguments") or ""
                try:
                    code = json.loads(args).get("code") or args
                except (json.JSONDecodeError, AttributeError):
                    code = args
                body.append("(code) " + cut(code, 1500 * scale, 500 * scale))
            parts.append("[agent]\n" + "\n".join(body))
        elif role == "tool":
            parts.append("[tool result]\n" + cut(strip_notes(text_of(m.get("content")), block), 1200 * scale, 600 * scale))
        else:
            parts.append("[environment]\n" + cut(strip_notes(text_of(m.get("content")), block), 6000 * scale, 2000 * scale))
    keep_head, out = 2, list(parts)
    while sum(map(len, out)) > total and len(out) > keep_head + 6:
        del out[keep_head]
    if len(out) < len(parts):
        out.insert(keep_head, f"[… {len(parts) - len(out)} earlier messages left out …]")
    return "\n\n".join(out)


def grid_text(grid) -> str:
    return "\n".join(" ".join(str(x) for x in row) for row in grid)


# -- moments ------------------------------------------------------------------------------------------------------------

def is_refine(messages: list[dict[str, Any]]) -> bool:
    return bool(messages) and messages[0].get("role") == "system" and text_of(messages[0].get("content")).startswith(REFINE)


def moment_id(play: str, user: int, i: int) -> str:
    return f"{play}-u{user:03d}-c{i:04d}"


def game_moments(ep: dict, play: Path) -> list[dict]:
    """Every candidate moment of one game: the agent's calls whose new messages are an INCORRECT verdict, or a failed
    check with two or more demos in hand."""
    out, demos = [], 0
    for i, c in enumerate(ep["calls"]):
        if not (c.get("request") and c.get("response")) or internal(c) or render.is_compaction(c):
            continue
        msgs = c["request"]["messages"]
        if is_refine(msgs):
            continue
        last = max((j for j, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        new = msgs[last + 1:]
        texts = [(m.get("role"), text_of(m.get("content"))) for m in new]
        for _, t in texts:
            demos = max([demos] + [int(n) for n in DEMOS_RECEIVED.findall(t)])
        kind = None
        if any(r == "user" and INCORRECT in t for r, t in texts):
            kind = "incorrect"
        elif demos >= 2 and any(r == "tool" and CHECK_FALSE.search(t) for r, t in texts):
            kind = "check_false"
        if kind:
            r = ep["record"]
            out.append({"id": moment_id(play.name, ep["user"], i), "play": str(play.resolve()), "user": ep["user"], "i": i,
                        "kind": kind, "turn": c.get("turn"), "task_id": ep["task_id"], "task_label": ep.get("task_label"),
                        "source": r.get("source"), "instance_seed": r.get("instance_seed"),
                        "exposure": r.get("exposure_index"), "solved": bool(ep.get("solved")), "demos_seen": demos})
    return out


def cmd_moments(a) -> None:
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    chosen, found = [], Counter()
    for play in map(Path, a.plays):
        records = {r.get("user"): r for r in map(json.loads, open(play / "records.jsonl"))}
        for f in sorted((play / "captured").glob("u*.json.gz")):   # one game in memory at a time
            with gzip.open(f, "rt") as fp:
                ep = json.load(fp)
            ep["record"] = records.get(ep["user"], {})
            ms = game_moments(ep, play)
            found.update(m["kind"] for m in ms)
            pick = []
            for kind in KINDS:   # one of each kind first, then any
                of_kind = [m for m in ms if m["kind"] == kind]
                if of_kind:
                    pick.append(rng.choice(of_kind))
            rest = [m for m in ms if m not in pick]
            rng.shuffle(rest)
            chosen += (pick + rest)[: a.per_game]
    if len(chosen) > a.max:
        chosen = rng.sample(chosen, a.max)
    chosen.sort(key=lambda m: m["id"])
    with open(out / "moments.jsonl", "w") as f:
        for m in chosen:
            f.write(json.dumps(m) + "\n")
    print(json.dumps({"candidates": dict(found), "chosen": dict(Counter(m["kind"] for m in chosen)),
                      "games": len({(m["play"], m["user"]) for m in chosen}), "tasks": len({m["task_id"] for m in chosen})}))


def load_moments(d: Path, limit: int | None = None) -> list[dict]:
    ms = [json.loads(line) for line in open(d / "moments.jsonl")]
    return ms[:limit] if limit else ms


def moment_call(m: dict) -> dict:
    with gzip.open(Path(m["play"]) / "captured" / f"u{m['user']:03d}.json.gz", "rt") as fp:
        return json.load(fp)["calls"][m["i"]]


# -- the oracle --------------------------------------------------------------------------------------------------------

_endpoints: dict[str, Any] = {}
_endpoint_lock = threading.Lock()


def oracle(spec: str):
    """The OpenRouter endpoint for ``spec`` ("model", or "model|provider,provider|effort": only those hosts, no
    fallback to others)."""
    with _endpoint_lock:
        if spec not in _endpoints:
            from ask_sd.thoughts import Endpoint
            model, providers, effort = (spec.split("|") + ["", ""])[:3]
            routing = {"order": providers.split(","), "allow_fallbacks": False} if providers else None
            _endpoints[spec] = Endpoint(ORACLE_URL, model, ORACLE_KEY, routing=routing, effort=effort or None, retries=8)
            if not _endpoints[spec].key:
                raise SystemExit(f"no {ORACLE_KEY}: set it in the environment")
    return _endpoints[spec]


def ask_json(messages: list[dict[str, str]], spec: str, *, max_tokens: int = 3000) -> Any:
    """The first JSON value in ``spec``'s answer; a reasoning model's budget covers its reasoning too, so it gets more."""
    from ask_sd.thoughts import first_json
    reasoning = "|" in spec and spec.split("|")[-1]
    text = oracle(spec).ask(messages, max_tokens=max_tokens * 5 if reasoning else max_tokens, temperature=0.0) or ""
    value = first_json(text)
    if value is None:   # GLM-5.3 puts raw newlines inside its strings at times: JSON that allows them
        starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
        if starts:
            try:
                value = json.JSONDecoder(strict=False).raw_decode(text[min(starts):])[0]
            except ValueError:
                pass
    return value


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def run_all(items: list, fn, workers: int) -> None:
    done = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status in pool.map(fn, items):
            done[status] += 1
    print(json.dumps(dict(done)), flush=True)


# -- rule cards ---------------------------------------------------------------------------------------------------------

# Re-ARC loads its flat modules by swapping them into sys.modules for a moment, so two threads loading it at once break
# each other's `from dsl import *` (NameError Tuple): the task data is built under this lock, with one handle
_task_lock = threading.Lock()
_rearc = None


def reference_program(task_id: str) -> str | None:
    """A Re-ARC task's verifier program (the generator's reference rule, in Re-ARC's DSL), or None."""
    global _rearc
    from continual_arc.rearc import ReArc
    if _rearc is None:
        _rearc = ReArc()
    rearc = _rearc
    try:
        if not rearc.has_task(task_id):
            return None
        return inspect.getsource(rearc._verifiers[task_id])
    except Exception:  # noqa: BLE001  (no Re-ARC data, or not a Re-ARC task: the card goes without it)
        return None


RUNNER = """import json, sys
ns = {}
exec(open("program.py").read(), ns)
out = []
for grid in json.load(open("inputs.json")):
    try:
        out.append([[int(x) for x in row] for row in ns["transform"]([row[:] for row in grid])])
    except Exception as e:
        out.append({"error": repr(e)[:300]})
print(json.dumps(out))
"""


def check_program(source: str, pairs: list) -> dict:
    """The writer's ``transform`` run on every pair's input, in a process of its own (no environment, an empty folder,
    30 seconds): which pairs it reproduces exactly."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "program.py").write_text(source)
        (Path(tmp) / "inputs.json").write_text(json.dumps([[list(r) for r in pr.input] for pr in pairs]))
        (Path(tmp) / "run.py").write_text(RUNNER)
        try:
            run = subprocess.run([sys.executable, "-I", "run.py"], cwd=tmp, capture_output=True, text=True, timeout=30,
                                 env={"PATH": "/usr/bin:/bin"})
            outs = json.loads(run.stdout.strip().splitlines()[-1]) if run.stdout.strip() else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
            outs = None
    if not isinstance(outs, list) or len(outs) != len(pairs):
        return {"ok": [False] * len(pairs), "error": "the program did not run"}
    ok = [o == [list(r) for r in pr.output] for o, pr in zip(outs, pairs)]
    errors = [o["error"] for o in outs if isinstance(o, dict)]
    return {"ok": ok, **({"error": errors[0]} if errors else {})}


def cmd_cards(a) -> None:
    from continual_arc.tasks import TaskRegistry
    d = Path(a.dir)
    registry = TaskRegistry.default()
    by_task = {}
    for m in load_moments(d, a.limit):
        by_task.setdefault(m["task_id"], m)

    def card(m: dict) -> str:
        path = d / "cards" / f"{m['task_id']}.json"
        if path.exists():
            return "kept"
        with _task_lock:
            registry.assign(m["task_id"], m["source"])
            test = registry.test_instance(m["task_id"], m["instance_seed"])
            demos = registry.demos(m["task_id"], m["instance_seed"], 0, 4, test)
            program = reference_program(m["task_id"])
        shown = "\n\n".join(f"Demo {k} input:\n{grid_text(p.input)}\nDemo {k} output:\n{grid_text(p.output)}"
                            for k, p in enumerate(demos, 1))
        user = ("Below are demonstration pairs of one ARC task, a test input and its correct output"
                + (", and the reference program that produces this task's outputs (Re-ARC's DSL)" if program else "")
                + ".\n\n" + shown + f"\n\nTest input:\n{grid_text(test.input)}\nTest output (correct):\n"
                + grid_text(test.output) + (f"\n\nReference program:\n{program}" if program else "")
                + "\n\nAnswer as JSON with these keys, nothing else:\n"
                  '"rule": the transformation, precisely, in two to six sentences, enough for a reader to tell whether '
                  "someone's hypothesis about it is on the right track;\n"
                  '"key_evidence": what in the grids reveals the rule, what a solver must compare or notice, in one to '
                  "three sentences;\n"
                  '"traps": two or three tempting wrong framings of this task, each with why it fails, as a list of '
                  "strings;\n"
                  '"program": Python source (standard library only) defining transform(grid) -> grid, grids as lists of '
                  "lists of ints, that implements the rule and reproduces every pair above exactly.")
        messages = [{"role": "system", "content": "You are an expert at ARC grid puzzles."}, {"role": "user", "content": user}]
        pairs = [*demos, test]
        best = None
        for attempt in range(2):   # the rule checked as a program on the demos and the test pair; once more on a failure
            # 8000 (x5 for a reasoning writer): GLM-5.3 at medium effort reasoned 15,000 tokens on 3979b1a8 and answered nothing
            answer = ask_json(messages, WRITER, max_tokens=8000)
            if not isinstance(answer, dict) or not answer.get("rule"):
                continue
            check = check_program(str(answer.get("program") or ""), pairs)
            if best is None or sum(check["ok"]) > sum(best[1]["ok"]):
                best = (answer, check)
            if all(check["ok"]):
                break
            failed = [f"demo {k}" for k, ok in enumerate(check["ok"][:-1], 1) if not ok] + ([] if check["ok"][-1] else ["the test pair"])
            messages += [{"role": "assistant", "content": json.dumps(answer)},
                         {"role": "user", "content": f"Your program does not reproduce {', '.join(failed)}"
                          + (f" ({check['error']})" if check.get("error") else "")
                          + ". Look at the grids again, correct the rule and the program, and answer in the same form."}]
        if best is None:
            return "failed"
        answer, check = best
        save(path, {"task_id": m["task_id"], "source": m["source"], "has_program": bool(program), "writer": WRITER,
                    "verified": all(check["ok"]), "reproduces": f"{sum(check['ok'])}/{len(pairs)}", **answer})
        return "verified" if all(check["ok"]) else "unverified"

    run_all(list(by_task.values()), card, a.workers)


def card_text(card: dict) -> str:
    traps = card.get("traps") or []
    traps = "\n".join(f"- {t}" for t in traps) if isinstance(traps, list) else str(traps)
    return (f"The task's actual rule: {card['rule']}\nWhat reveals it: {card.get('key_evidence', '')}\n"
            f"Tempting wrong framings:\n{traps}")


# -- hints and thoughts ------------------------------------------------------------------------------------------------

KIND_TEXT = {"incorrect": "its submission was just judged INCORRECT",
             "check_false": "one of its own checks just printed a failed comparison"}
HINT_RULES = ("never state the rule; never give an output grid, a cell value or a count the agent has not computed "
              "itself; refer only to grids and messages the agent has already seen; no tool call, no action")


def hint_prompt(card: dict, m: dict, context: str) -> list[dict[str, str]]:
    user = (f"An agent is solving one ARC grid puzzle through a Python tool, one instance of a task: it may request "
            f"demonstration pairs and submit an output grid. {card_text(card)}\n\n"
            "The conversation as the agent sees it (its system prompt left out, tool results and reasoning cut):\n"
            f"=== conversation ===\n{context}\n=== end of the conversation ===\n\n"
            f"At this point {KIND_TEXT[m['kind']]}. Write what would get it onto the right track without giving the "
            "answer: something visible in the grids or the feedback it has already seen that its current idea misses "
            "(what to compare, what to look at).\nAnswer as JSON with these keys, nothing else:\n"
            '"hint": one or two sentences addressed to the agent;\n'
            '"thought": the same pointer as the agent\'s own thinking at this point, in the first person, two to four '
            "sentences, as if it had noticed this itself.\nFor both: " + HINT_RULES + ".")
    return [{"role": "system", "content": "You help an agent solve an ARC puzzle. You know the puzzle's rule; the agent "
                                          "does not, and must find it from what it sees."},
            {"role": "user", "content": user}]


REPHRASE = ("Below is a note of what you are thinking at one point of a piece of work, where you are an agent solving "
            "grid puzzles with a Python tool. Say it again in your own words, in the first person, as your own "
            "thinking: keep every point and add nothing (no rule, no grid, nothing the note does not say), in two to "
            "four sentences. Reply with the restated thinking and nothing else: no action, no tool call, no "
            "preamble.\n\nThe note:\n")


def mean_logp(server: Any, call: Any, ids: list[int]) -> float | None:
    from ask_sd.thoughts import thought_logprob
    out = thought_logprob(server, call, ids, model=STUDENT)
    return (out or {}).get("mean")


def cmd_hints(a) -> None:
    from ask_sd.calls import Call, thought_ids
    from ask_sd.server import Server
    from transformers import AutoTokenizer
    d = Path(a.dir)
    server = Server(a.server)
    tok = AutoTokenizer.from_pretrained(loop.MODEL)
    block = loop.context_block(NOTES.read_text())

    def one(m: dict) -> str:
        path = d / "hints" / f"{m['id']}.json"
        if path.exists():
            return "kept"
        card_path = d / "cards" / f"{m['task_id']}.json"
        if not card_path.exists():
            return "no card"
        c = moment_call(m)
        answer = ask_json(hint_prompt(json.loads(card_path.read_text()), m,
                                      render_context(c["request"]["messages"], block=block)), WRITER)
        if not isinstance(answer, dict) or not answer.get("hint") or not answer.get("thought"):
            return "no hint"
        out = server.chat({"model": loop.MODEL, "messages": [{"role": "user", "content": REPHRASE + answer["thought"]}],
                           "max_tokens": 600, "temperature": 0.7, "n": 1, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}})
        worded = (((out or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        worded = worded.strip() or answer["thought"]
        call = Call.from_exchange(c["request"], c["response"])
        ids = thought_ids(tok, worded)
        own = call.reply_ids[: len(ids)] if call else []
        save(path, {"id": m["id"], "writer": WRITER, "hint": answer["hint"], "thought_authored": answer["thought"], "thought": worded,
                    "thought_tokens": len(ids), "logp_thought": mean_logp(server, call, ids) if call else None,
                    "logp_own": mean_logp(server, call, own) if call else None})
        return "written"

    run_all(load_moments(d, a.limit), one, a.workers)


# -- fair notes ---------------------------------------------------------------------------------------------------------

def fair_prompt(messages: list[dict[str, Any]], block: str) -> list[dict[str, str]]:
    """The student's view at the call, for the fair oracle to play the next step from: its system prompt and the
    conversation, the notes block taken out, only very long messages cut (at ten times the judge's caps: at four, GLM-5.3
    said the demos it read were cut)."""
    system = "\n\n".join(strip_notes(text_of(m.get("content")), block) for m in messages if m.get("role") == "system")
    context = render_context(messages, block=block, total=300_000, scale=10)
    user = ("You are the agent in the game below: you solve one ARC grid puzzle through a Python tool, and you may "
            "request demonstration pairs and submit an output grid. First your instructions, then everything you have "
            "seen and done so far in this game (long parts cut).\n\n"
            f"=== your instructions ===\n{system}\n=== end of your instructions ===\n\n"
            f"=== the game so far ===\n{context}\n=== end of the game so far ===\n\n"
            "It is your turn. Decide your next step from what you have seen. Answer as JSON with these keys, nothing "
            "else:\n"
            '"reasoning": what you notice at this point and why you take your next step, in two to six sentences;\n'
            '"plan": your next step in one sentence (what you will do, not the code);\n'
            '"step": the next step exactly as you would send it: the Python code you run, or the action JSON.')
    return [{"role": "system", "content": "You are an expert at ARC grid puzzles, playing one through a Python tool."},
            {"role": "user", "content": user}]


REFUSED = re.compile(r"(?i)terms of service|this request was blocked|violate .{0,40}usage polic")
MOVE_FIELDS = re.compile(r'"reasoning"\s*:\s*"?(.*?)"?\s*,\s*"plan"\s*:\s*"(.*?)"\s*,\s*"step"\s*:\s*(.*)', re.S)


def parse_move(text: str) -> dict | None:
    """The fair oracle's answer as a dict with its reasoning: strict JSON first; then JSON with raw newlines inside its
    strings (GLM-5.3 writes the step's code that way); then the three fields read by pattern (it also drops the
    reasoning's opening quote at times). Only the reasoning and the plan make the note; the step is kept as text."""
    from ask_sd.thoughts import first_json
    answer = first_json(text)
    if isinstance(answer, dict) and str(answer.get("reasoning") or "").strip():
        return answer
    start = text.find("{")
    if start >= 0:
        try:
            answer = json.JSONDecoder(strict=False).raw_decode(text[start:])[0]
            if isinstance(answer, dict) and str(answer.get("reasoning") or "").strip():
                return answer
        except ValueError:
            pass
    found = MOVE_FIELDS.search(text)
    if found and found.group(1).strip():
        step = re.sub(r"\s*\}?\s*(```)?\s*$", "", found.group(3)).strip()
        return {"reasoning": found.group(1).strip(), "plan": found.group(2).strip(), "step": step, "parsed": "pattern"}
    return None


def cmd_fair(a) -> None:
    """A fair note per moment: the fair oracle's reasoning and plan for its next step, worded by the base model as the
    model's own thought (as ``hints`` words its thoughts), and the note's mean log-probability under the student. The
    oracle's answer is kept under fair-raw/ first, so ``--oracle-only`` asks the oracle without a server and a later
    run words what it left. A refused request is saved under fair-refused/ and counted, never asked again in other
    words."""
    d = Path(a.dir)
    block = loop.context_block(NOTES.read_text())
    reasoning_model = "|" in FAIR and bool(FAIR.split("|")[-1])
    if not a.oracle_only:
        from ask_sd.calls import Call, thought_ids
        from ask_sd.server import Server
        from transformers import AutoTokenizer
        server = Server(a.server)
        tok = AutoTokenizer.from_pretrained(loop.MODEL)

    def one(m: dict) -> str:
        path, raw = d / "fair" / f"{m['id']}.json", d / "fair-raw" / f"{m['id']}.json"
        if path.exists():
            return "kept"
        if (d / "fair-refused" / f"{m['id']}.json").exists():
            return "refused"
        c = moment_call(m)
        if raw.exists():
            answer = json.loads(raw.read_text())
        else:
            # 40,000 for a reasoning model: GLM-5.3 at medium effort reasons up to ~15,000 tokens (the cards)
            text = oracle(FAIR).ask(fair_prompt(c["request"]["messages"], block),
                                    max_tokens=40_000 if reasoning_model else 4000,
                                    temperature=1.0 if reasoning_model else 0.0)
            answer = parse_move(text or "")
            if answer is None:
                if text and REFUSED.search(text):
                    save(d / "fair-refused" / f"{m['id']}.json", {"id": m["id"], "writer": FAIR, "text": text})
                    return "refused"
                return "no move"
            save(raw, answer)
        if a.oracle_only:
            return "asked"
        note = str(answer["reasoning"]).strip() + (" Next: " + str(answer["plan"]).strip() if answer.get("plan") else "")
        out = server.chat({"model": loop.MODEL, "messages": [{"role": "user", "content": REPHRASE + note}],
                           "max_tokens": 600, "temperature": 0.7, "n": 1, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}})
        worded = (((out or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        worded = worded.strip() or note
        call = Call.from_exchange(c["request"], c["response"])
        ids = thought_ids(tok, worded)
        own = call.reply_ids[: len(ids)] if call else []
        save(path, {"id": m["id"], "writer": FAIR, "reasoning": answer["reasoning"], "plan": answer.get("plan"),
                    "step": answer.get("step"), "thought_authored": note, "thought": worded, "thought_tokens": len(ids),
                    "logp_thought": mean_logp(server, call, ids) if call else None,
                    "logp_own": mean_logp(server, call, own) if call else None})
        return "written"

    run_all(load_moments(d, a.limit), one, a.workers)
    print(json.dumps({"oracle_calls": oracle(FAIR).calls, "usage": oracle(FAIR).usage}), flush=True)


def cmd_subset(a) -> None:
    """SRC's moments that are play calls (its /refine calls left out), with their rule cards; with --with-replies, also
    their replies and labels, so that a report here shows the earlier setups beside the new ones."""
    import shutil
    src, d = Path(a.src), Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    kept = [m for m in load_moments(src) if not is_refine(moment_call(m)["request"]["messages"])]
    with open(d / "moments.jsonl", "w") as f:
        for m in kept:
            f.write(json.dumps(m) + "\n")
    copied = Counter()
    for m in kept:
        card = src / "cards" / f"{m['task_id']}.json"
        if card.exists() and not (d / "cards" / card.name).exists():
            (d / "cards").mkdir(exist_ok=True)
            shutil.copy2(card, d / "cards" / card.name)
            copied["cards"] += 1
        if a.with_replies:
            for sub in ("replies", "judged"):
                s = src / sub / m["id"] if sub == "replies" else src / sub / f"{m['id']}.json"
                t = d / sub / s.name
                if s.exists() and not t.exists():
                    t.parent.mkdir(parents=True, exist_ok=True)
                    (shutil.copytree if s.is_dir() else shutil.copy2)(s, t)
                    copied[sub] += 1
    print(json.dumps({"moments": len(kept), "left out": sum(1 for _ in open(src / "moments.jsonl")) - len(kept), **copied}))


# -- the replay --------------------------------------------------------------------------------------------------------

def setup_request(request: dict, setup: str, *, block: str, notes: str, hint: dict | None, n: int,
                  max_tokens: int, student: str = STUDENT) -> dict | None:
    """The call's request as ``setup`` sends it: the student's exactly as served (any notes block taken out), from
    ``student``; the base model's with the notes block, with the hint inside it, with the thought opening its
    reasoning, or with the thought alone; ``fair`` as ``thought``, its thought the fair note (passed as ``hint``)."""
    p = copy.deepcopy(request)
    for k in ("stream_options", "max_completion_tokens", "store"):
        p.pop(k, None)
    p.update({"n": n, "stream": False, "max_tokens": max_tokens, "return_token_ids": True})
    for msg in p.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = content.replace(block, "")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = part["text"].replace(block, "")
    if setup == "student":
        p["model"] = student
        return p
    p["model"] = loop.MODEL
    if setup in ("hint", "thought", "thought_only", "fair") and not hint:
        return None
    if setup != "thought_only":
        text = notes.strip() + ("\n\nAbout this point of the work: " + hint["hint"].strip() if setup == "hint" else "")
        loop.inject(p, loop.context_block(text))
    if setup in ("thought", "thought_only", "fair"):
        loop.inject_thought(p, hint["thought"])
    return p


def sample_rows(out: dict | None, keep_ids: bool) -> list[dict]:
    rows = []
    for ch in (out or {}).get("choices") or []:
        msg = ch.get("message") or {}
        row = {"reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
               "content": msg.get("content") or "",
               "tool_calls": [{"name": (tc.get("function") or {}).get("name"),
                               "arguments": (tc.get("function") or {}).get("arguments")} for tc in msg.get("tool_calls") or []],
               "finish_reason": ch.get("finish_reason"), "tokens": len(ch.get("token_ids") or [])}
        if keep_ids:
            row["token_ids"] = ch.get("token_ids")
        rows.append(row)
    return rows


def cmd_replay(a) -> None:
    from ask_sd.server import Server
    d = Path(a.dir)
    server = Server(a.server)
    notes = NOTES.read_text()
    block = loop.context_block(notes)
    # a trained student (stage B) is replayed as the student setup under a label of its own
    setups = [a.label] if a.label else a.setups
    jobs = [(m, s) for m in load_moments(d, a.limit) for s in setups]

    def one(job) -> str:
        m, label = job
        path = d / "replies" / m["id"] / f"{label}.json"
        if path.exists():
            return "kept"
        setup = "student" if a.label else label
        hint_path = d / ("fair" if setup == "fair" else "hints") / f"{m['id']}.json"
        hint = json.loads(hint_path.read_text()) if hint_path.exists() else None
        req = setup_request(moment_call(m)["request"], setup, block=block, notes=notes, hint=hint, n=a.n,
                            max_tokens=a.max_tokens, student=a.student)
        if req is None:
            return f"{label}: no hint"
        # the server runs with seed 0, which gives every request without a seed of its own the same random stream: the
        # same request then gets the same reply, and n > 1 near-copies. A seed per moment and setup gives n
        # independent samples, and the same ones on a rerun
        req["seed"] = int(hashlib.sha256(f"{m['id']}:{label}".encode()).hexdigest()[:8], 16)
        rows = sample_rows(server.chat(req, retries=6, backoff=20.0), keep_ids=setup in TEACHER_SETUPS)
        if not rows:
            return f"{label}: failed"
        save(path, {"id": m["id"], "setup": label, "model": req["model"], "samples": rows})
        return label

    run_all(jobs, one, a.workers)


# -- the judge ---------------------------------------------------------------------------------------------------------

LABELS = ("same", "new_wrong", "right_track", "no_progress")


def continuation_text(s: dict) -> str:
    parts = [f"(reasoning) {cut(s['reasoning'], 3000, 1500)}" if s["reasoning"] else "(no reasoning)"]
    if s["content"].strip():
        parts.append(f"(reply) {cut(s['content'], 800, 300)}")
    for tc in s["tool_calls"][:2]:
        args = tc.get("arguments") or ""
        try:
            code = json.loads(args).get("code") or args
        except (json.JSONDecodeError, AttributeError):
            code = args
        parts.append(f"(code) {cut(code, 1500, 500)}")
    if s["finish_reason"] == "length":
        parts.append("(cut off at the length cap)")
    return "\n".join(parts)


def judge_prompt(card: dict, m: dict, context: str, chunk: list[dict]) -> list[dict[str, str]]:
    blocks = "\n\n".join(f"=== continuation {k} ===\n{continuation_text(s)}" for k, s in enumerate(chunk, 1))
    user = (f"An agent is solving one ARC grid puzzle through a Python tool. {card_text(card)}\n\n"
            "The conversation as the agent saw it (system prompt left out, tool results and reasoning cut):\n"
            f"=== conversation ===\n{context}\n=== end of the conversation ===\n\n"
            f"At this point {KIND_TEXT[m['kind']]}. Below are {len(chunk)} continuations of the agent from this point, "
            "each written independently (its next reply: reasoning, then text or code). Label each one:\n"
            "- same: it keeps the idea of the rule it held before this point (adjusting details inside that idea "
            "counts as same), or it resends a grid already judged INCORRECT;\n"
            "- new_wrong: it replaces its idea with a substantially different one that is still not the actual rule;\n"
            "- right_track: it states, tests or computes the key part of the actual rule (see what reveals it), even "
            "if incomplete;\n"
            "- no_progress: nothing about the rule (protocol only, re-reading data, empty, or cut off before any idea).\n"
            "Also mark leap: true when it asserts something that matches the actual rule or answer with nothing in "
            "the conversation or in its own continuation that would lead it there (it reads as if it knew), else "
            "false.\n"
            "Also mark good_step: true when the step it takes next (its code or its action) is a sensible move toward "
            "solving from what it has seen, judged with the actual rule in mind: for example it requests a demo it "
            "needs, tests an idea against the demos, fixes a real error, or submits a grid its own checks support; "
            "false when it resends a grid already judged INCORRECT, submits without checking, repeats a step that "
            "already failed, works on the wrong idea, gives up, or takes no step.\n\n" + blocks + "\n\n"
            'Answer as a JSON array, one object per continuation in order, nothing else: '
            '[{"k": 1, "label": "...", "leap": false, "good_step": true, "why": "<a few words>"}, ...]')
    return [{"role": "system", "content": "You judge an agent's reasoning on an ARC puzzle against the puzzle's actual "
                                          "rule. You answer in exactly the form asked."},
            {"role": "user", "content": user}]


def cmd_judge(a) -> None:
    d = Path(a.dir)
    block = loop.context_block(NOTES.read_text())

    def one(m: dict) -> str:
        """Labels every reply of the moment not labelled yet (replies added later, a trained student's, get theirs
        on a rerun), the setups mixed and unlabelled; a chunk the judge answers badly is left for the next run."""
        path = d / JUDGED / f"{m['id']}.json"
        labels = json.loads(path.read_text())["labels"] if path.exists() else []
        done = {(r["setup"], r["k"]) for r in labels if r.get("label")}
        card_path = d / "cards" / f"{m['task_id']}.json"
        replies = sorted((d / "replies" / m["id"]).glob("*.json"))
        if not card_path.exists() or not replies:
            return "not ready"
        pool = [(r["setup"], k, s) for f in replies for r in [json.loads(f.read_text())]
                for k, s in enumerate(r["samples"]) if (r["setup"], k) not in done]
        if not pool:
            return "kept"
        card = json.loads(card_path.read_text())
        random.Random(m["id"] + str(len(done))).shuffle(pool)
        context = render_context(moment_call(m)["request"]["messages"], block=block, total=30_000)
        labels = [r for r in labels if r.get("label")]
        status = "judged"
        for start in range(0, len(pool), a.chunk):
            chunk = pool[start:start + a.chunk]
            answer = ask_json(judge_prompt(card, m, context, [s for _, _, s in chunk]), JUDGE, max_tokens=2500)
            if not isinstance(answer, list) or len(answer) != len(chunk):
                status = "partly unparsed"
                continue
            for (setup, k, _), row in zip(chunk, answer):
                label = row.get("label") if isinstance(row, dict) else None
                labels.append({"setup": setup, "k": k, "label": label if label in LABELS else None,
                               "leap": bool(row.get("leap")) if isinstance(row, dict) else None,
                               "good_step": row.get("good_step") if isinstance(row, dict) else None,
                               "why": row.get("why") if isinstance(row, dict) else None})
        save(path, {"id": m["id"], "judge": JUDGE, "labels": labels})
        return status

    run_all(load_moments(d, a.limit), one, a.workers)


# -- stage B's training plays -------------------------------------------------------------------------------------------

TAKEOVER_FROM = {"takeover": "notes", "fairtake": "fair"}   # a takeover mode -> the setup whose replies it learns
THOUGHT_FROM = {"thoughts": "hints", "fair": "fair"}          # a thought mode -> the folder of its thoughts
TEACHER_SETUPS = tuple(TAKEOVER_FROM.values())                # their replies keep their token ids


def cmd_trainset(a) -> None:
    """A play folder of the moments' calls only, as train.py reads plays (captured/uNNN.json.gz, one per game):
    ``control``, each call as the student played it (on-policy training draws its own replies there); ``thoughts``
    and ``fair``, the same with the moment's thought (or fair note) on the call, which opens the teacher's reasoning;
    ``takeover``, one call per reply the notes teacher wrote at the moment (the replay's ``notes`` setup), as the reply
    played after the student's own prompt, so that off-policy training learns the teacher's own words there;
    ``fairtake``, the same with the replies the teacher wrote with the fair note opening its reasoning (``fair``), the
    note's tokens kept as the reply's opening, so that the student learns the whole reasoning in the base model's words.
    A teacher reply that cites the notes, or was cut off, is left out: it would teach the student to refer to notes it
    never sees."""
    from ask_sd.calls import thought_ids
    d, out = Path(a.dir), Path(a.out)
    games, counts = defaultdict(list), Counter()
    meta = {}
    tok = None
    for m in load_moments(d):
        c = moment_call(m)
        key = (Path(m["play"]).name, m["user"])
        meta[key] = {"task_id": m["task_id"], "solved": m["solved"]}
        if a.mode in TAKEOVER_FROM:
            rp = d / "replies" / m["id"] / f"{TAKEOVER_FROM[a.mode]}.json"
            if not rp.exists():
                counts["no teacher replies"] += 1
                continue
            opening = []
            if a.mode == "fairtake":
                if tok is None:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(loop.MODEL)
                opening = thought_ids(tok, json.loads((d / "fair" / f"{m['id']}.json").read_text())["thought"])
            for s in json.loads(rp.read_text())["samples"]:
                f = flags(s)
                if f["cites_notes"] or f["cut_off"] or not s.get("token_ids"):
                    counts["teacher reply left out"] += 1
                    continue
                choice = {"index": 0, "finish_reason": s["finish_reason"], "token_ids": opening + s["token_ids"],
                          "message": {"role": "assistant", "content": s["content"]}}
                games[key].append({"request": c["request"], "turn": c.get("turn"),
                                   "response": {"prompt_token_ids": c["response"]["prompt_token_ids"], "choices": [choice]}})
                counts["calls"] += 1
        else:
            call = {"request": c["request"], "response": c["response"], "turn": c.get("turn")}
            if a.mode in THOUGHT_FROM:
                hp = d / THOUGHT_FROM[a.mode] / f"{m['id']}.json"
                if not hp.exists():
                    counts["no thought"] += 1
                    continue
                call["thought"] = json.loads(hp.read_text())["thought"]
            games[key].append(call)
            counts["calls"] += 1
    (out / "captured").mkdir(parents=True, exist_ok=True)
    for n, (key, calls) in enumerate(sorted(games.items())):
        with gzip.open(out / "captured" / f"u{n:03d}.json.gz", "wt") as fp:
            json.dump({"user": n, "task_id": meta[key]["task_id"], "solved": meta[key]["solved"], "calls": calls,
                       "from": {"play": key[0], "user": key[1]}}, fp)
    print(json.dumps({"mode": a.mode, "games": len(games), **counts}))


# -- the report --------------------------------------------------------------------------------------------------------

def flags(s: dict) -> dict:
    text = s["reasoning"] + "\n" + s["content"]
    return {"cites_notes": bool(CITES.search(text)), "action": bool(ACTION.search(s["content"])),
            "tool_call": bool(s["tool_calls"]), "cut_off": s["finish_reason"] == "length"}


def bootstrap_ci(diffs: list[float], rng: random.Random, n: int = 2000) -> tuple[float, float]:
    if not diffs:
        return (float("nan"), float("nan"))
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def cmd_report(a) -> None:
    d = Path(a.dir)
    rows = defaultdict(lambda: defaultdict(list))   # setup -> moment -> per-sample dicts
    hints = {}
    for m in load_moments(d):
        jp = d / JUDGED / f"{m['id']}.json"
        if not jp.exists():
            continue
        judged = {(r["setup"], r["k"]): r for r in json.loads(jp.read_text())["labels"]}
        for f in (d / "replies" / m["id"]).glob("*.json"):
            r = json.loads(f.read_text())
            for k, s in enumerate(r["samples"]):
                j = judged.get((r["setup"], k))
                if j and j["label"]:
                    rows[r["setup"]][m["id"]].append({"label": j["label"], "leap": j["leap"],
                                                      "good_step": j.get("good_step") is True, "kind": m["kind"], **flags(s)})
        hp = d / "hints" / f"{m['id']}.json"
        if hp.exists():
            hints[m["id"]] = json.loads(hp.read_text())

    def share(samples, key, value=True):
        return statistics.fmean(1.0 if s[key] == value else 0.0 for s in samples) if samples else float("nan")

    extra = sorted(set(rows) - set(SETUPS))   # stage B's trained students, each replayed under its label
    lines = [f"Judge: {JUDGE} (labels in {JUDGED}/).", "",
             "| Setup | Moments | Replies | Right track | New, wrong | Same idea | No progress | Good step | Leap "
             "| Cites notes | Action | Tool call | Cut off |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for setup in (*SETUPS, *extra):
        per = rows.get(setup) or {}
        flat = [s for ss in per.values() for s in ss]
        if not flat:
            continue
        pct = lambda key, value=True: f"{100 * share(flat, key, value):.0f}%"  # noqa: E731
        lines.append(f"| {setup} | {len(per)} | {len(flat)} | {pct('label', 'right_track')} | {pct('label', 'new_wrong')} "
                     f"| {pct('label', 'same')} | {pct('label', 'no_progress')} | {pct('good_step')} | {pct('leap')} "
                     f"| {pct('cites_notes')} | {pct('action')} | {pct('tool_call')} | {pct('cut_off')} |")
    rng = random.Random(0)
    lines += ["", "Paired by moment (each moment's share of right-track replies, and of good steps, the difference "
              "averaged over the moments both setups have; 95% bootstrap intervals):", "",
              "| Comparison | Moments | Right track | 95% interval | Good step | 95% interval |", "|---|---|---|---|---|---|"]
    pairs = [("notes", "student"), ("hint", "notes"), ("thought", "notes"), ("thought", "hint"), ("thought_only", "notes"),
             ("fair", "notes"), ("fair", "student"), ("fair", "thought")]
    versus = ["student", *(a.versus or [])]   # a trained student against the student, and against same-day baselines
    for left, right in pairs + [(label, base) for label in extra for base in versus if base != label]:
        both = sorted(set(rows.get(left, {})) & set(rows.get(right, {})))
        if not both:
            continue
        cells = []
        for key, value in (("label", "right_track"), ("good_step", True)):
            diffs = [share(rows[left][i], key, value) - share(rows[right][i], key, value) for i in both]
            lo, hi = bootstrap_ci(diffs, rng)
            cells += [f"{100 * statistics.fmean(diffs):+.1f} pts", f"{100 * lo:+.1f} to {100 * hi:+.1f}"]
        lines.append(f"| {left} − {right} | {len(both)} | " + " | ".join(cells) + " |")
    for name, notes_ in (("thoughts", hints.values()),
                         ("fair notes", [json.loads(f.read_text()) for f in (d / "fair").glob("*.json")])):
        gaps = [(h["logp_thought"] - h["logp_own"]) for h in notes_
                if h.get("logp_thought") is not None and h.get("logp_own") is not None]
        if gaps:
            lines += ["", f"The {name}' mean log-probability per token under the student, minus its own reasoning's at "
                          f"the same call: median {statistics.median(gaps):+.2f} over {len(gaps)} (0: as likely as its "
                          f"own words)."]
    refused = len(list((d / "fair-refused").glob("*.json")))
    if refused or (d / "fair").exists():
        lines += ["", f"Fair notes: {len(list((d / 'fair').glob('*.json')))} written, {refused} refused by the oracle."]
    cards = [json.loads(f.read_text()) for f in (d / "cards").glob("*.json")]
    lines += ["", f"Rule cards checked as programs on the demos and the test pair: {sum(bool(c.get('verified')) for c in cards)} "
                  f"of {len(cards)} reproduce every pair."]
    kinds = Counter(m["kind"] for m in load_moments(d))
    lines += ["", f"Moments by kind: {dict(kinds)}."]
    text = "\n".join(lines)
    (d / "report.md").write_text(text + "\n")
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("moments")
    s.add_argument("dir")
    s.add_argument("plays", nargs="+")
    s.add_argument("--per-game", type=int, default=2)
    s.add_argument("--max", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("trainset")
    s.add_argument("dir")
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=("control", "thoughts", "takeover", "fair", "fairtake"), required=True)
    s = sub.add_parser("subset")
    s.add_argument("src")
    s.add_argument("dir")
    s.add_argument("--with-replies", action="store_true")
    for name in ("cards", "hints", "fair", "replay", "judge", "report"):
        s = sub.add_parser(name)
        s.add_argument("dir")
        s.add_argument("--limit", type=int, default=None, help="the first N moments only (a smoke run)")
        s.add_argument("--workers", type=int, default=8)
        if name in ("hints", "fair", "replay"):
            s.add_argument("--server", default="http://localhost:18101/v1")
        if name == "fair":
            s.add_argument("--oracle-only", action="store_true", help="ask the oracle only (no server): fair-raw/")
        if name == "report":
            s.add_argument("--versus", nargs="*", default=[], help="labels each trained student is also paired against")
        if name == "replay":
            s.add_argument("--setups", nargs="+", choices=SETUPS, default=list(SETUPS))
            s.add_argument("--n", type=int, default=8)
            s.add_argument("--max-tokens", type=int, default=REPLAY_MAX_TOKENS)
            s.add_argument("--student", default=STUDENT, help="the served name of the student the student setup asks")
            s.add_argument("--label", default=None,
                           help="replay --student as the student setup under this label (a stage B block's adapter)")
        if name == "judge":
            s.add_argument("--chunk", type=int, default=8)
    a = p.parse_args()
    {"moments": cmd_moments, "cards": cmd_cards, "hints": cmd_hints, "fair": cmd_fair, "replay": cmd_replay,
     "judge": cmd_judge, "trainset": cmd_trainset, "report": cmd_report, "subset": cmd_subset}[a.cmd](a)


if __name__ == "__main__":
    main()
