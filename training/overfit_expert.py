"""The recall overfit's dynamic teacher thoughts (overfit.py): on-policy distillation in which the teacher reads, at every
call, the task's card and then a correction for the state the student is in, so it keeps steering the student back to
the card's procedure wherever the student strayed (a dynamic prompt for the teacher, still OPSD).

A scripted expert decides the right next move at any state of one of the five tasks' games. It knows each task's
verified program (overfit_cards.json) and reads the call's messages:

- the game is over (a correct verdict, or the environment says so): finish;
- no demo pair yet: request one (c1's rule 3, the card's plan);
- a printed grid in a tool result equal to the program's output on the latest test input: submit that grid;
- a message that bars tool calls ("Decide now") with no such grid printed: request a demo (c1's rule 2);
- otherwise: one ipython call that loads every grid with c1's loader, defines the task's ``transform``, checks it on
  each demo pair and prints ``ANSWER`` and the test output.

The move is written as the teacher's own thought after the card, in the first person, with the exact action line or
code, so the teacher's next tokens put their weight on taking the step rather than describing it (most rejected
replies announce the next step and end the turn). ``thoughts`` writes a play's calls with that thought on each
(train.py's per-call thought, opening the teacher's reasoning after the run's context, c1).

Its :func:`latest_test_input` and :func:`run_program` also serve oracle_hint.py's state and checks.

Subcommands (in prime-rl's environment on the node, or any with ask_sd importable):
  thoughts PLAY... --cards-dir DIR --c1 FILE --out DIR   every call with the card and the expert's move as its thought
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import overfit  # noqa: E402

ANSWER = "ANSWER"
GRID_HEAD = re.compile(r"Test input \((\d+)x(\d+)\):\n")
OVER = ("Your submission was CORRECT", "The instance is over")
NO_TOOLS = ("Decide now",)
DEMO_SEEN = re.compile(r"Demo pair \d+ input \(\d+x\d+\):")
GRID_JSON = re.compile(r"\[\s*\[[\d,\s\[\]]*\]\s*\]")


def c1_loader(c1: str) -> str:
    """The grid loader c1 teaches (its rule 4's code block), word for word."""
    return re.search(r"```\n(.*?)```", c1, re.S).group(1)


def expert_code(loader: str, program: str) -> str:
    return (loader.rstrip() + "\n\n" + program.rstrip() + "\n\n"
            "import json\n"
            "pairs = sorted({int(k.split()[2]) for k in G if k.startswith('Demo pair')})\n"
            "ok = [transform(G[f'Demo pair {k} input']) == G[f'Demo pair {k} output'] for k in pairs]\n"
            "print('demo pairs', pairs, 'match', ok)\n"
            f"print('{ANSWER}', json.dumps(transform(G['Test input']), separators=(',', ':')))\n")


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(p.get("text") or "" for p in content if isinstance(p, dict))
    return content or ""


def latest_test_input(messages: list[dict]) -> list[list[int]] | None:
    """The test input grid as the newest message that shows it gives it."""
    for m in reversed(messages):
        text = text_of(m)
        found = list(GRID_HEAD.finditer(text))
        if not found:
            continue
        h, w = int(found[-1][1]), int(found[-1][2])
        rows = [r.split() for r in text[found[-1].end():].split("\n")[:h]]
        if len(rows) == h and all(len(r) == w and all(x.isdigit() for x in r) for r in rows):
            return [[int(x) for x in r] for r in rows]
    return None


def run_program(program: str, grid: list[list[int]]) -> list[list[int]] | None:
    ns: dict[str, Any] = {}
    try:
        exec(program, ns)  # noqa: S102  (the cards' own verified programs)
        return [[int(x) for x in row] for row in ns["transform"]([row[:] for row in grid])]
    except Exception:  # noqa: BLE001
        return None


def printed(messages: list[dict], grid: list[list[int]]) -> bool:
    """Whether a tool result shows ``grid`` as JSON (the expert's ANSWER line, or the student's own print)."""
    for m in messages:
        if m.get("role") != "tool":
            continue
        for found in GRID_JSON.findall(text_of(m)):
            try:
                if json.loads(found) == grid:
                    return True
            except ValueError:
                continue
    return False


def action_line(action: dict) -> str:
    return json.dumps(action, separators=(", ", ": "))


def demo_shown(messages: list[dict]) -> bool:
    return any(DEMO_SEEN.search(text_of(m)) for m in messages)


def expert_move(messages: list[dict], program: str, loader: str, demos_before: bool = False) -> tuple[str, str]:
    """The right next move after ``messages``: its kind and, in the first person, what the reply does (with the exact
    action line or code). ``demos_before``: an earlier call of the game showed a demo pair, which the harness's
    compaction may since have taken out of the messages (3,105 of 4,393 demo requests the expert asked for came
    after a demo had been shown; c1's loader reads the session's whole log, so the code still finds the demos)."""
    last = text_of(messages[-1]) if messages else ""
    if any(o in last for o in OVER):
        return "finish", ("The instance is solved. I reply with only the action line, nothing else:\n"
                          + action_line({"action": "finish"}))
    test = latest_test_input(messages)
    expected = run_program(program, test) if test else None
    demos = demos_before or demo_shown(messages)
    if not demos:
        return "request_demos", ("I have no demo pair yet, so this reply makes no tool call and ends with this action "
                                 "line:\n" + action_line({"action": "request_demos"}))
    if expected is not None and printed(messages, expected):
        return "submit", ("My transform's output for the test input is printed above. I do not describe the next step: "
                          "this reply ends now with the submission, the printed grid copied exactly, on the last line:\n"
                          + action_line({"action": "submit", "grid": expected}))
    if any(n in last for n in NO_TOOLS):
        return "request_demos", ("I must reply without a tool call, and my transform's output is not printed yet, so "
                                 "this reply ends with this action line:\n" + action_line({"action": "request_demos"}))
    return "code", ("I have a demo pair. I do not describe the step, I take it: one ipython call now that loads every "
                    "grid, defines my transform, checks it on each demo pair and prints the test output:\n```python\n"
                    + expert_code(loader, program) + "```")


def cmd_thoughts(a: argparse.Namespace) -> None:
    cards = overfit.load_cards(Path(a.cards))
    card_text = {t: Path(a.cards_dir, f"card-{t}.md").read_text().strip() for t in overfit.TASKS}
    loader = c1_loader(Path(a.c1).read_text())
    out = Path(a.out)
    stats: collections.Counter = collections.Counter()
    for run in a.runs:
        run_dir = Path(run)
        for name, doc in overfit.read_docs(run_dir):
            doc = copy.deepcopy(doc)
            task = doc["task_id"]
            doc["calls"] = [c for c in doc["calls"] if overfit.chat_call(c)]
            first = overfit.agent_calls(doc)[:1]
            seen = False   # a demo pair shown at any call so far (the calls are in the game's order)
            for c in doc["calls"]:
                kind, move = expert_move(c["request"]["messages"], cards[task]["program"], loader, demos_before=seen)
                seen = seen or demo_shown(c["request"]["messages"])
                head = overfit.RESTATE + " " if any(c is f for f in first) else ""
                c["thought"] = card_text[task] + "\n\n" + head + move
                stats[f"kind:{kind}"] += 1
                stats["calls"] += 1
            overfit.write_doc(out, f"u-{run_dir.name}-{name}", doc)
    (out / "thoughts.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("thoughts")
    s.add_argument("runs", nargs="+")
    s.add_argument("--cards", default=str(HERE / "overfit_cards.json"))
    s.add_argument("--cards-dir", required=True)
    s.add_argument("--c1", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_thoughts)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
