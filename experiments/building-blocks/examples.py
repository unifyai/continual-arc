"""How the building-block guidance's examples were chosen, and the check that they still hold (README.md, "The
examples"). The author, 8 Nov 2026: "if we could actually find some primitives among the 160 in the rearc generators that are
not used in any of those 25, 50, or 100 tasks, then we could even include some examples in context of the kind of
primitives that are worth storing ... We could even then potentially use one or two of the actual rules that, again,
don't appear in training or held out".

A task's rule is its verifier in Re-ARC's verifiers.py, a program of the DSL's 160 primitives (dsl.py), which name
their primitives as plain identifiers. This prints how many primitives the rules of the stream's tasks use at one, two
and four users (each user's 25 tasks, the four-user schedule's own lists, user 0's being the one-user stream's), and
which no rule uses; then checks the examples continual-arc-research shows (``protocol.BLOCK_LIBRARY_EXAMPLES``): every
function they define is a primitive no rule of the one-user stream uses, each program's task is a training task in no
user's stream (the held-out tasks are ARC-AGI-1's evaluation tasks, which Re-ARC has no rules for), and each program
reproduces every pair ARC gives its task. Exits 1 when a check fails.

    python blocks/examples.py [--rearc ~/.cache/continual-arc/re-arc] [--arc ~/.cache/continual-arc/ARC-AGI/data/training]
        [--schedule ~/continual-arc-results/runs/gpu-nodes/REF_ROWS_NODE_E/glm-hpl-u4-own/play/schedule.json]

run where continual_arc_baselines with the guidance imports (~/cab-blocks/.venv/bin/python on the machine).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path

EXAMPLE_TASKS = ("7468f01a", "b91ae062")   # the programs' tasks, in the order the examples show them


def rules(rearc: Path) -> tuple[list[str], dict[str, set[str]]]:
    """The DSL's primitives in dsl.py's order, and the primitives each training task's rule (its verifier) uses."""
    prims = re.findall(r"^def (\w+)\(", (rearc / "dsl.py").read_text(), re.M)
    source = (rearc / "verifiers.py").read_text()
    used = {m.group(1): {p for p in prims if re.search(rf"\b{p}\b", m.group(0))}
            for m in re.finditer(r"^def verify_(\w+)\(.*?(?=^def |\Z)", source, re.M | re.S)}
    return prims, used


def users_tasks(schedule: Path) -> list[list[str]]:
    """Each user's stream tasks in the four-user disjoint schedule, user 0's being the one-user stream's."""
    return json.loads(schedule.read_text())["meta"]["users"]["tasks"]


def programs(examples: str) -> tuple[types.ModuleType, list]:
    """The examples' functions as the module lib, and each task's transform composed from it, in the order shown."""
    code = examples.split("```python\n", 1)[1].rsplit("```", 1)[0]
    functions, rest = code.split("# one task's program", 1)
    lib = types.ModuleType("lib")
    exec(functions, lib.__dict__)
    sys.modules["lib"] = lib
    first, *others = ("# one task's program" + rest).split("# another's")
    shown = []
    for part in (first, *("# another's" + other for other in others)):
        scope: dict = {}
        exec(part, scope)
        shown.append(scope["transform"])
    return lib, shown


def main() -> None:
    from continual_arc_baselines.systems.protocol import BLOCK_LIBRARY_EXAMPLES

    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rearc", type=Path, default=home / ".cache/continual-arc/re-arc")
    ap.add_argument("--arc", type=Path, default=home / ".cache/continual-arc/ARC-AGI/data/training")
    ap.add_argument("--schedule", type=Path, default=home / "continual-arc-results/runs/gpu-nodes/REF_ROWS_NODE_E/"
                    "glm-hpl-u4-own/play/schedule.json")
    a = ap.parse_args()
    prims, used = rules(a.rearc)
    users = users_tasks(a.schedule)
    free = {}
    for n in (1, 2, 4):
        tasks = [t for u in users[:n] for t in u]
        free[n] = [p for p in prims if not any(p in used[t] for t in tasks)]
        print(f"{n} user(s), {len(tasks)} tasks: their rules use {len(prims) - len(free[n])} of the {len(prims)} "
              f"primitives; none uses {', '.join(free[n])}")
    failures = []
    lib, shown = programs(BLOCK_LIBRARY_EXAMPLES)
    names = [n for n in vars(lib) if isinstance(getattr(lib, n), types.FunctionType)]
    taken = [n for n in names if n not in free[1]]
    print(f"the examples' functions: {', '.join(names)}; free at one user: {not taken}; free at two and four users: "
          f"{', '.join(n for n in names if n in free[4]) or 'none'}")
    if taken:
        failures.append(f"the examples define primitives that the one-user stream's rules use: {', '.join(taken)}")
    streamed = {t for u in users for t in u}
    for task, transform in zip(EXAMPLE_TASKS, shown, strict=True):
        pairs = (lambda d: d["train"] + d["test"])(json.loads((a.arc / f"{task}.json").read_text()))
        right = sum(transform(p["input"]) == p["output"] for p in pairs)
        print(f"{task}: a training task {'in a user' + chr(39) + 's stream' if task in streamed else 'in no stream'}, "
              f"its program reproduces {right} of its {len(pairs)} pairs")
        if task in streamed or task not in used or right != len(pairs):
            failures.append(f"{task}: in a stream, not a training rule, or {right} of {len(pairs)} pairs")
    for f in failures:
        print("FAILED:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
