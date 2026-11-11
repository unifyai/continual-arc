"""The library ceiling's modules (README.md): the Re-ARC DSL's primitives that the rules of the stream's tasks use, as one
module each for the one-user and the four-user stream, which continual-arc-research's baselines show GLM-5.3 Flash
and load into its Python kernel (``systems/primitives/stream_u1.py`` and ``stream_u4.py``, as ``arc_dsl``). The author, 8
Nov 2026: "we only show it the functions that are involved in the creation of the tasks that were seen during the
stream. And then it's more kind of like a maximum ceiling on what harness-based learning could deduce."

A task's rule is its verifier in Re-ARC's verifiers.py, a program of the DSL's 160 primitives (dsl.py), which name
their primitives as plain identifiers. A stream's module is dsl.py's header (its typing imports, type aliases and
constants) and, in dsl.py's order, every primitive the rules of the stream's tasks use, with every primitive those call
in turn, so that it imports whole: 131 at one user (the 25 tasks' rules use 122) and 153 at four (100 tasks). Each
stream task's rule, run on nothing but its module, must give on every pair ARC gives the task what it gives on the whole
of dsl.py. Three of the rules (7e0986d6, e5062a87 and 29ec7d0e) miss one or two of ARC's original pairs on the whole
dsl.py too: Re-ARC wrote its verifiers for its generators' instances, which the benchmark serves, so this is said, not
failed.

    python ceiling/primitives.py write DIR    the modules into DIR (the baselines' systems/primitives/), then check them
    python ceiling/primitives.py check DIR    that DIR's modules are what this writes, and that they hold

Options: --rearc (~/.cache/continual-arc/re-arc, the copy the benchmark pins), --arc (ARC-AGI-1's training tasks),
--schedule (the four-user disjoint stream, whose users' task lists, user 0's being the one-user stream's, name each
stream's tasks). Exits 1 when a check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

STREAMS = {"stream_u1": 1, "stream_u4": 4}   # each module by the number of users whose stream's rules it serves
NOTICE = "# From Re-ARC's dsl.py, Copyright (c) 2024 Michael Hodel, MIT License.\n"


def dsl(rearc: Path) -> tuple[str, dict[str, str]]:
    """dsl.py's header, everything before its first function, and each primitive's source by name, in its order."""
    source = (rearc / "dsl.py").read_text()
    first = source.index("\ndef ") + 1
    return source[:first].rstrip() + "\n", {m.group(1): m.group(0).rstrip() + "\n" for m in
                                            re.finditer(r"^def (\w+)\(.*?(?=^def |\Z)", source[first:], re.M | re.S)}


def rules(rearc: Path) -> dict[str, str]:
    """Each training task's rule, its verifier's source, by task id."""
    return {m.group(1): m.group(0) for m in
            re.finditer(r"^def verify_(\w+)\(.*?(?=^def |\Z)", (rearc / "verifiers.py").read_text(), re.M | re.S)}


def uses(code: str, names) -> set[str]:
    return {n for n in names if re.search(rf"\b{n}\b", code)}


def module(header: str, prims: dict[str, str], tasks: list[str], verifiers: dict[str, str]) -> tuple[str, set, set]:
    """The module for ``tasks``: the header, then every primitive their rules use and every primitive those call, in
    dsl.py's order; with the primitives the rules use, and all it holds."""
    used = set().union(*(uses(verifiers[t], prims) for t in tasks))
    held, todo = set(), list(used)
    while todo:
        p = todo.pop()
        if p not in held:
            held.add(p)
            todo += [q for q in uses(prims[p], prims) if q != p]
    body = "\n\n".join(prims[p].rstrip() for p in prims if p in held)
    return NOTICE + header + "\n\n" + body + "\n", used, held


def run(space: dict, verifier: str, task: str, grid: list) -> list | str:
    """The task's rule run in ``space`` on ``grid``, as lists, or what it raised."""
    scope = dict(space)
    exec(verifier, scope)
    try:
        return [list(row) for row in scope[f"verify_{task}"](tuple(tuple(row) for row in grid))]
    except Exception as error:  # noqa: BLE001 - what a rule raises is compared, as its output is
        return f"raised {type(error).__name__}"


def holds(text: str, whole: str, tasks: list[str], verifiers: dict[str, str], arc: Path) -> tuple[list[str], list[str]]:
    """What fails of ``text``: it does not import, or a task's rule run on it alone gives on one of ARC's pairs what it
    does not give on the ``whole`` of dsl.py; and the tasks whose rule misses one of ARC's pairs either way."""
    space: dict = {}
    try:
        exec(compile(text, "arc_dsl.py", "exec"), space)
    except Exception as error:  # noqa: BLE001 - any failure to import is the finding
        return [f"does not import: {error!r}"], []
    full: dict = {}
    exec(compile(whole, "dsl.py", "exec"), full)
    failures, missed = [], []
    for task in tasks:
        data = json.loads((arc / f"{task}.json").read_text())
        pairs = data["train"] + data["test"]
        mine = [run(space, verifiers[task], task, p["input"]) for p in pairs]
        if mine != [run(full, verifiers[task], task, p["input"]) for p in pairs]:
            failures.append(f"{task}: its rule gives otherwise on the module than on the whole of dsl.py")
        if any(m != p["output"] for m, p in zip(mine, pairs, strict=True)):
            missed.append(task)
    return failures, missed


def main() -> None:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("write", "check"))
    ap.add_argument("dir", type=Path)
    ap.add_argument("--rearc", type=Path, default=home / ".cache/continual-arc/re-arc")
    ap.add_argument("--arc", type=Path, default=home / ".cache/continual-arc/ARC-AGI/data/training")
    ap.add_argument("--schedule", type=Path, default=home / "continual-arc-results/runs/gpu-nodes/REF_ROWS_NODE_E/"
                    "glm-hpl-u4-own/play/schedule.json")
    a = ap.parse_args()
    header, prims = dsl(a.rearc)
    verifiers = rules(a.rearc)
    users = json.loads(a.schedule.read_text())["meta"]["users"]["tasks"]
    print(f"Re-ARC: dsl.py sha256 {hashlib.sha256((a.rearc / 'dsl.py').read_bytes()).hexdigest()[:12]}, "
          f"{len(prims)} primitives; verifiers.py sha256 "
          f"{hashlib.sha256((a.rearc / 'verifiers.py').read_bytes()).hexdigest()[:12]}")
    failures = []
    for name, n in STREAMS.items():
        tasks = [t for u in users[:n] for t in u]
        text, used, held = module(header, prims, tasks, verifiers)
        path = a.dir / f"{name}.py"
        if a.action == "write":
            path.write_text(text)
        elif not path.exists() or path.read_text() != text:
            failures.append(f"{path}: not what the Re-ARC copy and the stream give")
        wrong, missed = holds(text, (a.rearc / "dsl.py").read_text(), tasks, verifiers, a.arc)
        failures += [f"{name}: {w}" for w in wrong]
        print(f"{name}: the rules of the {len(tasks)} tasks of the {n}-user stream use {len(used)} primitives, "
              f"{len(held)} with those they call; {len(text)} characters; every rule gives on it alone what it gives "
              f"on the whole of dsl.py: {not wrong}; rules that miss one of ARC's original pairs either way: "
              f"{', '.join(missed) or 'none'}")
    for f in failures:
        print("FAILED:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
