"""A paper row's learning curve beside the paper's own baselines, to read after every round: the
share of a visit's budget (``continual_arc.env.EnvConfig().max_cost``, 16) the row spent

- per round: every instance it played that round, how many it solved, and its spend on a user's first visit to a
  task and on the returning ones;
- by visit k (a user's own k-th visit to a task, ``user_exposure_index``) over every round so far: the measure the
  paper's exposure curves and the launch's chart plot.

The baselines are the paper's runs of the program library, each harness with its memory kept and wiped, and the bare
model, on GPT-6 Luna, each on one user's stream (the individual track): pooled over the stream's three orders (ws0,
seq0, rnd0) as the launch's chart pools them, and on the workstreams order
alone (ws0), the order a row plays. A row with more users (each walking a stream like theirs) is set beside them as
the launch's chart sets it: the difference is stated, never corrected for. User 0's stream is exactly the baselines'
ws0 stream, so its instances are also set beside the baselines' own runs of the same instances.

The baselines' records come from a checkout of the run records (https://github.com/unifyai/continual-arc-runs), whose
``runs/arc1/<prefix>-luna6-<effort>-<order>/records.jsonl`` holds each one (:data:`BASELINES`).

Usage: python3 paper_compare.py RECORDS.jsonl --results ~/continual-arc-runs [--effort medium] [--name DAgger]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

VISITS = 8   # a task's visits per user in arc1-small's stream, served to any number of users
ORDERS = ("ws0", "seq0", "rnd0")
BASELINES = {   # the column's name: the paper's run prefix (the run records' runs/arc1/<prefix>-luna6-<effort>-<order>)
    "program library": "M-BareProgramLibraryLearner",
    "Prime Agent": "H-prime-agent",
    "Hermes Agent": "H-hermes",
    "OpenClaw": "H-openclaw",
    "Prime Agent wiped": "H0-prime-agent",
    "Hermes wiped": "H0-hermes",
    "OpenClaw wiped": "H0-openclaw",
    "bare model": "F-BareModelLearner",
}


def stream_records(path: Path) -> list[dict[str, Any]]:
    """A run's stream instances (``records.jsonl``): the benchmark's episodes outside the epilogue, a trailing partial
    line left out (a round still being written)."""
    out = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("phase") == "stream" and r.get("episode_type", "benchmark") == "benchmark":
            out.append(r)
    return out


def share(recs: list[dict[str, Any]], max_cost: float) -> float | None:
    """The mean share of a visit's budget the records spent, in percent; None for no record."""
    return 100 * sum(r["cost"] for r in recs) / max_cost / len(recs) if recs else None


def by_visit(recs: list[dict[str, Any]], max_cost: float) -> list[tuple[float | None, int]]:
    """For visit k = 1..VISITS (``user_exposure_index`` k - 1), the share of the budget spent and the instances."""
    out = []
    for k in range(VISITS):
        rs = [r for r in recs if r["user_exposure_index"] == k]
        out.append((share(rs, max_cost), len(rs)))
    return out


def by_round(recs: list[dict[str, Any]], max_cost: float) -> list[dict[str, Any]]:
    """Per round: the instances, the share solved, the share of the budget spent, on first visits and returning."""
    rounds: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in recs:
        rounds[r["round"]].append(r)
    return [{"round": n, "instances": len(rs), "solved": 100 * sum(bool(r["solved"]) for r in rs) / len(rs),
             "spent": share(rs, max_cost),
             "first": share([r for r in rs if r["user_exposure_index"] == 0], max_cost),
             "returning": share([r for r in rs if r["user_exposure_index"] > 0], max_cost),
             "returning_n": sum(r["user_exposure_index"] > 0 for r in rs)}
            for n, rs in sorted(rounds.items())]


def baseline_records(results: Path, effort: str, orders: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    """Each baseline's stream records over ``orders``; a baseline missing any of their runs is left out."""
    out = {}
    for name, prefix in BASELINES.items():
        runs = [results / "runs" / "arc1" / f"{prefix}-luna6-{effort}-{o}" / "records.jsonl" for o in orders]
        if all(p.exists() for p in runs):
            out[name] = [r for p in runs for r in stream_records(p)]
    return out


def baseline_curves(baselines: dict[str, list[dict[str, Any]]], max_cost: float) -> dict[str, list[float | None]]:
    """Each baseline's share of the budget by visit."""
    return {name: [s for s, _ in by_visit(recs, max_cost)] for name, recs in baselines.items()}


def same_instances(recs: list[dict[str, Any]], baselines: dict[str, list[dict[str, Any]]],
                   max_cost: float) -> dict[str, tuple[float | None, int, int]]:
    """On the instances user 0 has played (the baselines' own ws0 stream, instance for instance), the row's and each
    baseline's share of the budget spent, instances solved and instances: a baseline's run of an instance is matched
    by its task and seed."""
    ours = [r for r in recs if r["user"] == 0]
    keys = {(r["task_id"], r["instance_seed"]) for r in ours}
    out = {"row": (share(ours, max_cost), sum(bool(r["solved"]) for r in ours), len(ours))}
    for name, theirs in baselines.items():
        same = [r for r in theirs if (r["task_id"], r["instance_seed"]) in keys]
        out[name] = (share(same, max_cost), sum(bool(r["solved"]) for r in same), len(same))
    return out


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def visit_table(recs: list[dict[str, Any]], curves: dict[str, list[float | None]], max_cost: float,
                name: str) -> list[str]:
    lines = ["| visit | " + name + " (n) | " + " | ".join(curves) + " |", "|---|---|" + "---|" * len(curves)]
    for k, (ours, n) in enumerate(by_visit(recs, max_cost)):
        lines.append(f"| {k + 1} | {fmt(ours)} ({n}) | " + " | ".join(fmt(c[k]) for c in curves.values()) + " |")
    return lines


def report(recs: list[dict[str, Any]], pooled: dict[str, list[dict[str, Any]]],
           ws0: dict[str, list[dict[str, Any]]], max_cost: float, name: str, effort: str) -> str:
    rounds = by_round(recs, max_cost)
    lines = [f"## {name}: {len(recs)} instances over rounds 0-{rounds[-1]['round'] if rounds else '-'}", "",
             "Per round (share of a visit's budget of 16 spent, percent):", "",
             "| round | instances | solved | spent | first visits | returning visits (n) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['round']} | {r['instances']} | {r['solved']:.0f}% | {fmt(r['spent'])} | {fmt(r['first'])} | "
              f"{fmt(r['returning'])} ({r['returning_n']}) |" for r in rounds]
    lines += ["", f"By visit, every round so far, beside the paper's GPT-6 Luna ({effort} effort) baselines, each on one "
                  "user's stream, pooled over its three orders:", ""]
    lines += visit_table(recs, baseline_curves(pooled, max_cost), max_cost, name)
    lines += ["", "The same on the workstreams order alone (ws0), the order this row plays:", ""]
    lines += visit_table(recs, baseline_curves(ws0, max_cost), max_cost, name)
    same = same_instances(recs, ws0, max_cost)
    lines += ["", "User 0's instances so far, which are the baselines' ws0 stream instance for instance (share of the "
                  "budget spent, solved of played):", "",
              "| " + " | ".join([name, *ws0]) + " |", "|" + "---|" * (len(ws0) + 1),
              "| " + " | ".join(f"{fmt(s)} ({k}/{n})" for s, k, n in same.values()) + " |"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("records")
    p.add_argument("--results", required=True,
                   help="a checkout of the run records (https://github.com/unifyai/continual-arc-runs): the baselines' "
                        "records are read from its runs/arc1/<prefix>-luna6-<effort>-<order>/records.jsonl")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--name", default="DAgger")
    a = p.parse_args()
    from continual_arc.env import EnvConfig
    max_cost = EnvConfig().max_cost
    recs = stream_records(Path(a.records).expanduser())
    results = Path(a.results).expanduser()
    print(report(recs, baseline_records(results, a.effort, ORDERS), baseline_records(results, a.effort, ("ws0",)),
                 max_cost, a.name, a.effort))


if __name__ == "__main__":
    main()
