"""Screen ARC-GEN's generators for the second track and record the pool the benchmark reads.

    uv run continual-arc data fetch --datasets arc2
    uv run python scripts/screen_arcgen.py [--n 1000] [--jobs 6] [--out src/continual_arc/arcgen_screen.json]

Every ARC-AGI-2 training task new to the second edition (``data.arc2_new_task_ids``) that ARC-GEN has a generator
for is screened (``arcgen.screen_task``: ARC-GEN's validate reproduces the task's published pairs, every draw keeps
what the published pairs have alike, and the held-out season's mechanical filters hold over ``--n`` draws), each in
a worker process of its own, so that a call's CPU backstop is a timer in that process's main thread. The record
names the pinned commits it ran on, the counts, the pool (the tasks that passed) and every task's readings; the
script prints one line for each task that failed, then the counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

from continual_arc import arcgen
from continual_arc.data import (
    ARC2_COMMIT,
    ARCGEN_COMMIT,
    arc2_new_task_ids,
    load_arc_tasks,
    require_data,
)


@cache
def _training() -> dict[str, dict]:
    return load_arc_tasks("training", dataset="arc2")


def _screen(args: tuple[str, int]) -> tuple[str, dict]:
    task_id, n = args
    return task_id, arcgen.screen_task(arcgen.ArcGen(), task_id, _training()[task_id], n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=arcgen.N_DRAWS, help="draws per generator")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", type=Path, default=arcgen.SCREEN_FILE)
    a = parser.parse_args(argv)
    require_data("arc2")
    covered = arcgen.ArcGen().task_ids
    new = arc2_new_task_ids("training")
    screened = sorted(set(new) & set(covered))
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        tasks = dict(pool.map(_screen, [(t, a.n) for t in screened]))
    passed = sorted(t for t, r in tasks.items() if r["ok"])
    record = {
        "arc_gen": ARCGEN_COMMIT,
        "arc_agi_2": ARC2_COMMIT,
        "draws": a.n,
        "filters": {
            "max_random": arcgen.MAX_RANDOM, "cpu_seconds_per_draw": arcgen.CPU_SECONDS_PER_DRAW,
            "min_input_side": arcgen.MIN_INPUT_SIDE, "min_distinct_inputs": arcgen.MIN_DISTINCT_INPUTS,
            "min_distinct_outputs": arcgen.MIN_DISTINCT_OUTPUTS, "max_identity": arcgen.MAX_IDENTITY,
            "repeated_draws": arcgen.N_REPEAT,
        },
        "counts": {
            "arc_gen_generators": len(covered),
            "arc_agi_2_training_covered": len(set(covered) & set(_training())),
            "new_to_second_edition": len(new),
            "screened": len(screened),
            "validate_reproduces": sum(r["faithful"] for r in tasks.values()),
            "passed": len(passed),
        },
        "pool": passed,
        "tasks": tasks,
    }
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    for task_id, r in tasks.items():
        if not r["ok"]:
            print(f"FAIL {task_id}: {'; '.join(r['reasons'])}")
    print(json.dumps(record["counts"]))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
