"""Check a run submitted to the Continual-ARC leaderboard, and print the numbers it would show.

    uv run python scripts/check_result.py results/<track>/<name> [more runs...]

A submitted run is the directory the benchmark's runner wrote (`records.jsonl`, `summary.json`, `schedule.json`,
`config.json`, `meta.json`) with a `submission.json` beside them. The check passes when the schedule is the one the
benchmark builds for the run's configuration, order and seed, the records cover every slot of it in order, the
summary is the one the records give, and `submission.json` names what the leaderboard shows. It then prints the
budget used on the stream and on the held-out epilogue (100 times the mean charge over the cap, lower is better) and
the share solved within the cap, which are the leaderboard's numbers. Nothing here calls a model or the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from continual_arc.configs import build_schedule, protocol_of
from continual_arc.metrics import summarize
from continual_arc.runner import load_run

RUN_FILES = ("records.jsonl", "summary.json", "schedule.json", "config.json", "meta.json")
SUBMISSION_FIELDS = ("name", "learner", "model", "code", "contact")


def budget(records, cap: float) -> tuple[float, float] | None:
    if not records:
        return None
    used = 100 * sum(r.cost for r in records) / len(records) / cap
    solved = 100 * sum(1 for r in records if r.solved) / len(records)
    return used, solved


def check(run_dir: Path) -> list[str]:
    problems = [f"missing {f}" for f in (*RUN_FILES, "submission.json") if not (run_dir / f).exists()]
    if problems:
        return problems
    schedule, config, records = load_run(run_dir)
    protocol = protocol_of(schedule, config)
    expected = build_schedule(protocol["config"], protocol["mode"], protocol["seed"],
                              epilogue=bool(protocol.get("epilogue", True)))
    if [s.to_json() for s in schedule.slots] != [s.to_json() for s in expected.slots]:
        problems.append(f"schedule.json is not the schedule the benchmark builds for {protocol['config']}, "
                        f"{protocol['mode']}, seed {protocol['seed']}")
    benchmark = [r for r in records if r.episode_type == "benchmark"]
    if len(benchmark) != len(expected.slots):
        problems.append(f"{len(benchmark)} records for {len(expected.slots)} slots: the run is not complete")
    for r, slot in zip(benchmark, expected.slots):
        if (r.task_id, r.instance_seed, r.phase) != (slot.task_id, slot.instance_seed, slot.phase):
            problems.append(f"record {r.step} is not slot {r.step} of the schedule")
            break
    reported = json.loads((run_dir / "summary.json").read_text())
    recomputed = summarize(records, config)
    for block in ("overall", "stream", "epilogue"):
        if reported.get(block) != recomputed.get(block):
            problems.append(f"summary.json's {block} block is not what the records give")
    submission = json.loads((run_dir / "submission.json").read_text())
    for field in SUBMISSION_FIELDS:
        if not isinstance(submission.get(field), str) or not submission[field].strip():
            problems.append(f"submission.json needs a {field}")

    cap = config.max_cost
    stream = budget([r for r in benchmark if r.phase == "stream"], cap)
    epilogue = budget([r for r in benchmark if r.phase == "epilogue"], cap)
    line = f"{run_dir}: {protocol['config']} {protocol['mode']} seed {protocol['seed']}, "
    line += "canonical" if protocol["canonical"] else "not canonical (" + "; ".join(protocol["deviations"]) + ")"
    print(line)
    if stream:
        print(f"  stream:   budget used {stream[0]:5.1f}   solved {stream[1]:5.1f}%")
    if epilogue:
        print(f"  epilogue: budget used {epilogue[0]:5.1f}   solved {epilogue[1]:5.1f}%")
    return problems


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    failed = 0
    for p in paths:
        problems = check(Path(p))
        for problem in problems:
            print(f"  problem: {problem}")
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
