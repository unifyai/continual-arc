"""Check a run submitted to the Continual-ARC leaderboard, and print the numbers it would show.

    uv run python scripts/check_result.py results/<track>/<name> [more runs...]

A submitted run is the directory the benchmark's runner wrote for the learner's stream (`records.jsonl`,
`summary.json`, `schedule.json`, `config.json`, `meta.json`) with a `submission.json` beside them. `submission.json`
names what the leaderboard shows, the demo-cache floor's run on the same model and stream (`floor`), which the share
learned is read against, and the frozen tests played after the stream (`tests`): for the held-out test and, if it was
played, the seen test, the learner's own test, played from where its stream left it, and its two references, the
same learner reset before each instance and the bare model (the stateless floor), each played on its own. Every
directory is relative to this repository.

The check passes when every run's schedule is the one the benchmark builds for its configuration, order or test, and
seed, its records cover every slot of it, its summary is the one its records give, the learner's tests were played
under its stream's task names (`alias-salt`), and the references are runs of the same learner and of the stateless
floor. It then prints the leaderboard's numbers: T on the held-out test first (the points of the budget saved against
the stronger of the two references, higher is better), T on the seen test, the share learned over the stream (the
share of the learnable budget saved beyond the floor) and the budget used and share solved on the stream (100 times
the mean charge over the cap, lower is better). Nothing here calls a model or the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from continual_arc.aliases import SALT_FILE
from continual_arc.configs import build_schedule, protocol_of, test_schedule, with_users
from continual_arc.metrics import share_learned, summarize, transfer
from continual_arc.runner import load_run
from continual_arc.schedule import TEST_HELD_OUT, TESTS

ROOT = Path(__file__).resolve().parent.parent

RUN_FILES = ("records.jsonl", "summary.json", "schedule.json", "config.json", "meta.json")
SUBMISSION_FIELDS = ("name", "learner", "model", "code", "contact")
FLOOR = "BareDemoCacheLearner"
BARE = "BareModelLearner"
ROLES = ("run", "reset", "floor")
REFERENCE = {"reset": "learner reset", "floor": "bare model"}


def budget(records, cap: float) -> tuple[float, float] | None:
    if not records:
        return None
    used = 100 * sum(r.cost for r in records) / len(records) / cap
    solved = 100 * sum(1 for r in records if r.solved) / len(records)
    return used, solved


def learner_of(run_dir: Path) -> str:
    """The learner a run played, by its class; the baselines wrap every learner in their tracking proxy, so a run of
    theirs names it by meta's "learner"."""
    meta = json.loads((run_dir / "meta.json").read_text())
    cls = meta.get("learner_class", "")
    return meta.get("learner", "") if not cls or cls.endswith(".TrackedLearner") else cls


def check_run(run_dir: Path, expected, label: str) -> tuple[list[str], tuple]:
    """``run_dir`` against the schedule the benchmark builds for it; returns the problems and the loaded run."""
    problems = [f"{label}: missing {f}" for f in RUN_FILES if not (run_dir / f).exists()]
    if problems:
        return problems, (None, None, None)
    schedule, config, records = load_run(run_dir)
    if [s.to_json() for s in schedule.slots] != [s.to_json() for s in expected.slots]:
        problems.append(f"{label}: schedule.json is not the schedule the benchmark builds for it")
    benchmark = [r for r in records if r.episode_type == "benchmark"]
    if len(benchmark) != len(expected.slots):
        problems.append(f"{label}: {len(benchmark)} records for {len(expected.slots)} slots: the run is not complete")
    for r in benchmark:
        slot = expected.slots[r.step] if r.step < len(expected.slots) else None
        if slot is None or (r.task_id, r.instance_seed, r.phase) != (slot.task_id, slot.instance_seed, slot.phase):
            problems.append(f"{label}: record {r.step} is not slot {r.step} of the schedule")
            break
    reported = json.loads((run_dir / "summary.json").read_text())
    recomputed = summarize(records, config)
    for block in recomputed:
        if block in reported and reported[block] != recomputed[block]:
            problems.append(f"{label}: summary.json's {block} block is not what the records give")
    return problems, (schedule, config, records)


def check_test(test: str, dirs: dict, run_dir: Path, protocol: dict, learner: str) -> tuple[list[str], dict | None]:
    """One frozen test's three runs; returns the problems and T, or None if they cannot be read."""
    problems, loaded = [], {}
    for role in ROLES:
        if not dirs.get(role):
            problems.append(f"submission.json's {test} test names no {role} directory")
            continue
        d = ROOT / str(dirs[role])
        label = f"{test} test {role} ({dirs[role]})"
        if not (d / "schedule.json").exists():
            problems.append(f"{label}: no run there")
            continue
        meta = json.loads((d / "schedule.json").read_text()).get("meta", {})
        expected = test_schedule(protocol["config"], protocol["seed"], test, width=meta.get("width"))
        found, (_, config, records) = check_run(d, expected, label)
        problems += found
        loaded[role] = records
        who = learner_of(d)
        if role == "floor" and BARE not in who:
            problems.append(f"{label}: not a run of the stateless floor ({BARE})")
        if role == "reset" and who != learner:
            problems.append(f"{label}: not a run of the learner the stream ran ({learner})")
        if role == "run" and (d / SALT_FILE).read_text() != (run_dir / SALT_FILE).read_text():
            problems.append(f"{label}: not played under the stream's task names (copy its {SALT_FILE} first)")
    if problems or len(loaded) < len(ROLES):
        return problems, None
    return problems, transfer(loaded["run"], loaded["reset"], loaded["floor"], config, test=test)


def check(run_dir: Path) -> list[str]:
    problems = [f"missing {f}" for f in (*RUN_FILES, "submission.json") if not (run_dir / f).exists()]
    if problems:
        return problems
    schedule, config, records = load_run(run_dir)
    protocol = protocol_of(schedule, config)
    expected = build_schedule(with_users(protocol["config"], protocol["users"]), protocol["mode"], protocol["seed"],
                              epilogue=bool(protocol.get("epilogue")))
    problems, _ = check_run(run_dir, expected, "the stream")
    submission = json.loads((run_dir / "submission.json").read_text())
    for field in SUBMISSION_FIELDS:
        if not isinstance(submission.get(field), str) or not submission[field].strip():
            problems.append(f"submission.json needs a {field}")
    learned = None
    floor_dir = ROOT / str(submission.get("floor", ""))
    if not submission.get("floor") or not (floor_dir / "records.jsonl").exists():
        problems.append("submission.json needs a floor: the directory of the demo-cache floor's run on the same model and stream")
    else:
        if FLOOR not in learner_of(floor_dir):
            problems.append(f"{submission['floor']} is not a run of the demo-cache floor ({FLOOR})")
        _, _, floor_records = load_run(floor_dir)
        try:
            learned = share_learned(records, floor_records, config)
        except ValueError as e:
            problems.append(str(e))
    tests = submission.get("tests") or {}
    if TEST_HELD_OUT not in tests:
        problems.append("submission.json's tests name no held-out test, whose T is the headline")
    results = {}
    for test in TESTS:
        if test in tests:
            found, results[test] = check_test(test, tests[test], run_dir, protocol, learner_of(run_dir))
            problems += found

    stream = budget([r for r in records if r.episode_type == "benchmark" and r.phase == "stream"], config.max_cost)
    line = f"{run_dir}: {protocol['config']} {protocol['mode']} seed {protocol['seed']}, "
    line += "canonical" if protocol["canonical"] else "not canonical (" + "; ".join(protocol["deviations"]) + ")"
    print(line)
    for test, t in results.items():
        if t and t["transfer"] is not None:
            lo, hi = t["ci95"] or (float("nan"), float("nan"))
            print(f"  {test} test: T {t['transfer']:+.1f} points (95% interval {lo:+.1f} to {hi:+.1f}) against the "
                  f"{REFERENCE[t['reference']]}, budget used {t['run_budget']:.1f}, solved {100 * t['solved']['run']:.1f}%")
    if learned and learned["share_learned"] is not None:
        lo, hi = learned["ci95"] or (float("nan"), float("nan"))
        print(f"  share learned {learned['share_learned']:+.2f} ±{learned['se']:.2f} (95% interval {lo:+.2f} to {hi:+.2f}), "
              f"learnable budget {learned['learnable_budget']:.1f} points")
    if stream:
        print(f"  stream: budget used {stream[0]:5.1f}   solved {stream[1]:5.1f}%")
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
