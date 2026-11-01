"""Check a run submitted to the Continual-ARC leaderboard, and print the numbers it would show.

    uv run python scripts/check_result.py results/<track>/<name> [more runs...]

A submitted run is the directory the benchmark's runner wrote for the learner's stream (`records.jsonl`,
`summary.json`, `schedule.json`, `config.json`, `meta.json`) with a `submission.json` beside them. `submission.json`
names what the leaderboard shows: the learning method (`name`, `learner`, `code`), the model that plays (`student`) at
its reasoning `effort` and the model that teaches it (`teacher`, or null), each a published checkpoint used unchanged
and named exactly, an OpenRouter model id or open weights as their Hugging Face repository at a commit
(`org/name@commit`), since a result is compared only with others on the same student and teacher; the learner reset on
the same stream (`reset`: `schedule make --reset-each`), which the share learned is read against, and the two frozen
tests played after the stream (`tests`): for each, the learner's own test, played from where its stream left it, the
same learner reset before each instance, which the test is read against, and the bare model (the stateless floor),
shown beside it, each played on its own. Every directory is relative to this repository.

The check passes when every run's schedule is the one the benchmark builds for its configuration, order or test, and
seed, its records cover every slot of it, its summary is the one its records give, the learner's tests were played
under its stream's task names (`alias-salt`), the resets are runs of the same learner and the bare runs of the
stateless floor. It then prints the leaderboard's numbers: the learner's place on the plane of the two tests, the
share of its reset's spend it saves on the seen test (memorization) and on the held-out test (generalization), each
with its 95 percent interval, the points saved (T) and the budget used and share solved by the learner and its reset
beside it, flagged when the reset uses more of the budget than the bare model; then the share learned over the stream
(the share of the learnable budget saved over the reset) and the budget used and share solved on the stream (100 times
the mean charge over the cap, lower is better). Nothing here calls a model or the network.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from continual_arc.aliases import SALT_FILE
from continual_arc.configs import build_schedule, protocol_of, test_schedule, with_users
from continual_arc.metrics import share_learned, summarize, transfer
from continual_arc.runner import load_run
from continual_arc.schedule import TEST_HELD_OUT, TEST_SEEN, TESTS

ROOT = Path(__file__).resolve().parent.parent

RUN_FILES = ("records.jsonl", "summary.json", "schedule.json", "config.json", "meta.json")
# A published checkpoint, named exactly: an OpenRouter model id (vendor/model, with an optional :variant), or open
# weights as their Hugging Face repository at a commit (org/name@commit). A model changed before the stream is another
# student, compared only with results on it, since one made to start badly would save a large share by recovering
# alone; this checks the name's form, and a verified run checks an OpenRouter id against OpenRouter's list.
PUBLISHED = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*(?::[\w.-]+|@[0-9a-f]{7,40})?")
SUBMISSION_FIELDS = ("name", "learner", "student", "effort", "code", "contact")
BARE = "BareModelLearner"
ROLES = ("run", "reset", "bare")


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


def check_test(test: str, dirs: dict, run_dir: Path, protocol: dict,
               learner: str) -> tuple[list[str], tuple[dict, dict] | None]:
    """One frozen test's three runs; returns the problems and the learner's reading against its reset
    (``metrics.transfer``: the share saved, T and both sides' budget and share solved), with the bare model read against
    the same reset (a positive T there is a reset that uses more of the budget than the bare model, so plays below it),
    or None if they cannot be read."""
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
        if role == "bare" and BARE not in who:
            problems.append(f"{label}: not a run of the stateless floor ({BARE})")
        if role == "reset" and who != learner:
            problems.append(f"{label}: not a run of the learner the stream ran ({learner})")
        if role == "run" and (d / SALT_FILE).read_text() != (run_dir / SALT_FILE).read_text():
            problems.append(f"{label}: not played under the stream's task names (copy its {SALT_FILE} first)")
    if problems or len(loaded) < len(ROLES):
        return problems, None
    return problems, (transfer(loaded["run"], loaded["reset"], config, test=test),
                      transfer(loaded["bare"], loaded["reset"], config, test=test, reps=0))


def check(run_dir: Path) -> list[str]:
    problems = [f"missing {f}" for f in (*RUN_FILES, "submission.json") if not (run_dir / f).exists()]
    if problems:
        return problems
    schedule, config, records = load_run(run_dir)
    protocol = protocol_of(schedule, config)
    expected = build_schedule(with_users(protocol["config"], protocol["users"]), protocol["seed"],
                              order=protocol["mode"], epilogue=bool(protocol.get("epilogue")))
    problems, _ = check_run(run_dir, expected, "the stream")
    submission = json.loads((run_dir / "submission.json").read_text())
    for field in SUBMISSION_FIELDS:
        if not isinstance(submission.get(field), str) or not submission[field].strip():
            problems.append(f"submission.json needs a {field}")
    teacher = submission.get("teacher", "")
    if teacher is not None and (not isinstance(teacher, str) or not teacher.strip()):
        problems.append("submission.json needs a teacher: the model that teaches the student, or null for none")
    for role, name in (("student", submission.get("student")), ("teacher", teacher)):
        if isinstance(name, str) and name.strip() and not PUBLISHED.fullmatch(name):
            problems.append(f"submission.json's {role} ({name}) is not a published checkpoint named exactly: an OpenRouter "
                            "model id, or open weights as their Hugging Face repository at a commit (org/name@commit)")
    learned = None
    reset_dir = ROOT / str(submission.get("reset", ""))
    if not submission.get("reset") or not (reset_dir / "records.jsonl").exists():
        problems.append("submission.json needs a reset: the directory of your learner's run of the same stream, wiped "
                        "before every instance (schedule make --reset-each)")
    else:
        found, (_, _, reset_records) = check_run(reset_dir, expected.reset_each(), f"the reset ({submission['reset']})")
        problems += found
        if learner_of(reset_dir) != learner_of(run_dir):
            problems.append(f"{submission['reset']} is not a run of the learner the stream ran ({learner_of(run_dir)})")
        if reset_records:
            learned = share_learned(records, reset_records, config)
    tests = submission.get("tests") or {}
    results = {}
    for test in TESTS:
        if test not in tests:
            problems.append(f"submission.json's tests name no {test} test: the headline is the plane of both")
            continue
        found, results[test] = check_test(test, tests[test], run_dir, protocol, learner_of(run_dir))
        problems += found

    stream = budget([r for r in records if r.episode_type == "benchmark" and r.phase == "stream"], config.max_cost)
    line = f"{run_dir}: {protocol['config']} {protocol['mode']} seed {protocol['seed']}, "
    line += "canonical" if protocol["canonical"] else "not canonical (" + "; ".join(protocol["deviations"]) + ")"
    print(line)
    for test in (TEST_SEEN, TEST_HELD_OUT):
        read = results.get(test)
        if not read or read[0]["share"] is None:
            continue
        t, bare = read
        lo, hi = t["share_ci95"] or (float("nan"), float("nan"))
        t_lo, t_hi = t["ci95"] or (float("nan"), float("nan"))
        print(f"  {test} test: share saved {t['share']:+.3f} (95% interval {lo:+.3f} to {hi:+.3f}) of the learner "
              f"reset's spend, T {t['transfer']:+.1f} points ({t_lo:+.1f} to {t_hi:+.1f})")
        print(f"    budget used {t['run_budget']:.1f}, solved {100 * t['solved']['run']:.1f}%; the reset's "
              f"{t['reset_budget']:.1f}, solved {100 * t['solved']['reset']:.1f}%")
        if bare["transfer"] is not None:
            flag = ": flagged, the reset plays below the bare model" if bare["transfer"] > 0 else ""
            print(f"    the bare model used {bare['run_budget']:.1f}{flag}")
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
