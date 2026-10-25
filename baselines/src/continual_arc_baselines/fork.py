"""A run taken up again from the end of any of its rounds, with the state its learner had then, instead of played
again.

:func:`fork` makes a new run directory that has played the first rounds of an old one, which logged its state
(``arc-baselines run --log-state``), up to and including a chosen round:

* their records, as the old run wrote them;
* the old run's task aliases (``alias-salt``), since a learner's memory names tasks by them;
* a new attempt whose ``system/state`` is the commit :mod:`continual_arc_baselines.statelog` made at the end of that
  round, its repository carrying the state's history up to it, so the commits go on from there;
* a schedule that serves every carried slot exactly as the old run did and may serve anything after it, such as a
  different epilogue after the stream (the old schedule when none is given).

``arc-baselines run --resume`` with the old run's own arguments, which :func:`fork` returns as a command line, then plays
on from the next round. Run it from the folder the old run was launched from, since a new attempt is made under
``.workbench`` there, and on the same path layout as the old run, since a restored virtual environment keeps its
absolute paths.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from continual_arc.aliases import SALT_FILE
from continual_arc.schedule import Schedule

from continual_arc_baselines.evidence import Evidence
from continual_arc_baselines.statelog import GIT_ENV, _git, commit_of, restore

# Arguments of the old launch that fork replaces: where its schedule came from, its run directory, its pilot stop.
_DROPPED = {"--config": 1, "--mode": 1, "--seed": 1, "--schedule": 1, "--out": 1, "--stop-after": 1, "--resume": 0}


def _served(schedule: Schedule, n: int) -> list[tuple]:
    return [(s.task_id, s.instance_seed, s.phase, s.reset_before, s.user) for s in schedule.slots[:n]]


def fork(run_dir: str | Path, round_index: int, out_dir: str | Path, schedule: Schedule | None = None) -> list[str]:
    """Make ``out_dir`` the run ``run_dir`` was at the end of round ``round_index``, to go on under ``schedule``;
    returns the command line that plays on."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    old = Schedule.load(run_dir / "schedule.json")
    new = schedule or old
    carried = sum(1 for r in old.rounds() if r <= round_index)
    lines = [line for line in (run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
    if len(lines) < carried:
        raise ValueError(f"{run_dir} holds {len(lines)} records, fewer than the {carried} of rounds 0 to {round_index}")
    if _served(new, carried) != _served(old, carried):
        raise ValueError(f"the schedule must serve the {carried} slots it carries exactly as {run_dir} served them")
    source = Path(json.loads((run_dir / "attempt.json").read_text())["attempt_dir"]) / "system"
    if not (source / "state-log.jsonl").is_file():
        raise ValueError(f"{run_dir} did not log its system's state, so no round has a commit to fork: play it with "
                         "arc-baselines run --log-state")
    commit = commit_of(source / "state-log.jsonl", round_index)

    # A new attempt, named as the old one was (``.workbench/runs/adhoc-<name>-<stamp>/attempts/<id>``).
    evidence = Evidence.from_environment(name=re.sub(r"^adhoc-|-\d+$", "", source.parent.parent.parent.name))
    target = evidence.directory / "system"
    target.mkdir(parents=True, exist_ok=True)
    repo = target / "state.git"
    subprocess.run(["git", "clone", "--quiet", "--bare", str(source / "state.git"), str(repo)],
                   env={**os.environ, **GIT_ENV}, check=True)
    _git(repo, "config", "core.bare", "false")
    _git(repo, "config", "gc.auto", "0")
    restore(repo, commit, target / "state")
    _git(repo, "reset", "--quiet", commit, tree=target / "state")   # the branch and the index at the round's commit
    kept = [line for line in (source / "state-log.jsonl").read_text().splitlines()
            if json.loads(line)["round"] <= round_index]
    (target / "state-log.jsonl").write_text("".join(line + "\n" for line in kept))

    out_dir.mkdir(parents=True)
    (out_dir / "records.jsonl").write_text("".join(line + "\n" for line in lines[:carried]))
    (out_dir / SALT_FILE).write_text((run_dir / SALT_FILE).read_text())
    new.save(out_dir / "schedule.json")
    (out_dir / "attempt.json").write_text(json.dumps({"attempt_dir": str(evidence.directory)}))
    (out_dir / "fork.json").write_text(json.dumps({"from": str(run_dir.resolve()), "round": round_index,
                                                   "commit": commit, "records": carried}, indent=1))

    argv = json.loads((run_dir / "launches.jsonl").read_text().splitlines()[0])["argv"]
    kept_args, k = [], 0
    while k < len(argv):
        if argv[k] in _DROPPED:
            k += 1 + _DROPPED[argv[k]]
            continue
        kept_args.append(argv[k])
        k += 1
    return ["arc-baselines", *kept_args, "--schedule", str(out_dir / "schedule.json"), "--out", str(out_dir), "--resume"]
