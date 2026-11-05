"""The teacher's rows (hg, hgpl and hplg): the teacher plays the whole stream in the harness itself and writes everything
in it, and the student plays each instance from the harness as the teacher left it before that instance, keeping nothing.
This is the most supervision a harness can be given, and more than any teacher would give in practice, since every game
is played twice, once by each model; so these rows are an upper bound on what a teacher can pass on through a harness, and
not learners the benchmark would verify, the teacher's demonstrations and verdicts going uncharged.

The teacher's own run is its reference row (:mod:`continual_arc_training.references`), GLM-5.3 Flash playing row h or hpl
at one user, whose learner committed its one state at the end of every round (user 0's ``system/state.git``, each round's
commit named in ``state-log.jsonl``). The student, Qwen3.5-9B, plays as a reference row of its own, the same preset,
effort, proxy, sandbox and rules (:func:`references.play_round`). :data:`ROWS` gives each row's two: hg plays row h from the
teacher's row h, hgpl row h from the teacher's row hpl, so that the teacher alone was given the guidance, and hplg row hpl
from the teacher's row hpl, both given it.

The stream: instance r is played from the state the teacher's harness held when its own round r began, the commit of round
r - 1 (nothing for round 0), on the task and instance seed the teacher met there. Nothing the student writes is kept, so
the 200 instances are independent: :data:`PART` consecutive rounds play as one round of PART users, user u playing round
first + u, each forked from its own commit (:func:`seed`, the baselines' ``frozen.fork_user``), into
``RUN/frozen/stream-FFF/play``. The schedule says so (``meta["forked"]``), so the learner keeps the users apart rather than
giving them one state. The two frozen tests play as every run's do, ``RUN/frozen/TEST[-K]/play``, every user forked from
the commit of the teacher's last round. Each play is a benchmark run of its own, with ``frozen.json``, what each user was
forked from. Plays started together meet the providers' rate limits, and a game whose calls fail plays differently, so
``teacher-rows.sh`` starts them minutes apart.

``RUN/play`` is the stream as a row's is read (:func:`cmd_view`): the teacher's schedule, and each stream slot's record from
the part that played it, its play as recorded and its place in the stream the slot's.

The row's reset is the student's own wiped row, h0 or hpl0, played by ``reference.sh``, best at the same hours as the row's
stream, since a provider's serving can change between runs.

Subcommands (``python -m continual_arc_training.teacher_rows ...``, with OPENROUTER_API_KEY in the environment):

  play  --row ROW --teacher TEACHER --run RUN --name NAME (--first F | --test TEST): one play into RUN/frozen/NAME/play
  view  --teacher TEACHER --run RUN: RUN/play from the stream's parts, once both are recorded

TEACHER is the teacher's run folder, as its driver spelt it and never resolved, since its state names its folder by that
spelling.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from continual_arc_training import frozen, play, references

TEACHER, STUDENT = "glm", "qwen"
# each row's teacher's own row, whose states it plays from, and the row the student plays as
ROWS: dict[str, tuple[str, str]] = {"hg": ("h", "h"), "hgpl": ("hpl", "h"), "hplg": ("hpl", "hpl")}
ROUNDS = 200
PART = 100   # stream rounds a play serves at once, one user each: a test's width


def teacher_system(teacher: Path) -> Path:
    return teacher / "play" / "users" / "u000" / "system"


def stream_of(teacher: Path) -> Any:
    from continual_arc.schedule import Schedule
    return Schedule.load(teacher / "play" / "schedule.json")


def part_schedule(stream: Any, first: int) -> Any:
    """Rounds ``first`` to ``first + PART - 1`` of the stream as one round of PART users, user u serving slot first + u,
    its task, instance seed and phase the stream's, every user a fork that nothing after the round reads."""
    from continual_arc.schedule import Schedule
    slots = stream.slots[first:first + PART]
    if stream.users != 1 or len(slots) != PART:
        raise SystemExit(f"{stream.name} is not a one-user stream holding rounds {first} to {first + PART - 1}")
    return Schedule(name=f"{stream.name}-rounds-{first:03d}-{first + PART - 1:03d}", seed=stream.seed,
                    slots=[replace(s, user=u, reset_before=False) for u, s in enumerate(slots)],
                    sources=dict(stream.sources), users=PART,
                    meta={**stream.meta, "forked": True, "first_round": first, "rounds": PART})


def seed(play_dir: Path, teacher: Path, starts: list[int | None]) -> list[dict[str, Any]]:
    """``play_dir`` as its round begins, from nothing: the teacher's task names, and user u's ``system/`` a fork of the
    teacher's state at the end of round ``starts[u]``, from nothing for None, as the baselines' ``frozen.fork_user`` forks
    a test user, its every absolute path naming the teacher's folder renamed to the user's. Returns what each was forked
    from."""
    from continual_arc_baselines import frozen as forks
    from continual_arc_baselines.statelog import commit_of, restore

    system = teacher_system(teacher)
    shutil.rmtree(play_dir, ignore_errors=True)
    play_dir.mkdir(parents=True)
    shutil.copy2(teacher / "play" / frozen.SALT_FILE, play_dir / frozen.SALT_FILE)
    objects = forks.object_store(play_dir / "objects.git", system)
    scratch = play_dir / "naming"

    def fork(user: int) -> dict[str, Any]:
        target, start = forks.user_system(play_dir, user), starts[user]
        if start is None:
            forks.fork_user(target, objects, None, None)
            return {"user": user, "round": None, "commit": None}
        commit = commit_of(system / "state-log.jsonl", start)
        restore(system / "state.git", commit, scratch / str(user))
        found = forks.naming(scratch / str(user), str(system))
        shutil.rmtree(scratch / str(user))
        forks.fork_user(target, objects, (system, start + 1, commit), found)
        return {"user": user, "round": start, "commit": commit, "relocated": len(found[0]) + len(found[1]),
                "databases": len(found[2])}

    with ThreadPoolExecutor(forks.FORKS) as pool:
        forked = list(pool.map(fork, range(len(starts))))
    shutil.rmtree(scratch, ignore_errors=True)
    return forked


def cmd_play(a: argparse.Namespace) -> None:
    from continual_arc.schedule import Schedule

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run, teacher = Path(a.run).expanduser(), Path(a.teacher).expanduser()
    play_dir = run / frozen.FROZEN / a.name / "play"
    if a.test:
        schedule = Schedule.load(frozen.schedule_path(a.test))
        if schedule.meta.get("test") != a.test or set(schedule.rounds()) != {0}:
            raise SystemExit(f"{frozen.schedule_path(a.test)} is not one round of the {a.test} test")
        starts: list[int | None] = [ROUNDS - 1] * schedule.users
    else:
        schedule = part_schedule(stream_of(teacher), a.first)
        starts = [a.first + u - 1 if a.first + u else None for u in range(PART)]
    if play.round_recorded(play_dir, schedule):
        print(json.dumps({"play": a.name, "played": True}), flush=True)
        return
    forked = seed(play_dir, teacher, starts)
    (play_dir / "frozen.json").write_text(json.dumps(
        {"run": str(run), "row": a.row, "player": STUDENT, "teacher": str(teacher), "test": a.test or "stream",
         "schedule": schedule.name, "users": schedule.users, "from_users": forked,
         "started": datetime.now(UTC).isoformat()}, indent=1))
    result = references.play_round(STUDENT, ROWS[a.row][1], play_dir, schedule,
                                   api_key=os.environ["OPENROUTER_API_KEY"])
    print(json.dumps({"play": a.name, "records": len(result.records), "solved": sum(r.solved for r in result.records)}),
          flush=True)


def cmd_view(a: argparse.Namespace) -> None:
    """``RUN/play``, the stream as a row's is read (the module's docstring), from its parts, all recorded."""
    from continual_arc.configs import protocol_of
    from continual_arc.env import EnvConfig, InstanceRecord
    from continual_arc.metrics import summarize

    run, teacher = Path(a.run).expanduser(), Path(a.teacher).expanduser()
    stream = stream_of(teacher)
    infos = stream.annotate()
    records: list[dict[str, Any]] = []
    parts = [run / frozen.FROZEN / f"stream-{first:03d}" / "play" for first in range(0, ROUNDS, PART)]
    for first, part in zip(range(0, ROUNDS, PART), parts):
        played = [json.loads(line) for line in (part / "records.jsonl").read_text().splitlines() if line.strip()]
        if len(played) != PART:
            raise SystemExit(f"{part} holds {len(played)} of its {PART} records")
        for rec in played:
            r = first + rec["user"]
            slot, info = stream.slots[r], infos[r]
            if (rec["task_id"], rec["instance_seed"]) != (slot.task_id, slot.instance_seed):
                raise SystemExit(f"{part}: user {rec['user']} played {rec['task_id']} at {rec['instance_seed']}, not "
                                 f"round {r}'s {slot.task_id} at {slot.instance_seed}")
            records.append({**rec, "step": r, "round": r, "user": slot.user, "phase": slot.phase,
                            "reset_before": slot.reset_before, "exposure_index": info.exposure_index, "gap": info.gap,
                            "user_exposure_index": info.user_exposure_index, "user_gap": info.user_gap})
    records.sort(key=lambda rec: rec["step"])
    out = run / "play"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("schedule.json", frozen.SALT_FILE):
        shutil.copy2(teacher / "play" / name, out / name)
    shutil.copy2(parts[0] / "config.json", out / "config.json")
    config = EnvConfig.from_json(json.loads((out / "config.json").read_text()))
    (out / "records.jsonl").write_text("".join(json.dumps(rec) + "\n" for rec in records))
    summary = summarize([InstanceRecord.from_json(rec) for rec in records], config)
    summary["protocol"] = protocol_of(stream, config)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    driver = json.loads((parts[0] / "driver.json").read_text())
    (out / "driver.json").write_text(json.dumps(
        {**driver, "teacher": str(teacher), "played": f"each stream instance from the teacher's state when its round "
                                                      f"began, {PART} rounds per play, one user each"}, indent=1))
    print(json.dumps({"records": len(records), "solved": sum(bool(rec["solved"]) for rec in records)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("play", help="one play: a part of the stream or a frozen test, into RUN/frozen/NAME/play")
    s.add_argument("--row", required=True, choices=ROWS)
    s.add_argument("--teacher", required=True, help="the teacher's run folder, as its driver spelt it")
    s.add_argument("--run", required=True, help="this row's run folder")
    s.add_argument("--name", required=True)
    which = s.add_mutually_exclusive_group(required=True)
    which.add_argument("--first", type=int, choices=range(0, ROUNDS, PART), help="the first stream round of the part")
    which.add_argument("--test", choices=frozen.TESTS)
    s.set_defaults(fn=cmd_play)
    s = sub.add_parser("view", help="RUN/play from the stream's parts")
    s.add_argument("--teacher", required=True)
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_view)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
