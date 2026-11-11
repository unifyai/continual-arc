"""Qwen in GLM's harness (the author, 5 Nov 2026): Qwen3.5-9B plays the paper's stream and its two frozen tests from GLM-5.3
Flash's own prime-agent harness, so that nothing stands between the student and what its teacher learnt but the harness
itself. In GLM's own rows, h and hpl (reference.py), GLM drove the harness and wrote every note, memory and program in
it; here Qwen only reads and uses what GLM kept, and whatever Qwen writes is thrown away. The paper's teacher-notes rows
(hg, hgpl, hplg) have GLM write only after Qwen's own games; this is the version with no Qwen-written state at all.

Each source is GLM's final one-user run of a row (:data:`SOURCES`), whose stream committed its harness state at the end
of every round (baselines' StateLog: ``play/users/u000/system/state.git``, the commit of each round named in
``state-log.jsonl``). Qwen plays a source as one of its own rows (``--row``), exactly as reference.py's ``play_round``
plays Qwen's own rows (the same preset, effort, proxy, sandbox, budget and rules), so that the three replays mirror the
paper's three teacher-notes rows, which differ in who is given the expert guidance: row h, prime-agent as it ships, from
GLM's h (beside H-T) and from GLM's hpl (beside H-T-E: GLM had the guidance, Qwen has not), and row hpl, the paper's H-E,
told c1 and the program library's procedure, from GLM's hpl (beside H-E-T-E). The first two are read against Qwen's
reset H0, the third against H-E0. A wiped row (h0, hpl0: reference.py's ``STATELESS_ROWS``) takes no source: every
instance plays from nothing, as the row is wiped before every instance. Played on the same 200 stream rounds beside the
replays, on the same machine and the same OpenRouter route at the same hours, these are the replays' same-window resets:
Together changed how it returns a call to an undeclared tool at 21:00 UTC on 29 Oct 2026, which the paper's own resets'
streams straddle (continual-arc-paper's notes/providers.md).

The stream: instance r is played from the state GLM's harness held when GLM's own round r began, the commit of round
r - 1 (an empty state for round 0), on the task and instance seed GLM met there (slot r of GLM's schedule, the stream at
seed 0). Nothing Qwen writes is kept, so the 200 instances are independent: :data:`PART` consecutive rounds play as one
round of PART users, user u playing round first + u, each forked from its own commit (:func:`seed`, baselines'
``frozen.fork_user`` but for the commit and the path renamed: the state's every absolute path naming GLM's folder, as
GLM's driver spelt it (:data:`SPELT`), renamed to the user's), into ``RUN/frozen/stream-FFF/play``. The two frozen tests
play the same way from GLM's end state, the commit of round 199, every user from the same commit, as ref-tests.sh's
TEST-qwen plays of GLM's h did, into ``RUN/frozen/TEST[-K]/play``, the paper's names for a row's plays
(continual-arc-paper's ``_play_names``). Each play is a benchmark run of its own, played by ``reference.play_round``
itself: its records, schedule, summary, driver.json, every user's harness and every model call
(``rounds/r0000/captured``), and ``frozen.json``, what each user was forked from. Once its round is recorded, each
user's harness logs and the diff of what it wrote (``users/uNNN/harness/``) are kept beside its calls and its working
tree goes, as ref-tests.sh leaves a test: the tree is in the user's repository whole.

``RUN/play`` is the stream as the paper reads a row's (:func:`view`): GLM's schedule, and each stream slot's record from
the part that played it, its play as recorded (solved, cost, attempts, demonstrations, resources) and its place in the
stream the slot's (step, round, user, phase, exposure index and gap, from the stream's own annotation), since a part
serves every instance in one round; ``view.json`` names the play and user behind each slot. ``RUN/source/play`` holds
GLM's run as the seeding reads it: its state history (repacked: the same objects), state log, schedule and alias salt,
the task names GLM's notes use, which every play takes. ``RUN/live.jsonl`` gets a row as each instance ends, for the
dashboard.

Run on the code GLM's and Qwen's rows and their resets played on, the final -turns tree (refs-deploy.sh's ~/loop-turns,
~/cab-turns and ~/benchmark-turns), as ref-tests.sh and extra-tests.sh run frozen.py: from ~/cab-turns with its .env,
``PYTHONPATH=~/loop-turns uv run --no-sync --env-file .env python glm_harness.py ...``:

  play     --run RUN --row ROW [--source SOURCE] --name NAME (--first F | --test TEST): one play into RUN/frozen/NAME/play
  view     --run RUN: RUN/play from the stream's parts, once all are recorded
  summary  --run RUN: RUN/frozen/summary.json (frozen.py summary)
  plan     --run RUN --row ROW [--source SOURCE] [--streams | --tests]: the plays a run needs, a line each, as
           glm_harness.sh's jobs file takes them

A run's stream and its tests may play on different machines, each holding the run's folder under the same name with the
source the plays need, as the paper's registry reads a row's tests from more than one node (TEST_FOLDERS).
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# GLM-5.3 Flash's final one-user runs, by row, as <node>/<run> under the artifacts bucket's runs/gpu-nodes/
SOURCES = {"h": "REF_ROWS_NODE_A/glm-h-u1-turns", "hpl": "REF_ROWS_NODE_B/glm-hpl-u1-turns"}
BUCKET = "gs://ARTIFACTS_BUCKET/results/runs/gpu-nodes"
PLAYER = "qwen"
ROWS = ("h", "hpl", "h0", "hpl0")   # Qwen's rows: h and hpl from a source's states, h0 and hpl0 wiped, from nothing
# GLM's system folder as its driver spelt it (reference.sh's RUNS/paper/<run>/play), the path its state names itself by
SPELT = "/home/NODE_USER/runs/paper/{run}/play/users/u000/system"
ROUNDS = 200
PART = 100      # stream rounds a play serves at once, one user each: a test's width
PLAYS = 4       # plays of each frozen test, as every one-user row of the paper has
TESTS = ("held-out", "seen")
SOURCE = "source"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"   # git's empty tree: what a user forked from nothing starts at
HARNESS_LOGS = ("adapter.log", "prime-agent.log", "rpc-events.jsonl", "turns.jsonl")
SKIP_DIFF = ":(exclude)agent/kernel-venv"   # the kernel's copy of its Python environment, which no instance changes


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def spelt(source: str) -> str:
    return SPELT.format(run=SOURCES[source].split("/", 1)[1])


def test_schedule(test: str) -> Path:
    """The paper's frozen test after arc1-small's stream at seed 0 and one user, as every reference row played it."""
    import paper
    return Path(paper.__file__).resolve().parent / "schedules" / f"arc1-small-{test}-test-seed0.json"


def stream_of(run: Path) -> Any:
    from continual_arc.schedule import Schedule
    return Schedule.load(run / SOURCE / "play" / "schedule.json")


def part_schedule(stream: Any, first: int) -> Any:
    """Rounds ``first`` to ``first + PART - 1`` of the stream as one round of PART users, user u serving slot first + u,
    its task, instance seed and phase the stream's."""
    from continual_arc.schedule import Schedule
    slots = stream.slots[first:first + PART]
    if stream.users != 1 or len(slots) != PART:
        raise SystemExit(f"{stream.name} is not a one-user stream holding rounds {first} to {first + PART - 1}")
    return Schedule(name=f"{stream.name}-rounds-{first:03d}-{first + PART - 1:03d}", seed=stream.seed,
                    slots=[replace(s, user=u, reset_before=False) for u, s in enumerate(slots)],
                    sources=dict(stream.sources), users=PART,
                    meta={**stream.meta, "glm_harness": {"stream": stream.name, "first_round": first, "rounds": PART}})


def state_log(system: Path) -> list[str]:
    return [line for line in (system / "state-log.jsonl").read_text().splitlines() if line.strip()]


def seed(play_dir: Path, source: Path, old: str | None, starts: list[int | None]) -> list[dict[str, Any]]:
    """``play_dir`` as its round begins, from nothing: the source's alias salt, and user u's ``system/`` a fork of the
    source's state at the end of round ``starts[u]`` (from nothing, for None), as baselines' ``frozen.fork_user`` forks a
    test user from its stream's last commit: a repository whose objects are the play's store (``objects.git``, reading
    the source's history as an alternate), its tree that commit's with every absolute path naming ``old``, the folder
    the source's state names itself by, renamed to the user's own (its naming found in its own tree), committed, and the
    source's state log up to it, so that the play's commit is numbered the round the user plays. Returns what each user
    was forked from. A play whose every user starts from nothing reads no history (a wiped row's)."""
    from continual_arc_baselines import frozen as forks
    from continual_arc_baselines.statelog import GIT_ENV, _git, restore

    system = source / "users" / "u000" / "system"
    forking = any(start is not None for start in starts)
    log = state_log(system) if forking else []
    commits = {json.loads(line)["round"]: json.loads(line)["commit"] for line in log}
    shutil.rmtree(play_dir, ignore_errors=True)
    play_dir.mkdir(parents=True)
    shutil.copy2(source / "alias-salt", play_dir / "alias-salt")
    objects = forks.object_store(play_dir / "objects.git", [system] if forking else [])

    def fork(user: int) -> dict[str, Any]:
        target, start = forks.user_system(play_dir, user), starts[user]
        if start is None:
            forks.fork_user(target, objects, None, None)
            return {"user": user, "round": None, "commit": None}
        repo, tree, commit = target / "state.git", target / "state", commits[start]
        target.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet", "--bare", str(repo)], env={**os.environ, **GIT_ENV}, check=True)
        _git(repo, "config", "core.bare", "false")
        _git(repo, "config", "gc.auto", "0")
        shutil.rmtree(repo / "objects")
        (repo / "objects").symlink_to(os.path.relpath(objects, repo.resolve()))
        restore(repo, commit, tree)
        _git(repo, "update-ref", "HEAD", commit)
        _git(repo, "reset", "--quiet", commit, tree=tree)
        (target / "state-log.jsonl").write_text("".join(line + "\n" for line in log
                                                        if json.loads(line)["round"] <= start))
        found = forks.naming(tree, old)
        forks.relocate(tree, old, str(target), found)
        _git(repo, "add", "--all", "--force", ".", tree=tree)
        _git(repo, "commit", "--quiet", "--allow-empty", "--no-verify", "-m", f"relocated from {old}", tree=tree)
        return {"user": user, "round": start, "commit": commit, "relocated": len(found[0]) + len(found[1]),
                "databases": len(found[2])}

    with ThreadPoolExecutor(forks.FORKS) as pool:
        return list(pool.map(fork, range(len(starts))))


@contextlib.contextmanager
def live_rows(path: Path, row_of):
    """Every instance's record as it ends, ``row_of(record)`` appended to ``path``: the runner's own progress callback,
    passed to the run reference.play_round starts (which gives none), the play otherwise unchanged."""
    import continual_arc.runner as runner
    run = runner.run

    def reporting(*args: Any, **kwargs: Any) -> Any:
        def progress(record: Any, _steps: int) -> None:
            with path.open("a") as fp:
                fp.write(json.dumps(row_of(record)) + "\n")
        return run(*args, **{**kwargs, "progress": progress})

    runner.run = reporting
    try:
        yield
    finally:
        runner.run = run


def keep_harness(play_dir: Path, users: int) -> None:
    """Once the play's round is recorded: each user's harness logs and the diff of what it wrote in its instance (its
    round's commit against the one it was forked at) into ``users/uNNN/harness/``, beside its calls, and its working
    tree deleted (``rm -rf``, as ref-tests.sh deletes a test's), every file of it in the user's repository. A user whose
    tree is gone is done, so a play taken up again after its round was recorded only finishes this."""
    from continual_arc_baselines import frozen as forks
    from continual_arc_baselines.statelog import GIT_ENV

    def git(repo: Path, tree: Path, *args: str) -> bytes:
        return subprocess.run(["git", f"--git-dir={repo}", f"--work-tree={tree}", *args], cwd=tree, check=True,
                              capture_output=True, env={**os.environ, **GIT_ENV}).stdout

    def one(user: int) -> None:
        system = forks.user_system(play_dir, user)
        repo, tree, kept = system / "state.git", system / "state", system.parent / "harness"
        if not tree.exists():
            return
        kept.mkdir(exist_ok=True)
        for name in HARNESS_LOGS:
            if (system / name).is_file():
                shutil.copy2(system / name, kept / name)
        base = "HEAD~1" if int(git(repo, tree, "rev-list", "--count", "HEAD")) > 1 else EMPTY_TREE
        (kept / "written.stat").write_bytes(git(repo, tree, "diff", "--stat=200", base, "HEAD", "--", SKIP_DIFF))
        with gzip.open(kept / "written.diff.gz", "wb") as fp:
            fp.write(git(repo, tree, "diff", "--binary", base, "HEAD", "--", SKIP_DIFF))
        subprocess.run(["rm", "-rf", str(tree)], check=True)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, range(users)))


def cmd_play(a: argparse.Namespace) -> None:
    import logging

    import paper
    import reference
    from continual_arc.schedule import Schedule

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = Path(a.run).expanduser()   # as given, never resolved: the harness names its folder by this spelling
    source = run / SOURCE / "play"
    play_dir = run / "frozen" / a.name / "play"
    wiped = a.row in reference.STATELESS_ROWS
    if wiped != (a.source is None):
        raise SystemExit("a wiped row plays every instance from nothing and takes no --source; any other row plays "
                         "from the --source's states")
    if a.test:
        schedule = Schedule.load(test_schedule(a.test))
        if schedule.meta.get("test") != a.test or set(schedule.rounds()) != {0}:
            raise SystemExit(f"{test_schedule(a.test)} is not one round of the {a.test} test")
        starts: list[int | None] = [None if wiped else ROUNDS - 1] * schedule.users
        infos = None
    else:
        stream = stream_of(run)
        schedule = part_schedule(stream, a.first)
        starts = [None if wiped or a.first + u == 0 else a.first + u - 1 for u in range(PART)]
        infos = stream.annotate()
    if paper.round_recorded(play_dir, schedule):
        keep_harness(play_dir, schedule.users)
        print(json.dumps({"play": a.name, "played": True}), flush=True)
        return
    forked = seed(play_dir, source, spelt(a.source) if a.source else None, starts)
    (play_dir / "frozen.json").write_text(json.dumps(
        {"run": str(run), "row": a.row, "player": PLAYER, "test": a.test or "stream", "schedule": schedule.name,
         "users": schedule.users, "source": f"{BUCKET}/{SOURCES[a.source]}" if a.source else None,
         "memory": ("none: every user from an empty state, the row wiped before every instance" if wiped
                    else "test user i a fork of GLM's state at its commit of the stream's last round, relocated" if a.test
                    else f"user u a fork of GLM's state at its commit of round {a.first} + u - 1, relocated: the state "
                         f"its harness held when its own round {a.first} + u began (none for round 0)"),
         **({} if a.test else {"first_round": a.first}), "from": forked, "started": now()}, indent=1))

    def row_of(record: Any) -> dict[str, Any]:
        out = {"play": a.name, "test": a.test or "stream", "step": record.step, "task_id": record.task_id,
               "instance_seed": record.instance_seed, "solved": bool(record.solved), "cost": record.cost,
               "usd": (record.resources or {}).get("llm_cost_usd"), "at": now()}
        if infos is not None:   # a part of the stream: the slot's round and visit in the stream
            r = a.first + record.user
            out.update(round=r, exposure_index=infos[r].exposure_index)
        return out

    with live_rows(run / "live.jsonl", row_of):
        result = reference.play_round(PLAYER, a.row, play_dir, schedule, api_key=os.environ["OPENROUTER_API_KEY"])
    keep_harness(play_dir, schedule.users)
    print(json.dumps({"play": a.name, "records": len(result.records),
                      "solved": sum(r.solved for r in result.records)}), flush=True)


def cmd_view(a: argparse.Namespace) -> None:
    """``RUN/play``, the stream as the paper reads it (the module's docstring), from its parts, all recorded."""
    from continual_arc.configs import protocol_of
    from continual_arc.env import EnvConfig, InstanceRecord
    from continual_arc.metrics import summarize

    run = Path(a.run).expanduser()
    source = run / SOURCE / "play"
    stream = stream_of(run)
    infos = stream.annotate()
    records: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    parts = [run / "frozen" / f"stream-{first:03d}" / "play" for first in range(0, ROUNDS, PART)]
    for first, part in zip(range(0, ROUNDS, PART), parts):
        played = [json.loads(line) for line in (part / "records.jsonl").read_text().splitlines() if line.strip()]
        if len(played) != PART:
            raise SystemExit(f"{part} holds {len(played)} of its {PART} records")
        for rec in played:
            r = first + rec["user"]
            slot, info = stream.slots[r], infos[r]
            if (rec["task_id"], rec["instance_seed"], rec["step"]) != (slot.task_id, slot.instance_seed, rec["user"]):
                raise SystemExit(f"{part}: user {rec['user']} played {rec['task_id']} at {rec['instance_seed']}, "
                                 f"not round {r}'s {slot.task_id} at {slot.instance_seed}")
            records.append({**rec, "step": r, "round": info.round, "user": slot.user, "phase": slot.phase,
                            "reset_before": slot.reset_before, "exposure_index": info.exposure_index,
                            "gap": info.gap, "user_exposure_index": info.user_exposure_index,
                            "user_gap": info.user_gap})
            slots.append({"round": r, "play": str(part.relative_to(run)), "user": rec["user"]})
    records.sort(key=lambda rec: rec["step"])
    out = run / "play"
    out.mkdir(exist_ok=True)
    for name in ("schedule.json", "alias-salt"):
        shutil.copy2(source / name, out / name)
    shutil.copy2(parts[0] / "config.json", out / "config.json")
    config = EnvConfig.from_json(json.loads((out / "config.json").read_text()))
    (out / "records.jsonl").write_text("".join(json.dumps(rec) + "\n" for rec in records))
    summary = summarize([InstanceRecord.from_json(rec) for rec in records], config)
    summary["protocol"] = protocol_of(stream, config)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    driver = json.loads((parts[0] / "driver.json").read_text())
    played_from = json.loads((parts[0] / "frozen.json").read_text())
    start = "nothing, the row wiped" if played_from["source"] is None else "GLM's state when its round began"
    (out / "driver.json").write_text(json.dumps(
        {**driver, "source": played_from["source"],
         "played": f"each stream instance from {start}, {PART} rounds a play, one user each (glm_harness.py)"},
        indent=1))
    (out / "meta.json").write_text(json.dumps(
        {"learner": run.name, "protocol": summary["protocol"],
         "parts": [json.loads((p / "meta.json").read_text()) for p in parts], "written": now()}, indent=1))
    (out / "view.json").write_text(json.dumps(
        {"what": "the stream as the paper reads a row's: each slot's record from the play that served it, its place "
                 "in the stream the slot's", "schedule": stream.name, "slots": slots, "written": now()}, indent=1))
    (run / "play.done").touch()
    print(json.dumps({"records": len(records), "solved": sum(bool(rec["solved"]) for rec in records)}), flush=True)


def cmd_summary(a: argparse.Namespace) -> None:
    import frozen
    frozen.cmd_summary(argparse.Namespace(run=a.run, out=None))


def cmd_plan(a: argparse.Namespace) -> None:
    """The plays RUN needs, ``RUN SOURCE ROW NAME FIRST|TEST`` a line (SOURCE - for a wiped row), the stream's parts
    first, or only those (``--streams``) or only the tests (``--tests``)."""
    head = f"{a.run} {a.source or '-'} {a.row}"
    lines = [] if a.tests else [f"{head} stream-{first:03d} {first}" for first in range(0, ROUNDS, PART)]
    for test in () if a.streams else TESTS:
        lines += [f"{head} {test if k == 1 else f'{test}-{k}'} {test}" for k in range(1, PLAYS + 1)]
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("play", help="one play: a part of the stream or a frozen test, into RUN/frozen/NAME/play")
    s.add_argument("--run", required=True, help="the run's folder, ~/runs/paper/<run>, with GLM's source in source/")
    s.add_argument("--row", required=True, choices=ROWS, help="the row Qwen plays")
    s.add_argument("--source", choices=SOURCES, help="the GLM run whose states it plays from; none for a wiped row")
    s.add_argument("--name", required=True)
    which = s.add_mutually_exclusive_group(required=True)
    which.add_argument("--first", type=int, choices=range(0, ROUNDS, PART), help="the first stream round of the part")
    which.add_argument("--test", choices=TESTS)
    s.set_defaults(fn=cmd_play)
    for name, fn, text in (("view", cmd_view, "RUN/play from the stream's parts"),
                           ("summary", cmd_summary, "RUN/frozen/summary.json")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--run", required=True)
        s.set_defaults(fn=fn)
    s = sub.add_parser("plan", help="the plays a run needs, as glm_harness.sh's jobs take them")
    s.add_argument("--run", required=True)
    s.add_argument("--row", required=True, choices=ROWS)
    s.add_argument("--source", choices=SOURCES)
    only = s.add_mutually_exclusive_group()
    only.add_argument("--streams", action="store_true", help="the stream's parts alone")
    only.add_argument("--tests", action="store_true", help="the tests alone")
    s.set_defaults(fn=cmd_plan)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
