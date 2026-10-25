"""A run's two frozen tests after its stream, each played as one round of users (``arc-baselines test-collective``), as
the paper's tests of its collective rows and of its GPT-6 Luna cells were played. Submitters play a harness's tests with
``arc-baselines test`` instead (:mod:`continual_arc_baselines.frozen`), one instance at a time from the stream's state
put back where it was.

The tests (continual-arc's ``configs.test_schedule``) are the held-out test, one instance of each of the held-out tasks,
and the seen test, new instances of the stream's own tasks. Here each is one round of a hundred users played at once
(:class:`CollectiveHarnessLearner`, a system of the run's own kind for every user, ``--width 100``), every user starting
from the system as the stream left it, so that nothing one test instance does reaches another and nothing is learnt in
between.

The system as the stream left it is the commit :mod:`continual_arc_baselines.statelog` made at the end of the stream's
last round, so the run must have logged its state: a run of ``arc-baselines run --log-state`` has one system, its
attempt's; a collective run (:class:`CollectiveHarnessLearner`, which commits every user's state at the end of every
round) has one per user, ``users/uNNN/system``, and test user i starts from the run's user i mod U, of its U. Each test
user's ``system/`` starts as a fork of that commit: its ``state`` the commit's tree, its ``state.git`` on the commit,
and its ``state-log.jsonl`` the run's rounds up to it, so that the test round's commit comes next in the user's history.
The tree is relocated first (:func:`relocate`): a harness names its own folder in its files (a Python environment's
scripts, a daemon's records, a session's log, openclaw's and hermes-agent's SQLite databases: their workspaces,
sessions' working folders and events), and a copy that kept those paths would read and write the run's own state, so
every absolute path naming the system folder it forks is renamed to the user's, and the renamed tree is committed before
the test, so that the test round's commit shows what the user wrote in its instance and nothing else. A run with no
stream (a fresh cell, the epilogue's references) starts every user from nothing. A stateless system (rows H0, HPL0 and
M0) is forked the same way and wipes what the stream's last instance left before its test instance, as before every
instance (``SystemLearner``), so its users play from nothing.

The users of a test keep their objects in one store, ``objects.git``, which reads the run's own as alternates: a
user's repository is its own (its branch, index and log) but its ``objects`` is the store's, so a hundred forks cost
what the files they wrote cost, and a harness's runtime that every user installs alike is kept once. The test's round
ends with every user's state committed as its next round, as a run's are at the end of each, and once the round is
recorded the users' working trees go: every one is in its repository, whole
(:func:`continual_arc_baselines.statelog.restore`).

The system and how it reaches its model are the run's own: the learner and its settings, the upstream, its key and the
request defaults of the run's first launch (``launches.jsonl``), as ``arc-baselines run`` parsed them.

    arc-baselines test-collective RUN_DIR --schedule TEST.json [--out OUT]   (OUT: RUN_DIR/frozen/<its name>/play)

A test's run directory is ``RUN_DIR/frozen/<the test>/play`` beside the run's own, and ``<the test>-reference/play``
for a run with no stream, whose users start from an empty state: a play from nothing, which a row's measures are read
against.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from continual_arc.aliases import SALT_FILE
from continual_arc.schedule import PHASE_STREAM, Schedule

from continual_arc_baselines.frozen import first_launch
from continual_arc_baselines.statelog import GIT_ENV, _git, commit_of, restore

# how many users are forked at once (a hermes-agent state is some 100 MB, an openclaw one 500 MB)
FORKS = 16
SQLITE = b"SQLite format 3\0"

Found = tuple[list[str], list[tuple[str, str]], list[str]]


def user_system(run_dir: Path, user: int) -> Path:
    """A user's ``system/`` in a run :class:`CollectiveHarnessLearner` plays (its ``user_dir``): a collective run's, or
    a test's."""
    return Path(run_dir) / "users" / f"u{user:03d}" / "system"


def stream_end(run_dir: Path) -> tuple[list[Path], int] | None:
    """The run's system folders, user by user (the module's docstring), and how many rounds its stream has, refused
    while the stream is still being played; None for a run whose schedule has no stream."""
    run_dir = Path(run_dir)
    schedule = Schedule.load(run_dir / "schedule.json")
    rounds = sum(slot.phase == PHASE_STREAM for slot in schedule.slots) // schedule.users
    if not rounds:
        return None
    played = sum(json.loads(line)["phase"] == PHASE_STREAM for line in (run_dir / "records.jsonl").read_text().splitlines())
    if played < rounds * schedule.users:
        raise SystemExit(f"{run_dir} has played {played} of its stream's {rounds * schedule.users} instances")
    attempt = run_dir / "attempt.json"
    if attempt.exists():
        system = Path(json.loads(attempt.read_text())["attempt_dir"]) / "system"
        if not (system / "state-log.jsonl").is_file():
            raise SystemExit(f"{run_dir} did not log its system's state, so its stream's end has no commit to fork: "
                             "play the stream with arc-baselines run --log-state, or test it with arc-baselines test")
        return [system], rounds
    return [user_system(run_dir, user) for user in range(schedule.users)], rounds


def naming(tree: Path, old: str) -> Found:
    """The files under ``tree`` that name ``old`` (an absolute path) in their text, its links that point under it, and
    its SQLite databases, which may name it in their rows or their write-ahead logs, each by its path relative to
    ``tree``. Any other file holding a NUL byte is binary and passed over."""
    needle = old.encode()
    files, links, databases = [], [], []
    for root, dirs, names in os.walk(tree):
        for path in (Path(root) / name for name in dirs + names):
            if path.is_symlink():
                target = os.readlink(path)
                if target == old or target.startswith(old + "/"):
                    links.append((str(path.relative_to(tree)), target))
            elif path.is_file():
                data = path.read_bytes()
                if data.startswith(SQLITE):
                    databases.append(str(path.relative_to(tree)))
                elif needle in data and b"\0" not in data:
                    files.append(str(path.relative_to(tree)))
    return files, links, databases


def relocate(tree: Path, old: str, new: str, found: Found) -> None:
    """Rename ``old`` to ``new`` wherever ``found`` (:func:`naming`, of the tree ``tree`` is a copy of) says ``tree``
    names it: in each file's text, as a whole path or the start of one, in each link's target, and in every text value
    of each database's own tables (:func:`relocate_database`)."""
    files, links, databases = found
    pattern = re.compile(re.escape(old.encode()) + rb"(?![\w.-])")
    for rel in files:
        path = tree / rel
        path.write_bytes(pattern.sub(new.encode(), path.read_bytes()))
    for rel, target in links:
        path = tree / rel
        path.unlink()
        path.symlink_to(new + target[len(old):])
    for rel in databases:
        relocate_database(tree / rel, old, new)


def relocate_database(path: Path, old: str, new: str) -> None:
    """Rename ``old`` to ``new`` in every text value of the SQLite database at ``path``, through SQLite itself, which
    takes in its write-ahead log first and checkpoints the result into the file. A virtual table and its shadow tables
    (a full-text index) are left as they are: their rows are kept in step with their content's by the database's
    own triggers, and an index that still finds an old path finds what it found before."""
    pattern = re.compile(re.escape(old) + r"(?![\w.-])")
    con = sqlite3.connect(path)
    try:
        con.create_function("relocated", 1, lambda value: pattern.sub(new, value), deterministic=True)
        tables = con.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
        virtual = [name for name, sql in tables if (sql or "").upper().startswith("CREATE VIRTUAL TABLE")]
        for name, _ in tables:
            if name.startswith("sqlite_") or any(name == v or name.startswith(v + "_") for v in virtual):
                continue
            for column in [row[1] for row in con.execute(f'PRAGMA table_info("{name}")')]:
                con.execute(f'UPDATE "{name}" SET "{column}" = relocated("{column}") '
                            f'WHERE typeof("{column}") = \'text\' AND instr("{column}", ?) > 0', (old,))
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()


def object_store(store: Path, systems: list[Path]) -> Path:
    """The test's object store, a bare repository reading the objects of each of ``systems`` (the run's, none for a
    fresh run) as an alternate, by paths relative to its own, so that a workbench copied off whole reads the same
    wherever it lands; returns its objects folder."""
    if not (store / "HEAD").exists():
        subprocess.run(["git", "init", "--quiet", "--bare", str(store)], env={**os.environ, **GIT_ENV}, check=True)
        _git(store, "config", "gc.auto", "0")
    objects = (store / "objects").resolve()
    if systems:
        (objects / "info" / "alternates").write_text("".join(
            os.path.relpath((system / "state.git" / "objects").resolve(), objects) + "\n" for system in systems))
    return objects


def fork_user(system: Path, objects: Path, source: tuple[Path, int, str] | None, found: Found | None) -> None:
    """``system`` (a test user's ``system/``) as the module's docstring says: a repository whose objects are the test's
    store, and with a ``source`` (a system folder of the run's, its stream's rounds and the commit of the last) that
    commit's tree relocated and committed, and that system's state log up to it."""
    repo, tree = system / "state.git", system / "state"
    system.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", "--bare", str(repo)], env={**os.environ, **GIT_ENV}, check=True)
    _git(repo, "config", "core.bare", "false")
    _git(repo, "config", "gc.auto", "0")
    shutil.rmtree(repo / "objects")
    (repo / "objects").symlink_to(os.path.relpath(objects, repo.resolve()))
    if source is None:
        return
    old, rounds, commit = source
    restore(repo, commit, tree)
    _git(repo, "update-ref", "HEAD", commit)
    _git(repo, "reset", "--quiet", commit, tree=tree)
    lines = [line for line in (old / "state-log.jsonl").read_text().splitlines() if json.loads(line)["round"] < rounds]
    (system / "state-log.jsonl").write_text("".join(line + "\n" for line in lines))
    relocate(tree, str(old), str(system), found)
    _git(repo, "add", "--all", "--force", ".", tree=tree)
    _git(repo, "commit", "--quiet", "--allow-empty", "--no-verify", "-m", f"relocated from {old}", tree=tree)


def seed(run_dir: Path, test_dir: Path, users: int) -> dict:
    """``test_dir`` as its round begins, from nothing: the run's task names and every user's ``system/`` (the
    module's docstring), test user i a fork of the run's system i mod U, of its U. Returns what the users start from:
    the stream's last round and each system's commit of it, in the order the users take them."""
    end = stream_end(run_dir)
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True)
    shutil.copy2(Path(run_dir) / SALT_FILE, test_dir / SALT_FILE)
    systems, rounds = end or ([], 0)
    objects = object_store(test_dir / "objects.git", systems)
    sources: list[tuple[tuple[Path, int, str], Found]] = []
    for system in systems:
        commit = commit_of(system / "state-log.jsonl", rounds - 1)
        scratch = test_dir / "naming"
        restore(system / "state.git", commit, scratch)
        sources.append(((system, rounds, commit), naming(scratch, str(system))))
        shutil.rmtree(scratch)

    def fork(user: int) -> None:
        source, found = sources[user % len(sources)] if sources else (None, None)
        fork_user(user_system(test_dir, user), objects, source, found)

    with ThreadPoolExecutor(FORKS) as pool:
        list(pool.map(fork, range(users)))
    return {"from": None if not sources else {
        "round": rounds - 1,
        "systems": [{"system": str(system), "commit": commit, "relocated": len(found[0]) + len(found[1]),
                     "databases": len(found[2])} for (system, _, commit), found in sources]}}


def play_name(run_dir: Path, test: str) -> str:
    """The folder a run's test goes into under its ``frozen/``: the test's name, or ``<test>-reference`` for a run with
    no stream, which plays it from an empty state."""
    return test if stream_end(run_dir) else f"{test}-reference"


def test(run_dir: Path, schedule: Path, out: Path | None = None) -> Path:
    """Play the frozen test ``schedule`` of the run in ``run_dir`` into ``out`` (default
    ``run_dir/frozen/<its name>/play``, the module's docstring), unless its round is recorded there already; returns
    ``out``. A test cut short is played again from its seeding."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, prior_records, run

    from continual_arc_baselines import secrets
    from continual_arc_baselines.cli import _import
    from continual_arc_baselines.collective import CollectiveHarnessLearner

    run_dir = Path(run_dir).resolve()
    sched = Schedule.load(schedule)
    out = Path(out).resolve() if out else run_dir / "frozen" / play_name(run_dir, sched.meta["test"]) / "play"
    if set(sched.rounds()) != {0}:
        raise SystemExit(f"{schedule} is not a frozen test: one round, every instance a user of its own")
    if (out / "records.jsonl").exists() and len(prior_records(out)) >= len(sched.slots):
        return out
    args = first_launch(run_dir)
    started = seed(run_dir, out, sched.users)
    config = EnvConfig.from_json(json.loads(Path(args.env_config).read_text())) if args.env_config else EnvConfig()
    kwargs = json.loads(args.learner_kwargs) if args.learner_kwargs else {}
    defaults = {**json.loads(args.llm_defaults or "{}")}
    defaults.setdefault("session_id", f"{run_dir.name}-{sched.meta['test']}")
    (out / "frozen.json").write_text(json.dumps({"run": str(run_dir), "test": sched.meta["test"],
                                                 "schedule": sched.name, "users": sched.users,
                                                 "learner": args.learner, "learner_kwargs": kwargs,
                                                 "llm_defaults": defaults, **started}, indent=1))
    learner = CollectiveHarnessLearner(lambda user: _import(args.learner, "learner")(**kwargs), out, schedule=sched,
                                       upstream=args.upstream,
                                       api_key=secrets.get(args.upstream_key_env, default=None),
                                       body_defaults=defaults,
                                       declared_context_length=args.declared_context_length or None)
    registry = _import(args.registry, "registry")() if args.registry else None
    try:   # the round's end commits every user's state (CollectiveHarnessLearner.end_round)
        run(learner, sched, config=config, out_dir=out, registry=registry, learner_name=args.name or
            args.learner.rsplit(":", 1)[-1], env=build_env(sched, config, registry), resume=True,
            alias_style=args.alias_style)
    finally:
        learner.close()
    for user in range(sched.users):
        shutil.rmtree(user_system(out, user) / "state")
    return out
