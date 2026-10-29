"""A run's two frozen tests, played after its stream from the state the stream left.

After its stream a learner plays two frozen tests (continual-arc's ``configs.test_schedule``): the held-out test, one
instance of each held-out task, and the seen test, new instances of the stream's own tasks. Every instance starts from
the learner as the stream left it, so that nothing one test instance does reaches another and nothing is learnt in
between. ``arc-baselines test RUN --schedule TEST.json [--out OUT] [--collective]`` plays a test of a harness's run,
the run directory ``arc-baselines run --out`` wrote, with the harness the run's first launch built (``launches.jsonl``:
the learner and its settings, the upstream, its key and the request defaults), in one of two ways.

**One instance at a time** (the default, :func:`play_alone`), from a test made with ``--width 1``, every instance a
round of its own with a reset before it. A harness (:class:`~continual_arc_baselines.systems.base.SystemLearner`)
keeps everything it carries between instances as files under its state directory, and before every instance its reset
puts that directory back as the stream left it instead of wiping it (``SystemLearner.freeze``). The state goes back
where it was, at the same path, so every absolute path the harness wrote into its own files (its workspace, its home,
its sessions' folders, the rows of its databases) still names that state, and no file is rewritten, whatever the
harness. The run must be a stream played whole and nothing after it (``--no-epilogue``). For the test the stream's own
state directory is set aside whole, renamed ``system/stream-end-<instances played>`` beside it, and a copy of it is put
in its place before every instance (:func:`put_back`). When the test ends, or is stopped with Ctrl-C, ``kill`` or a
hangup, the copy is deleted and the stream's directory renamed back. A test stopped harder leaves it aside, and the
next test, or the stream taken up again with ``arc-baselines run --resume``, puts it back first (:func:`settle`). One
process at a time plays a run, its stream or a test of it (:func:`holding`), so a run's two tests play one after the
other. The test's logs and model-call journal go to an attempt of its own.

**As one round of users** (``--collective``, :func:`play_forks`), from a test made at its full width, every instance a
user of its own and all of them served at once by
:class:`~continual_arc_baselines.collective.CollectiveHarnessLearner`, a system of the run's own kind for every user.
The system as the stream left it is the commit :mod:`~continual_arc_baselines.statelog` made at the end of the stream's
last round, so the run must have logged its state (``arc-baselines run --log-state``). Such a run has one system, its
attempt's; a collective run has one per user, ``users/uNNN/system``, and test user i starts from the run's user i mod
U, of its U. Each test user's ``system/`` starts as a fork of that commit (:func:`seed`): its ``state`` the commit's
tree, its ``state.git`` on the commit, and its ``state-log.jsonl`` the run's rounds up to it, so that the test round's
commit comes next in the user's history. The tree is relocated first (:func:`relocate`): a harness names its own
folder in its files (a Python environment's scripts, a daemon's records, a session's log, openclaw's and hermes-agent's
SQLite databases), and a copy that kept those paths would read and write the run's own state, so every absolute path
naming the system folder it forks is renamed to the user's, and the renamed tree is committed before the test, so that
the test round's commit shows what the user wrote in its instance and nothing else. The users keep their objects in one
store, ``objects.git``, which reads the run's own as alternates: a user's repository is its own (its branch, index and
log) but its objects are the store's, so a hundred forks cost what the files they wrote cost. The test's round ends
with every user's state committed as its next round, and once the round is recorded the users' working trees go: every
one is in its repository, whole (:func:`continual_arc_baselines.statelog.restore`). A run with no stream starts every
user from nothing, a play that a row's measures are read against.

Either way a stateless system (rows H0, HPL0 and PL0) wipes its state before its test instance, as before every
instance, so it plays from nothing. The test's records go to a run directory of its own, by default
``RUN/frozen/<the test>/play`` (``<the test>-reference/play`` for a collective test of a run with no stream), holding
the run's ``alias-salt``, so that the harness meets its tasks under the names it learnt them by, and ``frozen.json``,
what the test was played from. A test played whole is not played again. One played one instance at a time and cut
short goes on from where it stopped, and is refused once the stream has gone on since; a round of users cut short is
played again from its seeding.

A learner that is not a harness adapter, such as a submitted one (``HostedLearner``), plays a test in the launch that
plays its stream (``arc-baselines run --test``): the test's one round with the same learner, from the state the stream
left it, or, for a learner that keeps everything as files, the same instances one at a time, its files put back before
each (:func:`one_at_a_time`).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from continual_arc.aliases import SALT_FILE
from continual_arc.schedule import PHASE_STREAM, TEST_SEEN, TESTS, Schedule

from continual_arc_baselines.statelog import GIT_ENV, _git, commit_of, restore

HELD = "stream-end"
"""What a stream's state directory is renamed to, beside it, while a test plays from copies of it."""
PLAYED_FROM = "frozen.json"
"""What a test was played from, in its run directory."""
FORKS = 16
"""How many test users are forked at once (a hermes-agent state is some 100 MB, an openclaw one 500 MB)."""
SQLITE = b"SQLite format 3\0"

Found = tuple[list[str], list[tuple[str, str]], list[str]]


def first_launch(run_dir: Path):
    """The run's first launch, as ``arc-baselines run`` parsed it (``launches.jsonl``)."""
    from continual_arc_baselines.cli import build_parser

    path = Path(run_dir) / "launches.jsonl"
    if not path.is_file():
        raise SystemExit(f"{run_dir} has no launches.jsonl: a test follows a stream played with arc-baselines run --out")
    return build_parser().parse_args(json.loads(path.read_text().splitlines()[0])["argv"])


# -- a state directory saved and put back ---------------------------------------


def put_back(saved: Path, state: Path) -> None:
    """Make ``state`` a copy of ``saved``, whole: deleted and copied again, links as links. A copy whole takes a second
    or two for a harness's few hundred megabytes, and takes in any change an instance made, one that kept a file's size
    and modification time or changed only the case of its name included. Anything that is neither a file, a directory
    nor a link, such as the socket of a process that has stopped, is not copied."""
    if state.exists():
        delete(state)
    shutil.copytree(saved, state, symlinks=True, ignore=_special)


def _special(folder: str, names: list[str]) -> list[str]:
    def kind(name: str) -> int:
        return stat.S_IFMT(os.lstat(os.path.join(folder, name)).st_mode)

    return [name for name in names if kind(name) not in (stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK)]


def delete(path: Path) -> None:
    """``shutil.rmtree``, through directories made read-only (a Go module cache's) or unreadable, which refuse to lose
    an entry, or to be listed, until they are made writable and readable again."""
    def unlocked(function, failed, error):
        if not isinstance(error, PermissionError):
            raise error
        for folder in (os.path.dirname(failed), failed):
            if os.path.isdir(folder) and not os.path.islink(folder):
                os.chmod(folder, os.stat(folder).st_mode | stat.S_IRWXU)
        if os.path.isdir(failed) and not os.path.islink(failed):
            shutil.rmtree(failed, onexc=unlocked)
        else:
            os.unlink(failed)

    shutil.rmtree(path, onexc=unlocked)


# -- one instance at a time -------------------------------------------------------


def check(run_dir: Path, test: Schedule, name: str) -> int:
    """Refuse, before anything plays, the test ``test`` (the schedule at ``name``) unless it is a frozen test played one
    instance at a time, every instance after the first reset before it, of the configuration and seed of the run in
    ``run_dir``, whose schedule is a stream alone, played whole; returns how many instances the stream played."""
    from continual_arc.runner import prior_records

    if test.meta.get("test") not in TESTS:
        raise SystemExit(f"{name} is not a frozen test: make one with `continual-arc schedule make --test`")
    if test.users != 1 or not all(slot.reset_before for slot in test.slots[1:]):
        raise SystemExit(f"{name} does not play one instance at a time, each from the stream's end, and a harness keeps "
                         "one conversation: make the test with --width 1, or play it as one round of users with "
                         "--collective")
    stream = Schedule.load(Path(run_dir) / "schedule.json")
    if PHASE_STREAM not in stream.phases():
        raise SystemExit(f"{run_dir} has no stream for a test to follow")
    if stream.phases() != [PHASE_STREAM]:
        raise SystemExit(f"{run_dir} plays its stream and then {', '.join(stream.phases()[1:])}, so its system's state "
                         "is not the state the stream left: play the stream alone (arc-baselines run --no-epilogue)")
    played = len(prior_records(run_dir))
    if played < len(stream.slots):
        raise SystemExit(f"{run_dir} has played {played} of its stream's {len(stream.slots)} instances")

    def made_for(schedule: Schedule) -> tuple:
        return schedule.seed, (schedule.meta.get("config") or {}).get("name")

    if made_for(test) != made_for(stream):
        raise SystemExit(f"{name} is the test of {made_for(test)[1]} at seed {made_for(test)[0]}, and {run_dir}'s "
                         f"stream is {made_for(stream)[1]} at seed {made_for(stream)[0]}")
    seen_for(test, stream, name, run_dir)
    return played


def seen_for(test: Schedule, stream: Schedule, name: str, run_dir: Path) -> None:
    """Refuse the seen test ``test`` unless it was made for as many users as ``stream`` serves: each user brings tasks
    of their own (``continual_arc.configs.user_tasks``), so the seen test after U users' stream is over their tasks."""
    if test.meta.get("test") == TEST_SEEN and test.meta.get("stream_users", 1) != stream.users:
        raise SystemExit(f"{name} is the seen test after {test.meta.get('stream_users', 1)} users' stream, and "
                         f"{run_dir}'s stream serves {stream.users}: make it with `continual-arc schedule make "
                         f"--test seen --users {stream.users}`")


def system_of(run_dir: Path) -> Path:
    """The run's system folder, in the attempt its launches wrote to (``attempt.json``)."""
    return Path(json.loads((Path(run_dir) / "attempt.json").read_text())["attempt_dir"]) / "system"


@contextmanager
def holding(run_dir: Path) -> Iterator[None]:
    """The run directory ``run_dir`` for one process at a time, its stream or a test of it, which sets the stream's
    state aside, taken before either writes anything; the lock is the directory's own, and goes with the process."""
    folder = os.open(run_dir, os.O_RDONLY)
    try:
        try:
            fcntl.flock(folder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another process is playing {run_dir}, its stream or a test of it: one at a time plays "
                             "a run") from None
        yield
    finally:
        os.close(folder)


def hold_aside(system: Path, played: int) -> Path:
    """The stream's state directory, ``system/state``, renamed aside for a test from the stream's end after ``played``
    instances (:data:`HELD`); returns where it is."""
    held = system / f"{HELD}-{played}"
    state = system / "state"
    state.mkdir(exist_ok=True)   # a stream that kept nothing
    state.rename(held)
    return held


def give_back(system: Path, held: Path) -> None:
    """The stream's state directory renamed back from where a test set it aside, the test's copy in its place
    deleted."""
    state = system / "state"
    if state.exists():
        delete(state)
    held.rename(state)


def settle(system: Path, played: int) -> None:
    """Put back the stream's state directory that a test stopped hard left aside, when the stream had played
    ``played`` instances, the stream's count now; one set aside at any other point is refused, since the state played
    on since is not the stream's."""
    for held in system.glob(f"{HELD}-*"):
        if held.name != f"{HELD}-{played}":
            raise SystemExit(f"{held} is the stream's state as a test set it aside, and the stream has played {played} "
                             f"instances since, from the test's copy in {system / 'state'}")
        give_back(system, held)


def play_alone(run_dir: Path, schedule: Path, out: Path | None = None, *, quiet: bool = False,
               argv: list[str] | None = None) -> int:
    """Play the frozen test ``schedule`` of the harness's stream in ``run_dir`` one instance at a time into ``out``
    (default ``run_dir/frozen/<the test>/play``), every instance from a copy of the state the stream left, at the
    stream's own path (the module's docstring); ``argv`` is the command line, for the test's ``launches.jsonl``."""
    from continual_arc.runner import prior_records

    from continual_arc_baselines.cli import import_object, play
    from continual_arc_baselines.systems.base import SystemLearner

    run_dir = Path(run_dir).resolve()
    test = Schedule.load(schedule)
    played = check(run_dir, test, str(schedule))
    out = Path(out) if out else run_dir / "frozen" / test.meta["test"] / "play"
    played_from = {"stream": str(run_dir), "instances": played}
    if prior_records(out):
        if not (out / PLAYED_FROM).is_file() or json.loads((out / PLAYED_FROM).read_text()) != played_from:
            raise SystemExit(f"{out} holds a test played from another stream end than {run_dir}'s after {played} "
                             "instances: give this test an --out of its own")
        if len(prior_records(out)) == len(test.slots) and Schedule.load(out / "schedule.json").slots == test.slots:
            print(f"{out} holds the whole test already", file=sys.stderr)
            return 0
    launch = first_launch(run_dir)
    learner = import_object(launch.learner, "learner")
    if not (isinstance(learner, type) and issubclass(learner, SystemLearner) and learner.state_in_files):
        raise SystemExit(f"{launch.learner} keeps no state as files in a state directory, so nothing of its stream's "
                         "end is on disk to play a test from: a harness adapter (SystemLearner) does. Play a Python "
                         "learner's tests from a copy of it, as continual-arc's README shows")
    # the test alone, into a run directory of its own, from copies of the stream's state, which are never committed
    launch.config, launch.out, launch.resume, launch.stop_after, launch.log_state = None, str(out), True, None, False
    launch.quiet, launch.argv = quiet, list(argv or [])
    # stopped with kill or a hangup, a test still gives the stream's state back as it ends, as it does on Ctrl-C
    stops = {sig: signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
             for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        with holding(run_dir):
            out.mkdir(parents=True, exist_ok=True)
            (out / PLAYED_FROM).write_text(json.dumps(played_from))
            shutil.copy2(run_dir / SALT_FILE, out / SALT_FILE)   # the names the stream gave its tasks
            return play(launch, test, frozen_from=(system_of(run_dir), played))
    finally:
        for sig, handler in stops.items():
            signal.signal(sig, handler)


# -- as one round of users, each forked from the stream's last commit ----------------


def user_system(run_dir: Path, user: int) -> Path:
    """A user's ``system/`` in a run :class:`~continual_arc_baselines.collective.CollectiveHarnessLearner` plays (its
    ``user_dir``): a collective run's, or a test's."""
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
                             "play the stream with arc-baselines run --log-state, or its tests one instance at a time "
                             "(arc-baselines test without --collective)")
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
    run with no stream) as an alternate, by paths relative to its own, so that a run directory copied whole reads the
    same wherever it lands; returns its objects folder."""
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


def play_forks(run_dir: Path, schedule: Path, out: Path | None = None) -> Path:
    """Play the frozen test ``schedule`` of the run in ``run_dir`` as one round of users into ``out`` (default
    ``run_dir/frozen/<the test>/play``, or ``<the test>-reference/play`` for a run with no stream), every user forked
    from the stream's last commit (the module's docstring), unless its round is recorded there already; returns
    ``out``."""
    from continual_arc.runner import build_env, prior_records, run

    from continual_arc_baselines.cli import import_object, launch_settings
    from continual_arc_baselines.collective import CollectiveHarnessLearner

    run_dir = Path(run_dir).resolve()
    test = Schedule.load(schedule)
    if out is None:
        name = test.meta["test"] if stream_end(run_dir) else f"{test.meta['test']}-reference"
        out = run_dir / "frozen" / name / "play"
    out = Path(out).resolve()
    if set(test.rounds()) != {0}:
        raise SystemExit(f"{schedule} is not one round, every instance a user of its own: make the test at its full "
                         "width, or play it one instance at a time without --collective")
    if stream_end(run_dir):
        seen_for(test, Schedule.load(run_dir / "schedule.json"), str(schedule), run_dir)
    if (out / "records.jsonl").exists() and len(prior_records(out)) >= len(test.slots):
        return out
    launch = first_launch(run_dir)
    started = seed(run_dir, out, test.users)
    config, kwargs, registry, learner_name = launch_settings(launch, test)
    defaults = json.loads(launch.llm_defaults or "{}")
    defaults.setdefault("session_id", f"{run_dir.name}-{test.meta['test']}")
    (out / PLAYED_FROM).write_text(json.dumps({"run": str(run_dir), "test": test.meta["test"], "schedule": test.name,
                                               "users": test.users, "learner": launch.learner,
                                               "learner_kwargs": kwargs, "llm_defaults": defaults, **started},
                                              indent=1))
    learner = CollectiveHarnessLearner(lambda user: import_object(launch.learner, "learner")(**kwargs), out,
                                       schedule=test, upstream=launch.upstream,
                                       api_key=os.environ.get(launch.upstream_key_env) or None, body_defaults=defaults,
                                       declared_context_length=launch.declared_context_length or None)
    try:   # the round's end commits every user's state (CollectiveHarnessLearner.end_round)
        run(learner, test, config=config, out_dir=out, registry=registry, learner_name=learner_name,
            env=build_env(test, config, registry), resume=True, alias_style=launch.alias_style)
    finally:
        learner.close()
    for user in range(test.users):
        shutil.rmtree(user_system(out, user) / "state")
    return out


# -- a test after the stream, in the launch that plays it -----------------------------


def load_test(path: str) -> Schedule:
    """The frozen test ``arc-baselines run --test`` names (continual-arc's ``configs.test_schedule``), refused unless it
    is one round, every instance a user of its own, as a test is made by default: a learner whose state lives in its
    own process plays it from the state its stream left it with no snapshot to restore, and one that can be frozen
    plays the same instances one at a time (:func:`one_at_a_time`)."""
    test = Schedule.load(path)
    if test.meta.get("test") not in TESTS:
        raise SystemExit(f"{path} is not a frozen test (`continual-arc schedule make --test`): its meta names no test")
    if set(test.rounds()) != {0}:
        raise SystemExit(f"{path} is not one round: a test played after the stream by the same learner must serve "
                         "every instance to a user of its own at once")
    return test


def one_at_a_time(test: Schedule) -> Schedule:
    """``test``'s instances in its order, every one a round of its own with the learner put back before it, as
    ``continual-arc schedule make --test ... --width 1`` lays it out: the same test, served one instance at a time."""
    return Schedule(name=test.name, seed=test.seed, slots=[replace(s, user=0, reset_before=True) for s in test.slots],
                    sources=dict(test.sources), users=1, meta={**test.meta, "width": 1})
