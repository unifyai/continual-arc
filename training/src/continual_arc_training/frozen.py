"""The two frozen tests after a run's stream, both played frozen and at once: a trained run's (its model on the node's
vLLM), played on its own node once its stream has ended (frozen.sh), or a reference row's (its model on OpenRouter,
continual_arc_training.references), played where the row's memory is (ref-tests.sh). The benchmark defines them
(``configs.test_schedule``; the two for ``arc1-small`` at seed 0 are in this package's ``schedules/``): the held-out
test, one instance of each of the 100 held-out tasks, and the seen test, four new instances of each of the stream's
25 tasks. Each is one round of 100 users, every user starting from the end-of-stream state, so that nothing one test
instance does reaches another and nothing is learnt:

  * the model: a trained run's last phase, trained on every round the stream played, or another adapter given; or a
    reference row's player, or another given; or the base model, for the trained runs' references, rows h0 (the
    learner reset) and bare (the bare model), played from nothing on the stack the trained runs play on (``--row``);
  * the memory: test user i starts from what stream user i mod U, of the run's U users, ended the stream with: its
    harness state, in which every absolute path that named the stream user's folder names the test user's instead (a
    prime-agent state names its folder in its Python environments, its daemon's records and its sessions, and a state
    that kept them would read and write the stream user's own files), a bare program library's kept programs and pairs
    (its ``library/``), and the team's demonstration cache as the stream left it (the demo-cache floor's ``demo-cache/``);
  * the task names: the run's own (its ``alias-salt``), by which the memory names tasks.

A run whose stream committed every user's harness state at the end of every round (the baselines'
CollectiveHarnessLearner keeps each user's ``system/state.git`` and ``state-log.jsonl``) has each test user forked from
its stream user's commit of the stream's last round by the baselines' own seeding, so ``git diff HEAD~1 HEAD`` in a
test user's ``system/state.git`` is what it wrote in its test instance. A run without commits has its users'
``system/state`` folders copied and renamed instead (:func:`copy_states`). Either way the path renamed is the run's
``play/`` as its driver spelt it, never resolved, since a harness names its folder by that spelling.

Each test is a benchmark run of its own, ``OUT/TEST/play`` (OUT the run's ``frozen/`` unless given): records.jsonl,
schedule.json, summary.json, every user's harness and every model call, played by the run's own driver as its stream
played a round, and ``frozen.json``, what it was played from. The runner records the round only whole, so a test cut
short is played again from the end-of-stream state, its users seeded afresh. A reference play is the same test from an
empty state, into ``TEST-reference``: a wiped row's or a floor's second, independent play (``--reference``), whose
spread from its own test is the noise of playing a test twice, or a trained run's references from nothing (``--row``).
A trained run's weights-alone play (``--no-memory``) is its test from an empty state too, but on the run's own adapter
and under its own task names, into ``TEST-weights``: what the weights carry without anything the harness kept.

Subcommands (``python -m continual_arc_training.frozen ...``):
  model    the served name and the path of a trained run's last phase's adapter
  play     one test of a run (``--run RUN --test TEST``, with ``--base-url --served --adapter [--no-memory]`` for a
           trained run or ``--player P [--reference]`` for a reference row), or ``--row ROW --out OUT --base-url URL
           --served NAME`` for a reference from nothing on the node's server
  summary  OUT/summary.json: each play's solves and costs and its frozen.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TESTS = ("held-out", "seen")
SCHEDULES = Path(__file__).resolve().parent / "schedules"
SALT_FILE = "alias-salt"
FROZEN = "frozen"   # a run's tests, beside its play/
# how many users' states are copied at once (a prime-agent state is about 350 MB in some 10,000 files)
COPIES = 16
SQLITE = b"SQLite format 3\0"

Found = tuple[list[str], list[tuple[str, str]], list[str]]


def schedule_path(test: str) -> Path:
    """The test's schedule, ``arc1-small`` at seed 0 (the benchmark's ``configs.test_schedule``)."""
    return SCHEDULES / f"arc1-small-{test}-test-seed0.json"


def stream_users(run: Path) -> int:
    """How many users the run's stream was served to (its schedule's)."""
    return json.loads((run / "play" / "schedule.json").read_text())["users"]


def stream_rounds(run: Path) -> int:
    """How many rounds the run's stream played (its schedule, cut at the round it played last)."""
    return len(json.loads((run / "play" / "schedule.json").read_text())["slots"]) // stream_users(run)


def final_phase(run: Path) -> dict[str, Any]:
    """The run's last phase (``train/latest.json``), refused unless it trained on every round the stream played: the
    model the stream ended with."""
    latest = json.loads((run / "train" / "latest.json").read_text())
    rounds = stream_rounds(run)
    if latest["rounds"] != rounds:
        raise SystemExit(f"{run}'s last phase, {latest['phase']}, trained on {latest['rounds']} rounds, not the "
                         f"{rounds} its stream played")
    return latest


def naming(tree: Path, old: str) -> Found:
    """The files under ``tree`` that name ``old`` (an absolute path) in their text, its links that point under it, and
    its SQLite databases, which may name it in their rows or their write-ahead logs, each by its path relative to
    ``tree``. Any other file holding a NUL byte is binary and passed over: the one kind seen to name its folder is a
    session's saved Python namespace, which prime-agent reloads only into that same session, and each test instance
    is a session of its own."""
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
    (a full-text index) are left as they are: their rows are kept in step with their content's by the database's own
    triggers, and an index that still finds an old path finds what it found before."""
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


def committed(run: Path) -> bool:
    """Whether the run's stream committed its users' harness states round by round (each user's
    ``system/state-log.jsonl``)."""
    return any((run / "play" / "users").glob("u*/system/state-log.jsonl"))


def copy_states(run: Path, test_dir: Path, users: int) -> dict[str, Any]:
    """``test_dir`` from nothing with the run's alias-salt and, for each of its ``users``, a copy of stream user i mod U's
    harness state, renamed: the seeding of a run whose users' states were never committed. Returns what each stream
    user's state renamed its folder in, by user."""
    played = run / "play"
    n = stream_users(run)
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True)
    shutil.copy2(played / SALT_FILE, test_dir / SALT_FILE)
    sources = [played / "users" / f"u{u:03d}" for u in range(n)]
    found = {u: naming(src / "system" / "state", str(src)) for u, src in enumerate(sources)
             if (src / "system" / "state").is_dir()}

    def one(user: int) -> None:
        src, dst = sources[user % n], test_dir / "users" / f"u{user:03d}"
        if user % n not in found:
            return
        (dst / "system").mkdir(parents=True)
        subprocess.run(["cp", "-a", str(src / "system" / "state"), str(dst / "system" / "state")], check=True)
        relocate(dst / "system" / "state", str(src), str(dst), found[user % n])

    with ThreadPoolExecutor(COPIES) as pool:
        list(pool.map(one, range(users)))
    return {f"u{u:03d}": {"files": len(f[0]), "links": len(f[1]), "databases": len(f[2])} for u, f in found.items()}


def seed(run: Path, test_dir: Path, users: int) -> dict[str, Any]:
    """``test_dir`` as its round begins (the module's docstring), from nothing: the run's alias-salt, and for each of its
    ``users`` what stream user i mod U ended the stream with: its harness state, forked from its commit of the stream's
    last round when the run has commits (:func:`committed`) and copied otherwise (:func:`copy_states`), its kept
    programs and pairs, and the team's demonstration cache. Returns what the test starts from, for its frozen.json."""
    played = run / "play"
    n, last = stream_users(run), stream_rounds(run) - 1
    if committed(run):
        from continual_arc_baselines import frozen as forks
        # play/ as the run's driver spelt it, never resolved: the harnesses name their folders by that path
        memory = {"memory": "test user i a fork of stream user i mod stream_users at its commit of the stream's last "
                            "round, relocated", **forks.seed(played, test_dir, users)}
    else:
        memory = {"memory": "test user i from stream user i mod stream_users, relocated",
                  "relocated": copy_states(run, test_dir, users)}
    if (played / "library").is_dir():   # a bare program library's memory, by round and user
        (test_dir / "library").mkdir()
        for user in range(users):
            shutil.copy2(played / "library" / f"r{last:04d}-u{user % n:03d}.json",
                         test_dir / "library" / f"r{last:04d}-u{user:03d}.json")
    if (played / "demo-cache").is_dir():   # the team's demonstration cache, a file a round
        shutil.copytree(played / "demo-cache", test_dir / "demo-cache")
    return {**memory, "stream_users": n, "stream_rounds": last + 1}


def check_served(base_url: str, served: str, adapter: Path) -> None:
    """Refuse to play unless the server answers for ``served`` from the adapter's weights. An adapter no step has
    changed is the base model's own weights, which the check cannot tell apart, and passes."""
    from continual_arc_training.play import MODEL
    from continual_arc_training.server import Server, adapter_is_untrained, check_served_adapter
    if adapter_is_untrained(adapter):
        return
    check = check_served_adapter(Server(base_url), MODEL, served)
    if not (check["answered"] and check["differs"]):
        raise SystemExit(f"{served} failed the served-adapter check at {base_url}: {check}")


def cmd_model(o: argparse.Namespace) -> None:
    from continual_arc_training.realtime import adapter_name
    run = Path(o.run).expanduser()
    latest = final_phase(run)
    print(adapter_name(run.name, latest["phase"]), latest["adapter"])


def cmd_play(o: argparse.Namespace) -> None:
    from continual_arc.schedule import Schedule

    from continual_arc_training import play
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    served = bool(o.base_url and o.served)
    if o.row:
        if o.run or o.player or o.adapter or o.reference or o.no_memory or not (o.out and served):
            raise SystemExit("a reference on the node's server takes --row, --out, --base-url and --served alone")
    elif (not o.run or bool(o.player) == (served and bool(o.adapter)) or (o.reference and not o.player)
          or (o.no_memory and o.player)):
        raise SystemExit("give --run and --player [--reference] (a reference row) or --base-url, --served and "
                         "--adapter [--no-memory] (a trained run)")
    reference = o.reference or bool(o.row)
    run = Path(o.run).expanduser() if o.run else None
    out = Path(o.out).expanduser() if o.out else run / FROZEN
    name = f"{o.test}-reference" if reference else f"{o.test}-weights" if o.no_memory else o.test
    play_dir = out / (o.name or name) / "play"
    schedule = Schedule.load(o.schedule or schedule_path(o.test))
    if schedule.meta.get("test") != o.test or set(schedule.rounds()) != {0}:
        raise SystemExit(f"{o.schedule} is not one round of the {o.test} test")
    if play.round_recorded(play_dir, schedule):
        print(json.dumps({"test": o.test, "played": True}), flush=True)
        return
    if o.adapter:
        check_served(o.base_url, o.served, Path(o.adapter))
    if reference:
        shutil.rmtree(play_dir, ignore_errors=True)
        play_dir.mkdir(parents=True)
        memory: dict[str, Any] = {"memory": "none: played from an empty state"}
    elif o.no_memory:   # the task names stay the run's, which the weights may have learnt to know a task by
        shutil.rmtree(play_dir, ignore_errors=True)
        play_dir.mkdir(parents=True)
        shutil.copy2(run / "play" / SALT_FILE, play_dir / SALT_FILE)
        memory = {"memory": "none: the run's weights alone, from an empty state under the run's task names"}
    else:
        memory = seed(run, play_dir, schedule.users)
    row = o.row or json.loads((run / "play" / "driver.json").read_text())["row"]
    model = {"player": o.player} if o.player else {"served": o.served, **({"adapter": o.adapter} if o.adapter else {})}
    (play_dir / "frozen.json").write_text(json.dumps(
        {"run": run and str(run), "row": row, "test": o.test, "schedule": schedule.name, "users": schedule.users,
         **model, **memory, "started": datetime.now(UTC).isoformat()}, indent=1))
    if not o.player:
        result = play.play_round(row, play_dir, schedule, base_url=o.base_url, served_name=o.served)
    else:
        from continual_arc_training import references
        key = os.environ["OPENROUTER_API_KEY"]
        if row in references.BARE_ROWS:
            result = references.play_bare_round(o.player, row, play_dir, schedule, api_key=key,
                                                rounds_done=0 if reference else stream_rounds(run))
        else:
            result = references.play_round(o.player, row, play_dir, schedule, api_key=key)
    print(json.dumps({"test": o.test, "records": len(result.records),
                      "solved": sum(r.solved for r in result.records)}), flush=True)


def cmd_summary(o: argparse.Namespace) -> None:
    out = Path(o.out).expanduser() if o.out else Path(o.run).expanduser() / FROZEN
    doc: dict[str, Any] = {"run": o.run and str(Path(o.run).expanduser()), "tests": {}}
    for records in sorted(out.glob("*/play/records.jsonl")):
        games = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
        doc["tests"][records.parent.parent.name] = {
            **json.loads((records.parent / "frozen.json").read_text()), "instances": len(games),
            "solved": sum(bool(g["solved"]) for g in games),
            "mean_cost": round(sum(g["cost"] for g in games) / max(1, len(games)), 3)}
    (out / "summary.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({t: {k: v[k] for k in ("instances", "solved", "mean_cost")} for t, v in doc["tests"].items()}),
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model", help="the served name and path of the run's last phase's adapter")
    m.add_argument("--run", required=True, help="the run's folder (realtime.sh's RUNS/rt/NAME)")
    m.set_defaults(fn=cmd_model)
    s = sub.add_parser("play", help="one frozen test of the run, seeded from its end-of-stream state")
    s.add_argument("--run", help="the run whose end-of-stream state the test starts from")
    s.add_argument("--row", help="a reference played from nothing instead (h0 or bare), into --out")
    s.add_argument("--test", choices=TESTS, required=True)
    s.add_argument("--schedule", help="the test's schedule (default this package's, arc1-small at seed 0)")
    s.add_argument("--base-url", dest="base_url", help="a trained run's: the server its model is served on")
    s.add_argument("--served", help="a trained run's: the name the server serves the adapter under")
    s.add_argument("--adapter", help="a trained run's: the adapter's own folder, for the served-adapter check")
    s.add_argument("--player", help="a reference row's: who plays on OpenRouter, its own player or another")
    s.add_argument("--reference", action="store_true",
                   help="a reference row's: the test played from an empty state, into TEST-reference")
    s.add_argument("--no-memory", dest="no_memory", action="store_true",
                   help="a trained run's: its weights alone, from an empty state under the run's task names, into "
                        "TEST-weights")
    s.add_argument("--out", help="where the tests go (default RUN/frozen)")
    s.add_argument("--name", help="the play's folder under OUT (default TEST, TEST-reference or TEST-weights)")
    s.set_defaults(fn=cmd_play)
    u = sub.add_parser("summary", help="OUT/summary.json")
    u.add_argument("--run", help="the run the tests are of; none for a reference from nothing")
    u.add_argument("--out", help="where the tests are (default RUN/frozen)")
    u.set_defaults(fn=cmd_summary)
    o = p.parse_args()
    o.fn(o)


if __name__ == "__main__":
    main()
