"""Per-round memory-use counters for every harness row: how much of a harness's state persists across rounds, and how
often a round's games draw on it.

Two readings:

- **State on disk** (:func:`snapshot_state`, :func:`diff_snapshot`): a snapshot (path, size, mtime, sha256) of a user's
  harness state (``users/uNNN/system/state``, prime-agent's state folder), grouped into the areas prime-agent writes
  under it (:func:`area_of`), but only the files a game writes on purpose. The harness's own bookkeeping (session
  transcripts, daemon journals, logs, session leases, its per-session ledger, its own config: :data:`SNAPSHOT_EXCLUDE`)
  changes every game by construction and is left out, since counting it would show a frozen row saving something every
  round. Against the previous round's snapshot of the same user, three counts per area: files added, changed (a
  different hash or, past :data:`SNAPSHOT_MAX_FILE_BYTES`, a different size or mtime) and removed. A wiped row's
  snapshot is taken after its round's instance (the wipe runs at the next call's start), so it shows that round's own
  churn; what the wipe guarantees is that no file survives unchanged into the next round, so ``changed`` stays 0 in
  every area for a wiped row.
- **Reads in play** (:func:`read_areas`): what each of a game's captured calls drew from those areas. Only the
  student's own calls in play count (:func:`in_play`: phase "turn", with tools offered), never the closing turns after
  the verdict, where a teacher's storer writes, nor the harness's own subsystems on the student's model (its context
  summarizer, its refinement), which offer no tools. A program is read only when its source comes back: the game's own
  task's program (:func:`program_path`), as the folder held it when the game began (:func:`kept_at_start`), in the tool
  results the call answers (:func:`answered`, every tool message its request ends with) -- two of its distinctive
  lines (:func:`marks`), or its one. A path string is not enough: the student names the path when it writes over,
  deletes or lists the program and in calls to tools that do not exist, the summarizer names it too, and a path built
  with ``os.path.join`` never spells it out. The other areas are read by a reference to them (:data:`READ_PATTERNS`) in
  the call's own text (:func:`_call_text`), never in the system prompt or the benchmark's game messages: a heuristic,
  since a write names its path as a read does.

:func:`round_report` ties both together into ``rounds/rNNNN/memory.json``: per user and in total, files added, changed
and removed per area, reads per area, and two per-game signals (:func:`game_memory_outcome`): ``reused_program`` (its
task's program came back to it at or before its first submission) and ``solved_without_demo`` (it solved without
requesting a demonstration). ``workspace_other`` (the kernel's per-game working files) and ``other`` (anything left once
the known bookkeeping is pruned) are scratch, kept in ``memory.json`` but never summed into the headline
(:func:`summary_line`), since a round's scratch churn is large and per-game by nature.

``python -m continual_arc_training.memory line MEMORY.json`` prints a report's one-line summary (paper.sh), and
``python -m continual_arc_training.memory recount PLAY_DIR`` counts a played run's reads again into its rounds'
``memory.json`` (:func:`recount`).
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

# What a snapshot leaves out: runtimes and caches the harness installs for itself, not what it has learned
# ("kernel-venv" to "*.lock"; "kernel-venv" is the copy of the kernel's Python environment that the states earlier runs
# of the prime-agent adapter saved hold, the paper's among them, and that the kernel never ran on), and its own
# bookkeeping, which changes every game by construction: per-session state and
# logs ("sessions", "session-artifacts"), the daemon's command journals ("daemon-workers"), its logs ("logs"), its
# session leases ("session-leases"), its per-session ledger ("rlm-ledger") and the supervisor's lock folder
# ("supervisor-owners").
SNAPSHOT_EXCLUDE = ("kernel-venv", "node_modules", ".cache", "__pycache__", ".npm", "tmp", "*.pyc", "*.sock", "*.lock",
                   "sessions", "session-artifacts", "daemon-workers", "logs", "session-leases", "supervisor-owners",
                   "rlm-ledger")
# The harness's own provider and model config, excluded by its path (a bare name would sweep up a saved program's own
# data file too): it is rewritten with a fresh proxy port on every process start, so counting it would show every row
# changing its harness store every round.
SNAPSHOT_EXCLUDE_PATHS = ("agent/auth.json", "agent/models.json", "agent/settings.json")
SNAPSHOT_MAX_FILE_BYTES = 2_000_000
_HASH_CHUNK_BYTES = 1 << 20

# The areas a save/change/removal is grouped into. The first four are deliberate (:data:`DELIBERATE_AREAS`, what
# the headline reports, and the areas a read is counted in); the last two are scratch, reported but never summed into
# it (the module docstring).
AREAS = ("programs", "skills", "harness_store", "notes", "workspace_other", "other")
DELIBERATE_AREAS = ("programs", "skills", "harness_store", "notes")
SCRATCH_AREAS = ("workspace_other", "other")


def system_dir(run_dir: Path, user: int) -> Path:
    """User ``user``'s harness folder in a run, where the baselines' ``CollectiveHarnessLearner`` attaches its system:
    its state (``state/``) and the state's history, a commit at the end of every round (``state.git``, listed by round
    in ``state-log.jsonl``: :mod:`continual_arc_baselines.statelog`)."""
    return Path(run_dir) / "users" / f"u{user:03d}" / "system"


def _hash_file(path: Path, max_bytes: int) -> str | None:
    """``path``'s sha256, or None past ``max_bytes`` or an unreadable file (removed or locked mid-walk): a changed
    file whose hash is unknown on either side falls back to its size and mtime (:func:`_changed`)."""
    try:
        if path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def snapshot_state(state_dir: Path, *, exclude: tuple[str, ...] = SNAPSHOT_EXCLUDE,
                   exclude_paths: tuple[str, ...] = SNAPSHOT_EXCLUDE_PATHS,
                   max_file_bytes: int = SNAPSHOT_MAX_FILE_BYTES) -> dict[str, dict[str, Any]]:
    """Every file under ``state_dir`` (posix-relative path -> {"size", "mtime", "hash"}), except symlinks,
    ``exclude``'s runtimes/caches/bookkeeping (by bare name, pruned at any depth) and ``exclude_paths``' bookkeeping
    (by its exact relative path, since these names are not safe to prune wherever they appear); empty if
    ``state_dir`` does not exist (a user never started, or a wiped row between rounds)."""
    out: dict[str, dict[str, Any]] = {}
    if not state_dir.is_dir():
        return out
    for root, dirs, files in os.walk(state_dir):
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pat) for pat in exclude)]
        for name in files:
            if any(fnmatch.fnmatch(name, pat) for pat in exclude):
                continue
            path = Path(root) / name
            if path.is_symlink():
                continue
            rel = path.relative_to(state_dir).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in exclude_paths):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue   # removed or locked mid-walk: not there to count
            out[rel] = {"size": stat.st_size, "mtime": stat.st_mtime, "hash": _hash_file(path, max_file_bytes)}
    return out


def area_of(rel_path: str) -> str:
    """Which of :data:`AREAS` a state-relative path belongs to: a note (a file named ``AGENTS.md``, wherever it sits);
    the program library's (``workspace/programs/``); personal skills (``agent/skills/``, also reachable through the
    ``home/.prime/agent`` link, which a snapshot never follows); the global store prime-agent's refinement writes
    (``agent/harness/``); the rest of the kernel's working folder (``workspace/``: per-game scratch); or "other" for
    anything left once the known bookkeeping is pruned, scratch too, so that an unforeseen leftover cannot inflate the
    deliberate count."""
    parts = Path(rel_path).parts
    if Path(rel_path).name == "AGENTS.md":
        return "notes"
    if parts[:2] == ("workspace", "programs"):
        return "programs"
    if parts[:1] == ("workspace",):
        return "workspace_other"
    if parts[:2] == ("agent", "skills"):
        return "skills"
    if parts[:2] == ("agent", "harness"):
        return "harness_store"
    return "other"


def _changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if before.get("hash") is not None and after.get("hash") is not None:
        return before["hash"] != after["hash"]
    return before["size"] != after["size"] or before["mtime"] != after["mtime"]


def diff_snapshot(previous: dict[str, dict[str, Any]],
                  current: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """{area: {"added", "changed", "removed"}} between two :func:`snapshot_state` results, over :data:`AREAS`."""
    out = {area: {"added": 0, "changed": 0, "removed": 0} for area in AREAS}
    for path, meta in current.items():
        area = area_of(path)
        if path not in previous:
            out[area]["added"] += 1
        elif _changed(previous[path], meta):
            out[area]["changed"] += 1
    for path in previous:
        if path not in current:
            out[area_of(path)]["removed"] += 1
    return out


# A read's patterns for the deliberate areas but programs, which are read by their source (:func:`opened`).
# prime-agent's own system prompt lists its bundled skills by their installed path (".../prime-agent/skills/<name>/
# SKILL.md", in which "agent/skills/" is a substring), so the lookbehind rejects a match preceded by a word character or
# a hyphen while accepting a bare "agent/skills/" or ".prime/agent/skills/".
READ_PATTERNS: dict[str, str] = {
    "skills": r"(?<![\w-])agent/skills/",
    "harness_store": r"(?<![\w-])agent/harness/",
    "notes": r"AGENTS\.md",
}
# Where a harness keeps its programs (area_of's "programs"), one per task, named by the task's label.
PROGRAMS = "workspace/programs"
# Lines common to many grid programs, which a tool result can carry without the kept program in it.
GENERIC = frozenset({"def transform(grid):", "return out", "return result", "return output", "import numpy as np",
                     "from collections import Counter", "from collections import deque"})


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def in_play(call: dict[str, Any]) -> bool:
    """Whether a captured call is one of the student's own in the game (phase "turn") with tools offered: the harness's
    context summarizer and its refinement call the student's model offering none, and the closing turns after the
    verdict (phase "after") are where a teacher's storer writes, after the game has drawn on all it will."""
    return call.get("phase") == "turn" and "tools" in (call.get("request") or {})


def answered(call: dict[str, Any]) -> str:
    """The tool results the call answers: every tool message its request ends with, since a reply that makes several
    tool calls at once gets their results back as that many messages. A request whose ``messages`` is no list (a body
    the model's own sandbox code posted through the proxy, captured beside the harness's calls) answers none."""
    messages = (call.get("request") or {}).get("messages")
    results: list[str] = []
    for message in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(message, dict) or message.get("role") != "tool":
            break
        results.append(_text_of(message.get("content")))
    return "\n".join(reversed(results))


def _call_text(call: dict[str, Any]) -> str:
    """The text a reference to an area could show up in, for one captured call: the model's own output for this call
    (its reply's content and reasoning, and any tool call's code) and the tool results it answers (:func:`answered`),
    never the system prompt or the benchmark's game messages further back. Reading only the call's own addition, not
    the whole conversation every call carries again, keeps this linear in the game's length."""
    choices = (call.get("response") or {}).get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    parts = [_text_of(message.get("content")), _text_of(message.get("reasoning"))]
    parts += [json.dumps((tc or {}).get("function") or {}, default=str) for tc in message.get("tool_calls") or []]
    parts.append(answered(call))
    return "\n".join(parts)


def program_path(doc: dict[str, Any]) -> str:
    """Where a harness keeps the program of a game's own task: named by the task's label, its id as the agent sees it
    (``task_label``, which the captured game records)."""
    return f"{PROGRAMS}/{doc['task_label']}.py"


def marks(program: str) -> list[str]:
    """Up to twelve of the program's lines that a tool result carries only with the program: 25 characters or more,
    stripped, and not :data:`GENERIC`."""
    return [ln for ln in (s.strip() for s in program.split("\n")) if len(ln) >= 25 and ln not in GENERIC][:12]


def opened(call: dict[str, Any], program_marks: list[str]) -> bool:
    """Whether the program came back in the tool results the call answers: two of its marks, or its one (a program
    with no mark is never counted as opened)."""
    if not program_marks:
        return False
    result = answered(call)
    return sum(m in result for m in program_marks) >= min(2, len(program_marks))


def _committed(system: Path, round_: int) -> str:
    """The commit of a user's harness state at the end of round ``round_``, from its ``state-log.jsonl``."""
    commits = {e["round"]: e["commit"] for e in map(json.loads, (system / "state-log.jsonl").read_text().splitlines())}
    return commits[round_]


def kept_at_start(system: Path, round_: int, path: str, stateless: bool) -> str | None:
    """The file at ``path`` (relative to the state folder) as a user's harness folder, ``system`` (:func:`system_dir`),
    held it when round ``round_``'s game began: the state's commit at the end of the round before. None when the folder
    held no such file, and always for round 0 and for a row wiped before every game (``stateless``), whose games begin
    from nothing (the commit of the round before is the state its wipe then removed)."""
    if stateless or round_ == 0:
        return None
    repo, commit = f"--git-dir={system / 'state.git'}", _committed(system, round_ - 1)
    held = subprocess.run(["git", repo, "ls-tree", "--name-only", commit, "--", path], check=True,
                          capture_output=True, text=True).stdout
    if not held.strip():
        return None
    shown = subprocess.run(["git", repo, "show", f"{commit}:{path}"], check=True, capture_output=True).stdout
    return shown.decode("utf-8", "replace")


def read_areas(call: dict[str, Any], program_marks: list[str]) -> set[str]:
    """Which of :data:`DELIBERATE_AREAS` this one call reads, the game's own task's program given by its
    ``program_marks`` (:func:`marks` of the program as the folder held it when the game began; none if it held none):
    nothing for a call not :func:`in_play`; "programs" if the program came back to it (:func:`opened`); the other areas
    by their :data:`READ_PATTERNS` in the call's own text (:func:`_call_text`)."""
    if not in_play(call):
        return set()
    text = _call_text(call)
    areas = {area for area, pattern in READ_PATTERNS.items() if re.search(pattern, text)}
    if opened(call, program_marks):
        areas.add("programs")
    return areas


def count_reads(call_areas: list[set[str]]) -> dict[str, int]:
    """One game's reads per area, from its calls' :func:`read_areas`: at most one per area per call (a call that both
    opens a skill and has its program come back counts once in each), summed over every call of the game."""
    counts = {area: 0 for area in DELIBERATE_AREAS}
    for areas in call_areas:
        for area in areas:
            counts[area] += 1
    return counts


def game_memory_outcome(doc: dict[str, Any], call_areas: list[set[str]]) -> dict[str, bool]:
    """Two per-game signals, from this game's calls' :func:`read_areas` (``call_areas``, one set per call in order):

    - ``reused_program``: its task's program came back to it (a ``programs`` read) at or before the game's first
      submission (the turns' ``Submit`` action): it had its stored program before answering. A game that never submits
      counts any ``programs`` read.
    - ``solved_without_demo``: the game solved and never requested a demonstration (``RequestDemos``)."""
    turns = doc.get("turns") or []
    first_submit = next((i for i, t in enumerate(turns) if ((t or {}).get("action") or {}).get("type") == "Submit"),
                        None)
    requested_demo = any(((t or {}).get("action") or {}).get("type") == "RequestDemos" for t in turns)
    reused_program = False
    for call, areas in zip(doc.get("calls") or [], call_areas, strict=True):
        if "programs" not in areas:
            continue
        turn = call.get("turn")
        if first_submit is None or turn is None or turn <= first_submit:
            reused_program = True
            break
    return {"reused_program": reused_program, "solved_without_demo": bool(doc.get("solved")) and not requested_demo}


def game_reads(captured: Path, system: Path, round_: int, stateless: bool) -> dict[str, Any]:
    """One user's game of round ``round_`` (its captured game, ``captured``): reads per area (:func:`count_reads`) and
    :func:`game_memory_outcome`'s two signals, its task's program as the user's harness folder, ``system``, held it when
    the game began (:func:`kept_at_start`). A user with no captured game (the round never reached it) has no reads and
    neither signal."""
    if not captured.is_file():
        return {"reads": {area: 0 for area in DELIBERATE_AREAS}, "reused_program": False, "solved_without_demo": False}
    with gzip.open(captured, "rt") as fp:
        doc = json.load(fp)
    program = kept_at_start(system, round_, program_path(doc), stateless)
    program_marks = marks(program) if program is not None else []
    call_areas = [read_areas(c, program_marks) for c in doc.get("calls") or []]
    return {"reads": count_reads(call_areas), **game_memory_outcome(doc, call_areas)}


def _total(per_user: dict[str, Any]) -> dict[str, Any]:
    users = list(per_user.values())
    return {"memory": {area: {k: sum(u["memory"][area][k] for u in users) for k in ("added", "changed", "removed")}
                       for area in AREAS},
            "reads": {area: sum(u["reads"][area] for u in users) for area in DELIBERATE_AREAS},
            "reused_program": sum(int(u["reused_program"]) for u in users),
            "solved_without_demo": sum(int(u["solved_without_demo"]) for u in users)}


def round_report(run_dir: Path, round_: int, users: int, round_dir_of: Callable[[Path, int], Path],
                 stateless: bool) -> dict[str, Any]:
    """Build and write ``round_dir_of(run_dir, round_) / "memory.json"``: this round's per-user and total files added,
    changed and removed per area (each user's state, :func:`system_dir`, against the previous round's snapshot of it, or
    nothing for round 0), reads per area, and how many games show :func:`game_memory_outcome`'s two signals
    (:func:`game_reads`; ``stateless`` for a row wiped before every game). Each user's snapshot is kept beside it
    (``memory/uNNN.json``) for the next round's diff, so this only ever needs the previous round's files. A user with
    no captured game this round is counted on its snapshot alone, with no reads and neither signal."""
    this_dir = round_dir_of(run_dir, round_)
    snap_dir = this_dir / "memory"
    snap_dir.mkdir(parents=True, exist_ok=True)
    prev_snap_dir = round_dir_of(run_dir, round_ - 1) / "memory" if round_ > 0 else None

    per_user: dict[str, Any] = {}
    for u in range(users):
        system = system_dir(run_dir, u)
        snapshot = snapshot_state(system / "state")
        (snap_dir / f"u{u:03d}.json").write_text(json.dumps(snapshot))
        previous: dict[str, dict[str, Any]] = {}
        if prev_snap_dir is not None:
            prev_file = prev_snap_dir / f"u{u:03d}.json"
            if prev_file.is_file():
                previous = json.loads(prev_file.read_text())
        per_user[str(u)] = {"memory": diff_snapshot(previous, snapshot),
                            **game_reads(this_dir / "captured" / f"u{u:03d}.json.gz", system, round_, stateless)}

    report = {"round": round_, "users": per_user, "total": _total(per_user)}
    (this_dir / "memory.json").write_text(json.dumps(report, indent=1))
    return report


def recount(run_dir: Path) -> list[dict[str, Any]]:
    """Count a played run's reads again (:func:`game_reads`) into every round's ``memory.json`` under
    ``run_dir/rounds``, in place of the reads and signals it held, each round's files added, changed and removed kept as
    its snapshot recorded them, so that a round's ``memory.json`` reads as :func:`round_report` writes it whatever code
    wrote it. Whether the run's harness was wiped before every game is its ``driver.json``'s ``stateless``. Returns the
    rewritten reports in round order."""
    run_dir = Path(run_dir)
    stateless = json.loads((run_dir / "driver.json").read_text())["stateless"]
    reports = []
    for path in sorted((run_dir / "rounds").glob("r[0-9]*/memory.json")):
        report = json.loads(path.read_text())
        for user, entry in report["users"].items():
            u = int(user)
            entry.update(game_reads(path.parent / "captured" / f"u{u:03d}.json.gz", system_dir(run_dir, u),
                                    report["round"], stateless))
        report["total"] = _total(report["users"])
        path.write_text(json.dumps(report, indent=1))
        reports.append(report)
    return reports


def summary_line(report: dict[str, Any]) -> str:
    """The one-line summary a round's progress log gets: files added, changed and removed over the deliberate areas
    alone (programs, skills, the harness store, notes), reads over every area, and the two per-game signals, totalled
    over the round."""
    total = report["total"]
    deliberate = [total["memory"][area] for area in DELIBERATE_AREAS]
    added = sum(a["added"] for a in deliberate)
    changed = sum(a["changed"] for a in deliberate)
    removed = sum(a["removed"] for a in deliberate)
    reads = sum(total["reads"].values())
    return (f"round {report['round']} memory: {added} added, {changed} changed, {removed} removed, {reads} reads, "
           f"{total['reused_program']} games reused a saved program, {total['solved_without_demo']} solved "
           f"without a demo")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="a round's memory report as one line, or a played run's reads counted again")
    sub = p.add_subparsers(dest="cmd", required=True)
    line = sub.add_parser("line")
    line.add_argument("report", help="a round's memory.json")
    again = sub.add_parser("recount", help="count a played run's reads again into its rounds' memory.json, printing "
                                           "each round's summary line")
    again.add_argument("run_dir", help="the run's play folder: rounds/, users/ and driver.json")
    a = p.parse_args()
    if a.cmd == "line":
        print(summary_line(json.loads(Path(a.report).read_text())))
    else:
        for report in recount(Path(a.run_dir).expanduser()):
            print(summary_line(report))


if __name__ == "__main__":
    main()
