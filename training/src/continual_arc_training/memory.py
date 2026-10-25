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
- **Reads in play** (:func:`read_areas`): a game's captured calls scanned for a reference to one of those areas, only
  in the model's own output (a reply's text, reasoning and tool-call code) and in tool results, never in the system
  prompt or the benchmark's game messages.

:func:`round_report` ties both together into ``rounds/rNNNN/memory.json``: per user and in total, files added, changed
and removed per area, reads per area, and two per-game signals (:func:`game_memory_outcome`): ``reused_program`` (it
read or ran something from ``programs/`` at or before its first submission) and ``solved_without_demo`` (it solved
without requesting a demonstration). ``workspace_other`` (the kernel's per-game working files) and ``other`` (anything
left once the known bookkeeping is pruned) are scratch, kept in ``memory.json`` but never summed into the headline
(:func:`summary_line`), since a round's scratch churn is large and per-game by nature. :data:`READ_PATTERNS` is a
heuristic over a call's own text, not a parse of prime-agent's tool protocol.

``python -m continual_arc_training.memory line MEMORY.json`` prints a report's one-line summary (paper.sh).
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

# What a snapshot leaves out: runtimes and caches the harness installs for itself (prime-agent copies a whole Python
# environment into its state, not what it has learned: "kernel-venv" to "*.lock") and its own bookkeeping, which changes
# every game by construction: per-session state and logs ("sessions", "session-artifacts"), the daemon's command
# journals ("daemon-workers"), its logs ("logs"), its session leases ("session-leases"), its per-session ledger
# ("rlm-ledger") and the supervisor's lock folder ("supervisor-owners").
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
# the headline reports); the last two are scratch, reported but never summed into it (the module docstring).
AREAS = ("programs", "skills", "harness_store", "notes", "workspace_other", "other")
DELIBERATE_AREAS = ("programs", "skills", "harness_store", "notes")
SCRATCH_AREAS = ("workspace_other", "other")


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
    ``state_dir`` does not exist (a user never started, or a wiped row between rounds). On a 100-user run 325 of
    572,887 files survive the exclusions: the snapshot never goes near the kernel's Python environment."""
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


# A read's patterns, one per deliberate area. prime-agent's own system prompt lists its bundled skills by their
# installed path (".../prime-agent/skills/<name>/SKILL.md", in which "agent/skills/" is a substring), so the lookbehind
# rejects a match preceded by a word character or a hyphen while accepting a bare "agent/skills/" or
# ".prime/agent/skills/"; "programs/" needs a ``<name>.py`` right after it, not just the agent's plan ("I'll write this
# to programs/").
READ_PATTERNS: dict[str, str] = {
    "programs": r"programs/[\w.-]+\.py",
    "skills": r"(?<![\w-])agent/skills/",
    "harness_store": r"(?<![\w-])agent/harness/",
    "notes": r"AGENTS\.md",
}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _call_text(call: dict[str, Any]) -> str:
    """The text a read could show up in, for one captured call: the model's own output for this call (its reply's
    content and reasoning, and any tool call's code) and, if the newest message in the request is a tool result, that
    result's text, never the system prompt or the benchmark's game messages further back. Reading only the call's own
    addition, not the whole conversation every call carries again, keeps this linear in the game's length."""
    parts: list[str] = []
    choices = (call.get("response") or {}).get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    parts.append(_text_of(message.get("content")))
    parts.append(_text_of(message.get("reasoning")))
    for tc in message.get("tool_calls") or []:
        parts.append(json.dumps((tc or {}).get("function") or {}, default=str))
    messages = (call.get("request") or {}).get("messages")
    # a list of chat messages only: the model's own sandbox code can post any JSON body through the proxy,
    # "messages" included as a dict or a string, captured beside the harness's calls
    if isinstance(messages, list) and messages and isinstance(messages[-1], dict) \
            and messages[-1].get("role") == "tool":
        parts.append(_text_of(messages[-1].get("content")))
    return "\n".join(parts)


def read_areas(call: dict[str, Any]) -> set[str]:
    """Which of :data:`READ_PATTERNS`' areas this one call's own text (:func:`_call_text`) references."""
    text = _call_text(call)
    return {area for area, pattern in READ_PATTERNS.items() if re.search(pattern, text)}


def _reads_from_areas(call_areas: list[set[str]]) -> dict[str, int]:
    counts = {area: 0 for area in READ_PATTERNS}
    for areas in call_areas:
        for area in areas:
            counts[area] += 1
    return counts


def count_reads(calls: list[dict[str, Any]]) -> dict[str, int]:
    """One game's reads per area (:data:`READ_PATTERNS`): at most one per area per call (a call that both opens a
    skill and runs a program counts once in each), summed over every call of the game."""
    return _reads_from_areas([read_areas(c) for c in calls])


def game_memory_outcome(doc: dict[str, Any], call_areas: list[set[str]]) -> dict[str, bool]:
    """Two per-game signals, from this game's calls' :func:`read_areas` (``call_areas``, one set per call in order):

    - ``reused_program``: a ``programs/`` read at or before the game's first submission (the turns' ``Submit``
      action): it tried its stored program before answering. A game that never submits counts any ``programs/`` read.
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


def round_report(run_dir: Path, round_: int, users: int, state_dir_of: Callable[[int], Path],
                 round_dir_of: Callable[[Path, int], Path]) -> dict[str, Any]:
    """Build and write ``round_dir_of(run_dir, round_) / "memory.json"``: this round's per-user and total files added,
    changed and removed per area (against the previous round's snapshot of the same user, or nothing for round 0),
    reads per area, and how many games show :func:`game_memory_outcome`'s two signals. Each user's snapshot is kept
    beside it (``memory/uNNN.json``) for the next round's diff, so this only ever needs the previous round's files. A
    user with no captured game this round is counted on its snapshot alone, with no reads and neither signal."""
    this_dir = round_dir_of(run_dir, round_)
    snap_dir = this_dir / "memory"
    snap_dir.mkdir(parents=True, exist_ok=True)
    prev_snap_dir = round_dir_of(run_dir, round_ - 1) / "memory" if round_ > 0 else None

    per_user: dict[str, Any] = {}
    totals_memory = {area: {"added": 0, "changed": 0, "removed": 0} for area in AREAS}
    totals_reads = {area: 0 for area in READ_PATTERNS}
    reused_program_total = solved_without_demo_total = 0

    for u in range(users):
        snapshot = snapshot_state(state_dir_of(u))
        (snap_dir / f"u{u:03d}.json").write_text(json.dumps(snapshot))
        previous: dict[str, dict[str, Any]] = {}
        if prev_snap_dir is not None:
            prev_file = prev_snap_dir / f"u{u:03d}.json"
            if prev_file.is_file():
                previous = json.loads(prev_file.read_text())
        diff = diff_snapshot(previous, snapshot)

        doc_path = this_dir / "captured" / f"u{u:03d}.json.gz"
        reads = {area: 0 for area in READ_PATTERNS}
        outcome = {"reused_program": False, "solved_without_demo": False}
        if doc_path.is_file():
            with gzip.open(doc_path, "rt") as fp:
                doc = json.load(fp)
            call_areas = [read_areas(c) for c in doc.get("calls") or []]
            reads = _reads_from_areas(call_areas)
            outcome = game_memory_outcome(doc, call_areas)

        per_user[str(u)] = {"memory": diff, "reads": reads, **outcome}
        for area in AREAS:
            for k in ("added", "changed", "removed"):
                totals_memory[area][k] += diff[area][k]
        for area in READ_PATTERNS:
            totals_reads[area] += reads[area]
        reused_program_total += int(outcome["reused_program"])
        solved_without_demo_total += int(outcome["solved_without_demo"])

    report = {"round": round_, "users": per_user,
             "total": {"memory": totals_memory, "reads": totals_reads, "reused_program": reused_program_total,
                       "solved_without_demo": solved_without_demo_total}}
    (this_dir / "memory.json").write_text(json.dumps(report, indent=1))
    return report


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

    p = argparse.ArgumentParser(description="a round's memory report as one line")
    sub = p.add_subparsers(dest="cmd", required=True)
    line = sub.add_parser("line")
    line.add_argument("report", help="a round's memory.json")
    a = p.parse_args()
    print(summary_line(json.loads(Path(a.report).read_text())))


if __name__ == "__main__":
    main()
