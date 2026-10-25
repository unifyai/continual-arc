"""Per-round memory-use counters, for every row alike (H, H0, PG, OPSD): how much of a harness's state persists
across rounds, and how often a round's games actually draw on it.

Two independent readings:

- **State on disk** (:func:`snapshot_state`, :func:`diff_snapshot`): a lightweight snapshot (path, size, mtime,
  sha256) of a user's harness state directory (``<run>/users/uNNN/system/state``, prime-agent's
  ``PrimeAgentLearner.state_dir``), grouped into the areas prime-agent is known to write under it (:func:`area_of`)
  -- but only the files a game writes on purpose: the harness's own bookkeeping (session transcripts, daemon-worker
  journals, logs, session leases, its per-session ledger, its own config; see :data:`SNAPSHOT_EXCLUDE`) changes
  every game by construction and is excluded from the snapshot entirely, never counted as a save against any
  area, since counting it would show a frozen row saving something every round -- the opposite of the truth.
  Compared against the previous round's own snapshot of the same user, three counts per area: files added, changed
  (same path, a different hash or, past :data:`SNAPSHOT_MAX_FILE_BYTES`, a different size or mtime) and removed.
  A wiped row's snapshot is taken *after* that round's instance (the wipe itself runs at the *next* call's start,
  ``paper._wipe_stateless_users``), so it still shows that round's own churn (files added and removed); what the
  wipe actually guarantees is that no file survives *unchanged* into the next round -- ``changed`` stays 0 in
  every area for a wiped row, which is the invariant to check, not an empty diff.
- **Reads in play** (:func:`count_reads`): a game's captured calls (the raw doc ``paper.save_instance_hook`` and
  ``loop.save_instance`` both write, before ``train.load_calls`` turns it into training items) scanned for a
  reference to one of those areas -- but only in the model's own output (a call's own reply and tool-call code)
  and in tool results, never in the system prompt or the benchmark's own game messages (:func:`read_areas`).

:func:`round_report` ties both together into ``round_dir(r)/memory.json``: per user and in total, files added,
changed and removed per **deliberate** area (:data:`DELIBERATE_AREAS`: ``programs``, ``skills``, ``harness_store``,
``notes`` -- the areas that carry the "learned to save and fetch" claim), reads per area, and two per-game signals
(:func:`game_memory_outcome`): ``reused_program`` (it read or ran something from ``programs/`` at or before its
first submission) and ``solved_without_demo`` (it solved without ever requesting a demo -- the cleanest form of
reuse). ``workspace_other`` (the kernel's own per-game working files: submissions, scratch grids, one-off scripts)
and ``other`` (anything left over once the known bookkeeping is pruned) are **scratch**, kept in ``memory.json``
for the curious but never summed into the headline (:func:`summary_line`, ``plot_rows.py``): a round's scratch
churn is large and per-game by nature, and would otherwise drown out the few deliberate saves the claim rests on.

**Checked against a real run** (row H's run, round 0 in progress): the real state
directory has neither an ``agent/skills/`` nor an ``agent/harness/`` entry yet (nobody has saved a skill or run
``refine.run(global_=True)``); the only files surviving the exclusions are ``agent/auth.json``, ``agent/models.json``
and ``agent/settings.json`` (the harness's own provider/model config, rewritten -- a fresh proxy port -- on every
process start, hence excluded as bookkeeping, not counted as ``harness_store``) and ``workspace/``'s per-game
scratch (``other`` and ``workspace_other``, confirmed empty of anything under those three area names). This
module's own text search (:data:`READ_PATTERNS`) is a heuristic over a call's own content, not a verified parse
of prime-agent's tool protocol: update it here if a game turns out to refer to these areas some other way.
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

# What a snapshot leaves out: runtimes and caches the harness installs for itself (prime-agent copies a whole
# Python environment into its state, not what it has learned: "kernel-venv".."lock", the same patterns
# continual_arc_baselines.collective.SNAPSHOT_EXCLUDE uses for its own (byte) snapshot of "state_before") and its
# own bookkeeping, which changes every game by construction and would otherwise look like a frozen row saving
# something every round: per-session state and logs ("sessions", "session-artifacts"), the daemon's own command
# journals ("daemon-workers"), its logs ("logs"), its session leases ("session-leases"), its per-session ledger
# ("rlm-ledger", one hash-named file a session), and the supervisor's lock directory ("supervisor-owners") --
# confirmed against a real run's state directory (the module docstring).
SNAPSHOT_EXCLUDE = ("kernel-venv", "node_modules", ".cache", "__pycache__", ".npm", "tmp", "*.pyc", "*.sock", "*.lock",
                   "sessions", "session-artifacts", "daemon-workers", "logs", "session-leases", "supervisor-owners",
                   "rlm-ledger")
# The harness's own provider/model config, excluded by its path (not by a bare name, since an unrelated "*.json"
# elsewhere -- a saved program's own data file, say -- must not be swept up too): it is rewritten, a fresh proxy
# port, on every process start (measured: 100 distinct hashes across 100 users, one per port), so counting it
# would show every row, H0 and frozen H included, "changing its harness store" every round.
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
    ``state_dir`` does not exist (a user never started, or a wiped row between rounds). Measured on a real run (the
    module docstring): 325 of 572,887 files survive the exclusions and are walked and hashed, for all 100 users, in
    0.04 s -- the snapshot never goes near the kernel venv's own bulk."""
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
    """Which of :data:`AREAS` a state-relative path belongs to, on the real layout under ``state_dir`` (the
    module docstring's "checked against a real run"): an AGENTS.md-style note (a file literally named
    ``AGENTS.md``, wherever it sits); the program-library convention (``workspace/programs/``); personal skills
    (``agent/skills/``, also reachable through the ``home/.prime/agent`` link, which a snapshot never follows: it
    is a symlink); the global store ``refine.run(global_=True)`` writes, exactly ``agent/harness/`` and nothing
    else under ``agent/`` (measured: no other ``agent/*`` file exists today but the config :data:`SNAPSHOT_EXCLUDE_
    PATHS` already drops); the rest of the kernel's own working directory (``workspace/`` otherwise -- per-game
    scratch: submissions, one-off scripts, "workspace_other"); or "other" for anything left over once the known
    bookkeeping is pruned (the private HOME outside its ``.prime`` link, say) -- scratch too, never
    ``harness_store``, so an unanticipated leftover cannot silently inflate the headline's deliberate count."""
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


# A read's patterns, one per deliberate area (:data:`DELIBERATE_AREAS`; "workspace_other" and "other" are scratch,
# never something a game runs or opens by name). Anchored against the one false positive measured on a real run:
# prime-agent's own system prompt lists its bundled skills by their installed path
# ("/home/.../prime-agent/skills/<name>/SKILL.md" -- note "agent/skills/" is a literal substring of
# "prime-**agent/skills/**"), which a plain substring search over the whole conversation matched on 3,400 of 3,861
# calls, every one of them in the system prompt, none of them a real read. The lookbehind rejects a match directly
# preceded by a word character or hyphen (as "prime-agent/skills/" is, by "-") while still accepting a bare
# "agent/skills/" or ".prime/agent/skills/" (preceded by "/" or nothing); "programs/" also requires an actual
# ``<name>.py`` immediately after it, not just the agent's own plan text ("I'll write this to programs/").
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
    """Only the text a read could actually show up in, for one captured call (the raw dict shape
    ``paper.save_instance_hook``/``loop.save_instance`` write: a request/response body, not an ask_sd ``Call``):
    the model's own output for this call (its reply's content and reasoning, and any tool call's code) and, if the
    newest message already in the request is a tool result, that result's text -- never the system prompt or the
    benchmark's own game messages further back. Reading only this call's own addition, not the whole re-serialised
    conversation every call re-carries, also keeps this O(1) in the game's length rather than O(calls) per call."""
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
    if call.get("thought"):
        parts.append(str(call["thought"]))
    if call.get("note"):
        parts.append(str(call["note"]))
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
    """Two per-game signals the paper's claim rests on, from this game's calls' precomputed :func:`read_areas`
    (``call_areas``, one set per ``doc["calls"]`` in order -- computed once by the caller and passed to both this
    and :func:`count_reads`, rather than redone here, which is why this takes the sets rather than the calls):

    - ``reused_program``: a ``programs/`` read at or before the game's first submission (``continual_arc.env.
      Submit``, the action type ``doc["turns"]`` itself names) -- evidence it tried its stored program before
      answering. A game that never submits has no "before" to compare against, so any ``programs/`` read at all
      counts. (Not "before the first demo request": a demo request can come after the program already ran, or
      never at all when the program worked -- see ``solved_without_demo`` below -- so gating on it excluded most
      games by construction; a real round found 0 of 100 could ever satisfy it.)
    - ``solved_without_demo``: the game solved and never once requested a demo (``continual_arc.env.
      RequestDemos``) -- the cleanest form of reuse, needing no hint at all."""
    turns = doc.get("turns") or []
    first_submit = next((i for i, t in enumerate(turns) if ((t or {}).get("action") or {}).get("type") == "Submit"),
                        None)
    requested_demo = any(((t or {}).get("action") or {}).get("type") == "RequestDemos" for t in turns)
    reused_program = False
    for call, areas in zip(doc.get("calls") or [], call_areas):
        if "programs" not in areas:
            continue
        turn = call.get("turn")
        if first_submit is None or turn is None or turn <= first_submit:
            reused_program = True
            break
    return {"reused_program": reused_program, "solved_without_demo": bool(doc.get("solved")) and not requested_demo}


def round_report(run_dir: Path, round_: int, users: int, state_dir_of: Callable[[int], Path],
                 round_dir_of: Callable[[Path, int], Path]) -> dict[str, Any]:
    """Build and write ``round_dir_of(run_dir, round_) / "memory.json"``: this round's per-user and total files
    added/changed/removed per area (against the previous round's own snapshot of the same user, or nothing for
    round 0), reads per area, and how many games show :func:`game_memory_outcome`'s two signals. A small snapshot
    of each user's current state is kept alongside it (``round_dir_of(run_dir, round_) / "memory" / "uNNN.json"``),
    read back for the next round's own diff -- so this only ever needs the previous round's file, never the whole
    run's history. A user with no captured doc for this round (the round never reached them: an early kill) is
    counted on its state snapshot alone, with no reads and neither signal."""
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
    """The one-line summary a round's progress log gets: files added/changed/removed over the deliberate areas
    alone (:data:`DELIBERATE_AREAS`: programs, skills, the harness store, notes -- never the per-game scratch under
    workspace_other/other, which would otherwise drown this out), reads over every area (already deliberate-only:
    :data:`READ_PATTERNS` has no scratch entry), and the two per-game signals, total over the round."""
    total = report["total"]
    deliberate = [total["memory"][area] for area in DELIBERATE_AREAS]
    added = sum(a["added"] for a in deliberate)
    changed = sum(a["changed"] for a in deliberate)
    removed = sum(a["removed"] for a in deliberate)
    reads = sum(total["reads"].values())
    return (f"round {report['round']} memory: {added} added, {changed} changed, {removed} removed, {reads} reads, "
           f"{total['reused_program']} games reused a saved program, {total['solved_without_demo']} solved "
           f"without a demo")
