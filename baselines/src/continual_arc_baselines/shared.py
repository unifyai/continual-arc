"""One state for every user of a collective run: the copy of it each user's instance starts from, and the merge of what
the instances wrote back into it as the round ends.

In the collective regime a learner keeps one state, whichever user it serves: a harness's whole store (its memories,
skills, programs, sessions and workspace), a floor's demonstration cache and a program library's programs, as a trained
row's weights are one model for every user (Dan, 5 Nov 2026). ARC holds nothing personal, and a store kept per user set
a harness that saw its own user's tasks alone against weights trained on every user's.
:class:`~continual_arc_baselines.collective.CollectiveHarnessLearner` keeps the one state where a run of one user keeps
its user's, in user 0's folder: ``users/u000/system/state``, committed at the end of every round to ``state.git`` and
listed in ``state-log.jsonl`` (:mod:`continual_arc_baselines.statelog`), so that a run of one user is what it always
was. With more users a round goes:

* **As it begins** (:meth:`SharedState.begin_round`), the one state is read file by file as the round's base, every
  file's content kept in the object store of ``state.git``, and every other user's ``system/state`` becomes a copy of
  it, every absolute path in it that names user 0's ``system`` folder renamed to the user's own, in text files, links
  and SQLite databases, as a frozen test's fork renames it (:func:`continual_arc_baselines.frozen.relocate`), since a
  harness names its own folder in what it writes. User 0's harness works in the one state itself. Every user's instance
  thus starts from the same state, the one state as the round began.
* **As it plays**, each user's harness writes its own folder alone, so nothing one user writes in a round reaches
  another user's instance of it: the state changes only between rounds (``continual_arc.learner``).
* **As it ends** (:meth:`SharedState.end_round`), with every harness stopped, what each user wrote is read against the
  folder they started the round from, its paths named as user 0's again, and merged into the one state in user order:
  user 0's writes are in place, user 1's are merged into them, then user 2's, and so on. The merged state is the round's
  commit, and every user starts the next round from it.

A round is open from its beginning to its commit, and :data:`JOURNAL` holds what it began from. The runner writes a
round's records before the learner's ``end_round``, so a process stopped in an open round leaves one of two rounds,
which the next round's beginning settles (:meth:`SharedState.recover`): a round stopped while it played, which the
runner plays again, is undone, the one state put back as the round began; a round stopped once the runner had recorded
it is merged then, from what every user wrote in it, still on disk, and committed before the next round begins.

The merge goes path by path against the round's base, each user's writes in turn:

1. A path that one user alone changed (added, rewrote or removed) takes that change, so whatever users write under
   names of their own, a session's transcript, a task's program, a skill, is all kept.
2. A path that several users changed to the same content keeps it.
3. A path that several users changed differently keeps what each wrote wherever its kind can hold both:

   * a **JSON** file, one whose versions all parse as an object or an array (a ``.jsonl`` or ``.ndjson`` log is text),
     is merged as JSON (:func:`merge_json`): a key that one side added, changed or removed takes that change, an object
     both changed is merged key by key, and an array keeps the earlier user's items, less those the later user removed,
     then the items the later user added; any other value both changed takes the later user's;
   * a **SQLite database** is merged row by row wherever the merge can read the whole of it as the database itself
     keeps it (:func:`merge_database`): every table keyed by its primary key, or by its rows' whole content where it has
     none; every virtual table a full-text index that the database's own triggers write as its rows change; and no
     trigger that changes a table's rows. A row takes the change of whichever user changed it, the later user's where
     both did, and the rows the later user added to a table whose integer ids the database gives (an
     ``INTEGER PRIMARY KEY``) are numbered after the merged database's own when their ids clash, the foreign keys the
     database declares to them following (a reference it does not declare is not followed), so that both users'
     sessions and their messages are kept. Any other database (such as openclaw's agent database, whose own code keeps
     its full-text indexes and whose triggers rewrite its rows), and one whose schemas differ or whose merge would
     break one of its constraints, takes the later user's whole;
   * any other **text** file, one with no NUL byte, is merged line by line as ``git merge-file`` merges it, and where
     both changed the same lines both versions are kept, the earlier user's first: an append-only log keeps every
     user's lines;
   * anything else, a binary file, a link, a file against a link, or a file of a git repository's own (under a ``.git``
     folder, which no line-by-line merge keeps whole), takes the later user's.

4. A removal yields to a change: a path one user removed and another changed keeps the change.
5. Nothing is written through a link or in a folder's place: where the merged state holds a link or a file on the way
   to a path, or a folder at it, it keeps what it holds, and the later user's entry is left out.

:data:`MERGES` beside the state log has a line for every round's merge: each user's paths added, changed and removed
against the round's base, and every path that several users changed differently, with the users and the rule that
settled it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continual_arc_baselines.frozen import (
    FORKS,
    SQLITE,
    delete,
    naming,
    put_back,
    relocate,
    relocate_database,
)
from continual_arc_baselines.statelog import GIT_ENV, ensure_repo

MERGES = "state-merges.jsonl"
"""Every round's merge, a line each, in user 0's system folder beside the state log."""
JOURNAL = "state-round.json.gz"
"""The open round, in user 0's system folder: its base and every user's start as it began, then, once the runner has
recorded it, its index and user 0's own end of it. It goes once the round is committed."""
MERGED = "state-merged.json.gz"
"""The one state file by file as the last merge left it, beside the journal: what the next round's base is read
against, so that only a file changed since is read again."""
LINE_LOGS = (".jsonl", ".ndjson")
"""Files of one JSON record a line, merged as text however their lines parse."""
LATER = "the later user's"
MISSING = object()   # a key or a version that is not there

Listing = dict[str, list]
"""A folder's files and links by path: ``["file", size, mtime_ns, executable]`` or ``["link", target]``."""
Manifest = dict[str, list]
"""A folder's files and links by path, each file with its object: ``["file", size, mtime_ns, executable, oid]`` or
``["link", target]``."""


@dataclass(frozen=True)
class Entry:
    """What a path holds: a file's bytes and whether it is executable, or a link's target."""

    data: bytes
    link: bool = False
    executable: bool = False

    @property
    def mode(self) -> str:
        """The entry's mode as git records it."""
        return "120000" if self.link else "100755" if self.executable else "100644"


def read_entry(path: Path) -> Entry | None:
    """The file or link at ``path``; None for nothing, a folder, or anything else (a stopped process's socket)."""
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if stat.S_ISLNK(info.st_mode):
        return Entry(os.fsencode(os.readlink(path)), link=True)
    if stat.S_ISREG(info.st_mode):
        return Entry(Path(path).read_bytes(), executable=bool(info.st_mode & stat.S_IXUSR))
    return None


def listing(tree: Path) -> Listing:
    """Every file and link under ``tree`` by its path relative to it, with what shows that it changed: a file's size,
    modification time and executable bit, a link's target. A link to a folder is a link and is never followed; anything
    that is neither a file nor a link is no part of a state."""
    out: Listing = {}
    for root, dirs, files in os.walk(tree):
        for name in (*dirs, *files):
            path = os.path.join(root, name)
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                out[os.path.relpath(path, tree)] = ["link", os.readlink(path)]
            elif stat.S_ISREG(info.st_mode):
                out[os.path.relpath(path, tree)] = ["file", info.st_size, info.st_mtime_ns,
                                                     bool(info.st_mode & stat.S_IXUSR)]
    return out


def clear_way(tree: Path, rel: str) -> Path | None:
    """``tree / rel`` where no folder on the way to it under ``tree`` is a link or a file, so that nothing read or
    written there leads out of the state; None where one is."""
    where = tree
    for part in rel.split("/")[:-1]:
        where = where / part
        if where.is_symlink() or (os.path.lexists(where) and not where.is_dir()):
            return None
    return tree / rel


def _writable(folder: Path) -> None:
    """``folder`` made writable to its owner where it is not: a folder an agent made read-only (a Go module cache's)
    refuses to gain or lose an entry until it is."""
    if not os.access(folder, os.W_OK):
        folder.chmod(folder.stat().st_mode | stat.S_IRWXU)


def _unlink(path: Path) -> None:
    """``path``, a file or a link, removed, whatever its folder's mode."""
    _writable(path.parent)
    path.unlink()


def write_entry(path: Path, entry: Entry) -> None:
    """``entry`` at ``path``, whose folder is made if it is not there, in place of whatever file or link is there."""
    if os.path.lexists(path):
        _unlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _writable(path.parent)
    if entry.link:
        path.symlink_to(os.fsdecode(entry.data))
    else:
        path.write_bytes(entry.data)
        mode = path.stat().st_mode
        path.chmod(mode | 0o111 if entry.executable else mode & ~0o111)


def place(tree: Path, rel: str, entry: Entry | None) -> bool:
    """``entry`` written at ``rel`` under ``tree``, or whatever is there removed for None; False, with nothing changed,
    where a link or a file stands on the way to it or a folder holds it (rule 5 of the module's docstring)."""
    path = clear_way(tree, rel)
    if path is None or (path.is_dir() and not path.is_symlink()):
        return False
    if entry is None:
        if os.path.lexists(path):
            _unlink(path)
    else:
        write_entry(path, entry)
    return True


def settle_databases(tree: Path) -> None:
    """Every SQLite database under ``tree`` made one file, its write-ahead log taken into it and its side files removed,
    so that a database a harness stopped with is read whole wherever its log stood. Nothing may hold one open."""
    for root, _, names in os.walk(tree):
        for name in names:
            path = Path(root) / name
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as fp:
                if fp.read(len(SQLITE)) != SQLITE:
                    continue
            con = sqlite3.connect(path)
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                con.close()
            for side in ("-wal", "-shm"):
                Path(f"{path}{side}").unlink(missing_ok=True)


# -- the one state's objects ----------------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", f"--git-dir={repo}", *args], env={**os.environ, **GIT_ENV}, check=True,
                          capture_output=True).stdout


def blob_id(entry: Entry, algorithm: str) -> str:
    """The id git gives ``entry``'s content, by the repository's ``algorithm`` (sha1 or sha256)."""
    digest = hashlib.new(algorithm)
    digest.update(b"blob %d\0" % len(entry.data))
    digest.update(entry.data)
    return digest.hexdigest()


def store_blob(repo: Path, data: bytes, algorithm: str) -> str:
    """``data`` kept in ``repo``'s object store as git keeps a file's content, a loose object, whatever path it had: a
    path git keeps out of a tree (one inside a git repository of the state's own) has its content kept too."""
    oid = blob_id(Entry(data), algorithm)
    path = repo / "objects" / oid[:2] / oid[2:]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, made = tempfile.mkstemp(dir=path.parent, prefix=".blob-")
        packer = zlib.compressobj(1)
        with os.fdopen(fd, "wb") as fp:
            fp.write(packer.compress(b"blob %d\0" % len(data)))
            fp.write(packer.compress(data))
            fp.write(packer.flush())
        os.chmod(made, 0o444)
        os.replace(made, path)
    return oid


def load_blob(repo: Path, oid: str) -> bytes:
    """The content of the object ``oid`` in ``repo``: a loose object read directly, a packed one through git."""
    path = repo / "objects" / oid[:2] / oid[2:]
    if path.exists():
        body = zlib.decompress(path.read_bytes())
        return body[body.index(b"\0") + 1:]
    return _git(repo, "cat-file", "blob", oid)


def _holds(entry: list) -> tuple:
    """What a manifest's entry holds, whenever it was written: a link's target, or a file's mode and object."""
    return tuple(entry) if entry[0] == "link" else ("file", entry[3], entry[4])


# -- the merge of one path ------------------------------------------------------------------------------------------


def merge_json(base: Any, ours: Any, theirs: Any) -> Any:
    """``ours`` and ``theirs``, two changes of ``base`` (:data:`MISSING` where there was none), merged as JSON: a key one
    side added, changed or removed takes that change, an object both changed is merged key by key, an array keeps ours'
    items less those theirs removed and then the items theirs added, and any other value both changed takes theirs."""
    if ours == theirs or theirs == base:
        return ours
    if ours == base:
        return theirs
    if isinstance(ours, dict) and isinstance(theirs, dict):
        was = base if isinstance(base, dict) else {}
        merged = {}
        for key in [*ours, *(k for k in theirs if k not in ours)]:
            before, mine, other = was.get(key, MISSING), ours.get(key, MISSING), theirs.get(key, MISSING)
            if (mine is MISSING and other == before) or (other is MISSING and mine == before):
                continue   # one side removed it and the other left it as it was
            if mine is MISSING or other is MISSING:
                merged[key] = other if mine is MISSING else mine   # added, or kept by a change over a removal
            else:
                merged[key] = merge_json(before, mine, other)
        return merged
    if isinstance(ours, list) and isinstance(theirs, list):
        was = base if isinstance(base, list) else []
        return [x for x in ours if x in theirs or x not in was] + [x for x in theirs if x not in was and x not in ours]
    return theirs


def _json(entry: Entry | None) -> Any:
    """``entry`` as an object or an array of JSON, :data:`MISSING` for anything else."""
    if entry is None or entry.link:
        return MISSING
    try:
        value = json.loads(entry.data)
    except ValueError:
        return MISSING
    return value if isinstance(value, (dict, list)) else MISSING


def dump_json(value: Any, like: bytes) -> bytes:
    """``value`` as JSON laid out as ``like`` is: indented when it spans lines, ending with a newline when it does."""
    text = json.dumps(value, ensure_ascii=False, indent=2 if b"\n" in like.strip() else None)
    return (text + ("\n" if like.endswith(b"\n") else "")).encode()


def merge_text(base: bytes, ours: bytes, theirs: bytes, scratch: Path) -> bytes:
    """``ours`` and ``theirs``, two changes of ``base``, merged line by line as ``git merge-file --union`` merges them:
    where both changed the same lines, ours and then theirs."""
    folder = Path(tempfile.mkdtemp(dir=scratch))
    for name, data in (("ours", ours), ("base", base), ("theirs", theirs)):
        (folder / name).write_bytes(data)
    done = subprocess.run(["git", "merge-file", "-p", "--union", str(folder / "ours"), str(folder / "base"),
                           str(folder / "theirs")], env={**os.environ, **GIT_ENV}, capture_output=True)
    shutil.rmtree(folder)
    if done.returncode >= 128:   # 255: git could not read or merge them
        raise subprocess.CalledProcessError(done.returncode, done.args, done.stdout, done.stderr)
    return done.stdout


def merge_entries(rel: str, base: Entry | None, ours: Entry, theirs: Entry, scratch: Path) -> tuple[Entry, str]:
    """The path ``rel`` that two sides changed differently, ``ours`` (the users before) and ``theirs`` (the later user),
    from ``base`` (None where the round began without it), merged by its kind (rule 3 of the module's docstring); and the
    rule that settled it."""
    if ours.link or theirs.link or ".git" in rel.split("/"):
        return theirs, LATER
    before = None if base is None or base.link else base.data
    if ours.data.startswith(SQLITE) and theirs.data.startswith(SQLITE):
        try:
            merged = merge_database(before if before and before.startswith(SQLITE) else None, ours.data, theirs.data,
                                    scratch)
        except Unmergeable as why:
            return theirs, f"{LATER}: {why}"
        return Entry(merged, executable=theirs.executable), "merged row by row"
    mine, other = (MISSING, MISSING) if rel.endswith(LINE_LOGS) else (_json(ours), _json(theirs))
    if mine is not MISSING and other is not MISSING:
        merged = merge_json(_json(base), mine, other)
        return Entry(dump_json(merged, theirs.data), executable=theirs.executable), "merged as JSON"
    if all(b"\0" not in data for data in (before or b"", ours.data, theirs.data)):
        return Entry(merge_text(before or b"", ours.data, theirs.data, scratch), executable=theirs.executable), (
            "merged line by line")
    return theirs, LATER


# -- the merge of a database, row by row ----------------------------------------------------------------------------


class Unmergeable(Exception):
    """Two versions of a database whose rows the merge cannot merge: it cannot read the database whole as the database
    keeps itself, their schemas differ, or the merge breaks a constraint."""


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class _Table:
    name: str
    columns: tuple[str, ...]
    key: tuple[str, ...]   # the primary key's columns; none for a table without one, whose rows are their content
    allocated: bool        # its key an INTEGER PRIMARY KEY, the id the database gives each new row
    foreign: tuple[tuple[str, str, str | None], ...]   # (column, parent table, parent column): one-column foreign keys


def _schema(con: sqlite3.Connection, db: str) -> list[tuple]:
    return con.execute(f"SELECT type, name, tbl_name, sql FROM {db}.sqlite_master WHERE name NOT LIKE 'sqlite\\_%' "
                       "ESCAPE '\\' ORDER BY type, name").fetchall()


WRITES = re.compile(r"\b(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE(?:\s+OR\s+\w+)?|DELETE\s+FROM)\s+"
                    r"(?:\w+\s*\.\s*)?[\"`\[]?([^\s\"`\](,;]+)", re.I)
"""A statement that changes a table's rows, and the table it changes, in a trigger's body."""


def _unreadable(con: sqlite3.Connection) -> str | None:
    """Why the merge cannot read the merged database's rows whole as the database itself keeps them (the module's
    docstring), or None: a virtual table that is no full-text index, a full-text index none of the database's triggers
    writes, which its application keeps itself, or a trigger that changes a table's rows."""
    found = _schema(con, "main")
    virtual = {name: sql or "" for kind, name, _, sql in found
               if kind == "table" and (sql or "").lstrip().upper().startswith("CREATE VIRTUAL TABLE")}
    bodies = [(name, sql[match.end():]) for kind, name, _, sql in found
              if kind == "trigger" and (match := re.search(r"\bBEGIN\b", sql or "", re.I))]
    written = {target for _, body in bodies for target in WRITES.findall(body)}
    for name, sql in virtual.items():
        if not re.search(r"\bUSING\s+fts[345]\b", sql, re.I):
            return f"its virtual table {name} is no full-text index"
        if name not in written:
            return f"no trigger of its own keeps its full-text index {name}"
    for name, body in bodies:
        for target in WRITES.findall(body):
            if target not in virtual:
                return f"its trigger {name} changes the rows of {target}"
    return None


def _tables(con: sqlite3.Connection) -> list[_Table]:
    """The tables whose rows are merged: every table of the merged database but SQLite's own and a full-text index's
    (a virtual table and its shadow tables), which the database's triggers write as the rows change."""
    found = con.execute("SELECT name, sql FROM main.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' "
                        "ESCAPE '\\' ORDER BY name").fetchall()
    virtual = [name for name, sql in found if (sql or "").lstrip().upper().startswith("CREATE VIRTUAL TABLE")]
    tables = []
    for name, sql in found:
        if name in virtual or any(name.startswith(v + "_") for v in virtual):
            continue
        info = con.execute(f"PRAGMA main.table_info({_quote(name)})").fetchall()   # cid, name, type, notnull, dflt, pk
        key = tuple(row[1] for row in sorted((row for row in info if row[5]), key=lambda row: row[5]))
        allocated = (len(key) == 1 and not re.search(r"\bWITHOUT\s+ROWID\b", sql or "", re.I)
                     and next(row[2] for row in info if row[1] == key[0]).upper() == "INTEGER")
        groups: dict[int, list] = {}
        for fk in con.execute(f"PRAGMA main.foreign_key_list({_quote(name)})").fetchall():   # id, seq, table, from, to
            groups.setdefault(fk[0], []).append(fk)
        tables.append(_Table(name, tuple(row[1] for row in info), key, allocated,
                             tuple((fks[0][3], fks[0][2], fks[0][4]) for fks in groups.values() if len(fks) == 1)))
    return tables


def _numbering(con: sqlite3.Connection, table: _Table, in_base: bool) -> dict[int, int]:
    """The ids theirs' new rows of a table whose ids the database gives take: their own while none clashes with a
    different row the merged database has, else every one in turn after the largest id it has or has given. A new row
    the merged database already holds as it is keeps its id."""
    t, k, cols = _quote(table.name), _quote(table.key[0]), ", ".join(_quote(c) for c in table.columns)
    at = table.columns.index(table.key[0])
    added = con.execute(f"SELECT {cols} FROM t.{t} WHERE {k} NOT IN (SELECT {k} FROM b.{t}) ORDER BY {k}" if in_base
                        else f"SELECT {cols} FROM t.{t} ORDER BY {k}").fetchall()
    held = {row[at]: con.execute(f"SELECT {cols} FROM main.{t} WHERE {k} = ?", (row[at],)).fetchone() for row in added}
    new = [row for row in added if held[row[at]] != row]
    if not any(held[row[at]] is not None for row in new):
        return {}
    top = con.execute(f"SELECT max({k}) FROM main.{t}").fetchone()[0] or 0
    if con.execute("SELECT 1 FROM main.sqlite_master WHERE name = 'sqlite_sequence'").fetchone():
        given = con.execute("SELECT seq FROM main.sqlite_sequence WHERE name = ?", (table.name,)).fetchone()
        top = max(top, given[0] if given else 0)
    return {row[at]: top + n for n, row in enumerate(new, 1)}


def _merge_table(con: sqlite3.Connection, table: _Table, in_base: bool, numbered: dict[str, dict[int, int]],
                 keys: dict[str, tuple[str, ...]]) -> None:
    """Theirs' changes of one table (``t``) against the base (``b``) applied to the merged database (``main``)."""
    t, cols = _quote(table.name), ", ".join(_quote(c) for c in table.columns)
    marks = ", ".join("?" * len(table.columns))
    order = ", ".join(str(i + 1) for i in range(len(table.columns)))
    position = {column: i for i, column in enumerate(table.columns)}
    changed = con.execute(f"SELECT {cols} FROM t.{t} EXCEPT SELECT {cols} FROM b.{t} ORDER BY {order}" if in_base
                          else f"SELECT DISTINCT {cols} FROM t.{t} ORDER BY {order}").fetchall()

    def followed(row: tuple) -> list:
        """``row`` with every foreign key to a row of theirs that took a new id naming it by that id."""
        out = list(row)
        for column, parent, to in table.foreign:
            ids = numbered.get(parent)
            if ids and to in (None, *keys.get(parent, ())[:1]) and out[position[column]] in ids:
                out[position[column]] = ids[out[position[column]]]
        return out

    if not table.key:   # a row is its content
        same = " AND ".join(f"{_quote(c)} IS ?" for c in table.columns)
        if in_base:
            for row in con.execute(f"SELECT {cols} FROM b.{t} EXCEPT SELECT {cols} FROM t.{t} ORDER BY {order}"):
                con.execute(f"DELETE FROM main.{t} WHERE rowid = (SELECT rowid FROM main.{t} WHERE {same} LIMIT 1)", row)
        for row in changed:
            row = followed(row)
            if con.execute(f"SELECT 1 FROM main.{t} WHERE {same} LIMIT 1", row).fetchone() is None:
                con.execute(f"INSERT INTO main.{t} ({cols}) VALUES ({marks})", row)
        return
    at = [position[k] for k in table.key]
    same = " AND ".join(f"{_quote(k)} IS ?" for k in table.key)
    if in_base:
        kept = ", ".join(_quote(k) for k in table.key)
        for key in con.execute(f"SELECT {kept} FROM b.{t} EXCEPT SELECT {kept} FROM t.{t} ORDER BY "
                               + ", ".join(str(i + 1) for i in range(len(table.key)))).fetchall():
            before = con.execute(f"SELECT {cols} FROM b.{t} WHERE {same}", key).fetchone()
            if con.execute(f"SELECT {cols} FROM main.{t} WHERE {same}", key).fetchone() == before:
                con.execute(f"DELETE FROM main.{t} WHERE {same}", key)   # unchanged before them: their removal holds
    ids = numbered.get(table.name, {})
    for row in changed:
        own = row[at[0]]
        row = followed(row)
        if own in ids:   # a row theirs added, numbered after the merged database's own
            row[at[0]] = ids[own]
            con.execute(f"INSERT INTO main.{t} ({cols}) VALUES ({marks})", row)
            continue
        key = [row[i] for i in at]
        now = con.execute(f"SELECT {cols} FROM main.{t} WHERE {same}", key).fetchone()
        if now is None:
            con.execute(f"INSERT INTO main.{t} ({cols}) VALUES ({marks})", row)
        elif list(now) != row:
            sets = ", ".join(f"{_quote(c)} = ?" for c in table.columns)
            con.execute(f"UPDATE main.{t} SET {sets} WHERE {same}", [*row, *key])


def merge_database(base: bytes | None, ours: bytes, theirs: bytes, scratch: Path) -> bytes:
    """Two changes of the SQLite database ``base`` (None where the round began without it), ``ours`` and ``theirs``,
    merged row by row (the module's docstring) into a copy of ours, the database's own triggers keeping its full-text
    indexes as the rows change; raises :class:`Unmergeable` where the merge cannot read the database whole
    (:func:`_unreadable`), where their schemas differ, where a row breaks one of the database's constraints, or where the
    merged database fails SQLite's integrity check."""
    folder = Path(tempfile.mkdtemp(dir=scratch))
    merged, other, before = folder / "merged", folder / "theirs", folder / "base"
    merged.write_bytes(ours)
    other.write_bytes(theirs)
    if base is not None:
        before.write_bytes(base)
    con = sqlite3.connect(merged, isolation_level=None)
    try:
        con.execute("ATTACH DATABASE ? AS t", (str(other),))
        con.execute("ATTACH DATABASE ? AS b", (str(before) if base is not None else ":memory:",))
        if _schema(con, "main") != _schema(con, "t"):
            raise Unmergeable("their schemas differ")
        why = _unreadable(con)
        if why is not None:
            raise Unmergeable(why)
        tables = _tables(con)
        in_base = {table.name for table in tables
                   if [row[1] for row in con.execute(f"PRAGMA b.table_info({_quote(table.name)})")] == list(table.columns)}
        numbered = {table.name: _numbering(con, table, table.name in in_base) for table in tables if table.allocated}
        keys = {table.name: table.key for table in tables}
        con.execute("BEGIN")
        for table in tables:
            _merge_table(con, table, table.name in in_base, numbered, keys)
        con.execute("COMMIT")
        if con.execute("PRAGMA main.integrity_check").fetchone()[0] != "ok":
            raise Unmergeable("the merged database fails SQLite's integrity check")
        con.execute("PRAGMA main.wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as error:
        raise Unmergeable(f"{type(error).__name__}: {error}") from error
    finally:
        con.close()
    data = merged.read_bytes()
    shutil.rmtree(folder)
    return data


# -- the one state ---------------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else None


def _write_json(path: Path, value: Any) -> None:
    made = path.with_name(path.name + ".partial")
    made.write_bytes(gzip.compress(json.dumps(value).encode(), 1))
    made.replace(path)


class SharedState:
    """The one state of a collective run, kept in user 0's ``system/state`` (the module's docstring). ``system(user)`` is
    a user's system folder, whose path the user's harness names in what it writes, and ``users`` how many users the run
    serves. No harness may be running while a round begins or ends."""

    def __init__(self, system: Callable[[int], Path], users: int) -> None:
        self.system, self.users = system, users
        self.repo = system(0).absolute() / "state.git"
        self.journal_path, self.merged_path = system(0) / JOURNAL, system(0) / MERGED
        self.algorithm = "sha1"
        self.base: Manifest = {}
        self.starts: dict[int, Listing] = {}
        self.round: int | None = None
        self.own: Manifest | None = None

    def state(self, user: int) -> Path:
        return self.system(user) / "state"

    # -- a round -----------------------------------------------------------------------------------------------------

    def recover(self, recorded: int) -> int | None:
        """Settle the round a stopped process left open (the module's docstring), the runner's records holding
        ``recorded`` whole rounds: one stopped while it played is undone, the one state put back as the round began; one
        the runner had recorded, whether or not the learner had marked it so (:meth:`ending`), is merged now, user 0's
        own writes put back as they ended it first, and its index returned, for the caller to commit before
        :meth:`close_round`, unless its commit was made already. None when nothing is left to commit."""
        journal = _read_json(self.journal_path)
        if journal is None:
            return None
        self._open()
        self.base, self.round, self.own = journal["base"], journal["round"], journal["own"]
        self.starts = {int(user): start for user, start in journal["starts"].items()}
        if self.round is None and recorded > self.merged_rounds():   # recorded in the moment before it was marked
            self.round = recorded - 1
        if self.round is None:
            self.restore(self.state(0), self.base)
        elif not self.committed(self.round):
            if self.own is not None:
                self.restore(self.state(0), self.own)
            self._merge(self.round)
            return self.round
        self.close_round()
        return None

    def begin_round(self) -> None:
        """The one state read as the round's base and every other user's state a copy of it, its paths renamed to the
        user's own; every user's start taken, and the round opened (:data:`JOURNAL`)."""
        self._open()
        one, old = self.state(0), str(self.system(0))
        one.mkdir(parents=True, exist_ok=True)
        settle_databases(one)   # a write-ahead log a stopped harness left, in the base and every copy alike
        self.base = self.manifest(one, _read_json(self.merged_path) or {})
        found = naming(one, old)

        def copy(user: int) -> None:
            self.system(user).mkdir(parents=True, exist_ok=True)
            put_back(one, self.state(user))
            relocate(self.state(user), old, str(self.system(user)), found)

        with ThreadPoolExecutor(FORKS) as pool:
            list(pool.map(copy, range(1, self.users)))
        self._start()

    def rebase(self) -> None:
        """The round begun again from what every user's state holds now, after a reset has wiped them all."""
        self.base = self.manifest(self.state(0), self.base)
        self._start()

    def ending(self, round_index: int) -> None:
        """The runner has recorded round ``round_index``: a process stopped from now on leaves it to be merged, not
        played again."""
        self.round = round_index
        self._journal()

    def end_round(self, round_index: int) -> dict[str, Any]:
        """What every user wrote in round ``round_index`` merged into the one state, user by user (the module's
        docstring), user 0's own end of it kept first; returns the round's line of :data:`MERGES`, which it appends."""
        self.round = round_index
        settle_databases(self.state(0))
        self.own = self.manifest(self.state(0), self.base)
        self._journal()
        return self._merge(round_index)

    def close_round(self) -> None:
        """The round committed: nothing of it is left to settle."""
        self.journal_path.unlink(missing_ok=True)
        self.round, self.own = None, None

    def committed(self, round_index: int) -> bool:
        """Whether the one state's log holds a commit of round ``round_index`` (a state log numbers its rounds from 0)."""
        log = self.system(0) / "state-log.jsonl"
        return log.is_file() and len(log.read_text().splitlines()) > round_index

    def merged_rounds(self) -> int:
        """How many rounds have been merged: one more than the last :data:`MERGES` records, none before the first."""
        path = self.system(0) / MERGES
        return 1 + max((json.loads(line)["round"] for line in path.read_text().splitlines()), default=-1) if (
            path.is_file()) else 0

    # -- the round's files -------------------------------------------------------------------------------------------

    def _open(self) -> None:
        ensure_repo(self.repo)
        self.algorithm = _git(self.repo, "rev-parse", "--show-object-format").decode().strip()

    def _start(self) -> None:
        with ThreadPoolExecutor(FORKS) as pool:
            self.starts = dict(zip(range(1, self.users), pool.map(lambda u: listing(self.state(u)),
                                                                   range(1, self.users)), strict=True))
        self.round, self.own = None, None
        self._journal()

    def _journal(self) -> None:
        _write_json(self.journal_path, {"base": self.base, "starts": {str(u): s for u, s in self.starts.items()},
                                        "round": self.round, "own": self.own})

    def manifest(self, tree: Path, known: Manifest) -> Manifest:
        """Every file and link under ``tree`` with each file's object, its content kept in the one state's repository: a
        file whose size, modification time and executable bit are as ``known`` has them is taken to hold what it held
        then, and any other is read."""
        out: Manifest = {}
        for rel, seen in listing(tree).items():
            before = known.get(rel)
            if seen[0] == "link":
                out[rel] = seen
            elif before is not None and before[:4] == seen:
                out[rel] = before
            else:
                out[rel] = [*seen, store_blob(self.repo, (tree / rel).read_bytes(), self.algorithm)]
        return out

    def restore(self, tree: Path, manifest: Manifest) -> None:
        """``tree`` made to hold what ``manifest`` lists and nothing else: what a stopped round wrote there undone."""
        now = listing(tree)
        for rel in sorted(set(now) - set(manifest), reverse=True):
            _unlink(tree / rel)
        for rel, entry in sorted(manifest.items()):
            if now.get(rel) == entry[:4]:
                continue
            path = tree / rel
            if path.is_dir() and not path.is_symlink():
                delete(path)
            elif os.path.lexists(path):
                _unlink(path)
            write_entry(path, Entry(os.fsencode(entry[1]), link=True) if entry[0] == "link"
                        else Entry(load_blob(self.repo, entry[4]), executable=entry[3]))

    def _merge(self, round_index: int) -> dict[str, Any]:
        one, written, conflicts = self.state(0), [], []
        if self.own is None:   # stopped before user 0's own end of the round was kept: its folder holds it still
            settle_databases(one)
            self.own = self.manifest(one, self.base)
        mine = self.summary_of(self.own)
        writers: dict[str, list[int]] = {rel: [0] for paths in mine.values() for rel in paths}
        with ThreadPoolExecutor(FORKS) as pool:
            list(pool.map(lambda user: settle_databases(self.state(user)), range(1, self.users)))
        with tempfile.TemporaryDirectory(dir=self.system(0)) as scratch:
            written.append(mine)
            for user in range(1, self.users):
                writes = self.writes(user, Path(scratch))
                written.append(self.summary(writes))
                for rel, theirs in sorted(writes.items()):
                    earlier = writers.setdefault(rel, [])
                    way = clear_way(one, rel)
                    ours = read_entry(way) if way is not None else None
                    merged, rule = self.merge(rel, ours, theirs, Path(scratch))
                    if merged != ours and not place(one, rel, merged):
                        rule = "left out: a link or a file stands on its way, or a folder holds it"
                    if rule is not None:
                        conflicts.append({"path": rel, "users": [*earlier, user], "rule": rule})
                    earlier.append(user)
        record = {"round": round_index, "users": [{"user": user, **paths} for user, paths in enumerate(written)],
                  "conflicts": conflicts}
        with (self.system(0) / MERGES).open("a") as fp:
            fp.write(json.dumps(record) + "\n")
        _write_json(self.merged_path, self.manifest(one, self.own))
        return record

    # -- one user's writes ---------------------------------------------------------------------------------------

    def same(self, entry: Entry | None, rel: str) -> bool:
        """Whether ``entry`` is what the round's base holds at ``rel``."""
        base = self.base.get(rel)
        if entry is None or base is None:
            return entry is None and base is None
        if base[0] == "link":
            return entry.link and os.fsdecode(entry.data) == base[1]
        return not entry.link and entry.executable == base[3] and blob_id(entry, self.algorithm) == base[4]

    def base_entry(self, rel: str) -> Entry | None:
        base = self.base.get(rel)
        if base is None:
            return None
        if base[0] == "link":
            return Entry(os.fsencode(base[1]), link=True)
        return Entry(load_blob(self.repo, base[4]), executable=base[3])

    def canonical(self, entry: Entry | None, user: int, scratch: Path) -> Entry | None:
        """``entry``, read from ``user``'s state, with every absolute path that names the user's system folder naming
        user 0's, as the one state names it."""
        if entry is None or user == 0:
            return entry
        old, new = str(self.system(user)), str(self.system(0))
        if entry.link:
            target = os.fsdecode(entry.data)
            if target == old or target.startswith(old + "/"):
                return Entry(os.fsencode(new + target[len(old):]), link=True)
            return entry
        if entry.data.startswith(SQLITE):
            copy = Path(tempfile.mkdtemp(dir=scratch)) / "database"
            copy.write_bytes(entry.data)
            relocate_database(copy, old, new)
            data = copy.read_bytes()
            shutil.rmtree(copy.parent)
            return Entry(data, executable=entry.executable)
        if b"\0" in entry.data:
            return entry
        named = new.encode()
        return Entry(re.sub(re.escape(old.encode()) + rb"(?![\w.-])", lambda _: named, entry.data),
                     executable=entry.executable)

    def writes(self, user: int, scratch: Path) -> dict[str, Entry | None]:
        """What ``user`` (one of the users after user 0) wrote in the round: every path of their state that changed
        since they started the round and now differs from the base, holding what it holds as user 0's folder names it,
        and None for every path of the base they removed."""
        tree, start, now = self.state(user), self.starts.get(user, {}), listing(self.state(user))
        out: dict[str, Entry | None] = {}
        for rel, seen in now.items():
            if start.get(rel) != seen:
                entry = self.canonical(read_entry(tree / rel), user, scratch)
                if entry is not None and not self.same(entry, rel):
                    out[rel] = entry
        for rel in start:
            if rel not in now and rel in self.base:
                out[rel] = None
        return out

    def summary(self, writes: dict[str, Entry | None]) -> dict[str, list[str]]:
        """A user's writes as the record of the round lists them: the paths added, changed and removed."""
        return {"added": sorted(rel for rel, e in writes.items() if e is not None and rel not in self.base),
                "changed": sorted(rel for rel, e in writes.items() if e is not None and rel in self.base),
                "removed": sorted(rel for rel, e in writes.items() if e is None)}

    def summary_of(self, own: Manifest) -> dict[str, list[str]]:
        """User 0's writes, its own end of the round against the base, as the record of the round lists them."""
        return {"added": sorted(set(own) - set(self.base)),
                "changed": sorted(rel for rel in set(own) & set(self.base)
                                  if _holds(own[rel]) != _holds(self.base[rel])),
                "removed": sorted(set(self.base) - set(own))}

    def merge(self, rel: str, ours: Entry | None, theirs: Entry | None,
              scratch: Path) -> tuple[Entry | None, str | None]:
        """The path ``rel`` once a later user's write, ``theirs``, is merged into what the one state holds, ``ours``; and
        the rule that settled it, None where only one side changed it or both changed it alike."""
        if self.same(ours, rel):
            return theirs, None
        if ours == theirs:
            return ours, None
        if ours is None or theirs is None:
            return (theirs if ours is None else ours), "a removal yields to a change"
        return merge_entries(rel, self.base_entry(rel), ours, theirs, scratch)
