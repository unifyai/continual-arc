"""Every run a script of this package starts keeps its own record in its folder and goes on only as itself, with one
flag (the shared rule "Experiments keep everything": "Every run resumes"). The scripts call this through lib.sh's
``begin`` as they start, with the machine's own ``python3`` and nothing but its standard library:
``python3 -m continual_arc_training.launch begin|end ...``.

A run's folder holds, beside what the run writes:

  settings.json   what the run computes with, as it was first launched: the script's settings that decide it, and its
                  code, the commit of the checkout it ran from and, where that checkout held changes, their digest
  code.diff       those changes (tracked and untracked) to the code: the package, the benchmark and the baselines
  manifest.json   its record (brain's docs/operations/runs.md): the repository, the run's place under RUNS, why it was
                  made, and each launch: the script, its process, the code it ran, the machine and its GPUs, the
                  settings a launch may change, when it started and ended and how
  resumes.jsonl   a line for each time the run went on from where it stopped: each launch that resumed it, and each loop
                  of it that went on from its files (:func:`resumed`)
  console.log     the launching script's stdout and stderr, each launch's after the last

A launch without RESUME=1 starts a run, and is refused in a folder that already holds one and without a purpose
(PURPOSE, one line of at least six words, or the purpose of the run whose folder holds this one). With RESUME=1 it goes
on with the run in its folder: it is refused where there is none, while another launch of it is going, and when a
setting differs from the run's own, but for those a launch may change, which decide where and how the run is served,
never what it computes (its GPUs, its ports, how often its supervisor restarts a loop, its spend caps, which bind only by
stopping it). A run runs its own code: every launch starts its script again from a checkout of the run's commit with
its changes, made once beside the repository (``CODE_HOME``, default ``~/continual-arc-code/COMMIT``), the code alone
exported from the clone, which it never changes, so that neither a pull in the clone nor a resume from a clone at other
code changes the code a run runs, the loops its supervisor restarts included; a resume compares the settings there,
where the script's defaults are the run's own. A launch that may not start again (``--stay``: ref-tests.sh's, for each
row it tests) is refused at other code.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = "continual-arc"
# a run's code: what its processes import (the package, the benchmark and the baselines) and the scripts it runs, but
# the compiled modules Python writes beside them as it runs
CODE = ("training", "src", "baselines", "pyproject.toml")
NOT_CODE = (":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.pyc")
PURPOSE_WORDS, PURPOSE_CHARS = 6, 40
# what a launcher writes in a run's folder before its launch is recorded, or as it is refused
OWN = ("progress", "console.log")
# what a checkout of a run's own code holds of the code it is (checkout_at), which code_of reads
STAMP = ".launch-code.json"


class Refused(Exception):
    """A launch that may not go on, and why."""


def now() -> str:
    # timezone.utc, not datetime.UTC: the machine's own python3 runs this, 3.10 on Ubuntu 22.04
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def write_json(path: Path, doc: Any) -> None:
    """``doc`` at ``path`` whole or not at all."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), "-c", "core.quotepath=off", *args], capture_output=True, text=True)


def code_of(root: Path) -> dict[str, str] | None:
    """The code of the checkout at ``root``: its commit and its changes to :data:`CODE` (a diff git applies, untracked
    files as new ones), as a checkout of a run's own code says them (its :data:`STAMP`) or git does; None where
    ``root`` is neither."""
    if (root / STAMP).is_file():
        return json.loads((root / STAMP).read_text())
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode:
        return None
    diff = _git(root, "diff", "--no-color", "--no-ext-diff", "--binary", "HEAD", "--", *CODE, *NOT_CODE).stdout
    for name in _git(root, "ls-files", "--others", "--exclude-standard", "--", *CODE, *NOT_CODE).stdout.splitlines():
        diff += _git(root, "diff", "--no-color", "--no-ext-diff", "--binary", "--no-index", "/dev/null", name).stdout
    return {"commit": head.stdout.strip(), "diff": diff}


def digest(text: str) -> str | None:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def checkout_path(home: Path, commit: str, diff: str) -> Path:
    """Where the checkout of ``commit`` with ``diff`` is kept under ``home``."""
    return home / (commit + (f"-{digest(diff)[:12]}" if diff else ""))


def checkout_at(root: Path, commit: str, diff: str, home: Path) -> Path:
    """A checkout of :data:`CODE` at ``commit`` with ``diff`` applied, made once under ``home`` (:func:`checkout_path`)
    from the clone at ``root`` and found there after: the commit's code exported (``git archive``, which only reads the
    clone), the changes applied, and its :data:`STAMP`, written last, saying which code it holds."""
    out = checkout_path(home, commit, diff)
    if (out / STAMP).is_file():
        return out
    if (root / STAMP).is_file():
        raise Refused(f"the run's code is {commit}{' with changes' if diff else ''}, and {root}, a checkout of other "
                      "code, holds no history to make it from: launch the run from a clone")
    if _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode:
        _git(root, "fetch", "--quiet", "origin", commit)
    held = _git(root, "ls-tree", "--name-only", commit, "--", *CODE).stdout.split()   # those of CODE the commit has
    if not held:
        raise Refused(f"the run's code, {commit}, is not in {root}: launch the run from a clone that has it")
    archive = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit, "--", *held],
                             capture_output=True)
    if archive.returncode:
        raise Refused(f"the run's code, {commit}, cannot be read from {root}: {archive.stderr.decode().strip()}")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)   # one a stop cut short
    tmp.mkdir(parents=True)
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive.stdout, check=True)
    if diff:   # applied in a repository of the export's own, so that git reads its paths from the export's top
        for step in (("init", "-q"), ("apply", "--binary", "-")):
            made = subprocess.run(["git", "-C", str(tmp), *step], input=diff if step[0] == "apply" else None,
                                  capture_output=True, text=True)
            if made.returncode:
                raise Refused(f"the run's changes (code.diff) do not apply to {commit}: {made.stderr.strip()}")
        shutil.rmtree(tmp / ".git")
    (tmp / STAMP).write_text(json.dumps({"commit": commit, "diff": diff}))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


def purpose_of(folder: Path, runs: Path | None) -> str:
    """The purpose of the run whose folder holds ``folder`` (under ``runs``), said as this run's: a test of a run, say.
    Empty where no run holds it."""
    for parent in folder.parents:
        if runs is not None and (parent == runs or runs not in parent.parents):
            break
        manifest = parent / "manifest.json"
        if manifest.exists():
            outer = json.loads(manifest.read_text())
            return f"part of {outer['run']} ({folder.relative_to(parent)}): {outer['purpose']}"
    return ""


def _alive(pid: int, by: str) -> bool:
    """Whether process ``pid`` lives and runs the script ``by``."""
    shown = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return by in shown


def gpus() -> list[str]:
    """The machine's GPUs as nvidia-smi names them, none without it."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def resumed(folder: Path, **where: Any) -> None:
    """A line in the run's ``resumes.jsonl``: a launch or a loop of the run going on from where it stopped, and from
    where (``where``)."""
    with open(folder / "resumes.jsonl", "a") as fp:
        fp.write(json.dumps({"at": now(), **where}) + "\n")


def say(log: Path, by: str, message: str) -> None:
    """A line of the run's progress log, as the scripts write theirs, where its folder is."""
    if log.parent.is_dir():
        with open(log, "a") as fp:
            fp.write(f"{datetime.now(timezone.utc):%m-%d %H:%M} {by}: {message}\n")  # noqa: UP017


def begin(folder: Path, *, run: str, by: str, pid: int, resume: bool, settings: dict[str, str], free: set[str],
          root: Path, purpose: str = "", moved: bool = False, stay: bool = False, home: Path | None = None,
          runs: Path | None = None) -> Path | None:
    """The launch of the run in ``folder`` (its place ``run`` under ``runs``) by the script ``by`` (process ``pid``),
    from the checkout at ``root``, with ``settings`` (``free`` those a launch may change), as the module's docstring
    says: recorded, and None, for the script to go on; or the checkout of the run's own code (:func:`checkout_at`),
    for the script to start again from, unless it runs from there already, has started again (``moved``) or may not
    (``stay``), when it must hold the run's code. Raises :class:`Refused`."""
    record_path, manifest_path = folder / "settings.json", folder / "manifest.json"
    home = home or Path.home() / "continual-arc-code"
    code = code_of(root)
    if code is None:
        raise Refused(f"a run keeps the commit of its code, and {root} is no git checkout: launch it from a clone")
    strict = {k: v for k, v in settings.items() if k not in free}
    if not resume:
        held = sorted(p.name for p in folder.iterdir() if p.name not in OWN) if folder.is_dir() else []
        if held:
            raise Refused(f"{folder} already holds a run ({', '.join(held[:3])}): RESUME=1 goes on with it from where "
                          "it stopped, and a new run takes a folder of its own")
        purpose = purpose or purpose_of(folder, runs)
        if "\n" in purpose or len(purpose.split()) < PURPOSE_WORDS or len(purpose) < PURPOSE_CHARS:
            raise Refused(f"a new run says why it is made: PURPOSE, one line of at least {PURPOSE_WORDS} words and "
                          f"{PURPOSE_CHARS} characters, not {purpose!r}")
        commit, diff = code["commit"], code["diff"]
    else:
        if not record_path.exists():
            raise Refused(f"RESUME=1 goes on with a run, and {folder} holds none (no settings.json): without RESUME a "
                          "launch starts one")
        record = json.loads(record_path.read_text())
        commit = record["code"]["commit"]
        diff = (folder / "code.diff").read_text() if record["code"]["diff_sha256"] else ""
    # the run's processes run a checkout of its code that no pull changes, every loop a supervisor restarts included
    if not (moved or stay) and root.resolve() != checkout_path(home, commit, diff).resolve():
        return checkout_at(root, commit, diff, home)
    if (code["commit"], digest(code["diff"])) != (commit, digest(diff)):
        if moved:
            raise Refused(f"the checkout at {root}, made for the run's own code, does not hold it")
        raise Refused(f"the run began on {commit}{' with changes' if diff else ''}, and this checkout is at other "
                      "code: go on with it from a checkout of its own")
    if not resume:
        folder.mkdir(parents=True, exist_ok=True)
        if diff:
            (folder / "code.diff").write_text(diff)
        write_json(record_path, {"settings": strict, "code": {"commit": commit, "diff_sha256": digest(diff)}})
        manifest: dict[str, Any] = {"repo": REPO, "run": run, "purpose": purpose, "created": now(), "launches": []}
    else:
        differ = sorted(k for k in strict.keys() | record["settings"].keys()
                        if strict.get(k) != record["settings"].get(k))
        if differ:
            raise Refused("RESUME=1 goes on with the run's own settings, and these differ from its settings.json: "
                          + ", ".join(f"{k} {record['settings'].get(k)!r}, not {strict.get(k)!r}" for k in differ))
        manifest = json.loads(manifest_path.read_text())
        going = [launch for launch in manifest["launches"]
                 if launch.get("ended_at") is None and _alive(launch["pid"], launch["by"])]
        if going:
            raise Refused(f"the run goes on already: its launch {going[-1]['id']} by {going[-1]['by']}, process "
                          f"{going[-1]['pid']}, has not ended")
    launch = {"id": len(manifest["launches"]) + 1, "by": by, "pid": pid, "resume": resume, "started_at": now(),
              "ended_at": None, "status": "running", "commit": commit, "diff_sha256": digest(diff), "code": str(root),
              "host": socket.gethostname(), "user": getpass.getuser(), "gpus": gpus(),
              "settings": {k: v for k, v in settings.items() if k in free},
              **({"purpose": purpose} if resume and purpose else {})}
    manifest["launches"].append(launch)
    write_json(manifest_path, manifest)
    if resume:
        resumed(folder, launch=launch["id"], by=by)
    return None


def end(folder: Path, pid: int, status: int) -> None:
    """The end of the launch of the run in ``folder`` by process ``pid``, with the script's exit ``status``."""
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    launch = next(launch for launch in reversed(manifest["launches"]) if launch["pid"] == pid)
    launch.update(ended_at=now(), status="done" if status == 0 else "failed", exit=status)
    write_json(path, manifest)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("begin", help="record a launch, refuse it, or print the checkout of the run's code to go on from")
    b.add_argument("folder", type=Path)
    b.add_argument("--run", required=True, help="the run's place under RUNS")
    b.add_argument("--by", required=True, help="the launching script")
    b.add_argument("--pid", type=int, required=True, help="the launching script's process")
    b.add_argument("--resume", default="0", help="RESUME: 1 goes on with the run in the folder")
    b.add_argument("--free", default="", help="the settings a launch may change, one word each")
    b.add_argument("--root", type=Path, required=True, help="the checkout the script runs from")
    b.add_argument("--purpose", default="")
    b.add_argument("--moved", default="0", help="1: the script already started again from the run's own code")
    b.add_argument("--stay", action="store_true", help="refuse a resume at other code rather than move to the run's")
    b.add_argument("--runs", type=Path, help="RUNS, the top of every run's folder")
    b.add_argument("--log", type=Path, help="the script's progress log (default the folder's progress)")
    b.add_argument("settings", nargs="*", help="NAME=VALUE, each of the script's settings")
    e = sub.add_parser("end", help="record how a launch ended")
    e.add_argument("folder", type=Path)
    e.add_argument("--pid", type=int, required=True)
    e.add_argument("--status", type=int, required=True)
    a = p.parse_args(argv)
    if a.cmd == "end":
        end(a.folder, a.pid, a.status)
        return
    settings = dict(item.split("=", 1) for item in a.settings)
    home = Path(os.environ["CODE_HOME"]) if os.environ.get("CODE_HOME") else None
    log = a.log or a.folder / "progress"
    try:
        moved_to = begin(a.folder, run=a.run, by=a.by, pid=a.pid, resume=a.resume == "1", settings=settings,
                         free=set(a.free.split()), root=a.root, purpose=a.purpose, moved=a.moved == "1",
                         stay=a.stay, home=home, runs=a.runs)
    except Refused as why:
        say(log, a.by, f"refused: {why}")
        print(f"{a.by}: refused: {why}", file=sys.stderr)
        raise SystemExit(1) from None
    if moved_to is not None:
        say(log, a.by, f"starts again from {moved_to}, a checkout of the run's own code")
        print(moved_to)


if __name__ == "__main__":
    main()
