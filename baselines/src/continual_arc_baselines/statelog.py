"""Every round's end state of a system, kept as one git commit, so that a run can be taken up again from any round.

At the end of every round :meth:`StateLog.commit` commits the system's whole state directory (``workdir/state``,
everything it persists between instances) into a repository beside it (``workdir/state.git``) and appends the round
and its commit to ``workdir/state-log.jsonl``. Git stores a file once however many rounds it lasts, and a delta for one
that changes, so a run's every state costs about what its last one does. Nothing in the directory is left out: ignore
files a system writes there are overridden, so the commit is the state as it was, caches and environments included.

A system may keep working between instances (a consolidation on a schedule, a refinement of its own), so the run's
other processes are stopped with SIGSTOP for the moment the commit takes and continued after it, and no file is caught
half written. :func:`restore` writes a commit's state into a new directory, from which a run is continued. Git does not
keep empty directories, so a restored state lacks the ones the system had made and left empty, and a system that keeps
nothing (a floor's) restores as an empty directory.

A run that resumes puts its system's state back as it was last committed before the round the run goes on from
(:meth:`StateLog.saved`, which the runner keeps beside the round's records, and :meth:`StateLog.go_back`): the state
directory as that commit holds it, the repository's branch and index on it, and the log's later rounds forgotten, so
that the next commit follows it as though nothing had come between.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path

# The repository's own identity and settings, whatever git configuration the host has: a global ignore file or hook
# must not change what a commit holds.
GIT_ENV = {"GIT_AUTHOR_NAME": "continual-arc", "GIT_AUTHOR_EMAIL": "state@continual-arc",
           "GIT_COMMITTER_NAME": "continual-arc", "GIT_COMMITTER_EMAIL": "state@continual-arc",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def _git(repo: Path, *args: str, tree: Path | None = None) -> str:
    """Run git on ``repo``; with ``tree``, from inside it as the work tree, so that a pathspec of "." means all of it. A
    failure raises with what git wrote on stderr, which says why and which a CalledProcessError's message leaves out."""
    work = [f"--work-tree={tree}"] if tree is not None else []
    done = subprocess.run(["git", f"--git-dir={repo}", *work, *args], env={**os.environ, **GIT_ENV}, cwd=tree,
                          capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)} exited {done.returncode} on {repo}: {done.stderr.strip()}")
    return done.stdout


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


def ensure_repo(repo: Path) -> None:
    """A state's repository at ``repo``, made if there is none: bare, but for a work tree named at every call, and never
    repacked by itself, since a repack mid-run would hold the run stopped while it ran."""
    if not (repo / "HEAD").exists():
        subprocess.run(["git", "init", "--quiet", "--bare", str(repo)], env={**os.environ, **GIT_ENV}, check=True)
        _git(repo, "config", "core.bare", "false")
        _git(repo, "config", "gc.auto", "0")


def _descendants(root: int) -> list[int]:
    """Every live process below ``root``, from Linux's /proc; none on a host without it."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            stat = (entry / "stat").read_text()
        except OSError:  # the process ended while the table was read
            continue
        children.setdefault(int(stat.rsplit(")", 1)[1].split()[1]), []).append(int(entry.name))
    found, todo = [], [root]
    while todo:
        for child in children.get(todo.pop(), []):
            found.append(child)
            todo.append(child)
    return found


def _signal(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:  # ended since it was listed
            pass


class StateLog:
    """The commits of one system's state directory, one per round, numbered from the rounds a log already holds, so a
    run resumed after a crash goes on counting where it stopped."""

    def __init__(self, tree: Path, repo: Path, log: Path):
        self.tree, self.repo, self.log = Path(tree).resolve(), Path(repo).resolve(), Path(log).resolve()
        self.tree.mkdir(parents=True, exist_ok=True)
        ensure_repo(self.repo)
        self.rounds = sum(1 for _ in self.log.open()) if self.log.exists() else 0

    def commit(self) -> str:
        """The state at the end of the round just played, as the next commit; returns its hash."""
        stopped = _descendants(os.getpid())
        _signal(stopped, signal.SIGSTOP)
        try:
            _git(self.repo, "add", "--all", "--force", ".", tree=self.tree)
            _git(self.repo, "commit", "--quiet", "--allow-empty", "--no-verify", "-m", f"round {self.rounds}",
                 tree=self.tree)
            commit = _git(self.repo, "rev-parse", "HEAD").strip()
        finally:
            _signal(stopped, signal.SIGCONT)
        with self.log.open("a") as fp:
            fp.write(json.dumps({"round": self.rounds, "commit": commit, "at": time.time()}) + "\n")
        self.rounds += 1
        return commit

    def commit_of(self, round_index: int) -> str:
        return commit_of(self.log, round_index)

    def head(self) -> str | None:
        """The commit the repository's branch is on: the last round's, or one made on it (a fork's relocation); None
        before the first."""
        found = subprocess.run(["git", f"--git-dir={self.repo}", "rev-parse", "--verify", "--quiet", "HEAD"],
                               env={**os.environ, **GIT_ENV}, capture_output=True, text=True)
        return found.stdout.strip() or None

    def saved(self) -> dict:
        """Where the state was last committed: the rounds the log lists and the commit the branch is on, for
        :meth:`go_back`."""
        return {"rounds": self.rounds, "commit": self.head()}

    def go_back(self, saved: dict) -> None:
        """The state directory, the repository's branch and index and the log as they were when :meth:`saved` gave
        ``saved``: the directory written afresh from its commit, or emptied where there was none, unless it holds that
        commit's tree already, and the log's later rounds forgotten. Nothing of the system may be running."""
        commit, head = saved["commit"], self.head()
        if commit is None or head != commit or not self.tree.exists() or self.changed():
            if self.tree.exists():   # a system's wipe may take its state directory with it
                delete(self.tree)
            if commit is None:
                self.tree.mkdir(parents=True)
                if head is not None:
                    _git(self.repo, "update-ref", "-d", "HEAD")
                (self.repo / "index").unlink(missing_ok=True)
            else:
                restore(self.repo, commit, self.tree)
                _git(self.repo, "reset", "--quiet", commit, tree=self.tree)
        lines = self.log.read_text().splitlines()[:saved["rounds"]] if self.log.exists() else []
        self.log.write_text("".join(line + "\n" for line in lines))
        self.rounds = saved["rounds"]

    def changed(self) -> bool:
        """Whether the state directory differs from the commit the branch is on, in any file, ignored ones included,
        since every commit takes them."""
        return bool(_git(self.repo, "status", "--porcelain", "--untracked-files=all", "--ignored", tree=self.tree))


def commit_of(log: Path, round_index: int) -> str:
    """The commit holding the state at the end of round ``round_index``, from a state log."""
    for line in Path(log).read_text().splitlines():
        entry = json.loads(line)
        if entry["round"] == round_index:
            return entry["commit"]
    raise KeyError(f"no commit for round {round_index} in {log}")


def restore(repo: Path, commit: str, into: Path) -> None:
    """Write the state ``commit`` holds into ``into``, a directory that must not exist yet: git reads the commit's tree
    into an index of its own, never the repository's, and checks every file out of that index into ``into``."""
    repo, into = Path(repo).resolve(), Path(into).resolve()
    into.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, **GIT_ENV, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        for args in (["read-tree", commit], ["checkout-index", "--all"]):
            subprocess.run(["git", f"--git-dir={repo}", f"--work-tree={into}", *args], env=env, check=True,
                           capture_output=True)
