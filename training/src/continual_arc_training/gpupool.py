"""The trainer GPUs real-time runs on one node share. Runs given the same TRAIN_GPUS (realtime.sh) share them as one
pool; a run whose list overlaps another's without being the same is refused. A run's trainer needs GPUs only while it
trains a phase, and waits the rest of the time for its rounds' labels and play, so the pool's GPUs go to whichever runs
have a phase to train: a phase takes an even share and keeps it to its end, giving GPUs up only to a run whose phase
needs them, at a step checkpoint (a stop costs the step in flight and a restart), and taking more only when its next
phase starts. A run alone on its GPUs always holds all of them.

The pool's state is ``RUNS/gpupool/<gpus>.json`` (``RUNS`` from the environment, as realtime.sh exports it; ``~/runs``
by default), read and written under an flock by every run's trainer (the real-time driver's ``trainer``, which runs
the phases in a ``train`` process on the GPUs it is given): per run, whether it wants GPUs (a phase to train), whether
a phase is in flight, the GPUs its ``train`` process holds and that process's pid, and when the run last wrote. Every
run computes the same :func:`split` of it, so they agree on who gets which GPU without talking. A run that has not
written for :data:`STALE_S` and whose process is gone is dropped.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("RUNS") or Path.home() / "runs").expanduser() / "gpupool"
STALE_S = 600


def parse(gpus: str) -> list[int]:
    return sorted({int(g) for g in gpus.split(",") if g.strip()})


def name(gpus: list[int]) -> str:
    return "-".join(map(str, gpus))


def split(gpus: list[int], runs: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    """Each run's GPUs from the pool ``gpus``. The GPUs go evenly to the runs that want them (``want``), the odd ones to
    the runs holding more already; a run whose phase is in flight (``busy``) never gets more than it holds (it grows
    when its next phase starts), and what it cannot take goes to the others. Each run keeps the GPUs it holds where its
    share allows, and takes free ones in order, so a split moves as few GPUs as it can. A run that wants none gets none:
    what an idle run's process holds is free for the others."""
    wanting = sorted((r for r, s in runs.items() if s["want"]), key=lambda r: (-len(runs[r]["held"]), r))
    count: dict[str, int] = {}
    left, todo = len(gpus), list(wanting)
    while todo:   # water-filling: the capped runs first, each at most an even share of what is left
        caps = {r: len(runs[r]["held"]) if runs[r].get("busy") else len(gpus) for r in todo}
        even, extra = divmod(left, len(todo))
        capped = [r for r in todo if caps[r] < even + (1 if todo.index(r) < extra else 0)]
        if not capped:
            for i, r in enumerate(todo):
                count[r] = even + (1 if i < extra else 0)
            break
        for r in capped:
            count[r] = caps[r]
            left -= caps[r]
            todo.remove(r)
    out: dict[str, list[int]] = {r: [] for r in runs}
    taken: set[int] = set()
    for r in wanting:   # each keeps what it holds, up to its share
        keep = [g for g in sorted(runs[r]["held"]) if g in gpus and g not in taken][:count[r]]
        out[r] = keep
        taken.update(keep)
    free = [g for g in gpus if g not in taken]
    for r in wanting:
        need = count[r] - len(out[r])
        out[r] = sorted(out[r] + free[:need])
        free = free[need:]
    return out


# what a trainer's or a train process's command line holds (python -m continual_arc_training.realtime ...)
DRIVER = b"continual_arc_training.realtime"


def alive(pid: int | None) -> bool:
    """Whether ``pid`` is a live process of the real-time driver: the state file outlives a machine's restart, after
    which a pid it names may be any other process's."""
    if not pid:
        return False
    try:
        return DRIVER in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


class Pool:
    """One node's pool of trainer GPUs (the module docstring), seen from run ``run``."""

    def __init__(self, gpus: list[int], run: str, root: Path | None = None) -> None:
        self.gpus, self.run = gpus, run
        root = root or ROOT
        root.mkdir(parents=True, exist_ok=True)
        for other in root.glob("*.json"):   # one pool per GPU list: an overlapping list would hand a GPU out twice
            theirs = [int(g) for g in other.stem.split("-")]
            if theirs != gpus and set(theirs) & set(gpus) and self._members(other):
                raise SystemExit(f"TRAIN_GPUS {name(gpus)} overlaps the pool {other.stem} other runs on this node "
                                 f"train on: give runs that share GPUs the same list")
        self.path = root / f"{name(gpus)}.json"

    @staticmethod
    def _members(path: Path) -> dict[str, Any]:
        try:
            runs = json.loads(path.read_text())["runs"]
        except (FileNotFoundError, ValueError, KeyError):
            return {}
        return {r: s for r, s in runs.items() if alive(s.get("pid")) or time.time() - s["seen"] < STALE_S}

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        with open(self.path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = {"gpus": self.gpus, "runs": self._members(self.path)}
            yield state
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(self.path)

    def update(self, want: bool, busy: bool, held: list[int], pid: int | None) -> dict[str, Any]:
        """This run's entry written, and the pool as every run now sees it: each run's share (:func:`split`) and
        each one's ``held`` (with its process's pid)."""
        with self._locked() as state:
            state["runs"][self.run] = {"want": want, "busy": busy, "held": sorted(held), "pid": pid,
                                       "seen": time.time()}
            runs = state["runs"]
        return {"share": split(self.gpus, runs), "runs": runs}

    def entry(self) -> dict[str, Any] | None:
        """This run's entry as the pool holds it, read and left as it is."""
        with self._locked() as state:
            return state["runs"].get(self.run)

    def leave(self) -> None:
        with self._locked() as state:
            state["runs"].pop(self.run, None)

    def share_if_starting(self) -> list[int]:
        """This run's share were it to start a phase now (its ``train`` process asks before each phase): its entry read
        as wanting and not busy, every other run's as it is."""
        with self._locked() as state:
            runs = dict(state["runs"])
            mine = runs.get(self.run, {"held": [], "pid": None, "seen": time.time()})
            runs[self.run] = {**mine, "want": True, "busy": False}
        return split(self.gpus, runs)[self.run]

    def claim(self, gpus: list[int], pid: int) -> bool:
        """``gpus`` taken for this run, by ``pid`` (its trainer's, until the train process it starts on them has its
        own), if no other run's live process holds any of them: checked and written under one lock, so two runs never
        start on the same GPU, whatever each saw a moment before."""
        with self._locked() as state:
            if self.held_by_others(gpus, state["runs"]):
                return False
            mine = state["runs"].setdefault(self.run, {"want": True, "busy": False, "held": [], "pid": None})
            mine.update(held=sorted(gpus), pid=pid, seen=time.time())
            return True

    def held_by_others(self, gpus: list[int], runs: dict[str, Any]) -> set[int]:
        """Which of ``gpus`` another run's live process holds."""
        return {g for r, s in runs.items() if r != self.run and alive(s.get("pid")) for g in s["held"]} & set(gpus)
