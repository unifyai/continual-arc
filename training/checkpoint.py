"""A training phase's step checkpoints: a phase cut short, by a preempted node or a killed trainer, goes on from its last
checkpoint, on the same batches in the same order, instead of training again from its start.

``OUT/checkpoints/`` (OUT a phase's folder: train.py's ``--out``, realtime.py's ``train/pNNNN``) holds the trainer's
saves, a folder each (``cNNNN``, ``Trainer.save``'s files), and ``state.json``, which names the newest and the best of
them and holds what the phase's loop needs to go on as it would have: the step, the random generator's state now (the
steps themselves are planned again from its state as the phase began, train.plan), the steps' records (each with a
digest of its batch's calls), the held-out curve and its patience count, the counts so far and the seconds spent.
``state.json`` is written last and atomically, after the folder it names is whole on disk, every file synced: a node
that stops at any moment leaves the last checkpoint whole, and a folder no ``state.json`` names, one cut short, goes at
the next start. At most three saves are there at once (3.7 GB each for Qwen3.5-9B at rank 128): the best, the newest
and the one being written.

Only the save itself holds training up (the trainer's files written to the page cache, about 4 s): the sync to disk
and the commit run on a thread of their own while the next steps train, since a Nebius boot disk takes some 15 s to
sync 3.7 GB (250 MB/s). One commit at a time, in order; the phase's end, a resume's state and SIGTERM's checkpoint wait
for it.

A checkpoint is written once SIGTERM has come (:func:`catch_sigterm`: Nebius's notice, 60 s before it stops a
preemptible machine), after the step in flight, when the process exits (code 143), and every :data:`EVERY` steps, for a
stop with no notice in time. A phase resumes from a checkpoint only if it is its own: the same settings, starting
adapter, training and held-out calls (:func:`fingerprint`), and batches that replay as recorded (:func:`replays`); any
other is dropped and the phase trains again whole.

Under a held-out stop the best adapter is the checkpoint of the check that found it, or the adapter the phase started
from (``start``); it becomes ``OUT/adapter`` when the phase ends (:meth:`Checkpoints.write_adapter`, its files linked,
not copied), and ``OUT/checkpoints`` goes once ``OUT/DONE`` is written (:meth:`Checkpoints.clear`).

All of it but the best adapter's keeping waits on :data:`ON`.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Whether phases checkpoint their steps and go on from them after a stop, SIGTERM waiting for the step in flight. Off,
# a phase writes only its best adapter's checkpoint, as it saved that adapter before, a phase cut short trains again
# whole and SIGTERM ends the trainer at once. On: train.fit's checkpoints passed their GPU check (docs/method-search.md,
# "Step checkpoints"), so every run started from this code takes them; the paper's nine launch runs trained without
# them, their nodes keeping the code they were started with, from before the checkpoints, to the end.
ON = True
FORMAT = 1
# steps between a phase's checkpoints with ON, whatever its held-out cadence. Each holds training up 4 s, 3% of DAgger's
# steps at one every 5 (26 s a step), and a preemption comes once in some 22 machine-hours with SIGTERM's
# own checkpoint, so these are for a stop with no notice in time and come sparsely: at 25, 0.6% of DAgger's training,
# and such a stop loses at most 11 minutes of it
EVERY = 25
# the exit code after SIGTERM's checkpoint: 128 + the signal, as a process the signal itself ended
SIGTERM_EXIT = 128 + signal.SIGTERM
_TERM = threading.Event()


def catch_sigterm() -> None:
    """With :data:`ON`, SIGTERM asks the phase in flight to checkpoint after its step and exit
    (:meth:`Checkpoints.step`), rather than end the process mid-step; the trainer's data-parallel workers ignore it, so
    that every rank gets through the step (ask_sd.parallel's worker)."""
    if ON:
        signal.signal(signal.SIGTERM, lambda signum, frame: _TERM.set())


def stopping() -> bool:
    """Whether SIGTERM has come."""
    return _TERM.is_set()


def leave(step: int | None) -> None:
    """The process's end after SIGTERM: its output flushed, then gone at once, without the interpreter's exit, which
    would wait for an on-policy draw still in flight on a server that may be gone."""
    print(f"SIGTERM: {'checkpointed at step ' + str(step) if step is not None else 'no phase in flight'}; exiting",
          flush=True)
    sys.stderr.flush()
    os._exit(SIGTERM_EXIT)


def digest(doc: Any) -> str:
    """A short, stable hash of a JSON-able document."""
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _meta(x: Any) -> dict[str, Any]:
    """An item's or a call's labels: a TokenItem's ``label``, an ask_sd Call's ``meta``."""
    return getattr(x, "label", None) or getattr(x, "meta", None) or {}


def keys(batch: Any) -> Any:
    """A batch's calls by their place in their runs (run, user, i), nested as the batch is (a step of chain pieces is a
    list of lists)."""
    if isinstance(batch, (list, tuple)):
        return [keys(x) for x in batch]
    m = _meta(batch)
    return [m.get("run"), m.get("user"), m.get("i")]


def batch_digest(batch: Any) -> str:
    """The digest a step's record keeps of its batch (:func:`keys`), which a resumed phase's replayed batches must
    match (:func:`replays`)."""
    return digest(keys(batch))


def _shape(x: Any) -> list[Any]:
    """An item's or a call's place and lengths: what a phase's own calls share with its checkpoint's."""
    first = getattr(x, "student_ids", None) or getattr(x, "prompt_ids", None) or ()
    second = getattr(x, "teacher_ids", None) or getattr(x, "reply_ids", None) or ()
    return [*keys(x), len(first), len(second)]


def fingerprint(a: Any, init: Path | None, *groups: list[Any]) -> str:
    """What a checkpoint must share with the phase that resumes from it: the settings but the GPUs (a phase's steps do
    not depend on them, train.share, so it goes on on however many the pool now gives it: gpupool.py), the adapter it
    started from, and its items, training and held-out alike, each by its place and lengths."""
    return digest({"format": FORMAT, "args": {k: v for k, v in vars(a).items() if k != "gpus"},
                   "init": str(init) if init else None,
                   "items": [[_shape(x) for x in g] for g in groups]})


def replays(batches: list[Any], records: list[dict[str, Any]]) -> bool:
    """Whether ``batches``, drawn again, begin with the batches the steps of ``records`` trained on."""
    return len(records) <= len(batches) and all(r.get("batch") == batch_digest(b) for r, b in zip(records, batches))


def rng_state(saved: list[Any]) -> tuple[Any, ...]:
    """A ``random.Random`` state from its JSON form (``getstate()``'s tuple, read back as lists)."""
    version, internal, gauss = saved
    return version, tuple(internal), gauss


def saved_step(out: Path) -> int | None:
    """The step of the phase's newest checkpoint in ``out`` it would go on from, None without one or without
    :data:`ON`."""
    path = out / "checkpoints" / "state.json"
    return json.loads(path.read_text())["step"] if ON and path.exists() else None


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync(folder: Path) -> None:
    """Every file under ``folder`` and every folder's entries on disk, not only in the page cache: a machine stopped
    outright loses what was not."""
    for root, _, files in os.walk(folder):
        for name in files:
            _fsync(Path(root) / name)
        _fsync(Path(root))


def _write(path: Path, text: str) -> None:
    """``text`` at ``path`` atomically and on disk."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fp:
        fp.write(text)
        fp.flush()
        os.fsync(fp.fileno())
    tmp.rename(path)
    _fsync(path.parent)


def _log(doc: dict[str, Any]) -> None:
    """One line of the trainer's log, written whole (a commit's thread writes them too)."""
    sys.stdout.write(json.dumps(doc) + "\n")
    sys.stdout.flush()


def link_tree(src: Path, dst: Path, skip: tuple[str, ...] = ()) -> None:
    """``src``'s files at ``dst`` as hard links, but the top-level names in ``skip``. Nothing here writes a file of a
    save again once it is whole (a save goes to a new folder, and pruning only unlinks), so the two never differ."""
    shutil.copytree(src, dst, copy_function=os.link, ignore=lambda folder, names: skip if Path(folder) == src else ())


class Checkpoints:
    """A phase's checkpoints (the module docstring): ``trainer`` saves and loads them, ``rng`` is the phase's random
    generator, ``state()`` the loop's own state for ``state.json`` (JSON-able), and ``extra(folder)``, if given, writes
    what else the phase keeps with each checkpoint into ``folder`` (its ``extra/``: an on-policy phase's drawing adapter,
    train.SelfDistillation.keep)."""

    def __init__(self, out: Path, trainer: Any, rng: random.Random, fingerprint: str,
                 state: Callable[[], dict[str, Any]], extra: Callable[[Path], None] | None = None) -> None:
        self.root = out / "checkpoints"
        self.trainer, self.rng, self.fingerprint = trainer, rng, fingerprint
        self.state, self.extra = state, extra
        self.seq = 0
        self.dir: str | None = None    # the newest checkpoint's folder
        self.best: str | None = None   # the best adapter's: "start", a checkpoint's folder, or None without a held-out stop
        self.at: int | None = None     # the newest checkpoint's step
        self.seconds, self.t0 = 0.0, time.time()
        self.resumed: list[int] = []   # the steps this phase went on from
        self._committer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self._commit: Future | None = None

    # -- reading back ----------------------------------------------------------------------------------------------

    def load(self) -> dict[str, Any] | None:
        """This phase's newest checkpoint (its ``state.json``), or None, the folder cleared, when there is none or it
        is another phase's (other settings, adapter or calls: code deployed between the stop and the start, say), or
        without :data:`ON`."""
        path = self.root / "state.json"
        if not (ON and path.exists()):
            shutil.rmtree(self.root, ignore_errors=True)
            return None
        saved = json.loads(path.read_text())
        if saved.get("format") != FORMAT or saved.get("fingerprint") != self.fingerprint:
            self.discard(f"the checkpoint at step {saved.get('step')} is another phase's (settings, adapter or calls)")
            return None
        for leftover in self.root.iterdir():   # a save cut short, or one a later save replaced
            if leftover.is_dir() and leftover.name not in (saved["dir"], saved["best"]):
                shutil.rmtree(leftover)
        return saved

    def resume(self, saved: dict[str, Any]) -> None:
        """The trainer, the generator and the clock back where ``saved`` (:meth:`load`) left them."""
        self.trainer.load(self.root / saved["dir"])
        self.rng.setstate(rng_state(saved["rng"]))
        self.seq, self.dir, self.best, self.at = saved["seq"], saved["dir"], saved["best"], saved["step"]
        self.seconds, self.t0 = saved["seconds"], time.time()
        self.resumed = saved["resumed"] + [saved["step"]]
        print(json.dumps({"resumed": {"step": saved["step"], "checkpoint": str(self.root / saved["dir"]),
                                      "best": saved["best"], "minutes": round(saved["seconds"] / 60, 1)}}), flush=True)

    def extra_dir(self, saved: dict[str, Any], name: str) -> Path:
        """The folder kept as ``name`` with the checkpoint ``saved`` (:meth:`load`)."""
        return self.root / saved["dir"] / "extra" / name

    def discard(self, why: str) -> None:
        """Every checkpoint of the phase gone: it trains again from its start."""
        print(json.dumps({"checkpoint dropped": why}), flush=True)
        shutil.rmtree(self.root, ignore_errors=True)

    # -- writing ---------------------------------------------------------------------------------------------------

    def minutes(self) -> float:
        """Minutes of training so far, those before any resume included."""
        return (self.seconds + time.time() - self.t0) / 60

    def due(self, step: int) -> bool:
        """Whether a checkpoint falls after ``step`` steps (every :data:`EVERY`)."""
        return step % EVERY == 0

    def save(self, step: int, *, best: bool = False, wait: bool = False) -> None:
        """A checkpoint of the phase after ``step`` steps; with ``best``, the best adapter so far too (a held-out check
        found it). The trainer's files are written now, and synced and committed on the commit thread, or here with
        ``wait``. The newest checkpoint already holds the trainer at ``step`` when it was written there (the check after
        a resume's first save, say), so then only ``state.json`` is written again. Without :data:`ON` only a best is
        kept."""
        if not (ON or best):
            return
        self.wait()
        t = time.time()
        stale = {self.dir, self.best}
        fresh = None
        if self.at != step:
            self.seq += 1
            name = f"c{self.seq:04d}"
            tmp = self.root / f".{name}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            self.trainer.save(tmp)
            if self.extra:
                (tmp / "extra").mkdir()
                self.extra(tmp / "extra")
            fresh = (tmp, self.root / name)
            self.dir, self.at = name, step
        if best:
            self.best = self.dir
        # the loop's state as it is now, which the steps after go on changing
        state = json.dumps({
            "format": FORMAT, "fingerprint": self.fingerprint, "seq": self.seq, "step": step, "dir": self.dir,
            "best": self.best, "rng": self.rng.getstate(),
            "seconds": round(self.seconds + time.time() - self.t0, 1), "resumed": self.resumed,
            "loop": self.state()}, default=str)   # as phase.json takes the steps' records
        folder, keep, saved = self.dir, {self.dir, self.best}, time.time() - t

        def commit() -> None:
            t = time.time()
            if fresh is not None:
                _sync(fresh[0])
                fresh[0].rename(fresh[1])
            _write(self.root / "state.json", state)
            for gone in stale - keep - {None}:
                shutil.rmtree(self.root / gone, ignore_errors=True)
            _log({"checkpoint": {"step": step, "dir": folder, "best": best, "save_s": round(saved, 1),
                                 "sync_s": round(time.time() - t, 1)}})

        if wait:
            commit()
        else:
            self._commit = self._committer.submit(commit)

    def wait(self) -> None:
        """The commit in flight, if any, done (an error of its raised here)."""
        if self._commit is not None:
            commit, self._commit = self._commit, None
            commit.result()

    def start_is_best(self, init: Path | None) -> None:
        """The adapter the phase started from is the best so far (the held-out check before the first step found it):
        ``init``'s, which stays on disk while the phase runs, or, with none, the trainer's untrained adapter, saved
        now, since its random A matrices are this process's own."""
        if init is None:
            tmp = self.root / ".start"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            self.trainer.save(tmp)
            _sync(tmp)
            shutil.rmtree(self.root / "start", ignore_errors=True)
            tmp.rename(self.root / "start")
        self.best = "start"

    def step(self, step: int) -> None:
        """Between steps (``step`` of them taken): once SIGTERM has come, a checkpoint, unless the newest is at this
        step or none has been taken, and the process's end (:func:`leave`)."""
        if stopping():
            if step and self.at != step:
                self.save(step, wait=True)
            self.wait()
            leave(step)

    # -- the end ---------------------------------------------------------------------------------------------------

    def best_dir(self, init: Path | None) -> Path | None:
        """The folder the best adapter so far is in: ``init``, the untrained one's (``start``) or a checkpoint's; None
        without a held-out stop."""
        if self.best is None:
            return None
        return init if self.best == "start" and init is not None else self.root / self.best

    def write_adapter(self, dst: Path, init: Path | None) -> None:
        """``dst`` (``OUT/adapter``) the phase's adapter: the best one's files, linked from where they are
        (:meth:`best_dir`), or, with no held-out stop, the trainer's as training left it. Written beside it first, so
        that ``dst`` is never half of one."""
        self.wait()
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        best = self.best_dir(init)
        if best is None:
            tmp.mkdir(parents=True)
            self.trainer.save(tmp)
        else:
            link_tree(best, tmp, skip=("extra",))
        _sync(tmp)
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
        _fsync(dst.parent)

    def clear(self) -> None:
        """The checkpoints gone, once the phase's DONE is written."""
        self.wait()
        self._committer.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)

    def record(self) -> dict[str, Any]:
        """What the phase's summary says of its checkpoints: the steps it resumed from, and its minutes of training."""
        return {"resumed": self.resumed, "minutes": round(self.minutes(), 1)}
