"""The attempt directory every ``arc-baselines run`` writes beside its run directory.

The benchmark's runner writes the run directory (schedule, configuration, records, summary). The attempt holds what a
run produces around it:

* ``manifest.json``: the run's and the attempt's ids and when the attempt was made;
* ``events.jsonl``: one canonical JSON line per event, appended under a lock and synced: the run's parameters, its
  metrics, every artifact registered, and the model calls the tracking proxy reports
  (:class:`~continual_arc_baselines.tracking.JournalSink`);
* ``artifacts/<sha256>/<name>``: the run directory's files, copied in when the run ends;
* ``costs.jsonl``: the cost journal (:mod:`continual_arc_baselines.costs`);
* ``system/``: the learner's own folder, a harness's state and logs among it
  (:meth:`~continual_arc_baselines.systems.base.SystemLearner.attach`).

A run makes its attempt under ``.workbench/runs/adhoc-<name>-<unix seconds>/attempts/<id>/`` in the folder it is
launched from, unless an outer driver names one it made in ``WORKBENCH_RUN_DIR``; that driver may also list in
``WORKBENCH_REDACT_NAMES`` (a JSON list) environment variables whose values are replaced by ``[REDACTED]`` in every
event.

Metric names (``step`` is the schedule step, zero-based):

* ``instance/cost``, ``instance/solved``, ``instance/wrong_attempts``, ``instance/demo_requests``,
  ``instance/gave_up``, ``instance/exposure_index``
* ``instance/resource/<key>`` for every counter the learner reports
* ``running/mean_cost`` and ``running/score`` over the stream so far
* ``summary/<phase>/<stat>`` and ``summary/resource/<key>/total`` once, at the end, from the run's summary
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = ".workbench"
"""The folder a run makes its attempt under, in the folder it is launched from."""
GIVEN = "WORKBENCH_RUN_DIR"
"""An attempt an outer driver made, which a run writes into instead of making its own."""
REDACTED = "WORKBENCH_REDACT_NAMES"
"""The environment variables whose values no event may hold, as a JSON list of their names."""

_INSTANCE_FIELDS = ("cost", "solved", "wrong_attempts", "demo_requests", "gave_up", "exposure_index")
_SUMMARY_PHASES = ("overall", "stream", "epilogue")
_SUMMARY_STATS = ("n", "mean_cost", "score", "solve_rate", "mean_wrong_attempts", "mean_demo_requests",
                  "give_up_rate", "invalid_rate")
_RUN_FILES = ("schedule.json", "config.json", "meta.json", "records.jsonl", "summary.json", "events.jsonl")


def plain(value: Any) -> Any:
    """``value`` as JSON can hold it: dataclasses as dicts, tuples as lists, paths as strings, keys as strings."""
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value: Any) -> bytes:
    """One line of JSON with sorted keys and no spaces, the form every event and manifest is written in."""
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def slug(value: str) -> str:
    """``value`` as a lowercase kebab-case identifier that starts with a letter."""
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value if value and value[0].isalpha() else "run-" + value


def _atomic_write(path: Path, data: bytes) -> None:
    """``data`` at ``path`` whole or not at all: written beside it, synced, renamed over it, and the rename synced."""
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def locked(path: Path):
    """An exclusive advisory lock on the file ``path`` (made if missing), for writers in any process."""
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Attempt:
    """An attempt directory, opened on one that exists (:meth:`create` makes one), and the metrics of the run that
    writes into it. ``max_cost``, the configuration's whole budget, gives the running score."""

    def __init__(self, directory: str | os.PathLike[str], max_cost: float | None = None):
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.run_id, self.attempt_id = manifest["run_id"], manifest["attempt_id"]
        names = json.loads(os.environ.get(REDACTED, "[]"))
        self._secrets = sorted((os.environ[name] for name in names if os.environ.get(name)), key=len, reverse=True)
        self.max_cost = max_cost
        self._n = 0
        self._total_cost = 0.0

    @classmethod
    def create(cls, name: str, root: str | os.PathLike[str] = ROOT, max_cost: float | None = None) -> Attempt:
        """A new attempt of a new run, ``adhoc-<name>-<unix seconds>``, under ``root``."""
        run_id = slug(f"adhoc-{name}-{int(time.time())}")
        attempt_id = uuid.uuid4().hex
        directory = Path(root).resolve() / "runs" / run_id / "attempts" / attempt_id
        directory.mkdir(parents=True)
        _atomic_write(directory / "manifest.json", canonical({
            "experiment_id": "adhoc", "project_key": "continual-arc-baselines", "recovery": "restart-only",
            "resume_performed": False, "run_id": run_id, "attempt_id": attempt_id,
            "created_at": datetime.now(UTC).isoformat(), "schema_version": 1}))
        return cls(directory, max_cost=max_cost)

    @classmethod
    def from_environment(cls, name: str, max_cost: float | None = None) -> Attempt:
        """The attempt ``WORKBENCH_RUN_DIR`` names, or a new one (:meth:`create`)."""
        given = os.environ.get(GIVEN)
        return cls(given, max_cost=max_cost) if given else cls.create(name, max_cost=max_cost)

    @property
    def given(self) -> bool:
        """Whether an outer driver made this attempt (``WORKBENCH_RUN_DIR``)."""
        return bool(os.environ.get(GIVEN))

    # -- events --------------------------------------------------------------

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self._redact(str(k)): self._redact(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self._redact(v) for v in value]
        return value

    def emit(self, kind: str, payload: dict) -> dict:
        """Append one event to ``events.jsonl``; refused while the log ends in a line cut short, which needs reading
        by hand before anything is added."""
        event = {"event_id": uuid.uuid4().hex, "run_id": self.run_id, "attempt_id": self.attempt_id, "kind": kind,
                 "at": datetime.now(UTC).isoformat(), "payload": self._redact(payload)}
        data = canonical(event) + b"\n"
        with locked(self.directory / ".events.lock"), (self.directory / "events.jsonl").open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell():
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    raise ValueError("events.jsonl ends in a line cut short; recover it before appending")
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def metric(self, name: str, value: float, step: int = 0) -> dict:
        if not name or not math.isfinite(value):
            raise ValueError("a metric needs a name and a finite value")
        if type(step) is not int or step < 0:
            raise ValueError("a metric's step is a non-negative integer")
        return self.emit("metric", {"name": name, "value": value, "step": step})

    def parameters(self, values: dict[str, Any]) -> dict:
        return self.emit("parameters", {"values": values})

    def artifact(self, path: Path) -> dict:
        """A copy of the file ``path`` under ``artifacts/<its sha256>/``, verified after the copy, since the file may
        be changing as it is read."""
        path = Path(path).resolve()
        sha = _sha256(path)
        dest = self.directory / "artifacts" / sha / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            pending = dest.parent / (".pending-" + uuid.uuid4().hex)
            shutil.copyfile(path, pending)
            if _sha256(pending) != sha:
                pending.unlink()
                raise ValueError(f"{path} changed while it was copied")
            os.replace(pending, dest)
        payload = {"name": path.name, "path": str(dest.relative_to(self.directory)), "sha256": sha,
                   "size": dest.stat().st_size}
        self.emit("artifact", payload)
        return payload

    # -- the run's metrics ---------------------------------------------------

    def prime(self, records) -> None:
        """The running mean taken up from the stream records a resumed run already holds."""
        for record in records:
            if record.phase == "stream":
                self._n += 1
                self._total_cost += float(record.cost)

    def progress(self, record, n_steps: int) -> None:
        """The metrics of one record, as :func:`continual_arc.runner.run` reports it (its ``progress``)."""
        step = int(record.step)
        for name in _INSTANCE_FIELDS:
            self.metric(f"instance/{name}", float(getattr(record, name)), step=step)
        for key, value in (record.resources or {}).items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            self.metric(f"instance/resource/{slug(key)}", value, step=step)
        if record.phase == "stream":
            self._n += 1
            self._total_cost += float(record.cost)
            mean = self._total_cost / self._n
            self.metric("running/mean_cost", mean, step=step)
            if self.max_cost:
                self.metric("running/score", 100.0 * (1.0 - mean / self.max_cost), step=step)

    def finish(self, result) -> dict[str, float]:
        """The summary's metrics, and the run directory's files as artifacts; returns the metrics emitted."""
        emitted: dict[str, float] = {}
        summary = result.summary
        for phase in _SUMMARY_PHASES:
            block = summary.get(phase) or {}
            for stat in _SUMMARY_STATS:
                value = block.get(stat)
                if isinstance(value, int | float) and value == value:
                    name = f"summary/{phase}/{stat}"
                    self.metric(name, float(value))
                    emitted[name] = float(value)
        for key, block in (summary.get("resources") or {}).items():
            total = block.get("total")
            if isinstance(total, int | float):
                name = f"summary/resource/{slug(key)}/total"
                self.metric(name, float(total))
                emitted[name] = float(total)
        if result.out_dir is not None:
            for filename in _RUN_FILES:
                path = Path(result.out_dir) / filename
                if path.is_file():
                    self.artifact(path)
        self.emit("continual_arc_summary", {
            "out_dir": str(result.out_dir) if result.out_dir is not None else None,
            "n_records": len(result.records),
            "overall_score": (summary.get("overall") or {}).get("score"),
        })
        return emitted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
