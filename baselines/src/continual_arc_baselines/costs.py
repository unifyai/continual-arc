"""The cost journal: ``costs.jsonl`` in a run's attempt, every observation of every model call the tracking proxy made.

Each call is observed twice under one ``request_attempt_id``, its id at the proxy: ``request_started`` as it is
forwarded, then ``response`` or ``transport_error`` as it ends (:class:`~continual_arc_baselines.tracking.JournalSink`).
Lines are appended under a lock and synced, never edited. Charges are decimal strings in the provider's credits (USD for
OpenRouter), and what the provider did not report stays null: nothing is estimated.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path

TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")
COST_FIELDS = ("account_charge", "upstream_inference_cost")
_RANKS = {"request_started": 0, "transport_error": 1, "response": 2}


class JournalCorruptionError(ValueError):
    """The journal ends in a line cut short or holds one that is not a record: it needs reading by hand, since nothing
    is truncated or skipped on its own."""


def _money(value) -> str | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"a charge must be a decimal, not {value!r}") from None
    if not result.is_finite() or result < 0:
        raise ValueError(f"a charge must be finite and not negative, not {value!r}")
    return format(result, "f")


def _count(value) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"a token count must be a non-negative integer or null, not {value!r}")
    return value


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CostJournal:
    """The journal in ``directory``, safe to write from any thread or process on a local filesystem."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "costs.jsonl"

    @contextmanager
    def _locked(self):
        fd = os.open(self.directory / ".costs.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def record(self, record: dict) -> dict:
        """Append one observation, its charges as decimal strings and its tokens as counts or null, identified by the
        sha256 of its own canonical form (``source_id``); returns it as written."""
        value = dict(record)
        for field in COST_FIELDS:
            value[field] = _money(value.get(field))
        value["tokens"] = {k: _count(value.get("tokens", {}).get(k)) for k in TOKEN_FIELDS}
        if not value.get("provider") or not value.get("origin"):
            raise ValueError("an observation needs its provider and its origin")
        value.setdefault("source_id", hashlib.sha256(_canonical(value).encode()).hexdigest())
        line = _canonical(value).encode() + b"\n"
        with self._locked():
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a+b") as handle:
                handle.seek(0, 2)
                if handle.tell():
                    handle.seek(-1, 2)
                    if handle.read(1) != b"\n":
                        raise JournalCorruptionError(f"{self.path} ends in a line cut short")
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return value

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self._locked(), self.path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                if not line.endswith(b"\n"):
                    raise JournalCorruptionError(f"{self.path} ends in a line cut short at line {number}")
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    record = None
                if not isinstance(record, dict) or not record.get("source_id"):
                    raise JournalCorruptionError(f"{self.path} line {number} is not a record")
                out.append(record)
        return out

    def summary(self) -> dict:
        """The journal as one entity per call, each call's later observations filling in what its earlier ones left
        null: how many observations and calls, and for each token field and each charge the known total and how many
        calls left it unknown."""
        calls: dict[tuple[str, str], dict] = {}
        records = self.records()
        for row in sorted(records, key=lambda r: _RANKS.get(r["origin"], 1)):
            call = calls.setdefault((row["provider"], row.get("request_attempt_id") or row["source_id"]), {
                "currency": row.get("currency"), "unit": row.get("unit"), "tokens": dict.fromkeys(TOKEN_FIELDS),
                **dict.fromkeys(COST_FIELDS)})
            for field in COST_FIELDS:
                if row.get(field) is not None:
                    call[field] = row[field]
            for field in TOKEN_FIELDS:
                if row["tokens"].get(field) is not None:
                    call["tokens"][field] = row["tokens"][field]
        entities = list(calls.values())
        charges = []
        for currency, unit in sorted({(e["currency"], e["unit"]) for e in entities}, key=str):
            subset = [e for e in entities if (e["currency"], e["unit"]) == (currency, unit)]
            charge = {"currency": currency, "unit": unit}
            for field in COST_FIELDS:
                total = _total([None if e[field] is None else Decimal(e[field]) for e in subset], Decimal(0))
                charge[field] = {**total, "known_total": format(total["known_total"], "f")}
            charges.append(charge)
        return {"records": len(records), "entities": len(entities),
                "tokens": {field: _total([e["tokens"][field] for e in entities], 0) for field in TOKEN_FIELDS},
                "charges": charges}


def _total(values: list, zero):
    known = [v for v in values if v is not None]
    return {"known_total": sum(known, zero), "unknown_entities": len(values) - len(known)}
