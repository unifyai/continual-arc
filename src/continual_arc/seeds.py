"""Deterministic seed derivation shared by schedules and task sources."""

from __future__ import annotations

import hashlib


def derive_seed(*parts: object) -> int:
    """Derive a 63-bit integer seed from arbitrary parts, stable across runs."""
    payload = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
