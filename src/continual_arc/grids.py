"""Grid representation and validation.

Grids are plain ``list[list[int]]`` with values in 0..9 and dimensions in 1..30,
matching the ARC-AGI-1 format. Re-ARC produces tuples of tuples; ``normalize_grid``
accepts either (and anything with a ``tolist()`` method, e.g. numpy arrays).
"""

from __future__ import annotations

import hashlib
from typing import Any

Grid = list[list[int]]

MAX_DIM = 30
MIN_DIM = 1
NUM_COLOURS = 10


class InvalidGrid(ValueError):
    """Raised when a value cannot be interpreted as a valid ARC grid."""


def normalize_grid(value: Any) -> Grid:
    """Coerce ``value`` to a validated ``list[list[int]]``.

    Raises :class:`InvalidGrid` if the value is not a rectangular grid of ints in
    ``0..9`` with each dimension in ``1..30``.
    """
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise InvalidGrid(f"grid must be a list of rows, got {type(value).__name__}")
    rows: Grid = []
    for row in value:
        if hasattr(row, "tolist") and not isinstance(row, (list, tuple)):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise InvalidGrid(f"row must be a list, got {type(row).__name__}")
        cells: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                # bool is a subclass of int; reject it explicitly.
                try:
                    if isinstance(cell, bool):
                        raise TypeError
                    if float(cell) != int(cell):
                        raise TypeError
                    cell = int(cell)
                except (TypeError, ValueError):
                    raise InvalidGrid(f"cell must be an int, got {cell!r}") from None
            if not 0 <= cell < NUM_COLOURS:
                raise InvalidGrid(f"cell value {cell} out of range 0..{NUM_COLOURS - 1}")
            cells.append(int(cell))
        rows.append(cells)
    if not MIN_DIM <= len(rows) <= MAX_DIM:
        raise InvalidGrid(f"grid height {len(rows)} not in {MIN_DIM}..{MAX_DIM}")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise InvalidGrid("grid rows have differing widths")
    width = widths.pop()
    if not MIN_DIM <= width <= MAX_DIM:
        raise InvalidGrid(f"grid width {width} not in {MIN_DIM}..{MAX_DIM}")
    return rows


def is_valid_grid(value: Any) -> bool:
    try:
        normalize_grid(value)
    except InvalidGrid:
        return False
    return True


def grids_equal(a: Any, b: Any) -> bool:
    """Exact equality after normalisation. Invalid inputs compare unequal."""
    try:
        return normalize_grid(a) == normalize_grid(b)
    except InvalidGrid:
        return False


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def grid_hash(grid: Grid) -> str:
    """Stable content hash, used to detect duplicate instances."""
    payload = ";".join(",".join(str(c) for c in row) for row in grid)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def grid_to_text(grid: Grid) -> str:
    """Compact single-digit-per-cell rendering, one row per line."""
    return "\n".join("".join(str(c) for c in row) for row in grid)
