"""lf52 at level 0: peg solitaire on a board of 6-pixel floor tiles. Clicking a peg (ACTION6) that can jump shows
its landing cells, two tiles away across an orthogonally adjacent peg onto empty floor; clicking a landing cell makes
the jump and removes the peg jumped. The level is won when one peg is left; 64 actions lose. Level 0 has five pegs on
an L of floor, and at each of its first three jumps one of the two jumps open leads to a dead end.

The game ignores its Level and reads its board from the module table ``kciatvszkc["grid1"]``, so a level here is
level 0's placeholder Level carrying a fresh board as ``module_patch``, of level 0's tiles (floor, and pegs on
floor). A board is two or three rectangles of floor joined into one shape, placed anywhere on the 8 x 9 tiles the
screen shows, and four to six pegs (level 0: five) put on it by playing the game backwards from one peg on a random
cell, so it is solvable by construction. A board is kept when, as in level 0, its first move is a choice and a run of
random jumps wins about as often as on level 0 (an eighth of the time). The game's tutorial cues for level 0 are keyed
to level 0's cells (a dead-end flag on landing at (0, 2), (2, 2) or (5, 1), and a ring round (1, 2), its first peg, on
a click that misses every peg while five are left), so those four cells are kept off every board, where the cues
would mislead; this also keeps level 0's own board from being drawn.

solve is a depth-first search over sets of pegs, jumps tried in a fixed order and dead sets remembered; it returns two
clicks a jump (the peg, then its landing cell), checked in the real engine."""
from __future__ import annotations

import random
from functools import cache

from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "lf52"
TABLE, ENTRY = "kciatvszkc", "grid1"
PEG, FLOOR, EMPTY = "x", ".", " "
TILE = 6
ORIGIN = (10, 5)  # the game places level 0's board with its tile (0, 0) at this pixel
COLS, ROWS = 8, 9  # the tiles of a board at ORIGIN that the 64 x 64 screen shows whole
# Cells level 0's tutorial cues name: the game flags a jump landing on (0, 2), (2, 2) or (5, 1) as a dead end, and a
# click off every peg while five are left rings (1, 2), level 0's first peg to move. Boards leave them off.
CUE_CELLS = frozenset({(0, 2), (2, 2), (5, 1), (1, 2)})
STEPS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # the game's own order of jump directions
PEGS = (4, 5, 6)
WIN_SHARE = (1 / 12, 3 / 16)  # how often random jumps win: level 0's eighth, within a factor of 1.5
FLOOR_CELLS = (15, 40)  # floor cells a board has (level 0: 33)
TRIES = 400

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_BASE = getattr(_MODULE, TABLE)[ENTRY]
_BASE_ROWS = tuple(_BASE.ojilieuwrah)  # row y of the board, y = 0 at the top


class Board:
    """A board in the shape the game reads from its table (its rows, legend, tile size and merged tile groups, as
    level 0's), whose repr is its rows, so that a level's layout key reads the board itself."""

    def __init__(self, rows: list[str]):
        self.ojilieuwrah = list(rows)
        self.vyebgdxovnm = {k: list(v) for k, v in _BASE.vyebgdxovnm.items()}
        self.tuxqgponvlq = dict(_BASE.tuxqgponvlq)
        self.qgfifoygrvi = {k: [list(g) for g in v] for k, v in _BASE.qgfifoygrvi.items()}

    @property
    def tile_size(self) -> tuple[int, int]:
        return self.tuxqgponvlq["tile_size"]

    @property
    def hiaiflnslto(self) -> int:
        return max(len(r) for r in self.ojilieuwrah)

    @property
    def jlkfqcwrspr(self) -> int:
        return len(self.ojilieuwrah)

    @property
    def image_groups(self) -> list[list[str]]:
        return self.qgfifoygrvi["image_groups"]

    def __repr__(self) -> str:
        return "Board(" + "|".join(self.ojilieuwrah) + ")"


def _rows(level) -> tuple[str, ...]:
    """The board a level plays: its patched table's, or level 0's own."""
    patch = getattr(level, "module_patch", {})
    return tuple(patch[TABLE][ENTRY].ojilieuwrah) if TABLE in patch else _BASE_ROWS


def _parse(rows: tuple[str, ...]) -> tuple[frozenset, frozenset]:
    """(floor cells, peg cells) of a board; a peg stands on floor."""
    cells = {(x, y): c for y, r in enumerate(rows) for x, c in enumerate(r)}
    return (frozenset(p for p, c in cells.items() if c in (FLOOR, PEG)),
            frozenset(p for p, c in cells.items() if c == PEG))


def _jumps(floor: frozenset, pegs: frozenset) -> list[tuple]:
    """Every jump (from, over, to) open with ``pegs`` on ``floor``, pegs in reading order, directions in the game's."""
    out = []
    for x, y in sorted(pegs, key=lambda p: (p[1], p[0])):
        for dx, dy in STEPS:
            over, to = (x + dx, y + dy), (x + 2 * dx, y + 2 * dy)
            if over in pegs and to in floor and to not in pegs:
                out.append(((x, y), over, to))
    return out


def _after(pegs: frozenset, jump: tuple) -> frozenset:
    start, over, to = jump
    return pegs - {start, over} | {to}


def _solution(floor: frozenset, pegs: frozenset) -> list[tuple] | None:
    """The first jump sequence, in the fixed order, that leaves one peg."""

    @cache
    def search(state: frozenset) -> tuple | None:
        if len(state) == 1:
            return ()
        for jump in _jumps(floor, state):
            rest = search(_after(state, jump))
            if rest is not None:
                return (jump, *rest)
        return None

    found = search(pegs)
    return None if found is None else list(found)


def _win_share(floor: frozenset, pegs: frozenset) -> float:
    """How often jumps chosen at random among the open ones win."""

    @cache
    def share(state: frozenset) -> float:
        if len(state) == 1:
            return 1.0
        moves = _jumps(floor, state)
        return sum(share(_after(state, j)) for j in moves) / len(moves) if moves else 0.0

    return share(pegs)


def _shape(rng: random.Random) -> frozenset | None:
    """Two or three rectangles of floor, each joined to the one before along an edge, within the screen's tiles and
    clear of the cue cells."""
    cells: set[tuple[int, int]] = set()
    for i in range(rng.choice([2, 2, 3])):
        w, h = rng.randint(2, 7), rng.randint(1, 4)
        if rng.random() < 0.5:
            w, h = h, w
        if w > COLS or h > ROWS:
            return None
        for _ in range(30):
            x0, y0 = rng.randint(0, COLS - w), rng.randint(0, ROWS - h)
            rect = {(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)}
            if rect & CUE_CELLS or rect <= cells:
                continue
            touches = any((x + dx, y + dy) in cells for x, y in rect for dx, dy in STEPS)
            if i == 0 or touches:
                cells |= rect
                break
        else:
            return None
    return frozenset(cells) if FLOOR_CELLS[0] <= len(cells) <= FLOOR_CELLS[1] else None


def _backwards(rng: random.Random, floor: frozenset, count: int) -> frozenset | None:
    """``count`` pegs from one on a random cell, by undoing random jumps: a peg at ``to`` goes back to ``start`` and
    puts back the peg it jumped at ``over``."""
    pegs = frozenset({rng.choice(sorted(floor))})
    while len(pegs) < count:
        undo = [(to, (to[0] - dx, to[1] - dy), (to[0] - 2 * dx, to[1] - 2 * dy))
                for to in sorted(pegs) for dx, dy in STEPS]
        undo = [(to, over, start) for to, over, start in undo
                if over in floor and start in floor and over not in pegs and start not in pegs]
        if not undo:
            return None
        to, over, start = rng.choice(undo)
        pegs = pegs - {to} | {over, start}
    return pegs


def _board_rows(floor: frozenset, pegs: frozenset) -> tuple[str, ...]:
    height = max(y for _, y in floor) + 1
    rows = []
    for y in range(height):
        xs = [x for x, yy in floor if yy == y]
        width = max(xs) + 1 if xs else 0
        rows.append("".join(PEG if (x, y) in pegs else FLOOR if (x, y) in floor else EMPTY for x in range(width)))
    return tuple(rows)


def _level(rows: tuple[str, ...]):
    level = _LEVEL0.clone()
    level.module_patch = {TABLE: {ENTRY: Board(list(rows))}}
    return level


def generate(rng: random.Random):
    count = rng.choice(PEGS)
    for _ in range(TRIES):
        floor = _shape(rng)
        if floor is None:
            continue
        pegs = _backwards(rng, floor, count)
        if pegs is None or len(_jumps(floor, pegs)) < 2:
            continue
        if not WIN_SHARE[0] <= _win_share(floor, pegs) <= WIN_SHARE[1]:
            continue
        rows = _board_rows(floor, pegs)
        if rows != _BASE_ROWS:
            return _level(rows)
    return None


def _click(cell: tuple[int, int]) -> ActionInput:
    return ActionInput(id=GameAction.ACTION6, data={"x": ORIGIN[0] + TILE * cell[0] + 3, "y": ORIGIN[1] + TILE * cell[1] + 3})


def solve(level) -> list[ActionInput] | None:
    rows = _rows(level)
    jumps = _solution(*_parse(rows))
    if jumps is None:
        return None
    actions = [_click(cell) for start, _, to in jumps for cell in (start, to)]
    return actions if levels.replay(GAME, level, actions) else None
