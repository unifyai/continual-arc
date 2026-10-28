"""bp35 at level 0: a player stands at the top of a walled shaft of 6-pixel tiles with a gem at its floor. ACTION3
and ACTION4 step it left or right; stepping off an edge drops it straight down to the next solid tile, and clicking a
breakable block (ACTION6) removes it, dropping the player through when the block is the one beneath it. Reaching the
gem, by a step or a fall, wins; the spiked ceiling above sinks a row on each step without a fall that is an
even-numbered action, and its reaching the player's row, or 64 actions, loses.

The game ignores its Level and reads its board from the module table ``tjdtolkmxo["grid1"]``, so a level here is
level 0's placeholder Level carrying a fresh board as ``module_patch``. A board keeps level 0's frame (11 tiles wide,
two-tile side walls, the ceiling seven rows above the player's start under five rows of slab, a solid floor) and
level 0's tiles (wall, block, gem, player, ceiling, slab), and draws the rest anew: the player's row and column, the
gem's row and column, and strata two to four rows apart between them, each a wall with holes and block runs, a wall
whose only way through is a block run (sometimes two blocks thick), a floating block bar or a wall shelf. Every
stratum can be passed from anywhere above it, so, as in level 0, no fall leads to a dead end. A draw aims at a
solution length round level 0's and keeps a board that the search below wins within a step of it, breaking at least
three blocks and dropping at least three times (level 0: five of each); level 0's own board is never kept.

solve is a breadth-first search over (player tile, broken blocks at or below it, ceiling depth, action parity), with
the steps, drops, clicks and ceiling read from the game's code (:class:`_Rules`); it returns the shortest win in a few
milliseconds, and its answer is checked in the real engine."""
from __future__ import annotations

import random

from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "bp35"
TABLE, ENTRY = "tjdtolkmxo", "grid1"
WALL, BLOCK, GEM, PLAYER, CEILING, SLAB, EMPTY = "o", "x", "+", "n", "m", "w", " "
TILE = 6
CEILING_GAP = 7  # rows from the player's start up to the ceiling, as in level 0
MAX_ACTIONS = 63  # the 64th action loses
MAX_STATES = 200_000
GENERATE_STATES = 20_000
TRIES = 1000

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_BASE = getattr(_MODULE, TABLE)[ENTRY]
_BASE_ROWS = tuple(_BASE.mycafmfscyz)  # row y of the board, y = 0 at the floor; the player falls towards y = 0
WIDTH = _BASE.width
SLAB_ROWS = 5  # level 0's slab above the ceiling
INTERIOR = range(2, WIDTH - 2)


class Board:
    """A board in the shape the game reads from its table (its rows, legend, tile size and merged tile groups, as
    level 0's), whose repr is its rows, so that a level's layout key reads the board itself."""

    def __init__(self, rows: list[str]):
        self.mycafmfscyz = list(rows)
        self.rfbmxxrnrwi = {k: list(v) for k, v in _BASE.rfbmxxrnrwi.items()}
        self.ymorirjinte = dict(_BASE.ymorirjinte)
        self.ismbglrrplx = {k: [list(g) for g in v] for k, v in _BASE.ismbglrrplx.items()}

    @property
    def qmcjluiioz(self) -> tuple[int, int]:
        return self.ymorirjinte["vxruyoesvkf"]

    @property
    def width(self) -> int:
        return max(len(r) for r in self.mycafmfscyz)

    @property
    def height(self) -> int:
        return len(self.mycafmfscyz)

    @property
    def lcsducsxuc(self) -> list[list[str]]:
        return self.ismbglrrplx["jibupgvgfzf"]

    def __repr__(self) -> str:
        return "Board(" + "|".join(self.mycafmfscyz) + ")"


def _rows(level) -> tuple[str, ...]:
    """The board a level plays: its patched table's, or level 0's own."""
    patch = getattr(level, "module_patch", {})
    return tuple(patch[TABLE][ENTRY].mycafmfscyz) if TABLE in patch else _BASE_ROWS


class _Rules:
    """bp35's rules at level 0 over an abstract state (player x, player y, broken blocks at or below the player as a
    bit mask, rows the ceiling has sunk, action parity), read from the game's code: a step into a free tile drops the
    player to the next solid tile below it, a click removes a block (and drops the player through it when it is the
    block beneath), and each second action that is a step without a drop sinks the ceiling a row, losing if it
    reaches the player's row. An idle click on a wall spends an action to shift that parity. States whose ceiling
    has come down to the player's row are dropped, as the game loses them next. Clicks are tried only on the blocks
    the player touches, the one beneath it and those beside it in its row: a click costs one action wherever the
    player stands, so breaking a block when the player reaches it is never longer than breaking it ahead, and the
    search stays small."""

    def __init__(self, rows: tuple[str, ...]):
        self.blocks = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == BLOCK]
        self.index = {b: i for i, b in enumerate(self.blocks)}
        (self.gem,) = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == GEM]
        (player,) = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == PLAYER]
        (self.ceiling,) = [y for y, r in enumerate(rows) if CEILING in r]
        self.solid = {(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c in (WALL, SLAB, CEILING)}
        # Blocks at or below each row: a block above the player is never reached again, so it leaves the state.
        self.below = [sum(1 << i for i, b in enumerate(self.blocks) if b[1] <= y) for y in range(len(rows))]
        self.start = (player[0], player[1], 0, 0, 0)

    def free(self, x: int, y: int, mask: int) -> bool:
        if (x, y) in self.solid or (x, y) == self.gem:
            return False
        return (x, y) not in self.index or bool(mask >> self.index[(x, y)] & 1)

    def drop(self, x: int, y: int, mask: int) -> tuple[int, bool]:
        """Where a player in the free tile (x, y) comes to rest, and whether that is on the gem."""
        while self.free(x, y - 1, mask):
            y -= 1
        return y, (x, y - 1) == self.gem

    def successors(self, state: tuple) -> list[tuple[tuple, tuple | str]]:
        """Each step from ``state`` and the state it leads to, or "win"."""
        px, py, mask, sunk, parity = state
        odd = parity ^ 1
        out: list[tuple[tuple, tuple | str]] = []
        for dx in (-1, 1):
            tx = px + dx
            if (tx, py) == self.gem:
                out.append((("move", dx), "win"))
                continue
            nx = px
            if self.free(tx, py, mask):
                ry, on_gem = self.drop(tx, py, mask)
                if on_gem:
                    out.append((("move", dx), "win"))
                    continue
                if ry < py:
                    out.append((("move", dx), (tx, ry, mask & self.below[ry], sunk, odd)))
                    continue
                nx = tx
            if odd:
                out.append((("move", dx), (nx, py, mask, sunk, odd)))
            elif py < self.ceiling - sunk - 1:  # an even step without a drop sinks the ceiling, not onto the player
                out.append((("move", dx), (nx, py, mask, sunk + 1, odd)))
        for b in ((px, py - 1), (px - 1, py), (px + 1, py)):
            if b not in self.index or mask >> self.index[b] & 1:
                continue
            broken = mask | 1 << self.index[b]
            if b == (px, py - 1):
                ry, on_gem = self.drop(px, py - 1, broken)
                out.append((("click", b), "win" if on_gem else (px, ry, broken & self.below[ry], sunk, odd)))
            else:
                out.append((("click", b), (px, py, broken, sunk, odd)))
        out.append((("idle",), (px, py, mask, sunk, odd)))
        return out


def _search(rows: tuple[str, ...], max_states: int) -> list[tuple] | None:
    """Shortest list of steps ("move", -1 | 1), ("click", (x, y)) and ("idle",) that wins the board within the game's
    63 actions, by breadth-first search over :class:`_Rules`; None when ``max_states`` expansions find none."""
    rules = _Rules(rows)
    parent = {rules.start: None}
    frontier, expanded = [rules.start], 0
    for _ in range(MAX_ACTIONS):
        nxt_frontier = []
        for state in frontier:
            if expanded >= max_states:
                return None
            expanded += 1
            for step, nxt in rules.successors(state):
                if nxt == "win":
                    path = [step]
                    while parent[state] is not None:
                        state, prior = parent[state]
                        path.append(prior)
                    return path[::-1]
                if nxt not in parent:
                    parent[nxt] = (state, step)
                    nxt_frontier.append(nxt)
        frontier = nxt_frontier
    return None


def _trace(rows: tuple[str, ...], path: list[tuple]) -> list[tuple[int, int]]:
    """The player's tile before each step of a search path, and after the last."""
    (px, py) = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == PLAYER][0]
    removed: set[tuple[int, int]] = set()

    def free(x: int, y: int) -> bool:
        return rows[y][x] in (EMPTY, PLAYER) or (rows[y][x] == BLOCK and (x, y) in removed)

    places = [(px, py)]
    for step in path:
        if step[0] == "move" and free(px + step[1], py):
            px += step[1]
            while free(px, py - 1):
                py -= 1
        elif step[0] == "click":
            removed.add(step[1])
            if step[1] == (px, py - 1):
                while free(px, py - 1):
                    py -= 1
        places.append((px, py))
    return places


def _actions(rows: tuple[str, ...], path: list[tuple]) -> list[ActionInput]:
    """The engine's actions for a search path. The camera keeps the player's row at screen rows 36-41 (it is placed
    at the player's row times 6, less 36, at the start and after every drop), so a tile (x, y) is clicked at screen
    (6x + 3, 6(y - player row) + 39); an idle click lands on the left wall in the player's row."""
    actions = []
    for step, (_, py) in zip(path, _trace(rows, path), strict=False):
        if step[0] == "move":
            actions.append(ActionInput(id=GameAction.ACTION3 if step[1] < 0 else GameAction.ACTION4))
        elif step[0] == "click":
            bx, by = step[1]
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": TILE * bx + 3, "y": TILE * (by - py) + 39}))
        else:
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": 3, "y": 39}))
    return actions


def _stratum(rng: random.Random, kind: str) -> list[str]:
    """One stratum's row across the interior."""
    n = len(INTERIOR)
    if kind == "bar":  # a floating block bar in open space, perhaps with a short wall beside it
        row = [EMPTY] * n
        length = rng.randint(2, 4)
        at = rng.randint(0, n - length)
        row[at:at + length] = [BLOCK] * length
        if rng.random() < 0.4:
            side = rng.choice([i for i in range(n) if not at - 1 <= i <= at + length])
            row[side] = WALL
        return row
    if kind == "shelf":  # a wall run from one side, the rest open or blocks
        row = [EMPTY] * n
        length = rng.randint(3, 5)
        if rng.random() < 0.5:
            row[:length] = [WALL] * length
        else:
            row[n - length:] = [WALL] * length
        return row
    row = [WALL] * n
    if kind == "holes":  # a wall with one or two holes, and perhaps a block run
        for i in rng.sample(range(n), rng.choice([1, 1, 2])):
            row[i] = EMPTY
        if rng.random() < 0.5:
            length = rng.randint(1, 3)
            at = rng.randint(0, n - length)
            row[at:at + length] = [BLOCK if c == WALL else c for c in row[at:at + length]]
        return row
    length = rng.randint(2, 4)  # "floor": a wall whose only way through is a block run
    at = rng.randint(0, n - length)
    row[at:at + length] = [BLOCK] * length
    return row


def _draw(rng: random.Random) -> list[list[str]]:
    """A board's tiles, grid[y][x] with y = 0 at the floor, in level 0's frame: side walls two tiles thick, solid
    rows under the gem's, open rows up to the ceiling seven rows above the player, and five rows of slab above it."""
    start = rng.randint(21, 24)
    ceiling = start + CEILING_GAP
    floor = rng.randint(6, 8)  # the gem's row; the two rows above it are open
    grid = [[WALL] * WIDTH for _ in range(ceiling + 1 + SLAB_ROWS)]
    for y in range(ceiling + 1, len(grid)):
        grid[y] = [SLAB] * WIDTH
    grid[ceiling] = [CEILING] * WIDTH
    for y in range(floor, ceiling):
        for x in INTERIOR:
            grid[y][x] = EMPTY
    grid[floor][rng.choice(INTERIOR)] = GEM
    # Strata from the player's own floor down to the gem room's roof, two to four rows apart.
    rows = [start - 1]
    while rows[-1] - (floor + 2) >= 5:
        rows.append(rows[-1] - rng.choice([d for d in (2, 3, 3, 4) if rows[-1] - d - (floor + 2) >= 2]))
    rows.append(floor + 2)
    for i, y in enumerate(rows):
        if i == 0:
            kind = rng.choice(["holes", "holes", "floor", "shelf"])
        elif i == len(rows) - 1:
            kind = rng.choice(["holes", "floor", "floor"])
        else:
            kind = rng.choice(["holes", "floor", "bar", "bar", "shelf"])
        for x, c in zip(INTERIOR, _stratum(rng, kind), strict=True):
            grid[y][x] = c
        if kind == "floor" and 0 < i < len(rows) - 1 and rng.random() < 0.4:  # a thicker block plug, as level 0's
            for x in INTERIOR:
                if grid[y][x] == BLOCK and grid[y + 1][x] == EMPTY and y + 1 < rows[i - 1]:
                    grid[y + 1][x] = BLOCK
    stand = [x for x in INTERIOR if grid[start - 1][x] in (WALL, BLOCK)]
    grid[start][rng.choice(stand)] = PLAYER
    return grid


def _board_rows(grid: list[list[str]]) -> tuple[str, ...]:
    return tuple("".join(r) for r in grid)


_LEVEL0_PATH = _search(_BASE_ROWS, MAX_STATES)
# Solution lengths a draw aims at, about level 0's either way (15: 11 to 20); a draw takes a target in this range and
# keeps the first board whose solution is within a step of it, so the kept lengths center on level 0's, where raw
# boards would mostly be shorter.
LENGTH = (round(len(_LEVEL0_PATH) / 1.35), round(len(_LEVEL0_PATH) * 1.35))
WINDOW = 1
MIN_CLICKS = MIN_DROPS = 3  # level 0's solution breaks five blocks and drops five times


def _level(rows: tuple[str, ...]):
    level = _LEVEL0.clone()
    level.module_patch = {TABLE: {ENTRY: Board(list(rows))}}
    return level


def generate(rng: random.Random):
    target = rng.randint(*LENGTH)
    for _ in range(TRIES):
        rows = _board_rows(_draw(rng))
        if rows == _BASE_ROWS:
            continue
        path = _search(rows, GENERATE_STATES)
        if path is None or abs(len(path) - target) > WINDOW:
            continue
        places = _trace(rows, path)
        clicks = sum(1 for step in path if step[0] == "click")
        drops = sum(1 for a, b in zip(places, places[1:], strict=False) if b[1] < a[1])
        if clicks >= MIN_CLICKS and drops >= MIN_DROPS:
            return _level(rows)
    return None


def solve(level) -> list[ActionInput] | None:
    rows = _rows(level)
    path = _search(rows, MAX_STATES)
    if path is None:
        return None
    actions = _actions(rows, path)
    return actions if levels.replay(GAME, level, actions) else None
