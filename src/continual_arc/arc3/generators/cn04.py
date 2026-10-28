"""cn04 at level 0: two pieces on a 20 x 20 board each carry two connector cells, one of color 8 and one of 13, both
drawn as 8. A click selects a piece (the one nearest the corner starts selected), ACTION1-4 slide it a cell within the
board and ACTION5 turns it a quarter clockwise, pushed back inside the board; the level is won when each connector lies
on the other piece's connector of its own color, and the MaxSteps-th action (75 at level 0) loses.

A layout places both pieces anew, each at any quarter turn anywhere on the board, their boxes apart. It is kept only
when the search below solves it in 10 to 17 actions (level 0 takes 13), a cheap one-piece plan screening draws first;
MaxSteps scales with that length, so the slack over the shortest solution is level 0's.

solve is a breadth-first search over an abstract state read from the game's code (which piece is selected, each
piece's place and turn), with the game's win rule; a solution of 10 to 17 actions takes about 0.1 to 0.6 s."""
from __future__ import annotations

import random
from collections import deque

import numpy as np

from continual_arc.arc3 import levels

GAME = "cn04"
LENGTH = (10, 17)  # shortest solution lengths kept; level 0's is 13
MAX_STATES = 400_000

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_NAMES = [s.name for s in _LEVEL0.get_sprites()]
_W, _H = _LEVEL0.grid_size


def _shape(name: str, rotation: int) -> tuple[np.ndarray, frozenset]:
    """The piece as drawn at ``rotation`` (the engine's clockwise quarter turns) and its connectors (x, y, color)."""
    pixels = np.rot90(_MODULE.sprites[name].pixels, k=(-rotation % 360) // 90)
    return pixels, frozenset((x, y, int(pixels[y, x])) for y, x in zip(*np.nonzero((pixels == 8) | (pixels == 13)), strict=False))


_SHAPES = {(n, r): _shape(n, r) for n in _NAMES for r in (0, 90, 180, 270)}


def _connectors(name: str, x: int, y: int, r: int) -> frozenset:
    return frozenset((x + cx, y + cy, c) for cx, cy, c in _SHAPES[name, r][1])


def _won(pieces: tuple) -> bool:
    """Every connector shares its cell with the other piece's connector of its color (the game's check: each such
    cell holds exactly two same-colored connectors; each piece has one connector of each color)."""
    (ax, ay, ar), (bx, by, br) = pieces
    return _connectors(_NAMES[0], ax, ay, ar) == _connectors(_NAMES[1], bx, by, br)


def _owner(pieces: tuple, gx: int, gy: int) -> int | None:
    """The piece a click on grid cell (gx, gy) selects: the first in the level's order drawn there."""
    for i, (name, (x, y, r)) in enumerate(zip(_NAMES, pieces, strict=False)):
        pixels = _SHAPES[name, r][0]
        if 0 <= gx - x < pixels.shape[1] and 0 <= gy - y < pixels.shape[0] and pixels[gy - y, gx - x] >= 0:
            return i
    return None


def _click_cell(pieces: tuple, i: int) -> tuple[int, int]:
    """A cell whose click selects piece ``i``. One always exists: neither piece's drawn cells fit inside the other's
    (the second piece's seven-cell bar is longer than the first piece is wide or tall)."""
    x, y, r = pieces[i]
    pixels = _SHAPES[_NAMES[i], r][0]
    (cell,) = [(x + int(px), y + int(py)) for py, px in zip(*np.nonzero(pixels >= 0), strict=False)
               if _owner(pieces, x + px, y + py) == i][:1]
    return cell


def _start(level) -> tuple[int, tuple]:
    by_name = {s.name: s for s in level.get_sprites()}
    pieces = tuple((by_name[n].x, by_name[n].y, by_name[n].rotation) for n in _NAMES)
    selected = min(range(len(pieces)), key=lambda i: pieces[i][0] ** 2 + pieces[i][1] ** 2)
    return selected, pieces


def _turned(name: str, x: int, y: int, r: int) -> tuple[int, int, int]:
    """A quarter turn clockwise, pushed back inside the board as the game does."""
    r = (r + 90) % 360
    pixels = _SHAPES[name, r][0]
    return max(0, min(x, _W - pixels.shape[1])), max(0, min(y, _H - pixels.shape[0])), r


def _fits(name: str, x: int, y: int, r: int) -> bool:
    pixels = _SHAPES[name, r][0]
    return x >= 0 and y >= 0 and x + pixels.shape[1] <= _W and y + pixels.shape[0] <= _H


_MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _search(selected: int, pieces: tuple, limit: int) -> list[tuple] | None:
    """Shortest list of steps, ("move", action id), ("turn",) or ("select", piece), that wins in fewer than
    ``limit`` actions; None when none does or MAX_STATES states are expanded first."""
    start = (selected, pieces)
    prev = {start: None}
    queue = deque([(start, 0)])
    expanded = 0
    while queue and expanded < MAX_STATES:
        state, depth = queue.popleft()
        expanded += 1
        if depth + 1 >= limit:
            continue
        sel, pcs = state
        name, (x, y, r) = _NAMES[sel], pcs[sel]
        nexts = [(("move", a), (x + dx, y + dy, r)) for a, (dx, dy) in _MOVES.items() if _fits(name, x + dx, y + dy, r)]
        nexts.append((("turn",), _turned(name, x, y, r)))
        for step, piece in nexts:
            nxt = (sel, pcs[:sel] + (piece,) + pcs[sel + 1:])
            if nxt in prev:
                continue
            prev[nxt] = (state, step)
            if _won(nxt[1]):
                path = []
                while prev[nxt] is not None:
                    nxt, step = prev[nxt]
                    path.append(step if step[0] != "select" else ("select", *_click_cell(nxt[1], step[1])))
                return path[::-1]
            queue.append((nxt, depth + 1))
        for i in range(len(pcs)):
            nxt = (i, pcs)
            if i != sel and nxt not in prev:
                prev[nxt] = (state, ("select", i))
                queue.append((nxt, depth + 1))
    return None


def _estimate(selected: int, pieces: tuple) -> int | None:
    """The length of the best plan that turns and slides one piece onto the other where it lies: an upper bound on
    the shortest solution, and usually equal to it."""
    best = None
    for m, o in ((0, 1), (1, 0)):
        target = _connectors(_NAMES[o], *pieces[o])
        x, y, r = pieces[m]
        for turns in range(4):
            (cx, cy, c), *_ = sorted(_SHAPES[_NAMES[m], r][1])
            (tx, ty, _), = [(px - cx, py - cy, pc) for px, py, pc in target if pc == c]
            if _fits(_NAMES[m], tx, ty, r) and _connectors(_NAMES[m], tx, ty, r) == target:
                cost = (m != selected) + turns + abs(tx - x) + abs(ty - y)
                best = cost if best is None else min(best, cost)
            x, y, r = _turned(_NAMES[m], x, y, r)
    return best


def _actions(steps: list[tuple]) -> list:
    from arcengine import ActionInput, GameAction

    scale = min(64 // _W, 64 // _H)
    ox, oy = (64 - _W * scale) // 2, (64 - _H * scale) // 2
    out = []
    for step in steps:
        if step[0] == "move":
            out.append(ActionInput(id=GameAction.from_id(step[1])))
        elif step[0] == "turn":
            out.append(ActionInput(id=GameAction.ACTION5))
        else:
            out.append(ActionInput(id=GameAction.ACTION6, data={"x": step[1] * scale + ox + 1,
                                                                 "y": step[2] * scale + oy + 1}))
    return out


def solve(level):
    steps = _search(*_start(level), limit=level.get_data("MaxSteps"))
    return None if steps is None else _actions(steps)


def _level(pieces: tuple, max_steps: int):
    from arcengine import Level

    data = dict(_LEVEL0._data, MaxSteps=max_steps)
    return Level(sprites=[_MODULE.sprites[n].clone().set_position(x, y).set_rotation(r)
                          for n, (x, y, r) in zip(_NAMES, pieces, strict=False)], grid_size=_LEVEL0.grid_size, data=data)


def _apart(pieces: tuple) -> bool:
    """The pieces' boxes neither overlap nor touch, as at level 0."""
    (ax, ay, ar), (bx, by, br) = pieces
    (pa, _), (pb, _) = _SHAPES[_NAMES[0], ar], _SHAPES[_NAMES[1], br]
    return (ax + pa.shape[1] < bx or bx + pb.shape[1] < ax or ay + pa.shape[0] < by or by + pb.shape[0] < ay)


_L0 = len(_search(*_start(_LEVEL0), limit=_LEVEL0.get_data("MaxSteps")))


def generate(rng: random.Random):
    for _ in range(100):
        pieces = []
        for name in _NAMES:
            r = rng.choice((0, 90, 180, 270))
            pixels = _SHAPES[name, r][0]
            pieces.append((rng.randint(0, _W - pixels.shape[1]), rng.randint(0, _H - pixels.shape[0]), r))
        pieces = tuple(pieces)
        if not _apart(pieces):
            continue
        selected = min(range(len(pieces)), key=lambda i: pieces[i][0] ** 2 + pieces[i][1] ** 2)
        estimate = _estimate(selected, pieces)
        if estimate is None or not LENGTH[0] <= estimate <= LENGTH[1]:
            continue
        steps = _search(selected, pieces, limit=LENGTH[1] + 1)
        if steps is not None and len(steps) >= LENGTH[0]:
            return _level(pieces, round(_LEVEL0.get_data("MaxSteps") * len(steps) / _L0))
    return None
