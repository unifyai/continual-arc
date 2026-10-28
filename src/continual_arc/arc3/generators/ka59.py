"""ka59 at level 0: two 3 x 3 pieces, two target frames and a strip of gulch on a walled board of 3-pixel cells. A
click (ACTION6) selects a piece; ACTION1-4 move the selected piece one cell unless a wall or the gulch is there, and a
move into another piece pushes it instead: the pushed piece slides five cells, pushing any piece in front of it,
stops early at a wall, and keeps sliding while it is on the gulch. The level is won when every target frame holds a
piece; each action spends one of 100 steps. The gulch is the only way between the board's two halves and no piece
crosses it by moving, so one piece must push the other across.

generate draws, in one orientation, a gulch strip (2 x 7 cells) anywhere on the 15 x 15 board, walls off its columns
beyond it, carves a room on each side with an L-shaped corridor from each room to the strip, and places both pieces and
one target on the near side and the other target on the far side; the board is then turned and mirrored at random. A
layout is kept when the shortest solution, by the search solve runs, is 8 to 16 actions (level 0's is 11) and, as in
level 0, no solution leaves the first piece selected throughout; the two sides meet only through the gulch, so every
kept layout needs a push across it.

solve is a breadth-first search over (selected piece's cell, other pieces' cells) by the game's rules, read from the
level's own sprites: the walls from the wall sprite's pixels, the gulch from its sprites, and the piece that starts
selected from the game after its reset. It takes milliseconds."""
from __future__ import annotations

import random
from collections import deque

import numpy as np
from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "ka59"
WALL, GULCH, PIECE, TARGET = "0028lydaygyjbu", "0014ysspdlqsqg", "0021xplppqqmfb", "0009ouocpihipp"
CELL = 3      # a move's length in pixels (zsqdfmgyjo), the board's cell
N = 15        # cells a side; the wall sprite covers cells -1 to N, its outer ring wall
SLIDE = 5     # cells a pushed piece slides before it may stop (zuhizjvlpo)
STRIP = (2, 7)  # the gulch sprite in cells, unrotated (6 x 21 pixels)
LENGTH = (8, 16)
TRIES = 300
MAX_STATES = 200_000

MOVES = ((GameAction.ACTION1, 0, -1), (GameAction.ACTION2, 0, 1), (GameAction.ACTION3, -1, 0),
         (GameAction.ACTION4, 1, 0))

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_BASE_LAYOUT = levels.layout(_BASE)


def _slide(pushed: tuple, d: tuple, pieces: set, walls: set, gulch: set) -> set:
    """Where the pieces end after ``pushed`` is pushed one way: it moves a cell a frame, the run of pieces in front
    of it moving with it, for SLIDE frames and then while it is on the gulch, and stops when the run meets a wall."""
    pieces = set(pieces)
    frame = 0
    while frame < SLIDE or pushed in gulch:
        run = [pushed]
        while (run[-1][0] + d[0], run[-1][1] + d[1]) in pieces:
            run.append((run[-1][0] + d[0], run[-1][1] + d[1]))
        if (run[-1][0] + d[0], run[-1][1] + d[1]) in walls:
            break
        pieces -= set(run)
        pieces |= {(i + d[0], j + d[1]) for i, j in run}
        pushed = (pushed[0] + d[0], pushed[1] + d[1])
        frame += 1
    return pieces


def _search(walls: set, gulch: set, selected: tuple, others: tuple, targets: frozenset,
            clicks: bool = True) -> list | None:
    """The shortest actions that cover every target: each a move ("m", action) or, when ``clicks``, a click on a
    piece ("c", cell)."""
    start = (selected, tuple(sorted(others)))
    parent = {start: None}
    queue = deque([start])
    while queue and len(parent) < MAX_STATES:
        state = queue.popleft()
        sel, rest = state
        nexts = []
        for action, dx, dy in MOVES:
            to = (sel[0] + dx, sel[1] + dy)
            if to in walls or to in gulch:
                continue
            if to in rest:
                after = _slide(to, (dx, dy), set(rest), walls, gulch)
                nexts.append((("m", action), (sel, tuple(sorted(after)))))
            else:
                nexts.append((("m", action), (to, rest)))
        for cell in rest if clicks else ():
            nexts.append((("c", cell), (cell, tuple(sorted({*rest, sel} - {cell})))))
        for step, nxt in nexts:
            if nxt in parent:
                continue
            parent[nxt] = (state, step)
            if targets <= {nxt[0], *nxt[1]}:
                path = []
                while parent[nxt] is not None:
                    nxt, step = parent[nxt]
                    path.append(step)
                return path[::-1]
            queue.append(nxt)
    return None


def _cells(sprite) -> set:
    """The board cells a sprite's opaque pixels touch."""
    ys, xs = np.nonzero(sprite.render() != -1)
    return {((sprite.x + x) // CELL, (sprite.y + y) // CELL) for x, y in zip(xs.tolist(), ys.tolist(), strict=True)}


def _model(level) -> tuple:
    """(walls, gulch, selected, others, targets) in cells, from the level's sprites and its game after the reset."""
    game, _ = levels.new_game(GAME, level)
    board = {(i, j) for i in range(N) for j in range(N)}
    walls = {(i, j) for i in range(-1, N + 1) for j in range(-1, N + 1)} - board
    gulch = set()
    for s in game.current_level.get_sprites():
        if s.name == WALL:
            walls |= _cells(s)
        elif s.name == GULCH:
            gulch |= _cells(s)
    pieces = []
    for s in game.current_level.get_sprites_by_tag("0022vrxelxosfy"):
        assert s.x % CELL == 0 and s.y % CELL == 0 and (s.width, s.height) == (CELL, CELL), "pieces sit on cells"
        pieces.append(((s.x // CELL, s.y // CELL), s is game.prkgpeyexo))
    targets = frozenset(((s.x + 1) // CELL, (s.y + 1) // CELL) for s in game.current_level.get_sprites_by_tag(
        "0010xzmuziohuf"))
    (selected,) = [c for c, chosen in pieces if chosen]
    return walls, gulch, selected, tuple(c for c, chosen in pieces if not chosen), targets


def _click(level, cell: tuple) -> ActionInput:
    w, h = level.grid_size
    scale = min(64 // w, 64 // h)
    pad_x, pad_y = (64 - w * scale) // 2, (64 - h * scale) // 2
    x, y = CELL * cell[0] + 1, CELL * cell[1] + 1
    return ActionInput(id=GameAction.ACTION6, data={"x": pad_x + x * scale, "y": pad_y + y * scale})


def solve(level) -> list | None:
    path = _search(*_model(level))
    if path is None:
        return None
    return [ActionInput(id=arg) if kind == "m" else _click(level, arg) for kind, arg in path]


def _rect(rng: random.Random, cols: range, rows: range) -> set:
    w = rng.randint(2, min(5, len(cols)))
    h = rng.randint(2, min(6, len(rows)))
    i0 = rng.randint(cols.start, cols.stop - w)
    j0 = rng.randint(rows.start, rows.stop - h)
    return {(i, j) for i in range(i0, i0 + w) for j in range(j0, j0 + h)}


def _corridor(rng: random.Random, room: set, end: tuple) -> set:
    """An L-shaped corridor from a cell of ``room`` to ``end``, turning at the room's cell's column."""
    ci, cj = rng.choice(sorted(room))
    ei, ej = end
    return ({(ci, j) for j in range(min(cj, ej), max(cj, ej) + 1)}
            | {(i, ej) for i in range(min(ci, ei), max(ci, ei) + 1)})


def _carve(rng: random.Random) -> tuple:
    """A board with the gulch upright at columns gx, gx + 1: (near cells, far cells, gulch cells), near on the left."""
    gx = rng.randint(3, N - 5)
    gy = rng.randint(0, N - STRIP[1])
    gulch = {(gx + a, gy + b) for a in range(STRIP[0]) for b in range(STRIP[1])}
    rows = range(gy, gy + STRIP[1])
    near = _rect(rng, range(0, gx), range(N))
    near |= _corridor(rng, near, (gx - 1, rng.choice(rows)))
    far = _rect(rng, range(gx + 2, N), range(N))
    far |= _corridor(rng, far, (gx + 2, rng.choice(rows)))
    return near, far, gulch


def _turn(cell: tuple, transpose: bool, flip_x: bool, flip_y: bool) -> tuple:
    i, j = (cell[1], cell[0]) if transpose else cell
    return (N - 1 - i if flip_x else i, N - 1 - j if flip_y else j)


def _build(free: set, gulch: set, pieces: list, targets: list):
    pixels = np.full(((N + 2) * CELL, (N + 2) * CELL), -1, dtype=np.int8)
    for i in range(-1, N + 1):
        for j in range(-1, N + 1):
            if (i, j) not in free:
                pixels[CELL * (j + 1):CELL * (j + 2), CELL * (i + 1):CELL * (i + 2)] = 2
    wall = _SPRITES[WALL].clone().set_position(-CELL, -CELL)
    wall.pixels = pixels
    gi, gj = min(gulch)
    strip = _SPRITES[GULCH].clone().set_position(CELL * gi, CELL * gj)
    if max(i for i, _ in gulch) - gi + 1 != STRIP[0]:
        strip.set_rotation(90)
    sprites = [_SPRITES[TARGET].clone().set_position(CELL * i - 1, CELL * j - 1) for i, j in targets]
    sprites += [strip]
    sprites += [_SPRITES[PIECE].clone().set_position(CELL * i, CELL * j) for i, j in pieces]
    sprites += [wall]
    return type(_BASE)(sprites=sprites, grid_size=_BASE.grid_size, data=dict(_BASE._data), name=_BASE.name)


def generate(rng: random.Random):
    for _ in range(TRIES):
        near, far, gulch = _carve(rng)
        turn = (rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5)
        near, far, gulch = ({_turn(c, *turn) for c in cells} for cells in (near, far, gulch))
        pieces = rng.sample(sorted(near), 2)
        target_near = rng.choice(sorted(near - set(pieces)))
        target_far = rng.choice(sorted(far))
        walls = {(i, j) for i in range(-1, N + 1) for j in range(-1, N + 1)} - near - far - gulch
        model = (walls, gulch, pieces[0], (pieces[1],), frozenset({target_near, target_far}))
        path = _search(*model)
        if path is None or not LENGTH[0] <= len(path) <= LENGTH[1] or _search(*model, clicks=False) is not None:
            continue
        level = _build(near | far | gulch, gulch, pieces, [target_near, target_far])
        if levels.layout(level) == _BASE_LAYOUT:
            continue
        return level
    return None
