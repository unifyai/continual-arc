"""ar25 at level 0: one L-shaped piece moves a cell at a time (ACTION1-4) on a 21 x 21 board split by a fixed vertical
mirror, and the level is won when the piece and its reflection together cover every target cell; the target is the
piece's mirror image, so only the reflection can cover it. Each move spends one of the 64 steps and running out loses.

A layout draws the mirror's column, which side the target lies on, the target's place there, and the piece's start
on the other side; the piece's winning place is the target reflected back, so every layout is solvable, and draws
whose shortest solution is far from level 0's are redrawn.

solve is a breadth-first search over the piece's position, with the win rule read from the game (piece cells plus
their reflections cover the targets), and its answer is checked in the real engine."""
from __future__ import annotations

import random
from collections import deque

from continual_arc.arc3 import levels

GAME = "ar25"
TARGET, PIECE, MIRROR = "0001sruqbuvukh", "0007arvfmhagbj", "0055nwhypaamix"
SIZE = 21
LENGTH = (10, 22)  # shortest solution lengths kept; level 0's is 15

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]


def _cells(name: str) -> list[tuple[int, int]]:
    pixels = _MODULE.sprites[name].pixels
    return [(x, y) for y in range(pixels.shape[0]) for x in range(pixels.shape[1]) if pixels[y, x] != -1]


_SHAPE = _cells(PIECE)
_W = max(x for x, _ in _SHAPE) + 1
_H = max(y for _, y in _SHAPE) + 1


def _covered(px: int, py: int, mx: int) -> set[tuple[int, int]]:
    cells = {(px + x, py + y) for x, y in _SHAPE}
    return cells | {(2 * mx - x, y) for x, y in cells}


def _search(start: tuple[int, int], mx: int, targets: frozenset) -> list[int] | None:
    """Shortest list of moves (1 up, 2 down, 3 left, 4 right) that brings the piece to a place where it and its
    reflection cover ``targets``; moves that would leave the board are no-ops in the game and are skipped."""
    steps = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    prev = {start: None}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if targets <= _covered(*pos, mx):
            path = []
            while prev[pos] is not None:
                pos, move = prev[pos]
                path.append(move)
            return path[::-1]
        for move, (dx, dy) in steps.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if 0 <= nxt[0] <= SIZE - _W and 0 <= nxt[1] <= SIZE - _H and nxt not in prev:
                prev[nxt] = (pos, move)
                queue.append(nxt)
    return None


def _level(mx: int, piece: tuple[int, int], targets: list[tuple[int, int]]):
    from arcengine import Level

    sprites = _MODULE.sprites
    return Level(
        sprites=[sprites[TARGET].clone().set_position(x, y) for x, y in targets]
        + [sprites[PIECE].clone().set_position(*piece), sprites[MIRROR].clone().set_position(mx, 0)],
        grid_size=_LEVEL0.grid_size,
        data=dict(_LEVEL0._data),
    )


def generate(rng: random.Random):
    for _ in range(200):
        mx = rng.randint(5, SIZE - 6)
        right = rng.random() < 0.5
        # The target is the piece reflected: reflect a piece placed on the far side.
        wx = rng.randint(0, mx - _W) if right else rng.randint(mx + 1, SIZE - _W)
        wy = rng.randint(0, SIZE - _H)
        targets = sorted((2 * mx - (wx + x), wy + y) for x, y in _SHAPE)
        if not all(0 <= x < SIZE for x, _ in targets):
            continue
        sx = rng.randint(0, mx - _W) if right else rng.randint(mx + 1, SIZE - _W)
        sy = rng.randint(0, SIZE - _H)
        path = _search((sx, sy), mx, frozenset(targets))
        if path is None or not LENGTH[0] <= len(path) <= LENGTH[1]:
            continue
        return _level(mx, (sx, sy), targets)
    return None


def solve(level) -> list | None:
    from arcengine import ActionInput, GameAction

    (piece,) = [s for s in level.get_sprites() if s.name == PIECE]
    (mirror,) = [s for s in level.get_sprites() if s.name == MIRROR]
    targets = frozenset((s.x, s.y) for s in level.get_sprites() if s.name == TARGET)
    path = _search((piece.x, piece.y), mirror.x, targets)
    if path is None:
        return None
    actions = [ActionInput(id=GameAction.from_id(m)) for m in path]
    return actions if levels.replay(GAME, level, actions) else None
