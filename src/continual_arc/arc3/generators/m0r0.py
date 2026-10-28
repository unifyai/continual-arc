"""m0r0 at level 0: two identical blocks on an 11 x 11 map whose walls are one sprite (wahtyt-Level6). ACTION1-4 move
one block up, down, left or right and the other the same way vertically but mirrored horizontally; a block that
would enter a wall or leave the grid stays put while the other still moves. The level is won when the blocks meet,
by landing on one cell or by swapping two side-by-side cells; nothing else ends it before the 150-action limit.

generate keeps level 0's data and its one wall sprite, turned to any of its four rotations (level 0 is turned 180),
and drops the normal and the mirrored block on two free cells anywhere on the map. It keeps the layout when the
fewest moves that make them meet, by the same search solve runs, are 11 to 22 (level 0's are 15), which makes every
kept level solvable and about as long; layouts the blocks can never meet from are redrawn.

solve is a breadth-first search over the two blocks' cells, derived from the game's code: each block moves alone
against the walls and the grid's edge, a swap of side-by-side blocks merges them, and the search ends when they
share a cell. It takes a few milliseconds.
"""
from __future__ import annotations

import random
from collections import deque

from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "m0r0"

NORMAL, MIRRORED, WALLS = "pikgci-toljda-leklkn", "pikgci-toljda-rivmdg", "wahtyt-Level6"
MOVES = ((GameAction.ACTION1, 0, -1), (GameAction.ACTION2, 0, 1), (GameAction.ACTION3, -1, 0),
         (GameAction.ACTION4, 1, 0))
ROTATIONS = (0, 90, 180, 270)
LENGTH = (11, 22)
TRIES = 400

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_DATA = dict(_BASE._data)
_BASE_LAYOUT = levels.layout(_BASE)


def _free(walls, size: tuple[int, int]) -> set[tuple[int, int]]:
    """The grid's cells no wall pixel covers, as the game's pixel-perfect collision reads the wall sprite."""
    width, height = size
    pixels = walls.render()
    return {(x, y) for x in range(width) for y in range(height)
            if not (0 <= x - walls.x < pixels.shape[1] and 0 <= y - walls.y < pixels.shape[0]
                    and pixels[y - walls.y, x - walls.x] >= 0)}


def _search(free: set, normal: tuple, mirrored: tuple) -> list | None:
    """Fewest moves that make the blocks meet, by the game's step (its pyhtlpzlmnr branch)."""
    def go(cell, dx, dy):
        nxt = (cell[0] + dx, cell[1] + dy)
        return nxt if nxt in free else cell

    start = (normal, mirrored)
    parent = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        a, b = state
        for action, dx, dy in MOVES:
            na, nb = go(a, dx, dy), go(b, -dx, dy)
            side_by_side = abs(a[0] - b[0]) == 1 and a[1] == b[1]
            if na == nb or side_by_side and (na == b or nb == a):
                path = [action]
                while parent[state] is not None:
                    state, act = parent[state]
                    path.append(act)
                return path[::-1]
            nxt = (na, nb)
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    return None


def generate(rng: random.Random):
    for _ in range(TRIES):
        walls = _SPRITES[WALLS].clone().set_rotation(rng.choice(ROTATIONS))
        normal, mirrored = rng.sample(sorted(_free(walls, _BASE.grid_size)), 2)
        path = _search(_free(walls, _BASE.grid_size), normal, mirrored)
        if path is None or not LENGTH[0] <= len(path) <= LENGTH[1]:
            continue
        level = type(_BASE)(sprites=[_SPRITES[NORMAL].clone().set_position(*normal),
                                     _SPRITES[MIRRORED].clone().set_position(*mirrored), walls],
                            grid_size=_BASE.grid_size, data=dict(_DATA))
        if levels.layout(level) == _BASE_LAYOUT:
            continue
        return level
    return None


def solve(level) -> list | None:
    def one(name: str):
        (s,) = [s for s in level.get_sprites() if s.name == name]
        return s

    free = _free(one(WALLS), level.grid_size)
    path = _search(free, (one(NORMAL).x, one(NORMAL).y), (one(MIRRORED).x, one(MIRRORED).y))
    return None if path is None else [ActionInput(id=a) for a in path]
