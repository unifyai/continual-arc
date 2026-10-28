"""ls20 at level 0: a maze on a grid of 5-pixel cells. ACTION1-4 move the player one cell up, down, left or right
unless a wall is there; the goal cell lets the player in only once the key in the bottom-left box matches the goal's
shape, color and rotation, and entering it wins. The key starts a quarter turn off (StartRotation 270, GoalRotation
0), and each entry onto the rotator cell turns it 90 degrees, so the player must visit the rotator once and then reach
the goal within the 42-move counter; running out costs one of three lives and restarts the maze.

generate carves a fresh maze into the 10 x 9 interior inside level 0's border walls: three to five rooms of 1-5 x 1-4
cells joined in a random order by L-shaped corridors, sometimes with a dead-end spur. It places the goal, the rotator
and the player on three free cells anywhere in it and keeps the layout when the shortest win, by the same search solve
runs, is 9 to 19 moves (level 0's is 13), so every kept level is solvable well within the counter. Level 0's data are
kept as they are; only the goal's decorations move with it, the notch of its frame facing an open neighbor.

solve is a breadth-first search over (player cell, key rotation) derived from the game's code: walls block, the goal
blocks until the rotation matches, and entering the rotator advances the rotation. It returns the shortest action
sequence in a few milliseconds.
"""
from __future__ import annotations

import random
from collections import deque

from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "ls20"

CELL = 5
X0, Y0 = 4, 0          # the grid's origin: cell (i, j) is at pixel (X0 + CELL*i, Y0 + CELL*j)
COLS, ROWS = 12, 12    # the whole grid; its outer ring is wall, as in level 0
INTERIOR_I = range(1, 11)
INTERIOR_J = range(1, 10)  # row 10 (y=50) stays wall, as in level 0, clear of the bottom-left key box
LENGTH = (9, 19)
TRIES = 400

MOVES = ((GameAction.ACTION1, 0, -1), (GameAction.ACTION2, 0, 1), (GameAction.ACTION3, -1, 0),
         (GameAction.ACTION4, 1, 0))
ROTATIONS = (0, 90, 180, 270)

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_DATA = dict(_BASE._data)
_BASE_LAYOUT = levels.layout(_BASE)


def _cell(x: int, y: int) -> tuple[int, int]:
    return (x - X0) // CELL, (y - Y0) // CELL


def _pixel(i: int, j: int) -> tuple[int, int]:
    return X0 + CELL * i, Y0 + CELL * j


def _search(free: set, player: tuple, rotator: tuple, goal: tuple, start_rot: int, goal_rot: int) -> list | None:
    """Shortest moves from the player to the goal over (cell, rotation index), by the game's rules."""
    start = (player, start_rot)
    parent = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        (i, j), rot = state
        for action, dx, dy in MOVES:
            cell = (i + dx, j + dy)
            if cell not in free:
                continue
            if cell == goal:
                if rot != goal_rot:
                    continue
                path = [action]
                while parent[state] is not None:
                    state, act = parent[state]
                    path.append(act)
                return path[::-1]
            nrot = (rot + 1) % 4 if cell == rotator else rot
            nxt = (cell, nrot)
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    return None


def _rule() -> tuple[int, int]:
    return ROTATIONS.index(_DATA["StartRotation"]), ROTATIONS.index(_DATA["GoalRotation"])


def _carve(rng: random.Random) -> set:
    free: set = set()
    rooms = []
    for _ in range(rng.randint(3, 5)):
        w, h = rng.randint(1, 5), rng.randint(1, 4)
        i0 = rng.randint(INTERIOR_I.start, INTERIOR_I.stop - w)
        j0 = rng.randint(INTERIOR_J.start, INTERIOR_J.stop - h)
        cells = [(i, j) for i in range(i0, i0 + w) for j in range(j0, j0 + h)]
        free.update(cells)
        rooms.append(cells)
    for a, b in zip(rooms, rooms[1:], strict=False):
        (ai, aj), (bi, bj) = rng.choice(a), rng.choice(b)
        if rng.random() < 0.5:
            free.update((i, aj) for i in range(min(ai, bi), max(ai, bi) + 1))
            free.update((bi, j) for j in range(min(aj, bj), max(aj, bj) + 1))
        else:
            free.update((ai, j) for j in range(min(aj, bj), max(aj, bj) + 1))
            free.update((i, bj) for i in range(min(ai, bi), max(ai, bi) + 1))
    if rng.random() < 0.5:
        i, j = rng.choice(sorted(free))
        _, dx, dy = rng.choice(MOVES)
        for _ in range(rng.randint(2, 4)):
            if i + dx not in INTERIOR_I or j + dy not in INTERIOR_J:
                break
            i, j = i + dx, j + dy
            free.add((i, j))
    return free


def _build(free: set, player: tuple, rotator: tuple, goal: tuple, notch: int) -> object:
    gx, gy = _pixel(*goal)
    sprites = [
        _SPRITES["eqatonpohu"].clone().set_position(1, 53),
        _SPRITES["ghizzeqtoh"].clone().set_position(1, 53),
        _SPRITES["hoswmpiqkw"].clone().set_position(gx - 1, gy - 1),
    ]
    sprites += [_SPRITES["ihdgageizm"].clone().set_position(*_pixel(i, j))
                for j in range(ROWS) for i in range(COLS) if (i, j) not in free]
    sprites += [
        _SPRITES["kvynsvxbpi"].clone().set_position(gx + 1, gy + 1),
        _SPRITES["nszegiawib"].clone().set_position(gx - 2, gy - 2).set_rotation(notch),
        _SPRITES["rhsxkxzdjz"].clone().set_position(*_pixel(*rotator)),
        _SPRITES["rjlbuycveu"].clone().set_position(gx, gy),
        _SPRITES["sfqyzhzkij"].clone().set_position(*_pixel(*player)),
        _SPRITES["vjotnebuqo"].clone().set_position(gx - 1, gy - 1),
        _SPRITES["wgmbtyhvbc"].clone().set_position(3, 55).set_scale(2),
        _SPRITES["xvrpzkggig"].clone(),
    ]
    return type(_BASE)(sprites=sprites, grid_size=_BASE.grid_size, data=dict(_DATA))


def generate(rng: random.Random):
    start_rot, goal_rot = _rule()
    for _ in range(TRIES):
        free = _carve(rng)
        if len(free) < 12:
            continue
        player, rotator, goal = rng.sample(sorted(free), 3)
        path = _search(free, player, rotator, goal, start_rot, goal_rot)
        if path is None or not LENGTH[0] <= len(path) <= LENGTH[1]:
            continue
        openings = [rot for rot, (dx, dy) in zip(ROTATIONS, ((0, -1), (1, 0), (0, 1), (-1, 0)), strict=False)
                    if (goal[0] + dx, goal[1] + dy) in free]
        level = _build(free, player, rotator, goal, rng.choice(openings))
        if levels.layout(level) == _BASE_LAYOUT:
            continue
        return level
    return None


def solve(level) -> list | None:
    def one(name: str) -> tuple:
        (s,) = [s for s in level.get_sprites() if s.name == name]
        return _cell(s.x, s.y)

    free = {(i, j) for i in range(COLS) for j in range(ROWS)}
    free -= {_cell(s.x, s.y) for s in level.get_sprites() if s.name == "ihdgageizm"}
    data = level._data
    path = _search(free, one("sfqyzhzkij"), one("rhsxkxzdjz"), one("rjlbuycveu"),
                   ROTATIONS.index(data["StartRotation"]), ROTATIONS.index(data["GoalRotation"]))
    return None if path is None else [ActionInput(id=a) for a in path]
