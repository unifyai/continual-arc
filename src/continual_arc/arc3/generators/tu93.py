"""tu93 at level 0: a player walks a maze of 31 nodes, six pixels apart, joined by corridors; ACTION1-4 step it one
node up, down, left or right where a corridor leads, and the level is won when it stands on the exit. Every action,
blocked ones included, spends one of the 50 steps, and running out loses.

A layout is level 0's maze in one of its eight symmetries (its pixels turned or mirrored, so the walls the game reads
are the walls it draws), with the player and the exit on two of its nodes whose shortest path is 12 to 18 steps
(level 0's corner-to-corner path, the maze's longest, is 18): a draw takes such a pair of places uniformly, then one of
the symmetries that gives it. The maze is connected, so every layout is solvable.

solve is a breadth-first search over the maze's nodes, with the corridor test read from the game (the pixel half a
step away is a corridor's), and its answer is checked in the real engine."""
from __future__ import annotations

import random
from collections import deque

import numpy as np

from continual_arc.arc3 import levels

GAME = "tu93"
MAZE, EXIT, PLAYER = "0004iwcrtzivmj", "0014mzhhvzrazi", "0016ihgrljrgpq"
NODE = 6  # pixels between nodes (hwthhtvyki * 2 in the game)
HALF = NODE // 2
DISTANCE = (12, 18)  # shortest paths kept; level 0's is 18
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
(_MAZE0,) = [s for s in _LEVEL0.get_sprites() if s.name == MAZE]


def _nodes(pixels: np.ndarray) -> list[tuple[int, int]]:
    h, w = pixels.shape
    return [(x, y) for y in range(0, h, NODE) for x in range(0, w, NODE) if pixels[y, x] >= 0]


def _distances(pixels: np.ndarray, start: tuple[int, int]) -> dict:
    """Shortest move lists from ``start`` to every node, each node (maze-relative) mapped to its moves."""
    h, w = pixels.shape
    paths = {start: []}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for move, (dx, dy) in MOVES.items():
            cx, cy = x + HALF * dx, y + HALF * dy
            if 0 <= cx < w and 0 <= cy < h and pixels[cy, cx] == 2:
                nxt = (x + NODE * dx, y + NODE * dy)
                if nxt not in paths:
                    paths[nxt] = [*paths[(x, y)], move]
                    queue.append(nxt)
    return paths


def _layouts() -> dict[tuple, list[np.ndarray]]:
    """Every (start, exit) pair of places whose shortest path is in DISTANCE in some of the maze's eight symmetries,
    level 0's own pair left out, mapped to the symmetries' pixels that give it."""
    base = np.array(_MODULE.sprites[MAZE].pixels)
    (start0,) = [(s.x - _MAZE0.x, s.y - _MAZE0.y) for s in _LEVEL0.get_sprites() if s.name == PLAYER]
    (exit0,) = [(s.x - _MAZE0.x, s.y - _MAZE0.y) for s in _LEVEL0.get_sprites() if s.name == EXIT]
    out: dict[tuple, list[np.ndarray]] = {}
    for turns in range(4):
        for mirror in (False, True):
            pixels = np.rot90(base, turns)
            pixels = np.ascontiguousarray(np.fliplr(pixels) if mirror else pixels)
            for start in _nodes(pixels):
                for goal, path in _distances(pixels, start).items():
                    if DISTANCE[0] <= len(path) <= DISTANCE[1] and (start, goal) != (start0, exit0):
                        out.setdefault((start, goal), []).append(pixels)
    return out


_LAYOUTS = _layouts()
_PAIRS = sorted(_LAYOUTS)


def generate(rng: random.Random):
    from arcengine import Level

    start, goal = rng.choice(_PAIRS)
    maze = _MODULE.sprites[MAZE].clone().set_position(_MAZE0.x, _MAZE0.y)
    maze.pixels = rng.choice(_LAYOUTS[(start, goal)]).copy()
    return Level(
        sprites=[
            maze,
            _MODULE.sprites[EXIT].clone().set_position(_MAZE0.x + goal[0], _MAZE0.y + goal[1]),
            _MODULE.sprites[PLAYER].clone().set_position(_MAZE0.x + start[0], _MAZE0.y + start[1])
            .set_rotation(rng.choice([0, 90, 180, 270])),
        ],
        grid_size=_LEVEL0.grid_size,
        data=dict(_LEVEL0._data),
    )


def solve(level) -> list | None:
    from arcengine import ActionInput, GameAction

    (maze,) = [s for s in level.get_sprites() if s.name == MAZE]
    (player,) = [s for s in level.get_sprites() if s.name == PLAYER]
    (goal,) = [s for s in level.get_sprites() if s.name == EXIT]
    paths = _distances(np.asarray(maze.pixels), (player.x - maze.x, player.y - maze.y))
    path = paths.get((goal.x - maze.x, goal.y - maze.y))
    if path is None:
        return None
    actions = [ActionInput(id=GameAction.from_id(m)) for m in path]
    return actions if levels.replay(GAME, level, actions) else None
