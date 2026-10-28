"""su15 at level 0: a click (ACTION6) pulls every fruit within 8 cells of it to the clicked point over four frames, and
each click spends one of 32 steps. The level is won when a level-2 fruit (the 3x3 one) has its center inside the
9x9 target; it is lost when the steps run out. A "+" beside the fruit marks a first click.

A layout puts the target, with its center marker, anywhere it leaves room for a fruit 6 to 8 clicks away (level 0's
takes 7), then the one fruit at a random place that many clicks from it; the "+" sits at the first click of that
route. Level 0's dotted guide line, a fixed diagonal 43 cells long that only joins level 0's own corners, is left out.
Every such layout is solvable, since a click within reach always lands the fruit's center on it.

solve reads the game's rule as a reach kernel (click offsets whose distance to the fruit's 3x3 box is at most 8),
grows click-distance layers back from the target's cells by dilation, walks down them from the fruit, each click the
one nearest the target's center, and checks the clicks win in the real engine.
"""
from __future__ import annotations

import copy
import random

import numpy as np
from arcengine import ActionInput, GameAction, Level

from continual_arc.arc3 import levels

GAME = "su15"
_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_DATA = _MODULE.levels[0]._data
_REACH = 8                       # a click pulls a fruit whose box is this close (the game's kacsjmxae)
_X = (1, 62)                     # where a 3x3 fruit's center can be: x 0-61, y 10-61 for its corner
_Y = (11, 62)
_TARGET = 9
_CLICKS = (6, 8)                 # level 0 takes 7, and no two places on the board are more than 8 apart
_OFFSETS = [(dx, dy) for dx in range(-_REACH - 1, _REACH + 2) for dy in range(-_REACH - 1, _REACH + 2)
            if max(abs(dx) - 1, 0) ** 2 + max(abs(dy) - 1, 0) ** 2 <= _REACH ** 2]


def _layers(tx: int, ty: int) -> np.ndarray:
    """Clicks from each fruit center (indexed [y, x]) into the target at (tx, ty); -1 where out of the play area."""
    dist = np.full((64, 64), -1, dtype=np.int32)
    play = np.zeros((64, 64), dtype=bool)
    play[_Y[0]:_Y[1] + 1, _X[0]:_X[1] + 1] = True
    frontier = np.zeros_like(play)
    frontier[ty:ty + _TARGET, tx:tx + _TARGET] = True
    frontier &= play
    dist[frontier] = 0
    k = 0
    while frontier.any():
        k += 1
        grown = np.zeros_like(play)
        for dx, dy in _OFFSETS:
            shifted = np.zeros_like(play)
            shifted[max(dy, 0):64 + min(dy, 0), max(dx, 0):64 + min(dx, 0)] = \
                frontier[max(-dy, 0):64 - max(dy, 0), max(-dx, 0):64 - max(dx, 0)]
            grown |= shifted
        frontier = grown & play & (dist < 0)
        dist[frontier] = k
    return dist


def _route(cx: int, cy: int, tx: int, ty: int) -> list[tuple[int, int]] | None:
    """The clicks that bring a fruit centered at (cx, cy) into the target, fewest first, each the one nearest the
    target's center."""
    dist = _layers(tx, ty)
    if dist[cy, cx] < 0:
        return None
    mx, my, clicks = tx + _TARGET // 2, ty + _TARGET // 2, []
    while dist[cy, cx] > 0:
        steps = [(cx + dx, cy + dy) for dx, dy in _OFFSETS
                 if _X[0] <= cx + dx <= _X[1] and _Y[0] <= cy + dy <= _Y[1] and dist[cy + dy, cx + dx] == dist[cy, cx] - 1]
        cx, cy = min(steps, key=lambda p: ((p[0] - mx) ** 2 + (p[1] - my) ** 2, p))
        clicks.append((cx, cy))
    return clicks


def _parts(level: Level) -> tuple[int, int, int, int]:
    (fruit,) = level.get_sprites_by_tag("zmlxwcvwb")
    (target,) = level.get_sprites_by_tag("xkstxyqbs")
    return fruit.x + 1, fruit.y + 1, target.x, target.y


def generate(rng: random.Random) -> Level | None:
    while True:
        tx, ty = rng.randint(0, 64 - _TARGET), rng.randint(10, 63 - _TARGET)
        dist = _layers(tx, ty)
        counts = [k for k in range(_CLICKS[0], _CLICKS[1] + 1) if (dist == k).any()]
        if counts:
            break
    ys, xs = np.nonzero(dist == rng.choice(counts))
    i = rng.randrange(len(xs))
    cx, cy = int(xs[i]), int(ys[i])
    hx, hy = _route(cx, cy, tx, ty)[0]
    return Level(
        sprites=[
            _SPRITES["2"].clone().set_position(cx - 1, cy - 1),
            _SPRITES["0008hngjwqibfi"].clone().set_position(tx, ty),
            _SPRITES["0009mbvjylwely"].clone(),
            _SPRITES["0016fnrcxwfvtx"].clone().set_position(30, 4),
            _SPRITES["0020qkplcbivxi"].clone().set_position(hx - 2, hy - 1),
            _SPRITES["0022xgjoqvpfrf"].clone(),
            _SPRITES["0027blsxsorecj"].clone().set_position(tx + 3, ty + 3),
        ],
        grid_size=(64, 64),
        data=copy.deepcopy(_DATA),
    )


def solve(level: Level) -> list[ActionInput] | None:
    clicks = _route(*_parts(level))
    if clicks is None:
        return None
    actions = [ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}) for x, y in clicks]
    return actions if levels.replay(GAME, level, actions) else None
