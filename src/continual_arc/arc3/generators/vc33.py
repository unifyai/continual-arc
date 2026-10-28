"""vc33 at level 0: two black tanks of liquid rest against the right edge of a 32 x 32 board (gravity pulls right,
level 0's Gravity [2, 0]), one above and one below a gray wall with a mark on it. ACTION6 on the button at one
tank's foot draws two columns of liquid into that tank from the other, so it rises while the other falls; a tank
cannot rise past the wall's left end, and the other must not be empty. A float rests on one tank's surface and moves with it, and the
level is won when the float's yellow strip lines up with the wall's mark, within the 50-step counter.

generate draws where the wall lies (any lane from 8 to 22), how full each tank is, which tank carries the float and
where in its lane, and the mark's place on the wall, as two to five pours from the float's start in either direction
that the other tank can supply or take (level 0's is three upward pours). The tanks are level 0's black rectangles
with the extent drawn (the game itself resizes them as it pours); every other sprite keeps level 0's pixels, and the
buttons sit at each tank's foot beside the wall, where the game wires them, as in level 0.

solve is levels.bfs: the two buttons are the only actions, and the shortest answer is at most five clicks.
"""
from __future__ import annotations

import random

import numpy as np
from arcengine import Level

from continual_arc.arc3 import levels

GAME = "vc33"
MARK, FLOAT, BUTTON, WALL = "0009vwghyhtilb", "0015cltmlkogkc", "0021ayhektiwsl", "0024oklzljpbjf"
TANKS = ("0042ueboolxadn", "0044auurrayaab")
ROTATION = 270          # level 0's: the tanks' raw rows are their rendered columns
SIZE = 32
POUR = 2                # columns a click moves, level 0's Gravity
WALL_X = 10             # the wall spans x 10-31 and no tank rises past its left end
BUTTON_X = 30
LANES = range(8, 23, 2)  # the wall's row; level 0's is 14
FILL = range(2, 21, 2)  # a tank's starting width
POURS = range(2, 6)     # clicks to win; level 0's is 3
TRIES = 200

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_SPRITES = _MODULE.sprites
_BASE_LAYOUT = levels.layout(_LEVEL0)
_FLOAT_W = _SPRITES[FLOAT].clone().set_rotation(ROTATION).width
_FLOAT_H = _SPRITES[FLOAT].clone().set_rotation(ROTATION).height


def _sprite(name: str, x: int, y: int):
    return _SPRITES[name].clone().set_rotation(ROTATION).set_position(x, y)


def _tank(name: str, width: int, lane: int, height: int):
    tank = _sprite(name, SIZE - width, lane)
    tank.pixels = np.zeros((width, height), dtype=np.int8)
    return tank


def generate(rng: random.Random) -> Level | None:
    for _ in range(TRIES):
        wall = rng.choice(LANES)
        upper, lower = rng.sample(TANKS, 2)
        widths = {upper: rng.choice(FILL), lower: rng.choice(FILL)}
        carrier, other = rng.sample(TANKS, 2)
        pours = rng.choice(POURS)
        rise = rng.random() < 0.5
        start = SIZE - widths[carrier]
        end = start - POUR * pours if rise else start + POUR * pours
        # Rising needs the other tank to hold the pours and the carrier to stay right of the wall's end; falling
        # needs the carrier to hold them and the other tank to have room.
        if rise and (widths[other] < POUR * pours or end < WALL_X):
            continue
        if not rise and (widths[carrier] - POUR * pours < POUR or SIZE - widths[other] - POUR * pours < WALL_X):
            continue
        mark_x = end - _FLOAT_W
        if mark_x < WALL_X:
            continue
        lane, height = (0, wall) if carrier == upper else (wall + 2, SIZE - wall - 2)
        float_y = rng.randint(lane, lane + height - _FLOAT_H)
        sprites = [
            _sprite(MARK, mark_x, wall),
            _sprite(FLOAT, start - _FLOAT_W, float_y),
            _sprite(BUTTON, BUTTON_X, wall - 2),
            _sprite(BUTTON, BUTTON_X, wall + 2),
            _sprite(WALL, WALL_X, wall),
            _tank(upper, widths[upper], 0, wall),
            _tank(lower, widths[lower], wall + 2, SIZE - wall - 2),
        ]
        level = Level(sprites=sprites, grid_size=_LEVEL0.grid_size, data=dict(_LEVEL0._data))
        if levels.layout(level) != _BASE_LAYOUT:
            return level
    return None
