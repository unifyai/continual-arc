"""sp80 at level 0: a spout drips liquid straight down; ACTION1-4 slide the selected bar (the one nearest the top-left
is selected at the start, ACTION6 selects another) and ACTION5 pours. Liquid that lands on a bar runs along it and
falls off both ends; the level is won when every cup catches liquid through its open top and none reaches the floor,
and lost when the 30 steps run out or a fifth pour is tried.

A layout puts the spout, two cups on the floor and the one five-cell bar anywhere on the 16x16 board, under the rule
that makes level 0 solvable: the cups' openings are six columns apart and the spout's column lies over the bar's
place between them, so sliding the bar there (2 to 5 cells from where it starts, at any height clear of the cups)
and pouring wins.

solve is the harness's breadth-first search over the real engine, bounded by states expanded.
"""
from __future__ import annotations

import random

from arcengine import Level

from continual_arc.arc3 import levels

GAME = "sp80"
_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_DATA = _MODULE.levels[0]._data
_SIZE = 16
_CUP_Y = 13          # cups stand on the floor (y 15), two rows tall
_SHIFT = (2, 5)      # how far the bar starts from the place that wins, level 0's being 3


def generate(rng: random.Random) -> Level:
    left = rng.randint(1, 8)                 # the left cup's opening; the right one is six columns on
    target = left + 1                        # the bar's x that spills into both openings
    spout = rng.randint(target, target + 4)
    starts = [x for x in range(0, _SIZE - 5 + 1) if _SHIFT[0] <= abs(x - target) <= _SHIFT[1]]
    bar_x, bar_y = rng.choice(starts), rng.randint(3, _CUP_Y - 2)
    return Level(
        sprites=[
            _SPRITES["bodekplurlf16"].clone().set_position(-1, -1),
            _SPRITES["liolfvkveqg"].clone().set_position(spout, 1),
            _SPRITES["plzwjbfyfli-5"].clone().set_position(bar_x, bar_y),
            _SPRITES["repwkzbkhxl"].clone().set_position(left - 1, _CUP_Y),
            _SPRITES["repwkzbkhxl"].clone().set_position(left + 5, _CUP_Y),
            _SPRITES["sowlljgtjvn"].clone().set_position(spout, 0),
            _SPRITES["waoewejnqzc"].clone().set_position(0, 15),
        ],
        grid_size=(_SIZE, _SIZE),
        data=dict(_DATA),
    )


def solve(level: Level) -> list | None:
    return levels.bfs(GAME, level, max_states=100_000)
