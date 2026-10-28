"""sp80 for the baselines' tests, as the benchmark's tests/arc3_generators/sp80.py: level 0's spout, cups and floor,
its one bar started 2 to 5 cells either side of the place under the spout that spills into both cups (x 6), at a
height of its own; solved in the bar's distance plus one pour, by levels.bfs."""
from __future__ import annotations

import random

from continual_arc.arc3 import levels

GAME = "sp80"
BAR = "plzwjbfyfli-5"
TARGET = 6
STARTS = [x for x in range(0, 12) if 2 <= abs(x - TARGET) <= 5]

_BASE = levels.level0(GAME)


def generate(rng: random.Random):
    level = _BASE.clone()
    x, y = rng.choice(STARTS), rng.randint(3, 8)
    if (x, y) == (3, 4):   # level 0's own
        return None
    (bar,) = [s for s in level.get_sprites() if s.name == BAR]
    bar.set_position(x, y)
    return level
