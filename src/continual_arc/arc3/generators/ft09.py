"""ft09 at level 0: a 32 x 32 board in four quadrants. Three hold worked examples, a 3 x 3 pattern of 0s and 2s amid
eight tiles colored to match it; the fourth holds the puzzle, the same pattern sprite (wmW) amid tiles that all
start as color 9. A click on a tile cycles it through cwU (9, 8); the level is won when every tile four cells from
the pattern matches the pattern's center color (8) where the pattern shows 0 and differs where it shows 2, and the
kCv-th (32nd) tile click that does not win loses.

A layout puts the puzzle in any quadrant, its ring of tiles anywhere inside the puzzle's frame, and the three
examples in the other three quadrants in any order and anywhere inside them, the background sprites covering the
non-puzzle quadrants; it leaves out up to two of the eight tiles, never more than one that needs a click. A tile
that is absent is not checked, so every layout is solvable.

solve reads the rule from the game's code: each tile next to the pattern needs the clicks that bring its color to
what the pattern asks, here one or none; the clicks are those tiles in any order, which is the shortest solution."""
from __future__ import annotations

import random

from continual_arc.arc3 import levels

GAME = "ft09"
PATTERN, TILE = "wmW", "Hkx"
FRAME, BRACKETS = "seE", "UEq"  # the puzzle's quadrant: its edge, and its corner brackets that flash on a miss
HALF, QUARTER = "dvQ", "cgN"  # the backgrounds that cover the three quadrants without the puzzle
EXAMPLES = (("wCl", "sxj"), ("TOK", "MEF"), ("vxk", "qqX"))  # each example's tiles and its pattern, four cells in
PITCH = 4  # a tile's distance from the pattern, as the win check reads it

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_RING = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]


def _needs_click(dx: int, dy: int) -> bool:
    """Whether the ring tile at (dx, dy) must change from the start color cwU[0] to satisfy the pattern."""
    pixels = _MODULE.sprites[PATTERN].pixels
    start, center = _LEVEL0.get_data("cwU")[0], pixels[1][1]
    return (start == center) != (pixels[1 + dy][1 + dx] == 0)


def solve(level):
    from arcengine import ActionInput, GameAction

    colors = level.get_data("cwU")
    assert level.get_data("elp") == [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "a click turns only the tile clicked"
    tiles = {(s.x, s.y) for s in level.get_sprites() if s.name == TILE}
    clicks = []
    for target in (s for s in level.get_sprites() if "bsT" in s.tags):
        center = target.pixels[1][1]
        for dx, dy in _RING:
            at = (target.x + PITCH * dx, target.y + PITCH * dy)
            if at not in tiles:
                continue
            same = target.pixels[1 + dy][1 + dx] == 0
            n = next(k for k in range(len(colors)) if (colors[k] == center) == same)
            clicks += [at] * n
    scale = min(64 // level.grid_size[0], 64 // level.grid_size[1])
    return [ActionInput(id=GameAction.ACTION6, data={"x": (x + 1) * scale, "y": (y + 1) * scale}) for x, y in clicks]


def generate(rng: random.Random):
    from arcengine import Level

    sprites = _MODULE.sprites
    quadrants = [(0, 0), (16, 0), (0, 16), (16, 16)]
    qx, qy = rng.choice(quadrants)
    others = [q for q in quadrants if q != (qx, qy)]
    # The half background covers the two quadrants in the row without the puzzle, the quarter the other one in its row.
    half_y = 16 - qy
    quarter = (16 - qx, max(0, qy - 2))
    cx, cy = qx + 2 + PITCH + rng.randint(-1, 1), qy + 2 + PITCH + rng.randint(-1, 1)
    clicked = [d for d in _RING if _needs_click(*d)]
    plain = [d for d in _RING if not _needs_click(*d)]
    dropped = rng.sample(plain, rng.choice((0, 0, 1, 1, 2)))
    if rng.random() < 0.25:
        dropped = dropped[:1] + [rng.choice(clicked)]
    ring = [d for d in _RING if d not in dropped]
    order = rng.sample(EXAMPLES, len(EXAMPLES))
    placed = [sprites[QUARTER].clone().set_position(*quarter), sprites[HALF].clone().set_position(0, half_y)]
    placed += [sprites[TILE].clone().set_position(cx + PITCH * dx, cy + PITCH * dy) for dx, dy in ring]
    for (tiles, pattern), (ex, ey) in zip(order, others, strict=False):
        x, y = ex + rng.randint(1, 4), ey + rng.randint(1, 4)
        placed += [sprites[tiles].clone().set_position(x, y), sprites[pattern].clone().set_position(x + PITCH, y + PITCH)]
    placed += [sprites[FRAME].clone().set_position(qx, qy), sprites[BRACKETS].clone().set_position(qx, qy),
               sprites[PATTERN].clone().set_position(cx, cy)]
    return Level(sprites=placed, grid_size=_LEVEL0.grid_size, data=dict(_LEVEL0._data), name=_LEVEL0.name)
