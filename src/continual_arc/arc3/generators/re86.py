"""re86 at level 0: two plus-shaped crosses of two colors lie on the board over a picture of eight small framed dots,
four of each cross's color. ACTION1-4 slide the selected cross (its center black) three cells up, down, left or
right, ACTION5 passes the selection to the other cross, and the level is won when every dot is covered by its own
color's cross (the later cross drawn on top); 100 actions lose.

A layout picks where each cross must end, anywhere on the board, and draws the picture from level 0's own dots: one
dot of the cross's color on each of its four arms, at four to eleven or thirteen cells from its center, which pins
that end place. Each cross then starts a whole number of slides away from its end, so the picture is reachable by
construction; layouts whose end picture the other cross spoils, whose dots overlap, or whose shortest solution is far
from level 0's are redrawn.

solve reads the crosses and the dots from the level, finds every place a cross covers all its color's dots, checks
each pairing of such places on the game's own canvas, and plays the cheapest: the selected cross's slides, one
ACTION5, the other's slides; its answer is checked in the real engine."""
from __future__ import annotations

import itertools
import random

import numpy as np
from arcengine import ActionInput, GameAction, Level

from continual_arc.arc3 import levels

GAME = "re86"
CROSS_TAG, GOAL_TAG = "0031cppcuvqlbi", "0054xnsuqceejm"
GOAL, CANVAS = "0053jffbwmoiua", "0000wbshgbbxxc"
STEP = 3  # cells a slide moves a cross (the game's ognwltlfta)
FRAME, SELECTED = 4, 0  # a dot's frame color, which the win test ignores; the selected cross's center color
NEAREST = 4  # a dot's least distance from its cross's center (level 0's are five to eleven)
LENGTH = (14, 28)  # shortest solution lengths kept; level 0's is 20
TRIES = 400

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_LEVEL0 = _MODULE.levels[0]


def _crosses(level: Level) -> list:
    return [s for s in level.get_sprites() if CROSS_TAG in s.tags]


def _color(cross) -> int:
    p = cross.pixels
    return int(p[(p != SELECTED) & (p != -1)][0])


def _dots(level: Level) -> dict[tuple[int, int], int]:
    """The picture's dots: board cell -> the color the win test asks of it."""
    out = {}
    for goal in (s for s in level.get_sprites() if GOAL_TAG in s.tags):
        ys, xs = np.nonzero((goal.pixels != -1) & (goal.pixels != FRAME))
        out.update({(int(x) + goal.x, int(y) + goal.y): int(goal.pixels[y, x]) for y, x in zip(ys, xs, strict=False)})
    return out


def _center(cross) -> tuple[int, int]:
    return cross.x + cross.width // 2, cross.y + cross.height // 2


def _covers(center: tuple[int, int], half: int, cell: tuple[int, int]) -> bool:
    return (cell[0] == center[0] and abs(cell[1] - center[1]) <= half) or \
        (cell[1] == center[1] and abs(cell[0] - center[0]) <= half)


def _canvas_ok(crosses, centers, selected: int, dots) -> bool:
    """The game's win test with the crosses at ``centers`` and cross ``selected`` selected: paint every cross in
    order on the blank canvas, then compare every dot."""
    canvas = _SPRITES[CANVAS].pixels.astype(int).copy()
    for i, (cross, (cx, cy)) in enumerate(zip(crosses, centers, strict=False)):
        p = cross.pixels.astype(int).copy()
        p[cross.height // 2, cross.width // 2] = SELECTED if i == selected else _color(cross)
        x0, y0 = cx - cross.width // 2, cy - cross.height // 2
        for (y, x) in zip(*np.nonzero(p != -1), strict=False):
            if 0 <= x0 + x < 64 and 0 <= y0 + y < 64:
                canvas[y0 + y, x0 + x] = p[y, x]
    return all(canvas[y, x] == c for (x, y), c in dots.items())


def _route(start: tuple[int, int], end: tuple[int, int]) -> list[GameAction]:
    dx, dy = (end[0] - start[0]) // STEP, (end[1] - start[1]) // STEP
    return [GameAction.ACTION4 if dx > 0 else GameAction.ACTION3] * abs(dx) + \
        [GameAction.ACTION2 if dy > 0 else GameAction.ACTION1] * abs(dy)


def _plan(level: Level) -> list[GameAction] | None:
    """The cheapest winning play: crosses visited in ACTION5's cyclic order from the selected one, each slid
    straight to its end place."""
    crosses, dots = _crosses(level), _dots(level)
    colors = [_color(c) for c in crosses]
    if len(set(colors)) != len(colors):
        return None
    starts = [_center(c) for c in crosses]
    selected = next(i for i, c in enumerate(crosses) if c.pixels[c.height // 2, c.width // 2] == SELECTED)
    options = []
    for cross, color, (sx, sy) in zip(crosses, colors, starts, strict=False):
        half, mine = cross.width // 2, [cell for cell, c in dots.items() if c == color]
        options.append([(x, y) for y in range(sy % STEP, 64, STEP) for x in range(sx % STEP, 64, STEP)
                        if all(_covers((x, y), half, cell) for cell in mine)])
    best = None
    n = len(crosses)
    for ends in itertools.product(*options):
        visit = [(selected + k) % n for k in range(n)]
        moving = [k for k, i in enumerate(visit) if ends[i] != starts[i]]
        last = visit[moving[-1]] if moving else selected
        if not _canvas_ok(crosses, ends, last, dots):
            continue
        play = []
        for k, i in enumerate(visit[:moving[-1] + 1] if moving else []):
            if k:
                play.append(GameAction.ACTION5)
            play.extend(_route(starts[i], ends[i]))
        if best is None or len(play) < len(best):
            best = play
    return best


def _dot_block(color: int) -> np.ndarray:
    """Level 0's own 3 x 3 dot of ``color``, cut from its picture."""
    p = _SPRITES[GOAL].pixels
    ys, xs = np.nonzero(p == color)
    return p[ys[0] - 1:ys[0] + 2, xs[0] - 1:xs[0] + 2].copy()


def generate(rng: random.Random) -> Level | None:
    template = [s for s in _LEVEL0.get_sprites() if CROSS_TAG in s.tags]
    names, halves = [s.name for s in template], [s.width // 2 for s in template]
    colors = [_color(_SPRITES[name]) for name in names]
    for _ in range(TRIES):
        ends, cells = [], []
        for half in halves:
            ex, ey = rng.randrange(2, 62), rng.randrange(2, 61)
            arms = []
            for ux, uy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                far = [d for d in range(NEAREST, half + 1) if 1 <= ex + ux * d <= 62 and 1 <= ey + uy * d <= 61]
                if not far:
                    break
                d = rng.choice(far)
                arms.append((ex + ux * d, ey + uy * d))
            if len(arms) < 4:
                break
            ends.append((ex, ey))
            cells.append(arms)
        if len(ends) < len(halves):
            continue
        flat = [c for arms in cells for c in arms]
        if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 4 for a, b in itertools.combinations(flat, 2)):
            continue
        starts = []
        for ex, ey in ends:
            xs = [x for x in range(ex % STEP, 64, STEP) if 2 <= x <= 61]
            ys = [y for y in range(ey % STEP, 64, STEP) if 2 <= y <= 61]
            starts.append((rng.choice(xs), rng.choice(ys)))
        if any(abs(s[0] - e[0]) + abs(s[1] - e[1]) < 2 * STEP for s, e in zip(starts, ends, strict=False)):
            continue
        goal = _SPRITES[GOAL].clone()
        pixels = np.full_like(goal.pixels, -1)
        for color, arms in zip(colors, cells, strict=False):
            block = _dot_block(color)
            for x, y in arms:
                pixels[y - 1:y + 2, x - 1:x + 2] = block
        goal.pixels = pixels
        sprites = [_SPRITES[name].clone().set_position(sx - half, sy - half)
                   for name, half, (sx, sy) in zip(names, halves, starts, strict=False)]
        level = Level(sprites=[*sprites, goal], grid_size=(64, 64), data=dict(_LEVEL0._data))
        play = _plan(level)
        if play is None or not LENGTH[0] <= len(play) <= LENGTH[1]:
            continue
        return level
    return None


def solve(level: Level) -> list[ActionInput] | None:
    play = _plan(level)
    if play is None:
        return None
    actions = [ActionInput(id=a) for a in play]
    return actions if levels.replay(GAME, level, actions) else None
