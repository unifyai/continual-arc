"""s5i5 at level 0: two arms, one of color 11 that grows downward from its base and one of color 14 that grows
rightward, each carrying a tip at its end, and two control panels, each showing one arm's color. A click (ACTION6) on
the far half of a panel (its lower half if upright, its right half if lying) lengthens its arm by one 3-pixel unit, a
click on the near half shortens it to no less than one unit, and a change that would make two arms overlap is undone;
an off-screen block in line with each arm stops it at the board's edge. The level is won when every target holds a
tip; each click spends one of 50 steps.

generate keeps level 0's sprites as they are, arms at their own lengths, and lays them out anew: it puts each arm's
base anywhere on a 3-pixel lattice where the two arms' full reaches never cross, a target on each arm's line within
its reach, its block in line just past the board's edge, and the two panels anywhere clear of both reaches, the
targets and each other. Each target is on its arm's line within reach and the arms cannot meet, so every layout is
solvable; one is kept when its shortest solution is 9 to 18 clicks (level 0's is 13).

solve reads the arms, their tips, the panels and which arms each panel moves from the game after its reset, and runs
a breadth-first search over the arms' lengths by the game's rules: a click moves every arm its panel shows, an overlap
of two arms undoes it, and tips move with their arm's free end. It takes milliseconds."""
from __future__ import annotations

import random
from collections import deque

from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "s5i5"
DOWN_ARM, RIGHT_ARM, BLOCK = "0000amgeiiukhc", "0002ljvyffrskg", "0016dewzzopnkn"
TIP, TARGET = "0063ylopfyonpu", "0086sibroshteg"
UPRIGHT_PANEL, LYING_PANEL = "0070lwmhddtryt", "0069zlulamypif"  # the panels showing colors 11 and 14
ARM_TAG, TIP_TAG, PANEL_TAG, TARGET_TAG = "0001qwdmnlybkb", "0064ocqkuqacti", "0066ghlkyvdbgg", "0087vvmblxkzdi"
UNIT = 3      # an arm's unit of length (vjqemmsfmx)
EDGE = 64     # the board's side; the blocks sit just past it, as in level 0
LAST = 60     # the last lattice place a 3 x 3 tip fits above the step bar in row 63
LENGTH = (9, 18)
TRIES = 400
MAX_STATES = 100_000
# Each direction an arm can grow (gnpdxxlhrp's rotation): the unit step of its free end.
GROWTH = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_BASE_LAYOUT = levels.layout(_BASE)


def _rect(base: tuple, growth: int, units: int) -> tuple:
    """An arm's box (x0, y0, x1, y1), exclusive ends, from the corner its fixed end sits at."""
    x, y = base
    gx, gy = GROWTH[growth]
    if gx:
        return (x, y, x + UNIT * units, y + UNIT) if gx > 0 else (x - UNIT * (units - 1), y, x + UNIT, y + UNIT)
    return (x, y, x + UNIT, y + UNIT * units) if gy > 0 else (x, y - UNIT * (units - 1), x + UNIT, y + UNIT)


def _overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _model(level) -> tuple:
    """The level as solve searches it, read from its game after the reset: (arms, panels, tips, targets). An arm is
    (fixed-end corner, growth, units); a panel is (its arms, its lengthening click, its shortening click); a tip is
    (its arm, its place at the arm's start length)."""
    assert not level.get_data("Children"), "tips ride only on the arm they touch"
    game, _ = levels.new_game(GAME, level)
    sprites = game.current_level.get_sprites_by_tag(ARM_TAG)
    arms = []
    for s in sprites:
        growth = game.gnpdxxlhrp(s)
        units = max(s.width, s.height) // UNIT
        gx, gy = GROWTH[growth]
        corner = (s.x + (s.width - UNIT if gx < 0 else 0), s.y + (s.height - UNIT if gy < 0 else 0))
        arms.append((corner, growth, units))
    panels = []
    for p in game.current_level.get_sprites_by_tag(PANEL_TAG):
        moved = tuple(sprites.index(s) for s in game.pigtralzpb[p])
        if not moved:
            continue
        if p.width > p.height:
            near, far = (p.x + 1, p.y + p.height // 2), (p.x + p.width - 2, p.y + p.height // 2)
        else:
            near, far = (p.x + p.width // 2, p.y + 1), (p.x + p.width // 2, p.y + p.height - 2)
        panels.append((moved, far, near))
    tips = [(i, (t.x, t.y)) for i, s in enumerate(sprites) for t in game.uricqfoplr[s]]
    targets = frozenset((t.x, t.y) for t in game.current_level.get_sprites_by_tag(TARGET_TAG))
    return arms, panels, tips, targets


def _search(arms: list, panels: list, tips: list, targets: frozenset) -> list | None:
    """The shortest clicks, as display places, that bring a tip onto every target."""
    start = tuple(units for _, _, units in arms)

    def tip_places(state: tuple) -> set:
        out = set()
        for i, (x, y) in tips:
            gx, gy = GROWTH[arms[i][1]]
            d = UNIT * (state[i] - start[i])
            out.add((x + gx * d, y + gy * d))
        return out

    parent = {start: None}
    queue = deque([start])
    while queue and len(parent) < MAX_STATES:
        state = queue.popleft()
        for moved, far, near in panels:
            for click, delta in ((far, 1), (near, -1)):
                nxt = list(state)
                for i in moved:
                    if delta > 0 or nxt[i] > 1:
                        nxt[i] += delta
                boxes = [_rect(corner, growth, n) for (corner, growth, _), n in zip(arms, nxt, strict=True)]
                if any(_overlap(a, b) for k, a in enumerate(boxes) for b in boxes[k + 1:]):
                    continue
                nxt = tuple(nxt)
                if nxt in parent:
                    continue
                parent[nxt] = (state, click)
                if targets <= tip_places(nxt):
                    path = []
                    while parent[nxt] is not None:
                        nxt, click = parent[nxt]
                        path.append(click)
                    return path[::-1]
                queue.append(nxt)
    return None


def solve(level) -> list | None:
    path = _search(*_model(level))
    if path is None:
        return None
    return [ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}) for x, y in path]


def _place(rng: random.Random, size: tuple, taken: list) -> tuple | None:
    """A free place for a panel of ``size`` (w, h) on the board, a pixel clear of every box in ``taken``."""
    w, h = size
    for _ in range(100):
        x, y = rng.randint(0, EDGE - w), rng.randint(0, EDGE - 1 - h)
        box = (x - 1, y - 1, x + w + 1, y + h + 1)
        if not any(_overlap(box, t) for t in taken):
            return x, y
    return None


def generate(rng: random.Random):
    down, right = _SPRITES[DOWN_ARM], _SPRITES[RIGHT_ARM]
    down_units, right_units = down.height // UNIT, right.width // UNIT
    lattice = range(0, LAST + 1, UNIT)
    for _ in range(TRIES):
        down_at = (rng.choice(lattice), rng.choice(lattice[: -down_units]))
        right_at = (rng.choice(lattice[: -right_units]), rng.choice(lattice))
        down_reach = (EDGE - down_at[1]) // UNIT
        right_reach = (EDGE - right_at[0]) // UNIT
        down_box = _rect(down_at, 180, down_reach)
        right_box = _rect(right_at, 90, right_reach)
        if _overlap(down_box, right_box):
            continue
        down_goal = rng.choice([n for n in range(2, down_reach + 1) if n != down_units])
        right_goal = rng.choice([n for n in range(2, right_reach + 1) if n != right_units])
        if not LENGTH[0] <= abs(down_goal - down_units) + abs(right_goal - right_units) <= LENGTH[1]:
            continue
        down_target = (down_at[0], down_at[1] + UNIT * (down_goal - 1))
        right_target = (right_at[0] + UNIT * (right_goal - 1), right_at[1])
        taken = [down_box, right_box]
        upright = _place(rng, (_SPRITES[UPRIGHT_PANEL].width, _SPRITES[UPRIGHT_PANEL].height), taken)
        if upright is None:
            continue
        taken.append((*upright, upright[0] + _SPRITES[UPRIGHT_PANEL].width, upright[1] + _SPRITES[UPRIGHT_PANEL].height))
        lying = _place(rng, (_SPRITES[LYING_PANEL].width, _SPRITES[LYING_PANEL].height), taken)
        if lying is None:
            continue
        sprites = [
            down.clone().set_position(*down_at),
            right.clone().set_position(*right_at),
            _SPRITES[BLOCK].clone().set_position(down_at[0], EDGE),
            _SPRITES[BLOCK].clone().set_position(EDGE, right_at[1]),
            _SPRITES[TIP].clone().set_position(down_at[0], down_at[1] + UNIT * (down_units - 1)),
            _SPRITES[TIP].clone().set_position(right_at[0] + UNIT * (right_units - 1), right_at[1]),
            _SPRITES[LYING_PANEL].clone().set_position(*lying),
            _SPRITES[UPRIGHT_PANEL].clone().set_position(*upright),
            _SPRITES[TARGET].clone().set_position(*down_target),
            _SPRITES[TARGET].clone().set_position(*right_target),
        ]
        level = type(_BASE)(sprites=sprites, grid_size=_BASE.grid_size, data=dict(_BASE._data), name=_BASE.name)
        if levels.layout(level) == _BASE_LAYOUT:
            continue
        return level
    return None
