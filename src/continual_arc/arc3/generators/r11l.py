"""r11l at level 0: a body sits at the centroid of its two legs, and a target ring waits elsewhere on a walled board.
Clicking a leg (ACTION6) selects it (the leg nearest the top-left corner starts selected), and clicking open floor
moves the selected leg there unless it would touch the wall, the body following to the legs' new centroid; the level
is won when the body touches the target, and 60 actions lose.

A layout turns and mirrors the wall at random, puts the target anywhere clear of it and the two legs anywhere on open
floor, and keeps it only if neither leg alone can carry the body onto the target while one moving both can, so its
shortest solution is level 0's three clicks (move, select, move).

solve searches the clicks over masks of the wall, legs, body and target read from the game's own sprites: one move of
the selected leg, then a select and one move of the other, then a move of each, the first that lands the body on the
target, and its answer is checked in the real engine."""
from __future__ import annotations

import random

import numpy as np
from arcengine import ActionInput, GameAction, Level

from continual_arc.arc3 import levels

GAME = "r11l"
TARGET, BODY, LEG, WALL = "flkdtg-pumlzd", "roefwu-pumlzd", "roefwulewcui-pumlzd", "wakneh-Level1"
LEG_SIZE, TARGET_SIZE = 5, 7
CLICK_X, CLICK_Y = (3, 62), (2, 62)  # clicks that keep a leg on the board and off the step bar in column 0
PAD = 24  # the wall mask's margin around the 64 x 64 board
TRIES = 400
SPREAD = 8  # how far either side of the target solve sets the legs

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_LEVEL0 = _MODULE.levels[0]


def _mask(sprite) -> np.ndarray:
    return sprite.render() != -1


_LEG = _mask(_SPRITES[LEG])
_BODY = _mask(_SPRITES[BODY])
_TARGET = _mask(_SPRITES[TARGET])
# The body's top-left offsets from the target's top-left at which their pixels touch.
_HIT = {(dx, dy) for dx in range(-LEG_SIZE + 1, TARGET_SIZE) for dy in range(-LEG_SIZE + 1, TARGET_SIZE)
        if np.any(_BODY[max(0, -dy):min(LEG_SIZE, TARGET_SIZE - dy), max(0, -dx):min(LEG_SIZE, TARGET_SIZE - dx)]
                  & _TARGET[max(0, dy):min(TARGET_SIZE, LEG_SIZE + dy), max(0, dx):min(TARGET_SIZE, LEG_SIZE + dx)])}


def _wall_canvas(wall) -> np.ndarray:
    """The wall's pixels on the board, with a margin of PAD all round: canvas[y + PAD, x + PAD]."""
    canvas = np.zeros((64 + 2 * PAD, 64 + 2 * PAD), bool)
    m = _mask(wall)
    h, w = m.shape
    canvas[wall.y + PAD:wall.y + PAD + h, wall.x + PAD:wall.x + PAD + w] |= m
    return canvas


def _free(canvas: np.ndarray, mask: np.ndarray, origin: int = 0) -> np.ndarray:
    """free[y, x]: a sprite of ``mask`` with its top-left at (x - origin, y - origin), for x, y in 0..63, touches no
    wall pixel."""
    h, w = mask.shape
    view = np.lib.stride_tricks.sliding_window_view(canvas[PAD - origin:PAD - origin + 63 + h,
                                                           PAD - origin:PAD - origin + 63 + w], (h, w))
    return ~np.any(view & mask, axis=(2, 3))


def _leg_free(canvas: np.ndarray) -> np.ndarray:
    """ok[cy, cx]: a click at (cx, cy) may move a leg there (its top-left at (cx - 2, cy - 2)) clear of the wall."""
    ok = np.zeros((64, 64), bool)
    free = _free(canvas, _LEG, LEG_SIZE // 2)
    ok[slice(*CLICK_Y), slice(*CLICK_X)] = free[slice(*CLICK_Y), slice(*CLICK_X)]
    return ok


def _in_box(cx: np.ndarray, cy: np.ndarray, leg: tuple[int, int]) -> np.ndarray:
    return (leg[0] <= cx) & (cx < leg[0] + LEG_SIZE) & (leg[1] <= cy) & (cy < leg[1] + LEG_SIZE)


def _body(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """The body's top-left over legs with top-lefts a and b, as the game places it."""
    half = LEG_SIZE // 2
    return ((a[0] + half + b[0] + half) // 2 - half, (a[1] + half + b[1] + half) // 2 - half)


def _hits(body: tuple[int, int], target: tuple[int, int]) -> bool:
    return (body[0] - target[0], body[1] - target[1]) in _HIT


def _click(x: int, y: int) -> ActionInput:
    return ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)})


def _parts(level: Level):
    target = next(s for s in level.get_sprites() if s.name == TARGET)
    wall = next(s for s in level.get_sprites() if s.name == WALL)
    legs = sorted(((s.x, s.y) for s in level.get_sprites() if s.name == LEG), key=lambda p: (p[0] ** 2 + p[1] ** 2) ** 0.5)
    return (target.x, target.y), wall, legs


def _one_leg(ok: np.ndarray, moving: tuple[int, int], still: tuple[int, int], target: tuple[int, int],
             blocked: list[tuple[int, int]]) -> tuple[int, int] | None:
    """A click that moves ``moving`` so the body lands on the target, ``still`` staying put; clicks inside a leg in
    ``blocked`` would select it instead."""
    half = LEG_SIZE // 2
    for cy in range(*CLICK_Y):
        for cx in range(*CLICK_X):
            if ok[cy, cx] and not any(_in_box(cx, cy, b) for b in blocked) \
                    and _hits(_body((cx - half, cy - half), still), target):
                return cx, cy
    return None


def _select_click(leg: tuple[int, int], before: list[tuple[int, int]]) -> tuple[int, int] | None:
    """A pixel of ``leg``'s box in no box of the legs the game checks before it."""
    for dy in (2, 1, 3, 0, 4):
        for dx in (2, 1, 3, 0, 4):
            cx, cy = leg[0] + dx, leg[1] + dy
            if not any(_in_box(cx, cy, b) for b in before):
                return cx, cy
    return None


def _plan(target, wall, legs, ok=None) -> tuple[int, list[tuple[int, int]]] | None:
    """The shortest click plan from the reset, as (length, clicks): 1 moves the selected leg, 2 selects and moves the
    other, 3 moves both."""
    ok = _leg_free(_wall_canvas(wall)) if ok is None else ok
    first, second = legs
    half = LEG_SIZE // 2
    c = _one_leg(ok, first, second, target, legs)
    if c:
        return 1, [c]
    sel = _select_click(second, [first])
    if sel:
        c = _one_leg(ok, second, first, target, legs)
        if c:
            return 2, [sel, c]
    # Both legs move: the body must reach a top-left that touches the target, so the two click centers sum to twice
    # its center (give or take the floor of the halving). The first leg is tried nearest a spot a little way from the
    # target on its own side, so the legs end astride the target as a person would set them.
    cys, cxs = np.nonzero(ok)
    tcx, tcy = target[0] + TARGET_SIZE // 2, target[1] + TARGET_SIZE // 2
    away = np.array([first[0] + half - tcx, first[1] + half - tcy], float)
    ideal = np.array([tcx, tcy]) + SPREAD * away / max(np.hypot(*away), 1.0)
    order = np.lexsort((cxs, cys, np.round((cxs - ideal[0]) ** 2 + (cys - ideal[1]) ** 2, 6)))
    cxs, cys = cxs[order], cys[order]
    goals = sorted(((target[0] + dx, target[1] + dy) for dx, dy in _HIT),
                   key=lambda b: ((b[0] + half - tcx) ** 2 + (b[1] + half - tcy) ** 2, b))
    usable = ~(_in_box(cxs, cys, first) | _in_box(cxs, cys, second))
    for bx, by in goals:
        for px in (0, 1):
            for py in (0, 1):
                qx = 2 * (bx + half) + px - cxs
                qy = 2 * (by + half) + py - cys
                inside = (qx >= 0) & (qx < 64) & (qy >= 0) & (qy < 64)
                good = usable & inside
                good[good] &= ok[qy[good], qx[good]]
                for i in np.nonzero(good)[0]:
                    p = (int(cxs[i]), int(cys[i]))
                    moved = (p[0] - half, p[1] - half)
                    q = (int(qx[i]), int(qy[i]))
                    if _in_box(q[0], q[1], moved) or _in_box(q[0], q[1], second):
                        continue
                    if _hits(_body(moved, second), target):
                        continue
                    sel = _select_click(second, [moved])
                    if sel is None or _body(moved, (q[0] - half, q[1] - half)) != (bx, by):
                        continue
                    return 3, [p, sel, q]
    return None


def generate(rng: random.Random) -> Level | None:
    half = LEG_SIZE // 2
    for _ in range(TRIES):
        rotation, mirror = rng.choice((0, 90, 180, 270)), rng.random() < 0.5
        wall = _SPRITES[WALL].clone().set_rotation(rotation).set_mirror_lr(mirror)
        h, w = wall.render().shape
        wall.set_position(-(w - 64) // 2 + rng.randint(-2, 2), -(h - 64) // 2 + rng.randint(-2, 2))
        canvas = _wall_canvas(wall)
        ok = _leg_free(canvas)
        target_free = _free(canvas, _TARGET)
        tys, txs = np.nonzero(target_free[1:58, 1:58])
        cys, cxs = np.nonzero(ok)
        if not len(txs) or len(cxs) < 2:
            continue
        i = rng.randrange(len(txs))
        target = (int(txs[i]) + 1, int(tys[i]) + 1)
        a, b = rng.sample(range(len(cxs)), 2)
        legs = [(int(cxs[a]) - half, int(cys[a]) - half), (int(cxs[b]) - half, int(cys[b]) - half)]
        if abs(legs[0][0] - legs[1][0]) < LEG_SIZE + 2 and abs(legs[0][1] - legs[1][1]) < LEG_SIZE + 2:
            continue
        if legs[0][0] ** 2 + legs[0][1] ** 2 == legs[1][0] ** 2 + legs[1][1] ** 2:
            continue
        body = _body(*legs)
        body_free = _free(canvas, _BODY)
        if not (0 <= body[0] < 64 and 0 <= body[1] < 64 and body_free[body[1], body[0]]) or _hits(body, target):
            continue
        legs.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2) ** 0.5)
        plan = _plan(target, wall, legs, ok)
        if plan is None or plan[0] != 3:
            continue
        sprites = [_SPRITES[TARGET].clone().set_position(*target), _SPRITES[BODY].clone().set_position(*body),
                   *(_SPRITES[LEG].clone().set_position(*leg) for leg in legs), wall]
        return Level(sprites=sprites, grid_size=(64, 64), data=dict(_LEVEL0._data))
    return None


def solve(level: Level) -> list[ActionInput] | None:
    plan = _plan(*_parts(level))
    if plan is None:
        return None
    actions = [_click(*c) for c in plan[1]]
    return actions if levels.replay(GAME, level, actions) else None
