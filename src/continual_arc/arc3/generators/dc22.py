"""dc22 at level 0: a 2 x 2 player walks a path of floor pieces on the left half of a 64 x 44 board, two pixels a
move (ACTION1-4), and reaching the goal on the far pad wins. Clicking a button on the right-hand panel (ACTION6)
toggles every piece of its letter: button b turns each checkered wall tile into floor and each floor tile into a hole,
and button a swings the bridge between its horizontal arm and its vertical arm, which share the corner they turn on.
A move off the floor or into a wall does nothing, a click that drops the player into a hole costs 20 of the step
counter and is undone, and the counter (128 at level 0, a move one step and a click two) loses the game when it runs
out. Level 0 needs b to open the wall, b again to close it and reopen the far tile, and a once to turn the bridge.

generate places the bridge anywhere in the play area, grows a corridor of 2 x 2 tiles from the left end of its
horizontal arm and another from the top of its vertical arm, each a random induced walk (two to four tiles in all, as
level 0 has three), and ends each with a 6 x 6 pad: one holds the player, the other the goal, either way round, so
the bridge is turned once or twice. Two or three of the tiles are b tiles, the first met on the way a wall and at
least one after it an open tile, so b is pressed more than once; the rest are plain floor. No piece touches any but its
neighbors on the path, so the path is the only way. A layout is kept when its shortest win is 14 to 28 actions (they
come out 18 to 24, level 0's 20 the median) and replays to a win; its step counter is level 0's 128 scaled by the win's
step cost over level 0's 23.

solve builds the game, reads the floor and the walls from its sprites under each combination of button presses (the
toggle as the game's click does it), and runs a breadth-first search over (player position, presses of each button
mod 2): a move is allowed onto floor clear of walls, a click only if the player still stands on floor after it. It
returns the shortest action sequence in well under a second.
"""
from __future__ import annotations

import copy
import itertools
import random
from collections import deque

import numpy as np
from arcengine import ActionInput, GameAction
from arcengine.enums import BlockingMode, InteractionMode

from continual_arc.arc3 import levels

GAME = "dc22"

U = 2                      # a move, in pixels: the unit of every position below
COLS, ROWS = 16, 22        # the play area left of the button panel, in moves (32 x 44 pixels)
PAD, TILE, SHUT, OPEN, BRIDGE = "merged-sprite-2", "tacugo-plelvb", "tovemc-plelvb1", "tovemc-plelvb2", "refgps-plelvb1"
SIZE = {PAD: 3, TILE: 2, SHUT: 2, OPEN: 2, BRIDGE: 5}
TILES = (2, 4)             # tiles in the two corridors together, at least one each (level 0: 3)
LENGTH = (14, 28)          # the shortest win in actions
BASE_COST = 23             # level 0's win in steps: 17 moves and 3 clicks of two
TRIES = 400

MOVES = ((GameAction.ACTION1, 0, -1), (GameAction.ACTION2, 0, 1), (GameAction.ACTION3, -1, 0),
         (GameAction.ACTION4, 1, 0))
DIRS = tuple((dx, dy) for _, dx, dy in MOVES)

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_DATA = dict(_BASE._data)
_BASE_LAYOUT = levels.layout(_BASE)
_PANEL = [s for s in _BASE.get_sprites() if s.x >= 32]   # buttons, their hint frames, the panel and its divider


def _box(kind: str, x: int, y: int) -> set:
    n = SIZE[kind]
    return {(x + i, y + j) for i in range(n) for j in range(n)}


def _fits(cells: set, taken: set, neighbor: set) -> bool:
    """Whether a piece on ``cells`` lies in the play area, overlaps nothing and touches only ``neighbor``."""
    if any(not (0 <= x < COLS and 0 <= y < ROWS) or (x, y) in taken for x, y in cells):
        return False
    return all((x + dx, y + dy) in neighbor or (x + dx, y + dy) not in taken or (x + dx, y + dy) in cells
               for x, y in cells for dx, dy in DIRS)


def _leg(rng: random.Random, first: tuple, heading: tuple, length: int, taken: set, anchor: set) -> list | None:
    """``length`` tiles from ``first`` (touching only ``anchor``), each one tile on from the last, then a pad: the
    tiles' and the pad's top-left corners, or None when the walk gets stuck."""
    tiles, last_cells = [], anchor
    pos, step = first, heading
    for k in range(length):
        if k:
            options = [(dx, dy) for dx, dy in DIRS if (dx, dy) != (-step[0], -step[1])]
            if rng.random() < 0.5 and step in options:
                options = [step]
            rng.shuffle(options)
            for dx, dy in options:
                nxt = (pos[0] + 2 * dx, pos[1] + 2 * dy)
                if _fits(_box(TILE, *nxt), taken, last_cells):
                    pos, step = nxt, (dx, dy)
                    break
            else:
                return None
        elif not _fits(_box(TILE, *pos), taken, last_cells):
            return None
        tiles.append(pos)
        last_cells = _box(TILE, *pos)
        taken |= last_cells
    x, y = pos
    pads = []
    for dx, dy in DIRS:
        for o in (0, 1):
            corner = {(1, 0): (x + 2, y - o), (-1, 0): (x - 3, y - o), (0, 1): (x - o, y + 2), (0, -1): (x - o, y - 3)}
            pads.append(corner[(dx, dy)])
    rng.shuffle(pads)
    for pad in pads:
        if _fits(_box(PAD, *pad), taken, last_cells):
            taken |= _box(PAD, *pad)
            return [*tiles, pad]
    return None


def _build(pieces: list, start_pad: tuple, goal_pad: tuple, counter: int) -> object:
    sprites = [s.clone() for s in _PANEL]
    sx, sy = start_pad
    gx, gy = goal_pad
    sprites.append(_SPRITES["goknoi"].clone().set_position(U * gx + 2, U * gy + 2))
    sprites.append(_SPRITES["plflho1"].clone().set_position(U * sx + 2, U * sy + 2))
    sprites += [_SPRITES[kind].clone().set_position(U * x, U * y) for kind, x, y in pieces]
    sprites.sort(key=lambda s: (s.name, s.x, s.y))
    return type(_BASE)(sprites=sprites, grid_size=_BASE.grid_size, data={**_DATA, "StepCounter": counter})


def _draw(rng: random.Random) -> object | None:
    bx, by = rng.randint(2, COLS - 5), rng.randint(2, ROWS - 5)
    arm = {(bx + i, by + j) for i in range(5) for j in (3, 4)}
    upright = {(bx + i, by + j) for i in (3, 4) for j in range(5)}
    taken = _box(BRIDGE, bx, by)
    tiles = rng.randint(*TILES)
    on_left = rng.randint(1, tiles - 1)
    left = _leg(rng, (bx - 2, by + 3), (-1, 0), on_left, taken, arm)
    if left is None:
        return None
    top = _leg(rng, (bx + 3, by - 2), (0, -1), tiles - on_left, taken, upright)
    if top is None:
        return None
    if rng.random() < 0.5:
        start_leg, goal_leg = left, top
    else:
        start_leg, goal_leg = top, left
    route = [*start_leg[-2::-1], *goal_leg[:-1]]          # the tiles from the start pad to the goal pad
    shut = rng.randrange(len(route))
    later = list(range(shut + 1, len(route)))
    if not later:
        return None
    opened = set(rng.sample(later, rng.randint(1, min(2, len(later)))))
    kinds = [SHUT if k == shut else OPEN if k in opened else TILE for k in range(len(route))]
    pieces = [(BRIDGE, bx, by), (PAD, *start_leg[-1]), (PAD, *goal_leg[-1])]
    pieces += [(kind, *pos) for kind, pos in zip(kinds, route, strict=True)]
    level = _build(pieces, start_leg[-1], goal_leg[-1], _DATA["StepCounter"])
    solution = solve(level)
    if solution is None or not LENGTH[0] <= len(solution) <= LENGTH[1]:
        return None
    cost = sum(2 if a.id == GameAction.ACTION6 else 1 for a in solution)
    level = _build(pieces, start_leg[-1], goal_leg[-1], round(_DATA["StepCounter"] * cost / BASE_COST))
    if levels.layout(level) == _BASE_LAYOUT or not levels.replay(GAME, level, solution):
        return None
    return level


def generate(rng: random.Random):
    for _ in range(TRIES):
        level = _draw(rng)
        if level is not None:
            return level
    return None


def _toggle(game, letter: str) -> None:
    """What clicking button ``letter`` does to the pieces, as the game's step does it: each piece of the letter gives
    way to its next variant at the same place, whose tags set how it interacts."""
    (button,) = [s for s in game.current_level.get_sprites_by_tag("buezna") if letter in s.tags]
    for piece in game.ilvrmetiiv(letter, button):
        name = game.lcdavdtabp(piece.name)
        other = game.rcorpfmgqi(piece.x, piece.y, name) if name else None
        if other is None:
            continue
        piece.set_interaction(InteractionMode.REMOVED)
        if "omvz" in other.tags or "buezna" in other.tags:
            other.set_interaction(InteractionMode.INTANGIBLE)
        elif "inzejtible" in other.tags:
            other.set_interaction(InteractionMode.INVISIBLE)
        else:
            other.set_interaction(InteractionMode.TANGIBLE)


def _masks(game, player) -> tuple:
    """(wall, floor) over the grid: pixels of sprites the player collides with, and of sprites it may stand on."""
    width, height = game.current_level.grid_size
    wall, floor = np.zeros((height, width), bool), np.zeros((height, width), bool)
    for s in game.current_level.get_sprites():
        if s is player or "ignore" in s.tags:
            continue
        solid = s.is_collidable and s._blocking != BlockingMode.NOT_BLOCKED
        ground = s.interaction == InteractionMode.INTANGIBLE and not {"crzsjq", "vcha"} & set(s.tags)
        if not solid and not ground:
            continue
        pixels = s.render() >= 0
        x0, y0 = max(s.x, 0), max(s.y, 0)
        x1, y1 = min(s.x + pixels.shape[1], width), min(s.y + pixels.shape[0], height)
        if x0 >= x1 or y0 >= y1:
            continue
        part = pixels[y0 - s.y:y1 - s.y, x0 - s.x:x1 - s.x]
        if solid:
            wall[y0:y1, x0:x1] |= part
        if ground:
            floor[y0:y1, x0:x1] |= part
    return wall, floor


def solve(level) -> list | None:
    game, _ = levels.new_game(GAME, level)
    player, goal = game.qnnpcoyzd, game.hfuqkxulm
    pw, ph = player.width, player.height
    buttons = sorted(game.current_level.get_sprites_by_tag("buezna"), key=lambda s: s.name)
    letters = [next(t for t in b.tags if len(t) == 1) for b in buttons]
    scale, ox, oy = game.camera._calculate_scale_and_offset()
    clicks = []
    for b in buttons:
        ys, xs = np.nonzero(b.render() >= 0)
        clicks.append(ActionInput(id=GameAction.ACTION6, data={"x": (b.x + int(xs[0])) * scale + ox,
                                                                "y": (b.y + int(ys[0])) * scale + oy}))
    masks = {}
    for bits in itertools.product((0, 1), repeat=len(letters)):
        copied = copy.deepcopy(game)
        for letter, bit in zip(letters, bits, strict=True):
            if bit:
                _toggle(copied, letter)
        masks[bits] = _masks(copied, copied.qnnpcoyzd)
    height, width = masks[tuple(0 for _ in letters)][1].shape

    def stands(x: int, y: int, bits: tuple) -> bool:
        return 0 <= x < width and 0 <= y < height and bool(masks[bits][1][y, x])

    def clear(x: int, y: int, bits: tuple) -> bool:
        return not masks[bits][0][max(y, 0):y + ph, max(x, 0):x + pw].any()

    start = (player.x, player.y, tuple(0 for _ in letters))
    parent = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        x, y, bits = state
        if (x, y) == (goal.x, goal.y):
            path = []
            while parent[state] is not None:
                state, action = parent[state]
                path.append(action)
            return path[::-1]
        options = []
        for action, dx, dy in MOVES:
            nx, ny = x + U * dx, y + U * dy
            if clear(nx, ny, bits) and stands(nx, ny, bits):
                options.append(((nx, ny, bits), ActionInput(id=action)))
        for k, click in enumerate(clicks):
            flipped = tuple(b ^ (i == k) for i, b in enumerate(bits))
            if stands(x, y, flipped):
                options.append(((x, y, flipped), click))
        for nxt, action in options:
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    return None
