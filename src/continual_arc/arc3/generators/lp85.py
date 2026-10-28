"""lp85 at level 0: twenty 2 x 2 tiles sit on a closed loop of cells (the track of the level's level_name), and two
buttons turn the loop. ACTION6 on button_A_R moves every tile on the loop one cell forward, on button_A_L one cell
back; the level is won when the goal tile (the same color as the target's corner marks) sits inside the target
marker. Every click that does not win uses one of 13 steps, and the game is over when they run out.

generate keeps level 0's track and data (level_name selects the track from the game's table, so it and the 13 steps
stay as they are) and lays the level out anew: the target marker on any of the loop's 20 cells, the goal tile 3 to 7
cells from it in either direction (level 0's is 5, and no loop distance exceeds 10, so the steps always suffice), each
other cell filled with probability 0.85 by a tile of one of level 0's five colors, and the two buttons anywhere on
the board clear of the loop, the marker, the step bar's column and each other.

solve is a breadth-first search over the loop's turn (0 to 19 cells) derived from the game's code, clicking each
button where the engine reports it clickable; it takes well under a millisecond.
"""
from __future__ import annotations

import random
from collections import deque

from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "lp85"

UNIT = 3               # a track cell's pixel pitch (the game's crxpafuiwp)
TILES = ("aydooxtcli", "ekpijbeoel", "kisflvfvsy", "slefrnjymo", "tddzizwhzf")
GOAL, MARKER = "odkpvwbihk", "bghvgbtwcb"
BUTTONS = {"ahdesifykt": -1, "zpxgvxwdex": 1}  # button_A_L turns the loop back a cell, button_A_R forward
DISTANCE = (3, 7)
FILL = 0.85
TRIES = 200

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_DATA = dict(_BASE._data)
_BASE_LAYOUT = levels.layout(_BASE)


def _track(level_name: str) -> list[tuple[int, int]]:
    """The loop's cells in order, as pixel positions (the game's izutyjcpih table, track A)."""
    grid = _MODULE.izutyjcpih[level_name]["A"]
    cells = {n: (x * UNIT, y * UNIT) for y, row in enumerate(grid) for x, n in enumerate(row) if n != -1}
    return [cells[n] for n in sorted(cells)]


def _boxes_overlap(a: tuple, b: tuple, margin: int = 1) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw + margin and bx < ax + aw + margin and ay < by + bh + margin and by < ay + ah + margin


def _button_spots(taken: list[tuple], size: tuple[int, int], width: int, height: int) -> list[tuple[int, int]]:
    """Where a button fits clear of ``taken``; column 0 is left free, as the step bar is drawn over it."""
    w, h = size
    return [(x, y) for x in range(1, width - w + 1) for y in range(height - h + 1)
            if not any(_boxes_overlap((x, y, w, h), box) for box in taken)]


def generate(rng: random.Random):
    track = _track(_DATA["level_name"])
    n = len(track)
    width, height = _BASE.grid_size
    for _ in range(TRIES):
        target = rng.randrange(n)
        goal = (target + rng.choice((-1, 1)) * rng.randint(*DISTANCE)) % n
        tx, ty = track[target]
        sprites = [_SPRITES[MARKER].clone().set_position(tx - 1, ty - 1),
                   _SPRITES[GOAL].clone().set_position(*track[goal])]
        sprites += [_SPRITES[rng.choice(TILES)].clone().set_position(*track[i])
                    for i in range(n) if i not in (target, goal) and rng.random() < FILL]
        if rng.random() < FILL:
            sprites.append(_SPRITES[rng.choice(TILES)].clone().set_position(tx, ty))
        taken = [(x, y, 2, 2) for x, y in track] + [(tx - 1, ty - 1, 4, 4)]
        for name in rng.sample(sorted(BUTTONS), len(BUTTONS)):
            button = _SPRITES[name]
            spots = _button_spots(taken, (button.width, button.height), width, height)
            x, y = rng.choice(spots)
            sprites.append(button.clone().set_position(x, y))
            taken.append((x, y, button.width, button.height))
        level = type(_BASE)(sprites=sorted(sprites, key=lambda s: s.name), grid_size=_BASE.grid_size,
                            data=dict(_DATA))
        if levels.layout(level) == _BASE_LAYOUT:
            continue
        return level
    return None


def solve(level) -> list | None:
    track = _track(level._data["level_name"])
    index = {cell: i for i, cell in enumerate(track)}
    n = len(track)
    goals = [index[(s.x, s.y)] for s in level.get_sprites() if s.name == GOAL and (s.x, s.y) in index]
    targets = {index[(s.x + 1, s.y + 1)] for s in level.get_sprites() if s.name == MARKER}
    game, _ = levels.new_game(GAME, level)
    scale, x_off, y_off = game.camera._calculate_scale_and_offset()
    clicks = []
    for s in game.current_level.get_sprites():
        if s.name in BUTTONS:
            px = s.render()
            dy, dx = next((r, c) for r in range(px.shape[0]) for c in range(px.shape[1]) if px[r, c] >= 0)
            clicks.append((BUTTONS[s.name], {"x": (s.x + dx) * scale + x_off, "y": (s.y + dy) * scale + y_off}))
    clicks.sort(key=lambda c: c[0])
    parent = {0: None}
    queue = deque([0])
    while queue:
        turn = queue.popleft()
        for step, data in clicks:
            nxt = (turn + step) % n
            if nxt in parent:
                continue
            parent[nxt] = (turn, data)
            if targets <= {(g + nxt) % n for g in goals}:
                path, state = [], nxt
                while parent[state] is not None:
                    state, data = parent[state]
                    path.append(ActionInput(id=GameAction.ACTION6, data=dict(data)))
                return path[::-1]
            queue.append(nxt)
    return None
