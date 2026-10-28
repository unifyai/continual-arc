"""cd82 at level 0: a bucket sits at one of eight places around a blank 10 x 10 canvas (ACTION1-4 move it round the
ring), clicking one of two paint pots (ACTION6) picks its color, and ACTION5 pours: from a side it paints the near
half of the canvas, from a corner the near triangle. The level is won when the canvas matches the picture in the top
bar everywhere off its two diagonals; 100 actions lose.

A layout keeps level 0's pots, white (0) and the bucket's starting color (15), placed in either order anywhere along
the top bar with the pot marker under 15. Its picture is drawn from every picture those pots can paint, swept at
import, by first drawing a shortest painting length from 3 to 10 (level 0's is 5; the gate's band round it) and then a
picture of that length, so it is solvable by construction; blank pictures and level 0's own are left out.

solve is a breadth-first search over (bucket place, color, canvas), with the pour regions and the win test read from
the game, and its answer is checked in the real engine."""
from __future__ import annotations

import random
from collections import deque

import numpy as np

from continual_arc.arc3 import levels

GAME = "cd82"
PICTURE, BUCKET, POT, CANVAS, MARKER = "eoqnvkspoa-pqwme1-1", "oaoosfneq-oaanwen", "pqkenviek", "xytrjjbyib", "ydiwkzjkgl"
START_COLOR = 15  # the game sets the bucket's color to 15 on every level
LENGTH = (3, 10)  # shortest solution lengths kept: the gate's band round level 0's 5
MAX_STATES = 200_000

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]


def _regions() -> list[np.ndarray]:
    """Where a pour from each of the eight places paints: halves from the sides (0 top, 2 right, 4 bottom, 6 left)
    and triangles, their diagonal included, from the corners (1 top right, 3 bottom right, 5 bottom left, 7 top
    left), as the game's pour paints them."""
    masks = [np.zeros((10, 10), bool) for _ in range(8)]
    masks[0][0:5, :] = True
    masks[2][:, 5:10] = True
    masks[4][5:10, :] = True
    masks[6][:, 0:5] = True
    for i in range(10):
        masks[1][i, i:10] = True
        masks[3][i, 9 - i:10] = True
        masks[5][i, 0:i + 1] = True
        masks[7][i, 0:10 - i] = True
    return masks


_REGIONS = _regions()
_OFF_DIAGONAL = np.ones((10, 10), bool)
for _i in range(10):
    _OFF_DIAGONAL[_i, _i] = _OFF_DIAGONAL[_i, 9 - _i] = False
# The bucket's ring: place -> (row, column) on a 3 x 3 grid whose center is barred; ACTION1 and ACTION2 step the row
# up and down, ACTION3 and ACTION4 the column left and right, and a step off the ring or onto the center does nothing.
_CELL = {0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2), 4: (2, 1), 5: (2, 0), 6: (1, 0), 7: (0, 0)}
_PLACE = {v: k for k, v in _CELL.items()}
_STEP = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def _move(place: int, action: int) -> int:
    r, c = _CELL[place]
    dr, dc = _STEP[action]
    cell = (min(2, max(0, r + dr)), min(2, max(0, c + dc)))
    return _PLACE.get(cell, place)


def _search(canvas: np.ndarray, picture: np.ndarray, pots: list[int]) -> list[tuple] | None:
    """Shortest list of moves ("move", 1-4), pot clicks ("pot", index) and pours ("pour",) that paints ``canvas``
    into ``picture`` off its diagonals, the bucket starting at place 0 in START_COLOR."""
    goal = picture[_OFF_DIAGONAL]
    start = (0, START_COLOR, canvas.tobytes())
    prev = {start: None}
    queue = deque([start])
    expanded = 0
    while queue and expanded < MAX_STATES:
        state = queue.popleft()
        expanded += 1
        place, color, raw = state
        moves = [(("move", a), (_move(place, a), color, raw)) for a in _STEP]
        moves += [(("pot", i), (place, c, raw)) for i, c in enumerate(pots)]
        painted = np.frombuffer(raw, dtype=canvas.dtype).reshape(10, 10).copy()
        painted[_REGIONS[place]] = color
        moves.append((("pour",), (place, color, painted.tobytes())))
        for action, nxt in moves:
            if nxt in prev:
                continue
            prev[nxt] = (state, action)
            if action == ("pour",) and np.array_equal(painted[_OFF_DIAGONAL], goal):
                path = []
                while prev[nxt] is not None:
                    nxt, step = prev[nxt]
                    path.append(step)
                return path[::-1]
            queue.append(nxt)
    return None


def _level(picture: np.ndarray, pot_colors: list[int], pot_x: list[int], pot_y: int):
    from arcengine import Level

    sprites = _MODULE.sprites
    target = sprites[PICTURE].clone().set_position(3, 3)
    target.pixels = picture.copy()
    pots = [sprites[POT].clone().set_position(x, pot_y).color_remap(0, c) for x, c in zip(pot_x, pot_colors, strict=True)]
    marker_x = pot_x[pot_colors.index(START_COLOR)]
    return Level(
        sprites=[target, sprites["gkyfnkvrty"].clone(),
                 sprites[BUCKET].clone().set_position(25, 24).set_rotation(180), *pots,
                 sprites[CANVAS].clone().set_position(27, 34),
                 sprites[MARKER].clone().set_position(marker_x, pot_y + 5)],
        grid_size=_LEVEL0.grid_size,
        data=dict(_LEVEL0._data),
    )


def _pictures(pots: tuple[int, ...]) -> dict[int, list[np.ndarray]]:
    """Every picture paintable on the blank canvas with ``pots`` whose shortest painting lies within LENGTH, by that
    length: a breadth-first sweep of (bucket place, color, canvas), keeping the first pour that leaves each canvas
    as it reads off its diagonals."""
    blank = _MODULE.sprites[CANVAS].pixels.copy()
    start = (0, START_COLOR, blank.tobytes())
    depth, queue, found = {start: 0}, deque([start]), {}
    while queue:
        state = queue.popleft()
        place, color, raw = state
        if depth[state] >= LENGTH[1]:
            continue
        painted = np.frombuffer(raw, dtype=blank.dtype).reshape(10, 10).copy()
        painted[_REGIONS[place]] = color
        found.setdefault(painted[_OFF_DIAGONAL].tobytes(), (depth[state] + 1, painted))
        for nxt in [(_move(place, a), color, raw) for a in _STEP] + [(place, c, raw) for c in pots] + \
                [(place, color, painted.tobytes())]:
            if nxt not in depth:
                depth[nxt] = depth[state] + 1
                queue.append(nxt)
    by_length: dict[int, list[np.ndarray]] = {}
    for length, picture in found.values():
        if LENGTH[0] <= length and not np.array_equal(picture[_OFF_DIAGONAL], blank[_OFF_DIAGONAL]):
            by_length.setdefault(length, []).append(picture)
    return by_length


# Level 0's pots: white (0) and the bucket's starting color.
_BASE_POTS = tuple(int(s.pixels[2, 2]) for s in _LEVEL0.get_sprites() if s.name == POT)
(_BASE_PICTURE,) = [s.pixels for s in _LEVEL0.get_sprites() if s.name == PICTURE]
_PICTURES = {n: [p for p in ps if not np.array_equal(p[_OFF_DIAGONAL], _BASE_PICTURE[_OFF_DIAGONAL])]
             for n, ps in _pictures(_BASE_POTS).items()}


def generate(rng: random.Random):
    picture = rng.choice(_PICTURES[rng.choice(sorted(_PICTURES))]).copy()
    pots = list(_BASE_POTS)
    rng.shuffle(pots)
    gap = rng.randint(1, 7)
    x0 = rng.randint(19, 63 - 5 * len(pots) - gap * (len(pots) - 1))
    xs = [x0 + i * (5 + gap) for i in range(len(pots))]
    return _level(picture, pots, xs, rng.randint(1, 3))


def solve(level) -> list | None:
    from arcengine import ActionInput, GameAction

    by_name = {}
    for s in level.get_sprites():
        by_name.setdefault(s.name, []).append(s)
    pots = by_name[POT]
    path = _search(by_name[CANVAS][0].pixels.copy(), by_name[PICTURE][0].pixels, [int(p.pixels[2, 2]) for p in pots])
    if path is None:
        return None
    actions = []
    for step in path:
        if step[0] == "move":
            actions.append(ActionInput(id=GameAction.from_id(step[1])))
        elif step[0] == "pot":
            # The level is 64 x 64, so the camera maps a grid cell to the same display pixel.
            pot = pots[step[1]]
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": pot.x + 2, "y": pot.y + 2}))
        else:
            actions.append(ActionInput(id=GameAction.ACTION5))
    return actions if levels.replay(GAME, level, actions) else None
