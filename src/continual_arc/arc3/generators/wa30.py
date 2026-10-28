"""wa30 at level 0: a player on a 16 x 16 board of 4-pixel cells and three boxes, with one three-cell target strip.
ACTION1-4 step the player (and turn it to face that way while it holds nothing; a blocked step only turns it), ACTION5
grabs the box the player faces or lets go of the one it holds, and a held box moves rigidly with the player, pulled
or pushed. The level is won when every box lies on the strip and none is held, within the 200-step counter.

generate places the strip anywhere on the board, lying flat as in level 0 or standing on end, then the three boxes
and the player on free cells within four cells of it, as level 0's lie, and keeps the layout when solve's answer is
18 to 38 actions (level 0's is 26), well inside the counter; the board has no walls, so every layout is solvable.

solve is a uniform-cost search over the moments the player holds nothing: (player cell, facing, box cells). Each move
of it walks to one side of a box not yet on the strip (breadth-first over cell and facing, a blocked step turning the
player), grabs it, carries it to a free strip cell (breadth-first over the box's cell with the player beside it) and
lets go. The cheapest sequence is checked in the real engine; it takes about 0.15 s.
"""
from __future__ import annotations

import heapq
import random
from collections import deque

from arcengine import ActionInput, GameAction, Level

from continual_arc.arc3 import levels

GAME = "wa30"
PLAYER, BOX, STRIP = "wppuejnwhl", "pktgsotzmw", "jigtxgzhwt"
CELL, SIDE = 4, 16      # cells are 4 pixels; the board is 16 x 16 cells
PLACE_ROWS = 15         # the bottom row carries the step bar, so nothing is placed there
BOXES = 3
NEAR = 4                # boxes and player start within this many cells (Chebyshev) of the strip
LENGTH = (18, 38)       # solve's answer kept; level 0's is 26
TRIES = 12

# ACTION1-4 and the facing each gives: the game's rotations 0, 180, 270 and 90 point up, down, left and right.
MOVES = ((GameAction.ACTION1, (0, -1)), (GameAction.ACTION2, (0, 1)), (GameAction.ACTION3, (-1, 0)),
         (GameAction.ACTION4, (1, 0)))
FACING = {0: (0, -1), 180: (0, 1), 270: (-1, 0), 90: (1, 0)}

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_BASE_LAYOUT = levels.layout(_LEVEL0)


def _inside(c: tuple[int, int]) -> bool:
    return 0 <= c[0] < SIDE and 0 <= c[1] < SIDE


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _walks(start: tuple, facing: tuple, boxes: frozenset) -> dict:
    """Shortest walks over (cell, facing) from the player's start while it holds nothing: a step into a box or the
    board's edge only turns it. Returns each reached state's parent link (state, action)."""
    parent = {(start, facing): None}
    queue = deque([(start, facing)])
    while queue:
        state = queue.popleft()
        cell, _ = state
        for action, d in MOVES:
            nxt = _add(cell, d)
            new = (nxt, d) if _inside(nxt) and nxt not in boxes else (cell, d)
            if new not in parent:
                parent[new] = (state, action)
                queue.append(new)
    return parent


def _carries(box: tuple, offset: tuple, others: frozenset) -> dict:
    """Shortest carries of a box held from ``offset`` (box minus player): both cells must stay on the board and off
    the other boxes. Returns each reached box cell's parent link (cell, action)."""
    parent = {box: None}
    queue = deque([box])
    while queue:
        cell = queue.popleft()
        for action, d in MOVES:
            nxt = _add(cell, d)
            player = (nxt[0] - offset[0], nxt[1] - offset[1])
            if nxt in parent or not (_inside(nxt) and _inside(player)) or nxt in others or player in others:
                continue
            parent[nxt] = (cell, action)
            queue.append(nxt)
    return parent


def _trace(parent: dict, end: object) -> list:
    path = []
    while parent[end] is not None:
        end, action = parent[end]
        path.append(action)
    return path[::-1]


def _search(player: tuple, facing: tuple, boxes: frozenset, strip: frozenset) -> list | None:
    """Cheapest actions that leave every box on the strip, by uniform-cost search over (player, facing, boxes)
    between carries."""
    start = (player, facing, boxes)
    best = {start: 0}
    links = {start: None}
    heap = [(0, 0, start)]
    tie = 0
    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost > best[state]:
            continue
        player, facing, boxes = state
        if boxes <= strip:
            plan = []
            while links[state] is not None:
                state, actions = links[state]
                plan[:0] = actions
            return plan
        walks = _walks(player, facing, boxes)
        free_strip = strip - boxes
        for box in boxes - strip:
            others = boxes - {box}
            for d in FACING.values():
                stand = (box[0] - d[0], box[1] - d[1])
                if (stand, d) not in walks:
                    continue
                walk = _trace(walks, (stand, d))
                carries = _carries(box, d, others)
                for goal in free_strip & carries.keys():
                    actions = [*walk, GameAction.ACTION5, *_trace(carries, goal), GameAction.ACTION5]
                    nxt = ((goal[0] - d[0], goal[1] - d[1]), d, others | {goal})
                    if cost + len(actions) < best.get(nxt, 1 << 30):
                        best[nxt] = cost + len(actions)
                        links[nxt] = (state, actions)
                        tie += 1
                        heapq.heappush(heap, (best[nxt], tie, nxt))
    return None


def _cells(sprite) -> list[tuple[int, int]]:
    return [((sprite.x + dx) // CELL, (sprite.y + dy) // CELL)
            for dy in range(0, sprite.height, CELL) for dx in range(0, sprite.width, CELL)]


def _read(level) -> tuple:
    sprites = level.get_sprites()
    (player,) = [s for s in sprites if s.name == PLAYER]
    strip = frozenset(c for s in sprites if s.name == STRIP for c in _cells(s))
    boxes = frozenset((s.x // CELL, s.y // CELL) for s in sprites if s.name == BOX)
    return (player.x // CELL, player.y // CELL), FACING[player.rotation], boxes, strip


def _level(strip: tuple, rotation: int, boxes: list, player: tuple) -> Level:
    sprites = _MODULE.sprites
    return Level(
        sprites=[sprites[STRIP].clone().set_rotation(rotation).set_position(strip[0] * CELL, strip[1] * CELL)]
        + [sprites[BOX].clone().set_position(x * CELL, y * CELL) for x, y in boxes]
        + [sprites[PLAYER].clone().set_position(player[0] * CELL, player[1] * CELL)],
        grid_size=_LEVEL0.grid_size,
        data=dict(_LEVEL0._data),
    )


def generate(rng: random.Random) -> Level | None:
    for _ in range(TRIES):
        rotation = rng.choice((0, 90))
        w, h = (3, 1) if rotation == 0 else (1, 3)
        corner = (rng.randrange(SIDE - w + 1), rng.randrange(PLACE_ROWS - h + 1))
        strip = {(corner[0] + i, corner[1] + j) for i in range(w) for j in range(h)}
        near = [(x, y) for x in range(SIDE) for y in range(PLACE_ROWS) if (x, y) not in strip
                and min(max(abs(x - sx), abs(y - sy)) for sx, sy in strip) <= NEAR]
        *boxes, player = rng.sample(near, BOXES + 1)
        plan = _search(player, FACING[0], frozenset(boxes), frozenset(strip))
        if plan is None or not LENGTH[0] <= len(plan) <= LENGTH[1]:
            continue
        level = _level(corner, rotation, boxes, player)
        if levels.layout(level) != _BASE_LAYOUT:
            return level
    return None


def solve(level) -> list | None:
    plan = _search(*_read(level))
    if plan is None:
        return None
    actions = [ActionInput(id=a) for a in plan]
    return actions if levels.replay(GAME, level, actions) else None
