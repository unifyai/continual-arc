"""sc25 at level 0: a big (4 x 4 pixel) player in a maze of pixel-perfect walls must touch the exit, which lies at
the end of a passage only 2 pixels wide. ACTION1-4 move the player 4 pixels (2 when 4 are blocked, 2 when small),
and ACTION6 clicks: drawing the size spell's pattern (a plus without its center) in the 3 x 3 grid at the bottom casts
it, which shrinks the player to 2 x 2 or grows it back where it fits. Every move and click spends one of 50 steps.

generate redraws the maze rows (0-45) of level 0's one wall sprite on a 2-pixel lattice of random parity: a random
walk of one to seven 4 x 4 cells, sometimes with side cells, from the player's start, and from its last cell a passage
one block (2 pixels) wide of two to eight blocks with random turns that touches no other floor, so the big player
cannot enter it. The exit sits at the passage's end, turned so its open side faces it, as level 0's does. A layout is
kept when it is solvable, not solvable without casting the spell, and its shortest solution is 9 to 21 actions (level
0's is 13). The step budget scales with the solution's steps, as level 0's 50 is to its 12; the spell grid, its
button and the step bar stay where level 0 has them.

solve is a uniform-cost search over (x, y, scale) derived from the game's code: moves collide pixel-perfectly with
every collidable sprite and win when the blocked move touches the exit; a cast (four clicks) shrinks in place or grows
at the first free offset the game tries, and costs four actions and five steps. The first action of a level only
plays the spell's demonstration, so the solution repeats its first action once. It takes a few milliseconds.
"""
from __future__ import annotations

import heapq
import random

import numpy as np
from arcengine import ActionInput, BlockingMode, GameAction

from continual_arc.arc3 import levels

GAME = "sc25"

PLAYER, EXIT, WALL = "pluyoo", "exydhv", "duvwsv-2"
SIZE = 64
MAZE_ROWS = 46             # rows 0-45 are drawn anew; rows 46-63 keep level 0's wall around the spell grid
FLOOR_X, FLOOR_Y = (2, 59), (2, 43)    # floor pixels stay inside these (inclusive), so a wall rings the maze
EXIT_X, EXIT_Y = (0, 61), (0, 46)      # the exit stays clear of the step bar (x 62-63) and of the spell grid
LENGTH = (9, 21)
TRIES = 400
MAX_STATES = 50_000
PAD = 4                    # a move or a grow looks at most 4 pixels beyond the player

MOVES = ((GameAction.ACTION1, 0, -1), (GameAction.ACTION2, 0, 1), (GameAction.ACTION3, -1, 0),
         (GameAction.ACTION4, 1, 0))
FACING = {GameAction.ACTION1: 180, GameAction.ACTION2: 0, GameAction.ACTION3: 90, GameAction.ACTION4: 270}
# The exit beside the passage's last block, its open side towards it, for the direction the passage last ran:
# (rotation, dx, dy) from the block's corner. Level 0's (rotation 0) lies to the left of its passage.
EXIT_AT = {(-1, 0): (0, -5, -2), (1, 0): (180, 2, -2), (0, -1): (90, -2, -5), (0, 1): (270, -2, 2)}
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_DATA = dict(_BASE._data)
_BASE_LAYOUT = levels.layout(_BASE)
_UI = [s for s in _BASE.get_sprites() if s.name not in (PLAYER, EXIT, WALL)]
(_BASE_WALL,) = [np.array(s.pixels) for s in _BASE.get_sprites() if s.name == WALL]
WALL_COLOR = int(_BASE_WALL.max())
_GAME = levels.new_game(GAME, _BASE)[0]
_PATTERN = _GAME.zzpoabuniyn[_DATA["efvw"]]
# The spell's clicks, the centers of its grid slots as the game lists them (bmmtkvkbcdd, row by row).
_CAST = [_GAME.bmmtkvkbcdd[3 * r + c] for r in range(3) for c in range(3) if _PATTERN[r][c]]
CAST_STEPS = len(_CAST) + 1    # each click spends a step, and the change of size one more


def _paint(mask: np.ndarray, sprite) -> None:
    px = sprite.render() != -1
    h, w = px.shape
    x0, y0 = max(sprite.x, 0), max(sprite.y, 0)
    x1, y1 = min(sprite.x + w, SIZE), min(sprite.y + h, SIZE)
    if x0 < x1 and y0 < y1:
        mask[PAD + y0:PAD + y1, PAD + x0:PAD + x1] |= px[y0 - sprite.y:y1 - sprite.y, x0 - sprite.x:x1 - sprite.x]


def _search(level, casts: bool = True) -> tuple[list, int] | None:
    """The fewest actions from the reset to the exit, and the steps they spend, by the game's rules (no demonstration
    action); None when the budget or MAX_STATES allows none. Without casts the player keeps its size."""
    shape = (SIZE + 2 * PAD, SIZE + 2 * PAD)
    block, exit_, walls = np.zeros(shape, bool), np.zeros(shape, bool), np.ones(shape, bool)
    walls[PAD:PAD + SIZE, PAD:PAD + SIZE] = False   # a grow treats leaving the grid as blocked; a move does not
    for s in level.get_sprites():
        if s.name == PLAYER or not s.is_collidable or s.blocking == BlockingMode.NOT_BLOCKED:
            continue
        _paint(exit_ if s.name == EXIT else block, s)
        if s.name.startswith("duvwsv-"):
            _paint(walls, s)

    def hit(mask: np.ndarray, x: int, y: int, k: int) -> bool:
        return bool(mask[PAD + y:PAD + y + k, PAD + x:PAD + x + k].any())

    def move(x: int, y: int, scale: int, dx: int, dy: int) -> tuple[tuple, bool]:
        k = 2 * scale
        for d in ((2,) if scale == 1 else (4, 2)):    # a big player blocked at 4 pixels tries 2
            nx, ny = x + dx * d, y + dy * d
            if not (hit(block, nx, ny, k) or hit(exit_, nx, ny, k)):
                return (nx, ny, scale), False
        return (x, y, scale), hit(exit_, nx, ny, k)

    def cast(x: int, y: int, scale: int) -> tuple | None:
        if scale == 2:
            return x, y, 1
        for tx in (-2, -1, 0):      # the game's hsrovbjzdx: the first offset where the grown player clears the walls
            for ty in (-2, -1, 0):
                if not hit(walls, x + tx, y + ty, 4):
                    return x + tx, y + ty, 2
        return None

    (p,) = level.get_sprites_by_name(PLAYER)
    budget = int(level.get_data("slfh"))
    start = (p.x, p.y, p.scale)
    best = {start: (0, 0)}
    parent: dict = {start: None}
    heap = [(0, 0, start)]
    expanded = 0
    while heap and expanded < MAX_STATES:
        cost, steps, state = heapq.heappop(heap)
        if best[state] < (cost, steps):
            continue
        expanded += 1
        options = []
        for action, dx, dy in MOVES:
            nxt, win = move(*state, dx, dy)
            if win:
                path = [action]
                while parent[state] is not None:
                    state, act = parent[state]
                    path.append(act)
                return path[::-1], steps
            if nxt != state:
                options.append((nxt, cost + 1, steps + 1, action))
        if casts and (nxt := cast(*state)) is not None:
            options.append((nxt, cost + len(_CAST), steps + CAST_STEPS, "cast"))
        for nxt, c, s, act in options:
            if s <= budget and (nxt not in best or (c, s) < best[nxt]):
                best[nxt] = (c, s)
                parent[nxt] = (state, act)
                heapq.heappush(heap, (c, s, nxt))
    return None


def _actions(path: list) -> list:
    out = []
    for act in path:
        if act == "cast":
            out += [ActionInput(id=GameAction.ACTION6, data=dict(a.data)) for a in _CAST]
        else:
            out.append(ActionInput(id=act))
    return out


def solve(level) -> list | None:
    found = _search(level)
    if found is None:
        return None
    actions = _actions(found[0])
    # Level 0's first action only plays the spell's demonstration (the game's qytejzcythm), so it is spent twice.
    return [ActionInput(id=actions[0].id, data=dict(actions[0].data or {})), *actions]


_BASE_STEPS = _search(_BASE)[1]


def _draw(rng: random.Random):
    """A maze on a 2-pixel lattice: (floor pixel mask, player corner, exit corner and rotation), or None."""
    ox, oy = rng.randint(0, 1), rng.randint(0, 1)

    def corner(a: int, b: int) -> tuple[int, int]:
        return ox + 2 * a, oy + 2 * b

    def inside(a: int, b: int) -> bool:
        x, y = corner(a, b)
        return FLOOR_X[0] <= x and x + 1 <= FLOOR_X[1] and FLOOR_Y[0] <= y and y + 1 <= FLOOR_Y[1]

    a0, b0 = rng.randint(0, 1), rng.randint(0, 1)

    def blocks(cell: tuple[int, int]) -> list[tuple[int, int]]:
        i, j = cell
        return [(a0 + 2 * i + u, b0 + 2 * j + v) for u in (0, 1) for v in (0, 1)]

    def usable(cell: tuple[int, int]) -> bool:
        return all(inside(*b) for b in blocks(cell))

    cells = [c for c in ((i, j) for i in range(16) for j in range(12)) if usable(c)]
    walk = [rng.choice(cells)]
    heading = rng.choice(DIRS)
    for _ in range(rng.randint(0, 6)):
        turns = [heading] * 3 + [d for d in DIRS if d != heading and d != (-heading[0], -heading[1])]
        rng.shuffle(turns)
        for d in turns:
            nxt = (walk[-1][0] + d[0], walk[-1][1] + d[1])
            if usable(nxt) and nxt not in walk:
                walk.append(nxt)
                heading = d
                break
        else:
            break
    room = set(walk)
    for _ in range(rng.choice((0, 0, 1, 2, 3))):
        base = rng.choice(walk[:-1] or walk)
        d = rng.choice(DIRS)
        nxt = (base[0] + d[0], base[1] + d[1])
        if usable(nxt):
            room.add(nxt)
    floor = {b for c in room for b in blocks(c)}

    # The passage leaves the walk's last cell through one of its sides, from one of that side's two blocks. Each of
    # its blocks touches no floor but the block before it, so it stays one block wide and the big player stays out.
    last = walk[-1]
    sides = [d for d in DIRS if (last[0] + d[0], last[1] + d[1]) not in room]
    if not sides:
        return None
    d = rng.choice(sides)
    prev = rng.choice([b for b in blocks(last) if (b[0] + d[0], b[1] + d[1]) not in floor])
    passage: list[tuple[int, int]] = []

    def lonely(b: tuple[int, int], before: tuple[int, int]) -> bool:
        around = [(b[0] + e[0], b[1] + e[1]) for e in DIRS]
        return all(n == before or (n not in floor and n not in passage) for n in around)

    for _ in range(rng.randint(2, 8)):
        turns = [d] * 2 + [t for t in DIRS if t != d and t != (-d[0], -d[1])]
        rng.shuffle(turns)
        for t in turns:
            b = (prev[0] + t[0], prev[1] + t[1])
            if inside(*b) and b not in floor and b not in passage and lonely(b, prev):
                passage.append(b)
                prev, d = b, t
                break
        else:
            break
    if len(passage) < 2:
        return None
    floor |= set(passage)

    rotation, ex, ey = EXIT_AT[d]
    bx, by = corner(*passage[-1])
    exit_xy = (bx + ex, by + ey)
    w, h = (5, 6) if rotation in (0, 180) else (6, 5)
    if not (EXIT_X[0] <= exit_xy[0] and exit_xy[0] + w - 1 <= EXIT_X[1]
            and EXIT_Y[0] <= exit_xy[1] and exit_xy[1] + h - 1 <= EXIT_Y[1]):
        return None
    mask = np.zeros((SIZE, SIZE), bool)
    for a, b in floor:
        x, y = corner(a, b)
        mask[y:y + 2, x:x + 2] = True
    if mask[exit_xy[1]:exit_xy[1] + h, exit_xy[0]:exit_xy[0] + w].any():
        return None
    return mask, corner(*blocks(walk[0])[0]), exit_xy, rotation


def _level(mask: np.ndarray, player: tuple[int, int], exit_xy: tuple[int, int], exit_rotation: int,
           facing: int, budget: int):
    pixels = _BASE_WALL.copy()
    pixels[:MAZE_ROWS] = WALL_COLOR
    pixels[:MAZE_ROWS][mask[:MAZE_ROWS]] = -1
    wall = _SPRITES[WALL].clone()
    wall.pixels = pixels.astype(np.int8)
    sprites = [s.clone() for s in _UI] + [
        wall,
        _SPRITES[EXIT].clone().set_position(*exit_xy).set_rotation(exit_rotation),
        _SPRITES[PLAYER].clone().set_position(*player).set_scale(2).set_rotation(facing),
    ]
    return type(_BASE)(sprites=sprites, grid_size=_BASE.grid_size, data={**_DATA, "slfh": budget})


def generate(rng: random.Random):
    for _ in range(TRIES):
        drawn = _draw(rng)
        if drawn is None:
            continue
        mask, player, exit_xy, rotation = drawn
        level = _level(mask, player, exit_xy, rotation, 90, _DATA["slfh"])
        found = _search(level)
        if found is None or not LENGTH[0] <= 1 + len(_actions(found[0])) <= LENGTH[1]:
            continue
        if _search(level, casts=False) is not None:
            continue
        path, steps = found
        first = next(a for a in path if a != "cast")
        level = _level(mask, player, exit_xy, rotation, FACING[first], round(_DATA["slfh"] * steps / _BASE_STEPS))
        if levels.layout(level) == _BASE_LAYOUT:
            continue
        return level
    return None
