"""g50t at level 0: a maze of 7-pixel floor cells on a 6-pixel lattice. ACTION1-4 move the player one cell, and
reaching the goal wins. A wire joins a pressure plate to a gate that blocks the only way to the goal: the gate slides
aside while anyone stands on the plate and slides back when the plate is left, crushing the player if under it.
ACTION5 rewinds the player to the start and leaves a ghost that replays the moves made since, in step with the
player's, so the player walks to the plate, records that walk with ACTION5, then walks through the gate the ghost holds
open (level 0: 4 moves, ACTION5, 12 moves). Level 0's two ghost slots allow one ghost at a time; the timer bar loses
the game after 130 actions.

generate draws a lattice of 5-10 x 5-9 cells anywhere on the board and grows a tree of floor on it: an induced path
from the start through the gate to the goal, a branch off it before the gate to the plate, and up to three dead-end
spurs. Every path is induced, so the floor is a tree and the gate cuts the start from the goal. The gate's open
place is a bare cell beside it, and the wire runs from there to the plate through bare cells, drawn as level 0 draws
it, with its floor-colored track beneath. A layout is kept when the ghost reaches the plate before the player can
reach the gate (as in level 0, so no waiting is needed), its shortest win is 13 to 22 actions (level 0's 17 the
median), and that win replays to a win in the engine.

solve reads the level's cells, gates, plates and their wires from the built game and searches an abstract model of
the game's code: a breadth-first search from the start that treats every gate as shut finds each plate; with a ghost
parked on plate P after a moves, a second search over (cell, moves made, capped at a) lets the player through P's
gates from move a + 1 on. The best of a direct walk and walk-to-plate, ACTION5, walk-through is returned, in
milliseconds.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "g50t"

STEP = 6                   # a move, the game's jarvstobjt: one lattice cell
CELL = 7                   # a cell's floor block; neighboring blocks share a pixel
FLOOR, WIRE, HIDDEN = 5, 8, -2
X_MIN, X_MAX = 0, 63       # the board: the goal's 9 x 9 floor reaches a pixel past its cell
Y_MIN, Y_MAX = 6, 62       # under the ghost slots (rows 0-5), above the timer bar (row 63)
COLS, ROWS = (5, 10), (5, 9)
TO_PLATE = (2, 6)          # moves from the start to the plate
LENGTH = (13, 22)          # the shortest win, ACTION5 included, drawn so that level 0's 17 is the median
TRIES = 300

MOVES = ((GameAction.ACTION1, 0, -1), (GameAction.ACTION2, 0, 1), (GameAction.ACTION3, -1, 0),
         (GameAction.ACTION4, 1, 0))
DIRS = tuple((dx, dy) for _, dx, dy in MOVES)
GATE_ROTATION = {(0, 1): 0, (-1, 0): 90, (0, -1): 180, (1, 0): 270}  # the side a gate slides to, as the game reads it

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_BASE_LAYOUT = levels.layout(_BASE)


def _near(c: tuple) -> list:
    return [(c[0] + dx, c[1] + dy) for dx, dy in DIRS]


def _walk(rng: random.Random, start: tuple, length: int, taken: set, inside) -> list | None:
    """An induced path of ``length`` moves from ``start`` (in ``taken``): no new cell touches any cell of ``taken`` or
    of the path other than the one it steps from, so walking it is the only way between its ends."""
    path = [start]
    for _ in range(length):
        last = path[-1]
        options = [c for c in _near(last) if inside(c) and c not in taken and c not in path
                   and not any(n != last and (n in taken or n in path) for n in _near(c))]
        if not options:
            return None
        if len(path) > 1 and rng.random() < 0.5:
            ahead = (2 * last[0] - path[-2][0], 2 * last[1] - path[-2][1])
            if ahead in options:
                options = [ahead]
        path.append(rng.choice(options))
    return path


def _route(rng: random.Random, start: tuple, ends: set, bare) -> list | None:
    """A shortest path of bare cells from ``start`` to one of ``ends``, ties broken at random."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        if c in ends:
            path = [c]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        options = [n for n in _near(c) if bare(n) and n not in parent]
        rng.shuffle(options)
        for n in options:
            parent[n] = c
            queue.append(n)
    return None


def _segment_pixels(points: list) -> list:
    """Every pixel of the axis-aligned polyline through ``points``, in order, without repeats."""
    out = [points[0]]
    for (x1, y1), (x2, y2) in zip(points, points[1:], strict=False):
        dx, dy = (x2 > x1) - (x2 < x1), (y2 > y1) - (y2 < y1)
        x, y = x1, y1
        while (x, y) != (x2, y2):
            x, y = x + dx, y + dy
            out.append((x, y))
    return out


def _sprite(name: str, pixels: dict, default: int = -1):
    """Level 0's template ``name`` with its pixels replaced by ``pixels`` ({(x, y): color}), placed at their box."""
    xs, ys = [x for x, _ in pixels], [y for _, y in pixels]
    bx, by = min(xs), min(ys)
    grid = np.full((max(ys) - by + 1, max(xs) - bx + 1), default, dtype=np.int8)
    for (x, y), v in pixels.items():
        grid[y - by, x - bx] = v
    sprite = _SPRITES[name].clone().set_position(bx, by)
    sprite.pixels = grid
    return sprite


def _build(origin: tuple, floor: set, start: tuple, goal: tuple, gate: tuple, opening: tuple, plate: tuple,
           wire: list) -> object:
    ox, oy = origin

    def at(c: tuple) -> tuple:
        return ox + STEP * c[0], oy + STEP * c[1]

    floor_px = {}
    for c in floor:
        x, y = at(c)
        floor_px.update({(x + u, y + v): FLOOR for u in range(CELL) for v in range(CELL)})
    gx, gy = at(goal)
    floor_px.update({(gx - 1 + u, gy - 1 + v): FLOOR for u in range(CELL + 2) for v in range(CELL + 2)})

    line = _segment_pixels([(at(c)[0] + CELL // 2, at(c)[1] + CELL // 2) for c in wire])
    px, py = at(plate)
    qx, qy = at(gate)
    wire_px = {}
    for x, y in line:
        under_plate = abs(x - px - CELL // 2) <= 1 and abs(y - py - CELL // 2) <= 1
        under_gate = qx <= x < qx + CELL and qy <= y < qy + CELL
        wire_px[(x, y)] = HIDDEN if under_plate or under_gate else WIRE
    track_px = {(x + u, y + v): FLOOR for x, y in line for u in (-1, 0, 1) for v in (-1, 0, 1)
                if (x + u, y + v) not in floor_px}

    rotation = GATE_ROTATION[(opening[0] - gate[0], opening[1] - gate[1])]
    sprites = [
        _sprite("awhgrxsdnu", track_px),
        _SPRITES["gehwhuvxqq"].clone().set_position(1, 1),
        _SPRITES["gilbljmfbc"].clone().set_position(gx - 1, gy - 1),
        _SPRITES["gpkhwmwioo"].clone(),
        _SPRITES["gpkhwmwioo"].clone().set_position(4, 0),
        _SPRITES["kjrcloicja"].clone().set_position(qx, qy).set_rotation(rotation),
        _SPRITES["medyellngi"].clone().set_position(px, py),
        _sprite("oanvksd0", wire_px),
        _SPRITES["ovhuyqtghw"].clone(),
        _SPRITES["ppfvilwwnk"].clone().set_position(0, 63),
        _SPRITES["qftsebtxuc"].clone().set_position(*at(start)),
        _sprite("vsrojdvivb", floor_px),
        _SPRITES["xulolghsby"].clone().set_position(0, 63),
    ]
    return type(_BASE)(sprites=sprites, grid_size=_BASE.grid_size, data=dict(_BASE._data))


def _draw(rng: random.Random) -> object | None:
    cols, rows = rng.randint(*COLS), rng.randint(*ROWS)
    origin = (rng.randint(X_MIN + 1, X_MAX - 1 - STEP * (cols - 1) - CELL),
              rng.randint(Y_MIN + 1, Y_MAX - STEP * (rows - 1) - CELL))

    def inside(c: tuple) -> bool:
        return 0 <= c[0] < cols and 0 <= c[1] < rows

    def on_board(c: tuple) -> bool:
        x, y = origin[0] + STEP * c[0], origin[1] + STEP * c[1]
        return X_MIN <= x - 1 and x + CELL <= X_MAX and Y_MIN <= y - 1 and y + CELL <= Y_MAX

    total = rng.randint(*LENGTH)
    to_plate = rng.randint(TO_PLATE[0], min(TO_PLATE[1], (total - 4) // 2))
    to_gate = rng.randint(to_plate + 1, min(to_plate + 4, total - to_plate - 3))
    main_len = total - 1 - to_plate
    start = (rng.randrange(cols), rng.randrange(rows))
    main = _walk(rng, start, main_len, {start}, inside)
    if main is None:
        return None
    attach = rng.randint(max(0, to_plate - 4), min(to_plate - 1, to_gate - 1))
    floor = set(main)
    branch = _walk(rng, main[attach], to_plate - attach, floor, inside)
    if branch is None:
        return None
    floor.update(branch)
    for _ in range(rng.randint(0, 3)):
        spur = _walk(rng, rng.choice(sorted(floor)), rng.randint(1, 3), floor, inside)
        if spur is not None:
            floor.update(spur)
    gate, goal, plate = main[to_gate], main[-1], branch[-1]

    def bare(c: tuple) -> bool:
        return on_board(c) and c not in floor

    openings = [c for c in _near(gate) if bare(c)]
    if not openings:
        return None
    opening = rng.choice(openings)
    ends = {c for c in _near(plate) if bare(c)}
    wire = _route(rng, opening, ends, bare)
    if wire is None:
        return None
    level = _build(origin, floor, start, goal, gate, opening, plate, [gate, *wire, plate])
    if levels.layout(level) == _BASE_LAYOUT:
        return None
    solution = solve(level)
    if solution is None or not LENGTH[0] <= len(solution) <= LENGTH[1]:
        return None
    if not levels.replay(GAME, level, solution):
        return None
    return level


def generate(rng: random.Random):
    for _ in range(TRIES):
        level = _draw(rng)
        if level is not None:
            return level
    return None


def _bfs(start: tuple, goal_test, passable) -> list | None:
    """Shortest list of (action, state) steps from ``start`` to a state passing ``goal_test``; ``passable(state,
    dx, dy)`` gives the next state or None."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if goal_test(state):
            path = []
            while parent[state] is not None:
                state, action = parent[state]
                path.append(action)
            return path[::-1]
        for action, dx, dy in MOVES:
            nxt = passable(state, dx, dy)
            if nxt is not None and nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    return None


def solve(level) -> list | None:
    game, _ = levels.new_game(GAME, level)
    world = game.vgwycxsxjz
    floor, player, goal = world.afbbgvkpip, world.dzxunlkwxt, world.whftgckbcu
    half = player.width // 2
    gates = {(g.x, g.y): g for g in world.uwxkstolmf}
    plates = [p for p in world.hamayflsib if hasattr(p, "nexhtmlmxh")]
    start, target = (player.x, player.y), (goal.x + 1, goal.y + 1)

    def walkable(cell: tuple) -> bool:
        return floor.esidlbhbhw(cell[0] + half, cell[1] + half) != -1

    def opened(plate) -> set:
        wire = plate.nexhtmlmxh
        return {(g.x, g.y) for g in wire.ytztewxdin} if wire is not None else set()

    def slid(cell: tuple) -> tuple:
        dx, dy = gates[cell].hluvhlvimq()
        return cell[0] + dx * STEP, cell[1] + dy * STEP

    def shut_step(cell: tuple, dx: int, dy: int) -> tuple | None:
        nxt = (cell[0] + dx * STEP, cell[1] + dy * STEP)
        return nxt if walkable(nxt) and nxt not in gates else None

    best = _bfs(start, lambda c: c == target, shut_step)
    if len(world.drofvwhbxb) >= 2:
        for plate in plates:
            spot = (plate.x, plate.y)
            to_plate = _bfs(start, lambda c, spot=spot: c == spot, shut_step)
            if not to_plate:
                continue
            a, held = len(to_plate), opened(plate)
            moved_aside = {slid(c) for c in held}

            def held_step(state: tuple, dx: int, dy: int, a=a, held=held, moved_aside=moved_aside) -> tuple | None:
                (x, y), made = state
                nxt = (x + dx * STEP, y + dy * STEP)
                is_open = made >= a
                if not walkable(nxt) or (nxt in gates and not (nxt in held and is_open)):
                    return None
                if is_open and nxt in moved_aside:
                    return None
                return nxt, min(made + 1, a)

            through = _bfs((start, 0), lambda s: s[0] == target, held_step)
            if through is not None:
                plan = [*to_plate, GameAction.ACTION5, *through]
                if best is None or len(plan) < len(best):
                    best = plan
    return None if best is None else [ActionInput(id=a) for a in best]
