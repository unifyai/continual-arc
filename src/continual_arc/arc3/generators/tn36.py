"""tn36 at level 0: a piece on a 9 x 8 checkerboard of 4-pixel cells, its outline drawn on another cell, and a program
of five instructions below it, each two clickable bits (ACTION6 on a bit toggles it): no bit does nothing, the bar
alone steps the piece left, the stem alone steps it right, both step it down. Clicking the round button runs the
program from the piece's start; it wins if the piece ends on its outline, and otherwise the piece goes back to start.
Every click shortens the timer bar along the top by one pixel, and the 61st click loses the level.

generate keeps level 0's frame (the board, the program bar, the button, the timer and the unused panel off the left
edge, as level 0 has them) and draws the outline's cell anywhere on the board and the piece's on a cell five such
steps reach it from (at most five steps in all, none upward); the bits start five to ten flips away from one program
that wins, its steps in a random order. It keeps the layout when the fewest clicks that win, toggles plus the run,
are 5 to 10 (level 0's are 7).

solve tries all 1024 programs on the piece's start, by the moves the game's code gives each instruction, keeps those
that end on the outline, and picks the one fewest bit toggles away (the first such in a fixed order); it clicks those
bits and then the button, and the answer is checked in the real engine. It takes a few milliseconds.
"""
from __future__ import annotations

import itertools
import random

from arcengine import ActionInput, GameAction, Level

from continual_arc.arc3 import levels

GAME = "tn36"
PIECE, OUTLINE, RUN = "bltjrl4", "taptxx", "sucqgkbuojsa"
BITS = ("Maidxzhowygv", "Maidxzvenlef")  # the bar (bit 0) and the stem (bit 1) of an instruction
SET = 5                 # a set bit's color, which the game reads at pixel [1, 2]
COLUMNS = (19, 24, 29, 34, 39)
BIT_Y = (41, 44)
CELL = 4
BOARD_X, BOARD_Y, BOARD_W, BOARD_H = 14, 9, 9, 8
OUTLINE_I, OUTLINE_J = range(1, 8), range(0, 7)  # cells whose outline clears the board's frame and the program bar
STEP = {0: (0, 0), 1: (-CELL, 0), 2: (CELL, 0), 3: (0, CELL)}
FLIPS = range(5, 11)    # bits flipped from a winning program; level 0's start is six from its only one
CLICKS = (5, 10)        # fewest winning clicks kept; level 0's are 7
TRIES = 200

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_SPRITES = _MODULE.sprites
_FRAME = [s for s in _LEVEL0.get_sprites() if s.name not in (PIECE, OUTLINE, *BITS)]
_BASE_LAYOUT = levels.layout(_LEVEL0)
_PROGRAMS = list(itertools.product(range(4), repeat=len(COLUMNS)))


def _cell(i: int, j: int) -> tuple[int, int]:
    return BOARD_X + CELL * i, BOARD_Y + CELL * j


def _plan(start: tuple, goal: tuple, bits: list[list[bool]]) -> tuple[int, ...] | None:
    """The winning program fewest toggles from ``bits`` (bits[column][0 bar, 1 stem]), or None."""
    best, best_cost = None, None
    for program in _PROGRAMS:
        x, y = start
        for op in program:
            x, y = x + STEP[op][0], y + STEP[op][1]
        if (x, y) != goal:
            continue
        cost = sum((op >> b & 1) != bits[c][b] for c, op in enumerate(program) for b in range(2))
        if best_cost is None or cost < best_cost:
            best, best_cost = program, cost
    return best


def _toggles(program: tuple, bits: list[list[bool]]) -> list[tuple[int, int]]:
    return [(c, b) for c, op in enumerate(program) for b in range(2) if (op >> b & 1) != bits[c][b]]


def _level(piece: tuple, outline: tuple, bits: list[list[bool]]) -> Level:
    sprites = [s.clone() for s in _FRAME]
    sprites.append(_SPRITES[PIECE].clone().set_position(*piece))
    sprites.append(_SPRITES[OUTLINE].clone().set_position(outline[0] - 1, outline[1] - 1))
    for column, x in enumerate(COLUMNS):
        for b, (name, y) in enumerate(zip(BITS, BIT_Y, strict=True)):
            bit = _SPRITES[name].clone().set_position(x, y)
            sprites.append(bit.color_remap(None, SET) if bits[column][b] else bit)
    return Level(sprites=sprites, grid_size=_LEVEL0.grid_size, data=dict(_LEVEL0._data))


def generate(rng: random.Random) -> Level | None:
    for _ in range(TRIES):
        goal_i, goal_j = rng.choice(OUTLINE_I), rng.choice(OUTLINE_J)
        down = rng.randint(0, min(goal_j, len(COLUMNS)))
        across = rng.randint(down - len(COLUMNS), len(COLUMNS) - down)
        start_i, start_j = goal_i - across, goal_j - down
        if (down == 0 and across == 0) or not 0 <= start_i < BOARD_W:
            continue
        # The bits start a few flips away from one program that wins, its steps in a random order.
        ops = [3] * down + [2 if across > 0 else 1] * abs(across)
        ops += [0] * (len(COLUMNS) - len(ops))
        rng.shuffle(ops)
        bits = [[bool(op >> b & 1) for b in range(2)] for op in ops]
        for c, b in rng.sample([(c, b) for c in range(len(COLUMNS)) for b in range(2)], rng.choice(FLIPS)):
            bits[c][b] = not bits[c][b]
        start, goal = _cell(start_i, start_j), _cell(goal_i, goal_j)
        program = _plan(start, goal, bits)
        if not CLICKS[0] <= len(_toggles(program, bits)) + 1 <= CLICKS[1]:
            continue
        level = _level(start, goal, bits)
        if levels.layout(level) != _BASE_LAYOUT:
            return level
    return None


def _read(level) -> tuple:
    sprites = level.get_sprites()
    (piece,) = [s for s in sprites if s.name == PIECE]
    (outline,) = [s for s in sprites if s.name == OUTLINE]
    (run,) = [s for s in sprites if s.name == RUN]
    bits = {(s.x, s.y): s for s in sprites if s.name in BITS}
    grid = [[bits[(x, y)] for y in BIT_Y] for x in COLUMNS]
    goal = (outline.x + outline.scale, outline.y + outline.scale)
    return (piece.x, piece.y), goal, grid, run


def solve(level) -> list | None:
    start, goal, grid, run = _read(level)
    bits = [[int(s.pixels[1, 2]) == SET for s in column] for column in grid]
    program = _plan(start, goal, bits)
    if program is None:
        return None

    def click(sprite) -> ActionInput:
        return ActionInput(id=GameAction.ACTION6, data={"x": sprite.x + sprite.width // 2,
                                                        "y": sprite.y + sprite.height // 2})

    actions = [click(grid[c][b]) for c, b in _toggles(program, bits)] + [click(run)]
    return actions if levels.replay(GAME, level, actions) else None
