"""sb26 at level 0: a row of colored targets at the top, a frame of four empty slots, and four colored blocks in the
palette under the energy bar. Clicking a block then a slot (or another block) moves or swaps it, one energy each;
ACTION5 runs the frame, which walks its slots left to right against the targets and wins once every target is
matched, and a mismatch or an empty slot fails the run. Energy is 64; running out loses.

A layout puts the target row, the frame and the palette at fresh places on the board: the target row's x, the
frame's x and y anywhere between the targets and the energy bar, the palette's x and spacing. It draws four distinct
colors for the targets and lays the same four colors out in the palette in another order, so every level is
solvable in the same way as level 0.

solve places, for each slot in turn, the palette block whose color is that slot's target (two clicks), then runs the
frame: 4 x 2 + 1 = 9 actions, verified in the real engine.
"""
from __future__ import annotations

import random

from continual_arc.arc3 import levels

GAME = "sb26"

COLORS = (6, 9, 11, 12, 14, 15)
PALETTE_Y = 56  # below the energy bar (row 53), where the game puts a return spot under each block
TARGET_STEP = 7  # the target row's pitch, as every level has it
SLOT_STEP = 6  # the game's slot pitch inside a frame (kojduumcap)


def _sprites():
    return levels.load_module(GAME).sprites


def generate(rng: random.Random):
    from arcengine import Level

    sprites = _sprites()
    colors = rng.sample(COLORS, 4)
    palette = colors[:]
    while palette == colors:
        rng.shuffle(palette)
    target_x = rng.randint(1, 64 - 1 - 4 * TARGET_STEP)
    frame_x, frame_y = rng.randint(1, 64 - 1 - 28), rng.randint(11, 42)
    step = rng.randint(7, 11)
    palette_x = rng.randint(1, 64 - 1 - (3 * step + 6))
    placed = [sprites["zpwrpmkvsv"].clone().set_position(0, 53),
              sprites["qdmvvkvhaz4"].clone().set_position(frame_x, frame_y)]
    for i, color in enumerate(colors):
        x = target_x + i * TARGET_STEP
        placed.append(sprites["quhhhthrri"].clone().set_position(x, 1).color_remap(None, color))
        placed.append(sprites["uzxwqmkrmk"].clone().set_position(x - 1, 0))
        placed.append(sprites["susublrply"].clone().set_position(frame_x + 2 + i * SLOT_STEP, frame_y + 2))
    for i, color in enumerate(palette):
        placed.append(sprites["lngftsryyw"].clone().set_position(palette_x + i * step, PALETTE_Y)
                      .color_remap(None, color))
    return Level(sprites=placed, grid_size=(64, 64))


def _plan(level):
    """The clicks that fill each slot with its target's color, then the run."""
    from arcengine import ActionInput, GameAction

    def by_name(name):
        return sorted(level.get_sprites_by_name(name), key=lambda s: (s.y, s.x))
    targets = by_name("quhhhthrri")
    (frame,) = by_name("qdmvvkvhaz4")
    blocks = by_name("lngftsryyw")
    actions = []
    for i, target in enumerate(targets):
        color = int(target.pixels[0, 0])
        block = next(b for b in blocks if int(b.pixels[1, 1]) == color)
        blocks.remove(block)
        actions.append(ActionInput(id=GameAction.ACTION6, data={"x": block.x + 2, "y": block.y + 2}))
        actions.append(ActionInput(id=GameAction.ACTION6,
                                   data={"x": frame.x + 2 + i * SLOT_STEP + 2, "y": frame.y + 2 + 2}))
    actions.append(ActionInput(id=GameAction.ACTION5))
    return actions


def solve(level):
    actions = _plan(level)
    return actions if levels.replay(GAME, level, actions) else None
