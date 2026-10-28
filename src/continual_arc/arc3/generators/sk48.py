"""sk48 at level 0: an arm on a 5 x 5 board of 6-pixel cells reaches right from a head that slides along a rail
down the board's left side. ACTION4 extends the arm a cell and ACTION3 retracts it, ACTION1/2 move head and arm a row
up or down; the arm pushes the colored blocks it meets, carries those lying on it, and slides under a block that
cannot move further, which is how a block gets onto it. The level is won when the blocks on the arm, read from the
head out, have the colors of the row shown below the board (level 0: three blocks, 8, 9 and 14, to be held as 8, 14,
9).

generate keeps level 0's board, rail, step bar and display, and draws the head's row, the arm's starting length (one to
three segments), a cell for each of level 0's three blocks anywhere on the board off the arm, and the display's order
of their colors. Almost every such layout is solvable; one is kept when the search below finds a win of 9 to 21 moves
(level 0's is 14) among its first DRAW_STATES states, which keeps a draw fast and its length near level 0's.

solve is a breadth-first search over (head, arm segments, block cells), stepping a port of the game's own move code
(hgivzuhjvj, bnrdrdiakd, qzvlbxkjgk, gvtmoopqgy) on plain positions; moves that leave the game's pending-push list set
without moving anything (it would swallow the next action) are not taken. The port matched the engine on 36,000
random moves. A kept level is solved in well under a tenth of a second.
"""
from __future__ import annotations

import random
from collections import deque
from itertools import permutations

from arcengine import ActionInput, GameAction

from continual_arc.arc3 import levels

GAME = "sk48"

HEAD, SEG, BLOCK, RAIL, BOARD = "ejlpqgojjt", "qtjqovumxf", "elmjchdqcn", "irkeobngyh", "rtwdndlhdf"
STEP = 6                                    # udenqlsrfq: a cell, and how far a move takes the arm
DIRS = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}   # hhvuoijeua
DISPLAY_Y = 53                              # fzjeqdahvs: heads below it only show a target
MOVES = ((GameAction.ACTION1, 0, -1), (GameAction.ACTION2, 0, 1), (GameAction.ACTION3, -1, 0),
         (GameAction.ACTION4, 1, 0))
LENGTH = (9, 21)
TRIES = 40
DRAW_STATES = 4_000      # a draw whose shortest win needs more of the search than this is redrawn
MAX_STATES = 200_000

_MODULE = levels.load_module(GAME)
_SPRITES = _MODULE.sprites
_BASE = _MODULE.levels[0]
_BASE_LAYOUT = levels.layout(_BASE)


class _Thing:
    """A head, segment or block of the port: where it is, which way a segment lies, a block's color."""
    __slots__ = ("x", "y", "flat", "color")

    def __init__(self, x: int, y: int, flat: bool = True, color: int = -1):
        self.x, self.y, self.flat, self.color = x, y, flat, color


class _Rules:
    """What a level fixes: the board, the rails, the arm's direction, the target and the sprites that never move."""

    def __init__(self, level):
        heads = level.get_sprites_by_tag("epdquznwmq")
        head = heads[0]                                  # the game starts with the first head selected
        (pair,) = [h for h in heads if h is not head and h.pixels[2, 2] == head.pixels[2, 2]]
        self.flat = head.rotation in (0, 180)
        self.dir = DIRS[head.rotation]
        (board,) = level.get_sprites_by_tag("jtteddgeyl")
        self.box = (board.x, board.y, board.x + board.pixels.shape[1] * STEP, board.y + board.pixels.shape[0] * STEP)
        self.rails = [(r.x, r.y, r.x + r.width, r.y + r.height) for r in level.get_sprites_by_tag(RAIL)]
        segs = level.get_sprites_by_name(SEG)
        blocks = level.get_sprites_by_tag(BLOCK)
        self.blocks = [b for b in blocks if b.y < DISPLAY_Y]
        self.colors = [int(b.pixels[1, 1]) for b in self.blocks]
        self.fixed_blocks = {(b.x, b.y): _Thing(b.x, b.y, color=int(b.pixels[1, 1]))
                             for b in blocks if b.y >= DISPLAY_Y}
        self.arm = self.chain(head, segs)
        shown = self.chain(pair, segs)
        self.fixed_segs = {(s.x, s.y, s.rotation in (0, 180)): _Thing(s.x, s.y, s.rotation in (0, 180))
                           for s in segs if s.y >= DISPLAY_Y}
        on_display = [self.fixed_blocks.get((s.x, s.y)) for s in shown]
        self.target = [b.color for b in on_display if b is not None][:len(shown) - 1]
        self.start = ((head.x, head.y), tuple((s.x, s.y) for s in self.arm),
                      tuple((b.x, b.y) for b in self.blocks))

    def chain(self, head, segs: list) -> list:
        """The segments from a head outwards, as on_set_level finds them."""
        dx, dy = DIRS[head.rotation]
        flat, out, x, y = head.rotation in (0, 180), [], head.x, head.y
        while (seg := next((s for s in segs if (s.x, s.y) == (x, y) and (s.rotation in (0, 180)) == flat), None)):
            out.append(seg)
            x, y = x + dx * STEP, y + dy * STEP
        return out


class _Board:
    """One state of the port, stepped by the game's own rules. The game moves nothing until every push of a move is
    decided, so the lookups by place, built once, hold for the whole move (a retracted segment included)."""

    def __init__(self, rules: _Rules, state: tuple):
        (hx, hy), segs, blocks = state
        self.rules = rules
        self.head = _Thing(hx, hy)
        self.segs = [_Thing(x, y, rules.flat) for x, y in segs]
        self.blocks = [_Thing(x, y, color=c) for (x, y), c in zip(blocks, rules.colors, strict=True)]
        self.blocks_at = {**rules.fixed_blocks, **{(b.x, b.y): b for b in self.blocks}}
        self.segs_at = {**rules.fixed_segs, **{(s.x, s.y, s.flat): s for s in self.segs}}
        self.pending: list[_Thing] = []              # pzzwlsmdt

    def state(self) -> tuple:
        return ((self.head.x, self.head.y), tuple((s.x, s.y) for s in self.segs),
                tuple((b.x, b.y) for b in self.blocks))

    def off(self, t: _Thing, mx: int, my: int) -> bool:             # qzvlbxkjgk (level 0 has no walls)
        x0, y0, x1, y1 = self.rules.box
        x, y = t.x + mx * STEP, t.y + my * STEP
        return x < x0 or x + STEP > x1 or y < y0 or y + STEP > y1

    def block_at(self, x: int, y: int) -> _Thing | None:            # ebribtrdgw
        return self.blocks_at.get((x, y))

    def seg_at(self, x: int, y: int, flat: bool, skip: _Thing | None = None) -> _Thing | None:   # pptqisyill
        seg = self.segs_at.get((x, y, flat))
        return None if seg is skip else seg

    def push(self, t: _Thing, mx: int, my: int, moved: set, pusher: _Thing | None = None) -> bool:   # bnrdrdiakd
        if t in moved:
            return True
        is_seg = t.color < 0
        if self.off(t, mx, my) and not (is_seg and (self.rules.dir == (-mx, -my) or self.off(t, 0, 0))):
            return False
        dx, dy = mx * STEP, my * STEP
        if is_seg:
            for ox, oy in ((0, 0), (dx, dy)):
                if b := self.block_at(t.x + ox, t.y + oy):
                    if self.push(b, mx, my, moved, pusher=t):
                        moved.add(b)
                    elif (self.rules.dir[0] == 0) != (mx == 0):
                        return False
                    elif not self.seg_at(b.x, b.y, mx != 0):
                        self.pending.append(b)
        else:
            flat = mx != 0
            for ox, oy in ((0, 0), (dx, dy)):
                if self.seg_at(t.x + ox, t.y + oy, not flat, skip=pusher):
                    return False                 # a segment across the move holds the block
            ahead = self.block_at(t.x + dx, t.y + dy)
            if ahead and not self.push(ahead, mx, my, moved):
                return False
            if not self.seg_at(t.x, t.y, flat) and self.seg_at(t.x + dx, t.y + dy, flat):
                self.pending.append(t)
        moved.add(t)
        return True

    def move(self, mx: int, my: int) -> str:
        """hgivzuhjvj and the frames after it: 'moved', 'still' (nothing changes) or 'stuck' (nothing moved but the
        game's pending list is set, so it would swallow the next action)."""
        moved: set = set()
        new = None
        if (mx, my) == self.rules.dir:
            if self.off(self.segs[-1], mx, my):
                return "still"
            for s in list(self.segs):
                self.push(s, mx, my, moved)
            new = _Thing(self.head.x, self.head.y, self.rules.flat)
        elif (mx, my) == (-self.rules.dir[0], -self.rules.dir[1]):
            if len(self.segs) == 1:
                return "still"
            self.segs.pop(0)
            for s in list(self.segs):
                self.push(s, mx, my, moved)
        elif self.rail_at(self.head.x + 2 + mx * STEP // 2, self.head.y + 2 + my * STEP // 2):
            for s in list(self.segs):
                if not self.push(s, mx, my, moved):
                    return "stuck" if self.pending else "still"
            moved.add(self.head)
        else:
            return "still"
        if not moved:
            return "stuck"
        for t in moved:
            t.x, t.y = t.x + mx * STEP, t.y + my * STEP
        if new is not None:
            self.segs.insert(0, new)
        return "moved"

    def rail_at(self, x: int, y: int) -> bool:
        return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in self.rules.rails)

    def won(self) -> bool:                                           # gvtmoopqgy, after a move
        at = {**self.rules.fixed_blocks, **{(b.x, b.y): b for b in self.blocks}}
        held = [b.color for s in self.segs if (b := at.get((s.x, s.y)))]
        return held[:len(self.rules.target)] == self.rules.target


def _search(rules: _Rules, max_states: int = MAX_STATES) -> list | None:
    """The fewest moves to a win, by breadth-first search over the port's states; None once ``max_states`` states
    are found without one."""
    start = rules.start
    parent: dict = {start: None}
    queue = deque([start])
    while queue and len(parent) < max_states:
        state = queue.popleft()
        for action, mx, my in MOVES:
            board = _Board(rules, state)
            if board.move(mx, my) != "moved":
                continue
            nxt = board.state()
            if nxt in parent:
                continue
            parent[nxt] = (state, action)
            if board.won():
                path = []
                while parent[nxt] is not None:
                    nxt, act = parent[nxt]
                    path.append(act)
                return path[::-1]
            queue.append(nxt)
    return None


def solve(level) -> list | None:
    path = _search(_Rules(level))
    return None if path is None else [ActionInput(id=a) for a in path]


def _level(head_row: int, arm: int, cells: list[tuple[int, int]], order: tuple[int, ...]):
    """Level 0's board, rail, bar and display, with the head on ``head_row``, an arm of ``arm`` segments, level 0's
    blocks (_COLORS) on ``cells`` (column, row) and the display showing their colors in ``order``."""
    sprites = _BASE.get_sprites()
    (head, shown) = [s for s in sprites if s.name == HEAD]      # level 0 lists the playing head first
    (board,) = [s for s in sprites if s.name == BOARD]
    shown_cells = sorted((s.x, s.y) for s in sprites if s.name == BLOCK and s.y >= DISPLAY_Y)
    hx, hy = head.x, board.y + STEP * head_row
    dx, dy = DIRS[head.rotation]
    out = [_SPRITES[HEAD].clone().set_position(hx, hy).set_rotation(head.rotation)]
    out += [s.clone() for s in sprites if s.name not in (HEAD, BLOCK, SEG)]
    out.append(shown.clone())
    out += [s.clone() for s in sprites if s.name == SEG and s.y >= DISPLAY_Y]
    out += [_SPRITES[SEG].clone().set_position(hx + dx * STEP * k, hy + dy * STEP * k).set_rotation(head.rotation)
            for k in range(arm)]
    out += [_SPRITES[BLOCK].clone().set_position(board.x + STEP * c, board.y + STEP * r).color_remap(None, color)
            for color, (c, r) in zip(_COLORS, cells, strict=True)]
    out += [_SPRITES[BLOCK].clone().set_position(x, y).color_remap(None, color)
            for color, (x, y) in zip(order, shown_cells, strict=True)]
    return type(_BASE)(sprites=out, grid_size=_BASE.grid_size, data=dict(_BASE._data))


_COLS = _BASE.get_sprites_by_name(BOARD)[0].pixels.shape[1]
_ROWS = _BASE.get_sprites_by_name(BOARD)[0].pixels.shape[0]
_COLORS = [int(b.pixels[1, 1]) for b in _BASE.get_sprites_by_name(BLOCK) if b.y < DISPLAY_Y]


def generate(rng: random.Random):
    orders = sorted(permutations(sorted(_COLORS)))
    for _ in range(TRIES):
        head_row = rng.randrange(_ROWS)
        arm = rng.randint(1, 3)
        free = [(c, r) for c in range(_COLS) for r in range(_ROWS) if not (r == head_row and c < arm - 1)]
        cells = rng.sample(free, len(_COLORS))
        level = _level(head_row, arm, cells, rng.choice(orders))
        path = _search(_Rules(level), DRAW_STATES)
        if path is None or not LENGTH[0] <= len(path) <= LENGTH[1]:
            continue
        if levels.layout(level) == _BASE_LAYOUT:
            continue
        return level
    return None
