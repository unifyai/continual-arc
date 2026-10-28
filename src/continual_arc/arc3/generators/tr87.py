"""tr87 at level 0: the top half shows rules, each a frame of one A symbol, an arrow and a frame of one B symbol; the
bottom half shows a row of A symbols and below it a row of B symbols. ACTION3/4 move the cursor along the B row
(wrapping), ACTION1/2 cycle the B symbol under it back or forward through its seven shapes, and the level is won when
the B row is the A row translated symbol by symbol through the rules. Every action spends one of 128 steps.

A layout draws five or six rules (distinct A symbols to distinct B symbols, from level 0's), puts them in a random
order into the two columns of three slots, shifts that block across the top half, draws a source row of four to six
of the rules' A symbols and a B row of level 0's B symbols (the engine then turns each by its fixed per-place shift),
and places both rows at a random column. Every source symbol has a rule, so every layout is solvable; draws whose
shortest solution is far from level 0's are redrawn.

solve reads the rules, the source row and the B row's shapes after the reset from the real engine, translates the
source, takes the shorter way round each symbol's cycle of seven, and finds the shortest cursor tour of the places to
change by a breadth-first search over (cursor, places done); the answer is checked in the real engine."""
from __future__ import annotations

import random
from collections import deque

from continual_arc.arc3 import levels

GAME = "tr87"
SYM = "nxkictbbvzt"
CYCLE = 7  # shapes per symbol family (kjgicbtgrt in the game)
STEP = 7   # pixels between neighboring symbols of a row (iokndxodxw)
LENGTH = (10, 19)  # shortest solution lengths kept; level 0's is 14

_MODULE = levels.load_module(GAME)
_LEVEL0 = _MODULE.levels[0]
_A = sorted({s.name for s in _LEVEL0.get_sprites() if s.name.startswith(SYM + "A")})
_B = sorted({s.name for s in _LEVEL0.get_sprites() if s.name.startswith(SYM + "B")})
# Level 0's six rule slots (frame A's corner): two columns of three.
_SLOTS = [(12 + 23 * c, 4 + 9 * r) for c in range(2) for r in range(3)]


def _level(rules: list[tuple[str, str]], slots: list[tuple[int, int]], source: list[str], target: list[str], x0: int):
    from arcengine import Level

    sp = _MODULE.sprites
    out = [sp["background"].clone().set_position(0, 34).set_scale(32)]
    for (a, b), (fx, fy) in zip(rules, slots, strict=False):
        out += [
            sp["gyrdjxybtcmA"].clone().set_position(fx, fy),
            sp["gyrdjxybtcmB"].clone().set_position(fx + 10, fy),
            sp["iqrduxrukrk"].clone().set_position(fx + 3, fy + 3),
            sp[a].clone().set_position(fx + 1, fy + 1),
            sp[b].clone().set_position(fx + 11, fy + 1),
        ]
    for i, (a, b) in enumerate(zip(source, target, strict=False)):
        x = x0 + STEP * i
        out += [
            sp["gyrdjxybtcmA"].clone().set_position(x - 1, 40),
            sp["gyrdjxybtcmB"].clone().set_position(x - 1, 51),
            sp[a].clone().set_position(x, 41),
            sp[b].clone().set_position(x, 52),
        ]
    return Level(sprites=out, grid_size=_LEVEL0.grid_size, data=dict(_LEVEL0._data))


def _number(name: str) -> int:
    return int(name[-1])


def _plan(game) -> list[int] | None:
    """The shortest action ids that win from the game's reset state, or None when a source symbol has no rule."""
    rules = {}
    for lhs, rhs in game.cifzvbcuwqe:
        if len(lhs) != 1 or len(rhs) != 1:
            return None
        rules.setdefault(lhs[0].name, rhs[0].name)  # the game takes the first matching rule
    source = [s.name for s in game.zvojhrjxxm]
    target = [s.name for s in game.ztgmtnnufb]
    if any(a not in rules for a in source) or len(source) != len(target):
        return None
    want = [rules[a] for a in source]
    if any(w[:-1] != t[:-1] for w, t in zip(want, target, strict=False)):
        return None
    turns = []
    for w, t in zip(want, target, strict=False):
        d = (_number(w) - _number(t)) % CYCLE
        turns.append([2] * d if d <= CYCLE - d else [1] * (CYCLE - d))
    n = len(target)
    need = sum(1 << i for i, t in enumerate(turns) if t)
    if not need:
        return None  # already translated: no change would be checked
    # Cursor tour: breadth-first over (cursor, places reached), a place counting as done on arrival.
    start = (0, 1 & need)
    prev = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state[1] == need:
            break
        pos, done = state
        for move, step in ((3, -1), (4, 1)):
            nxt_pos = (pos + step) % n
            nxt = (nxt_pos, done | ((1 << nxt_pos) & need))
            if nxt not in prev:
                prev[nxt] = (state, move)
                queue.append(nxt)
    moves = []
    while prev[state] is not None:
        state, move = prev[state]
        moves.append(move)
    moves.reverse()
    actions, pos, visited = list(turns[0]), 0, {0}
    for move in moves:
        actions.append(move)
        pos = (pos + (1 if move == 4 else -1)) % n
        if pos not in visited:
            visited.add(pos)
            actions += turns[pos]
    return actions


def generate(rng: random.Random):
    for _ in range(50):
        k = rng.randint(5, len(_A))
        rules = list(zip(rng.sample(_A, k), rng.sample(_B, k), strict=False))
        dx, dy = rng.randint(-12, 11), rng.randint(-4, 4)
        slots = [(x + dx, y + dy) for x, y in rng.sample(_SLOTS, k)]
        n = rng.randint(4, 6)
        source = [rng.choice(rules)[0] for _ in range(n)]
        target = [rng.choice(_B) for _ in range(n)]
        x0 = rng.randint(1, 64 - STEP * n)
        level = _level(rules, slots, source, target, x0)
        game, _ = levels.new_game(GAME, level)
        plan = _plan(game)
        if plan is not None and LENGTH[0] <= len(plan) <= LENGTH[1]:
            return level
    return None


def solve(level) -> list | None:
    from arcengine import ActionInput, GameAction

    game, _ = levels.new_game(GAME, level)
    plan = _plan(game)
    if plan is None:
        return None
    actions = [ActionInput(id=GameAction.from_id(a)) for a in plan]
    return actions if levels.replay(GAME, level, actions) else None
