"""ARC-AGI-3 levels for the third track: load a pinned public game, play any level of its first level's rules in the
real engine, replay a solution, search for one, and check that a game's generator draws fresh, solvable levels of
about one difficulty.

A game is one module of ARC Prize's ``arcengine`` (MIT), pinned in the data root. Its class reads the module's
``levels`` list when it is built, so a level of our own plays unchanged by setting that list first. A task of the
track is a game's rules at its first level (level 0): its sprite types and its level data. An instance is a fresh
level of those rules, drawn by the game's generator (``continual_arc.arc3.generators.<game>``):

    GAME = "ls20"
    def generate(rng: random.Random) -> Level: ...        # level 0's sprite types and data, a new layout
    def solve(level: Level) -> list[ActionInput] | None:  # optional: a winning sequence; bfs() when absent
    REDRAWS = 0                                           # optional: layouts the last generate() rejected itself

Most games keep a level's layout in its sprites. A few keep it in tables of the game module instead (bp35 reads its
board from a module-level table keyed by level index); a level of such a game carries the tables it replaces as
``level.module_patch``, a dict of module attribute names to values, which :func:`new_game` sets on the fresh module
before the game is built, and which :func:`layout` reads. Every search is bounded by states expanded, never by time, so
that a level is solved, or not, the same way on every machine; :func:`check` times generate and solve only as a speed
filter on top. :func:`check` is the gate a generator passes to enter the track (the filters above SEEDS).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import random
import signal
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from continual_arc.data import data_root

_SEQUENCE = itertools.count()

# A generator's gate (check): over SEEDS draws, a level within GENERATE_S, redraws (draws that gave no level, and the
# layouts a generator rejected itself, its REDRAWS) under REDRAW_SHARE of all layouts drawn, every kept level solved
# within SOLVE_S and won on replay, its solution LENGTH_BAND times level 0's, DISTINCT_SHARE of layouts distinct and none
# level 0's, and only level 0's sprite types and data keys.
SEEDS = 50
GENERATE_S = 2.0
REDRAW_SHARE = 0.2
SOLVE_S = 60.0
LENGTH_BAND = (0.5, 2.0)
DISTINCT_SHARE = 0.9


def games_dir() -> Path:
    """Where the pinned public games live (``continual-arc data fetch --datasets arc3`` puts them there)."""
    return data_root() / "arc-agi-3" / "environment_files"


def game_file(game_id: str) -> Path:
    """The game's pinned source: its one version folder under :func:`games_dir`."""
    (path,) = (games_dir() / game_id).glob("*/*.py")
    return path


def metadata(game_id: str) -> dict:
    (path,) = (games_dir() / game_id).glob("*/metadata.json")
    return json.loads(path.read_text())


def baseline(game_id: str) -> int:
    """The human action count ARC Prize publishes for the game's level 0."""
    return int(metadata(game_id)["baseline_actions"][0])


def load_module(game_id: str) -> Any:
    """A fresh copy of the game's module, so that setting its ``levels`` touches no other game built from it."""
    path = game_file(game_id)
    spec = importlib.util.spec_from_file_location(f"continual_arc_arc3_{game_id}_{next(_SEQUENCE)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def game_class(module: Any) -> type:
    """The module's game: its one class built on arcengine's ``ARCBaseGame``."""
    (cls,) = (v for v in vars(module).values() if isinstance(v, type) and v.__module__ == module.__name__
              and any(b.__name__ == "ARCBaseGame" for b in v.__mro__))
    return cls


def level0(game_id: str) -> Any:
    """The game's public level 0, whose rules a task of the track is."""
    return load_module(game_id).levels[0]


def new_game(game_id: str, level: Any) -> tuple[Any, Any]:
    """The game built on ``level`` alone, after its reset: (game, frame data). The level's ``module_patch``, if it has
    one, replaces those attributes of the fresh module first, for a game whose layout lives in module tables."""
    from arcengine import ActionInput, GameAction

    module = load_module(game_id)
    module.levels = [level]
    for name, value in getattr(level, "module_patch", {}).items():
        setattr(module, name, copy.deepcopy(value))
    game = game_class(module)()
    return game, game.perform_action(ActionInput(id=GameAction.RESET))


def won(frame: Any) -> bool:
    """Whether a step completed the level: a game of one level is won when its level is."""
    from arcengine import GameState

    return frame.state == GameState.WIN or getattr(frame, "levels_completed", 0) >= 1


def replay(game_id: str, level: Any, actions: list[Any]) -> bool:
    """Whether ``actions``, played from the level's reset, win it."""
    game, frame = new_game(game_id, level)
    for action in actions:
        frame = game.perform_action(action)
        if won(frame):
            return True
    return False


def state_key(game: Any, frame: Any) -> bytes:
    """A search state: the frame after the action and where every sprite of the level is."""
    import numpy as np

    h = hashlib.blake2b(np.asarray(frame.frame, dtype=np.int16).tobytes(), digest_size=16)
    for s in game.current_level.get_sprites():
        h.update(f"{s.name}:{s.x},{s.y};".encode())
    return h.digest()


def bfs(game_id: str, level: Any, max_states: int = 200_000) -> list[Any] | None:
    """A shortest winning sequence of the engine's own valid actions, by breadth-first search over deep copies of the
    game with states deduplicated (:func:`state_key`); None when ``max_states`` expansions find none, the same answer
    on every machine. Solves games whose states are few, such as small mazes and click puzzles; a game whose valid
    actions or states are too many needs a ``solve`` of its own."""
    from arcengine import GameState

    game, frame = new_game(game_id, level)
    seen, queue, expanded = {state_key(game, frame)}, deque([(game, [])]), 0
    while queue and expanded < max_states:
        current, path = queue.popleft()
        for action in current._get_valid_actions():
            nxt = copy.deepcopy(current)
            out = nxt.perform_action(action)
            expanded += 1
            if won(out):
                return [*path, action]
            if out.state == GameState.GAME_OVER:
                continue
            key = state_key(nxt, out)
            if key not in seen:
                seen.add(key)
                queue.append((nxt, [*path, action]))
    return None


def signature(level: Any) -> tuple[frozenset, frozenset]:
    """A level's rules as the track reads them: its sprite types (names) and its data keys."""
    return (frozenset(s.name for s in level.get_sprites()), frozenset(level._data))


def _plain(value: Any) -> Any:
    """``value`` with every array as nested lists, so that a layout's key sees every cell (an array's str elides)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def layout(level: Any) -> tuple:
    """A level's layout, for telling levels apart: every sprite's type, place, orientation and pixels (a level may
    carry its content in a sprite's pixels, as cd82's picture is), its data, and the module tables it replaces."""
    import numpy as np

    sprites = tuple(sorted((s.name, s.x, s.y, s.rotation, s.mirror_ud, s.mirror_lr, np.shape(s.pixels),
                            hashlib.blake2b(np.ascontiguousarray(s.pixels).tobytes(), digest_size=8).hexdigest())
                           for s in level.get_sprites()))
    return (sprites, json.dumps(_plain(level._data), sort_keys=True, default=str),
            json.dumps(_plain(getattr(level, "module_patch", {})), sort_keys=True, default=str))


@dataclass
class Report:
    """What :func:`check` found for one generator."""
    game: str
    level0_length: int | None
    draws: int = 0
    redraws: int = 0
    kept: int = 0
    solved: int = 0
    lengths: list[int] = field(default_factory=list)
    distinct: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def line(self) -> str:
        band = f"{min(self.lengths)}-{max(self.lengths)}" if self.lengths else "-"
        return (f"{self.game}: {'PASS' if self.passed else 'FAIL'}; level 0 solved in {self.level0_length}; {self.kept} "
                f"kept of {self.draws}, {self.redraws} redrawn inside, {self.solved} solved, lengths {band}, "
                f"{self.distinct} distinct" + (f"; {'; '.join(self.failures[:4])}" if self.failures else ""))


@contextmanager
def _deadline(seconds: float):
    """Stop the call inside with TimeoutError once ``seconds`` of wall time pass, by SIGALRM, so that a solve that
    hangs fails the gate rather than holding it (check runs in a process's main thread)."""
    def expire(signum: int, frame: Any) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _timed(seconds: float, fn: Any, *args: Any) -> tuple[Any, bool]:
    """``fn(*args)`` and whether it finished within ``seconds``."""
    try:
        with _deadline(seconds):
            return fn(*args), True
    except TimeoutError:
        return None, False


def check(generator: Any, seeds: int = SEEDS) -> Report:
    """Whether ``generator`` (a module with GAME, generate and optionally solve and REDRAWS) passes the track's gate
    over ``seeds`` draws: every filter above SEEDS, each failure named in the report."""
    game = generator.GAME
    solve = getattr(generator, "solve", None) or (lambda level: bfs(game, level))
    base = level0(game)
    base_solution, in_time = _timed(SOLVE_S, solve, base)
    report = Report(game, len(base_solution) if base_solution else None)
    if not in_time or base_solution is None or not replay(game, base, base_solution):
        report.failures.append(f"level 0 itself not solved within {SOLVE_S:.0f} s")
        return report
    lo, hi = (f * len(base_solution) for f in LENGTH_BAND)
    sig, layouts = signature(base), set()
    for seed in range(seeds):
        rng = random.Random(seed)
        level, in_time = _timed(GENERATE_S, generator.generate, rng)
        report.draws += 1
        if not in_time:
            report.failures.append(f"seed {seed}: generate ran past {GENERATE_S:.0f} s")
            continue
        report.redraws += getattr(generator, "REDRAWS", 0)
        if level is None:
            continue
        report.kept += 1
        if signature(level)[0] - sig[0] or signature(level)[1] != sig[1]:
            report.failures.append(f"seed {seed}: sprite types or data keys not level 0's")
        if layout(level) == layout(base):
            report.failures.append(f"seed {seed}: level 0 itself")
        layouts.add(layout(level))
        solution, in_time = _timed(SOLVE_S, solve, level)
        if not in_time or solution is None:
            report.failures.append(f"seed {seed}: not solved within {SOLVE_S:.0f} s")
            continue
        if not replay(game, level, solution):
            report.failures.append(f"seed {seed}: its solution does not win on replay")
            continue
        report.solved += 1
        report.lengths.append(len(solution))
        if not lo <= len(solution) <= hi:
            report.failures.append(f"seed {seed}: solution of {len(solution)} outside {lo:.0f}-{hi:.0f}")
    report.distinct = len(layouts)
    if report.kept and report.distinct < DISTINCT_SHARE * report.kept:
        report.failures.append(f"{report.distinct} distinct layouts of {report.kept}")
    rejected, drawn = report.draws - report.kept + report.redraws, report.draws + report.redraws
    if drawn and rejected > REDRAW_SHARE * drawn:
        report.failures.append(f"{rejected} of {drawn} layouts drawn were rejected")
    return report
