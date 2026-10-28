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

:func:`check` is the gate a generator passes to enter the track (FILTERS).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from continual_arc.data import data_root

_SEQUENCE = itertools.count()

# A generator's gate (check): over SEEDS draws, a level within GENERATE_S, redraws under REDRAW_SHARE, every kept level
# solved within SOLVE_S and won on replay, its solution LENGTH_BAND times level 0's, DISTINCT_SHARE of layouts distinct
# and none level 0's, and only level 0's sprite types and data keys.
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
    """The game built on ``level`` alone, after its reset: (game, frame data)."""
    from arcengine import ActionInput, GameAction

    module = load_module(game_id)
    module.levels = [level]
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


def bfs(game_id: str, level: Any, max_states: int = 200_000, max_seconds: float = SOLVE_S) -> list[Any] | None:
    """A shortest winning sequence of the engine's own valid actions, by breadth-first search over deep copies of the
    game with states deduplicated (:func:`state_key`); None when the budget runs out first. Solves games whose states
    are few, such as small mazes and click puzzles; a game whose valid actions or states are too many needs a
    ``solve`` of its own."""
    from arcengine import GameState

    game, frame = new_game(game_id, level)
    seen, queue, started, expanded = {state_key(game, frame)}, deque([(game, [])]), time.monotonic(), 0
    while queue and expanded < max_states and time.monotonic() - started < max_seconds:
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


def layout(level: Any) -> tuple:
    """A level's layout, for telling levels apart: every sprite's type and place, and its data."""
    return (tuple(sorted((s.name, s.x, s.y) for s in level.get_sprites())), json.dumps(level._data, sort_keys=True,
                                                                                         default=str))


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
                f"kept of {self.draws}, {self.solved} solved, lengths {band}, {self.distinct} distinct"
                + (f"; {'; '.join(self.failures[:4])}" if self.failures else ""))


def check(generator: Any, seeds: int = SEEDS) -> Report:
    """Whether ``generator`` (a module with GAME, generate and optionally solve) passes the track's gate over
    ``seeds`` draws: every filter of FILTERS, each failure named in the report."""
    game = generator.GAME
    solve = getattr(generator, "solve", None) or (lambda level: bfs(game, level))
    base = level0(game)
    base_solution = solve(base)
    report = Report(game, len(base_solution) if base_solution else None)
    if base_solution is None or not replay(game, base, base_solution):
        report.failures.append("level 0 itself not solved")
        return report
    lo, hi = (f * len(base_solution) for f in LENGTH_BAND)
    sig, layouts = signature(base), set()
    for seed in range(seeds):
        rng = random.Random(seed)
        started = time.monotonic()
        level = generator.generate(rng)
        report.draws += 1
        report.redraws += getattr(generator, "REDRAWS", 0)
        if time.monotonic() - started > GENERATE_S:
            report.failures.append(f"seed {seed}: generate took {time.monotonic() - started:.1f} s")
            continue
        if level is None:
            continue
        report.kept += 1
        if signature(level)[0] - sig[0] or signature(level)[1] != sig[1]:
            report.failures.append(f"seed {seed}: sprite types or data keys not level 0's")
        if layout(level) == layout(base):
            report.failures.append(f"seed {seed}: level 0 itself")
        layouts.add(layout(level))
        started = time.monotonic()
        solution = solve(level)
        if solution is None or time.monotonic() - started > SOLVE_S:
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
    if report.draws and (report.draws - report.kept) > REDRAW_SHARE * report.draws:
        report.failures.append(f"{report.draws - report.kept} draws gave no level")
    return report
