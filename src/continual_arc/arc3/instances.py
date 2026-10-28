"""Instances of the ARC-AGI-3 track: fresh levels of a game's level-0 rules, their solutions, and the solver's
demonstrations, with every solution kept on disk.

An instance's seed draws its level: ``generate(random.Random(derive_seed("arc3-draw", seed, k)))`` for k = 0, 1, ...,
the first draw that gives a level the game's solver solves within :data:`continual_arc.arc3.levels.LENGTH_BAND` of
level 0's own solution length (the gate's band, so that every instance of a game is about as hard). The game's
``solve`` gives that length, and the instance's baseline is level 0's human baseline scaled by it: ARC Prize's action
count for level 0 times the instance's solution length over level 0's. A demonstration is the solver's play of another
fresh level of the game, drawn from a seed of its own (the instance's seed and the request's index), any layout equal
to the instance's own skipped, and shown as its first frame and each action with the cells it changed
(:class:`Demonstration`).

Generating a level is deterministic and fast (the gate holds it under 2 s), so a level is drawn again from its seed
whenever it is needed; solving one can take up to a minute, so only solutions are kept, one file a draw, under
``<data root>/arc-agi-3/solutions/<game>/<version>/<generator digest>/<draw seed>.json``: keyed by the game, its pinned
version, a digest of the generator module's source and the draw's seed, with a digest of the drawn layout to catch a
generator that is not deterministic. Every solve is bounded by states expanded, never by time, so a level is solved,
or not, the same way on every machine, and a cached solution is the one any machine would find.
:func:`precompute` draws a configuration's every instance and demonstration ahead of a run, so that a run never waits
on a solver.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from continual_arc.arc3 import levels
from continual_arc.data import data_root
from continual_arc.seeds import derive_seed

Frame = list[list[int]]
"""One 64x64 frame of colors 0 to 15, rows top to bottom."""

COMPLEX_ACTION = 6
"""ACTION6 is a click, with ``x`` (column) and ``y`` (row) in 0..63; every other action is simple."""
MAX_DRAWS = 50
"""Draws of one seed before it is given up on: the gate holds a generator's rejected draws under one in five."""


@dataclass(frozen=True)
class Act:
    """One engine action: ``action`` 1 to 7, with ``x`` and ``y`` for ACTION6."""

    action: int
    x: int | None = None
    y: int | None = None

    def engine(self) -> Any:
        """The action as the engine takes it."""
        from arcengine import ActionInput, GameAction

        data = {"x": int(self.x), "y": int(self.y)} if self.action == COMPLEX_ACTION else {}
        return ActionInput(id=GameAction.from_id(self.action), data=data)

    @classmethod
    def of(cls, action_input: Any) -> Act:
        """The engine's ``ActionInput`` as an action of the benchmark."""
        data = action_input.data or {}
        return cls(int(action_input.id.value), data.get("x"), data.get("y"))

    def to_json(self) -> list[int]:
        return [self.action] if self.x is None else [self.action, self.x, self.y]

    @classmethod
    def from_json(cls, d: list[int]) -> Act:
        return cls(*d)


def changes(before: Frame, after: Frame) -> tuple[tuple[int, int, int], ...]:
    """Every cell ``after`` changed, as ``(x, y, color)``, row by row."""
    return tuple((x, y, c) for y, (old, new) in enumerate(zip(before, after, strict=True))
                 for x, (o, c) in enumerate(zip(old, new, strict=True)) if o != c)


@dataclass(frozen=True)
class DemoStep:
    action: Act
    changes: tuple[tuple[int, int, int], ...]
    """The cells the action changed, ``(x, y, color)``, from the frame before it to the last frame it rendered."""


@dataclass(frozen=True)
class Demonstration:
    """The solver's play of a fresh level of the task, from its reset to the win: the first frame, then each action
    with the cells it changed."""

    start: Frame
    steps: tuple[DemoStep, ...]

    def frames(self) -> list[Frame]:
        """Every frame of the play: the start, then the frame after each action."""
        out = [self.start]
        for step in self.steps:
            frame = [list(row) for row in out[-1]]
            for x, y, c in step.changes:
                frame[y][x] = c
            out.append(frame)
        return out

    @property
    def final(self) -> Frame:
        return self.frames()[-1]

    def to_json(self) -> dict:
        return {"start": self.start, "steps": [{"action": s.action.to_json(), "changes": [list(c) for c in s.changes]}
                                               for s in self.steps]}

    @classmethod
    def from_json(cls, d: dict) -> Demonstration:
        return cls(d["start"], tuple(DemoStep(Act.from_json(s["action"]), tuple(tuple(c) for c in s["changes"]))
                                     for s in d["steps"]))


def play(game_id: str, level: Any, solution: Sequence[Act]) -> Demonstration:
    """``solution`` played on ``level`` in the real engine, as a :class:`Demonstration`; it must win the level."""
    game, frame = levels.new_game(game_id, level)
    start = shown = frame.frame[-1]
    steps = []
    for act in solution:
        frame = game.perform_action(act.engine())
        after = frame.frame[-1] if frame.frame else shown
        steps.append(DemoStep(act, changes(shown, after)))
        shown = after
    if not levels.won(frame):
        raise RuntimeError(f"{game_id}: a solution did not win its level when played for a demonstration")
    return Demonstration(start, tuple(steps))


@dataclass(frozen=True)
class Drawn:
    """An instance's level, drawn from ``seed`` at its ``draw``-th try, and its solver's solution."""

    game: str
    seed: int
    draw: int
    level: Any
    solution: tuple[Act, ...]
    layout: tuple
    baseline: float
    """Level 0's human action count times this level's solution length over level 0's."""


def _digest(value: Any) -> str:
    return hashlib.blake2b(repr(value).encode(), digest_size=8).hexdigest()


class LevelSource:
    """Where an ARC-AGI-3 instance's level and demonstrations come from (the module docstring). ``generators`` maps a
    game to the import path of its generator module, ``continual_arc.arc3.generators.<game>`` for any game it does not
    name; ``cache`` is the folder solutions are kept in."""

    def __init__(self, generators: dict[str, str] | None = None, cache: str | Path | None = None):
        self.generators = dict(generators or {})
        self.cache = Path(cache) if cache is not None else data_root() / "arc-agi-3" / "solutions"
        self._level0: dict[str, int] = {}

    def generator(self, game_id: str) -> ModuleType:
        return importlib.import_module(self.generators.get(game_id, f"continual_arc.arc3.generators.{game_id}"))

    def folder(self, game_id: str) -> Path:
        """The cache's folder for ``game_id``'s generator as it is now: its version and its source's digest."""
        source = Path(self.generator(game_id).__file__).read_bytes()
        return (self.cache / game_id / levels.game_file(game_id).parent.name
                / hashlib.blake2b(source, digest_size=8).hexdigest())

    def solve(self, game_id: str, key: str, level: Any) -> tuple[Act, ...] | None:
        """The solution of ``level``, drawn under ``key``, from the cache or else from the game's ``solve``
        (``levels.bfs`` when it has none), kept only if it wins the level on replay; None when there is none."""
        path = self.folder(game_id) / f"{key}.json"
        layout = _digest(levels.layout(level))
        if path.exists():
            kept = json.loads(path.read_text())
            if kept["layout"] != layout:
                raise RuntimeError(f"{game_id}: draw {key} gave another layout than when it was solved; its generator "
                                   "is not deterministic")
            return None if kept["solution"] is None else tuple(Act.from_json(a) for a in kept["solution"])
        solve = getattr(self.generator(game_id), "solve", None) or (lambda lv: levels.bfs(game_id, lv))
        found = solve(level)
        solution = (tuple(Act.of(a) for a in found)
                    if found is not None and levels.replay(game_id, level, found) else None)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"layout": layout, "solution": None if solution is None
                                   else [a.to_json() for a in solution]}))
        tmp.replace(path)
        return solution

    def level0_length(self, game_id: str) -> int:
        """The length of level 0's own solution by the game's solver, which an instance's length is held to."""
        if game_id not in self._level0:
            solution = self.solve(game_id, "level0", levels.level0(game_id))
            if solution is None:
                raise RuntimeError(f"{game_id}: its solver does not solve level 0")
            self._level0[game_id] = len(solution)
        return self._level0[game_id]

    def draw(self, game_id: str, seed: int, exclude: Iterable[tuple] = ()) -> Drawn:
        """The level ``seed`` draws (the module docstring), skipping any whose layout is in ``exclude``."""
        length = self.level0_length(game_id)
        lo, hi = (f * length for f in levels.LENGTH_BAND)
        excluded = set(exclude)
        generator = self.generator(game_id)
        for k in range(MAX_DRAWS):
            key = derive_seed("arc3-draw", seed, k)
            level = generator.generate(random.Random(key))
            if level is None:
                continue
            layout = levels.layout(level)
            if layout in excluded:
                continue
            solution = self.solve(game_id, str(key), level)
            if solution is None or not lo <= len(solution) <= hi:
                continue
            return Drawn(game_id, seed, k, level, solution, layout,
                         levels.baseline(game_id) * len(solution) / length)
        raise RuntimeError(f"{game_id}: no solvable level in {MAX_DRAWS} draws of seed {seed}")

    def instance(self, game_id: str, instance_seed: int) -> Drawn:
        """The level a slot's instance seed serves."""
        return self.draw(game_id, derive_seed("arc3-instance", instance_seed))

    def demonstration(self, game_id: str, instance_seed: int, request_index: int, instance_layout: tuple) -> Demonstration:
        """The ``request_index``-th demonstration of the instance: the solver's play of a level of its own seed, never
        the instance's own layout."""
        drawn = self.draw(game_id, derive_seed("arc3-demo", instance_seed, request_index), exclude=(instance_layout,))
        return play(game_id, drawn.level, drawn.solution)


def _prepare(job: tuple[str, int, int, dict[str, str], str]) -> str:
    """One instance of a precompute, in a worker: its level and its first ``demos`` demonstrations, solved and cached."""
    game_id, instance_seed, demos, generators, cache = job
    source = LevelSource(generators, cache)
    drawn = source.instance(game_id, instance_seed)
    for k in range(demos):
        source.draw(game_id, derive_seed("arc3-demo", instance_seed, k), exclude=(drawn.layout,))
    return game_id


def _level0(job: tuple[str, dict[str, str], str]) -> int:
    game_id, generators, cache = job
    return LevelSource(generators, cache).level0_length(game_id)


def precompute(source: LevelSource, instances: Iterable[tuple[str, int]], demos: int, workers: int = 4,
               progress: Any = None) -> int:
    """Solve and cache in ``source``'s cache, ahead of a run, every one of ``instances`` (game, instance seed) and its
    first ``demos`` demonstrations, ``workers`` processes at a time: each game's level 0 first, so that no two workers
    solve it at once. Returns how many instances were prepared; ``progress(done, total, game)`` is called after
    each."""
    cache = str(source.cache)
    jobs = sorted({(game, seed) for game, seed in instances})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_level0, [(game, source.generators, cache) for game in sorted({g for g, _ in jobs})]))
        work = [(game, seed, demos, source.generators, cache) for game, seed in jobs]
        for done, game in enumerate(pool.map(_prepare, work), 1):
            if progress is not None:
                progress(done, len(jobs), game)
    return len(jobs)
