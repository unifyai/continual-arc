"""Seeded access to the ARC-GEN generators, and the screen that decides which of them the second track serves.

ARC-GEN (github.com/google/arc-gen, Apache-2.0) is a flat top-level module ``common`` and one module per task,
``tasks/task_<id>.py``, with two functions: ``generate(**params)``, which draws every parameter it is not given
through ``common``'s use of Python's ``random`` and returns ``{"input", "output"}``, and ``validate()``, which calls
``generate`` with the parameters of the task's published pairs and returns them as a task, ``{"train", "test"}``.
This wrapper:

* loads ``common`` from the pinned checkout once, and a task's module on first use, without leaving ``common``
  registered in ``sys.modules`` (it is a common name);
* hands ``common`` a ``random.Random`` of its own for every call in place of the ``random`` module, seeded by the
  call, so a pair is a pure function of ``(task_id, seed)`` and nothing a learner does with the global ``random``
  reaches it (``random.Random(s)`` draws exactly what ``random.seed(s)`` would);
* stops a call that has taken :data:`MAX_RANDOM` random numbers: a generator redraws inside one call until a layout
  fits, and on some seeds some never stop, so counting what a call draws ends it at the same point on every machine;
* redraws, from a derived seed, a pair whose input is excluded, so that a demonstration never shares the test's
  input or another demonstration's, and a call that raises, runs past its random numbers or makes an invalid grid.

ARC-GEN ships no solver or verifier, so a drawn pair is not checked against an independent program, as Re-ARC's
are. The second track instead serves only the generators that passed :func:`screen_task`, run on every ARC-AGI-2
training task new to the second edition that ARC-GEN covers, and recorded in :data:`SCREEN_FILE`
(``scripts/screen_arcgen.py`` writes it; :func:`pool` reads it).
"""

from __future__ import annotations

import importlib.util
import json
import random
import signal
import sys
import types
from functools import cache
from pathlib import Path

from continual_arc.data import arcgen_dir, require_data
from continual_arc.grids import InvalidGrid, grid_hash, normalize_grid
from continual_arc.seeds import derive_seed
from continual_arc.tasks import Pair

SCREEN_FILE = Path(__file__).with_name("arcgen_screen.json")

MAX_RANDOM = 10_000_000
"""Random numbers one call of a generator may take, a few seconds of work. A call takes hundreds as a rule; the most
each generator took in one of the screen's draws is in the screen's record (``max_random``)."""


class GenerationError(RuntimeError):
    """Raised when a generator cannot produce an allowed pair within the tries."""


class TooLong(RuntimeError):
    """A call of a generator took more than its random numbers."""


class _Counted(random.Random):
    """A ``random.Random`` that raises :class:`TooLong` once it has given ``budget`` random numbers. It draws exactly
    what ``random.Random`` draws, since every method of it reaches one of the two it counts."""

    def __new__(cls, seed: int, budget: int):
        # random.Random's own __new__ takes the seed alone before Python 3.11, which the package supports
        return super().__new__(cls, seed)

    def __init__(self, seed: int, budget: int):
        super().__init__(seed)
        self.budget = budget
        self.used = 0

    def _spend(self) -> None:
        self.used += 1
        if self.used > self.budget:
            raise TooLong(f"took more than {self.budget} random numbers")

    def random(self) -> float:
        self._spend()
        return super().random()

    def getrandbits(self, k: int) -> int:
        self._spend()
        return super().getrandbits(k)


class ArcGen:
    """Lazy handle on the ARC-GEN generators of a checkout (the pinned one by default)."""

    def __init__(self, path: Path | None = None, max_tries: int = 64, max_random: int = MAX_RANDOM):
        self._path = path
        self._common: types.ModuleType | None = None
        self._modules: dict[str, types.ModuleType] = {}
        self.max_tries = max_tries
        self.max_random = max_random

    # -- loading -----------------------------------------------------------

    @property
    def path(self) -> Path:
        if self._path is None:
            require_data("arc2")
            self._path = arcgen_dir()
        return Path(self._path)

    @property
    def task_ids(self) -> list[str]:
        """Every task ARC-GEN has a generator for."""
        return sorted(p.stem.removeprefix("task_") for p in (self.path / "tasks").glob("task_*.py"))

    def _load(self, name: str, file: Path) -> types.ModuleType:
        """Execute ``file`` as module ``name`` with this checkout's ``common`` importable, then put back whatever
        ``sys.modules`` held under ``common`` before."""
        saved = sys.modules.get("common")
        try:
            if self._common is not None:
                sys.modules["common"] = self._common
            spec = importlib.util.spec_from_file_location(name, file)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load ARC-GEN module {file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if saved is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = saved
        return module

    def module(self, task_id: str) -> types.ModuleType:
        if self._common is None:
            self._common = self._load("common", self.path / "common.py")
        if task_id not in self._modules:
            file = self.path / "tasks" / f"task_{task_id}.py"
            if not file.is_file():
                raise KeyError(f"no ARC-GEN generator for task {task_id!r}")
            self._modules[task_id] = self._load(f"arcgen_task_{task_id}", file)
        return self._modules[task_id]

    # -- generation --------------------------------------------------------

    def _call(self, task_id: str, seed: int, fn: str) -> tuple[object, int]:
        """``fn`` of the task's module with ``common`` drawing from its own random numbers at ``seed``, and how many
        it took."""
        module = self.module(task_id)
        rng = _Counted(seed, self.max_random)
        self._common.random = rng  # type: ignore[union-attr]
        try:
            return getattr(module, fn)(), rng.used
        finally:
            self._common.random = random  # type: ignore[union-attr]

    def draw(self, task_id: str, seed: int) -> tuple[Pair, int]:
        """One call of the task's generator at ``seed``, unchecked but for the grids being valid, and how many random
        numbers it took."""
        example, used = self._call(task_id, seed, "generate")
        return Pair(input=normalize_grid(example["input"]), output=normalize_grid(example["output"])), used

    def generate(self, task_id: str, seed: int,
                 exclude_input_hashes: frozenset[str] | set[str] = frozenset()) -> Pair:
        """One pair for ``task_id``, a pure function of the arguments: the generator's draw at a seed derived from
        ``seed``, drawn again at the next derived seed while its input is in ``exclude_input_hashes``, or while the
        call raises, runs past its random numbers or makes a grid the benchmark refuses. A screened generator did
        none of these in any of the screen's draws, but a run draws far more than the screen, and a draw that failed
        each time the run resumed would stop it for good; a generator that fails on every seed still fails loud."""
        self.module(task_id)   # an unknown task is a KeyError, not a draw to retry
        error: Exception | None = None
        for attempt in range(self.max_tries):
            try:
                pair, _ = self.draw(task_id, derive_seed("arcgen", task_id, seed, attempt))
            except Exception as exc:  # noqa: BLE001 (a rare bad draw of a screened generator; see above)
                error = exc
                continue
            if grid_hash(pair.input) not in exclude_input_hashes:
                return pair
        raise GenerationError(f"task {task_id}: no allowed pair after {self.max_tries} tries (seed={seed})") from error

    def validate(self, task_id: str) -> dict:
        """ARC-GEN's ``validate()``: the task's published pairs as its generator draws them from their parameters."""
        return self._call(task_id, 0, "validate")[0]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------

N_DRAWS = 1000
"""Draws a generator is screened on: more than the held-out season's 300, since a run draws thousands of pairs of
each of its tasks, and a generator that repeats its inputs or fails on a rare seed shows it only at that scale."""
CPU_SECONDS_PER_DRAW = 10.0
"""A backstop for a call that loops without drawing: CPU seconds, so that how busy the machine is does not move it."""
MIN_INPUT_SIDE = 3
MIN_DISTINCT_INPUTS = 0.9
MIN_DISTINCT_OUTPUTS = 0.1
MAX_IDENTITY = 0.05
MAX_FAILED = 3
"""Calls that ran past their random numbers or CPU seconds after which a generator's screen stops: it has failed,
and each such call costs seconds."""
N_REPEAT = 5
"""Draws made again at the end of the screen, which must come out as they did at the start."""


def _d8(g: list[list[int]]) -> list[list[list[int]]]:
    """The eight symmetries of the square applied to ``g``, identity first."""
    out = []
    for _ in range(4):
        out.append(g)
        out.append([list(reversed(row)) for row in g])
        g = [list(row) for row in zip(*g[::-1], strict=True)]
    return out


class _Timeout(BaseException):
    """Raised by the CPU timer; a BaseException so that nothing a generator catches can swallow it."""


def _alarm(signum, frame):
    raise _Timeout


def _pairs(task: dict) -> list[tuple[list, list]]:
    return [(normalize_grid(p["input"]), normalize_grid(p["output"])) for p in task["train"] + task["test"]]


def _traits(i: list[list[int]], o: list[list[int]]) -> dict[str, object]:
    """Four things about a pair that a task's rule usually fixes."""
    return {
        "same_shape": (len(i), len(i[0])) == (len(o), len(o[0])),
        "output_shape": (len(o), len(o[0])),
        "scale": (len(o) / len(i), len(o[0]) / len(i[0])),
        "output_colors_in_input": {v for row in o for v in row} <= {v for row in i for v in row},
    }


def _shared(pairs: list[tuple[list, list]]) -> dict[str, object]:
    """The traits (:func:`_traits`) every one of ``pairs`` has alike, a yes-or-no trait only when it holds of every
    pair: that no published output has its input's shape, or that each adds a color, can be chance (an output whose
    size counts the input's layers can come out the input's size), where an output that always keeps its input's
    shape or colors is the rule."""
    traits = [_traits(i, o) for i, o in pairs]
    return {k: v for k, v in traits[0].items() if v is not False and all(t[k] == v for t in traits)}


def screen_task(arcgen: ArcGen, task_id: str, published: dict, n: int = N_DRAWS) -> dict:
    """Every reading of the screen for ``task_id`` and whether it passes. Run it in a process's main thread: the
    CPU backstop on a call is a ``SIGPROF``.

    Fidelity, in two parts. ARC-GEN's ``validate()`` reproduces every published pair of the task, ``train`` and
    ``test``, exactly (``published`` is the task's JSON at the pinned ARC-AGI-2 commit), so the code that draws a fresh
    pair is code that draws the task's own. And every draw keeps what all the published pairs have alike of four
    traits a rule usually fixes (:func:`_traits`): whether the output has the input's shape, the output's shape, its
    size over the input's, and whether its colors are all the input's. A draw that breaks one has left the published
    rule: an output color the input lacks, for instance, is one the input hid, so the answer cannot be read from it.
    Then the held-out season's mechanical filters, over ``n`` draws at seeds of the screen's own: every call
    finishes, within :data:`MAX_RANDOM` random numbers and :data:`CPU_SECONDS_PER_DRAW`,
    and raises nothing; every grid is valid and every input at least :data:`MIN_INPUT_SIDE` by
    :data:`MIN_INPUT_SIDE`; at least 90 percent of inputs and 10 percent of outputs are distinct; the output differs
    from the input in at least 95 percent of draws and is not always the input under one fixed rotation or
    reflection; and :data:`N_REPEAT` draws made again at the end come out as they did, so that a pair depends on its
    seed alone."""
    signal.signal(signal.SIGPROF, _alarm)
    reasons: list[str] = []
    try:
        faithful = _pairs(arcgen.validate(task_id)) == _pairs(published)
    except Exception as exc:  # noqa: BLE001 (the record names the generator's own bug)
        faithful = False
        reasons.append(f"validate raised {type(exc).__name__}: {exc}"[:160])
    if not faithful and not reasons:
        reasons.append("validate does not reproduce the published pairs")
    pairs: list[tuple[list, list]] = []
    used: list[int] = []
    failures: dict[str, int] = {}
    too_long = f"call ran past {arcgen.max_random:,} random numbers"
    too_slow = f"call ran past {CPU_SECONDS_PER_DRAW:g} CPU seconds"

    def fail(key: str) -> None:
        failures[key] = failures.get(key, 0) + 1

    for k in range(n):
        if failures.get(too_long, 0) + failures.get(too_slow, 0) >= MAX_FAILED:
            fail(f"stopped after {k} draws")
            break
        signal.setitimer(signal.ITIMER_PROF, CPU_SECONDS_PER_DRAW)
        try:
            pair, count = arcgen.draw(task_id, derive_seed("arcgen-screen", task_id, k))
        except _Timeout:
            fail(too_slow)
            continue
        except TooLong:
            fail(too_long)
            continue
        except InvalidGrid as exc:
            fail(f"invalid grid: {exc}"[:160])
            continue
        except Exception as exc:  # noqa: BLE001 (the record names the generator's own bug)
            fail(f"generate raised {type(exc).__name__}: {exc}"[:160])
            continue
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
        pairs.append((pair.input, pair.output))
        used.append(count)
    reasons.extend(f"{v}x {k}" for k, v in sorted(failures.items(), key=lambda kv: -kv[1])[:3])
    shared = _shared(_pairs(published))
    broken = {k: 0 for k in shared}
    for i, o in pairs:
        for k, v in _traits(i, o).items():
            if k in shared and v != shared[k]:
                broken[k] += 1
    reasons.extend(f"{v} draws break {k}, which every published pair keeps" for k, v in broken.items() if v)
    small = sum(1 for i, _ in pairs if len(i) < MIN_INPUT_SIDE or len(i[0]) < MIN_INPUT_SIDE)
    if small:
        reasons.append(f"{small} inputs smaller than {MIN_INPUT_SIDE}x{MIN_INPUT_SIDE}")
    distinct_in = len({grid_hash(i) for i, _ in pairs}) / max(1, len(pairs))
    distinct_out = len({grid_hash(o) for _, o in pairs}) / max(1, len(pairs))
    identity = sum(1 for i, o in pairs if i == o) / max(1, len(pairs))
    if distinct_in < MIN_DISTINCT_INPUTS:
        reasons.append(f"only {distinct_in:.0%} of inputs distinct")
    if distinct_out < MIN_DISTINCT_OUTPUTS:
        reasons.append(f"only {distinct_out:.0%} of outputs distinct")
    if identity > MAX_IDENTITY:
        reasons.append(f"output equals input in {identity:.0%} of draws")
    for k in range(1, 8):
        if pairs and all(_d8(i)[k] == o for i, o in pairs):
            reasons.append(f"output is always symmetry {k} of the input")
    if not failures:
        again = [arcgen.draw(task_id, derive_seed("arcgen-screen", task_id, k))[0] for k in range(min(N_REPEAT, n))]
        if [(p.input, p.output) for p in again] != pairs[:len(again)]:
            reasons.append("a draw made again differs: the generator depends on more than its seed")
    cells = [len(i) * len(i[0]) for i, _ in pairs]
    return {
        "ok": not reasons,
        "reasons": reasons,
        "faithful": faithful,
        "shared": sorted(shared),
        "draws": len(pairs),
        "distinct_inputs": round(distinct_in, 3),
        "distinct_outputs": round(distinct_out, 3),
        "identity": round(identity, 3),
        "mean_input_cells": round(sum(cells) / max(1, len(cells)), 1),
        "max_input_side": max((max(len(i), len(i[0])) for i, _ in pairs), default=0),
        "mean_random": round(sum(used) / max(1, len(used))),
        "max_random": max(used, default=0),
    }


@cache
def screen_record() -> dict:
    """The screen's record (:data:`SCREEN_FILE`): what it ran on and every task's readings."""
    return json.loads(SCREEN_FILE.read_text())


def pool() -> list[str]:
    """The second track's stream pool: the ARC-AGI-2 training tasks new to the second edition whose ARC-GEN
    generator passed the screen."""
    return list(screen_record()["pool"])
