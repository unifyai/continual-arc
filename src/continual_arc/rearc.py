"""Seeded, verified access to the Re-ARC generators.

Re-ARC is a set of flat top-level modules (``dsl``, ``utils``, ``generators``,
``verifiers``) that draw from Python's global ``random`` module. This wrapper:

* loads those modules from the pinned checkout without leaving ``dsl``/``utils``
  registered in ``sys.modules`` (they are common names);
* stubs ``matplotlib`` if it is not installed, since ``utils`` imports it at the top;
* seeds the global RNG per call and restores its state afterwards, so instance
  generation is a pure function of ``(task_id, seed, difficulty bounds)``;
* verifies each generated pair with the task's verifier and retries on failure.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from collections.abc import Callable
from pathlib import Path

from continual_arc.data import rearc_dir, require_data
from continual_arc.grids import Grid, InvalidGrid, grid_hash, normalize_grid
from continual_arc.seeds import derive_seed
from continual_arc.tasks import Pair

_REARC_MODULES = ("dsl", "utils", "generators", "verifiers")


class GenerationError(RuntimeError):
    """Raised when a generator cannot produce a verified pair after retries."""



def _stub_matplotlib() -> None:
    if "matplotlib" in sys.modules:
        return
    try:
        import matplotlib  # noqa: F401

        return
    except ImportError:
        pass
    mpl = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    colors = types.ModuleType("matplotlib.colors")
    colors.ListedColormap = object  # type: ignore[attr-defined]
    colors.Normalize = object  # type: ignore[attr-defined]
    mpl.pyplot = pyplot  # type: ignore[attr-defined]
    mpl.colors = colors  # type: ignore[attr-defined]
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = pyplot
    sys.modules["matplotlib.colors"] = colors


def _load_modules(path: Path) -> dict[str, types.ModuleType]:
    """Import Re-ARC's flat modules from ``path`` without polluting ``sys.modules``."""
    _stub_matplotlib()
    saved = {name: sys.modules.get(name) for name in _REARC_MODULES}
    loaded: dict[str, types.ModuleType] = {}
    try:
        for name in _REARC_MODULES:
            spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load Re-ARC module {name} from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module  # so `from dsl import *` inside resolves
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return loaded


class ReArc:
    """Lazy handle on the Re-ARC generators and verifiers."""

    def __init__(self, path: Path | None = None, max_tries: int = 64):
        self._path = path
        self._modules: dict[str, types.ModuleType] | None = None
        self._generators: dict[str, Callable] = {}
        self._verifiers: dict[str, Callable] = {}
        self.max_tries = max_tries

    # -- loading -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._modules is not None:
            return
        path = self._path
        if path is None:
            require_data()
            path = rearc_dir()
        self._modules = _load_modules(Path(path))
        gens = self._modules["generators"]
        vers = self._modules["verifiers"]
        self._generators = {
            n[len("generate_"):]: getattr(gens, n) for n in dir(gens) if n.startswith("generate_")
        }
        self._verifiers = {
            n[len("verify_"):]: getattr(vers, n) for n in dir(vers) if n.startswith("verify_")
        }

    @property
    def task_ids(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._generators)

    def has_task(self, task_id: str) -> bool:
        self._ensure_loaded()
        return task_id in self._generators

    # -- generation --------------------------------------------------------

    def verify(self, task_id: str, input_grid: Grid) -> Grid | None:
        """Run the task's verifier program on ``input_grid``; None if it fails."""
        self._ensure_loaded()
        verifier = self._verifiers[task_id]
        try:
            out = verifier(tuple(tuple(r) for r in input_grid))
            return normalize_grid(out)
        except Exception:  # verifiers can raise on out-of-distribution inputs
            return None

    def generate(
        self,
        task_id: str,
        seed: int,
        diff_lb: float = 0.0,
        diff_ub: float = 1.0,
        exclude_input_hashes: frozenset[str] | set[str] = frozenset(),
    ) -> Pair:
        """Produce one verified input/output pair for ``task_id``.

        The result is a deterministic function of the arguments. Generation is
        retried with derived sub-seeds until the verifier agrees with the generated
        output and the input is not in ``exclude_input_hashes``.
        """
        self._ensure_loaded()
        if task_id not in self._generators:
            raise KeyError(f"no Re-ARC generator for task {task_id!r}")
        generator = self._generators[task_id]
        utils = self._modules["utils"]  # type: ignore[index]
        state = random.getstate()
        try:
            for attempt in range(self.max_tries):
                random.seed(derive_seed("rearc", task_id, seed, attempt))
                utils.rng = []  # generator difficulty trace; reset so it cannot grow
                try:
                    example = generator(diff_lb, diff_ub)
                    inp = normalize_grid(example["input"])
                    out = normalize_grid(example["output"])
                except (InvalidGrid, Exception):
                    continue
                if grid_hash(inp) in exclude_input_hashes:
                    continue
                verified = self.verify(task_id, inp)
                if verified is None or verified != out:
                    continue
                return Pair(input=inp, output=out)
        finally:
            random.setstate(state)
            utils.rng = []
        raise GenerationError(
            f"task {task_id}: no verified pair after {self.max_tries} tries (seed={seed})"
        )
