"""Task sources: where test instances and demonstration pairs come from.

Two kinds of source exist:

* :class:`ReArcSource` — generative. Every instance and every demo pair is freshly
  sampled from the task's Re-ARC generator, seeded by the slot, so no two slots
  share a test input and demos differ across requests.
* :class:`FixedArcSource` — the original ARC-AGI-1 JSON tasks. Used for the
  evaluation split (which has no generators) in the held-out epilogue. Test
  instances cycle through the task's ``test`` pairs; demos are the task's own
  ``train`` pairs, always the same, and nothing after them.

A :class:`TaskRegistry` maps each task id to the source it should be served from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from continual_arc.grids import Grid, grid_hash, normalize_grid
from continual_arc.seeds import derive_seed

SOURCE_REARC = "rearc"
SOURCE_ARC_TRAIN = "arc_train"
SOURCE_ARC_EVAL = "arc_eval"
SOURCE_NAMES = (SOURCE_REARC, SOURCE_ARC_TRAIN, SOURCE_ARC_EVAL)


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid
    note: str | None = field(default=None, compare=False)
    """Shown to a learner with the pair when a source sets it."""

    def to_json(self) -> dict:
        d = {"input": self.input, "output": self.output}
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_json(cls, d: dict) -> Pair:
        return cls(input=normalize_grid(d["input"]), output=normalize_grid(d["output"]))


class TaskSource(Protocol):
    name: str

    def task_ids(self) -> list[str]: ...

    def test_instance(self, task_id: str, instance_seed: int) -> Pair: ...

    def demos(
        self,
        task_id: str,
        instance_seed: int,
        request_index: int,
        n: int,
        exclude: Pair,
    ) -> list[Pair]: ...

    # Optional: ``demo_supply(task_id, instance_seed, exclude) -> int | None``, the number of demonstrations
    # the source can serve for the instance, or None when it never runs out (a generator).


def _supply(in_frame: list[Pair], exclude: Pair) -> int:
    """How many of ``in_frame`` :func:`_drip` can serve: every pair whose input is not the test's."""
    ex = grid_hash(exclude.input)
    return sum(1 for p in in_frame if grid_hash(p.input) != ex)


def _drip(own: list[Pair], exclude: Pair, start: int, n: int) -> list[Pair]:
    """Demonstrations ``start`` to ``start + n`` of an instance's sequence: the task's own pairs, in order,
    and nothing after them. A pair whose input is the test's is never served."""
    ex = grid_hash(exclude.input)
    return [p for p in own if grid_hash(p.input) != ex][start:start + n]


class ReArcSource:
    """Generative source backed by the Re-ARC generators."""

    name = SOURCE_REARC

    def __init__(self, rearc=None, diff_lb: float = 0.0, diff_ub: float = 1.0):
        if rearc is None:
            from continual_arc.rearc import ReArc

            rearc = ReArc()
        self._rearc = rearc
        self.diff_lb = diff_lb
        self.diff_ub = diff_ub

    def task_ids(self) -> list[str]:
        return self._rearc.task_ids

    def test_instance(self, task_id: str, instance_seed: int) -> Pair:
        return self._rearc.generate(
            task_id, derive_seed("test", instance_seed), self.diff_lb, self.diff_ub
        )

    def demos(
        self, task_id: str, instance_seed: int, request_index: int, n: int, exclude: Pair
    ) -> list[Pair]:
        seen = {grid_hash(exclude.input)}
        pairs: list[Pair] = []
        for k in range(n):
            pair = self._rearc.generate(
                task_id,
                derive_seed("demo", instance_seed, request_index, k),
                self.diff_lb,
                self.diff_ub,
                exclude_input_hashes=frozenset(seen),
            )
            seen.add(grid_hash(pair.input))
            pairs.append(pair)
        return pairs


class FixedArcSource:
    """Source backed by the original ARC JSON tasks, which have no generators: an instance's demonstrations
    are the task's own pairs and nothing after them, so its demo cap is the number of pairs it has left
    (every task without generators since 27 Sep 2026). ``dataset`` loads another edition's tasks under a
    ``name`` of the caller's."""

    def __init__(self, split: str, tasks: dict[str, dict] | None = None, dataset: str = "arc1",
                 name: str | None = None):
        if split not in ("training", "evaluation"):
            raise ValueError(split)
        if name is None and dataset != "arc1":
            raise ValueError(f"a source over {dataset!r} needs a name of its own")
        self.split = split
        self.dataset = dataset
        self.name = name or (SOURCE_ARC_TRAIN if split == "training" else SOURCE_ARC_EVAL)
        self._tasks = tasks

    def _load(self) -> dict[str, dict]:
        if self._tasks is None:
            from continual_arc.data import load_arc_tasks

            self._tasks = load_arc_tasks(self.split, dataset=self.dataset)
        return self._tasks

    def task_ids(self) -> list[str]:
        return sorted(self._load())

    def test_instance(self, task_id: str, instance_seed: int) -> Pair:
        tests = self._load()[task_id]["test"]
        return Pair.from_json(tests[instance_seed % len(tests)])

    def _own(self, task_id: str) -> list[Pair]:
        return [Pair.from_json(p) for p in self._load()[task_id]["train"]]

    def demos(
        self, task_id: str, instance_seed: int, request_index: int, n: int, exclude: Pair
    ) -> list[Pair]:
        # One request buys ``n`` more of the task's own demonstrations (26 Sep: eight requests of one pair
        # each, and a task without generators has fewer pairs than that).
        return _drip(self._own(task_id), exclude, request_index * n, n)

    def demo_supply(self, task_id: str, instance_seed: int, exclude: Pair) -> int:
        return _supply(self._own(task_id), exclude)


class TaskRegistry:
    """Maps task ids to sources and serves instances through a uniform interface."""

    def __init__(self, sources: dict[str, TaskSource] | None = None):
        self._sources: dict[str, TaskSource] = dict(sources or {})
        self._assignment: dict[str, str] = {}

    @classmethod
    def default(cls, diff_lb: float = 0.0, diff_ub: float = 1.0) -> TaskRegistry:
        """Every named source. Each loads its data lazily, so a registry costs
        nothing for the datasets a schedule does not touch."""
        return cls(
            {
                SOURCE_REARC: ReArcSource(diff_lb=diff_lb, diff_ub=diff_ub),
                SOURCE_ARC_TRAIN: FixedArcSource("training"),
                SOURCE_ARC_EVAL: FixedArcSource("evaluation"),
            }
        )

    def add_source(self, source: TaskSource) -> None:
        self._sources[source.name] = source

    def assign(self, task_id: str, source_name: str) -> None:
        if source_name not in self._sources:
            raise KeyError(f"unknown source {source_name!r}; have {sorted(self._sources)}")
        self._assignment[task_id] = source_name

    def assign_many(self, mapping: dict[str, str]) -> None:
        for task_id, source_name in mapping.items():
            self.assign(task_id, source_name)

    def source_for(self, task_id: str) -> TaskSource:
        try:
            return self._sources[self._assignment[task_id]]
        except KeyError:
            raise KeyError(f"task {task_id!r} is not assigned to a source") from None

    def test_instance(self, task_id: str, instance_seed: int) -> Pair:
        return self.source_for(task_id).test_instance(task_id, instance_seed)

    def demos(self, task_id: str, instance_seed: int, request_index: int, n: int, exclude: Pair) -> list[Pair]:
        return self.source_for(task_id).demos(task_id, instance_seed, request_index, n, exclude)

    def demo_supply(self, task_id: str, instance_seed: int, exclude: Pair) -> int | None:
        """How many demonstrations the task's source can serve for the instance; None when unlimited."""
        supply = getattr(self.source_for(task_id), "demo_supply", None)
        if supply is None:
            return None
        return supply(task_id, instance_seed, exclude)
