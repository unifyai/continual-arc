"""Task sources: where test instances and demonstration pairs come from.

Two kinds of source exist:

* :class:`ReArcSource` — generative. Every instance and every demo pair is freshly
  sampled from the task's Re-ARC generator, seeded by the slot, so no two slots
  share a test input and demos differ across requests.
* :class:`FixedArcSource` — the original ARC-AGI-1 (or ARC-AGI-2) JSON tasks.
  Used for the evaluation split (which has no generators) in the held-out
  epilogue. Test instances cycle through the task's ``test`` pairs; demos are the
  task's ``train`` pairs, always the same.
* :class:`AugmentedArcSource` — JSON tasks served under seeded augmentations
  (:mod:`continual_arc.augment`), for task sets with no generators that must
  still recur with inputs a learner cannot cache. This is how the ARC-AGI-2
  evaluation tasks enter the stream (the track draws its stream and its epilogue
  from the evaluation tasks new to the second edition; see ``data.arc2_new_task_ids``).
  On the track each task is drawn only under the symmetries and recolourings its
  rule survives (per-task labels, :data:`INVARIANCES_FILE`), and an instance serves
  only the task's own pairs (Dan, 26 Sep 2026).

A :class:`TaskRegistry` maps each task id to the source it should be served from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from continual_arc.grids import Grid, grid_hash, normalize_grid
from continual_arc.seeds import derive_seed

SOURCE_REARC = "rearc"
SOURCE_ARC_TRAIN = "arc_train"
SOURCE_ARC_EVAL = "arc_eval"
SOURCE_ARC2_TRAIN_AUG = "arc2_train_aug"
SOURCE_ARC2_EVAL_AUG = "arc2_eval_aug"
SOURCE_ARC2_EVAL = "arc2_eval"
SOURCE_NAMES = (SOURCE_REARC, SOURCE_ARC_TRAIN, SOURCE_ARC_EVAL, SOURCE_ARC2_TRAIN_AUG, SOURCE_ARC2_EVAL_AUG, SOURCE_ARC2_EVAL)


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid
    note: str | None = field(default=None, compare=False)
    """Set on a padded demonstration (:data:`PADDED_NOTE`), which a learner is shown with the pair."""

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
    # the source can serve for the instance, or None when it never runs out (a generator, or padding).
    # A source whose methods take ``exposure`` (the task's visit index in the schedule) is given it by the
    # registry; the ARC-AGI-2 source serves each visit of a task in a frame of its own.


INVARIANCES_FILE = "arc2_invariances.json"
"""Per-task labels of the ARC-AGI-2 pool (the 114 evaluation tasks new to the edition, from which every
configuration of the track draws its stream and its epilogue): the symmetries of the square under which
the task's rule is unchanged and the colours with a fixed role in it (Dan, 26 Sep 2026)."""


def invariances_path(name: str = INVARIANCES_FILE) -> Path:
    return Path(__file__).with_name(name)


def load_invariances(name: str = INVARIANCES_FILE) -> dict[str, dict]:
    """``{task_id: {"symmetries": [names], "fixed_colours": [colours]}, ...}`` from the labels file."""
    return json.loads(invariances_path(name).read_text())["tasks"]


def invariances_digest(name: str = INVARIANCES_FILE) -> str:
    """The first twelve hex digits of the labels file's SHA-256, which a schedule records."""
    return hashlib.sha256(invariances_path(name).read_bytes()).hexdigest()[:12]


MAX_FRAMES = 64
"""The most distinct frames a task keeps (its visits cycle through them)."""

MIN_STREAM_FRAMES = 8
"""The ARC-AGI-2 stream draws only tasks with at least this many distinct frames: the visits a task gets in
the small configuration, so that no visit shows a grid an earlier visit showed (Dan, 26 Sep 2026)."""


def _supply(in_frame: list[Pair], exclude: Pair) -> int:
    """How many of ``in_frame`` :func:`_drip` can serve: every pair whose input is not the test's."""
    ex = grid_hash(exclude.input)
    return sum(1 for p in in_frame if grid_hash(p.input) != ex)


PADDED_NOTE = ("a transformed copy: one of this task's demonstration pairs under a different rotation or reflection "
               "and recolouring, so the rule may appear rotated, reflected or recoloured relative to the test")
"""What a padded demonstration says about itself. Padding (a source built without ``own_pairs_only``) served
a task's own pairs under other augmentations once its pairs ran out; no track of the benchmark pads since
27 Sep 2026, when every source without generators went to its own pairs only (Dan)."""


def _drip(in_frame: list[Pair], pad, exclude: Pair, start: int, n: int) -> list[Pair]:
    """Demonstrations ``start`` to ``start + n`` of an instance's sequence: the pairs in the test's frame
    first, then padded pairs from ``pad(j)`` (``None`` when the task has nothing to pad with). A pair
    whose input is the test's is never served."""
    ex = grid_hash(exclude.input)
    seq = [p for p in in_frame if grid_hash(p.input) != ex]
    j = 0
    while len(seq) < start + n and j < 20 * (start + n) + 20:
        p = pad(j)
        j += 1
        if p is None:
            break
        if grid_hash(p.input) != ex:
            seq.append(p)
    return seq[start:start + n]


def _pad_augmentation(seed_parts: tuple, avoid, *, symmetries: bool = True, colours: bool = True,
                      keep_background: bool = True):
    """A seeded augmentation other than ``avoid`` (the test's own, or the identity for an unaugmented task)."""
    from continual_arc.augment import Augmentation

    for r in range(64):
        aug = Augmentation.sample(derive_seed("aug-pad", *seed_parts, r), symmetries=symmetries, colours=colours,
                                  keep_background=keep_background)
        if aug != avoid:
            return aug
    return aug


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


def _fixed_source_name(split: str, dataset: str) -> str:
    if dataset == "arc1":
        return SOURCE_ARC_TRAIN if split == "training" else SOURCE_ARC_EVAL
    if dataset == "arc2":
        if split == "training":
            raise ValueError("the ARC-AGI-2 training split is served augmented; use AugmentedArcSource")
        return SOURCE_ARC2_EVAL
    raise ValueError(f"unknown dataset {dataset!r}")


class FixedArcSource:
    """Source backed by the original ARC JSON tasks (no generators, no augmentation).

    ``own_pairs_only`` serves a task's own demonstrations and nothing after them (the ARC-AGI-2 epilogue,
    26 Sep 2026); otherwise its own pairs are followed by marked copies under other augmentations."""

    def __init__(self, split: str, tasks: dict[str, dict] | None = None, dataset: str = "arc1",
                 name: str | None = None, *, own_pairs_only: bool = False):
        if split not in ("training", "evaluation"):
            raise ValueError(split)
        self.split = split
        self.dataset = dataset
        self.name = name or _fixed_source_name(split, dataset)
        self.own_pairs_only = own_pairs_only
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

    def demos(
        self, task_id: str, instance_seed: int, request_index: int, n: int, exclude: Pair
    ) -> list[Pair]:
        # One request buys ``n`` more of the task's demonstrations: its own pairs first, then those pairs
        # under other augmentations, marked with PADDED_NOTE (26 Sep: eight requests of one pair each on
        # every grid track, and a task without generators has fewer pairs than that).
        from continual_arc.augment import Augmentation

        own = [Pair.from_json(p) for p in self._load()[task_id]["train"]]

        def pad(j: int) -> Pair | None:
            if not own or self.own_pairs_only:
                return None
            aug = _pad_augmentation((task_id, instance_seed, j), Augmentation.identity())
            return replace(aug.pair(own[j % len(own)]), note=PADDED_NOTE)

        return _drip(own, pad, exclude, request_index * n, n)

    def demo_supply(self, task_id: str, instance_seed: int, exclude: Pair) -> int | None:
        if not self.own_pairs_only:
            return None
        return _supply([Pair.from_json(p) for p in self._load()[task_id]["train"]], exclude)


class AugmentedArcSource:
    """JSON tasks served under seeded augmentations.

    An instance is one of the task's pairs (demonstrations and tests pooled)
    under an augmentation drawn from the instance seed: one of the eight
    symmetries of the square and a permutation of colours 1..9, or with
    ``invariances`` only those the task's rule survives. Demonstrations for the
    instance are the task's *other* pairs under the same augmentation, one per
    request (``demos_per_request``), so the pairs a learner sees within an instance
    describe one consistent rule.
    Across instances the augmentation changes, so the same underlying pair
    returns as a different grid and a learner that has cached outputs gains
    nothing; a learner that has learned the rule up to symmetry and relabelling
    gains everything. The number of distinct instances per task is eight times
    the number of pairs times the number of colour permutations, which is more
    than any schedule asks for.

    ``symmetries`` and ``colours`` switch either half of the augmentation off,
    and ``keep_background`` fixes colour 0. All three are recorded in the run.

    ``invariances`` restricts the draw per task: ``"labels"`` reads :data:`INVARIANCES_FILE` (or pass the
    mapping itself), and a task is then drawn only under the symmetries its rule is unchanged by, with only
    the colours outside its fixed ones relabelled, so a rule a learner kept from one instance is the rule
    of the next. ``own_pairs_only`` serves the task's other pairs in the instance's frame and nothing after
    them. The ARC-AGI-2 track uses both (Dan, 26 Sep 2026).
    """

    def __init__(self, split: str = "training", tasks: dict[str, dict] | None = None,
                 dataset: str = "arc2", name: str | None = None, *, symmetries: bool = True,
                 colours: bool = True, keep_background: bool = True,
                 invariances: dict[str, dict] | str | None = None, own_pairs_only: bool = False):
        if split not in ("training", "evaluation"):
            raise ValueError(split)
        self.split = split
        self.dataset = dataset
        if name is None:
            name = {("arc2", "training"): SOURCE_ARC2_TRAIN_AUG, ("arc2", "evaluation"): SOURCE_ARC2_EVAL_AUG}.get(
                (dataset, split), f"{dataset}_{split}_aug")
        self.name = name
        self.symmetries = symmetries
        self.colours = colours
        self.keep_background = keep_background
        self.own_pairs_only = own_pairs_only
        self._invariances_spec = invariances
        self._invariances: dict[str, dict] | None = invariances if isinstance(invariances, dict) else None
        self._tasks = tasks

    def _load(self) -> dict[str, dict]:
        if self._tasks is None:
            from continual_arc.data import load_arc_tasks

            self._tasks = load_arc_tasks(self.split, dataset=self.dataset)
        return self._tasks

    def task_ids(self) -> list[str]:
        return sorted(self._load())

    def pairs(self, task_id: str) -> list[Pair]:
        task = self._load()[task_id]
        return [Pair.from_json(p) for p in task["train"]] + [Pair.from_json(p) for p in task["test"]]

    def label(self, task_id: str) -> dict | None:
        """The task's invariance label, or None when the draw is free."""
        if self._invariances_spec is None:
            return None
        if self._invariances is None:
            self._invariances = load_invariances() if self._invariances_spec == "labels" \
                else load_invariances(self._invariances_spec)
        try:
            return self._invariances[task_id]
        except KeyError:
            raise KeyError(f"task {task_id!r} has no invariance label in {INVARIANCES_FILE}") from None

    def _allowed(self, label: dict) -> tuple[list[int], list[int]]:
        from continual_arc.augment import SYMMETRY_NAMES

        syms = [SYMMETRY_NAMES.index(n) for n in label["symmetries"]] if self.symmetries else [0]
        fixed = list(label["fixed_colours"]) if self.colours else list(range(1, 10))
        return syms, fixed

    def frames(self, task_id: str) -> list:
        """The task's distinct frames, in the order its visits take them: seeded draws within its label, each
        kept only if no grid of the task looks under it as under a frame already kept, so a pair never shows
        the same grid on two visits (up to :data:`MAX_FRAMES`). A task whose label allows one look has one."""
        cache = self.__dict__.setdefault("_frames", {})
        if task_id not in cache:
            from continual_arc.augment import Augmentation

            syms, fixed = self._allowed(self.label(task_id))
            grids = [g for p in self.pairs(task_id) for g in (p.input, p.output)]
            kept, seen, last = [], set(), 0
            for j in range(64 * MAX_FRAMES):
                if j - last > 512:   # the label's looks are exhausted
                    break
                aug = Augmentation.sample_within(derive_seed("frame", task_id, j), syms, fixed)
                looks = {grid_hash(aug.grid(g)) for g in grids}
                if looks & seen:
                    continue
                kept.append(aug)
                seen |= looks
                last = j
                if len(kept) == MAX_FRAMES:
                    break
            cache[task_id] = kept
        return cache[task_id]

    def augmentation(self, task_id: str, instance_seed: int, exposure: int | None = None):
        """The instance's frame: the free draw without labels; with them, the task's frame for this visit
        (``exposure``, when the environment says which visit it is), else a seeded draw within the label."""
        from continual_arc.augment import Augmentation

        label = self.label(task_id)
        if label is None:
            return Augmentation.sample(
                derive_seed("aug", task_id, instance_seed), symmetries=self.symmetries,
                colours=self.colours, keep_background=self.keep_background,
            )
        if exposure is not None:
            frames = self.frames(task_id)
            return frames[exposure % len(frames)]
        syms, fixed = self._allowed(label)
        return Augmentation.sample_within(derive_seed("aug", task_id, instance_seed), syms, fixed)

    def _pick(self, task_id: str, instance_seed: int) -> int:
        pairs = self.pairs(task_id)
        return derive_seed("aug-pair", task_id, instance_seed) % len(pairs)

    def test_instance(self, task_id: str, instance_seed: int, exposure: int | None = None) -> Pair:
        pairs = self.pairs(task_id)
        return self.augmentation(task_id, instance_seed, exposure).pair(pairs[self._pick(task_id, instance_seed)])

    def demos(
        self, task_id: str, instance_seed: int, request_index: int, n: int, exclude: Pair,
        exposure: int | None = None,
    ) -> list[Pair]:
        # One request buys ``n`` more demonstrations: the task's other pairs under the instance's own
        # augmentation first (the test's frame), then those pairs under other augmentations, marked with
        # PADDED_NOTE because their frame can differ from the test's (26 Sep: eight requests of one pair
        # each on every grid track; a task has only 3 to 7 public pairs).
        pairs = self.pairs(task_id)
        held = self._pick(task_id, instance_seed)
        aug = self.augmentation(task_id, instance_seed, exposure)
        others = [pairs[i] for i in range(len(pairs)) if i != held]
        in_frame = [aug.pair(p) for p in others]

        def pad(j: int) -> Pair | None:
            if not others or self.own_pairs_only:
                return None
            other = _pad_augmentation((task_id, instance_seed, j), aug, symmetries=self.symmetries,
                                      colours=self.colours, keep_background=self.keep_background)
            return replace(other.pair(others[j % len(others)]), note=PADDED_NOTE)

        return _drip(in_frame, pad, exclude, request_index * n, n)

    def demo_supply(self, task_id: str, instance_seed: int, exclude: Pair, exposure: int | None = None) -> int | None:
        if not self.own_pairs_only:
            return None
        pairs = self.pairs(task_id)
        held = self._pick(task_id, instance_seed)
        aug = self.augmentation(task_id, instance_seed, exposure)
        return _supply([aug.pair(pairs[i]) for i in range(len(pairs)) if i != held], exclude)

    def describe(self) -> dict:
        d = {"dataset": self.dataset, "split": self.split, "symmetries": self.symmetries,
             "colours": self.colours, "keep_background": self.keep_background, "own_pairs_only": self.own_pairs_only}
        if self._invariances_spec is not None:
            d["invariances"] = self._invariances_spec if isinstance(self._invariances_spec, str) else "mapping"
        return d


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
                # Tasks without generators serve their own pairs only, on every track (the ARC-AGI-1 epilogue
                # from 27 Sep; its marked transformed copies before that could show the rule in another frame).
                SOURCE_ARC_TRAIN: FixedArcSource("training", own_pairs_only=True),
                SOURCE_ARC_EVAL: FixedArcSource("evaluation", own_pairs_only=True),
                SOURCE_ARC2_TRAIN_AUG: AugmentedArcSource("training", dataset="arc2"),
                # The ARC-AGI-2 track: each task drawn only under what its rule survives, and only the
                # task's own pairs served, in the stream and in the epilogue (Dan, 26 Sep 2026).
                SOURCE_ARC2_EVAL_AUG: AugmentedArcSource("evaluation", dataset="arc2", invariances="labels",
                                                         own_pairs_only=True),
                SOURCE_ARC2_EVAL: FixedArcSource("evaluation", dataset="arc2", own_pairs_only=True),
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

    @staticmethod
    def _visit(method, exposure: int | None) -> dict:
        """``{"exposure": ...}`` for a source method that takes the visit index, else nothing."""
        import inspect

        if exposure is None:
            return {}
        try:
            return {"exposure": exposure} if "exposure" in inspect.signature(method).parameters else {}
        except (TypeError, ValueError):
            return {}

    def test_instance(self, task_id: str, instance_seed: int, exposure: int | None = None) -> Pair:
        method = self.source_for(task_id).test_instance
        return method(task_id, instance_seed, **self._visit(method, exposure))

    def demos(
        self, task_id: str, instance_seed: int, request_index: int, n: int, exclude: Pair,
        exposure: int | None = None,
    ) -> list[Pair]:
        method = self.source_for(task_id).demos
        return method(task_id, instance_seed, request_index, n, exclude, **self._visit(method, exposure))

    def demo_supply(self, task_id: str, instance_seed: int, exclude: Pair, exposure: int | None = None) -> int | None:
        """How many demonstrations the task's source can serve for the instance; None when unlimited."""
        supply = getattr(self.source_for(task_id), "demo_supply", None)
        if supply is None:
            return None
        return supply(task_id, instance_seed, exclude, **self._visit(supply, exposure))
