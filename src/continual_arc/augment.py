"""Grid augmentations for task sets that have no procedural generators.

ARC-AGI-2 ships only its public demonstration and test pairs. To serve a task
repeatedly with inputs a learner cannot cache, an instance is one of the task's
pairs under an *augmentation*: a symmetry of the square (one of the eight
elements of D8) composed with a permutation of the non-background colours.
The same augmentation is applied to the input and the output, and within one
instance to the demonstrations served for it, so the pairs a learner sees
describe one consistent rule.

The free draw (:meth:`Augmentation.sample`, any of the eight symmetries and any
relabelling of colours 1..9) is the standard test-time-training augmentation
(Akyürek et al., 2024). It assumes the task's rule commutes with the symmetry and
with the relabelling, which about a third of ARC rules do: a rule that names "the
red cells", or "gravity downwards", is changed by it, so a rule a learner kept
from one instance would be wrong in the next. The ARC-AGI-2 track therefore draws
each task's augmentation only from the symmetries its rule is unchanged by and
relabels only the colours with no fixed role in it (:meth:`Augmentation.sample_within`),
from per-task labels (``arc2_invariances.json``; Dan, 26 Sep 2026).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from continual_arc.grids import Grid, normalize_grid
from continual_arc.seeds import derive_seed
from continual_arc.tasks import Pair

N_SYMMETRIES = 8
BACKGROUND = 0
SYMMETRY_NAMES = ("identity", "rot90", "rot180", "rot270", "flip_lr", "anti_transpose", "flip_ud", "transpose")
"""Element ``i`` of D8 as :func:`symmetry` applies it: ``i % 4`` clockwise quarter turns, after a left-right
mirror when ``i >= 4`` (so 5 reflects across the anti-diagonal, 6 mirrors top to bottom, 7 transposes)."""


def _rotate(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    return [list(row) for row in zip(*grid[::-1], strict=True)]


def _flip(grid: Grid) -> Grid:
    """Mirror left to right."""
    return [row[::-1] for row in grid]


def symmetry(grid: Grid, index: int) -> Grid:
    """Apply element ``index`` (0..7) of D8: ``index % 4`` clockwise quarter turns,
    preceded by a horizontal flip when ``index >= 4``."""
    if not 0 <= index < N_SYMMETRIES:
        raise ValueError(f"symmetry index must be in 0..{N_SYMMETRIES - 1}, got {index}")
    out = [list(row) for row in grid]
    if index >= 4:
        out = _flip(out)
    for _ in range(index % 4):
        out = _rotate(out)
    return out


def compose(first: int, then: int) -> int:
    """The element of D8 that applies ``first`` and then ``then``."""
    probe = [[0, 1, 2], [3, 4, 5]]
    target = symmetry(symmetry(probe, first), then)
    return next(i for i in range(N_SYMMETRIES) if symmetry(probe, i) == target)


def is_subgroup(elements) -> bool:
    """Whether ``elements`` (indices of D8) contain the identity and are closed under composition."""
    s = set(elements)
    return 0 in s and all(compose(a, b) in s for a in s for b in s)


def largest_subgroup_within(elements) -> tuple[int, ...]:
    """The largest subgroup of D8 inside ``elements``; when two of the same size fit, only what they share."""
    s = set(elements) | {0}
    from itertools import combinations

    best: list[set[int]] = []
    for k in range(N_SYMMETRIES, 0, -1):
        best = [set(c) for c in combinations(sorted(s), k) if is_subgroup(c)]
        if best:
            break
    shared = set.intersection(*best) if best else {0}
    return tuple(sorted(shared))


def recolour(grid: Grid, mapping: tuple[int, ...]) -> Grid:
    """Relabel colours: cell ``c`` becomes ``mapping[c]``."""
    return [[mapping[c] for c in row] for row in grid]


def colour_permutation(seed: int, keep_background: bool = True) -> tuple[int, ...]:
    """A seeded permutation of the ten colours, fixing 0 unless told otherwise."""
    rng = random.Random(seed)
    if keep_background:
        rest = list(range(1, 10))
        rng.shuffle(rest)
        return (BACKGROUND, *rest)
    cols = list(range(10))
    rng.shuffle(cols)
    return tuple(cols)


def colour_permutation_within(seed: int, fixed=()) -> tuple[int, ...]:
    """A seeded permutation of the ten colours that moves only colours 1..9 outside ``fixed``."""
    rng = random.Random(seed)
    free = [c for c in range(1, 10) if c not in set(fixed)]
    shuffled = free[:]
    rng.shuffle(shuffled)
    mapping = list(range(10))
    for a, b in zip(free, shuffled, strict=True):
        mapping[a] = b
    return tuple(mapping)


@dataclass(frozen=True)
class Augmentation:
    symmetry: int
    colours: tuple[int, ...]

    @classmethod
    def identity(cls) -> Augmentation:
        return cls(0, tuple(range(10)))

    @classmethod
    def sample(cls, seed: int, *, symmetries: bool = True, colours: bool = True,
               keep_background: bool = True) -> Augmentation:
        rng = random.Random(derive_seed("augment", seed))
        sym = rng.randrange(N_SYMMETRIES) if symmetries else 0
        cols = colour_permutation(derive_seed("colours", seed), keep_background) if colours else tuple(range(10))
        return cls(sym, cols)

    @classmethod
    def sample_within(cls, seed: int, symmetries, fixed_colours=()) -> Augmentation:
        """An augmentation drawn only from ``symmetries`` (indices of D8, a subgroup) that relabels only
        the colours 1..9 outside ``fixed_colours``: a task's rule survives every draw (Dan, 26 Sep 2026)."""
        allowed = tuple(sorted(set(symmetries)))
        if not is_subgroup(allowed):
            raise ValueError(f"symmetries {allowed} are not a subgroup of D8")
        rng = random.Random(derive_seed("augment-within", seed))
        return cls(allowed[rng.randrange(len(allowed))],
                   colour_permutation_within(derive_seed("colours-within", seed), fixed_colours))

    def grid(self, grid: Grid) -> Grid:
        return normalize_grid(recolour(symmetry(grid, self.symmetry), self.colours))

    def pair(self, pair: Pair) -> Pair:
        return Pair(input=self.grid(pair.input), output=self.grid(pair.output))

    def to_json(self) -> dict:
        return {"symmetry": self.symmetry, "colours": list(self.colours)}
