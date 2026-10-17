import pytest

from continual_arc.augment import (
    N_SYMMETRIES,
    SYMMETRY_NAMES,
    Augmentation,
    colour_permutation,
    compose,
    is_subgroup,
    largest_subgroup_within,
    symmetry,
)
from continual_arc.env import ContinualArcEnv, EnvConfig, RequestDemos, Submit
from continual_arc.schedule import RANDOM, stream_schedule
from continual_arc.tasks import (
    PADDED_NOTE,
    SOURCE_ARC2_EVAL,
    SOURCE_ARC2_EVAL_AUG,
    SOURCE_ARC_EVAL,
    AugmentedArcSource,
    FixedArcSource,
    Pair,
    TaskRegistry,
    load_invariances,
)

G = [[1, 2, 3], [4, 5, 6]]


def test_symmetries_are_the_dihedral_group():
    images = {tuple(map(tuple, symmetry(G, i))) for i in range(N_SYMMETRIES)}
    assert len(images) == N_SYMMETRIES  # all distinct on an asymmetric grid
    assert symmetry(G, 0) == G
    r = G
    for _ in range(4):
        r = symmetry(r, 1)
    assert r == G  # four quarter turns
    assert symmetry(symmetry(G, 4), 4) == G  # a flip is an involution


def test_colour_permutation_keeps_background_and_is_a_bijection():
    perm = colour_permutation(11)
    assert perm[0] == 0
    assert sorted(perm) == list(range(10))
    assert colour_permutation(11) == perm
    assert colour_permutation(12) != perm


def test_augmentation_is_seeded_and_applies_to_both_halves():
    a = Augmentation.sample(5)
    assert a == Augmentation.sample(5)
    p = a.pair(Pair([[1, 0]], [[0, 1]]))
    assert p.input == a.grid([[1, 0]])
    assert p.output == a.grid([[0, 1]])
    assert Augmentation.sample(5, symmetries=False).symmetry == 0
    assert Augmentation.sample(5, colours=False).colours == tuple(range(10))


TASKS = {
    "t1": {
        "train": [
            {"input": [[1, 0], [0, 2]], "output": [[2, 0], [0, 1]]},
            {"input": [[3, 3], [0, 0]], "output": [[0, 0], [3, 3]]},
            {"input": [[4, 0, 0]], "output": [[0, 0, 4]]},
        ],
        "test": [{"input": [[5, 6]], "output": [[6, 5]]}],
    }
}


def test_augmented_source_serves_consistent_instances():
    src = AugmentedArcSource(tasks=TASKS)
    assert src.task_ids() == ["t1"]
    seen = set()
    for seed in range(40):
        test = src.test_instance("t1", seed)
        aug = src.augmentation("t1", seed)
        # the test pair is one of the task's pairs under the instance's augmentation
        assert any(aug.pair(p) == test for p in src.pairs("t1"))
        others = len(src.pairs("t1")) - 1
        demos = [d for k in range(8) for d in src.demos("t1", seed, k, 1, exclude=test)]
        assert len(demos) == 8 and all(d.input != test.input for d in demos)
        # the task's other pairs under the same augmentation come first, unmarked
        for d in demos[:others]:
            assert d.note is None and any(aug.pair(p) == d and aug.pair(p) != test for p in src.pairs("t1"))
        # then the padding: other pairs under other augmentations, marked as transformed copies
        assert all(d.note == PADDED_NOTE for d in demos[others:])
        seen.add(tuple(map(tuple, test.input)))
    assert len(seen) > 8  # recurrence does not repeat the same grid


def test_demonstrations_drip_one_pair_per_request_and_are_seeded():
    src = AugmentedArcSource(tasks=TASKS)
    test = src.test_instance("t1", 3)
    one_by_one = [d for k in range(8) for d in src.demos("t1", 3, k, 1, exclude=test)]
    in_pairs = src.demos("t1", 3, 0, 2, exclude=test) + src.demos("t1", 3, 1, 2, exclude=test)
    assert one_by_one[:4] == in_pairs  # a request of n pairs is n consecutive demonstrations
    assert one_by_one == [d for k in range(8) for d in src.demos("t1", 3, k, 1, exclude=test)]


def test_fixed_source_serves_its_pairs_then_marked_padding():
    tasks = {"f": {"train": [{"input": [[1, 0]], "output": [[0, 1]]}, {"input": [[2, 0]], "output": [[0, 2]]}],
                   "test": [{"input": [[3, 0]], "output": [[0, 3]]}]}}
    src = FixedArcSource("evaluation", tasks=tasks, dataset="arc1")
    test = src.test_instance("f", 0)
    demos = [d for k in range(8) for d in src.demos("f", 0, k, 1, exclude=test)]
    assert len(demos) == 8
    assert [d.note for d in demos[:2]] == [None, None] and demos[0].input == [[1, 0]]
    assert all(d.note == PADDED_NOTE for d in demos[2:])


def test_named_symmetries_and_subgroups():
    g = [[1, 2, 3], [4, 5, 6]]
    assert symmetry(g, SYMMETRY_NAMES.index("flip_ud")) == [[4, 5, 6], [1, 2, 3]]
    assert symmetry(g, SYMMETRY_NAMES.index("transpose")) == [[1, 4], [2, 5], [3, 6]]
    assert symmetry(g, SYMMETRY_NAMES.index("anti_transpose")) == [[6, 3], [5, 2], [4, 1]]
    assert compose(SYMMETRY_NAMES.index("flip_lr"), SYMMETRY_NAMES.index("flip_ud")) == SYMMETRY_NAMES.index("rot180")
    assert is_subgroup([0, 4]) and is_subgroup(range(8)) and is_subgroup([0, 1, 2, 3])
    assert not is_subgroup([0, 4, 6]) and not is_subgroup([0, 1])
    assert largest_subgroup_within([0, 4, 6]) == (0,)   # two candidates of one size: only what they share
    assert largest_subgroup_within([0, 2, 4, 6, 7]) == (0, 2, 4, 6)


def _gravity(grid):
    """Every non-zero cell falls to the bottom of its column: a rule that a quarter turn changes."""
    h, w = len(grid), len(grid[0])
    cols = []
    for c in range(w):
        cells = [grid[r][c] for r in range(h) if grid[r][c]]
        cols.append([0] * (h - len(cells)) + cells)
    return [[cols[c][r] for c in range(w)] for r in range(h)]


_GRAVITY_INPUTS = ([[1, 0, 0], [0, 0, 2], [0, 0, 0]], [[0, 3, 0], [0, 0, 0], [0, 0, 0]],
                   [[0, 0, 0], [4, 0, 0], [0, 5, 0]], [[6, 0, 7], [0, 0, 0], [0, 0, 0]])
GRAVITY = {"g": {"train": [{"input": x, "output": _gravity(x)} for x in _GRAVITY_INPUTS],
                 "test": [{"input": [[0, 8, 0], [9, 0, 0], [0, 0, 0]], "output": _gravity([[0, 8, 0], [9, 0, 0], [0, 0, 0]])}]}}


def test_the_rule_kept_from_one_instance_holds_in_every_instance_of_a_restricted_draw():
    """Gravity to the bottom survives a left-right mirror and any recolouring but not a quarter turn: under
    its label every instance, test and demonstrations, follows the rule stated once; under the free draw
    some instances do not."""
    label = {"g": {"symmetries": ["identity", "flip_lr"], "fixed_colours": []}}
    restricted = AugmentedArcSource(tasks=GRAVITY, invariances=label, own_pairs_only=True)
    free = AugmentedArcSource(tasks=GRAVITY)
    broken = 0
    for seed in range(64):
        test = restricted.test_instance("g", seed)
        assert _gravity(test.input) == test.output
        demos = [d for k in range(8) for d in restricted.demos("g", seed, k, 1, exclude=test)]
        assert demos and all(_gravity(d.input) == d.output for d in demos)
        t = free.test_instance("g", seed)
        broken += _gravity(t.input) != t.output
    assert broken > 0


def test_a_restricted_draw_moves_only_allowed_symmetries_and_free_colours():
    label = {"t1": {"symmetries": ["identity", "flip_lr", "flip_ud", "rot180"], "fixed_colours": [1, 2]}}
    src = AugmentedArcSource(tasks=TASKS, invariances=label)
    syms, moved = set(), set()
    for seed in range(200):
        aug = src.augmentation("t1", seed)
        syms.add(aug.symmetry)
        assert aug.colours[:3] == (0, 1, 2) and sorted(aug.colours) == list(range(10))
        moved |= {c for c in range(3, 10) if aug.colours[c] != c}
    assert syms == {0, 2, 4, 6} and moved == set(range(3, 10))
    with pytest.raises(KeyError):
        src.augmentation("unlabelled", 0)
    with pytest.raises(ValueError):   # a label that is not a subgroup is refused, not guessed at
        AugmentedArcSource(tasks=TASKS, invariances={"t1": {"symmetries": ["identity", "rot90"],
                                                           "fixed_colours": []}}).augmentation("t1", 0)


F_TASKS = {"f": {"train": [{"input": [[1, 0]], "output": [[0, 1]]}, {"input": [[2, 0]], "output": [[0, 2]]}],
                 "test": [{"input": [[3, 0]], "output": [[0, 3]]}]}}


def test_own_pairs_only_serves_the_task_s_pairs_and_nothing_after():
    src = AugmentedArcSource(tasks=TASKS, own_pairs_only=True)
    for seed in range(20):
        test = src.test_instance("t1", seed)
        supply = src.demo_supply("t1", seed, test)
        assert supply == len(src.pairs("t1")) - 1
        demos = [src.demos("t1", seed, k, 1, exclude=test) for k in range(8)]
        assert all(len(d) == 1 and d[0].note is None for d in demos[:supply])
        assert all(d == [] for d in demos[supply:])
    own = FixedArcSource("evaluation", tasks=F_TASKS, dataset="arc2", own_pairs_only=True)
    test = own.test_instance("f", 0)
    assert own.demo_supply("f", 0, test) == 2 and own.demos("f", 0, 2, 1, exclude=test) == []
    assert FixedArcSource("evaluation", tasks=F_TASKS, dataset="arc1").demo_supply("f", 0, test) is None
    assert AugmentedArcSource(tasks=TASKS).demo_supply("t1", 0, test) is None


def test_an_instance_s_demo_cap_is_what_its_task_can_serve():
    """A source that runs out caps its instances at its supply, refuses the next request at no cost and
    records the cap; a padded source keeps the configuration's cap and records nothing new."""
    for source, cap in ((FixedArcSource("evaluation", tasks=F_TASKS, dataset="arc2", own_pairs_only=True, name="own"), 2),
                        (FixedArcSource("evaluation", tasks=F_TASKS, dataset="arc1", name="padded"), 8)):
        sched = stream_schedule(["f"], 1, seed=0, source=source.name, params=RANDOM, mode="random")
        env = ContinualArcEnv(sched, registry=TaskRegistry({source.name: source}), config=EnvConfig())
        obs = env.reset()
        assert obs.demo_cap == cap
        for _ in range(cap):
            r = env.step(RequestDemos())
            assert not r.feedback.refused and r.cost_delta == 1.0 and len(r.feedback.pairs) == 1
        if cap < 8:
            r = env.step(RequestDemos())
            assert r.feedback.refused and r.cost_delta == 0.0 and r.feedback.pairs == []
        rec = env.step(Submit([[0, 3]])).record
        assert rec.solved and rec.cost == cap
        assert rec.metadata.get("demo_cap") == (cap if cap < 8 else None)
    assert EnvConfig().max_cost == 16   # the bound is the configuration's either way


def test_the_arc2_track_draws_under_its_labels_and_serves_only_own_pairs():
    sources = TaskRegistry.default()._sources
    stream, epilogue, arc1_epilogue = sources[SOURCE_ARC2_EVAL_AUG], sources[SOURCE_ARC2_EVAL], sources[SOURCE_ARC_EVAL]
    assert stream.own_pairs_only and stream.describe()["invariances"] == "labels"
    assert epilogue.own_pairs_only
    assert arc1_epilogue.own_pairs_only   # the ARC-AGI-1 epilogue too, since 27 Sep


def test_the_labels_file_is_well_formed():
    labels = load_invariances()
    assert labels
    for tid, lab in labels.items():
        assert is_subgroup([SYMMETRY_NAMES.index(n) for n in lab["symmetries"]]), tid
        assert set(lab["fixed_colours"]) <= set(range(1, 10)), tid


@pytest.mark.data
def test_every_task_of_the_arc2_pool_is_labelled():
    from continual_arc.data import arc2_new_task_ids

    assert set(load_invariances()) == set(arc2_new_task_ids("evaluation"))


def test_each_visit_of_a_task_takes_a_frame_of_its_own():
    """Through the environment, the k-th visit of a task is served in the task's k-th frame, so no visit's
    test grid is one an earlier visit showed, as a test or as a demonstration."""
    from continual_arc.grids import grid_hash
    from continual_arc.schedule import sequential_params

    label = {"g": {"symmetries": ["identity", "flip_lr"], "fixed_colours": []}}
    src = AugmentedArcSource(tasks=GRAVITY, invariances=label, own_pairs_only=True, name="g-src")
    frames = src.frames("g")
    assert len(frames) >= 8 and len(set(frames)) == len(frames)
    looks = [{grid_hash(f.grid(g)) for p in src.pairs("g") for g in (p.input, p.output)} for f in frames]
    assert all(not (looks[i] & looks[j]) for i in range(len(looks)) for j in range(i))
    sched = stream_schedule(["g"], 8, seed=0, source=src.name, params=sequential_params(8), mode="sequential")
    env = ContinualArcEnv(sched, registry=TaskRegistry({src.name: src}), config=EnvConfig())
    obs, shown = env.reset(), set()
    for visit in range(8):
        assert grid_hash(obs.input) not in shown
        shown.add(grid_hash(obs.input))
        while True:
            r = env.step(RequestDemos())
            if r.feedback.refused:
                break
            shown |= {grid_hash(p.input) for p in r.feedback.pairs}
        r = env.step(Submit(_gravity(obs.input)))
        assert r.record.solved and r.record.exposure_index == visit
        obs = r.observation
    assert obs is None


def test_a_task_with_one_look_has_one_frame():
    label = {"g": {"symmetries": ["identity"], "fixed_colours": list(range(1, 10))}}
    assert len(AugmentedArcSource(tasks=GRAVITY, invariances=label).frames("g")) == 1
