"""The second track's source: ARC-GEN's generators, their screen and the pool it records.

The first tests run on a checkout of three toy generators written here, so they need no data; the rest need the
second track's pinned data (``continual-arc data fetch --datasets arc2``) and are skipped without it.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
from conftest import needs_arc2

from continual_arc.arcgen import (
    ArcGen,
    GenerationError,
    TooLong,
    _Counted,
    pool,
    screen_record,
    screen_task,
)
from continual_arc.data import ARC2_COMMIT, ARCGEN_COMMIT
from continual_arc.grids import grid_hash
from continual_arc.seeds import derive_seed
from continual_arc.tasks import ArcGenSource

COMMON = """\
import random


def randint(a, b):
  return random.randint(a, b)
"""

# sorts each row of a random grid: a rule no rotation or reflection gives
SORTS = """\
import common


def generate(size=None):
  if size is None:
    size = common.randint(3, 6)
  grid = [[common.randint(0, 9) for _ in range(size)] for _ in range(size)]
  return {"input": grid, "output": [sorted(row) for row in grid]}


def validate():
  return {"train": [generate(size=3), generate(size=4)], "test": [generate(size=5)]}
"""

# mirrors each row: always the input's reflection, which the screen refuses
MIRRORS = SORTS.replace("[sorted(row) for row in grid]", "[row[::-1] for row in grid]")

# never stops drawing
SPINS = """\
import common


def generate():
  while True:
    common.randint(0, 9)


def validate():
  return {"train": [], "test": []}
"""

# fails on half its seeds
FLAKY = """\
import common


def generate():
  if common.randint(0, 1):
    raise ValueError("a layout that does not fit")
  return {"input": [[common.randint(0, 9) for _ in range(3)] for _ in range(3)], "output": [[1]]}


def validate():
  return {"train": [], "test": []}
"""


@pytest.fixture
def toy(tmp_path: Path) -> ArcGen:
    (tmp_path / "common.py").write_text(COMMON)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for task_id, code in (("0000000a", SORTS), ("0000000b", MIRRORS), ("0000000c", SPINS), ("0000000d", FLAKY)):
        (tasks / f"task_{task_id}.py").write_text(code)
    return ArcGen(tmp_path, max_random=10_000)


def test_the_counted_random_draws_exactly_what_random_does():
    for seed in range(20):
        a, b = _Counted(seed, 10**9), random.Random(seed)
        draws = [lambda r: r.randint(0, 9), lambda r: r.choice(range(30)), lambda r: r.sample(range(40), 7),
                 lambda r: r.random(), lambda r: r.choices(range(5), k=4), lambda r: r.getrandbits(70)]
        assert [d(a) for d in draws] == [d(b) for d in draws]
    with pytest.raises(TooLong):
        counted = _Counted(0, 5)
        for _ in range(6):
            counted.randint(0, 9)


def test_a_pair_depends_on_its_task_and_seed_alone(toy):
    state = random.getstate()
    a = toy.generate("0000000a", 5)
    assert random.getstate() == state   # the global random is never drawn from, nor reseeded
    random.random()
    assert toy.generate("0000000a", 5) == a != toy.generate("0000000a", 6)
    assert "common" not in sys.modules
    assert a.output == [sorted(row) for row in a.input]


def test_a_call_past_its_random_numbers_stops_and_a_failing_draw_is_redrawn(toy):
    with pytest.raises(TooLong):
        toy.draw("0000000c", 0)
    with pytest.raises(GenerationError) as err:
        toy.generate("0000000c", 0)
    assert isinstance(err.value.__cause__, TooLong)
    pairs = [toy.generate("0000000d", s) for s in range(20)]   # half the calls raise; each seed still gives a pair
    assert pairs == [toy.generate("0000000d", s) for s in range(20)] and len({grid_hash(p.input) for p in pairs}) == 20
    with pytest.raises(KeyError):
        toy.generate("0000000e", 0)


def test_a_demonstration_never_shares_an_input_with_the_test_or_another(toy):
    source = ArcGenSource(toy)
    test = source.test_instance("0000000a", 3)
    assert test == toy.generate("0000000a", derive_seed("test", 3)) == source.test_instance("0000000a", 3)
    for request in range(3):
        demos = source.demos("0000000a", 3, request, 4, test)
        assert demos == source.demos("0000000a", 3, request, 4, test)
        hashes = [grid_hash(p.input) for p in demos]
        assert len(set(hashes)) == 4 and grid_hash(test.input) not in hashes
    # an input the generator would draw is excluded: the next derived seed's pair is served in its place
    first = toy.generate("0000000a", 11)
    assert toy.generate("0000000a", 11, exclude_input_hashes={grid_hash(first.input)}).input != first.input


def test_the_screen_passes_a_faithful_generator_and_names_what_fails(toy):
    good = screen_task(toy, "0000000a", toy.validate("0000000a"), n=60)
    assert good["ok"] and good["faithful"] and good["draws"] == 60, good["reasons"]
    assert set(good["shared"]) == {"same_shape", "output_colors_in_input", "scale"}
    other = toy.validate("0000000b")   # published pairs the generator does not draw
    unfaithful = screen_task(toy, "0000000a", other, n=20)
    assert not unfaithful["faithful"] and "validate does not reproduce the published pairs" in unfaithful["reasons"]
    mirror = screen_task(toy, "0000000b", toy.validate("0000000b"), n=20)
    assert mirror["faithful"] and "output is always symmetry 1 of the input" in mirror["reasons"]
    spins = screen_task(toy, "0000000c", toy.validate("0000000a"), n=20)
    assert not spins["ok"] and any("random numbers" in r for r in spins["reasons"]) and spins["draws"] == 0
    flaky = screen_task(toy, "0000000d", toy.validate("0000000a"), n=20)
    assert any("generate raised ValueError" in r for r in flaky["reasons"])
    assert any("inputs distinct" in r or "outputs distinct" in r for r in flaky["reasons"])   # one output, [[1]]


def test_a_draw_that_breaks_what_every_published_pair_keeps_fails_the_screen(toy):
    published = toy.validate("0000000a")
    for pair in published["train"] + published["test"]:   # every published output is 1 by 1
        pair["output"] = [[pair["output"][0][0]]]
    record = screen_task(toy, "0000000a", published, n=20)
    assert "20 draws break output_shape, which every published pair keeps" in record["reasons"]


# -- the pool -------------------------------------------------------------------

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "arc2-pool-golden.json").read_text())
"""The pool's digest, and the digests of the second track's streams over it, as the first track's are frozen."""


def _stream_digest(cfg: str, seed: int) -> str:
    from continual_arc.configs import build_schedule
    from continual_arc.schedule import PHASE_STREAM

    s = build_schedule(cfg, seed, eval_task_ids=[f"e{i:03d}" for i in range(114)])   # the stream over pool()
    stream = [(x.task_id, x.instance_seed, x.phase, x.reset_before) for x in s.slots if x.phase == PHASE_STREAM]
    return hashlib.sha256(json.dumps(stream).encode()).hexdigest()


def test_the_pool_is_what_the_screen_recorded_at_the_pinned_commits():
    rec = screen_record()
    assert rec["arc_gen"] == ARCGEN_COMMIT and rec["arc_agi_2"] == ARC2_COMMIT
    assert pool() == rec["pool"] == sorted(t for t, r in rec["tasks"].items() if r["ok"])
    counts = rec["counts"]
    assert counts["screened"] == len(rec["tasks"]) == 172 and counts["new_to_second_edition"] == 233
    assert counts["passed"] == len(pool()) >= 25 + 8
    assert all(r["faithful"] for t, r in rec["tasks"].items() if t in set(pool()))
    # frozen as the streams are: a new pool is a new track
    assert hashlib.sha256(json.dumps(pool()).encode()).hexdigest() == GOLDEN["pool"]


def test_the_second_tracks_streams_over_its_pool_are_frozen():
    keys = [k for k in GOLDEN if k != "pool"]
    assert len(keys) == 6
    for key in keys:
        cfg, _, seed = key.split("/")
        assert _stream_digest(cfg, int(seed)) == GOLDEN[key], key


@needs_arc2
@pytest.mark.data
def test_the_screen_covered_every_new_training_task_arc_gen_has_and_no_first_edition_task():
    from continual_arc.data import arc1_task_ids, arc2_new_task_ids

    new = arc2_new_task_ids("training")
    assert len(new) == 233 and len(arc2_new_task_ids("evaluation")) == 114
    assert set(screen_record()["tasks"]) == set(new) & set(ArcGen().task_ids)
    assert not set(pool()) & arc1_task_ids()


@needs_arc2
@pytest.mark.data
def test_pool_tasks_pass_the_screen_again():
    from continual_arc.data import load_arc_tasks

    training = load_arc_tasks("training", dataset="arc2")
    arcgen = ArcGen()
    for task_id in pool()[::40]:
        again = screen_task(arcgen, task_id, training[task_id], n=100)
        assert again["ok"] and again["faithful"], (task_id, again["reasons"])


@needs_arc2
@pytest.mark.data
def test_the_environment_serves_the_second_tracks_sources():
    from continual_arc.data import arc2_new_task_ids, load_arc_tasks
    from continual_arc.env import ContinualArcEnv, EnvConfig, RequestDemos, Submit
    from continual_arc.schedule import RANDOM, epilogue_schedule, stream_schedule
    from continual_arc.tasks import SOURCE_ARC2_EVAL, SOURCE_ARCGEN

    stream_ids, held_out = pool()[:3], arc2_new_task_ids("evaluation")[:2]
    sched = stream_schedule(stream_ids, 6, seed=0, source=SOURCE_ARCGEN, params=RANDOM, order="random").concat(
        epilogue_schedule(held_out, seed=0, source=SOURCE_ARC2_EVAL))
    env = ContinualArcEnv(sched, config=EnvConfig(demos_per_request=2))
    evaluation = load_arc_tasks("evaluation", dataset="arc2")
    obs, seen = env.reset(), set()
    while obs is not None:
        h = grid_hash(obs.input)
        assert h not in seen
        seen.add(h)
        demos = env.step(RequestDemos()).feedback.pairs
        if obs.phase == "epilogue":   # the task's own pairs
            own = [p["input"] for p in evaluation[obs.task_id]["train"]]
            assert [p.input for p in demos] == own[:2]
        else:
            assert len(demos) == 2 and all(grid_hash(p.input) != h for p in demos)
        res = env.step(Submit(env._open[obs.step].pair.output))
        assert res.record.solved and res.record.cost == 1
        obs = res.observation
    assert len(env.records) == 8
