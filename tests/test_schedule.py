import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from conftest import as_users

from continual_arc.configs import build_schedule
from continual_arc.schedule import (
    PHASE_EPILOGUE,
    PHASE_STREAM,
    RANDOM,
    Schedule,
    Slot,
    StreamParams,
    epilogue_schedule,
    sequential_params,
    stream_schedule,
)

TASKS = [f"t{i:02d}" for i in range(20)]


def workstreams_schedule(tasks, length, seed, source, params=None):
    return stream_schedule(tasks, length, seed, source, params, "workstreams")


def test_sequential_is_blocked_and_keeps_state():
    s = stream_schedule(TASKS, 100, seed=1, source="fake", params=sequential_params(5), mode="sequential")
    assert len(s) == 100
    assert not any(slot.reset_before for slot in s.slots)  # a mode orders instances; it never touches the learner
    # each block of 5 is one task, and every task gets exactly one block
    blocks = [{slot.task_id for slot in s.slots[i : i + 5]} for i in range(0, 100, 5)]
    assert all(len(b) == 1 for b in blocks)
    assert sorted(next(iter(b)) for b in blocks) == TASKS
    infos = s.annotate()
    assert [i.exposure_index for i in infos[:5]] == [0, 1, 2, 3, 4]
    assert len({slot.instance_seed for slot in s.slots}) == len(s)


def test_sequential_stalls_past_the_supply():
    with pytest.raises(RuntimeError, match="stalled"):
        stream_schedule(TASKS, 101, seed=1, source="fake", params=sequential_params(5), mode="sequential")


def test_random_is_uniform_from_the_first_step():
    s = stream_schedule(TASKS, 2000, seed=3, source="fake", params=RANDOM, mode="random")
    assert len(s) == 2000
    first_seen = {}
    for i, slot in enumerate(s.slots):
        first_seen.setdefault(slot.task_id, i)
    assert set(first_seen) == set(TASKS)
    assert max(first_seen.values()) < 150  # every task live at once: all seen early, no arrivals
    counts = Counter(slot.task_id for slot in s.slots)
    assert min(counts.values()) > 60 and max(counts.values()) < 140  # near 100 each
    gaps = [i.gap for i in s.annotate() if i.gap is not None]
    assert 15 < sum(gaps) / len(gaps) < 25  # geometric with mean n_tasks


def test_modes_share_the_slot_seeds_and_differ_in_order():
    seq = stream_schedule(TASKS, 100, seed=1, source="fake", params=sequential_params(5), mode="sequential")
    rnd = stream_schedule(TASKS, 100, seed=1, source="fake", params=RANDOM, mode="random")
    ws = stream_schedule(TASKS, 100, seed=1, source="fake")
    assert [x.instance_seed for x in seq.slots] == [x.instance_seed for x in rnd.slots] == [x.instance_seed for x in ws.slots]
    assert len({tuple(x.task_id for x in s.slots) for s in (seq, rnd, ws)}) == 3
    assert {s.meta["mode"] for s in (seq, rnd, ws)} == {"sequential", "random", "workstreams"}


def test_workstreams_streams_are_frozen():
    """The canonical streams must never change under refactoring: cells already run on them."""
    gold = json.loads((Path(__file__).parent / "fixtures" / "stream-golden.json").read_text())
    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    for key, digest in gold.items():
        cfg, mode, seed = key.split("/")
        s = build_schedule(cfg, mode, int(seed), rearc_task_ids=pool, eval_task_ids=evals)
        stream = [(x.task_id, x.instance_seed, x.phase, x.reset_before) for x in s.slots]
        assert hashlib.sha256(json.dumps(stream).encode()).hexdigest() == digest, key


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_workstreams_properties(seed):
    params = StreamParams(concurrency=3, burst_mean=5, dormancy_median=20, dormancy_sigma=0.8, arrival_interval=8)
    s = workstreams_schedule(TASKS, 600, seed=seed, source="fake", params=params)
    assert len(s) == 600
    assert all(slot.phase == PHASE_STREAM for slot in s.slots)
    counts = Counter(slot.task_id for slot in s.slots)
    # every task is introduced and most recur
    assert set(counts) == set(TASKS)
    assert s.meta["n_tasks_introduced"] == len(TASKS)
    assert sum(1 for c in counts.values() if c > 1) >= len(TASKS) - 1
    # the set of seen tasks grows over the stream rather than all appearing at once
    first_seen = {}
    for i, slot in enumerate(s.slots):
        first_seen.setdefault(slot.task_id, i)
    introductions = sorted(first_seen.values())
    # 17 arrivals after the initial fill at ~8 steps each: expect the last well past step 60
    assert introductions[-1] > 60
    assert introductions[len(TASKS) // 2] > 20
    assert introductions[params.concurrency] > 0  # only the initial fill happens at step 0
    # gaps exist across a range of scales
    gaps = [i.gap for i in s.annotate() if i.gap is not None]
    assert min(gaps) >= 1
    assert max(gaps) > 30


def test_workstreams_deterministic():
    a = workstreams_schedule(TASKS, 300, seed=5, source="fake")
    b = workstreams_schedule(TASKS, 300, seed=5, source="fake")
    c = workstreams_schedule(TASKS, 300, seed=6, source="fake")
    assert a.slots == b.slots
    assert a.slots != c.slots


def test_workstreams_burstiness():
    params = StreamParams(concurrency=2, burst_mean=8, dormancy_median=50, dormancy_sigma=0.5, arrival_interval=10)
    s = workstreams_schedule(TASKS, 400, seed=2, source="fake", params=params)
    gaps = [i.gap for i in s.annotate() if i.gap is not None]
    short = sum(1 for g in gaps if g <= 4)
    long = sum(1 for g in gaps if g >= 25)
    # bursty: many short gaps (within a burst) and a decent number of long ones (dormancy)
    assert short > len(gaps) * 0.5
    assert long > 10


def test_workstreams_single_task_does_not_stall():
    s = workstreams_schedule(["only"], 50, seed=0, source="fake")
    assert len(s) == 50


def test_stream_validates_params():
    with pytest.raises(ValueError):
        workstreams_schedule(TASKS, 10, 0, "fake", StreamParams(concurrency=0))
    with pytest.raises(ValueError):
        workstreams_schedule(TASKS, 10, 0, "fake", StreamParams(burst_mean=2.5, burst_fixed=True))


def test_epilogue_reset_flag():
    e = epilogue_schedule(TASKS[:5], seed=1, source="eval", reset_each=True)
    assert all(slot.reset_before for slot in e.slots)
    assert all(slot.phase == PHASE_EPILOGUE for slot in e.slots)
    e2 = epilogue_schedule(TASKS[:5], seed=1, source="eval", instances_per_task=2, reset_each=True)
    assert [slot.reset_before for slot in e2.slots] == [True, False] * 5
    e3 = epilogue_schedule(TASKS[:5], seed=1, source="eval")
    assert not any(slot.reset_before for slot in e3.slots)


def test_concat_and_json_roundtrip(tmp_path):
    a = stream_schedule(TASKS[:5], 20, seed=1, source="fake", params=RANDOM, mode="random")
    e = epilogue_schedule(["e1", "e2"], seed=1, source="eval")
    s = a.concat(e, name="combo")
    assert s.phases() == ["stream", "epilogue"]
    assert s.sources["e1"] == "eval" and s.sources["t00"] == "fake"
    path = tmp_path / "s.json"
    s.save(path)
    back = Schedule.load(path)
    assert back.slots == s.slots
    assert back.sources == s.sources
    assert back.name == "combo"
    summary = back.summary()
    assert summary["n_steps"] == 22
    assert summary["phases"] == {"stream": 20, "epilogue": 2}


def test_concat_rejects_conflicting_sources():
    a = stream_schedule(["x"], 3, seed=1, source="fake", params=RANDOM, mode="random")
    b = stream_schedule(["x"], 3, seed=1, source="other", params=RANDOM, mode="random")
    with pytest.raises(ValueError):
        a.concat(b)


def test_from_json_rejects_unknown_format():
    with pytest.raises(ValueError):
        Schedule.from_json(json.loads('{"format": "nope"}'))


def test_rounds_group_slots_by_users_and_break_at_phases():
    stream = stream_schedule(TASKS[:4], 7, seed=1, source="fake", params=RANDOM, mode="random")
    epi = epilogue_schedule(["e1", "e2", "e3"], seed=1, source="eval")
    s = stream.concat(epi)
    assert s.users == 1 and s.rounds() == list(range(10))  # the individual regime: a round per slot
    s.users = 3
    assert s.rounds() == [0, 0, 0, 1, 1, 1, 2, 3, 3, 3]  # the last stream round is short; the epilogue starts a round
    assert s.round_starts() == [True, False, False, True, False, False, True, True, False, False]
    assert s.summary()["n_rounds"] == 4 and s.summary()["users"] == 3
    with pytest.raises(ValueError, match="users in order"):
        s.check_users()  # every slot still says user 0
    as_users(s, 3).check_users()


def test_annotate_counts_exposures_and_gaps_in_rounds():
    slots = [Slot(t, i, PHASE_STREAM) for i, t in enumerate("AABACAAB")]
    s = Schedule("x", 0, slots, {t: "fake" for t in "ABC"}, users=1)
    assert [(i.exposure_index, i.gap) for i in s.annotate()] == [
        (0, None), (1, 1), (0, None), (2, 2), (0, None), (3, 2), (4, 1), (1, 5)]
    assert [(i.user_exposure_index, i.user_gap) for i in s.annotate()] == [
        (i.exposure_index, i.gap) for i in s.annotate()]  # one user: the team is the user
    as_users(s, 4)  # rounds AABA | CAAB: within a round a task's slots share exposure and gap
    infos = s.annotate()
    assert [i.round for i in infos] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [(i.exposure_index, i.gap) for i in infos] == [
        (0, None), (0, None), (0, None), (0, None), (0, None), (1, 1), (1, 1), (1, 1)]
    # users 0..3 in each round: user 1 met A in round 0 and meets it again; the others meet a task new to them
    assert [(i.user_exposure_index, i.user_gap) for i in infos] == [
        (0, None), (0, None), (0, None), (0, None), (0, None), (1, 1), (0, None), (0, None)]


def test_users_survive_json_and_concat(tmp_path):
    s = as_users(stream_schedule(TASKS[:3], 6, seed=1, source="fake", params=RANDOM, mode="random"), 2)
    path = tmp_path / "s.json"
    s.save(path)
    back = Schedule.load(path)
    assert back.users == 2 and back.annotate() == s.annotate() and [x.user for x in back.slots] == [0, 1] * 3
    assert s.concat(epilogue_schedule(["e1"], seed=1, source="eval")).users == 2
    legacy = {("width" if k == "users" else k): v for k, v in s.to_json().items()}
    assert Schedule.from_json(legacy).users == 2  # files written before the rename
    assert Schedule.from_json({k: v for k, v in s.to_json().items() if k != "users"}).users == 1
    one = stream_schedule(TASKS[:3], 6, seed=1, source="fake", params=RANDOM, mode="random")
    assert all("user" not in x for x in one.to_json()["slots"])  # an individual schedule's file is unchanged


def test_interleave_users_lays_out_rounds_of_every_user():
    from continual_arc.schedule import interleave_users

    a = stream_schedule(TASKS[:3], 6, seed=1, source="fake", params=RANDOM, mode="random")
    b = stream_schedule(TASKS[:3], 6, seed=2, source="fake", params=RANDOM, mode="random")
    team = interleave_users([a.concat(epilogue_schedule(["e1", "e2"], seed=1, source="eval", reset_each=True)),
                             b.concat(epilogue_schedule(["e1", "e2"], seed=2, source="eval"))], seed=7, name="team")
    assert team.users == 2 and len(team) == 16 and team.seed == 7
    assert [x.user for x in team.slots] == [0, 1] * 8
    assert [x.task_id for x in team.slots[0:12:2]] == [x.task_id for x in a.slots]
    assert [x.task_id for x in team.slots[1:12:2]] == [x.task_id for x in b.slots]
    assert team.rounds() == [r for r in range(8) for _ in (0, 1)]
    # a wipe marked by any user belongs to the round, on its first slot
    assert [x.reset_before for x in team.slots] == [False] * 12 + [True, False, True, False]
    with pytest.raises(ValueError, match="phases"):
        interleave_users([a, epilogue_schedule(["e1"] * 1, seed=1, source="eval")], seed=0, name="bad")


def test_unconstrained_streams_are_prefix_consistent():
    """Without a supply, a longer horizon is a superset of a shorter one. With a supply (every
    configuration but ARC-AGI-3) the stream depends on its length, and a checkpoint is a prefix
    read of one run rather than a shorter configuration."""
    for mode, params in (("workstreams", None), ("random", RANDOM), ("sequential", sequential_params(30))):
        short = stream_schedule(TASKS, 200, seed=4, source="fake", params=params, mode=mode)
        long = stream_schedule(TASKS, 600, seed=4, source="fake", params=params, mode=mode)
        assert long.slots[:200] == short.slots, mode
    n = len(TASKS)
    short = stream_schedule(TASKS, 8 * n, seed=4, source="fake", params=RANDOM, mode="random", instances_per_task=8)
    long = stream_schedule(TASKS, 24 * n, seed=4, source="fake", params=RANDOM, mode="random", instances_per_task=24)
    assert long.slots[:8 * n] != short.slots


def test_even_visits_in_every_mode():
    """With a supply, the three modes are one multiset of instances in three orders: every
    task exactly instances_per_task times, every task introduced, and no stall."""
    n = len(TASKS)
    for mode, params in (("workstreams", StreamParams(concurrency=4, burst_mean=3.0, dormancy_median=30.0,
                                                       dormancy_sigma=1.0, arrival_interval=2.0)),
                         ("random", RANDOM), ("sequential", sequential_params(8))):
        for seed in (0, 1, 2):
            s = stream_schedule(TASKS, 8 * n, seed=seed, source="fake", params=params, mode=mode, instances_per_task=8)
            counts = Counter(slot.task_id for slot in s.slots)
            assert set(counts) == set(TASKS) and set(counts.values()) == {8}, (mode, seed)
            assert s.meta["instances_per_task"] == 8 and s.meta["n_tasks_introduced"] == n
    # the supply fixes the length
    with pytest.raises(ValueError, match="supply"):
        stream_schedule(TASKS, 8 * n + 1, seed=0, source="fake", params=RANDOM, mode="random", instances_per_task=8)
    # sequential is the same schedule with or without the supply
    a = stream_schedule(TASKS, 8 * n, seed=5, source="fake", params=sequential_params(8), mode="sequential")
    b = stream_schedule(TASKS, 8 * n, seed=5, source="fake", params=sequential_params(8), mode="sequential", instances_per_task=8)
    assert a.slots == b.slots


def test_first_rounds_is_a_literal_prefix():
    stream = stream_schedule(TASKS[:4], 7, seed=1, source="fake", params=RANDOM, mode="random")
    epi = epilogue_schedule(["e1", "e2", "e3"], seed=1, source="eval")
    full = as_users(stream.concat(epi), 3)
    assert full.rounds() == [0, 0, 0, 1, 1, 1, 2, 3, 3, 3]  # round 2 is a short, phase-ending round

    cut = full.first_rounds(3)
    assert cut.slots == full.slots[:7]  # rounds 0, 1 and the short round 2, nothing of the epilogue
    assert cut.users == 3 and cut.sources == full.sources and cut.seed == full.seed
    assert cut.meta == {**full.meta, "cut_to_round": 3, "full_rounds": 4}

    assert full.first_rounds(0).slots == []
    assert full.first_rounds(4).slots == full.slots  # every round: the whole schedule
    with pytest.raises(ValueError, match="between 0 and"):
        full.first_rounds(5)
    with pytest.raises(ValueError, match="between 0 and"):
        full.first_rounds(-1)


def test_first_rounds_on_the_individual_regime():
    s = stream_schedule(TASKS[:5], 20, seed=2, source="fake", params=RANDOM, mode="random")
    assert s.users == 1
    cut = s.first_rounds(8)
    assert cut.slots == s.slots[:8]  # one user: a round is one slot
    assert cut.users == 1


def test_small_configs_visit_every_task_equally():
    for cfg in ("arc1-small", "arc2-small"):
        for mode in ("workstreams", "random", "sequential"):
            pool = [f"t{i:03d}" for i in range(500)]
            s = build_schedule(cfg, mode, 0, rearc_task_ids=pool, eval_task_ids=[f"e{i:03d}" for i in range(500)])
            stream = [x for x in s.slots if x.phase == PHASE_STREAM]
            counts = Counter(x.task_id for x in stream)
            assert len(stream) == 200 and set(counts.values()) == {8} and len(counts) == 25, (cfg, mode)
