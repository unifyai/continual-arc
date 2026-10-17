import pytest
from conftest import needs_arc2

from continual_arc.configs import CONFIGS, MODES, build_schedule, get_config, select_tasks

POOL = [f"r{i:03d}" for i in range(50)]
EVAL = [f"e{i:03d}" for i in range(40)]


@pytest.mark.parametrize("mode", MODES)
def test_build_schedule_modes(mode):
    sched = build_schedule("arc1-smoke", mode, seed=3, rearc_task_ids=POOL, eval_task_ids=EVAL)
    cfg = CONFIGS["arc1-smoke"]
    stream = [s for s in sched.slots if s.phase != "epilogue"]
    epi = [s for s in sched.slots if s.phase == "epilogue"]
    assert len(epi) == cfg.n_epilogue_tasks
    assert all(sched.sources[s.task_id] == "arc_eval" for s in epi)
    if mode == "fresh":
        # the transfer reference: the held-out tasks alone, a fresh learner before each
        assert not stream and all(s.reset_before for s in epi)
    else:
        assert len({s.task_id for s in stream}) == cfg.n_tasks
        assert all(sched.sources[s.task_id] == "rearc" for s in stream)
        assert not any(s.reset_before for s in sched.slots)  # a mode is an order, never a reset
        if mode == "sequential":
            assert len(stream) == cfg.n_tasks * cfg.instances_per_task == cfg.stream_length
        else:
            assert len(stream) == cfg.stream_length
    assert sched.meta["mode"] == mode and sched.meta["config"]["name"] == "arc1-smoke"


def test_same_task_set_across_modes():
    sets = []
    for mode in MODES:
        sched = build_schedule("arc1-smoke", mode, seed=11, rearc_task_ids=POOL, eval_task_ids=EVAL)
        sets.append(frozenset(s.task_id for s in sched.slots if s.phase != "epilogue"))
        sets.append(frozenset(s.task_id for s in sched.slots if s.phase == "epilogue"))
    assert sets[0] == sets[2] == sets[4]
    assert sets[1] == sets[3] == sets[5]


def test_no_epilogue():
    sched = build_schedule("arc1-smoke", "random", seed=0, rearc_task_ids=POOL, eval_task_ids=EVAL, epilogue=False)
    assert sched.phases() == ["stream"]


def test_select_tasks_deterministic_and_bounded():
    assert select_tasks(POOL, 5, 1, "x") == select_tasks(POOL, 5, 1, "x")
    assert select_tasks(POOL, 5, 1, "x") != select_tasks(POOL, 5, 2, "x")
    with pytest.raises(ValueError):
        select_tasks(POOL, 51, 0, "x")


def test_unknown_config_and_mode():
    with pytest.raises(KeyError):
        get_config("huge")
    with pytest.raises(ValueError):
        build_schedule("arc1-smoke", "sideways", 0, rearc_task_ids=POOL, eval_task_ids=EVAL)


def test_named_configs_arrivals_fit_in_stream():
    for cfg in CONFIGS.values():
        expected_arrival_span = cfg.n_tasks * cfg.workstreams.arrival_interval
        assert expected_arrival_span <= 0.75 * cfg.stream_length, cfg.name


def test_arc2_config_uses_augmented_stream_and_arc2_epilogue():
    from continual_arc.configs import build_schedule, get_config
    from continual_arc.tasks import SOURCE_ARC2_EVAL, SOURCE_ARC2_EVAL_AUG

    cfg = get_config("arc2-smoke")
    assert cfg.dataset == "arc2"
    pool = [f"a2-{i:03d}" for i in range(30)]
    evals = [f"e2-{i:03d}" for i in range(20)]
    sched = build_schedule(cfg, "workstreams", 0, rearc_task_ids=pool, eval_task_ids=evals)
    sources = set(sched.sources.values())
    assert sources == {SOURCE_ARC2_EVAL_AUG, SOURCE_ARC2_EVAL}
    assert sched.meta["dataset"] == "arc2"
    stream = [s for s in sched.slots if s.phase != "epilogue"]
    assert all(sched.sources[s.task_id] == SOURCE_ARC2_EVAL_AUG for s in stream)
    # one pool: the epilogue is drawn from it with the stream's tasks removed
    shared = build_schedule(cfg, "workstreams", 0, rearc_task_ids=evals)
    st = {s.task_id for s in shared.slots if s.phase != "epilogue"}
    ep = {s.task_id for s in shared.slots if s.phase == "epilogue"}
    assert st and ep and not st & ep and st | ep <= set(evals)


def test_dataset_is_validated():
    import pytest

    from continual_arc.configs import BenchmarkConfig

    with pytest.raises(ValueError):
        BenchmarkConfig(name="x", n_tasks=1, stream_length=1, n_epilogue_tasks=0, dataset="arc9")
    with pytest.raises(ValueError, match="multiple"):
        BenchmarkConfig(name="x", n_tasks=3, stream_length=10, n_epilogue_tasks=0)


def test_bare_names_are_refused_because_no_track_is_the_default():
    import pytest

    from continual_arc.configs import get_config

    with pytest.raises(KeyError, match="does not name its track"):
        get_config("small")
    assert get_config("arc1-small").track == "arc1"
    assert get_config("arc2-small").track == "arc2"
    assert get_config("arc3-small").track == "arc3"


def test_canonical_protocol_per_track():
    from continual_arc.configs import (
        CANONICAL_SEEDS,
        build_schedule,
        canonical_protocol,
        protocol_of,
    )
    from continual_arc.env import EnvConfig

    for track in ("arc1", "arc2", "arc3"):
        proto = canonical_protocol(track)
        assert proto["config"] == f"{track}-small" and proto["mode"] == "workstreams" and proto["epilogue"]
        assert proto["seeds"] == list(CANONICAL_SEEDS)
    pool = [f"t{i:03d}" for i in range(40)]
    evals = [f"e{i:03d}" for i in range(40)]
    sched = build_schedule("arc1-small", "workstreams", 0, rearc_task_ids=pool, eval_task_ids=evals)
    assert sched.meta["track"] == "arc1"
    p = protocol_of(sched, EnvConfig())
    assert p["canonical"] and p["deviations"] == []
    p = protocol_of(sched, EnvConfig(demo_cost=0.5))
    assert not p["canonical"] and any("demo_cost" in d for d in p["deviations"])
    sched = build_schedule("arc1-small", "sequential", 7, rearc_task_ids=pool, eval_task_ids=evals, epilogue=False)
    p = protocol_of(sched)
    assert not p["canonical"]
    assert any("sequential" in d for d in p["deviations"]) and any("seed 7" in d for d in p["deviations"])
    assert any("no epilogue" in d for d in p["deviations"])


def test_collective_config_is_a_hundred_users_each_on_their_own_stream():
    from collections import Counter

    from continual_arc.configs import (
        REGIME_COLLECTIVE,
        REGIME_INDIVIDUAL,
        BenchmarkConfig,
        canonical_protocol,
        protocol_of,
    )

    cfg = get_config("arc1-collective")
    assert cfg.users == 100 and cfg.regime == REGIME_COLLECTIVE and cfg.instances_per_task == 8
    assert all(c.regime == REGIME_INDIVIDUAL for n, c in CONFIGS.items() if n != "arc1-collective")
    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    sched = build_schedule(cfg, "workstreams", 0, rearc_task_ids=pool, eval_task_ids=evals)
    assert sched.users == 100 and sched.meta["config"]["users"] == 100 and len(sched.meta["users"]["seeds"]) == 100
    rounds = sched.rounds()
    assert rounds[-1] + 1 == 200 + 25  # 200 stream rounds, then every user's 25 held-out tasks
    assert [s.user for s in sched.slots] == [i % 100 for i in range(len(sched))]  # a round is every user, in order
    sched.check_users()
    stream = [s for s in sched.slots if s.phase != "epilogue"]
    assert len(stream) == 20_000 and len(sched) == 22_500
    assert set(Counter(s.task_id for s in stream).values()) == {800}  # every task 800 times over the team
    assert set(Counter((s.user, s.task_id) for s in stream).values()) == {8}  # and 8 times per user
    # the same tasks as the small configuration, stream and epilogue alike
    small = build_schedule("arc1-small", "workstreams", 0, rearc_task_ids=pool, eval_task_ids=evals)
    for phase in ("stream", "epilogue"):
        assert {s.task_id for s in sched.slots if s.phase == phase} == {s.task_id for s in small.slots if s.phase == phase}
    # user 0 walks exactly the small configuration's stream and epilogue at the same seed
    mine = [(s.task_id, s.instance_seed, s.phase, s.reset_before) for s in sched.slots if s.user == 0]
    assert mine == [(s.task_id, s.instance_seed, s.phase, s.reset_before) for s in small.slots]
    # the others walk their own draws of the same process: different orders, different instances
    one = [s for s in sched.slots if s.user == 1]
    assert [s.task_id for s in one] != [s.task_id for s in sched.slots if s.user == 0]
    assert not {s.instance_seed for s in one} & {s.instance_seed for s in sched.slots if s.user == 0}
    epi_orders = {tuple(s.task_id for s in sched.slots if s.user == u and s.phase == "epilogue") for u in range(100)}
    assert len(epi_orders) > 90  # a held-out task reaches the team gradually, in each user's own order
    p = protocol_of(sched)
    assert p["regime"] == REGIME_COLLECTIVE and p["users"] == 100 and not p["canonical"]
    assert any("collective" in d for d in p["deviations"])
    assert canonical_protocol("arc1")["regime"] == REGIME_INDIVIDUAL
    with pytest.raises(ValueError, match="users"):
        BenchmarkConfig(name="x", n_tasks=1, stream_length=1, n_epilogue_tasks=0, users=0)


def test_collective_exposure_counts_the_team_and_each_user():
    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    sched = build_schedule("arc1-collective", "workstreams", 1, rearc_task_ids=pool, eval_task_ids=evals)
    infos = sched.annotate()
    seen: dict[tuple[int, str], int] = {}
    for slot, info in zip(sched.slots, infos, strict=True):
        key = (slot.user, slot.task_id)
        assert info.user_exposure_index == seen.get(key, 0)  # the user's own earlier meetings
        assert info.exposure_index >= info.user_exposure_index  # the team's rounds include the user's
        seen[key] = seen.get(key, 0) + 1
    # a task is in many users' hands at once, so the team meets it far more often than any one user
    last = [i for s, i in zip(sched.slots, infos, strict=True) if s.phase == "stream"][-100:]
    assert max(i.exposure_index for i in last) > 50 and max(i.user_exposure_index for i in last) <= 7


def test_collective_fresh_reference_wipes_before_every_round():
    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    sched = build_schedule("arc1-collective", "fresh", 0, rearc_task_ids=pool, eval_task_ids=evals)
    assert len(sched) == 2_500 and sched.rounds()[-1] + 1 == 25
    assert [s.reset_before for s in sched.slots] == [i % 100 == 0 for i in range(2_500)]  # once per round


def test_epilogue_for_users_matches_the_full_schedules_own_epilogue():
    from continual_arc.configs import BenchmarkConfig, epilogue_for_users

    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    full = build_schedule("arc1-collective", "workstreams", 3, rearc_task_ids=pool, eval_task_ids=evals)
    k = 10
    epi = epilogue_for_users("arc1-collective", 3, k, rearc_task_ids=pool, eval_task_ids=evals)
    assert epi.users == k and epi.phases() == ["epilogue"]
    assert epi.rounds()[-1] + 1 == 25  # as many rounds as the full schedule's own epilogue
    for u in range(k):
        mine = [(s.task_id, s.instance_seed, s.phase, s.reset_before)
                for s in full.slots if s.phase == "epilogue" and s.user == u]
        theirs = [(s.task_id, s.instance_seed, s.phase, s.reset_before)
                 for s in epi.slots if s.user == u]
        assert mine == theirs, u
    # users must be between 1 and the configuration's own user count
    with pytest.raises(ValueError, match="between 1 and"):
        epilogue_for_users("arc1-collective", 3, 0)
    with pytest.raises(ValueError, match="between 1 and"):
        epilogue_for_users("arc1-collective", 3, 101)
    no_epilogue = BenchmarkConfig(name="x", n_tasks=2, stream_length=4, n_epilogue_tasks=0, users=2)
    with pytest.raises(ValueError, match="has no epilogue"):
        epilogue_for_users(no_epilogue, 0, 1)


def test_protocol_labels_a_cut_stream_and_a_k_user_epilogue():
    from continual_arc.configs import epilogue_for_users, protocol_of

    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    full = build_schedule("arc1-collective", "workstreams", 0, rearc_task_ids=pool, eval_task_ids=evals)

    cut = full.first_rounds(20)
    p = protocol_of(cut)
    assert not p["canonical"] and p["users"] == 100
    assert any("stream cut at round 20" in d for d in p["deviations"])
    assert any("collective" in d for d in p["deviations"])
    assert p["config"] == "arc1-collective"  # meta.config survives the cut, so the label stays readable

    epi = epilogue_for_users("arc1-collective", 0, 10, rearc_task_ids=pool, eval_task_ids=evals)
    p2 = protocol_of(epi)
    assert not p2["canonical"] and p2["users"] == 10 and p2["epilogue"] is True
    assert any("epilogue alone, users 0–9 of 100" in d for d in p2["deviations"])


def test_checkpoints_are_horizons_within_the_stream():
    from continual_arc.configs import BenchmarkConfig

    assert get_config("arc1-collective").checkpoints == (2_000, 6_000, 20_000)
    with pytest.raises(ValueError, match="checkpoints"):
        BenchmarkConfig(name="x", n_tasks=2, stream_length=10, n_epilogue_tasks=0, checkpoints=(4, 20))
    with pytest.raises(ValueError, match="checkpoints"):
        BenchmarkConfig(name="x", n_tasks=2, stream_length=10, n_epilogue_tasks=0, checkpoints=(6, 4))
    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    sched = build_schedule("arc1-collective", "random", 1, rearc_task_ids=pool, eval_task_ids=evals)
    assert list(sched.meta["config"]["checkpoints"]) == [2_000, 6_000, 20_000]


@needs_arc2
def test_arc2_pools_hold_only_tasks_new_to_the_second_edition():
    from continual_arc.configs import epilogue_task_pool, stream_task_pool
    from continual_arc.data import arc1_task_ids, arc2_new_task_ids, arc_task_ids

    old = arc1_task_ids()
    stream = stream_task_pool(get_config("arc2-small"))
    epi = epilogue_task_pool(get_config("arc2-small"))
    assert len(old) == 800
    assert len(arc_task_ids("training", "arc2")) == 1000 and len(arc2_new_task_ids("training")) == 233
    # the epilogue may hold out any of the 114 new tasks; the stream draws only those whose invariance labels
    # allow enough distinct frames for no visit to repeat a grid (26 Sep 2026)
    from continual_arc.tasks import MIN_STREAM_FRAMES, AugmentedArcSource

    assert len(arc_task_ids("evaluation", "arc2")) == 120 and len(epi) == 114 and epi == arc2_new_task_ids("evaluation")
    assert set(stream) < set(epi) and len(stream) == 100
    src = AugmentedArcSource("evaluation", dataset="arc2", invariances="labels")
    assert all(len(src.frames(t)) >= MIN_STREAM_FRAMES for t in stream)
    assert all(len(src.frames(t)) < MIN_STREAM_FRAMES for t in set(epi) - set(stream))
    assert not set(epi) & old
    sched = build_schedule("arc2-small", "workstreams", 0)
    st = {s.task_id for s in sched.slots if s.phase != "epilogue"}
    ep = {s.task_id for s in sched.slots if s.phase == "epilogue"}
    assert len(st) == 25 and len(ep) == 25 and not st & ep and not (st | ep) & old
    assert st <= set(stream) and (st | ep) <= set(arc_task_ids("evaluation", "arc2"))
    assert sched.meta["pools"] == {"stream": 100, "epilogue": 114, "shared": True, "arc2_new_only": True}


def test_every_grid_track_caps_demo_requests_at_one():
    for name in ("arc1-small", "arc1-medium", "arc1-collective", "arc2-smoke", "arc2-small"):
        cfg = get_config(name)
        assert cfg.env.demo_cap == 8 and cfg.env.demos_per_request == 1 and cfg.env.attempt_cap == 8 and cfg.env.max_cost == 16, name
