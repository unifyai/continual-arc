import pytest

from continual_arc.configs import CONFIGS, FRESH, ORDERS, build_schedule, get_config, select_tasks

POOL = [f"r{i:03d}" for i in range(50)]
EVAL = [f"e{i:03d}" for i in range(40)]


@pytest.mark.parametrize("order", (*ORDERS, FRESH))
def test_build_schedule_orders(order):
    sched = build_schedule("arc1-smoke", 3, order=order, rearc_task_ids=POOL, eval_task_ids=EVAL)
    cfg = CONFIGS["arc1-smoke"]
    stream = [s for s in sched.slots if s.phase != "epilogue"]
    epi = [s for s in sched.slots if s.phase == "epilogue"]
    assert len(epi) == cfg.n_epilogue_tasks
    assert all(sched.sources[s.task_id] == "arc_eval" for s in epi)
    if order == "fresh":
        # the transfer reference: the held-out tasks alone, a fresh learner before each
        assert not stream and all(s.reset_before for s in epi)
    else:
        assert len({s.task_id for s in stream}) == cfg.n_tasks
        assert all(sched.sources[s.task_id] == "rearc" for s in stream)
        assert not any(s.reset_before for s in sched.slots)  # an order arranges instances, never resets the learner
        if order == "sequential":
            assert len(stream) == cfg.n_tasks * cfg.instances_per_task == cfg.stream_length
        else:
            assert len(stream) == cfg.stream_length
    assert sched.meta["mode"] == order and sched.meta["config"]["name"] == "arc1-smoke"


def test_same_task_set_across_orders():
    sets = []
    for order in (*ORDERS, FRESH):
        sched = build_schedule("arc1-smoke", 11, order=order, rearc_task_ids=POOL, eval_task_ids=EVAL)
        sets.append(frozenset(s.task_id for s in sched.slots if s.phase != "epilogue"))
        sets.append(frozenset(s.task_id for s in sched.slots if s.phase == "epilogue"))
    assert sets[0] == sets[2] == sets[4]
    assert sets[1] == sets[3] == sets[5]


def test_no_epilogue():
    sched = build_schedule("arc1-smoke", 0, order="random", rearc_task_ids=POOL, eval_task_ids=EVAL, epilogue=False)
    assert sched.phases() == ["stream"]


def test_select_tasks_deterministic_and_bounded():
    assert select_tasks(POOL, 5, 1, "x") == select_tasks(POOL, 5, 1, "x")
    assert select_tasks(POOL, 5, 1, "x") != select_tasks(POOL, 5, 2, "x")
    with pytest.raises(ValueError):
        select_tasks(POOL, 51, 0, "x")


def test_unknown_config_and_order():
    with pytest.raises(KeyError):
        get_config("huge")
    with pytest.raises(ValueError):
        build_schedule("arc1-smoke", 0, order="sideways", rearc_task_ids=POOL, eval_task_ids=EVAL)


def test_named_configs_arrivals_fit_in_stream():
    for cfg in CONFIGS.values():
        expected_arrival_span = cfg.n_tasks * cfg.workstreams.arrival_interval
        assert expected_arrival_span <= 0.75 * cfg.stream_length, cfg.name


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


def test_canonical_protocol_per_track():
    from continual_arc.configs import (
        CANONICAL_SEEDS,
        build_schedule,
        canonical_protocol,
        protocol_of,
    )
    from continual_arc.env import EnvConfig

    for track in ("arc1",):
        proto = canonical_protocol(track)
        assert proto["config"] == f"{track}-small" and proto["mode"] == "workstreams"
        assert proto["tests"] == ["held-out", "seen"]
        assert proto["seeds"] == list(CANONICAL_SEEDS)
    pool = [f"t{i:03d}" for i in range(40)]
    evals = [f"e{i:03d}" for i in range(120)]
    sched = build_schedule("arc1-small", 0, rearc_task_ids=pool, eval_task_ids=evals)
    assert sched.meta["track"] == "arc1"
    p = protocol_of(sched, EnvConfig())
    assert p["canonical"] and p["deviations"] == []
    p = protocol_of(sched, EnvConfig(demo_cost=0.5))
    assert not p["canonical"] and any("demo_cost" in d for d in p["deviations"])
    sched = build_schedule("arc1-small", 7, order="sequential", rearc_task_ids=pool, eval_task_ids=evals, epilogue=False)
    p = protocol_of(sched)
    assert not p["canonical"]
    assert any("sequential" in d for d in p["deviations"]) and any("seed 7" in d for d in p["deviations"])
    # the frozen tests replaced the epilogue, so a stream without one is the canonical protocol
    sched = build_schedule("arc1-small", 0, rearc_task_ids=pool, eval_task_ids=evals, epilogue=False)
    p = protocol_of(sched, EnvConfig())
    assert p["canonical"] and p["deviations"] == [] and not p["epilogue"]


def test_a_configuration_served_to_several_users_gives_each_a_stream_of_their_own():
    from collections import Counter

    from continual_arc.configs import (
        CANONICAL_USERS,
        REGIME_COLLECTIVE,
        REGIME_INDIVIDUAL,
        BenchmarkConfig,
        canonical_protocol,
        protocol_of,
        with_users,
    )

    assert CANONICAL_USERS == (1, 2, 4, 8, 16) and canonical_protocol("arc1")["users"] == [1, 2, 4, 8, 16]
    assert all(c.users == 1 and c.regime == REGIME_INDIVIDUAL for c in CONFIGS.values())   # users are the run's
    cfg = with_users("arc1-small", 16)
    assert cfg.name == "arc1-small" and cfg.users == 16 and cfg.regime == REGIME_COLLECTIVE
    assert cfg.instances_per_task == 8
    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    sched = build_schedule(cfg, 0, rearc_task_ids=pool, eval_task_ids=evals)
    assert sched.users == 16 and sched.meta["config"]["users"] == 16 and len(sched.meta["users"]["seeds"]) == 16
    rounds = sched.rounds()
    assert rounds[-1] + 1 == 200 + 100  # 200 stream rounds, then every user's 100 held-out tasks
    assert [s.user for s in sched.slots] == [i % 16 for i in range(len(sched))]  # a round is every user, in order
    sched.check_users()
    stream = [s for s in sched.slots if s.phase != "epilogue"]
    assert len(stream) == 3_200 and len(sched) == 4_800
    assert set(Counter(s.task_id for s in stream).values()) == {128}  # every task 128 times over the team
    assert set(Counter((s.user, s.task_id) for s in stream).values()) == {8}  # and 8 times per user
    # the same tasks as one user alone, stream and epilogue alike
    small = build_schedule("arc1-small", 0, rearc_task_ids=pool, eval_task_ids=evals)
    for phase in ("stream", "epilogue"):
        assert {s.task_id for s in sched.slots if s.phase == phase} == {s.task_id for s in small.slots if s.phase == phase}
    # user 0 walks exactly one user's stream and epilogue at the same seed
    mine = [(s.task_id, s.instance_seed, s.phase, s.reset_before) for s in sched.slots if s.user == 0]
    assert mine == [(s.task_id, s.instance_seed, s.phase, s.reset_before) for s in small.slots]
    # the others walk their own draws of the same process: different orders, different instances
    one = [s for s in sched.slots if s.user == 1]
    assert [s.task_id for s in one] != [s.task_id for s in sched.slots if s.user == 0]
    assert not {s.instance_seed for s in one} & {s.instance_seed for s in sched.slots if s.user == 0}
    epi_orders = {tuple(s.task_id for s in sched.slots if s.user == u and s.phase == "epilogue") for u in range(16)}
    assert len(epi_orders) == 16  # a held-out task reaches the team gradually, in each user's own order
    # fewer users are more users cut to their first ones: user u's stream does not depend on how many there are
    four = build_schedule(with_users("arc1-small", 4), 0, rearc_task_ids=pool, eval_task_ids=evals)
    for u in range(4):
        assert ([(s.task_id, s.instance_seed) for s in four.slots if s.user == u]
                == [(s.task_id, s.instance_seed) for s in sched.slots if s.user == u])
    p = protocol_of(sched)
    assert p["regime"] == REGIME_COLLECTIVE and p["users"] == 16 and p["canonical"] and p["deviations"] == []
    ten = protocol_of(build_schedule(with_users("arc1-small", 10), 0, rearc_task_ids=pool,
                                     eval_task_ids=evals))
    assert not ten["canonical"] and any("10 users is not one of the canonical" in d for d in ten["deviations"])
    with pytest.raises(ValueError, match="users"):
        BenchmarkConfig(name="x", n_tasks=1, stream_length=1, n_epilogue_tasks=0, users=0)


def test_collective_exposure_counts_the_team_and_each_user():
    from continual_arc.configs import with_users

    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    sched = build_schedule(with_users("arc1-small", 16), 1, rearc_task_ids=pool, eval_task_ids=evals)
    infos = sched.annotate()
    seen: dict[tuple[int, str], int] = {}
    for slot, info in zip(sched.slots, infos, strict=True):
        key = (slot.user, slot.task_id)
        assert info.user_exposure_index == seen.get(key, 0)  # the user's own earlier meetings
        assert info.exposure_index >= info.user_exposure_index  # the team's rounds include the user's
        seen[key] = seen.get(key, 0) + 1
    # a task is in several users' hands at once, so the team meets it more often than any one user
    last = [i for s, i in zip(sched.slots, infos, strict=True) if s.phase == "stream"][-16:]
    assert max(i.exposure_index for i in last) > 2 * max(i.user_exposure_index for i in last)


def test_collective_fresh_reference_wipes_before_every_round():
    from continual_arc.configs import with_users

    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    sched = build_schedule(with_users("arc1-small", 16), 0, order="fresh", rearc_task_ids=pool, eval_task_ids=evals)
    assert len(sched) == 1_600 and sched.rounds()[-1] + 1 == 100
    assert [s.reset_before for s in sched.slots] == [i % 16 == 0 for i in range(1_600)]  # once per round


def test_epilogue_for_users_matches_the_full_schedules_own_epilogue():
    from continual_arc.configs import BenchmarkConfig, epilogue_for_users, with_users

    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    cfg = with_users("arc1-small", 16)
    full = build_schedule(cfg, 3, rearc_task_ids=pool, eval_task_ids=evals)
    k = 4
    epi = epilogue_for_users(cfg, 3, k, rearc_task_ids=pool, eval_task_ids=evals)
    assert epi.users == k and epi.phases() == ["epilogue"]
    assert epi.rounds()[-1] + 1 == 100  # as many rounds as the full schedule's own epilogue
    for u in range(k):
        mine = [(s.task_id, s.instance_seed, s.phase, s.reset_before)
                for s in full.slots if s.phase == "epilogue" and s.user == u]
        theirs = [(s.task_id, s.instance_seed, s.phase, s.reset_before)
                 for s in epi.slots if s.user == u]
        assert mine == theirs, u
    # users must be between 1 and the number the configuration is served to
    with pytest.raises(ValueError, match="between 1 and"):
        epilogue_for_users(cfg, 3, 0)
    with pytest.raises(ValueError, match="between 1 and"):
        epilogue_for_users(cfg, 3, 17)
    no_epilogue = BenchmarkConfig(name="x", n_tasks=2, stream_length=4, n_epilogue_tasks=0, users=2)
    with pytest.raises(ValueError, match="has no epilogue"):
        epilogue_for_users(no_epilogue, 0, 1)


def test_protocol_labels_a_cut_stream_and_a_k_user_epilogue():
    from continual_arc.configs import epilogue_for_users, protocol_of, with_users

    pool = [f"t{i:03d}" for i in range(500)]
    evals = [f"e{i:03d}" for i in range(500)]
    cfg = with_users("arc1-small", 16)
    full = build_schedule(cfg, 0, rearc_task_ids=pool, eval_task_ids=evals)

    cut = full.first_rounds(20)
    p = protocol_of(cut)
    assert not p["canonical"] and p["users"] == 16
    assert p["deviations"] == ["stream cut at round 20"]
    assert p["config"] == "arc1-small"  # meta.config survives the cut, so the label stays readable

    epi = epilogue_for_users(cfg, 0, 4, rearc_task_ids=pool, eval_task_ids=evals)
    p2 = protocol_of(epi)
    assert not p2["canonical"] and p2["users"] == 4 and p2["epilogue"] is True
    assert any("epilogue alone, users 0–3 of 16" in d for d in p2["deviations"])


def test_checkpoints_are_horizons_within_the_stream_and_scale_with_the_users():
    from continual_arc.configs import BenchmarkConfig, with_users

    with pytest.raises(ValueError, match="checkpoints"):
        BenchmarkConfig(name="x", n_tasks=2, stream_length=10, n_epilogue_tasks=0, checkpoints=(4, 20))
    with pytest.raises(ValueError, match="checkpoints"):
        BenchmarkConfig(name="x", n_tasks=2, stream_length=10, n_epilogue_tasks=0, checkpoints=(6, 4))
    one = BenchmarkConfig(name="x", n_tasks=2, stream_length=10, n_epilogue_tasks=0, checkpoints=(4, 10))
    assert with_users(one, 8).checkpoints == (32, 80) and with_users(with_users(one, 8), 2).checkpoints == (8, 20)


def test_every_configuration_caps_demo_requests_at_one():
    for name in ("arc1-small", "arc1-medium"):
        cfg = get_config(name)
        assert cfg.env.demo_cap == 8 and cfg.env.demos_per_request == 1 and cfg.env.attempt_cap == 8 and cfg.env.max_cost == 16, name


def test_the_held_out_test_is_user_0s_epilogue_played_at_once_from_the_end_state():
    from continual_arc.configs import epilogue_for_users, protocol_of, test_schedule, with_users
    pool, evals = [f"t{i:03d}" for i in range(40)], [f"e{i:03d}" for i in range(120)]
    test = test_schedule(with_users("arc1-small", 4), 0, "held-out", rearc_task_ids=pool, eval_task_ids=evals)
    epilogue = epilogue_for_users("arc1-small", 0, 1, rearc_task_ids=pool, eval_task_ids=evals)
    assert [(s.task_id, s.instance_seed) for s in test.slots] == [(s.task_id, s.instance_seed) for s in epilogue.slots]
    # one round: every instance a user of its own, each starting from its own copy of the end state
    assert test.users == len(test.slots) == 100 and [s.user for s in test.slots] == list(range(100))
    assert {s.phase for s in test.slots} == {"held-out-test"} and not any(s.reset_before for s in test.slots)
    assert test.meta["test"] == "held-out" and protocol_of(test)["canonical"]


def test_the_seen_test_draws_new_instances_of_the_streams_tasks():
    from continual_arc.configs import (
        SEEN_TEST_PER_TASK,
        config_tasks,
        get_config,
        test_schedule,
        with_users,
    )
    pool, evals = [f"t{i:03d}" for i in range(40)], [f"e{i:03d}" for i in range(120)]
    test = test_schedule("arc1-small", 0, "seen", rearc_task_ids=pool, eval_task_ids=evals)
    tasks = config_tasks(get_config("arc1-small"), pool, 0)
    assert sorted(s.task_id for s in test.slots) == sorted(tasks * SEEN_TEST_PER_TASK) and len(test.slots) == 100
    assert {s.phase for s in test.slots} == {"seen-test"} and set(test.sources.values()) == {"rearc"}
    stream = build_schedule(with_users("arc1-small", 16), 0, rearc_task_ids=pool, eval_task_ids=evals)
    seeds = [s.instance_seed for s in test.slots]
    assert len(set(seeds)) == 100 and not set(seeds) & {s.instance_seed for s in stream.slots}


def test_a_narrower_test_restores_the_end_state_before_every_instance():
    from continual_arc.configs import test_schedule
    pool, evals = [f"t{i:03d}" for i in range(40)], [f"e{i:03d}" for i in range(120)]
    wide = test_schedule("arc1-small", 0, "seen", rearc_task_ids=pool, eval_task_ids=evals)
    narrow = test_schedule("arc1-small", 0, "seen", width=25, rearc_task_ids=pool, eval_task_ids=evals)
    assert narrow.users == 25 and [s.user for s in narrow.slots] == [i % 25 for i in range(100)]
    assert all(s.reset_before for s in narrow.slots)   # each user plays four, every one from the end state
    assert [(s.task_id, s.instance_seed) for s in narrow.slots] == [(s.task_id, s.instance_seed) for s in wide.slots]
    with pytest.raises(ValueError, match="width"):
        test_schedule("arc1-small", 0, "seen", width=30, rearc_task_ids=pool, eval_task_ids=evals)
    with pytest.raises(ValueError, match="test must be"):
        test_schedule("arc1-small", 0, "epilogue", rearc_task_ids=pool, eval_task_ids=evals)
