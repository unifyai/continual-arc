"""Tests for fleet.py: a run's rounds read from its files, the lag-1 pipeline simulated, and plans. Run with
``python3 -m pytest test_fleet.py -q``."""
from __future__ import annotations

import json

import pytest

import fleet

FLAT = {g: 1.0 for g in range(1, 9)}   # a trainer exactly linear in its GPUs


def run(play: float, tail: float, work: float, rounds: int = 60, warm: int = 0) -> fleet.Trace:
    return fleet.Trace("r", [play] * rounds, [tail] * rounds, [0.0] * warm + [work] * (rounds - warm))


@pytest.mark.parametrize("play,tail,train", [(30, 5, 10), (10, 5, 40), (10, 20, 20)])
def test_a_round_takes_the_longest_of_play_training_and_the_two_round_cycle(play, tail, train):
    """Under the one-round lag round r+2 waits for phase r, which waits for round r's labels: a round takes
    max(P, T, (P + L + T) / 2), play-bound, train-bound and cycle-bound in turn here."""
    rounds = 200
    got = fleet.simulate([run(play, tail, train * 4, rounds)], gpus=[4], efficiency=FLAT)[0]["minutes"]
    period = max(play, train, (play + tail + train) / 2)
    assert got / rounds == pytest.approx(period, rel=0.02)


def test_more_trainer_gpus_shorten_a_round_only_while_training_holds_it_up():
    one = fleet.simulate([run(10, 5, 160)], gpus=[2], efficiency=FLAT)[0]["minutes"]   # 80 min phases: train-bound
    two = fleet.simulate([run(10, 5, 160)], gpus=[4], efficiency=FLAT)[0]["minutes"]
    assert two < 0.6 * one
    fast = fleet.simulate([run(30, 0, 8)], gpus=[2], efficiency=FLAT)[0]["minutes"]    # play-bound
    assert fleet.simulate([run(30, 0, 8)], gpus=[8], efficiency=FLAT)[0]["minutes"] == pytest.approx(fast, rel=0.01)


def test_warm_up_phases_take_no_gpu_from_a_pool():
    """A run warming up while another trains leaves the other its whole pool: its phases need no GPU."""
    busy, warming = run(5, 0, 120, rounds=20), run(5, 0, 0.0, rounds=20, warm=20)
    alone = fleet.simulate([busy], pool=6, efficiency=FLAT)[0]["minutes"]
    shared = fleet.simulate([busy, warming], pool=6, efficiency=FLAT)[0]["minutes"]
    assert shared == pytest.approx(alone)


def test_a_pool_lends_an_idle_trainers_gpus_so_two_runs_go_faster_than_on_a_fixed_split():
    a, b = run(20, 25, 80), run(20, 25, 80)
    b.play = [p * 1.3 for p in b.play]   # out of step with a, so their phases mostly come apart
    split = fleet.simulate([a, b], gpus=[3, 3], efficiency=FLAT)
    pooled = fleet.simulate([a, b], pool=6, efficiency=FLAT, restart=1.0)
    assert sum(x["minutes"] for x in pooled) < sum(x["minutes"] for x in split)


def _progress(run, lines):
    run.mkdir(parents=True, exist_ok=True)
    (run / "progress").write_text("\n".join(lines) + "\n")


def test_profile_reads_each_rounds_play_label_tail_and_phase_and_the_players_waits(tmp_path):
    run = tmp_path / "dagger-u4-lag1"
    _progress(run, [
        "10-20 06:24 realtime.sh: started: dagger, 4 users, 200 rounds, lag 1, LR 1e-4, serving on 0,1,2,3, training on 4,5,6,7",
        "10-20 06:24 realtime: play round 0 started (Qwen/Qwen3.5-9B, attempt 1)",
        "10-20 06:29 realtime: play round 0 done (4/4)",
        "10-20 06:29 realtime: play round 1 started (Qwen/Qwen3.5-9B, attempt 1)",
        "10-20 06:35 realtime: play round 1 done (8/8)",
        "10-20 06:35 realtime: play round 2 waits for phase 0",
        "10-20 06:43 realtime: round 0 labelled: {}",
        "10-20 06:43 realtime: phase 0 done: rounds 0-0, not trained: 4 games played, training starts at 4",
        "10-20 06:43 realtime: play round 2 started (dagger-u4-lag1-p0000, attempt 1)",
        "10-20 06:45 realtime: round 1 labelled: {}",
        "10-20 06:45 realtime: phase 1 started: rounds 0-1, 2 labels, 0 held-out games, 1 pass(es)",
        "10-20 06:47 realtime: play round 2 done (12/12)",
        "10-20 07:05 realtime: phase 1 done: rounds 0-1, 10 steps in 20.0 min",
    ])
    (run / "train" / "p0001").mkdir(parents=True)
    (run / "train" / "p0001" / "DONE").write_text(json.dumps({"steps": 10, "minutes": 20.0}))
    (run / "train" / "p0001" / "phase.json").write_text(json.dumps({"args": {"gpus": 4}}))
    p = fleet.profile(run)
    assert (p["row"], p["users"]) == ("dagger", 4)
    r0, r1, r2 = p["rounds"]
    assert (r0["play"], r0["label_tail"], r0["phase"]["trained"]) == (5, 14, False)
    assert (r1["play"], r1["label_tail"], r1["phase"]["minutes"], r1["phase"]["gpus"]) == (6, 10, 20.0, 4)
    assert r1["phase"]["work"] == pytest.approx(20 * 4 * fleet.EFFICIENCY[4])
    assert r2["wait"] == 8 and r2["label_tail"] is None   # round 2 waited for phase 0, and is not labelled yet
    assert p["pace"]["player_waited_hours"] == pytest.approx(8 / 60, abs=0.05)


def test_the_cheapest_partition_puts_runs_together_only_where_a_group_costs_less_than_its_runs_apart():
    cost = {(0,): 10.0, (1,): 10.0, (2,): 10.0, (3,): 10.0, (0, 1): 12.0, (2, 3): None, (0, 2): 15.0, (1, 3): 11.0,
            (0, 3): 25.0, (1, 2): 25.0, (0, 1, 2): 30.0}
    assert fleet.cheapest(4, cost) == (26.0, [(0, 2), (1, 3)])   # 15 + 11, against 12 + 10 + 10 or 40 apart
    assert fleet.cheapest(2, {(0,): None, (1,): 1.0}) is None


HOURS = {(0,): 10.0, (1,): 10.0, (2,): 12.0, (0, 1): 14.0, (0, 2): 20.0, (1, 2): 13.0, (0, 1, 2): 30.0}


def test_the_frontier_runs_from_the_fastest_plan_to_the_cheapest_each_cheaper_and_slower_than_the_one_before():
    """A node is paid until its last run ends: apart, the last run is done at 12 h on 32 node-hours; runs 1 and 2
    together finish at 13 h on 23; no other grouping is both cheaper and as fast as one of those."""
    assert fleet.frontier(3, HOURS) == [(12.0, 32.0, [(0,), (1,), (2,)]), (13.0, 23.0, [(0,), (1, 2)])]
    assert fleet.frontier(2, {(0,): 5.0, (1,): 7.0, (0, 1): 9.0}) == [(7.0, 12.0, [(0,), (1,)]), (9.0, 9.0, [(0, 1)])]


def test_a_deadline_takes_the_cheapest_plan_done_in_time_and_otherwise_every_run_goes_apart_within_any_budget():
    plans = [{"hours": h, "cost": c} for h, c in ((12, 320), (13, 230), (20, 200))]
    got = {"apart": {"hours": 12, "cost": 330}, "plans": plans}
    assert fleet.choose(got) is got["apart"]                       # every run at its fastest, on a node of its own
    assert fleet.choose(got, deadline=15) == plans[1]
    assert fleet.choose(got, deadline=11) is None
    assert fleet.choose(got, budget=400) is got["apart"]
    assert fleet.choose(got, budget=325) == plans[0]              # as soon, sharing nodes to fit the budget
    assert fleet.choose(got, budget=250) == plans[1]
    assert fleet.choose(got, deadline=15, budget=220) is None     # done in time only at 230


def test_a_layouts_gpus_put_the_server_first_then_a_shared_pool_or_each_runs_own():
    assert fleet.gpus_of({"tp": 2, "pool": 6}, 2, 8) == ("0,1", ["2,3,4,5,6,7"] * 2)
    assert fleet.gpus_of({"tp": 1, "gpus": [3, 4]}, 2, 8) == ("0", ["1,2,3", "4,5,6,7"])


def test_a_shared_or_smaller_server_slows_play_by_its_per_call_time_against_the_reference():
    spec = {"serve": {"4": {"1": 4.0, "8": 6.0}, "2": {"1": 5.0, "8": 8.0}}, "reference": {"tp": 4},
            "model_share": 0.5}
    assert fleet.play_factor(spec, 4, 1, 1, 0.0) == 1.0
    assert fleet.play_factor(spec, 2, 1, 1, 0.0) == pytest.approx(1 + 0.5 * (5 / 4 - 1))
    assert fleet.play_factor(spec, 4, 5, 1, 0.0) == pytest.approx(1 + 0.5 * ((4 + 2 * 4 / 7) / 4 - 1))
    assert fleet.per_call(spec, 4, 15) == pytest.approx(6.0 + 2 / 7 * 7)   # past the table: its last slope


def test_an_on_policy_runs_draws_slow_the_other_runs_on_its_server_never_its_own_play():
    p = {"users": 2, "trainer": {"busy_fraction_since_first_trained_phase": 0.5}}
    assert fleet.drawing({"draws": 64}, p, 2) == pytest.approx(0.5 * fleet.DRAW_SHARE[2]) and fleet.drawing({}, p, 2) == 0
    spec = {"serve": {"2": {"1": 5.0, "8": 5.0}, "4": {"1": 5.0, "8": 5.0}}, "reference": {"tp": 4}}
    beside = fleet.play_factor(spec, 2, 3, 1, fleet.drawing({"draws": 64}, p, 2))
    assert beside == pytest.approx(1 + fleet.DRAW_SLOWDOWN[2] * 0.5 * fleet.DRAW_SHARE[2])
