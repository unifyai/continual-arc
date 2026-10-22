from continual_arc.env import EnvConfig, InstanceRecord
from continual_arc.metrics import (
    cost_by_exposure,
    learning_curve,
    retention_by_gap,
    score,
    share_learned,
    summarize,
    transfer,
)
from continual_arc.report import transfer_report


def rec(step, task, phase="stream", exp=0, gap=None, cost=0.0, solved=True, demos=0, seed=None):
    return InstanceRecord(
        step=step, task_id=task, phase=phase, source="fake", exposure_index=exp, gap=gap,
        instance_seed=seed if seed is not None else step, reset_before=False, solved=solved,
        wrong_attempts=int(cost - demos), invalid_submissions=0, demo_requests=demos,
        gave_up=False, cost=cost,
    )


CFG = EnvConfig(attempt_cap=8, demo_cap=8, demo_cost=1.0)


def test_score_bounds():
    assert score([], CFG)["score"] is None
    assert score([rec(0, "a", cost=0)], CFG)["score"] == 100
    assert score([rec(0, "a", cost=16, solved=False)], CFG)["score"] == 0
    s = score([rec(0, "a", cost=4), rec(1, "b", cost=0)], CFG)
    assert s["mean_cost"] == 2 and s["score"] == 87.5 and s["n"] == 2


def test_buckets():
    recs = [
        rec(0, "a", exp=0, cost=3),
        rec(1, "a", exp=1, gap=1, cost=1),
        rec(50, "a", exp=2, gap=49, cost=2),
        rec(400, "a", exp=3, gap=350, cost=5),
        rec(401, "e", phase="epilogue", exp=0, cost=7),
    ]
    ret = {r["gap"]: r for r in retention_by_gap(recs)}
    assert ret["1-10"]["n"] == 1 and ret["1-10"]["mean_cost"] == 1
    assert ret["31-100"]["mean_cost"] == 2
    assert ret["301+"]["mean_cost"] == 5
    assert ret["11-30"]["n"] == 0 and ret["11-30"]["mean_cost"] is None
    exp = {r["exposure"]: r for r in cost_by_exposure(recs)}
    assert exp["0"]["mean_cost"] == 3  # epilogue excluded
    assert exp["3-5"]["mean_cost"] == 5


def test_learning_curve_window():
    recs = [rec(i, "a", cost=float(i)) for i in range(5)]
    lc = learning_curve(recs, window=2)
    assert [p["cumulative_mean_cost"] for p in lc] == [0, 0.5, 1, 1.5, 2]
    assert [p["window_mean_cost"] for p in lc] == [0, 0.5, 1.5, 2.5, 3.5]


def test_summarize_shape():
    s = summarize([rec(0, "a", cost=1), rec(1, "x", phase="epilogue", cost=2)], CFG)
    assert s["stream"]["n"] == 1 and s["epilogue"]["n"] == 1 and s["overall"]["n"] == 2
    assert set(s["by_phase"]) == {"stream", "epilogue"}
    assert s["per_task"]["a"]["first_cost"] == 1


def held(costs, start=0, phase="held-out-test"):
    """One instance of the held-out test per held-out task, ``e0``, ``e1``, ..., unsolved at the cap."""
    return [rec(start + i, f"e{i}", phase=phase, cost=c, solved=c < 16, seed=100 + i) for i, c in enumerate(costs)]


def test_transfer_counts_a_runs_frozen_test_alone_and_takes_a_fresh_runs_epilogue_as_a_reference():
    # the run's epilogue played while still learning never counts; a fresh run's epilogue, wiped before every task,
    # is frozen by construction and serves as the reset's play of the held-out test
    run, reset = held([2, 2, 2, 2]), held([4, 4, 4, 4], phase="epilogue")
    assert transfer(run, reset, CFG, reps=0)["transfer"] == 100 * (4 - 2) / 16
    assert transfer(held([2, 2, 2, 2], phase="epilogue"), reset, CFG, reps=0)["n"] == 0


def test_the_seen_test_reads_new_instances_of_the_streams_tasks_and_clusters_by_task():
    def seen(costs):
        # two instances of each of two stream tasks, so the bootstrap draws tasks, not instances
        return [rec(i, f"t{i // 2}", phase="seen-test", cost=c, solved=c < 16, seed=500 + i) for i, c in enumerate(costs)]
    run, reset = seen([1, 1, 3, 3]), seen([5, 5, 5, 5])
    t = transfer(run, reset, CFG, test="seen", reps=200)
    assert t["test"] == "seen" and t["n"] == 4 and t["tasks"] == 2 and t["transfer"] == 100 * (5 - 2) / 16
    assert transfer_report(t).startswith("Transfer on the seen test")
    assert t["ci95"] == [100 * 2 / 16, 100 * 4 / 16]   # a draw takes each task's two instances together
    assert transfer(run, reset, CFG, reps=0)["n"] == 0   # none of them is a held-out instance


def test_transfer_reads_the_learner_reset_alone():
    # the bare model is no reference, so a reset that fails every task at the cap sets T on its own; a verified run
    # flags such a reset against the bare model's play of the same instances (docs/HELD-OUT.md)
    run, reset = held([2, 2, 2, 2]), held([16, 16, 16, 16])
    t = transfer(run, reset, CFG, reps=0)
    assert t["n"] == 4 and t["tasks"] == 4 and t["transfer"] == 100 * (16 - 2) / 16
    assert t["solved"] == {"run": 1.0, "reset": 0.0}


def test_transfer_matches_instances_and_bootstraps_over_tasks():
    run = held([1, 1, 1, 1, 1, 1])
    reset = held([3, 3, 3, 3, 3]) + [rec(50, "stream-task", cost=9)]  # one held-out instance short, and a stream slot
    t = transfer(run, reset, CFG, reps=200)
    assert t["n"] == 5 and t["transfer"] == 100 * 2 / 16
    assert t["ci95"][0] == t["ci95"][1] == t["transfer"]  # every task transfers alike, so every draw agrees
    assert transfer(run, [], CFG)["transfer"] is None


def test_checkpoint_scores_read_the_stream_prefix():
    from continual_arc.metrics import checkpoint_scores

    recs = [rec(i, "a", cost=4.0 if i < 4 else 0.0) for i in range(8)] + [rec(8, "x", phase="epilogue", cost=8)]
    rows = checkpoint_scores(recs, CFG, (2, 4, 8, 12))
    assert [r["instances"] for r in rows] == [2, 4, 8, 12]
    assert [r["reached"] for r in rows] == [True, True, True, False]
    assert rows[0]["mean_cost"] == 4.0 and rows[2]["mean_cost"] == 2.0 and rows[3]["n"] == 8  # the epilogue is not counted
    assert summarize(recs, CFG, checkpoints=(4,))["checkpoints"][0]["score"] == rows[1]["score"]
    assert summarize(recs, CFG)["checkpoints"] == []


def _stream(costs_by_task):
    """Records of a stream visiting each task in turn: task t's k-th visit costs costs_by_task[t][k]."""
    out, step = [], 0
    visits = max(len(c) for c in costs_by_task.values())
    for k in range(visits):
        for t, costs in costs_by_task.items():
            if k < len(costs):
                out.append(rec(step, t, exp=k, gap=None if k == 0 else 2, cost=float(costs[k]), seed=100 + step))
                step += 1
    return out


RESET = {"a": [8, 4, 4, 4], "b": [6, 6, 2, 2], "c": [4, 4, 4, 4]}


def test_share_learned_is_zero_for_the_reset_and_one_for_perfect_memory():
    reset = _stream(RESET)
    same = share_learned(reset, reset, CFG, reps=50)
    assert same["share_learned"] == 0 and same["lift"] == 0 and same["same_instances"] == len(reset)
    perfect = _stream({t: [c[0]] + [0] * (len(c) - 1) for t, c in RESET.items()})
    full = share_learned(perfect, reset, CFG, reps=50)
    assert abs(full["share_learned"] - 1) < 1e-12
    # the learnable budget is the reset's charge on returns: (12 + 10 + 12) of the 12 instances, over the cap
    assert abs(full["learnable_budget"] - 100 * 34 / 16 / 12) < 1e-12
    assert full["ci95"][0] <= 1 <= full["ci95"][1] + 1e-12


def test_share_learned_reads_halfway_and_worse_than_the_reset():
    reset = _stream(RESET)
    half = _stream({t: [c[0]] + [x / 2 for x in c[1:]] for t, c in RESET.items()})
    assert abs(share_learned(half, reset, CFG, reps=0)["share_learned"] - 0.5) < 1e-12
    worse = _stream({t: [c + 2 for c in cs] for t, cs in RESET.items()})
    assert share_learned(worse, reset, CFG, reps=0)["share_learned"] < 0


def test_share_learned_needs_room_to_learn():
    saturated = _stream({"a": [5, 0, 0], "b": [3, 0, 0]})
    assert share_learned(saturated, saturated, CFG, reps=50)["share_learned"] is None


def test_share_learned_reads_an_instance_the_reset_never_met_against_its_mean_on_the_task():
    reset = _stream(RESET)
    # other draws of the same tasks, as another user's instances are: each is read against the reset's mean on its task
    other = [InstanceRecord(**{**r.to_json(), "instance_seed": r.instance_seed + 1000}) for r in reset]
    o = share_learned(other, reset, CFG, reps=0)
    assert o["same_instances"] == 0 and o["n"] == len(reset)
    assert abs(o["lift"]) < 1e-12 and abs(o["share_learned"]) < 1e-12   # the same costs as the reset's, task by task
    # a task the reset has not met is left out
    assert share_learned(other + [rec(99, "z", cost=16.0, seed=999)], reset, CFG, reps=0)["n"] == len(reset)


def test_share_learned_reads_a_run_in_progress_on_the_slots_it_reached_and_a_reset_in_progress_by_its_means():
    reset = _stream(RESET)
    partial = share_learned(reset[:6], reset, CFG, reps=0)
    assert partial["n"] == 6 and partial["share_learned"] == 0
    ahead = share_learned(reset, reset[:6], CFG, reps=0)   # the reset has played each task's first two visits
    assert ahead["n"] == 12 and ahead["same_instances"] == 6
