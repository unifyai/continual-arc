from continual_arc.env import EnvConfig, InstanceRecord
from continual_arc.metrics import (
    cost_by_exposure,
    forward_transfer,
    learning_curve,
    retention_by_gap,
    score,
    summarize,
)


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


def test_forward_transfer():
    run_recs = [rec(9, "e1", phase="epilogue", cost=1, seed=100), rec(10, "e2", phase="epilogue", cost=2, seed=101)]
    ref_recs = [rec(9, "e1", phase="epilogue", cost=4, seed=100), rec(10, "e2", phase="epilogue", cost=2, seed=101),
                rec(11, "e3", phase="epilogue", cost=9, seed=102)]
    t = forward_transfer(run_recs, ref_recs, CFG)
    assert t["n_matched"] == 2
    assert t["mean_transfer"] == 1.5
    assert t["transfer_score_points"] == 100 * 1.5 / 16
    assert t["per_task"]["e1"]["transfer"] == 3


def test_checkpoint_scores_read_the_stream_prefix():
    from continual_arc.metrics import checkpoint_scores

    recs = [rec(i, "a", cost=4.0 if i < 4 else 0.0) for i in range(8)] + [rec(8, "x", phase="epilogue", cost=8)]
    rows = checkpoint_scores(recs, CFG, (2, 4, 8, 12))
    assert [r["instances"] for r in rows] == [2, 4, 8, 12]
    assert [r["reached"] for r in rows] == [True, True, True, False]
    assert rows[0]["mean_cost"] == 4.0 and rows[2]["mean_cost"] == 2.0 and rows[3]["n"] == 8  # the epilogue is not counted
    assert summarize(recs, CFG, checkpoints=(4,))["checkpoints"][0]["score"] == rows[1]["score"]
    assert summarize(recs, CFG)["checkpoints"] == []
