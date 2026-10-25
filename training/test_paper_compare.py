"""paper_compare.py on hand-made records: the per-round and per-visit shares of a visit's budget, and the report's
tables. Run with ``python3 -m pytest test_paper_compare.py -q``."""
from __future__ import annotations

import json

import paper_compare


def rec(round_, user, k, cost, solved=True, phase="stream", task="t0", seed=0):
    return {"round": round_, "user": user, "user_exposure_index": k, "cost": cost, "solved": solved, "phase": phase,
            "episode_type": "benchmark", "task_id": task, "instance_seed": seed}


def test_shares_by_round_and_by_visit(tmp_path):
    recs = [rec(0, 0, 0, 8.0), rec(0, 1, 0, 16.0, solved=False), rec(1, 0, 1, 4.0), rec(1, 1, 0, 8.0),
            rec(1, 2, 0, 0.0, phase="epilogue")]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n{\"round\": 2, \"cost\"")   # a partial last line
    stream = paper_compare.stream_records(path)
    assert len(stream) == 4   # the epilogue and the partial line left out
    rounds = paper_compare.by_round(stream, 16.0)
    assert [r["instances"] for r in rounds] == [2, 2] and rounds[0]["solved"] == 50 and rounds[0]["spent"] == 75
    assert rounds[1]["first"] == 50 and rounds[1]["returning"] == 25 and rounds[1]["returning_n"] == 1
    visits = paper_compare.by_visit(stream, 16.0)
    assert visits[0] == (200 / 3, 3) and visits[1] == (25, 1) and visits[2] == (None, 0)
    pooled = {"program library": [rec(0, 0, 0, 4.0), rec(1, 0, 1, 1.6)]}
    ws0 = {"program library": [rec(0, 0, 0, 2.0), rec(1, 0, 1, 1.6, task="t1"),
                               rec(2, 0, 0, 16.0, solved=False, task="t2")]}
    text = paper_compare.report(stream, pooled, ws0, 16.0, "DAgger", "medium")
    assert "| 0 | 2 | 50% | 75 | 75 | - (0) |" in text and "| 2 | 25 (1) | 10 |" in text
    assert "| 1 | 67 (3) | 25 |" in text and "| 1 | 67 (3) | 56 |" in text   # pooled, then ws0 alone
    assert text.endswith("| 38 (2/2) | 12 (1/1) |")   # user 0 played t0 (seed 0) twice; the baseline played it once


def test_same_instances_matches_user_0s_instances_to_the_baselines_runs_of_them():
    ours = [rec(0, 0, 0, 8.0, task="a", seed=1), rec(0, 1, 0, 4.0, task="b", seed=9), rec(1, 0, 1, 16.0, solved=False,
                                                                                         task="a", seed=2)]
    theirs = {"program library": [rec(0, 0, 0, 2.0, task="a", seed=1), rec(1, 0, 1, 1.0, task="a", seed=2),
                                  rec(2, 0, 0, 4.0, task="c", seed=3)]}
    same = paper_compare.same_instances(ours, theirs, 16.0)
    assert same["row"] == (75.0, 1, 2)   # user 1's instance is another stream's
    assert same["program library"] == (9.375, 2, 2)
