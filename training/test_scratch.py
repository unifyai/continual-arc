"""scratch.py's summary off any GPU: both adapters' held-out loss read from their DONE, and their frozen plays compared
on the instances both played: the validation puzzles, and each frozen test, the continual student's in the run's frozen/
and the fresh adapter's in scratch/frozen/; and the continual student's tests from its end-of-stream memory beside its
weights alone in scratch/weights/. Run with ``python3 -m pytest test_scratch.py -q``.

Its trainer settings come from the run's own node's realtime.py, which this folder's has moved on from (its trainer
takes --train-gpus), so they were checked against the three DAgger nodes' realtime.py instead (docs/method-search.md,
"The from-scratch retrain").
"""
from __future__ import annotations

import json

import scratch


def _done(path, devs, best_step, steps):
    path.mkdir(parents=True)
    (path / "DONE").write_text(json.dumps({"dev": [{"dev_nll": d} for d in devs], "best_step": best_step,
                                           "steps": steps, "minutes": 3.5}))


def _play(path, games):
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text("".join(json.dumps({"task_id": t, "instance_seed": s, "solved": ok, "cost": c,
                                                             "round": 0}) + "\n" for t, s, ok, c in games))


def test_the_summary_compares_both_adapters_on_the_same_held_out_loss_and_instances(tmp_path):
    root = tmp_path / "dagger-u1-lag1" / "scratch"
    _done(root / "continual", [0.66, 0.659], 5, 15)
    _done(root / "model", [1.2, 0.8, 0.62, 0.63], 50, 100)
    _play(root / "plays" / "val-continual", [("a", 1, True, 2), ("b", 2, False, 16), ("c", 3, True, 1)])
    _play(root / "plays" / "val-scratch", [("a", 1, True, 1), ("b", 2, True, 3), ("c", 3, False, 16),
                                           ("d", 4, True, 1)])
    _play(root.parent / "frozen" / "seen" / "play", [("e", 5, True, 1), ("f", 6, False, 16)])
    _play(root / "frozen" / "seen" / "play", [("e", 5, False, 16), ("f", 6, False, 16)])
    _play(root.parent / "frozen" / "held-out" / "play", [("g", 7, True, 1)])
    _play(root / "weights" / "seen-weights" / "play", [("e", 5, False, 16), ("f", 6, True, 3)])
    scratch.cmd_summary(type("O", (), {"dir": str(root)})())
    doc = json.loads((root / "summary.json").read_text())
    assert doc["held_out"] == {"continual": {"start": 0.66, "best": 0.659, "best_step": 5, "steps": 15, "minutes": 3.5},
                               "scratch": {"start": 1.2, "best": 0.62, "best_step": 50, "steps": 100, "minutes": 3.5}}
    # d was played by the fresh adapter alone, so it is left out of the pairs
    assert doc["val"] == {"instances": 3, "continual": {"solved": 2, "mean_cost": 6.333},
                          "scratch": {"solved": 2, "mean_cost": 6.667}, "only_continual": 1, "only_scratch": 1}
    assert doc["seen"] == {"instances": 2, "continual": {"solved": 1, "mean_cost": 8.5},
                           "scratch": {"solved": 0, "mean_cost": 16.0}, "only_continual": 1, "only_scratch": 0}
    assert "held-out" not in doc   # the fresh adapter has not played it yet
    # the continual student with its end-of-stream memory and with its weights alone, on the same instances
    assert doc["seen-weights"] == {"instances": 2, "with_memory": {"solved": 1, "mean_cost": 8.5},
                                   "weights_alone": {"solved": 1, "mean_cost": 9.5}, "only_with_memory": 1,
                                   "only_weights_alone": 1}
    assert "held-out-weights" not in doc   # its weights alone have not played it yet


def test_a_play_is_read_by_task_and_instance_seed(tmp_path):
    _play(tmp_path / "p", [("a", 1, True, 2), ("a", 2, False, 16)])
    assert scratch.outcomes(tmp_path / "p") == {("a", 1): {"solved": True, "cost": 2},
                                                ("a", 2): {"solved": False, "cost": 16}}
