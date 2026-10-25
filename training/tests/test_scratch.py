"""Training once at the end (continual_arc_training.scratch) off any GPU: the retrain trains the run's phase as the
real-time driver trains a phase, from no adapter, and the summary reads both adapters' held-out loss from their DONE
and compares their frozen plays on the instances both played: each frozen test, the continual student's in the run's
frozen/ and the fresh adapter's in scratch/frozen/, and the continual student's tests from its end-of-stream memory
beside its weights alone in scratch/weights/."""
from __future__ import annotations

import json
import types

from continual_arc_training import phase, realtime, scratch


def _done(path, held_out, best_step, steps):
    path.mkdir(parents=True)
    (path / "DONE").write_text(json.dumps({"held_out": held_out, "best_step": best_step, "steps": steps,
                                           "minutes": 3.5}))


def _play(path, games):
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text("".join(json.dumps({"task_id": t, "instance_seed": s, "solved": ok, "cost": c,
                                                             "round": 0}) + "\n" for t, s, ok, c in games))


def test_the_summary_compares_both_adapters_on_the_same_held_out_loss_and_instances(tmp_path):
    root = tmp_path / "dagger-u1-lag1" / "scratch"
    _done(root / "continual", [0.66, 0.659], 5, 15)
    _done(root / "model", [1.2, 0.8, 0.62, 0.63], 50, 100)
    _play(root.parent / "frozen" / "seen" / "play", [("e", 5, True, 1), ("f", 6, False, 16)])
    _play(root / "frozen" / "seen" / "play", [("e", 5, False, 16), ("f", 6, False, 16), ("g", 9, True, 1)])
    _play(root.parent / "frozen" / "held-out" / "play", [("g", 7, True, 1)])
    _play(root / "weights" / "seen-weights" / "play", [("e", 5, False, 16), ("f", 6, True, 3)])
    scratch.cmd_summary(types.SimpleNamespace(dir=str(root)))
    doc = json.loads((root / "summary.json").read_text())
    assert doc["held_out"] == {"continual": {"start": 0.66, "best": 0.659, "best_step": 5, "steps": 15, "minutes": 3.5},
                               "scratch": {"start": 1.2, "best": 0.62, "best_step": 50, "steps": 100, "minutes": 3.5}}
    # g was played by the fresh adapter alone, so it is left out of the pairs
    assert doc["seen"] == {"instances": 2, "continual": {"solved": 1, "mean_cost": 8.5},
                           "scratch": {"solved": 0, "mean_cost": 16.0}, "only_continual": 1, "only_scratch": 0}
    assert "held-out" not in doc   # the fresh adapter has not played it yet
    # the continual student with its end-of-stream memory and with its weights alone, on the same instances
    assert doc["seen-weights"] == {"instances": 2, "with_memory": {"solved": 1, "mean_cost": 8.5},
                                   "weights_alone": {"solved": 1, "mean_cost": 9.5}, "only_with_memory": 1,
                                   "only_weights_alone": 1}
    assert "held-out-weights" not in doc


def test_a_play_is_read_by_task_and_instance_seed(tmp_path):
    _play(tmp_path / "p", [("a", 1, True, 2), ("a", 2, False, 16)])
    assert scratch.outcomes(tmp_path / "p") == {("a", 1): {"solved": True, "cost": 2},
                                                ("a", 2): {"solved": False, "cost": 16}}


def test_the_retrain_trains_the_runs_phase_from_no_adapter_with_its_own_stop(tmp_path, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(phase, "make_trainer", lambda targs: seen.setdefault("trainer", ("the trainer", targs.gpus)))

    def train_phase(a, run, targs, trainer, k, out, init):
        seen.update(run=run, k=k, out=out, init=init, trainer_given=trainer,
                    stop=(targs.eval_steps, targs.max_steps, targs.max_minutes, targs.patience, targs.lr))
        return {"steps": 300, "minutes": 100.0, "best_step": 275, "held_out": [1.0]}

    monkeypatch.setattr(realtime, "train_phase", train_phase)
    scratch.cmd_train(types.SimpleNamespace(run=str(tmp_path / "view"), phase=199, out=str(tmp_path / "model"),
                                            gpus=8))
    assert (seen["run"], seen["k"], seen["out"], seen["init"]) == (tmp_path / "view", 199, tmp_path / "model", None)
    assert seen["trainer_given"] == ("the trainer", 8) and (tmp_path / "model").is_dir()
    assert seen["stop"] == (scratch.EVAL_STEPS, scratch.MAX_STEPS, scratch.MINUTES, 2, 1e-4)
    assert json.loads(capsys.readouterr().out) == {"steps": 300, "minutes": 100.0, "best_step": 275}
