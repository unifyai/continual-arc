"""longtrain.py: a real-time run's last phase trained again from the adapter it ended with, as realtime.py's trainer
trained it but with a budget that lets its held-out measure turn (Experiments Keep Everything: train until it stops
improving; the author, 5 Nov 2026, of DAgger at four users coming in no better than two: whether the fixed per-phase cap and
early stop cut its training short).

Phase K, the run's last (train/latest.json's), is trained exactly as realtime.py's ``train_phase`` trains a DAgger or
OPSD phase: the row's calls of rounds 0..K under the run's dose, its held-out games drawn with seed K up to
``--val-max-items`` calls (the same held-out split the run's phase K measured), the steps planned from
``random.Random(K)``, every setting the phase's own (``train/pKKKK/phase.json``'s ``args``), from the phase's adapter,
which holds the optimizer's moments too (ask-sd's ``Trainer.load``). Only the stop moves: ``--max-steps``,
``--patience`` (measurements in a row without improving) and ``--max-minutes`` replace the phase's, and ``--gpus``
the GPUs it trains on, which changes none of its steps (train.plan). So its first steps are the phase's own, and it
goes on where the phase stopped.

Into OUT/train: fit's phase.json, DONE and adapter (the best held-out measure's, the starting one while none improves
on it), steps.jsonl and held_out.jsonl, every step's record and every held-out measure as phase.json gains them (the
run's own train.py keeps no JSON Lines), and every measurement's adapter in OUT/train/evals/sNNNN/ (PEFT's files alone;
the starting one is the phase's own); fit's step checkpoints let a run stopped by preemption go on from its last one
when started again. OUT/longtrain.json records what it trained from.

Usage (on the node, from ~/prime-rl, CUDA_VISIBLE_DEVICES the GPUs):
  uv run --no-sync python ~/more-plays/longtrain.py --run ~/runs/rt/dagger-u4-disjoint \
      --out ~/runs/rt/dagger-u4-longtrain --gpus 8 --max-steps 1600 --patience 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="the real-time run's folder (realtime.sh's ~/runs/rt/NAME)")
    p.add_argument("--out", required=True, help="this run's folder; the training goes into OUT/train")
    p.add_argument("--gpus", type=int, required=True, help="the GPUs CUDA_VISIBLE_DEVICES gives this process")
    p.add_argument("--max-steps", type=int, required=True, dest="max_steps")
    p.add_argument("--patience", type=int, required=True, help="held-out measurements in a row without improving")
    p.add_argument("--max-minutes", type=float, default=24 * 60, dest="max_minutes")
    # the run's realtime.sh settings that a phase's args do not hold (realtime.py's defaults, which the final runs kept)
    p.add_argument("--row", default="dagger", choices=("dagger", "opsd"))
    p.add_argument("--dose", default="aggregate", choices=("aggregate", "replay", "recent"))
    p.add_argument("--max-epochs", type=int, default=20, dest="max_epochs")
    p.add_argument("--val-max-items", type=int, default=1000, dest="val_max_items")
    p.add_argument("--replay-ratio", type=float, default=1.0, dest="replay_ratio")
    p.add_argument("--first-epochs", type=int, default=2, dest="first_epochs")
    o = p.parse_args()

    import checkpoint
    import kernel_stack
    import realtime
    import train

    checkpoint.catch_sigterm()   # a GPU cloud's notice before it takes a preemptible machine back: a checkpoint, then out
    run, out = Path(o.run).expanduser(), Path(o.out).expanduser()
    latest = realtime.latest(run)
    k = latest["phase"]
    init = realtime.phase_dir(run, k) / "adapter"   # by the run's folder: latest.json names it by its node's absolute path
    phase = json.loads((realtime.phase_dir(run, k) / "phase.json").read_text())
    targs = argparse.Namespace(**{**phase["args"], "gpus": o.gpus, "max_steps": o.max_steps, "patience": o.patience,
                                  "max_minutes": o.max_minutes})
    root = out / "train"
    if (root / "phase.json").exists() and checkpoint.saved_step(root) is None and not (root / "DONE").exists():
        # cut short before its first step checkpoint, so fit trains it again from its first step: what it wrote is set
        # aside whole, its measurements' adapters among them, rather than mixed with the run that replaces it
        root.rename(root.with_name(f"train.stopped-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"))
    root.mkdir(parents=True, exist_ok=True)
    (out / "longtrain.json").write_text(json.dumps(
        {"run": str(run), "phase": k, "init": str(init), "rounds": latest["rounds"],
         "phase_args": phase["args"], "args": vars(targs), "settings": vars(o),
         "started": datetime.now(timezone.utc).isoformat()}, indent=1, default=str))
    trainer = train.make_trainer(targs)
    kernel_stack.record(root)

    # the phase's own setup (realtime.train_phase for DAgger and OPSD), into this run's folder
    rng = random.Random(k)
    targs.seed = k
    per_round, held = realtime.labelled_rounds(run, k)
    calls, targs.epochs, stopped = realtime.dose(o, k, init, per_round, bool(held))
    update = realtime.row_update(o, run, targs, trainer.tokenizer, calls,
                                 train.whole_games(held, o.val_max_items, k) if stopped else [], {})
    measured, stepped = update.held_out, trainer.step
    # The run's train.py writes phase.json whole after every step, before it measures, and keeps no JSON Lines of its
    # own (its steps.jsonl and held_out.jsonl came later), so the steps and measurements are journaled from it here:
    # a step once phase.json holds it, which is before the next step's optimizer step or the measurement after it.
    journaled = {"steps": 0, "best": None}

    def journal_steps() -> int:
        steps = json.loads((root / "phase.json").read_text())["steps"] if (root / "phase.json").exists() else []
        if len(steps) < journaled["steps"]:   # a resumed run's phase.json starts from its checkpoint's
            journaled["steps"] = 0
        with open(root / "steps.jsonl", "a") as fp:
            for record in steps[journaled["steps"]:]:
                fp.write(json.dumps(record) + "\n")
        journaled["steps"] = len(steps)
        return len(steps)

    def step_journaled(*args, **kwargs):
        journal_steps()
        return stepped(*args, **kwargs)
    trainer.step = step_journaled

    def held_out_kept(tr):
        """The held-out measure, journaled, and the adapter it measured kept beside fit's best, PEFT's files alone
        (with the optimizer's moments each would be four times the size); the steps phase.json holds name it, and the
        starting adapter is the phase's own."""
        point = measured(tr)
        n = journal_steps()
        best = journaled["best"] is None or point["loss"] < journaled["best"]
        journaled["best"] = point["loss"] if best else journaled["best"]
        with open(root / "held_out.jsonl", "a") as fp:
            fp.write(json.dumps({"step": n, "loss": round(point["loss"], 6), "best": best}) + "\n")
        dst = root / "evals" / f"s{n:04d}"
        if n and not (dst / "adapter_model.safetensors").exists():
            getattr(tr, "local", tr).model.save_pretrained(str(dst))
        return point
    update.held_out = held_out_kept
    print(json.dumps({"phase": k, "init": str(init), "labels": len(calls), "held_out_games": len(held),
                      "epochs": targs.epochs, "held_out_stop": stopped, "max_steps": o.max_steps,
                      "patience": o.patience}), flush=True)
    done = train.fit(targs, trainer, update, root, init, rng)
    journal_steps()
    (out / "longtrain.json").write_text(json.dumps(
        {**json.loads((out / "longtrain.json").read_text()), "done": done,
         "ended": datetime.now(timezone.utc).isoformat()}, indent=1, default=str))
    print("longtrain done", json.dumps({k: done[k] for k in ("steps", "best_step", "held_out_stop", "time_capped")}),
          flush=True)
    trainer.close() if hasattr(trainer, "close") else None
    sys.exit(0)


if __name__ == "__main__":
    main()
