"""longtrain.py: a real-time run's last phase trained again, as realtime.py's trainer trained it but with a budget that
lets its held-out measure turn (Experiments Keep Everything: train until it stops improving; the author, 5 Nov 2026, of DAgger
at four users coming in no better than two: whether the fixed per-phase cap and early stop cut its training short). It
starts from the adapter the run ended with or, with --from-base, from the base model, so that of the stream only its
labels are kept: one long fit over all of them, shuffled, with no phases, warm starts or rollbacks. With --fresh-lora it
starts from the function the run's adapter computes but trains a fresh LoRA of the same rank beside it, the run's adapter
frozen, with a fresh optimizer and a linear warm-up of --warmup steps (fresh_lora.py), so that what the run's adapter
held besides its function, its weights' norms and directions and its optimizer's moments, is gone.

Phase K, the run's last (train/latest.json's), is trained exactly as realtime.py's ``train_phase`` trains a DAgger or
OPSD phase: the row's calls of rounds 0..K under the run's dose, its held-out games drawn with seed K up to
``--val-max-items`` calls (the same held-out split the run's phase K measured), the steps planned from
``random.Random(K)``, every setting the phase's own (``train/pKKKK/phase.json``'s ``args``), from the phase's adapter,
which holds the optimizer's moments too (ask-sd's ``Trainer.load``), or from the base model. Only the start and the
stop move: ``--max-steps``, ``--patience`` (measurements in a row without improving) and ``--max-minutes`` replace the
phase's, ``--eval-steps`` how often it is measured (the phase's own unless given), and ``--gpus`` the GPUs it trains on,
which changes none of its steps (train.plan). So from the adapter its first steps are the phase's own, and it goes on
where the phase stopped.

Into OUT/train: fit's phase.json, DONE and adapter (the best held-out measure's, the starting one while none improves
on it), steps.jsonl and held_out.jsonl, every step's record and every held-out measure as phase.json gains them (the
run's own train.py keeps no JSON Lines), and every measurement's adapter in OUT/train/evals/sNNNN/ (PEFT's files alone;
the starting one, if any, is the phase's own); fit's step checkpoints let a run stopped by preemption go on from its last one
when started again. OUT/longtrain.json records what it trained from.

Usage (on the node, from ~/prime-rl, CUDA_VISIBLE_DEVICES the GPUs):
  uv run --no-sync python ~/more-plays/longtrain.py --run ~/runs/rt/dagger-u4-disjoint \
      --out ~/runs/rt/dagger-u4-longtrain --gpus 8 --max-steps 1600 --patience 20
With --fresh-lora the trainer is built at twice the phase's rank and alpha (fresh_lora.py), and what it saves, its
measurements' adapters among them, is the composed adapter, which vLLM serves with --max-lora-rank twice the phase's.
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
    start = p.add_mutually_exclusive_group()
    start.add_argument("--from-base", action="store_true", dest="from_base",
                       help="train from the base model rather than from the adapter the run ended with")
    start.add_argument("--fresh-lora", action="store_true", dest="fresh_lora",
                       help="train a fresh LoRA of the phase's rank beside the run's adapter, frozen (fresh_lora.py)")
    p.add_argument("--warmup", type=int, default=50,
                   help="with --fresh-lora, the steps of the fresh optimizer's linear warm-up to the phase's rate")
    p.add_argument("--eval-steps", type=int, dest="eval_steps", help="steps between held-out measures, the phase's own unless given")
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
    # by the run's folder: latest.json names the adapter by its node's absolute path
    adapter = realtime.phase_dir(run, k) / "adapter"
    # a fresh LoRA starts as fit starts one from the base model (Trainer.reset), the run's adapter composed into it
    init = None if o.from_base or o.fresh_lora else adapter
    phase = json.loads((realtime.phase_dir(run, k) / "phase.json").read_text())
    wide = {"rank": 2 * phase["args"]["rank"], "alpha": 2 * phase["args"]["alpha"]} if o.fresh_lora else {}
    targs = argparse.Namespace(**{**phase["args"], "gpus": o.gpus, "max_steps": o.max_steps, "patience": o.patience,
                                  "max_minutes": o.max_minutes, "eval_steps": o.eval_steps or phase["args"]["eval_steps"],
                                  **wide})
    root = out / "train"
    if (root / "phase.json").exists() and checkpoint.saved_step(root) is None and not (root / "DONE").exists():
        # cut short before its first step checkpoint, so fit trains it again from its first step: what it wrote is set
        # aside whole, its measurements' adapters among them, rather than mixed with the run that replaces it
        root.rename(root.with_name(f"train.stopped-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"))
    root.mkdir(parents=True, exist_ok=True)
    (out / "longtrain.json").write_text(json.dumps(
        {"run": str(run), "phase": k, "rounds": latest["rounds"],
         "init": f"a fresh LoRA beside {adapter}, frozen" if o.fresh_lora else str(init or "the base model"),
         "phase_args": phase["args"], "args": vars(targs), "settings": vars(o),
         "started": datetime.now(timezone.utc).isoformat()}, indent=1, default=str))
    trainer = train.make_trainer(targs)
    composed = None
    if o.fresh_lora:
        import fresh_lora
        composed = fresh_lora.compose_all(trainer, adapter, phase["args"]["rank"])
        print(json.dumps({"fresh_lora": composed}), flush=True)
    kernel_stack.record(root)

    # the phase's own setup (realtime.train_phase for DAgger and OPSD), into this run's folder
    rng = random.Random(k)
    targs.seed = k
    per_round, held = realtime.labelled_rounds(run, k)
    calls, targs.epochs, stopped = realtime.dose(o, k, init, per_round, bool(held))
    update = realtime.row_update(o, run, targs, trainer.tokenizer, calls,
                                 train.whole_games(held, o.val_max_items, k) if stopped else [], {})
    measured, stepped = update.held_out, trainer.step
    if o.fresh_lora:
        # the rate each step takes, set before its gradient by fit's own step count, which a resumed run keeps
        accumulate = update.step

        def warmed(tr, k, generator):
            tr.set_lr(fresh_lora.warmup_lr(targs.lr, k, o.warmup))
            return accumulate(tr, k, generator)
        update.step = warmed
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
    print(json.dumps({"phase": k, "init": "fresh LoRA" if o.fresh_lora else str(init or "the base model"),
                      "labels": len(calls), "held_out_games": len(held),
                      "epochs": targs.epochs, "held_out_stop": stopped, "max_steps": o.max_steps,
                      "patience": o.patience}), flush=True)
    done = train.fit(targs, trainer, update, root, init, rng)
    journal_steps()
    (out / "longtrain.json").write_text(json.dumps(
        {**json.loads((out / "longtrain.json").read_text()), "done": done, "fresh_lora": composed,
         "ended": datetime.now(timezone.utc).isoformat()}, indent=1, default=str))
    print("longtrain done", json.dumps({k: done[k] for k in ("steps", "best_step", "held_out_stop", "time_capped")}),
          flush=True)
    trainer.close() if hasattr(trainer, "close") else None
    sys.exit(0)


if __name__ == "__main__":
    main()
