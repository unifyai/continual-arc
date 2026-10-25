"""A training phase, the same for every trained row but for its update (:mod:`continual_arc_training.updates`):
:func:`fit`, its steps planned before the first (:func:`plan`), the held-out stop, the step checkpoints, and the
phase's record on disk; and the trainer it runs on (:func:`make_trainer`)."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from continual_arc_training import checkpoints
from continual_arc_training.chains import TokenItem, chain_batches, find_chains

# torch's compiled kernels (FlexAttention's, for every shape a step's chains bring) kept on the boot disk, which a
# preempted machine keeps, not in torch's default /tmp, which its restart clears: a trainer whose cache was cleared
# spent 203, 106 and 59 s on its first three steps compiling them again, against 25, 25 and 18 s from this cache
INDUCTOR_CACHE = Path.home() / ".cache" / "torchinductor"


def make_trainer(a: argparse.Namespace) -> Any:
    """The trainer the settings ask for: data parallel over ``a.gpus`` GPUs (the trainer alone in this process on one),
    its compiled kernels cached in :data:`INDUCTOR_CACHE` (the data-parallel workers take this process's environment)."""
    from continual_arc_training.parallel import DataParallelTrainer

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(INDUCTOR_CACHE))
    return DataParallelTrainer(a.model, world_size=a.gpus, device="cuda", rank=a.rank, alpha=a.alpha, lr=a.lr,
                               max_seq_len=a.max_seq_len, lora_targets=tuple(a.lora_targets))


def plan(units: list[TokenItem], a: argparse.Namespace, rng: random.Random) -> list[tuple[int, list[list[TokenItem]]]]:
    """A phase's steps, every pass planned before the first step: ``a.epochs`` passes over ``units``, each pass whole
    chains of a game's calls (:func:`continual_arc_training.chains.find_chains`) in an order of its own, cut into pieces
    each GPU can take within the window, in steps of ``a.batch`` calls, at most ``a.max_steps`` in all; each step its
    pass (from 1) and its pieces. Every update plans so, so the same units give the same steps whatever the update draws
    from ``rng`` after."""
    chained = find_chains(units, a.chain_slack)
    steps: list[tuple[int, list[list[TokenItem]]]] = []
    for p in range(1, a.epochs + 1):
        steps += [(p, step) for step in chain_batches(chained, a.batch, a.chain_calls, a.max_seq_len, rng,
                                                      slack=a.chain_slack)]
    return steps[: a.max_steps or len(steps)]


def fit(a: argparse.Namespace, trainer: Any, update: Any, out: Path, init: Path | None,
        rng: random.Random) -> dict[str, Any]:
    """A phase of training from ``init`` (the base model when None) into ``out``: the steps :func:`plan` draws over the
    update's units, each the update's gradient (:meth:`Update.step`) and one optimizer step. With held-out units the
    update's held-out loss is measured before the first step, every ``a.eval_steps`` steps and after the last;
    ``out/adapter`` is the adapter at the best of them (the starting one while none improves on it), and training stops
    once ``a.patience`` in a row have not improved on the best. Without, ``out/adapter`` is the adapter as training left
    it. ``a.max_minutes`` stops it too. Then ``out/phase.json`` (the update's counts, every step's record, every
    held-out measurement, the settings) and ``out/DONE``, whose summary is returned. ``trainer`` is the caller's, so one
    trainer trains phase after phase, each from the adapter it is given.

    The best adapter is kept as a checkpoint (:mod:`continual_arc_training.checkpoints`), and so is the phase on
    SIGTERM and every ``checkpoints.EVERY`` steps: a phase cut short goes on from its last one, its steps planned again
    from ``rng``'s state at the start, as they were (each step's record keeps its batch's digest, ``batch``, which the
    plan must match), and the steps already trained skipped. The summary names the steps the phase resumed from
    (``resumed``)."""
    if init:
        trainer.load(init)
    else:
        trainer.reset()
    # the loop's state, all of it in the checkpoint: the steps taken, the held-out curve, its best and patience count,
    # the update's counts, and whether a stop or the time cap has ended training
    run: dict[str, Any] = {"steps": [], "evals": [], "best": None, "since_best": 0, "stats": update.stats,
                           "capped": False, "stopped": False}
    ck = checkpoints.Checkpoints(out, trainer, rng, checkpoints.fingerprint(a, init, update.units, update.held_items),
                                 lambda: run, extra=update.keep)
    steps = plan(update.units, a, rng)
    saved = ck.load()
    if saved and not checkpoints.replays([step for _, step in steps], saved["loop"]["steps"]):
        ck.discard("its batches do not replay")
        saved = None
    if saved:
        run.update(saved["loop"])
        ck.resume(saved)
    records, evals = run["steps"], run["evals"]
    update.stats = run["stats"]
    update.start(trainer, [step for _, step in steps], out, len(records),
                 ck.extra_dir(saved, "drawer") if saved else None, lambda: ck.step(len(records)))
    print(json.dumps({"objective": update.objective, update.key: update.stats, "held_out_items": update.held_stats,
                      "steps": len(steps)}), flush=True)

    def write_phase() -> None:
        (out / "phase.json").write_text(json.dumps({"objective": update.objective, update.key: update.stats,
                                                    "held_out_items": update.held_stats, "steps": records,
                                                    "held_out": evals, "args": vars(a)}, indent=1, default=str))

    def measure(final: bool = False) -> bool:
        """The held-out loss now, into the measurements, and a checkpoint after it when the loss is the best so far, to
        hold the best adapter, or one is due where the phase goes on; whether ``a.patience`` measurements in a row have
        now not improved on the best."""
        n = len(records)
        point = {"step": n, **update.held_out(trainer)}
        point["best"] = run["best"] is None or point["loss"] < run["best"]
        if point["best"]:
            run["best"], run["since_best"] = point["loss"], 0
        else:
            run["since_best"] += 1
        point["loss"] = round(point["loss"], 6)
        evals.append(point)
        print(json.dumps({"held_out": point}), flush=True)
        write_phase()
        stop = bool(a.patience) and run["since_best"] >= a.patience
        if not n:
            ck.start_is_best(init)   # before the first step: the best is the adapter the phase starts from
        elif point["best"] or (ck.due(n) and not (final or stop or run["capped"] or n == len(steps))):
            ck.save(n, best=point["best"])
        return stop

    def after_step() -> None:
        """The held-out measurement or the checkpoint a step's end is due, once (a resume ends where one was due)."""
        n = len(records)
        if update.holds_out and n % a.eval_steps == 0:
            if evals[-1]["step"] != n and measure():
                run["stopped"] = True
                print(f"held-out stop at step {n}: {a.patience} measurements in a row above {run['best']:.6f}",
                      flush=True)
        elif ck.due(n) and n < len(steps):
            ck.save(n)

    if not saved and update.holds_out:
        measure()
    elif saved and records and not (run["stopped"] or run["capped"]):
        after_step()
    for k in range(len(records), len(steps)):
        if run["stopped"] or run["capped"]:
            break
        ck.step(k)   # SIGTERM in the last step's measurement: the process's end
        t = time.time()
        record = {"step": k + 1, "pass": steps[k][0], **update.step(trainer, k, rng)}
        taken = trainer.step()
        record.update(grad_norm=round(taken["grad_norm"], 5) if "grad_norm" in taken else None,
                      train_s=round(time.time() - t, 1), batch=checkpoints.batch_digest(steps[k][1]))
        records.append(record)
        write_phase()
        if ck.minutes() > a.max_minutes:
            print(f"time cap {a.max_minutes} min at step {k + 1}", flush=True)
            run["capped"] = True
        ck.step(k + 1)   # SIGTERM: a checkpoint, and the process's end
        if not run["capped"]:
            after_step()
        if not (run["stopped"] or run["capped"]) and k + 1 < len(steps):
            record.update(update.stepped(trainer, k + 1))
        print(json.dumps(record), flush=True)
    update.end()
    if update.holds_out and evals[-1]["step"] != len(records):   # the steps since the last measurement, measured once
        measure(final=True)
    write_phase()
    ck.write_adapter(out / "adapter", init)   # the best one's, or with nothing held out the adapter as training left it
    update.close()
    best_step = max((e["step"] for e in evals if e["best"]), default=None)
    summary = {"objective": update.objective, "steps": len(records), "time_capped": run["capped"],
               "held_out_stop": run["stopped"], update.key: update.stats, "held_out": [e["loss"] for e in evals],
               "best_step": best_step, "last": records[-1] if records else None, **ck.record()}
    (out / "DONE").write_text(json.dumps(summary))
    ck.clear()
    print("done", json.dumps(summary), flush=True)
    return summary
