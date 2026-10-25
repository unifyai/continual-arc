"""The from-scratch retrain at a real-time run's stream end: DAgger's phase K trained from the base model on all the
training data the stream accumulated, beside the continual student's phase K, which started from phase K-1's adapter,
every phase warm-started from the last. Both train on the same label sets of rounds 0..K under the same held-out games,
so the comparison asks whether the stream's early phases, each a few steps on little data, leave the continual student
worse off than a clean start on everything (docs/method-search.md, "The from-scratch retrain").

It runs on the run's own node once every run there is done (scratch.sh), with the node's own code on the path: its
realtime.py and train.py as the run trained, its settings read through the node's own realtime.py, so the retrain
differs from phase K in its start and its stop alone. That code is the paper's DAgger runs' own, older than this
folder's realtime.py and train.py in two ways, which TRAINER_ARGV and :func:`held_out` follow: its trainer takes no
``--train-gpus``, and its phases' DONE keeps the held-out curve as ``dev``. With this folder's code on the path, the
trainer needs ``--train-gpus`` too and train.fit's DONE keeps the curve as ``held_out``. The start is the base model.
The stop is the same held-out loss and patience, measured every EVAL_STEPS steps instead of 5, with no step cap and at
most MINUTES of training: a fresh adapter needs many more steps than a warm-started phase, which mostly kept its 5th or
10th.

  train    scratch.py train --run VIEW --phase K --out OUT [--gpus 8]: VIEW is a folder whose play/ is the run's
  summary  scratch.py summary O: O/summary.json from O/model/DONE, O/continual/DONE, the validation plays in O/plays/,
           and each frozen test (frozen.py) played by both, the continual student's in the run's frozen/ and the fresh
           one's in O/frozen/; and the continual student's tests from the end-of-stream memory beside its weights alone
           from an empty one, in O/weights/ (scratch.sh's weights stage: what the weights carry without the notes)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The paper's DAgger runs' trainer arguments, the same for dagger-u1, -u2 and -u4 (read off their nodes' realtime.py
# trainer processes), but --run, --gpus and the stop's settings, which the retrain sets itself
TRAINER_ARGV = ["--rounds", "200", "--row", "dagger", "--lag", "1", "--lr", "1e-4", "--batch", "64",
                "--patience", "2", "--hold-every", "10", "--val-max-items", "1000", "--skip-first", "3",
                "--start-after", "24", "--dose", "aggregate", "--replay-ratio", "1", "--first-epochs", "2"]
# the held-out loss every this many steps (a pass over the final runs' label sets is 230 to 950 steps of 64)
EVAL_STEPS = 25
# no step cap: train.train_sft stops on the held-out loss, after at most --max-epochs passes or MINUTES
MAX_STEPS = 1_000_000
MINUTES = 720


def trainer_args(run: Path, gpus: int) -> argparse.Namespace:
    """The run's trainer settings as the node's own realtime.py parses them, defaults included, with the retrain's
    GPUs and stop: ``main`` binds each subcommand when it runs, so the trainer's is caught in place of being run."""
    import realtime
    seen: list[argparse.Namespace] = []
    realtime.cmd_trainer = seen.append
    argv = sys.argv
    sys.argv = ["realtime.py", "trainer", "--run", str(run), *TRAINER_ARGV, "--gpus", str(gpus),
                "--eval-steps", str(EVAL_STEPS), "--max-steps", str(MAX_STEPS), "--max-minutes", str(MINUTES)]
    try:
        realtime.main()
    finally:
        sys.argv = argv
    (a,) = seen
    return a


def cmd_train(o: argparse.Namespace) -> None:
    """Phase ``--phase`` of the run in ``--run`` from the base model into ``--out`` (OUT/adapter, OUT/DONE), as the
    node's realtime.py trains a phase (``train_phase``), ``init`` None."""
    import realtime
    import train
    run, out = Path(o.run), Path(o.out)
    a = trainer_args(run, o.gpus)
    targs = realtime.train_args(a)
    trainer = train.make_trainer(targs)
    out.mkdir(parents=True, exist_ok=True)
    done = realtime.train_phase(a, run, targs, trainer, o.phase, out, None)
    print(json.dumps({k: done.get(k) for k in ("steps", "minutes", "best_step")}), flush=True)


def held_out(done: dict[str, Any]) -> dict[str, Any]:
    """A DAgger phase's held-out loss (its DONE's ``dev`` measurements): where it began, its best, and when."""
    dev = [e["dev_nll"] for e in done["dev"]]
    return {"start": dev[0], "best": min(dev), "best_step": done["best_step"], "steps": done["steps"],
            "minutes": done.get("minutes")}


def outcomes(play: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """A play's games by task and instance seed: whether each solved and what it cost."""
    games = {}
    for line in (play / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        games[(r["task_id"], r["instance_seed"])] = {"solved": bool(r["solved"]), "cost": r["cost"]}
    return games


def compare(plays: dict[str, dict[tuple[str, int], dict[str, Any]]]) -> dict[str, Any]:
    """Two plays, by name, on the instances both played: each one's solves and mean cost, and the pairs (McNemar's
    discordant counts are what a one-run difference rests on)."""
    (a, first), (b, second) = plays.items()
    both = sorted(set(first) & set(second))
    x, y = [first[k] for k in both], [second[k] for k in both]
    return {"instances": len(both),
            a: {"solved": sum(g["solved"] for g in x), "mean_cost": round(sum(g["cost"] for g in x) / max(1, len(x)), 3)},
            b: {"solved": sum(g["solved"] for g in y), "mean_cost": round(sum(g["cost"] for g in y) / max(1, len(y)), 3)},
            f"only_{a}": sum(g["solved"] and not h["solved"] for g, h in zip(x, y)),
            f"only_{b}": sum(h["solved"] and not g["solved"] for g, h in zip(x, y))}


def cmd_summary(o: argparse.Namespace) -> None:
    """O/summary.json: both adapters' held-out loss, and their frozen plays of the 100 validation puzzles and of
    each frozen test, compared instance by instance; and, under TEST-weights, the continual student's test from its
    end-of-stream memory beside its weights alone."""
    root = Path(o.dir)
    doc = {"held_out": {"continual": held_out(json.loads((root / "continual" / "DONE").read_text())),
                        "scratch": held_out(json.loads((root / "model" / "DONE").read_text()))}}
    plays = {"val": {"continual": root / "plays" / "val-continual", "scratch": root / "plays" / "val-scratch"},
             **{test: {"continual": root.parent / "frozen" / test / "play", "scratch": root / "frozen" / test / "play"}
                for test in ("held-out", "seen")},
             **{f"{test}-weights": {"with_memory": root.parent / "frozen" / test / "play",
                                    "weights_alone": root / "weights" / f"{test}-weights" / "play"}
                for test in ("held-out", "seen")}}
    for kind, pair in plays.items():
        if all((play / "records.jsonl").exists() for play in pair.values()):
            doc[kind] = compare({name: outcomes(play) for name, play in pair.items()})
    (root / "summary.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="the run's phase K from the base model")
    t.add_argument("--run", required=True, help="a folder whose play/ is the run's (scratch.sh's VIEW)")
    t.add_argument("--phase", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--gpus", type=int, default=8)
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("summary", help="O/summary.json from the retrain's and the plays' outputs")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_summary)
    o = p.parse_args()
    o.fn(o)


if __name__ == "__main__":
    main()
