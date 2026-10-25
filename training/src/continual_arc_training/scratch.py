"""Training once at the end: a DAgger run's phase K (the stream's last, by default) trained once more from the base
model on all the data the stream gathered, beside the continual student's phase K, which started from phase K-1's
adapter, every phase warm-started from the last. Both train on the same label sets of rounds 0..K under the same
held-out games, loss, learning rate, batch and patience, so the comparison asks whether learning over the stream left
the student where training once at its end would: the retrain differs in its start and its stop alone. The stop is the
same held-out loss and patience measured every :data:`EVAL_STEPS` steps instead of 5, with no step cap and at most
:data:`MINUTES` of training, since a fresh adapter needs far more steps than a warm-started phase. The retrain trains
on the aggregate of every round's labels, as the runs' first phases did.

scratch.sh runs it on the run's own node once every run there and their frozen tests are done, then plays the fresh
adapter's two frozen tests from the run's end-of-stream memory, beside the continual student's; before that, the
continual student's weights alone on both tests (frozen.sh with MEMORY=none).

  train    scratch train --run VIEW --phase K --out OUT [--gpus 8]: VIEW is a folder whose play/ is the run's
  summary  scratch summary O: O/summary.json from O/model/DONE, O/continual/DONE and the frozen tests, each compared
           instance by instance: the continual student's in the run's frozen/ against the fresh one's in O/frozen/, and
           the continual student's tests from the end-of-stream memory against its weights alone in O/weights/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# The DAgger runs' trainer settings (the same at 1, 2 and 4 users), but the run, the GPUs and the stop's, which the
# retrain sets itself
TRAINER_ARGV = ["--rounds", "200", "--row", "dagger", "--lag", "1", "--lr", "1e-4", "--batch", "64",
                "--patience", "2", "--hold-every", "10", "--val-max-items", "1000", "--skip-first", "3",
                "--start-after", "24", "--dose", "aggregate", "--replay-ratio", "1"]
# the held-out loss every this many steps (a pass over a run's label sets is 230 to 950 steps of 64)
EVAL_STEPS = 25
# no step cap: the phase stops on the held-out loss, after at most --max-epochs passes or MINUTES
MAX_STEPS = 1_000_000
MINUTES = 720


def trainer_args(run: Path, gpus: int) -> argparse.Namespace:
    """The run's trainer settings as the real-time driver parses them, defaults included, with the retrain's GPUs and
    stop."""
    from continual_arc_training import realtime
    return realtime.parser().parse_args(
        ["train", "--run", str(run), "--train-gpus", ",".join(map(str, range(gpus))), *TRAINER_ARGV, "--gpus",
         str(gpus), "--eval-steps", str(EVAL_STEPS), "--max-steps", str(MAX_STEPS), "--max-minutes", str(MINUTES)])


def cmd_train(o: argparse.Namespace) -> None:
    """Phase ``--phase`` of the run in ``--run`` from the base model into ``--out`` (OUT/adapter, OUT/DONE), as the
    real-time driver trains a phase, from no adapter."""
    from continual_arc_training import phase, realtime
    run, out = Path(o.run), Path(o.out)
    a = trainer_args(run, o.gpus)
    targs = realtime.train_args(a)
    trainer = phase.make_trainer(targs)
    out.mkdir(parents=True, exist_ok=True)
    done = realtime.train_phase(a, run, targs, trainer, o.phase, out, None)
    print(json.dumps({k: done.get(k) for k in ("steps", "minutes", "best_step")}), flush=True)


def held_out(done: dict[str, Any]) -> dict[str, Any]:
    """A phase's held-out loss (its DONE's measurements): where it began, its best, and when."""
    return {"start": done["held_out"][0], "best": min(done["held_out"]), "best_step": done["best_step"],
            "steps": done["steps"], "minutes": done.get("minutes")}


def outcomes(played: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """A play's games by task and instance seed: whether each solved and what it cost."""
    games = {}
    for line in (played / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        games[(r["task_id"], r["instance_seed"])] = {"solved": bool(r["solved"]), "cost": r["cost"]}
    return games


def compare(plays: dict[str, dict[tuple[str, int], dict[str, Any]]]) -> dict[str, Any]:
    """Two plays, by name, on the instances both played: each one's solves and mean cost, and the instances only one of
    them solved (McNemar's discordant counts)."""
    (a, first), (b, second) = plays.items()
    both = sorted(set(first) & set(second))
    x, y = [first[k] for k in both], [second[k] for k in both]
    return {"instances": len(both),
            a: {"solved": sum(g["solved"] for g in x), "mean_cost": round(sum(g["cost"] for g in x) / max(1, len(x)), 3)},
            b: {"solved": sum(g["solved"] for g in y), "mean_cost": round(sum(g["cost"] for g in y) / max(1, len(y)), 3)},
            f"only_{a}": sum(g["solved"] and not h["solved"] for g, h in zip(x, y, strict=True)),
            f"only_{b}": sum(h["solved"] and not g["solved"] for g, h in zip(x, y, strict=True))}


def cmd_summary(o: argparse.Namespace) -> None:
    """O/summary.json: both adapters' held-out loss, their frozen plays of each test compared instance by instance,
    and, under TEST-weights, the continual student's test from its end-of-stream memory beside its weights alone."""
    root = Path(o.dir)
    doc: dict[str, Any] = {"held_out": {"continual": held_out(json.loads((root / "continual" / "DONE").read_text())),
                                        "scratch": held_out(json.loads((root / "model" / "DONE").read_text()))}}
    plays = {**{test: {"continual": root.parent / "frozen" / test / "play", "scratch": root / "frozen" / test / "play"}
                for test in ("held-out", "seen")},
             **{f"{test}-weights": {"with_memory": root.parent / "frozen" / test / "play",
                                    "weights_alone": root / "weights" / f"{test}-weights" / "play"}
                for test in ("held-out", "seen")}}
    for kind, pair in plays.items():
        if all((p / "records.jsonl").exists() for p in pair.values()):
            doc[kind] = compare({name: outcomes(p) for name, p in pair.items()})
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
    s = sub.add_parser("summary", help="O/summary.json from the retrain's and the tests' outputs")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_summary)
    o = p.parse_args()
    o.fn(o)


if __name__ == "__main__":
    main()
