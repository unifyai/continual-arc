"""frontier_check.py: whether DAgger's held-out stop judges its phases on the wrong set, and which training mix lowers
the loss on the states the student meets next (21 Oct 2026, method-search.md "Why DAgger's phases stop"). Two in three
of the final DAgger runs' phases keep the adapter they started from: the held-out loss rises about half a percent in
the first five steps, and the stop (--patience 2, every 5 steps) ends the phase at step 10. The held-out set is one game
in ten of every round so far, and every phase trains on every round's labels so far, so both are mostly rounds the
adapter has long fitted, and the newest round is a few percent of either.

So a stalled phase is played on past its stop, from the adapter it started from, and at each step of --measure the
adapter's loss is measured on three sets:
  held_out  the run's own held-out set, as the stop reads it (train.held_out_calls of the phase's held-out sides,
            --val-max-items, the phase's seed)
  frontier  every label of the rounds after the phase, which the adapter has not trained on: the states the student
            meets next (whole games drawn up to CAPS)
  recent    the training side of the last rounds, from --since to the phase's own
on one of three training mixes:
  aggregate every round's labels so far, as the run trains (realtime.py --dose aggregate); in the phase's own order
            (--order the phase), its first measurements are the run's own, a check that this plays the phase as the run
            did
  recent    the last rounds' labels, from --since on, and earlier games' labels drawn at random until there are as many
            (dagger.replay_plan's draw at --replay-ratio 1, the mix --dose replay takes with one round)
  uniform   whole games drawn at random from every round's labels until there are as many as the recent mix holds:
            the recent mix's size without its weight on the last rounds, so that what the recent mix gains for being
            small (more passes over less) is told from what it gains for being recent
each in passes as needed to fill --steps, in the batch order --order seeds.

It runs against the code of the run it reads (the final runs' ~/loop of 19-20 Oct, whose train_sft it plays step for
step), from a copy of that ~/loop holding this file in search/: on a node, cd ~/prime-rl && CUDA_VISIBLE_DEVICES=0,1,2,3
uv run --no-sync python COPY/search/frontier_check.py RUN --phase K --since S --init ADAPTER --mix aggregate --out OUT.
RUN holds the run's play/rounds/rNNNN/labels/split of every round labelled so far; ADAPTER is the one phase K started
from (with its optimizer); OUT/<--name>.jsonl gets the sets' sizes, then a line per step trained and per measurement.
--dry builds the sets on the CPU, with the model's tokenizer, and stops before the trainer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dagger  # noqa: E402
import realtime  # noqa: E402
import train  # noqa: E402

# the trainer's settings as the final DAgger runs' trainers have them: their command line (realtime.py trainer --row
# dagger --lr 1e-4 --batch 64 --gpus 4 --eval-steps 5 --patience 2 --max-steps 120 --val-max-items 1000) and its defaults
RUN_ARGS = dict(row="dagger", lr=1e-4, batch=64, gpus=4, lora_rank=128, max_seq_len=65536, max_minutes=240,
                max_steps=120, eval_steps=5, patience=2, val_max_items=1000)
CAPS = {"frontier": 1000, "recent": 500}     # whole games drawn up to this many calls, as the held-out set is


def round_of(side: str) -> int:
    """A label side's round, from its folder's name (realtime.sides: train-rNNNN, dev-rNNNN)."""
    return int(Path(side).name.rsplit("-r", 1)[1])


def labelled(run: Path) -> int:
    """How many rounds from round 0 on have their labels split."""
    n = 0
    while (realtime.split_dir(run, n) / "split.json").exists():
        n += 1
    return n


def calls_of(sides: list[str]) -> list[Any]:
    return [call for side in sides for call in train.load_calls(Path(side))[0]]


def drawn(games: list[list[Any]], want: int, seed: int) -> list[Any]:
    """Whole games drawn at random until their calls reach ``want``."""
    games = games[:]
    random.Random(seed).shuffle(games)
    picked: list[Any] = []
    for game in games:
        if len(picked) >= want:
            break
        picked += game
    return picked


def chains_of(tok: Any, calls: list[Any], a: argparse.Namespace) -> tuple[list[Any], int]:
    """A set's calls as train_sft reads its held-out set: the items, chained and cut into pieces; and their count."""
    items, _ = train.sft_items(tok, calls, a.max_seq_len)
    return ([train.chain_item(piece, a.chain_slack) for chain in train.find_chains(items, a.chain_slack)
             for piece in train.chain_pieces(chain, a.batch, a.max_seq_len, a.chain_slack)], len(items))


def nll(trainer: Any, chains: list[Any]) -> float:
    """The set's replies' mean negative log-likelihood per token under the adapter as it is now (train_sft's dev_nll)."""
    res = [x for x in trainer.nll_chains(chains)["items"] if x]
    return -sum(x["logp"] for x in res) / max(1, sum(x["tokens"] for x in res))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="a folder holding the run's play/rounds/rNNNN/labels/split")
    p.add_argument("--phase", type=int, required=True, help="the stalled phase K, which trains rounds 0..K")
    p.add_argument("--since", type=int, required=True, help="the first of the last rounds (the recent mix and set)")
    p.add_argument("--init", required=True, help="the adapter phase K started from, with its optimizer")
    p.add_argument("--mix", choices=("aggregate", "recent", "uniform"), required=True)
    p.add_argument("--order", type=int, default=None, help="the batch order's seed (default the phase's own)")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--measure", default="0,5,10,15,20,30,40,50,60", help="the steps after which to measure")
    p.add_argument("--name", default=None, help="the output's name (default the mix)")
    p.add_argument("--out", required=True)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    run, k = Path(a.run).expanduser(), a.phase
    order = k if a.order is None else a.order
    at = {int(s) for s in a.measure.split(",")}
    targs = realtime.train_args(argparse.Namespace(**RUN_ARGS))
    targs.seed = k
    rng = random.Random(order)   # train_phase's random.Random(k) for the phase's own order
    train_sides, dev_sides = realtime.phase_runs(run, k + 1)
    per_round = {r: train.load_calls(Path(side))[0] for r, side in train_sides.items()}
    recent = [c for r in sorted(per_round) if r >= a.since for c in per_round[r]]
    earlier = [c for r in sorted(per_round) if r < a.since for c in per_round[r]]
    recent_mix = dagger.replay_plan(recent, earlier, True, 1.0, random.Random(1000 + k))
    if a.mix == "aggregate":
        calls = [c for r in sorted(per_round) for c in per_round[r]]
    elif a.mix == "recent":
        calls = recent_mix
    else:
        calls = drawn(train.by_game([c for r in sorted(per_round) for c in per_round[r]]), len(recent_mix), 2000 + k)
    upto = labelled(run)
    later_train, later_dev = realtime.phase_runs(run, upto)
    frontier = calls_of([side for r, side in later_train.items() if r > k] + [s for s in later_dev if round_of(s) > k])
    sets = {"held_out": train.held_out_calls(dev_sides, targs.val_max_items, k),
            "frontier": train.whole_games(train.by_game(frontier), CAPS["frontier"], k),
            "recent": train.whole_games(train.by_game(recent), CAPS["recent"], k)}
    if a.dry:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(targs.model)
    else:
        trainer = train.make_trainer(targs)
        trainer.load(a.init)
        tok = trainer.tokenizer
    measured = {name: chains_of(tok, got, targs) for name, got in sets.items()}
    items, istats = train.sft_items(tok, calls, targs.max_seq_len)
    chains = train.find_chains(items, targs.chain_slack)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = (out / f"{a.name or a.mix}.jsonl").open("a")

    def write(line: dict[str, Any]) -> None:
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(json.dumps(line), flush=True)

    write({"run": run.name, "mix": a.mix, "phase": k, "since": a.since, "order": order, "init": a.init,
           "labels": len(calls), "items": istats, "chains": len(chains), "frontier_rounds": list(range(k + 1, upto)),
           "sets": {name: {"calls": len(sets[name]), "items": n} for name, (_, n) in measured.items()},
           "settings": {key: getattr(targs, key) for key in ("lr", "batch", "gpus", "rank", "max_seq_len")}})
    if a.dry:
        return
    t0 = time.time()

    def measure(step: int) -> None:
        write({"mix": a.mix, "step": step, "minutes": round((time.time() - t0) / 60, 1),
               **{name: round(nll(trainer, got), 6) for name, (got, _) in measured.items()}})

    measure(0)
    steps = 0
    while steps < a.steps:
        for batch in train.chain_batches(chains, targs.batch, targs.gpus, 1, targs.max_seq_len, rng,
                                         slack=targs.chain_slack):
            res = trainer.sft_chains([train.chain_item(piece, targs.chain_slack) for piece in batch], rng=rng)
            trainer.step()
            steps += 1
            loss = res["nll_per_token"]
            write({"mix": a.mix, "trained": steps, "nll_per_token": None if loss is None else round(loss, 6)})
            if steps in at:
                measure(steps)
            if steps >= a.steps:
                break


if __name__ == "__main__":
    main()
