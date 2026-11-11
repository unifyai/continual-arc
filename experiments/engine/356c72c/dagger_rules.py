"""An offline study of the DAgger row's training rule: live runs' saved label sets replayed round by round under a rule,
every phase's adapter scored on later rounds' labels that no rule trains on. Nothing is played or served.

Rules (``--rule``):
  current  realtime.py's DAgger phase exactly (``realtime.train_args``, ``train.fit`` with ``train.Imitation``): phase k
           trains on the training sides of rounds 0..k from phase k-1's best adapter, one pass while no held-out side
           exists, then up to ``--max-epochs`` passes with the held-out loss on every held-out side so far measured
           every ``--eval-steps`` steps and the phase stopped after ``--patience`` measurements without a new best, the
           best adapter kept.
  replay   nothing is trained while the training sides of rounds 0..k hold fewer than ``--warmup-labels`` labels; the
           first phase that reaches it makes one pass over all of them; every later phase makes one pass over round
           k's training side and whole earlier games drawn at random until they hold ``--replay-ratio`` times as many
           labels, from the last phase's adapter (:func:`replay_plan`). The held-out loss is measured, never acted on.
  joint    the reference: nothing until the stream's last phase, which trains the base model on every training label of
           the stream, ``--joint-epochs`` passes with no held-out stop (what one clean training on all of the data
           reaches, the joint-training reference of continual learning).

The stream: ``--runs`` names one or more runs' folders (copied off their nodes); round r of the stream, for r below
``--test-from``, is every run's round r, so several runs replay as one run with all their users, each game kept apart by
its run (:func:`load`). Every round each run has labelled from ``--test-from`` on is the test set.

Scored after every phase (and once for the base model, phase -1): the test set's reply negative log-likelihood per
token, over all labels and by the label's move (``dagger.coarse``: code, request, submit, other), and the held-out
sides' so far. Written to OUT/RULE/results.jsonl,
one line a phase, the adapters to OUT/RULE/pNNNN.

On the GPU node, from ~/prime-rl: ``CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python ~/loop/dagger_rules.py
--runs ~/study/dagger-u1-lag1 ~/study/dagger-u2-lag1 ~/study/dagger-u4-lag1 --test-from 9 --rule replay --out
~/study/out --lr 1e-4 --gpus 4``.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import time
from pathlib import Path
from typing import Any

import dagger


def sides(run: Path, round_: int) -> tuple[list[Path], list[Path]]:
    """A run's round's training side and held-out side, each when it holds a game (``split.json``, written last by the
    labeller, marks the round labelled; a round not yet labelled has neither)."""
    split = run / "play" / "rounds" / f"r{round_:04d}" / "labels" / "split"
    if not (split / "split.json").exists():
        return [], []
    done = json.loads((split / "split.json").read_text())
    train_side, dev_side = f"train-r{round_:04d}", f"dev-r{round_:04d}"
    return ([split / train_side] if done[train_side]["games"] else [],
            [split / dev_side] if done[dev_side]["games"] else [])


def labelled(run: Path, round_: int) -> bool:
    return (run / "play" / "rounds" / f"r{round_:04d}" / "labels" / "split" / "split.json").exists()


def stream(runs: list[Path], test_from: int) -> tuple[list[dict[str, list[Path]]], list[Path]]:
    """The runs pooled: rounds 0..test_from-1, each every run's training and held-out sides (every run must have
    labelled them), and the test set, both sides of every round each run has labelled from ``test_from`` on."""
    rounds = []
    for r in range(test_from):
        short = [run.name for run in runs if not labelled(run, r)]
        if short:
            raise SystemExit(f"round {r} is not labelled in {', '.join(short)}: --test-from at most {r}")
        train_sides, dev_sides = [], []
        for run in runs:
            t, d = sides(run, r)
            train_sides += t
            dev_sides += d
        rounds.append({"train": train_sides, "dev": dev_sides})
    test = []
    for run in runs:
        r = test_from
        while labelled(run, r):
            t, d = sides(run, r)
            test += t + d
            r += 1
    return rounds, test


def load(sides_: list[Path]) -> list[Any]:
    """The label sets' calls (``train.load_calls``), each call's ``run`` its run and side (``dagger-u2-lag1/train-r0005``)
    so that games of different runs with the same user and round stay apart in chains and replay, and its ``move`` the
    label's coarse move (``dagger.coarse``)."""
    import train
    calls = []
    for side in sides_:
        moves = {}
        for f in sorted((side / "captured").glob("u*.json.gz")):
            with gzip.open(f, "rt") as fp:
                doc = json.load(fp)
            for c in doc["calls"]:
                choices = (c.get("response") or {}).get("choices") or []
                if c.get("call_id") and choices:
                    moves[c["call_id"]] = dagger.coarse(choices[0]["message"])
        got, _ = train.load_calls(side)
        name = f"{side.parents[5].name}/{side.name}"
        for call in got:
            call.meta["run"] = name
            call.meta["move"] = moves.get(call.meta.get("call_id"), "other")
        calls += got
    return calls


def replay_plan(new: list[Any], earlier: list[Any], trained_before: bool, warmup: int, ratio: float,
                rng: random.Random) -> list[Any] | None:
    """The replay rule's calls for a phase: None (no training) while ``new`` and ``earlier`` together hold fewer than
    ``warmup`` labels, and :func:`dagger.replay_plan`'s dose after that."""
    if len(new) + len(earlier) < warmup:
        return None
    return dagger.replay_plan(new, earlier, trained_before, ratio, rng)


def score(trainer: Any, items: list[Any], batch: int, max_seq_len: int) -> dict[str, Any]:
    """The items' reply negative log-likelihood per token under the adapter as it is now (``Trainer.nll_chains``, each
    chain read once), over all of them and by the move in each item's label."""
    import train
    pieces = [p for ch in train.find_chains(items) for p in train.chain_pieces(ch, batch, max_seq_len)]
    res = trainer.nll_chains([train.chain_item(p) for p in pieces])["items"]
    flat = [it for p in pieces for it in p]
    tot: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0, 0])
    for it, r in zip(flat, res, strict=True):
        if r is None:
            continue
        for key in ("all", it.label.get("move", "other")):
            tot[key][0] -= r["logp"]
            tot[key][1] += r["tokens"]
            tot[key][2] += 1
    return {k: {"nll": round(v[0] / v[1], 6) if v[1] else None, "tokens": v[1], "labels": v[2]}
            for k, v in sorted(tot.items())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="the runs' folders, their rounds pooled")
    p.add_argument("--test-from", type=int, required=True, dest="test_from",
                   help="the first round scored and never trained on; the rounds before it are the stream")
    p.add_argument("--rule", choices=("current", "replay", "joint"), required=True)
    p.add_argument("--warmup-labels", type=int, default=2000, dest="warmup")
    p.add_argument("--replay-ratio", type=float, default=1.0, dest="ratio")
    p.add_argument("--joint-epochs", type=int, default=2, dest="joint_epochs")
    p.add_argument("--name", help="the rule's folder under OUT (default: the rule's own name and settings)")
    p.add_argument("--out", required=True)
    # realtime.py trainer's own settings, so that the current rule trains exactly as the live runs do
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--lora-rank", type=int, default=128, dest="lora_rank")
    p.add_argument("--max-seq-len", type=int, default=65536, dest="max_seq_len")
    p.add_argument("--eval-steps", type=int, default=5, dest="eval_steps")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=120, dest="max_steps")
    p.add_argument("--max-epochs", type=int, default=20, dest="max_epochs")
    p.add_argument("--val-max-items", type=int, default=1000, dest="val_max_items")
    p.add_argument("--skip-first", type=int, default=3, dest="skip_first")
    p.add_argument("--max-minutes", type=float, default=240, dest="max_minutes")
    a = p.parse_args()
    a.row = "dagger"

    import realtime
    import train
    runs = [Path(r).expanduser() for r in a.runs]
    st, test_sides = stream(runs, a.test_from)
    if not test_sides:
        raise SystemExit(f"no run has labelled round {a.test_from}: no test set")
    name = a.name or {"current": "current", "replay": f"replay-w{a.warmup}-r{a.ratio:g}",
                      "joint": f"joint-e{a.joint_epochs}"}[a.rule]
    root = Path(a.out).expanduser() / name
    root.mkdir(parents=True, exist_ok=True)
    targs = realtime.train_args(a)
    trainer = train.make_trainer(targs)
    tok = trainer.tokenizer
    test_items, _ = train.sft_items(tok, load(test_sides), a.max_seq_len)
    results = root / "results.jsonl"

    def write(line: dict[str, Any]) -> None:
        print(json.dumps(line), flush=True)
        with open(results, "a") as fp:
            fp.write(json.dumps(line) + "\n")

    trainer.reset()
    base_test = score(trainer, test_items, a.batch, a.max_seq_len)
    write({"rule": name, "phase": -1, "trained": False, "test": base_test, "test_from": a.test_from,
           "test_sides": [f"{s.parents[5].name}/{s.name}" for s in test_sides], "runs": [r.name for r in runs]})
    earlier: list[Any] = []
    adapter: Path | None = None
    for k in range(a.test_from):
        new = load(st[k]["train"])
        dev_sides = [s for r in st[: k + 1] for s in r["dev"]]
        out = root / f"p{k:04d}"
        out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(k)
        targs.seed = k
        t0 = time.time()
        held: list[Any] = []
        if a.rule == "current":
            calls = earlier + new
            held = train.held_out_calls([str(s) for s in dev_sides], a.val_max_items, k)
            targs.epochs = a.max_epochs if dev_sides else 1
        elif a.rule == "replay":
            calls = replay_plan(new, earlier, adapter is not None, a.warmup, a.ratio, random.Random(1000 + k))
            targs.epochs = 1
        else:
            calls = earlier + new if k == a.test_from - 1 else None
            targs.epochs = a.joint_epochs
        line: dict[str, Any] = {"rule": name, "phase": k, "labels_so_far": len(earlier) + len(new),
                                "trained": calls is not None}
        if calls is not None:
            done = train.fit(targs, trainer, train.Imitation(targs, tok, calls, held), out, adapter, rng)
            adapter = out / "adapter"
            trainer.load(adapter)   # the phase's kept adapter (the current rule's best), which round k+2 would play on
            line.update(labels_trained=len(calls), steps=done["steps"], best_step=done.get("best_step"),
                        held_out_during=done["held_out"],
                        train_nll=[s["nll_per_token"] for s in json.loads((out / "phase.json").read_text())["steps"]])
        if dev_sides:
            dev_items, _ = train.sft_items(tok, load(dev_sides), a.max_seq_len)
            line["held_out"] = score(trainer, dev_items, a.batch, a.max_seq_len)["all"]
        line["test"] = score(trainer, test_items, a.batch, a.max_seq_len) if calls is not None else base_test
        line["minutes"] = round((time.time() - t0) / 60, 1)
        write(line)
        earlier += new


if __name__ == "__main__":
    main()
