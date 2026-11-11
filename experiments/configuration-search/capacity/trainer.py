"""trainer.py RUN --root ROOT: a capacity run's continual trainer (README.md), on 8 GPUs (Modal's, modal_capacity.py), its
phases taken up where the last job left them. Each training set the generator made (ROOT/sets/bKKKK.tgz, shipped there
by the operator's capacity task) is a phase, in order:

  1. measure  the loss on the new batch's calls, games no phase has trained on, under the adapter as the last phase
              left it (the base model before phase 0): the mean negative log-likelihood per reply token, each chain
              read once (Trainer.nll_chains). This is the stop's signal, the loss on what it has not seen; the loss on
              what it has trained on keeps falling as it learns GLM's exact text.
  2. train    one pass over every batch so far, warm-started from the last phase (its adapter and optimizer moments),
              in steps of BATCH calls at LR, chains read whole (train.py's chain batches), nothing held out, no early
              stop and nothing chosen by a validation set; once every batch so far holds more than CAP calls, the new
              batch whole and a draw of the older games to CAP (seeded by the phase).
  3. keep     ROOT/phases/pKKKK/adapter (what vLLM loads, and the trainer's weights and moments for the next phase, which
              only the newest phase keeps) and phase.json; then ROOT/state.json, every phase's numbers and whether the
              run has plateaued (capacity.plateaued: three phases without a 1% gain on the best new-batch loss), with
              the same a line a phase in ROOT/phases.jsonl, and ROOT/steps.jsonl, every step's loss.

It stops when no set is left to train, when the run has plateaued, or when the next phase would not end within
--hours of its start (Modal ends a job at 24). Run from ~/prime-rl with the bundle's ~/loop on the path:
  cd ~/prime-rl && PYTHONPATH=$HOME/loop uv run --no-sync python ~/gen/tools/capacity/trainer.py t25 --root /cap/t25
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import capacity

MODEL = "Qwen/Qwen3.5-9B"
LR, BATCH, RANK = 5e-5, 16, 128   # R's recipe's rate and batch; train.py's LoRA defaults, every projection
CAP = 16_000   # calls a phase trains on at most: about 70 minutes on 8 H100s
GPUS = 8
MAX_SEQ_LEN = 65_536


def now() -> str:
    return f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"


def trainer_args() -> argparse.Namespace:
    return argparse.Namespace(model=MODEL, rank=RANK, alpha=RANK, lr=LR, max_seq_len=MAX_SEQ_LEN, gpus=GPUS)


def phase_calls(per_set: list[list[Any]], k: int, cap: int = CAP) -> tuple[list[Any], str]:
    """Phase ``k``'s training calls from every set so far (``per_set[i]``, set i's calls): all of them, or once they
    hold more than ``cap``, the newest set whole and the older sets' games drawn whole, seeded by the phase, until
    ``cap``. Returns the calls and which it was."""
    import train
    every = [c for calls in per_set for c in calls]
    if len(every) <= cap:
        return every, "every batch"
    newest, older = per_set[-1], train.by_game([c for calls in per_set[:-1] for c in calls])
    random.Random(k).shuffle(older)
    picked, n = [], len(newest)
    for game in older:
        if n >= cap:
            break
        picked += game
        n += len(game)
    return newest + picked, f"the newest batch and {len(picked)} older calls drawn"


class Run:
    def __init__(self, run: str, root: Path, hours: float):
        self.run, self.root, self.hours = run, root, hours
        self.sets, self.phases = root / "sets", root / "phases"
        self.phases.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, message: str) -> None:
        with open(self.root / "progress", "a") as f:
            f.write(f"{datetime.now(timezone.utc):%m-%d %H:%M} trainer.py {self.run}: {message}\n")
        print(message, flush=True)

    def state(self) -> dict[str, Any]:
        f = self.root / "state.json"
        return json.loads(f.read_text()) if f.exists() else {"run": self.run, "tasks": capacity.RUNS[self.run],
                                                             "phases": [], "plateaued": False}

    def done(self) -> int:
        return len(self.state()["phases"])

    def set_dir(self, k: int) -> Path:
        """Batch ``k``'s training set, a play folder named after its batch, unpacked once from its tarball."""
        name = f"b{k:04d}"
        if not (self.sets / name / "captured").is_dir():
            subprocess.run(["tar", "-C", str(self.sets), "-xzf", str(self.sets / f"{name}.tgz")], check=True)
        return self.sets / name

    def waiting(self, k: int) -> bool:
        return (self.sets / f"b{k:04d}.tgz").exists()

    def phase(self, trainer: Any, k: int) -> dict[str, Any]:
        import train
        rng = random.Random(k)
        per_set = [train.load_calls(self.set_dir(i))[0] for i in range(k + 1)]
        tok = trainer.tokenizer
        new_items, new_stats = train.sft_items(tok, per_set[-1], MAX_SEQ_LEN)
        chains = [train.chain_item(p, train.CHAIN_SLACK) for ch in train.find_chains(new_items, train.CHAIN_SLACK)
                  for p in train.chain_pieces(ch, BATCH, MAX_SEQ_LEN, train.CHAIN_SLACK)]
        t = time.time()
        res = [x for x in trainer.nll_chains(chains)["items"] if x]
        tokens = sum(x["tokens"] for x in res)
        new_loss = -sum(x["logp"] for x in res) / max(1, tokens)
        measure_min = (time.time() - t) / 60
        with open(self.root / "new-batch.jsonl", "a") as log:   # each chain's own result, beside the phase's mean
            log.write(json.dumps({"at": now(), "phase": k, "new_loss": new_loss, "tokens": tokens, "items": res},
                                 default=str) + "\n")
        self.log(f"phase {k}: batch {k}'s {len(new_items)} calls ({tokens} reply tokens) at {new_loss:.4f} a token "
                 f"under {'phase ' + str(k - 1) if k else 'the base model'}, in {measure_min:.1f} min")
        calls, dose = phase_calls(per_set, k)
        items, stats = train.sft_items(tok, calls, MAX_SEQ_LEN)
        steps = train.chain_batches(train.find_chains(items, train.CHAIN_SLACK), BATCH, GPUS, 1, MAX_SEQ_LEN, rng,
                                    slack=train.CHAIN_SLACK)
        self.log(f"phase {k}: training on {len(items)} calls ({dose}), {len(steps)} steps")
        t, losses = time.time(), []
        with open(self.root / "steps.jsonl", "a") as log:
            for n, step in enumerate(steps, 1):
                s = time.time()
                r = trainer.sft_chains([train.chain_item(piece, train.CHAIN_SLACK) for piece in step], rng=rng)
                taken = trainer.step()
                if r["nll_per_token"] is not None:
                    losses.append(r["nll_per_token"])
                log.write(json.dumps({"at": now(), "phase": k, "step": n, "nll_per_token": r["nll_per_token"],
                                      "items": r["n_examples"], "tokens": r["tokens"], "train_s": round(time.time() - s, 1),
                                      "grad_norm": round(taken.get("grad_norm", 0.0), 5)}) + "\n")
                log.flush()
        train_min = (time.time() - t) / 60
        out = self.phases / f"p{k:04d}"
        trainer.save(out / "adapter")
        summary = {"phase": k, "at": now(), "new_loss": round(new_loss, 6), "new_tokens": tokens,
                   "new_calls": len(new_items), "new_too_long": new_stats["too_long"], "trained_calls": len(items),
                   "trained_too_long": stats["too_long"], "dose": dose, "steps": len(steps),
                   "train_loss_first": round(losses[0], 6) if losses else None,
                   "train_loss_last": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 6) if losses else None,
                   "measure_min": round(measure_min, 1), "train_min": round(train_min, 1),
                   "sets": k + 1, "games": sum(len(train.by_game(c)) for c in per_set)}
        (out / "phase.json").write_text(json.dumps(summary, indent=1))
        if k:   # the trainer's weights and moments, a few GB, are kept by the newest phase alone
            self.prune(self.phases / f"p{k - 1:04d}" / "adapter")
        return summary

    @staticmethod
    def prune(adapter: Path) -> None:
        for f in adapter.iterdir():
            if f.name not in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.rmtree(f) if f.is_dir() else f.unlink()

    def write_state(self, state: dict[str, Any]) -> None:
        tmp = self.root / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.root / "state.json")

    def loop(self) -> int:
        import train
        state = self.state()
        k = len(state["phases"])
        if state["plateaued"] or not self.waiting(k):
            self.log(f"nothing to train: {k} phases done, {'plateaued' if state['plateaued'] else 'no new set'}")
            return 0
        trainer = train.make_trainer(trainer_args())
        if k:
            trainer.load(self.phases / f"p{k - 1:04d}" / "adapter")
        self.log(f"started: phase {k} on, from {'phase ' + str(k - 1) if k else 'the base model'}")
        while self.waiting(k) and not state["plateaued"]:
            last = state["phases"][-1] if state["phases"] else None
            need = (last["measure_min"] + last["train_min"]) * 1.2 + 10 if last else 30
            if (time.time() - self.t0) / 60 + need > self.hours * 60:
                self.log(f"phase {k} would not end within {self.hours} h of the job's start: left for the next job")
                break
            summary = self.phase(trainer, k)
            state["phases"].append(summary)
            state["plateaued"] = capacity.plateaued([p["new_loss"] for p in state["phases"]])
            state["updated"] = now()
            self.write_state(state)
            with open(self.root / "phases.jsonl", "a") as f:
                f.write(json.dumps({**summary, "plateaued": state["plateaued"]}) + "\n")
            self.log(f"phase {k} done: {json.dumps(summary)}{'; plateaued' if state['plateaued'] else ''}")
            k += 1
        return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", choices=capacity.RUNS)
    p.add_argument("--root", required=True, help="the run's folder: sets/ in, phases/ and state.json out")
    p.add_argument("--hours", type=float, default=22.5, help="no phase starts that would end after this many hours")
    a = p.parse_args()
    return Run(a.run, Path(a.root), a.hours).loop()


if __name__ == "__main__":
    sys.exit(main())
