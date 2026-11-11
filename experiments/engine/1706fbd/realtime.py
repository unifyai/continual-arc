"""The paper's trained rows, DAgger, OPSD and PG, with a fixed lag: the student plays `arc1-small` served to its users
round after round while the oracle labels its games (DAgger and OPSD) and one resident trainer trains on them, and
round r is played by the model trained on rounds 0..r-1-LAG, no more and no less (the author, 19 Oct 2026: a fixed one-round
lag, LAG=1, so that which model plays which round never depends on how fast the hardware is; LAG=0 is the strict,
synchronous protocol). Play waits for that model when training is behind; training never waits for play.

Three loops (two for PG, which needs no oracle) run side by side on one node, each its own process, handing each other
files (realtime.sh starts them, restarts one that dies, and stops the run once all of them are done):

  player    round r = 0, 1, ...: phase r-1-LAG's adapter (the base model while r-1-LAG < 0), waited for until the
            trainer has finished it, is loaded into the model server, which stays up on its own GPUs, checked
            (paper.check_served_adapter: vLLM silently serves the base model under an adapter's name when its layers
            are misnamed), named in round_dir(r)/served.json, and plays round r (paper.play_round over
            paper.stream_cut(r)); then round_dir(r)/play.done. Phase k trains on rounds 0..k exactly, so in the
            notation round r is played by pi_(r-1-LAG), which has learnt from every round but the LAG before r.
  labeller  DAgger and OPSD, round by round: the oracle's own reply to every player call of the round's games as they
            land (forward.py points/label: each game is asked once while play goes on, then the whole round once more
            when it is played, so a point that failed gets a second ask). DAgger then builds the label set
            (label-set) and splits it into a training side and a held-out side (split --hold-every: held_game's one
            game in HOLD_EVERY, played and scored like any other), round_dir(r)/labels/split/split.json, written
            last, marking the round labelled; OPSD's labels are its demonstrations as they stand, labels/done marking
            the round labelled.
  trainer   one process for the whole run, holding the model on its own GPUs. Phase k starts once round k is ready
            (DAgger and OPSD: labelled; PG: played), from phase k-1's adapter (the base model for phase 0):
              dagger  supervised fine-tuning (train.train_sft) on every round's training side so far, the held-out
                      loss on every held-out side so far measured every EVAL_STEPS steps and the phase stopped once
                      PATIENCE measurements in a row have not improved on it, the best adapter kept (DAgger's
                      aggregate, uniform; until a held-out game exists, one pass);
              opsd    on-policy self-distillation in SDFT's form (train.train_sdft with --demos): one pass over round
                      k's own calls, each call's teacher, the frozen base model, reading the oracle's reply at it as a
                      demonstration, grading a reply drawn fresh from the current student at every step (forward KL
                      over the full vocabulary, the first SKIP_FIRST ids ungraded);
              pg      plain policy gradient (train.train_pg): one pass over round k's games, each game's reward
                      against the mean of the run's last 32 games, every ratio clipped against the adapter that
                      played round k (phase k-1-LAG, named in its served.json);
            then train/pNNNN/{adapter,DONE,rounds.json} and train/latest.json.

Every stage resumes from its files: a round whose play was cut short is played again whole (the runner's own resume),
labels already written are kept, and a phase cut short is trained again from the last finished phase's adapter.

Subcommands (on the GPU node, from ~/prime-rl: ``uv run --no-sync python ~/loop/realtime.py ...``): player,
labeller, trainer; realtime.sh passes each its settings.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import loop  # noqa: E402
import paper  # noqa: E402

ROWS = ("dagger", "opsd", "pg")
LABELLED_ROWS = ("dagger", "opsd")   # the rows whose games the oracle labels
SERVER = "http://localhost:8101/v1"
POLL_S = 15


# -- the run's files --------------------------------------------------------------------------------------------

def play_dir(run: Path) -> Path:
    """The benchmark run the player plays into (``records.jsonl`` and every round's ``rounds/rNNNN``)."""
    return run / "play"


def round_dir(run: Path, round_: int) -> Path:
    return paper.round_dir(play_dir(run), round_)


def labels_dir(run: Path, round_: int) -> Path:
    return round_dir(run, round_) / "labels"


def split_dir(run: Path, round_: int) -> Path:
    return labels_dir(run, round_) / "split"


def phase_dir(run: Path, phase: int) -> Path:
    return run / "train" / f"p{phase:04d}"


def sides(round_: int) -> tuple[str, str]:
    """A round's training and held-out sides' folder names, which train.load_calls takes as their games' run."""
    return f"train-r{round_:04d}", f"dev-r{round_:04d}"


def say(run: Path, message: str) -> None:
    with open(run / "progress", "a") as fp:
        fp.write(f"{datetime.now(timezone.utc):%m-%d %H:%M} realtime: {message}\n")


def write_json(path: Path, doc: Any) -> None:
    """``doc`` at ``path`` atomically: another loop reading it sees the old file or the new one, never half of one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.rename(path)


def latest(run: Path) -> dict[str, Any] | None:
    """The newest finished phase (``train/latest.json``: its number k, its adapter, and the k+1 rounds it trained on,
    rounds 0..k), or None before the first."""
    path = run / "train" / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None


def finished(run: Path, phase: int) -> dict[str, Any] | None:
    """Phase ``phase`` once the trainer has finished it (its rounds.json, written after its DONE): its number, adapter
    and the rounds it trained on; None until then."""
    path = phase_dir(run, phase) / "rounds.json"
    if not path.exists():
        return None
    return {"phase": phase, "adapter": str(phase_dir(run, phase) / "adapter"),
            "rounds": json.loads(path.read_text())["rounds"]}


def ready(run: Path, round_: int, row: str) -> bool:
    """Whether round ``round_`` is ready for its phase: labelled and split (DAgger), labelled (OPSD), played (PG)."""
    if row == "dagger":
        return (split_dir(run, round_) / "split.json").exists()
    if row == "opsd":
        return (labels_dir(run, round_) / "done").exists()
    return (round_dir(run, round_) / "play.done").exists()


def ready_rounds(run: Path, rounds: int, row: str) -> int:
    """How many rounds from round 0 on are ready (:func:`ready`), in a row."""
    n = 0
    while n < rounds and ready(run, n, row):
        n += 1
    return n


def phase_runs(run: Path, rounds: int) -> tuple[list[str], list[str]]:
    """DAgger's rounds 0..rounds-1: their training sides and held-out sides that hold a game (a round with no
    held-out game, or whose games all failed labelling, has an empty side)."""
    train_runs, dev_runs = [], []
    for r in range(rounds):
        done = json.loads((split_dir(run, r) / "split.json").read_text())
        for name, out in zip(sides(r), (train_runs, dev_runs)):
            if done[name]["games"]:
                out.append(str(split_dir(run, r) / name))
    return train_runs, dev_runs


# -- player ----------------------------------------------------------------------------------------------------

def adapter_name(name: str, phase: int) -> str:
    return f"{name}-p{phase:04d}"


def served_models() -> list[str]:
    with urllib.request.urlopen(SERVER + "/models", timeout=60) as resp:
        return [m["id"] for m in json.loads(resp.read())["data"]]


def serve(a: argparse.Namespace, run: Path, phase: dict[str, Any] | None, current: str) -> str:
    """The model round plays on: ``phase``'s adapter, loaded (serve.sh, which also starts the server on
    ``--serve-gpus`` if it is down) and checked, every other adapter of this run's player unloaded; the base model when
    ``phase`` is None. An adapter that fails the served-adapter check stops the player (``SystemExit``, with a line in
    the progress log): the protocol names the model each round plays on, and a round played on another would break it,
    as one played on base weights served under a trained adapter's name would."""
    from ask_sd.server import Server
    serve_sh = ["bash", str(Path.home() / "loop" / "serve.sh"), a.serve_gpus, str(a.serve_dp)]
    if phase is None:
        subprocess.run(serve_sh, check=True)
        return loop.MODEL
    name = adapter_name(a.name, phase["phase"])
    if name == current:
        return current
    subprocess.run(serve_sh + [name, phase["adapter"]], check=True)
    server = Server(SERVER)
    if not paper.adapter_is_untrained(Path(phase["adapter"])):
        check = paper.check_served_adapter(server, loop.MODEL, name, paper.SERVED_ADAPTER_PROMPT)
        if not (check["answered"] and check["differs"]):
            say(run, f"phase {phase['phase']}'s adapter failed the served-adapter check ({check}): the player stops")
            server.unload_adapter(name)
            raise SystemExit(1)
    for other in served_models():
        if other.startswith(a.name + "-p") and other != name:
            server.unload_adapter(other)
    return name


def cmd_player(a: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = Path(a.run).expanduser()
    current = ""
    for r in range(a.rounds):
        rd = round_dir(run, r)
        if (rd / "play.done").exists():
            continue
        need = r - 1 - a.lag   # the phase that plays round r: trained on rounds 0..need
        phase = None
        if need >= 0:
            phase = finished(run, need)
            if phase is None:
                say(run, f"play round {r} waits for phase {need}")
                while phase is None:
                    time.sleep(POLL_S)
                    phase = finished(run, need)
        current = serve(a, run, phase, current)
        rd.mkdir(parents=True, exist_ok=True)
        write_json(rd / "served.json", {"model": current, "phase": need if phase else None,
                                        "trained_rounds": need + 1 if phase else 0, "lag": a.lag,
                                        "at": datetime.now(timezone.utc).isoformat()})
        schedule = paper.stream_cut(r, users=a.users)
        want = len(schedule.slots)
        have = 0
        for attempt in range(1, 4):
            say(run, f"play round {r} started ({current}, attempt {attempt})")
            try:
                paper.play_round(a.row, play_dir(run), schedule, base_url=SERVER, served_name=current)
            except Exception:  # noqa: BLE001  (a crashed play is retried: the runner replays the round whole)
                logging.exception("round %s's play failed", r)
            records = play_dir(run) / "records.jsonl"
            have = sum(1 for _ in open(records)) if records.exists() else 0
            if have >= want:
                break
            say(run, f"play round {r} short ({have}/{want}) after attempt {attempt}")
        if have < want:
            say(run, f"play round {r} failed after 3 attempts")
            raise SystemExit(1)
        (rd / "play.done").touch()
        say(run, f"play round {r} done ({have}/{want})")
    (run / "player.done").touch()


# -- labeller --------------------------------------------------------------------------------------------------

def label_pass(a: argparse.Namespace, run: Path, round_: int, guidelines: Path, games: set[tuple[str, int]] | None,
               final: bool) -> set[tuple[str, int]]:
    """One labelling pass over round ``round_``'s games that have landed: every point of them (``final``, the round
    played: forward.py's own points file, every game's, which label-set then reads) or only the points of games not in
    ``games`` (asked once while play goes on). Returns the games it asked about. The run's label spend is checked
    first, and the final pass fails under ``--min-labelled-pct``."""
    import forward
    out = labels_dir(run, round_)
    out.mkdir(parents=True, exist_ok=True)
    spent = forward.spent([labels_dir(run, r) for r in range(round_ + 1) if labels_dir(run, r).exists()])
    if spent >= a.label_run_budget:
        say(run, f"round {round_} not labelled: the run's labels have spent ${spent:.2f}, over {a.label_run_budget}")
        raise SystemExit(1)
    points = forward.points([round_dir(run, round_).resolve()])
    asked = {(p["play"], p["user"]) for p in points}
    if not final:
        points = [p for p in points if (p["play"], p["user"]) not in (games or set())]
    with open(out / "points.jsonl", "w") as fp:
        for p in points:
            fp.write(json.dumps(p) + "\n")
    if points:
        forward.cmd_label(argparse.Namespace(
            out=str(out), expert=a.label_model, expert_url=f"http://127.0.0.1:{a.relay_port}/v1", effort=a.label_effort,
            reasoning=json.dumps({"enabled": True, "effort": a.label_effort}), budget=a.label_round_budget,
            workers=a.label_workers, guidelines=str(guidelines), guidelines_at="message", rule_retries=3,
            thinking=True, need_thinking=a.need_thinking, limit=0, seed=0,
            min_labelled_pct=a.min_labelled_pct if final else 0))
    return asked


def finish_round(a: argparse.Namespace, run: Path, round_: int) -> dict[str, Any]:
    """DAgger: round ``round_``'s label set and its split (split.json, written last, marks the round labelled)."""
    import forward
    out = labels_dir(run, round_)
    shutil.rmtree(out / "set", ignore_errors=True)   # a set half-written by a killed attempt is rebuilt whole
    forward.cmd_label_set(argparse.Namespace(out=str(out), model=loop.MODEL, workers=a.label_set_workers))
    split = split_dir(run, round_)
    shutil.rmtree(split, ignore_errors=True)
    forward.cmd_split(argparse.Namespace(set=str(out / "set"), held=None, hold_every=a.hold_every, round=round_,
                                         sides=",".join(sides(round_)), out=str(split)))
    return json.loads((split / "split.json").read_text())


def cmd_labeller(a: argparse.Namespace) -> None:
    import oracle_hint
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = Path(a.run).expanduser()
    guidelines = run / "guidelines.md"
    guidelines.write_text(oracle_hint.library_rules(Path(a.c1).expanduser().read_text()))
    r, asked = ready_rounds(run, a.rounds, a.row), set()
    while r < a.rounds:
        rd = round_dir(run, r)
        played = (rd / "play.done").exists()
        landed = {(rd.name, int(f.name[1:4])) for f in (rd / "captured").glob("u*.json.gz")} if rd.exists() else set()
        if played:
            for attempt in range(1, 4):
                try:
                    label_pass(a, run, r, guidelines, asked, final=True)
                    break
                except SystemExit as error:
                    say(run, f"round {r}'s labels, attempt {attempt}: {error}")
                    if attempt == 3:
                        raise
                    time.sleep(300)
            if a.row == "dagger":
                say(run, f"round {r} labelled: {json.dumps(finish_round(a, run, r), separators=(',', ':'))}")
            else:
                (labels_dir(run, r) / "done").touch()
                say(run, f"round {r} labelled")
            r, asked = r + 1, set()
        elif landed - asked:
            asked |= label_pass(a, run, r, guidelines, asked, final=False)
        else:
            time.sleep(POLL_S)
    (run / "labeller.done").touch()


# -- trainer ---------------------------------------------------------------------------------------------------

KEEP_EVERY = 20   # every 20th phase's served adapter outlives pruning, for reading the run back later
# the adapter files a phase needs only while a later phase may still start from it or clip against it: the next phase
# loads the newest whole (its optimizer's state and the trainer's own copy), PG clips against the adapter that played
# its round, and the player converts and loads the one its round plays on (adapter.vllm)
RESUME_FILES = ("optimizer.pt", "trainer_adapter.safetensors", "trainer_state.json")


def serving(run: Path) -> int | None:
    """The phase the player's newest round is played on (its served.json), None before it has served one."""
    found = sorted(round_dir(run, 0).parent.glob("r*/served.json"))
    return json.loads(found[-1].read_text())["phase"] if found else None


def prune(root: Path, newest: int, served: int | None, lag: int) -> None:
    """Frees what phases older than ``newest - lag`` no longer need (4.4 GB a phase on Qwen3.5-9B at rank 128, so a
    1-user run that trains every round would fill a node's disk within a day): each one's vLLM copy and the files a
    phase resumes from go, and its whole adapter too unless it is a KEEP_EVERY-th phase. DONE, phase.json and
    rounds.json, every phase's record of its training, stay. Phases newest-lag..newest stay whole: the next phase
    starts from the newest, and PG's clips against the one that played its round, ``lag`` older. The ``served`` phase
    stays whole however old: a round plays on one phase to its end, and vLLM reads its adapter again whenever another
    adapter's load pushes it out of the server's cache (18 Oct 2026: a 1-user round outlasted three phases, and the
    reload of its pruned adapter killed the server under all four runs)."""
    for phase in range(newest - lag):
        if phase == served:
            continue
        out = root / f"p{phase:04d}"
        shutil.rmtree(out / "adapter.vllm", ignore_errors=True)
        if phase % KEEP_EVERY:
            shutil.rmtree(out / "adapter", ignore_errors=True)
            continue
        for name in RESUME_FILES:
            (out / "adapter" / name).unlink(missing_ok=True)


def train_args(a: argparse.Namespace) -> argparse.Namespace:
    """train.py's settings for every phase of the row (train.train_sft, train_sdft or train_pg read them); the seed,
    and DAgger's held-out runs and pass count, OPSD's demonstrations, are each phase's own (:func:`cmd_trainer`)."""
    import train
    common = dict(model=loop.MODEL, lr=a.lr, batch=a.batch, gpus=a.gpus, rank=a.lora_rank, alpha=a.lora_rank,
                  max_seq_len=a.max_seq_len, chain_slack=train.CHAIN_SLACK, seed=0, epochs=1, max_steps=None,
                  max_minutes=a.max_minutes, dry_run=False)
    if a.row == "dagger":
        return argparse.Namespace(**{**common, "max_steps": a.max_steps}, objective=train.Objective.sft, chains=True,
                                  eval_steps=a.eval_steps, patience=a.patience, val_run=None,
                                  val_max_items=a.val_max_items, divergence="forward_kl")
    if a.row == "opsd":
        # SDFT's form (paper.sh's own OPSD flags): the frozen base model as teacher, no anchor, forward KL, fresh
        # replies from the server every step, nothing measured on the side (no validation plays)
        return argparse.Namespace(**common, objective=train.Objective.sdft, demos=None, ema_decay=None,
                                  teacher_adapter=None, anchor_mu=0.0, divergence="forward_kl", skip_first=a.skip_first,
                                  on_policy=True, server=SERVER, placement="message", val_run=None,
                                  eval_every=10**9, eval_items=256, no_initial_eval=True)
    return argparse.Namespace(**common, objective=train.Objective.pg, per_call=False)


def train_phase(a: argparse.Namespace, run: Path, targs: argparse.Namespace, trainer: Any, k: int, out: Path,
                init: Path | None) -> dict[str, Any]:
    """Phase ``k``, its row's training (the module docstring) from ``init``, into ``out``: DONE's summary."""
    import train
    rng = random.Random(k)
    targs.seed = k
    if a.row == "dagger":
        train_runs, dev_runs = phase_runs(run, k + 1)
        calls, compactions = [], 0
        for path in train_runs:
            got, left_out = train.load_calls(Path(path))
            calls += got
            compactions += left_out
        targs.val_run = dev_runs or None
        targs.epochs = a.max_epochs if dev_runs else 1   # no held-out game yet: one pass, nothing to stop on
        say(run, f"phase {k} started: rounds 0-{k}, {len(calls)} labels, {len(dev_runs)} held-out sides")
        return train.train_sft(targs, trainer, calls, compactions, out, init, rng)
    calls, compactions = train.load_calls(round_dir(run, k))
    if a.row == "opsd":
        calls, demos = train.attach_demos(trainer.tokenizer, calls, labels_dir(run, k))
        targs.demos = [str(labels_dir(run, k))]
        say(run, f"phase {k} started: round {k}, {demos['with_demo']} of {demos['calls']} calls with a demonstration")
        return train.train_sdft(targs, trainer, calls, compactions, out, init, rng, demo_stats=demos)
    played = json.loads((round_dir(run, k) / "served.json").read_text())["phase"]
    played_by = phase_dir(run, played) / "adapter" if played is not None else None
    records = {round_dir(run, k).name: train.instance_records(round_dir(run, k))}
    say(run, f"phase {k} started: round {k}, played by {f'phase {played}' if played is not None else 'the base model'}")
    return train.train_pg(targs, trainer, calls, records, compactions, out, init, played_by, rng)


def phase_line(row: str, done: dict[str, Any]) -> str:
    """A finished phase's summary for the progress log."""
    if row == "dagger":
        dev = [e["dev_nll"] for e in done["dev"]]
        held = f", held-out {dev[0]} to {min(dev)} (best at step {done['best_step']})" if dev else ""
        return f"{done['steps']} steps in {done['minutes']} min{held}"
    if row == "pg":
        games = done["games"]
        return (f"{games['episodes_used']}/{games['episodes']} games trained ({games['solved_used']} solved), "
                f"{done['steps']} steps in {done['minutes']} min")
    items = done["items"]
    return f"{items.get('drawn', 0)} replies drawn, {done['steps']} steps"


def cmd_trainer(a: argparse.Namespace) -> None:
    import kernel_stack
    import train
    run = Path(a.run).expanduser()
    root = run / "train"
    root.mkdir(parents=True, exist_ok=True)
    targs = train_args(a)
    trainer = train.make_trainer(targs)
    kernel_stack.record(root)
    while True:
        newest = latest(run)
        k = newest["rounds"] if newest else 0   # phase k trains round k: one round a phase, rounds 0..k in all
        if k == a.rounds:
            break
        if newest and newest["phase"] != k - 1:
            raise SystemExit(f"train/latest.json names phase {newest['phase']} over {k} rounds: phase k trains rounds "
                             f"0..k, one round a phase")
        if not ready(run, k, a.row):
            time.sleep(POLL_S)
            continue
        out = root / f"p{k:04d}"
        shutil.rmtree(out, ignore_errors=True)   # a phase cut short is trained again whole
        out.mkdir()
        done = train_phase(a, run, targs, trainer, k, out, Path(newest["adapter"]) if newest else None)
        write_json(out / "rounds.json", {"rounds": k + 1})
        write_json(root / "latest.json", {"phase": k, "adapter": str(out / "adapter"), "rounds": k + 1})
        prune(root, k, serving(run), a.lag)
        say(run, f"phase {k} done: rounds 0-{k}, {phase_line(a.row, done)}")
    (run / "trainer.done").touch()


# -- CLI -------------------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--run", required=True, help="the run's folder (realtime.sh's ~/runs/rt/NAME)")
        s.add_argument("--rounds", type=int, required=True)
        s.add_argument("--row", choices=ROWS, required=True)

    s = sub.add_parser("player")
    common(s)
    s.add_argument("--users", type=int, required=True, help="the users the stream is served to (paper.served)")
    s.add_argument("--name", required=True, help="the run's name, its adapters' prefix on the server")
    s.add_argument("--lag", type=int, required=True,
                   help="round r plays on the phase trained on rounds 0..r-1-LAG (1: the paper's; 0: synchronous)")
    s.add_argument("--serve-gpus", required=True, dest="serve_gpus")
    s.add_argument("--serve-dp", type=int, required=True, dest="serve_dp")
    s.set_defaults(fn=cmd_player)

    s = sub.add_parser("labeller")
    common(s)
    s.add_argument("--c1", required=True, help="the oracle's playing rules (oracle_hint.library_rules adds the "
                                               "program library's)")
    s.add_argument("--label-model", required=True, dest="label_model")
    s.add_argument("--label-effort", default="medium", dest="label_effort")
    s.add_argument("--relay-port", type=int, required=True, dest="relay_port")
    s.add_argument("--label-workers", type=int, default=256, dest="label_workers")
    s.add_argument("--label-set-workers", type=int, default=32, dest="label_set_workers")
    s.add_argument("--need-thinking", type=int, default=4, dest="need_thinking")
    s.add_argument("--label-round-budget", type=float, required=True, dest="label_round_budget")
    s.add_argument("--label-run-budget", type=float, required=True, dest="label_run_budget")
    s.add_argument("--min-labelled-pct", type=float, default=40, dest="min_labelled_pct")
    s.add_argument("--hold-every", type=int, default=10, dest="hold_every")
    s.set_defaults(fn=cmd_labeller)

    s = sub.add_parser("trainer")
    common(s)
    s.add_argument("--lag", type=int, required=True, help="the player's lag: how many phases back PG's plays are")
    s.add_argument("--lr", type=float, required=True)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--gpus", type=int, required=True, help="the GPUs CUDA_VISIBLE_DEVICES gives this process")
    s.add_argument("--lora-rank", type=int, default=128, dest="lora_rank")
    s.add_argument("--max-seq-len", type=int, default=65536, dest="max_seq_len")
    s.add_argument("--eval-steps", type=int, default=5, dest="eval_steps", help="dagger: steps between held-out checks")
    s.add_argument("--patience", type=int, default=2, help="dagger: checks in a row without improving that stop a phase")
    s.add_argument("--max-steps", type=int, default=120, dest="max_steps", help="dagger: a phase's safety cap")
    s.add_argument("--max-epochs", type=int, default=20, dest="max_epochs")
    s.add_argument("--val-max-items", type=int, default=1000, dest="val_max_items")
    s.add_argument("--skip-first", type=int, default=3, dest="skip_first",
                   help="opsd: each reply's first ids left ungraded (SDFT's code: 3)")
    s.add_argument("--max-minutes", type=float, default=240, dest="max_minutes")
    s.set_defaults(fn=cmd_trainer)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
