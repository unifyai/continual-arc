"""The real-time driver of the paper's trained rows, DAgger, OPSD and PG: the student plays ``arc1-small`` served to
its users round after round while the oracle labels its games (DAgger and OPSD) and one resident trainer trains on
them, and round r is played by the model trained on rounds 0..r-1-LAG, no more and no less. With LAG=1, the paper's,
which model plays which round never depends on how fast the hardware is; LAG=0 is the strict, synchronous protocol.
Play waits for its model when training is behind; training never waits for play.

A run plays ``--rounds`` rounds of the schedule (:func:`continual_arc_training.play.full_schedule`): realtime.sh's
default is the stream's 200, after which frozen.sh plays the run's two frozen tests from its end-of-stream state
(continual_arc_training.frozen). The schedule goes on with the epilogue's 100 rounds, which a run given 300 rounds
plays and trains on as the stream, and which no measure reads.

Three loops (two for PG, which needs no oracle) run side by side on one node, each its own process, handing each other
files in the run's folder; realtime.sh starts them, restarts one that dies, and stops the run once all are done:

  player    round r = 0, 1, ...: phase r-1-LAG's adapter (the base model while r-1-LAG < 0, or when that phase saved
            none), waited for until the trainer has finished it, is loaded into the model server, which stays up on
            its own GPUs, checked (vLLM silently serves the base model under an adapter's name when its layers are
            misnamed: :func:`continual_arc_training.server.check_served_adapter`), named in the round's
            ``served.json``, and plays round r; then the round's ``play.done``.
  labeller  DAgger and OPSD alike: the oracle's own reply to every call of the round's games as they land, each game
            asked once while play goes on and the whole round once more when it is played, under the working rules
            ``--oracle-rules`` names (continual_arc_training.oracle), kept in the run's ``guidelines.md``, which a
            labeller started again must find unchanged; then the round's label set and its split into a training side
            and a held-out side, one game in ``--hold-every`` held out (continual_arc_training.labeller), the split's
            ``split.json``, written last, marking the round labelled.
  trainer   the run's phases, run by a ``train`` process that holds the model on the GPUs the node's pool of
            ``--train-gpus`` gives the run (continual_arc_training.gpupool: runs given the same TRAIN_GPUS share them,
            each phase on an even share of the runs with one to train). Phase k starts once round k is ready
            (DAgger and OPSD: labelled; PG: played), from phase k-1's adapter. While rounds 0..k hold fewer than
            ``--start-after`` games, phase k trains nothing, saves no adapter and takes no GPU, so the base model plays
            on. Then every row's phase is one setup (:func:`continual_arc_training.phase.fit`), only its update the
            row's own (continual_arc_training.updates): the row's calls of rounds 0..k under ``--dose`` (aggregate:
            every round's so far; recent: the last RECENT_ROUNDS rounds' and as many from whole earlier games drawn at
            random), the held-out games' calls for the held-out stop once one exists, the held-out loss measured every
            ``--eval-steps`` steps and the phase stopped once ``--patience`` measurements in a row have not improved on
            the best, the best adapter kept, at most ``--max-steps`` steps; until a held-out game exists, one pass. Then
            the phase's ``rounds.json`` and ``train/latest.json``.

Every stage resumes from its files: a round whose play was cut short is played again whole (the runner's own resume),
labels already written are kept, and a phase cut short goes on from its last step checkpoint
(continual_arc_training.checkpoints: one on SIGTERM, a preemptible machine's notice, and one every
``checkpoints.EVERY`` steps), on the same batches in the same order, or trains again from its start when it had none.

Subcommands (``python -m continual_arc_training.realtime ...``, from the node's environment): player, labeller,
trainer; realtime.sh passes each its settings, and the trainer runs ``train`` processes on its GPUs.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from continual_arc_training import checkpoints, gpupool, labeller, oracle, play
from continual_arc_training.calls import Call, by_game, load_calls

ROWS = ("dagger", "opsd", "pg")
LABELLED_ROWS = ("dagger", "opsd")   # the rows whose games the oracle labels
SERVER = "http://localhost:8101/v1"
SERVE_SH = Path(__file__).resolve().parents[2] / "scripts" / "serve.sh"
POLL_S = 15
# every KEEP_EVERY-th phase's served adapter outlives pruning, for reading the run back later
KEEP_EVERY = 20
# the adapter files a phase needs only while a later phase may start from it: the next phase loads the newest whole (its
# optimizer's state and the trainer's own copy)
RESUME_FILES = ("optimizer.pt", "trainer_adapter.safetensors")
# the recent dose's window: a phase trains on the last RECENT_ROUNDS rounds' labels and as many earlier games'
RECENT_ROUNDS = 10
# a train process's exit when the pool's share for its next phase is not the GPUs it holds
REGROW = 75
SUPERVISE_S = 5


# -- the run's files -----------------------------------------------------------------------------------------------

def play_dir(run: Path) -> Path:
    """The benchmark run the player plays into (``records.jsonl`` and every round's ``rounds/rNNNN``)."""
    return run / "play"


def round_dir(run: Path, round_: int) -> Path:
    return play.round_dir(play_dir(run), round_)


def labels_dir(run: Path, round_: int) -> Path:
    return round_dir(run, round_) / "labels"


def split_dir(run: Path, round_: int) -> Path:
    return labels_dir(run, round_) / "split"


def phase_dir(run: Path, phase: int) -> Path:
    return run / "train" / f"p{phase:04d}"


def sides(round_: int) -> tuple[str, str]:
    """A round's training and held-out sides' folder names, which training takes as their games' run."""
    return f"train-r{round_:04d}", f"dev-r{round_:04d}"


def say(run: Path, message: str) -> None:
    with open(run / "progress", "a") as fp:
        fp.write(f"{datetime.now(UTC):%m-%d %H:%M} realtime: {message}\n")


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
    (None for a warm-up phase, which trained nothing and leaves the base model in place) and the rounds it trained on;
    None until then."""
    path = phase_dir(run, phase) / "rounds.json"
    if not path.exists():
        return None
    adapter = phase_dir(run, phase) / "adapter"
    return {"phase": phase, "adapter": str(adapter) if adapter.exists() else None,
            "rounds": json.loads(path.read_text())["rounds"]}


def ready(run: Path, round_: int, row: str) -> bool:
    """Whether round ``round_`` is ready for its phase: labelled, its label set split (DAgger and OPSD), or played
    (PG)."""
    if row in LABELLED_ROWS:
        return (split_dir(run, round_) / "split.json").exists()
    return (round_dir(run, round_) / "play.done").exists()


def ready_rounds(run: Path, rounds: int, row: str) -> int:
    """How many rounds from round 0 on are ready (:func:`ready`), in a row."""
    n = 0
    while n < rounds and ready(run, n, row):
        n += 1
    return n


def phase_runs(run: Path, rounds: int) -> tuple[dict[int, str], list[str]]:
    """DAgger's and OPSD's rounds 0..rounds-1: their training sides that hold a game, by round, and their held-out
    sides that hold one."""
    train_runs, dev_runs = {}, []
    for r in range(rounds):
        done = json.loads((split_dir(run, r) / "split.json").read_text())
        train_side, dev_side = sides(r)
        if done[train_side]["games"]:
            train_runs[r] = str(split_dir(run, r) / train_side)
        if done[dev_side]["games"]:
            dev_runs.append(str(split_dir(run, r) / dev_side))
    return train_runs, dev_runs


# -- player ------------------------------------------------------------------------------------------------------

def adapter_name(name: str, phase: int) -> str:
    return f"{name}-p{phase:04d}"


def serve(a: argparse.Namespace, run: Path, phase: dict[str, Any] | None, current: str) -> str:
    """The model the round plays on: ``phase``'s adapter, loaded (serve.sh, which also starts the server on
    ``--serve-gpus`` if it is down) and checked, every other adapter of this run's player unloaded; the base model when
    ``phase`` is None or saved no adapter. An adapter that fails the served-adapter check stops the player
    (``SystemExit``, with a line in the progress log): the protocol names the model each round plays on, and a round
    played on base weights served under a trained adapter's name would break it."""
    from continual_arc_training.server import Server, adapter_is_untrained, check_served_adapter
    serve_sh = ["bash", str(SERVE_SH), a.serve_gpus, str(a.serve_dp)]
    if phase is None or phase["adapter"] is None:
        subprocess.run(serve_sh, check=True)
        return play.MODEL
    name = adapter_name(a.name, phase["phase"])
    if name == current:
        return current
    subprocess.run(serve_sh + [name, phase["adapter"]], check=True)
    server = Server(SERVER)
    if not adapter_is_untrained(Path(phase["adapter"])):
        check = check_served_adapter(server, play.MODEL, name)
        if not (check["answered"] and check["differs"]):
            say(run, f"phase {phase['phase']}'s adapter failed the served-adapter check ({check}): the player stops")
            server.unload_adapter(name)
            raise SystemExit(1)
    for other in server.models():
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
                                        "trained_rounds": need + 1 if phase and phase["adapter"] else 0, "lag": a.lag,
                                        "at": datetime.now(UTC).isoformat()})
        schedule = play.schedule_cut(r, users=a.users)
        want = len(schedule.slots)
        have = 0
        for attempt in range(1, 4):
            say(run, f"play round {r} started ({current}, attempt {attempt})")
            try:
                play.play_round(a.row, play_dir(run), schedule, base_url=SERVER, served_name=current)
            except Exception:  # noqa: BLE001  (a crashed play is retried: the runner plays the round again whole)
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


# -- labeller ----------------------------------------------------------------------------------------------------

def label_pass(a: argparse.Namespace, run: Path, round_: int, rules: str, games: set[tuple[str, int]] | None,
               final: bool) -> set[tuple[str, int]]:
    """One labelling pass over round ``round_``'s games that have landed: every point of them (``final``, the round
    played: the points file then holds every game's, which the label set reads) or only the points of games not in
    ``games`` (asked once while play goes on). Returns the games it asked about. The run's label spend is checked
    first, and the final pass fails under ``--min-labelled-pct``."""
    out = labels_dir(run, round_)
    out.mkdir(parents=True, exist_ok=True)
    spent = labeller.spent([labels_dir(run, r) for r in range(round_ + 1) if labels_dir(run, r).exists()])
    if spent >= a.label_run_budget:
        say(run, f"round {round_} not labelled: the run's labels have spent ${spent:.2f}, over {a.label_run_budget}")
        raise SystemExit(1)
    pts = labeller.points([round_dir(run, round_).resolve()])
    asked = {(p["play"], p["user"]) for p in pts}
    if not final:
        pts = [p for p in pts if (p["play"], p["user"]) not in (games or set())]
    with open(out / "points.jsonl", "w") as fp:
        for p in pts:
            fp.write(json.dumps(p) + "\n")
    if pts:
        ask = oracle.Oracle(f"http://127.0.0.1:{a.relay_port}/v1", a.label_model, a.label_effort).ask
        labeller.label(out, ask, rules, budget=a.label_round_budget, workers=a.label_workers,
                       need_thinking=a.need_thinking, min_labelled_pct=a.min_labelled_pct if final else 0)
    return asked


def finish_round(a: argparse.Namespace, run: Path, round_: int) -> dict[str, Any]:
    """Round ``round_``'s label set and its split (split.json, written last, marks the round labelled)."""
    out = labels_dir(run, round_)
    shutil.rmtree(out / "set", ignore_errors=True)   # a set half-written by a killed attempt is built again whole
    print(json.dumps(labeller.label_set(out, play.MODEL, a.label_set_workers)), flush=True)
    split = split_dir(run, round_)
    shutil.rmtree(split, ignore_errors=True)
    labeller.split(out / "set", split, round_, a.hold_every, sides(round_))
    return json.loads((split / "split.json").read_text())


def started_oracle_rules(run: Path) -> str:
    """What the run's oracle is told (realtime.sh's ORACLE_RULES), as the run's last "started:" line names it, the line
    a relaunch takes the run's settings from; ``library``, realtime.sh's default, for a line that names none."""
    started = re.findall(r"realtime\.sh: started: .*", (run / "progress").read_text())[-1]
    named = re.search(r", oracle rules ([a-z0-9]+)(?:, patience \d+)?$", started)
    return named[1] if named else "library"


def cmd_labeller(a: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = Path(a.run).expanduser()
    guidelines = run / "guidelines.md"
    rules = oracle.oracle_rules(Path(a.c1).expanduser().read_text(), a.oracle_rules)
    # a relaunch under other rules, or a c1 changed under it, would label every later round under other rules than the
    # earlier ones, changing the oracle mid-run
    if not guidelines.exists():
        tmp = guidelines.with_name(guidelines.name + ".tmp")   # whole or not at all, since it is never written again
        tmp.write_text(rules)
        tmp.rename(guidelines)
    elif guidelines.read_text() != rules:
        say(run, f"the labeller stops: guidelines.md, the rules the run's oracle has labelled under, is not what "
                 f"--oracle-rules {a.oracle_rules} gives with {a.c1} (to change them on purpose, remove guidelines.md)")
        raise SystemExit(1)
    r, asked = ready_rounds(run, a.rounds, a.row), set()
    while r < a.rounds:
        rd = round_dir(run, r)
        played = (rd / "play.done").exists()
        landed = {(rd.name, int(f.name[1:4])) for f in (rd / "captured").glob("u*.json.gz")} if rd.exists() else set()
        if played:
            for attempt in range(1, 4):
                try:
                    label_pass(a, run, r, rules, asked, final=True)
                    break
                except SystemExit as error:
                    say(run, f"round {r}'s labels, attempt {attempt}: {error}")
                    if attempt == 3:
                        raise
                    time.sleep(300)
            say(run, f"round {r} labelled: {json.dumps(finish_round(a, run, r), separators=(',', ':'))}")
            r, asked = r + 1, set()
        elif landed - asked:
            asked |= label_pass(a, run, r, rules, asked, final=False)
        else:
            time.sleep(POLL_S)
    (run / "labeller.done").touch()


# -- trainer -----------------------------------------------------------------------------------------------------

def serving(run: Path) -> int | None:
    """The phase the player's newest round is played on (its served.json), None before it has served one."""
    found = sorted(round_dir(run, 0).parent.glob("r*/served.json"))
    return json.loads(found[-1].read_text())["phase"] if found else None


def prune(root: Path, newest: int, served: int | None, lag: int) -> None:
    """Frees what phases older than ``newest - lag`` no longer need (4.4 GB a phase on Qwen3.5-9B at rank 128, so a
    1-user run that trains every round would fill a node's disk within a day): each one's vLLM copy and the files a
    phase resumes from go, and its whole adapter too unless it is a KEEP_EVERY-th phase. DONE, phase.json and
    rounds.json, every phase's record of its training, stay. Phases newest-lag..newest stay whole: the next phase
    starts from the newest, and PG's ratios are taken under the one that played its round, ``lag`` older. The
    ``served`` phase stays whole however old: a round plays on one phase to its end, and vLLM reads its adapter again
    whenever another adapter's load pushes it out of the server's cache."""
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
    """Every row's training settings, one configuration for all three (:func:`continual_arc_training.phase.fit`, and of
    the updates OPSD's alone reads the server it draws its replies from and the ids it leaves ungraded); each phase sets
    its own seed and passes (:func:`train_phase`). ``gpus`` is the train process's own; nothing a phase trains on
    depends on it."""
    from continual_arc_training.chains import CHAIN_CALLS, CHAIN_SLACK
    from continual_arc_training.trainer import LORA_TARGETS
    return argparse.Namespace(model=play.MODEL, lr=a.lr, batch=a.batch, gpus=a.gpus, rank=a.lora_rank,
                              alpha=a.lora_rank, lora_targets=list(LORA_TARGETS), max_seq_len=a.max_seq_len,
                              chain_slack=CHAIN_SLACK, chain_calls=CHAIN_CALLS, seed=0, epochs=1,
                              max_steps=a.max_steps, max_minutes=a.max_minutes, eval_steps=a.eval_steps,
                              patience=a.patience, skip_first=a.skip_first, server=SERVER)


def games_played(run: Path, rounds: int) -> int:
    """How many games rounds 0..rounds-1 have played, every user's (their captured games)."""
    return sum(len(list((round_dir(run, r) / "captured").glob("u*.json.gz"))) for r in range(rounds))


def warming_up(run: Path, k: int, start_after: int) -> bool:
    """Whether phase ``k`` trains nothing, rounds 0..k having played fewer than ``start_after`` games: it saves no
    adapter, and the round it is for plays on the base model. The count only grows with k, so every such phase comes
    before the first that trains, and the model it leaves in place is the base model."""
    return games_played(run, k + 1) < start_after


def whole_games(games: list[list[Call]], max_items: int, seed: int) -> list[Call]:
    """The calls of ``games`` (each a game's calls): every game's, or, with ``max_items``, those of games drawn with
    ``seed`` until their calls reach it (a held-out set that grows every round, measured at a cost that does not)."""
    if max_items:
        games = games[:]
        random.Random(seed).shuffle(games)
        kept, n = [], 0
        for game in games:
            if n >= max_items:
                break
            kept.append(game)
            n += len(game)
        games = kept
    return [call for game in games for call in game]


def train_phase(a: argparse.Namespace, run: Path, targs: argparse.Namespace, trainer: Any, k: int, out: Path,
                init: Path | None) -> dict[str, Any]:
    """Phase ``k`` from ``init`` (the base model when None) into ``out``, every row's one setup (the module docstring):
    the row's calls of rounds 0..k under ``--dose``, the held-out games' calls for the held-out stop once one exists,
    and the phase's plan, steps and stop, with only the update the row's own (:func:`row_update`). DONE's summary."""
    from continual_arc_training import phase
    rng = random.Random(k)
    targs.seed = k
    if a.row == "pg":
        per_round, held, logps = pg_rounds(a, run, k, trainer, targs)
        # PG's loss weighs a held-out failure 0, so only a held-out win can say whether its training generalises
        stops = [game for game in held if game[0].meta["solved"]]
    else:
        (per_round, held), logps = labelled_rounds(run, k), {}
        stops = held
    calls, targs.epochs, stopped = dose(a, k, per_round, bool(stops))
    update = row_update(a, run, targs, trainer.tokenizer, calls,
                        whole_games(stops, a.val_max_items, k) if stopped else [], logps)
    trains = (f"{update.stats['wins']} of {update.stats['episodes']} games won, {len(update.units)} calls, "
              f"{len(held)} held-out games ({len(stops)} won)" if a.row == "pg"
              else f"{len(calls)} labels, {len(held)} held-out games")
    say(run, f"phase {k} started: rounds 0-{k}, {trains}, {targs.epochs} pass(es)"
             f"{'' if stopped else ', no held-out stop'}")
    return phase.fit(targs, trainer, update, out, init, rng)


def row_update(a: argparse.Namespace, run: Path, targs: argparse.Namespace, tok: Any, calls: list[Call],
               held: list[Call], logps: dict[tuple, list[float]]) -> Any:
    """The row's update over a phase's calls and its held-out calls: DAgger's imitation of the labels, OPSD's
    self-distillation from the same labels, PG's policy gradient on its won games' calls, each against the adapter that
    played it (``logps``)."""
    from continual_arc_training import updates
    if a.row == "dagger":
        return updates.Imitation(targs, tok, calls, held)
    if a.row == "opsd":
        return updates.SelfDistillation(targs, tok, calls, plays(run, held, tok))
    items, games = updates.pg_items(tok, calls, targs.max_seq_len)
    val = updates.pg_items(tok, held, targs.max_seq_len)[0]
    updates.attach_behaviour(items + val, logps)
    return updates.PolicyGradient(targs, items, games, val)


def recent_plan(new: list[Call], earlier: list[Call], ratio: float, rng: random.Random) -> list[Call]:
    """The recent dose's calls for a phase: every one of ``new``, and whole earlier games drawn at random until they
    hold at least ``ratio`` times ``len(new)`` labels (all of them when there are fewer). Whole games keep a game's
    calls together, so that a chain still reads its trunk once."""
    by: dict[tuple, list[Call]] = {}
    for c in earlier:
        by.setdefault((c.meta["run"], c.meta["user"]), []).append(c)
    pool = [sorted(by[key], key=lambda c: c.meta["i"]) for key in sorted(by, key=str)]
    rng.shuffle(pool)
    picked: list[Call] = []
    for game in pool:
        if len(picked) >= ratio * len(new):
            break
        picked += game
    return new + picked


def dose(a: argparse.Namespace, k: int, per_round: dict[int, list[Call]], held_out: bool) -> tuple[list[Call], int, bool]:
    """Phase ``k``'s training calls under ``--dose``, every row's alike, from ``per_round`` (each round's training side
    so far): the calls, their passes (``--max-epochs`` under the held-out stop, one pass while there is no held-out game
    to stop on), and whether the held-out stop runs."""
    epochs = a.max_epochs if held_out else 1
    if a.dose == "aggregate":
        return [c for r in sorted(per_round) for c in per_round[r]], epochs, held_out
    since = k + 1 - RECENT_ROUNDS
    recent = [c for r in sorted(per_round) if r >= since for c in per_round[r]]
    earlier = [c for r in sorted(per_round) if r < since for c in per_round[r]]
    return recent_plan(recent, earlier, a.replay_ratio, random.Random(1000 + k)), epochs, held_out


def labelled_rounds(run: Path, k: int) -> tuple[dict[int, list[Call]], list[list[Call]]]:
    """DAgger's and OPSD's rounds 0..k, as the labeller split their label sets (:func:`phase_runs`): each round's
    training side's calls, by round, and every held-out side's games, a list of calls each."""
    train_runs, dev_runs = phase_runs(run, k + 1)
    return ({r: load_calls(Path(path))[0] for r, path in train_runs.items()},
            [game for path in dev_runs for game in by_game(load_calls(Path(path))[0])])


def plays(run: Path, calls: list[Call], tok: Any) -> list[Call]:
    """Labelled calls (a label set's) as they were played: each the play's own call at that point (its round's captured
    game), whose served prompt and played reply OPSD's held-out loss scores, carrying the label's demonstration. A
    labelled call whose play came back without its token ids has no played reply to score, and is left out."""
    from continual_arc_training import updates
    played: dict[int, dict[tuple[int, int], Call]] = {}
    out = []
    for call in calls:
        r = int(call.meta["run"].rsplit("-r", 1)[1])   # its side's round (sides)
        if r not in played:
            played[r] = {(c.meta["user"], c.meta["i"]): c for c in load_calls(round_dir(run, r))[0]}
        own = played[r].get(labeller.source(call.meta["call_id"]))
        if own is None:
            continue
        own.meta["demo"] = updates.demo(tok, call)
        out.append(own)
    return out


def pg_rounds(a: argparse.Namespace, run: Path, k: int, trainer: Any,
              targs: argparse.Namespace) -> tuple[dict[int, list[Call]], list[list[Call]], dict[tuple, list[float]]]:
    """PG's rounds 0..k split as the labeller splits DAgger's and OPSD's (one game in ``--hold-every`` held out): each
    round's training-side calls, the held-out games' calls (a list per game), and every won game's calls' reply
    log-probabilities under the adapter that played its round (:func:`pg_behaviour`), the only calls PG trains on or
    measures."""
    per_round, held, logps = {}, [], {}
    for r in range(k + 1):
        got, _ = load_calls(round_dir(run, r))
        logps.update(pg_behaviour(run, r, [c for c in got if c.meta["solved"]], trainer, targs, a.start_after))
        per_round[r] = [c for c in got if not labeller.held_game(r, c.meta["user"], a.hold_every)]
        held += by_game([c for c in got if labeller.held_game(r, c.meta["user"], a.hold_every)])
    return per_round, held, logps


def pg_behaviour(run: Path, r: int, calls: list[Call], trainer: Any, targs: argparse.Namespace,
                 start_after: int) -> dict[tuple, list[float]]:
    """Round ``r``'s ``calls``' reply log-probabilities under the adapter that played the round (its served.json's
    phase, the base model before any phase trained), keyed by run, user and place: read from the round's
    ``pg_behaviour.json.gz``, or taken now and written there; none for a round with no calls given (no game won). A
    round played after the first phase that trains is first trained in phase ``r``, while the adapter that played it,
    ``LAG`` phases older, is still on disk (prune keeps it), so every later phase takes its ratios against the adapter
    that played each game, however long ago it was pruned. A round played in the warm-up is first trained in the first
    phase that trains, by when prune has deleted most of the warm-up's adapters; its player trained nothing, so it
    served the base model, which is what its ratios are taken against."""
    from continual_arc_training import updates
    from continual_arc_training.server import adapter_is_untrained
    path = round_dir(run, r) / "pg_behaviour.json.gz"
    name = round_dir(run, r).name
    if path.exists():
        with gzip.open(path, "rt") as fp:
            return {(name, int(u), int(i)): logps for key, logps in json.load(fp)["logps"].items()
                    for u, i in [key.split(":")]}
    if not calls:
        return {}
    played = json.loads((round_dir(run, r) / "served.json").read_text())["phase"]
    adapter = phase_dir(run, played) / "adapter" if played is not None else None
    if adapter is None or warming_up(run, played, start_after) or adapter_is_untrained(adapter):
        trainer.configure_behaviour(None)
    else:
        trainer.configure_behaviour(adapter)
    got = updates.behaviour_logps(trainer, targs, updates.played_items(trainer.tokenizer, calls, targs.max_seq_len))
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as fp:
        json.dump({"played_by": played, "logps": {f"{u}:{i}": logps for (_, u, i), logps in got.items()}}, fp)
    tmp.replace(path)
    return got


def phase_line(row: str, done: dict[str, Any]) -> str:
    """A finished phase's summary for the progress log."""
    if done.get("skipped"):
        return f"not trained: {done['games']} games played, training starts at {done['start_after']}"
    lead = ""
    if row == "opsd":
        lead = f"{done['items']['drawn']} replies drawn, "
    elif row == "pg":
        lead = f"{done['games']['wins']}/{done['games']['episodes']} games won and trained on, "
    held = done["held_out"]
    held_line = f", held-out {held[0]} to {min(held)} (best at step {done['best_step']})" if held else ""
    resumed = f", resumed from step {', '.join(map(str, done['resumed']))}" if done.get("resumed") else ""
    return f"{lead}{done['steps']} steps in {done['minutes']} min{held_line}{resumed}"


def close_phase(a: argparse.Namespace, run: Path, k: int, done: dict[str, Any]) -> None:
    """Phase ``k`` finished: its rounds.json (marking it finished for the player), train/latest.json (its adapter, None
    for a warm-up phase), the older phases' adapters pruned, and its line in the progress log."""
    out, root = phase_dir(run, k), run / "train"
    write_json(out / "rounds.json", {"rounds": k + 1})
    write_json(root / "latest.json", {"phase": k, "adapter": str(out / "adapter") if (out / "adapter").exists() else None,
                                      "rounds": k + 1})
    prune(root, k, serving(run), a.lag)
    say(run, f"phase {k} done: rounds 0-{k}, {phase_line(a.row, done)}")


def skip_phase(a: argparse.Namespace, run: Path, k: int) -> None:
    """Warm-up phase ``k`` (:func:`warming_up`), which needs no model and no GPU: its DONE says so, and the base model
    plays its round."""
    out = phase_dir(run, k)
    out.mkdir(parents=True, exist_ok=True)
    done = {"skipped": True, "games": games_played(run, k + 1), "start_after": a.start_after}
    write_json(out / "DONE", done)
    close_phase(a, run, k, done)


def next_phase(run: Path) -> int:
    """The phase the run trains next: phase k trains round k, one round a phase, rounds 0..k in all."""
    newest = latest(run)
    k = newest["rounds"] if newest else 0
    if newest and newest["phase"] != k - 1:
        raise SystemExit(f"train/latest.json names phase {newest['phase']} over {k} rounds: phase k trains rounds 0..k, "
                         f"one round a phase")
    return k


def active(run: Path) -> Path:
    """The file a train process keeps while a phase is in flight (its number and GPUs), for the trainer to see."""
    return run / "train" / "active.json"


def cmd_train(a: argparse.Namespace) -> None:
    """The run's phases on this process's GPUs (CUDA_VISIBLE_DEVICES, ``--gpus`` of them), which the run's trainer took
    from the node's pool (:func:`cmd_trainer`): the model loaded once, for every phase the pool lets it keep them.
    Before each phase it asks the pool for the run's share were it to start now, and ends with REGROW if that is not the
    GPUs it holds, for the trainer to start it again on its share; the warm-up's phases it leaves to the trainer, which
    needs no GPU for them."""
    from continual_arc_training import phase, stack
    checkpoints.catch_sigterm()   # a phase in flight checkpoints after its step and exits; between phases it just exits
    run = Path(a.run).expanduser()
    root = run / "train"
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / ".train.lock", "a")   # one train process a run: a second waits for the first to end
    fcntl.flock(lock, fcntl.LOCK_EX)
    mine = gpupool.parse(os.environ["CUDA_VISIBLE_DEVICES"])
    pool = gpupool.Pool(gpupool.parse(a.train_gpus), run.name)
    targs = train_args(a)
    trainer = phase.make_trainer(targs)
    stack.record(root)
    while True:
        k = next_phase(run)
        if k == a.rounds:
            break
        if not ready(run, k, a.row) or warming_up(run, k, a.start_after):
            if checkpoints.stopping():
                checkpoints.leave(None)
            time.sleep(POLL_S)
            continue
        if checkpoints.stopping():   # SIGTERM between phases: nothing to save
            checkpoints.leave(None)
        share = pool.share_if_starting()
        if share != mine:
            print(f"phase {k}: the pool's share is GPUs {share}, this process holds {mine}: ending for the trainer to "
                  f"start one on its share", flush=True)
            sys.exit(REGROW)
        write_json(active(run), {"phase": k, "gpus": mine, "pid": os.getpid()})
        out = phase_dir(run, k)
        if (out / "DONE").exists():   # trained to its end, its adapter in place, before it stopped
            done = json.loads((out / "DONE").read_text())
            shutil.rmtree(out / "checkpoints", ignore_errors=True)   # the stop came before they went
        else:
            step = checkpoints.saved_step(out)
            if step is None:   # cut short before its first checkpoint: trained again whole
                shutil.rmtree(out, ignore_errors=True)
                out.mkdir()
            else:
                say(run, f"phase {k} goes on from its checkpoint at step {step} on GPUs {','.join(map(str, mine))}")
            newest = latest(run)
            init = newest["adapter"] if newest else None
            done = train_phase(a, run, targs, trainer, k, out, Path(init) if init else None)
        close_phase(a, run, k, done)
        active(run).unlink(missing_ok=True)
    (run / "trainer.done").touch()


def tick(run: str, share: dict[str, list[int]], held: list[int], busy: bool, want: bool, alive: bool,
         asked: bool, others_hold: set[int]) -> str:
    """What the run's trainer does with its train process now, from the pool's split (``share``, every run's): "stop"
    it when another run's share takes GPUs it holds (between phases at once, mid-phase at its next step checkpoint),
    "start" one when the run wants GPUs and holds none and no other run's process holds any of its share, or "wait"."""
    if alive:
        theirs = {g for r, gpus in share.items() if r != run for g in gpus}
        if set(held) & theirs and not asked:
            return "stop"
        return "wait"
    if want and share[run] and not set(share[run]) & others_hold:
        return "start"
    return "wait"


def cmd_trainer(a: argparse.Namespace) -> None:
    """The run's trainer, as realtime.sh supervises it: the warm-up's phases written here, with no GPU, and every phase
    that trains run by a ``train`` process (:func:`cmd_train`) on the GPUs the node's pool of ``--train-gpus`` gives
    the run, started when the run has a phase to train and none runs, stopped when another run's phase needs its GPUs
    (:func:`tick`), and started again on its new share; its own end ends the run's training (trainer.done). A train
    process that ends any other way than asked or REGROW ends this one with its code, for realtime.sh to restart."""
    run = Path(a.run).expanduser()
    (run / "train").mkdir(parents=True, exist_ok=True)
    pool = gpupool.Pool(gpupool.parse(a.train_gpus), run.name)
    stopping: list[int] = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopping.append(signum))
    orphan = pool.entry()   # a train process an earlier trainer of this run left running, if one did
    child: subprocess.Popen | None = None
    held: list[int] = []
    asked = False
    argv = [sys.executable, "-m", "continual_arc_training.realtime", "train",
            *sys.argv[sys.argv.index("trainer") + 1:]]
    try:
        while True:
            if child is not None and child.poll() is not None:
                code = child.returncode
                active(run).unlink(missing_ok=True)
                if code == 0:
                    break
                if not (code == REGROW or (asked and code in (checkpoints.SIGTERM_EXIT, -signal.SIGTERM))):
                    raise SystemExit(f"the train process on GPUs {held} ended with {code}")
                child, held, asked = None, [], False
            if stopping:   # the run's own SIGTERM: its train process checkpoints (or ends) and this one with it
                if child is not None:
                    child.send_signal(signal.SIGTERM)
                    child.wait()
                raise SystemExit(checkpoints.SIGTERM_EXIT)
            k = next_phase(run)
            if k == a.rounds and child is None:
                break
            if child is None and k < a.rounds and ready(run, k, a.row) and warming_up(run, k, a.start_after):
                skip_phase(a, run, k)
                continue
            want = k < a.rounds and ready(run, k, a.row) and not warming_up(run, k, a.start_after)
            busy = child is not None and active(run).exists()
            view = pool.update(want, busy, held, child.pid if child else None)
            if orphan and gpupool.alive(orphan.get("pid")):   # it checkpoints (or ends) before this run starts another
                os.kill(orphan["pid"], signal.SIGTERM)
                while gpupool.alive(orphan["pid"]):
                    time.sleep(1)
            orphan = None
            action = tick(run.name, view["share"], held, busy, want, child is not None, asked,
                          pool.held_by_others(view["share"][run.name], view["runs"]))
            if action == "stop":
                child.send_signal(signal.SIGTERM)
                asked = True
                say(run, f"trainer: GPUs {held} go to another run's phase{' (at its next step checkpoint)' if busy else ''}")
            elif action == "start" and pool.claim(view["share"][run.name], os.getpid()):
                held = view["share"][run.name]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, held))}
                child = subprocess.Popen([*argv, "--gpus", str(len(held))], env=env)
                pool.update(want, False, held, child.pid)
                if a.train_gpus != ",".join(map(str, held)):
                    say(run, f"trainer: phase {k} trains on GPUs {','.join(map(str, held))} of the pool {a.train_gpus}")
            time.sleep(SUPERVISE_S)
    finally:
        pool.leave()
    (run / "trainer.done").touch()


# -- the command line ------------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--run", required=True, help="the run's folder (realtime.sh's RUNS/rt/NAME)")
        s.add_argument("--rounds", type=int, required=True)
        s.add_argument("--row", choices=ROWS, required=True)

    s = sub.add_parser("player")
    common(s)
    s.add_argument("--users", type=int, required=True, help="the users the schedule is served to")
    s.add_argument("--name", required=True, help="the run's name, its adapters' prefix on the server")
    s.add_argument("--lag", type=int, required=True,
                   help="round r plays on the phase trained on rounds 0..r-1-LAG (1: the paper's; 0: synchronous)")
    s.add_argument("--serve-gpus", required=True, dest="serve_gpus")
    s.add_argument("--serve-dp", type=int, required=True, dest="serve_dp")
    s.set_defaults(fn=cmd_player)

    s = sub.add_parser("labeller")
    common(s)
    s.add_argument("--c1", default=str(oracle.RULES), help="the oracle's playing rules")
    s.add_argument("--oracle-rules", choices=oracle.ORACLE_RULES, default="library", dest="oracle_rules",
                   help="what the oracle is told: c1 and the program library's procedure (library), or c1 alone (c1)")
    s.add_argument("--label-model", required=True, dest="label_model")
    s.add_argument("--label-effort", default="medium", dest="label_effort")
    s.add_argument("--relay-port", type=int, required=True, dest="relay_port")
    s.add_argument("--label-workers", type=int, default=256, dest="label_workers")
    s.add_argument("--label-set-workers", type=int, default=32, dest="label_set_workers")
    s.add_argument("--need-thinking", type=int, default=4, dest="need_thinking",
                   help="asks again, up to this many more times, when a label has no reasoning or makes no move")
    s.add_argument("--label-round-budget", type=float, required=True, dest="label_round_budget")
    s.add_argument("--label-run-budget", type=float, required=True, dest="label_run_budget")
    s.add_argument("--min-labelled-pct", type=float, default=40, dest="min_labelled_pct")
    s.add_argument("--hold-every", type=int, default=10, dest="hold_every")
    s.set_defaults(fn=cmd_labeller)

    def training(s: argparse.ArgumentParser) -> None:   # the trainer's settings, which it passes its train process
        common(s)
        s.add_argument("--train-gpus", required=True, dest="train_gpus",
                       help="the node's GPUs this run trains on, shared with every run given the same list")
        s.add_argument("--lag", type=int, required=True, help="the player's lag: how many phases back PG's plays are")
        s.add_argument("--lr", type=float, required=True)
        s.add_argument("--batch", type=int, default=64)
        s.add_argument("--lora-rank", type=int, default=128, dest="lora_rank")
        s.add_argument("--max-seq-len", type=int, default=65536, dest="max_seq_len")
        s.add_argument("--eval-steps", type=int, default=5, dest="eval_steps", help="steps between held-out measurements")
        s.add_argument("--patience", type=int, default=2,
                       help="measurements in a row without improving on the best that stop a phase")
        s.add_argument("--max-steps", type=int, default=120, dest="max_steps", help="a phase's cap")
        s.add_argument("--hold-every", type=int, default=10, dest="hold_every",
                       help="pg: the games held out, as the labeller holds out DAgger's and OPSD's")
        s.add_argument("--max-epochs", type=int, default=20, dest="max_epochs",
                       help="a phase's passes under the held-out stop")
        s.add_argument("--val-max-items", type=int, default=1000, dest="val_max_items",
                       help="the held-out set: whole games drawn until their calls reach this")
        s.add_argument("--skip-first", type=int, default=3, dest="skip_first",
                       help="opsd: each reply's first ids left ungraded (SDFT's code: 3)")
        s.add_argument("--max-minutes", type=float, default=240, dest="max_minutes", help="a phase's time cap")
        s.add_argument("--start-after", type=int, default=0, dest="start_after",
                       help="games played (every user's) before any phase trains: the base model plays until then")
        s.add_argument("--dose", choices=("aggregate", "recent"), default="aggregate",
                       help="a phase trains on every round's so far (aggregate), or on the last RECENT_ROUNDS rounds' "
                            "and as many from whole earlier games (recent)")
        s.add_argument("--replay-ratio", type=float, default=1.0, dest="replay_ratio",
                       help="--dose recent: earlier calls per recent call")

    s = sub.add_parser("trainer")
    training(s)
    s.set_defaults(fn=cmd_trainer)

    s = sub.add_parser("train")
    training(s)
    s.add_argument("--gpus", type=int, required=True, help="the GPUs CUDA_VISIBLE_DEVICES gives this process")
    s.set_defaults(fn=cmd_train)
    return p


def main() -> None:
    a = parser().parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
