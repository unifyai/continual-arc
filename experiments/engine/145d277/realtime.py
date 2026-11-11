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
            the round labelled (its trainer holds out the games this split would, forward.held_game).
  trainer   one process for the whole run, holding the model on its own GPUs. Phase k starts once round k is ready
            (DAgger and OPSD: labelled; PG: played), from phase k-1's adapter (the base model for phase 0). While
            rounds 0..k hold fewer than --start-after games played (episodes), phase k trains nothing and keeps the
            untrained adapter, so the base model plays on (every row's learning-starts threshold); then:
              dagger  supervised fine-tuning (train.train_sft). --dose aggregate: on every round's training side so
                      far, the held-out loss on every held-out side so far measured every EVAL_STEPS steps and the
                      phase stopped once PATIENCE measurements in a row have not improved on it, the best adapter kept
                      (DAgger's aggregate, uniform; until a held-out game exists, one pass). --dose replay: the first
                      phase that trains takes every training side so far for --first-epochs passes under that same
                      held-out stop, and every later one one pass over round k's training side and whole earlier
                      games drawn at random until they hold --replay-ratio times as many labels (dagger.replay_plan),
                      with no held-out stop, so that a phase's training stays the same size however long the stream;
              opsd    DAgger's phase with on-policy self-distillation's loss in place of imitation (train.train_sdft
                      with --demos): each call's teacher, the frozen base model, reads the oracle's reply at it as a
                      demonstration and grades a reply drawn fresh from the current student at every step (forward KL
                      over the full vocabulary, the first SKIP_FIRST ids ungraded). Everything else is DAgger's: the
                      games its labeller holds out (forward.held_game) are held out, every other call with a
                      demonstration is on the training side, either dose takes them as DAgger's takes its labels, and
                      the held-out loss, that same loss over the held-out games' calls, stops a phase and picks its
                      adapter as DAgger's does. So the two rows differ in their loss alone, and both train on every
                      episode so far, as little data as the stream gives (the author, 19 Oct 2026; SDFT's own form, one pass
                      over each round's calls with nothing held out, trained 2 to 8 steps a round);
              pg      plain policy gradient (train.train_policy_gradient) in DAgger's phase: REINFORCE with no
                      baseline, a game's reward 1 if it solved and 0 if not, so only won games train (train.pg_items),
                      every round's so far but those DAgger would hold out, as either dose takes them, every ratio
                      clipped against the adapter that played the game, whose reply log-probabilities are taken
                      once, while it exists (pg_behaviour); the held-out loss, PG's own loss over the held-out
                      games, which is their wins' clipped likelihood, stops a phase and picks its adapter as
                      DAgger's does once a held-out game has won (until then a phase makes one pass, as every row's
                      phases do before their first held-out game), the held-out wins' likelihood per token logged
                      beside it (the author, 19 Oct 2026: as much data as possible at every step, and as much training as
                      it supports, "without overfitting"; 20 Oct 2026: a win 1, a failure 0, no baseline);
            then train/pNNNN/{adapter,DONE,rounds.json} and train/latest.json.

Every stage resumes from its files: a round whose play was cut short is played again whole (the runner's own resume),
or, with paper.RESUME_CALLS, taken up from its last model call (paper.Resume: its calls replayed, its users' memory put
back as the round began, the progress line saying how), labels already written are kept, and a phase cut short is
trained again from the last finished phase's adapter.

Subcommands (on the GPU node, from ~/prime-rl: ``uv run --no-sync python ~/loop/realtime.py ...``): player,
labeller, trainer; realtime.sh passes each its settings.
"""
from __future__ import annotations

import argparse
import gzip
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


def phase_runs(run: Path, rounds: int) -> tuple[dict[int, str], list[str]]:
    """DAgger's rounds 0..rounds-1: their training sides that hold a game, by round, and their held-out sides that
    hold one (a round with no held-out game, or whose games all failed labelling, has an empty side)."""
    train_runs, dev_runs = {}, []
    for r in range(rounds):
        done = json.loads((split_dir(run, r) / "split.json").read_text())
        train_side, dev_side = sides(r)
        if done[train_side]["games"]:
            train_runs[r] = str(split_dir(run, r) / train_side)
        if done[dev_side]["games"]:
            dev_runs.append(str(split_dir(run, r) / dev_side))
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


def resumed(rd: Path) -> str:
    """How round ``rd``'s play was taken up from its last model call after an attempt was cut short (paper.Resume's
    resume.json), for the progress line; "" for a round played in one attempt."""
    f = rd / "resume.json"
    if not f.exists():
        return ""
    out = json.loads(f.read_text())
    if len(out["attempts"]) < 2:
        return ""
    total = {k: sum(c.get(k, 0) for c in out["counts"].values()) for k in ("replayed", "live", "handed_off", "drifted")}
    return (f", taken up at attempt {len(out['attempts'])}: {total['replayed']} calls replayed, {total['live']} asked of "
            f"the model, {total['handed_off']} games handed over at their last call, {total['drifted']} drifted")


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
        say(run, f"play round {r} done ({have}/{want}){resumed(rd)}")
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
    the pass count, DAgger's held-out runs and OPSD's demonstrations are each phase's own (:func:`train_phase`)."""
    import train
    common = dict(model=loop.MODEL, lr=a.lr, batch=a.batch, gpus=a.gpus, rank=a.lora_rank, alpha=a.lora_rank,
                  max_seq_len=a.max_seq_len, chain_slack=train.CHAIN_SLACK, seed=0, epochs=1, max_steps=None,
                  max_minutes=a.max_minutes, dry_run=False)
    if a.row == "dagger":
        return argparse.Namespace(**{**common, "max_steps": a.max_steps}, objective=train.Objective.sft, chains=True,
                                  eval_steps=a.eval_steps, patience=a.patience, val_run=None,
                                  val_max_items=a.val_max_items, divergence="forward_kl")
    if a.row == "opsd":
        # SDFT's loss (paper.sh's own OPSD flags): the frozen base model as teacher, no anchor, forward KL, fresh
        # replies from the server every step; DAgger's cap and held-out stop, over the held-out games' calls
        # (train_phase hands them over), every one of them scored
        return argparse.Namespace(**{**common, "max_steps": a.max_steps}, objective=train.Objective.sdft, demos=None,
                                  ema_decay=None, teacher_adapter=None, anchor_mu=0.0, divergence="forward_kl",
                                  skip_first=a.skip_first, on_policy=True, server=SERVER, placement="message",
                                  val_run=None, eval_every=a.eval_steps, patience=a.patience, eval_items=0,
                                  no_initial_eval=False)
    # DAgger's cap and held-out stop, over the held-out games' calls (train_phase hands them over)
    return argparse.Namespace(**{**common, "max_steps": a.max_steps}, objective=train.Objective.pg, per_call=False,
                              eval_steps=a.eval_steps, patience=a.patience)


def games_played(run: Path, rounds: int) -> int:
    """How many games rounds 0..rounds-1 have played, every user's (their captured games)."""
    return sum(len(list((round_dir(run, r) / "captured").glob("u*.json.gz"))) for r in range(rounds))


def train_phase(a: argparse.Namespace, run: Path, targs: argparse.Namespace, trainer: Any, k: int, out: Path,
                init: Path | None) -> dict[str, Any]:
    """Phase ``k``, its row's training (the module docstring) from ``init``, into ``out``: DONE's summary."""
    import train
    rng = random.Random(k)
    targs.seed = k
    played_games = games_played(run, k + 1)
    if played_games < a.start_after:
        trainer.reset()
        trainer.save(out / "adapter")
        return {"skipped": True, "games": played_games, "start_after": a.start_after}
    if a.row == "dagger":
        train_runs, dev_runs = phase_runs(run, k + 1)
        per_round, compactions = {}, 0
        for r, path in train_runs.items():
            per_round[r], left_out = train.load_calls(Path(path))
            compactions += left_out
        calls, targs.epochs, stopped = dose(a, k, init, per_round, bool(dev_runs))
        targs.val_run = dev_runs if stopped else None
        say(run, f"phase {k} started: rounds 0-{k}, {len(calls)} labels, {len(dev_runs)} held-out sides, "
                 f"{targs.epochs} pass(es){'' if stopped else ', no held-out stop'}")
        return train.train_sft(targs, trainer, calls, compactions, out, init, rng)
    if a.row == "opsd":
        per_round, held, compactions, demos = opsd_rounds(a, run, k, trainer.tokenizer)
        calls, targs.epochs, stopped = dose(a, k, init, per_round, bool(held))
        targs.demos = [str(labels_dir(run, r)) for r in range(k + 1)]
        say(run, f"phase {k} started: rounds 0-{k}, {len(calls)} calls with a demonstration, {len(held)} held-out "
                 f"games, {targs.epochs} pass(es){'' if stopped else ', no held-out stop'}")
        return train.train_sdft(targs, trainer, calls, compactions, out, init, rng,
                                val_calls=train.whole_games(held, a.val_max_items, k) if stopped else None,
                                demo_stats=demos)
    per_round, held, compactions, logps = pg_rounds(a, run, k, trainer, targs)
    # a held-out failure weighs 0 in PG's loss, so only a held-out win can say whether training generalises
    won = [game for game in held if game[0].meta["solved"]]
    calls, targs.epochs, stopped = dose(a, k, init, per_round, bool(won))
    items, games = train.pg_items(trainer.tokenizer, calls, targs.max_seq_len)
    val = (train.pg_items(trainer.tokenizer, train.whole_games(won, a.val_max_items, k), targs.max_seq_len)[0]
           if stopped else [])
    train.attach_behaviour(items + val, logps)
    say(run, f"phase {k} started: rounds 0-{k}, {games['wins']} of {games['episodes']} games won, {len(items)} "
             f"calls, {len(held)} held-out games ({len(won)} won), {targs.epochs} pass(es)"
             f"{'' if stopped else ', no held-out stop'}")
    return train.train_policy_gradient(targs, trainer, calls, lambda tok: (items, games), "games", compactions, out,
                                       init, None, rng, val_items=val)


def dose(a: argparse.Namespace, k: int, init: Path | None, per_round: dict[int, list[Any]],
         held_out: bool) -> tuple[list[Any], int, bool]:
    """Phase ``k``'s training calls under ``--dose``, DAgger's and OPSD's alike, from ``per_round`` (each round's
    training side so far): the calls, their passes, and whether the held-out stop runs (the module docstring)."""
    import dagger
    every = [c for r in sorted(per_round) for c in per_round[r]]
    if a.dose == "aggregate":
        return every, a.max_epochs if held_out else 1, held_out   # no held-out game yet: one pass, nothing to stop on
    if init is None or paper.adapter_is_untrained(init):   # the replay dose's first phase that trains
        return every, a.first_epochs, held_out
    earlier = [c for r in sorted(per_round) if r != k for c in per_round[r]]
    return dagger.replay_plan(per_round.get(k, []), earlier, True, a.replay_ratio, random.Random(1000 + k)), 1, False


def opsd_rounds(a: argparse.Namespace, run: Path, k: int,
                tok: Any) -> tuple[dict[int, list[Any]], list[list[Any]], int, dict[str, int]]:
    """OPSD's rounds 0..k split as DAgger's labeller splits its (forward.held_game, one game in ``--hold-every``): each
    round's training-side calls with a demonstration (train.attach_demos), the held-out games' calls with theirs, a
    list per game, the compaction calls left out, and the demonstrations' counts."""
    import forward
    import train
    per_round, held, compactions = {}, [], 0
    stats = {"calls": 0, "with_demo": 0, "no_demo": 0, "no_move": 0, "unrendered": 0}
    for r in range(k + 1):
        got, left_out = train.load_calls(round_dir(run, r))
        compactions += left_out
        got, got_stats = train.attach_demos(tok, got, labels_dir(run, r))
        for key in stats:
            stats[key] += got_stats[key]
        per_round[r] = [c for c in got if not forward.held_game(r, c.meta["user"], a.hold_every)]
        held += train.by_game([c for c in got if forward.held_game(r, c.meta["user"], a.hold_every)])
    return per_round, held, compactions, stats


def pg_rounds(a: argparse.Namespace, run: Path, k: int, trainer: Any,
              targs: argparse.Namespace) -> tuple[dict[int, list[Any]], list[list[Any]], int, dict[tuple, list[float]]]:
    """PG's rounds 0..k split as DAgger's labeller splits its (forward.held_game, one game in ``--hold-every``): each
    round's training-side calls, the held-out games' calls (a list per game), the compaction calls left out, and every
    won game's calls' reply log-probabilities under the adapter that played its round (:func:`pg_behaviour`), the only
    calls PG trains on or measures."""
    import forward
    import train
    per_round, held, compactions, logps = {}, [], 0, {}
    for r in range(k + 1):
        got, left_out = train.load_calls(round_dir(run, r))
        compactions += left_out
        logps.update(pg_behaviour(run, r, [c for c in got if c.meta["solved"]], trainer, targs))
        per_round[r] = [c for c in got if not forward.held_game(r, c.meta["user"], a.hold_every)]
        held += train.by_game([c for c in got if forward.held_game(r, c.meta["user"], a.hold_every)])
    return per_round, held, compactions, logps


def pg_behaviour(run: Path, r: int, calls: list[Any], trainer: Any,
                 targs: argparse.Namespace) -> dict[tuple, list[float]]:
    """Round ``r``'s ``calls``' reply log-probabilities under the adapter that played the round (its served.json's
    phase, the base model before any phase trained), keyed as train.behaviour_logps keys them: read from the round's
    pg_behaviour.json.gz, or taken now (train.behaviour_logps) and written there; none for a round with no calls given
    (no game won). A round is first trained in phase ``r``, while the adapter that played it, ``LAG`` phases older, is
    still on disk (prune keeps it), so every phase after can take its ratios against the adapter that played each
    game, however long ago it was pruned."""
    import train
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
    if adapter is None or paper.adapter_is_untrained(adapter):
        trainer.configure_behaviour(base=True)
    else:
        trainer.configure_behaviour(adapter)
    got = train.behaviour_logps(trainer, targs, train.played_items(trainer.tokenizer, calls, targs.max_seq_len))
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as fp:
        json.dump({"played_by": played, "logps": {f"{u}:{i}": logps for (_, u, i), logps in got.items()}}, fp)
    tmp.replace(path)
    return got


def phase_line(row: str, done: dict[str, Any]) -> str:
    """A finished phase's summary for the progress log."""
    if done.get("skipped"):
        return f"not trained: {done['games']} games played, training starts at {done['start_after']}"
    if row == "dagger":
        dev = [e["dev_nll"] for e in done["dev"]]
        held = f", held-out {dev[0]} to {min(dev)} (best at step {done['best_step']})" if dev else ""
        return f"{done['steps']} steps in {done['minutes']} min{held}"
    if row == "pg":
        games, held = done["games"], done["held_out"]
        held_line = f", held-out {held[0]} to {min(held)} (best at step {done['best_step']})" if held else ""
        return (f"{games['wins']}/{games['episodes']} games won and trained on, {done['steps']} steps in "
                f"{done['minutes']} min{held_line}")
    held = done["held_out"]
    held_line = f", held-out {held[0]} to {min(held)} (best at step {done['best_step']})" if held else ""
    return f"{done['items'].get('drawn', 0)} replies drawn, {done['steps']} steps{held_line}"


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
    s.add_argument("--eval-steps", type=int, default=5, dest="eval_steps",
                   help="dagger, opsd and pg: steps between held-out checks")
    s.add_argument("--patience", type=int, default=2,
                   help="dagger, opsd and pg: checks in a row without improving that stop a phase")
    s.add_argument("--max-steps", type=int, default=120, dest="max_steps", help="dagger, opsd and pg: a phase's safety cap")
    s.add_argument("--hold-every", type=int, default=10, dest="hold_every",
                   help="opsd and pg: the games held out, as DAgger's labeller holds them out (forward.held_game)")
    s.add_argument("--max-epochs", type=int, default=20, dest="max_epochs")
    s.add_argument("--val-max-items", type=int, default=1000, dest="val_max_items")
    s.add_argument("--skip-first", type=int, default=3, dest="skip_first",
                   help="opsd: each reply's first ids left ungraded (SDFT's code: 3)")
    s.add_argument("--max-minutes", type=float, default=240, dest="max_minutes")
    s.add_argument("--start-after", type=int, default=0, dest="start_after",
                   help="games played (every user's) before any phase trains: the base model plays until then")
    s.add_argument("--dose", choices=("aggregate", "replay"), default="aggregate",
                   help="every row: a phase trains on every round's so far (aggregate) or on its round's and a "
                        "replay of earlier games (replay)")
    s.add_argument("--replay-ratio", type=float, default=1.0, dest="replay_ratio",
                   help="--dose replay: earlier calls replayed per new call")
    s.add_argument("--first-epochs", type=int, default=2, dest="first_epochs",
                   help="--dose replay: passes of the first phase that trains, under the held-out stop")
    s.set_defaults(fn=cmd_trainer)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
