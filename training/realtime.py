"""The paper's trained rows, DAgger, OPSD and PG, with a fixed lag: the student plays `arc1-small` served to its users
round after round while the oracle labels its games (DAgger and OPSD) and one resident trainer trains on them, and
round r is played by the model trained on rounds 0..r-1-LAG, no more and no less (unify fixes the lag at one round,
LAG=1, so that which model plays which round never depends on how fast the hardware is; LAG=0 is the strict,
synchronous protocol). Play waits for that model when training is behind; training never waits for play.

A run plays --rounds rounds of the schedule (paper.full_schedule): realtime.sh's default is the stream's 200, after
which frozen.sh plays the run's two frozen tests from its end-of-stream state (frozen.py), which transfer, the
benchmark's headline, reads. The schedule goes on with the epilogue's 100 rounds, in which every user meets the 100
held-out tasks once each; a run given 300 rounds plays, labels and trains on them as on the stream, each user's harness
memory carried over, and no measure reads them.

Three loops (two for PG, which needs no oracle) run side by side on one node, each its own process, handing each other
files (realtime.sh starts them, restarts one that dies, and stops the run once all of them are done):

  player    round r = 0, 1, ...: phase r-1-LAG's adapter (the base model while r-1-LAG < 0), waited for until the
            trainer has finished it, is loaded into the model server, which stays up on its own GPUs, checked
            (paper.check_served_adapter: vLLM silently serves the base model under an adapter's name when its layers
            are misnamed), named in round_dir(r)/served.json, and plays round r (paper.play_round over
            paper.schedule_cut(r)); then round_dir(r)/play.done. Phase k trains on rounds 0..k exactly, so in the
            notation round r is played by pi_(r-1-LAG), which has learnt from every round but the LAG before r.
  labeller  DAgger and OPSD alike, round by round: the oracle's own reply to every player call of the round's games as
            they land (forward.py points/label: each game is asked once while play goes on, then the whole round once
            more when it is played, so a point that failed gets a second ask), under the working rules --oracle-rules
            names (oracle_hint.oracle_rules: c1 and the program library's procedure, or c1 alone), kept in the run's
            guidelines.md, which a labeller started again must find unchanged, since rounds labelled under other rules
            would change the oracle mid-run; then the label set (forward.py label-set: each label as the student
            writes it, a label that makes no move or that the template cannot render left out) and its split into a
            training side and a held-out side (split --hold-every: held_game's one game in HOLD_EVERY, played and
            scored like any other), round_dir(r)/labels/split/split.json, written last, marking the round labelled.
  trainer   the run's phases, trained by a ``train`` process that holds the model on the GPUs the node's pool of
            --train-gpus gives the run (gpupool.py: runs given the same TRAIN_GPUS share them, a phase taking an even
            share of the runs that have one to train, and a run alone holds all of them, as one resident process).
            Phase k starts once round k is ready (DAgger and OPSD: labelled; PG: played), from phase k-1's adapter (the
            base model for the first phase that trains). While rounds 0..k hold fewer than --start-after games played
            (episodes), phase k trains nothing, saves no adapter and takes no GPU, written by the trainer itself, so the
            base model plays on (every row's learning-starts threshold). Then every row's phase is one setup, only its
            update the row's own (train.fit: OPSD is DAgger with its distillation step in place of imitation, and every
            row trains on every game so far, as much data as the stream gives): the row's calls of rounds 0..k under
            --dose (aggregate: every round's so far; replay: the
            first phase that trains takes every one so far for --first-epochs passes, every later one round k's and
            whole earlier games drawn at random until they hold --replay-ratio times as many, dagger.replay_plan, with
            no held-out stop, so that a phase's training stays the same size however long the stream; recent: the last
            RECENT_ROUNDS rounds' and whole earlier games' drawn the same way, under the held-out stop, which lowered
            the loss on the rounds still to come where every round's did not, docs/method-search.md "Why DAgger's
            phases stop"), the games
            forward.held_game holds out (one in HOLD_EVERY), the held-out loss measured every EVAL_STEPS steps and the
            phase stopped once PATIENCE measurements in a row have not improved on it, the best adapter kept (until a
            held-out game exists, one pass), at most MAX_STEPS steps, and the steps planned alike (train.plan). The
            rows' calls and updates:
              dagger  the label sets' training sides, each label imitated (train.Imitation: supervised fine-tuning,
                      every token weighing alike); the held-out sides' targets' likelihood is the held-out loss;
              opsd    the same calls, so the same steps, with SDFT's loss in place of imitation
                      (train.SelfDistillation): at each call the current student's own reply, drawn fresh, graded by
                      the frozen base model reading the call's label as a demonstration (forward KL over the full
                      vocabulary, the first SKIP_FIRST ids ungraded, each reply weighing alike); the held-out loss is
                      that loss over the held-out sides' calls as they were played (plays). So the two rows differ
                      in their update alone (docs/method-search.md, "The OPSD audit");
              pg      every round's played calls but those of the games held out (pg_rounds), REINFORCE with no
                      baseline (train.PolicyGradient over train.pg_items): a game's reward 1 if it solved and 0 if
                      not, so only won games train, every ratio clipped against the adapter that played the game,
                      whose reply log-probabilities are taken once, while it exists (pg_behaviour); its held-out loss
                      is its own loss over the held-out games, their wins' clipped likelihood, so its stop runs once a
                      held-out game has won, the held-out wins' likelihood per token logged beside it;
            then train/pNNNN/{adapter,DONE,rounds.json} and train/latest.json. A ``train`` process holds its GPUs
            between phases until another run's phase needs them, gives them up to such a phase at its next step
            checkpoint (with checkpoint.ON; without, at its phase's end), and is started again on its new share, the
            phase going on from its checkpoint; it takes more only when its next phase starts. A phase's steps never
            depend on how many GPUs train it (train.plan), so its checkpoint resumes on any number of them.

Every stage resumes from its files: a round whose play was cut short is played again whole (the runner's own resume),
or, with paper.RESUME_CALLS, taken up from its last model call (paper.Resume: its calls replayed, its users' memory put
back as the round began, the progress line saying how), labels already written are kept, and a phase cut short is
trained again from the last finished phase's adapter, or, with checkpoint.ON, goes on from its last
step checkpoint (checkpoint.py: one on SIGTERM, Nebius's notice that it is taking the machine back, and one every
checkpoint.EVERY steps for a stop with no notice in time), on the same batches in the same order, so a stop costs at most
the steps since, and one stopped after its DONE only has its rounds.json written.

Subcommands (on the GPU node, from prime-rl's checkout: ``uv run --no-sync python
~/continual-arc/training/realtime.py ...``): player, labeller, trainer; realtime.sh passes each its settings, and the
trainer runs ``train`` on its GPUs.
"""
from __future__ import annotations

import argparse
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkpoint  # noqa: E402
import gpupool  # noqa: E402
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
    sides that hold one (a round with no held-out game, or whose games all failed labelling, has an empty side)."""
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
    ``phase`` is None or saved no adapter (the warm-up). An adapter that fails the served-adapter check stops the player
    (``SystemExit``, with a line in the progress log): the protocol names the model each round plays on, and a round
    played on another would break it, as one played on base weights served under a trained adapter's name would."""
    from ask_sd.server import Server
    serve_sh = ["bash", str(HERE / "serve.sh"), a.serve_gpus, str(a.serve_dp)]
    if phase is None or phase["adapter"] is None:
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
                                        "trained_rounds": need + 1 if phase and phase["adapter"] else 0, "lag": a.lag,
                                        "at": datetime.now(timezone.utc).isoformat()})
        schedule = paper.schedule_cut(r, users=a.users)
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
    """Round ``round_``'s label set and its split (split.json, written last, marks the round labelled), DAgger's and
    OPSD's alike."""
    import forward
    out = labels_dir(run, round_)
    shutil.rmtree(out / "set", ignore_errors=True)   # a set half-written by a killed attempt is rebuilt whole
    forward.cmd_label_set(argparse.Namespace(out=str(out), model=loop.MODEL, workers=a.label_set_workers))
    split = split_dir(run, round_)
    shutil.rmtree(split, ignore_errors=True)
    forward.cmd_split(argparse.Namespace(set=str(out / "set"), held=None, hold_every=a.hold_every, round=round_,
                                         sides=",".join(sides(round_)), out=str(split)))
    return json.loads((split / "split.json").read_text())


def started_oracle_rules(run: Path) -> str:
    """What the run's oracle is told (realtime.sh's ORACLE_RULES), as the run's last "started:" line names it, the
    line a relaunch takes the run's settings from; ``library``, realtime.sh's default, for a line that names none (the
    nine launch runs' lines were written before realtime.sh named it)."""
    started = re.findall(r"realtime\.sh: started: .*", (run / "progress").read_text())[-1]
    named = re.search(r", oracle rules ([a-z0-9]+)(?:, patience \d+)?$", started)
    return named[1] if named else "library"


def cmd_labeller(a: argparse.Namespace) -> None:
    import oracle_hint
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = Path(a.run).expanduser()
    guidelines = run / "guidelines.md"
    rules = oracle_hint.oracle_rules(Path(a.c1).expanduser().read_text(), a.oracle_rules)
    # a relaunch without the run's own ORACLE_RULES, or a c1 changed under it, would label every later round under
    # other rules than the earlier ones, the procedure told to an oracle that was not told it or the other way round
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
                    label_pass(a, run, r, guidelines, asked, final=True)
                    break
                except SystemExit as error:
                    say(run, f"round {r}'s labels, attempt {attempt}: {error}")
                    if attempt == 3:
                        raise
                    time.sleep(300)
            say(run, f"round {r} labelled: {json.dumps(finish_round(a, run, r), separators=(',', ':'))}")
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
    adapter's load pushes it out of the server's cache (a 1-user round once outlasted three phases, and the reload of
    its pruned adapter killed the server under all four runs on the node)."""
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
    """Every row's training settings, one configuration for all three (train.fit, and of the updates OPSD's alone reads
    the server it draws its replies from and the ids it leaves ungraded); each phase sets its own seed and passes
    (:func:`train_phase`). ``gpus`` is the train process's own; nothing a phase trains on depends on it. The adapter
    trains ask-sd's default projections (``train.LORA_TARGETS``), as every trained row's has."""
    import train
    return argparse.Namespace(model=loop.MODEL, lr=a.lr, batch=a.batch, gpus=a.gpus, rank=a.lora_rank,
                              alpha=a.lora_rank, lora_targets=list(train.LORA_TARGETS), max_seq_len=a.max_seq_len,
                              chains=True, per_call=False, chain_slack=train.CHAIN_SLACK, chain_calls=train.CHAIN_CALLS,
                              seed=0, epochs=1,
                              max_steps=a.max_steps, max_minutes=a.max_minutes, eval_steps=a.eval_steps,
                              patience=a.patience, divergence="forward_kl", skip_first=a.skip_first, server=SERVER)


def games_played(run: Path, rounds: int) -> int:
    """How many games rounds 0..rounds-1 have played, every user's (their captured games)."""
    return sum(len(list((round_dir(run, r) / "captured").glob("u*.json.gz"))) for r in range(rounds))


def warming_up(run: Path, k: int, start_after: int) -> bool:
    """Whether phase ``k`` trains nothing, rounds 0..k having played fewer than ``start_after`` games: it saves no
    adapter, and the round it is for plays on the base model. The count only grows with k, so every such phase comes
    before the first that trains, and the model it leaves in place is the base model."""
    return games_played(run, k + 1) < start_after


def train_phase(a: argparse.Namespace, run: Path, targs: argparse.Namespace, trainer: Any, k: int, out: Path,
                init: Path | None) -> dict[str, Any]:
    """Phase ``k`` from ``init`` (the base model when None) into ``out``, every row's one setup (the module docstring):
    the row's calls of rounds 0..k under --dose, the held-out games' calls for the held-out stop once one exists, and
    train.fit's plan, steps and stop, with only the update the row's own (:func:`row_update`). DONE's summary."""
    import train
    rng = random.Random(k)
    targs.seed = k
    if a.row == "pg":
        per_round, held, logps = pg_rounds(a, run, k, trainer, targs)
        # PG's loss weighs a held-out failure 0, so only a held-out win can say whether its training generalises
        stops = [game for game in held if game[0].meta["solved"]]
    else:
        (per_round, held), logps = labelled_rounds(run, k), {}
        stops = held
    calls, targs.epochs, stopped = dose(a, k, init, per_round, bool(stops))
    update = row_update(a, run, targs, trainer.tokenizer, calls,
                        train.whole_games(stops, a.val_max_items, k) if stopped else [], logps)
    trains = (f"{update.stats['wins']} of {update.stats['episodes']} games won, {len(update.units)} calls, "
              f"{len(held)} held-out games ({len(stops)} won)" if a.row == "pg"
              else f"{len(calls)} labels, {len(held)} held-out games")
    say(run, f"phase {k} started: rounds 0-{k}, {trains}, {targs.epochs} pass(es)"
             f"{'' if stopped else ', no held-out stop'}")
    return train.fit(targs, trainer, update, out, init, rng)


def row_update(a: argparse.Namespace, run: Path, targs: argparse.Namespace, tok: Any, calls: list[Any],
               held: list[Any], logps: dict[tuple, list[float]]) -> Any:
    """The row's update (the module docstring) over a phase's calls and its held-out calls: DAgger's imitation of the
    labels, OPSD's self-distillation from the same labels, PG's policy gradient on its won games' calls, each
    clipped against the adapter that played it (``logps``)."""
    import train
    if a.row == "dagger":
        return train.Imitation(targs, tok, calls, held)
    if a.row == "opsd":
        return train.SelfDistillation(targs, tok, calls, plays(run, held, tok))
    items, games = train.pg_items(tok, calls, targs.max_seq_len)
    val = train.pg_items(tok, held, targs.max_seq_len)[0]
    train.attach_behaviour(items + val, logps)
    return train.PolicyGradient(targs, train.Objective.pg, items, games, "games", val)


# the recent dose's window: a phase trains on the last RECENT_ROUNDS rounds' labels and as many earlier games'
RECENT_ROUNDS = 10


def dose(a: argparse.Namespace, k: int, init: Path | None, per_round: dict[int, list[Any]],
         held_out: bool) -> tuple[list[Any], int, bool]:
    """Phase ``k``'s training calls under ``--dose``, every row's alike, from ``per_round`` (each round's training side
    so far): the calls, their passes, and whether the held-out stop runs (the module docstring)."""
    import dagger
    every = [c for r in sorted(per_round) for c in per_round[r]]
    if a.dose == "aggregate":
        return every, a.max_epochs if held_out else 1, held_out   # no held-out game yet: one pass, nothing to stop on
    if a.dose == "recent":
        since = k + 1 - RECENT_ROUNDS
        recent = [c for r in sorted(per_round) if r >= since for c in per_round[r]]
        earlier = [c for r in sorted(per_round) if r < since for c in per_round[r]]
        return (dagger.replay_plan(recent, earlier, True, a.replay_ratio, random.Random(1000 + k)),
                a.max_epochs if held_out else 1, held_out)
    if init is None:   # the replay dose's first phase that trains
        return every, a.first_epochs, held_out
    earlier = [c for r in sorted(per_round) if r != k for c in per_round[r]]
    return dagger.replay_plan(per_round.get(k, []), earlier, True, a.replay_ratio, random.Random(1000 + k)), 1, False


def labelled_rounds(run: Path, k: int) -> tuple[dict[int, list[Any]], list[list[Any]]]:
    """DAgger's and OPSD's rounds 0..k, as the labeller split their label sets (:func:`phase_runs`): each round's
    training side's calls, by round, and every held-out side's games, a list of calls each."""
    import train
    train_runs, dev_runs = phase_runs(run, k + 1)
    return ({r: train.load_calls(Path(path))[0] for r, path in train_runs.items()},
            [game for path in dev_runs for game in train.by_game(train.load_calls(Path(path))[0])])


def plays(run: Path, calls: list[Any], tok: Any) -> list[Any]:
    """Labelled calls (a label set's) as they were played: each the play's own call at that point (its round's captured
    game), whose served prompt and played reply OPSD's held-out loss scores, carrying the label's demonstration
    (train.demo). A labelled call whose response came back without its token ids has no played reply to score:
    forward.points labels it, train.load_calls leaves it out."""
    import dagger
    import train
    played: dict[int, dict[tuple[int, int], Any]] = {}
    out = []
    for call in calls:
        r = int(call.meta["run"].rsplit("-r", 1)[1])   # its side's round (sides)
        if r not in played:
            played[r] = {(c.meta["user"], c.meta["i"]): c for c in train.load_calls(round_dir(run, r))[0]}
        play = played[r].get(dagger.source(call.meta["call_id"]))
        if play is None:
            continue
        play.meta["demo"] = train.demo(tok, call)
        out.append(play)
    return out


def pg_rounds(a: argparse.Namespace, run: Path, k: int, trainer: Any,
              targs: argparse.Namespace) -> tuple[dict[int, list[Any]], list[list[Any]], dict[tuple, list[float]]]:
    """PG's rounds 0..k split as the labeller splits DAgger's and OPSD's (forward.held_game, one game in
    ``--hold-every``): each round's training-side calls, the held-out games' calls (a list per game), and every won
    game's calls' reply log-probabilities under the adapter that played its round (:func:`pg_behaviour`), the only
    calls PG trains on or measures."""
    import forward
    import train
    per_round, held, logps = {}, [], {}
    for r in range(k + 1):
        got, _ = train.load_calls(round_dir(run, r))
        logps.update(pg_behaviour(run, r, [c for c in got if c.meta["solved"]], trainer, targs, a.start_after))
        per_round[r] = [c for c in got if not forward.held_game(r, c.meta["user"], a.hold_every)]
        held += train.by_game([c for c in got if forward.held_game(r, c.meta["user"], a.hold_every)])
    return per_round, held, logps


def pg_behaviour(run: Path, r: int, calls: list[Any], trainer: Any, targs: argparse.Namespace,
                 start_after: int) -> dict[tuple, list[float]]:
    """Round ``r``'s ``calls``' reply log-probabilities under the adapter that played the round (its served.json's
    phase, the base model before any phase trained), keyed as train.behaviour_logps keys them: read from the round's
    pg_behaviour.json.gz, or taken now (train.behaviour_logps) and written there; none for a round with no calls given
    (no game won). A round played after the first phase that trains is first trained in phase ``r``, while the
    adapter that played it, ``LAG`` phases older, is still on disk (prune keeps it), so every phase after can take its
    ratios against the adapter that played each game, however long ago it was pruned. A round played in the warm-up
    (:func:`warming_up`, under ``start_after``) is first trained in the first phase that trains, by when prune has
    deleted most of the warm-up's adapters; its player trained nothing, so it served the base model, which is what
    the ratios are taken against, without reading its adapter."""
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
    if adapter is None or warming_up(run, played, start_after) or paper.adapter_is_untrained(adapter):
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


REGROW = 75   # a train process's end when the pool's share for its next phase is not the GPUs it holds
SUPERVISE_S = 5


def active(run: Path) -> Path:
    """The file a train process keeps while a phase is in flight (its number and GPUs), for the trainer to see."""
    return run / "train" / "active.json"


def cmd_train(a: argparse.Namespace) -> None:
    """The run's phases on this process's GPUs (CUDA_VISIBLE_DEVICES, --gpus of them), which the run's trainer took
    from the node's pool (:func:`cmd_trainer`): the model loaded once, for every phase the pool lets it keep them.
    Before each phase it asks the pool for the run's share were it to start now (gpupool.Pool.share_if_starting), and
    ends with REGROW if that is not the GPUs it holds, for the trainer to start it again on its share; the warm-up's
    phases it leaves to the trainer, which needs no GPU for them."""
    import fcntl
    import kernel_stack
    import train
    checkpoint.catch_sigterm()   # a phase in flight checkpoints after its step and exits; between phases it just exits
    run = Path(a.run).expanduser()
    root = run / "train"
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / ".train.lock", "a")   # one train process a run: a second waits for the first to end
    fcntl.flock(lock, fcntl.LOCK_EX)
    mine = gpupool.parse(os.environ["CUDA_VISIBLE_DEVICES"])
    pool = gpupool.Pool(gpupool.parse(a.train_gpus), run.name)
    targs = train_args(a)
    trainer = train.make_trainer(targs)
    kernel_stack.record(root)
    while True:
        k = next_phase(run)
        if k == a.rounds:
            break
        if not ready(run, k, a.row) or warming_up(run, k, a.start_after):
            if checkpoint.stopping():
                checkpoint.leave(None)
            time.sleep(POLL_S)
            continue
        if checkpoint.stopping():   # SIGTERM between phases: nothing to save
            checkpoint.leave(None)
        share = pool.share_if_starting()
        if share != mine:
            print(f"phase {k}: the pool's share is GPUs {share}, this process holds {mine}: ending for the trainer to "
                  f"start one on its share", flush=True)
            sys.exit(REGROW)
        write_json(active(run), {"phase": k, "gpus": mine, "pid": os.getpid()})
        out = phase_dir(run, k)
        if checkpoint.ON and (out / "DONE").exists():   # trained to its end, its adapter in place, before it stopped
            done = json.loads((out / "DONE").read_text())
            shutil.rmtree(out / "checkpoints", ignore_errors=True)   # the stop came before they went
        else:
            step = checkpoint.saved_step(out)
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
         asked: bool, others_hold: set[int], checkpoints: bool) -> str:
    """What the run's trainer does with its train process now, from the pool's split (``share``, every run's): "stop"
    it when another run's share takes GPUs it holds (at once between phases; mid-phase only with step ``checkpoints``,
    since without them the phase would start again from nothing: it then ends at the phase's end, REGROW), "start" one
    when the run wants GPUs and holds none and no other run's process holds any of its share, or "wait"."""
    if alive:
        theirs = {g for r, gpus in share.items() if r != run for g in gpus}
        if set(held) & theirs and not asked and (not busy or checkpoints):
            return "stop"
        return "wait"
    if want and share[run] and not set(share[run]) & others_hold:
        return "start"
    return "wait"


def cmd_trainer(a: argparse.Namespace) -> None:
    """The run's trainer, as realtime.sh supervises it: the warm-up's phases written here, with no GPU, and every phase
    that trains run by a ``train`` process (:func:`cmd_train`) on the GPUs the node's pool of --train-gpus gives the run
    (gpupool.py), started when the run has a phase to train and none runs, stopped when another run's phase needs its
    GPUs (:func:`tick`), and started again on its new share; its own end ends the run's training (trainer.done). A train
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
    argv = [sys.executable, str(HERE / "realtime.py"), "train", *sys.argv[sys.argv.index("trainer") + 1:]]
    try:
        while True:
            if child is not None and child.poll() is not None:
                code = child.returncode
                active(run).unlink(missing_ok=True)
                if code == 0:
                    break
                if not (code == REGROW or (asked and code in (checkpoint.SIGTERM_EXIT, -signal.SIGTERM))):
                    raise SystemExit(f"the train process on GPUs {held} ended with {code}")
                child, held, asked = None, [], False
            if stopping:   # the run's own SIGTERM: its train process checkpoints (or ends) and this one with it
                if child is not None:
                    child.send_signal(signal.SIGTERM)
                    child.wait()
                raise SystemExit(checkpoint.SIGTERM_EXIT)
            k = next_phase(run)
            if k == a.rounds and child is None:
                break
            if child is None and k < a.rounds and ready(run, k, a.row) and warming_up(run, k, a.start_after):
                skip_phase(a, run, k)
                continue
            want = k < a.rounds and ready(run, k, a.row) and not warming_up(run, k, a.start_after)
            busy = child is not None and active(run).exists()
            view = pool.update(want, busy, held, child.pid if child else None)
            if orphan and gpupool._alive(orphan.get("pid")):   # it checkpoints (or ends) before this run starts another
                os.kill(orphan["pid"], signal.SIGTERM)
                while gpupool._alive(orphan["pid"]):
                    time.sleep(1)
            orphan = None
            action = tick(run.name, view["share"], held, busy, want, child is not None, asked,
                          pool.held_by_others(view["share"][run.name], view["runs"]), checkpoint.ON)
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


# -- CLI -------------------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--run", required=True, help="the run's folder (realtime.sh's RUNS/rt/NAME)")
        s.add_argument("--rounds", type=int, required=True)
        s.add_argument("--row", choices=ROWS, required=True)

    s = sub.add_parser("player")
    common(s)
    s.add_argument("--users", type=int, required=True, help="the users the schedule is served to (paper.served)")
    s.add_argument("--name", required=True, help="the run's name, its adapters' prefix on the server")
    s.add_argument("--lag", type=int, required=True,
                   help="round r plays on the phase trained on rounds 0..r-1-LAG (1: the paper's; 0: synchronous)")
    s.add_argument("--serve-gpus", required=True, dest="serve_gpus")
    s.add_argument("--serve-dp", type=int, required=True, dest="serve_dp")
    s.set_defaults(fn=cmd_player)

    s = sub.add_parser("labeller")
    common(s)
    s.add_argument("--c1", required=True, help="the oracle's playing rules")
    s.add_argument("--oracle-rules", choices=("library", "c1"), default="library", dest="oracle_rules",
                   help="what the oracle is told (oracle_hint.oracle_rules): c1 and the program library's procedure "
                        "(library), or c1 alone (c1)")
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

    def training(s: argparse.ArgumentParser) -> None:   # the trainer's settings, which it passes its train process
        common(s)
        s.add_argument("--train-gpus", required=True, dest="train_gpus",
                       help="the node's GPUs this run trains on, shared with every run given the same list (gpupool.py)")
        s.add_argument("--lag", type=int, required=True, help="the player's lag: how many phases back PG's plays are")
        s.add_argument("--lr", type=float, required=True)
        s.add_argument("--batch", type=int, default=64)
        s.add_argument("--lora-rank", type=int, default=128, dest="lora_rank")
        s.add_argument("--max-seq-len", type=int, default=65536, dest="max_seq_len")
        s.add_argument("--eval-steps", type=int, default=5, dest="eval_steps",
                       help="dagger, opsd and pg: steps between held-out checks")
        s.add_argument("--patience", type=int, default=2,
                       help="dagger, opsd and pg: checks in a row without improving that stop a phase")
        s.add_argument("--max-steps", type=int, default=120, dest="max_steps", help="dagger, opsd and pg: a phase's safety cap")
        s.add_argument("--hold-every", type=int, default=10, dest="hold_every",
                       help="pg: the games held out, as the labeller holds out DAgger's and OPSD's (forward.held_game)")
        s.add_argument("--max-epochs", type=int, default=20, dest="max_epochs")
        s.add_argument("--val-max-items", type=int, default=1000, dest="val_max_items")
        s.add_argument("--skip-first", type=int, default=3, dest="skip_first",
                       help="opsd: each reply's first ids left ungraded (SDFT's code: 3)")
        s.add_argument("--max-minutes", type=float, default=240, dest="max_minutes")
        s.add_argument("--start-after", type=int, default=0, dest="start_after",
                       help="games played (every user's) before any phase trains: the base model plays until then")
        s.add_argument("--dose", choices=("aggregate", "replay", "recent"), default="aggregate",
                       help="every row: a phase trains on every round's so far (aggregate), on its round's and a "
                            "replay of earlier games (replay), or on the last RECENT_ROUNDS rounds' and a replay of "
                            "earlier games under the held-out stop (recent)")
        s.add_argument("--replay-ratio", type=float, default=1.0, dest="replay_ratio",
                       help="--dose replay: earlier calls replayed per new call")
        s.add_argument("--first-epochs", type=int, default=2, dest="first_epochs",
                       help="--dose replay: passes of the first phase that trains, under the held-out stop")

    s = sub.add_parser("trainer")
    training(s)
    s.set_defaults(fn=cmd_trainer)

    s = sub.add_parser("train")
    training(s)
    s.add_argument("--gpus", type=int, required=True, help="the GPUs CUDA_VISIBLE_DEVICES gives this process")
    s.set_defaults(fn=cmd_train)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
