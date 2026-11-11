"""The real-time DAgger row: the student plays `arc1-small` served to its users round after round and never waits for
training, while
the oracle labels its games and one resident trainer fine-tunes on every label so far (the author, 18 Oct 2026: play never
waits, and the trainer stays up between phases).

Three loops run side by side on one node, each its own process, handing each other files (realtime.sh starts them,
restarts one that dies, and stops the run once all three are done):

  player    round r = 0, 1, ...: the newest adapter the trainer has finished (the base model before the first) is
            loaded into the model server, which stays up on its own GPUs, checked (paper.check_served_adapter: vLLM
            silently serves the base model under an adapter's name when its layers are misnamed), named in
            round_dir(r)/served.json, and plays round r (paper.play_round over paper.stream_cut(r)); then
            round_dir(r)/play.done. In the notation, round r is played by pi_k(r), k(r) the last phase finished when
            round r starts; served.json records k(r) and the rounds that phase had trained on.
  labeller  round by round: the oracle's own reply to every player call of the round's games as they land
            (forward.py points/label: each game is asked once while play goes on, then the whole round once more when
            it is played, so a point that failed gets a second ask), the label set (label-set), and its split into a
            training side and a held-out side (split --hold-every: held_game's one game in HOLD_EVERY, played and
            scored like any other); round_dir(r)/labels/split/split.json, written last, marks the round labelled.
  trainer   one process for the whole run, holding the model on its own GPUs: whenever rounds are labelled that the
            last phase did not train on, phase k trains (train.train_sft) from phase k-1's best adapter on every
            labelled round's training side, measuring the held-out loss on every held-out side so far every
            EVAL_STEPS steps and stopping once it turns (DAgger's aggregate, uniform; until a held-out game exists, one
            pass); then train/pNNNN/{adapter,DONE,rounds.json} and train/latest.json, which the player reads.

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

ROW = "dagger"
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
    """The newest finished phase (``train/latest.json``: its number, adapter and how many rounds it trained on, rounds
    0..rounds-1), or None before the first."""
    path = run / "train" / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None


def labelled_rounds(run: Path, rounds: int) -> int:
    """How many rounds from round 0 on are labelled and split, in a row: the rounds a phase can train on."""
    n = 0
    while n < rounds and (split_dir(run, n) / "split.json").exists():
        n += 1
    return n


def phase_runs(run: Path, rounds: int) -> tuple[list[str], list[str]]:
    """Rounds 0..rounds-1's training sides and held-out sides that hold a game (a round with no held-out game, or
    whose games all failed labelling, has an empty side)."""
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


def serve(a: argparse.Namespace, run: Path, newest: dict[str, Any] | None, current: str) -> str:
    """The model the next round plays: the newest phase's adapter, loaded (serve.sh, which also starts the server on
    ``--serve-gpus`` if it is down) and checked, every other adapter of this run unloaded; the base model before the
    first phase. An adapter that fails the served-adapter check is unloaded and ``current`` kept, with a line in the
    progress log: the round plays on, never on base weights served under a trained adapter's name."""
    from ask_sd.server import Server
    serve_sh = ["bash", str(Path.home() / "loop" / "serve.sh"), a.serve_gpus, str(a.serve_dp)]
    if newest is None:
        subprocess.run(serve_sh, check=True)
        return loop.MODEL
    name = adapter_name(a.name, newest["phase"])
    if name == current:
        return current
    subprocess.run(serve_sh + [name, newest["adapter"]], check=True)
    server = Server(SERVER)
    if not paper.adapter_is_untrained(Path(newest["adapter"])):
        check = paper.check_served_adapter(server, loop.MODEL, name, paper.SERVED_ADAPTER_PROMPT)
        if not (check["answered"] and check["differs"]):
            say(run, f"phase {newest['phase']}'s adapter failed the served-adapter check ({check}): "
                     f"round plays on {current}")
            server.unload_adapter(name)
            return current
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
        newest = latest(run)
        current = serve(a, run, newest, current)
        rd.mkdir(parents=True, exist_ok=True)
        write_json(rd / "served.json", {"model": current, "phase": newest["phase"] if newest else None,
                                        "trained_rounds": newest["rounds"] if newest else 0,
                                        "at": datetime.now(timezone.utc).isoformat()})
        schedule = paper.stream_cut(r, users=a.users)
        want = len(schedule.slots)
        have = 0
        for attempt in range(1, 4):
            say(run, f"play round {r} started ({current}, attempt {attempt})")
            try:
                paper.play_round(ROW, play_dir(run), schedule, base_url=SERVER, served_name=current)
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
    """Round ``round_``'s label set and its split (split.json, written last, marks the round labelled)."""
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
    r, asked = labelled_rounds(run, a.rounds), set()
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
# the adapter files a phase needs only while it is the newest or the one before it: the next phase loads the newest
# whole (its optimizer's state and the trainer's own copy), and the player converts and loads it (adapter.vllm)
RESUME_FILES = ("optimizer.pt", "trainer_adapter.safetensors", "trainer_state.json")


def serving(run: Path) -> int | None:
    """The phase the player's newest round is played on (its served.json), None before it has served one."""
    found = sorted(round_dir(run, 0).parent.glob("r*/served.json"))
    return json.loads(found[-1].read_text())["phase"] if found else None


def prune(root: Path, newest: int, served: int | None) -> None:
    """Frees what phases older than ``newest - 1`` no longer need (4.4 GB a phase on Qwen3.5-9B at rank 128, so a
    1-user run that trains every round would fill a node's disk within a day): each one's vLLM copy and the files a
    phase resumes from go, and its whole adapter too unless it is a KEEP_EVERY-th phase. DONE, phase.json and
    rounds.json, every phase's record of its training, stay. The ``served`` phase stays whole however old: a round
    plays on one phase to its end, and vLLM reads its adapter again whenever another adapter's load pushes it out of
    the server's cache (18 Oct 2026: a 1-user round outlasted three phases, and the reload of its pruned adapter
    killed the server under all four runs)."""
    for phase in range(newest - 1):
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
    """train.py's settings for every phase (train.train_sft reads them); the held-out runs, seed and pass count are
    each phase's own (:func:`cmd_trainer`)."""
    import train
    return argparse.Namespace(
        model=loop.MODEL, objective=train.Objective.sft, lr=a.lr, batch=a.batch, gpus=a.gpus, rank=a.lora_rank,
        alpha=a.lora_rank, max_seq_len=a.max_seq_len, chains=True, chain_slack=train.CHAIN_SLACK, seed=0,
        eval_steps=a.eval_steps, patience=a.patience, max_steps=a.max_steps, epochs=a.max_epochs, val_run=None,
        val_max_items=a.val_max_items, max_minutes=a.max_minutes, divergence="forward_kl", dry_run=False)


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
        trained = newest["rounds"] if newest else 0
        ready = labelled_rounds(run, a.rounds)
        if trained == a.rounds:
            break
        train_runs, dev_runs = phase_runs(run, ready) if ready > trained else ([], [])
        if not train_runs:
            time.sleep(POLL_S)
            continue
        phase = newest["phase"] + 1 if newest else 0
        out = root / f"p{phase:04d}"
        shutil.rmtree(out, ignore_errors=True)   # a phase cut short is trained again whole
        out.mkdir()
        calls, compactions = [], 0
        for path in train_runs:
            got, left_out = train.load_calls(Path(path))
            calls += got
            compactions += left_out
        targs.val_run, targs.seed = dev_runs or None, phase
        targs.epochs = a.max_epochs if dev_runs else 1   # no held-out game yet: one pass, nothing to stop on
        say(run, f"phase {phase} started: rounds 0-{ready - 1}, {len(calls)} labels, {len(dev_runs)} held-out sides")
        done = train.train_sft(targs, trainer, calls, compactions, out, Path(newest["adapter"]) if newest else None,
                               random.Random(phase))
        write_json(out / "rounds.json", {"rounds": ready})
        write_json(root / "latest.json", {"phase": phase, "adapter": str(out / "adapter"), "rounds": ready})
        prune(root, phase, serving(run))
        dev = [e["dev_nll"] for e in done["dev"]]
        held = f", held-out {dev[0]} to {min(dev)} (best at step {done['best_step']})" if dev else ""
        say(run, f"phase {phase} done: rounds 0-{ready - 1}, {done['steps']} steps in {done['minutes']} min{held}")
    (run / "trainer.done").touch()


# -- CLI -------------------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--run", required=True, help="the run's folder (realtime.sh's ~/runs/rt/NAME)")
        s.add_argument("--rounds", type=int, required=True)

    s = sub.add_parser("player")
    common(s)
    s.add_argument("--users", type=int, required=True, help="the users the stream is served to (paper.served)")
    s.add_argument("--name", required=True, help="the run's name, its adapters' prefix on the server")
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
    s.add_argument("--lr", type=float, required=True)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--gpus", type=int, required=True, help="the GPUs CUDA_VISIBLE_DEVICES gives this process")
    s.add_argument("--lora-rank", type=int, default=128, dest="lora_rank")
    s.add_argument("--max-seq-len", type=int, default=65536, dest="max_seq_len")
    s.add_argument("--eval-steps", type=int, required=True, dest="eval_steps")
    s.add_argument("--patience", type=int, default=1)
    s.add_argument("--max-steps", type=int, required=True, dest="max_steps", help="a phase's safety cap")
    s.add_argument("--max-epochs", type=int, default=20, dest="max_epochs")
    s.add_argument("--val-max-items", type=int, default=1000, dest="val_max_items")
    s.add_argument("--max-minutes", type=float, default=240, dest="max_minutes")
    s.set_defaults(fn=cmd_trainer)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
