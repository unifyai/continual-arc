"""The capacity and task-count ablation (README.md): GLM-5.3-flash's own games of a run's tasks, each of its calls a
label, and Qwen3.5-9B trained on them from the base model until the loss on games it has never trained on stops
falling. This module holds what the generator, the trainer and the tests share: the runs, their tasks, the instances
each batch plays, the labels made from a batch's games, and the stop.

  tasks RUN                   the run's tasks, one a line
  schedules RUN K --out DIR   batch K's games, one schedule each (DIR/gNNN/schedule.json), a game a fresh user
  labels BATCH                a played batch's labels (BATCH/view, BATCH/labels and its training set BATCH/labels/set)
  plateau STATE               whether the losses in a trainer's state.json have stopped falling
  cells RUN PHASE             the run's tests of PHASE's adapter as two generalisation cells (README.md, "The tests")

The benchmark and the loops' code come from the machine's trees (setup.sh's ~/loop and the rest), so run it from
~/prime-rl with ~/loop on the path: ``PYTHONPATH=~/loop uv run --no-sync python capacity.py ...``.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

HOME = Path.home()
# Each run's tasks: a benchmark configuration's at seed 0, nested (configs.config_tasks): arc1-small's 25 lie within
# arc1-medium's 100, which lie within arc1-full's 400, every Re-ARC task
RUNS = {"t25": "arc1-small", "t100": "arc1-medium", "t400": "arc1-full"}
SEED = 0
# Games a batch plays, a phase's new data: the seen test's 100, whatever the run, so that every run's phases add alike;
# its tasks come round evenly (a run's tasks in a seeded order, again and again), so a batch holds each of 25 tasks 4
# times, each of 100 once, and a quarter of 400 once each
BATCH = 100
LABEL = "capacity"   # the instances' own seeds, derive_seed(SEED, LABEL, run, batch, game), which no test draws
SEEN_TEST = "seen-test"   # the seen test's instance seeds, derive_seed(SEED, SEEN_TEST, i) (configs.test_schedule)
SEEN_TEST_SIZE = 100
# The stop (the trainer's): the loss on each phase's new games, measured before training on them; the run has
# plateaued once PATIENCE phases in a row have not lowered it by MIN_GAIN of the best so far
MIN_GAIN = 0.01
PATIENCE = 3


def tasks(run: str, pool: list[str] | None = None) -> list[str]:
    """The run's tasks, sorted, from ``pool`` (the Re-ARC task ids; the benchmark's own by default)."""
    from continual_arc.configs import config_tasks, get_config
    if pool is None:
        from continual_arc.rearc import ReArc
        pool = ReArc().task_ids
    return sorted(config_tasks(get_config(RUNS[run]), pool, SEED))


def order(run: str, ids: list[str], k: int) -> list[str]:
    """Batch ``k``'s tasks: positions BATCH*k to BATCH*(k+1) of the run's tasks dealt round after round, each round in
    an order of its own (seeded by the run and the round), so that every task comes once in each round."""
    n, out = len(ids), []
    for pos in range(BATCH * k, BATCH * (k + 1)):
        rnd, at = divmod(pos, n)
        dealt = sorted(ids)
        random.Random(f"{LABEL}-{run}-round-{rnd}").shuffle(dealt)
        out.append(dealt[at])
    return out


def instances(run: str, ids: list[str], k: int) -> list[tuple[str, int]]:
    """Batch ``k``'s games: each its task and its instance seed, a seed of its own that the seen test never draws."""
    from continual_arc.seeds import derive_seed
    seen = {derive_seed(SEED, SEEN_TEST, i) for i in range(SEEN_TEST_SIZE)}
    out = [(task, derive_seed(SEED, LABEL, run, k, j)) for j, task in enumerate(order(run, ids, k))]
    if seen & {s for _, s in out}:
        raise SystemExit(f"batch {k} of {run} draws a seen-test instance")
    return out


def schedule(run: str, k: int, j: int, task: str, seed: int) -> Any:
    """Game ``j`` of batch ``k``: one instance for a fresh user, its task served from Re-ARC as the stream serves it."""
    from continual_arc.configs import get_config
    from continual_arc.schedule import PHASE_STREAM, Schedule, Slot
    cfg = get_config(RUNS[run])
    return Schedule(name=f"{LABEL}-{run}-b{k:04d}-g{j:03d}", seed=SEED, slots=[Slot(task, seed, PHASE_STREAM)],
                    sources={task: cfg.stream_source}, users=1,
                    meta={"config": cfg.to_json(), "seed": SEED, "dataset": cfg.dataset, "track": cfg.track,
                          "capacity": {"run": run, "batch": k, "game": j}})


def write_schedules(run: str, k: int, out: Path, ids: list[str]) -> list[Path]:
    """Batch ``k``'s games' folders under ``out`` (gNNN), each holding its schedule.json, from the run's tasks ``ids``."""
    folders = []
    for j, (task, seed) in enumerate(instances(run, ids, k)):
        game = out / f"g{j:03d}"
        game.mkdir(parents=True, exist_ok=True)
        if not (game / "schedule.json").exists():
            schedule(run, k, j, task, seed).save(game / "schedule.json")
        folders.append(game)
    return folders


# -- labels: each of GLM's calls its own label, at its own state, in the student's view -------------------------------

def student_view(message: dict[str, Any]) -> dict[str, Any]:
    """An assistant message of GLM's as the student's own would stand in the conversation: what the student is trained
    to write for it (dagger.student_message: all its prose as the reasoning, only the action, with any tool calls, as
    the visible text), so that a call's earlier replies read in training as the student's own read when it plays."""
    import dagger
    return dagger.student_message({"message": message, "reasoning": message.get("reasoning")
                                   or message.get("reasoning_content")})


def viewed(doc: dict[str, Any]) -> dict[str, Any]:
    """A captured game as the student would have been served it: every request's earlier replies in the student's view
    (:func:`student_view`) and the student's chat-template settings (loop.TEMPLATE_KWARGS, its thinking mode), which
    forward.py label-set renders each call with; each call's own reply, its label, is kept as GLM gave it."""
    import loop
    calls = []
    for c in doc["calls"]:
        req = c.get("request")
        if req and isinstance(req.get("messages"), list):
            req = {**req, "chat_template_kwargs": dict(loop.TEMPLATE_KWARGS),
                   "messages": [student_view(m) if m.get("role") == "assistant" else m for m in req["messages"]]}
        calls.append({**c, "request": req})
    return {**doc, "calls": calls}


def own_label(call: dict[str, Any]) -> dict[str, Any]:
    """A call's own reply as forward.py label writes a label (labels/ID.json): the message and its reasoning."""
    message = call["response"]["choices"][0]["message"]
    usage = call["response"].get("usage") or {}
    return {"message": {k: message.get(k) for k in ("role", "content", "tool_calls")},
            "reasoning": message.get("reasoning") or message.get("reasoning_content") or "", "usage": usage,
            "from": "own", "asked": 1, "spent": usage.get("cost") or 0.0}


def label_batch(batch: Path) -> dict[str, Any]:
    """A played batch's training set: each game (BATCH/games/gNNN, played by glm_val.py) in the student's view as a
    play of its own (BATCH/view/gNNN), its points (forward.points: every call the agent made, prime-agent's compaction
    and internal calls left out) and each point's own reply as its label (BATCH/labels, forward.py label's layout),
    then forward.py label-set's training set (BATCH/labels/set, rendered with the student's template). A game the
    benchmark recorded no instance for, a play that never finished, is left out. Returns the counts."""
    import forward
    out = batch / "labels"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    views, games, spent = [], collections.Counter(), 0.0
    for game in sorted((batch / "games").glob("g*")):
        doc_file = game / "captured" / "u000.json.gz"
        if not (game / "records.jsonl").exists() or not doc_file.exists():
            games["unplayed"] += 1
            continue
        with gzip.open(doc_file, "rt") as fp:
            doc = json.load(fp)
        view = batch / "view" / game.name
        (view / "captured").mkdir(parents=True, exist_ok=True)
        with gzip.open(view / "captured" / "u000.json.gz", "wt") as fp:
            json.dump(viewed(doc), fp)
        views.append(view)
        games["solved" if doc.get("solved") else "unsolved"] += 1
        spent += sum(((c.get("response") or {}).get("usage") or {}).get("cost") or 0.0 for c in doc["calls"])
    points = forward.points(views)
    with open(out / "points.jsonl", "w") as fp:
        for p in points:
            fp.write(json.dumps(p) + "\n")
    import dagger
    for p in points:
        path = out / "labels" / f"{p['id']}.json"
        if not path.exists():
            path.write_text(json.dumps(own_label(dagger.point_call(p))))
    subprocess.run([sys.executable, str(HOME / "loop/forward.py"), "label-set", "--out", str(out)], check=True)
    made = json.loads((out / "set.json").read_text())
    return {"games": dict(games), "points": len(points), "examples": made.get("examples", 0),
            "no_move": made.get("no_move", 0), "render_failed": made.get("render_failed", 0),
            "cost_usd": round(spent, 4)}


# -- the tests: the generalisation search's cells, on its turn-based bundle --------------------------------------------

MACHINE = "NODE-cap-gen"   # the generators' machine, whose hourly backup holds each run's play config
TRAINED = "modal-capacity"   # where brain's capacity task copies each phase's adapter, under the backup's folder


def cells(run: str, phase: int) -> dict[str, dict[str, Any]]:
    """Phase ``phase``'s tests of ``run``, as generalisation cells: the paper's two frozen tests of its weights alone,
    from an empty memory under the run's task names (the run's play config, which generate.py writes and the backup
    copies), and R's 100 puzzles."""
    name, adapter = f"cap-{run}-p{phase:04d}", f"{TRAINED}/cap-{run}/phases/p{phase:04d}/adapter"
    what = (f"the capacity ablation's {run} ({RUNS[run]}'s tasks), its phase {phase}: Qwen3.5-9B trained from the base "
            "model on GLM-5.3-flash's own games of the run's tasks (search/capacity)")
    return {f"{name}-frozen": {"name": f"{name}-frozen", "kind": "frozen", "bundle": "code-dagger-rows-turns",
                               "what": f"{what}, on the seen and held-out tests, weights alone under the run's task names",
                               "run": f"{MACHINE}/cap-{run}", "adapters": {f"p{phase:04d}": adapter}},
            f"{name}-val": {"name": f"{name}-val", "kind": "val", "bundle": "code-dagger-rows-turns",
                            "what": f"{what}, on R's 100 validation puzzles", "adapters": {f"p{phase:04d}": adapter}}}


# -- the stop ---------------------------------------------------------------------------------------------------------

def plateaued(losses: list[float], min_gain: float = MIN_GAIN, patience: int = PATIENCE) -> bool:
    """Whether the new games' losses, phase by phase, have stopped falling: the last ``patience`` of them each failed to
    come ``min_gain`` (relative) under the best before it."""
    if len(losses) <= patience:
        return False
    best = min(losses[:-patience])
    return all(x > best * (1 - min_gain) for x in losses[-patience:])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("run", choices=RUNS)
    s = sub.add_parser("schedules")
    s.add_argument("run", choices=RUNS)
    s.add_argument("k", type=int)
    s.add_argument("--out", required=True)
    lab = sub.add_parser("labels")
    lab.add_argument("batch")
    pl = sub.add_parser("plateau")
    pl.add_argument("state")
    c = sub.add_parser("cells")
    c.add_argument("run", choices=RUNS)
    c.add_argument("phase", type=int)
    a = p.parse_args()
    if a.cmd == "tasks":
        print("\n".join(tasks(a.run)))
    elif a.cmd == "schedules":
        print(len(write_schedules(a.run, a.k, Path(a.out).expanduser(), tasks(a.run))))
    elif a.cmd == "labels":
        print(json.dumps(label_batch(Path(a.batch).expanduser())))
    elif a.cmd == "cells":
        folder = Path(__file__).resolve().parents[1] / "generalisation" / "cells"
        for name, spec in cells(a.run, a.phase).items():
            (folder / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
            print(folder / f"{name}.json")
    else:
        state = json.loads(Path(a.state).read_text())
        print(json.dumps({"plateaued": plateaued([ph["new_loss"] for ph in state["phases"]])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
