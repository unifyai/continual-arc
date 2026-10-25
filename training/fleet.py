"""Where real-time runs' GPUs go: each run's measured pace, the hardware's speed, and the plans of a set of runs on
nodes from the fastest to the cheapest, so that the GPUs are used as fully as the cost allows, for any training
configuration and any number of runs, an experiment wanting the cheapest plan, the fastest or one between.
docs/method-search.md, "GPU layouts", has the measurements and the plans this tool gave.

A run (realtime.py) is three loops handing each other files. Per round r: play takes P_r on the model server's GPUs,
the oracle's labels land L_r after play ends (DAgger and OPSD; no GPU), and phase r trains for T_r on the trainer's
GPUs. Under the one-round lag round r+2 waits for phase r, which waits for round r's labels, so a run's period per
round is about max(P, T, (P + L + T) / 2): play alone when P >= L + T, training alone when T >= P + L, and otherwise
the two-round cycle in which a minute of play, labelling or training costs half a minute. The trainer's speed is
close to linear in its GPUs (about 4,000 to 5,000 tokens read a second per H100, 1 to 4 GPUs, from 11,000 steps of the
runs of 18-20 Oct), so a phase's work is about the same GPU-minutes on any number of them, and a run given fewer
trainer GPUs trains proportionally slower; the model server, which a run keeps 0-5 requests in flight on, is the
slack. Runs on one node share one server (realtime.sh), and their trainers share their TRAIN_GPUS as one pool
(gpupool.py): a phase takes a fair share of the pool's GPUs and gives some back when another run's phase needs them.

Subcommands:
  profile RUN...    each run's rounds from its own files (progress, train/pNNNN/{phase.json,DONE}): per round its play,
                    label tail and phase minutes, the phase's GPUs and work in GPU-minutes, the player's waits, and the
                    run's pace and how busy its trainer was. On a copy of the run's files or on the node.
  plan SPEC         SPEC's runs on SPEC's nodes, every layout simulated over the runs' profiles (simulate): every run
                    on a node of its own at its fastest, and the plans from the fastest (the last run done soonest)
                    to the cheapest, sharing nodes; then the one to run, with each run's realtime.sh command line
                    (--launch): the cheapest done within --deadline hours, or else every run apart at its fastest, or
                    the fastest plan within --budget dollars when that costs more.
  bench-serve       (on a node, system Python) the model server's speed: captured games replayed against it by
                    --users users at once, each call asked for its recorded reply's length; per call its first
                    token's wait and its decode speed.
  bench-train       (on a node, from prime-rl's checkout) the trainer's speed on --gpus GPUs: --steps steps of a run's
                    phase, with the held-out measurements its settings make, from a copy of the run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import itertools
import json
import math
import random
import re
import statistics
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# -- a run's own pace, from its files ----------------------------------------------------------------------------

LINE = re.compile(r"^(\d\d-\d\d \d\d:\d\d) (realtime(?:\.sh)?): (.*)$")
# the trainer's speed per GPU against one GPU's, by GPUs (a phase's work divided by GPUs and this is its minutes):
# DAgger's steps of 18-20 Oct 2026 on H100s read 4.8k, 4.1k, 4.9k and 4.1k tokens a second per GPU on 1-4 GPUs, OPSD's
# 3.7k, 3.8k, 3.9k and 3.1k; on a leased 8x H100 the same 6 steps of DAgger, every kernel cached, took 34.3 s a step
# on 4 GPUs and 24.8 on 6 (5 and 7-8 between and past them). bench-train measures a node's own (SPEC's efficiency)
EFFICIENCY = {1: 1.0, 2: 0.86, 3: 0.95, 4: 0.85, 5: 0.81, 6: 0.78, 7: 0.76, 8: 0.74}
SKIPPED_PHASE_MIN = 0.5   # a warm-up phase, which trains nothing and saves no adapter
CHAIN_CALLS = 16          # train.CHAIN_CALLS: the most consecutive calls of one game a step's piece holds


def _minute(stamp: str) -> float:
    """A progress line's time (MM-DD HH:MM, UTC) in minutes; the year never matters, only differences do."""
    return datetime.strptime("2026-" + stamp, "%Y-%m-%d %H:%M").timestamp() / 60


def profile(run: Path) -> dict[str, Any]:
    """A run's rounds as its files tell them (the module docstring). Times come from progress, to the minute, but a
    phase's from its DONE's minutes when it has them; a phase's GPUs from its phase.json. A round cut short and played
    again counts from its last start."""
    events: dict[str, dict[int, float]] = {k: {} for k in ("start", "done", "labelled", "wait", "phase_start",
                                                            "phase_done")}
    skipped: set[int] = set()
    row = users = None
    for line in (run / "progress").read_text(errors="replace").splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        at, msg = _minute(m[1]), m[3]
        if (g := re.match(r"started: ([a-z]+), (\d+) users", msg)):
            row, users = g[1], int(g[2])
        for key, pattern in (("start", r"play round (\d+) started"), ("done", r"play round (\d+) done"),
                             ("wait", r"play round (\d+) waits for phase"), ("labelled", r"round (\d+) labelled"),
                             ("phase_start", r"phase (\d+) started"), ("phase_done", r"phase (\d+) done")):
            if (g := re.match(pattern, msg)):
                events[key][int(g[1])] = at
        if (g := re.match(r"phase (\d+) done: rounds 0-\d+, not trained", msg)):
            skipped.add(int(g[1]))
    rounds = []
    for r in sorted(events["done"]):
        start, done = events["start"].get(r), events["done"][r]
        ready = done if row == "pg" else events["labelled"].get(r)
        phase = {"trained": r not in skipped and r in events["phase_done"]}
        if phase["trained"]:
            folder = run / "train" / f"p{r:04d}"
            summary = json.loads((folder / "DONE").read_text()) if (folder / "DONE").exists() else {}
            began = events["phase_start"].get(r)
            minutes = summary.get("minutes") or (events["phase_done"][r] - began if began is not None else None)
            gpus = json.loads((folder / "phase.json").read_text())["args"]["gpus"] if (folder / "phase.json").exists() \
                else None
            phase.update(minutes=minutes, gpus=gpus, steps=summary.get("steps"),
                         work=minutes * gpus * EFFICIENCY[gpus] if minutes and gpus else None)
        wait = start - events["wait"][r] if start is not None and r in events["wait"] else 0.0
        rounds.append({"round": r, "play": done - start if start is not None else None,
                       "label_tail": ready - done if ready is not None else None, "phase": phase, "wait": wait})
    trained = [x for x in rounds if x["phase"]["trained"]]
    out: dict[str, Any] = {"run": run.name, "row": row, "users": users, "rounds": rounds}
    if rounds and events["start"]:
        span = max(events["done"].values()) - min(events["start"].values())
        out["pace"] = {"rounds_per_hour": round(len(rounds) / (span / 60), 2) if span else None,
                       "hours": round(span / 60, 1), "player_waited_hours": round(sum(x["wait"] for x in rounds) / 60, 1)}
    if trained:
        first = events["phase_start"].get(trained[0]["round"]) or events["done"][trained[0]["round"]]
        last = max(events["phase_done"].values())
        busy = sum(x["phase"]["minutes"] or 0 for x in trained)
        out["trainer"] = {"phases": len(trained), "busy_fraction_since_first_trained_phase":
                          round(busy / (last - first), 2) if last > first else None}
    return out


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 1) if xs else None


def cmd_profile(a: argparse.Namespace) -> None:
    for path in a.runs:
        p = profile(Path(path).expanduser())
        if a.json:
            print(json.dumps(p))
            continue
        trained = [x["phase"] for x in p["rounds"] if x["phase"]["trained"]]
        print(f"{p['run']} ({p['row']}, {p['users']} users): {len(p['rounds'])} rounds, {p.get('pace')}, "
              f"trainer {p.get('trainer')}")
        print(f"  per round, minutes: play {mean([x['play'] for x in p['rounds']])}, label tail "
              f"{mean([x['label_tail'] for x in p['rounds']])}, trained phase {mean([t['minutes'] for t in trained])} "
              f"on {sorted({t['gpus'] for t in trained if t['gpus']})} GPUs, work {mean([t['work'] for t in trained])} "
              f"GPU-minutes")


# -- the pipeline, simulated -----------------------------------------------------------------------------------

@dataclass
class Trace:
    """A run to simulate: per round its play minutes (on the server its profile played on), its label tail and its
    phase's work in GPU-minutes (0 for a phase that trains nothing)."""
    name: str
    play: list[float]
    label_tail: list[float]
    work: list[float]
    lag: int = 1


def trace(p: dict[str, Any], rounds: int, start_after: int = 24, seed: int = 0,
          work_from: list[dict[str, Any]] | None = None, work_scale: float = 1.0) -> Trace:
    """A run of ``rounds`` rounds drawn from profile ``p``'s own rounds: each round's play and label tail drawn from
    its rounds', each phase's work from the trained phases of ``work_from`` (profiles of runs that train alike; ``p``
    alone by default), none while rounds 0..k hold fewer than ``start_after`` games (realtime's warm-up). A phase's work
    hardly depends on the run's users, since every row's phase runs until its held-out loss stops improving, so a run
    with few trained phases yet draws better from its row's every run. ``work_scale`` scales every phase's work: a
    profile of a run's first rounds has small held-out sets, whose measurements grow with the run."""
    rng = random.Random(seed)
    play = [x["play"] for x in p["rounds"] if x["play"] is not None]
    tail = [x["label_tail"] for x in p["rounds"] if x["label_tail"] is not None] or [0.0]
    work = [x["phase"]["work"] for q in (work_from or [p]) for x in q["rounds"] if x["phase"].get("work")]
    if not work:
        raise SystemExit(f"{p['run']}: no trained phase to draw training from")
    return Trace(p["run"], [rng.choice(play) for _ in range(rounds)], [rng.choice(tail) for _ in range(rounds)],
                 [0.0 if (k + 1) * p["users"] < start_after else rng.choice(work) * work_scale for k in range(rounds)])


@dataclass
class _Run:
    t: Trace
    play_factor: float
    own: int = 0                    # static: its trainer's GPUs
    r: int = 0                      # the next round to play
    play_end: float | None = None   # while a round plays
    ready: dict[int, float] = field(default_factory=dict)   # each played round's labels, when they land
    k: int = 0                      # the next phase to train
    left: float | None = None       # while a phase trains: its work left, GPU-minutes
    gpus: int = 0                   # while a phase trains: its GPUs
    stall: float = 0.0              # until when its trainer (re)starts and trains nothing
    resident: int = 0               # pooled: the GPUs its trainer holds between phases (0: a cold start next)
    done_at: float | None = None
    busy: float = 0.0


def simulate(traces: list[Trace], pool: int | None = None, gpus: list[int] | None = None,
             play_factors: list[float] | None = None, efficiency: dict[int, float] = EFFICIENCY,
             restart: float = 1.5, shrink: bool = True) -> list[dict[str, float]]:
    """Each run's minutes to its last phase and its trainer's busy minutes, for runs on one node, realtime.py's loops
    as events: round r plays once phase r-1-LAG is done (each run's play scaled by ``play_factors``, a slower or busier
    server), its labels land its label tail after it, and phase r trains once they have. With ``gpus`` each run's
    trainer has its own; with ``pool`` all of them share that many as gpupool.py shares them: split evenly over the runs
    with a phase to train (the odd ones to whoever holds more), a phase starting on another number than its idle
    trainer holds paying ``restart`` minutes, and, with ``shrink``, a phase in flight giving GPUs up to a newcomer at the
    same cost and never growing before its end; an idle trainer keeps its GPUs until another phase needs them. A phase
    that trains nothing (the warm-up) takes SKIPPED_PHASE_MIN and no GPU."""
    n = len(traces)
    runs = [_Run(t, (play_factors or [1.0] * n)[i], (gpus or [0] * n)[i]) for i, t in enumerate(traces)]
    now = 0.0

    def rate(x: _Run) -> float:
        return x.gpus * efficiency[x.gpus] if x.gpus else 0.0

    def startable(x: _Run) -> bool:
        return x.done_at is None and x.left is None and x.ready.get(x.k, math.inf) <= now

    def start(x: _Run, g: int) -> None:
        x.left, x.gpus = x.t.work[x.k], g
        if not x.t.work[x.k]:
            x.gpus, x.stall = 0, now + SKIPPED_PHASE_MIN
        elif pool and g != x.resident:
            x.stall, x.resident = now + restart, g

    def share() -> None:
        for x in runs:   # the warm-up's phases need no GPU
            if startable(x) and not x.t.work[x.k]:
                start(x, 0)
        wanting = [x for x in runs if x.done_at is None and x.t.work[min(x.k, len(x.t.work) - 1)]
                   and (x.left is not None or startable(x))]
        if not wanting:
            return
        base, extra = divmod(pool, len(wanting))
        order = sorted(wanting, key=lambda x: -max(x.gpus, x.resident))
        target = {id(x): base + (i < extra) for i, x in enumerate(order)}
        for x in wanting:
            if x.left is not None and shrink and target[id(x)] < x.gpus:
                x.gpus = x.resident = target[id(x)]
                x.stall = max(x.stall, now) + restart
        free = pool - sum(x.gpus for x in runs if x.left is not None)
        for x in sorted(wanting, key=lambda x: x.ready.get(x.k, 0.0)):
            if x.left is None and free > 0:
                g = min(target[id(x)], free)
                start(x, g)
                free -= g
        held = sorted((x for x in runs if x.left is None and x.resident), key=lambda x: -x.resident)
        for x in held:   # an idle trainer whose GPUs a phase took has none left to stay on
            if free >= x.resident:
                free -= x.resident
            else:
                x.resident = 0

    while any(x.done_at is None for x in runs):
        for x in runs:
            if x.play_end is None and x.r < len(x.t.play) and x.r - 1 - x.t.lag < x.k:
                x.play_end = now + x.t.play[x.r] * x.play_factor
        if pool:
            share()
        else:
            for x in runs:
                if startable(x):
                    start(x, x.own)
        times = []
        for x in runs:
            if x.play_end is not None:
                times.append(x.play_end)
            if x.left is not None:
                times.append(x.stall if x.stall > now else now + x.left / rate(x) if rate(x) else math.inf)
            elif x.done_at is None and x.ready.get(x.k, -math.inf) > now:
                times.append(x.ready[x.k])
        t = min(times)
        if t == math.inf:
            raise RuntimeError("the simulation stalled: a phase waits for GPUs nobody will free")
        for x in runs:
            if x.left is not None and rate(x) and t > max(now, x.stall):
                x.left -= (t - max(now, x.stall)) * rate(x)
                x.busy += t - max(now, x.stall)
        now = t
        for x in runs:
            if x.play_end is not None and x.play_end <= now:
                x.ready[x.r] = max(now + x.t.label_tail[x.r], x.ready.get(x.r - 1, 0.0))
                x.r, x.play_end = x.r + 1, None
            if x.left is not None and x.left <= 1e-9 and now >= x.stall:
                x.k, x.left, x.gpus = x.k + 1, None, 0
                if x.k == len(x.t.work):
                    x.done_at = now
    return [{"minutes": x.done_at, "trainer_busy": x.busy, "rounds": len(x.t.play)} for x in runs]


# -- plans -----------------------------------------------------------------------------------------------------

def compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """Every way of giving ``parts`` runs at least one of ``total`` GPUs each, all of them used."""
    return [c for c in itertools.product(range(1, total + 1), repeat=parts) if sum(c) == total]


def cheapest(n: int, cost: dict[tuple[int, ...], float | None]) -> tuple[float, list[tuple[int, ...]]] | None:
    """The cheapest partition of runs 0..n-1 into groups each with a ``cost`` (None: that group may not share a node),
    and its cost: a dynamic programme over the sets of runs left, each step placing the lowest run left with some of
    the others, so a plan of 15 or 20 runs is as quick as one of 6."""
    import functools

    @functools.lru_cache(maxsize=None)
    def best(left: frozenset[int]) -> tuple[float, tuple[tuple[int, ...], ...]] | None:
        if not left:
            return 0.0, ()
        first = min(left)
        found = None
        for group, c in cost.items():
            if c is None or group[0] != first or not set(group) <= left:
                continue
            rest = best(left - set(group))
            if rest is not None and (found is None or c + rest[0] < found[0]):
                found = (c + rest[0], (group, *rest[1]))
        return found

    got = best(frozenset(range(n)))
    return (got[0], list(got[1])) if got else None


def frontier(n: int, hours: dict[tuple[int, ...], float]) -> list[tuple[float, float, list[tuple[int, ...]]]]:
    """Every grouping of runs 0..n-1 onto nodes that no other beats on both time and node-hours, the fastest first, as
    (the hours until its last run is done, its node-hours, its groups). ``hours`` is each group's on one node in its
    fastest layout, which is also its cheapest, since a node is paid until its last run ends. Under a time limit the
    cheapest grouping of the groups within it is :func:`cheapest`'s, and bisection over the groups' hours finds each
    limit that buys a grouping at least 1% cheaper than the one before, so many runs take few programmes."""
    limits = sorted(set(hours.values()))
    found: dict[float, tuple[float, list[tuple[int, ...]]] | None] = {}

    def within(limit: float) -> tuple[float, list[tuple[int, ...]]] | None:
        if limit not in found:
            found[limit] = cheapest(n, {g: h if h <= limit else None for g, h in hours.items()})
        return found[limit]

    def first(ok: Callable[[float], bool], lo: int) -> int | None:   # ok holds from some limit on, or for none
        hi = len(limits)
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(limits[mid]) else (mid + 1, hi)
        return lo if lo < len(limits) else None

    out = []
    at = first(lambda t: within(t) is not None, 0)
    while at is not None:
        cost, groups = within(limits[at])
        out.append((max(hours[g] for g in groups), cost, groups))
        at = first(lambda t, cost=cost: within(t)[0] < 0.99 * cost, at + 1)
    return out


def choose(got: dict[str, Any], deadline: float | None = None, budget: float | None = None) -> dict[str, Any] | None:
    """The plan to run from :func:`plan`'s ``got``. With a ``deadline`` (hours), the cheapest plan whose last run is
    done within it. Without one, every run on a node of its own at its fastest (``apart``), which leaves every run all
    the margin it can have; with a ``budget`` too small for that, the fastest plan within it. Never one costing more
    than ``budget``; None when no plan meets them."""
    ok = [p for p in got["plans"]
          if (deadline is None or p["hours"] <= deadline) and (budget is None or p["cost"] <= budget)]
    if deadline is not None:
        return ok[-1] if ok else None
    if budget is None or got["apart"]["cost"] <= budget:
        return got["apart"]
    return ok[0] if ok else None


# an on-policy trainer's draws (OPSD: 64 replies a step, 740 tokens each, drawn while the step before trains) slow the
# other calls on its server. Measured with bench-serve's --draws (64 calls of a step's chain pieces every
# 30 s) beside users replaying games: on 2 GPUs a call took 16.2 s per 1k tokens against 6.15 without at 5 users and
# 15.6 against 4.6 at 1, a batch in flight 28-30 s of every 30; on 4 GPUs 9.3 against 4.84 at 5 users, 6.7 against 4.84
# at 4 and 4.5 against 3.7 at 1, a batch 20.5, 14.7 and 9.8 s. DRAW_SLOWDOWN is the extra a call takes per unit of the
# time the draws are in flight (2.1 on 2 GPUs, 0.95 on 4: the means), DRAW_SHARE that share of an OPSD step (35 s) a
# batch takes on the server. They slow the other runs' play, never the run's own: its player mostly waits for the phase
# its trainer is drawing for (the lag), and the final OPSD runs' games generated 94-112% of the DAgger runs' tokens a
# second at the same users. Beside an OPSD run on 2 GPUs, the pool test's DAgger games ran 1.05-1.45 times slower
DRAW_SLOWDOWN = {4: 0.95, 2: 2.1, 1: 3.0}
DRAW_SHARE = {4: 0.43, 2: 0.83, 1: 1.0}


def per_call(spec: dict[str, Any], tp: int, load: float) -> float:
    """A call's seconds per thousand tokens on a server of ``tp`` GPUs with ``load`` requests in flight, interpolated
    in SPEC's ``serve`` (bench-serve's, by tensor-parallel size and users): held at the lightest load below it, and
    past the heaviest carried on along its last two points' slope, so a load nobody measured is never taken for a free
    one."""
    row = {int(k): v for k, v in spec["serve"][str(tp)].items()}
    known = sorted(row)
    load = max(load, known[0])
    if load >= known[-1]:
        if len(known) == 1:
            return row[known[0]]
        a, b = known[-2], known[-1]
        return row[b] + (row[b] - row[a]) * (load - b) / (b - a)
    lo, hi = max(k for k in known if k <= load), min(k for k in known if k >= load)
    return row[lo] if lo == hi else row[lo] + (row[hi] - row[lo]) * (load - lo) / (hi - lo)


def drawing(member: dict[str, Any], p: dict[str, Any], tp: int) -> float:
    """The share of the time an on-policy run's draws are in flight on a server of ``tp`` GPUs (0 for any other run):
    the share its trainer trains (its profile's) times :data:`DRAW_SHARE`."""
    if not member.get("draws"):
        return 0.0
    return ((p.get("trainer") or {}).get("busy_fraction_since_first_trained_phase") or 0.0) * DRAW_SHARE[tp]


def play_factor(spec: dict[str, Any], tp: int, users: int, own_users: int, others_drawing: float) -> float:
    """A run's play on a server of ``tp`` GPUs that ``users`` users of every run on it share, beside other runs' draws
    in flight ``others_drawing`` of the time (:func:`drawing`, summed), against its profile's own (SPEC's reference
    server, the run alone): the per-call times' ratio, weighed by the share of play spent waiting on the model
    (``model_share``: a game generated a median 240 tokens a second of its wall time, as fast as a lone call decodes,
    so all of it)."""
    ratio = (per_call(spec, tp, users) * (1 + DRAW_SLOWDOWN[tp] * others_drawing)
             / per_call(spec, spec["reference"]["tp"], own_users))
    return 1 + spec.get("model_share", 1.0) * (ratio - 1)


def node_layouts(spec: dict[str, Any], members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every layout of ``members`` (runs) on one node: a server of each tensor-parallel size SPEC allows, and the
    other GPUs either split among the runs' trainers or shared as one pool."""
    out = []
    for tp in spec["tp"]:
        left = spec["gpus_per_node"] - tp
        if left < len(members):
            continue
        out.append({"tp": tp, "pool": left})
        out += [{"tp": tp, "gpus": list(c)} for c in compositions(left, len(members))]
    return out


def evaluate(spec: dict[str, Any], members: list[dict[str, Any]], layout: dict[str, Any],
             profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``layout``'s hours for each of ``members`` over SPEC's seeds (the mean), and the node's hours (its slowest)."""
    users = [profiles[m["profile"]]["users"] for m in members]
    draws = [drawing(m, profiles[m["profile"]], layout["tp"]) for m in members]
    factors = [play_factor(spec, layout["tp"], sum(users), u, sum(draws) - d) for u, d in zip(users, draws)]
    hours = [[] for _ in members]
    for seed in range(spec.get("seeds", 3)):
        traces = [trace(profiles[m["profile"]], m["rounds"], m.get("start_after", 24), seed=seed * 101 + i,
                        work_from=[profiles[w] for w in m.get("work_from", [m["profile"]])],
                        work_scale=spec.get("work_scale", 1.0))
                  for i, m in enumerate(members)]
        got = simulate(traces, pool=layout.get("pool"), gpus=layout.get("gpus"), play_factors=factors,
                       efficiency={int(k): v for k, v in spec.get("efficiency", EFFICIENCY).items()},
                       restart=spec.get("restart_minutes", 1.5))
        for i, g in enumerate(got):
            hours[i].append(g["minutes"] / 60)
    each = [statistics.fmean(h) for h in hours]
    return {"layout": layout, "hours": each, "node_hours": max(each)}


def alone(spec: dict[str, Any], member: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> float:
    """The run's hours alone on a node in SPEC's reference layout (its server, its trainer GPUs)."""
    ref = spec["reference"]
    return evaluate(spec, [member], {"tp": ref["tp"], "gpus": [ref["train_gpus"]]}, profiles)["hours"][0]


def gpus_of(layout: dict[str, Any], n: int, per_node: int) -> tuple[str, list[str]]:
    """A node layout's SERVE_GPUS and each of its ``n`` runs' TRAIN_GPUS: the server on the first GPUs, then the pool
    (every run the same list) or each run's own GPUs in turn."""
    tp = layout["tp"]
    serve = ",".join(map(str, range(tp)))
    if "pool" in layout:
        return serve, [",".join(map(str, range(tp, per_node)))] * n
    out, at = [], tp
    for g in layout["gpus"]:
        out.append(",".join(map(str, range(at, at + g))))
        at += g
    return serve, out


def plan(spec: dict[str, Any]) -> dict[str, Any]:
    """SPEC's runs on SPEC's nodes: every layout of every group of at most ``most_per_node`` runs simulated once
    (:func:`evaluate`, over SPEC's ``seeds``), each group in its fastest, a node paid SPEC's ``price_per_node_hour``
    until its last run ends. ``apart`` is every run on a node of its own at its fastest; ``plans`` run from the
    fastest to the cheapest (:func:`frontier`), sharing nodes where that costs less without the last run done later.
    Each run's hours are also given against its own alone in SPEC's reference layout (:func:`alone`), the layout its
    profile was measured in."""
    profiles = {name: profile(Path(path).expanduser()) for name, path in spec["profiles"].items()}
    members = spec["runs"]
    base = [alone(spec, m, profiles) for m in members]
    best = {}
    for size in range(1, spec.get("most_per_node", 3) + 1):
        for group in itertools.combinations(range(len(members)), size):
            runs = [members[i] for i in group]
            best[group] = min((evaluate(spec, runs, layout, profiles) for layout in node_layouts(spec, runs)),
                              key=lambda x: x["node_hours"])
    price = spec["price_per_node_hour"]

    def laid_out(groups: list[tuple[int, ...]]) -> dict[str, Any]:
        node_hours = sum(best[g]["node_hours"] for g in groups)
        return {"hours": max(best[g]["node_hours"] for g in groups), "node_hours": node_hours,
                "cost": node_hours * price,
                "nodes": [{"runs": [members[i]["name"] for i in g], **best[g],
                           "against_reference": [h / base[i] - 1 for h, i in zip(best[g]["hours"], g)]}
                          for g in groups]}

    return {"reference": {"hours": base, "cost": sum(base) * price},
            "apart": laid_out([(i,) for i in range(len(members))]),
            "plans": [laid_out(groups) for _, _, groups in
                      frontier(len(members), {g: b["node_hours"] for g, b in best.items()})]}


def cmd_plan(a: argparse.Namespace) -> None:
    spec = json.loads(Path(a.spec).read_text())
    got = plan(spec)
    members = {m["name"]: m for m in spec["runs"]}
    n, ref = len(members), got["reference"]

    def summary(p: dict[str, Any]) -> str:
        shared = ["+".join(node["runs"]) for node in p["nodes"] if len(node["runs"]) > 1]
        return (f"the last done in {p['hours']:.0f} h ({p['hours'] / 24:.1f} days), {len(p['nodes'])} nodes, "
                f"{p['node_hours']:.0f} node-hours, ${p['cost']:,.0f} (${p['cost'] / n:,.0f} a run)"
                + (f", sharing {', '.join(shared)}" if shared else ""))

    print(f"each run alone in the reference layout (server on {spec['reference']['tp']} GPUs, trainer on "
          f"{spec['reference']['train_gpus']}): " + ", ".join(f"{m} {h:.0f}h" for m, h in zip(members, ref["hours"]))
          + f"; the last done in {max(ref['hours']):.0f} h, ${ref['cost']:,.0f} (${ref['cost'] / n:,.0f} a run)")
    print("apart, every run on a node of its own at its fastest: " + summary(got["apart"]))
    print("the plans, the fastest first, sharing nodes where that costs less (every run started at once):")
    for k, p in enumerate(got["plans"]):
        before = got["plans"][k - 1] if k else None
        print(f"  {summary(p)}" + (f": {p['hours'] - before['hours']:+.0f} h for ${before['cost'] - p['cost']:,.0f} "
                                     "less than the one above" if before else ""))
    chosen = choose(got, a.deadline, a.budget)
    asked = " and ".join(([f"done within {a.deadline:g} h"] if a.deadline is not None else [])
                         + ([f"at most ${a.budget:,.0f}"] if a.budget is not None else []))
    if chosen is None:
        print(f"no plan is {asked}")
        return
    print(("apart" if chosen is got["apart"] else f"the {'cheapest' if a.deadline is not None else 'fastest'} plan "
           f"{asked}") + ": " + summary(chosen))
    hosts = a.hosts.split(",") if a.hosts else []
    for i, node in enumerate(chosen["nodes"]):
        layout = node["layout"]
        trainers = f"a pool of {layout['pool']}" if "pool" in layout else "+".join(map(str, layout["gpus"]))
        print(f"  server on {layout['tp']}, trainers {trainers}: " + ", ".join(
            f"{r} {h:.0f}h ({s:+.0%})" for r, h, s in zip(node["runs"], node["hours"], node["against_reference"])))
        if a.launch:   # each run's realtime.sh command line, to run on its node
            host = hosts[i] if i < len(hosts) else f"NODE{i + 1}"
            serve, train_gpus = gpus_of(layout, len(node["runs"]), spec["gpus_per_node"])
            for r, tg in zip(node["runs"], train_gpus):
                print(f"    ssh {host} '{launch_line(members[r], serve, tg)}'")
    if a.json:
        print(json.dumps(got))


def launch_line(member: dict[str, Any], serve: str, train_gpus: str) -> str:
    """The shell line that starts ``member`` (a SPEC run) on its node: realtime.sh detached, with SPEC's settings as its
    variables, its ``launch_name``, if any, as its NAME, and the layout's GPUs."""
    values = {**({"NAME": member["launch_name"]} if member.get("launch_name") else {}), **member["settings"],
              "SERVE_GPUS": serve, "TRAIN_GPUS": train_gpus}
    env = " ".join(f"{k}={v}" for k, v in values.items())
    return f"cd ~ && ({env} setsid nohup bash ~/continual-arc/training/realtime.sh > /dev/null 2>&1 < /dev/null &)"


# -- the server's speed (bench-serve) ------------------------------------------------------------------------

def replayable(game: Path) -> list[dict[str, Any]]:
    """A captured game's calls that came back whole (every call kept as served: paper.captured), in order."""
    calls = json.load(gzip.open(game))["calls"]
    return [c for c in calls if c.get("response") and (c["response"].get("usage") or {}).get("completion_tokens")]


def ask(server: str, call: dict[str, Any], model: str, timeout: float = 1800, tokens: int | None = None) -> dict[str, float]:
    """One captured call asked again, as it was served but for its model, made to write exactly as many tokens as it
    did (min_tokens and max_tokens both its count, its end ignored; ``tokens`` instead, if given), streamed: when its
    first token came and how fast the rest did."""
    body = dict(call["request"])
    n = tokens or call["response"]["usage"]["completion_tokens"]
    body.update(model=model, max_completion_tokens=n, min_tokens=n, ignore_eos=True, stream=True,
                stream_options={"include_usage": True}, return_token_ids=False)
    req = urllib.request.Request(server.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if first is None and any(c.get("delta") for c in chunk.get("choices") or []):
                first = time.time()
    end = time.time()
    first = first or end
    out = usage.get("completion_tokens") or n
    return {"seconds": end - t0, "first_token_s": first - t0, "tokens": out, "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tps": (out - 1) / (end - first) if end > first else None}


def cmd_bench_serve(a: argparse.Namespace) -> None:
    games = [replayable(Path(g)) for g in a.games]
    models = a.models.split(",")
    stop = time.time() + a.seconds
    lock, done, batches = threading.Lock(), [], []

    def drawer() -> None:   # an on-policy trainer's draws: --draws calls at once, every --draw-every s, as a step plans
        # them (train.plan): runs of consecutive calls of one game, at most train.CHAIN_CALLS each, which share their
        # prompts' beginnings, so the server prefills a few conversations, not --draws unrelated ones
        rng = random.Random(0)
        while time.time() < stop:
            batch: list[dict[str, Any]] = []
            while len(batch) < a.draws:
                g = rng.choice(games)
                at = rng.randrange(len(g))
                batch += g[at:at + min(CHAIN_CALLS, a.draws - len(batch))]
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(a.draws) as ex:
                list(ex.map(lambda c: ask(a.server, c, models[0], tokens=a.draw_tokens), batch))
            batches.append(time.time() - t0)
            time.sleep(max(0.0, a.draw_every - (time.time() - t0)))

    def user(i: int) -> None:   # one user: one game's calls in order, from a different start each user
        calls = games[i % len(games)]
        j = (i * 7) % len(calls)
        while time.time() < stop:
            got = ask(a.server, calls[j % len(calls)], models[i % len(models)])
            with lock:
                done.append({"user": i, "end": time.time(), **got})
            j += 1

    with concurrent.futures.ThreadPoolExecutor(a.users + 1) as pool:
        for f in [pool.submit(user, i) for i in range(a.users)] + ([pool.submit(drawer)] if a.draws else []):
            f.result()
    # calls that began before the clock stopped, all users busy throughout (the last call each user finished counts)
    tps = [d["decode_tps"] for d in done if d["decode_tps"]]
    out = {"tp": a.tp, "users": a.users, "calls": len(done), "models": models,
           "seconds_per_call": round(statistics.fmean(d["seconds"] for d in done), 2),
           "first_token_s": round(statistics.median(d["first_token_s"] for d in done), 2),
           "decode_tps_median": round(statistics.median(tps), 1),
           "seconds_per_1k_tokens": round(1000 * sum(d["seconds"] for d in done) / sum(d["tokens"] for d in done), 2),
           "tokens_per_s_all_users": round(sum(d["tokens"] for d in done) / a.seconds, 1)}
    if a.draws:   # the users' calls' times above are with the draws beside them
        out.update(draws=a.draws, draw_every_s=a.draw_every, draw_tokens=a.draw_tokens,
                   draw_batch_s=round(statistics.fmean(batches), 1) if batches else None)
    print(json.dumps(out))
    if a.out:
        with open(a.out, "a") as fp:
            fp.write(json.dumps({**out, "detail": done}) + "\n")


# -- the trainer's speed (bench-train) -----------------------------------------------------------------------

def cmd_bench_train(a: argparse.Namespace) -> None:
    """--steps steps of the run's phase --phase on this process's GPUs (CUDA_VISIBLE_DEVICES), its held-out loss
    measured every --eval-steps, into --out (never the run's own train/): realtime.train_phase's own code, so its
    calls, plan and update are the run's."""
    import realtime
    import train
    run, out = Path(a.run).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ra = argparse.Namespace(row=a.row, lr=1e-4, batch=a.batch, gpus=a.gpus, lora_rank=128, max_seq_len=65536,
                            max_steps=a.steps, max_minutes=240, eval_steps=a.eval_steps, patience=0, skip_first=3,
                            start_after=0, dose="aggregate", replay_ratio=1.0, first_epochs=2, max_epochs=20,
                            hold_every=10, val_max_items=1000)
    targs = realtime.train_args(ra)
    t0 = time.time()
    trainer = train.make_trainer(targs)
    loaded = time.time() - t0
    init = run / "train" / f"p{a.phase - 1:04d}" / "adapter"
    t0 = time.time()
    done = realtime.train_phase(ra, run, targs, trainer, a.phase, out, init if init.exists() else None)
    total = time.time() - t0
    steps = json.loads((out / "phase.json").read_text())["steps"]
    train_s = sum(s["train_s"] for s in steps)
    print(json.dumps({"gpus": a.gpus, "row": a.row, "phase": a.phase, "load_s": round(loaded, 1),
                      "steps": len(steps), "step_s": round(train_s / len(steps), 1),
                      "tokens_read_per_step": round(statistics.fmean((s.get("read") or {}).get("tokens_read", 0)
                                                                     for s in steps)),
                      "phase_s": round(total, 1), "outside_steps_s": round(total - train_s, 1),
                      "held_out_measurements": len(done.get("held_out") or [])}))


# -- CLI -------------------------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("profile")
    s.add_argument("runs", nargs="+")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("plan")
    s.add_argument("spec", help="JSON: profiles, runs, gpus_per_node, tp, serve, reference, price_per_node_hour "
                                "(docs/method-search.md, \"GPU layouts\")")
    s.add_argument("--deadline", type=float, help="hours from the start: the cheapest plan whose last run is done "
                                                  "within them (without: every run on a node of its own at its fastest)")
    s.add_argument("--budget", type=float, help="dollars: the plan costs at most this")
    s.add_argument("--launch", action="store_true", help="each run's realtime.sh command line on its node: SPEC's "
                                                         "settings (realtime.sh's variables), launch_name as its NAME, "
                                                         "and its GPUs")
    s.add_argument("--hosts", help="the nodes' ssh names, in the plan's order (NODE1, NODE2, ... without)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("bench-serve")
    s.add_argument("--server", default="http://localhost:8101/v1")
    s.add_argument("--games", nargs="+", required=True, help="captured games (rounds/rNNNN/captured/uNNN.json.gz)")
    s.add_argument("--models", required=True, help="the served names the users ask, round robin")
    s.add_argument("--users", type=int, required=True)
    s.add_argument("--seconds", type=float, default=240)
    s.add_argument("--tp", type=int, required=True, help="the server's tensor-parallel GPUs, for the record")
    s.add_argument("--draws", type=int, default=0, help="an on-policy trainer's draws beside the users: so many calls "
                                                       "at once (OPSD: 64), each writing --draw-tokens")
    s.add_argument("--draw-every", type=float, default=30.0, dest="draw_every", help="seconds from one batch to the next")
    s.add_argument("--draw-tokens", type=int, default=740, dest="draw_tokens")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_bench_serve)
    s = sub.add_parser("bench-train")
    s.add_argument("--run", required=True, help="a copy of a real-time run")
    s.add_argument("--row", required=True, choices=("dagger", "opsd", "pg"))
    s.add_argument("--phase", type=int, required=True)
    s.add_argument("--gpus", type=int, required=True)
    s.add_argument("--steps", type=int, default=10)
    s.add_argument("--eval-steps", type=int, default=5, dest="eval_steps")
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_bench_train)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
