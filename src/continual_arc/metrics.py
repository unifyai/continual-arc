"""Scoring and diagnostics.

Every measure starts from the charge of each instance: wrong attempts plus demonstrations,
against a cap. The budget used is 100 times the mean charge over the cap (lower is better);
``summary.json`` carries its complement as ``score``. The headline is transfer
(:func:`transfer`): the budget a run saves on held-out tasks it meets after its stream, against
the stronger of the same learner reset and the bare model on the same tasks. No cache helps on a
task never seen, and both references play the run's own model, so a stronger model does not
score higher for being stronger. Second comes the share learned (:func:`share_learned`): the
budget used on the stream read against the demo-cache floor on the same model and stream, as the
share of the learnable budget a run saves beyond a cache of its demonstrations.

Diagnostics, all derived from the same per-instance records:

* acquisition — cost on first exposure to a task;
* retention   — cost on re-exposure, as a function of the gap (in rounds) since last seen;
* exposure curve — cost as a function of how many earlier rounds held the task;
* learning curve — cumulative and windowed mean cost over the stream;
* checkpoints — the score over the first k stream instances for the horizons a
  configuration names, so runs of different lengths compare where both reached;
* epilogue    — cost on the held-out phase, and forward transfer against one
  reference run (:func:`forward_transfer`), of which the headline takes the stronger of two;
* share learned — the share of the learnable budget a run saves beyond a cache of its demonstrations,
  against the demo-cache floor on the same model and stream (:func:`share_learned`);
* users       — in the collective regime, each user's own view: the spread of per-user stream
  scores, and cost by how many times the user had met the task before (their own exposure,
  the curve an individual run of the same stream draws), beside the team's exposure above.

The horizon of a configuration is an observation window, not a ceiling: streams
are prefix-consistent, so a longer run is a superset of a shorter one and a
checkpoint score never changes when the horizon is extended.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from continual_arc.env import EnvConfig, InstanceRecord
from continual_arc.schedule import (
    PHASE_EPILOGUE,
    PHASE_HELD_OUT_TEST,
    PHASE_SEEN_TEST,
    PHASE_STREAM,
    TEST_HELD_OUT,
    TEST_PHASE,
    TEST_REFERENCE_PHASES,
)

GAP_BUCKETS: tuple[tuple[int, int | None], ...] = ((1, 10), (11, 30), (31, 100), (101, 300), (301, None))
EXPOSURE_BUCKETS: tuple[tuple[int, int | None], ...] = ((0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, None))


def _bucket_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    if lo == hi:
        return str(lo)
    return f"{lo}-{hi}"


def _mean(xs: Sequence[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def score(records: Sequence[InstanceRecord], config: EnvConfig) -> dict:
    """Aggregate cost statistics and the 0..100 score for a set of records."""
    n = len(records)
    if n == 0:
        return {
            "n": 0, "mean_cost": None, "score": None, "solve_rate": None,
            "mean_wrong_attempts": None, "mean_demo_requests": None,
            "give_up_rate": None, "invalid_rate": None,
        }
    mean_cost = sum(r.cost for r in records) / n
    return {
        "n": n,
        "mean_cost": mean_cost,
        "score": 100.0 * (1.0 - mean_cost / config.max_cost),
        "solve_rate": sum(r.solved for r in records) / n,
        "mean_wrong_attempts": sum(r.wrong_attempts for r in records) / n,
        "mean_demo_requests": sum(r.demo_requests for r in records) / n,
        "give_up_rate": sum(r.gave_up for r in records) / n,
        "invalid_rate": sum(r.invalid_submissions for r in records) / n,
    }


def by_phase(records: Sequence[InstanceRecord], config: EnvConfig) -> dict[str, dict]:
    groups: dict[str, list[InstanceRecord]] = defaultdict(list)
    for r in records:
        groups[r.phase].append(r)
    return {phase: score(rs, config) for phase, rs in groups.items()}


def learning_curve(records: Sequence[InstanceRecord], window: int = 50) -> list[dict]:
    """Cumulative and trailing-window mean cost after every instance."""
    out: list[dict] = []
    total = 0.0
    costs: list[float] = []
    for i, r in enumerate(records):
        total += r.cost
        costs.append(r.cost)
        lo = max(0, i + 1 - window)
        out.append(
            {
                "step": r.step,
                "phase": r.phase,
                "cumulative_mean_cost": total / (i + 1),
                "window_mean_cost": sum(costs[lo:]) / (i + 1 - lo),
            }
        )
    return out


def _stream_records(records: Iterable[InstanceRecord]) -> list[InstanceRecord]:
    return [r for r in records if r.episode_type == "benchmark" and r.phase == PHASE_STREAM]


def _benchmark_records(records: Iterable[InstanceRecord]) -> list[InstanceRecord]:
    return [r for r in records if r.episode_type == "benchmark"]


def acquisition(records: Sequence[InstanceRecord], config: EnvConfig) -> dict:
    """Cost on first exposure, over the non-epilogue phases."""
    firsts = [r for r in _stream_records(records) if r.exposure_index == 0]
    return score(firsts, config)


def retention_by_gap(records: Sequence[InstanceRecord]) -> list[dict]:
    """Mean cost on re-exposure, bucketed by the number of rounds since last seen."""
    rows: list[dict] = []
    reexposed = [r for r in _stream_records(records) if r.exposure_index > 0 and r.gap is not None]
    for lo, hi in GAP_BUCKETS:
        rs = [r for r in reexposed if r.gap >= lo and (hi is None or r.gap <= hi)]
        rows.append(
            {
                "gap": _bucket_label(lo, hi),
                "n": len(rs),
                "mean_cost": _mean([r.cost for r in rs]),
                "solve_rate": _mean([float(r.solved) for r in rs]),
                "mean_demo_requests": _mean([float(r.demo_requests) for r in rs]),
            }
        )
    return rows


def cost_by_exposure(records: Sequence[InstanceRecord]) -> list[dict]:
    """Mean cost as a function of how many earlier rounds held the task."""
    rows: list[dict] = []
    stream = _stream_records(records)
    for lo, hi in EXPOSURE_BUCKETS:
        rs = [r for r in stream if r.exposure_index >= lo and (hi is None or r.exposure_index <= hi)]
        rows.append(
            {
                "exposure": _bucket_label(lo, hi),
                "n": len(rs),
                "mean_cost": _mean([r.cost for r in rs]),
                "solve_rate": _mean([float(r.solved) for r in rs]),
                "mean_demo_requests": _mean([float(r.demo_requests) for r in rs]),
            }
        )
    return rows


def cost_by_user_exposure(records: Sequence[InstanceRecord]) -> list[dict]:
    """Mean cost as a function of how many earlier rounds the instance's own user met the task in."""
    rows: list[dict] = []
    stream = _stream_records(records)
    for lo, hi in EXPOSURE_BUCKETS:
        rs = [r for r in stream if r.user_exposure_index >= lo and (hi is None or r.user_exposure_index <= hi)]
        rows.append(
            {
                "exposure": _bucket_label(lo, hi),
                "n": len(rs),
                "mean_cost": _mean([r.cost for r in rs]),
                "solve_rate": _mean([float(r.solved) for r in rs]),
                "mean_demo_requests": _mean([float(r.demo_requests) for r in rs]),
            }
        )
    return rows


def by_user(records: Sequence[InstanceRecord], config: EnvConfig) -> dict:
    """The collective regime's per-user view: how many users, the spread of their stream scores,
    the cost of a user's first meeting with a task (their own acquisition, whether or not a
    teammate met it first) and cost by the user's own exposure. Empty with one user."""
    stream = _stream_records(records)
    groups: dict[int, list[InstanceRecord]] = defaultdict(list)
    for r in _benchmark_records(records):
        groups[r.user].append(r)
    if len(groups) <= 1:
        return {}
    scores = sorted(s for s in (score([r for r in rs if r.phase == PHASE_STREAM], config)["score"]
                                for rs in groups.values()) if s is not None)
    return {
        "n": len(groups),
        "stream_score": {"min": scores[0] if scores else None, "median": _quantile(scores, 0.5),
                         "max": scores[-1] if scores else None},
        "acquisition": score([r for r in stream if r.user_exposure_index == 0], config),
        "cost_by_user_exposure": cost_by_user_exposure(records),
    }


def _quantile(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def checkpoint_scores(records: Sequence[InstanceRecord], config: EnvConfig, checkpoints: Sequence[int]) -> list[dict]:
    """The stream score after the first ``k`` stream instances, for each ``k`` in
    ``checkpoints``. A checkpoint the run has not reached is scored over what it has
    and marked ``reached: False``."""
    stream = _stream_records(records)
    rows: list[dict] = []
    for k in checkpoints:
        rows.append({"instances": int(k), "reached": len(stream) >= k, **score(stream[:k], config)})
    return rows


def per_task(records: Sequence[InstanceRecord]) -> dict[str, dict]:
    groups: dict[str, list[InstanceRecord]] = defaultdict(list)
    for r in records:
        if r.episode_type != "benchmark":
            continue
        groups[r.task_id].append(r)
    out: dict[str, dict] = {}
    for tid, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r.step)
        out[tid] = {
            "n": len(rs),
            "phase": rs[0].phase,
            "mean_cost": _mean([r.cost for r in rs]),
            "first_cost": rs[0].cost,
            "last_cost": rs[-1].cost,
            "solve_rate": _mean([float(r.solved) for r in rs]),
        }
    return out


def resources_summary(records: Sequence[InstanceRecord]) -> dict[str, dict]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        for k, v in (r.resources or {}).items():
            totals[k] += v
            counts[k] += 1
    return {k: {"total": totals[k], "mean": totals[k] / counts[k], "n": counts[k]} for k in sorted(totals)}


def summarize(
    records: Sequence[InstanceRecord], config: EnvConfig, window: int = 50, checkpoints: Sequence[int] = ()
) -> dict:
    """Everything a run report needs, as plain JSON-able data. ``checkpoints`` are
    the configuration's horizons (:attr:`continual_arc.configs.BenchmarkConfig.checkpoints`)."""
    stream = _stream_records(records)
    benchmark = _benchmark_records(records)
    epilogue = [r for r in benchmark if r.phase == PHASE_EPILOGUE]
    addon_records = [r for r in records if r.episode_type != "benchmark"]
    users = by_user(records, config)
    return {
        "config": config.to_json(),
        "max_cost": config.max_cost,
        "overall": score(records, config),
        "benchmark": score(benchmark, config),
        "addons": score(addon_records, config),
        "stream": score(stream, config),
        "held_out_test": score([r for r in benchmark if r.phase == PHASE_HELD_OUT_TEST], config),
        "seen_test": score([r for r in benchmark if r.phase == PHASE_SEEN_TEST], config),
        "epilogue": score(epilogue, config),
        "by_phase": by_phase(records, config),
        "acquisition": acquisition(records, config),
        "retention_by_gap": retention_by_gap(records),
        "cost_by_exposure": cost_by_exposure(records),
        "learning_curve": learning_curve(records, window),
        "checkpoints": checkpoint_scores(records, config, checkpoints),
        "per_task": per_task(records),
        "resources": resources_summary(records),
        **({"users": users} if users else {}),
    }


def forward_transfer(
    run_records: Sequence[InstanceRecord],
    reference_records: Sequence[InstanceRecord],
    config: EnvConfig,
    test: str = TEST_HELD_OUT,
) -> dict:
    """The cost of ``run`` on a frozen test (``test``: held-out or seen) relative to a ``reference`` run on the
    same instances.

    Positive ``mean_transfer`` means the run needed less cost per instance than the reference.
    Instances are matched by task and instance seed, so both runs must serve the same test.
    """
    ref = {(r.task_id, r.instance_seed): r for r in reference_records if r.phase in TEST_REFERENCE_PHASES[test]}
    matched: list[tuple[InstanceRecord, InstanceRecord]] = []
    for r in run_records:
        if r.phase != TEST_PHASE[test]:
            continue
        key = (r.task_id, r.instance_seed)
        if key in ref:
            matched.append((r, ref[key]))
    diffs = [ref_r.cost - run_r.cost for run_r, ref_r in matched]
    return {
        "n_matched": len(matched),
        "run_mean_cost": _mean([a.cost for a, _ in matched]),
        "reference_mean_cost": _mean([b.cost for _, b in matched]),
        "mean_transfer": _mean(diffs),
        "transfer_score_points": (100.0 * _mean(diffs) / config.max_cost) if diffs else None,
        "per_task": {a.task_id: {"run": a.cost, "reference": b.cost, "transfer": b.cost - a.cost} for a, b in matched},
    }


def transfer(
    run_records: Sequence[InstanceRecord],
    reset_records: Sequence[InstanceRecord],
    floor_records: Sequence[InstanceRecord],
    config: EnvConfig,
    test: str = TEST_HELD_OUT,
    reps: int = 2000,
    seed: int = 0,
) -> dict:
    """The headline: the budget ``run`` saves on a frozen test after its stream (``test``: the held-out
    tasks, or new instances of the stream's own; :func:`continual_arc.configs.test_schedule`), against
    the stronger of two references that never had the stream.

    Only ``run``'s records of the test's own phase count, so an epilogue it played while still
    learning never does; a reference's held-out instances may also be a fresh run's epilogue, which
    wipes the learner before each and so is frozen by construction.

    ``reset`` is the same learner wiped before each held-out task (its ``fresh`` run), and
    ``floor`` is the bare model on the same held-out instances (the demo-cache floor's epilogue or
    fresh run, which are alike, since a cache holds nothing for a task never seen)::

        transfer = min(U(reset), U(floor)) - U(run)

    where U is the budget used, 100 times the mean charge over the cap, over the held-out
    instances all three played; positive is better. Held-out tasks are new, so no cache of
    programs or demonstrations helps on them, and both references play the run's own model, so a
    stronger model is not credited for being stronger.

    The reference is whichever of the two used less budget. A learner judged against its own reset
    copy alone could play dumb whenever its memory is empty, answering garbage until it has seen
    a stream, and count the difference as transfer; against the stronger reference, the most
    holding back can earn is the learner's edge over the bare model on new tasks (Dan, 20 Oct
    2026). Taking the stronger of two noisy references understates transfer by at most the noise
    between them and never overstates it. The 95 percent interval is a bootstrap over held-out
    tasks that chooses the reference again in each draw. All three runs must serve the same
    epilogue: instances are matched by task and instance seed."""
    runs = {"run": run_records, "reset": reset_records, "floor": floor_records}
    phases = {"run": (TEST_PHASE[test],), "reset": TEST_REFERENCE_PHASES[test], "floor": TEST_REFERENCE_PHASES[test]}
    held = {
        name: {(r.task_id, r.instance_seed): r for r in records if r.phase in phases[name]}
        for name, records in runs.items()
    }
    keys = sorted(set(held["run"]) & set(held["reset"]) & set(held["floor"]))
    if not keys:
        return {"test": test, "n": 0, "transfer": None}
    cells: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # run, reset, floor, count
    for key in keys:
        c = cells[key[0]]
        for i, name in enumerate(runs):
            c[i] += held[name][key].cost
        c[3] += 1

    def point(tasks: Sequence[str]) -> tuple[float, float, float]:
        n = sum(cells[t][3] for t in tasks)
        run, reset, floor = (100.0 * sum(cells[t][i] for t in tasks) / config.max_cost / n for i in range(3))
        return run, reset, floor

    tasks = sorted(cells)
    run, reset, floor = point(tasks)
    out = {
        "test": test,
        "n": len(keys),
        "tasks": len(tasks),
        "run_budget": run,
        "reset_budget": reset,
        "floor_budget": floor,
        "reference": "reset" if reset <= floor else "floor",
        "reference_budget": min(reset, floor),
        "transfer": min(reset, floor) - run,
        "solved": {name: sum(held[name][k].solved for k in keys) / len(keys) for name in runs},
        "ci95": None,
        "se": None,
        "reps": reps,
    }
    if reps < 10:
        return out
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        b_run, b_reset, b_floor = point([tasks[rng.randrange(len(tasks))] for _ in tasks])
        draws.append(min(b_reset, b_floor) - b_run)
    draws.sort()
    mean = sum(draws) / len(draws)
    out["ci95"] = [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]]
    out["se"] = (sum((x - mean) ** 2 for x in draws) / (len(draws) - 1)) ** 0.5
    return out


def share_learned(
    run_records: Sequence[InstanceRecord],
    floor_records: Sequence[InstanceRecord],
    config: EnvConfig,
    reps: int = 2000,
    seed: int = 0,
) -> dict:
    """The share of the learnable budget ``run`` saves beyond a cache of its demonstrations.

    ``floor`` is the demo-cache floor on the same model and the same stream: the bare model,
    shown again at no cost every pair it has bought for a task. A cache needs no learning, and
    with a strong enough model, re-reading the kept pairs costs only compute, which is never
    charged, so the floor is the zero of the scale. Its one is perfect memory on the same model:
    the floor's own charge on each task's first round, and nothing on any return. Between them
    lies the learnable budget, which shrinks as the model gets stronger; a run's share learned
    is the part of it the run saves::

        lift = U(floor) - U(run)
        learnable budget = U(floor) - U(perfect memory)
        share learned = lift / learnable budget

    where U is the budget used, 100 times the mean charge over the cap, over the stream slots
    both runs played (a run in progress is read as far as it goes). Because both ends of the
    scale are the same model's, a stronger model scores no higher for being stronger, and
    knowledge a learner carries into the stream counts only where it beats the cache. The
    95 percent interval is a bootstrap over tasks, since a task's instances rise and fall
    together. ``share_learned`` is None when the floor leaves nothing to learn.

    Both runs must serve the same schedule: slots are matched by step, and a slot whose task
    or instance seed differs raises."""
    floor = {r.step: r for r in _stream_records(floor_records)}
    cells: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # run, floor, first visits, count
    for r in _stream_records(run_records):
        f = floor.get(r.step)
        if f is None:
            continue
        if (f.task_id, f.instance_seed) != (r.task_id, r.instance_seed):
            raise ValueError(f"slot {r.step} differs between the run ({r.task_id}) and the floor ({f.task_id}): "
                             "the floor must have played the same schedule")
        c = cells[r.task_id]
        c[0] += r.cost
        c[1] += f.cost
        c[2] += f.cost if f.exposure_index == 0 else 0.0
        c[3] += 1

    def point(tasks: Sequence[str]) -> tuple[float, float, float, float]:
        n = sum(cells[t][3] for t in tasks)
        run, flo, first = (100.0 * sum(cells[t][i] for t in tasks) / config.max_cost / n for i in range(3))
        return run, flo, first, n

    tasks = sorted(cells)
    if not tasks:
        return {"n": 0, "share_learned": None}
    run, flo, first, n = point(tasks)
    learnable = flo - first
    out = {
        "n": int(n),
        "run_budget": run,
        "floor_budget": flo,
        "perfect_memory_budget": first,
        "learnable_budget": learnable,
        "lift": flo - run,
        "share_learned": (flo - run) / learnable if learnable > 0 else None,
        "ci95": None,
        "se": None,
        "reps": reps,
    }
    if learnable <= 0 or reps < 10:
        return out
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        b_run, b_flo, b_first, _ = point(sample)
        if b_flo > b_first:
            draws.append((b_flo - b_run) / (b_flo - b_first))
    draws.sort()
    mean = sum(draws) / len(draws)
    out["ci95"] = [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]]
    out["se"] = (sum((x - mean) ** 2 for x in draws) / (len(draws) - 1)) ** 0.5
    return out
