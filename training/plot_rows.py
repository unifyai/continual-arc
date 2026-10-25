"""Plot the paper rows against each other from one or more run folders, while they are still playing.

Four groups of figures, one PNG plus a CSV of its own numbers each, every row overlaid in the same, consistent
colour (:data:`ROW_COLORS`):

- per-round mean cost and solve rate (:func:`per_round_cost_and_solve_rate`), from each row's own ``records.jsonl``
  (``continual_arc.runner.prior_records``, which already drops a trailing partial line);
- cost by a user's own k-th exposure to a task, U(k) (:func:`cost_by_user_exposure`,
  ``continual_arc.metrics.cost_by_user_exposure`` itself -- never re-derived);
- per-round training curves for a trained row (:func:`training_series`, ``train.py``'s own ``phase.json`` under
  ``--train``'s directory): steps taken, the last step's clip fraction, approx KL and entropy, and the items its
  item-builder kept (``pg_items``'/``opd_items``' stats dicts both carry an ``"items"`` count, under ``"games"``
  or ``"items"`` respectively: the one count comparable across the two objectives, read in place of reporting
  "items for opd, games for pg" as two differently-shaped series);
- per-round memory counters (:func:`memory_series`, ``memory.py``'s own ``round_dir(r)/memory.json``): files
  added, changed and removed over the *deliberate* areas alone (``memory.DELIBERATE_AREAS``: programs, skills,
  the harness store, notes -- ``workspace_other``/``other`` are per-game scratch, plotted separately as
  ``scratch_churn``, never summed into the deliberate figures), reads over every area, and the two per-game
  signals (``reused_program``, ``solved_without_demo``).

Every reader tolerates a partial run (a row played fewer rounds than another, or not trained at all yet): a
missing ``records.jsonl``, ``rounds/``, ``memory.json`` or ``phase.json`` simply contributes no points, never an
error, so this can be run again and again while the rows it compares are still going.

Usage (paper.sh's own PLAY_DIR/TRAIN_DIR, one --run per row, --train only for a trained one):
  python3 plot_rows.py --run h0=~/runs/paper/h0/play --run h=~/runs/paper/h/play \\
      --run pg=~/runs/paper/pg/play --train pg=~/runs/paper/pg/train \\
      --run opd=~/runs/paper/opd/play --train opd=~/runs/paper/opd/train --out ~/runs/paper/plots
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

import memory

# A fixed colour per row this paper already names, so a figure's colours agree across every call and every metric;
# any other row name (a future one, or a smoke run's own) gets a stable fallback by its position among the rows
# actually passed, not by name, so it stays the same across this call's own figures.
ROW_COLORS = {"h0": "#4C72B0", "pg": "#DD8452", "opd": "#55A868", "h": "#C44E52", "opsd": "#8172B2"}
_FALLBACK_COLORS = ("#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD")


def color_of(row: str, rows_order: list[str]) -> str:
    if row in ROW_COLORS:
        return ROW_COLORS[row]
    return _FALLBACK_COLORS[rows_order.index(row) % len(_FALLBACK_COLORS)]


# -- readers: records.jsonl, memory.json, phase.json ----------------------------------------------------------

def per_round_cost_and_solve_rate(records: list[Any]) -> dict[int, dict[str, float]]:
    """{round: {"cost": mean, "solve_rate": fraction}}, over however many rounds ``records`` actually holds.
    Filtered through ``continual_arc.metrics``' own stream filter (episode_type "benchmark", not the epilogue
    phase), the same one :func:`cost_by_user_exposure` goes through via ``cost_by_user_exposure`` itself, so the
    two figures on the same page are always computed over the same record set -- never re-derived here."""
    from continual_arc.metrics import _stream_records
    by_round: dict[int, list[Any]] = collections.defaultdict(list)
    for r in _stream_records(records):
        by_round[r.round].append(r)
    return {rnd: {"cost": sum(r.cost for r in rs) / len(rs), "solve_rate": sum(bool(r.solved) for r in rs) / len(rs)}
           for rnd, rs in by_round.items()}


def cost_by_user_exposure(records: list[Any]) -> list[dict[str, Any]]:
    """``continual_arc.metrics.cost_by_user_exposure`` itself, one row per exposure bucket (``n`` 0 where ``records``
    has nothing there yet): the benchmark's own U(k), not re-derived here."""
    from continual_arc.metrics import cost_by_user_exposure as _cost_by_user_exposure
    return _cost_by_user_exposure(records)


def _numbered_subdirs(parent: Path, prefix: str) -> list[tuple[int, Path]]:
    """Every ``parent/{prefix}NNNN`` that exists, in round order; empty if ``parent`` does not exist (a run not
    started, or not trained)."""
    out = []
    if not parent.is_dir():
        return out
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.startswith(prefix) and child.name[len(prefix):].isdigit():
            out.append((int(child.name[len(prefix):]), child))
    return sorted(out)


def memory_series(play_dir: Path) -> dict[int, dict[str, Any]]:
    """Per round, :mod:`memory`'s own ``round_dir(r)/memory.json`` totals: files added/changed/removed summed over
    the *deliberate* areas alone (``memory.DELIBERATE_AREAS``), the per-game scratch areas summed separately into
    one ``scratch_churn`` number (``memory.SCRATCH_AREAS``, added+changed+removed together: scratch is per-game
    noise, not worth breaking down further here), reads summed over every area, and the two per-game signals
    (``reused_program``, ``solved_without_demo``). A round whose memory.json is not there yet (still playing, or
    an older run before this feature) is skipped."""
    out = {}
    for round_, path in _numbered_subdirs(play_dir / "rounds", "r"):
        mem_file = path / "memory.json"
        if not mem_file.is_file():
            continue
        total = json.loads(mem_file.read_text())["total"]
        deliberate = [total["memory"][a] for a in memory.DELIBERATE_AREAS]
        scratch = [total["memory"][a] for a in memory.SCRATCH_AREAS]
        out[round_] = {"added": sum(a["added"] for a in deliberate),
                      "changed": sum(a["changed"] for a in deliberate),
                      "removed": sum(a["removed"] for a in deliberate),
                      "scratch_churn": sum(a["added"] + a["changed"] + a["removed"] for a in scratch),
                      "reads": sum(total["reads"].values()),
                      "reused_program": total["reused_program"],
                      "solved_without_demo": total["solved_without_demo"]}
    return out


def training_series(train_dir: Path) -> dict[int, dict[str, Any]]:
    """Per round, ``train.py``'s own ``phase.json`` (whichever ``train_dir/rN`` rounds have one so far): the steps
    taken, the last step's clip fraction, approx KL and entropy, and the item-builder's kept-item count (see the
    module docstring)."""
    out = {}
    for round_, path in _numbered_subdirs(train_dir, "r"):
        phase_file = path / "phase.json"
        if not phase_file.is_file():
            continue
        phase = json.loads(phase_file.read_text())
        steps = phase.get("steps") or []
        last = steps[-1] if steps else {}
        block = phase.get("games") or phase.get("items") or {}
        out[round_] = {"steps": len(steps), "clip_fraction": last.get("clip_fraction"),
                       "approx_kl": last.get("approx_kl"), "entropy": last.get("entropy_mean"),
                       "items": block.get("items")}
    return out


# -- figures: one PNG + one CSV per metric ---------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_per_round_metric(out_dir: Path, name: str, ylabel: str, series_by_row: dict[str, dict[int, float]],
                          rows_order: list[str]) -> None:
    """One figure over round number, every row with any points overlaid; ``name.png`` and ``name.csv`` (round,
    row, value), skipping a row with nothing to plot (no run given it, or not trained, or no round yet)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    csv_rows = []
    for row in rows_order:
        series = series_by_row.get(row) or {}
        if not series:
            continue
        xs = sorted(series)
        ax.plot(xs, [series[x] for x in xs], marker="o", label=row, color=color_of(row, rows_order))
        csv_rows.extend({"round": x, "row": row, "value": series[x]} for x in xs)
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(name)
    if csv_rows:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _write_csv(out_dir / f"{name}.csv", csv_rows, ["round", "row", "value"])


def plot_cost_by_user_exposure(out_dir: Path, exposure_by_row: dict[str, list[dict[str, Any]]],
                               rows_order: list[str]) -> None:
    """One figure over the shared exposure buckets (``continual_arc.metrics.EXPOSURE_BUCKETS``: every row's own
    list is one entry per bucket, in the same order, so the buckets line up across rows by position); a row's own
    empty bucket (``n`` 0) is left out of its line, not plotted as 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    csv_rows, labels = [], None
    for row in rows_order:
        buckets = exposure_by_row.get(row) or []
        if not buckets:
            continue
        if labels is None:
            labels = [b["exposure"] for b in buckets]
        present = [(i, b) for i, b in enumerate(buckets) if b["n"] > 0]
        if present:
            ax.plot([i for i, _ in present], [b["mean_cost"] for _, b in present], marker="o", label=row,
                   color=color_of(row, rows_order))
        csv_rows.extend({"exposure": b["exposure"], "row": row, "n": b["n"], "mean_cost": b["mean_cost"],
                        "solve_rate": b["solve_rate"]} for _, b in present)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("user's own exposure to the task")
    ax.set_ylabel("mean cost")
    ax.set_title("cost_by_user_exposure")
    if csv_rows:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cost_by_user_exposure.png")
    plt.close(fig)
    _write_csv(out_dir / "cost_by_user_exposure.csv", csv_rows, ["exposure", "row", "n", "mean_cost", "solve_rate"])


# -- orchestration ---------------------------------------------------------------------------------------------

TRAINING_METRICS = (("steps", "steps"), ("clip_fraction", "clip fraction"), ("approx_kl", "approx KL"),
                   ("entropy", "entropy"), ("items", "items kept"))
MEMORY_METRICS = (("added", "files added (deliberate)"), ("changed", "files changed (deliberate)"),
                  ("removed", "files removed (deliberate)"), ("scratch_churn", "scratch churn"),
                  ("reads", "reads"), ("reused_program", "games that reused a saved program"),
                  ("solved_without_demo", "games solved without a demo"))


def plot_runs(runs: dict[str, Path], trains: dict[str, Path], out_dir: Path) -> None:
    """Every figure this module makes, for ``runs`` (row -> play directory) and ``trains`` (row -> train
    directory, only for the rows that have one), into ``out_dir``. Each row's own ``records.jsonl``, ``phase.json``
    series and ``memory.json`` series are each read once (not once per metric plotted from them)."""
    from continual_arc.runner import prior_records

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_order = sorted(runs)   # one fixed order, so every figure's legend and colours agree with each other

    records_by_row = {row: prior_records(path) for row, path in runs.items()}
    training_by_row = {row: training_series(trains[row]) for row in rows_order if row in trains}
    memory_by_row = {row: memory_series(runs[row]) for row in rows_order}

    plot_per_round_metric(out_dir, "per_round_mean_cost", "mean cost",
                          {row: {r: v["cost"] for r, v in per_round_cost_and_solve_rate(recs).items()}
                           for row, recs in records_by_row.items()}, rows_order)
    plot_per_round_metric(out_dir, "per_round_solve_rate", "solve rate",
                          {row: {r: v["solve_rate"] for r, v in per_round_cost_and_solve_rate(recs).items()}
                           for row, recs in records_by_row.items()}, rows_order)
    plot_cost_by_user_exposure(out_dir, {row: cost_by_user_exposure(recs) for row, recs in records_by_row.items()
                                        if recs}, rows_order)

    for metric, ylabel in TRAINING_METRICS:
        series = {row: {r: v[metric] for r, v in rounds.items() if v[metric] is not None}
                 for row, rounds in training_by_row.items()}
        plot_per_round_metric(out_dir, f"training_{metric}", ylabel, series, rows_order)

    for metric, ylabel in MEMORY_METRICS:
        series = {row: {r: v[metric] for r, v in rounds.items()} for row, rounds in memory_by_row.items()}
        plot_per_round_metric(out_dir, f"memory_{metric}", ylabel, series, rows_order)


def _parse_pairs(pairs: list[str]) -> dict[str, Path]:
    out = {}
    for pair in pairs or []:
        name, sep, path = pair.partition("=")
        if not sep:
            raise ValueError(f"expected ROW=PATH, got {pair!r}")
        out[name] = Path(path).expanduser()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", action="append", required=True, metavar="ROW=PLAY_DIR",
                  help="a row's play directory (paper.sh's PLAY_DIR), e.g. pg=~/runs/paper/pg/play; repeatable")
    p.add_argument("--train", action="append", default=[], metavar="ROW=TRAIN_DIR",
                  help="a trained row's train directory (paper.sh's TRAIN_DIR), e.g. "
                       "pg=~/runs/paper/pg/train; repeatable, only for rows with training curves to plot")
    p.add_argument("--out", required=True, help="where the PNGs and CSVs land")
    a = p.parse_args()
    plot_runs(_parse_pairs(a.run), _parse_pairs(a.train), Path(a.out).expanduser())
    print(f"wrote plots to {Path(a.out).expanduser()}")


if __name__ == "__main__":
    main()
