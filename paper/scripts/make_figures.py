"""The paper's generated figures and tables, computed from the runs' records in runs/ at the repository's root.
Nothing in them is drawn or typed by hand.

Run it from paper/, where uv finds the benchmark at the repository's root, so that the runs are read with the code that
wrote them:

    uv run --extra plots python scripts/make_figures.py [names] [--out DIR]

With no names it makes every one of GENERATORS. The figures are written to DIR/figures/ and the tables to
DIR/sections/tables/, the paper's own layout, DIR being paper/build/ unless given, so that each can be compared with the
file the paper sets (scripts/compare.py). Every run a figure or table reads is listed in DIR/figures/SOURCES.json.

The GPT-6 Luna runs are those of runs/runs.json; the open models' runs on ARC-AGI-1 are TEST_ROWS below and Kimi K3.1
Pro's on the second and third tracks TRACK_ROWS, every one of them listed in runs/open-models.json. runs/README.md says
how a run's folder is laid out. Figures follow unify's brand (brand.py): paper, ink, rules instead of fills,
Instrument Sans labels and IBM Plex Mono numerals.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import random
import re
import statistics
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, patheffects  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MarkerPath  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, MultipleLocator, NullLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brand  # noqa: E402
from brand import GREEN, INK, LINE_STRONG, STONE, STONE2  # noqa: E402

brand.apply()

ROOT = Path(__file__).resolve().parent.parent   # paper/
REPO = ROOT.parent   # the repository, whose runs/ holds the runs and from whose root every registry's paths resolve
OUT = ROOT / "build"
FIG = OUT / "figures"
TABLES = OUT / "sections" / "tables"
SOURCES = FIG / "SOURCES.json"
WIDTH = 5.5  # ICLR text width in inches
BOOT_REPS = 400  # task resamples behind every error bar on a single stream
EXPOSURE_BUCKETS = tuple((k, k) for k in range(8)) + ((8, None),)   # every index shown; arc1 serves each task eight times

# The GPT-6 Luna runs, one entry per run: its folder (path), row, model, track, seed, order (mode), label, the harness of
# a harness row other than prime-agent, and "ablation": true for Appendix F's rows (runs/README.md, "Layout").
RUNS: list[dict] = json.loads((REPO / "runs" / "runs.json").read_text())

TRACKS = ("arc1",)
HARNESSES = ("prime-agent", "hermes-agent", "openclaw")
# The orders every stream measure of the GPT-6 Luna runs pools: the track's own and the side experiment's two.
STREAM_ORDERS = ("workstreams", "sequential", "random")

# The ten ARC colours, in the order the grids use them. These are semantic and stay.
ARC_PALETTE = ListedColormap([
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
])


def _rule_axes(ax):
    """Bottom and left rules only, in stone; no ticks poking out."""
    ax.tick_params(length=2.5, width=0.6, colors=STONE, labelcolor=STONE)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(STONE2)


def _mono_ticks(ax, axis="both", size=6.5):
    labs = []
    if axis in ("x", "both"):
        labs += ax.get_xticklabels()
    if axis in ("y", "both"):
        labs += ax.get_yticklabels()
    for lab in labs:
        lab.set_fontproperties(brand.mono(size))


# ---------------------------------------------------------------------------
# Figure: the interaction protocol (one instance)
# ---------------------------------------------------------------------------

PROTOCOL_TITLE, PROTOCOL_NOTE = 7.0, 5.6   # the protocol's box names and its notes, in points


def _box(ax, x, y, w, h, name, note, edge, fill, note_color):
    """A box of the protocol, in points: its name in ink above a note in mono in ``note_color``."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", linewidth=0.8,
                                edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h * 0.64, name, ha="center", va="center", fontsize=PROTOCOL_TITLE, color=INK)
    ax.text(x + w / 2, y + h * 0.3, note, ha="center", va="center", fontproperties=brand.mono(PROTOCOL_NOTE),
            color=note_color)


def _arrow(ax, x, y0, y1, color):
    """A vertical arrow of the protocol at ``x`` from ``y0`` to ``y1``, in points."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                                                mutation_scale=6, shrinkA=0, shrinkB=0))


def _tint(hex_colour, alpha=0.13):
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(hex_colour)
    return (r, g, b, alpha)


def _mix(colour, other, share: float) -> tuple[float, float, float]:
    """``colour`` moved ``share`` of the way to ``other``, opaque."""
    a, b = matplotlib.colors.to_rgb(colour), matplotlib.colors.to_rgb(other)
    return tuple(x + (y - x) * share for x, y in zip(a, b))


def fig_protocol() -> Path:
    """figures/protocol.pdf, Figure 1: one instance, at the width of the figure that sits beside Section 2's text: two
    actions, each a loop from the learner through the environment and back, in the fewest words that say it, its caps
    and the budget left to the text and Equation 1.

    In the brand's palette at the strength of the plots, every box alike, a faint tint of its colour under an
    outline in it with its name in ink, so that no box reads as heavier than the rest: the learner in dusk, the submit
    loop in deep coral and the demonstration loop in deep aqua, so the two costs are told apart at a glance, and the
    environment in haze and slate. Return labels run along their arrows, rotated, so nothing crosses a line or a box.
    The axes' data units are points, so the layout is set in the sizes it prints at.
    """
    w, h, hb = 180.0, 104.0, 24.0
    red, aqua = brand.CORAL_DEEP, brand.AQUA_DEEP
    fig = plt.figure(figsize=(w / 72, h / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    low, high = 0.5 + hb, h - 0.5 - hb
    mid = (low + high) / 2
    _box(ax, 0.5, high, w - 1, hb, "environment", "serves task id + fresh input grid", brand.SLATE, brand.HAZE, STONE)
    _box(ax, 0.5, 0.5, w - 1, hb, "learner", "persists state between instances", brand.DUSK, _tint(brand.DUSK, 0.12),
         brand.DUSK)
    for left, name, note, colour, back, label in ((4, "submit grid", "wrong: cost 1", red, 78, "correct?"),
                                                  (94, "request demo", "cost 1", aqua, 168, "demo pair")):
        _box(ax, left, mid - hb / 2, 64, hb, name, note, colour, _tint(colour, 0.12), colour)
        _arrow(ax, left + 32, low, mid - hb / 2, colour)
        _arrow(ax, left + 32, mid + hb / 2, high, colour)
        _arrow(ax, back, high, low, colour)
        ax.text(back + 2.5, mid, label, ha="left", va="center", rotation=90,
                fontproperties=brand.mono(PROTOCOL_NOTE), color=colour)
    out = FIG / "protocol.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure: the three modes as timelines on arc1
# ---------------------------------------------------------------------------

def fig_modes() -> Path:
    """figures/modes.pdf, Figure 2: arc1's stream at the track's seed, as the GPT-6 Luna runs were served it
    (served_runs), without the frozen tests that follow it: one row per task in the order the stream first meets them,
    a mark per instance and a hairline in the task's color over each spell, across the page and short, since it sits
    beside the text that defines the stream. Each mark is as wide as an instance's slot allows, 1.8 points, and half as
    tall again, 2.7, and each spell 1.3 points thick, so that the stream does not read as thin on a screen. The tasks
    take the brand's hues in turn, at full strength, 40 percent toward white and 30 percent toward ink, so that
    neighboring rows differ and the whole reads as the brand's rather than a rainbow."""
    served = served_runs("arc1")
    if "workstreams" not in served:
        raise SystemExit(f"arc1: no registered run at seed {TRACK_SEED} was served the stream")
    run = served["workstreams"]
    sched = run.schedule
    stream = [(i, s) for i, s in enumerate(sched.slots) if s.phase == "stream"]
    order: list[str] = []
    for _, s in stream:
        if s.task_id not in order:
            order.append(s.task_id)
    row = {t: i for i, t in enumerate(order)}
    hues = (brand.DUSK, brand.CORAL_DEEP, brand.AQUA_DEEP, brand.MAUVE_DEEP, brand.GOLD, brand.SKY, brand.VERMILLION,
            GREEN, brand.ROSE)
    palette = ([matplotlib.colors.to_rgb(c) for c in hues] + [_mix(c, "#ffffff", 0.4) for c in hues]
               + [_mix(c, INK, 0.3) for c in hues])
    colour = {t: palette[i % len(palette)] for t, i in row.items()}

    fig = plt.figure(figsize=(WIDTH, 1.65))
    ax = fig.add_axes([0.055, 0.215, 0.94, 0.765])
    # the generator records every spell as [task, first step, one past the last step]
    parts = sched.meta.get("parts") or [sched.meta]
    spells = next((pt["spells"] for pt in parts if pt and "spells" in pt), [])
    for t, x0, x1 in spells:
        ax.plot([x0 - 0.5, x1 - 0.5], [row[t], row[t]], color=colour[t], alpha=0.4, lw=1.3, solid_capstyle="butt",
                zorder=1)
    # a rectangle marker, whose taller side is the marker's size, so that its width stays mark_w at any height
    mark_w, mark_h = 1.8, 2.7
    r = mark_w / mark_h
    mark = MarkerPath([(-r, -1), (r, -1), (r, 1), (-r, 1), (-r, -1)], closed=True)
    ax.scatter([i for i, _ in stream], [row[s.task_id] for _, s in stream], s=mark_h ** 2,
               c=[colour[s.task_id] for _, s in stream], marker=mark, linewidths=0, rasterized=True, zorder=2)
    n_tasks = len(order)
    ax.set_xlim(-2, ROUNDS + 2)
    ax.set_ylim(n_tasks - 0.5, -0.7)
    ticks = [0] + list(range(4, n_tasks, 5))
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(k + 1) for k in ticks], fontproperties=brand.mono(5.2))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _rule_axes(ax)
    ax.set_xticks(range(0, ROUNDS + 1, 25))
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(brand.mono(5.2))
    ax.set_xlabel("instance", fontsize=6.5, color=STONE, labelpad=1.5)
    ax.set_ylabel("task", fontsize=6.5, color=STONE, labelpad=1.5)
    out = FIG / "modes.pdf"
    fig.savefig(out, dpi=300)
    _report("modes", [run])
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure: an input grid does not determine its task
# ---------------------------------------------------------------------------

def _draw_grid(ax, grid):
    import numpy as np
    a = np.array(grid)
    ax.imshow(a, cmap=ARC_PALETTE, vmin=0, vmax=9, interpolation="nearest")
    h, w = a.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_edgecolor(STONE2)


def find_ambiguous_pairs(n_inputs: int = 2, max_cells: int = 100, want: int = 3):
    """Inputs generated for task A that task B's verifier also accepts with a different output.

    A verifier that raises on an input is treated as 'rule does not apply'. We keep
    pairs where B accepts every sampled input of A, gives a valid grid, and that grid
    differs from A's output on every sample. Symmetric pairs (A's verifier also accepts
    B's inputs) come first, then small grids; both tasks of a shown pair are distinct.
    """
    from continual_arc.rearc import ReArc

    re = ReArc()
    ids = re.task_ids
    samples: dict[str, list] = {}
    for t in ids:
        try:
            samples[t] = [re.generate(t, seed=s) for s in range(n_inputs)]
        except Exception:
            continue
    cands = []
    for a, pairs in samples.items():
        if any(len(p.input) * len(p.input[0]) > max_cells for p in pairs):
            continue
        for b in ids:
            if b == a:
                continue
            outs, ok = [], True
            for p in pairs:
                o = re.verify(b, p.input)
                if o is None or o == p.output or o == p.input:
                    ok = False; break
                outs.append(o)
            if ok:
                cells = sum(len(p.input) * len(p.input[0]) for p in pairs)
                cands.append({"a": a, "b": b, "cells": cells, "input": pairs[0].input,
                              "out_a": pairs[0].output, "out_b": outs[0]})
    accepted = {(c["a"], c["b"]) for c in cands}
    for c in cands:
        c["symmetric"] = (c["b"], c["a"]) in accepted
    cands.sort(key=lambda c: (not c["symmetric"], c["cells"], c["a"], c["b"]))
    chosen, used = [], set()
    for c in cands:
        if c["a"] in used or c["b"] in used:
            continue
        chosen.append(c); used.update((c["a"], c["b"]))
        if len(chosen) == want:
            break
    return chosen, len(cands), len(samples)


def fig_taskid() -> Path:
    chosen, n_cands, n_tasks = find_ambiguous_pairs()
    (FIG / "taskid.json").write_text(json.dumps(
        {"n_tasks_sampled": n_tasks, "n_ordered_pairs": n_tasks * (n_tasks - 1),
         "n_ambiguous_pairs": n_cands,
         "shown": [{k: c[k] for k in ("a", "b", "symmetric")} for c in chosen]}, indent=1))
    n = len(chosen)
    # compact: rows a little over an inch, a narrow label column, the three grids close together
    fig = plt.figure(figsize=(4.4, 1.05 * n + 0.3))
    ratios = [0.78, 1, 1, 1]
    gs = fig.add_gridspec(n, 4, width_ratios=ratios, wspace=0.08, hspace=0.12,
                          left=0.01, right=0.99, top=0.93, bottom=0.01)
    span = 0.98 / sum(ratios)
    for c, title in enumerate(["input", "output under A", "output under B"]):
        x = 0.01 + span * (ratios[0] + c + 0.5)
        fig.text(x, 0.95, title, ha="center", va="bottom", fontsize=8.5, color=STONE)
    for r, c in enumerate(chosen):
        lab = fig.add_subplot(gs[r, 0]); lab.axis("off")
        lab.text(0.0, 0.5, f"A = {c['a']}\nB = {c['b']}", ha="left", va="center",
                 fontproperties=brand.mono(7.5), color=STONE, linespacing=1.4)
        for k, grid in enumerate((c["input"], c["out_a"], c["out_b"])):
            ax = fig.add_subplot(gs[r, k + 1])
            _draw_grid(ax, grid)
            ax.set_anchor("C")
    out = FIG / "taskid.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _in_bucket(x: int, lo: int, hi: int | None) -> bool:
    return x >= lo and (hi is None or x <= hi)


@dataclass
class Run:
    spec: dict
    path: Path
    records: list            # InstanceRecord, in order, the dropped ones left out
    summary: dict
    config: object           # EnvConfig, or Arc3Config on ARC-AGI-3
    protocol: dict
    n_slots: int             # length of the schedule served
    epilogue_start: int | None  # first schedule index of the epilogue, if the schedule has one
    schedule: object = None  # the Schedule served
    dropped: tuple = ()      # the records an outage played through (DROPPED), which no measure reads

    @property
    def row(self) -> str: return self.spec["row"]
    @property
    def model(self) -> str: return self.spec["model"]
    @property
    def track(self) -> str: return self.spec["track"]
    @property
    def mode(self) -> str: return self.spec["mode"]
    @property
    def seed(self): return self.spec["seed"]
    @property
    def label(self) -> str: return self.spec.get("label") or self.path.name
    @property
    def harness(self) -> str:
        return self.spec.get("harness") or ("prime-agent" if self.row.startswith("H") else "")
    @property
    def complete(self) -> bool: return self.n_slots > 0 and self.written >= self.n_slots
    @property
    def written(self) -> int:
        """How far the run has played: its records, the dropped ones among them."""
        return len(self.records) + len(self.dropped)
    @property
    def max_cost(self) -> float: return float(self.summary.get("max_cost") or getattr(self.config, "max_cost", 16.0))

    def stream_records(self) -> list:
        return [r for r in self.records if r.episode_type == "benchmark" and r.phase != "epilogue"]

    def complete_buckets(self) -> list[bool]:
        """One flag per bucket of EXPOSURE_BUCKETS: True when every stream slot of the schedule that falls in the
        bucket has a record, so that a figure can mark a visit that not every task has had."""
        last = {}
        for i, (slot, info) in enumerate(zip(self.schedule.slots, self.schedule.annotate())):
            if slot.phase != "epilogue":
                last[info.exposure_index] = i
        return [all(i < self.written for x, i in last.items() if _in_bucket(x, lo, hi)) for lo, hi in EXPOSURE_BUCKETS]

    def source_entry(self) -> dict:
        return {"path": str(self.path.relative_to(REPO)), "label": self.label, "row": self.row,
                "harness": self.harness or None, "model": self.model, "track": self.track, "seed": self.seed,
                "mode": self.mode, "n_records": len(self.records), "n_dropped": len(self.dropped),
                "n_slots": self.n_slots, "complete": self.complete}


# The instances an outage played through, by the folder of their play from the repository's root (data/outages.json):
# every instance with a model call while the provider refused requests, found in the plays' call logs. No measure reads
# them, whatever their outcome, since a selection by outcome would move the measures it cleans. Their runs played on
# past them, so each is dropped alone; the measures pair runs instance by instance, so a dropped instance leaves every
# comparison it was in. A call the provider refused outright is in no call log, so an instance whose every call was
# refused is found from its record instead (_refused).
DROPPED: dict[str, set[int]] = defaultdict(set)
for _outage in json.loads((ROOT / "data" / "outages.json").read_text()):
    for _play, _steps in _outage["dropped"].items():
        DROPPED[_play].update(_steps)


def _refused(record) -> bool:
    """Whether the provider refused, or answered with nothing, every model call of the record's instance: calls made,
    no token read or written, nothing charged, unsolved. Such an instance measures the provider, as an outage's do, and
    the run played on past it."""
    res = record.resources or {}
    got = (res.get("llm_output_tokens", 0) or 0) + (res.get("llm_input_tokens", 0) or 0)
    return (not record.solved) and (res.get("llm_requests", 0) or 0) > 0 and not (res.get("llm_cost_usd", 0) or 0) \
        and not got


def _load_run(spec: dict, path: Path) -> Run | None:
    """The run whose benchmark files are in ``path`` (records.jsonl, summary.json, schedule.json, config.json and
    meta.json), its dropped instances (DROPPED, _refused) set apart; None where it holds no records."""
    from continual_arc.arc3.env import Arc3Config
    from continual_arc.configs import protocol_of
    from continual_arc.env import EnvConfig, InstanceRecord
    from continual_arc.metrics import summarize
    from continual_arc.schedule import Schedule

    rec_path = path / "records.jsonl"
    if not rec_path.exists():
        return None
    records = [InstanceRecord.from_json(json.loads(line)) for line in rec_path.read_text().splitlines() if line.strip()]
    if not records:
        return None
    gone = DROPPED.get(str(path.relative_to(REPO)), set()) | {r.step for r in records if _refused(r)}
    dropped = tuple(r for r in records if r.step in gone)
    records = [r for r in records if r.step not in gone]
    # an ARC-AGI-3 run charges by its games' baselines (Arc3Config), every other by attempts and demonstrations
    cfg = json.loads((path / "config.json").read_text())
    config = (Arc3Config if "cap_multiplier" in cfg else EnvConfig).from_json(cfg)
    schedule = Schedule.load(path / "schedule.json")
    epilogue_start = next((i for i, s in enumerate(schedule.slots) if s.phase == "epilogue"), None)
    summary = None
    if (path / "summary.json").exists():
        summary = json.loads((path / "summary.json").read_text())
        if (summary.get("overall") or {}).get("n") != len(records):
            summary = None  # written before the run's last records, or counting its dropped instances
    if summary is None:
        summary = summarize(records, config)
    protocol = summary.get("protocol")
    if not protocol:
        meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
        protocol = meta.get("protocol")
    protocol = protocol or protocol_of(schedule, config)
    spec = dict(spec, path=str(path))
    if spec.get("seed") is None:
        spec["seed"] = protocol.get("seed")
    spec.setdefault("mode", protocol.get("mode") or "workstreams")
    spec.setdefault("track", protocol.get("track") or "arc1")
    for key in ("track", "mode"):
        if protocol.get(key) and protocol[key] != spec[key]:
            print(f"  warning {path}: the registry says {key} {spec[key]}, the run {protocol[key]}", file=sys.stderr)
    return Run(spec, path, records, summary, config, protocol, len(schedule), epilogue_start, schedule, dropped)


# Appendix F's ablation rows, marked "ablation": true in runs.json, which only its table reads
ABLATION_ROWS = ("HA", "HPL", "HPL0", "M0 program library")


def load_runs(ablations: bool = False) -> list[Run]:
    """Every GPT-6 Luna run of RUNS but the ablation rows, or with ``ablations`` the ablation rows alone."""
    return [_load_run(entry, REPO / entry["path"]) for entry in RUNS
            if bool(entry.get("ablation") or entry["row"] in ABLATION_ROWS) == ablations]


@functools.cache
def registered_runs() -> list[Run]:
    return load_runs()


@functools.cache
def ablation_runs() -> list[Run]:
    """Appendix F's ablation rows (ABLATION_ROWS), which registered_runs leaves out."""
    return load_runs(ablations=True)


def served_runs(track: str) -> dict[str, Run]:
    """Per order, a GPT-6 Luna run of ``track`` at the track's seed, whose schedule (Run.schedule, read from its own
    schedule.json) is the one the figures draw for that order. Every run of an order must have been served the same
    stream; what follows it differs (a learning-on epilogue of 25 or 100 held-out tasks, or none), and no figure reads
    it."""
    from dataclasses import astuple

    by_mode: dict[str, dict[tuple, Run]] = defaultdict(dict)
    for r in registered_runs() + ablation_runs():
        if r.track == track and str(r.seed) == str(TRACK_SEED):
            by_mode[r.mode].setdefault(tuple(astuple(s) for s in r.schedule.slots if s.phase == "stream"), r)
    out = {}
    for mode, kinds in by_mode.items():
        if len(kinds) > 1:
            raise SystemExit(f"{track} {mode}: its runs were served {len(kinds)} different schedules, as "
                             f"{', '.join(r.path.name for r in kinds.values())} were")
        out[mode] = next(iter(kinds.values()))
    return out


def _boot_task_means(cells: dict[str, list[tuple[float, int]]], n_buckets: int, reps: int = BOOT_REPS):
    """Bootstrap over tasks. ``cells[task]`` holds (sum of cost, count) per bucket; returns the
    per-bucket list of resampled means (None where the resample has no record)."""
    import random as _random
    tasks = sorted(cells)
    rng = _random.Random(0)
    out: list[list[float]] = [[] for _ in range(n_buckets)]
    if not tasks:
        return out
    for _ in range(reps):
        draw = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        for b in range(n_buckets):
            s = sum(cells[t][b][0] for t in draw); n = sum(cells[t][b][1] for t in draw)
            if n:
                out[b].append(s / n)
    return out


def _boot_buckets(run: "Run", block: str, n_buckets: int, buckets=None):
    """16th and 84th percentiles over task resamples of the bucket means of one run."""
    from continual_arc.metrics import GAP_BUCKETS
    if buckets is None:
        buckets = GAP_BUCKETS if block == "retention_by_gap" else EXPOSURE_BUCKETS
    cells: dict[str, list[list[float]]] = {}
    for r in run.stream_records():
        if block == "retention_by_gap":
            if r.exposure_index <= 0 or r.gap is None:
                continue
            x = r.gap
        else:
            x = r.exposure_index
        row = cells.setdefault(r.task_id, [[0.0, 0] for _ in buckets])
        for i, (lo, hi) in enumerate(buckets):
            if _in_bucket(x, lo, hi):
                row[i][0] += float(r.cost); row[i][1] += 1
    dists = _boot_task_means({t: [tuple(c) for c in row] for t, row in cells.items()}, len(buckets))
    lo, hi = [], []
    for d in dists[:n_buckets]:
        if len(d) < 10:
            lo.append(None); hi.append(None); continue
        d = sorted(d)
        lo.append(d[int(0.16 * len(d))]); hi.append(d[int(0.84 * len(d))])
    return lo, hi


def _to_budget(vals, cap: float):
    """Costs to the share of the feedback budget used, in percent: 100 cost / cap; None kept as None."""
    return [None if v is None else 100.0 * float(v) / cap for v in vals]


TRACK_SEED = 0   # a track is one fixed stream, its configuration's draw at seed 0


def _report(name: str, runs: list[Run]) -> None:
    """Print which runs a figure or table read, and record them in SOURCES.json."""
    print(f"{name}: {len(runs)} run(s)")
    for r in runs:
        state = "complete" if r.complete else f"{r.written} of {r.n_slots}"
        print(f"  {r.row:<18} {r.harness or '-':<13} {r.model:<24} {r.mode:<15} seed {r.seed!s:<4} {state}: "
              f"{r.path.relative_to(REPO)}")
    sources = json.loads(SOURCES.read_text()) if SOURCES.exists() else {}
    sources[name] = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "runs": [r.source_entry() for r in runs]}
    SOURCES.write_text(json.dumps(sources, indent=1) + "\n")


_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _effort_rank(model: str) -> int:
    """Panels and legends run from the least to the most reasoning effort: 'GPT-6 Luna (high)' -> 4."""
    m = re.search(r"\((none|minimal|low|medium|high|xhigh|max)\)", model)
    return _EFFORTS.index(m.group(1)) if m else len(_EFFORTS)


def table_status() -> Path:
    """sections/tables/status.tex, the macros the captions and the text print while a run they read is short of its
    schedule: \\partialnote, \\partialcurvenote, \\inprog and \\partialtextnote for the GPT-6 Luna runs, and \\ablinprog and
    \\ablationnote for Appendix F's rows, each empty once every run is complete."""
    partial = [r for r in registered_runs() if not r.complete]
    note = ("* marks a run in progress, over the instances it reached." if partial else "")
    curve = ("* marks a run in progress; a curve stops where a run behind it has not met every instance."
             if partial else "")
    text = ("Numbers marked * come from runs still in progress, over the instances they reached." if partial else "")
    abl = [r for r in ablation_runs() if not r.complete]
    abl_note = ("Numbers marked * come from runs still in progress, over the instances they have reached." if abl else "")
    out = TABLES / "status.tex"
    out.write_text("% generated by scripts/make_figures.py status; do not edit\n"
                   f"\\newcommand{{\\partialnote}}{{{note}}}\n"
                   f"\\newcommand{{\\partialcurvenote}}{{{curve}}}\n"
                   f"\\newcommand{{\\inprog}}{{{'*' if partial else ''}}}\n"
                   f"\\newcommand{{\\partialtextnote}}{{{text}}}\n"
                   f"\\newcommand{{\\ablinprog}}{{{'*' if abl else ''}}}\n"
                   f"\\newcommand{{\\ablationnote}}{{{abl_note}}}\n")
    return out


def _pooled_gain(pairs: list[tuple["Run | None", "Run | None"]], keep=None):
    """What run a saves against run b, pooled over several pairs of runs (the orders of a cell), in points of the
    budget, on the stream slots both have reached, with a bootstrap standard error over tasks, the number of matched
    slots, and whether a run behind it is in progress. ``keep`` picks slots by run a's record. None when a run is
    missing or no slot matches."""
    cells: dict[str, list[tuple[float, int]]] = {}
    n, partial = 0, False
    for a, b in pairs:
        if a is None or b is None:
            return None
        partial |= not (a.complete and b.complete)
        other = {rec.step: rec for rec in b.stream_records()}
        for rec in a.stream_records():
            o = other.get(rec.step)
            if o is None or (keep is not None and not keep(rec)):
                continue
            s, k = cells.get(rec.task_id, [(0.0, 0)])[0]
            cells[rec.task_id] = [(s + 100 * (float(o.cost) - float(rec.cost)) / a.max_cost, k + 1)]
            n += 1
    if not n:
        return None
    mean = sum(s for (s, _), in cells.values()) / n
    d = _boot_task_means(cells, 1)[0]
    se = None
    if len(d) >= 10:
        m = sum(d) / len(d); se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    return mean, se, n, partial


# Appendix F's comparisons: (row a, row b, label) for each harness, then the bare model's; what a saves against b
ABLATION_HARNESS_ROWS = [("H", "H0", "H, as it ships"), ("HA", "H0", "H-A, asked to store"),
                         ("HPL0", "H0", "H-E\\textsubscript{0}, expert guidance, nothing kept"),
                         ("HPL", "H0", "H-E, expert guidance"),
                         ("HPL", "HPL0", "H-E against H-E\\textsubscript{0}, what keeping adds"),
                         ("HPL", "M program library", "H-E against PL")]
ABLATION_BARE_ROWS = [("M0 program library", "F stateless", "PL\\textsubscript{0} against the stateless floor"),
                      ("M program library", "F stateless", "PL against the stateless floor"),
                      ("M program library", "M0 program library", "PL against PL\\textsubscript{0}")]


def ablation_comparisons(track: str = "arc1"):
    """[(group, label, row a, row b, {effort model: _pooled_gain})] for Appendix F, pooled over STREAM_ORDERS."""
    pool = [r for r in registered_runs() + ablation_runs() if r.track == track]
    models = sorted({r.model for r in pool if r.row in ABLATION_ROWS}, key=_effort_rank)
    orders = STREAM_ORDERS

    def find(row, harness, model, mode):
        c = [r for r in pool if r.row == row and (r.harness == harness or not harness) and r.model == model
             and r.mode == mode]
        c = [r for r in c if r.seed in (0, "0")] or c
        return c[0] if c else None
    out = []
    for h in ("prime-agent", "hermes-agent", "openclaw"):
        for a, b, label in ABLATION_HARNESS_ROWS:
            hb = "" if b.startswith(("M ", "F ")) else h
            out.append((h, label, a, b, {m: _pooled_gain([(find(a, h, m, o), find(b, hb, m, o)) for o in orders])
                                         for m in models}))
    for a, b, label in ABLATION_BARE_ROWS:
        out.append(("bare model", label, a, b, {m: _pooled_gain([(find(a, "", m, o), find(b, "", m, o)) for o in orders])
                                                for m in models}))
    return models, out


def table_ablations(track: str = "arc1") -> Path:
    """sections/tables/ablations-arc1.tex: Appendix F's table, the budget each ablation row saves against its reference
    on the same stream slots, pooled over STREAM_ORDERS, with a bootstrap standard error over tasks in small type
    and * for a comparison with a run still in progress."""
    models, rows = ablation_comparisons(track)
    efforts = [m[m.find("(") + 1:m.find(")")] if "(" in m else m for m in models]
    lines = ["% generated by scripts/make_figures.py ablations; do not edit",
             "\\begin{tabular}{ll" + "r" * len(models) + "}", "\\toprule",
             " & method & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{e}}}" for e in efforts) + " \\\\", "\\midrule"]
    # each effort's largest saving bold, its standard error with it, among the savings whose 95 percent interval,
    # 1.96 standard errors either side, lies above zero (_bold_measurable)
    strong = {}
    for m in models:
        flags = _bold_measurable([vals[m][0] if vals.get(m) else None for _, _, _, _, vals in rows],
                                 [vals[m][0] - 1.96 * vals[m][1] if vals.get(m) and vals[m][1] is not None else None
                                  for _, _, _, _, vals in rows], 1)
        strong.update({(i, m): f for i, f in enumerate(flags)})
    last = None
    for i, (group, label, _, _, vals) in enumerate(rows):
        if last is not None and group != last:
            lines.append("\\midrule")
        cells = []
        for m in models:
            g = vals.get(m)
            if g is None:
                cells.append("--")
                continue
            mean, se, _, partial = g
            value, bar = f"{mean:+.1f}", (f"\\scriptsize$\\pm${se:.1f}" if se is not None else "")
            if strong[i, m]:
                value, bar = "\\textbf{" + value + "}", ("\\textbf{" + bar + "}" if bar else "")
            spread = f"\\makebox[2.3em][l]{{\\hspace{{0.45em}}{bar}}}" if bar else "\\makebox[2.3em][l]{}"
            cells.append(f"\\makebox[2.15em][r]{{{value}}}{spread}" + ("\\rlap{*}" if partial else ""))
        lines.append(f"{group if group != last else ''} & {label} & " + " & ".join(cells) + " \\\\")
        last = group
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = TABLES / f"ablations-{track}.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# The share learned (continual_arc.metrics.share_learned): every system read against itself reset, the same system
# wiped before every instance of the same stream, as the share of the learnable budget it saves over it; the harness is
# the environment a system learns in, so it is the baseline
# ---------------------------------------------------------------------------

LEARNED_REPS = 2000   # task resamples behind each 95 percent interval and standard error
ADJUSTED_REPS = 200_000   # direct Bonferroni tails of the paired task bootstrap, for the 2x2 planes
# each system's reset: the same system wiped before every instance, on the same model, harness and stream; a system
# that keeps nothing between instances is its own
LEARNED_RESET = {"F stateless": "F stateless", "F demo-cache": "F stateless", "H0": "H0", "H": "H0",
                 "M program library": "M0 program library"}
_LEARNED_CACHE: dict[tuple[str, str], dict] = {}


def _learned(run: "Run", reset: "Run") -> dict:
    """continual_arc.metrics.share_learned of a run against its reset, cached by the two run directories."""
    from continual_arc.metrics import share_learned
    key = (str(run.path), str(reset.path))
    if key not in _LEARNED_CACHE:
        _LEARNED_CACHE[key] = share_learned(run.records, reset.records, run.config, reps=LEARNED_REPS)
    return _LEARNED_CACHE[key]


# ---------------------------------------------------------------------------
# The frozen tests: Qwen3.5-9B's and GLM-5.3 Flash's rows, notes in the harness against weights, each read on its stream
# and on the two tests it plays from its end state, the seen test (new instances of the stream's tasks: memorization)
# and the held-out test (tasks never met: generalization). These runs are runs/open-models.json's.
# ---------------------------------------------------------------------------

PLAYERS = {"qwen": "Qwen3.5-9B", "glm": "GLM-5.3 Flash (medium)"}
FROZEN_TESTS = ("seen", "held-out")   # continual_arc.schedule.TESTS, in the order the figures read them
TEST_NAMES = {"seen": "seen test", "held-out": "held-out test"}
STREAM_CONFIG = "arc1"   # every row plays its workstreams stream at seed 0, at the row's number of users


@dataclass(frozen=True)
class TestRow:
    """One row at one number of users, and the runs its measures are read against, each named by its folder from the
    repository's root (runs/arc1/<run>), its stream's files there and each play of a frozen test in
    <run>/frozen/<play>/play/ (runs/README.md). Every measure reads the row against itself reset, the same system wiped
    before each instance, since the harness is the environment it learns in; the bare model is no reference.
    ``reset`` is the run whose plays from an empty state are that reset on the frozen tests: the row's wiped twin of
    the same player at one user, or for a trained row the base model in the same harness on the trained rows' own
    server. Each test is played several times on the same instances with fresh sampling (test_plays): the run's own
    plays from its end-of-stream state, frozen/<test>/ and frozen/<test>-K/, and a reset's plays from an empty state,
    frozen/<test>-reference/ and frozen/<test>-reference-K/, so that a wiped row is read half its plays against the
    other half, and the spread of such rows is the noise of playing a test. ``twin`` is the run whose stream is that
    reset on the stream, the zero of the share learned: the same wiped twin (for a trained row, the base model in the
    same harness through OpenRouter), at the row's number of users (continual_arc.metrics.share_learned). ``pooled``
    names the resets whose plays of the held-out test, the same at any number of users, are plays of the same wiped
    system too: a row at 2 or 4 users reads its reset's one play of that test with the four of its reset at one
    user."""
    player: str
    row: str
    users: int
    run: str
    reset: str
    twin: str
    pooled: tuple[str, ...] = ()


def _arc1(name: str) -> str:
    return f"runs/arc1/{name}"


# Each one-user row's reset: its wiped twin, its stream replayed from round 0 and four plays of each frozen test, its own
# or, for a reset, from an empty state.
_Q_H0, _Q_HPL0, _Q_M0, _Q_F, _Q_DC = map(_arc1, (
    "Q0-prime-agent-qwen3.5-9b-ws0-u1", "QE0-prime-agent-qwen3.5-9b-ws0-u1", "QP0-qwen3.5-9b-ws0-u1",
    "QC0-qwen3.5-9b-ws0-u1", "QC-qwen3.5-9b-ws0-u1"))
_G_H0, _G_HPL0, _G_M0, _G_F, _G_DC = map(_arc1, (
    "G0-prime-agent-glm-5.3-flash-ws0-u1", "GE0-prime-agent-glm-5.3-flash-ws0-u1", "GP0-glm-5.3-flash-ws0-u1",
    "GC0-glm-5.3-flash-ws0-u1", "GC-glm-5.3-flash-ws0-u1"))
# The trained rows' reset on the frozen tests: the base model in row h0's harness on the trained rows' own server, at
# the model card's sampling that the trained rows play at, four plays of each test from an empty state, and one of each
# at two and four users, whose seen tests serve every user's own tasks.
_Q_SERVED = _arc1("Q0-prime-agent-qwen3.5-9b-served-tests")
_Q_SERVED_U = {users: f"{_Q_SERVED}-u{users}" for users in (2, 4)}
# The teacher's rows: the teacher plays the whole stream in the harness and writes everything in it, its own rows H and
# H-E, and the student plays each instance from the harness as the teacher left it before that instance, keeping
# nothing. Each stream, and the reset that is its zero, played in the same window, since the providers that served the
# other resets' streams changed under them; each row's tests are four plays from the teacher's end state.
_T_H, _T_H_FROM_E, _T_E = map(_arc1, ("QT-prime-agent-qwen3.5-9b-ws0-u1", "QTE-prime-agent-qwen3.5-9b-ws0-u1",
                                      "QETE-prime-agent-qwen3.5-9b-ws0-u1"))
_T_H0, _T_E0 = map(_arc1, ("Q0-prime-agent-qwen3.5-9b-ws0-u1-same-window",
                           "QE0-prime-agent-qwen3.5-9b-ws0-u1-same-window"))


def _own(row: str, name: str, twin: str, one: str) -> list["TestRow"]:
    """Qwen3.5-9B's row at two and four users, runs/arc1/<name>-ws0-u<n>: its stream replayed from round 0 with every
    user on 25 tasks of their own, read on the stream and on both tests against its wiped twin ``twin`` at the same
    number of users, and on the held-out test against ``one``, the same twin at one user, too. The seen test then
    serves every user's own tasks (configs.test_schedule), which only a run with the same users played; the held-out
    test is the same at any number of users, so each of its plays by the wiped system, at whatever number of users, is
    a play of the same reset. A row that keeps nothing is its own twin."""
    return [TestRow("qwen", row, n, _arc1(f"{name}-ws0-u{n}"), _arc1(f"{twin}-ws0-u{n}"), _arc1(f"{twin}-ws0-u{n}"),
                    (one,)) for n in (2, 4)]


# The registry of the open models' rows on ARC-AGI-1: one entry per row and number of users, with the runs each is read
# against (runs/open-models.json lists the same). Only DAgger and Qwen3.5-9B's baselines play at two and four users, the
# wiped harness Q0 among them as DAgger's reset there. At two and four users every row and its reset play each test
# once (_own), so a row that keeps nothing has no second play of the seen test to read itself against there, and the
# tables of every number of users list it at one user alone.
TEST_ROWS = [
    # Qwen3.5-9B, notes in the harness: its own (h, ha, hpl), GLM's (hg, hgpl, hplg), and the program library (m)
    TestRow("qwen", "h", 1, _arc1("Q-prime-agent-qwen3.5-9b-ws0-u1"), _Q_H0, _Q_H0),
    TestRow("qwen", "ha", 1, _arc1("QA-prime-agent-qwen3.5-9b-ws0-u1"), _Q_H0, _Q_H0),
    TestRow("qwen", "hpl", 1, _arc1("QE-prime-agent-qwen3.5-9b-ws0-u1"), _Q_HPL0, _Q_HPL0),
    TestRow("qwen", "hg", 1, _T_H, _Q_H0, _T_H0),
    TestRow("qwen", "hgpl", 1, _T_H_FROM_E, _Q_H0, _T_H0),
    TestRow("qwen", "hplg", 1, _T_E, _Q_HPL0, _T_E0),
    TestRow("qwen", "m", 1, _arc1("QP-qwen3.5-9b-ws0-u1"), _Q_M0, _Q_M0),
    *_own("m", "QP-qwen3.5-9b", "QP0-qwen3.5-9b", _Q_M0),
    # Qwen3.5-9B, weights: LoRA trained on the stream by DAgger, its users sharing one adapter, read on the tests
    # against the base model in the same harness on its own server, at the same number of users, and on the stream
    # against the wiped harness through OpenRouter
    TestRow("qwen", "DAgger", 1, _arc1("QDE-qwen3.5-9b-ws0-u1"), _Q_SERVED, _Q_H0),
    TestRow("qwen", "DAgger", 2, _arc1("QDE-qwen3.5-9b-ws0-u2"), _Q_SERVED_U[2],
            _arc1("Q0-prime-agent-qwen3.5-9b-ws0-u2"), (_Q_SERVED,)),
    TestRow("qwen", "DAgger", 4, _arc1("QDE-qwen3.5-9b-ws0-u4"), _Q_SERVED_U[4],
            _arc1("Q0-prime-agent-qwen3.5-9b-ws0-u4"), (_Q_SERVED,)),
    # Qwen3.5-9B, the floors and the wiped rows, which carry nothing to a new task; at one user each is read against
    # its own plays, so their spread on the held-out test is the noise every other row is read against
    TestRow("qwen", "f", 1, _Q_F, _Q_F, _Q_F),
    *_own("f", "QC0-qwen3.5-9b", "QC0-qwen3.5-9b", _Q_F),
    TestRow("qwen", "dc", 1, _Q_DC, _Q_F, _Q_F),
    *_own("dc", "QC-qwen3.5-9b", "QC0-qwen3.5-9b", _Q_F),
    TestRow("qwen", "h0", 1, _Q_H0, _Q_H0, _Q_H0),
    *_own("h0", "Q0-prime-agent-qwen3.5-9b", "Q0-prime-agent-qwen3.5-9b", _Q_H0),   # DAgger's twin at 2 and 4 users
    TestRow("qwen", "hpl0", 1, _Q_HPL0, _Q_HPL0, _Q_HPL0),
    TestRow("qwen", "m0", 1, _Q_M0, _Q_M0, _Q_M0),
    *_own("m0", "QP0-qwen3.5-9b", "QP0-qwen3.5-9b", _Q_M0),
    # GLM-5.3 Flash, the trained rows' teacher, playing the same rows itself
    TestRow("glm", "h", 1, _arc1("G-prime-agent-glm-5.3-flash-ws0-u1"), _G_H0, _G_H0),
    TestRow("glm", "ha", 1, _arc1("GA-prime-agent-glm-5.3-flash-ws0-u1"), _G_H0, _G_H0),
    TestRow("glm", "hpl", 1, _arc1("GE-prime-agent-glm-5.3-flash-ws0-u1"), _G_HPL0, _G_HPL0),
    TestRow("glm", "m", 1, _arc1("GP-glm-5.3-flash-ws0-u1"), _G_M0, _G_M0),
    TestRow("glm", "f", 1, _G_F, _G_F, _G_F),
    TestRow("glm", "dc", 1, _G_DC, _G_F, _G_F),
    TestRow("glm", "h0", 1, _G_H0, _G_H0, _G_H0),
    TestRow("glm", "hpl0", 1, _G_HPL0, _G_HPL0, _G_HPL0),
    TestRow("glm", "m0", 1, _G_M0, _G_M0, _G_M0),
]
TEST_BY_RUN = {r.run: r for r in TEST_ROWS}   # every row by its run folder, each row's twin among them
# the teacher's rows' twins, wiped resets played beside them that no table lists
TEST_BY_RUN |= {run: TestRow("qwen", row, 1, run, run, run) for run, row in ((_T_H0, "h0"), (_T_E0, "hpl0"))}


NOTES, TEACHER_NOTES, WEIGHTS, FLOORS = "notes in the harness", "notes the teacher writes", "weights", "baselines"
TEACHER_PLAYS = "the teacher plays"   # a family alone: its rows are the notes group's, with the stronger model playing
# Each row's name in the paper, its group, and its line, told apart within its group by its dash and its marker; a
# wiped row is hollow. A hyphen sets off each part of a name: who keeps the notes, H the student in its harness or T
# the teacher writing them, each followed by what it is given, A asked to store or E expert guidance, the program
# library's procedure. So H-T-E is the teacher alone given the guidance and H-E-T-E both, as the glossary of Figure 3
# has it. The names are the method's parts, not its models or domain, so that the other tracks, with other models and
# other domains' expert procedures, name their rows the same way.
# A row's marker says what its name says: a diamond wherever there is expert guidance (E), a square where the harness is
# asked to store (A) and a circle otherwise, a plus for a baseline, hollow for a wiped row. Every line is solid but the
# stateless floor's, dotted to single it out, since each row has a color and a marker of its own.
SOLID, DOTTED = "-", (0, (1, 1.5))
TEST_LOOK = {
    "h": ("H", NOTES, dict(ls=SOLID, marker="o")),
    "ha": ("H-A", NOTES, dict(ls=SOLID, marker="s")),
    "hpl": ("H-E", NOTES, dict(ls=SOLID, marker="D")),
    "hg": ("H-T", TEACHER_NOTES, dict(ls=SOLID, marker="o")),
    "hgpl": ("H-T-E", TEACHER_NOTES, dict(ls=SOLID, marker="D")),
    "hplg": ("H-E-T-E", TEACHER_NOTES, dict(ls=SOLID, marker="D")),
    "DAgger": ("DAgger", WEIGHTS, dict(ls=SOLID, marker="D")),
    "m": ("PL", FLOORS, dict(ls=SOLID, marker="P")),
    "f": ("C0 stateless", FLOORS, dict(ls=DOTTED, marker="P")),
    "dc": ("C demo-cache", FLOORS, dict(ls=SOLID, marker="P")),
    "h0": ("H0", FLOORS, dict(ls=SOLID, marker="o", hollow=True)),
    "hpl0": ("H-E0", FLOORS, dict(ls=SOLID, marker="D", hollow=True)),
    "m0": ("PL0", FLOORS, dict(ls=SOLID, marker="P", hollow=True)),
}
# Each family's colors, a ramp from light to dark, a row the deeper the more guidance it is given: reds for the notes
# the student keeps on its own, greens for notes the teacher writes for the student, blues for a stronger model playing
# and keeping its own notes (the teacher, or GPT-6 Luna), so that red, green and blue are the rows that keep notes in
# the harness, and purple, off those primaries, for the one method that changes the weights. Each ramp steps evenly in
# CIELAB lightness, about 11 to 14 points a shade, and stops at about L 46 to 50 rather than near-black, so that the
# guided rows read as the same family. Every baseline is gray, since the baselines are not methods but the bounds of
# what a simple system reaches: the program library, the upper bound, solid and darker than the stateless floor, the
# lower bound, which alone is dotted, its wiped twin paler, and the demo-cache floor the stateless floor's gray; the two
# stay about 14 points of lightness apart, so that their test points, which have no line to tell them by, still read
# as two. Only the methods have hues, so the rows that learn, DAgger among them, stand out. A point of Figure 3 carries
# no letter: its name beside it says what it is.
REDS = ("#f28b82", "#e75b48", "#bb463b")
GREENS = ("#7fcf8e", "#4aab61", "#36864e")
BLUES = ("#86b4ee", "#4f8ce0", "#2563c9")
PURPLE, GRAY, DARK_GRAY, PALE_GRAY = "#7a4fc4", "#a1a19c", "#7d7c78", "#c9c8c3"
FAMILY_LOOK = {NOTES: (REDS[1], True), TEACHER_NOTES: (GREENS[1], True), TEACHER_PLAYS: (BLUES[2], True),
               WEIGHTS: (PURPLE, True),
               FLOORS: (DARK_GRAY, False)}
# Each row's color by the model that plays it, in the plots where several rows share a panel, its family's shade
# deepening with the guidance the row is given: the wiped twin palest, the plain harness light, the harness asked to
# save in the middle and the harness given the expert guidance darkest.
_BASELINE_COLOR = {"m": DARK_GRAY, "m0": PALE_GRAY, "dc": GRAY, "f": GRAY}
ROW_COLOR = {
    "qwen": {"h": REDS[0], "ha": REDS[1], "hpl": REDS[2], "hg": GREENS[0], "hgpl": GREENS[1], "hplg": GREENS[2],
             "DAgger": PURPLE, "h0": "#e9aaa2", "hpl0": "#f6cdc8", **_BASELINE_COLOR},
    "glm": {"h": BLUES[0], "ha": BLUES[1], "hpl": BLUES[2], "h0": "#9dbce8", "hpl0": "#cfe0f6", **_BASELINE_COLOR},
}
ROW_COLOR["luna"] = ROW_COLOR["glm"]
# The general key shows each family as a circle, then guidance as a black diamond: color identifies where learning
# is kept and shape identifies guidance, including QDE's expert-guided teacher.
PLANE_KEY = ((FLOORS, "o", None, "baseline"), (NOTES, "o", None, "keeps notes in the harness"),
             (TEACHER_NOTES, "o", None, "teacher writes them"), (TEACHER_PLAYS, "o", None, "teacher plays"),
             (WEIGHTS, "o", None, "teacher trains the weights"), (NOTES, "D", INK, "expert guidance"))
# A row's name as the paper prints it, one letter for each concept, since an open-ended benchmark has no default player
# or harness: the model that plays, in the harness unless a letter says otherwise, then what the row adds. So QE is
# Qwen3.5-9B given the expert guidance, QT the teacher writing its notes, QTE the teacher given the guidance and QETE
# both; QDE is DAgger with an expert-guided teacher. GPT-6 Luna, which plays three harnesses and whose letter the share
# learned L has taken, keeps TEST_LOOK's names.
PLAYER_LETTER = {"qwen": "Q", "glm": "G"}
ROW_LETTERS = {"h": "", "ha": "A", "hpl": "E", "hg": "T", "hgpl": "TE", "hplg": "ETE", "DAgger": "DE", "m": "P", "f": "C0",
               "dc": "C", "h0": "0", "hpl0": "E0", "m0": "P0"}
# a key says in words what a figure with no glossary of its own could not otherwise say
ROW_WORDS = {"f": "stateless floor", "dc": "demo cache", "m": "program library", "h0": "wiped", "hpl0": "wiped",
             "m0": "wiped"}


def row_name(row: str, player: str) -> str:
    """The row's name as the paper prints it (PLAYER_LETTER, ROW_LETTERS)."""
    return TEST_LOOK[row][0] if player == "luna" else PLAYER_LETTER[player] + ROW_LETTERS[row]


def _key_name(row: str, player: str) -> str:
    """The row's name in a figure's key, with its words (ROW_WORDS); GPT-6 Luna's baselines by their words alone."""
    word = ROW_WORDS.get(row)
    if word is None:
        return row_name(row, player)
    return word if player == "luna" and row in ("f", "dc", "m") else f"{row_name(row, player)} {word}"


def _row_names(row: str, players) -> str:
    """The row's name for each of ``players``, as a table with a column per model prints it: "Q, G"."""
    return ", ".join(_tex_name(row_name(row, p)) for p in players)


# The rows the main text leaves to the appendix: H-A, the harness asked after every verdict to save liberally, so
# Figure 3, Table 1 and Figure 4 go without it, and the appendix's tables by users and on solved instances keep it.
APPENDIX_ROWS = ("ha",)


def _row_family(row: str, player: str = "qwen") -> str:
    """The family a row is drawn in: its group, except that a stronger model keeping its own notes is its own."""
    group = TEST_LOOK[row][1]
    return TEACHER_PLAYS if group == NOTES and player != "qwen" else group


def _tex_name(name: str) -> str:
    """A row's name as the paper sets it, a wiped row's 0 as a subscript."""
    return re.sub(r"0$", r"\\textsubscript{0}", name)


TEST_MIN_POINT = 5   # Figure 4 draws a visit once this many of its instances are in, hollow until every task has had it
VISITS = 8           # arc1 serves each of its 25 tasks eight times
ROUNDS = 200         # its stream, in rounds


@functools.cache
def _stream_of(users: int):
    """The stream slots of arc1's schedule at seed 0 and ``users`` users: its workstreams stream, the stream every row
    of the registry plays slot for slot."""
    from continual_arc.configs import build_schedule, get_config, with_users
    full = build_schedule(with_users(get_config(STREAM_CONFIG), users), TRACK_SEED, order="workstreams")
    return replace(full, slots=[s for s in full.slots if s.phase == "stream"])


@functools.cache
def _test_of(test: str, users: int = 1) -> frozenset:
    """The (task, instance seed) pairs of one frozen test after arc1's stream at ``users`` users
    (configs.test_schedule): the held-out test is the same at any number of users, and the seen test serves every
    user's own tasks."""
    from continual_arc.configs import get_config, test_schedule, with_users
    schedule = test_schedule(with_users(get_config(STREAM_CONFIG), users), TRACK_SEED, test)
    return frozenset((s.task_id, s.instance_seed) for s in schedule.slots)


@functools.cache
def played(row: TestRow) -> Run | None:
    """The row's stream, judged against the whole stream, after a check that it is the one the row was served; None
    for a run that played the frozen tests alone."""
    path = REPO / row.run
    spec = {"row": row.row, "model": PLAYERS[row.player], "track": "arc1", "seed": TRACK_SEED, "mode": "workstreams",
            "users": row.users, "label": f"{row.player} {row.row} u{row.users}"}
    run = _load_run(spec, path)
    if run is None:
        return None
    full = _stream_of(row.users)
    served = [(s.task_id, s.instance_seed, s.user) for s in run.schedule.slots if s.phase == "stream"]
    if served != [(s.task_id, s.instance_seed, s.user) for s in full.slots[:len(served)]]:
        raise SystemExit(f"{path}: its schedule is not {STREAM_CONFIG}'s workstreams stream at seed {TRACK_SEED} and "
                         f"{row.users} users")
    return replace(run, records=[r for r in run.records if r.phase == "stream"],
                   dropped=tuple(r for r in run.dropped if r.phase == "stream"), schedule=full, n_slots=len(full),
                   epilogue_start=None)


def _test_users(run: Run) -> int:
    """The number of users of the stream a play of a frozen test follows, which its schedule records for a seen test
    at more than one user (configs.test_schedule's stream_users); 1 otherwise."""
    return (run.schedule.meta or {}).get("stream_users", 1)


def _whole_test(run: Run, path: Path, test: str) -> Run | None:
    """``run`` once it holds every instance of the frozen test and nothing else, None while it is short."""
    from continual_arc.schedule import TEST_PHASE
    want = _test_of(test, _test_users(run))
    got = {(r.task_id, r.instance_seed) for r in (*run.records, *run.dropped) if r.phase == TEST_PHASE[test]}
    if got - want:
        raise SystemExit(f"{path}: instances that are not in {STREAM_CONFIG}'s {TEST_NAMES[test]} at "
                         f"{_test_users(run)} users")
    return run if got == want else None


REPLAY = ".replay"   # beside a play, the games it played again after a context overrun


def _replayed(path: Path, run: Run) -> Run:
    """``run``, the frozen test played in ``path``, with each game a context overrun cut in place of the test's own
    once their replay beside it is whole: ``<play>.replay/play``, every game in which the trained rows' server refused
    a call because its prompt and the 65,536-token reply it asked passed the 131,072-token window, played again from
    that call on with the harness fixed, the calls before it served as recorded (the appendix's "The window on our own
    server"). A replayed game takes the cut game's step, matched by task and instance; a replay not yet whole leaves
    the test as played."""
    from continual_arc.env import InstanceRecord
    from continual_arc.metrics import summarize
    folder = path.parent.with_name(path.parent.name + REPLAY) / "play"
    if not (folder / "records.jsonl").exists():
        return run
    again = [InstanceRecord.from_json(json.loads(line)) for line in (folder / "records.jsonl").read_text().splitlines()
             if line.strip()]
    want = json.loads((folder.parent / "schedule.json").read_text())["users"]
    if len(again) != want:
        print(f"  warning {folder}: {len(again)} of its {want} games replayed, so {path} is read as played",
              file=sys.stderr)
        return run
    key = lambda r: (r.task_id, r.instance_seed)  # noqa: E731
    by = {key(r): r for r in again}
    was = {key(r): r for r in (*run.records, *run.dropped)}
    if len(by) != want or not by.keys() <= was.keys():
        raise SystemExit(f"{folder}: replays of games that {path} did not play")
    kept = [r for r in run.records if key(r) not in by]
    dropped = [r for r in run.dropped if key(r) not in by]
    for k, r in by.items():   # a replayed game the provider refused whole is left out as _load_run leaves one out
        r = replace(r, step=was[k].step)
        (dropped if _refused(r) else kept).append(r)
    records = sorted(kept, key=lambda r: r.step)
    summary = summarize(records, run.config)
    if run.summary.get("protocol"):
        summary["protocol"] = run.summary["protocol"]
    return replace(run, records=records, summary=summary, spec={**run.spec, "replayed": len(by)},
                   dropped=tuple(sorted(dropped, key=lambda r: r.step)))


def _frozen_play(path: Path, test: str, spec: dict) -> Run | None:
    """One play of a frozen test in ``path``, its cut games replayed (_replayed), once every instance of it is in;
    None before."""
    loaded = _load_run(spec, path)
    return None if loaded is None else _whole_test(_replayed(path, loaded), path, test)


def _play_names(frozen: Path, test: str, reference: bool) -> list[str]:
    """The names of a run's plays of ``test`` in its frozen/ folder, the first and then by number: its own from its
    end-of-stream state, TEST and TEST-K, or with ``reference`` its plays from an empty state, TEST-reference and
    TEST-reference-K."""
    stem = f"{test}-reference" if reference else test
    name = re.compile(re.escape(stem) + r"(?:-([2-9]|[1-9][0-9]+))?")
    found = [(int(m.group(1) or 1), p.name) for p in frozen.glob(stem + "*")
             if p.is_dir() and (m := name.fullmatch(p.name))]
    return [n for _, n in sorted(found)]


@functools.cache
def test_plays(run: str, test: str, reference: bool = False) -> tuple[Run, ...]:
    """Every whole play of a frozen test by ``run`` (a TestRow's run or reset), its own from its end-of-stream state
    or, with ``reference``, from an empty state (_play_names), in order; a play still short is left out."""
    frozen = REPO / run / "frozen"
    plays = []
    for name in _play_names(frozen, test, reference):
        spec = {"row": run, "model": "", "track": "arc1", "seed": TRACK_SEED, "mode": f"{test} test",
                "label": f"{run}, {name}"}
        play = _frozen_play(frozen / name / "play", test, spec)
        if play is not None:
            plays.append(play)
    return tuple(plays)


def _wiped_plays(reset: str, test: str) -> tuple[Run, ...]:
    """Every whole play of a frozen test by a reset: its plays from an empty state and, where the reset is a row of the
    registry, that row's own plays of the test, which a system wiped before every instance plays as it would from
    nothing, the own plays first: independent plays of the same wiped system."""
    plays = test_plays(reset, test, reference=True)
    return test_plays(reset, test) + plays if reset in TEST_BY_RUN else plays


def reset_plays(row: TestRow, test: str) -> tuple[Run, ...]:
    """Every whole play of a frozen test by the row's reset (_wiped_plays) and, on the held-out test, which is the same
    at any number of users, by the resets it pools with it (TestRow.pooled)."""
    pooled = row.pooled if test == "held-out" else ()
    return tuple(p for reset in (row.reset, *pooled) for p in _wiped_plays(reset, test))


def _mean_records(plays: tuple[Run, ...], test: str, phase: str) -> list:
    """One record for each instance of the test that every play holds, under ``phase``, its charge and its solve the
    means of the plays'."""
    from continual_arc.schedule import TEST_REFERENCE_PHASES
    by: dict[tuple[str, int], list] = defaultdict(list)
    for play in plays:
        for r in play.records:
            if r.phase in TEST_REFERENCE_PHASES[test]:
                by[(r.task_id, r.instance_seed)].append(r)
    return [replace(rs[0], phase=phase, cost=statistics.mean(r.cost for r in rs),
                    solved=statistics.mean(r.solved for r in rs)) for rs in by.values() if len(rs) == len(plays)]


def _sides(row: TestRow, test: str) -> tuple[tuple[Run, ...], tuple[Run, ...]]:
    """The row's plays of a frozen test and its reset's, the two sides each measure reads: a row that keeps nothing is
    its own reset and reads its plays taken alternately, half against the other half. Either side is empty until it has
    a whole play; two sides that played different instances, a seen test at another number of users, stop the build."""
    resets = reset_plays(row, test)
    plays, resets = (resets[0::2], resets[1::2]) if row.run == row.reset else (test_plays(row.run, test), resets)
    if plays and resets and {_test_users(p) for p in (*plays, *resets)} != {_test_users(plays[0])}:
        raise SystemExit(f"{row.run}: its plays of the {TEST_NAMES[test]} and its reset {row.reset}'s follow streams "
                         "with different numbers of users")
    return plays, resets


def _transfer(plays: tuple[Run, ...], resets: tuple[Run, ...], test: str) -> dict:
    """continual_arc.metrics.transfer of the mean of ``plays`` against the mean of ``resets``."""
    from continual_arc.metrics import transfer
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    return transfer(_mean_records(plays, test, TEST_PHASE[test]),
                    _mean_records(resets, test, TEST_REFERENCE_PHASES[test][0]), plays[0].config, test=test,
                    reps=LEARNED_REPS)


@functools.cache
def test_transfer(row: TestRow, test: str) -> dict | None:
    """T on a frozen test (continual_arc.metrics.transfer): the budget the row saves on it against its reset, the same
    system wiped before each instance, on the same instances, in points, with the 95 percent interval over tasks. Each
    side's charge on an instance is the mean over its plays (test_plays, reset_plays), so that no one play's luck
    decides where the row stands, nor lends itself to every row read against the same reset. A row that keeps nothing
    is its own reset, and reads its plays taken alternately, half against the other half. None until the row and its
    reset have each played the whole test."""
    plays, resets = _sides(row, test)
    if not plays or not resets:
        return None
    t = _transfer(plays, resets, test)
    return dict(t, runs=[*plays, *resets], plays=(len(plays), len(resets))) if t["n"] else None


def _charge(record) -> float:
    """What an instance costs a learner: its charge."""
    return record.cost


def _solve(record) -> float:
    """How far an instance was solved, the mean over its plays where it is the mean of several."""
    return float(record.solved)


def _share_cells(run_records: list, reset_records: list, test: str, cost=_charge) -> dict[str, list[float]]:
    """Each task's ``cost`` on a frozen test over the instances both sides played, matched by task and instance seed as
    continual_arc.metrics.transfer matches them: [the run's sum, its reset's sum]."""
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    run, reset = ({(r.task_id, r.instance_seed): cost(r) for r in records if r.phase in phases}
                  for records, phases in ((run_records, (TEST_PHASE[test],)), (reset_records, TEST_REFERENCE_PHASES[test])))
    cells: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for key in sorted(set(run) & set(reset)):
        cells[key[0]][0] += run[key]
        cells[key[0]][1] += reset[key]
    return dict(cells)


def _share_of(cells: dict[str, list[float]], tasks: list[str]) -> float:
    return 1.0 - sum(cells[t][0] for t in tasks) / sum(cells[t][1] for t in tasks)


def _draws(stat, tasks: list[str]) -> list[float]:
    """``stat`` over LEARNED_REPS resamples of ``tasks``, seed 0, as continual_arc.metrics.transfer resamples them, and
    the 2.5th and 97.5th percentiles of the draws."""
    rng = random.Random(0)
    draws = sorted(stat([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(LEARNED_REPS))
    return [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]]


def _share_saved(cells: dict[str, list[float]]) -> dict:
    """The share saved on a frozen test, 1 - U(run) / U(reset): the share of its reset's spend that the run no longer
    pays, with its 95 percent interval over task resamples. On the seen test every task has been met in the stream, so
    perfect memory spends nothing there and the reset's whole spend is the learnable budget: there it is the share
    learned of Equation 3. On the held-out test it is how much cheaper the run became on tasks it never met,
    as a share of what it spent before."""
    tasks = sorted(cells)
    return {"share": _share_of(cells, tasks), "ci95": _draws(lambda ts: _share_of(cells, ts), tasks),
            "tasks": len(tasks), "_cells": cells, "_metric": "share"}


def _solved_ratio(cells: dict[str, list[float]]) -> dict:
    """How many times as much of a frozen test the run solves as its reset, from _share_cells of _solve, with its 95
    percent interval over task resamples: the solve rate's reading against the reset, 1 for no change, and small for a
    model already near the cap, where a share of its few failures would read one instance as a large gain. A resample
    on which the reset solved nothing has nothing to read and is drawn again."""
    def ratio(ts: list[str]) -> float:
        return sum(cells[t][0] for t in ts) / sum(cells[t][1] for t in ts)
    tasks = sorted(cells)
    if not tasks or sum(cells[t][1] for t in tasks) <= 0:
        return {"ratio": None, "ci95": None, "tasks": len(tasks), "_cells": cells, "_metric": "solved_ratio"}
    rng = random.Random(0)
    draws: list[float] = []
    while len(draws) < LEARNED_REPS:
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        if sum(cells[t][1] for t in sample) > 0:
            draws.append(ratio(sample))
    draws.sort()
    return {"ratio": ratio(tasks), "ci95": [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]],
            "tasks": len(tasks), "_cells": cells, "_metric": "solved_ratio"}


def _adjusted_ci(cells: dict[str, list[float]], family_size: int, metric: str = "share") -> tuple[float, float] | None:
    """Direct paired-task bootstrap tails at .025/k and 1-.025/k for a family of k learners.

    Each cell holds the learner's and reset's sums for one task, after averaging that side's frozen plays. Both sides
    resample the same tasks. Families are separate for each model/effort, user setting, track, test and metric. These
    intervals describe task uncertainty conditional on the recorded plays and final learned state; plays and training
    runs are not resampled. The existing pointwise 95 percent intervals are calculated separately and unchanged.
    """
    if family_size < 1:
        raise ValueError("an adjusted interval needs at least one learner in its family")
    if metric not in ("share", "solved_ratio"):
        raise ValueError(f"unknown relative metric: {metric}")
    data = tuple((task, float(cell[0]), float(cell[1])) for task, cell in sorted(cells.items()))
    return _adjusted_ci_cached(data, family_size, metric)


@functools.cache
def _adjusted_ci_cached(data: tuple[tuple[str, float, float], ...], family_size: int,
                        metric: str) -> tuple[float, float] | None:
    import numpy as np
    values = np.asarray([(own, reset) for _, own, reset in data], dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("task charges and solve counts must be finite and nonnegative")
    if values[:, 1].sum() <= 0:
        return None
    rng = np.random.default_rng(0)
    draws = np.empty(ADJUSTED_REPS, dtype=float)
    written = 0
    while written < ADJUSTED_REPS:
        # Bounded temporary arrays even for a wide test. Zero-reference samples have no defined relative measure.
        size = min(4096, ADJUSTED_REPS - written)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        own = values[:, 0][indices].sum(axis=1)
        reset = values[:, 1][indices].sum(axis=1)
        valid = reset > 0
        ratios = own[valid] / reset[valid]
        batch = 1.0 - ratios if metric == "share" else ratios
        draws[written:written + len(batch)] = batch
        written += len(batch)
    tail = 0.025 / family_size
    lo, hi = np.quantile(draws, (tail, 1.0 - tail))
    return float(lo), float(hi)


def _row_share_cells(row: TestRow, test: str) -> dict[str, list[float]] | None:
    """_share_cells of the row's plays against its reset's, each side's charge on an instance the mean of its plays,
    and a row that keeps nothing read half its plays against the other half, as test_transfer reads them."""
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    plays, resets = _sides(row, test)
    if not plays or not resets:
        return None
    return _share_cells(_mean_records(plays, test, TEST_PHASE[test]),
                        _mean_records(resets, test, TEST_REFERENCE_PHASES[test][0]), test)


@functools.cache
def test_share(row: TestRow, test: str) -> dict | None:
    """The share saved on a frozen test (_share_saved) from the plays test_transfer reads; None until it has T."""
    t = test_transfer(row, test)
    return None if t is None else dict(_share_saved(_row_share_cells(row, test)), runs=t["runs"], plays=t["plays"])


def frozen_measure(row: TestRow, test: str) -> dict | None:
    """What the paper reports for a row on a frozen test, as ``value`` with its ``ci95``: the share saved
    (test_share), a fraction of the reset's own spend on the test, so that a system whose reset already spends little
    is not read as having gained less (Section 2.4); None until both sides have played the whole test."""
    d = test_share(row, test)
    return None if d is None else dict(d, value=d["share"])


# what each test's measure is called on an axis, and its digits in a table
MEASURE_AXIS = {"seen": "share saved on the seen test", "held-out": "share saved on the held-out test"}
MEASURE_DIGITS = {"seen": 2, "held-out": 2}


def stream_learned(row: TestRow) -> dict | None:
    """The share learned on the stream (_learned) against the row's twin, itself reset before every instance of the
    stream, once both have played the whole stream; None before."""
    run, twin = played(row), played(TEST_BY_RUN[row.twin])
    if run is None or twin is None or not (run.complete and twin.complete):
        return None
    return dict(_learned(run, twin), runs=[run, twin])


def _chance(player: str, cost=_charge, read=_share_saved) -> tuple[float, float] | None:
    """Chance on the seen and on the held-out test for ``player`` at one user (_chance_of): its rows of CHANCE_ROWS, each
    its own reset (reset_plays)."""
    rows = [r for r in TEST_ROWS if r.player == player and r.users == 1 and r.row in CHANCE_ROWS]
    return _chance_of([functools.partial(reset_plays, r) for r in rows], cost, read)


def _chance_of(sides, cost=_charge, read=_share_saved) -> tuple[float, float] | None:
    """How far from zero the share saved (frozen_measure), or with ``cost`` and ``read`` another share of its kind, falls
    95 times in 100 on the seen and on the held-out test for a learner that keeps nothing between instances, played as
    often as its reset, n plays against n. Each of ``sides`` gives one such learner's plays of a test. A learner that
    keeps nothing is its own reset, so its n plays split into halves in every way, and the half-width of the 95 percent
    interval of the measure of one half against the other, its mean over the splits scaled from n/2 plays a side to n
    (sqrt(1/2): a mean of plays has the variance of one play over their number), is that learner's chance; the result
    is the mean over the learners. None until every one has played each test at least twice."""
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    out = []
    for test in FROZEN_TESTS:
        halves = []
        for plays_of in sides:
            plays = plays_of(test)
            n = len(plays)
            if n < 2:
                return None
            widths = []
            for a in itertools.combinations(range(1, n), n // 2 - 1):   # each split once: play 0 on side a
                side = {0, *a}
                one, other = tuple(plays[i] for i in sorted(side)), tuple(plays[i] for i in range(n) if i not in side)
                lo, hi = read(_share_cells(_mean_records(one, test, TEST_PHASE[test]),
                                           _mean_records(other, test, TEST_REFERENCE_PHASES[test][0]), test, cost))["ci95"]
                k = len(side)
                widths.append((hi - lo) / 2 * ((2 / n) / (1 / k + 1 / (n - k))) ** 0.5)
            halves.append(statistics.mean(widths))
        out.append(statistics.mean(halves))
    return tuple(out)


# Each model/effort, user setting, track, test and metric has its own family of displayed, measured persistent learners.
# Its direct paired-task bootstrap tails at .025/k and 1-.025/k give the Bonferroni intervals drawn on the 2x2 planes
# and used to judge gains. Tables keep their pointwise 95 percent intervals.
def widening(k: int) -> float:
    """The normal multiplier that widens a 95 percent half-width to a Bonferroni one over ``k`` learners, which sets
    the broad frames of the second and third tracks' figures and the chance macros of chance-numbers.tex."""
    nd = statistics.NormalDist()
    return nd.inv_cdf(1 - 0.025 / k) / nd.inv_cdf(0.975)


def _plane_learners(player: str) -> list[TestRow]:
    """The learners Figure 3's panel for ``player`` draws and judges: its one-user rows outside the appendix that keep
    something between instances and have both tests' measures."""
    return [r for r in TEST_ROWS if r.player == player and r.users == 1 and r.row not in APPENDIX_ROWS
            and r.row not in CHANCE_ROWS and frozen_measure(r, "seen") is not None
            and frozen_measure(r, "held-out") is not None]


def _paired_cells(plays, resets, test: str, metric: str) -> dict[str, list[float]]:
    """Paired task sums for a relative metric, with each instance first averaged over its side's plays."""
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    return _share_cells(_mean_records(tuple(plays), test, TEST_PHASE[test]),
                        _mean_records(tuple(resets), test, TEST_REFERENCE_PHASES[test][0]), test,
                        _charge if metric == "share" else _solve)


@functools.cache
def _open_metric_cells(row: TestRow, test: str, metric: str) -> dict[str, list[float]]:
    if metric == "share":
        d = test_share(row, test)
        return d["_cells"] if d else {}
    plays, resets = _sides(row, test)
    return _paired_cells(plays, resets, test, metric) if plays and resets else {}


@functools.cache
def _open_interval_family(player: str, users: int, test: str, metric: str) -> tuple[TestRow, ...]:
    """Measured persistent settings displayed for this model and user count, independently for each test/metric.
    The one-user scope is _plane_learners' main rows; at two/four users only USER_ROWS are displayed. Raw availability
    checks avoid a dependency on the figure's entries, which themselves need the family size.
    """
    return tuple(r for r in TEST_ROWS if r.player == player and r.users == users
                 and r.row not in APPENDIX_ROWS and r.row not in CHANCE_ROWS and r.run != r.reset
                 and (users == 1 or r.row in USER_ROWS)
                 and all(test_share(r, side) is not None for side in FROZEN_TESTS)
                 and sum(c[1] for c in _open_metric_cells(r, test, metric).values()) > 0)


def _cross(player: str, cost=_charge, read=_share_saved) -> tuple[float, float] | None:
    """Pooled noise half-widths on the seen and held-out tests for ``player``, widened for the learners its panel
    judges (widening): the chance macros, and the bounds of the second and third tracks' broad frames. No figure draws
    them as bands; each learner is judged by its own interval (_adjusted_ci)."""
    c = _chance(player, cost, read)
    if c is None:
        return None
    w = widening(len(_plane_learners(player)))
    return (c[0] * w, c[1] * w)


def _fewest_plays(player: str) -> int:
    """The fewest plays of a frozen test that a one-user row of ``player`` or its reset holds: the number of plays the
    text gives (\\testPlays)."""
    counts = []
    for r in TEST_ROWS:
        if r.player == player and r.users == 1:
            for test in FROZEN_TESTS:
                counts.append(len(reset_plays(r, test)))
                if r.run != r.reset:
                    counts.append(len(test_plays(r.run, test)))
    return min(counts)


def _distinct(runs: list[Run]) -> list[Run]:
    """Each run once, in the order first read: a reference serves many rows."""
    return list({str(r.path): r for r in runs}.values())


def _line(row: str, player: str = "qwen") -> dict:
    """The row's line, with its dash and marker (TEST_LOOK), in its color as its player plays it (ROW_COLOR)."""
    look = TEST_LOOK[row][2]
    color = ROW_COLOR[player][row]
    lw = 1.0 if TEST_LOOK[row][1] == FLOORS else 1.4
    return dict(color=color, ls=look["ls"], lw=lw, marker=look["marker"], ms=2.8, mew=0.7, mec=color,
                mfc="white" if look.get("hollow") else color)


CHANCE_ROWS = ("f", "h0", "hpl0", "m0")   # the rows that keep nothing between instances
# the rows Figure 3 also draws at two and four users, each user bringing tasks of their own, named with their number of
# users, since what more users' more varied tasks add on unseen tasks is a result the plane shows
USER_ROWS = ("DAgger",)
POINT = 6.5   # a point's width on Figure 3 and in its key, in points


def _point(ax, x: float, y: float, family: str, size: float = POINT, transform=None, zorder: float = 4,
           marker: str = "o", color: str | tuple[str, str] | None = None) -> None:
    """A point in its family's color (FAMILY_LOOK) unless given one, in its row's symbol: a learner's filled, with a
    white rim so that two learners at nearly one place read as two, and a baseline's hollow; a pair of colors fills
    its left and right halves."""
    family_color, filled = FAMILY_LOOK[family]
    color = color or family_color
    kw = {} if transform is None else {"transform": transform, "clip_on": False}
    rim = 0.7 if filled else 1.0
    if isinstance(color, tuple):   # no rim, which would split the halves with a white seam
        kw.update(fillstyle="left", markerfacecoloralt=color[1])
        color, rim = color[0], 0.0
    ax.plot([x], [y], ls="", marker=marker, ms=size * (0.85 if marker == "D" else 1.0),
            mfc=color if filled else "white", mec="white" if filled else color, mew=rim, zorder=zorder, **kw)


def _family_key(fig, rect, items: tuple[tuple[str, str, str | tuple[str, str] | None, str], ...], size: float = 6.6,
                gap: float = 10.0) -> None:
    """One line of key centered across ``rect`` (figure coordinates): each (family, markers, color, words) item's
    points side by side, one per marker code in ``markers``, in the family's color unless given one, then what it means
    in the figure, the items ``gap`` points apart; a key wider than ``rect`` is an error."""
    key = fig.add_axes(rect); key.axis("off"); key.set_xlim(0, 1); key.set_ylim(0, 1)
    renderer = fig.canvas.get_renderer()
    width_in = rect[2] * fig.get_size_inches()[0]
    step, air, space = 1.15 * POINT / 72.0 / width_in, 0.75 * POINT / 72.0 / width_in, gap / 72.0 / width_in
    font = font_manager.FontProperties(family=brand.SANS, size=size)
    def text_width(words):
        t = key.text(0, 0, words, fontproperties=font, transform=key.transAxes)
        w = t.get_window_extent(renderer).width / fig.dpi / width_in
        t.remove()
        return w
    spans = [(len(markers) - 1) * step + 2 * air + text_width(words) for _, markers, _, words in items]
    total = sum(spans) + space * (len(items) - 1)
    if total > 1:
        raise ValueError(f"the key runs {total:.2f} of its width")
    x = (1 - total) / 2
    for (family, markers, color, words), span in zip(items, spans):
        for k, marker in enumerate(markers):
            _point(key, x + air / 2 + k * step, 0.5, family, transform=key.transAxes, marker=marker, color=color)
        key.text(x + (len(markers) - 1) * step + 2 * air, 0.5, words, va="center", ha="left", color=brand.INK,
                 transform=key.transAxes, fontproperties=font)
        x += span + space


# Figure 3's glossary: every row it draws by name, a line for the student's own notes, then the teacher's, then the
# rest, so that a reader who has found a family's color in the key above reads its rows in that order
GLOSSARY = ((("Q", "Qwen3.5-9B (student) plays in the harness (prime-agent)"), ("G", "GLM-5.3 Flash (teacher) plays in it")),
            (("T", "teacher writes the notes"), ("E", "expert guidance, for the letter before it"),
             ("D", "DAgger, imitates the teacher")),
            (("P", "program library, in place of the harness"), ("C", "demo cache, in place of the harness"),
             ("(n)", "n concurrent users")))
def _glossary(fig, x0: float, y: float, width: float, lines, size: float = 6.0) -> float:
    """``code gloss`` pairs from (x0, y) down in figure coordinates, a line for each of ``lines``, each code in ink and
    its gloss after it in stone; a line that runs past ``width`` is an error. Returns the step from one line to the
    next, in inches."""
    renderer = fig.canvas.get_renderer()
    font = font_manager.FontProperties(family=brand.SANS, size=size)
    w_in, h_in = fig.get_size_inches()
    em, step = size / 72.0 / w_in, size * 1.65 / 72.0 / h_in
    to_fig = fig.transFigure.inverted().transform
    for pairs in lines:
        x = x0
        for code, gloss in pairs:
            for text, color in ((code, brand.INK), (gloss, STONE)):
                t = fig.text(x, y, text, fontproperties=font, color=color, va="baseline")
                x = to_fig((t.get_window_extent(renderer).x1, 0))[0] + 0.35 * em
            x += 1.3 * em
        if x - 1.65 * em > x0 + width:
            raise ValueError(f"a glossary line runs past its width: {pairs}")
        y -= step
    return step * h_in


def _clustered_legend(ax, handles: dict, columns, **kw):
    """A key of one column per cluster of rows, each cluster a family as Figure 3 groups them (the baselines, the
    student's own notes, the teacher's notes, the trained weights), so that a line's class reads at a glance beside its
    own color; ``handles`` maps each row to its (handle, name), and a row it lacks is left out."""
    columns = [[handles[row] for row in col if row in handles] for col in columns]
    columns = [col for col in columns if col]
    depth = max(len(col) for col in columns)
    blank = (plt.Line2D([], [], alpha=0), "")
    items = [item for col in columns for item in col + [blank] * (depth - len(col))]
    return ax.legend(*zip(*items), ncol=len(columns), frameon=False, **kw)


NOT_A_POINT = "not a point"   # the gid of a marker that only marks where a line starts, which _blend_markers passes over


def _marker_polygon(marker: str, size: float) -> "np.ndarray":
    """A marker's outline about its center, counterclockwise, in the units of its size: each shape convex, so that
    _blend_markers can clip one by another, the plus of a baseline by its hull, drawn too small for the difference to show."""
    import numpy as np
    half = size / 2
    if marker == "D":
        half *= math.sqrt(2)
        return np.array([(half, 0), (0, half), (-half, 0), (0, -half)])
    if marker == "^":
        return np.array([(half, -half), (0, half), (-half, -half)])
    if marker == "s":
        return np.array([(half, -half), (half, half), (-half, half), (-half, -half)])
    if marker == "P":
        arm = half / 3
        return np.array([(half, -arm), (half, arm), (arm, half), (-arm, half), (-half, arm), (-half, -arm), (-arm, -half), (arm, -half)])
    angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
    return half * np.column_stack([np.cos(angles), np.sin(angles)])


def _clip_convex(subject: "np.ndarray", clip: "np.ndarray") -> "np.ndarray":
    """``subject`` clipped by the convex, counterclockwise polygon ``clip`` (Sutherland and Hodgman)."""
    import numpy as np
    out = subject
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if len(out) == 0:
            return out
        inside = (b[0] - a[0]) * (out[:, 1] - a[1]) - (b[1] - a[1]) * (out[:, 0] - a[0]) >= 0
        kept = []
        for k in range(len(out)):
            p, q = out[k], out[(k + 1) % len(out)]
            if inside[k]:
                kept.append(p)
            if inside[k] != inside[(k + 1) % len(out)]:
                d, e = q - p, b - a
                t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / (d[0] * e[1] - d[1] * e[0])
                kept.append(p + t * d)
        out = np.array(kept) if kept else np.empty((0, 2))
    return out


def _offset_convex(poly: "np.ndarray", distance: float) -> "np.ndarray":
    """A convex, counterclockwise polygon with every edge moved outward by ``distance``, its corners mitered, as a marker's
    stroke is drawn."""
    import numpy as np
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.column_stack([edges[:, 1], -edges[:, 0]]) / np.hypot(edges[:, 0], edges[:, 1])[:, None]
    before = np.roll(normals, 1, axis=0)
    return poly + distance * (before + normals) / (1 + (before * normals).sum(axis=1))[:, None]


def _blend_markers(ax) -> None:
    """Where the points of different rows overlap, every one of them shows, so that no point hides another: a marker
    is a fill and an edge, a learner's white rim or a baseline's ring, each cut into convex pieces, and each region
    shared by n markers takes the mean of the n colors they have there, fill or edge, 1/n of each. So an edge that falls
    inside another marker shows through it as a mix, and outside the overlaps each marker keeps its own colors. Run once
    the axes' limits and place are final."""
    import numpy as np
    to_rgb = matplotlib.colors.to_rgb
    scale = ax.figure.dpi / 72.0
    markers = []
    for line in ax.get_lines():
        marker, size = line.get_marker(), line.get_markersize()
        if marker in (None, "None", "", " ") or size <= 0 or line.get_gid() == NOT_A_POINT:
            continue
        face = line.get_markerfacecolor()
        if isinstance(face, str) and face == "none":
            continue
        base = _marker_polygon(marker, size * scale)
        half = line.get_markeredgewidth() * scale / 2
        outer, inner = _offset_convex(base, half), _offset_convex(base, -half)
        edge = to_rgb(line.get_markeredgecolor())
        for xy in line.get_transform().transform(line.get_xydata()):
            if not np.isfinite(xy).all():
                continue
            ring = np.roll(outer, -1, axis=0) + xy, np.roll(inner, -1, axis=0) + xy
            pieces = [(inner + xy, np.array(to_rgb(face)))]
            pieces += [(np.array([outer[k] + xy, ring[0][k], ring[1][k], inner[k] + xy]), np.array(edge))
                       for k in range(len(outer))]
            markers.append(dict(z=line.get_zorder(), outer=outer + xy,
                                pieces=[(poly, color, poly.min(axis=0), poly.max(axis=0)) for poly, color in pieces]))
    def area(poly):
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    # every set of markers that shares a region, piece by piece: the pairs and then the larger sets within them, smallest
    # sets first, so that a region shared by more is painted over the regions of the fewer it lies in
    regions: list[tuple[int, "np.ndarray", "np.ndarray"]] = []
    frontier = [(k, 1, poly, color, low, high) for k, m in enumerate(markers) for poly, color, low, high in m["pieces"]]
    while frontier:
        grown = []
        for last, count, poly, total, low, high in frontier:
            for j in range(last + 1, len(markers)):
                for piece, color, piece_low, piece_high in markers[j]["pieces"]:
                    if (piece_low > high).any() or (piece_high < low).any():
                        continue
                    shared = _clip_convex(poly, piece)
                    if len(shared) >= 3 and area(shared) > 1e-3:
                        grown.append((j, count + 1, shared, total + color, shared.min(axis=0), shared.max(axis=0)))
                        regions.append((count + 1, shared, (total + color) / (count + 1)))
        frontier = grown
    if not regions:
        return
    to_data = ax.transData.inverted().transform
    top = max(m["z"] for m in markers)
    regions.sort(key=lambda region: region[0])
    for order, (_, poly, color) in enumerate(regions):
        # a hair of stroke in the region's own color, or a viewer's antialiasing shows a seam between regions that touch
        ax.add_artist(plt.Polygon(to_data(poly), closed=True, facecolor=color, edgecolor=color, lw=0.06, clip_on=False,
                                  zorder=top + 0.5 * (order + 1) / len(regions)))


def _save(fig, path: Path, crop_bottom: float = 0.0) -> None:
    """Save a figure of points that may overlap, with _blend_markers on each of its axes once its layout is final,
    leaving out the ``crop_bottom`` inches at its foot and keeping its width, so that it scales into the page as every
    other figure does."""
    fig.canvas.draw()
    for ax in fig.axes:
        _blend_markers(ax)
    w, h = fig.get_size_inches()
    fig.savefig(path, bbox_inches=Bbox.from_extents(0, crop_bottom, w, h) if crop_bottom else None)


def _zebra(ax, dash: float = 2.0) -> None:
    """Where two or more of a panel's lines run along the same segment, both its ends shared, draw that segment in
    dashes of each line's color in turn, zebra stripes, so that no line vanishes under another; a point that lines
    only meet at is left as it is. Each such line is broken across the shared segment, keeping its
    points and markers, and the stripes run just under its markers, at the widest of the lines' widths so that the
    dashes, which scale with the width, line up, each line's offset by its place among those sharing the segment."""
    def drawn(line) -> bool:
        return (line.get_transform() == ax.transData and line.get_linestyle() not in ("None", "", " ")
                and len(line.get_xdata()) > 1)
    def key(line, i: int) -> tuple | None:
        xs, ys = line.get_xdata(), line.get_ydata()
        ends = [float(v) for v in (xs[i], ys[i], xs[i + 1], ys[i + 1])]
        return tuple(round(v, 9) for v in ends) if all(math.isfinite(v) for v in ends) else None
    lines = [line for line in ax.get_lines() if drawn(line)]
    sharers: dict[tuple, list] = defaultdict(list)
    for line in lines:
        for i in range(len(line.get_xdata()) - 1):
            if (k := key(line, i)) is not None:
                sharers[k].append(line)
    shared = {k: group for k, group in sharers.items() if len(group) > 1}
    for line in lines:
        cut = {i for i in range(len(line.get_xdata()) - 1) if key(line, i) in shared}
        if cut:
            xs, ys = [], []
            for i, (x, y) in enumerate(zip(line.get_xdata(), line.get_ydata())):
                xs.append(float(x)); ys.append(float(y))
                if i in cut:
                    xs.append(float("nan")); ys.append(float("nan"))
            line.set_data(xs, ys)
    for (x0, y0, x1, y1), group in shared.items():
        width = max(line.get_linewidth() for line in group)
        for k, line in enumerate(group):
            ax.plot([x0, x1], [y0, y1], color=line.get_color(), alpha=line.get_alpha(), lw=width,
                    ls=(k * dash, (dash, (len(group) - 1) * dash)), dash_capstyle="butt",
                    zorder=min(other.get_zorder() for other in group) - 0.01, clip_on=line.get_clip_on())


def _key_headings(legend, headings: list[str], size: float = 4.8) -> None:
    """A small heading over each column of a clustered key (_clustered_legend), level with its column's handles, so
    that a reader who no longer has Figure 3's key in mind sees each cluster's family at a glance.
    Call it once the figure is drawn, since it reads where the key's entries landed."""
    fig = legend.figure
    renderer = fig.canvas.get_renderer()
    texts = legend.get_texts()
    depth = len(texts) // len(headings)
    # an entry's text starts this far right of its handle (matplotlib's HPacker of handle box and text)
    pad = (legend.handlelength + legend.handletextpad) * legend.prop.get_size_in_points() * fig.dpi / 72.0
    for k, heading in enumerate(headings):
        box = texts[k * depth].get_window_extent(renderer)
        x, y = fig.transFigure.inverted().transform((box.x0 - pad, box.y1 + 1.2 * fig.dpi / 72.0))
        fig.text(x, y, heading, ha="left", va="bottom", color=STONE, style="italic",
                 fontproperties=font_manager.FontProperties(family=brand.SANS, size=size, style="italic"))


def _text_shade(color: str, ink: float = 0.35) -> str:
    """``color`` mixed with ``ink`` of the brand's ink: the shade a point's name is written in, since the point's own
    hue reads too light as small text on white."""
    a, b = matplotlib.colors.to_rgb(color), matplotlib.colors.to_rgb(brand.INK)
    return matplotlib.colors.to_hex([(1 - ink) * x + ink * y for x, y in zip(a, b)])


def _better_wash(ax, strength: float = 0.10, at: tuple[float, float] | None = None) -> None:
    """A faint wash under a plane, white through the origin, or through ``at`` in fractions of the axes on a plane with
    no origin of its own, and deepening to green towards the top right, where a learner does well on both tests, and to
    red towards the bottom left: each axis is read as the share of its own reach on its side of that point, and the wash
    follows their mean, at ``strength`` opacity in the corners. The axes' limits stay as they were."""
    import numpy as np
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    fx, fy = at if at else (-x0 / (x1 - x0), -y0 / (y1 - y0))
    gx, gy = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 300))
    v = (np.where(gx > fx, (gx - fx) / (1 - fx), (gx - fx) / fx) + np.where(gy > fy, (gy - fy) / (1 - fy), (gy - fy) / fy)) / 2
    img = np.zeros(gx.shape + (4,))
    for sign, hue in ((1, brand.GREEN), (-1, brand.CORAL_DEEP)):
        side = sign * v > 0
        img[side, :3] = matplotlib.colors.to_rgb(hue)
        img[side, 3] = strength * np.clip(sign * v[side], 0, 1)
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="lower", aspect="auto", zorder=-1, interpolation="bilinear")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


# Where each name sits beside its row's point on Figure 3, in points from the point, (dx, dy, alignment, hairline): by
# hand, since with 14 rows in two panels a placing that minds only names and points lays them across the lines.
PLANE_BUDGET_NAMES = {("qwen", "h", 1): (-1, -8, "center", False), ("qwen", "hgpl", 1): (6, -1, "left", False),
                      ("qwen", "hg", 1): (6, 0, "left", False), ("qwen", "hpl", 1): (-6, 0, "right", False),
                      ("qwen", "hplg", 1): (6, -1, "left", False), ("qwen", "DAgger", 1): (6, -1, "left", False),
                      ("qwen", "DAgger", 2): (6, 1, "left", False), ("qwen", "DAgger", 4): (-8, -1, "right", False),
                      ("qwen", "m", 1): (5, 0, "left", False), ("qwen", "dc", 1): (0, -8, "center", False),
                      ("glm", "dc", 1): (-5, 7, "right", False), ("glm", "h", 1): (-7, 0, "right", False),
                      ("glm", "hpl", 1): (-9, 7, "right", True), ("glm", "m", 1): (-2, -11, "right", True)}
PLANE_SHARE_NAMES = {("qwen", "h", 1): (-6, 1, "right", False), ("qwen", "hpl", 1): (9, -11, "left", True),
                     ("qwen", "hgpl", 1): (1, -17, "center", True), ("qwen", "hg", 1): (4, 7, "left", False),
                     ("glm", "h", 1): (6, 1, "left", False), ("qwen", "hplg", 1): (1, -9, "center", False),
                     ("glm", "dc", 1): (6, -1, "left", False), ("qwen", "dc", 1): (0, 8, "center", False),
                     ("qwen", "m", 1): (-6, 1, "right", False), ("glm", "m", 1): (6, 0, "left", False),
                     ("glm", "hpl", 1): (5, 6, "left", False), ("qwen", "DAgger", 1): (6, 0, "left", False),
                     ("qwen", "DAgger", 2): (6, 0, "left", False), ("qwen", "DAgger", 4): (-6, 0, "right", False)}
PLANE_SOLVED_NAMES = {("qwen", "h", 1): (-6, 0, "right", False), ("qwen", "hgpl", 1): (2, -8, "center", False),
                      ("qwen", "hg", 1): (6, -4, "left", False), ("qwen", "hpl", 1): (-6, 5, "right", False),
                      ("qwen", "hplg", 1): (6, -1, "left", False), ("qwen", "DAgger", 1): (6, 0, "left", False),
                      ("qwen", "DAgger", 2): (-6, 0, "right", False), ("qwen", "DAgger", 4): (6, 2, "left", False),
                      ("qwen", "m", 1): (6, 0, "left", False), ("qwen", "dc", 1): (0, -8, "center", False),
                      ("glm", "dc", 1): (6, 0, "left", False), ("glm", "h", 1): (-7, -3, "right", False),
                      ("glm", "hpl", 1): (6, -2, "left", False), ("glm", "m", 1): (-14, 7, "right", True)}
PLANE_RATIO_NAMES = {("qwen", "h", 1): (-7, 0, "right", False), ("qwen", "hgpl", 1): (2, -12, "center", True),
                     ("qwen", "hg", 1): (6, 2, "left", False), ("qwen", "hpl", 1): (-8, -9, "right", True),
                     ("qwen", "hplg", 1): (6, -1, "left", False), ("qwen", "DAgger", 1): (6, 0, "left", False),
                     ("qwen", "DAgger", 2): (6, 1, "left", False), ("qwen", "DAgger", 4): (-6, 1, "right", False),
                     ("qwen", "m", 1): (6, 0, "left", False), ("qwen", "dc", 1): (6, -1, "left", False),
                     ("glm", "dc", 1): (9, -8, "left", True), ("glm", "h", 1): (-8, 8, "right", True),
                     ("glm", "hpl", 1): (0, 12, "center", True), ("glm", "m", 1): (6, 18, "left", True)}
# each reading's (left panel, right panel) placements
PLANE_NAMES = {"budget": (PLANE_BUDGET_NAMES, PLANE_SHARE_NAMES), "solved": (PLANE_SOLVED_NAMES, PLANE_RATIO_NAMES)}


# The percents the broad frames' left panels mark, each panel running the whole of 0 to 100 percent so that every
# figure drawn on its axes shares its frame
LEVEL_TICKS = (0, 20, 40, 60, 80, 100)


def _plane_name(ax, x: float, y: float, name: str, color: str, place: tuple) -> None:
    """A row's name beside its point (PLANE_NAMES), white-edged so that it reads over the lines,
    with a hairline to the point where it sits further off."""
    dx, dy, align, hairline = place
    text = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=align, va="center", zorder=8,
                       color=_text_shade(color), fontproperties=font_manager.FontProperties(family=brand.SANS, size=5.8),
                       arrowprops=dict(arrowstyle="-", color=color, lw=0.35, shrinkA=0, shrinkB=2.5) if hairline else None)
    text.set_path_effects([patheffects.withStroke(linewidth=1.3, foreground="white")])


def _reset_to_learner(ax, start: tuple[float, float], end: tuple[float, float], color: str, zorder: float) -> None:
    """A row's line on a plane's left panel from its reset, a small hollow dot, to the learner, faint at the reset and
    full at the learner, so that the many lines that leave nearly one place read apart and the rows' ends stand out. Its
    steps are taken in the units of the panel's scale, set before it is drawn, so that it is straight on any scale. The
    dot sits over every point, so that it still shows for a row that has barely moved from it."""
    import numpy as np
    steps = 28
    ends = ax.transScale.transform([start, end])
    xs, ys = ax.transScale.inverted().transform(np.linspace(ends[0], ends[1], steps + 1)).T
    segments = np.stack([np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1)
    t = np.linspace(0, 1, steps)
    rgb = matplotlib.colors.to_rgb(color)
    ax.add_collection(LineCollection(segments, colors=[(*rgb, 0.28 + 0.67 * u) for u in t], linewidths=0.5 + t,
                                     capstyle="round", zorder=zorder))
    ax.plot([start[0]], [start[1]], ls="", marker="o", ms=2.2, mfc="white", mec=color, mew=0.5, zorder=7, gid=NOT_A_POINT)


# A plane's two readings of the frozen tests, a row of panels each: the budget used, the benchmark's own measure, and
# the solve rate. Each left panel runs a line from the reset to the learner, and each right panel reads the learner
# against its reset: the share of the reset's spend it saves, 0 for no change, or how many times as many instances it
# solves, 1 for no change. Each panel takes the plainest scale on which its points stand apart, figure by figure.
# Figure 3's left panels crowd at two corners, GLM's rows within six points of every instance solved and Qwen's 15 to 32
# points above none, so they go on complementary log-log axes (_cloglog), which stretch low shares as a log scale does
# and high ones more gently. A logit, even with 50 percent in the middle, opens only GLM's corner, since to it Qwen's
# rows are mid-range points; within the Aranda-Ordaz family, from a log scale through the cloglog to the logit, the
# scale that keeps both corners' closest rows furthest apart in both panels at once (lambda 0.34) sits beside the
# cloglog (lambda 0), so the named scale stands in for a fitted constant. GPT-6 Luna's left panels (Figure 5) stand
# apart as they are and stay linear, each distance at face value. The right panels spread the points near no change and
# close up those far from it: the share saved on an asinh axis, linear within SHARE_LINEAR of zero, where a share can be
# negative, and logarithmic beyond, and the solve rate increase on a log2 axis, symmetric about x1. Each axis hugs its
# own points; GPT-6 Luna's ratios lie within a sixth of x1, where log2 is as good as linear and their points crowd at
# x1, so their axis also takes asinh of the log2, linear within LUNA_RATIO_LINEAR of x1, with the finer RATIO_FINE
# ticks. Both relative panels show each learner's own interval, since each learner has its own reset, and the no-change
# lines at 1 for solve ratios and 0 for share saved. The share-saved axes carry the paper's S_seen and S_held, so that a
# reader of the Results ties each axis to Table 1's columns.
SHARE_LINEAR = 0.1
SHARE_TICKS = (-0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.8)
RATIO_TICKS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RATIO_FINE = (0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25)
LUNA_RATIO_LINEAR = 1.01
CLOGLOG_TICKS = (1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 98, 99)
PLANE_READINGS = {
    "solved": ("Solve rate, from the reset to the learner", "Solve rate increase, against the reset", "solve rate",
               "solve rate increase"),
    "budget": ("Budget used, from the reset to the learner", "Share saved, against the reset", "budget used", "share saved"),
}
# The symbol each test's axis takes after its words, the letters in the body font and mathtext lowering the subscript
AXIS_SYMBOLS = {"share saved": (r", S$_{\mathrm{seen}}$", r", S$_{\mathrm{held}}$")}


@functools.cache
def _figure3() -> tuple[tuple[TestRow, ...], tuple[dict, ...], dict[str, tuple[float, float]]]:
    """Figure 3's rows, their points (_plane_reading) and their pooled noise widths (_cross), which bound the second and
    third tracks' broad frames: every learner at one user, and each of USER_ROWS also at two and four users, outside
    APPENDIX_ROWS, once both of its tests are played."""
    rows = tuple(r for r in TEST_ROWS if r.row not in APPENDIX_ROWS and r.row not in CHANCE_ROWS
                 and (r.users == 1 or r.row in USER_ROWS) and all(frozen_measure(r, test) for test in FROZEN_TESTS))
    points = []
    for r in rows:
        points.append({"name": row_name(r.row, r.player) + f"({r.users})" * (r.users > 1),
                       "family": _row_family(r.row, r.player), "row": r.row, "entry": _open_entry(r),
                       "places": {reading: tuple(names[r.player, r.row, r.users] for names in PLANE_NAMES[reading])
                                  for reading in PLANE_READINGS}})
    return rows, tuple(points), {PLAYER_LETTER[p]: arms for p in PLAYERS if (arms := _cross(p))}


def _plane_figure(points, frame, arms, key, glossary, out: Path, open_label: str | None = None, fit_axes: bool = True,
                  tight_axes: bool = False, cloglog_left: bool = False, ratio_linear: float | None = None) -> None:
    """Figure 3's layout, both readings of PLANE_READINGS on ``points`` (_plane_reading), axes fitted to
    their own data unless ``fit_axes=False`` uses the broad ``frame`` and ``arms``, the key above and the glossary below; with
    ``open_label``, each panel marks the space its learners leave open (_plane_reading). ``tight_axes`` gives Luna's
    panels exact endpoint bounds plus five percent padding, choosing ticks independently of those bounds;
    ``cloglog_left`` and ``ratio_linear`` choose the figure's scales (_plane_reading)."""
    fig = plt.figure(figsize=(WIDTH, 6.4))
    # Leave the right panels' vertical labels and fractional ticks a clear gutter beside the left plots.
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], wspace=0.28, hspace=0.3, left=0.085, right=0.965 if fit_axes else 0.985,
                            bottom=0.177, top=0.914)
    for i, reading in enumerate(PLANE_READINGS):
        _plane_reading(fig.add_subplot(grid[i, 0]), fig.add_subplot(grid[i, 1]), points, reading, frame, arms,
                       open_label, fit_axes=fit_axes, tight_axes=tight_axes, cloglog_left=cloglog_left,
                       ratio_linear=ratio_linear)
    _family_key(fig, [0.02, 0.953, 0.965, 0.044], key, size=5.7, gap=5.0)
    step = _glossary(fig, 0.085, 0.077, 0.9, glossary)
    # the layout has room for Figure 3's glossary, the longest; a figure with fewer lines leaves out the room for the
    # lines it lacks, so that its caption follows its last line as Figure 3's does rather than after a blank band
    _save(fig, out, crop_bottom=(len(GLOSSARY) - len(glossary)) * step)


def fig_tests() -> Path:
    """figures/tests.pdf, Figure 3: every learner at one user, and each of USER_ROWS also at two and four users, on the
    two frozen tests, the seen test across (memorization) and the held-out test up (generalization), whichever model
    plays, since an open-ended benchmark has no default player, in both readings of PLANE_READINGS, the solve rate above
    and the budget used below (_plane_reading), the left panels on complementary log-log axes, where its rows crowd at
    two corners. The key above names the families, and the glossary below takes the rows' names apart. Every row needs
    both tests played."""
    rows, points, _ = _figure3()
    out = FIG / "tests.pdf"
    _plane_figure(points, points, [], PLANE_KEY, GLOSSARY, out, cloglog_left=True)
    _report("tests", _distinct([run for r in rows for test in FROZEN_TESTS
                                for run in frozen_measure(r, test).get("runs", [])]))
    return out


# The second and third tracks' rows (Appendix K): Kimi K3.1 Pro at medium effort, at seed 0 and one user. The two floors
# played through OpenRouter, each stream once and then each frozen test four times; every other row played with every
# call pinned to Moonshot AI's own endpoint, each stream once and each test twice. Its weights are not open, so no row
# trains them or has a teacher write its notes, and on ARC-AGI-3 neither the program library nor the expert guidance
# plays, both being procedures for grids (Appendix K). Each row is its folder from the repository's root, its stream's
# files there and its plays of a test in <run>/frozen/<test>-p<k>/play/. Each track gets Figure 3's four panels with
# broad axes, so that its few rows sit where Figure 3's would and the rest of the plane shows what is left to run there.
TRACK_PLAYER = ("K", "Kimi K3.1 Pro")
# in Appendix K's table's order
TRACK_ROWS = {"arc2": {row: f"runs/arc2/{name}-kimi-k3.1-pro-ws0-u1" for row, name in (
                  ("f", "KC0"), ("dc", "KC"), ("m", "KP"), ("m0", "KP0"), ("h", "K-prime-agent"),
                  ("h0", "K0-prime-agent"), ("ha", "KA-prime-agent"), ("hpl", "KE-prime-agent"),
                  ("hpl0", "KE0-prime-agent"))},
              "arc3": {row: f"runs/arc3/{name}-kimi-k3.1-pro-ws0-u1" for row, name in (
                  ("f", "KC0"), ("dc", "KC"), ("h", "K-prime-agent"), ("h0", "K0-prime-agent"),
                  ("ha", "KA-prime-agent"))}}
# each row's reset, as TEST_ROWS reads the open models' rows against theirs; a floor or a wiped row is its own
TRACK_RESET = {"f": "f", "dc": "f", "m": "m0", "h": "h0", "ha": "h0", "hpl": "hpl0", "h0": "h0", "m0": "m0",
               "hpl0": "hpl0"}
# where each name sits beside its point, PLANE_NAMES's form by track and row: {reading: (left panel, right panel)}, a
# name to the right of its point unless placed
NAME_RIGHT, NAME_LEFT = (6, 0, "left", False), (-6, 0, "right", False)
# ARC-AGI-2's solve-rate ratios crowd about 1, so K and KA name themselves to the left, clear of KC's and KE's points
TRACK_NAMES: dict[str, dict[str, dict[str, tuple[tuple, tuple]]]] = {
    "arc2": {row: {"solved": (NAME_RIGHT, NAME_LEFT)} for row in ("h", "ha")}}
# what each part of a track row's name means, in GLOSSARY's words
TRACK_GLOSSARY = {"": ("K", "Kimi K3.1 Pro plays in the harness (prime-agent)"),
                  "C": ("C", "demo cache, in place of the harness"), "P": ("P", "program library, in place of the harness"),
                  "E": ("E", "expert guidance"), "A": ("A", "asked to store what it chooses")}


@functools.cache
def _track_plays(track: str, row: str, test: str) -> tuple[Run, ...]:
    """Every whole play of a frozen test by a row of TRACK_ROWS, in order: a play holds every instance of the track's test
    at seed 0 and one user (configs.test_schedule) and nothing else, one still short is left out, and one that holds
    anything else stops the build."""
    from continual_arc.configs import get_config, test_schedule
    from continual_arc.schedule import TEST_PHASE
    want = {(s.task_id, s.instance_seed) for s in test_schedule(get_config(track), TRACK_SEED, test).slots}
    frozen = REPO / TRACK_ROWS[track][row] / "frozen"
    plays = []
    for k in itertools.count(1):
        path = frozen / f"{test}-p{k}" / "play"
        if not (path / "records.jsonl").exists():
            return tuple(plays)
        play = _load_run({"row": f"kimi-{track}-{row}", "model": TRACK_PLAYER[1], "track": track, "seed": TRACK_SEED,
                          "mode": f"{test} test", "label": f"{TRACK_ROWS[track][row]}, {test} play {k}"}, path)
        got = {(r.task_id, r.instance_seed) for r in (*play.records, *play.dropped) if r.phase == TEST_PHASE[test]}
        if got - want:
            raise SystemExit(f"{path}: instances that are not in {track}'s {TEST_NAMES[test]} at seed {TRACK_SEED}")
        if got == want:
            plays.append(play)


@functools.cache
def _track_stream(track: str, row: str) -> Run | None:
    """A track row's stream (TRACK_ROWS), its stream's records alone, after a check that it is the track's workstreams
    stream at seed 0 and one user, slot for slot; None until it has played the whole stream."""
    from continual_arc.configs import build_schedule, get_config
    path = REPO / TRACK_ROWS[track][row]
    run = _load_run({"row": f"kimi-{track}-{row}", "model": TRACK_PLAYER[1], "track": track, "seed": TRACK_SEED,
                     "mode": "workstreams", "label": f"{TRACK_ROWS[track][row]}, stream"}, path)
    full = build_schedule(get_config(track), TRACK_SEED, order="workstreams")
    stream = [s for s in full.slots if s.phase == "stream"]
    if run is None:
        return None
    served = [(s.task_id, s.instance_seed) for s in run.schedule.slots if s.phase == "stream"]
    if served != [(s.task_id, s.instance_seed) for s in stream]:
        raise SystemExit(f"{path}: its schedule is not {track}'s workstreams stream at seed {TRACK_SEED}")
    records = [r for r in run.records if r.phase == "stream"]
    dropped = tuple(r for r in run.dropped if r.phase == "stream")
    if len(records) + len(dropped) < len(stream):
        return None
    return replace(run, records=records, dropped=dropped, schedule=replace(full, slots=stream), n_slots=len(stream),
                   epilogue_start=None)


@functools.cache
def _track_sides(track: str, row: str, test: str) -> tuple[tuple[Run, ...], tuple[Run, ...]]:
    own = _track_plays(track, row, test)
    reset = TRACK_RESET[row]
    return (own[0::2], own[1::2]) if reset == row else (own, _track_plays(track, reset, test))


@functools.cache
def _track_metric_cells(track: str, row: str, test: str, metric: str) -> dict[str, list[float]]:
    plays, resets = _track_sides(track, row, test)
    return _paired_cells(plays, resets, test, metric) if plays and resets else {}


@functools.cache
def _track_interval_family(track: str, test: str, metric: str) -> tuple[str, ...]:
    """The track's measured persistent displayed rows, independently for each test and metric; no entry recursion."""
    return tuple(row for row in TRACK_ROWS[track] if row not in CHANCE_ROWS and TRACK_RESET[row] != row
                 and all(all(_track_sides(track, row, side)) for side in FROZEN_TESTS)
                 and sum(c[1] for c in _track_metric_cells(track, row, test, metric).values()) > 0)


@functools.cache
def _track_entry(track: str, row: str) -> dict | None:
    """A track row on both tests in _open_entry's form, against its reset (TRACK_RESET), a row that keeps nothing its plays
    taken alternately, half against the other half, as _sides takes them; None until both sides have played both tests."""
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    tests = {}
    for test in FROZEN_TESTS:
        plays, resets = _track_sides(track, row, test)
        if not plays or not resets:
            return None
        mine = _mean_records(plays, test, TEST_PHASE[test])
        theirs = _mean_records(resets, test, TEST_REFERENCE_PHASES[test][0])
        before = _absolute(resets, test, TEST_REFERENCE_PHASES[test][0])
        share = _share_saved(_share_cells(mine, theirs, test))
        ratio = _solved_ratio(_share_cells(mine, theirs, test, _solve))
        tests["held" if test == "held-out" else "seen"] = {
            "share": _triple(share, "share"), "solved_ratio": _triple(ratio, "ratio"),
            **_absolute(plays, test, TEST_PHASE[test]), "reset_budget": before["budget"], "reset_solved": before["solved"],
            **_adjusted_fields(share, ratio, {metric: _track_interval_family(track, test, metric)
                                            for metric in ("share", "solved_ratio")}, row)}
    return tests


def _track_learners(track: str) -> list[str]:
    """The rows of a track that keep something between instances, those its panels draw and judge, once both of their
    tests are played against their reset's."""
    return [row for row in TRACK_ROWS[track] if row not in CHANCE_ROWS and _track_entry(track, row) is not None]


def fig_tests_track(track: str) -> Path:
    """figures/tests-<track>.pdf, Figures 11 and 12: Figure 3 for one of the second and third tracks (TRACK_ROWS), its
    learners each against its reset on broad axes, framed by Figure 3's points, with learner-specific intervals. The
    key and the glossary name only what it draws. Every panel marks with a dashed box the space no learner here reaches
    yet, left open for new methods: on a left panel all of it above the points drawn, and on a right panel where a
    learner that beats its reset on both tests would stand, since a model can reach the left panels' best corner
    without learning a thing."""
    _, figure3, figure3_chance = _figure3()
    letter = TRACK_PLAYER[0]
    learners = _track_learners(track)
    points = tuple({"name": letter + ROW_LETTERS[row], "family": TEST_LOOK[row][1], "row": row,
                    "entry": _track_entry(track, row),
                    "places": {reading: TRACK_NAMES.get(track, {}).get(row, {}).get(reading, (NAME_RIGHT, NAME_RIGHT))
                               for reading in PLANE_READINGS}} for row in learners)
    keep_nothing = [row for row in TRACK_ROWS[track] if row in CHANCE_ROWS]
    arms = _chance_of([functools.partial(_track_plays, track, row) for row in keep_nothing])
    chance = {letter: tuple(v * widening(len(learners)) for v in arms)} if arms and learners else {}
    families = {p["family"] for p in points}
    key = tuple(item for item, present in (
        ((FLOORS, "o", None, "baseline"), FLOORS in families),
        ((NOTES, "o", None, "keeps notes in the harness"), NOTES in families),
        ((NOTES, "D", INK, "expert guidance"), "hpl" in learners),
        ((NOTES, "s", INK, "asked to store"), "ha" in learners)) if present)
    # the player's own gloss is a line's width alone, so it takes the first line and the methods' letters follow in
    # pairs, at most Figure 3's three lines
    parts = [part for part in ("C", "P", "E", "A") if any(part in ROW_LETTERS[row] for row in learners)]
    glossary = ((TRACK_GLOSSARY[""],),) + tuple(tuple(TRACK_GLOSSARY[part] for part in parts[i:i + 2])
                                                for i in range(0, len(parts), 2))
    out = FIG / f"tests-{track}.pdf"
    _plane_figure(points, figure3 + points, list(figure3_chance.values()) + list(chance.values()), key,
                  glossary, out, open_label="Space for new methods", fit_axes=False)
    _report(f"tests-{track}", [play for row in TRACK_ROWS[track] for test in FROZEN_TESTS
                               for play in _track_plays(track, row, test)])
    return out


def _cloglog(budget: bool):
    """The complementary log-log, ln(-ln(1 - p)), as forward and inverse functions of a left panel's values, percent
    solved or, in the budget reading, minus percent used, of which p is the share solved or the share of the budget left:
    it stretches low shares as a log scale does and high ones more gently, and least at 1 - 1/e, about 63 percent."""
    import numpy as np
    share = (lambda v: 1.0 + np.asarray(v, float) / 100.0) if budget else (lambda v: np.asarray(v, float) / 100.0)
    level = (lambda p: 100.0 * (p - 1.0)) if budget else (lambda p: 100.0 * p)
    def forward(v):
        # Placing a y tick label, matplotlib passes its place across the panel, 0, through the x scale too and discards
        # what comes back (BlendedGenericTransform.transform_non_affine), so this meets shares of 0 and 1, where numpy
        # returns infinities; matplotlib's own log and logit transforms ignore the same errors there.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(-np.log1p(-share(v)))
    return forward, lambda u: level(-np.expm1(-np.exp(np.asarray(u, float))))


def _cloglog_extent(values, forward, inverse, to_abs):
    """Bounds for a left panel on complementary log-log axes (_cloglog), in its own values, padded past every reset and
    learner by a tenth of their span in the scale's units, room for the names of the rows at its edges, and ticks from
    CLOGLOG_TICKS, thinned outward from 63 percent, where the scale stretches least, so that no two print closer than a
    ninth of the panel."""
    units = forward(values)
    low, high = float(units.min()), float(units.max())
    pad = 0.1 * (high - low)
    low, high = low - pad, high + pad
    ticks = []
    for t in sorted((to_abs(c) for c in CLOGLOG_TICKS), key=lambda t: abs(float(forward(t)))):
        u = float(forward(t))
        if low <= u <= high and all(abs(u - float(forward(k))) >= (high - low) / 9 for k in ticks):
            ticks.append(t)
    return (float(inverse(low)), float(inverse(high))), sorted(ticks)


def _plane_spread_extent(scale, limits, values, origin: float, *, ratio: bool, ratio_linear: float | None = None):
    """Puts a relative panel's axis on a scale that spreads the points near no change and closes up those far from it,
    the solve rate increase (``ratio``) on log2, also on asinh within ``ratio_linear`` of x1 where one is given, and the
    share saved on asinh with SHARE_LINEAR, its limits padded a little past every point and interval and set before the
    scale, so that the scale never meets matplotlib's default limits, and returns its ticks, from RATIO_TICKS, or
    RATIO_FINE where the ratios span less than a factor of four, or SHARE_TICKS, thinned outward from no change so that no
    two print closer than an eighth of the panel."""
    import numpy as np
    values = [float(v) for v in values if v is not None and math.isfinite(v)] + [origin]
    if ratio:
        values = [v for v in values if v > 0]
        if ratio_linear is None:
            forward, inverse, kind = math.log2, (lambda u: 2.0 ** u), ("log", dict(base=2))
        else:
            core = math.log2(ratio_linear)
            forward = lambda r: core * math.asinh(math.log2(r) / core)
            inverse = lambda u: 2.0 ** (core * math.sinh(u / core))
            def spread(r):
                with np.errstate(divide="ignore", invalid="ignore"):   # a tick label's place, 0, as in _cloglog
                    return core * np.arcsinh(np.log2(r) / core)
            kind = ("function", dict(functions=(spread, lambda u: 2.0 ** (core * np.sinh(np.asarray(u, float) / core)))))
        coarse = [t for t in RATIO_TICKS if forward(min(values)) <= forward(t) <= forward(max(values))]
        candidates = RATIO_TICKS if len(coarse) >= 3 else RATIO_FINE
    else:
        forward = lambda v: SHARE_LINEAR * math.asinh(v / SHARE_LINEAR)
        inverse = lambda u: SHARE_LINEAR * math.sinh(u / SHARE_LINEAR)
        kind = ("asinh", dict(linear_width=SHARE_LINEAR))
        candidates = SHARE_TICKS
    low, high = forward(min(values)), forward(max(values))
    pad = 0.06 * (high - low)
    low, high = low - pad, high + pad
    limits(inverse(low), inverse(high))
    scale(kind[0], **kind[1])
    ticks = [origin]
    for side in (sorted(t for t in candidates if t > origin), sorted((t for t in candidates if t < origin), reverse=True)):
        last = forward(origin)
        for t in side:
            if low <= forward(t) <= high and abs(forward(t) - last) >= 0.12 * (high - low):
                ticks.append(t)
                last = forward(t)
    return sorted(ticks)


def _plane_extent(values, *, percentage: bool = False, ratio: bool = False, tight: bool = False):
    """Padded linear bounds and readable ticks from the values actually plotted, including CI endpoints."""
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        values = [0.0, 100.0 if percentage else 1.0]
    low, high = min(values), max(values)
    if tight:
        # Luna's rectangle hugs all plotted endpoints. Nice ticks do not move the limits out to tick boundaries.
        pad = 0.05 * (high - low if high > low else (1.0 if percentage else 0.01))
        low, high = low - pad, high + pad
        ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
        ticks = [float(v) for v in ticks if low <= v <= high and (not percentage or 0 <= v <= 100)]
        return (low, high), ticks
    span = max(high - low, 1.0 if percentage else 0.01)
    pad = max(0.08 * span, 2.0 if percentage else 0.01)
    low, high = low - pad, high + pad
    unit = 10 ** math.floor(math.log10((high - low) / 5))
    step = next(k * unit for k in (1, 2, 2.5, 5, 10) if k * unit >= (high - low) / 5)
    low = math.floor(low / step + 1e-10) * step
    high = math.ceil(high / step - 1e-10) * step
    if percentage:
        low, high = max(0.0, low), min(100.0, high)
        # Keep room for symbols and names at a percentage boundary.
        low = -4.0 if low == 0 else low
        high = 104.0 if high == 100 else high
    elif ratio:
        low = max(0.0, low)
    ticks = [round(k * step, 10) for k in range(math.ceil(low / step - 1e-10),
                                               math.floor(high / step + 1e-10) + 1)]
    if percentage:
        ticks = [v for v in ticks if 0 <= v <= 100]
    return (low, high), ticks


def _plane_reading(absolute, relative, points, reading: str, frame, arms, open_label: str | None = None,
                   fit_axes: bool = True, tight_axes: bool = False, cloglog_left: bool = False,
                   ratio_linear: float | None = None) -> None:
    """One row of a plane's panels in ``reading`` (PLANE_READINGS). On ``absolute``, what ``relative`` is read from: each
    row a line from its reset to the learner (_reset_to_learner), in percent with less of the budget used
    and more solved to the right and up, so that a line shows both how much a learner gained and how able it was before
    and after. On ``relative``, each row against its reset as a point, with its direct Bonferroni intervals on the two tests as a
    faint cross in its family's color. Up and right are better on both, under the same faint wash (_better_wash). Each of
    ``points`` is a row's {"name", "family", "row", "entry" (_open_entry's form), "places" (its name's place on each
    panel by reading)}, drawn in its family's color (FAMILY_LOOK) and its row's symbol, with its name beside it.
    Ordinary axes fit their own points and confidence intervals, the absolute panels on complementary log-log axes
    with ``cloglog_left`` (_cloglog) and linear ones otherwise, and the solve rate increase on asinh of its log2 within
    ``ratio_linear`` of x1 where one is given (_plane_spread_extent). With ``fit_axes=False``, the absolute panels cover
    0 to 100 and the relative panels use ``frame`` and ``arms`` to keep a broad frame. With ``open_label``, each panel
    boxes and names the space its points leave open: on ``absolute`` the panel's whole width above every point drawn,
    where a learner better on the held-out test than any here would stand, and on ``relative`` where one that beats its
    reset on both tests would, up and right of no change; each box stops just above anything drawn under it."""
    budget = reading == "budget"
    title, rel_title, words, rel_words = PLANE_READINGS[reading]
    to_abs = (lambda used: -used) if budget else float
    reset_key, rel_key = ("reset_budget", "share") if budget else ("reset_solved", "solved_ratio")
    interval_key = rel_key + "_adjusted"
    # The left panel's limits, scale and ticks come before anything is drawn on it, so that each reset-to-learner line is
    # straight on its scale, and its limits before its scale, so that the scale never meets matplotlib's default limits.
    for test, axis, limits, set_scale in (("seen", absolute.xaxis, absolute.set_xlim, absolute.set_xscale),
                                          ("held", absolute.yaxis, absolute.set_ylim, absolute.set_yscale)):
        values = [to_abs(p["entry"][test][key]) for p in points for key in (reset_key, reading)]
        if cloglog_left:
            forward, inverse = _cloglog(budget)
            bounds, ticks = _cloglog_extent(values, forward, inverse, to_abs)
        elif fit_axes:
            bounds, ticks = _plane_extent([abs(v) for v in values], percentage=True, tight=tight_axes)
            bounds, ticks = sorted(to_abs(v) for v in bounds), [to_abs(v) for v in ticks]
        else:   # the whole of 0 to 100 percent, with room past each end for the names of the rows near it
            bounds, ticks = sorted((to_abs(-4), to_abs(104))), [to_abs(t) for t in LEVEL_TICKS]
        limits(*bounds)
        if cloglog_left:
            set_scale("function", functions=(forward, inverse))
            axis.set_minor_locator(NullLocator())
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FixedFormatter([f"{abs(v):g}" for v in ticks]))
    # no change: no share saved, or as many solved as the reset
    origin = 0.0 if budget else 1.0
    relative.axhline(origin, color=STONE2, lw=0.6, zorder=1)
    relative.axvline(origin, color=STONE2, lw=0.6, zorder=1)
    # Horizontal extent and upper y of each point/interval, so invitation boxes clear the complete corrected bars.
    drawn_abs: list[tuple[float, float, float]] = []
    drawn_rel: list[tuple[float, float, float]] = []
    for k, point in enumerate(points):
        family, name = point["family"], point["name"]
        abs_place, rel_place = point["places"][reading]
        color = point.get("color") or FAMILY_LOOK[family][0]
        marker = "o" if family == FLOORS else TEST_LOOK[point["row"]][2]["marker"]
        seen, held = point["entry"]["seen"], point["entry"]["held"]
        start = (to_abs(seen[reset_key]), to_abs(held[reset_key]))
        end = (to_abs(seen[reading]), to_abs(held[reading]))
        _reset_to_learner(absolute, start, end, color, 3 + k / 100)
        drawn_abs += [(a, a, b) for a, b in (start, end)]
        _point(absolute, *end, family, zorder=4 + k / 100, marker=marker, color=point.get("color"))
        _plane_name(absolute, *end, name, color, abs_place)
        if seen[interval_key] is None or held[interval_key] is None:
            continue   # an all-zero reset solve count has no defined solve ratio
        (x, *sx_ci), (y, *sy_ci) = seen[interval_key], held[interval_key]
        drawn_rel.append((x, x, y))
        # a learner outside its interval family (_adjusted_fields) carries no interval to draw
        if None not in sx_ci:
            relative.plot(sx_ci, (y, y), color=color, lw=0.8, alpha=0.6, zorder=2, solid_capstyle="butt")
            drawn_rel.append((min(sx_ci), max(sx_ci), y))
        if None not in sy_ci:
            relative.plot((x, x), sy_ci, color=color, lw=0.8, alpha=0.6, zorder=2, solid_capstyle="butt")
            drawn_rel.append((x, x, max(sy_ci)))
        _point(relative, x, y, family, zorder=4 + k / 100, marker=marker, color=point.get("color"))
        _plane_name(relative, x, y, name, color, rel_place)
    if fit_axes:
        for test, axis, limits, scale in (("seen", relative.xaxis, relative.set_xlim, relative.set_xscale),
                                          ("held", relative.yaxis, relative.set_ylim, relative.set_yscale)):
            values = ([] if tight_axes else [origin]) + [v for p in points for v in (p["entry"][test][interval_key] or [])]
            ticks = _plane_spread_extent(scale, limits, values, origin, ratio=not budget, ratio_linear=ratio_linear)
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FixedFormatter([f"{v:g}" if budget else f"×{v:g}" for v in ticks]))
            axis.set_minor_locator(NullLocator())
    else:
        frame_arms = arms if budget else []
        xs = [origin] + [p["entry"]["seen"][rel_key][0] for p in frame if p["entry"]["seen"][rel_key]] + [s * hx for hx, _ in frame_arms for s in (-1, 1)]
        ys = [origin] + [p["entry"]["held"][rel_key][0] for p in frame if p["entry"]["held"][rel_key]] + [s * hy for _, hy in frame_arms for s in (-1, 1)]
        if budget:
            relative.set_xlim(min(xs) - 0.06, max(xs) + 0.15)
            relative.set_ylim(min(ys) - 0.07, max(ys) + 0.13)
            relative.xaxis.set_major_locator(MultipleLocator(0.2))
            relative.yaxis.set_major_locator(MultipleLocator(0.2))
        else:
            rx, ry = max(xs) - min(xs), max(ys) - min(ys)
            relative.set_xlim(min(xs) - 0.08 * rx, max(xs) + 0.16 * rx)
            relative.set_ylim(min(ys) - 0.13 * ry, max(ys) + 0.12 * ry)
            for axis, (lo, hi) in ((relative.xaxis, relative.get_xlim()), (relative.yaxis, relative.get_ylim())):
                ratios = [v for v in range(1, 9) if lo <= v <= hi]
                axis.set_major_locator(FixedLocator(ratios))
                axis.set_major_formatter(FixedFormatter([f"\u00d7{v:g}" for v in ratios]))
        # Preserve the broad sparse-track frame, extending it only where an actual corrected interval would be cut.
        for test, limits, current in (("seen", relative.set_xlim, relative.get_xlim()),
                                       ("held", relative.set_ylim, relative.get_ylim())):
            values = [v for p in points for v in (p["entry"][test][interval_key] or [])
                      if v is not None and math.isfinite(v)]
            if values:
                lo, hi = current
                pad = 0.02 * (hi - lo)
                limits(min(lo, min(values) - pad) if min(values) < lo else lo,
                       max(hi, max(values) + pad) if max(values) > hi else hi)
    if open_label:
        for ax, drawn, corner in ((absolute, drawn_abs, True), (relative, drawn_rel, False)):
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            w, h = x1 - x0, y1 - y0
            right, top = x1 - 0.04 * w, y1 - 0.04 * h
            left = x0 + 0.04 * w if corner else max(x0 + 0.05 * w, origin + 0.04 * w)
            floor = y0 + 0.05 * h if corner else origin + 0.05 * h
            under = [b for low, high, b in drawn if high >= left - 0.04 * w and low <= right + 0.04 * w]
            bottom = max([floor] + [b + 0.07 * h for b in under])
            if top - bottom >= 0.15 * h:
                ax.add_patch(FancyBboxPatch((left, bottom), right - left, top - bottom, fill=False, ls=(0, (3, 2)),
                                            lw=0.6, ec=STONE, zorder=1, boxstyle=f"round,pad=0,rounding_size={0.03 * h}"))
                ax.text((left + right) / 2, (bottom + top) / 2, open_label, ha="center", va="center", color=STONE,
                        zorder=2, fontproperties=font_manager.FontProperties(family=brand.SANS, size=6.4))
    # where no change falls, in fractions of the panel, on whatever scale it has
    at = tuple(relative.transAxes.inverted().transform(relative.transData.transform((origin, origin))))
    _better_wash(relative, at=at)
    _better_wash(absolute, at=at)
    for ax, heading, measure in ((absolute, title, words), (relative, rel_title, rel_words)):
        ax.set_title(heading, loc="left", color=INK, fontsize=7.2)
        seen, held = AXIS_SYMBOLS.get(measure, ("", ""))
        ax.set_xlabel(f"memorization: {measure} on the seen test{seen}", fontsize=6.6, color=STONE)
        ax.set_ylabel(f"generalization: {measure} on the held-out test{held}", fontsize=6.2, color=STONE)
        _rule_axes(ax); _mono_ticks(ax)
        if fit_axes and ax is relative:
            # Longer fractional ratio ticks and labels must stay inside the figure.
            ax.xaxis.label.set_ha("right")
            ax.xaxis.label.set_x(1.0)


def _visits(run: Run) -> tuple[list, list, list, list[bool]]:
    """Budget used by a user's visit to a task, 1 to VISITS, with the 16th to 84th percentile over task resamples, and
    whether every task has had the visit; a visit with fewer than TEST_MIN_POINT instances is None. A visit is drawn
    before every task has had it, which on this stream is not before round 98, and the last visit not before the last
    round."""
    sums = [[0.0, 0] for _ in EXPOSURE_BUCKETS]
    for r in run.records:
        for i, (lo, hi) in enumerate(EXPOSURE_BUCKETS):
            if _in_bucket(r.exposure_index, lo, hi):
                sums[i][0] += float(r.cost); sums[i][1] += 1
    mean = [s / n if n >= TEST_MIN_POINT else None for s, n in sums]
    lo, hi = _boot_buckets(run, "cost_by_exposure", len(EXPOSURE_BUCKETS), EXPOSURE_BUCKETS)
    keep = lambda v: [x if m is not None else None for x, m in zip(v, mean)][:VISITS]  # noqa: E731
    cap = run.max_cost
    return (_to_budget(keep(mean), cap), _to_budget(keep(lo), cap), _to_budget(keep(hi), cap),
            run.complete_buckets()[:VISITS])


def _visit_stats(runs: list[Run], measure: str) -> tuple[list[float], list[float], list[float]]:
    """Each visit's ``measure`` over the runs' stream records, in percent, with its 95 percent interval over task
    resamples: "solved", the share of instances solved within the cap; "solved budget", the budget per solved instance;
    any other, the budget used per instance. NaN for a visit with fewer than TEST_MIN_POINT instances. The stream's
    records alone, since some cells also hold the records of a test."""
    cells: dict[str, list[list[float]]] = {}
    for run in runs:
        for r in run.stream_records():
            if r.exposure_index >= VISITS or (measure == "solved budget" and not r.solved):
                continue
            row = cells.setdefault(r.task_id, [[0.0, 0] for _ in range(VISITS)])
            row[r.exposure_index][0] += 100.0 * (bool(r.solved) if measure == "solved" else float(r.cost) / run.max_cost)
            row[r.exposure_index][1] += 1
    sums = [sum(row[b][0] for row in cells.values()) for b in range(VISITS)]
    counts = [sum(row[b][1] for row in cells.values()) for b in range(VISITS)]
    dists = _boot_task_means({t: [tuple(c) for c in row] for t, row in cells.items()}, VISITS, reps=LEARNED_REPS)
    mean, lo, hi = [], [], []
    for d, total, n in zip(dists, sums, counts):
        d = sorted(d)
        whole = n >= TEST_MIN_POINT
        mean.append(total / n if whole else float("nan"))
        lo.append(d[int(0.025 * len(d))] if whole else float("nan"))
        hi.append(d[min(len(d) - 1, int(0.975 * len(d)))] if whole else float("nan"))
    return mean, lo, hi


def _visit_band(ax, runs: list[Run], measure: str, style: dict) -> None:
    """Each visit's 95 percent interval (_visit_stats) as a faint band in the row's color along its line."""
    _, lo, hi = _visit_stats(runs, measure)
    ax.fill_between(range(VISITS), lo, hi, color=style["color"], alpha=0.12, lw=0, zorder=1)


def _visit_shares(runs: list[Run]) -> list[float | None]:
    """The share of instances solved within the cap at each visit to a task, 1 to VISITS, in percent, over the runs'
    stream records; None for a visit with fewer than TEST_MIN_POINT instances."""
    sums = [[0, 0] for _ in range(VISITS)]
    for run in runs:
        for r in run.stream_records():
            if r.exposure_index < VISITS:
                sums[r.exposure_index][0] += bool(r.solved)
                sums[r.exposure_index][1] += 1
    return [100.0 * k / n if n >= TEST_MIN_POINT else None for k, n in sums]


def _visit_axis(ax, labelled: bool) -> None:
    """A panel's axis of visits to the task, 1 to VISITS, its ticks named and the axis labelled only where
    ``labelled``. The visits run edge to edge, the first on the axis and the last on its end, so that the lines and
    bands meet both with no margin, which would read as a padded border."""
    ax.set_xticks(range(VISITS))
    ax.set_xticklabels([str(k + 1) for k in range(VISITS)])
    ax.tick_params(axis="x", labelbottom=labelled)
    ax.set_xlim(0, VISITS - 1)
    ax.set_xlabel(r"$i$th task visit (mean over 25 tasks)" if labelled else "", fontsize=7.2, color=STONE,
                  labelpad=1.0)   # what a stream point is, said where the reader looks for it
    _rule_axes(ax); _mono_ticks(ax)
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(brand.mono(5.2))


def fig_notes_weights() -> Path:
    """figures/notes-weights.pdf, Figure 4: each row's learning curve on the stream, the student's column beside the
    teacher's, the two the same size, mirror images, with a rule between them and Figure 3's headings over each. In
    each column the share solved above, how well a row plays, and below it the budget, what that play cost, both by
    visit to the task, 1 to 8, so that both read against how much of a task a learner has seen, each on an axis fitted
    to its own values. The frozen tests are Figure 3's and Table 1's, so this figure is the learning curve alone. Each
    row's 95 percent interval runs along its line as a faint band (_visit_band), muddied where the lines cross but
    still telling a wide visit from a narrow one.
    Left, Qwen3.5-9B at one user: the rows that keep notes in the harness and DAgger, which also trains the weights and
    runs through the band the notes leave empty, beside the same baselines (the student wiped, the stateless floor and
    the program library). Right, GLM-5.3 Flash in prime-agent at one user, its budget on the instances it solves
    (_solved_panel), beside the same baselines. Each column's key sits below it, clustered by family
    (_clustered_legend), each row in its color (ROW_COLOR). A visit is hollow until every task has had it."""
    qwen = {r.row: r for r in TEST_ROWS if r.player == "qwen" and r.users == 1}
    glm = {r.row: r for r in TEST_ROWS if r.player == "glm" and r.users == 1}
    qwen_names = ["h", "hpl", "hg", "hgpl", "hplg", "DAgger", "h0", "f", "m"]
    glm_names = [row[1] for row in (SOLVED_FLOOR, *SOLVED_ROWS) if row[1] not in APPENDIX_ROWS]
    q_key = [["f", "m"], ["h0", "h", "hpl"], ["hg", "hgpl", "hplg"], ["DAgger"]]
    g_key = [["f", "m"], ["h0", "h", "hpl"]]
    fig = plt.figure(figsize=(WIDTH, 4.3))
    grid = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.1, left=0.1, right=0.985, bottom=0.25, top=0.9)
    qs, gs = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    qb, gb = fig.add_subplot(grid[1, 0], sharex=qs), fig.add_subplot(grid[1, 1], sharex=gs)
    used: list[Run] = []

    def draw(ax, ys, style):
        ax.plot(range(VISITS), [v if v is not None else float("nan") for v in ys], zorder=3, **style)

    def legend_name(row: str, player: str) -> str:
        # Figure 4 uses abbreviations alone; mathtext lowers only the zero, leaving the letters in the body font.
        return re.sub(r"0$", r"$_{0}$", row_name(row, player))

    q_handles: dict = {}
    for name in qwen_names:
        r, style = qwen[name], _line(name)
        run = played(r)
        if run is not None:
            used.append(run)
            mean, _, _, whole = _visits(run)
            draw(qb, mean, style)
            _visit_band(qb, [run], "budget", style)
            open_ = [(x, m) for x, (m, w) in enumerate(zip(mean, whole)) if m is not None and not w]
            if open_:
                qb.plot(*zip(*open_), ls="", marker=style["marker"], ms=style["ms"] + 0.4, mfc="white",
                        mec=style["color"], mew=0.7, zorder=4)
            draw(qs, _visit_shares([run]), style)
            _visit_band(qs, [run], "solved", style)
        q_handles[name] = (plt.Line2D([], [], **style), legend_name(name, "qwen"))
    g_handles, g_used = _solved_panel(gb)
    g_handles = {name: (handle, legend_name(name, "glm")) for name, (handle, _) in g_handles.items()}
    used += g_used
    for name in glm_names:
        r, style = glm[name], _line(name, "glm")
        run = played(r)
        if run is not None:
            draw(gs, _visit_shares([run]), style)
            _visit_band(gs, [run], "solved", style)
    for ax in (qs, qb):
        ax.set_ylim(0, 100)
    # GLM-5.3 Flash solves nearly every instance, so its share solved starts at the nearest ten below its lowest point,
    # as its budget per solve ends just above its highest (_solved_panel): each of its panels fitted to its own values,
    # so that the trends show
    gs.set_ylim(10 * math.floor(gs.dataLim.y0 / 10), 100)
    # every value lies within its panel's limits, so a point on an edge, a share of 100 or a budget of 0, is drawn whole
    # rather than cut in half by the axes
    for ax in (qs, gs, qb, gb):
        for line in ax.get_lines():
            line.set_clip_on(False)
        _zebra(ax)
    for ax, label in ((qb, "budget used per\ninstance (%)"), (qs, "solved (%)"),
                      (gb, "budget per solved\ninstance (%)"), (gs, "solved (%)")):
        ax.set_ylabel(label, fontsize=7)
    for ax, labelled in ((qs, False), (gs, False), (qb, True), (gb, True)):
        _visit_axis(ax, labelled)
    # the student's key spaces its columns wider than the teacher's, so that the heading over a column of short names,
    # "teacher writes" over QT, clears the next column's
    q_legend = _clustered_legend(qb, q_handles, q_key,
                      loc="upper left", bbox_to_anchor=(-0.27, -0.44), fontsize=5.2, handlelength=1.8,
                      handletextpad=0.5, columnspacing=3.3, labelspacing=0.3)
    g_legend = _clustered_legend(gb, g_handles, g_key, loc="upper left", bbox_to_anchor=(-0.27, -0.44),
                      fontsize=5.2, handlelength=1.8, handletextpad=0.5, columnspacing=0.9, labelspacing=0.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    _key_headings(q_legend, ["baselines", "student writes", "teacher writes", "teacher trains"])
    _key_headings(g_legend, ["baselines", "teacher plays"])
    # the rule between the student's column and the teacher's, halfway from the one to the other's axis labels, and
    # each column's heading from its own edge
    label_x0 = min(fig.transFigure.inverted().transform((ax.yaxis.label.get_window_extent(renderer).x0, 0))[0]
                   for ax in (gb, gs))
    rule = (qb.get_position().x1 + label_x0) / 2
    head = font_manager.FontProperties(family=brand.SANS, size=7.4)
    fig.text(0.012, 0.95, "Qwen3.5-9B (student) plays", ha="left", va="bottom", color=INK, fontproperties=head)
    fig.text(rule + 0.014, 0.95, "GLM-5.3 Flash (teacher) plays", ha="left", va="bottom", color=INK,
             fontproperties=head)
    fig.add_artist(plt.Line2D([rule, rule], [0.03, 0.985], transform=fig.transFigure, color=LINE_STRONG, lw=0.6))
    out = FIG / "notes-weights.pdf"
    _save(fig, out)
    plt.close(fig)
    _report("notes-weights", _distinct(used))
    return out


# The appendix's tables of every row of the registry, one float each, since a single table of the whole registry is
# taller than the page: file under sections/tables/ -> (player, groups). Qwen3.5-9B's rows with notes in the harness
# and GLM-5.3 Flash's rows, at one user, H-A and the rows that keep nothing among them (Table 6 in Appendix F), and
# Qwen3.5-9B's trained row beside its baselines at every number of users (Table 11 in Appendix H). Every row of the
# registry is in exactly one.
APPENDIX_TABLES = {"tests-notes": ("qwen", (NOTES, TEACHER_NOTES)),
                   "tests-users": ("qwen", (WEIGHTS, FLOORS)),
                   "tests-glm": ("glm", (NOTES, WEIGHTS, FLOORS))}
assert all(sum(r.player == player and TEST_LOOK[r.row][1] in groups for player, groups in APPENDIX_TABLES.values())
           == 1 for r in TEST_ROWS)


def _bold_best(values: list, higher: bool, digits: int) -> list[bool]:
    """Which of a table column's values to set in bold: every one that prints as the column's best, the highest or the
    lowest at ``digits`` places, so that a tie on the page is bold on each of its rows. Every row competes, the
    baselines as the learners do."""
    shown = [None if v is None else round(v, digits) for v in values]
    best = (max if higher else min)((v for v in shown if v is not None), default=None)
    return [v is not None and v == best for v in shown]


def _bold_measurable(values: list, lows: list, digits: int) -> list[bool]:
    """Which of a column of savings (higher is better) to set in bold: its best as printed (_bold_best), among only
    the values whose 95 percent interval lies above zero, so that bold never points at a saving within noise of none,
    and a column with no such value has no bold."""
    kept = [v if v is not None and lo is not None and lo > 0 else None for v, lo in zip(values, lows)]
    return _bold_best(kept, True, digits)


def table_tests() -> Path:
    """sections/tables/tests.tex, Table 1, a line per row of the registry at one user with Qwen3.5-9B's measures and
    then GLM-5.3 Flash's across, and the appendix's tables of every number of users, Tables 6 and 11, a line per
    player, row and number of users (APPENDIX_TABLES): the share learned on the stream (stream_learned, once the row
    and its twin have the whole stream) and the share saved on the seen and on the held-out test (frozen_measure), each
    against the row's reset and with its 95 percent interval in small type; -- where a value is not in, and in Table 1's
    cells of a row its player does not play. A row that keeps nothing between instances is its own twin, and its share
    learned prints as 0. Each table sets in bold each player's best value in each column among the values whose 95
    percent interval lies above zero (_bold_measurable), comparing values as printed, so that rows printing the same
    value are bold together; Table 1 names each group of rows in a narrow column beside them, wrapped onto as many
    lines as the group has rows. It also writes chance-numbers.tex, the macros the text quotes: \\testPlays, the
    number of plays of each test, and the chance half-widths of _cross."""
    def num(v, digits, sign=""):
        v = round(v, digits) + 0.0   # no -0.0
        return f"{v:{sign}.{digits}f}" if v else f"{0:.{digits}f}"
    def cell(d, key, digits, size="\\scriptsize", bold=False):
        if d is None or d.get(key) is None:
            return "-- &"
        bar = f"{{{size} [{num(d['ci95'][0], digits)}, {num(d['ci95'][1], digits)}]}}" if d["ci95"] else ""
        value = num(d[key], digits, '+')
        shown = "\\textbf{" + value + "}" if bold else value
        return f"{shown} & {bar}"
    used: list[Run] = []
    order = {g: i for i, g in enumerate((NOTES, TEACHER_NOTES, WEIGHTS, FLOORS))}
    def values(r: TestRow) -> list[float | None]:
        """The row's share learned on the stream and its measure on each test, None where not in yet."""
        learned = stream_learned(r)
        l_value = 0.0 if r.run == r.twin and learned is not None else (learned or {}).get("share_learned")
        return [l_value] + [(frozen_measure(r, test) or {}).get("value") for test in FROZEN_TESTS]
    def lows(r: TestRow) -> list[float | None]:
        """The lower ends of the row's 95 percent intervals, as values() orders them; None where there is no interval,
        as for a row that keeps nothing, whose share learned is 0 by definition."""
        learned = None if r.run == r.twin else stream_learned(r)
        return [((learned or {}).get("ci95") or [None])[0]] + [((frozen_measure(r, test) or {}).get("ci95") or [None])[0]
                                                            for test in FROZEN_TESTS]
    def measures(r: TestRow, size: str, bold=(False, False, False)) -> list[str]:
        """The row's share learned on the stream and its measure on each test, a value and its interval each."""
        learned = stream_learned(r)
        tests = [frozen_measure(r, test) for test in FROZEN_TESTS]
        for d in [learned, *tests]:
            used.extend((d or {}).get("runs", []))
        l_cell = (("\\textbf{0}" if bold[0] else "0") + " &" if r.run == r.twin and learned is not None
                  else cell(learned, "share_learned", 2, size, bold[0]))
        return [l_cell] + [cell(t, "value", MEASURE_DIGITS[test], size, b)
                           for test, t, b in zip(FROZEN_TESTS, tests, bold[1:])]
    head = ("\\multicolumn{2}{c}{$L$} & \\multicolumn{2}{c}{$S_{\\mathrm{seen}}$} & "
            "\\multicolumn{2}{c}{$S_{\\mathrm{held}}$}")
    digits = [2] + [MEASURE_DIGITS[test] for test in FROZEN_TESTS]
    def bold_flags(rows: list[TestRow]) -> dict[int, list[bool]]:
        """Each of ``rows``' bold flags by id, one per column (L, then each test): its printed value is its player's
        best in that column among the rows whose interval there lies above zero (_bold_measurable)."""
        got = {id(r): values(r) for r in rows}
        low = {id(r): lows(r) for r in rows}
        flags: dict[int, list[bool]] = {id(r): [] for r in rows}
        for player in {r.player for r in rows}:
            mine = [r for r in rows if r.player == player]
            for k, d in enumerate(digits):
                for r, flag in zip(mine, _bold_measurable([got[id(r)][k] for r in mine], [low[id(r)][k] for r in mine], d)):
                    flags[id(r)].append(flag)
        return flags
    def tabular(rows: list[TestRow]) -> str:
        """A line per row and number of users, the users column left out where every row is at one user."""
        lines, prev = [], None
        rows = sorted(rows, key=lambda r: (list(PLAYERS).index(r.player), order[TEST_LOOK[r.row][1]]))
        flags = bold_flags(rows)
        several = any(r.users > 1 for r in rows)
        for r in rows:
            group = TEST_LOOK[r.row][1]
            name = row_name(r.row, r.player)
            block = f"{PLAYERS[r.player].split(' (')[0]}, {group}"
            if block != prev:
                lines += ["\\midrule", f"\\multicolumn{{{8 if several else 7}}}{{l}}{{\\emph{{{block}}}}} \\\\"]
            users = f" & {r.users}" if several else ""
            lines.append(f"{_tex_name(name)}{users} & "
                         + " & ".join(measures(r, "\\scriptsize", flags[id(r)])) + " \\\\")
            prev = block
        return ("% generated by scripts/make_figures.py tests-table; do not edit\n"
                "\\begin{tabular}{l" + ("r" if several else "") + "r@{\\hspace{0.4em}}l" * 3 + "}\n\\toprule\n"
                "method" + (" & users" if several else "") + f" & {head} \\\\\n" + "\n".join(lines)
                + "\n\\bottomrule\n\\end{tabular}\n")
    def wide(rows: list[TestRow]) -> str:
        registry = [r.row for r in TEST_ROWS]
        by = {(r.player, r.row): r for r in rows}
        names = sorted({r.row for r in rows}, key=lambda row: (order[TEST_LOOK[row][1]], registry.index(row)))
        flags = bold_flags(list(by.values()))
        def bold(player: str, row: str) -> list[bool]:
            return flags[id(by[player, row])]
        groups: dict[str, list[str]] = {}
        for row in names:
            groups.setdefault(TEST_LOOK[row][1], []).append(row)
        lines = []
        for group, rows in groups.items():
            side = textwrap.wrap(group, 11)
            if len(side) > len(rows):
                raise SystemExit(f"tests-table: '{group}' takes {len(side)} lines beside its {len(rows)} rows")
            lines.append("\\midrule")
            label = "\\\\".join(side)
            for i, row in enumerate(rows):
                cells = [c for player in PLAYERS
                         for c in (measures(by[player, row], "\\tiny", bold(player, row)) if (player, row) in by
                                   else ["-- &"] * 3)]
                name = (f"\\multirow[t]{{{len(rows)}}}{{*}}{{\\tiny\\emph{{\\begin{{tabular}}[t]{{@{{}}l@{{}}}}"
                        f"{label}\\end{{tabular}}}}}}" if i == 0 else "")
                lines.append(f"{name} & {_row_names(row, [p for p in PLAYERS if (p, row) in by])} & " + " & ".join(cells) + " \\\\")
        players = " & ".join(f"\\multicolumn{{6}}{{c}}{{{PLAYERS[p].split(' (')[0]}}}" for p in PLAYERS)
        rules = "".join(f"\\cmidrule(lr){{{3 + 6 * i}-{8 + 6 * i}}}" for i in range(len(PLAYERS)))
        return ("% generated by scripts/make_figures.py tests-table; do not edit\n"
                "\\begin{tabular}{@{}l@{\\hspace{4pt}}l" + "r@{\\hspace{0.3em}}l" * 3 * len(PLAYERS)
                + "}\n\\toprule\n"
                f" & & {players} \\\\\n{rules}\n"
                " & method & " + " & ".join([head] * len(PLAYERS)) + " \\\\\n" + "\n".join(lines)
                + "\n\\bottomrule\n\\end{tabular}\n")
    for name, (player, groups) in APPENDIX_TABLES.items():
        # a row that keeps nothing at two or four users is the reset of the rows above it, with one play of each test
        # and nothing to read it against (_own), so the table lists it at one user alone
        rows = [r for r in TEST_ROWS if r.player == player and TEST_LOOK[r.row][1] in groups
                and (r.users == 1 or r.run != r.reset)]
        (TABLES / f"{name}.tex").write_text(tabular(rows))
    path = TABLES / "tests.tex"
    # Table 1 leaves out the rows that keep nothing, each its own reset, whose L is 0 by construction and whose shares
    # are chance; the appendix's tables carry them
    path.write_text(wide([r for r in TEST_ROWS if r.users == 1 and r.row not in APPENDIX_ROWS and r.run != r.twin]))
    macros = ["% generated by scripts/make_figures.py tests-table; do not edit"]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    plays = min(_fewest_plays(player) for player in PLAYERS)
    macros.append(f"\\newcommand{{\\testPlays}}{{{words.get(plays, plays)}}}")
    for player in PLAYERS:
        for test, half in zip(FROZEN_TESTS, _cross(player) or (None, None)):
            name = "\\chance" + player.capitalize() + ("Seen" if test == "seen" else "Held")
            macros.append(f"\\newcommand{{{name}}}{{{'--' if half is None else f'{half:.{MEASURE_DIGITS[test]}f}'}}}")
    (TABLES / "chance-numbers.tex").write_text("\n".join(macros) + "\n")
    _report("tests-table", _distinct(used))
    return path

# ---------------------------------------------------------------------------
# GPT-6 Luna's frozen tests: each system's workstreams cell plays both tests from its state at the stream's end, in
# <cell>/frozen/<test>/play/ and, played again, <test>-2/, <test>-3/ and on, and the fresh cell of its reset, the same
# system wiped, plays them from an empty state in <cell>/frozen/<test>-reference/play/ and <test>-reference-K/
# (runs/README.md). luna_test reads them as an open model's row is read, the mean of a system's plays against the mean
# of its reset's, four a side; a system that keeps nothing reads half its plays against the other half, so the spread
# of those rows is the noise of playing a test, as for the open models' rows.
# ---------------------------------------------------------------------------

# each system's reset on the frozen tests: LEARNED_RESET, with Appendix F's rows (HA, H-A, keeps what it is asked to on
# top of row H, so its reset is H0's; HPL's, H-E's, is HPL0, given the same guidance and wiped)
LUNA_TEST_RESET = {**LEARNED_RESET, "HA": "H0", "HPL": "HPL0", "HPL0": "HPL0", "M0 program library": "M0 program library"}
# the table's systems, in its order: (row, harness or None, group, name in the paper); the rows that keep nothing
# between instances first in each group
LUNA_TEST_SYSTEMS = [("F stateless", None, "floors", "stateless"), ("F demo-cache", None, "floors", "demo-cache"),
                     ("M0 program library", None, "program library", "PL\\textsubscript{0}"),
                     ("M program library", None, "program library", "PL"),
                     *[(row, h, h, name) for h in sorted(HARNESSES)
                       for row, name in (("H0", "H\\textsubscript{0}"), ("HPL0", "H-E\\textsubscript{0}"), ("H", "H"),
                                         ("HA", "H-A"), ("HPL", "H-E"))]]


def _luna_plays(cell: "Run", test: str, reference: bool) -> list["Run"]:
    """Every whole play of ``test`` in a GPT-6 Luna cell's frozen/ folder, the first and then by number (_play_names):
    its own from its end-of-stream state, or with ``reference`` its plays from an empty state."""
    frozen = cell.path / "frozen"
    spec = {"row": cell.row, "harness": cell.harness, "model": cell.model, "track": "arc1", "seed": TRACK_SEED,
            "mode": f"{test} test"}
    plays = []
    for name in _play_names(frozen, test, reference):
        play = _frozen_play(frozen / name / "play", test, dict(spec, label=f"{cell.label}, {TEST_NAMES[test]}, {name}"))
        if play is not None:
            plays.append(play)
    return plays


@functools.cache
def luna_test(row: str, harness: str, model: str, test: str) -> dict | None:
    """The share saved on a frozen test (frozen_measure) of one GPT-6 Luna system at one effort, as ``value``, read as
    an open model's row is (_row_share_cells): the mean of its workstreams cell's plays of the test, each from its
    end-of-stream state, against the mean of its reset's (LUNA_TEST_RESET), the reset's fresh cell's plays from an empty
    state and its workstreams cell's own, which a system that keeps nothing plays as it would from nothing, so that a
    system that keeps nothing reads half those plays against the other half. With the 95 percent interval over tasks,
    and the plays of each side; None until both sides have played the whole test."""
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    pool = [r for r in registered_runs() + ablation_runs() if r.track == "arc1" and str(r.seed) == str(TRACK_SEED)
            and r.model == model and not r.spec.get("repeat")]
    def cell(row_, mode):
        return next((r for r in pool if r.row == row_ and r.mode == mode
                     and r.harness == (harness if row_.startswith("H") else "")), None)
    reset_row = LUNA_TEST_RESET[row]
    run, own, fresh = cell(row, "workstreams"), cell(reset_row, "workstreams"), cell(reset_row, "fresh")
    if run is None or fresh is None:
        return None
    resets = (_luna_plays(own, test, False) if own else []) + _luna_plays(fresh, test, True)
    plays, resets = (resets[0::2], resets[1::2]) if row == reset_row else (_luna_plays(run, test, False), resets)
    if not plays or not resets:
        return None
    cells = _share_cells(_mean_records(tuple(plays), test, TEST_PHASE[test]),
                         _mean_records(tuple(resets), test, TEST_REFERENCE_PHASES[test][0]), test)
    if not cells:
        return None
    share = _share_saved(cells)
    return dict(share, runs=plays + resets, plays=plays, resets=resets, value=share["share"])


@functools.cache
def _luna_metric_cells(row: str, harness: str, model: str, test: str, metric: str) -> dict[str, list[float]]:
    d = luna_test(row, harness, model, test)
    if d is None:
        return {}
    return d["_cells"] if metric == "share" else _paired_cells(d["plays"], d["resets"], test, metric)


@functools.cache
def _luna_interval_family(model: str, test: str, metric: str) -> tuple[tuple[str, str], ...]:
    """One family per effort/test/metric: the measured persistent rows across all three harnesses and two baselines."""
    return tuple((row, h or "") for row, h, _, _ in LUNA_TEST_SYSTEMS if LUNA_TEST_RESET[row] != row
                 and all(luna_test(row, h or "", model, side) is not None for side in FROZEN_TESTS)
                 and sum(c[1] for c in _luna_metric_cells(row, h or "", model, test, metric).values()) > 0)


def table_luna_tests() -> Path:
    """sections/tables/tests-luna.tex, Table 12: the share saved on the seen and on the held-out test of every GPT-6
    Luna system at every effort, each against its reset (luna_test), with its 95 percent interval in small type; --
    where a value is not in. A row that keeps nothing between instances reads half its plays against the other half.
    Each column's best share saved is bold among those whose interval lies above zero (_bold_measurable)."""
    models = sorted({r.model for r in registered_runs() if r.track == "arc1" and r.model.startswith("GPT-6 Luna")},
                    key=_effort_rank)
    def eff(m): return m.split("(")[-1].rstrip(")")
    def num(v, digits):
        v = round(v, digits) + 0.0   # no -0.0
        return f"{v:+.{digits}f}" if v else f"{0:.{digits}f}"
    used: list[Run] = []
    strong = {}
    for test in FROZEN_TESTS:
        for m in models:
            got = [luna_test(row, h or "", m, test) for row, h, _, _ in LUNA_TEST_SYSTEMS]
            flags = _bold_measurable([d["value"] if d else None for d in got],
                                     [d["ci95"][0] if d and d.get("ci95") else None for d in got], MEASURE_DIGITS[test])
            strong.update({(row, h, test, m): f for (row, h, _, _), f in zip(LUNA_TEST_SYSTEMS, flags)})
    lines, prev = [], None
    for row, h, group, name in LUNA_TEST_SYSTEMS:
        if group != prev:
            lines += ["\\midrule", f"\\multicolumn{{{1 + 2 * len(FROZEN_TESTS) * len(models)}}}{{l}}{{\\emph{{{group}}}}} \\\\"]
        cells = []
        for test in FROZEN_TESTS:
            for m in models:
                d = luna_test(row, h or "", m, test)
                if d is None:
                    cells.append("-- &"); continue
                used.extend(d["runs"])
                digits = MEASURE_DIGITS[test]
                bar = f"{{\\scriptsize [{num(d['ci95'][0], digits)}, {num(d['ci95'][1], digits)}]}}" if d["ci95"] else ""
                value = num(d["value"], digits)
                value = "\\textbf{%s}" % value if strong[row, h, test, m] else value
                cells.append(f"{value} & {bar}")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        prev = group
    head = ("method & " + " & ".join(f"\\multicolumn{{{2 * len(models)}}}{{c}}{{{MEASURE_AXIS[t].split(' (')[0]}}}" for t in FROZEN_TESTS)
            + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 2 * len(models) * i}-{1 + 2 * len(models) * (i + 1)}}}"
                                     for i in range(len(FROZEN_TESTS)))
            + "\n GPT-6 Luna, effort & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{eff(m)}}}" for _ in FROZEN_TESTS for m in models)
            + " \\\\")
    path = TABLES / "tests-luna.tex"
    path.write_text("% generated by scripts/make_figures.py tests-luna; do not edit\n"
                    "\\begin{tabular}{l" + "r@{\\hspace{0.4em}}l" * (len(FROZEN_TESTS) * len(models)) + "}\n\\toprule\n"
                    + head + "\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _report("tests-luna", _distinct(used))
    return path


SOLVED_FROM = 100        # the table's rounds: every task has been met by then
SOLVED_LUNA = "GPT-6 Luna (high)"
# (name in the paper, GLM-5.3 Flash's TEST_ROWS row, GPT-6 Luna's registry row): the program library and the stateless
# floor are systems of their own, and GPT-6 Luna plays every other row in each of the three harnesses
SOLVED_ROWS = (("H0", "h0", "H0"), ("H", "h", "H"), ("H-A", "ha", "HA"), ("H-E", "hpl", "HPL"),
               ("PL", "m", "M program library"))
SOLVED_FLOOR = ("C\\textsubscript{0} stateless", "f", "F stateless")
SOLVED_MACRO = {"C\\textsubscript{0} stateless": "Stateless", "H0": "Hzero", "H": "H", "H-A": "Ha", "H-E": "He", "PL": "Pl"}
SOLVED_HARNESS_MACRO = {"prime-agent": "Prime", "hermes-agent": "Hermes", "openclaw": "Openclaw"}


def _solved_luna(row: str, harness: str) -> Run | None:
    """GPT-6 Luna's workstreams cell of ``row`` at high effort and seed 0; ``harness`` names the harness of an H row."""
    pool = [r for r in registered_runs() + ablation_runs() if r.track == "arc1" and str(r.seed) == str(TRACK_SEED)
            and r.model == SOLVED_LUNA and r.mode == "workstreams" and not r.spec.get("repeat")]
    return next((r for r in pool if r.row == row and r.harness == (harness if row.startswith("H") else "")), None)


def _solved_runs(side: str, row: tuple[str, str, str]) -> list[Run]:
    """The runs behind one system of the figure: GLM-5.3 Flash's row at one user, or GPT-6 Luna's in each harness."""
    name, glm_row, luna_row = row
    if side == "glm":
        found = [played(next(r for r in TEST_ROWS if r.player == "glm" and r.row == glm_row and r.users == 1))]
    elif luna_row.startswith("H"):
        found = [_solved_luna(luna_row, h) for h in HARNESSES]
    else:
        found = [_solved_luna(luna_row, "")]
    return [r for r in found if r is not None]


def _solved_cells(runs: list[Run], n: int, block, charge=None) -> dict[str, list[tuple[float, int]]]:
    """Per task, the (sum, count) of the budget each solved instance used, in percent of the budget, in each of ``n``
    buckets (``block`` maps a record to its bucket, or None to leave it out); ``charge`` replaces an instance's own
    charge, as perfect memory's does."""
    cells: dict[str, list[list[float]]] = {}
    for run in runs:
        for r in run.stream_records():
            b = block(r)
            if b is None or not r.solved:
                continue
            row = cells.setdefault(r.task_id, [[0.0, 0] for _ in range(n)])
            row[b][0] += 100.0 * (r.cost if charge is None else charge(r)) / run.max_cost
            row[b][1] += 1
    return {t: [tuple(c) for c in row] for t, row in cells.items()}


def _cell_means(cells: dict[str, list[tuple[float, int]]], n: int) -> list[float | None]:
    out = []
    for b in range(n):
        s, k = sum(c[b][0] for c in cells.values()), sum(c[b][1] for c in cells.values())
        out.append(s / k if k else None)
    return out


def _solved_panel(ax) -> tuple[dict, list[Run]]:
    """GLM-5.3 Flash's rows at one user on ``ax`` (Figure 4's lower right panel): the budget a solved instance used by
    visit to its task, 1 to VISITS, on the axis _visit_axis gives every panel, so that it reads against how much of a
    task the learner has seen rather than when the stream served it, each row in its color as GLM-5.3 Flash plays it
    (ROW_COLOR), beside the stateless floor as the student's panel is, each visit with its 95 percent interval as a
    band (_visit_band). Returns the lines' key entries by row and the runs read."""
    def block(r): return r.exposure_index if r.exposure_index < VISITS else None
    used: list[Run] = []
    handles: dict = {}
    for row in (SOLVED_FLOOR, *SOLVED_ROWS):
        if row[1] in APPENDIX_ROWS:
            continue
        runs = _solved_runs("glm", row)
        if not runs:
            continue
        used.extend(runs)
        mean = _cell_means(_solved_cells(runs, VISITS, block), VISITS)
        style = _line(row[1], "glm")
        ax.plot(range(VISITS), [m if m is not None else float("nan") for m in mean], zorder=3, **style)
        _visit_band(ax, runs, "solved budget", style)
        handles[row[1]] = (plt.Line2D([], [], **style), _key_name(row[1], "glm"))
    ax.set_ylim(0, None)
    return handles, used


def _solved_stats(run: Run) -> dict:
    """From round SOLVED_FROM on: the share of instances solved, the budget a solved instance used with its 95 percent
    interval over task resamples, the share of solves that used nothing, and the demonstrations and wrong answers
    behind a solve."""
    late = [r for r in run.stream_records() if r.round >= SOLVED_FROM]
    solved = [r for r in late if r.solved]
    cells = _solved_cells([run], 1, lambda r: 0 if r.round >= SOLVED_FROM else None)
    d = sorted(_boot_task_means(cells, 1, reps=LEARNED_REPS)[0])
    return {"solved": 100.0 * len(solved) / len(late), "cost": _cell_means(cells, 1)[0],
            "ci": (d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]),
            "free": 100.0 * statistics.mean(r.cost == 0 for r in solved),
            "demos": statistics.mean(r.demo_requests for r in solved),
            "wrong": statistics.mean(r.wrong_attempts for r in solved), "complete": run.complete}


def table_solved_budget() -> Path:
    """sections/tables/solved-budget.tex, Table 9 (GLM-5.3 Flash), and sections/tables/solved-numbers.tex with the
    macros the text quotes (\\sb<model><harness><row><measure>, and for GPT-6 Luna's harness rows the least and most
    over its three harnesses): _solved_stats for GLM-5.3 Flash's rows at one user, then GPT-6 Luna's at high effort,
    harness by harness. A run still playing is marked *; a system with no run is --."""
    systems = [("glm", "GLM-5.3 Flash (medium), prime-agent", None, row) for row in (SOLVED_FLOOR, *SOLVED_ROWS)]
    systems += [("luna", "GPT-6 Luna (high)", None, row) for row in (SOLVED_FLOOR, SOLVED_ROWS[-1])]
    systems += [("luna", f"GPT-6 Luna (high), {h}", h, row) for h in HARNESSES for row in SOLVED_ROWS[:-1]]
    def num(v, places=1): return f"{v:.{places}f}"
    used: list[Run] = []
    lines, macros, prev = [], [], None
    spread: dict[str, list[dict]] = defaultdict(list)
    entries = []
    for side, group, harness, row in systems:
        run = (_solved_runs("glm", row) or [None])[0] if side == "glm" else _solved_luna(row[2], harness or "")
        entries.append((side, group, harness, row, run, _solved_stats(run) if run is not None else None))
    # each table's best in each column (_bold_best): the most solved, the least budget a solve used, the most solved
    # at no cost, the fewest demonstrations and the fewest wrong answers behind a solve
    columns = (("solved", True, 0), ("cost", False, 1), ("free", True, 0), ("demos", False, 1), ("wrong", False, 1))
    strong: dict[tuple[int, str], bool] = {}
    for table in ("glm", "luna"):
        mine = [i for i, e in enumerate(entries) if e[0] == table]
        for measure, higher, places in columns:
            got = [entries[i][5][measure] if entries[i][5] else None for i in mine]
            strong.update({(i, measure): f for i, f in zip(mine, _bold_best(got, higher, places))})
    for i, (side, group, harness, row, run, s) in enumerate(entries):
        name = row[0]
        if group != prev:
            lines += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{group}}}}} \\\\"]
            prev = group
        label = _tex_name(row_name(row[1], "glm") if side == "glm" else name)
        key = "\\sb" + ("Glm" if side == "glm" else "Luna") + SOLVED_HARNESS_MACRO.get(harness, "") + SOLVED_MACRO[name]
        if run is None:
            lines.append(f"{label} & -- & -- & & -- & -- & -- \\\\")
            macros += [f"\\newcommand{{{key}{m}}}{{--}}" for m in ("Solved", "Cost", "Free", "Demos", "Wrong")]
            continue
        used.append(run)
        if harness:
            spread[SOLVED_MACRO[name]].append(s)
        star = "" if s["complete"] else "\\textsuperscript{*}"
        shown = {}
        for measure, _, places in columns:
            shown[measure] = num(s[measure], places)
            if strong[i, measure]:
                shown[measure] = "\\textbf{" + shown[measure] + "}"
        lines.append(f"{label}{star} & {shown['solved']} & {shown['cost']} & "
                     f"{{\\scriptsize [{num(s['ci'][0])}, {num(s['ci'][1])}]}} & {shown['free']} & "
                     f"{shown['demos']} & {shown['wrong']} \\\\")
        macros += [f"\\newcommand{{{key}Solved}}{{{num(s['solved'], 0)}}}", f"\\newcommand{{{key}Cost}}{{{num(s['cost'])}}}",
                   f"\\newcommand{{{key}Free}}{{{num(s['free'], 0)}}}", f"\\newcommand{{{key}Demos}}{{{num(s['demos'])}}}",
                   f"\\newcommand{{{key}Wrong}}{{{num(s['wrong'])}}}"]
    for row, stats in spread.items():
        for measure, places in (("solved", 0), ("cost", 1), ("free", 0), ("demos", 1), ("wrong", 1)):
            vals = [s[measure] for s in stats]
            macros += [f"\\newcommand{{\\sbLuna{row}{measure.capitalize()}Min}}{{{num(min(vals), places)}}}",
                       f"\\newcommand{{\\sbLuna{row}{measure.capitalize()}Max}}{{{num(max(vals), places)}}}"]
    head = ("system & solved (\\%) & \\multicolumn{2}{c}{budget a solve used (\\%)} & solved at no cost (\\%) & "
            "demonstrations & wrong answers \\\\")
    glm = lines[:next(i for i, l in enumerate(lines) if "GPT-6 Luna" in l) - 1]   # GPT-6 Luna's rows give macros alone
    path = TABLES / "solved-budget.tex"
    path.write_text("% generated by scripts/make_figures.py solved-budget-table; do not edit\n"
                    "\\begin{tabular}{lrr@{\\hspace{0.4em}}lrrr}\n\\toprule\n" + head + "\n" + "\n".join(glm)
                    + "\n\\bottomrule\n\\end{tabular}\n")
    (TABLES / "solved-numbers.tex").write_text(
        "% generated by scripts/make_figures.py solved-budget-table; do not edit\n" + "\n".join(macros) + "\n")
    _report("solved-budget-table", _distinct(used))
    return path
# ---------------------------------------------------------------------------
# Every learner on the two tests, as the planes and Table 10 read it: its share saved and solve rate increase against
# its reset, with their intervals, and the budget used and share solved on each side
# ---------------------------------------------------------------------------


def _absolute(plays: tuple[Run, ...], test: str, phase: str) -> dict:
    """The mean budget used, in percent, and the share solved, in percent, of the plays of one test, each instance the
    mean of its plays' charges."""
    rows = _mean_records(plays, test, phase)
    return {"budget": round(_to_budget([statistics.mean(r.cost for r in rows)], plays[0].max_cost)[0], 1),
            "solved": round(100 * statistics.mean(r.solved for r in rows), 1)}


def _triple(d: dict | None, key: str) -> list | None:
    """A measure and its 95 percent interval, to six places, at which the planes place points and read intervals."""
    if d is None or d.get(key) is None:
        return None
    return [round(d[key], 6) + 0.0] + ([round(x, 6) + 0.0 for x in d["ci95"]] if d.get("ci95") else [None, None])


def _adjusted_fields(share: dict, ratio: dict, families: dict[str, tuple], member) -> dict:
    """Each measure's value with its interval corrected over its family (_adjusted_ci), share_adjusted and
    solved_ratio_adjusted, and the family's size; a learner outside its family gets no interval. ``member`` is the row's
    identity in each family. The share and solved_ratio triples keep their pointwise 95 percent intervals."""
    out = {"interval_family_size": {}, "interval_family_member": {}}
    for metric, measure, key in (("share", share, "share"), ("solved_ratio", ratio, "ratio")):
        family = families[metric]
        value = measure.get(key, measure.get("value") if metric == "share" else None)
        judged = member in family and value is not None and bool(measure.get("_cells"))
        out["interval_family_size"][metric] = len(family)
        out["interval_family_member"][metric] = judged
        if value is None:
            out[metric + "_adjusted"] = None
            continue
        ci = _adjusted_ci(measure["_cells"], len(family), metric) if judged else None
        out[metric + "_adjusted"] = [round(value, 6) + 0.0] + (
            [round(v, 6) + 0.0 for v in ci] if ci is not None else [None, None])
    return out


def _open_entry(r: TestRow) -> dict | None:
    """One row of the open models at its number of users on both tests, against its reset (_sides); None until both
    sides have played both tests."""
    from continual_arc.schedule import TEST_PHASE, TEST_REFERENCE_PHASES
    tests = {}
    for test in FROZEN_TESTS:
        share = frozen_measure(r, test)
        if share is None:
            return None
        plays, resets = _sides(r, test)
        reset = _absolute(resets, test, TEST_REFERENCE_PHASES[test][0])
        mine = _absolute(plays, test, TEST_PHASE[test])
        ratio = _solved_ratio(_share_cells(_mean_records(plays, test, TEST_PHASE[test]),
                                           _mean_records(resets, test, TEST_REFERENCE_PHASES[test][0]), test, _solve))
        tests["held" if test == "held-out" else "seen"] = {"share": _triple(share, "value"),
                                                          "solved_ratio": _triple(ratio, "ratio"), **mine,
                                                          "reset_budget": reset["budget"],
                                                          "reset_solved": reset["solved"],
                                                          **_adjusted_fields(share, ratio, {
                                                              metric: _open_interval_family(r.player, r.users, test, metric)
                                                              for metric in ("share", "solved_ratio")}, r)}
    return tests


def _luna_entries() -> list[dict]:
    """Every GPT-6 Luna system at every effort on the two tests (luna_test), its share learned on the stream against
    its reset's workstreams cell, what its model calls cost per instance of the stream, and its absolute budget and
    solves beside its reset's, each the mean over that side's plays (luna_test), with how many times as much as its
    reset it solves."""
    models = sorted({r.model for r in registered_runs() if r.track == "arc1" and r.model.startswith("GPT-6 Luna")},
                    key=_effort_rank)
    pool = [r for r in registered_runs() + ablation_runs() if r.track == "arc1" and str(r.seed) == str(TRACK_SEED)
            and not r.spec.get("repeat")]
    names = {"F stateless": "stateless floor", "F demo-cache": "demo-cache floor", "M0 program library": "PL0",
             "M program library": "PL", "H0": "H0", "HPL0": "H-E0", "H": "H", "HA": "H-A", "HPL": "H-E"}
    out = []
    for m in models:
        for row, h, group, _ in LUNA_TEST_SYSTEMS:
            def cell(row_, mode):
                return next((r for r in pool if r.model == m and r.row == row_ and r.mode == mode
                             and r.harness == ((h or "") if row_.startswith("H") else "")), None)
            tests = {}
            for test in FROZEN_TESTS:
                d = luna_test(row, h or "", m, test)
                if d is None:
                    break
                mine, ref = d["plays"], d["resets"]
                ratio = _solved_ratio(_paired_cells(mine, ref, test, "solved_ratio"))
                tests["held" if test == "held-out" else "seen"] = {
                    "share": _triple(d, "value"),
                    "solved_ratio": _triple(ratio, "ratio"),
                    "budget": round(_to_budget([statistics.mean(x.cost for p in mine for x in p.records)], mine[0].max_cost)[0], 1),
                    "solved": round(100 * statistics.mean(x.solved for p in mine for x in p.records), 1),
                    "reset_budget": round(_to_budget([statistics.mean(x.cost for p in ref for x in p.records)], ref[0].max_cost)[0], 1),
                    "reset_solved": round(100 * statistics.mean(x.solved for p in ref for x in p.records), 1),
                    **_adjusted_fields(d, ratio, {metric: _luna_interval_family(m, test, metric)
                                                  for metric in ("share", "solved_ratio")}, (row, h or ""))}
            if len(tests) < 2:
                continue
            run, twin = cell(row, "workstreams"), cell(LUNA_TEST_RESET[row], "workstreams")
            keeps_nothing = LUNA_TEST_RESET[row] == row
            learned = None if keeps_nothing or run is None or twin is None else _triple(_learned(run, twin), "share_learned")
            stream = [x for x in run.records if x.phase == "stream"] if run else []
            usd = round(statistics.mean((x.resources or {}).get("llm_cost_usd", 0.0) for x in stream), 4) if stream else None
            out.append({"effort": m.split("(")[-1].rstrip(")"), "harness": h, "row": names[row],
                        "keeps_nothing": keeps_nothing, "learned": learned, "usd": usd, **tests})
    return out


# GPT-6 Luna's figures in Appendix I, Figures 5 and 10: one model in three harnesses, so each harness has a color of its
# own and each row its symbol as Figure 3 draws it, the program library and the demo-cache floor its baselines
HARNESS_COLOR = {"prime-agent": REDS[1], "hermes-agent": BLUES[2], "openclaw": GREENS[1]}
LUNA_CODE = {"demo-cache floor": "dc", "PL": "m", "H": "h", "H-A": "ha", "H-E": "hpl"}
LUNA_NAME = {"dc": "C", "m": "P"}
LUNA_KEY = ((FLOORS, "o", None, "baseline"),
            *((NOTES, "o", color, harness) for harness, color in HARNESS_COLOR.items()),
            (NOTES, "s", INK, "asked to store"), (NOTES, "D", INK, "expert guidance"))
LUNA_GLOSSARY = ((("P", "program library, in place of the harness"), ("C", "demo cache, in place of the harness")),)
# Figure 5 draws the highest effort played; Table 12 gives every effort.
LUNA_PLANE_EFFORT = "high"


def _luna_points(effort: str) -> list[dict]:
    """GPT-6 Luna's systems that keep something, at ``effort``, as _plane_reading's points: each harness's rows in its
    color (HARNESS_COLOR), the baselines named."""
    points = []
    for e in _luna_entries():
        if e["effort"] != effort or e["keeps_nothing"]:
            continue
        code = LUNA_CODE[e["row"]]
        floor = code in LUNA_NAME
        places = {reading: (NAME_RIGHT, NAME_RIGHT) for reading in PLANE_READINGS}
        if code == "m":
            places["budget"] = ((-6, -8, "right", False), NAME_RIGHT)
        points.append({"name": LUNA_NAME.get(code, ""), "family": FLOORS if floor else NOTES, "row": code,
                       "color": None if floor else HARNESS_COLOR[e["harness"]], "entry": e,
                       "places": places})
    return points


def fig_plane_luna() -> Path:
    """figures/plane-luna.pdf, Figure 5: Figure 3's four panels for GPT-6 Luna at LUNA_PLANE_EFFORT on axes fitted to
    its own data, every system that keeps something against its reset (luna_test), with each learner's direct corrected
    intervals."""
    model = f"GPT-6 Luna ({LUNA_PLANE_EFFORT})"
    points = _luna_points(LUNA_PLANE_EFFORT)
    out = FIG / "plane-luna.pdf"
    _plane_figure(points, points, [], LUNA_KEY, LUNA_GLOSSARY, out, tight_axes=True, ratio_linear=LUNA_RATIO_LINEAR)
    _report("plane-luna", _distinct([r for row, h, _, _ in LUNA_TEST_SYSTEMS for test in FROZEN_TESTS
                                     for d in [luna_test(row, h or "", model, test)] if d for r in d["runs"]]))
    return out


def fig_luna_cost() -> Path:
    """figures/luna-cost.pdf, Figure 10: the share learned on the stream of every GPT-6 Luna system that keeps
    something, with its 95 percent interval, against what its model calls cost per instance of the stream, on a log
    scale, as ARC Prize plots score against cost per task, a panel per effort, each harness in its color and each row
    in its symbol, the baselines named."""
    entries = [e for e in _luna_entries() if e["learned"] and e["usd"]]
    efforts = [eff for eff in ("low", "medium", "high") if any(e["effort"] == eff for e in entries)]
    fig = plt.figure(figsize=(WIDTH, 2.55))
    grid = fig.add_gridspec(1, len(efforts), wspace=0.08, left=0.1, right=0.985, bottom=0.3, top=0.8)
    low = min(e["learned"][1] for e in entries); high = max(e["learned"][2] for e in entries)
    spend = [e["usd"] for e in entries]
    for i, eff in enumerate(efforts):
        ax = fig.add_subplot(grid[0, i])
        ax.axhline(0, color=STONE2, lw=0.6, zorder=1)
        for e in (e for e in entries if e["effort"] == eff):
            code = LUNA_CODE[e["row"]]
            floor = code in LUNA_NAME
            family, color = (FLOORS, None) if floor else (NOTES, HARNESS_COLOR[e["harness"]])
            v, lo, hi = e["learned"]
            ax.plot((e["usd"], e["usd"]), (lo, hi), color=color or FAMILY_LOOK[family][0], lw=0.8, alpha=0.6, zorder=2,
                    solid_capstyle="butt")
            _point(ax, e["usd"], v, family, zorder=4, marker="o" if floor else TEST_LOOK[code][2]["marker"], color=color)
            if floor:
                _plane_name(ax, e["usd"], v, LUNA_NAME[code], color or FAMILY_LOOK[family][0], NAME_RIGHT)
        ax.set_xscale("log")
        ax.set_xlim(min(spend) / 1.6, max(spend) * 1.6)
        ax.set_ylim(low - 0.06, high + 0.06)
        ax.set_title(f"{eff} effort", loc="left", color=INK, fontsize=7.2)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("share learned on the stream", fontsize=6.6, color=STONE)
        _rule_axes(ax); _mono_ticks(ax)
    fig.text(0.54, 0.155, "model spend per instance of the stream, in US dollars (log scale)", ha="center", fontsize=6.6,
             color=STONE)
    _family_key(fig, [0.02, 0.9, 0.965, 0.08], LUNA_KEY, size=5.7, gap=5.0)
    _glossary(fig, 0.1, 0.045, 0.88, LUNA_GLOSSARY)
    out = FIG / "luna-cost.pdf"
    _save(fig, out)
    plt.close(fig)
    _report("luna-cost", [r for r in registered_runs() + ablation_runs() if r.track == "arc1" and r.mode == "workstreams"
                          and r.model.startswith("GPT-6 Luna") and str(r.seed) == str(TRACK_SEED)
                          and not r.spec.get("repeat")])
    return out


def table_tests_absolute() -> Path:
    """sections/tables/tests-absolute.tex, Table 10: every row of Table 1 in absolute terms, the budget used on each
    test and the share of the held-out test solved, the learner's own in normal type and its reset's beside it in small
    type, Qwen3.5-9B's then GLM-5.3 Flash's across, so that each share of Table 1 reads beside what it is a share of (a
    large share of a small spend is a small gain). A reset that uses more of a test's budget than the bare model, the
    player's stateless floor over all its plays, is marked with an asterisk, as Appendix J flags a test on which the
    reset plays below the bare model, and so is a learner that uses more of it than the bare model; the floors, whose
    reset is the bare model, are not."""
    from continual_arc.schedule import TEST_PHASE
    order = {g: i for i, g in enumerate((NOTES, TEACHER_NOTES, WEIGHTS, FLOORS))}
    rows = [r for r in TEST_ROWS if r.users == 1 and r.row not in APPENDIX_ROWS]
    registry = [r.row for r in TEST_ROWS]
    names = sorted({r.row for r in rows}, key=lambda row: (order[TEST_LOOK[row][1]], registry.index(row)))
    entries = {(r.player, r.row): _open_entry(r) for r in rows}
    floor = {p: next(r for r in rows if r.player == p and r.row == "f") for p in PLAYERS}
    bare = {(p, test): _absolute(reset_plays(floor[p], test), test, TEST_PHASE[test])["budget"]
            for p in PLAYERS for test in FROZEN_TESTS}
    # each player's best of each column (_bold_best), the row's own value and never its reset's
    columns = (("seen", "budget", False), ("held", "budget", False), ("held", "solved", True))
    strong: dict[tuple[str, str, int], bool] = {}
    for p in PLAYERS:
        found = [(row, entries.get((p, row))) for row in names]
        for k, (side, measure, higher) in enumerate(columns):
            for (row, _), flag in zip(found, _bold_best([e[side][measure] if e else None for _, e in found], higher, 1)):
                strong[p, row, k] = flag
    def cells(e, r):
        if e is None:
            return ["--"] * 3
        def pair(mine, ref, k, test=None):
            mark = "$^{*}$" if test and r.reset != floor[r.player].run and ref > bare[(r.player, test)] else ""
            own = "$^{*}$" if test and r.reset != floor[r.player].run and mine > bare[(r.player, test)] else ""
            value = "\\textbf{%.1f}" % mine if strong[r.player, r.row, k] else f"{mine:.1f}"
            return f"{value}{own} {{\\tiny {ref:.1f}{mark}}}"
        return [pair(e["seen"]["budget"], e["seen"]["reset_budget"], 0, "seen"),
                pair(e["held"]["budget"], e["held"]["reset_budget"], 1, "held-out"),
                pair(e["held"]["solved"], e["held"]["reset_solved"], 2)]
    lines, prev = [], None
    for row in names:
        group = TEST_LOOK[row][1]
        if group != prev:
            lines += ["\\midrule", f"\\multicolumn{{{1 + 3 * len(PLAYERS)}}}{{l}}{{\\emph{{{group}}}}} \\\\"]
            prev = group
        found = [(entries.get((p, row)), next((r for r in rows if r.player == p and r.row == row), None)) for p in PLAYERS]
        lines.append(f"{_row_names(row, [p for p in PLAYERS if (p, row) in entries])} & "
                     + " & ".join(c for e, r in found for c in cells(e, r)) + " \\\\")
    players = " & ".join(f"\\multicolumn{{3}}{{c}}{{{PLAYERS[p].split(' (')[0]}}}" for p in PLAYERS)
    rules = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(PLAYERS)))
    head = " & ".join(["$U_{\\mathrm{seen}}$ & $U_{\\mathrm{held}}$ & $P_{\\mathrm{held}}$"] * len(PLAYERS))
    path = TABLES / "tests-absolute.tex"
    path.write_text("% generated by scripts/make_figures.py tests-absolute; do not edit\n"
                    "\\begin{tabular}{l" + "r" * 3 * len(PLAYERS) + "}\n\\toprule\n"
                    f" & {players} \\\\\n{rules}\nmethod & {head} \\\\\n" + "\n".join(lines)
                    + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"{path.name}: {len(names)} rows")
    return path


def _weight_loss_rows(row: TestRow) -> dict:
    """The losses of each training phase's incoming adapter, read from the run's training records (runs/README.md):
    the token NLL of step 1, which the trainer logs before its optimizer update, and the held-out measure of step 0,
    which evaluates the incoming adapter. The training reading is one minibatch, not a full-set evaluation; the
    validation reading is over the phase's sampled games."""
    import hashlib
    folder = REPO / row.run
    digest = hashlib.sha256()

    def read(path: Path):
        raw = path.read_bytes()
        digest.update(str(path.relative_to(folder)).encode() + b"\0" + raw + b"\0")
        return json.loads(raw)

    splits = []
    for path in sorted((folder / "rounds").glob("r*/labels/split/split.json")):
        d = read(path)
        dev = d[f"dev-r{d['round']:04d}"]
        splits.append({"round": d["round"], "games": dev["games"], "calls": dev["examples"]})
    if [s["round"] for s in splits] != list(range(ROUNDS)):
        raise ValueError(f"{row.run}: incomplete validation-pool split records")

    phases = []
    for path in sorted((folder / "train").glob("p*/phase.json")):
        if not (path.parent / "DONE").exists():
            continue
        d, done = read(path), read(path.parent / "DONE")
        round_ = read(path.parent / "rounds.json")["rounds"] - 1
        first = next(s for s in d["steps"] if s["step"] == 1)
        held = next((s for s in d["held_out"] if s["step"] == 0), None)
        train, validation = first["nll_per_token"], held["loss"] if held else None
        if not all(math.isfinite(v) and v >= 0 for v in (train, validation) if v is not None):
            raise ValueError(f"{path}: invalid token NLL")
        phases.append({"round": round_, "training": train, "validation": validation,
                       "training_batch_calls": first["items"], "training_batch_tokens": first["tokens"],
                       "training_set": d["items"], "validation_subset": d["held_out_items"],
                       "selected_step": done["best_step"]})
    if not phases or [p["round"] for p in phases] != list(range(phases[0]["round"], ROUNDS)):
        raise ValueError(f"{row.run}: incomplete training phase records")
    return {"users": row.users, "run": row.run, "source_sha256": digest.hexdigest(), "phases": phases,
            "validation_pool_additions": [s for s in splits if s["calls"] > 0]}


def fig_weight_losses() -> Path:
    """figures/weight-losses.pdf, Figure 8: the training minibatch's and the sampled validation games' token NLL at
    each phase's incoming adapter, for DAgger at one, two and four users (_weight_loss_rows)."""
    runs = [_weight_loss_rows(r) for r in TEST_ROWS if r.player == "qwen" and r.row == "DAgger"]
    runs.sort(key=lambda r: r["users"])
    if [r["users"] for r in runs] != [1, 2, 4]:
        raise ValueError("weight-losses needs DAgger's runs at one, two and four users")
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.13, top=0.90, hspace=0.42)
    values = [p[k] for r in runs for p in r["phases"] for k in ("training", "validation") if p[k] is not None]
    lo, hi = min(values), max(values)
    bounds = (max(0.0, lo - 0.06 * (hi - lo)), hi + 0.06 * (hi - lo))
    for ax, run in zip(axes, runs):
        rows = run["phases"]
        xs = [p["round"] for p in rows]
        ax.plot(xs, [p["training"] for p in rows], color=STONE2, lw=0.85, label="training minibatch")
        ax.plot(xs, [p["validation"] for p in rows], color=PURPLE, lw=1.1, label="validation subset")
        ax.vlines([s["round"] for s in run["validation_pool_additions"]], -0.075, -0.025,
                  transform=ax.get_xaxis_transform(), color=STONE2, lw=0.65, clip_on=False)
        ax.set_xlim(0, ROUNDS - 1)
        ax.set_ylim(*bounds)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_title(f"{run['users']} user" + ("s" if run["users"] > 1 else ""), loc="left", fontsize=7.4, pad=5)
        last = rows[-1]
        ax.text(1, 1.065, f"final: train {last['training']:.3f}  |  val {last['validation']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontproperties=brand.mono(6.0), color=STONE)
        ax.grid(axis="y", color=brand.LINE, lw=0.45)
        ax.set_axisbelow(True)
        _rule_axes(ax); _mono_ticks(ax)
        print(f"  {run['users']} users: {len(rows)} phases; final training minibatch NLL {last['training']:.6f}, "
              f"validation subset NLL {last['validation']:.6f}; selected step {last['selected_step']}")
    axes[-1].set_xticks([0, 50, 100, 150, 199], ["0", "50", "100", "150", "199"])
    axes[-1].set_xlabel("round of the stream", fontsize=7)
    _mono_ticks(axes[-1])
    fig.text(0.015, 0.53, "negative log-likelihood per token", rotation=90, va="center", fontsize=7, color=STONE)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.28), ncol=2, fontsize=7, handlelength=2.0, borderaxespad=0)
    fig.text(0.105, 0.018, "Ticks below each panel: new validation games enter the pool", fontsize=6.3, color=STONE)
    out = FIG / "weight-losses.pdf"
    _save(fig, out)
    (FIG / "weight-losses.json").write_text(json.dumps({"measurement": "phase-start token NLL; training minibatch and validation subset",
                                                      "runs": runs}, indent=1) + "\n")
    sources = json.loads(SOURCES.read_text()) if SOURCES.exists() else {}
    sources["weight-losses"] = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "runs": [{"path": r["run"], "row": "DAgger", "model": "Qwen3.5-9B",
                                          "users": r["users"], "phases": len(r["phases"]), "complete": True,
                                          "source_sha256": r["source_sha256"]} for r in runs]}
    SOURCES.write_text(json.dumps(sources, indent=1) + "\n")
    return out


def _inference_price(tokens: dict, pricing: dict, cached_fraction: float) -> float:
    """What ``tokens`` cost at ``pricing``'s rates, ``cached_fraction`` of the input read from a cache; the output
    tokens already hold the reasoning tokens."""
    if not 0 <= cached_fraction <= 1:
        raise ValueError("cached input fraction must lie in [0, 1]")
    total = 0.0
    for model, counts in tokens.items():
        rate = pricing["models"][model]["rates"]
        input_rate = (1 - cached_fraction) * rate["input"] + cached_fraction * rate["cached_input"]
        total += (counts["input"] * input_rate + counts["output"] * rate["output"]) / 1e6
    return total


def fig_inference_cost() -> Path:
    """figures/inference-cost.pdf, Figure 13, and sections/tables/inference-pricing.tex, Table 17: Figure 3's learners'
    share learned on the stream against what their stream's tokens cost at the providers' prices (data/
    inference-pricing.json), the cache's share of the input as a range beside each point, apart from the interval over
    tasks."""
    from inference_costs import collect_figure3_inference
    entries, runs = collect_figure3_inference(sys.modules[__name__])
    pricing = json.loads((ROOT / "data" / "inference-pricing.json").read_text())
    cached = pricing["cached_input_fraction"]
    for entry in entries:
        entry["usd_per_instance"] = _inference_price(entry["tokens"], pricing, cached) / entry["instances"]
        entry["cache_sensitivity_usd_per_instance"] = sorted(
            _inference_price(entry["tokens"], pricing, fraction) / entry["instances"]
            for fraction in pricing["cache_sensitivity_fraction"])
        print(f"  {entry['name']}: ${entry['usd_per_instance']:.5f}/instance, "
              f"L={entry['learned'][0]:.3f}, cache range {entry['cache_sensitivity_usd_per_instance']}")
    fig, ax = plt.subplots(figsize=(WIDTH, 4.15))
    fig.subplots_adjust(left=0.115, right=0.97, bottom=0.18, top=0.88)
    ax.axhline(0, color=STONE2, lw=0.6, zorder=1)
    places = {"G": (7, 7, "left", False), "GE": (7, 7, "left", False),
              "QTE": (7, 18, "left", True), "QT": (7, -7, "left", False),
              "QETE": (7, 8, "left", False)}
    for e in entries:
        x, (y, lo, hi) = e["usd_per_instance"], e["learned"]
        color = FAMILY_LOOK[e["family"]][0]
        ax.plot((x, x), (lo, hi), color=color, lw=0.9, alpha=0.65, zorder=2)
        ax.plot(e["cache_sensitivity_usd_per_instance"], (y, y), color=color, lw=0.9, alpha=0.65, zorder=2)
        marker = "o" if e["family"] == FLOORS else TEST_LOOK[e["row"]][2]["marker"]
        _point(ax, x, y, e["family"], color=color, marker=marker)
        _plane_name(ax, x, y, e["name"], color, places.get(e["name"], NAME_RIGHT))
    spends = [x for e in entries for x in e["cache_sensitivity_usd_per_instance"]]
    low, high = min(e["learned"][1] for e in entries), max(e["learned"][2] for e in entries)
    ax.set_xscale("log")
    ax.set_xlim(min(spends) / 1.55, max(spends) * 1.55)
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1], ["0.005", "0.01", "0.02", "0.05", "0.10"])
    ax.set_ylim(low - 0.06, high + 0.07)
    ax.set_ylabel("share learned on the stream", fontsize=7, color=STONE)
    ax.set_xlabel("inference cost per stream instance, US dollars (log scale)", fontsize=7, color=STONE, labelpad=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color=brand.LINE, lw=0.45)
    ax.set_axisbelow(True)
    _rule_axes(ax); _mono_ticks(ax)
    _family_key(fig, [0.02, 0.94, 0.965, 0.044], PLANE_KEY, size=5.7, gap=5.0)
    fig.text(0.115, 0.045, "Common token prices; points assume 70% cached input for both models", fontsize=6.1, color=STONE)
    out = FIG / "inference-cost.pdf"
    _save(fig, out)
    plt.close(fig)
    (FIG / "inference-cost.json").write_text(json.dumps({"pricing": pricing, "entries": entries}, indent=1) + "\n")
    table = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Model & Provider & Input & Cached input & Output \\", r"\midrule"]
    for model in ("qwen", "glm"):
        p = pricing["models"][model]
        rate = p["rates"]
        provider = "PrimeIntellect" if model == "qwen" else "Z.ai"
        table.append(f"{p['model']} & {provider} & {rate['input']:.3f} & {rate['cached_input']:.3f} & {rate['output']:.3f} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "inference-pricing.tex").write_text("\n".join(table) + "\n")
    _report("inference-cost", runs)
    return out


# Every figure and table this module makes, by name, in the paper's order. weight_updates.py makes Figure 7,
# forgetting.py Figure 9 and experiment_costs.py Table 18.
GENERATORS = {
    "protocol": fig_protocol,                                   # Figure 1
    "modes": fig_modes,                                         # Figure 2
    "tests": fig_tests,                                         # Figure 3
    "notes-weights": fig_notes_weights,                         # Figure 4
    "plane-luna": fig_plane_luna,                               # Figure 5
    "taskid": fig_taskid,                                       # Figure 6
    "weight-losses": fig_weight_losses,                         # Figure 8
    "luna-cost": fig_luna_cost,                                 # Figure 10
    "tests-arc2": functools.partial(fig_tests_track, "arc2"),   # Figure 11
    "tests-arc3": functools.partial(fig_tests_track, "arc3"),   # Figure 12
    "inference-cost": fig_inference_cost,                       # Figure 13 and Table 17
    "tests-table": table_tests,                                 # Tables 1, 6 and 11, and chance-numbers.tex
    "ablations": table_ablations,                               # Table 7
    "solved-budget-table": table_solved_budget,                 # Table 9 and solved-numbers.tex
    "tests-absolute": table_tests_absolute,                     # Table 10
    "tests-luna": table_luna_tests,                             # Table 12
    "status": table_status,                                     # status.tex
}


def set_out(out: Path) -> None:
    """Write the figures to ``out``/figures/ and the tables to ``out``/sections/tables/."""
    global OUT, FIG, TABLES, SOURCES
    OUT = out.resolve()
    FIG, TABLES = OUT / "figures", OUT / "sections" / "tables"
    SOURCES = FIG / "SOURCES.json"
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"what to make; default all of: {', '.join(GENERATORS)}")
    ap.add_argument("--out", type=Path, default=OUT, help="write figures/ and sections/tables/ here (default build/)")
    args = ap.parse_args(argv)
    unknown = [n for n in args.names if n not in GENERATORS]
    if unknown:
        ap.error(f"unknown: {', '.join(unknown)}")
    set_out(args.out)
    for name in args.names or list(GENERATORS):
        print(name, "->", GENERATORS[name]())


if __name__ == "__main__":
    main()
