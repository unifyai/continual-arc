"""figures/forgetting.pdf, Figure 9: forgetting on the seen test, whether a row saves less on the tasks its stream left
longest ago.

Every task of the stream is visited 8 times at any number of users, in a burst whose span differs from task to task:
at one user the first task's visits end at round 44 and the second's at round 160. So a task's last visit gives how
long it went unpracticed before the frozen tests, with practice held equal, and its first visit how long it had been
known. A task's saving is read against the row's reset on the same task, from make_figures' own cells (each side's
charge on the task's instances, each the mean of its plays), so that the task's difficulty cancels.

For every row of Table 1, and DAgger's at two and four users, whose share saved on the seen test is above zero, this
reads that share over the tasks split into halves and into quarters by their last visit, and again by their first: the
newest group less the oldest, with its 95 percent interval over task resamples, each group resampled apart. It also
gives the slope of each task's share saved on its last, first and mean visit, per 100 rounds, weighted by the reset's
charge so that it agrees with the share over a group, with its interval over task resamples (Appendix H, "Forgetting").
The figure draws each task's share saved against its last visit for DAgger and the two demo caches.

    uv run --extra plots python scripts/forgetting.py [--out DIR]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

PAPER = ("QDE", "QDE(2)", "QDE(4)", "QC", "GC")   # the figure's rows: DAgger, and the demo caches, which cannot forget
REPS = mf.LEARNED_REPS


def visits(row: mf.TestRow) -> dict[str, list[int]]:
    """Each task's rounds in the row's own stream."""
    out = defaultdict(list)
    for line in (mf.REPO / row.run / "records.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["task_id"]].append(r["round"])
    return dict(out)


def name(row: mf.TestRow) -> str:
    return mf.row_name(row.row, row.player) + (f"({row.users})" if row.users > 1 else "")


def resample(rng: random.Random, tasks: list[str]) -> list[str]:
    return [tasks[rng.randrange(len(tasks))] for _ in tasks]


def interval(draws: list[float]) -> tuple[float, float]:
    draws = sorted(draws)
    return draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]


def gap(cells: dict, old: list[str], new: list[str]) -> tuple[float, float, float, float, float]:
    """The oldest group's share saved, the newest's, the newest less the oldest, and that difference's interval."""
    rng = random.Random(0)
    draws = [mf._share_of(cells, resample(rng, new)) - mf._share_of(cells, resample(rng, old)) for _ in range(REPS)]
    a, b = mf._share_of(cells, old), mf._share_of(cells, new)
    return (a, b, b - a, *interval(draws))


def slope(cells: dict, x: dict[str, float], tasks: list[str]) -> float:
    """Each task's share saved on x, per 100 rounds, weighted by the reset's charge on the task. ``tasks`` is a list,
    a resample's repeats included, so every sum runs over it."""
    total = sum(cells[t][1] for t in tasks)
    mx = sum(cells[t][1] * x[t] for t in tasks) / total
    my = sum(cells[t][1] - cells[t][0] for t in tasks) / total   # the weighted mean of 1 - own / reset
    sxx = sum(cells[t][1] * (x[t] - mx) ** 2 for t in tasks)
    sxy = sum((x[t] - mx) * (cells[t][1] - cells[t][0] - cells[t][1] * my) for t in tasks)
    return 100 * sxy / sxx


def slope_ci(cells: dict, x: dict[str, float], tasks: list[str]) -> tuple[float, float, float]:
    rng = random.Random(0)
    draws = []
    while len(draws) < REPS:
        sample = resample(rng, tasks)
        if len({x[t] for t in sample}) > 1:   # a resample of one visit round has no slope
            draws.append(slope(cells, x, sample))
    return (slope(cells, x, tasks), *interval(draws))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=mf.OUT, help="write figures/ here (default build/)")
    mf.set_out(ap.parse_args().out)
    rows = [r for r in mf.TEST_ROWS if r.users == 1 or r.row == "DAgger"]
    read = []
    print("Forgetting on the seen test: each row's share saved against its reset, over tasks split by their visits\n")
    print("Rows whose share saved on the seen test is above zero (pointwise 95 percent interval), with it:")
    for row in rows:
        s = mf.test_share(row, "seen")
        if s is None or row.reset == row.run:   # a row that keeps nothing is read half its plays against the other half
            continue
        lo, hi = s["ci95"]
        if lo > 0:
            read.append((row, s))
            print(f"  {name(row):8s} S_seen {s['share']:+.2f} [{lo:+.2f}, {hi:+.2f}] over {s['tasks']} tasks")
    figure = []
    for row, s in read:
        cells = s["_cells"]
        v = visits(row)
        tasks = sorted(t for t in cells if cells[t][1] > 0)
        assert set(tasks) <= set(v), f"{row.run}: seen-test tasks its stream never served"
        assert len({len(v[t]) for t in tasks}) == 1, f"{row.run}: tasks visited unequally"
        first = {t: min(v[t]) for t in tasks}
        last = {t: max(v[t]) for t in tasks}
        mean = {t: sum(v[t]) / len(v[t]) for t in tasks}
        print(f"\n{name(row)} ({row.run}, reset {row.reset}): {len(tasks)} tasks, {len(v[tasks[0]])} visits each, "
              f"last visits rounds {min(last.values())} to {max(last.values())}, first {min(first.values())} to "
              f"{max(first.values())}")
        for label, key in (("last visit", last), ("first visit", first)):
            order = sorted(tasks, key=lambda t: (key[t], first[t], t))
            n = len(order)
            for split, k in (("halves", n // 2), ("quarters", n // 4)):
                old, new = order[:k], order[n - (n - k if split == "halves" else k):]
                a, b, d, dlo, dhi = gap(cells, old, new)
                print(f"  by {label:11s} {split:8s} oldest {len(old):3d} tasks (rounds {key[old[0]]:3.0f}-"
                      f"{key[old[-1]]:3.0f}) {a:+.2f}, newest {len(new):3d} (rounds {key[new[0]]:3.0f}-"
                      f"{key[new[-1]]:3.0f}) {b:+.2f}; newest less oldest {d:+.2f} [{dlo:+.2f}, {dhi:+.2f}]")
        for label, key in (("last", last), ("first", first), ("mean", mean)):
            b, blo, bhi = slope_ci(cells, key, tasks)
            print(f"  slope on the {label:5s} visit: {b:+.3f} per 100 rounds [{blo:+.3f}, {bhi:+.3f}]")
        figure.append((row, cells, tasks, last, first))
    draw([f for f in figure if name(f[0]) in PAPER], mf.FIG / "forgetting.pdf")


def draw(figure: list, path: Path) -> None:
    """Each task's share saved against its last visit, a panel per row, the dot's area the reset's charge on the task
    and the line the slope weighted by it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(figure), figsize=(5.5, 1.75), sharey=True, squeeze=False)
    for j, (row, cells, tasks, last, _) in enumerate(figure):
        color = mf.ROW_COLOR[row.player].get(row.row, mf.DARK_GRAY)
        mean_reset = sum(cells[u][1] for u in tasks) / len(tasks)
        ax = axes[0][j]
        xs = [last[t] for t in tasks]
        ax.scatter(xs, [1 - cells[t][0] / cells[t][1] for t in tasks], s=[8 * cells[t][1] / mean_reset for t in tasks],
                   color=color, alpha=0.7, linewidth=0)
        b = slope(cells, last, tasks) / 100
        total = sum(cells[t][1] for t in tasks)
        mx = sum(cells[t][1] * last[t] for t in tasks) / total
        my = 1 - sum(cells[t][0] for t in tasks) / total
        lo, hi = min(xs), max(xs)
        ax.plot([lo, hi], [my + b * (lo - mx), my + b * (hi - mx)], color=color, lw=1.2)
        ax.axhline(0, color=mf.PALE_GRAY, lw=0.6, zorder=0)
        ax.set_xlim(-5, 204)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([0, 100, 200])
        ax.tick_params(labelsize=6)
        ax.set_title(name(row), fontsize=7)
        ax.set_xlabel("round of last visit", fontsize=6)
        if j == 0:
            ax.set_ylabel("share saved", fontsize=6)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    mf._save(fig, path)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
