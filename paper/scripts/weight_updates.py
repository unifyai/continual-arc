"""figures/weight-updates.pdf, Figure 7: how often DAgger's training phases changed the student's weights at one, two
and four users (Appendix H). A phase keeps the adapter at its best held-out measure, so it changed the weights exactly
when that best came after its first measure, or when it had no held-out games yet and so kept its last step. Above, a
dot at each round whose phase changed the weights, on a rule over the rounds that had a phase; below, how many had by
each round.

It reads each run's training records in runs/ (train/p<phase>/, runs/README.md), records the runs in
build/figures/SOURCES.json, and prints what Appendix H says of them: each run's count, its count per 50 rounds, and at
one user how far the held-out games' negative log-likelihood per token sat above that of the labels the adapter
trained on (a phase's first five steps), per 50 rounds.

    uv run --extra plots python scripts/weight_updates.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brand  # noqa: E402
import make_figures as mf  # noqa: E402

# The DAgger rows the paper reports, by the number of users served at once.
RUNS = {r.users: r.run for r in mf.TEST_ROWS if r.row == "DAgger"}
# Every run is DAgger, so each is drawn in the trained weights' purple, a triangle at each round whose phase changed the
# weights, as the figure's caption has it, and the number of users is told by position and label rather than by a
# shade, which in the paper's figures means guidance added.
LOOK = dict(mf.TEST_LOOK["DAgger"][2], marker="^")
ROUNDS = 200
BLOCK = 50


def phases(folder: Path) -> list[dict]:
    """Every finished phase that trained, in order: the rounds played before it, whether it changed the weights, and
    its first held-out measure beside the training labels' loss over its first five steps. A phase that never trained
    (before the base model had played 24 games) has no phase.json."""
    rows = []
    for d in sorted(folder.glob("p0*")):
        if not (d / "DONE").exists() or not (d / "phase.json").exists():
            continue
        phase = json.loads((d / "phase.json").read_text())
        held, first = phase["held_out"], phase["steps"][:5]
        best = min(held, key=lambda h: h["loss"])["step"] if held else None
        trained = sum(s["nll_per_token"] * s["tokens"] for s in first) / sum(s["tokens"] for s in first)
        rows.append({"rounds": json.loads((d / "rounds.json").read_text())["rounds"], "changed": best != 0,
                     "gap": held[0]["loss"] - trained if held else None})
    return rows


def label(users: int) -> str:
    return f"{users} user" + "s" * (users > 1)


def draw(runs: dict[int, list[dict]], out: Path) -> None:
    brand.apply()
    fig = plt.figure(figsize=(5.5, 3.1))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.28, left=0.13, right=0.83, bottom=0.14,
                            top=0.95)
    top, bottom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for y, users in enumerate(sorted(runs, reverse=True)):
        rows = runs[users]
        changed = [r["rounds"] - 1 for r in rows if r["changed"]]
        top.plot([rows[0]["rounds"] - 1, rows[-1]["rounds"] - 1], [y, y], color=brand.LINE_STRONG, lw=0.6,
                 solid_capstyle="butt", zorder=1)
        top.scatter(changed, [y] * len(changed), s=11, marker=LOOK["marker"], color=mf.PURPLE, edgecolors="white",
                    linewidths=0.4, zorder=3)
        top.text(-6, y, label(users), ha="right", va="center", fontsize=7, color=brand.INK)
        top.text(ROUNDS + 5, y, f"{len(changed)} of {len(rows)} ({100 * len(changed) / len(rows):.0f}%)", ha="left",
                 va="center", fontproperties=brand.mono(6.5), color=brand.GRAPHITE)
    top.set_xlim(-2, ROUNDS + 1)
    top.set_ylim(-0.6, len(runs) - 0.4)
    top.set_yticks([])
    top.spines["left"].set_visible(False)
    top.spines["bottom"].set_color(brand.STONE2)
    top.tick_params(axis="x", labelbottom=False, length=2.5, width=0.6, colors=brand.STONE)
    for users in sorted(runs):
        rows = runs[users]
        xs, ys = [rows[0]["rounds"] - 1], [0]
        for r in rows:
            xs.append(r["rounds"] - 1)
            ys.append(ys[-1] + r["changed"])
        bottom.step(xs, ys, where="post", color=mf.PURPLE, ls=LOOK["ls"], lw=1.2, zorder=3)
        # a run whose phases stop short of the stream's end says how far they got
        end = f"\n(to round {xs[-1]})" if rows[-1]["rounds"] < ROUNDS else ""
        bottom.text(ROUNDS + 5, ys[-1], label(users) + end, ha="left", va="center", fontsize=6.5, color=brand.INK,
                    linespacing=1.1)
    bottom.set_xlim(-2, ROUNDS + 1)
    bottom.set_ylim(0, None)
    bottom.set_xlabel("round of the stream")
    bottom.set_ylabel("phases that changed\nthe weights, so far")
    bottom.tick_params(length=2.5, width=0.6, colors=brand.STONE, labelcolor=brand.STONE)
    for side in ("left", "bottom"):
        bottom.spines[side].set_color(brand.STONE2)
    for tick in bottom.get_xticklabels() + bottom.get_yticklabels():
        tick.set_fontproperties(brand.mono(6.5))
    bottom.grid(axis="y", color=brand.LINE, lw=0.5)
    bottom.set_axisbelow(True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=mf.OUT, help="write figures/ here (default build/)")
    mf.set_out(ap.parse_args().out)
    runs = {users: phases(mf.REPO / RUNS[users] / "train") for users in sorted(RUNS)}
    draw(runs, mf.FIG / "weight-updates.pdf")
    for users, rows in runs.items():
        changed = sum(r["changed"] for r in rows)
        state = "complete" if rows[-1]["rounds"] == ROUNDS else f"to round {rows[-1]['rounds'] - 1}"
        blocks = [[r for r in rows if lo < r["rounds"] <= lo + BLOCK] for lo in range(0, ROUNDS, BLOCK)]
        per = ", ".join(f"{sum(r['changed'] for r in b)} of {len(b)}" for b in blocks)
        print(f"{label(users)}: {changed} of {len(rows)} phases changed the weights "
              f"({100 * changed / len(rows):.0f}%; {state}); per {BLOCK} rounds: {per}")
        if users == 1:
            gaps = [[r["gap"] for r in b if r["gap"] is not None] for b in blocks]
            print("  held-out minus trained-on loss per token, per 50 rounds: "
                  + ", ".join(f"{sum(g) / len(g):.3f}" for g in gaps if g))
    sources = json.loads(mf.SOURCES.read_text()) if mf.SOURCES.exists() else {}
    sources["weight-updates"] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": [{"path": RUNS[users], "row": "DAgger", "model": "Qwen3.5-9B", "users": users,
                  "phases": len(rows), "changed": sum(r["changed"] for r in rows),
                  "complete": rows[-1]["rounds"] == ROUNDS} for users, rows in runs.items()]}
    mf.SOURCES.write_text(json.dumps(sources, indent=1) + "\n")


if __name__ == "__main__":
    main()
