"""Human-readable reports and optional plots from a run summary."""

from __future__ import annotations

from pathlib import Path


def _fmt(x, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _table(headers: list[str], rows: list[list], nd: int = 2) -> str:
    cells = [[_fmt(c, nd) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in cells:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))
    return "\n".join(out)


def protocol_line(protocol: dict | None) -> str:
    """One line naming the track and saying whether the run is the canonical protocol."""
    if not protocol:
        return "protocol: unknown"
    head = (f"track: {protocol.get('track')}   protocol: {protocol.get('config')}/{protocol.get('mode')} "
            f"seed {protocol.get('seed')}" + ("" if protocol.get("epilogue") else " (no epilogue)"))
    if protocol.get("canonical"):
        return head + "   canonical: yes"
    return head + "   canonical: NO (" + "; ".join(protocol.get("deviations", [])) + ")"


def text_report(summary: dict, title: str = "Continual-ARC run") -> str:
    parts: list[str] = [title, "=" * len(title), ""]
    parts.append(protocol_line(summary.get("protocol")))
    ov = summary["overall"]
    parts.append(f"score: {_fmt(ov['score'])} / 100   (mean cost {_fmt(ov['mean_cost'])} of max {_fmt(summary['max_cost'])})")
    parts.append(
        f"instances: {ov['n']}   solve rate: {_fmt(ov['solve_rate'])}   "
        f"wrong attempts/instance: {_fmt(ov['mean_wrong_attempts'])}   demo requests/instance: {_fmt(ov['mean_demo_requests'])}"
    )
    parts.append("")
    parts.append("By phase")
    rows = [
        [phase, s["n"], s["score"], s["mean_cost"], s["solve_rate"], s["mean_wrong_attempts"], s["mean_demo_requests"], s["give_up_rate"]]
        for phase, s in summary["by_phase"].items()
    ]
    parts.append(_table(["phase", "n", "score", "cost", "solved", "wrong", "demos", "gave_up"], rows))
    parts.append("")
    acq = summary["acquisition"]
    parts.append(f"Acquisition (first exposure, stream phases): n={acq['n']} cost={_fmt(acq['mean_cost'])} solved={_fmt(acq['solve_rate'])}")
    parts.append("")
    parts.append("Retention by gap since last exposure (stream phases, re-exposures only)")
    rows = [[r["gap"], r["n"], r["mean_cost"], r["solve_rate"], r["mean_demo_requests"]] for r in summary["retention_by_gap"]]
    parts.append(_table(["gap", "n", "cost", "solved", "demos"], rows))
    parts.append("")
    parts.append("Cost by exposure index (stream phases)")
    rows = [[r["exposure"], r["n"], r["mean_cost"], r["solve_rate"], r["mean_demo_requests"]] for r in summary["cost_by_exposure"]]
    parts.append(_table(["exposure", "n", "cost", "solved", "demos"], rows))
    if summary.get("resources"):
        parts.append("")
        parts.append("Resources (logged, not scored)")
        rows = [[k, v["total"], v["mean"]] for k, v in summary["resources"].items()]
        parts.append(_table(["resource", "total", "per instance"], rows, nd=3))
    return "\n".join(parts) + "\n"


def transfer_report(transfer: dict) -> str:
    lines = [
        "Forward transfer on the held-out epilogue",
        "-----------------------------------------",
        f"matched instances: {transfer['n_matched']}",
        f"run mean cost:       {_fmt(transfer['run_mean_cost'])}",
        f"reference mean cost: {_fmt(transfer['reference_mean_cost'])}",
        f"mean transfer (reference - run): {_fmt(transfer['mean_transfer'])}  "
        f"({_fmt(transfer['transfer_score_points'])} score points)",
    ]
    return "\n".join(lines) + "\n"


def learned_report(learned: dict) -> str:
    lines = [
        "Share learned on the stream, against the demo-cache floor",
        "---------------------------------------------------------",
        f"matched instances: {learned['n']}",
    ]
    if not learned["n"]:
        return "\n".join(lines + ["no slot of the run is in the floor's records"]) + "\n"
    lines += [
        f"budget used:      run {_fmt(learned['run_budget'], 1)}   floor {_fmt(learned['floor_budget'], 1)}   "
        f"perfect memory {_fmt(learned['perfect_memory_budget'], 1)}",
        f"learnable budget: {_fmt(learned['learnable_budget'], 1)} points (the floor's budget less perfect memory's)",
        f"saved beyond the cache: {_fmt(learned['lift'], 1)} points",
    ]
    if learned["share_learned"] is None:
        lines.append("share learned: undefined, since the floor leaves nothing to learn on this model")
    else:
        ci = learned["ci95"]
        span = f"  (95% interval {_fmt(ci[0])} to {_fmt(ci[1])}, {learned['reps']} task resamples)" if ci else ""
        lines.append(f"share learned: {_fmt(learned['share_learned'])}{span}")
    return "\n".join(lines) + "\n"


def plot(summary: dict, out_dir: str | Path, prefix: str = "") -> list[Path]:
    """Write learning-curve, retention and exposure plots. Requires matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("plots need matplotlib: pip install 'continual-arc[plots]'") from e

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    lc = summary["learning_curve"]
    if lc:
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = [p["step"] for p in lc]
        ax.plot(steps, [p["cumulative_mean_cost"] for p in lc], label="cumulative mean cost")
        ax.plot(steps, [p["window_mean_cost"] for p in lc], label="windowed mean cost", alpha=0.7)
        phase_starts = {}
        for p in lc:
            phase_starts.setdefault(p["phase"], p["step"])
        for phase, s in list(phase_starts.items())[1:]:
            ax.axvline(s, color="grey", linestyle=":", linewidth=1)
            ax.text(s, ax.get_ylim()[1], phase, fontsize=8, va="top")
        ax.set_xlabel("instance")
        ax.set_ylabel("cost")
        ax.set_title("Learning curve (lower is better)")
        ax.legend()
        fig.tight_layout()
        path = out / f"{prefix}learning_curve.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    for key, xkey, title, fname in (
        ("retention_by_gap", "gap", "Retention: cost on re-exposure vs gap", "retention.png"),
        ("cost_by_exposure", "exposure", "Cost vs exposure index", "exposure.png"),
    ):
        rows = [r for r in summary[key] if r["n"] > 0]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([r[xkey] for r in rows], [r["mean_cost"] for r in rows])
        for i, r in enumerate(rows):
            ax.text(i, r["mean_cost"], f"n={r['n']}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel(xkey)
        ax.set_ylabel("mean cost")
        ax.set_title(title)
        fig.tight_layout()
        path = out / f"{prefix}{fname}"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written
