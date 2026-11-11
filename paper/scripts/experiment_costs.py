"""sections/tables/experiment-costs.tex, Table 18: the API charges the paper's run families recorded and an estimate of
DAgger's training compute, not a cloud invoice; and figures/experiment-costs.json, the manifest of every source it
sums, which the table is rendered from.

A play's API charge is the sum of its records' llm_cost_usd, read from runs/ for every play there: each run the open
models' rows on ARC-AGI-1 read (make_figures.TEST_ROWS, each row, its reset, its twin and the resets it pools) and the
page alone, their streams and every play of the frozen tests with the games replayed after a context overrun; each
GPT-6 Luna run of runs.json with its plays of the tests; and Kimi K3.1 Pro's rows on the second and third tracks
(make_figures.TRACK_ROWS) with theirs. A run several rows read is counted once. Each play is counted whole, the games
an outage played through and the learning-on epilogues included, since they spent what they recorded. What runs/ does
not hold is read from data/experiment-costs.json: the plays the measures leave out, the private set's calibration,
the charges of DAgger's teacher labels, and the corrections and the price behind its training estimate. A training
phase's GPU-hours are its minutes (DONE) times the GPUs it ended on (phase.json's args), plus the GPU-hours a resumed
phase ran on more GPUs before giving some up.

    uv run --extra plots python scripts/experiment_costs.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

DATA = json.loads((mf.ROOT / "data" / "experiment-costs.json").read_text())
GROUPS = {
    "open": "ARC-AGI-1, Qwen and GLM",
    "luna": "ARC-AGI-1, Luna (three efforts and orders)",
    "arc2": "ARC-AGI-2, Kimi methods",
    "arc3": "ARC-AGI-3, Kimi methods",
    "calibration": "Private-set and ARC-AGI-2 calibration",
}
PAGE_ALONE = "runs/arc1/Q0-prime-agent-qwen3.5-9b-ws0-u1-page-alone"   # Appendix H's untrained student given the page
# a play of a frozen test the measures read or leave out, or its games played again after a context overrun
PLAY = re.compile(r"(?:seen|held-out)(?:-reference)?(?:-\d+)?(?:\.replay)?")


def read(path: Path):
    return json.loads(path.read_text())


def manifest() -> dict[str, str]:
    """Every play in runs/ the table sums, by its folder, with its group."""
    plays: dict[str, str] = {}

    def add(play: Path, group: str) -> None:
        if (play / "records.jsonl").exists():
            key = str(play.relative_to(mf.REPO))
            if plays.setdefault(key, group) != group:
                raise SystemExit(f"{key} is in two groups, {plays[key]} and {group}")

    runs = {v for r in mf.TEST_ROWS for v in (r.run, r.reset, r.twin, *r.pooled)} | {PAGE_ALONE}
    for run in sorted(runs):
        add(mf.REPO / run, "open")
        for frozen in sorted((mf.REPO / run / "frozen").glob("*/play")):
            if PLAY.fullmatch(frozen.parent.name):
                add(frozen, "open")
    systems = {(row, harness or "") for row, harness, _, _ in mf.LUNA_TEST_SYSTEMS}
    for entry in mf.RUNS:
        harness = entry.get("harness") or ("prime-agent" if entry["row"].startswith("H") else "")
        if (entry["row"], harness) not in systems:
            continue
        base = mf.REPO / entry["path"]
        add(base, "luna")
        for test in mf.FROZEN_TESTS:
            for reference in (False, True):
                for name in mf._play_names(base / "frozen", test, reference):
                    add(base / "frozen" / name / "play", "luna")
    for track, rows in mf.TRACK_ROWS.items():
        for run in rows.values():
            add(mf.REPO / run, track)
            for play in sorted((mf.REPO / run / "frozen").glob("*-p*/play")):
                add(play, track)
    return plays


def record_costs(plays: dict[str, str]) -> list[dict]:
    """Each play's records and the sum of the charges they recorded, with the sha256 of its records.jsonl."""
    out = []
    for key, group in sorted(plays.items()):
        raw = (mf.REPO / key / "records.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        cost = sum((row.get("resources") or {}).get("llm_cost_usd") or 0 for row in rows)
        missing = sum("llm_cost_usd" not in (row.get("resources") or {}) for row in rows)
        out.append({"path": key, "group": group, "records": len(rows), "llm_cost_usd": cost,
                    "missing_cost_records": missing, "records_sha256": hashlib.sha256(raw).hexdigest()})
    return out


def training_costs() -> list[dict]:
    """Each DAgger run's teacher-label charges and its training GPU-hours, at DATA's price."""
    out = []
    for row in (r for r in mf.TEST_ROWS if r.row == "DAgger"):
        phases = [(read(p.parent / "DONE"), read(p)) for p in sorted((mf.REPO / row.run / "train").glob("p*/phase.json"))
                  if (p.parent / "DONE").exists()]
        recorded = sum(done["minutes"] * phase["args"]["gpus"] / 60 for done, phase in phases)
        given = DATA["training"]["runs"][row.run]
        hours = recorded + sum(c["extra_gpu_hours"] for c in given["resume_corrections"])
        out.append({"run": row.run, "users": row.users, "phases": len(phases),
                    "phase_hours": sum(done["minutes"] / 60 for done, _ in phases),
                    "gpu_hours_duration_times_final_allocation": recorded,
                    "resume_corrections": given["resume_corrections"], "gpu_hours_approx": hours,
                    "gpu_usd_per_hour": DATA["gpu_usd_per_hour"], "gpu_usd_approx": hours * DATA["gpu_usd_per_hour"],
                    "teacher_label_usd_recorded": given["teacher_label_usd"], "label_files": given["label_files"],
                    "unpriced_labels": given["unpriced_labels"]})
    return out


def render_table(document: dict) -> str:
    """The table, from its manifest alone."""
    totals, training = document["groups"], document["training"]
    api, gpu = document["recorded_api_usd"], document["training_gpu_hours_approx"]
    lines = ["% generated by scripts/experiment_costs.py; do not edit",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Experiment group & Recorded API (\$) & H100 hours & Est. training (\$) \\", r"\midrule"]
    for key, label in GROUPS.items():
        lines.append(f"{label} & {totals[key]['llm_cost_usd']:,.2f} & -- & -- " + r"\\")
    for row in training:
        prefix = r"$\geq$" if row["unpriced_labels"] else ""
        lines.append(f"QDE, {row['users']} user{'s' if row['users'] > 1 else ''}: labels and training & "
                     f"{prefix}{row['teacher_label_usd_recorded']:,.2f} & {row['gpu_hours_approx']:,.1f} & "
                     f"{row['gpu_usd_approx']:,.0f} " + r"\\")
    lines += [r"\midrule", f"Subtotal & {api:,.2f} & {gpu:,.1f} & {document['training_gpu_usd_approx']:,.0f} " + r"\\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=mf.OUT, help="write figures/ and sections/tables/ here (default build/)")
    mf.set_out(ap.parse_args().out)
    sources = record_costs(manifest())
    left_out = DATA["left_out"]["plays"]
    if any((mf.REPO / key).exists() for key in left_out):
        raise SystemExit("a play data/experiment-costs.json gives as left out of runs/ is in runs/")
    sources += [{"path": key, "group": "open", **given} for key, given in left_out.items()]
    sources.append({"path": "the private set's calibration reports", "group": "calibration",
                    **{k: DATA["calibration"][k] for k in ("records", "llm_cost_usd")}})
    training = training_costs()
    totals: dict[str, Counter] = defaultdict(Counter)
    for source in sources:
        totals[source["group"]]["records"] += source["records"]
        totals[source["group"]]["llm_cost_usd"] += source["llm_cost_usd"]
        totals[source["group"]]["sources"] += 1
    api = sum(t["llm_cost_usd"] for t in totals.values()) + sum(r["teacher_label_usd_recorded"] for r in training)
    gpu = sum(r["gpu_hours_approx"] for r in training)
    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "The final paper run families, their controls, frozen repeats, the GPT-6 Luna order comparisons and "
                 "learning-on epilogues, and the reported calibrations. A run several rows read is counted once.",
        "excluded": ["Model calls that recorded no charge, the teacher's moments it failed to label among them",
                     "Calls between instances, which no instance's record holds",
                     "GPU serving, idle allocation, startup and work before a saved checkpoint, and CPU, storage and "
                     "network",
                     "The search that chose DAgger's configuration, the other weight-based methods, exploratory "
                     "probes, the private set's authoring and follow-on runs",
                     "ARC-AGI-3's first-sight probes and the pilot of the program library and the expert guidance"],
        "accounting": {"api": "The sum of each record's resources.llm_cost_usd, the teacher's labels apart; a "
                              "harness's own system_cost_usd is not added.",
                       "gpu": "Completed phases' minutes times their final GPU count, plus each resumed phase's "
                              f"minutes on more GPUs, at ${DATA['gpu_usd_per_hour']:.2f} per H100-hour; an estimate, "
                              "not an invoice.",
                       "teacher_labels": "Every label file's recorded charge in the final runs."},
        "groups": {k: dict(v) for k, v in totals.items()}, "training": training, "sources": sources,
        "recorded_api_usd": api, "training_gpu_hours_approx": gpu,
        "training_gpu_usd_approx": gpu * DATA["gpu_usd_per_hour"],
        "recorded_api_plus_estimated_training_usd": api + gpu * DATA["gpu_usd_per_hour"],
    }
    (mf.FIG / "experiment-costs.json").write_text(json.dumps(document, indent=2) + "\n")
    (mf.TABLES / "experiment-costs.tex").write_text(render_table(document))
    print(json.dumps({k: v for k, v in document.items() if k not in ("sources", "training")}, indent=2))


if __name__ == "__main__":
    main()
