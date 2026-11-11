"""The numbers of Tables 15 and 16 (Appendix K), Kimi K3.1 Pro's rows on the second and third tracks, read from runs/
as make_figures' track code reads them for Figures 11 and 12.

Table 15: each row's share of its stream solved, P, in percent; its share learned on the stream, L, against its reset
(make_figures.TRACK_RESET), 0 for a row that keeps nothing; and its share saved on each frozen test, S_seen and
S_held, over every whole play of each side (make_figures._track_entry), a row that keeps nothing read with its plays
taken alternately, half against the other half; each with its 95 percent interval over task resamples.

Table 16: on ARC-AGI-3's stream, 120 instances, a first visit and seven returns of each of its 15 games, each row's
share of instances solved and its mean charge out of 8, overall, on the first visits (how many of the 15 it solved)
and on the returns, the pilot of the program library (KP) and the expert guidance (KE) beside Table 15's rows, every
record as its run wrote it.

    uv run --extra plots python scripts/tracks_tables.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

PILOT = {"m": "runs/arc3/KP-kimi-k3.1-pro-ws0-u1-pilot", "hpl": "runs/arc3/KE-prime-agent-kimi-k3.1-pro-ws0-u1-pilot"}
PILOT_ORDER = ("f", "dc", "m", "h", "h0", "ha", "hpl")   # Table 16's order


def name(row: str) -> str:
    return mf._tex_name(mf.TRACK_PLAYER[0] + mf.ROW_LETTERS[row])


def triple(v: float, ci, digits: int = 2) -> str:
    """A value and its interval as Table 15 prints them, +0.12 [0.03, 0.24]."""
    num = lambda x, sign="": f"{round(x, digits) + 0.0:{sign}.{digits}f}"  # noqa: E731
    return f"{num(v, '+') if round(v, digits) else num(0)} [{num(ci[0])}, {num(ci[1])}]"


def table_15() -> None:
    print("Table 15: method, then for each track P, L, S_seen and S_held")
    lines = {}
    for track, rows in mf.TRACK_ROWS.items():
        for row in rows:
            stream, reset = mf._track_stream(track, row), mf.TRACK_RESET[row]
            solved = 100 * statistics.mean(float(r.solved) for r in stream.records)
            if reset == row:
                learned = "0"
            else:
                d = mf._learned(stream, mf._track_stream(track, reset))
                learned = triple(d["share_learned"], d["ci95"])
            entry = mf._track_entry(track, row)
            tests = [triple(entry[side]["share"][0], entry[side]["share"][1:]) for side in ("seen", "held")]
            lines.setdefault(row, {})[track] = f"{solved:.1f} & {learned} & " + " & ".join(tests)
    for row in mf.TRACK_ROWS["arc2"]:
        cells = [lines[row].get(track, "-- & -- & -- & --") for track in mf.TRACK_ROWS]
        print(f"  {name(row)} & " + " & ".join(cells))


def table_16() -> None:
    print("Table 16: method, stream solved and charge, first visits solved (of 15) and charge, returns solved and charge")
    runs = {**mf.TRACK_ROWS["arc3"], **PILOT}
    for row in PILOT_ORDER:
        records = [json.loads(line) for line in (mf.REPO / runs[row] / "records.jsonl").read_text().splitlines()
                   if line.strip()]
        stream = [r for r in records if r["phase"] == "stream"]
        first = [r for r in stream if r["exposure_index"] == 0]
        returns = [r for r in stream if r["exposure_index"] > 0]
        solved = lambda rs: 100 * statistics.mean(r["solved"] for r in rs)  # noqa: E731
        charge = lambda rs: statistics.mean(r["cost"] for r in rs)  # noqa: E731
        print(f"  {name(row)} & {solved(stream):.1f} & {charge(stream):.2f} & {sum(r['solved'] for r in first)} & "
              f"{charge(first):.2f} & {solved(returns):.1f} & {charge(returns):.2f} ({len(stream)} instances, "
              f"{len(first)} first visits)")


if __name__ == "__main__":
    table_15()
    table_16()
