"""Per play, the behaviours the round-1 post-mortem traced the failures to, from what the environment received (its
turn records) and the agent's own calls (prime-agent's internal compaction and review calls left out). Counts cover the
games that have finished, so a play still running can be read. Plain json; read-only. probe.py reads its patterns
(CITES, internal, text_of).

Usage: python3 analysis/trend.py LABEL=PLAY_DIR ...  (prints one tab-separated row per play)
"""
import collections
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

CITES = re.compile(r"\b(the notes|notes say|my notes|the guidance|harness notes|notes:|rule [1-9]|note [0-9])", re.I)
KEYS = re.compile(r"'Demo pair \d+ (?:input|output)'")


def text_of(c) -> str:
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def internal(call) -> bool:
    msgs = (call.get("request") or {}).get("messages") or []
    return bool(msgs) and text_of(msgs[-1].get("content")).lstrip().startswith(("<conversation>", "<trigger>"))


def play(path: Path) -> dict:
    games = solved = gave_up = requests = refused = submits = calls = cites = loaded = 0
    after = collections.Counter()
    for f in sorted((path / "captured").glob("u*.json.gz")):
        doc = json.load(gzip.open(f, "rt"))
        games += 1
        solved += bool(doc.get("solved"))
        turns = doc.get("turns") or []
        kinds = [(t.get("action") or {}).get("type") for t in turns]
        gave_up += "GiveUp" in kinds
        requests += kinds.count("RequestDemos")
        submits += kinds.count("Submit")
        for k, t in enumerate(turns):
            fb = t.get("feedback") or {}
            if fb.get("type") != "DemosFeedback":
                continue
            if fb.get("refused"):
                refused += 1
            elif k + 1 < len(turns):
                after[kinds[k + 1] or "none"] += 1
        seen_grid = False
        for c in doc["calls"]:
            if not c.get("response") or internal(c):
                continue
            calls += 1
            m = ((c["response"].get("choices") or [{}])[0].get("message") or {})
            cites += bool(CITES.search((m.get("reasoning_content") or m.get("reasoning") or "") + (m.get("content") or "")))
            seen_grid = seen_grid or any(x.get("role") == "tool" and KEYS.search(text_of(x.get("content")))
                                         for x in (c.get("request") or {}).get("messages") or [])
        loaded += seen_grid
    served = sum(after.values()) or 1
    row = {"games": games, "solved": solved, "gave_up": gave_up, "demo_requests/game": requests / max(1, games),
           "refused/game": refused / max(1, games), "submits/game": submits / max(1, games),
           "after_demo_reask%": 100 * after["RequestDemos"] / served, "after_demo_submit%": 100 * after["Submit"] / served,
           "after_demo_giveup%": 100 * after["GiveUp"] / served, "calls/game": calls / max(1, games),
           "citing%": 100 * cites / max(1, calls), "grids_loaded%": 100 * loaded / max(1, games)}
    recs = path / "records.jsonl"
    if recs.exists() and recs.stat().st_size:
        res = [json.loads(line).get("resources") or {} for line in open(recs)]
        row["budget_hits/game"] = sum(r.get("system_budget_hits") or 0 for r in res) / len(res)
        row["parse_failures/game"] = sum(r.get("system_parse_failures") or 0 for r in res) / len(res)
        row["median_min"] = statistics.median((r.get("wall_seconds") or 0) / 60 for r in res)
    return row


if __name__ == "__main__":
    rows = [(spec.split("=", 1)[0], play(Path(spec.split("=", 1)[1]).expanduser())) for spec in sys.argv[1:]]
    cols = list(dict.fromkeys(k for _, r in rows for k in r))
    print("play\t" + "\t".join(cols))
    for label, r in rows:
        print(label + "\t" + "\t".join(f"{r[k]:.1f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in cols))
