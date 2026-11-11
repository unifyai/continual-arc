"""final_checks.py RUN_DIR [--last N]: a running trained row read against each thing that broke the 19 Oct final runs
(docs/method-search.md, "The final configuration"), from its own files: OK, WATCH or STOP a line, the last N rounds'
stream and labels and its newest training phase. The operator reads it every round of the gating DAgger run, and the author's go
for the other runs waits on every line being OK (the author, 26 Oct 2026).

  budget      the stream plays turn-based (driver.json's harness budget), so no action is cut at a time
  reply cap   the requests ask for the window-set cap (driver.json's max_tokens, paper.MAX_TOKENS), not prime-agent's 32,000
  clock cuts  no action stopped at a time (the harness's "action budget exceeded (Ns)" in its users' adapter.log; under
              the turn-based budget every stop is "N model calls"), and no game ended by the turn's liveness guard (the
              records' system_infra_failures)
  labels      the teacher's labels as R's taught: reasoning left to it, so an easy move gets next to none (labels under
              100 characters of reasoning: hers at least 59%, which had none; today every OpenRouter host gives GLM a
              14-50 character note on an easy move, since its reasoning cannot be switched off, and 500-2,000 characters
              on a hard one, 26 Oct 2026), short on average (hers 924 characters, ours 6,170 under required reasoning),
              no program-library ritual (hers 2%), and nearly every moment labelled
  student     the student acts rather than deliberates: completion tokens a call (her models 1,200-1,500, ours 2,300-7,400),
              reasoning's share of them (hers 58-62%, ours 82-96%, a student that stopped thinking near 0%), replies of
              32,000 tokens or more, which prime-agent's own cap and so R's validation cut (on her 100 puzzles turn-
              based: her models 0.24-0.38 a game; dagger-u4-lag1's phase 109 0.89, solving 55; its phase 159 3.34, solving
              43; the base model none), and replies cut at the 65,536-token cap, which only a loop that repeats itself
              reaches (the base model none; dagger-u4-final's phases 12-21 0.30 a game, 26 Oct 2026)
  dose        each phase trains on every label so far (the aggregate dose): its examples are most of every round's
              labelled examples so far (all but the held-out games), not a window of the last rounds (the recent dose)
The thresholds are the evidence's own numbers (25-26 Oct 2026), with room either side.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
from pathlib import Path

PROGRAMS = re.compile(r"program", re.I)
TIME_STOP = re.compile(r"action budget exceeded \([0-9.]+s\)")
NEXT_TO_NONE = 100   # characters of reasoning under which a label reasons next to nothing: an easy move's note
REPLY_CAP = 65_536
LONG = 32_000        # prime-agent's own reply cap, which R's validation plays under


def status(ok: bool, watch: bool = False) -> str:
    return "OK" if ok else ("WATCH" if watch else "STOP")


def rounds_of(run: Path, last: int) -> list[Path]:
    found = sorted((run / "play" / "rounds").glob("r[0-9][0-9][0-9][0-9]"))
    return found[-last:]


def stream(rounds: list[Path]) -> dict[str, float]:
    games = calls = tokens = reasoning = capped = long = 0
    for r in rounds:
        for f in glob.glob(str(r / "captured" / "*.json.gz")):
            g = json.load(gzip.open(f))
            games += 1
            for c in g["calls"]:
                if c.get("status") != "completed":
                    continue
                calls += 1
                capped += c.get("finish_reason") == "length"
                u = (c.get("response") or {}).get("usage") or {}
                tokens += u.get("completion_tokens") or 0
                long += (u.get("completion_tokens") or 0) >= LONG
                reasoning += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return {"games": games, "tokens_a_call": tokens / max(1, calls), "reasoning_share": reasoning / max(1, tokens),
            "capped_a_game": capped / max(1, games), "long_a_game": long / max(1, games)}


def clock(run: Path) -> tuple[int, int]:
    """Actions the harness stopped at a time, over the whole stream (its users' adapter.log), and games the turn's
    liveness guard ended (the records' system_infra_failures)."""
    time_stops = sum(len(TIME_STOP.findall(f.read_text(errors="replace")))
                     for f in (run / "play" / "users").glob("*/system/adapter.log"))
    records = run / "play" / "records.jsonl"
    stalls = sum(bool((json.loads(line).get("resources") or {}).get("system_infra_failures"))
                 for line in records.read_text().splitlines() if line.strip()) if records.exists() else 0
    return time_stops, stalls


def labels(rounds: list[Path]) -> dict[str, float] | None:
    n = no_reasoning = unlabelled = programs = chars = 0
    for r in rounds:
        s = r / "labels" / "set.json"
        if not s.exists():
            continue
        unlabelled += json.loads(s.read_text()).get("unlabelled", 0)
        for f in glob.glob(str(r / "labels" / "labels" / "*.json")):
            lab = json.loads(Path(f).read_text())
            # the teacher's own reasoning field, as her 59% without any was counted (not set.json's no_reasoning, which
            # takes a label's prose for reasoning too), under NEXT_TO_NONE characters counted as next to none
            reason = (lab.get("reasoning") or "").strip()
            n += 1
            no_reasoning += len(reason) < NEXT_TO_NONE
            chars += len(reason)
            programs += bool(PROGRAMS.search(reason + str((lab.get("message") or {}).get("content") or "")))
    if not n:
        return None
    return {"labels": n, "no_reasoning_share": no_reasoning / n, "reasoning_chars": chars / n,
            "programs_share": programs / n, "unlabelled_share": unlabelled / (n + unlabelled)}


def dose(run: Path) -> tuple[int, int] | None:
    """The newest phase's training examples, and every round's labelled examples up to its rounds (set.json's)."""
    phases = sorted((run / "train").glob("p[0-9][0-9][0-9][0-9]/phase.json"))
    if not phases:
        return None
    k = int(phases[-1].parent.name[1:])
    trained = (json.loads(phases[-1].read_text()).get("items") or {}).get("items", 0)
    labelled = sum(json.loads(f.read_text()).get("examples", 0)
                   for f in (run / "play" / "rounds").glob("r[0-9][0-9][0-9][0-9]/labels/set.json") if int(f.parent.parent.name[1:]) <= k)
    return trained, labelled


def checks(run: Path, last: int) -> list[tuple[str, str, str]]:
    out = []
    driver = json.loads((run / "play" / "driver.json").read_text())
    budget = (driver.get("harness_budget") or {}).get("name")
    out.append(("budget", status(budget == "turns"), f"harness budget {budget!r}"))
    cap = driver.get("max_tokens")
    out.append(("reply cap", status(cap == REPLY_CAP), f"max_tokens {cap}"))
    rounds = rounds_of(run, last)
    s = stream(rounds)
    span = f"rounds {rounds[0].name}-{rounds[-1].name}, {s['games']} games" if rounds else "no round yet"
    time_stops, stalls = clock(run)
    out.append(("clock cuts", status(time_stops == 0 and stalls == 0, time_stops == 0),
                f"{time_stops} actions stopped at a time, {stalls} games ended by the liveness guard (the whole stream)"))
    out.append(("student tokens", status(s["tokens_a_call"] < 2_000, s["tokens_a_call"] < 3_000),
                f"{s['tokens_a_call']:,.0f} completion tokens a call ({span})"))
    share = s["reasoning_share"]
    out.append(("student reasoning", status(0.40 <= share <= 0.75, 0.30 <= share <= 0.85),
                f"{share:.0%} of its tokens reasoning"))
    out.append(("long replies", status(s["long_a_game"] < 0.5, s["long_a_game"] < 2.0),
                f"{s['long_a_game']:.2f} replies of {LONG:,} tokens or more a game"))
    out.append(("reply cap hits", status(s["capped_a_game"] < 0.2, s["capped_a_game"] < 1.0),
                f"{s['capped_a_game']:.2f} replies cut at the cap a game"))
    lab = labels(rounds)
    if lab is not None:
        share, chars = lab["no_reasoning_share"], lab["reasoning_chars"]
        out.append(("label reasoning", status(0.40 <= share <= 0.85 and chars < 2_500, share >= 0.20 and chars < 4_000),
                    f"{lab['no_reasoning_share']:.0%} with next to none, {lab['reasoning_chars']:,.0f} characters on average "
                    f"({lab['labels']} labels)"))
        out.append(("label ritual", status(lab["programs_share"] < 0.05, lab["programs_share"] < 0.15),
                    f"{lab['programs_share']:.0%} mention programs"))
        out.append(("labelled", status(lab["unlabelled_share"] < 0.15, lab["unlabelled_share"] < 0.30),
                    f"{lab['unlabelled_share']:.0%} of moments unlabelled"))
    d = dose(run)
    if d is not None:
        out.append(("dose", status(d[0] >= 0.7 * d[1]), f"newest phase trained on {d[0]:,} of {d[1]:,} labelled examples so far"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--last", type=int, default=10, help="the stream's last N rounds")
    a = p.parse_args()
    rows = checks(a.run.expanduser(), a.last)
    for name, st, detail in rows:
        print(f"{st:5s}  {name:17s} {detail}")
    sys.exit(0 if all(st == "OK" for _, st, _ in rows) else 1)


if __name__ == "__main__":
    main()
