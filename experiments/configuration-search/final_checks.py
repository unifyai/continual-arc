"""final_checks.py RUN_DIR [--last N]: a running trained row read against each thing that broke the 19 Oct final runs
(docs/method-search.md, "The final configuration"), from its own files: OK, WATCH or STOP a line, the last N rounds'
stream and labels and its newest training phase. The operator reads it every round of the gating DAgger run, and the author's go
for the other runs waits on every line being OK (the author, 26 Oct 2026: "verified that the kind of things that tripped us up
last time don't trip us up this time").

  budget      the stream plays turn-based (driver.json's harness budget), so no action is cut at a time
  reply cap   the requests ask for the window-set cap (driver.json's max_tokens, paper.MAX_TOKENS), not prime-agent's 32,000
  clock cuts  no action stopped at a time (the harness's "action budget exceeded (Ns)" in its users' adapter.log; under
              the turn-based budget every stop is "N model calls"), and no game ended by the turn's liveness guard (the
              records' system_infra_failures)
  labels      the teacher's labels as R's taught: reasoning left to it, so an easy move gets next to none (labels under
              100 characters of reasoning: his at least 59%, which had none; today every OpenRouter host gives GLM a
              14-50 character note on an easy move, since its reasoning cannot be switched off, and 500-2,000 characters
              on a hard one, 26 Oct 2026), short on average (his 924 characters, ours 6,170 under required reasoning),
              no program-library ritual (his 2%), and nearly every moment labelled
  student     the student acts rather than deliberates: completion tokens a call (his models 1,200-1,500, ours 2,300-7,400),
              reasoning's share of them (his 58-62% and ours 82-96% at the paper's old sampling; at the model card's, every
              final run's base model 25-34% before its first phase, 28 Oct 2026; a student that stopped thinking near 0%), replies of
              32,000 tokens or more, which prime-agent's own cap and so R's validation cut (on his 100 puzzles turn-
              based: his models 0.24-0.38 a game; dagger-u4-lag1's phase 109 0.89, solving 55; its phase 159 3.34, solving
              43; the base model none), and replies cut at the 65,536-token cap, which only a loop that repeats itself
              reaches (the base model none; dagger-u4-final's phases 12-21 0.30 a game, 26 Oct 2026)
  dose        each phase trains on every label so far (the aggregate dose): its examples are most of every round's
              labelled examples so far (all but the held-out games), not a window of the last rounds (the recent dose);
              PG's, every game won so far
  kept        the run still changes its model: no long run of phases in a row that kept no update, their held-out
              stop keeping the adapter they started from or no step taken (the final DAgger runs' longest run was 15
              phases, 1 Nov 2026; PG's held-out stop kept none for 55 to 190 phases in a row in each of six runs,
              unseen for 12 days: docs/method-search.md, "PG without a held-out stop"). A model that has stopped
              improving is no fault of the run's, so this only ever says WATCH
  entropy     PG's student stays near what it plays: its next-token entropy on the wins it trains, 0.13-0.21 nats
              at the model card's sampling (pg-u4-disjoint's phases, 30 Oct-1 Nov 2026), 2.4-3.2 in the runs whose
              replies broke (20 Oct 2026); far under the first, its replies have collapsed onto a few
The thresholds are the evidence's own numbers (25-26 Oct 2026, and the dates above), with room either side.
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
KEPT_WATCH = 20      # phases in a row that kept no update
ENTROPY = (0.05, 1.0, 2.0)   # PG's entropy in nats: OK within the first two, STOP from the third


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
            # the teacher's own reasoning field, as his 59% without any was counted (not set.json's no_reasoning, which
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


def newest_phase(run: Path) -> tuple[int, dict] | None:
    """The newest phase that has trained (its phase.json) and its record."""
    phases = sorted((run / "train").glob("p[0-9][0-9][0-9][0-9]/phase.json"))
    return (int(phases[-1].parent.name[1:]), json.loads(phases[-1].read_text())) if phases else None


def dose(run: Path) -> tuple[str, int, int] | None:
    """The newest phase's training against what it should have taken: DAgger's and OPSD's examples against every
    round's labelled examples up to its rounds (set.json's), PG's won games against every game won in them (the
    records')."""
    newest = newest_phase(run)
    if newest is None:
        return None
    k, phase = newest
    if phase.get("objective") == "pg":
        won = sum(bool(json.loads(line).get("solved")) for line in (run / "play" / "records.jsonl").read_text().splitlines()
                  if line.strip() and json.loads(line).get("round", k + 1) <= k)
        return "won games", (phase.get("games") or {}).get("wins", 0), won
    trained = (phase.get("items") or {}).get("items", 0)
    labelled = sum(json.loads(f.read_text()).get("examples", 0)
                   for f in (run / "play" / "rounds").glob("r[0-9][0-9][0-9][0-9]/labels/set.json") if int(f.parent.parent.name[1:]) <= k)
    return "labelled examples", trained, labelled


def kept(run: Path) -> tuple[int, int, int | None] | None:
    """The phases that trained, how many of the newest in a row kept no update (their best adapter the one they
    started from, best_step 0, or no step taken), and the newest that kept one; None before any phase trains."""
    done = [(int(f.parent.name[1:]), json.loads(f.read_text())) for f in sorted((run / "train").glob("p[0-9][0-9][0-9][0-9]/DONE"))]
    trained = [(k, d) for k, d in done if not d.get("skipped")]
    if not trained:
        return None
    moved = [k for k, d in trained if d.get("steps") and d.get("best_step") != 0]
    streak = len([k for k, _ in trained if not moved or k > moved[-1]])
    return len(trained), streak, moved[-1] if moved else None


def entropy(run: Path) -> float | None:
    """PG's newest phase's mean next-token entropy over its steps' tokens, None for another row's or a phase with none."""
    newest = newest_phase(run)
    if newest is None or newest[1].get("objective") != "pg":
        return None
    steps = [st for st in newest[1].get("steps", []) if st.get("entropy_mean") is not None and st.get("tokens")]
    tokens = sum(st["tokens"] for st in steps)
    return sum(st["entropy_mean"] * st["tokens"] for st in steps) / tokens if tokens else None


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
    out.append(("student reasoning", status(0.20 <= share <= 0.75, 0.10 <= share <= 0.85),
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
        what, trained, so_far = d
        share = 1.0 if what == "won games" else 0.7   # PG holds no game out
        out.append(("dose", status(trained >= share * so_far), f"newest phase trained on {trained:,} of {so_far:,} {what} so far"))
    k = kept(run)
    if k is not None:
        n, streak, last = k
        out.append(("updates kept", status(streak < KEPT_WATCH, True),
                    f"the newest {streak} of {n} phases that trained kept no update "
                    f"({f'the last that kept one: phase {last}' if last is not None else 'none has kept one'})"))
    e = entropy(run)
    if e is not None:
        out.append(("entropy", status(ENTROPY[0] <= e <= ENTROPY[1], e < ENTROPY[2]),
                    f"{e:.2f} nats a token on the wins its newest phase trained"))
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
