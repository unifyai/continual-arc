"""final_checks.py RUN_DIR [--last N]: a running trained row read against each thing that broke the 19 Oct final runs
(docs/method-search.md, "The final configuration"), from its own files: OK, WATCH or STOP a line, the last N rounds'
stream and labels and its newest training phase. The operator reads it every round of the gating DAgger run, and the author's go
for the other runs waits on every line being OK (the author, 26 Oct 2026).

  budget      the stream plays turn-based (driver.json's harness budget), so no action is cut at a time
  reply cap   the requests ask for the window-set cap (driver.json's max_tokens, paper.MAX_TOKENS), not prime-agent's 32,000
  clock cuts  no action of the stream's games ended with a call cancelled before 40 calls (a cut at seconds)
  labels      the teacher's labels as R's taught: reasoning left to it, so an easy move gets next to none (labels under
              100 characters of reasoning: hers at least 59%, which had none; today every OpenRouter host gives GLM a
              14-50 character note on an easy move, since its reasoning cannot be switched off, and 500-2,000 characters
              on a hard one, 26 Oct 2026), short on average (hers 924 characters, ours 6,170 under required reasoning),
              no program-library ritual (hers 2%), and nearly every moment labelled
  student     the student acts rather than deliberates: completion tokens a call (her models 1,200-1,500, ours 2,300-7,400),
              reasoning's share of them (hers 58-62%, ours 82-96%, a student that stopped thinking near 0%), and replies cut
              at the reply cap (hers 0.1-0.2 a game, the long-winded checkpoints 1.3-2.3)
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
NEXT_TO_NONE = 100   # characters of reasoning under which a label reasons next to nothing: an easy move's note
REPLY_CAP = 65_536
CALL_CAP = 40


def status(ok: bool, watch: bool = False) -> str:
    return "OK" if ok else ("WATCH" if watch else "STOP")


def rounds_of(run: Path, last: int) -> list[Path]:
    found = sorted((run / "play" / "rounds").glob("r[0-9][0-9][0-9][0-9]"))
    return found[-last:]


def stream(rounds: list[Path]) -> dict[str, float]:
    games = calls = tokens = reasoning = capped = clock_cuts = 0
    for r in rounds:
        for f in glob.glob(str(r / "captured" / "*.json.gz")):
            g = json.load(gzip.open(f))
            games += 1
            per_turn, cancelled = {}, set()
            for c in g["calls"]:
                if c.get("status") == "cancelled":
                    cancelled.add(c["turn"])
                if c.get("status") != "completed":
                    continue
                per_turn[c["turn"]] = per_turn.get(c["turn"], 0) + 1
                calls += 1
                capped += c.get("finish_reason") == "length"
                u = (c.get("response") or {}).get("usage") or {}
                tokens += u.get("completion_tokens") or 0
                reasoning += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            clock_cuts += sum(1 for t in cancelled if per_turn.get(t, 0) < CALL_CAP)
    return {"games": games, "tokens_a_call": tokens / max(1, calls), "reasoning_share": reasoning / max(1, tokens),
            "capped_a_game": capped / max(1, games), "clock_cuts_a_game": clock_cuts / max(1, games)}


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
    out.append(("clock cuts", status(s["clock_cuts_a_game"] == 0, s["clock_cuts_a_game"] < 0.05),
                f"{s['clock_cuts_a_game']:.2f} a game ({span})"))
    out.append(("student tokens", status(s["tokens_a_call"] < 2_000, s["tokens_a_call"] < 3_000),
                f"{s['tokens_a_call']:,.0f} completion tokens a call"))
    share = s["reasoning_share"]
    out.append(("student reasoning", status(0.40 <= share <= 0.75, 0.30 <= share <= 0.85),
                f"{share:.0%} of its tokens reasoning"))
    out.append(("reply cap hits", status(s["capped_a_game"] < 0.1, s["capped_a_game"] < 0.3),
                f"{s['capped_a_game']:.2f} replies cut a game"))
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
