"""A play made readable: every episode of a model's play, in full and as a compact sample, and counts over all of
them. It was written for the proposer of the earlier context-distillation rounds (a model that read a round's games and
wrote the next round's notes), and its parsing of calls and actions is what the rows' code reads games with.

Usage: ``python render.py RUN_DIR OUT_DIR [--episodes 30]`` writes, in OUT_DIR:

- ``episodes/uNNN.md``: every episode in full, one file per user, grids and thinking kept, ending with each submission's
  verdict and, for a wrong one, the expected output and how the submission differs from it;
- ``episodes.md``: a sample of ``--episodes`` episodes rendered compactly (grids squeezed, long text clipped), to start
  from;
- ``calls.tsv``: one row per model call (what kind of reply, which action, tool errors, budget stops, ...), and
  ``submissions.tsv``: one row per submitted grid, so any count over the round is a filter over rows;
- ``stats.json``: the counts over every episode.

RUN_DIR is a play's run directory (``captured/uNNN.json[.gz]`` and ``records.jsonl``), copied off the node. Expected
outputs are rebuilt from each record's task, instance seed and source through continual-arc, and actions are read with
the environment's own parser, so run it where both are installed (the baselines' environment, ``../baselines``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
import warnings
from pathlib import Path

GRID_ROW = re.compile(r"^(?:\d(?: \d){2,})$")
LITERAL_INT = re.compile(r"(?<=[\[,\s])\d(?=\s*[,\]])")
READS_LOG = re.compile(r"RLM_SESSION_DIR|sessions/|\.jsonl")
BUDGET_STOP = "exceeded the time or model-call budget"
ERROR = re.compile(r"Traceback \(most recent call last\)|\b\w+Error\b:")
RETYPED_GRID_INTS = 40   # a tool call with this many literal digits retypes a grid by hand
# prime-agent's compaction call: its summarizer reads the conversation so far packed into one message that opens so; the
# environment never reads its reply, which is the summary the conversation continues from
COMPACTION = "<conversation>"
# the environment's parser's action classes, by the protocol's names
ACTION_NAMES = {"RequestDemos": "request_demos", "Submit": "submit", "Finish": "finish"}
# a submission of the right size with at most this share of its cells wrong: the rule nearly found, or a sloppy copy
NEAR_MISS_SHARE = 0.05
CALL_COLUMNS = ["user", "task", "solved", "call", "turn", "prompt_tokens", "reply_tokens", "kind", "action",
                "tool_calls", "tool_errors", "budget_stop", "empty_reply", "log_reads", "retyped_grids", "memory_calls"]
SUBMISSION_COLUMNS = ["user", "task", "turn", "valid", "correct", "submitted_size", "expected_size", "cells_wrong",
                      "cells", "diff"]


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def squeeze_grids(text: str) -> str:
    """Runs of grid rows collapsed to one marker, so a grid costs a line instead of thirty."""
    out, run = [], 0
    for line in text.splitlines():
        if GRID_ROW.match(line.strip()):
            run += 1
            continue
        if run:
            out.append(f"  [{run} grid rows]")
            run = 0
        out.append(line)
    if run:
        out.append(f"  [{run} grid rows]")
    return "\n".join(out)


def clip(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n // 2] + f" … [{len(text) - n} chars] … " + text[-n // 2:]


def load(run_dir: Path) -> list[dict]:
    """Every captured episode of a run, with its record (cost, solved) attached by user."""
    recs = {}
    if (run_dir / "records.jsonl").exists():
        for line in open(run_dir / "records.jsonl"):
            r = json.loads(line)
            recs[r.get("user", r.get("step"))] = r
    eps = []
    for f in sorted((run_dir / "captured").glob("u*.json*")):
        if f.suffix == ".tmp":   # a save that failed half-way (loop.py writes .tmp, then renames)
            continue
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt") as fp:
            doc = json.load(fp)
        doc["record"] = recs.get(doc["user"], {})
        eps.append(doc)
    return eps


def user_name(ep: dict) -> str:
    return f"u{ep['user']:03d}"


def task_name(ep: dict) -> str:
    """The task as the model saw it (its alias in the prompts), never the dataset's own id: the proposer's files name
    no real task, so nothing ties a note to a public puzzle."""
    return ep.get("task_label") or ""


def attach_expected(eps: list[dict]) -> None:
    """Each episode's expected test output as ``ep["expected"]``, rebuilt from its record's task, instance seed and
    source the way the environment built the instance (None for an episode without a record)."""
    from continual_arc.tasks import TaskRegistry
    registry = TaskRegistry.default()
    for ep in eps:
        r, ep["expected"] = ep["record"], None
        if r.get("task_id") and r.get("source"):
            registry.assign(r["task_id"], r["source"])
            pair = registry.test_instance(r["task_id"], r["instance_seed"])
            ep["expected"] = [list(row) for row in pair.output]


def grid_size(grid) -> str:
    """``HxW`` for a grid the environment accepts (continual-arc's own check), else "invalid"."""
    from continual_arc.grids import grid_shape, is_valid_grid
    return "{}x{}".format(*grid_shape(grid)) if is_valid_grid(grid) else "invalid"


def model_calls(ep: dict) -> list[dict]:
    """The episode's model calls that completed (a request and a response), in order: the calls the renders number."""
    return [c for c in ep["calls"] if c.get("request") and c.get("response")]


def is_compaction(call: dict) -> bool:
    messages = (call.get("request") or {}).get("messages")
    # a raw completion, or any other body the model's own code sent through the proxy (its "messages", if any, not a
    # list of chat messages), is never prime-agent's summarizer
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return False
    last = messages[-1]
    return last.get("role") == "user" and text_of(last.get("content")).lstrip().startswith(COMPACTION)


def submissions(ep: dict) -> list[dict]:
    """Every grid the episode submitted, from its protocol turns: the environment's verdict and, against the expected
    output, both sizes and how many cells differ (the columns of submissions.tsv)."""
    out, expected = [], ep.get("expected")
    for turn, t in enumerate(ep.get("turns") or []):
        action, feedback = t.get("action") or {}, t.get("feedback") or {}
        if action.get("type") != "Submit":
            continue
        grid = action.get("grid")
        row = {"user": user_name(ep), "task": task_name(ep), "turn": turn, "valid": int(bool(feedback.get("valid"))),
               "correct": int(bool(feedback.get("correct"))), "submitted_size": grid_size(grid),
               "expected_size": grid_size(expected), "cells_wrong": None, "cells": None}
        if row["submitted_size"] == "invalid":
            row["diff"] = "invalid grid"
        elif expected is None:
            row["diff"] = "correct" if row["correct"] else "no expected output"
        elif row["submitted_size"] != row["expected_size"]:
            row["diff"] = f"wrong size: {row['submitted_size']} against {row['expected_size']}"
        else:
            wrong = sum(a != b for ra, rb in zip(grid, expected) for a, b in zip(ra, rb))
            row.update(cells_wrong=wrong, cells=len(expected) * len(expected[0]))
            row["diff"] = "correct" if row["correct"] else f"{wrong} of {row['cells']} cells wrong"
        out.append(row)
    return out


def reply_action(content: str) -> str:
    """The action the environment reads from a text reply, by its own parser (lenient: any JSON action anywhere in the
    reply, under any usual spelling, a bare demo request, a grid written as rows), or "" when it finds none."""
    from continual_arc_baselines.systems import protocol
    try:
        with warnings.catch_warnings():   # the parser evaluates model-written literals, whose stray escapes warn
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = protocol.parse_action(content, finish=True)
    except protocol.ParseError:
        return ""
    first = parsed[0] if isinstance(parsed, list) else parsed
    return ACTION_NAMES.get(type(first).__name__, type(first).__name__)


def call_rows(ep: dict) -> list[dict]:
    """One row per model call of the episode (the columns of calls.tsv), computed once and kept on the episode: the
    reply's kind and the action the environment would read from it, and what its tool calls did. A tool call's error is
    its result's, which arrives in a later call's messages. A compaction call is kind ``compaction`` with no action."""
    if "_rows" in ep:
        return ep["_rows"]
    calls = model_calls(ep)
    errored: dict = {}
    for c in calls:   # every later request repeats earlier tool results; each is searched once
        for m in c["request"]["messages"]:
            if m.get("role") == "tool" and m.get("tool_call_id") not in errored:
                errored[m.get("tool_call_id")] = bool(ERROR.search(text_of(m.get("content"))))
    rows = []
    for i, c in enumerate(calls):
        choice = (c["response"].get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content, tcs, codes = (msg.get("content") or "").strip(), msg.get("tool_calls") or [], tool_codes(msg)
        last, compaction = c["request"]["messages"][-1], is_compaction(c)
        if compaction:
            kind = "compaction"
        else:
            kind = f"tool:{(tcs[0].get('function') or {}).get('name')}" if tcs else ("text" if content else "empty")
        rows.append({
            "user": user_name(ep), "task": task_name(ep), "solved": int(bool(ep["record"].get("solved"))), "call": i,
            "turn": c.get("turn"), "prompt_tokens": len(c["response"].get("prompt_token_ids") or []),
            "reply_tokens": len(choice.get("token_ids") or []), "kind": kind,
            # from the reply's text whether or not it also called a tool: the environment reads such a reply too
            "action": reply_action(content) if content and not compaction else "",
            "tool_calls": len(codes), "tool_errors": sum(errored.get(tc.get("id"), False) for tc in tcs),
            "budget_stop": int(last.get("role") == "user" and BUDGET_STOP in text_of(last.get("content"))),
            "empty_reply": int(not (content or tcs)),
            "log_reads": sum(bool(READS_LOG.search(code)) for code in codes),
            "retyped_grids": sum(len(LITERAL_INT.findall(code)) >= RETYPED_GRID_INTS for code in codes),
            "memory_calls": sum("rlm.harness" in code for code in codes)})
    ep["_rows"] = rows
    return rows


def tool_codes(message: dict) -> list[str]:
    codes = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        code = args.get("code") if isinstance(args, dict) else None
        codes.append(f"{fn.get('name')}: {code if code is not None else fn.get('arguments', '')}")
    return codes


def episode_stats(ep: dict) -> dict:
    """What one episode did: its calls.tsv rows summed."""
    rows = call_rows(ep)

    def total(k):
        return sum(r[k] for r in rows)

    return {"calls": len(rows), "tool_calls": total("tool_calls"), "tool_errors": total("tool_errors"),
            "budget_stops": total("budget_stop"), "log_reads": total("log_reads"),
            "retyped_grids": total("retyped_grids"), "memory_calls": total("memory_calls"),
            "empty_replies": total("empty_reply")}


def call_blocks(ep: dict, full: bool = False) -> list[tuple[str, str]]:
    """Each model call as (text, one-line gist): what the model was shown since its previous call, then its reasoning,
    its text and its tool calls. Compact by default (grids squeezed to a marker, long text clipped); ``full`` keeps
    every character."""
    def fit(text: str, n: int, squeeze: bool = True) -> str:
        return text.strip() if full else clip(squeeze_grids(text) if squeeze else text, n)

    blocks, prev_len = [], 0
    for i, c in enumerate(model_calls(ep)):
        msgs = c["request"]["messages"]
        # what the model was shown since its previous call: the messages after the last assistant message, or the
        # whole conversation at the first call (a compaction also restarts the count). A compaction call's input is
        # that whole conversation again, so it is named, not repeated: its reply, the summary, is what matters.
        last_asst = max((j for j, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        new = msgs[last_asst + 1:] if (last_asst >= 0 and len(msgs) >= prev_len) else msgs[1:]
        prev_len = len(msgs)
        lines, gist = [], []
        if is_compaction(c):
            chars = len(text_of(msgs[-1].get("content")))
            lines.append(f"[{i}] compaction: the harness has the conversation so far summarised ({chars} characters, "
                         "not repeated here)")
            gist.append("compaction")
            new = []
        for m in new:
            role, body = m.get("role"), text_of(m.get("content"))
            lines.append(f"[{i}] {role}: " + fit(body, 700 if role == "user" else 400))
            if role == "tool" and ERROR.search(body):
                err = next((ln for ln in reversed(body.splitlines()) if ERROR.search(ln)), "error")
                gist.append("error " + clip(err, 90))
            elif role == "user" and BUDGET_STOP in body:
                gist.append("budget stop")
        msg = (c["response"].get("choices") or [{}])[0].get("message") or {}
        if msg.get("reasoning"):
            lines.append(f"[{i}] model thinks: " + fit(msg["reasoning"], 500, squeeze=False))
        if msg.get("content"):
            lines.append(f"[{i}] model says: " + fit(msg["content"], 400))
        for code in tool_codes(msg):
            lines.append(f"[{i}] model runs {fit(code, 700, squeeze=False)}")
            first = next((ln for ln in code.split(": ", 1)[-1].splitlines() if ln.strip() and not ln.startswith("#")), "")
            gist.append("runs " + code.split(":", 1)[0] + " `" + clip(first, 70) + "`"
                        + (" [reads session log]" if READS_LOG.search(code) else "")
                        + (" [retypes grids]" if len(LITERAL_INT.findall(code)) >= RETYPED_GRID_INTS else ""))
        if not (msg.get("content") or msg.get("tool_calls")):
            gist.append("empty reply")
        elif not msg.get("tool_calls"):
            gist.append("says " + clip(msg.get("content", "").replace("\n", " "), 80))
        blocks.append(("\n".join(lines), f"[{i}] " + "; ".join(gist)))
    return blocks


def outcome(ep: dict) -> str:
    """The episode's outcome line, as both renders head it."""
    r, st = ep["record"], episode_stats(ep)
    return (f"outcome: {'solved' if r.get('solved') else 'not solved'}, cost {r.get('cost')}, wrong {r.get('wrong_attempts')}, "
            f"demo requests {r.get('demo_requests')}, gave up {r.get('gave_up')}; model calls {st['calls']}, tool errors "
            f"{st['tool_errors']}, budget stops {st['budget_stops']}\n")


def render_episode(k: int, ep: dict, budget: int = 15_000, keep_head: int = 3, keep_tail: int = 2) -> str:
    """One episode within about ``budget`` characters: every call in full when it fits; otherwise the first and last
    calls in full and the ones between as one line each."""
    head = f"## Episode {k} · {user_name(ep)} · {task_name(ep)} (episodes/{user_name(ep)}.md in full)\n" + outcome(ep)
    blocks = call_blocks(ep)
    if sum(len(b[0]) for b in blocks) <= budget or len(blocks) <= keep_head + keep_tail:
        body = [b[0] for b in blocks]
    else:
        body = ([b[0] for b in blocks[:keep_head]] + [f"… calls {keep_head}–{len(blocks) - keep_tail - 1}, one line each:"]
                + [b[1] for b in blocks[keep_head:len(blocks) - keep_tail]] + [b[0] for b in blocks[len(blocks) - keep_tail:]])
    return head + "\n".join(body) + "\n"


def render_full(ep: dict) -> str:
    """One episode with every character of every call (grids and thinking kept), then each submission's verdict and,
    for a wrong one, the expected output it missed."""
    from continual_arc_baselines.systems.protocol import render_grid
    head = f"# {user_name(ep)} · {task_name(ep)}\n" + outcome(ep)
    subs = submissions(ep)
    lines = [f"- turn {s['turn']}: {s['diff']}" for s in subs] or ["none"]
    if ep.get("expected") is not None and any(s["valid"] and not s["correct"] for s in subs):
        lines += [f"expected output ({grid_size(ep['expected'])}):"] + ["  " + row for row in render_grid(ep["expected"]).splitlines()]
    return head + "\n".join(b[0] for b in call_blocks(ep, full=True)) + "\n\n## Submissions\n" + "\n".join(lines) + "\n"


def summary(eps: list[dict]) -> dict:
    per = [episode_stats(e) for e in eps]
    recs = [e["record"] for e in eps if e["record"]]
    subs = [s for e in eps for s in submissions(e)]
    wrong = [s for s in subs if s["valid"] and not s["correct"]]

    def total(k):
        return sum(p[k] for p in per)

    return {
        "episodes": len(eps), "solved": sum(bool(r.get("solved")) for r in recs),
        "mean_cost": round(statistics.mean(r["cost"] for r in recs), 3) if recs else None,
        "gave_up": sum(bool(r.get("gave_up")) for r in recs),
        "demo_requests": {"total": sum(r.get("demo_requests", 0) for r in recs),
                          "episodes_with_none": sum(r.get("demo_requests", 0) == 0 for r in recs)},
        "wrong_attempts": sum(r.get("wrong_attempts", 0) for r in recs),
        "model_calls": {"total": total("calls"), "median_per_episode": statistics.median(p["calls"] for p in per) if per else 0},
        "tool_calls": total("tool_calls"), "tool_errors": total("tool_errors"),
        "budget_stops": total("budget_stops"), "empty_replies": total("empty_replies"),
        "episodes_reading_session_log": sum(p["log_reads"] > 0 for p in per), "session_log_reads": total("log_reads"),
        "episodes_retyping_grids": sum(p["retyped_grids"] > 0 for p in per), "retyped_grid_cells": total("retyped_grids"),
        "episodes_using_memory": sum(p["memory_calls"] > 0 for p in per),
        # against the expected output: a wrong size, a near miss (right size, at most NEAR_MISS_SHARE of cells wrong)
        # or a wrong rule; a correct verdict on a grid unlike the rebuilt output would mean the rebuild is wrong
        "submissions": {"total": len(subs), "correct": sum(s["correct"] for s in subs),
                        "invalid": sum(not s["valid"] for s in subs),
                        "wrong_size": sum(s["diff"].startswith("wrong size") for s in wrong),
                        "near_miss": sum(s["cells_wrong"] is not None and s["cells_wrong"] <= NEAR_MISS_SHARE * s["cells"]
                                         for s in wrong),
                        "far": sum(s["cells_wrong"] is not None and s["cells_wrong"] > NEAR_MISS_SHARE * s["cells"]
                                   for s in wrong),
                        # correct but not the rebuilt output: another size, cells apart, or nothing rebuilt
                        "expected_output_mismatches": sum(bool(s["correct"]) and s["cells_wrong"] != 0 for s in subs)},
    }


def pick(eps: list[dict], n: int) -> list[dict]:
    """A third solved (cheapest first), two thirds not solved (costliest first), at most one episode per task until
    every task is represented."""
    solved = sorted((e for e in eps if e["record"].get("solved")), key=lambda e: e["record"].get("cost", 0))
    failed = sorted((e for e in eps if not e["record"].get("solved")), key=lambda e: -e["record"].get("cost", 0))

    def spread(pool, k):
        out, tasks = [], set()
        for e in pool:
            if len(out) < k and e["task_id"] not in tasks:
                out.append(e)
                tasks.add(e["task_id"])
        for e in pool:
            if len(out) < k and e not in out:
                out.append(e)
        return out

    chosen = spread(solved, n // 3) + spread(failed, n - n // 3)
    return chosen[:n]


def write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=columns, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def render_round(run_dir: Path, out: Path, episodes: int = 30) -> dict:
    """Everything the proposer reads for one play (the files in the module docstring), written to ``out``; returns
    what was written."""
    eps = load(run_dir)
    attach_expected(eps)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    for ep in eps:
        (out / "episodes" / f"{user_name(ep)}.md").write_text(render_full(ep))
    write_tsv(out / "calls.tsv", CALL_COLUMNS, [r for ep in eps for r in call_rows(ep)])
    write_tsv(out / "submissions.tsv", SUBMISSION_COLUMNS, [s for ep in eps for s in submissions(ep)])
    stats = summary(eps)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    chosen = pick(eps, episodes)
    (out / "episodes.md").write_text("\n".join(render_episode(k, e) for k, e in enumerate(chosen, 1)))
    return {"episodes": len(eps), "sampled": len(chosen), "calls": stats["model_calls"]["total"],
            "submissions": stats["submissions"]["total"], "digest_chars": (out / "episodes.md").stat().st_size}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir")
    p.add_argument("out_dir")
    p.add_argument("--episodes", type=int, default=30, help="how many episodes the compact sample holds")
    a = p.parse_args()
    print(json.dumps(render_round(Path(a.run_dir), Path(a.out_dir), a.episodes)))


if __name__ == "__main__":
    main()
