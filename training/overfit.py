"""The recall overfit, an experiment from before the paper's rows: can on-policy distillation write five puzzles' rules
into the base model's weights, with the student stating each rule in its own reasoning? The trained rows reach this
module only through oracle_hint.py, which reads captured plays with :func:`read_docs` and :func:`agent_calls`.

The teacher is Qwen3.5-9B reading, as the opening of its reasoning at every call, c1 and a card C in the first person:
the task's alias, its rule, and a ``transform(grid)`` checked on 400 fresh Re-ARC pairs (``overfit_cards.json``). The
five tasks are stream tasks (ARC-1 training tasks with Re-ARC generators), so every game is a fresh instance: the test
users' instances are never trained on, and what the student can carry from training to them is the rule, not a grid.

A thought that opens the teacher's reasoning is never a target in on-policy distillation (train.py: the teacher reads
the prompt, the thought, then the student's reply; the loss covers the reply), so a student distilled from it acts on
the rule without stating it. To make the recall a target, the student is first trained by SFT on the teacher's solved
games with the boundary moved: the student reads the call without the thought, and writes C (on the game's first call
only) and the teacher's continuation (``sft``). On-policy distillation then goes on from that student, the teacher
reading c1 as the run's context and C as each call's own thought (``thoughts``).

Subcommands (with continual_arc, continual_arc_baselines and ask_sd importable: ``python training/overfit.py ...``):
  cards     every task's card C (card-TASK.md) and the teacher's context, c1 then C (teacher-TASK.md), each card's
            program checked on fresh pairs first
  schedule  one instance of each task (or of --tasks) for each of --users, one user per instance as loop.py's schedules
  sft       the teacher's solved games as SFT plays with the boundary moved (captured/, as train.py --objective sft reads)
  thoughts  a play's calls, each carrying its task's card as the teacher's thought (train.py's per-call thought)
  report    every play's solves, cost and recall rate, by arm and by task (report.md)
"""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import loop  # noqa: E402

TASKS = ("7837ac64", "63613498", "673ef223", "88a10436", "7e0986d6")   # no recorded play had solved the first four
TEST_USERS = list(range(80, 85))                                      # past every stream user the earlier runs played
DATA_USERS = list(range(85, 100)) + list(range(0, 5))                 # the teacher's games the SFT set is built from
OPSD_USERS_PER_BLOCK = 10


def opsd_users(block: int) -> list[int]:
    """The collective configuration's users whose instances block ``block`` (1, 2, ...) plays: ten a block from user 5,
    none of them a test user while there are at most seven blocks."""
    first = 5 + OPSD_USERS_PER_BLOCK * (block - 1)
    return list(range(first, first + OPSD_USERS_PER_BLOCK))


# The card as the teacher thinks it and the student learns to write it. The plan follows c1 (a demo pair before any
# submission, every grid loaded, the transform checked on the demos), so it holds at every call of a game.
CARD = ("This is task {alias}. I have solved this task before, and I recall its rule. {rule}\n\n"
        "The transform I wrote for it then, checked on many of its pairs:\n```python\n{program}```\n"
        "Plan: once I have a demo pair, load every grid, check that transform(demo input) equals the demo output for "
        "each pair, then submit the printed json.dumps(transform(G['Test input'])).")
RECALL_MARK = "I have solved this task before"   # in a reply's reasoning: the student wrote the recall itself
# The teacher's thought at a game's first call ends with this, so its next tokens are the recall itself: without it
# the teacher, having already thought the card, finds a student's recall redundant, and one block of on-policy
# distillation took the recall from 49 of 50 games to 0 and the right transforms from 12 to 0
RESTATE = ("I begin this reply's reasoning by writing out the recall above word for word, from \"This is task\" to "
           "the end of the plan, and then I act.")


def card_text(task: str, cards: dict[str, dict], aliases: Any) -> str:
    c = cards[task]
    return CARD.format(alias=aliases.alias(task), rule=c["rule"].strip(), program=c["program"].rstrip() + "\n")


def load_cards(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())


def fresh_pairs(task: str, users: list[int], demos: int = 3) -> list[Any]:
    """Each user's test pair and its first ``demos`` demonstration pairs of ``task``'s stream instance."""
    from continual_arc.tasks import TaskRegistry
    registry, pairs = TaskRegistry.default(), []
    for u in users:
        slot = next(s for s in loop.stream_slots(0, user=u) if s["task_id"] == task)
        registry.assign(task, slot["source"])
        test = registry.test_instance(task, slot["instance_seed"])
        pairs += [test, *registry.demos(task, slot["instance_seed"], 0, demos, test)]
    return pairs


# -- cards ---------------------------------------------------------------------------------------------------------

def cmd_cards(a: argparse.Namespace) -> None:
    from continual_arc.aliases import TaskAliases
    from probe import check_program
    cards = load_cards(Path(a.cards))
    aliases = TaskAliases(salt=Path(a.salt).read_text().strip(), style="hex")
    c1 = Path(a.c1).read_text().strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    for task in TASKS:
        if a.check_users:
            pairs = fresh_pairs(task, list(range(a.check_users)))
            ok = check_program(cards[task]["program"], pairs)["ok"]
            checks[task] = f"{sum(ok)}/{len(ok)}"
            if not all(ok):
                raise SystemExit(f"{task}: the card's program reproduces {checks[task]} fresh pairs")
        text = card_text(task, cards, aliases)
        (out / f"card-{task}.md").write_text(text + "\n")
        (out / f"teacher-{task}.md").write_text(c1 + "\n\n" + text + "\n")
    print(json.dumps({"out": str(out), "aliases": {t: aliases.alias(t) for t in TASKS}, "checked": checks}))


# -- schedule ------------------------------------------------------------------------------------------------------

def cmd_schedule(a: argparse.Namespace) -> None:
    from continual_arc.schedule import Schedule, Slot, interleave_users
    tasks = set(a.tasks or TASKS)
    picked = [s for u in a.users for s in loop.stream_slots(0, user=u) if s["task_id"] in tasks]
    per_user = [Schedule(name=f"u{i}", seed=0, slots=[Slot(x["task_id"], x["instance_seed"], "probe")],
                         sources={x["task_id"]: x["source"]}) for i, x in enumerate(picked)]
    sched = interleave_users(per_user, 0, name=Path(a.out).stem)
    sched.meta = {"mode": "probe", "seed": 0, "group": "overfit", "users": a.users, "tasks": sorted(tasks)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sched.save(out)
    print(json.dumps({"out": str(out), "instances": len(picked), "tasks": len({x["task_id"] for x in picked})}))


# -- captured plays ------------------------------------------------------------------------------------------------

def read_docs(run_dir: Path) -> list[tuple[str, dict]]:
    """A play's captured episodes (one a user), with their file names."""
    out = []
    for f in sorted((run_dir / "captured").glob("u*.json.gz")):
        with gzip.open(f, "rt") as fp:
            out.append((f.name, json.load(fp)))
    return out


def write_doc(run_dir: Path, name: str, doc: dict) -> None:
    (run_dir / "captured").mkdir(parents=True, exist_ok=True)
    with gzip.open(run_dir / "captured" / name, "wt") as fp:
        json.dump(doc, fp)


def chat_call(c: dict) -> bool:
    """A completed chat call: a request with messages and a response (not a raw completion, which the model's own code
    can send through the proxy from its sandbox, and which train.load_calls cannot read)."""
    messages = (c.get("request") or {}).get("messages") if isinstance(c.get("request"), dict) else None
    return bool(c.get("response") and isinstance(messages, list) and messages)


def agent_calls(doc: dict) -> list[dict]:
    """The episode's completed model calls made as the agent (train.load_calls's: not prime-agent's summarizer)."""
    import render
    return [c for c in doc["calls"] if chat_call(c) and not render.is_compaction(c)]


def reply_text(call: dict) -> str:
    """A call's reply as text: its reasoning, then its content."""
    choice = (call["response"].get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("reasoning") or message.get("reasoning_content") or "") + "\n" + (message.get("content") or "")


def prefill_start(tokenizer: Any, prompt_ids: list[int], prefill: str) -> int | None:
    """Where ``prefill`` (a reasoning prefill the template appended to the prompt's end) begins in the served prompt
    ids: its own tokens when the prompt ends with them, else the shortest tail that decodes to it; None when no tail
    does."""
    ids = tokenizer(prefill, add_special_tokens=False)["input_ids"]
    if ids and prompt_ids[-len(ids):] == ids:
        return len(prompt_ids) - len(ids)
    for i in range(len(prompt_ids) - 1, max(-1, len(prompt_ids) - len(ids) - 16), -1):
        if tokenizer.decode(prompt_ids[i:], skip_special_tokens=False) == prefill:
            return i
    return None


# the environment's answer to a reply it rejected, as the next call's last message opens or holds it
REJECTED = ("I could not find a valid action", "Your submission was INCORRECT")


def last_message_text(call: dict) -> str:
    content = call["request"]["messages"][-1].get("content")
    if isinstance(content, list):
        return "\n".join(p.get("text") or "" for p in content if isinstance(p, dict))
    return content or ""


def rejected(calls: list[dict]) -> list[bool]:
    """For each call, whether the environment rejected its reply: the next call opens with a reply that had no valid
    action, or reports a wrong submission."""
    nxt = [last_message_text(c) for c in calls[1:]] + [""]
    return [any(r in text for r in REJECTED) for text in nxt]


def moved_boundary(tokenizer: Any, doc: dict, card: str, stats: collections.Counter) -> dict:
    """An episode played with a reasoning prefill as one to imitate without it: every agent call's prompt is the served
    prompt with the prefill cut off, and its reply the teacher's continuation, opened on the episode's first agent
    call by ``card`` as the reply's own reasoning (the student learns to write the recall there, and only there). A
    call whose prefill cannot be found in its prompt ids is dropped and counted, and the whole episode when that call is
    its first (the card would otherwise open a later call). A reply the environment rejected (no valid action, or a
    wrong submission) is no target: its call is dropped and counted, and when it is the first call the episode has no
    card target; the later calls keep it in their history, so the recovery after it is still learned."""
    from ask_sd.calls import THOUGHT_SEP
    card_ids = tokenizer(card.strip() + THOUGHT_SEP, add_special_tokens=False)["input_ids"]
    out = {k: v for k, v in doc.items() if k != "calls"}
    out["calls"] = []
    calls = agent_calls(doc)
    bad = rejected(calls)
    for i, c in enumerate(calls):
        first = i == 0
        kwargs = c["request"].get("chat_template_kwargs") or {}
        prefill = kwargs.get("reasoning_prefill")
        prompt = c["response"]["prompt_token_ids"]
        cut = prefill_start(tokenizer, prompt, prefill) if prefill else None
        if cut is None:
            stats["no_prefill" if not prefill else "no_boundary"] += 1
            if first:
                stats["episodes_without_first_call"] += 1
                out["calls"] = []
                return out
            continue
        if bad[i]:
            stats["rejected_left_out"] += 1
            stats["first_call_rejected"] += int(first)
            continue
        call = copy.deepcopy(c)
        call["request"]["chat_template_kwargs"] = {k: v for k, v in kwargs.items() if k != "reasoning_prefill"}
        call["response"]["prompt_token_ids"] = prompt[:cut]
        choice = call["response"]["choices"][0]
        if first:
            choice["token_ids"] = card_ids + list(choice["token_ids"])
            message = choice.get("message") or {}
            key = "reasoning" if "reasoning" in message else "reasoning_content"
            message[key] = card.strip() + THOUGHT_SEP + (message.get(key) or "")
            stats["calls_with_card"] += 1
        call["moved_boundary"] = {"cut": cut, "card": first}
        out["calls"].append(call)
        stats["calls"] += 1
    return out


def cmd_sft(a: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    cards = {t: Path(a.cards_dir, f"card-{t}.md").read_text() for t in TASKS}
    out = Path(a.out)
    stats: collections.Counter = collections.Counter()
    for run in a.runs:
        run_dir = Path(run)
        for name, doc in read_docs(run_dir):
            stats["episodes"] += 1
            if not doc.get("solved"):
                stats["unsolved_left_out"] += 1
                continue
            moved = moved_boundary(tokenizer, doc, cards[doc["task_id"]], stats)
            if moved["calls"]:
                stats["episodes_used"] += 1
                stats[f"episodes_used:{doc['task_id']}"] += 1
                write_doc(out, f"u-{run_dir.name}-{name}", moved)
    (out / "sft.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def cmd_thoughts(a: argparse.Namespace) -> None:
    cards = {t: Path(a.cards_dir, f"card-{t}.md").read_text().strip() for t in TASKS}
    out = Path(a.out)
    n = 0
    for run in a.runs:
        run_dir = Path(run)
        for name, doc in read_docs(run_dir):
            doc = copy.deepcopy(doc)
            doc["calls"] = [c for c in doc["calls"] if chat_call(c)]
            first = agent_calls(doc)[:1]
            for c in doc["calls"]:
                card = cards[doc["task_id"]]
                c["thought"] = card + "\n\n" + RESTATE if a.restate and any(c is f for f in first) else card
                n += 1
            write_doc(out, f"u-{run_dir.name}-{name}", doc)
    print(json.dumps({"out": str(out), "calls": n}))


# -- report --------------------------------------------------------------------------------------------------------

def play_stats(run_dir: Path) -> dict[str, Any]:
    """A play's records (solves, cost, demo requests, wrong submissions) and, from its captured calls, how many
    episodes stated the recall in their first agent call's reply and in any reply, by task and over all."""
    recs = [json.loads(line) for line in open(run_dir / "records.jsonl")]
    first_recall, any_recall = collections.Counter(), collections.Counter()
    if (run_dir / "captured").exists():
        for _, doc in read_docs(run_dir):
            calls = agent_calls(doc)
            if calls and RECALL_MARK in reply_text(calls[0]):
                first_recall[doc["task_id"]] += 1
            if any(RECALL_MARK in reply_text(c) for c in calls):
                any_recall[doc["task_id"]] += 1

    def row(rs: list[dict], task: str | None) -> dict[str, Any]:
        n = max(1, len(rs))
        pick = (lambda c: c[task]) if task else (lambda c: sum(c.values()))
        return {"n": len(rs), "solved": sum(bool(r["solved"]) for r in rs), "cost": round(sum(r["cost"] for r in rs) / n, 2),
                "demos": round(sum(r["demo_requests"] for r in rs) / n, 2),
                "wrong": round(sum(r["wrong_attempts"] for r in rs) / n, 2),
                "recall_first_call": pick(first_recall), "recall_any_call": pick(any_recall)}
    by_task = collections.defaultdict(list)
    for r in recs:
        by_task[r["task_id"]].append(r)
    return {"all": row(recs, None), "tasks": {t: row(rs, t) for t, rs in sorted(by_task.items())}}


def arm_of(play: str) -> str:
    """A play's arm: its name without a trailing task id (the teacher plays one task a play)."""
    head, _, tail = play.rpartition("-")
    return head if tail in TASKS else play


def cmd_report(a: argparse.Namespace) -> None:
    runs = Path(a.runs)
    arms: dict[str, list[dict]] = collections.defaultdict(list)
    for d in sorted(p for p in runs.iterdir() if (p / "records.jsonl").exists()):
        arms[arm_of(d.name)].append(play_stats(d))
    cols = ("n", "solved", "cost", "demos", "wrong", "recall_first_call", "recall_any_call")

    def merged(stats: list[dict], task: str | None) -> dict[str, float]:
        rows = [s["all"] if task is None else s["tasks"][task] for s in stats if task is None or task in s["tasks"]]
        n = sum(r["n"] for r in rows)
        out = {c: sum(r[c] for r in rows) for c in ("n", "solved", "recall_first_call", "recall_any_call")}
        for c in ("cost", "demos", "wrong"):
            out[c] = round(sum(r[c] * r["n"] for r in rows) / max(1, n), 2)
        return out
    order = [arm for arm in a.arms if arm in arms] + sorted(set(arms) - set(a.arms or []))
    lines = ["# Recall overfit: plays", "", "| Arm | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for arm in order:
        m = merged(arms[arm], None)
        lines.append(f"| {arm} | " + " | ".join(str(m[c]) for c in cols) + " |")
    lines += ["", "## Solved by task", "", "| Arm | " + " | ".join(TASKS) + " |", "|" + "---|" * (len(TASKS) + 1)]
    for arm in order:
        cells = []
        for t in TASKS:
            m = merged(arms[arm], t)
            cells.append(f"{m['solved']}/{m['n']}" if m["n"] else "")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    Path(a.out).write_text(text)
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cards")
    s.add_argument("--cards", default=str(HERE / "overfit_cards.json"))
    s.add_argument("--salt", required=True, help="the alias-salt every play of the run takes (--alias-from)")
    s.add_argument("--c1", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--check-users", type=int, default=100,
                   help="check each program on these many stream users' test pair and 3 demos first (0: no check)")
    s.set_defaults(fn=cmd_cards)
    s = sub.add_parser("schedule")
    s.add_argument("--users", type=int, nargs="+", required=True)
    s.add_argument("--tasks", nargs="+", default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_schedule)
    s = sub.add_parser("sft")
    s.add_argument("runs", nargs="+", help="the teacher's plays (with a reasoning prefill at every call)")
    s.add_argument("--cards-dir", required=True)
    s.add_argument("--model", default=loop.MODEL)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sft)
    s = sub.add_parser("thoughts")
    s.add_argument("runs", nargs="+")
    s.add_argument("--cards-dir", required=True)
    s.add_argument("--restate", action="store_true", help="the first call's thought asks for the recall (RESTATE)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_thoughts)
    s = sub.add_parser("report")
    s.add_argument("--runs", required=True)
    s.add_argument("--arms", nargs="*", default=[], help="the arms' order in the tables")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
