"""Replays of the frozen-test games the context overrun cut (continual-arc-paper's
notes/readings/context-overrun-2026-11-04.txt): the games of a run's frozen test, or of a reference's, in which our vLLM
server refused a call because its prompt and the 65,536-token reply it asked (paper.MAX_TOKENS) together passed the
server's 131,072-token window, each played again from where the refusal cut it, on the harness with the fix: the proxy
sends such a call once more without the budget, so that the server gives the reply the room the window has left, as
the release's proxy does (a copy of the node's harness package with that change, put ahead of the installed one on
the Python path, never ~/continual-arc-baselines itself, which a running stream imports). The author, 4 Nov 2026: fix the
bug, replay the cut games, and read the tests as done correctly.

A test's replays are one round of their own, beside the test's play and never in it: ``<test>.replay/play`` under the
run's ``frozen/``, where TEST is the play's folder (held-out, seen-2, held-out-reference, ...). The round has one user
for each cut game, numbered from 0 in the order of the test's users, and each starts as the test's user did and plays
its task instance: a fork of what stream user i mod U ended the stream with, i its user in the test (frozen.py's
seeding, with the test's own commits and renaming, read from its frozen.json), or nothing for a reference played from
nothing. Every game is replayed through ``loop.replay_proxy`` from its recording: each model call that began before
the game's first refused call is served as recorded, the environment answering as it did since its instances are
seeded, and from the refused call on the game is the model's, on the fixed harness (the plan's ``cut``: the game is
handed to the model where its recording ends). A replay that drifts from its recording before then is played on by
the model from where it drifted. Either way the round's ``replay.jsonl`` says where each game went live. So a
replayed game is the game the test would have played had its harness not refused the call, the calls before it the
very ones it made; continual-arc-paper's figures read each in place of the cut game, by test, task and instance.

Run on the node from ~/prime-rl, as frozen.sh runs frozen.py, with the run's code and the fixed harness on the path
(PYTHONPATH=~/loop:~/overrun-fix/src) and the node's own frozen.py (~/frozen-tools) beside this file:
  plan  replay.py plan --test PLAY --users U[,U...] --out DIR
        PLAY is the test's play folder, holding its schedule.json, records.jsonl, frozen.json and driver.json, and of
        each cut game its user's calls.jsonl and captured game (rounds/r0000/captured/uNNN.json.gz). Writes
        DIR/schedule.json (the cut games' slots, the users numbered from 0), DIR/sources/uNNN.json.gz (each game's
        calls before its first refused call, in the form loop.recorded_calls gives), DIR/plan.jsonl (replay_proxy's
        plan) and DIR/replay.json (what is replayed: by replay user, the test's user, its task and instance, where its
        game was cut and how it ended)
  play  replay.py play --dir DIR --base-url URL [--served NAME --adapter A]
        DIR/play, the round played as frozen.py plays a test (paper.play_round, the run's own code, refused unless its
        settings are the test's driver.json's), under the test's own task names (its alias-salt), the run's stream
        read from where PLAY's run is (PLAY/../../..), and its last phase's adapter served under the test's own name
        unless another is given; refused once played if any game was served none of its recorded agent calls
  report  replay.py report DIR [DIR...]
        a JSON line per replayed game: how it ended against the cut game, where it went live (replay.jsonl), how many
        of its recorded calls were served, the kinds of those never asked for (prime-agent times its reviews by its own
        clock, which a replay runs ahead of: each /refine pass among them is shown with its recorded reply, since one
        that changed the harness makes the replay's harness differ from the game's), the calls whose prompt alone
        passed the reply budget's room (the fix's), and any still refused (a prompt past the window itself)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frozen

# A call the server refused for the room its reply asked: one of prime-agent's own (its system prompt opens the game's
# conversation, or the compaction summary of it) that failed before the server read a token of its prompt, and at once
# (vLLM answers it 400 in well under a second; a failure the proxy retries takes a minute or more). The reading of 4
# Nov 2026 matched every failed call of the paper's vLLM test plays to the server's answer in the play's log: each of
# prime-agent's own calls failed so was refused for the window, while the rule's other calls were the model's own code
# calling the server's URL from its kernel (other paths answered 404 or 405, bodies it got wrong answered 400), which
# is the game's own doing, cuts nothing and plays again as it did.
REFUSED_WITHIN_MS = 2000
HARNESS_PROMPTS = ("You are a general purpose agent", "You are a context summarization")
WINDOW = 131_072   # the tests' server window (frozen.sh's MAX_MODEL_LEN), which replay.sh serves too
CAPTURED = Path("rounds") / "r0000" / "captured"
FORKS = 16   # users forked at once, as baselines' frozen.seed


def calls_of(play: Path, user: int) -> list[dict[str, Any]]:
    """User ``user``'s model calls in the test ``play``, as its calls.jsonl records them."""
    path = play / "users" / f"u{user:03d}" / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def failed_at_once(call: dict[str, Any]) -> bool:
    """Whether a calls.jsonl row failed before the server read its prompt, at once (:data:`REFUSED_WITHIN_MS`)."""
    tokens = call.get("tokens") or {}
    return (call.get("status") == "failed" and not tokens.get("input")
            and float(call.get("latency_ms") or 0.0) < REFUSED_WITHIN_MS)


def harness_call(request: dict[str, Any] | None) -> bool:
    """Whether a captured request is one of prime-agent's own (:data:`HARNESS_PROMPTS`)."""
    messages = (request or {}).get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return False
    content = messages[0].get("content")
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return isinstance(content, str) and content.startswith(HARNESS_PROMPTS)


def recording(play: Path, user: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """User ``user``'s game in the test ``play`` up to its first refused call, as ``loop.replay_proxy`` serves a
    recording: every chat call that began before it and did not fail, in the order the captured game keeps them (the
    order they ended, as ``loop.recorded_calls`` reads a captured game), each with its ``latency_ms`` and an empty
    ``partial`` (a test plays turn-based, so no turn is cut at a budget of seconds); and where the game was cut."""
    rows = calls_of(play, user)
    timing = {r["call_id"]: r for r in rows}
    with gzip.open(play / CAPTURED / f"u{user:03d}.json.gz", "rt") as fp:
        doc = json.load(fp)
    if doc["user"] != user:
        raise SystemExit(f"{play}'s captured game u{user:03d} is user {doc['user']}'s")
    missing = [c["call_id"] for c in doc["calls"] if c["call_id"] not in timing]
    if missing:
        raise SystemExit(f"{play}, user {user}: captured calls {missing[:3]} are not in its calls.jsonl")
    cut = [timing[c["call_id"]] for c in doc["calls"]
           if failed_at_once(timing[c["call_id"]]) and harness_call(c.get("request"))]
    if not cut:
        raise SystemExit(f"{play}, user {user}: no call of prime-agent's own was refused, so the game was not cut")
    first = min(cut, key=lambda r: r["started"])
    calls = []
    for c in doc["calls"]:
        row = timing[c["call_id"]]
        chat = isinstance((c.get("request") or {}).get("messages"), list)
        if chat and c.get("status") != "failed" and row["started"] < first["started"]:
            calls.append({**c, "latency_ms": row.get("latency_ms"), "partial": ""})
    begun = sorted(r["started"] for r in rows)
    return calls, {"task_id": doc["task_id"], "calls": len(rows), "refused": len(cut),
                   "refused_call": first["call_id"], "refused_at": begun.index(first["started"]),
                   "served": len(calls), "unanswered": sum(1 for c in calls if not (c.get("response") or {}).get("choices"))}


def cmd_plan(o: argparse.Namespace) -> None:
    play, out = Path(o.test).expanduser().resolve(), Path(o.out).expanduser()
    users = sorted({int(u) for u in o.users.split(",")})
    if (out / "play").exists():
        raise SystemExit(f"{out}/play exists: a replay is never planned over one played")
    sched = json.loads((play / "schedule.json").read_text())
    records = {r["step"]: r for r in map(json.loads, (play / "records.jsonl").read_text().splitlines())}
    if len(records) != len(sched["slots"]):
        raise SystemExit(f"{play} records {len(records)} of its {len(sched['slots'])} games: the test is not whole")
    shutil.rmtree(out, ignore_errors=True)
    (out / "sources").mkdir(parents=True)
    slots, plan, games = [], [], []
    for j, u in enumerate(users):
        slot = sched["slots"][u]
        if slot.get("user", 0) != u:
            raise SystemExit(f"{play}'s slot {u} is user {slot.get('user', 0)}'s: a test's slot i is its user i's")
        calls, cut = recording(play, u)
        if cut["task_id"] != slot["task_id"] or records[u]["instance_seed"] != slot["instance_seed"]:
            raise SystemExit(f"{play}, user {u}: its game, record and slot are not of one instance")
        source = out / "sources" / f"u{j:03d}.json.gz"
        with gzip.open(source, "wt", compresslevel=1) as fp:
            json.dump(calls, fp)
        slots.append({k: v for k, v in slot.items() if k != "user"} | ({"user": j} if j else {}))
        plan.append({"user": j, "source": str(source.resolve()), "branch": None, "reply": None, "cut": True,
                     "calls": len(calls)})
        rec = records[u]
        games.append({"user": j, "of": u, "instance_seed": slot["instance_seed"], **cut,
                      "original": {k: rec[k] for k in ("solved", "gave_up", "cost")}})
    doc = {**sched, "name": f"{sched['name']}-replay", "users": len(users), "slots": slots,
           "sources": {s["task_id"]: sched["sources"][s["task_id"]] for s in slots},
           "meta": {**sched["meta"], "replay_of": str(play), "users_of": users}}
    (out / "schedule.json").write_text(json.dumps(doc, indent=1))
    (out / "plan.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plan))
    (out / "replay.json").write_text(json.dumps(
        {"test": str(play), "frozen": json.loads((play / "frozen.json").read_text()),
         "driver": json.loads((play / "driver.json").read_text()), "rule": f"the first of prime-agent's own calls to "
         f"fail with no input tokens, under {REFUSED_WITHIN_MS} ms", "games": games}, indent=1))
    print(json.dumps({"test": str(play), "games": len(games), "served": sum(g["served"] for g in games)}), flush=True)


def fork_user(system: Path, objects: Path, stream: Path, spelt: str, rounds: int, commit: str, found: Any) -> None:
    """baselines' ``frozen.fork_user`` for a stream kept somewhere other than where it played: ``system`` (a replay
    user's ``system/``) a repository on the test's store, holding ``commit``'s tree with every path that named the
    stream user's folder as the stream spelt it (``spelt``) renamed to its own, committed, and the state log of the
    stream user's ``system/`` at ``stream`` up to it."""
    from continual_arc_baselines import frozen as forks
    from continual_arc_baselines.statelog import GIT_ENV, _git, restore
    repo, tree = system / "state.git", system / "state"
    system.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", "--bare", str(repo)], env={**os.environ, **GIT_ENV}, check=True)
    _git(repo, "config", "core.bare", "false")
    _git(repo, "config", "gc.auto", "0")
    shutil.rmtree(repo / "objects")
    (repo / "objects").symlink_to(os.path.relpath(objects, repo.resolve()))
    restore(repo, commit, tree)
    _git(repo, "update-ref", "HEAD", commit)
    _git(repo, "reset", "--quiet", commit, tree=tree)
    lines = [line for line in (stream / "state-log.jsonl").read_text().splitlines() if json.loads(line)["round"] < rounds]
    (system / "state-log.jsonl").write_text("".join(line + "\n" for line in lines))
    forks.relocate(tree, spelt, str(system), found)
    _git(repo, "add", "--all", "--force", ".", tree=tree)
    _git(repo, "commit", "--quiet", "--allow-empty", "--no-verify", "-m", f"relocated from {spelt}", tree=tree)


def seed(run: Path, play_dir: Path, of: list[int], origin: dict[str, Any]) -> dict[str, Any]:
    """``play_dir`` as the replay's round begins, from nothing: the run's task names, and replay user j forked as the
    test forked its user of[j], from stream user of[j] mod U's commit of the stream's last round (the test's
    frozen.json, ``origin``: refused unless the stream kept here holds the same commits and the same files naming its
    folder), and the stream's kept programs and demonstration cache as frozen.py's ``seed`` copies them."""
    from continual_arc_baselines import frozen as forks
    from continual_arc_baselines.statelog import commit_of, restore
    play = frozen.play_of(run)
    systems, rounds = forks.stream_end(play)
    if origin["round"] != rounds - 1 or len(origin["systems"]) != len(systems):
        raise SystemExit(f"{play} is not the stream the test was forked from: {origin['round'] + 1} rounds and "
                         f"{len(origin['systems'])} users there, {rounds} and {len(systems)} here")
    shutil.rmtree(play_dir, ignore_errors=True)
    play_dir.mkdir(parents=True)
    shutil.copy2(play / frozen.SALT_FILE, play_dir / frozen.SALT_FILE)
    objects = forks.object_store(play_dir / "objects.git", systems)
    sources = []
    for system, was in zip(systems, origin["systems"]):
        commit = commit_of(system / "state-log.jsonl", rounds - 1)
        scratch = play_dir / "naming"
        restore(system / "state.git", commit, scratch)
        found = forks.naming(scratch, was["system"])
        shutil.rmtree(scratch)
        if commit != was["commit"] or (len(found[0]) + len(found[1]), len(found[2])) != (was["relocated"],
                                                                                        was["databases"]):
            raise SystemExit(f"{system}: commit {commit} renaming {len(found[0]) + len(found[1])} files and "
                             f"{len(found[2])} databases, where the test forked {was}")
        sources.append((system, was["system"], commit, found))

    def fork(j: int) -> None:
        system, spelt, commit, found = sources[of[j] % len(sources)]
        fork_user(forks.user_system(play_dir, j), objects, system, spelt, rounds, commit, found)

    with ThreadPoolExecutor(FORKS) as pool:
        list(pool.map(fork, range(len(of))))
    last, n = rounds - 1, len(systems)
    if (play / "library").is_dir():
        (play_dir / "library").mkdir()
        for j, u in enumerate(of):
            shutil.copy2(play / "library" / f"r{last:04d}-u{u % n:03d}.json",
                         play_dir / "library" / f"r{last:04d}-u{j:03d}.json")
    if (play / "demo-cache").is_dir():
        shutil.copytree(play / "demo-cache", play_dir / "demo-cache")
    return {"memory": "replay user j a fork of stream user (the test's user of[j]) mod stream_users at its commit of "
                      "the stream's last round, as the test forked it, relocated",
            "from": {**origin, "systems": [{**was, "kept_at": str(s)} for was, (s, *_) in zip(origin["systems"],
                                                                                              sources)]},
            "stream_users": n, "stream_rounds": rounds}


def take_salt(test: Path, play_dir: Path) -> None:
    """The test's own task names in ``play_dir``: its alias-salt. The test's recorded prompts name each task by its
    alias, so under any other salt no call of the agent's matches its recording, and the game would play live from its
    first call while the recording's internal calls, which are matched by kind and never compared, are served into it.
    A test forked from a stream has the stream's salt, which ``seed`` has copied, refused if it is not the test's; a
    reference played from nothing drew one of its own when it played."""
    salt = play_dir / frozen.SALT_FILE
    if salt.exists() and salt.read_text().strip() != (test / frozen.SALT_FILE).read_text().strip():
        raise SystemExit(f"{salt} is not the alias-salt of {test}, whose recorded prompts the replay must match")
    shutil.copy2(test / frozen.SALT_FILE, salt)


def check_settings(driver: dict[str, Any]) -> None:
    """Refuse to replay a test whose settings (its driver.json) the code on the path does not play: its row, sampling,
    harness, budget and reply cap must be this code's own, as must the fix to the window."""
    import loop
    import paper

    import continual_arc_baselines.proxy as proxies
    if not hasattr(proxies, "REPLY_BUDGETS"):
        raise SystemExit(f"{proxies.__file__} has no fix for the window: put the fixed harness ahead on PYTHONPATH")
    budget = driver.get("harness_budget") or {}
    ours = {"row": driver["row"] in paper.ROWS, "sampling": paper.SAMPLING, "max_tokens": paper.MAX_TOKENS,
            "stateless": driver["row"] in paper.STATELESS_ROWS}
    theirs = {"row": True, "sampling": driver["sampling"], "max_tokens": driver["max_tokens"],
              "stateless": driver["stateless"]}
    if "harness" in driver:
        ours["harness"], theirs["harness"] = dict(loop.HARNESS), driver["harness"]
        ours["budget"] = {"name": budget.get("name"), **loop.HARNESS_BUDGETS[budget.get("name")]}
        theirs["budget"] = budget
    if ours != theirs:
        raise SystemExit(f"the code on the path plays {ours}, not the test's {theirs}")


def cmd_play(o: argparse.Namespace) -> None:
    import logging

    import loop
    import paper
    from continual_arc.schedule import Schedule

    import continual_arc_baselines.proxy as proxies
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folder = Path(o.dir).expanduser().resolve()
    doc = json.loads((folder / "replay.json").read_text())
    test, driver, was = Path(doc["test"]), doc["driver"], doc["frozen"]
    schedule = Schedule.load(folder / "schedule.json")
    play_dir = folder / "play"
    if paper.round_recorded(play_dir, schedule):
        check_matched(folder, doc)
        print(json.dumps({"replay": str(folder), "played": True}), flush=True)
        return
    check_settings(driver)
    served = o.served or driver["served"]
    if o.adapter:
        frozen.check_served(o.base_url, served, Path(o.adapter))
    of = [g["of"] for g in doc["games"]]
    if was.get("from"):
        memory = seed(test.parent.parent.parent, play_dir, of, was["from"])
    else:   # a reference played from nothing
        shutil.rmtree(play_dir, ignore_errors=True)
        play_dir.mkdir(parents=True)
        memory = {"memory": was["memory"]}
    take_salt(test, play_dir)
    (play_dir / "frozen.json").write_text(json.dumps(
        {**{k: v for k, v in was.items() if k not in ("from", "started")}, "replay_of": str(test), "users_of": of,
         "users": len(of), "schedule": schedule.name, "served": served, **({"adapter": o.adapter} if o.adapter else {}),
         **memory, "harness": f"the window's fix, {proxies.__file__}", "started": datetime.now(timezone.utc).isoformat()},
        indent=1))
    log = paper.round_dir(play_dir, 0)
    log.mkdir(parents=True, exist_ok=True)
    proxies.LLMProxy = loop.replay_proxy(proxies.LLMProxy, folder / "plan.jsonl", log / "replay.jsonl")
    result = paper.play_round(driver["row"], play_dir, schedule, base_url=o.base_url, served_name=served)
    check_matched(folder, doc)
    print(json.dumps({"replay": str(folder), "records": len(result.records),
                      "solved": sum(r.solved for r in result.records)}), flush=True)


def check_matched(folder: Path, doc: dict[str, Any]) -> None:
    """Refuse a played replay any of whose games was served none of its recorded agent calls: its agent's prompts were
    not its recording's (another alias salt, say), so the game was the model's from its first call, with whatever
    internal calls of the recording replay_proxy served into it, and is no replay of the cut game."""
    import loop
    unmatched = []
    for g in doc["games"]:
        j = g["user"]
        with gzip.open(folder / "sources" / f"u{j:03d}.json.gz", "rt") as fp:
            agent = {c["call_id"] for c in json.load(fp) if loop.call_kind(c["request"].get("messages") or []) == "agent"}
        with gzip.open(folder / "play" / CAPTURED / f"u{j:03d}.json.gz", "rt") as fp:
            served = {(c.get("response") or {}).get("replayed") for c in json.load(fp)["calls"]}
        if agent and not agent & served:
            unmatched.append(j)
    if unmatched:
        raise SystemExit(f"{folder}: replay users {unmatched} were served none of their recorded agent calls")


def cmd_report(o: argparse.Namespace) -> None:
    import collections

    import loop
    import render
    for d in o.dirs:
        folder = Path(d).expanduser().resolve()
        doc = json.loads((folder / "replay.json").read_text())
        play_dir = folder / "play"
        done = {(r["task_id"], r["instance_seed"]): r for r in map(json.loads, (play_dir / "records.jsonl").read_text()
                                                                    .splitlines())} \
            if (play_dir / "records.jsonl").exists() else {}
        events = collections.defaultdict(list)
        if (play_dir / "rounds" / "r0000" / "replay.jsonl").exists():
            for line in (play_dir / "rounds" / "r0000" / "replay.jsonl").read_text().splitlines():
                event = json.loads(line)
                events[event["user"]].append({k: v for k, v in event.items() if k not in ("t", "user")})
        for g in doc["games"]:
            j = g["user"]
            with gzip.open(folder / "sources" / f"u{j:03d}.json.gz", "rt") as fp:
                source = json.load(fp)
            captured = play_dir / CAPTURED / f"u{j:03d}.json.gz"
            calls = []
            if captured.exists():
                with gzip.open(captured, "rt") as fp:
                    calls = json.load(fp)["calls"]
            served = {(c.get("response") or {}).get("replayed") for c in calls} - {None}
            unserved = [c for c in source if c["call_id"] not in served]
            rows = calls_of(play_dir, j) if (play_dir / "users" / f"u{j:03d}" / "calls.jsonl").exists() else []
            timing = {r["call_id"]: r for r in rows}
            refines = [{"call": c["call_id"], "reply": render.text_of(
                ((c.get("response") or {}).get("choices") or [{}])[0].get("message", {}).get("content"))[:300]}
                for c in unserved if loop.call_kind(c["request"].get("messages") or []) == "refine"]
            rec = done.get((g["task_id"], g["instance_seed"]))
            print(json.dumps({
                "replay": str(folder), "user": j, "of": g["of"], "task_id": g["task_id"],
                "instance_seed": g["instance_seed"], "original": g["original"],
                "replayed": {k: rec[k] for k in ("solved", "gave_up", "cost")} if rec else None,
                "events": events.get(j, []), "recorded": len(source), "served": len(served & {c["call_id"] for c in source}),
                "unserved": dict(collections.Counter(loop.call_kind(c["request"].get("messages") or [])
                                                     for c in unserved)), "refines_unserved": refines,
                "past_budget_room": sum(1 for r in rows if ((r.get("tokens") or {}).get("input") or 0)
                                        > WINDOW - doc["driver"]["max_tokens"]),
                "refused": sum(1 for c in calls if c["call_id"] in timing and failed_at_once(timing[c["call_id"]])
                               and harness_call(c.get("request")))}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan", help="the replay of a test's cut games: its schedule, recordings and plan")
    s.add_argument("--test", required=True, help="the test's play folder (RUN/frozen/TEST/play)")
    s.add_argument("--users", required=True, help="the test's users whose games were cut, comma-separated")
    s.add_argument("--out", required=True, help="the replay's folder (RUN/frozen/TEST.replay)")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("play", help="play a planned replay into its folder's play/")
    s.add_argument("--dir", required=True, help="the replay's folder, as planned")
    s.add_argument("--base-url", dest="base_url", required=True, help="the server the model is served on")
    s.add_argument("--served", help="the name it is served under (the test's own unless given)")
    s.add_argument("--adapter", help="the adapter's own folder, for the served-adapter check (none for the base model)")
    s.set_defaults(fn=cmd_play)
    s = sub.add_parser("report", help="each replayed game against its cut one, a JSON line each")
    s.add_argument("dirs", nargs="+", help="replay folders")
    s.set_defaults(fn=cmd_report)
    o = p.parse_args()
    o.fn(o)


if __name__ == "__main__":
    main()
