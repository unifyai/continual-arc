"""The player: a row's rounds through the benchmark's runner, one round a call, on the node's model server, and the
settings every row plays under.

Rows (``ROWS``):

  h0      the base model in prime-agent, its harness wiped before every instance (the learner reset, transfer's
          reference on the held-out tasks)
  h0c1    row h0 told c1, the page of playing rules DAgger's oracle labels under (:data:`TOLD_ROWS`), at the start of
          every conversation: what the page alone gives the untrained student, beside what training on replies written
          under it carries
  bare    the bare model with no harness (the baselines' ``BareModelLearner`` at the demo-cache floors' settings,
          :data:`FLOOR`), which keeps nothing from one instance to the next: the stateless floor
  dagger, opsd, pg  the trained rows, prime-agent's memory kept from round to round; the real-time driver plays their
          rounds (continual_arc_training.realtime), each on the adapter its round is served

Every harness row plays under prime-agent with the finish protocol, compacting at 48k tokens and keeping 8k
(:data:`HARNESS`), a budget of 40 model calls per action and no seconds (:data:`TURNS`), a 131,072-token window and
65,536-token replies, sampled at the model card's setting for thinking mode (:data:`SAMPLING`), thinking on.

:func:`play_round` plays a cut of the schedule (:func:`schedule_cut`: ``Schedule.first_rounds(r + 1)``, a literal
prefix) with the runner's own resume: every whole round already in the run directory's ``records.jsonl`` is restored and
the run goes on from the next, so a call for round r plays exactly that round, and a second call for it does nothing.
The run directory is one benchmark run (``records.jsonl``, ``schedule.json``, ``meta.json``, ``summary.json``), while a
round's games are kept under ``rounds/rNNNN/captured/`` (:func:`round_dir`): every user plays every round under the same
user id, and a round trains on its own games alone.

The schedule (:func:`full_schedule`) is ``arc1-small`` served to its users, workstreams order, seed 0: the stream's 200
rounds, then the epilogue's 100, in which every user meets the 100 held-out tasks once each. The held-out test's
instances are user 0's of the epilogue, so a row that keeps nothing between instances plays the held-out instances by
playing the epilogue alone (:func:`epilogue_cut`), dealt wider than its users to a round (:func:`wide_epilogue_cut`).

Run as ``python -m continual_arc_training.play --row ROW --round R ...`` (paper.sh) it plays one round.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_KWARGS = {"enable_thinking": True}   # the model card's thinking mode, at every call
STREAM_CONFIG = "arc1-small"
SEED = 0   # workstreams, seed 0: the schedule every row plays
# The users the whole schedule is built for when none is given: 16, as many as can each bring arc1-small's number of
# tasks of their own from the 400 (continual_arc.configs.user_tasks). Users 0..N-1 of any larger number are exactly N
# users' own, round by round (:func:`served`), so a run at 1, 2 or 4 users plays the first users of this schedule.
SCHEDULE_USERS = 16
# prime-agent's settings: the finish protocol, an empty reply asked again in the same conversation (a restart loses
# the conversation), and compaction at 48k tokens keeping 8k, where the untrained Qwen3.5-9B solved 16 of 48 games
# against 10 at 32k, every call still inside the trainer's 65,536 tokens.
HARNESS = {"compact_at": 49_152, "compact_keep": 8_000, "end": "finish", "empty_reply": "reprompt"}
# The action budget, counted in turns: 40 model calls per action and 6 in its follow-up, with no seconds (NEVER_S), so
# that no score depends on how fast the model is served; under budgets in seconds the same adapter's score moved with
# the serving layout. TURN_LIVENESS_S, six hours, is the harness's hard ceiling on one turn, which aborts a turn that
# has hung.
NEVER_S, TURN_LIVENESS_S = 30 * 86_400.0, 6 * 3_600.0
TURNS = {"max_model_calls": 40, "reprompt_max_model_calls": 6, "action_budget_s": NEVER_S,
         "reprompt_budget_s": NEVER_S, "timeout_s": TURN_LIVENESS_S}
# Qwen3.5-9B's model card for thinking mode on general tasks, forced on every call of every row the node serves, trained
# or not: at temperature 1 with nothing cut, a trained adapter's long replies ran on into incoherent text up to the reply
# cap, which they never did at the card's setting
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
# The served window (scripts/vllm-serve.sh's MAX_MODEL_LEN), declared to the harness as its model's, and the reply cap:
# the longest reply that fits in the window above the harness's compaction point with REPLY_HEADROOM to spare for the
# newest observation (131,072 - 49,152 - 16,384 = 65,536). A harness row's play sets it in the request itself
# (max_completion_tokens), since prime-agent asks for at most 32,000 whatever its own setting, and a reply cut at the cap
# carries no action. vLLM refuses a call whose prompt and budget together pass the window, so the proxy sends such a
# call again without its budget and the server gives the reply the room left (continual_arc_baselines.proxy).
CONTEXT_WINDOW, REPLY_HEADROOM = 131_072, 16_384
MAX_TOKENS = CONTEXT_WINDOW - HARNESS["compact_at"] - REPLY_HEADROOM
# The bare model's settings, every floor's: the harness rows' rules (end "finish"), an empty reply asked again, a wipe
# archived first
FLOOR = {"end": "finish", "empty_reply": "reprompt", "archive_wipes": True}
ROWS = ("h0", "h0c1", "bare", "dagger", "opsd", "pg")
STATELESS_ROWS = ("h0", "h0c1")
# the rows told a page of playing rules (rules/NAME.md) at the start of every conversation, as the baselines'
# ``rules_page``: h0c1 reads c1, the page DAgger's oracle is told, which the trained student never reads
TOLD_ROWS = {"h0c1": "c1"}
# rows that keep nothing between instances, whose held-out instances may be dealt wider than the run's users
TRANSFER_REFERENCES = ("h0", "bare")
# a captured call's fields, as a game is saved
CALL_FIELDS = ("call_id", "turn", "phase", "status", "finish_reason", "request", "response")


def rules_page(name: str) -> str:
    """rules/``name``.md, a page of playing rules (c1 is the one DAgger's oracle is told)."""
    return (Path(__file__).resolve().parent / "rules" / f"{name}.md").read_text().strip()


# -- schedules --------------------------------------------------------------------------------------------------

def served(users: int = SCHEDULE_USERS) -> Any:
    """``arc1-small`` served to ``users`` people at once, each over tasks of their own. User u's tasks and stream
    depend only on the seed and u, so the first N users of any larger number are exactly N users' own."""
    from continual_arc.configs import with_users
    return with_users(STREAM_CONFIG, users)


def full_schedule(seed: int = SEED, users: int = SCHEDULE_USERS) -> Any:
    """The whole schedule: the workstreams stream's 200 rounds of ``users`` instances, then the epilogue's 100."""
    from continual_arc.configs import build_schedule
    return build_schedule(served(users), seed)


def schedule_cut(round_: int, seed: int = SEED, users: int = SCHEDULE_USERS) -> Any:
    """The schedule cut at ``round_ + 1`` rounds: a literal prefix of :func:`full_schedule` (a shorter schedule built
    afresh would not be its prefix), so resuming under it after round ``round_ - 1`` plays exactly round ``round_``."""
    return full_schedule(seed, users).first_rounds(round_ + 1)


def epilogue_cut(round_: int, epilogue_users: int, seed: int = SEED, users: int = SCHEDULE_USERS) -> Any:
    """The epilogue alone, served to the first ``epilogue_users`` of ``users``, in the order and at the instance seeds
    the whole schedule's own epilogue gives them, cut at ``round_ + 1`` rounds."""
    from continual_arc.configs import epilogue_for_users
    return epilogue_for_users(served(users), seed, epilogue_users).first_rounds(round_ + 1)


def wide_epilogue_cut(round_: int, width: int, epilogue_users: int, seed: int = SEED,
                      users: int = SCHEDULE_USERS) -> Any:
    """:func:`epilogue_cut`'s instances, each the same task at the same instance seed, dealt ``width`` at a time into
    rounds of their own, each instance a user of its own, cut at ``round_ + 1`` rounds. A row that keeps nothing between
    instances meets each as it would in the epilogue, so dealing them wider changes only how many play at once: users
    0-3's 400 instances are 4 rounds of 100. A record's user is its instance's place in its round; its task and instance
    seed, which transfer matches on, are the epilogue's."""
    from continual_arc.configs import epilogue_for_users
    from continual_arc.schedule import Schedule, Slot
    epilogue = epilogue_for_users(served(users), seed, epilogue_users)
    if len(epilogue.slots) % width:
        raise ValueError(f"width {width} does not divide the epilogue's {len(epilogue.slots)} instances into whole "
                         "rounds")
    slots = [Slot(s.task_id, s.instance_seed, s.phase, user=i % width) for i, s in enumerate(epilogue.slots)]
    wide = Schedule(name=f"{epilogue.name}-wide{width}", seed=seed, slots=slots, sources=epilogue.sources,
                    meta={**epilogue.meta, "dealt": {"width": width, "from_users": epilogue_users}}, users=width)
    return wide.first_rounds(round_ + 1)


# -- a round's games ---------------------------------------------------------------------------------------------

def round_dir(run_dir: Path, round_: int) -> Path:
    """Where round ``round_``'s games live under ``run_dir`` (``captured/``, and what the round's training adds)."""
    return Path(run_dir) / "rounds" / f"r{round_:04d}"


def save_instance_hook(run_dir: Path) -> Any:
    """The learner's instance hook: each instance's game saved as it ends to its round's
    ``captured/uNNN.json.gz``, its calls in the order they ended. ``default=str``, and never fatal: a harness's turn
    record can hold values JSON has no form for, and one instance's save failing must not cost the round the rest."""

    def hook(t: Any) -> None:
        captured_dir = round_dir(run_dir, t.round) / "captured"
        captured_dir.mkdir(parents=True, exist_ok=True)
        doc = {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
               "phase": t.phase, "solved": getattr(t, "solved", None), "turns": t.turns,
               "calls": [{k: getattr(c, k) for k in CALL_FIELDS} for c in t.calls]}
        tmp = captured_dir / f"u{t.user:03d}.json.gz.tmp"
        try:
            with gzip.open(tmp, "wt") as fp:
                json.dump(doc, fp, default=str)
            tmp.rename(captured_dir / f"u{t.user:03d}.json.gz")
        except Exception:  # noqa: BLE001
            logging.exception("could not save instance of user %s, round %s", t.user, t.round)

    return hook


def round_recorded(run_dir: Path, schedule: Any) -> bool:
    """Whether ``run_dir``'s records already hold every slot of ``schedule``, so that playing it changes nothing."""
    from continual_arc.runner import prior_records, whole_rounds
    return len(whole_rounds(prior_records(run_dir), schedule)) >= len(schedule.slots)


def wipe_stateless_users(learner: Any, schedule: Any) -> None:
    """Every user's harness wiped before the round (the baselines' ``SystemLearner.reset``: stop if started, archive,
    wipe the state, a fresh transcript). A call plays one instance per user, so the harness's own in-process wipe,
    which is set at the end of the instance a call plays, never fires across calls; this is the wipe before every
    instance. Raises if a user's own reset count did not advance by exactly one."""
    for u in range(schedule.users):
        system = learner._user(u).system
        before = system.stats["resets"]
        system.reset()
        after = system.stats["resets"]
        if after != before + 1:
            raise RuntimeError(f"user {u}'s system was not wiped exactly once before the round (resets {before} "
                               f"-> {after})")


def refresh_meta_protocol(run_dir: Path, protocol: dict[str, Any]) -> None:
    """``meta.json``'s ``protocol`` block as ``summary.json``'s, recomputed by every call: the runner reloads
    ``meta.json`` whole on a resumed call, so its block would stay what the run's first cut saw."""
    path = Path(run_dir) / "meta.json"
    meta = json.loads(path.read_text())
    meta["protocol"] = protocol
    path.write_text(json.dumps(meta, indent=1))


def report_round_memory(learner: Any, schedule: Any, run_dir: Path) -> None:
    """The round's memory-use counters (:func:`continual_arc_training.memory.round_report`), the round being the cut's
    last. A failure here is logged and never costs the round its result."""
    from continual_arc_training import memory
    round_ = schedule.rounds()[-1]
    try:
        report = memory.round_report(run_dir, round_, schedule.users, lambda u: learner._user(u).system.state_dir,
                                     round_dir)
    except Exception:  # noqa: BLE001
        logging.exception("could not compute round %s's memory-use counters", round_)
        return
    logging.info(memory.summary_line(report))


def play_round(row: str, run_dir: Path, schedule: Any, *, base_url: str, served_name: str,
               registry: Any = None) -> Any:
    """Play ``schedule`` into ``run_dir`` as ``row`` (the module docstring), serving ``served_name`` (the base model's
    own id, or the adapter name the server holds the round's adapter under), from ``base_url``: the runner restores
    every whole round already recorded and plays only the rest. ``registry`` is the benchmark's by default. Returns the
    run's ``RunResult``."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env
    from continual_arc.runner import run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.proxy import THINKING_FIELD, VLLM_DROP, VLLM_ROLES

    if row not in ROWS:
        raise ValueError(f"row must be one of {ROWS}, got {row!r}")
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = {"model": served_name, "chat_template_kwargs": dict(TEMPLATE_KWARGS), **SAMPLING}
    if row == "bare":
        from continual_arc_baselines.systems.bare import BareModelLearner

        def system(user: int) -> Any:
            return BareModelLearner(MODEL, max_tokens=MAX_TOKENS, **FLOOR)
        setting: dict[str, Any] = {"floor": FLOOR, "max_tokens": MAX_TOKENS}
    else:
        from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
        told = {"rules_page": rules_page(TOLD_ROWS[row])} if row in TOLD_ROWS else {}
        kwargs = {"model": MODEL, "context_window": CONTEXT_WINDOW, "max_tokens": MAX_TOKENS,
                  "stateless": row in STATELESS_ROWS, **HARNESS, **TURNS, **told}

        def system(user: int) -> Any:
            return PrimeAgentLearner(**kwargs)
        overrides.update(return_token_ids=True, max_completion_tokens=MAX_TOKENS)
        setting = {"harness": dict(HARNESS), "harness_budget": {"name": "turns", **TURNS}, "max_tokens": MAX_TOKENS,
                   **({"told": TOLD_ROWS[row]} if row in TOLD_ROWS else {})}
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": row, "served": served_name, "stateless": row in STATELESS_ROWS, "model": MODEL, **setting,
         "sampling": SAMPLING}, indent=1))
    learner = CollectiveHarnessLearner(system, run_dir, schedule=schedule, upstream=base_url, body_overrides=overrides,
                                       body_drop=VLLM_DROP, role_map=VLLM_ROLES, thinking_field=THINKING_FIELD,
                                       declared_context_length=CONTEXT_WINDOW, capture=True, snapshot=False,
                                       instance_hook=save_instance_hook(run_dir), reset_hook=lambda: None)
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        if row in STATELESS_ROWS:
            wipe_stateless_users(learner, schedule)
        result = run_stream(learner, schedule, EnvConfig(), out_dir=run_dir, env=env, resume=True,
                            learner_name=f"paper-{row}")
        refresh_meta_protocol(run_dir, result.summary["protocol"])
        if row != "bare":   # the bare model has no harness, so no memory to count
            report_round_memory(learner, schedule, run_dir)
    finally:
        learner.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="play one round of a row's schedule, or of the epilogue alone; resumable")
    p.add_argument("--row", required=True, choices=ROWS)
    p.add_argument("--run-dir", required=True, dest="run_dir")
    p.add_argument("--base-url", required=True, dest="base_url", help="the served model's OpenAI-compatible base URL")
    p.add_argument("--served", required=True, help="the name the server serves the model under")
    p.add_argument("--round", type=int, required=True, help="0-indexed: the round this call plays")
    p.add_argument("--epilogue-users", type=int, default=None, dest="epilogue_users",
                   help="play the epilogue alone, served to this many users, instead of the whole schedule")
    p.add_argument("--width", type=int, default=None,
                   help="with --epilogue-users, for h0 and bare: the epilogue's instances dealt this many to a round")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--users", type=int, default=SCHEDULE_USERS, help="how many users the schedule is served to")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.width is not None:
        if a.row not in TRANSFER_REFERENCES or a.epilogue_users is None:
            raise SystemExit(f"--width deals the epilogue's instances wider: only with --epilogue-users and for a row "
                             f"that keeps nothing between instances, {TRANSFER_REFERENCES}, not {a.row}")
        schedule = wide_epilogue_cut(a.round, a.width, a.epilogue_users, a.seed, a.users)
    elif a.epilogue_users is not None:
        schedule = epilogue_cut(a.round, a.epilogue_users, a.seed, a.users)
    else:
        schedule = schedule_cut(a.round, a.seed, a.users)
    run_dir = Path(a.run_dir).expanduser()
    result = play_round(a.row, run_dir, schedule, base_url=a.base_url, served_name=a.served)
    print(json.dumps({"out": str(run_dir), "round": a.round, "records": len(result.records),
                      "summary": result.summary}, default=str))


if __name__ == "__main__":
    main()
