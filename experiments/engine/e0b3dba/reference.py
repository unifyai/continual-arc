"""Reference rows for the RL graph (the author, 19 Oct 2026): the paper's frozen rows, h (harness memory kept) and h0 (wiped),
and hpl, row h told the program library's procedure, all played exactly as :func:`paper.play_round` plays them -- the
same stream cut a round at a time, the same harness with its compaction point and call budgets (``loop.HARNESS``,
``loop.HARNESS_BUDGETS["block"]``), the same 131,072-token window and 32,768-token replies -- but with the model served
by OpenRouter instead of our own vLLM, so that no GPU and no machine of the trained rows' is touched (the author, 19 Oct 2026:
nothing may slow the trained rows down).

Row hpl is row h with baselines' ``program_library`` (the paper's ablation row HPL, as GPT-6 Luna's harnesses play it):
the rules gain ``PROGRAM_LIBRARY_RULES``, a program per task in the workspace's ``programs`` folder, run first when the
task returns, checked on every pair and saved after the verdict, and every verdict is followed by
``PROGRAM_LIBRARY_ASK``. The trained rows' oracle labels under that same procedure (``oracle_hint.library_rules``) while
their student is never told it, so qwen's hpl is what telling the untrained student gets, beside what training on the
told oracle's replies gets (the author, 19 Oct 2026).

Two players:

  glm   GLM-5.3-flash, the trained rows' oracle, at the labeller's own settings (expert.py through realtime.sh's relay:
        medium effort; gmicloud, parasail and fireworks in that order, no fallback; the provider's own sampling),
        playing the games itself: where the oracle's own play sits beside the students it teaches.
  qwen  Qwen3.5-9B, the trained rows' base model, untrained: rows h and h0 themselves, on Together first (its
        precision unstated), then Venice (fp8), then DeepInfra (bf16, the one OpenRouter provider of it known to be at
        full precision with tool calls), and no other. Speed decides the order: on 19 Oct 2026 Together generated it at
        about 120 tokens a second and Venice answered as fast, while DeepInfra ran at about 11 and refused calls as
        "temporarily rate-limited upstream" on OpenRouter's shared key, so with DeepInfra first a one-user row played
        about a game an hour against the trained rows' 5-10 minutes; SiliconFlow, the other fp8 host, took about 100 s
        a call where Together and Venice took 3. Each captured reply's generation id names the provider that served it
        (OpenRouter's /api/v1/generation). Sampled as ``paper.SAMPLING`` samples it (temperature 1, top_p 1,
        top_k off, which OpenRouter writes as 0) with its thinking on (``loop.TEMPLATE_KWARGS``). Its chat template and
        tool-call parser are the provider's, not serve.sh's: a trained row's first rounds, the base model on our own
        server, are the check that they serve it alike.

Both send a harness's developer prompt as a system one and every earlier reply's reasoning under OpenRouter's
``reasoning`` key, as the trained rows' proxies do for vLLM (carc_stream.team's VLLM_ROLES and THINKING_FIELD), and drop
prime-agent's own effort (xhigh unless set), so that the player's setting is the one that applies.

A row can also play at another compaction point (``--compact-at``/``--compact-keep``, :func:`loop.harness_of`; 0 leaves
prime-agent's own default) with every other setting unchanged: the probe of what the trained rows' 32,768-token point,
keeping 8,000, costs and gives against later ones for today's student (the author, 19 Oct 2026), as the 25 Sep probe did for
K2 Horizon (continual-arc-baselines docs/RUNNING.md, "shortening K2's context through prime-agent").

Each player also plays its demo-cache floor (``floor``; reference.sh's ROW=dc): the paper's row F demo-cache, the bare
model with the demonstrations it bought for a task shown again when the task returns, at the player's own request
settings and reply cap, on the rows' own stream at any number of users (``paper.stream_cut``; at one user the canonical
``arc1-small``, workstreams, seed 0), round by round as the rows play. Every user's bare model reads one cache for the
team (baselines' ``BareSharedDemoCacheLearner``), a round's purchases joining it at the round's end and written to the
run's ``demo-cache/`` (:func:`play_floor_round`), since the share learned's perfect memory charges a task once in the
whole stream, whichever user met it (20 Oct 2026). It is the zero of the share learned
(``continual_arc.metrics.share_learned``, the paper's headline since 19 Oct 2026): a row is read against the floor of
the model that plays it, so glm's row h against glm's floor, and the DAgger, OPSD and PG rows and qwen's rows against
qwen's, each at its own number of users: a trained row's teacher is training infrastructure, its cost in the
training-compute column (the author, 19 Oct 2026).

Usage (on the reference machine, from ~/continual-arc-baselines, OPENROUTER_API_KEY in its .env; reference.sh keeps it
going): uv run --no-sync --env-file .env python ~/loop/reference.py play --player glm --row h \\
    --run-dir ~/runs/paper/glm-h-u1/play --users 1 --rounds 200
  plays every round of the stream not yet in the run's records.jsonl, one call of play_round a round, as paper.sh does
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import loop
import paper

OPENROUTER = "https://openrouter.ai/api/v1"
# the trained rows' window and reply cap, which paper.play_round writes out as literals
CONTEXT_WINDOW, MAX_TOKENS = 131_072, 32_768
# each player's model on OpenRouter and the request fields that replace the harness's own
PLAYERS: dict[str, dict[str, Any]] = {
    "glm": {"model": "z-ai/glm-5.3-flash", "body": {
        "reasoning": {"effort": "medium"},
        "provider": {"order": ["gmicloud", "parasail", "fireworks"], "allow_fallbacks": False}}},
    "qwen": {"model": "qwen/qwen3.5-9b", "body": {
        **paper.SAMPLING, "top_k": 0, "reasoning": {"enabled": True},
        "provider": {"order": ["together", "venice", "deepinfra"], "allow_fallbacks": False}}},
}
# each reference row's additions to paper.play_round's harness settings (hpl: the module's docstring)
REFERENCE_ROWS: dict[str, dict[str, Any]] = {"h": {}, "h0": {}, "hpl": {"program_library": True}}
# the settings every paper floor ran with (api-launch.sh's floors), for the demo-cache floor (the module's docstring)
FLOOR_KWARGS: dict[str, Any] = {"end": "finish", "empty_reply": "reprompt", "archive_wipes": True}


def play_round(player: str, row: str, run_dir: Path, schedule: Any, *, api_key: str, registry: Any = None,
               harness: dict[str, Any] | None = None) -> Any:
    """:func:`paper.play_round` for a frozen ``row`` with ``player`` served by OpenRouter (the module's docstring): plays
    the rounds of ``schedule`` not yet in ``run_dir``'s records and returns the run's ``RunResult``. ``harness`` replaces
    :data:`loop.HARNESS` (a compaction probe's :func:`loop.harness_of`)."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner

    if row not in REFERENCE_ROWS:
        raise ValueError(f"a reference row is a frozen one, {tuple(REFERENCE_ROWS)}, not {row!r}")
    spec = PLAYERS[player]
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = loop.HARNESS_BUDGETS["block"]
    harness = harness or loop.HARNESS
    kwargs = {"model": spec["model"], "context_window": CONTEXT_WINDOW, "max_tokens": MAX_TOKENS,
              "stateless": row in paper.STATELESS_ROWS, **harness, **budget, **REFERENCE_ROWS[row]}
    overrides = {"model": spec["model"], **spec["body"]}
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": row, "player": player, "served": f"openrouter:{spec['model']}", "stateless": row in paper.STATELESS_ROWS,
         "model": spec["model"], "harness": {**harness, **REFERENCE_ROWS[row]},
         "harness_budget": {"name": "block", **budget}, "overrides": overrides}, indent=1))
    learner = CollectiveHarnessLearner(lambda user: PrimeAgentLearner(**kwargs), run_dir, schedule=schedule,
                                       upstream=OPENROUTER, api_key=api_key, body_overrides=overrides,
                                       body_drop=("reasoning_effort",),
                                       role_map={"developer": "system"}, thinking_field="reasoning",
                                       declared_context_length=CONTEXT_WINDOW, capture=True, snapshot=False,
                                       instance_hook=paper.save_instance_hook(run_dir), reset_hook=lambda: None)
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        if row in paper.STATELESS_ROWS:
            paper._wipe_stateless_users(learner, schedule)
        result = run_stream(learner, schedule, EnvConfig(), out_dir=run_dir, env=env, resume=True,
                            learner_name=f"reference-{player}-{row}")
        paper._refresh_meta_protocol(run_dir, result.summary["protocol"])
        paper._report_round_memory(learner, schedule, run_dir)
        return result
    finally:
        learner.close()


def cmd_play(a: argparse.Namespace) -> None:
    from continual_arc.runner import prior_records
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(a.run_dir).expanduser()
    key = os.environ["OPENROUTER_API_KEY"]
    harness = loop.harness_of(a.compact_at, a.compact_keep)
    for r in range(len(prior_records(run_dir)) // a.users, a.rounds):
        result = play_round(a.player, a.row, run_dir, paper.stream_cut(r, a.seed, a.users), api_key=key,
                            harness=harness)
        print(json.dumps({"round": r, "records": len(result.records)}), flush=True)


def play_floor_round(player: str, run_dir: Path, schedule: Any, *, api_key: str, rounds_done: int,
                     registry: Any = None) -> Any:
    """``player``'s demo-cache floor (the module's docstring) over the rounds of ``schedule`` not yet in ``run_dir``'s
    records, the first ``rounds_done`` of them in its records already: every user's bare model at the floor's settings
    (:data:`FLOOR_KWARGS`, the player's reply cap) behind a proxy of their own taking the player's request settings as
    its defaults, as the CLI's ``--llm-defaults`` gave them, and one cache for the team opened on the run's
    ``demo-cache/`` after those rounds (``SharedDemoCache``), whose round ends with the round. Returns the run's
    ``RunResult``."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.systems.bare import BareSharedDemoCacheLearner, SharedDemoCache

    spec = PLAYERS[player]
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = SharedDemoCache(run_dir / "demo-cache", rounds_done)
    floor = {"max_tokens": MAX_TOKENS, **FLOOR_KWARGS}
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": "dc", "player": player, "served": f"openrouter:{spec['model']}", "model": spec["model"],
         "learner": "continual_arc_baselines.systems.bare:BareSharedDemoCacheLearner", "floor": floor,
         "defaults": spec["body"]}, indent=1))
    learner = CollectiveHarnessLearner(
        lambda user: BareSharedDemoCacheLearner(spec["model"], shared=cache, user=user, **floor), run_dir,
        schedule=schedule, upstream=OPENROUTER, api_key=api_key, body_defaults=spec["body"], capture=False,
        snapshot=False, round_hook=lambda data: cache.end_round(), reset_hook=cache.clear)
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        return run_stream(learner, schedule, EnvConfig(), out_dir=run_dir, env=env, resume=True,
                          learner_name=f"floor-{player}-dc")
    finally:
        learner.close()


def cmd_floor(a: argparse.Namespace) -> None:
    from continual_arc.runner import prior_records
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(a.run_dir).expanduser()
    key = os.environ["OPENROUTER_API_KEY"]
    for r in range(len(prior_records(run_dir)) // a.users, a.rounds):
        result = play_floor_round(a.player, run_dir, paper.stream_cut(r, a.seed, a.users), api_key=key,
                                  rounds_done=r)
        print(json.dumps({"round": r, "records": len(result.records)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    play = sub.add_parser("play", help="play every round of a reference row's stream not yet played, resumable")
    play.add_argument("--player", required=True, choices=PLAYERS)
    play.add_argument("--row", required=True, choices=REFERENCE_ROWS)
    play.add_argument("--run-dir", required=True, dest="run_dir")
    play.add_argument("--users", type=int, default=1)
    play.add_argument("--rounds", type=int, default=200, help="the stream's rounds, all 200 by default")
    play.add_argument("--seed", type=int, default=paper.SEED)
    play.add_argument("--compact-at", type=int, default=loop.HARNESS["compact_at"], dest="compact_at",
                      help="prime-agent's compaction point in tokens (0: its own default); the trained rows' by default")
    play.add_argument("--compact-keep", type=int, default=loop.HARNESS["compact_keep"], dest="compact_keep",
                      help="what a compaction keeps word for word (0: its own default); the trained rows' by default")
    play.set_defaults(fn=cmd_play)
    floor = sub.add_parser("floor", help="play a player's demo-cache floor on the rows' stream, resumable")
    floor.add_argument("--player", required=True, choices=PLAYERS)
    floor.add_argument("--run-dir", required=True, dest="run_dir")
    floor.add_argument("--users", type=int, default=1)
    floor.add_argument("--rounds", type=int, default=200, help="the stream's rounds, all 200 by default")
    floor.add_argument("--seed", type=int, default=paper.SEED)
    floor.set_defaults(fn=cmd_floor)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
