"""Reference rows for the trained rows: the paper's frozen rows, h (harness memory kept) and h0 (wiped), and hpl,
row h told the program library's procedure, all played exactly as :func:`paper.play_round` plays them -- the same
stream cut a round at a time, the same harness with its compaction point and call budgets (``loop.HARNESS``,
``loop.HARNESS_BUDGETS["block"]``), the same 131,072-token window and 32,768-token replies -- but with the model served
by OpenRouter instead of our own vLLM, so that no GPU and no machine of the trained rows' is touched and nothing can
slow the trained rows down.

Row hpl is row h with baselines' ``program_library`` (the paper's ablation row HPL, as GPT-6 Luna's harnesses play it):
the rules gain ``PROGRAM_LIBRARY_RULES``, a program per task in the workspace's ``programs`` folder, run first when the
task returns, checked on every pair and saved after the verdict, and every verdict is followed by
``PROGRAM_LIBRARY_ASK``. The trained rows' oracle labels under that same procedure (``oracle_hint.library_rules``) while
their student is never told it, so qwen's hpl is what telling the untrained student gets, beside what training on the
told oracle's replies gets.

Row hg is row h with its storage handed to another model: the player plays every instance, and from its verdict on,
the finish protocol's turn in which the harness saves what it wants before it finishes, the row's storer
(:data:`STORERS`) drives the same session, its request fields in place of the player's on the user's own proxy
(:func:`hand_storage`), so it reads the player's whole game and keeps in the harness what it judges will help the player
later. qwen's hg, with glm storing, is the trained rows' teacher passing on what it learns through the harness alone,
beside the trained rows' passing it through the weights: the same student playing every move, the same teacher and the
same stream, only where the teaching is kept differing.

Rows hplg and hgpl give the storer what the trained rows' oracle has, the program library's procedure, which it labels
under (``oracle_hint.library_rules``) while the trained student is never told it: the oracle is told, and anything that
reaches the student comes through the learning algorithm. hplg is row hpl with its storage handed to the storer, so
player and storer both read the procedure. hgpl is row hg with the storer alone told it: the procedure and the ask to
save the task's program, as row hpl's player reads them, follow the verdict in the turn the storer takes
(:func:`library_lesson`), which the player never sees, since every instance is a session of its own. Whatever reaches
hgpl's player comes through what the storer keeps in the harness, as whatever reaches the trained student comes through
its weights.

Rows ha and hpl0 are the paper's ablations HA and HPL0 at the player's settings, which complete Qwen's and GLM's grid at
one user: ha is row h asked after every verdict to save liberally what could help with the task's later instances
(baselines' ``ask_to_store``), and hpl0 is row hpl wiped before every instance (:data:`STATELESS_ROWS`).

Two players:

  glm   GLM-5.3-flash, the trained rows' oracle, at the labeller's own settings (expert.py through realtime.sh's relay:
        medium effort; gmicloud, parasail and fireworks in that order, no fallback; the provider's own sampling),
        playing the games itself: where the oracle's own play sits beside the students it teaches.
  qwen  Qwen3.5-9B, the trained rows' base model, untrained: rows h and h0 themselves, on Together first (its
        precision unstated), then Venice (fp8), then DeepInfra (bf16, the one OpenRouter provider of it known to be at
        full precision with tool calls), and no other. Speed decides the order: when the rows started, Together
        generated it at about 120 tokens a second and Venice answered as fast, while DeepInfra ran at about 11 and
        refused calls as "temporarily rate-limited upstream" on OpenRouter's shared key, so with DeepInfra first a
        one-user row played about a game an hour against the trained rows' 5-10 minutes; SiliconFlow, the other fp8
        host, took about 100 s a call where Together and Venice took 3. Each captured reply's generation id names the
        provider that served it (OpenRouter's /api/v1/generation). Sampled as ``paper.SAMPLING`` samples it
        (temperature 1, top_p 1, top_k off, which OpenRouter writes as 0) with its thinking on
        (``loop.TEMPLATE_KWARGS``). Its chat template and tool-call parser are the provider's, not serve.sh's: a
        trained row's first rounds, the base model on our own server, are the check that they serve it alike.

Both send a harness's developer prompt as a system one and every earlier reply's reasoning under OpenRouter's
``reasoning`` key, as the trained rows' proxies do for vLLM (the baselines proxy's VLLM_ROLES and THINKING_FIELD), and drop
prime-agent's own effort (xhigh unless set), so that the player's setting is the one that applies.

A row can also play at another compaction point (``--compact-at``/``--compact-keep``, :func:`loop.harness_of`; 0 leaves
prime-agent's own default) with every other setting unchanged: the probe of what the trained rows' compaction point
costs and gives against later ones for the untrained student.

Each player also plays the paper's bare-model rows (``bare``; reference.sh's ROW=dc, f, m or m0, :data:`BARE_ROWS`),
the bare model behind the harness protocol at the player's own request settings and reply cap, on the rows' own stream
at any number of users (``paper.schedule_cut``; at one user the canonical ``arc1-small``, workstreams, seed 0), round
by round as the rows play (:func:`play_bare_round`): dc, the demo-cache floor, the bare model with the demonstrations
it bought for a task shown again when the task returns; f, the stateless floor, the bare model alone; m, the program
library, the demo-cache floor whose every submission is a program, the last that reproduces all the task's pairs kept
and run first when the task returns; and m0, the program library wiped before every instance. In dc every user's bare
model reads one cache for the team (baselines' ``BareSharedDemoCacheLearner``), a round's purchases joining it at the
round's end and written to the run's ``demo-cache/``, since the share learned's perfect memory charges a task once in
the whole stream, whichever user met it. Row m's users keep their cache and programs, which baselines' learner holds in
memory, in the run's ``library/`` between rounds (:func:`keep_library`), since every round is a process of its own.
The demo-cache floor is the zero of the share learned (``continual_arc.metrics.share_learned``): a row is read against
the floor of the model that plays it, so glm's row h against glm's floor, and the DAgger, OPSD and PG rows and qwen's
rows against qwen's, each at its own number of users, since a trained row's teacher is training infrastructure, its
cost in the training-compute column.

Usage (on the reference machine, from the baselines' folder, ``../baselines``, with OPENROUTER_API_KEY in its .env;
reference.sh keeps it going): uv run --no-sync --env-file .env python ~/continual-arc/training/reference.py play \\
    --player glm --row h --run-dir ~/runs/paper/glm-h-u1/play --users 1 --rounds 200
  plays every round of the stream not yet in the run's records.jsonl, one call of play_round a round, as paper.sh does
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import loop
import paper

OPENROUTER = "https://openrouter.ai/api/v1"
# each player's model on OpenRouter and the request fields that replace the harness's own
PLAYERS: dict[str, dict[str, Any]] = {
    "glm": {"model": "z-ai/glm-5.3-flash", "body": {
        "reasoning": {"effort": "medium"},
        "provider": {"order": ["gmicloud", "parasail", "fireworks"], "allow_fallbacks": False}}},
    "qwen": {"model": "qwen/qwen3.5-9b", "body": {
        **paper.SAMPLING, "top_k": 0, "reasoning": {"enabled": True},
        "provider": {"order": ["together", "venice", "deepinfra"], "allow_fallbacks": False}}},
}
# each reference row's additions to paper.play_round's harness settings (hpl, hg, ha, hpl0, hplg and hgpl: the module's
# docstring)
REFERENCE_ROWS: dict[str, dict[str, Any]] = {"h": {}, "h0": {}, "hpl": {"program_library": True}, "hg": {},
                                              "ha": {"ask_to_store": True}, "hpl0": {"program_library": True},
                                              "hplg": {"program_library": True}, "hgpl": {}}
# the reference rows wiped before every instance: the trained rows' own, and hpl0
STATELESS_ROWS: tuple[str, ...] = (*paper.STATELESS_ROWS, "hpl0")
# each player's bare-model rows (the module's docstring): baselines' learner and its settings beyond paper.FLOOR's
BARE_ROWS: dict[str, tuple[str, dict[str, Any]]] = {
    "dc": ("BareSharedDemoCacheLearner", {}), "f": ("BareModelLearner", {}),
    "m": ("BareProgramLibraryLearner", {}), "m0": ("BareProgramLibraryLearner", {"stateless": True})}
# the rows whose turn after each verdict, the finish protocol's storage, another player takes (hg, hplg and hgpl: the
# module's docstring)
STORERS: dict[str, str] = {"hg": "glm", "hplg": "glm", "hgpl": "glm"}
# the rows whose storer alone is told the program library's procedure, in its turn after each verdict (hgpl)
TOLD_STORERS: tuple[str, ...] = ("hgpl",)


def overrides_of(player: str) -> dict[str, Any]:
    """The request fields ``player``'s calls carry in place of the harness's own: its model and its settings."""
    return {"model": PLAYERS[player]["model"], **PLAYERS[player]["body"]}


def library_lesson(system: Any, obs: Any) -> str:
    """Row hgpl's storer's lesson (the module's docstring): the program library's procedure and the ask to save the
    task's program, as row hpl's player reads them for ``obs``'s task."""
    from continual_arc_baselines.systems.protocol import PROGRAM_LIBRARY_ASK, PROGRAM_LIBRARY_RULES

    programs = system.programs_dir()
    return (PROGRAM_LIBRARY_RULES.format(programs=programs) + "\n\n"
            + PROGRAM_LIBRARY_ASK.format(path=f"{programs}/{obs.task_id}.py"))


def hand_storage(user: Any, storer: dict[str, Any], *, told: bool = False) -> None:
    """Rows hg, hplg and hgpl (the module's docstring): ``user``'s harness takes its turn after each verdict, the
    finish protocol's storage, with ``storer``'s request fields on the user's own proxy in place of the player's, which
    the next instance's play has again; every call the harness makes in that turn is the storer's. ``told`` follows
    the verdict with the program library's lesson (:func:`library_lesson`), which only the storer reads."""
    system, proxy = user.system, user.proxy
    finish = system._finish_instance

    def stored(obs: Any, pending: list[str]) -> None:
        player = proxy.body_overrides
        proxy.body_overrides = dict(storer)
        try:
            finish(obs, [*pending, "", library_lesson(system, obs)] if told else pending)
        finally:
            proxy.body_overrides = player

    system._finish_instance = stored


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
    kwargs = {"model": spec["model"], "context_window": paper.CONTEXT_WINDOW, "max_tokens": paper.MAX_TOKENS,
              "stateless": row in STATELESS_ROWS, **harness, **budget, **REFERENCE_ROWS[row]}
    overrides = overrides_of(player)
    storer = overrides_of(STORERS[row]) if row in STORERS else None
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": row, "player": player, "served": f"openrouter:{spec['model']}", "stateless": row in STATELESS_ROWS,
         "model": spec["model"], "harness": {**harness, **REFERENCE_ROWS[row]},
         "harness_budget": {"name": "block", **budget}, "overrides": overrides,
         **({"storer": {"player": STORERS[row], "overrides": storer, "told": row in TOLD_STORERS}} if storer else {})},
        indent=1))
    learner = CollectiveHarnessLearner(lambda user: PrimeAgentLearner(**kwargs), run_dir, schedule=schedule,
                                       upstream=OPENROUTER, api_key=api_key, body_overrides=overrides,
                                       body_drop=("reasoning_effort",),
                                       role_map={"developer": "system"}, thinking_field="reasoning",
                                       declared_context_length=paper.CONTEXT_WINDOW, capture=True, snapshot=False,
                                       instance_hook=paper.save_instance_hook(run_dir, paper.played), reset_hook=lambda: None)
    if storer:
        for user in range(schedule.users):
            hand_storage(learner._user(user), storer, told=row in TOLD_STORERS)
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        if row in STATELESS_ROWS:
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
        result = play_round(a.player, a.row, run_dir, paper.schedule_cut(r, a.seed, a.users), api_key=key,
                            harness=harness)
        print(json.dumps({"round": r, "records": len(result.records)}), flush=True)


def keep_library(system: Any, path: Path) -> None:
    """Row m's memory once a round is played (the module's docstring): the user's demo cache and kept programs, which
    baselines' ``BareProgramLibraryLearner`` holds in memory, written to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"programs": system.library,
                                "pairs": {task: [pair.to_json() for pair in pairs] for task, pairs in system.cache.items()}}))


def restore_library(system: Any, path: Path) -> None:
    """The memory :func:`keep_library` wrote to ``path``, given to the next round's learner."""
    from continual_arc.tasks import Pair
    kept = json.loads(path.read_text())
    system.library = dict(kept["programs"])
    system.cache = {task: [Pair.from_json(pair) for pair in pairs] for task, pairs in kept["pairs"].items()}


def play_bare_round(player: str, row: str, run_dir: Path, schedule: Any, *, api_key: str, rounds_done: int,
                    registry: Any = None, max_tokens: int = paper.MAX_TOKENS) -> Any:
    """``player``'s bare-model ``row`` (:data:`BARE_ROWS`, the module's docstring) over the rounds of ``schedule`` not
    yet in ``run_dir``'s records, the first ``rounds_done`` of them in its records already: every user's bare model at
    the floor's settings (:data:`paper.FLOOR`) with a reply cap of ``max_tokens`` (the rows' 32,768; the demo-cache
    floors play at the paper's 128,000) behind a proxy of their own taking the player's
    request settings as its defaults, as the CLI's ``--llm-defaults`` gave them. Row dc's users read one cache for the
    team opened on the run's ``demo-cache/`` after those rounds (``SharedDemoCache``), whose round ends with the round;
    row m's take up the memory the round before kept in the run's ``library/`` and keep theirs once the round is played.
    Returns the run's ``RunResult``."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.systems import bare

    spec = PLAYERS[player]
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    name, settings = BARE_ROWS[row]
    floor = {"max_tokens": max_tokens, **paper.FLOOR, **settings}
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": row, "player": player, "served": f"openrouter:{spec['model']}", "model": spec["model"],
         "learner": f"continual_arc_baselines.systems.bare:{name}", "floor": floor, "defaults": spec["body"]}, indent=1))
    hooks: dict[str, Any] = {}
    if row == "dc":
        cache = bare.SharedDemoCache(run_dir / "demo-cache", rounds_done)

        def make(user: int) -> Any:
            return bare.BareSharedDemoCacheLearner(spec["model"], shared=cache, user=user, **floor)
        hooks = {"round_hook": lambda data: cache.end_round(), "reset_hook": cache.clear}
    else:
        def make(user: int) -> Any:
            return getattr(bare, name)(spec["model"], **floor)
    learner = CollectiveHarnessLearner(make, run_dir, schedule=schedule, upstream=OPENROUTER, api_key=api_key,
                                       body_defaults=spec["body"], capture=False, snapshot=False, **hooks)
    library = run_dir / "library"
    if row == "m" and rounds_done:
        for user in range(schedule.users):
            restore_library(learner._user(user).system, library / f"r{rounds_done - 1:04d}-u{user:03d}.json")
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        result = run_stream(learner, schedule, EnvConfig(), out_dir=run_dir, env=env, resume=True,
                            learner_name=f"bare-{player}-{row}")
        if row == "m":
            for user in range(schedule.users):
                keep_library(learner._user(user).system, library / f"r{schedule.rounds()[-1]:04d}-u{user:03d}.json")
        return result
    finally:
        learner.close()


def cmd_bare(a: argparse.Namespace) -> None:
    from continual_arc.runner import prior_records
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(a.run_dir).expanduser()
    key = os.environ["OPENROUTER_API_KEY"]
    for r in range(len(prior_records(run_dir)) // a.users, a.rounds):
        result = play_bare_round(a.player, a.row, run_dir, paper.schedule_cut(r, a.seed, a.users), api_key=key,
                                 rounds_done=r, max_tokens=a.max_tokens)
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
    bare = sub.add_parser("bare", help="play a player's bare-model row on the rows' stream, resumable")
    bare.add_argument("--player", required=True, choices=PLAYERS)
    bare.add_argument("--row", required=True, choices=BARE_ROWS)
    bare.add_argument("--run-dir", required=True, dest="run_dir")
    bare.add_argument("--users", type=int, default=1)
    bare.add_argument("--rounds", type=int, default=200, help="the stream's rounds, all 200 by default")
    bare.add_argument("--seed", type=int, default=paper.SEED)
    bare.add_argument("--max-tokens", type=int, default=paper.MAX_TOKENS, dest="max_tokens",
                      help="the reply cap: the rows' 32,768 by default; the demo-cache floors play at the paper's 128,000")
    bare.set_defaults(fn=cmd_bare)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
