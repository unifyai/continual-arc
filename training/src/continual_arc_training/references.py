"""The reference rows the trained rows are read against, played as :func:`continual_arc_training.play.play_round`
plays a row (the same stream cut a round at a time, the same harness with its compaction point and call budget, the
same 131,072-token window and 32,768-token replies), but with the model served by OpenRouter, so no GPU is needed.

Two players (:data:`PLAYERS`):

  glm   GLM-5.3 Flash, the trained rows' oracle, at the labeller's settings (medium effort, the provider's own
        sampling) but on Z.ai, then Fireworks, no fallback, since GMICloud's replies to prompts it has cached run to the
        length cap or come back empty; playing the games itself.
  qwen  Qwen3.5-9B, the trained rows' base model, untrained: on Together first, then Venice (fp8), then DeepInfra
        (bf16), and no other, sampled at temperature 1 with no top-p or top-k cutoff (OpenRouter writes top_k off as 0),
        thinking on. Speed decides the order: Together and Venice answered a call in about 3 s, DeepInfra was some ten
        times slower and refused calls as rate-limited. Its chat template and tool-call parser are the provider's, not
        serve.sh's: a trained row's first rounds, the base model on the node's own server, check that they serve it
        alike.

Both send a harness's developer prompt as a system one and every earlier reply's reasoning under OpenRouter's
``reasoning`` key, as the trained rows' proxies do for vLLM, and drop prime-agent's own effort, so that the player's
setting is the one that applies.

The harness rows (:data:`REFERENCE_ROWS`): h, prime-agent with its memory kept; h0, wiped before every instance; hpl,
row h with the baselines' ``program_library`` (``PROGRAM_LIBRARY_RULES``: a program per task in the workspace's
``programs`` folder, run first when the task returns, checked on every pair and saved after the verdict, and every
verdict followed by ``PROGRAM_LIBRARY_ASK``), the procedure the trained rows' oracle labels under while their student is
never told it; ha, row h asked after every verdict to save liberally what could help with the task's later instances;
hpl0, row hpl wiped before every instance; hg, row h with its storage handed to GLM: from the verdict on, in the finish
protocol's turn in which the harness saves what it wants before it finishes, the storer drives the same session, its
request fields in place of the player's (:func:`hand_storage`), so it reads the player's whole game and keeps in the
harness what it judges will help the player later; hplg, row hpl with its storage handed to GLM, both told the
procedure; and hgpl, row hg with the storer alone told it, the procedure and the ask to save the task's program
following the verdict in the turn the storer takes (:func:`library_lesson`), which the player never sees.

The bare-model rows (:data:`BARE_ROWS`, :func:`play_bare_round`), the bare model behind the harness protocol at the
player's request settings: dc, the demo-cache floor, the demonstrations it bought for a task shown again when the task
returns, one cache for the team (a round's purchases joining it at the round's end, written to the run's
``demo-cache/``); f, the stateless floor, the bare model alone; m, the program library on the bare model, every
submission a program, the last that reproduces all of a task's pairs kept and run first when the task returns, its
memory kept in the run's ``library/`` between rounds (:func:`keep_library`), since every round is a process of its own;
and m0, row m wiped before every instance.

Subcommands (``python -m continual_arc_training.references ...``, with OPENROUTER_API_KEY in the environment; reference.sh
keeps one going): ``play --player P --row R --run-dir D --users N``, every round of a harness row's stream not yet
played, and ``bare`` likewise for a bare-model row.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from continual_arc_training import play

OPENROUTER = "https://openrouter.ai/api/v1"
# each player's model on OpenRouter and the request fields that replace the harness's own
PLAYERS: dict[str, dict[str, Any]] = {
    "glm": {"model": "z-ai/glm-5.3-flash", "body": {
        "reasoning": {"effort": "medium"},
        "provider": {"order": ["z-ai", "fireworks"], "allow_fallbacks": False}}},
    "qwen": {"model": "qwen/qwen3.5-9b", "body": {
        **play.SAMPLING, "top_k": 0, "reasoning": {"enabled": True},
        "provider": {"order": ["together", "venice", "deepinfra"], "allow_fallbacks": False}}},
}
# each harness row's additions to the harness settings
REFERENCE_ROWS: dict[str, dict[str, Any]] = {"h": {}, "h0": {}, "hpl": {"program_library": True}, "hg": {},
                                              "ha": {"ask_to_store": True}, "hpl0": {"program_library": True},
                                              "hplg": {"program_library": True}, "hgpl": {}}
STATELESS_ROWS: tuple[str, ...] = (*play.STATELESS_ROWS, "hpl0")
# each bare-model row's learner (the baselines' systems.bare) and its settings beyond play.FLOOR's
BARE_ROWS: dict[str, tuple[str, dict[str, Any]]] = {
    "dc": ("BareSharedDemoCacheLearner", {}), "f": ("BareModelLearner", {}),
    "m": ("BareProgramLibraryLearner", {}), "m0": ("BareProgramLibraryLearner", {"stateless": True})}
# the rows whose turn after each verdict another player takes, and the rows whose storer alone is told the procedure
STORERS: dict[str, str] = {"hg": "glm", "hplg": "glm", "hgpl": "glm"}
TOLD_STORERS: tuple[str, ...] = ("hgpl",)


def overrides_of(player: str) -> dict[str, Any]:
    """The request fields ``player``'s calls carry in place of the harness's own: its model and its settings."""
    return {"model": PLAYERS[player]["model"], **PLAYERS[player]["body"]}


def library_lesson(system: Any, obs: Any) -> str:
    """Row hgpl's storer's lesson: the program library's procedure and the ask to save the task's program, as row hpl's
    player reads them for ``obs``'s task."""
    from continual_arc_baselines.systems.protocol import PROGRAM_LIBRARY_ASK, PROGRAM_LIBRARY_RULES

    programs = system.programs_dir()
    return (PROGRAM_LIBRARY_RULES.format(programs=programs) + "\n\n"
            + PROGRAM_LIBRARY_ASK.format(path=f"{programs}/{obs.task_id}.py"))


def hand_storage(user: Any, storer: dict[str, Any], *, told: bool = False) -> None:
    """``user``'s harness takes its turn after each verdict, the finish protocol's storage, with ``storer``'s request
    fields on the user's own proxy in place of the player's, which the next instance's play has again; every call the
    harness makes in that turn is the storer's. ``told`` follows the verdict with the program library's lesson
    (:func:`library_lesson`), which only the storer reads."""
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


def play_round(player: str, row: str, run_dir: Path, schedule: Any, *, api_key: str, registry: Any = None) -> Any:
    """A harness ``row`` played by ``player`` (the module docstring) over the rounds of ``schedule`` not yet in
    ``run_dir``'s records; the run's ``RunResult``."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env
    from continual_arc.runner import run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner

    if row not in REFERENCE_ROWS:
        raise ValueError(f"a reference row is one of {tuple(REFERENCE_ROWS)}, not {row!r}")
    spec = PLAYERS[player]
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"model": spec["model"], "context_window": play.CONTEXT_WINDOW, "max_tokens": play.MAX_TOKENS,
              "stateless": row in STATELESS_ROWS, **play.HARNESS, **play.BLOCK, **REFERENCE_ROWS[row]}
    overrides = overrides_of(player)
    storer = overrides_of(STORERS[row]) if row in STORERS else None
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": row, "player": player, "served": f"openrouter:{spec['model']}", "stateless": row in STATELESS_ROWS,
         "model": spec["model"], "harness": {**play.HARNESS, **REFERENCE_ROWS[row]},
         "harness_budget": {"name": "block", **play.BLOCK}, "overrides": overrides,
         **({"storer": {"player": STORERS[row], "overrides": storer, "told": row in TOLD_STORERS}} if storer else {})},
        indent=1))
    learner = CollectiveHarnessLearner(lambda user: PrimeAgentLearner(**kwargs), run_dir, schedule=schedule,
                                       upstream=OPENROUTER, api_key=api_key, body_overrides=overrides,
                                       body_drop=("reasoning_effort",), role_map={"developer": "system"},
                                       thinking_field="reasoning", declared_context_length=play.CONTEXT_WINDOW,
                                       capture=True, snapshot=False, instance_hook=play.save_instance_hook(run_dir),
                                       reset_hook=lambda: None)
    if storer:
        for user in range(schedule.users):
            hand_storage(learner._user(user), storer, told=row in TOLD_STORERS)
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        if row in STATELESS_ROWS:
            play.wipe_stateless_users(learner, schedule)
        result = run_stream(learner, schedule, EnvConfig(), out_dir=run_dir, env=env, resume=True,
                            learner_name=f"reference-{player}-{row}")
        play.refresh_meta_protocol(run_dir, result.summary["protocol"])
        play.report_round_memory(learner, schedule, run_dir)
        return result
    finally:
        learner.close()


def keep_library(system: Any, path: Path) -> None:
    """Row m's memory once a round is played: the user's demonstration cache and kept programs, which the baselines'
    ``BareProgramLibraryLearner`` holds in memory, written to ``path``."""
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
                    registry: Any = None, max_tokens: int = play.MAX_TOKENS) -> Any:
    """``player``'s bare-model ``row`` (the module docstring) over the rounds of ``schedule`` not yet in ``run_dir``'s
    records, the first ``rounds_done`` of them in its records already: every user's bare model at the floor's settings
    with a reply cap of ``max_tokens``, behind a proxy of its own taking the player's request settings as its defaults.
    Row dc's users read one cache for the team opened on the run's ``demo-cache/`` after those rounds, whose round ends
    with the round; row m's take up the memory the round before kept in the run's ``library/`` and keep theirs once the
    round is played. The run's ``RunResult``."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env
    from continual_arc.runner import run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.systems import bare

    spec = PLAYERS[player]
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    name, settings = BARE_ROWS[row]
    floor = {"max_tokens": max_tokens, **play.FLOOR, **settings}
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


def cmd_play(a: argparse.Namespace) -> None:
    from continual_arc.runner import prior_records
    run_dir = Path(a.run_dir).expanduser()
    key = os.environ["OPENROUTER_API_KEY"]
    for r in range(len(prior_records(run_dir)) // a.users, a.rounds):
        result = play_round(a.player, a.row, run_dir, play.schedule_cut(r, a.seed, a.users), api_key=key)
        print(json.dumps({"round": r, "records": len(result.records)}), flush=True)


def cmd_bare(a: argparse.Namespace) -> None:
    from continual_arc.runner import prior_records
    run_dir = Path(a.run_dir).expanduser()
    key = os.environ["OPENROUTER_API_KEY"]
    for r in range(len(prior_records(run_dir)) // a.users, a.rounds):
        result = play_bare_round(a.player, a.row, run_dir, play.schedule_cut(r, a.seed, a.users), api_key=key,
                                 rounds_done=r, max_tokens=a.max_tokens)
        print(json.dumps({"round": r, "records": len(result.records)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s: argparse.ArgumentParser, rows: Any) -> None:
        s.add_argument("--player", required=True, choices=PLAYERS)
        s.add_argument("--row", required=True, choices=rows)
        s.add_argument("--run-dir", required=True, dest="run_dir")
        s.add_argument("--users", type=int, default=1)
        s.add_argument("--rounds", type=int, default=200, help="the stream's rounds, all 200 by default")
        s.add_argument("--seed", type=int, default=play.SEED)

    s = sub.add_parser("play", help="play every round of a harness row's stream not yet played, resumable")
    common(s, REFERENCE_ROWS)
    s.set_defaults(fn=cmd_play)
    s = sub.add_parser("bare", help="play every round of a bare-model row's stream not yet played, resumable")
    common(s, BARE_ROWS)
    s.add_argument("--max-tokens", type=int, default=play.MAX_TOKENS, dest="max_tokens",
                   help="the reply cap: the rows' 32,768 by default; the demo-cache floors play at 128,000")
    s.set_defaults(fn=cmd_bare)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
