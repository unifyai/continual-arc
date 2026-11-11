"""The paper driver: plays `arc1-collective` one round per call, for the block's rows.

The paper rows are driven from here, not from baselines' own K2 team driver (`envs/carc_stream/carc_stream/team.py`),
because the loop already has what a paper row needs: a runner that resumes under a longer schedule
(`continual_arc.runner.run(..., resume=True)`), baselines' `CollectiveHarnessLearner` with prime-agent, every call
captured with token ids, and Qwen3.5's adapter-serving fix (`loop.served_adapter`).

Rows:
  h0    frozen, harness wiped before every instance (the control; its epilogue is the fresh reference for T)
  grpo  trained (train.grpo_items' groups: a round's games by task), harness memory kept
  opsd  trained (train.opd_items against the frozen base model's per-token scores of each call under its own
        hinted prompt, teacher_score.py's --hints), harness memory kept -- standard OPD (`train.py --objective opd`
        against a 27B teacher's plain scores, no hints) is dropped from the paper rows; opsd replaces it, though
        train.py/teacher_score.py's own OPD machinery stays, round.sh's dev loop still using it
  dagger  trained (DAgger: the oracle, GLM-5.3-flash, writes its own reply to every player call of the round, read
        exactly as the player read it with OPSD's playing and program-library rules at the end of the last message;
        `train.py --objective sft` on those replies, forward.py's points/label/label-set), harness memory kept
  h     frozen, harness memory kept (the "should" row, prime-agent's own continual harness)
Only h0 is wiped. The trained rows keep memory because the story under test is that training teaches the model to
store a solved puzzle's program and fetch it when the puzzle returns, so a returning puzzle gets cheaper, which a
wiped row could never show (opsd's hints and dagger's labels coach that habit: continual_arc_baselines.systems.
protocol.PROGRAM_LIBRARY_RULES, oracle_hint.py's module docstring). "Wiped" is baselines' own stateless mode
(`continual_arc_baselines.systems.base.SystemLearner.stateless`): before every instance the system is stopped,
everything it wrote is deleted, and it restarts; rows grpo, opsd, dagger and h leave it off.

The mechanism, one round r = 0, 1, 2, ...: paper.sh serves the adapter round r needs (the base model at r = 0, or
round r-1's saved adapter), calls this file's ``play`` for round r (below), and, for a trained row, scores round r
(opsd: the teacher is the frozen base model, read off the very same server play just used) or labels it (dagger:
the oracle's own replies) and trains on it (train.py), saving the new adapter for round r+1.

``play`` plays the real `arc1-collective` schedule (:func:`stream_cut`: ``Schedule.first_rounds(r + 1)``, a literal
prefix) with ``continual_arc.runner.run(resume=True)``: every whole round already in the run directory's
``records.jsonl`` is restored and the stream continues from the next one, so calling ``play`` again for the same
round is a cheap no-op and calling it for round r plays exactly that one new round. The run directory is one
continuous benchmark run (``records.jsonl``, ``schedule.json``, ``meta.json``, ``summary.json``: `continual-arc
report`/`compare`'s own format), so ``train.grpo_items``' grouping, which keys an episode's group by (run, task) with
``run`` the directory `train.load_calls` was pointed at, would otherwise mix every round's games into one group.
:func:`save_instance_hook` avoids that by sharding captured calls under ``rounds/rNNNN/`` by the round they were
played in (``round r``'s own `train.load_calls` call reads only that round, so (run, task) is (round, task)); the
benchmark's own ``records.jsonl`` at the run directory's root is unaffected and still holds every round in order.

The epilogue at round R (:func:`epilogue_cut`, ``continual_arc.configs.epilogue_for_users``): a second, separate run
directory, played the same way over the held-out tasks alone; a trained row starts it from round R's saved adapter
(``--init-adapter``, paper.sh's job) and keeps training between its own rounds, while H0 plays it wiped, as the fresh
reference for T.

Subcommands (on the GPU node, from ~/prime-rl, as loop.py's: ``uv run --no-sync python ~/loop/paper.py ...``):
  play          play round ``--round`` of ``--row``'s schedule (the stream, or ``--epilogue-users``' epilogue) under
                the served model ``--served``, into ``--run-dir``; resumable, see above. ``--config-users`` plays a
                cheaper cut, `arc1-collective` with that many users instead of the real 100's
                (:func:`collective_config`): users 0..N-1 are exactly the real configuration's own first N, so a
                smoke run's rows are a true prefix of what the paper rows would see, never a different draw
  check-served  the served-adapter check: the fixed prompt's reply through ``--served`` differs from the base
                model's (vLLM silently serves the base model when an adapter's layer names are wrong)
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import loop  # noqa: E402  (this repo's dev loop: MODEL, TEMPLATE_KWARGS, HARNESS, harness_of, HARNESS_BUDGETS)

STREAM_CONFIG = loop.STREAM_CONFIG  # "arc1-collective"
SEED = 0  # workstreams, seed 0: the schedule every row plays

ROWS = ("h0", "grpo", "opsd", "dagger", "h")
# Wiped before every instance (baselines' stateless mode): h0 alone, the control. grpo, opsd and dagger keep
# prime-agent's own memory (the harness's files), so a trained row can show a returning puzzle getting cheaper through
# what it saved; h is the frozen row that keeps memory, the trained rows' reference.
STATELESS_ROWS = ("h0",)

# The shared settings every row plays under: temperature 1, top_p 1, no top_k forced on every call, with
# return_token_ids. loop.HARNESS_BUDGETS["block"] already codifies the call cap and its time budget, and loop.HARNESS
# the compaction point (32,768 tokens, keeping 8,000) -- read from there, never hard-coded here.
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}

DEFAULT_EPILOGUE_ROUND = 20
DEFAULT_EPILOGUE_USERS = 10

# The served-adapter check's fixed prompt: vLLM silently serves the base model under an adapter's name when its
# layer names are wrong, so one chat turn, nothing from the benchmark itself, catches it before a whole round plays
# on an adapter that never loaded.
SERVED_ADAPTER_PROMPT = [{"role": "user",
                         "content": "In one short sentence, what should an agent do when it is unsure whether its "
                                    "last answer was correct?"}]
# The check reads the replies' tokens and, at each, the log-probabilities of the top SERVED_TOP_LOGPROBS candidates,
# never the reply's text: Qwen3.5 opens with its reasoning, which vLLM's reasoning parser moves out of ``content``,
# so 16 tokens leave ``content`` empty for any weights. When both replies are the same tokens, the largest gap between
# a shared candidate's two log-probabilities must reach SERVED_LOGPROB_GAP. 16 Oct 2026, on the node: the base model
# asked twice gave a largest gap of exactly 0.0 (one request at a time, the same kernels); OPD's adapter after 8 steps
# at 2e-5 gave the same 16 tokens with a largest gap of 0.375 (mean 0.108), while the chosen tokens' own
# log-probabilities, all near 0, moved by under 1e-3.
SERVED_TOP_LOGPROBS = 5
SERVED_LOGPROB_GAP = 1e-2


# -- schedules -------------------------------------------------------------------------------------------

def collective_config(config_users: int | None = None) -> Any:
    """`arc1-collective`'s own configuration, or, with ``config_users``, a copy with that many users instead of the
    real 100's: a cheaper cut for a smoke test. ``config_users is None`` or equal to the real 100 returns the
    configuration unchanged, so a paper row never serves a copy. Otherwise the copy's name changes
    (``"arc1-collective-uN"``), so ``continual_arc.configs.protocol_of`` always marks it non-canonical, and its
    ``checkpoints`` (set at the real 100 users' instance counts) are dropped, since they would otherwise fall outside
    a smaller copy's stream and fail the configuration's own check. Every other field, and so every task selected and
    every user's own stream and epilogue order, is unaffected: user u's stream depends only on the seed and u
    (``continual_arc.configs._user_seeds``), task selection only on the dataset and ``n_tasks`` (``config_tasks``),
    neither on ``users``. Users 0..``config_users`` - 1 of the copy are therefore exactly the real configuration's
    first ``config_users`` users, round by round (this file's own test)."""
    import dataclasses

    from continual_arc.configs import get_config
    cfg = get_config(STREAM_CONFIG)
    if config_users is None or config_users == cfg.users:
        return cfg
    if not 1 <= config_users <= cfg.users:
        raise ValueError(f"config_users must be between 1 and {cfg.name}'s {cfg.users} users, got {config_users}")
    return dataclasses.replace(cfg, users=config_users, name=f"{cfg.name}-u{config_users}", checkpoints=())


def full_schedule(seed: int = SEED, config_users: int | None = None) -> Any:
    """`arc1-collective`'s whole schedule (100 users x 200 rounds, workstreams, or, with ``config_users``, a cheaper
    cut: :func:`collective_config`), built once; :func:`stream_cut` slices it rather than rebuilding a shorter one,
    which would not be this schedule's own prefix (``Schedule.first_rounds``)."""
    from continual_arc.configs import build_schedule
    return build_schedule(collective_config(config_users), "workstreams", seed)


def stream_cut(round_: int, seed: int = SEED, config_users: int | None = None) -> Any:
    """The real stream cut at ``round_ + 1`` rounds: a literal prefix of :func:`full_schedule`, so resuming under it
    after round ``round_ - 1`` was played plays exactly round ``round_``."""
    return full_schedule(seed, config_users).first_rounds(round_ + 1)


def epilogue_cut(round_: int, users: int = DEFAULT_EPILOGUE_USERS, seed: int = SEED,
                 config_users: int | None = None) -> Any:
    """The K-user epilogue schedule, cut the same way as :func:`stream_cut`: round ``round_`` of the held-out
    tasks served to ``users`` of the configuration's own (or, with ``config_users``, :func:`collective_config`'s
    cheaper copy's), in the exact order and at the exact instance seeds the full schedule's own epilogue would give
    them (``continual_arc.configs.epilogue_for_users``)."""
    from continual_arc.configs import epilogue_for_users
    return epilogue_for_users(collective_config(config_users), seed, users).first_rounds(round_ + 1)


# -- capture, sharded by round ---------------------------------------------------------------------------

def round_dir(run_dir: Path, round_: int) -> Path:
    """Where round ``round_``'s own captured calls (and, once scored or trained, its teacher scores and adapter)
    live under ``run_dir``: a directory train.py's ``--train-runs`` and teacher_score.py's can be pointed at alone, so
    an episode's (run, task) group (``train.grpo_items``) is (round, task), never mixed with another round's."""
    return run_dir / "rounds" / f"r{round_:04d}"


def save_instance_hook(run_dir: Path):
    """An instance_hook (``CollectiveHarnessLearner``) that saves each instance exactly as ``loop.py``'s own
    ``save_instance`` does, under :func:`round_dir` instead of one flat ``captured/``: the played run directory is
    continuous (every round, for the whole stream), but a round's training must read only its own games."""

    def hook(t: Any) -> None:
        captured = round_dir(run_dir, t.round) / "captured"
        captured.mkdir(parents=True, exist_ok=True)
        doc = {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
               "phase": t.phase, "solved": getattr(t, "solved", None), "turns": t.turns,
               "calls": [{"call_id": c.call_id, "turn": c.turn, "phase": c.phase, "status": c.status,
                          "finish_reason": c.finish_reason, "request": c.request, "response": c.response}
                         for c in t.calls]}
        tmp = captured / f"u{t.user:03d}.json.gz.tmp"
        try:
            # default=str, never fatal: as loop.py's save_instance (a harness's turn record can hold values JSON has
            # no form for; one instance's save failing must not cost the round the rest of its instances)
            with gzip.open(tmp, "wt") as fp:
                json.dump(doc, fp, default=str)
            tmp.rename(captured / f"u{t.user:03d}.json.gz")
        except Exception:  # noqa: BLE001
            logging.exception("could not save instance of user %s, round %s", t.user, t.round)

    return hook


# -- play --------------------------------------------------------------------------------------------------

def _wipe_stateless_users(learner: Any, schedule: Any) -> None:
    """Baselines' own wipe (``SystemLearner.reset``: stop if started, archive, ``wipe_state()``, a fresh
    ``Transcript``) for every user of ``schedule``'s collective regime, called once per round: a ``play`` call plays
    exactly one instance per user, so the harness's own in-process wipe (``SystemLearner.act``, gated on
    ``_session_pending``, which is set only at the *end* of the instance this very call plays) never fires across
    separate processes -- it is one call too late, every call. Calling ``reset()`` directly reproduces "wiped before
    every instance" at the only point this driver can: here, before the round's one instance per user. Raises if any
    user's own ``stats["resets"]`` (``reset()``'s own counter; the automatic in-process path above bumps
    ``stats["wipes"]`` instead, which never applies here) did not advance by exactly one, since a system that
    silently failed to wipe would otherwise look identical to one that did."""
    for u in range(schedule.users):
        system = learner._user(u).system
        before = system.stats["resets"]
        system.reset()
        after = system.stats["resets"]
        if after != before + 1:
            raise RuntimeError(f"user {u}'s system was not wiped exactly once before the round (resets {before} "
                               f"-> {after})")


def _refresh_meta_protocol(run_dir: Path, protocol: dict[str, Any]) -> None:
    """``continual_arc.runner.run`` reloads ``meta.json`` wholesale on a resumed call rather than recomputing its
    ``protocol`` block, so it freezes at whatever cut the run's first call saw; ``summary.json``'s own protocol is
    recomputed every call and is always right. Since this driver resumes every round, rewrite ``meta.json``'s
    ``protocol`` from the just-recomputed one (``RunResult.summary["protocol"]``) after every call, so a reader of
    ``meta.json`` alone is not misled."""
    path = run_dir / "meta.json"
    meta = json.loads(path.read_text())
    meta["protocol"] = protocol
    path.write_text(json.dumps(meta, indent=1))


def play_round(row: str, run_dir: Path, schedule: Any, *, base_url: str, served_name: str,
              registry: Any = None) -> Any:
    """Play ``schedule`` into ``run_dir`` under ``row``'s harness memory setting, serving ``served_name`` (the base
    model's own id, or the adapter name paper.sh already loaded into the server): ``continual_arc.runner.run``'s own
    ``resume=True`` restores every whole round already recorded and plays only the rest, so calling this with a cut
    schedule one round longer than last time plays exactly that new round, and calling it again under the same cut
    does nothing. ``registry`` defaults to the real benchmark's (``TaskRegistry.default()``); tests pass their own.
    Returns the run's ``RunResult`` (``.records``, ``.summary``)."""
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, run as run_stream
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    from carc_stream.team import THINKING_FIELD, VLLM_DROP, VLLM_ROLES

    if row not in ROWS:
        raise ValueError(f"row must be one of {ROWS}, got {row!r}")
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    # the fixed harness settings every row plays under: loop.HARNESS's own compaction point (32,768 tokens, keeping
    # 8,000) and loop.HARNESS_BUDGETS["block"]'s call cap, never overridden per round
    budget = loop.HARNESS_BUDGETS["block"]
    kwargs = {"model": loop.MODEL, "context_window": 131_072, "max_tokens": 32_768,
              "stateless": row in STATELESS_ROWS, **loop.HARNESS, **budget}
    overrides = {"model": served_name, "chat_template_kwargs": dict(loop.TEMPLATE_KWARGS), "return_token_ids": True,
                 **SAMPLING}
    (run_dir / "driver.json").write_text(json.dumps(
        {"row": row, "served": served_name, "stateless": row in STATELESS_ROWS, "model": loop.MODEL,
         "harness": dict(loop.HARNESS), "harness_budget": {"name": "block", **budget}, "sampling": SAMPLING},
        indent=1))
    learner = CollectiveHarnessLearner(lambda user: PrimeAgentLearner(**kwargs), run_dir, upstream=base_url,
                                       body_overrides=overrides, body_drop=VLLM_DROP, role_map=VLLM_ROLES,
                                       thinking_field=THINKING_FIELD, declared_context_length=131_072, capture=True,
                                       snapshot=False, instance_hook=save_instance_hook(run_dir),
                                       reset_hook=lambda: None)
    env = build_env(schedule, EnvConfig(), registry if registry is not None else TaskRegistry.default())
    try:
        if row in STATELESS_ROWS:
            _wipe_stateless_users(learner, schedule)
        result = run_stream(learner, schedule, EnvConfig(), out_dir=run_dir, env=env, resume=True,
                            learner_name=f"paper-{row}")
        _refresh_meta_protocol(run_dir, result.summary["protocol"])
        _report_round_memory(learner, schedule, run_dir)
        return result
    finally:
        learner.close()


def _report_round_memory(learner: Any, schedule: Any, run_dir: Path) -> None:
    """Every row's memory-use counters for the round just played (:mod:`memory`'s ``round_report``), the round
    being ``schedule.rounds()[-1]`` -- ``Schedule.rounds()`` gives every slot's own round, O(len(slots)), and
    ``[-1]`` is the cut's last slot's, which is correct here only because the cut's last slot belongs to the cut's
    last round (a resumed call plays exactly that one, see :func:`play_round`'s docstring). Row-agnostic -- H, H0,
    GRPO and OPSD alike. A wiped row's own wipe runs at the *next* call's start (:func:`_wipe_stateless_users`),
    not before this snapshot, so its snapshot still holds this round's own instance; what the wipe does guarantee
    is that no file survives *unchanged* into the next round (:mod:`memory`'s own docstring). A failure here (a
    user's harness state unreadable, say) is logged and never costs the round its own result."""
    import memory
    round_ = schedule.rounds()[-1]
    try:
        report = memory.round_report(run_dir, round_, schedule.users, lambda u: learner._user(u).system.state_dir,
                                      round_dir)
    except Exception:  # noqa: BLE001
        logging.exception("could not compute round %s's memory-use counters", round_)
        return
    logging.info(memory.summary_line(report))


# -- the served-adapter check -------------------------------------------------------------------------------

def adapter_is_untrained(adapter_dir: Path) -> bool:
    """Whether every ``lora_B`` matrix of the PEFT adapter in ``adapter_dir`` is exactly zero: PEFT starts B at zero,
    so an adapter no optimizer step has touched (a round whose data gave no item, e.g. GRPO with no mixed group) adds
    nothing and is the base model's own weights, which the served-adapter check cannot tell from a base model served
    under the adapter's name."""
    from safetensors import safe_open
    with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), "pt") as st:
        return all(not st.get_tensor(k).any() for k in st.keys() if "lora_B" in k)


def check_served_adapter(server: Any, base_model: str, served_name: str, prompt: list[dict[str, str]],
                         max_tokens: int = 16) -> dict[str, Any]:
    """``prompt`` (a fixed chat prompt) sent at temperature 0 to ``base_model`` and to ``served_name``: ``differs`` is
    false exactly when vLLM answered both from the same weights, which is how it silently serves the base model when
    an adapter's layer names are wrong (``loop.served_adapter``) instead of refusing to load it. Reads each reply's
    tokens and their top :data:`SERVED_TOP_LOGPROBS` candidates' log-probabilities, never the reply's text (see
    :data:`SERVED_LOGPROB_GAP`'s note): different tokens differ outright; the same tokens differ when a candidate
    both replies list moved by at least :data:`SERVED_LOGPROB_GAP` (``logprob_gap_max``, the largest such move).
    ``answered`` is false when either reply came back with no tokens (a failed request), which says nothing about
    the weights. ``server`` is an ``ask_sd.server.Server`` (or, in tests, anything with its ``chat`` method)."""

    def reply(model: str) -> tuple[list[str], dict[tuple[int, str], float]]:
        out = server.chat({"model": model, "messages": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                           "logprobs": True, "top_logprobs": SERVED_TOP_LOGPROBS})
        choice = ((out or {}).get("choices") or [{}])[0]
        content = ((choice.get("logprobs") or {}).get("content")) or []
        tops = {(i, c["token"]): float(c["logprob"]) for i, t in enumerate(content) for c in t.get("top_logprobs") or []}
        return [t["token"] for t in content], tops

    (base_tokens, base_tops), (served_tokens, served_tops) = reply(base_model), reply(served_name)
    out: dict[str, Any] = {"base": "".join(base_tokens), "served": "".join(served_tokens),
                           "answered": bool(base_tokens) and bool(served_tokens)}
    if not out["answered"]:
        return {**out, "differs": False}
    if base_tokens != served_tokens:
        return {**out, "differs": True}
    gaps = [abs(base_tops[k] - served_tops[k]) for k in base_tops.keys() & served_tops.keys()]
    out["logprob_gap_max"] = max(gaps) if gaps else None
    return {**out, "differs": bool(gaps) and max(gaps) >= SERVED_LOGPROB_GAP}


# -- CLI -----------------------------------------------------------------------------------------------------

def cmd_play(a: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(a.run_dir).expanduser()
    schedule = (epilogue_cut(a.round, a.epilogue_users, a.seed, a.config_users) if a.epilogue_users is not None
                else stream_cut(a.round, a.seed, a.config_users))
    result = play_round(a.row, run_dir, schedule, base_url=a.base_url, served_name=a.served)
    print(json.dumps({"out": str(run_dir), "round": a.round, "records": len(result.records),
                      "summary": result.summary}, default=str))


def cmd_check_served(a: argparse.Namespace) -> None:
    if a.adapter and adapter_is_untrained(Path(a.adapter).expanduser()):
        print(json.dumps({"skipped": f"{a.adapter} has taken no optimizer step (every lora_B is zero): it is the "
                                     "base model's own weights, so there is nothing to tell apart"}))
        return
    from ask_sd.server import Server
    out = check_served_adapter(Server(a.vllm_url), a.base_model, a.served, SERVED_ADAPTER_PROMPT,
                               max_tokens=a.max_tokens)
    print(json.dumps(out))
    if not out["answered"]:
        raise SystemExit(f"no tokens came back from {a.base_model!r} or {a.served!r} at {a.vllm_url}: the check could "
                         "not compare them (is the server up, and is the adapter loaded under that name?)")
    if not out["differs"]:
        raise SystemExit(f"{a.served!r} answered exactly as the base model {a.base_model!r}: vLLM may be serving the "
                         "base weights under the adapter's name (check the adapter's layer names, loop.served_adapter)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    play = sub.add_parser("play", help="play one round (or the K-user epilogue) of a row's schedule, resumable")
    play.add_argument("--row", required=True, choices=ROWS)
    play.add_argument("--run-dir", required=True, dest="run_dir")
    play.add_argument("--base-url", required=True, dest="base_url", help="the served model's OpenAI-compatible base URL")
    play.add_argument("--served", required=True,
                      help="the vLLM-served model name for this round: the base model's own id at round 0, or the "
                           "adapter name serve.sh already loaded the previous round's adapter under")
    play.add_argument("--round", type=int, required=True, help="0-indexed: the round this call plays")
    play.add_argument("--epilogue-users", type=int, default=None, dest="epilogue_users",
                      help="play the K-user epilogue (continual_arc.configs.epilogue_for_users) instead of the stream")
    play.add_argument("--seed", type=int, default=SEED)
    play.add_argument("--config-users", type=int, default=None, dest="config_users",
                      help="a cheaper cut: play arc1-collective with this many users instead of the real 100's "
                           "(collective_config); omit for the real, canonical configuration")
    play.set_defaults(fn=cmd_play)

    check = sub.add_parser("check-served",
                           help="the served-adapter check: the fixed prompt's reply through --served differs from "
                                "the base model's")
    check.add_argument("--vllm-url", required=True, dest="vllm_url")
    check.add_argument("--base-model", default=loop.MODEL, dest="base_model")
    check.add_argument("--served", required=True)
    check.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    check.add_argument("--adapter", default=None,
                       help="the served adapter's own directory: an adapter no step has touched is skipped "
                            "(adapter_is_untrained), since it is the base model's weights by construction")
    check.set_defaults(fn=cmd_check_served)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
