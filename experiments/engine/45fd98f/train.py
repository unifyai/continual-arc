"""Context distillation for one round of the loop: the student learns to act without the context as its teacher acts
with it; or, with ``--objective pg``, outcome reinforcement learning on the same plays, and with ``--objective sft``,
supervised fine-tuning on plays' replies (below).

The training prompts are the model calls of the current model's stream episodes, except prime-agent's compaction calls
(its summarizer's, not the agent's). Each call is one item: the student reads the call as served, the teacher reads the
same call with the context, and the loss compares them on every token of the reply. Where the context sits for the
teacher (``--placement``):

- ``thought``: the opening of the teacher's own reasoning, right after ``<think>``
  (``ask_sd.calls.thought_ids``; play's ``--context-placement thought`` shows the model the same text). A model
  refers to a document it reads, not to its own thoughts: on round 0's calls the teacher's push toward reference
  words was 16.5% of all it pushes as a notes block and 1.2% as its own reasoning;
- ``message`` (default, as play's): appended to the last message, wrapped as notes (``loop.context_block``), the call
  rendered locally; a call the local template does not render exactly as served is skipped.

Which reply (``--on-policy``):

- on-policy: every step's replies are drawn fresh from the current student, served by vLLM (``--server``) with the
  adapter reloaded after every step, so a batch is at most one step stale. A habit of several steps is taught a step
  further whenever the student's own replies reach the next one (1 Oct: trained on replies written before training,
  the student learned only first steps and collapsed);
- off-policy: the reply that was played.

Which teacher: the frozen base model (``--ema-decay none``); with ``--teacher-adapter``, a saved adapter read once and
fixed for the run, the student a round starts from (cycles.sh: the last kept round's student, once a round is kept); or,
the default, SDFT's moving average (below).

Otherwise the round follows SDFT (Shenfeld et al. 2026): the teacher is the moving average of the student's adapter
(``--ema-decay``), carried from round to round in the saved adapter; the loss is the forward KL, KL(teacher || student),
over the full vocabulary (``--divergence``); an anchor adds ``--anchor-mu`` times the same divergence toward the adapter
the round started from (the base model in round 1); one pass over the prompts (``--epochs``), nothing selected. The
validation calls (``--val-run``: a play of tasks never trained on, the validation split's in cycles.sh) are scored
against the moving-average teacher and against the fixed base model with the context: the moving teacher drifts toward
the student, so only the second says whether the student moved toward the context's behaviour. With a fixed teacher
(``--ema-decay none``, with or without ``--teacher-adapter``) the two are the same score against that teacher, computed
once.

Writes ``OUT/adapter`` (weights, optimizer and any moving average, for the next block's or round's
``--init-adapter``), with the moving average on also ``OUT/ema`` (it as an adapter vLLM serves), ``OUT/phase.json``
(per step: the loss, the replies drawn and the seconds spent; every ``--eval-every`` steps: the validation scores) and
``OUT/DONE``. With ``--patience`` the validation scores are a held-out loss, as ``sft``'s are: ``OUT/adapter`` is the
adapter at the best of them, and training stops once that many in a row have not improved on it (realtime.py's OPSD
row, whose held-out calls are DAgger's held-out games'). Every objective first writes ``OUT/kernels.json`` (the versions and GPU the linear-attention layers
train on) and stops on a stack known to compute wrong gradients (``kernel_stack.py``).
``--no-initial-eval`` skips the validation before the first step: a block that continues from the previous one would
only repeat that block's last. On-policy, ``OUT/live/`` holds the adapters published to the server, the last two kept.

With ``--objective pg`` there is no context, teacher or validation: plain policy gradient with a running-average
baseline. Each game's reward is :func:`reward` (solved minus a weighted share of its instance's cost), and its
advantage is the reward minus the mean reward of the last ``PG_BASELINE_GAMES`` games its run played before it,
whatever their tasks (:func:`pg_items`): the learning rule never reads the benchmark's task ids, which a real stream
does not have. A run's first game has nothing to weigh it against and is left out. Every call of the games that stay
is one item, its reply pushed by its game's advantage through PPO's clipped ratio against the adapter that played it
(ask_sd's ``Trainer.policy_gradient``).

With ``--objective opd`` there is no context, outcome, baseline or validation: every call of the training runs is one
item (none left out but one the teacher's chain failed to score), carrying the teacher's per-token log-probabilities
(:func:`opd_items`, the scores a round's own scoring step wrote beside the play, ``opd_scores.json``: RL-03's
``teacher_score.py --out``); ``Trainer.policy_gradient`` takes the advantage from them itself, position by position
(the teacher's log-probability minus the token's own under the adapter that played it), the same clipped ratio as
``pg``.

Both ``pg`` and ``opd`` read a round's calls as chains by default (:func:`find_chains`, each game's calls whose
prompts each extend the one before read as one sequence: ``Trainer.policy_gradient_chains``), one pass per chain
instead of one per call; ``--per-call`` falls back to one item per call (``Trainer.policy_gradient``), for the
equivalence check and when a chain path misbehaves. Either way, ``--epochs`` passes over the round's calls in steps of
``--batch`` of them. ``OUT/phase.json`` holds the games and their baselines (``pg``) or the item counts (``opd``) and, per step, the
loss, the clipped share of tokens, the KL estimate to the playing adapter and the entropy (plus, for ``opd``, the mean,
absolute mean and spread of the advantage over scored positions, and the share of positions with no teacher value).

With ``--objective sft`` there is no context, teacher, outcome or validation either: every call of the training runs is
a reply to imitate (the runs hold only such calls), one item each, its prompt as served and its reply as played, and
the reply's tokens are raised by maximum likelihood, every token weighing alike (ask_sd's ``Trainer.sft``), for
``--epochs`` passes over them in steps of ``--batch`` calls. ``OUT/phase.json`` holds the item counts and, per step,
the reply tokens' mean negative log-likelihood, the loss and the entropy. With ``--val-run`` (held-out plays of the same
kind) the held-out loss is measured before the first pass and after each, ``OUT/adapter`` is the adapter at its lowest,
and ``--patience`` stops the passes once that many in a row have not lowered it: trained to fit, no further.

Run on the GPU node from ~/prime-rl, on the GPUs the server does not hold (CUDA_VISIBLE_DEVICES):
  uv run --no-sync python ~/loop/train.py --model Qwen/Qwen3.5-9B --train-runs ~/runs/loop/r0-stream \
      --val-run ~/runs/loop/r0-val --context-file ~/runs/loop/c1.md --out ~/runs/loop/b1-1 --on-policy \
      --ema-decay none [--teacher-adapter PATH] [--init-adapter PATH] [--gpus 4] [--dry-run]
  uv run --no-sync python ~/loop/train.py --objective pg --train-runs ~/runs/paper/pg-u1/play-u1/rounds/r0005 \
      --out ~/runs/paper/pg-u1/train-u1/r5 --lr 1e-4 [--init-adapter PATH] [--gpus 8] [--per-call] [--dry-run]
  uv run --no-sync python ~/loop/train.py --objective opd --train-runs ~/runs/loop/round-play1 --out ~/runs/loop/round1 \
      --lr 2e-5 [--init-adapter PATH] [--gpus 8] [--per-call] [--dry-run]   # round-play1/opd_scores.json must exist
  uv run --no-sync python ~/loop/train.py --objective sft --train-runs ~/runs/loop/s-play1 --out ~/runs/loop/s1 \
      [--init-adapter PATH] [--gpus 8] [--batch 64] [--epochs 1] [--max-steps S] [--lr LR] [--dry-run] \
      [--val-run HELD_OUT --patience 2]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import json
import random
import shutil
import sys
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import kernel_stack  # noqa: E402
import loop  # noqa: E402
import render  # noqa: E402
from ask_sd.calls import ToolSchema  # noqa: E402
from ask_sd.trainer import DIVERGENCE_BETA  # noqa: E402

# the loop's server is vLLM 0.30 (serve.sh), whose tool schema the local rendering must reproduce
TOOL_SCHEMA = ToolSchema.vllm_0_30
# a fresh reply's length cap: replies run to a few thousand tokens (Qwen3.5-9B, round 0: median 566, p90 2,068), and
# the longest reply of a batch sets how long the batch takes to draw
DRAW_MAX_TOKENS = 8192
# a publish that fails (the server restarting: resume.sh brings it back in a few minutes) is retried this often
PUBLISH_RETRIES, PUBLISH_BACKOFF_S = 20, 30
# pg's baseline: how many of a run's most recent games, other than the game itself, are averaged for it (pg_items). A
# window in games, not rounds, gives the mean the same noise at any number of users (32 rounds at 1 user, 8 at 4), and
# keeps it to the recent policy's games as the policy learns. The advantage is the reward minus that mean, unscaled: the reward
# is already on a fixed scale (a solve near +1, a failure between -0.25 and -0.5), and dividing by the window's
# spread would blow up a failure's cost-only difference whenever the window holds no solve
PG_BASELINE_GAMES = 32
# reward's cost term's weight: how much of max_cost's worth of cost offsets one full point of the solved term
COST_WEIGHT = 0.5
# pg_items drops a game with no cost on record rather than guess one (see pg_items' docstring); past this share of a
# round's scored games, train_pg refuses rather than train on whatever had a cost, as OPD_UNSCORED_MAX_FRAC already
# does for a dead teacher
PG_UNCOSTED_MAX_FRAC = 0.02


def reward(solved: bool, cost: float, max_cost: float, cost_on_failures: bool = True) -> float:
    """A game's reward: 1 if ``solved`` else 0, minus :data:`COST_WEIGHT` times its instance's ``cost`` as a share of
    ``max_cost`` (the benchmark's whole budget, ``continual_arc.env.EnvConfig.max_cost``: a solved instance costs its
    wrong attempts plus its demo requests, an unsolved one the whole budget). Solving always outranks failing: a
    solve's reward is at least ``1 - COST_WEIGHT * (max_cost - 1) / max_cost``, and every failure's is
    ``-COST_WEIGHT``. Within solves a cheaper game outranks a costlier one; failures all score alike, so failing
    cheaply, without asking for the demonstrations a policy cannot solve without, is never rewarded (19 Oct 2026:
    under the benchmark's earlier rule, an unsolved instance charged the attempt cap plus its demo requests,
    pg-u1-lag1 never asked again after its fifth game).

    With ``cost_on_failures`` False (solve first) every failure scores 0 instead of ``-COST_WEIGHT``: the same order,
    with a smaller step between a failure and a solve."""
    if not solved and not cost_on_failures:
        return 0.0
    return (1.0 if solved else 0.0) - COST_WEIGHT * cost / max_cost


class Objective(StrEnum):
    """What a run learns from (``--objective``, written into ``phase.json`` and ``DONE``): ``sdft`` distils the context
    into the student; ``pg`` learns from the plays' outcomes; ``opd`` learns from a teacher's per-token scores of the
    plays; ``sft`` imitates the plays' replies."""

    sdft = "sdft"
    pg = "pg"
    opd = "opd"
    sft = "sft"


def instance_records(run_dir: Path) -> list[Any]:
    """Every instance the run has played so far (``continual_arc.env.InstanceRecord``: its user, round, task, outcome
    and cost), read from the run's own ``records.jsonl`` (``continual_arc.runner.prior_records``, which already drops a
    trailing partial line rather than raising on one): the captured doc
    (``continual_arc_baselines.collective.InstanceTrace``, as ``paper.save_instance_hook``/``loop.save_instance`` write
    it) carries no cost of its own, and :func:`pg_items` weighs a game against the run's earlier ones. Checked at
    ``run_dir`` directly (a plain play, ``records.jsonl`` beside its ``captured/``) and, since ``paper.round_dir``
    shards captured calls under ``run_dir/rounds/rNNNN`` while the run's own ``records.jsonl`` stays at the run
    directory's root, two directories up as well. Empty wherever neither holds one (a dry run before any round
    ended, or a run with no cost recorded) -- which :func:`pg_items` treats as every game uncosted, not as every
    cost 0."""
    from continual_arc.runner import prior_records
    for candidate in (run_dir, run_dir.parent.parent):
        records = prior_records(candidate)
        if records:
            return records
    return []


def load_calls(run_dir: Path) -> tuple[list[Any], int]:
    """Every completed model call of a play that the model made as the agent, as ask_sd Calls labelled with user, round,
    task, position, whether the episode solved its instance and its instance's cost (:func:`instance_records`, since the
    captured doc carries no cost of its own), and how many compaction calls were left out: prime-agent's summarizer
    calls (:func:`render.is_compaction`), whose reply is a summary the environment never reads and whose one message
    packs the whole conversation, where the context would land inside the text being summarised."""
    from ask_sd.calls import Call
    calls, compactions = [], 0
    costs: dict[tuple[int, int, str], float] | None = None
    for f in sorted((run_dir / "captured").glob("u*.json.gz")):
        with gzip.open(f, "rt") as fp:
            doc = json.load(fp)
        for i, c in enumerate(doc["calls"]):
            # a completed chat call only: the model's own sandbox code can send a raw completion (a "prompt", or any
            # JSON body) through the tracking proxy, captured beside the harness's calls, and that is no move of the
            # agent's and has no messages to render (overfit.chat_call, the same test the hint watcher applies)
            if not (c.get("request") and c.get("response") and isinstance(c["request"].get("messages"), list)
                    and c["request"]["messages"]):
                continue
            if render.is_compaction(c):
                compactions += 1
                continue
            # a played instance's cost is in its run's records; a label set's game (dagger.write_set: the expert's
            # replies at a play's calls, for sft) is no played instance, carries no round, and has none
            cost = doc.get("cost")
            if cost is None and "round" in doc:
                if costs is None:
                    costs = {(r.user, r.round, r.task_id): r.cost for r in instance_records(run_dir)}
                cost = costs.get((doc["user"], doc["round"], doc["task_id"]))
            # a call can carry a thought of its own for the teacher (probe.py's training plays): build_items
            meta = {"user": doc["user"], "round": doc.get("round"), "task_id": doc["task_id"], "i": i,
                    "call_id": c.get("call_id"), "run": run_dir.name, "solved": doc.get("solved"), "cost": cost}
            if c.get("thought"):
                meta["thought"] = c["thought"]
            # or a note of its own, read after the context in the same notes block (placement message)
            if c.get("note"):
                meta["note"] = c["note"]
            call = Call.from_exchange(c["request"], c["response"], meta=meta)
            if call is not None:
                calls.append(call)
    return calls, compactions


# SDFT's teacher prompt (Shenfeld et al. 2026, section 3, as its code words it): the demonstration after the question, in
# the question's own turn; for a harness call, the end of its last message, the one the reply answers
SDFT_DEMO = "\n\nThis is an example for a response to the question:\n{demo}\nNow answer with a response of your own, including the thinking process."


def demo_text(tok: Any, request: dict[str, Any], label: dict[str, Any]) -> str | None:
    """The expert's reply at a call (a ``forward.py label`` file: its message and reasoning) as the student writes a reply
    there: the DAgger row's target for the same label (:func:`dagger.student_message`: all of the oracle's prose as its
    thinking, the action in view) rendered after the call's prompt by the student's own chat template
    (:func:`forward.render_reply`), its thinking opened as the template's generation prompt opens it, its end of turn left
    off. None when the template cannot render it."""
    import dagger
    import forward
    rendered = forward.render_reply(tok, request, dagger.student_message(label))
    if rendered is None:
        return None
    prompt, reply = rendered
    opened = tok.decode(prompt[-4:]).endswith("<think>\n")
    return ("<think>\n" if opened else "") + tok.decode(reply[:-1])


def attach_demos(tok: Any, calls: list[Any], labels_dir: Path) -> tuple[list[Any], dict[str, int]]:
    """The OPSD row's demonstrations (SDFT): each call with the expert's own reply at it, :func:`demo_text` in
    :data:`SDFT_DEMO`, as the text its teacher reads at the end of the last message (``meta["demo"]``,
    :func:`build_items`). That reply is the DAgger row's label for the same call (``forward.py label``: the same oracle,
    rules and settings), read from ``labels_dir`` (its points.jsonl and labels/) and matched by the call's own id. A
    call with no label, or one the template cannot render, is left out: its teacher would read no demonstration; and so
    is one whose label makes no move (dagger.makes_move), which would show the teacher a reply that does nothing."""
    points = {(p["user"], p["i"]): p for p in map(json.loads, open(labels_dir / "points.jsonl"))}
    import dagger
    kept, stats = [], {"calls": len(calls), "with_demo": 0, "no_demo": 0, "no_move": 0, "unrendered": 0}
    for call in calls:
        p = points.get((call.meta["user"], call.meta["i"]))
        f = labels_dir / "labels" / f"{p['id']}.json" if p and p["call_id"] == call.meta["call_id"] else None
        if f is None or not f.exists():
            stats["no_demo"] += 1
            continue
        label = json.loads(f.read_text())
        if not dagger.makes_move(dagger.student_message(label)):
            stats["no_move"] += 1
            continue
        demo = demo_text(tok, call.request, label)
        if demo is None:
            stats["unrendered"] += 1
            continue
        call.meta["demo"] = SDFT_DEMO.format(demo=demo)
        kept.append(call)
        stats["with_demo"] += 1
    return kept, stats


def played_reply(call: Any, eos: int | None) -> list[int]:
    """The reply's ids as the model ended it: the server leaves out the end-of-turn token of a reply that stopped
    (``stop``) or called a tool (``tool_calls``), which the model did sample, so it goes back on."""
    reply = list(call.reply_ids)
    if call.finish_reason in ("stop", "tool_calls") and eos is not None and reply[-1] != eos:
        reply.append(eos)
    return reply


def pg_items(tokenizer: Any, calls: list[Any], records: dict[str, list[Any]], max_seq_len: int, through_round: int,
             max_cost: float | None = None, cost_on_failures: bool = True) -> tuple[list[Any], dict[str, Any]]:
    """Plain policy gradient's items from plays' calls (:func:`load_calls`): each game (a run's user) has the reward
    :func:`reward` of its solved outcome and its instance's cost, and its advantage is that reward minus the mean
    reward of its run's :data:`PG_BASELINE_GAMES` most recent games up to round ``through_round``, the game itself
    never among them, whatever their tasks: a running-average baseline of the run's play as it now is, read from
    ``records`` (each run's own :func:`instance_records`, by run name), the run's games in round then user order. So a
    game of an earlier round trained on again counts as good only if it beats what the run now plays, and nothing
    played after ``through_round`` is read, however far play has run ahead. No group of games from one starting point
    is needed, so a run trains at any number of users, and nothing reads the benchmark's task ids. Left out and
    counted: a game with no other game to weigh it against (``no_history``: a one-user run's round 0); one whose
    outcome was never recorded (``unscored``); and one with no cost on record (``uncosted``) -- never defaulted to 0,
    the cheapest cost there is, which would make a bookkeeping gap the strongest reward of all. ``max_cost`` defaults
    to the benchmark's own bound on an instance's cost (``continual_arc.env.EnvConfig().max_cost``), read once here
    rather than hard-coded, so a different track's cap is picked up without a code change. Each call of a game that
    stays is one item, the call's prompt as served and its reply as played, with the game's advantage; a call longer
    than the window is skipped and counted. ``trained`` in the stats lists every game trained on: its reward, the
    baseline, how many games that averaged, and the advantage. ``cost_on_failures`` is :func:`reward`'s, for the game
    and its baseline's games alike."""
    from ask_sd.trainer import TokenItem
    if max_cost is None:
        from continual_arc.env import EnvConfig
        max_cost = EnvConfig().max_cost
    eos = tokenizer.eos_token_id
    episodes: dict[tuple[str, int], list[Any]] = collections.defaultdict(list)
    for call in calls:
        episodes[(call.meta["run"], call.meta["user"])].append(call)
    played = {run: sorted(rs, key=lambda r: (r.round, r.user)) for run, rs in records.items()}
    stats: dict[str, Any] = {"episodes": len(episodes), "unscored": 0, "uncosted": 0, "no_history": 0,
                             "episodes_used": 0, "solved_used": 0, "items": 0, "too_long": 0, "trained": []}
    items = []
    for (run, user), ep in sorted(episodes.items()):
        meta = ep[0].meta
        if meta["solved"] is None:
            stats["unscored"] += 1
            continue
        if meta["cost"] is None:
            stats["uncosted"] += 1
            continue
        before = [r for r in played.get(run, [])
                  if r.round <= through_round and (r.round, r.user) != (meta["round"], user)][-PG_BASELINE_GAMES:]
        if not before:
            stats["no_history"] += 1
            continue
        baseline = sum(reward(r.solved, r.cost, max_cost, cost_on_failures) for r in before) / len(before)
        solved = bool(meta["solved"])
        own = reward(solved, meta["cost"], max_cost, cost_on_failures)
        advantage = own - baseline
        stats["episodes_used"] += 1
        stats["solved_used"] += int(solved)
        stats["trained"].append({"run": run, "user": user, "round": meta["round"], "solved": solved,
                                 "reward": round(own, 4), "baseline": round(baseline, 4),
                                 "baseline_games": len(before), "advantage": round(advantage, 4)})
        for call in ep:
            reply = played_reply(call, eos)
            ids = list(call.prompt_ids) + reply
            if len(ids) > max_seq_len:
                stats["too_long"] += 1
                continue
            items.append(TokenItem(ids, ids, len(reply), label=dict(call.meta), advantage=advantage))
    stats["items"] = len(items)
    return items, stats

def load_opd_scores(run_dir: Path, model: str | None = None) -> dict[tuple[str, int, int], "list[float | None] | None"]:
    """A run's teacher scores (``teacher_score.py --out``, written beside its captured calls as ``opd_scores.json`` by
    ``round.sh``'s scoring step), keyed by (run, user, i) as :func:`load_calls` labels a call. With ``model``, refuses
    a file scored for a different student: ``played_reply`` depends on the student tokenizer's end-of-turn id, which
    ask-sd's length check alone would not catch a mismatch on."""
    doc = json.loads((run_dir / "opd_scores.json").read_text())
    student = doc.get("student_model")
    if model and student and student != model:
        raise ValueError(f"{run_dir / 'opd_scores.json'} was scored for student {student!r}, not {model!r}")
    return {(it["run"], it["user"], it["i"]): it["scores"] for it in doc["items"]}


_UNATTEMPTED = object()   # opd_items' own sentinel: a call load_opd_scores' dict has no key for at all, never
                          # sent to the teacher by design (OPSD's non-player and no-hint calls, teacher_score.py's
                          # own too_long/template_mismatch) -- distinct from a key present with scores None, which
                          # means the teacher was asked and came back with nothing (a live teacher-health signal)


def opd_items(tokenizer: Any, calls: list[Any],
              scores: dict[tuple[str, int, int], "list[float | None] | None"],
              max_seq_len: int) -> tuple[list[Any], dict[str, Any]]:
    """OPD's items from plays' calls (:func:`load_calls`) and the teacher's scores of them (:func:`load_opd_scores`):
    one item per call, its prompt as served and its reply as played, carrying the teacher's per-token log-probabilities
    (``Trainer.policy_gradient`` takes the advantage from them itself). A call absent from ``scores`` entirely is
    never counted as the teacher's failure (``excluded``): standard OPD scores every call it is given, so this is
    always 0 there, but OPSD's ``--hints`` deliberately never attempts a call with no hint (its own non-player and
    no-hint calls) or one :func:`train.build_items` itself drops (``teacher_score.score_hinted_items``), and none
    of those are a dead-teacher signal. A call present with ``scores`` None -- the teacher was asked and gave
    nothing -- is ``unscored``, and so is a call longer than the window."""
    from ask_sd.trainer import TokenItem
    eos = tokenizer.eos_token_id
    stats: dict[str, Any] = {"calls": len(calls), "items": 0, "excluded": 0, "unscored": 0, "too_long": 0}
    items = []
    for call in calls:
        score = scores.get((call.meta["run"], call.meta["user"], call.meta["i"]), _UNATTEMPTED)
        if score is _UNATTEMPTED:
            stats["excluded"] += 1
            continue
        if score is None:
            stats["unscored"] += 1
            continue
        reply = played_reply(call, eos)
        ids = list(call.prompt_ids) + reply
        if len(ids) > max_seq_len:
            stats["too_long"] += 1
            continue
        items.append(TokenItem(ids, ids, len(reply), label=dict(call.meta), teacher_logp=list(score)))
    stats["items"] = len(items)
    return items, stats


def sft_items(tokenizer: Any, calls: list[Any], max_seq_len: int) -> tuple[list[Any], dict[str, Any]]:
    """Supervised items from plays' calls (:func:`load_calls`), one per call: the call's prompt as served and its reply
    as played, the reply the target; every call is one to imitate, so none is left out but a call longer than the
    window, which is skipped and counted."""
    from ask_sd.trainer import TokenItem
    eos = tokenizer.eos_token_id
    stats: dict[str, Any] = {"calls": len(calls), "items": 0, "too_long": 0}
    items = []
    for call in calls:
        reply = played_reply(call, eos)
        ids = list(call.prompt_ids) + reply
        if len(ids) > max_seq_len:
            stats["too_long"] += 1
            continue
        items.append(TokenItem(ids, ids, len(reply), label=dict(call.meta)))
    stats["items"] = len(items)
    return items, stats


def splice_note(served: list[int], local: list[int], noted: list[int]) -> list[int] | None:
    """The teacher's prompt ids: ``served`` (a call's prompt as the server tokenised it) with the note that ``noted``
    (the local rendering with the note) adds to ``local`` (the local rendering without it), put in by position. The
    served ids can split an earlier turn's text otherwise than a fresh rendering of it would: they hold the replies as
    the model sampled them, and it can sample a split the tokenizer would not make (a word in a rare script, such as
    Devanagari), so the two renderings are not compared whole. They agree up to where the note starts, and what
    follows that point in ``local`` (the last message's end and the generation prompt) ends ``served`` too; the note
    goes in there, before it. None when ``served`` does not end that way."""
    cut = common_prefix(local, noted)
    tail = local[cut:]
    if len(tail) > len(served) or served[len(served) - len(tail):] != tail:
        return None
    return served[:len(served) - len(tail)] + noted[cut:]


def build_items(tokenizer: Any, calls: list[Any], context: str, placement: str,
                max_seq_len: int) -> tuple[list[Any], dict[str, Any]]:
    """Distillation items, one per call: the student's ids are the call's prompt as served and its reply; the
    teacher's, the same prompt with the context (``placement``) and the same reply. A call with a ``thought`` of its own
    (in its meta, from :func:`load_calls`) has that thought open the teacher's reasoning as well, after the context; with
    ``message``, a call with a ``note`` of its own has it in the notes block after the context (one block: what play's
    oracle hints send, oracle_hint.play_note), and one with a ``demo`` (:func:`attach_demos`) reads that text alone, as
    it is, at the end of its last message. With ``message`` the teacher's prompt is the served one with the note put
    in where the local template puts it (:func:`splice_note`), so teacher and student read the same ids before it; a
    call whose served prompt does not end as the template's does is skipped and counted (``template_mismatch``), and
    one whose earlier turns the served ids split otherwise than a fresh rendering would is counted in
    ``retokenised``. A call longer than the window is skipped and counted either way."""
    from ask_sd.calls import render_prompt_ids, thought_ids, with_note
    from ask_sd.trainer import TokenItem
    eos = tokenizer.eos_token_id
    thought = thought_ids(tokenizer, context) if placement == "thought" else None
    block = loop.context_block(context)
    stats: dict[str, Any] = {"calls": len(calls), "items": 0, "template_mismatch": 0, "too_long": 0, "render_error": 0,
                             "retokenised": 0}
    items = []
    for call in calls:
        if (call.meta.get("note") or call.meta.get("demo")) and placement != "message":
            raise ValueError("a call's note or demonstration needs placement message")
        if thought is not None:
            teacher = list(call.prompt_ids) + thought
        else:
            if call.meta.get("demo"):
                note = call.meta["demo"]   # SDFT's own wording, at the end of the last message as it stands
            elif call.meta.get("note"):
                note = loop.context_block(context.strip() + "\n\n" + call.meta["note"].strip())
            else:
                note = block
            try:
                local = render_prompt_ids(tokenizer, call.request, join_text_parts=True, tool_schema=TOOL_SCHEMA)
                noted = render_prompt_ids(tokenizer, call.request, with_note(call.request["messages"], note),
                                          join_text_parts=True, tool_schema=TOOL_SCHEMA)
            except Exception:  # noqa: BLE001  (a call the local template cannot render is skipped, as call_items does)
                stats["render_error"] += 1
                continue
            served = list(call.prompt_ids)
            teacher = splice_note(served, local, noted)
            if teacher is None:
                stats["template_mismatch"] += 1
                continue
            stats["retokenised"] += local != served
        if call.meta.get("thought"):
            teacher = teacher + thought_ids(tokenizer, call.meta["thought"])
        reply = played_reply(call, eos)
        item = TokenItem(list(call.prompt_ids) + reply, teacher + reply, len(reply), label=dict(call.meta))
        if max(len(item.student_ids), len(item.teacher_ids)) > max_seq_len:
            stats["too_long"] += 1
            continue
        items.append(item)
    stats["items"] = len(items)
    return items, stats


def mean_scores(trainer: Any, items: list[Any], divergence: str, anchor_mu: float,
                base_teacher: bool = False) -> dict[str, float | None]:
    """The items' means under the adapter as it is now: the training loss (the divergence to the teacher, plus the
    anchor's weighted term), the divergence to the teacher alone (``teacher_div``: the loss without that term),
    KL(student || teacher), and the divergence to the anchor (None without one); against the configured teacher, or
    with ``base_teacher`` against the fixed base model reading the same teacher prompts."""
    res = [x for x in trainer.evaluate(items, divergence=divergence, base_teacher=base_teacher)["items"] if x]
    n = max(1, len(res))
    anchor = [x["anchor_div"] for x in res if x["anchor_div"] is not None]
    return {"loss": sum(x["loss"] for x in res) / n,
            "teacher_div": sum(x["loss"] - anchor_mu * (x["anchor_div"] or 0.0) for x in res) / n,
            "teacher_kl": sum(x["teacher_kl"] for x in res) / n,
            "anchor_div": sum(anchor) / len(anchor) if anchor else None}


class LiveAdapter:
    """The student as the server serves it during on-policy training: after every step its adapter is saved under
    ``OUT/live`` and loaded under a new name, and the one before is retired once no draw uses it, so the server holds
    two of the student's at most (beside the adapter a cycle's play uses: serve.sh's --max-loras 4)."""

    def __init__(self, trainer: Any, server: Any, out: Path, model: str) -> None:
        self.trainer, self.server, self.model = trainer, server, model
        self.root = out / "live"
        # named for the run as well as the round (OUT's last three parts: a paper cut's row, train folder and round), so
        # cuts sharing one server never take each other's: each attempt's names are its own, since an attempt that died
        # (a crash) leaves its last adapters loaded and vLLM refuses to load a name it already serves
        tag = "-".join(out.resolve().parts[-3:])
        for name in self._served():
            if name.startswith(tag + "-"):
                self.server.unload_adapter(name)
        self.prefix = f"{tag}-t{int(time.time())}"
        self.published: list[tuple[str, list[Path]]] = []   # (served name, its files), oldest first

    def _served(self) -> list[str]:
        """The names the server serves now (the base model's and every adapter's)."""
        with urllib.request.urlopen(self.server.base + "/models", timeout=60) as resp:
            return [m["id"] for m in json.loads(resp.read())["data"]]

    def publish(self, step: int) -> str:
        """The adapter as it is now, loaded on the server; its served name. Every rank holds the same adapter, so
        rank 0's is the one saved."""
        path = self.root / f"s{step:04d}"
        shutil.rmtree(path, ignore_errors=True)
        self.trainer.model.save_pretrained(str(path))
        served = loop.served_adapter(self.model, path)
        name = f"{self.prefix}-s{step}"
        for attempt in range(PUBLISH_RETRIES + 1):
            try:
                self.server.load_adapter(name, served)
                break
            except Exception as error:  # noqa: BLE001  (a server that restarts refuses for a few minutes)
                if attempt == PUBLISH_RETRIES:
                    raise
                print(f"publishing {name} failed ({error}); retrying in {PUBLISH_BACKOFF_S} s", flush=True)
                time.sleep(PUBLISH_BACKOFF_S)
        self.published.append((name, sorted({path, served})))
        return name

    def retire(self, keep: int) -> None:
        """Unload and delete every published adapter but the newest ``keep``."""
        while len(self.published) > keep:
            name, paths = self.published.pop(0)
            self.server.unload_adapter(name)
            for path in paths:
                shutil.rmtree(path, ignore_errors=True)


def draw(pool: ThreadPoolExecutor, server: Any, calls: list[Any], model: str) -> list[Future]:
    """One fresh reply from ``model`` at each call's prompt, the call's own request otherwise (its sampling settings,
    tools and thinking), capped at DRAW_MAX_TOKENS; the draws run while the current step trains."""
    from ask_sd.calls import resample

    def one(call: Any) -> list[Any]:
        try:
            return resample(server, call, model, 1, max_tokens=DRAW_MAX_TOKENS)
        except Exception as error:  # noqa: BLE001  (a reply cut off mid-read is a failed draw, counted, not fatal)
            print(f"a draw failed: {type(error).__name__}: {error}", flush=True)
            return []
    return [pool.submit(one, c) for c in calls]


def make_trainer(a: argparse.Namespace) -> Any:
    """The trainer the arguments ask for: data parallel over ``--gpus`` GPUs, or one GPU."""
    kw = dict(rank=a.rank, alpha=a.alpha, lr=a.lr, max_seq_len=a.max_seq_len, system_prompt=None, share_prompts=True)
    if a.gpus > 1:
        from ask_sd.parallel import DataParallelTrainer
        return DataParallelTrainer(a.model, world_size=a.gpus, device="cuda", **kw)
    from ask_sd.trainer import Trainer
    return Trainer(a.model, device="cuda", **kw)


def epoch_batches(items: list[Any], a: argparse.Namespace, rng: random.Random) -> list[list[Any]]:
    """``--epochs`` passes over the items, each in its own shuffled order, in steps of ``--batch``, at most
    ``--max-steps`` of them."""
    batches = []
    for _ in range(a.epochs):
        order = items[:]
        rng.shuffle(order)
        batches += [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
    return batches[: a.max_steps or len(batches)]


# a call joins the chain of the call before it when its prompt begins with all but at most this many of that call's
# last prompt ids (16 Oct 2026: a reply with no reasoning renders in the next prompt as an empty think block, whose
# newlines tokenize as one id where the earlier prompt ended with one; 46% of the round-1 model's calls broke there)
CHAIN_SLACK = 8


def find_chains(items: list[Any], slack: int = CHAIN_SLACK) -> list[list[Any]]:
    """Supervised items (:func:`sft_items`) as chains: within each game (an item's ``run`` and ``user``), in call order
    (``i``), a call joins the chain of the call before it when its prompt is at least as long and begins with that
    call's prompt but for at most its last ``slack`` ids (a game's calls each extend the last, until a summary rewrites
    the conversation or the template renders an earlier message differently). Every item is in exactly one chain, in
    order; every prompt in a chain begins with all but at most ``slack`` of the chain's earlier prompts' ids, so each
    earlier call branches off the chain's last prompt with a head of at most ``slack`` ids (:func:`chain_item`)."""
    games: dict[tuple, list[Any]] = collections.defaultdict(list)
    for it in items:
        games[(it.label.get("run"), it.label.get("user"))].append(it)
    chains: list[list[Any]] = []
    for key in sorted(games, key=str):
        prev = None
        for it in sorted(games[key], key=lambda x: x.label.get("i", 0)):
            prompt = it.student_ids[:-it.n]
            base = len(prev) - slack if prev is not None else 0
            if prev is not None and len(prompt) >= len(prev) and base > 0 and prompt[:base] == prev[:base]:
                chains[-1].append(it)
            else:
                chains.append([it])
            prev = prompt
    return chains


def chain_item(calls: list[Any], slack: int = CHAIN_SLACK) -> Any:
    """Consecutive calls of one chain (:func:`find_chains` with the same ``slack``) as one
    :class:`ask_sd.trainer.ChainItem`: the last call's prompt is the trunk, and each call's reply a branch at the last id
    its prompt shares with the trunk, the rest of its prompt (at most ``slack`` ids) carried ungraded before it. A call
    with its own ``advantage``, ``teacher_logp`` or ``behaviour_logp`` (:func:`pg_items`'s or :func:`opd_items`'s
    items, :func:`attach_behaviour`'s) carries it onto its branch, for
    :meth:`ask_sd.trainer.Trainer.policy_gradient_chains`; plain :func:`sft_items` calls carry none."""
    from ask_sd.trainer import ChainItem
    last = calls[-1]
    trunk = list(last.student_ids[:-last.n])
    branches, heads = [], []
    for c in calls:
        prompt = c.student_ids[:-c.n]
        point = max(0, len(prompt) - slack)
        if trunk[:point] != list(prompt[:point]):
            raise ValueError("chain_item: a call's prompt leaves the chain's last prompt by more than the slack")
        while point < len(prompt) and trunk[point] == prompt[point]:
            point += 1
        if point == 0:
            raise ValueError("chain_item: a call's prompt shares no id with the chain's last prompt")
        branches.append((point, list(prompt[point:]) + list(c.student_ids[-c.n:])))
        heads.append(len(prompt) - point)
    # getattr, not c.teacher_logp: an item built against an ask-sd without OPD's TokenItem.teacher_logp (sft_items'
    # plain items, or any ask-sd before RL-02's per-token advantages) has no such attribute at all, not just None.
    # extra stays empty for sft_items' calls, so this is ChainItem(trunk, branches, labels, heads) exactly, as every
    # ask-sd version accepts: only pg_items'/opd_items' calls, which need the newer ChainItem fields, add to it
    advantages = [c.advantage for c in calls] if any(c.advantage is not None for c in calls) else None
    teacher_logps = ([getattr(c, "teacher_logp", None) for c in calls]
                     if any(getattr(c, "teacher_logp", None) is not None for c in calls) else None)
    behaviour_logps = ([c.behaviour_logp for c in calls]
                       if any(getattr(c, "behaviour_logp", None) is not None for c in calls) else None)
    extra = {}
    if advantages is not None:
        extra["advantages"] = advantages
    if teacher_logps is not None:
        extra["teacher_logps"] = teacher_logps
    if behaviour_logps is not None:
        extra["behaviour_logps"] = behaviour_logps
    return ChainItem(trunk, branches, [c.label for c in calls], heads if any(heads) else [], **extra)


def chain_pieces(chain: list[Any], max_calls: int, max_len: int, slack: int = CHAIN_SLACK) -> list[list[Any]]:
    """A chain cut at call boundaries into pieces of at most ``max_calls`` calls whose pass (the last call's prompt and
    every reply, each earlier one with a head of at most ``slack`` ids) is at most ``max_len`` tokens; a piece reads its
    own prompt from the start again."""
    pieces: list[list[Any]] = []
    cur: list[Any] = []
    for c in chain:
        cand = cur + [c]
        size = len(c.student_ids) - c.n + sum(x.n for x in cand) + slack * (len(cand) - 1)
        if cur and (len(cand) > max_calls or size > max_len):
            pieces.append(cur)
            cand = [c]
        cur = cand
    if cur:
        pieces.append(cur)
    return pieces


def chain_batches(chains: list[list[Any]], batch: int, ranks: int, epochs: int, max_len: int,
                  rng: random.Random, max_steps: int = 0, slack: int = CHAIN_SLACK) -> list[list[list[Any]]]:
    """``epochs`` passes over the chains' calls in steps of ``batch`` calls, each step made of whole chains drawn at
    random: each pass shuffles the chains (never single calls: two calls of one chain in one pass share their prefix, two
    calls drawn apart share nothing), cuts each into pieces of at most ``batch / ranks`` calls (so every one of
    ``ranks`` GPUs gets a piece) and the window (:func:`chain_pieces`), and fills steps with the pieces in order,
    splitting a piece at a call boundary where the step fills up. Every call is in exactly one step per pass, every step
    holds ``batch`` calls but a pass's last, and a step is a list of pieces (lists of consecutive calls of one chain).
    At most ``max_steps`` steps."""
    cap = max(1, -(-batch // max(1, ranks)))
    steps: list[list[list[Any]]] = []
    for _ in range(epochs):
        order = chains[:]
        rng.shuffle(order)
        step: list[list[Any]] = []
        room = batch
        for chain in order:
            for piece in chain_pieces(chain, cap, max_len, slack):
                while piece:
                    take, piece = piece[:room], piece[room:]
                    step.append(take)
                    room -= len(take)
                    if room == 0:
                        steps.append(step)
                        step, room = [], batch
        if step:
            steps.append(step)
    return steps[: max_steps or len(steps)]


def common_prefix(a: list[int], b: list[int]) -> int:
    """How many ids two lists share from their start."""
    n, i = min(len(a), len(b)), 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def teacher_chain_item(items: list[Any], chain: Any) -> Any:
    """The teacher's chain for ``Trainer.distill_chains`` over ``chain`` (:func:`chain_item` of the same ``items``): the
    same trunk, and each item's branch at the last id its teacher prompt shares with it, the rest of that prompt (the
    demonstration and what follows it to the reply) carried ungraded as its head before the same reply."""
    from ask_sd.trainer import ChainItem
    branches, heads = [], []
    for it in items:
        t_prompt = list(it.teacher_ids[:-it.n])
        at = min(common_prefix(chain.trunk, t_prompt), len(t_prompt) - 1)
        branches.append((at, t_prompt[at:] + list(it.teacher_ids[-it.n:])))
        heads.append(len(t_prompt) - at)
    return ChainItem(chain.trunk, branches, [it.label for it in items], heads)


def sdft_pieces(chain: list[Any], max_calls: int, max_len: int, slack: int = CHAIN_SLACK) -> list[list[Any]]:
    """:func:`chain_pieces` for ``Trainer.distill_chains``: pieces of at most ``max_calls`` calls whose student pass
    (as chain_pieces counts it) and teacher pass (the same trunk, and every reply after the rest of its teacher prompt,
    at most ``slack`` ids more than its own prompt shares) are both at most ``max_len`` tokens."""
    rest = {id(it): len(it.teacher_ids) - it.n - common_prefix(it.student_ids[:-it.n], it.teacher_ids[:-it.n]) + slack
            for it in chain}
    pieces: list[list[Any]] = []
    cur: list[Any] = []
    for c in chain:
        cand = cur + [c]
        trunk = len(c.student_ids) - c.n
        size = trunk + sum(x.n for x in cand) + max(slack * (len(cand) - 1), sum(rest[id(x)] for x in cand))
        if cur and (len(cand) > max_calls or size > max_len):
            pieces.append(cur)
            cand = [c]
        cur = cand
    if cur:
        pieces.append(cur)
    return pieces


def sdft_steps(calls: list[Any], a: argparse.Namespace, rng: random.Random) -> list[list[Any]]:
    """``--demos``' steps: ``--epochs`` passes over the calls in steps of ``--batch``, each step whole chains of a game's
    calls (:func:`chain_batches` over the calls as played), so a step's calls share their games' prompts; at most
    ``--max-steps`` steps. A step is a list of calls."""
    from types import SimpleNamespace
    played = [SimpleNamespace(student_ids=list(c.prompt_ids) + list(c.reply_ids), n=len(c.reply_ids), label=c.meta, call=c)
              for c in calls if c.reply_ids]
    steps = chain_batches(find_chains(played, a.chain_slack), a.batch, a.gpus, a.epochs, a.max_seq_len, rng,
                          a.max_steps or 0, a.chain_slack)
    return [[x.call for piece in step for x in piece] for step in steps]


def distill_chain_step(trainer: Any, items: list[Any], a: argparse.Namespace, rng: random.Random) -> dict[str, Any]:
    """One step's SDFT pass over ``items`` (:func:`build_items` of a step's calls with their demonstrations): their games'
    chains (:func:`find_chains`), cut so that both passes fit the window and every GPU gets a piece
    (:func:`sdft_pieces`), each piece's chain and teacher chain through ``Trainer.distill_chains``."""
    students, teachers = sdft_chain_pairs(items, a)
    return trainer.distill_chains(students, teachers, rng=rng, divergence=a.divergence, skip_first=a.skip_first)


def sdft_chain_pairs(items: list[Any], a: argparse.Namespace) -> tuple[list[Any], list[Any]]:
    """:func:`build_items`' items as ``Trainer.distill_chains`` takes them: their games' chains (:func:`find_chains`),
    cut so that both passes fit the window and every GPU gets a piece (:func:`sdft_pieces`), each piece's chain and its
    teacher chain."""
    cap = max(1, -(-a.batch // max(1, a.gpus)))
    students, teachers = [], []
    for chain in find_chains(items, a.chain_slack):
        for piece in sdft_pieces(chain, cap, a.max_seq_len, a.chain_slack):
            student = chain_item(piece, a.chain_slack)
            students.append(student)
            teachers.append(teacher_chain_item(piece, student))
    return students, teachers


def policy_gradient_batches(items: list[Any], a: argparse.Namespace, rng: random.Random) -> list[Any]:
    """``pg``'s or ``opd``'s items for the step loop: chains by default (:func:`find_chains`/:func:`chain_batches`,
    one pass per chain of a game's calls, each step a list of pieces of consecutive calls), or, with ``--per-call``,
    :func:`epoch_batches`'s flat calls."""
    if a.per_call:
        return epoch_batches(items, a, rng)
    chains = find_chains(items, a.chain_slack)
    batches = []
    for _ in range(a.epochs):
        batches += chain_batches(chains, a.batch, a.gpus, 1, a.max_seq_len, rng, slack=a.chain_slack)
    return batches[: a.max_steps or len(batches)]


def policy_gradient_step(trainer: Any, batch: Any, a: argparse.Namespace, rng: random.Random) -> dict[str, Any]:
    """One optimizer step's policy-gradient pass over ``batch`` (:func:`policy_gradient_batches`): per call with
    ``--per-call`` (``Trainer.policy_gradient``), or, by default, whole chain pieces (``Trainer.policy_gradient_chains``,
    each piece read as one sequence via :func:`chain_item`)."""
    if a.per_call:
        return trainer.policy_gradient(batch, rng=rng)
    return trainer.policy_gradient_chains([chain_item(piece, a.chain_slack) for piece in batch], rng=rng)


def policy_gradient_record(res: dict[str, Any], taken: dict[str, Any], step: int) -> dict[str, Any]:
    """One step's log line for ``pg`` and ``opd`` alike, from :func:`policy_gradient_step`'s result: the items and
    tokens trained on, the gradient step taken, the played tokens' mean log-probability under the adapter as the step
    began (``logp_per_token``) and the usual policy-gradient metrics, the share of tokens ask-sd's dual clip silenced
    and the step's token of largest |log r| (``log_ratio_worst``: its call's label, position, id and log-probabilities
    under the adapter training and the one that played it, which is pruned soon after, so that a step whose ratios ran
    away can be traced to its token from ``phase.json`` alone), plus ``opd`` (the mean, absolute mean and spread of the
    advantage over scored positions, and the share of positions with no teacher value; the two spread fields only once
    ask-sd's loss reports them) when the step held any OPD items."""
    record = {"step": step, "items": res["n_examples"], "tokens": res["tokens"], "skipped_long": res["skipped_long"],
              "grad_norm": round(taken["grad_norm"], 5) if "grad_norm" in taken else None,
              "logp_per_token": (round(sum(x for x in res["logp"] if x is not None) / res["tokens"], 5)
                                 if res["tokens"] else None),
              **{k: round(res[k], 6) if res[k] is not None else None
                 for k in ("loss_mean", "ratio_mean", "clip_fraction", "dual_clip_fraction", "approx_kl",
                           "entropy_mean")},
              "log_ratio_worst": res["log_ratio_worst"]}
    if res.get("opd"):
        record["opd"] = {"tokens": res["opd"]["tokens"],
                         **{k: round(res["opd"][k], 6) for k in
                            ("advantage_mean", "advantage_abs_mean", "advantage_std", "missing_frac")
                            if k in res["opd"]}}
    return record


# opd_items: a dead or partly-dead teacher scores some calls None (score_items swallows a chain's failure); above
# this share of a round's calls, train_opd refuses rather than train on whatever fraction of the round survived
OPD_UNSCORED_MAX_FRAC = 0.02


def policy_gradient_dry_run(a: argparse.Namespace, calls: list[Any],
                            build_items: Callable[[Any], tuple[list[Any], dict[str, Any]]], stats_key: str,
                            compactions: int) -> None:
    """``--objective pg`` or ``opd`` with ``--dry-run``: the round's items' counts, built with the tokenizer alone."""
    from transformers import AutoTokenizer
    items, item_stats = build_items(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True))
    print(json.dumps({"train_calls": len(calls), "compaction_calls_left_out": compactions, stats_key: item_stats,
                      "reply_len_median": sorted(it.n for it in items)[len(items) // 2] if items else None}))


def train_policy_gradient(a: argparse.Namespace, trainer: Any, calls: list[Any],
                          build_items: Callable[[Any], tuple[list[Any], dict[str, Any]]], stats_key: str,
                          compactions: int, out: Path, init: Path | None, played_by: Path | None,
                          rng: random.Random, *, val_items: list[Any] | None = None) -> dict[str, Any]:
    """The loop shared by ``--objective pg`` and ``opd``: ``build_items(tokenizer)`` gives the round's items
    (:func:`pg_items` or :func:`opd_items`, already bound to their own calls, and for ``opd`` its teacher scores),
    read as chains by default or, with ``--per-call``, one call per item (:func:`policy_gradient_batches`),
    ``--epochs`` passes over them in steps of ``--batch`` calls from ``init`` (``--init-adapter``, or the base model),
    every ratio taken against the adapter that played them: an item's own ``behaviour_logp``
    (:func:`attach_behaviour`), else ``played_by`` (the base model when None), in a synchronous round the adapter
    training starts from, and an older one when play runs ahead of training (realtime.py's lag).

    With ``val_items`` (held-out games' items, built as the training ones are) and ``--patience`` P, the held-out loss
    is DAgger's held-out stop for policy gradient (:func:`train_sft`): the training loss itself over the held-out
    games' chains (``Trainer.policy_gradient_scores_chains``), each span's dual-clipped surrogate against the adapter
    that played it, measured before the first step, every ``--eval-steps`` steps and after the last; ``OUT/adapter`` is
    the adapter at the best of those, and training stops once P in a row have not improved on it. Beside it, for
    reading, the held-out games that solved: their replies' negative log-likelihood per token
    (``Trainer.nll_chains``). Without, ``OUT/adapter`` is the adapter as training left it. Then ``OUT/phase.json`` and
    ``OUT/DONE``, whose summary is returned. ``stats_key`` is the item-builder's stats block's name in
    ``phase.json``/``DONE`` (``games`` for ``pg``, ``items`` for ``opd``); with no item (PG: a run's first game, with
    nothing to weigh it against) the adapter is saved as it was, so the next round plays with it again. ``trainer``
    (:func:`make_trainer`) is the caller's, so one trainer can train round after round (realtime.py's resident
    trainer)."""
    if init:
        trainer.load(init)
    else:
        trainer.reset()
    if played_by:
        trainer.configure_behaviour(played_by)
    else:
        trainer.configure_behaviour(base=True)
    items, item_stats = build_items(trainer.tokenizer)
    print(json.dumps({stats_key: item_stats, "compaction_calls_left_out": compactions}), flush=True)
    batches = policy_gradient_batches(items, a, rng)
    val = val_items or []
    held_out = bool(val and a.patience)   # the held-out loss picks the adapter and stops training
    cap = max(1, -(-a.batch // max(1, a.gpus)))
    val_chains = [chain_item(piece, a.chain_slack) for ch in find_chains(val, a.chain_slack)
                  for piece in chain_pieces(ch, cap, a.max_seq_len, a.chain_slack)]
    wins = [it for it in val if it.label.get("solved")]
    wins_chains = [chain_item(piece, a.chain_slack) for ch in find_chains(wins, a.chain_slack)
                   for piece in chain_pieces(ch, cap, a.max_seq_len, a.chain_slack)]
    t0, steps, evals, capped, stopped = time.time(), [], [], False, False
    best, since_best = None, 0

    def write_phase() -> None:
        (out / "phase.json").write_text(json.dumps({"objective": a.objective, stats_key: item_stats, "steps": steps,
                                                    "held_out": evals, "args": vars(a)}, indent=1, default=str))

    def measure() -> bool:
        """The held-out loss now (and the held-out wins' likelihood), into ``held_out``; the adapter saved when the
        loss is the best so far; whether ``--patience`` measurements in a row have now not improved on it."""
        nonlocal best, since_best
        res = [x for x in trainer.policy_gradient_scores_chains(val_chains)["items"] if x]
        point: dict[str, Any] = {"step": len(steps), "loss": round(sum(x["loss"] for x in res) / max(1, len(res)), 6),
                                 "spans": len(res)}
        if wins_chains:
            nll = [x for x in trainer.nll_chains(wins_chains)["items"] if x]
            point["wins_nll"] = round(-sum(x["logp"] for x in nll) / max(1, sum(x["tokens"] for x in nll)), 6)
        point["best"] = best is None or point["loss"] < best
        if point["best"]:
            best, since_best = point["loss"], 0
            trainer.save(out / "adapter")
        else:
            since_best += 1
        evals.append(point)
        print(json.dumps({"held_out": point}), flush=True)
        write_phase()
        return since_best >= a.patience

    if held_out:
        measure()
    for batch in batches:
        t = time.time()
        res = policy_gradient_step(trainer, batch, a, rng)
        taken = trainer.step()
        # logp_per_token and entropy_mean: for pg, failed games are the longer ones, so their push down
        # outweighs the solved ones' push up, and with no KL term a steady fall of the first and rise of the second
        # is the policy spreading away from what it plays
        record = {**policy_gradient_record(res, taken, len(steps) + 1), "train_s": round(time.time() - t, 1)}
        steps.append(record)
        print(json.dumps(record), flush=True)
        write_phase()
        if held_out and len(steps) % a.eval_steps == 0 and measure():
            stopped = True
            print(f"held-out stop at step {len(steps)}: {a.patience} measurements in a row above {best:.6f}",
                  flush=True)
            break
        if time.time() - t0 > 60 * a.max_minutes:
            print(f"time cap {a.max_minutes} min at step {len(steps)}", flush=True)
            capped = True
            break
    if held_out and evals[-1]["step"] != len(steps):   # the steps after the last measurement, measured once
        measure()
    write_phase()
    if not held_out:
        trainer.save(out / "adapter")
    best_step = max((e["step"] for e in evals if e["best"]), default=None)
    summary = {"objective": a.objective, "steps": len(steps), "time_capped": capped, "held_out_stop": stopped,
               stats_key: item_stats, "held_out": [e["loss"] for e in evals], "best_step": best_step,
               "last": steps[-1] if steps else None, "minutes": round((time.time() - t0) / 60, 1)}
    (out / "DONE").write_text(json.dumps(summary))
    print("done", json.dumps(summary), flush=True)
    return summary


def played_items(tokenizer: Any, calls: list[Any], max_seq_len: int) -> list[Any]:
    """Every call's reply as played after its prompt as served (:func:`played_reply`), one item a call, labelled with
    its meta, a call longer than the window left out: what :func:`behaviour_logps` scores."""
    from ask_sd.trainer import TokenItem
    eos = tokenizer.eos_token_id
    items = []
    for call in calls:
        reply = played_reply(call, eos)
        ids = list(call.prompt_ids) + reply
        if len(ids) <= max_seq_len:
            items.append(TokenItem(ids, ids, len(reply), label=dict(call.meta)))
    return items


def behaviour_logps(trainer: Any, a: argparse.Namespace, items: list[Any]) -> dict[tuple[str, int, int], list[float]]:
    """``items``' replies' token log-probabilities under the trainer's behaviour adapter (its configure_behaviour),
    read as chains (``Trainer.token_logps_chains``), keyed by each call's run, user and place in its game; a call in a
    chain longer than the window is left out, as training leaves it out."""
    cap = max(1, -(-a.batch // max(1, a.gpus)))
    pieces = [piece for ch in find_chains(items, a.chain_slack)
              for piece in chain_pieces(ch, cap, a.max_seq_len, a.chain_slack)]
    got = trainer.token_logps_chains([chain_item(piece, a.chain_slack) for piece in pieces],
                                     behaviour=True)["items"]
    calls = [it for piece in pieces for it in piece]
    return {(it.label["run"], it.label["user"], it.label["i"]): logps for it, logps in zip(calls, got)
            if logps is not None}


def attach_behaviour(items: list[Any], logps: dict[tuple[str, int, int], list[float]]) -> list[Any]:
    """Each item with its reply's log-probabilities under the adapter that played it (:func:`behaviour_logps`), which
    ``Trainer.policy_gradient_chains`` takes its ratios against; an item none were taken for, or of another length,
    is refused (ValueError): its ratios would be against whatever adapter the trainer holds."""
    for it in items:
        key = (it.label["run"], it.label["user"], it.label["i"])
        if key not in logps or len(logps[key]) != it.n:
            raise ValueError(f"attach_behaviour: no behaviour log-probabilities for {key} ({it.n} reply tokens)")
        it.behaviour_logp = logps[key]
    return items


def pg_build(a: argparse.Namespace, calls: list[Any], records: dict[str, list[Any]],
             through_round: int) -> Callable[[Any], tuple[list[Any], dict[str, Any]]]:
    """``--objective pg``'s item-builder for :func:`train_policy_gradient`: :func:`pg_items` over ``calls``, each game
    weighed against its run's play up to ``through_round``, refusing
    (``SystemExit``) a round where more than :data:`PG_UNCOSTED_MAX_FRAC` of its scored games have no cost on record:
    a missing or misplaced ``records.jsonl`` must stop the round, not quietly train every game as cost 0 (the best
    possible reward, see :func:`pg_items`'s docstring) -- the same reasoning :func:`opd_build` already applies to a
    dead teacher (:data:`OPD_UNSCORED_MAX_FRAC`)."""

    def build(tok: Any) -> tuple[list[Any], dict[str, Any]]:
        items, pstats = pg_items(tok, calls, records, a.max_seq_len, through_round,
                                 cost_on_failures=a.pg_reward == "cost")
        scored = pstats["episodes"] - pstats["unscored"]
        uncosted_frac = pstats["uncosted"] / scored if scored else 0.0
        if uncosted_frac > PG_UNCOSTED_MAX_FRAC:
            raise SystemExit(f"pg: {pstats['uncosted']}/{scored} scored games have no cost on record "
                             f"({uncosted_frac:.1%}), over the {PG_UNCOSTED_MAX_FRAC:.0%} limit: records.jsonl "
                             f"is likely missing or not where instance_records looked")
        return items, pstats

    return build


def train_pg(a: argparse.Namespace, trainer: Any, calls: list[Any], records: dict[str, list[Any]], compactions: int,
             out: Path, init: Path | None, played_by: Path | None, rng: random.Random) -> dict[str, Any]:
    """``--objective pg``: :func:`train_policy_gradient` over :func:`pg_build`'s items, each game weighed against its
    run's play up to the newest round among ``calls``, every ratio taken against ``played_by``, the adapter that played
    the games."""
    newest = max(c.meta["round"] for c in calls)
    return train_policy_gradient(a, trainer, calls, pg_build(a, calls, records, newest), "games", compactions, out,
                                 init, played_by, rng)


def opd_build(a: argparse.Namespace, calls: list[Any],
              scores: dict[tuple[str, int, int], Any]) -> Callable[[Any], tuple[list[Any], dict[str, Any]]]:
    """``--objective opd``'s item-builder for :func:`train_policy_gradient`: :func:`opd_items` over ``calls``,
    refusing (``SystemExit``) a round with no items to train on, or where the teacher failed to score more than
    :data:`OPD_UNSCORED_MAX_FRAC` of the calls it was actually asked to score: a dead or partly-dead teacher must stop
    the round, not quietly train on whatever survived and bias the learning-rate choice without failing. The
    fraction is over calls the teacher was asked to score (``calls - excluded``), never over every call `load_calls`
    kept: a call never asked by design (OPSD's own non-player and no-hint calls) says nothing about the teacher's
    health, standard OPD's own ``excluded`` always 0."""

    def build(tok: Any) -> tuple[list[Any], dict[str, Any]]:
        items, ostats = opd_items(tok, calls, scores, a.max_seq_len)
        if not items:
            raise SystemExit(f"opd: no items to train on ({ostats['calls']} calls, {ostats['excluded']} excluded "
                             f"by design, {ostats['unscored']} unscored, {ostats['too_long']} too long)")
        attempted = ostats["calls"] - ostats["excluded"]
        unscored_frac = ostats["unscored"] / attempted if attempted else 0.0
        if unscored_frac > OPD_UNSCORED_MAX_FRAC:
            raise SystemExit(f"opd: {ostats['unscored']}/{attempted} calls the teacher was asked to score came "
                             f"back unscored ({unscored_frac:.1%}), over the {OPD_UNSCORED_MAX_FRAC:.0%} limit: "
                             f"the teacher likely failed partway through scoring ({ostats['excluded']} further "
                             f"calls were never asked, by design, and are not counted here)")
        return items, ostats

    return build


def train_opd(a: argparse.Namespace, trainer: Any, calls: list[Any], scores: dict[tuple[str, int, int], Any],
              compactions: int, out: Path, init: Path | None, rng: random.Random) -> dict[str, Any]:
    """``--objective opd``: :func:`train_policy_gradient` over :func:`opd_build`'s items, from the adapter that played
    them (``init``: OPD's rounds are synchronous)."""
    return train_policy_gradient(a, trainer, calls, opd_build(a, calls, scores), "items", compactions, out, init,
                                 init, rng)


def held_out_calls(runs: list[str], max_items: int = 0, seed: int = 0) -> list[Any]:
    """The held-out plays' calls (:func:`load_calls`), whole games at a time (:func:`whole_games`)."""
    return whole_games([g for r in runs for g in by_game(load_calls(Path(r).expanduser())[0])], max_items, seed)


def by_game(calls: list[Any]) -> list[list[Any]]:
    """``calls`` (:func:`load_calls`') as their games' calls, a game a run's user's, in run then user order."""
    games: dict[tuple[str, int], list[Any]] = collections.defaultdict(list)
    for call in calls:
        games[call.meta["run"], call.meta["user"]].append(call)
    return [games[key] for key in sorted(games)]


def whole_games(games: list[list[Any]], max_items: int = 0, seed: int = 0) -> list[Any]:
    """The calls of ``games`` (each a game's calls): every game's, or, with ``max_items``, those of games drawn with
    ``seed`` until their calls reach it (a held-out set that grows every round, measured at a cost that does not)."""
    if max_items:
        games = games[:]
        random.Random(seed).shuffle(games)
        kept, n = [], 0
        for game in games:
            if n >= max_items:
                break
            kept.append(game)
            n += len(game)
        games = kept
    return [call for game in games for call in game]


def sft_dry_run(a: argparse.Namespace, calls: list[Any], compactions: int) -> None:
    """``--objective sft --dry-run``: the supervised items' counts, built with the tokenizer alone."""
    from transformers import AutoTokenizer
    items, istats = sft_items(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True), calls, a.max_seq_len)
    lens = sorted(it.n for it in items)
    print(json.dumps({"train_calls": len(calls), "compaction_calls_left_out": compactions, "items": istats,
                      "reply_len_median": lens[len(lens) // 2] if lens else None, "reply_tokens": sum(lens)}))


def train_sft(a: argparse.Namespace, trainer: Any, calls: list[Any], compactions: int, out: Path, init: Path | None,
              rng: random.Random) -> dict[str, Any]:
    """``--objective sft``: the plays' calls as supervised items (:func:`sft_items`), up to ``--epochs`` passes over
    them in steps of ``--batch`` calls, each reply's tokens raised by maximum likelihood (ask_sd's ``Trainer.sft``),
    starting from ``init`` (``--init-adapter``) or the base model; then ``OUT/adapter``, ``OUT/phase.json`` and
    ``OUT/DONE``, whose summary is returned. With ``--val-run`` (held-out plays, every call an item as here:
    :func:`held_out_calls`) the held-out replies' negative log-likelihood per token is measured before the first step,
    then after each pass or, with ``--eval-steps`` N, every N steps (and at the last step, unless it was just
    measured); ``OUT/adapter`` is the adapter at the best of those, and with ``--patience`` P training stops once P
    measurements in a row have not improved on it. With no item the adapter is saved as it was. ``trainer``
    (:func:`make_trainer`) is the caller's, so one trainer can train phase after phase (realtime.py's resident
    trainer), each phase loading the last one's best adapter as its ``init`` rather than going on from wherever the last
    phase's steps left the weights."""
    if init:
        trainer.load(init)
    items, istats = sft_items(trainer.tokenizer, calls, a.max_seq_len)
    print(json.dumps({"items": istats, "compaction_calls_left_out": compactions}), flush=True)
    dev: list[Any] = []
    if a.val_run:
        dev, dstats = sft_items(trainer.tokenizer, held_out_calls(a.val_run, a.val_max_items, a.seed), a.max_seq_len)
        print(json.dumps({"dev_items": dstats}), flush=True)
    t0, steps, evals, capped = time.time(), [], [], False

    dev_chains = ([chain_item(p, a.chain_slack) for ch in find_chains(dev, a.chain_slack)
                   for p in chain_pieces(ch, a.batch, a.max_seq_len, a.chain_slack)]
                  if a.chains and dev else [])

    def dev_nll() -> float:
        """The held-out replies' mean negative log-likelihood per token under the adapter as it is now (with
        ``--chains``, each chain read once: :meth:`ask_sd.trainer.Trainer.nll_chains`)."""
        if dev_chains:
            res = [x for x in trainer.nll_chains(dev_chains)["items"] if x]
        else:
            res = [x for x in trainer.evaluate(dev, divergence=a.divergence)["items"] if x]
        return -sum(x["logp"] for x in res) / max(1, sum(x["tokens"] for x in res))

    def write_phase() -> None:
        (out / "phase.json").write_text(json.dumps({"objective": a.objective, "items": istats, "steps": steps,
                                                    "dev": evals, "args": vars(a)}, indent=1, default=str))

    best, since_best, measured = None, 0, 0
    if dev:
        best = dev_nll()
        evals.append({"epoch": 0, "step": 0, "dev_nll": round(best, 6), "best": True})
        print(json.dumps(evals[-1]), flush=True)

    def measure(epoch: int) -> bool:
        """The held-out loss now, the adapter saved when it is the best so far; whether ``--patience`` measurements in
        a row have now not improved on it."""
        nonlocal best, since_best, measured
        nll = dev_nll()
        improved = nll < best
        measured = len(steps)
        evals.append({"epoch": epoch, "step": len(steps), "dev_nll": round(nll, 6), "best": improved})
        print(json.dumps(evals[-1]), flush=True)
        write_phase()
        if improved:
            best, since_best = nll, 0
            trainer.save(out / "adapter")
        else:
            since_best += 1
        return bool(a.patience) and since_best >= a.patience

    chains = find_chains(items, a.chain_slack) if a.chains else []
    if a.chains:
        print(json.dumps({"chains": len(chains), "calls": len(items),
                          "calls_per_chain": round(len(items) / max(1, len(chains)), 2)}), flush=True)
    stop = False
    for epoch in range(1, a.epochs + 1):
        if a.chains:
            batches = chain_batches(chains, a.batch, a.gpus, 1, a.max_seq_len, rng, slack=a.chain_slack)
        else:
            order = items[:]
            rng.shuffle(order)
            batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
        for batch in batches:
            t = time.time()
            if a.chains:
                res = trainer.sft_chains([chain_item(piece, a.chain_slack) for piece in batch], rng=rng)
            else:
                res = trainer.sft(batch, rng=rng)
            taken = trainer.step()
            # nll_per_token: the batch's reply tokens' mean negative log-likelihood under the adapter as the step began
            record = {"step": len(steps) + 1, "epoch": epoch, "items": res["n_examples"], "tokens": res["tokens"],
                      "train_s": round(time.time() - t, 1),
                      "grad_norm": round(taken["grad_norm"], 5) if "grad_norm" in taken else None,
                      **{k: round(res[k], 6) if res[k] is not None else None
                         for k in ("nll_per_token", "loss_mean", "entropy_mean")}}
            if a.chains:
                record["read"] = {k: res["chains"][k] for k in ("chains", "tokens_read", "tokens_per_call")}
            steps.append(record)
            print(json.dumps(record), flush=True)
            write_phase()
            if time.time() - t0 > 60 * a.max_minutes or (a.max_steps and len(steps) >= a.max_steps):
                capped = time.time() - t0 > 60 * a.max_minutes
                stop = True
                break
            if dev and a.eval_steps and len(steps) % a.eval_steps == 0 and measure(epoch):
                stop = True
                break
        if dev and not a.eval_steps and measure(epoch):   # one measurement per pass
            stop = True
        if stop:
            break
    if dev and a.eval_steps and measured != len(steps):   # the steps after the last measurement, measured once
        measure(epoch)
    write_phase()
    if not dev:
        trainer.save(out / "adapter")   # no held-out plays: the adapter as training left it
    elif not (out / "adapter").exists():
        # no pass improved on the starting adapter: it is the best
        if init:
            shutil.copytree(init, out / "adapter", dirs_exist_ok=True)
        else:
            trainer.reset()
            trainer.save(out / "adapter")
    best_epoch = max((e["epoch"] for e in evals if e["best"]), default=None)
    best_step = max((e["step"] for e in evals if e["best"]), default=None)
    summary = {"objective": a.objective, "steps": len(steps), "time_capped": capped, "items": istats,
               "best_epoch": best_epoch, "best_step": best_step, "dev": evals, "last": steps[-1] if steps else None,
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "DONE").write_text(json.dumps(summary))
    print("done", json.dumps(summary), flush=True)
    return summary


def train_sdft(a: argparse.Namespace, trainer: Any, train_calls: list[Any], compactions: int, out: Path,
               init: Path | None, rng: random.Random, *, context: str = "", val_calls: list[Any] | None = None,
               demo_stats: dict[str, int] | None = None) -> dict[str, Any]:
    """``--objective sdft``: the student distils its teacher's behaviour on ``train_calls``, from ``init`` (or the base
    model), in steps of ``--batch`` (the module docstring: the teacher, the placement of its context, on- or
    off-policy replies); each call's teacher reads ``context``, or, with ``--demos``, the demonstration
    :func:`attach_demos` gave it (``demo_stats``, its counts, go into ``phase.json`` and ``DONE``). ``val_calls`` are
    scored before the first step (unless ``--no-initial-eval``), every ``--eval-every`` steps and after the last: with
    ``--demos`` by the training loss itself over their chains (``Trainer.distill_scores_chains``, each prompt read once
    a chain), otherwise by each call's divergence and KL to the teacher (:func:`mean_scores`). With ``--patience`` P
    they are a held-out loss as DAgger's is (:func:`train_sft`): ``OUT/adapter`` is the adapter at the best of them,
    and training stops once P in a row have not improved on it; without, they are for reading only and ``OUT/adapter``
    is the adapter as training left it. Then ``OUT/adapter`` (and ``OUT/ema`` with the moving average on),
    ``OUT/phase.json`` and ``OUT/DONE``, whose summary is returned. ``trainer`` (:func:`make_trainer`) is the caller's,
    so one trainer can train round after round (realtime.py's resident trainer): each call configures its teacher, its
    anchor and its starting adapter afresh."""
    val_calls = val_calls or []
    # the EMA teacher before the load, so the load restores the average saved with the adapter; the anchor after it,
    # so it is the adapter the round starts from
    trainer.configure_ema_teacher(decay=a.ema_decay)
    trainer.configure_teacher_adapter(Path(a.teacher_adapter).expanduser() if a.teacher_adapter else None)
    if init:
        trainer.load(init)
    else:
        trainer.reset()
    trainer.configure_adapter_anchor(mu=a.anchor_mu)
    tok = trainer.tokenizer
    val, vstats = build_items(tok, val_calls, context, a.placement, a.max_seq_len)
    val = val[: a.eval_items] if a.eval_items else val
    val_chains = sdft_chain_pairs(val, a) if a.demos else ([], [])
    if a.on_policy:
        prompts, tstats = train_calls, {"calls": len(train_calls), "drawn": 0, "draws_failed": 0, "too_long": 0}
    elif a.demos:
        # built a step at a time (each step's calls into chains), so a step's teacher prompts exist only while it trains
        prompts, tstats = train_calls, {"calls": len(train_calls), "items": 0, "too_long": 0}
    else:
        prompts, tstats = build_items(tok, train_calls, context, a.placement, a.max_seq_len)
    if demo_stats:
        tstats["demos"] = demo_stats
    print(json.dumps({"train": tstats, "compaction_calls_left_out": compactions, "val": vstats, "val_used": len(val),
                      "placement": a.placement, "on_policy": a.on_policy}), flush=True)
    if not prompts or (a.val_run and not val):
        raise SystemExit(f"no {'training' if not prompts else 'validation'} items (counts above): nothing to distil")

    t0, step, curve, steps = time.time(), 0, [], []
    steps_per_epoch = -(-len(prompts) // a.batch)
    held_out = bool(val and a.patience)   # the scores pick the adapter and stop training
    best, since_best = None, 0

    def write_phase() -> None:
        (out / "phase.json").write_text(json.dumps({"train_items": tstats, "val_items": vstats, "curve": curve,
                                                    "steps": steps, "args": vars(a)}, indent=1, default=str))

    def save() -> None:
        trainer.save(out / "adapter")
        if a.ema_decay is not None:
            trainer.save_ema(out / "ema")

    def evaluate() -> bool:
        """The validation score now, into the curve; with ``--patience``, the adapter saved when it is the best so far,
        and whether P scores in a row have now not improved on it."""
        nonlocal best, since_best
        if not val:
            return False
        point: dict[str, Any] = {"step": step, "epoch": round(step / steps_per_epoch, 2),
                                 "minutes": round((time.time() - t0) / 60, 1)}
        if a.demos:
            res = [x for x in trainer.distill_scores_chains(*val_chains, divergence=a.divergence,
                                                            skip_first=a.skip_first)["items"] if x]
            point["val"] = {"loss": sum(x["loss"] for x in res) / max(1, len(res)), "replies": len(res)}
            print(f"step {step} epoch {point['epoch']} held-out loss {point['val']['loss']:.5f} over "
                  f"{len(res)} replies | {point['minutes']} min", flush=True)
        else:
            point["val"] = mean_scores(trainer, val, a.divergence, a.anchor_mu)
            # against the fixed base model with the context: a moving teacher drifts toward the student, so only this
            # says whether the student moved toward the context's behaviour; with the teacher fixed (the base model or
            # a saved adapter) it is the same score
            point["val_fixed_teacher"] = (point["val"] if a.ema_decay is None
                                          else mean_scores(trainer, val, a.divergence, a.anchor_mu, base_teacher=True))
            v, f = point["val"], point["val_fixed_teacher"]
            print(f"step {step} epoch {point['epoch']} val div {v['teacher_div']:.5f} kl {v['teacher_kl']:.5f} | "
                  f"fixed teacher div {f['teacher_div']:.5f} kl {f['teacher_kl']:.5f} | {point['minutes']} min",
                  flush=True)
        if held_out:
            point["best"] = best is None or point["val"]["loss"] < best
            if point["best"]:
                best, since_best = point["val"]["loss"], 0
                save()
            else:
                since_best += 1
        curve.append(point)
        write_phase()
        return held_out and since_best >= a.patience

    batches = sdft_steps(prompts, a, rng) if a.demos else epoch_batches(prompts, a, rng)
    live = pool = pending = name = None
    if a.on_policy:
        from ask_sd.server import Server
        server = Server(a.server)
        live = LiveAdapter(trainer, server, out, a.model)
        pool = ThreadPoolExecutor(max_workers=a.batch)
        name = live.publish(0)   # the round's starting student
        pending = draw(pool, server, batches[0], name)

    if not a.no_initial_eval:
        evaluate()
    capped = stopped = False
    for k, batch in enumerate(batches):
        record: dict[str, Any] = {"step": step + 1}
        if a.on_policy:
            t = time.time()
            fresh = [c for f in pending for c in f.result()]
            record["wait_s"] = round(time.time() - t, 1)
            if k + 1 < len(batches):
                # the next batch is drawn by the adapter as of now while this one trains: one step stale
                pending = draw(pool, server, batches[k + 1], name)
            items, istats = build_items(tok, fresh, context, a.placement, a.max_seq_len)
            finish = collections.Counter(c.finish_reason for c in fresh)
            skipped = {k: istats[k] for k in ("too_long", "template_mismatch", "render_error")}
            record.update(drawn=len(fresh), draws_failed=len(batch) - len(fresh), items=len(items), **skipped,
                          retokenised=istats["retokenised"],
                          reply_tokens_mean=round(sum(len(c.reply_ids) for c in fresh) / max(1, len(fresh)), 1),
                          finish=dict(finish))
            tstats["drawn"] += len(fresh)
            tstats["draws_failed"] += len(batch) - len(fresh)
            for k, v in {**skipped, "retokenised": istats["retokenised"]}.items():
                tstats[k] = tstats.get(k, 0) + v
            if not items:
                raise SystemExit(f"step {step + 1}: no reply was drawn ({record})")
        elif a.demos:
            items, istats = build_items(tok, batch, context, a.placement, a.max_seq_len)
            skipped = {k: istats[k] for k in ("too_long", "template_mismatch", "render_error")}
            record.update(items=len(items), **skipped, retokenised=istats["retokenised"])
            tstats["items"] += len(items)
            for k, v in {**skipped, "retokenised": istats["retokenised"]}.items():
                tstats[k] = tstats.get(k, 0) + v
        else:
            items = batch
        t = time.time()
        if a.demos:
            # a step's games each in one pass for the teacher and one for the student (Trainer.distill_chains)
            res = distill_chain_step(trainer, items, a, rng)
            record.update(skipped_long=res["skipped_long"], tokens=res["tokens"],
                          tokens_read=res["chains"]["tokens_read"],
                          teacher_tokens_read=res["chains"]["teacher_tokens_read"],
                          tokens_per_call=res["chains"]["tokens_per_call"])
        else:
            # no per-token credit logs: nothing here reads them, and they are a decode per reply token
            res = trainer.accumulate(items, rng=rng, divergence=a.divergence, credit_items=0)
        trainer.step()
        step += 1
        record.update(loss=round(res["loss_mean"], 5) if res["loss_mean"] is not None else None,
                      train_s=round(time.time() - t, 1))
        if "drift" in res:
            # the student's ratio to the anchor (the round's starting adapter) on the replies' own tokens: off-policy
            # the staleness of the played replies; on-policy how far the student has moved since the round began
            record["drift"] = res["drift"]
        if a.on_policy:
            t = time.time()
            # the adapter that drew the batch just trained on is done; the one drawing the next batch stays
            live.retire(1)
            name = live.publish(step)
            record["publish_s"] = round(time.time() - t, 1)
        steps.append(record)
        print(json.dumps(record), flush=True)
        if step % a.eval_every == 0:
            stopped = evaluate()
        else:
            write_phase()
        if stopped:
            print(f"held-out stop at step {step}: {a.patience} scores in a row above {best:.5f}", flush=True)
            break
        if time.time() - t0 > 60 * a.max_minutes:
            print(f"time cap {a.max_minutes} min at step {step}", flush=True)
            capped = True
            break
    if pool is not None:
        # a stop or a time cap leaves the next batch's draws running: they finish before their adapter is unloaded
        # and deleted
        concurrent.futures.wait(pending)
        pool.shutdown()
    if not curve or curve[-1]["step"] != step:
        evaluate()
    if not held_out:
        save()
    if live is not None:
        live.retire(0)
    best_step = max((p["step"] for p in curve if p.get("best")), default=None)
    summary = {"steps": step, "epochs_done": round(step / steps_per_epoch, 2), "time_capped": capped,
               "held_out_stop": stopped, "items": tstats, "held_out": [round(p["val"]["loss"], 6) for p in curve],
               "best_step": best_step, "last": curve[-1] if curve else (steps[-1] if steps else None)}
    (out / "DONE").write_text(json.dumps(summary))
    print("done", json.dumps(summary), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=loop.MODEL, help="the base model the adapter is trained on (Hugging Face id)")
    p.add_argument("--objective", type=Objective, choices=list(Objective), default=Objective.sdft,
                   help="sdft: distil the context (needs --val-run and --context-file); pg: learn from the plays' "
                        "outcomes; opd: learn from a teacher's per-token scores of the plays (needs "
                        "--train-runs' own opd_scores.json, round.sh's score_opd); sft: imitate every reply of the "
                        "plays")
    p.add_argument("--train-runs", nargs="+", required=True)
    p.add_argument("--val-run", nargs="+", default=None, help="held-out plays (sft: several are read as one set)")
    p.add_argument("--val-max-items", type=int, default=0, dest="val_max_items",
                   help="sft: the held-out set as whole games drawn with --seed until their calls reach this (0: all)")
    p.add_argument("--eval-steps", type=int, default=0, dest="eval_steps",
                   help="sft with --val-run: measure the held-out loss every this many steps, --patience counting "
                        "measurements, rather than after each pass (0)")
    p.add_argument("--context-file", default=None)
    p.add_argument("--demos", nargs="+", default=None,
                   help="sdft: one labels folder per --train-runs (forward.py label's points.jsonl and labels/): each "
                        "call's teacher reads the expert's reply at it as a demonstration (attach_demos), in place of "
                        "--context-file; no --val-run needed")
    p.add_argument("--out", required=True)
    p.add_argument("--init-adapter", default=None,
                   help="continue from this saved adapter (weights, optimizer and, with the EMA teacher, its average)")
    p.add_argument("--placement", choices=loop.PLACEMENTS, default="message",
                   help="where the teacher reads the context: thought, the opening of its own reasoning; message, the "
                        "end of the last message")
    p.add_argument("--on-policy", action="store_true",
                   help="draw every step's replies fresh from the current student (served at --server) instead of "
                        "training on the replies played")
    p.add_argument("--server", default="http://localhost:8101/v1", help="the vLLM server the student is published to")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64, help="calls per optimizer step (one reply each)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--divergence", choices=sorted(DIVERGENCE_BETA), default="forward_kl")
    p.add_argument("--ema-decay", type=lambda s: None if s.strip().lower() == "none" else float(s), default=0.99,
                   help="the EMA teacher's decay per optimizer step (SDFT's 0.01 mixing rate); none: the frozen base "
                        "model is the teacher")
    p.add_argument("--teacher-adapter", default=None,
                   help="a saved adapter (an OUT/adapter folder) the teacher reads with, fixed for the run: the student "
                        "a round starts from; needs --ema-decay none")
    p.add_argument("--skip-first", type=int, default=0, dest="skip_first",
                   help="sdft with --demos: each reply's first ids left ungraded (SDFT's code: 3)")
    p.add_argument("--anchor-mu", type=float, default=2e-3,
                   help="the adapter anchor's weight (iSDFT's, Khamis et al. 2026; untuned for LoRA on harness calls); "
                        "0: no anchor")
    # the validation curve is for reading only, so it is sampled sparsely: 256 validation calls every 25 steps, against
    # both teachers, cost about 10% of the training compute
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-items", type=int, default=256, help="sdft: the validation calls scored (0: every one)")
    # a safety cap, not a schedule: one epoch on Qwen3.5-9B's round-0 play is about 260 minutes on 4 H100s (1.7
    # minutes a step, 142 steps, and the validation passes)
    p.add_argument("--max-minutes", type=float, default=300)
    p.add_argument("--max-seq-len", type=int, default=65536)
    p.add_argument("--chains", action="store_true",
                   help="sft: read each chain of a game's calls once, every reply a branch off it (ask_sd.branching), "
                        "in steps of --batch calls drawn as whole chains (chain_batches); the same loss and the same "
                        "calls per step as the default, which passes each call alone")
    p.add_argument("--chain-slack", type=int, default=CHAIN_SLACK,
                   help="--chains/pg/opd: a call joins the chain when its prompt begins with all but at most this "
                        "many of the previous call's last prompt ids, the difference read ungraded in its branch "
                        "(find_chains)")
    p.add_argument("--per-call", action="store_true",
                   help="pg/opd: pass each call as its own item (policy_gradient) instead of reading a game's calls "
                        "as chains (policy_gradient_chains, the default for these two objectives), for the "
                        "equivalence check and as a fallback")
    p.add_argument("--pg-reward", choices=("cost", "solve-first"), default="cost", dest="pg_reward",
                   help="pg: cost ranks every game (reward), or solves only, every failure scoring 0 (solve-first)")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (a smoke run, a cycle's block)")
    p.add_argument("--patience", type=int, default=0,
                   help="sft with --val-run: stop once this many measurements in a row (passes, or every --eval-steps "
                        "steps) have not lowered the held-out loss (0: run every pass of --epochs); sdft: the same, "
                        "measured every --eval-every steps, the best adapter kept (0: the scores are for reading only)")
    p.add_argument("--no-initial-eval", action="store_true",
                   help="skip the validation before the first step (a block continuing from the previous block's end)")
    p.add_argument("--dry-run", action="store_true", help="build the items with the tokenizer only and report their counts")
    a = p.parse_args()
    init = Path(a.init_adapter).expanduser() if a.init_adapter else None

    if a.demos and (a.objective is not Objective.sdft or len(a.demos) != len(a.train_runs)):
        p.error("--demos is sdft's, one labels folder per --train-runs")
    if a.demos and a.anchor_mu:
        p.error("--demos trains through Trainer.distill_chains, which has no anchor: --anchor-mu 0")
    demo_tok = None
    if a.demos:
        from transformers import AutoTokenizer
        demo_tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    train_calls, compactions, demo_stats = [], 0, collections.Counter()
    for k, r in enumerate(a.train_runs):
        calls, left_out = load_calls(Path(r).expanduser())
        if a.demos:
            calls, stats = attach_demos(demo_tok, calls, Path(a.demos[k]).expanduser())
            demo_stats.update(stats)
        train_calls += calls
        compactions += left_out
    rng = random.Random(a.seed)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if not a.dry_run:
        # OUT/kernels.json: the versions and GPU the linear-attention layers train on; no training on a stack with
        # wrong gradients (kernel_stack.py, docs/fla-hopper-gradients.md)
        kernel_stack.record(out)
    if a.objective is Objective.pg:
        records = {Path(r).expanduser().name: instance_records(Path(r).expanduser()) for r in a.train_runs}
        if a.dry_run:
            policy_gradient_dry_run(a, train_calls, pg_build(a, train_calls, records,
                                                             max(c.meta["round"] for c in train_calls)),
                                    "games", compactions)
        else:
            # a synchronous round: the adapter training starts from is the one that played the round
            train_pg(a, make_trainer(a), train_calls, records, compactions, out, init, init, rng)
        return
    if a.objective is Objective.opd:
        scores: dict[tuple[str, int, int], Any] = {}
        for r in a.train_runs:
            scores.update(load_opd_scores(Path(r).expanduser(), a.model))
        if a.dry_run:
            policy_gradient_dry_run(a, train_calls, opd_build(a, train_calls, scores), "items", compactions)
        else:
            train_opd(a, make_trainer(a), train_calls, scores, compactions, out, init, rng)
        return
    if a.objective is Objective.sft:
        if a.dry_run:
            sft_dry_run(a, train_calls, compactions)
        else:
            train_sft(a, make_trainer(a), train_calls, compactions, out, init, rng)
        return
    if not (a.demos or (a.val_run and a.context_file)):
        p.error("--objective sdft needs --demos, or --val-run and --context-file")
    if a.teacher_adapter and a.ema_decay is not None:
        p.error("--teacher-adapter is the teacher in place of the moving average: it needs --ema-decay none")
    # with --demos each call's demonstration is its whole teacher context (its note); validation is for reading only
    context = Path(a.context_file).expanduser().read_text() if a.context_file else ""
    val_calls = held_out_calls(a.val_run) if a.val_run else []
    val_calls = rng.sample(val_calls, min(a.eval_items * 2, len(val_calls)))

    if a.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        sample = rng.sample(train_calls, min(200, len(train_calls)))
        items, stats = build_items(tok, sample, context, a.placement, a.max_seq_len)
        lens = sorted(len(it.teacher_ids) for it in items)
        print(json.dumps({"train_calls": len(train_calls), "compaction_calls_left_out": compactions,
                          "val_calls": len(val_calls), "sample": stats,
                          "teacher_len_median": lens[len(lens) // 2] if lens else None,
                          "teacher_len_max": lens[-1] if lens else None,
                          "reply_len_median": sorted(it.n for it in items)[len(items) // 2] if items else None}))
        return

    train_sdft(a, make_trainer(a), train_calls, compactions, out, init, rng, context=context, val_calls=val_calls,
               demo_stats=dict(demo_stats))


if __name__ == "__main__":
    main()
