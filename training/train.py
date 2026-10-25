"""Training: a phase of one of the paper's trained rows (:func:`fit`), the same for every row but for its update -- DAgger
imitating the oracle's labels (``--objective sft``, :class:`Imitation`), OPSD distilling from them (realtime.py's opsd
row, :class:`SelfDistillation`), plain policy gradient on won games (``--objective pg``) or per-token distillation from
a teacher's scores (``--objective opd``, not a row of the paper; both :class:`PolicyGradient`) -- and context
distillation for the earlier experiments' rounds loop (``--objective sdft``, :func:`train_sdft`).

A phase (:func:`fit`): from ``--init-adapter`` (the base model without one), ``--epochs`` passes over the update's
units, each a call's prompt and the reply it trains on, every pass in an order of its own, in steps of ``--batch``
units and at most ``--max-steps`` in all, every pass planned before the first step (:func:`plan`): whole chains of a
game's calls (:func:`find_chains`, each prompt extending the last, read in one pass with every reply a branch off it)
cut into pieces each GPU can take within the window (``--chains`` for sft, the default for pg and opd, OPSD's only
way), or with ``--per-call`` (pg, opd) single calls. Each step is the update's gradient, then one optimizer step. With
held-out units the update's own held-out loss is measured before the first step, every ``--eval-steps`` steps (after
each pass when 0) and after the last; ``OUT/adapter`` is the adapter at the best of them, and ``--patience`` P stops the
phase once P in a row have not improved on it: trained to fit, no further. Without, ``OUT/adapter`` is the adapter as
training left it. ``--max-minutes`` caps a phase. ``OUT/phase.json`` holds the update's counts, every step's record and
every held-out measurement, ``OUT/DONE`` its summary. The updates:

- :class:`Imitation` (``sft``, DAgger): each call's reply is its target, the oracle's label at that point as the
  student writes it (a label set's call, forward.py label-set: the prompt as the local template renders it, never
  compared with the served ids), raised by maximum likelihood, every token weighing alike (ask_sd's
  ``Trainer.sft_chains``, or ``Trainer.sft`` per call). Held out: the held-out targets' negative log-likelihood per
  token.
- :class:`SelfDistillation` (OPSD): DAgger's units, and at each the current student's own reply, drawn fresh from the
  server it is published to after every step (the settings' ``server``, realtime.py's; the next step's drawn while
  this one trains, so at most one step stale), graded by the frozen base model reading the call's label, DAgger's
  target, as a demonstration (:func:`demo`, at the end of the call's last message): the forward KL over the full
  vocabulary at each reply position, the first ``skip_first`` ungraded (realtime.py's ``--skip-first``, SDFT's 3),
  averaged over the reply, each reply weighing alike
  (``Trainer.distill_chains``: SDFT's own loss and reduction, Shenfeld et al. 2026). Held out: that loss over the
  held-out games' replies as they were played.
- :class:`PolicyGradient` (``pg``, ``opd``): each call's played reply raised through PPO's clipped ratio against the
  adapter that played it (an item's own behaviour log-probabilities, :func:`attach_behaviour`, or else the adapter the
  phase starts from, a synchronous round's) by its advantage. ``pg`` is plain REINFORCE with no baseline
  (:func:`pg_items`): a game's reward is 1 if it solved its instance and 0 if not, and is its advantage, so a failure
  pushes nothing and only won games' calls are units. Its rule reads neither the benchmark's task ids, which a real
  stream does not have, nor the cost, whose worth plays out over the stream (a demonstration bought now pays off when
  its task returns), where no game's own reward can credit it. ``opd`` takes every call, carrying the teacher's
  per-token log-probabilities (:func:`opd_items`, the scores a round's own scoring step wrote beside the play,
  ``opd_scores.json``; the teacher's scorer is not in this folder), from which ``Trainer.policy_gradient`` takes the
  advantage
  position by position (the teacher's log-probability less the token's own under the adapter that played it). Held
  out: that loss over the held-out games' chains, the held-out wins' likelihood per token logged beside it.

The training prompts are a play's model calls made as the agent (:func:`load_calls`), never prime-agent's compaction
calls (its summarizer's, whose one message packs the whole conversation).

Context distillation (``--objective sdft``, the rounds loop's): the student learns to act without the context as its
teacher acts with it. Each call is one item: the student reads the call as served, the teacher reads the same call
with the context, and the loss compares them on every token of the reply. Where the context sits for the teacher
(``--placement``):

- ``thought``: the opening of the teacher's own reasoning, right after ``<think>``
  (``ask_sd.calls.thought_ids``; play's ``--context-placement thought`` shows the model the same text). A model
  refers to a document it reads, not to its own thoughts: on round 0's calls the teacher's push toward reference
  words was 16.5% of all it pushes as a notes block and 1.2% as its own reasoning;
- ``message`` (default, as play's): appended to the last message, wrapped as notes (``loop.context_block``), the
  teacher's prompt the served one with the note put in where the local template puts it (:func:`splice_note`).

Which reply (``--on-policy``):

- on-policy: every step's replies are drawn fresh from the current student, served by vLLM (``--server``) with the
  adapter reloaded after every step, so a batch is at most one step stale. A habit of several steps is taught a step
  further whenever the student's own replies reach the next one (trained on replies written before training, the
  student learned only first steps and collapsed);
- off-policy: the reply that was played.

Which teacher: the frozen base model (``--ema-decay none``); with ``--teacher-adapter``, a saved adapter read once and
fixed for the run, the student a round starts from (in the rounds loop, the last kept round's student, once a round is
kept); or, the default, SDFT's moving average (below).

Otherwise the round follows SDFT (Shenfeld et al. 2026): the teacher is the moving average of the student's adapter
(``--ema-decay``), carried from round to round in the saved adapter; the loss is the forward KL, KL(teacher || student),
over the full vocabulary (``--divergence``); an anchor adds ``--anchor-mu`` times the same divergence toward the adapter
the round started from (the base model in round 1); one pass over the prompts (``--epochs``), nothing selected. The
validation calls (``--val-run``: a play of tasks never trained on, the validation split's in the rounds loop), for
reading
only, are scored against the moving-average teacher and against the fixed base model with the context: the moving
teacher drifts toward the student, so only the second says whether the student moved toward the context's behaviour.
With a fixed teacher (``--ema-decay none``, with or without ``--teacher-adapter``) the two are the same score against
that teacher, computed once. It writes ``OUT/adapter`` (weights, optimizer and any moving average, for the next block's
or round's ``--init-adapter``), with the moving average on also ``OUT/ema`` (it as an adapter vLLM serves),
``OUT/phase.json`` (per step: the loss, the replies drawn and the seconds spent; every ``--eval-every`` steps: the
validation scores) and ``OUT/DONE``. ``--no-initial-eval`` skips the validation before the first step: a block that
continues from the previous one would only repeat that block's last.

Every objective first writes ``OUT/kernels.json`` (the versions and GPU the linear-attention layers train on) and stops
on a stack known to compute wrong gradients (``kernel_stack.py``). On-policy, ``OUT/live/`` holds the adapters
published to the server, the last two kept.

Run on the GPU node from prime-rl's checkout, on the GPUs the server does not hold (CUDA_VISIBLE_DEVICES), with
T=~/continual-arc/training:
  uv run --no-sync python $T/train.py --model Qwen/Qwen3.5-9B --train-runs ~/runs/loop/r0-stream \
      --val-run ~/runs/loop/r0-val --context-file ~/runs/loop/c1.md --out ~/runs/loop/b1-1 --on-policy \
      --ema-decay none [--teacher-adapter PATH] [--init-adapter PATH] [--gpus 4] [--dry-run]
  uv run --no-sync python $T/train.py --objective pg --train-runs ~/runs/paper/pg-u1/play-u1/rounds/r0005 \
      --out ~/runs/paper/pg-u1/train-u1/r5 --lr 1e-4 [--init-adapter PATH] [--gpus 8] [--per-call] [--dry-run]
  uv run --no-sync python $T/train.py --objective opd --train-runs ~/runs/loop/round-play1 --out ~/runs/loop/round1 \
      --lr 2e-5 [--init-adapter PATH] [--gpus 8] [--per-call] [--dry-run]   # round-play1/opd_scores.json must exist
  uv run --no-sync python $T/train.py --objective sft --train-runs ~/runs/loop/s-play1 --out ~/runs/loop/s1 \
      [--init-adapter PATH] [--gpus 8] [--batch 64] [--epochs 1] [--max-steps S] [--lr LR] [--dry-run] \
      [--val-run HELD_OUT --patience 2]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import gzip
import json
import os
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
import checkpoint  # noqa: E402
import kernel_stack  # noqa: E402
import loop  # noqa: E402
import render  # noqa: E402
from ask_sd.calls import ToolSchema  # noqa: E402
from ask_sd.trainer import DIVERGENCE_BETA, LORA_TARGETS  # noqa: E402

# the loop's server is vLLM 0.30 (serve.sh), whose tool schema the local rendering must reproduce
TOOL_SCHEMA = ToolSchema.vllm_0_30
# a fresh reply's length cap: replies run to a few thousand tokens (Qwen3.5-9B, round 0: median 566, p90 2,068), and
# the longest reply of a batch sets how long the batch takes to draw
DRAW_MAX_TOKENS = 8192
# a publish that fails (a server restarting comes back in a few minutes) is retried this often
PUBLISH_RETRIES, PUBLISH_BACKOFF_S = 20, 30
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
    it) carries no cost of its own, so :func:`load_calls` joins each game's in from here. Checked at ``run_dir``
    directly (a plain play, ``records.jsonl`` beside its ``captured/``) and, since ``paper.round_dir`` shards captured
    calls under ``run_dir/rounds/rNNNN`` while the run's own ``records.jsonl`` stays at the run directory's root, two
    directories up as well. Empty wherever neither holds one (a dry run before any round ended)."""
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
            # agent's and has no messages to render (overfit.chat_call's own test)
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


def demo(tok: Any, call: Any) -> str:
    """OPSD's demonstration for a labelled call (a label set's call, forward.py label-set: its prompt as the local
    template renders it and its reply the DAgger row's target there, dagger.student_message's rendering of the oracle's
    label), as its teacher reads it at the end of the call's last message (:data:`SDFT_DEMO`): the target as text, its
    thinking opened as the template's generation prompt opens it, its end of turn left off."""
    opened = tok.decode(list(call.prompt_ids[-4:])).endswith("<think>\n")
    return SDFT_DEMO.format(demo=("<think>\n" if opened else "") + tok.decode(list(call.reply_ids[:-1])))


def played_reply(call: Any, eos: int | None) -> list[int]:
    """The reply's ids as the model ended it: the server leaves out the end-of-turn token of a reply that stopped
    (``stop``) or called a tool (``tool_calls``), which the model did sample, so it goes back on."""
    reply = list(call.reply_ids)
    if call.finish_reason in ("stop", "tool_calls") and eos is not None and reply[-1] != eos:
        reply.append(eos)
    return reply


def pg_items(tokenizer: Any, calls: list[Any], max_seq_len: int) -> tuple[list[Any], dict[str, Any]]:
    """Plain policy gradient's items from plays' calls (:func:`load_calls`): REINFORCE with no baseline, each game's
    reward its advantage, 1 if it solved its instance and 0 if not. A failure's advantage is 0 and pushes nothing, so
    only a won game's calls are items, each the call's prompt as served and its reply as played, with advantage 1; a
    call longer than the window is skipped and counted (``too_long``), and a game whose outcome was never recorded is
    left out and counted (``unscored``). No baseline is subtracted: with wins as rare as the stream's, any mean of the
    run's games sits above every failure, so every token a failure played is pushed down, the tool calls' own format
    with the rest, which broke every PG run trained so (docs/method-search.md). ``trained`` in the stats
    lists every won game trained on."""
    from ask_sd.trainer import TokenItem
    eos = tokenizer.eos_token_id
    episodes: dict[tuple[str, int], list[Any]] = collections.defaultdict(list)
    for call in calls:
        episodes[(call.meta["run"], call.meta["user"])].append(call)
    stats: dict[str, Any] = {"episodes": len(episodes), "unscored": 0, "wins": 0, "items": 0, "too_long": 0,
                             "trained": []}
    items = []
    for (run, user), ep in sorted(episodes.items()):
        solved = ep[0].meta["solved"]
        if solved is None:
            stats["unscored"] += 1
            continue
        if not solved:
            continue
        stats["wins"] += 1
        stats["trained"].append({"run": run, "user": user, "round": ep[0].meta["round"]})
        for call in ep:
            reply = played_reply(call, eos)
            ids = list(call.prompt_ids) + reply
            if len(ids) > max_seq_len:
                stats["too_long"] += 1
                continue
            items.append(TokenItem(ids, ids, len(reply), label=dict(call.meta), advantage=1.0))
    stats["items"] = len(items)
    return items, stats

def load_opd_scores(run_dir: Path, model: str | None = None) -> dict[tuple[str, int, int], "list[float | None] | None"]:
    """A run's teacher scores (written beside its captured calls as ``opd_scores.json`` by a scoring step, whose
    scorer is not in this folder), keyed by (run, user, i) as :func:`load_calls` labels a call. With ``model``, refuses
    a file scored for a different student: ``played_reply`` depends on the student tokenizer's end-of-turn id, which
    ask-sd's length check alone would not catch a mismatch on."""
    doc = json.loads((run_dir / "opd_scores.json").read_text())
    student = doc.get("student_model")
    if model and student and student != model:
        raise ValueError(f"{run_dir / 'opd_scores.json'} was scored for student {student!r}, not {model!r}")
    return {(it["run"], it["user"], it["i"]): it["scores"] for it in doc["items"]}


_UNATTEMPTED = object()   # opd_items' own sentinel: a call load_opd_scores' dict has no key for at all, never
                          # sent to the teacher (the scorer's own too_long) -- distinct from a key present
                          # with scores None, which means the teacher was asked and came back with nothing (a live
                          # teacher-health signal)


def opd_items(tokenizer: Any, calls: list[Any],
              scores: dict[tuple[str, int, int], "list[float | None] | None"],
              max_seq_len: int) -> tuple[list[Any], dict[str, Any]]:
    """OPD's items from plays' calls (:func:`load_calls`) and the teacher's scores of them (:func:`load_opd_scores`):
    one item per call, its prompt as served and its reply as played, carrying the teacher's per-token log-probabilities
    (``Trainer.policy_gradient`` takes the advantage from them itself). A call absent from ``scores`` entirely is
    never counted as the teacher's failure (``excluded``): the scorer scores every call it is given but one too long
    for its own window, which is no dead-teacher signal. A call present with ``scores`` None --
    the teacher was asked and gave nothing -- is ``unscored``, and so is a call longer than the window."""
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
    oracle hints send, oracle_hint.play_note), and one with a ``demo`` (:func:`demo`) reads that text alone, as
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
    two of the student's at most (beside the adapters other plays use: serve.sh's --max-loras)."""

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
        shutil.rmtree(self.root, ignore_errors=True)   # and its folders (a resume's drawer is its checkpoint's own)
        self.prefix = f"{tag}-t{int(time.time())}"
        self.published: list[tuple[str, Path, Path]] = []   # (served name, its folder, the folder vLLM read), oldest first

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
        return self._load(step, path)

    def publish_saved(self, step: int, src: Path) -> str:
        """The adapter saved at ``src`` (a checkpoint's copy of the one that drew the batch in flight when the phase
        stopped), loaded on the server as the student of ``step``; its served name."""
        path = self.root / f"s{step:04d}"
        shutil.rmtree(path, ignore_errors=True)
        checkpoint.link_tree(src, path)
        return self._load(step, path)

    def newest(self) -> Path:
        """The newest published adapter's folder."""
        return self.published[-1][1]

    def _load(self, step: int, path: Path) -> str:
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
        self.published.append((name, path, served))
        return name

    def retire(self, keep: int) -> None:
        """Unload and delete every published adapter but the newest ``keep``."""
        while len(self.published) > keep:
            name, path, served = self.published.pop(0)
            self.server.unload_adapter(name)
            for folder in {path, served}:
                shutil.rmtree(folder, ignore_errors=True)


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


# torch's compiled kernels (FlexAttention's, every shape a step's packs bring) kept on the boot disk, which a preempted
# machine keeps, not in torch's default /tmp/torchinductor_<user>, which its restart clears: a trainer started after
# one spent 203, 106 and 59 s on its first three steps compiling them again, against 25, 25 and 18 s from this cache
# (docs/method-search.md). The tuning of fla's kernels (~/.triton/cache) and TileLang's kernels
# (~/.tilelang) are on the boot disk already.
INDUCTOR_CACHE = Path.home() / ".cache" / "torchinductor"


def make_trainer(a: argparse.Namespace) -> Any:
    """The trainer the arguments ask for: data parallel over ``--gpus`` GPUs, or one GPU, its compiled kernels cached in
    :data:`INDUCTOR_CACHE` (the data-parallel workers take this process's environment)."""
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(INDUCTOR_CACHE))
    kw = dict(rank=a.rank, alpha=a.alpha, lr=a.lr, max_seq_len=a.max_seq_len, system_prompt=None, share_prompts=True,
              lora_targets=tuple(a.lora_targets))
    if a.gpus > 1:
        from ask_sd.parallel import DataParallelTrainer
        return DataParallelTrainer(a.model, world_size=a.gpus, device="cuda", **kw)
    from ask_sd.trainer import Trainer
    return Trainer(a.model, device="cuda", **kw)


# the most calls of one chain a piece holds (--chain-calls): a step's chains and the held-out ones are cut so whatever the
# GPUs that train them, so a phase's steps, and its checkpoints' batches, are the same on any number of them (a run's
# share of a node's pool changes between and within phases: gpupool.py). 16 is what a batch of 64 on 4 GPUs gave when
# the cut was batch/GPUs, as on the final runs' trainers but dagger-u1-lag1's and pg-u1-lag1's (2 GPUs: 32). A piece's
# tokens are bounded by the window either way, so its memory does not change with it
CHAIN_CALLS = 16


def share(a: argparse.Namespace) -> int:
    """The most calls of one chain a piece holds (:data:`CHAIN_CALLS`)."""
    return a.chain_calls


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
# last prompt ids (a reply with no reasoning renders in the next prompt as an empty think block, whose newlines
# tokenize as one id where the earlier prompt ended with one; 46% of the round-1 model's calls broke there)
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
    # getattr, not c.teacher_logp: an item built against an ask-sd whose TokenItem has no teacher_logp (sft_items'
    # plain items) has no such attribute at all, not just None. extra stays empty for sft_items' calls, so this is
    # ChainItem(trunk, branches, labels, heads) exactly: only pg_items'/opd_items' calls, which need ChainItem's
    # advantage and log-probability fields, add to it
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


def chain_batches(chains: list[list[Any]], batch: int, cap: int, epochs: int, max_len: int,
                  rng: random.Random, max_steps: int = 0, slack: int = CHAIN_SLACK) -> list[list[list[Any]]]:
    """``epochs`` passes over the chains' calls in steps of ``batch`` calls, each step made of whole chains drawn at
    random: each pass shuffles the chains (never single calls: two calls of one chain in one pass share their prefix, two
    calls drawn apart share nothing), cuts each into pieces of at most ``cap`` calls and the window
    (:func:`chain_pieces`), and fills steps with the pieces in order, splitting a piece at a call boundary where the step
    fills up; the data-parallel trainer deals a step's pieces to its GPUs by their cost, however many they are. Every
    call is in exactly one step per pass, every step holds ``batch`` calls but a pass's last, and a step is a list of
    pieces (lists of consecutive calls of one chain). At most ``max_steps`` steps."""
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


def sdft_chain_pairs(items: list[Any], a: argparse.Namespace) -> tuple[list[Any], list[Any]]:
    """:func:`build_items`' items as ``Trainer.distill_chains`` takes them: their games' chains (:func:`find_chains`),
    cut so that both passes fit the window and every GPU gets a piece (:func:`sdft_pieces`), each piece's chain and its
    teacher chain."""
    students, teachers = [], []
    for chain in find_chains(items, a.chain_slack):
        for piece in sdft_pieces(chain, share(a), a.max_seq_len, a.chain_slack):
            student = chain_item(piece, a.chain_slack)
            students.append(student)
            teachers.append(teacher_chain_item(piece, student))
    return students, teachers


def pg_record(res: dict[str, Any]) -> dict[str, Any]:
    """A policy-gradient step's record for ``pg`` and ``opd`` alike (:class:`PolicyGradient`; :func:`fit` adds the
    step's number and gradient norm), from ask-sd's result: the items and tokens trained on, the played tokens' mean
    log-probability under the adapter as the step began (``logp_per_token``) and the usual policy-gradient metrics,
    the share of tokens ask-sd's dual clip silenced
    and the step's token of largest |log r| (``log_ratio_worst``: its call's label, position, id and log-probabilities
    under the adapter training and the one that played it, which is pruned soon after, so that a step whose ratios ran
    away can be traced to its token from ``phase.json`` alone), plus ``opd`` (the mean, absolute mean and spread of the
    advantage over scored positions, and the share of positions with no teacher value; the two spread fields only once
    ask-sd's loss reports them) when the step held any OPD items."""
    record = {"items": res["n_examples"], "tokens": res["tokens"], "skipped_long": res["skipped_long"],
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
# this share of a round's calls, opd_build refuses rather than train on whatever fraction of the round survived
OPD_UNSCORED_MAX_FRAC = 0.02


def policy_gradient_dry_run(a: argparse.Namespace, calls: list[Any],
                            build_items: Callable[[Any], tuple[list[Any], dict[str, Any]]], stats_key: str,
                            compactions: int) -> None:
    """``--objective pg`` or ``opd`` with ``--dry-run``: the round's items' counts, built with the tokenizer alone."""
    from transformers import AutoTokenizer
    items, item_stats = build_items(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True))
    print(json.dumps({"train_calls": len(calls), "compaction_calls_left_out": compactions, stats_key: item_stats,
                      "reply_len_median": sorted(it.n for it in items)[len(items) // 2] if items else None}))


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
    pieces = [piece for ch in find_chains(items, a.chain_slack)
              for piece in chain_pieces(ch, share(a), a.max_seq_len, a.chain_slack)]
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


def opd_build(a: argparse.Namespace, calls: list[Any],
              scores: dict[tuple[str, int, int], Any]) -> Callable[[Any], tuple[list[Any], dict[str, Any]]]:
    """``--objective opd``'s item-builder for :class:`PolicyGradient`: :func:`opd_items` over ``calls``,
    refusing (``SystemExit``) a round with no items to train on, or where the teacher failed to score more than
    :data:`OPD_UNSCORED_MAX_FRAC` of the calls it was actually asked to score: a dead or partly-dead teacher must stop
    the round, not quietly train on whatever survived and bias the learning-rate choice without failing. The
    fraction is over calls the teacher was asked to score (``calls - excluded``), never over every call `load_calls`
    kept: a call never asked (one too long for the scorer's window) says nothing about the teacher's health."""

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


# -- a phase of the paper's trained rows: one plan, one loop, and the update the row's own --------------------------

def plan(units: list[Any], a: argparse.Namespace, rng: random.Random,
         chains: bool = True) -> list[tuple[int, list[list[Any]]]]:
    """A phase's steps, every pass planned before the first step: ``--epochs`` passes over ``units`` (items, each a
    call's prompt and the reply it trains on, labelled with its game and place), each in an order of its own, in steps
    of ``--batch`` units, at most ``--max-steps`` in all; each step its pass (from 1) and its pieces. With ``chains`` a
    pass shuffles whole chains of a game's calls (:func:`find_chains`), cut into pieces each GPU can take within the
    window (:func:`chain_batches`), so that a step's calls share their games' prompts; without, single units, a piece
    each. Every row plans so, so the same units give the same steps whatever the update draws from ``rng`` after."""
    chained = find_chains(units, a.chain_slack) if chains else []
    steps: list[tuple[int, list[list[Any]]]] = []
    for p in range(1, a.epochs + 1):
        if chains:
            got = chain_batches(chained, a.batch, share(a), 1, a.max_seq_len, rng, slack=a.chain_slack)
        else:
            order = units[:]
            rng.shuffle(order)
            got = [[[u] for u in order[i:i + a.batch]] for i in range(0, len(order), a.batch)]
        steps += [(p, step) for step in got]
    return steps[: a.max_steps or len(steps)]


def held_chains(items: list[Any], a: argparse.Namespace) -> list[Any]:
    """Held-out items as a held-out loss reads them: their games' chains (:func:`find_chains`) in pieces each GPU can
    take within the window, one :class:`ask_sd.trainer.ChainItem` each."""
    return [chain_item(piece, a.chain_slack) for chain in find_chains(items, a.chain_slack)
            for piece in chain_pieces(chain, share(a), a.max_seq_len, a.chain_slack)]


def place(meta: dict[str, Any]) -> tuple[str, int, int]:
    """A call's game and place in it (:func:`load_calls`' run, user and index)."""
    return meta["run"], meta["user"], meta["i"]


class Update:
    """What a row's phase does that is the row's own, all :func:`fit` leaves to it: its ``units``, the items the steps
    are planned over (:func:`plan`, as chain pieces when ``chains``), each step's gradient over them (:meth:`step`,
    accumulated for the optimizer step fit takes), and its held-out loss (:meth:`held_out`) over ``held_items``.
    :meth:`start`, :meth:`stepped`, :meth:`keep`, :meth:`end` and :meth:`close` are what it needs around the steps
    (OPSD serves the student its replies are drawn from), by default nothing but keeping the plan. ``stats`` are its
    counts (JSON-able, kept with a checkpoint), under ``key`` in phase.json and DONE, and ``held_stats`` its held-out
    ones."""

    objective: Objective
    key = "items"
    chains = True
    units: list[Any]
    held_items: list[Any]
    stats: dict[str, Any]
    held_stats: dict[str, Any] | None = None

    @property
    def holds_out(self) -> bool:
        return bool(self.held_items)

    def start(self, trainer: Any, steps: list[list[list[Any]]], out: Path, at: int, drawer: Path | None,
              pause: Callable[[], None]) -> None:
        """Before step ``at`` (0, or the step a checkpoint resumes after; ``drawer``, the checkpoint's kept adapter,
        :meth:`keep`); ``pause`` is where SIGTERM may end the process between steps (checkpoint.Checkpoints.step)."""
        self.steps = steps

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        """Step ``k``'s gradient (``self.steps[k]``), accumulated; its record's fields."""
        raise NotImplementedError

    def stepped(self, trainer: Any, n: int) -> dict[str, Any]:
        """After the optimizer step of a phase that goes on (``n`` steps taken); its record's fields."""
        return {}

    keep: Callable[[Path], None] | None = None   # what a checkpoint keeps beside the trainer's save, into its extra/

    def end(self) -> None:
        """After the last step, before the phase's last held-out measurement: what is still in flight finishes."""

    def close(self) -> None:
        """Once the phase's adapter is written: what the update served is taken down."""

    def held_out(self, trainer: Any) -> dict[str, Any]:
        """The held-out loss under the adapter as it is now, {"loss": ..., ...}."""
        raise NotImplementedError


class Imitation(Update):
    """DAgger's update (``--objective sft``): every call's reply is its target (a label set's call: the oracle's label
    at that point as the student writes it), raised by maximum likelihood, every token weighing alike (ask_sd's
    ``Trainer.sft_chains`` over a step's chain pieces, or ``Trainer.sft`` per call with ``--chains`` off); a call longer
    than the window is left out (:func:`sft_items`). Its held-out loss is the held-out targets' negative
    log-likelihood per token (``Trainer.nll_chains``, or ``Trainer.evaluate`` per call)."""

    objective = Objective.sft

    def __init__(self, a: argparse.Namespace, tok: Any, calls: list[Any], held: list[Any]) -> None:
        self.a, self.chains = a, a.chains
        self.units, self.stats = sft_items(tok, calls, a.max_seq_len)
        self.held_items, self.held_stats = sft_items(tok, held, a.max_seq_len)
        self.held_read = held_chains(self.held_items, a) if self.chains else self.held_items

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        if self.chains:
            res = trainer.sft_chains([chain_item(piece, self.a.chain_slack) for piece in self.steps[k]], rng=rng)
        else:
            res = trainer.sft([u for piece in self.steps[k] for u in piece], rng=rng)
        # nll_per_token: the step's reply tokens' mean negative log-likelihood under the adapter as the step began
        record = {"items": res["n_examples"], "tokens": res["tokens"],
                  **{name: round(res[name], 6) if res[name] is not None else None
                     for name in ("nll_per_token", "loss_mean", "entropy_mean")}}
        if self.chains:
            record["read"] = {name: res["chains"][name] for name in ("chains", "tokens_read", "tokens_per_call")}
        return record

    def held_out(self, trainer: Any) -> dict[str, Any]:
        if self.chains:
            res = [x for x in trainer.nll_chains(self.held_read)["items"] if x]
        else:
            res = [x for x in trainer.evaluate(self.held_items, divergence=self.a.divergence)["items"] if x]
        return {"loss": -sum(x["logp"] for x in res) / max(1, sum(x["tokens"] for x in res))}


class SelfDistillation(Update):
    """OPSD's update, DAgger's with SDFT's loss (realtime.py's opsd row): the units are DAgger's, the labelled calls as
    :class:`Imitation` takes them, so the same calls give OPSD the steps they give DAgger. At each, the current
    student's own reply is drawn fresh from the server (``--server``) it is published to after every step
    (:class:`LiveAdapter`; a step's replies are drawn while the step before it trains, so at most one step stale), and
    the frozen base model is its teacher, reading the call's label, DAgger's target there, as a demonstration
    (:func:`demo`, at the end of the call's last message: :func:`build_items`). A step's loss is SDFT's (Shenfeld et
    al. 2026, and its code's reduction): the forward KL over the full vocabulary at each position of a reply but its
    first ``skip_first`` (realtime.py's ``--skip-first``), averaged over the reply, each reply weighing alike
    (``Trainer.distill_chains``, a game's calls in one pass for the student and one for the teacher). A drawn reply
    whose student or teacher pass the window cannot hold is left out and counted (``drawn_too_long``). Its held-out
    loss is that loss over the replies the held-out games played (``held``: the plays of the held-out labelled calls,
    each carrying its label's demonstration), the same replies at every measurement (``Trainer.distill_scores_chains``).
    A checkpoint keeps the student that drew the batch in flight (:meth:`keep`), which draws it again on resume, so
    every batch is drawn by the student one step before the one it trains, as without the stop."""

    objective = Objective.sdft

    def __init__(self, a: argparse.Namespace, tok: Any, calls: list[Any], held: list[Any]) -> None:
        self.a, self.tok = a, tok
        self.units, self.stats = sft_items(tok, calls, a.max_seq_len)
        for call in calls:
            call.meta["demo"] = demo(tok, call)
        self.calls = {place(call.meta): call for call in calls}
        self.stats.update(drawn=0, draws_failed=0, drawn_too_long=0, template_mismatch=0, render_error=0,
                          retokenised=0)
        self.held_items = held
        items, self.held_stats = build_items(tok, held, "", "message", a.max_seq_len)
        self.held = sdft_chain_pairs(items, a)
        self.live: LiveAdapter | None = None
        self.drawer: Path | None = None   # the student drawing the batch in flight

    @property
    def holds_out(self) -> bool:
        return bool(self.held[0])   # a held-out call whose teacher pass the window cannot hold is no held-out item

    def start(self, trainer: Any, steps: list[list[list[Any]]], out: Path, at: int, drawer: Path | None,
              pause: Callable[[], None]) -> None:
        from ask_sd.server import Server
        super().start(trainer, steps, out, at, drawer, pause)
        trainer.configure_ema_teacher(decay=None)   # the frozen base model is the teacher
        trainer.configure_teacher_adapter(None)
        trainer.configure_adapter_anchor(mu=0.0)
        self.pause, self.pending = pause, []
        if at == len(steps):
            return
        self.server = Server(self.a.server)
        self.live = LiveAdapter(trainer, self.server, out, self.a.model)
        self.pool = ThreadPoolExecutor(max_workers=self.a.batch)
        # the batch in flight: the phase's first, drawn by its starting student, or on resume the one the stop left,
        # drawn again by the student that drew it then
        self.name = self.live.publish_saved(at - 1, drawer) if at else self.live.publish(0)
        self.pending = self.ask(at)
        if at:
            self.name = self.live.publish(at)

    def ask(self, k: int) -> list[Future]:
        """Step ``k``'s replies, drawn from the student as it is served now."""
        self.drawer = self.live.newest()
        return draw(self.pool, self.server, [self.calls[place(u.label)] for piece in self.steps[k] for u in piece],
                    self.name)

    def keep(self, folder: Path) -> None:
        """With each checkpoint, the student drawing its next batch."""
        if self.drawer is not None:
            checkpoint.link_tree(self.drawer, folder / "drawer")

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        t = time.time()
        fresh = [c for f in self.pending for c in f.result()]
        record: dict[str, Any] = {"wait_s": round(time.time() - t, 1)}
        # SIGTERM in the wait, the draws then likely failed with the server: the checkpoint is of the step before,
        # this batch to be drawn again by the same student
        self.pause()
        if k + 1 < len(self.steps):
            self.pending = self.ask(k + 1)   # drawn by the student as of now while this step trains: one step stale
        items, istats = build_items(self.tok, fresh, "", "message", self.a.max_seq_len)
        record.update(drawn=len(fresh), draws_failed=sum(len(piece) for piece in self.steps[k]) - len(fresh),
                      items=len(items), drawn_too_long=istats["too_long"],
                      **{name: istats[name] for name in ("template_mismatch", "render_error", "retokenised")},
                      reply_tokens_mean=round(sum(len(c.reply_ids) for c in fresh) / max(1, len(fresh)), 1),
                      finish=dict(collections.Counter(c.finish_reason for c in fresh)))
        for name in ("drawn", "draws_failed", "drawn_too_long", "template_mismatch", "render_error", "retokenised"):
            self.stats[name] += record[name]
        if not items:
            raise SystemExit(f"step {k + 1}: no reply was drawn ({record})")
        res = trainer.distill_chains(*sdft_chain_pairs(items, self.a), rng=rng, divergence=self.a.divergence,
                                     skip_first=self.a.skip_first)
        record.update(skipped_long=res["skipped_long"], tokens=res["tokens"],
                      **{name: res["chains"][name] for name in ("tokens_read", "teacher_tokens_read", "tokens_per_call")},
                      loss=round(res["loss_mean"], 5) if res["loss_mean"] is not None else None)
        return record

    def stepped(self, trainer: Any, n: int) -> dict[str, Any]:
        t = time.time()
        self.live.retire(1)   # the adapter that drew the step just trained on is done; the one drawing the next stays
        self.name = self.live.publish(n)
        return {"publish_s": round(time.time() - t, 1)}

    def end(self) -> None:
        if self.live is None:
            return
        # a stop or a time cap leaves the next step's draws running: they finish before their adapter is unloaded
        concurrent.futures.wait(self.pending)
        self.pool.shutdown()

    def close(self) -> None:
        # only now: the phase's last measurement can find a new best, whose checkpoint keeps the drawer (keep)
        if self.live is not None:
            self.live.retire(0)

    def held_out(self, trainer: Any) -> dict[str, Any]:
        res = [x for x in trainer.distill_scores_chains(*self.held, divergence=self.a.divergence,
                                                        skip_first=self.a.skip_first)["items"] if x]
        return {"loss": sum(x["loss"] for x in res) / max(1, len(res)), "replies": len(res)}


class PolicyGradient(Update):
    """PG's and OPD's update (``--objective pg``, ``opd``): every unit's played reply raised through PPO's clipped ratio
    against the adapter that played it, by its advantage (:func:`pg_items`' 1, a won game's reward, or
    :func:`opd_items`' teacher scores, from which ask_sd takes it position by position): ``Trainer.policy_gradient_chains``
    over a step's chain pieces, or with ``--per-call`` ``Trainer.policy_gradient``. A unit carrying its reply's
    log-probabilities under the adapter that played it (:func:`attach_behaviour`) is clipped against them, any other
    against ``played_by`` (the base model when None). Its held-out loss is the same loss over the held-out units' games
    (``Trainer.policy_gradient_scores_chains``), the held-out wins' negative log-likelihood per token logged beside it
    (``Trainer.nll_chains``)."""

    def __init__(self, a: argparse.Namespace, objective: Objective, items: list[Any], stats: dict[str, Any],
                 key: str, held: list[Any] | None = None, played_by: Path | None = None) -> None:
        self.a, self.objective, self.key, self.played_by = a, objective, key, played_by
        self.units, self.stats, self.chains = items, stats, not a.per_call
        self.held_items = held or []
        self.held = held_chains(self.held_items, a)
        self.wins = held_chains([it for it in self.held_items if it.label.get("solved")], a)
        self.held_stats = {"items": len(self.held_items)}

    def start(self, trainer: Any, steps: list[list[list[Any]]], out: Path, at: int, drawer: Path | None,
              pause: Callable[[], None]) -> None:
        super().start(trainer, steps, out, at, drawer, pause)
        if self.played_by:   # after any load, which drops it
            trainer.configure_behaviour(self.played_by)
        else:
            trainer.configure_behaviour(base=True)

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        if self.chains:
            res = trainer.policy_gradient_chains([chain_item(piece, self.a.chain_slack) for piece in self.steps[k]],
                                                 rng=rng)
        else:
            res = trainer.policy_gradient([u for piece in self.steps[k] for u in piece], rng=rng)
        return pg_record(res)

    def held_out(self, trainer: Any) -> dict[str, Any]:
        res = [x for x in trainer.policy_gradient_scores_chains(self.held)["items"] if x]
        point = {"loss": sum(x["loss"] for x in res) / max(1, len(res)), "spans": len(res)}
        if self.wins:
            nll = [x for x in trainer.nll_chains(self.wins)["items"] if x]
            point["wins_nll"] = round(-sum(x["logp"] for x in nll) / max(1, sum(x["tokens"] for x in nll)), 6)
        return point


def fit(a: argparse.Namespace, trainer: Any, update: Update, out: Path, init: Path | None,
        rng: random.Random) -> dict[str, Any]:
    """A phase of training, the same for every row but for ``update`` (:class:`Update`): from ``init``
    (``--init-adapter``, or the base model), the steps :func:`plan` draws over the update's units, each the update's
    gradient and one optimizer step. With held-out units the update's held-out loss is measured before the first step,
    every ``--eval-steps`` steps (after each pass when 0) and after the last; ``OUT/adapter`` is the adapter at the best
    of them, the starting one while none improves on it, and with ``--patience`` P training stops once P in a row have
    not improved on the best. Without, ``OUT/adapter`` is the adapter as training left it. ``--max-minutes`` stops it
    too. Then ``OUT/phase.json`` (the update's counts, every step's record, every held-out measurement) and
    ``OUT/DONE``, whose summary is returned. ``trainer`` (:func:`make_trainer`) is the caller's, so one trainer trains
    phase after phase (realtime.py's resident trainer), each from the adapter it is given.

    The best adapter is kept as a checkpoint (checkpoint.py), and with ``checkpoint.ON`` so is the phase on SIGTERM
    and every ``checkpoint.EVERY`` steps: a phase cut short goes on from its last one, its steps planned again from
    ``rng``'s state at the start, as they were (each step's record keeps its batch's digest, ``batch``, which the replay
    must match), and the steps already trained skipped. The summary names the steps the phase resumed from
    (``resumed``)."""
    if init:
        trainer.load(init)
    else:
        trainer.reset()
    # the loop's state, all of it in the checkpoint: the steps taken, the held-out curve, its best and patience count,
    # the update's counts, and whether a stop or the time cap has ended training
    run: dict[str, Any] = {"steps": [], "evals": [], "best": None, "since_best": 0, "stats": update.stats,
                           "capped": False, "stopped": False}
    ck = checkpoint.Checkpoints(out, trainer, rng, checkpoint.fingerprint(a, init, update.units, update.held_items),
                                lambda: run, extra=update.keep)
    steps = plan(update.units, a, rng, update.chains)
    saved = ck.load()
    if saved and not checkpoint.replays([step for _, step in steps], saved["loop"]["steps"]):
        ck.discard("its batches do not replay")
        saved = None
    if saved:
        run.update(saved["loop"])
        ck.resume(saved)
    records, evals = run["steps"], run["evals"]
    update.stats = run["stats"]
    update.start(trainer, [step for _, step in steps], out, len(records),
                 ck.extra_dir(saved, "drawer") if saved else None, lambda: ck.step(len(records)))
    print(json.dumps({"objective": update.objective, update.key: update.stats, "held_out_items": update.held_stats,
                      "steps": len(steps)}), flush=True)

    def write_phase() -> None:
        (out / "phase.json").write_text(json.dumps({"objective": update.objective, update.key: update.stats,
                                                    "held_out_items": update.held_stats, "steps": records,
                                                    "held_out": evals, "args": vars(a)}, indent=1, default=str))

    def due(n: int) -> bool:
        """Whether a held-out measurement falls after ``n`` steps: every --eval-steps, or at each pass's end."""
        return n % a.eval_steps == 0 if a.eval_steps else n == len(steps) or steps[n][0] != steps[n - 1][0]

    def measure(final: bool = False) -> bool:
        """The held-out loss now, into the measurements, and a checkpoint after it when the loss is the best so far,
        to hold the best adapter, or one is due where the phase goes on; whether ``--patience`` measurements in a row
        have now not improved on the best."""
        n = len(records)
        point = {"step": n, **update.held_out(trainer)}
        point["best"] = run["best"] is None or point["loss"] < run["best"]
        if point["best"]:
            run["best"], run["since_best"] = point["loss"], 0
        else:
            run["since_best"] += 1
        point["loss"] = round(point["loss"], 6)
        evals.append(point)
        print(json.dumps({"held_out": point}), flush=True)
        write_phase()
        stop = bool(a.patience) and run["since_best"] >= a.patience
        if not n:
            ck.start_is_best(init)   # before the first step: the best is the adapter the phase starts from
        elif point["best"] or (ck.due(n) and not (final or stop or run["capped"] or n == len(steps))):
            ck.save(n, best=point["best"])
        return stop

    def after_step() -> None:
        """The held-out measurement or the checkpoint a step's end is due, once (a resume ends where one was due)."""
        n = len(records)
        if update.holds_out and due(n):
            if evals[-1]["step"] != n and measure():
                run["stopped"] = True
                print(f"held-out stop at step {n}: {a.patience} measurements in a row above {run['best']:.6f}",
                      flush=True)
        elif ck.due(n) and n < len(steps):
            ck.save(n)

    if not saved and update.holds_out:
        measure()
    elif saved and records and not (run["stopped"] or run["capped"]):
        after_step()
    for k in range(len(records), len(steps)):
        if run["stopped"] or run["capped"]:
            break
        ck.step(k)   # SIGTERM in the last step's measurement: the process's end
        t = time.time()
        record = {"step": k + 1, "pass": steps[k][0], **update.step(trainer, k, rng)}
        taken = trainer.step()
        record.update(grad_norm=round(taken["grad_norm"], 5) if "grad_norm" in taken else None,
                      train_s=round(time.time() - t, 1), batch=checkpoint.batch_digest(steps[k][1]))
        records.append(record)
        write_phase()
        if ck.minutes() > a.max_minutes:
            print(f"time cap {a.max_minutes} min at step {k + 1}", flush=True)
            run["capped"] = True
        ck.step(k + 1)   # SIGTERM: a checkpoint, and the process's end
        if not run["capped"]:
            after_step()
        if not (run["stopped"] or run["capped"]) and k + 1 < len(steps):
            record.update(update.stepped(trainer, k + 1))
        print(json.dumps(record), flush=True)
    update.end()
    if update.holds_out and evals[-1]["step"] != len(records):   # the steps since the last measurement, measured once
        measure(final=True)
    write_phase()
    ck.write_adapter(out / "adapter", init)   # the best one's, or with nothing held out the adapter as training left it
    update.close()
    best_step = max((e["step"] for e in evals if e["best"]), default=None)
    summary = {"objective": update.objective, "steps": len(records), "time_capped": run["capped"],
               "held_out_stop": run["stopped"], update.key: update.stats, "held_out": [e["loss"] for e in evals],
               "best_step": best_step, "last": records[-1] if records else None, **ck.record()}
    (out / "DONE").write_text(json.dumps(summary))
    ck.clear()
    print("done", json.dumps(summary), flush=True)
    return summary


def train_sdft(a: argparse.Namespace, trainer: Any, train_calls: list[Any], compactions: int, out: Path,
               init: Path | None, rng: random.Random, *, context: str, val_calls: list[Any]) -> dict[str, Any]:
    """``--objective sdft``, context distillation (the module docstring: the teacher, the placement of its context, on-
    or off-policy replies): the student distils its teacher's behaviour on ``train_calls``, each call's teacher reading
    ``context``, from ``init`` (or the base model), in steps of ``--batch``. ``val_calls``, for reading, are scored
    before the first step (unless ``--no-initial-eval``), every ``--eval-every`` steps and after the last, by each
    call's divergence and KL to the teacher (:func:`mean_scores`). Then ``OUT/adapter`` as training left it (and
    ``OUT/ema`` with the moving average on), ``OUT/phase.json`` and ``OUT/DONE``, whose summary is returned.

    With ``checkpoint.ON`` the phase is checkpointed on SIGTERM and every ``checkpoint.EVERY`` steps (checkpoint.py): a
    phase cut short goes on from its last checkpoint, its batches drawn again from ``rng``'s state at the start, as
    they were, and the steps already trained skipped. On-policy, a checkpoint keeps the adapter that drew the batch in
    flight (its ``extra/drawer``), which draws it again on resume, so every batch is drawn by the student one step
    before the one it trains, as without the stop. Each step's record keeps its batch's digest (``batch``), and the
    summary the steps the phase resumed from (``resumed``)."""
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
    if a.on_policy:
        prompts, tstats = train_calls, {"calls": len(train_calls), "drawn": 0, "draws_failed": 0, "too_long": 0}
    else:
        prompts, tstats = build_items(tok, train_calls, context, a.placement, a.max_seq_len)
    print(json.dumps({"train": tstats, "compaction_calls_left_out": compactions, "val": vstats, "val_used": len(val),
                      "placement": a.placement, "on_policy": a.on_policy}), flush=True)
    if not prompts or not val:
        raise SystemExit(f"no {'training' if not prompts else 'validation'} items (counts above): nothing to distil")

    steps_per_epoch = -(-len(prompts) // a.batch)
    # the loop's state, all of it in the checkpoint: the steps taken, the validation curve, the counts so far, and
    # whether the time cap has ended training
    run: dict[str, Any] = {"step": 0, "steps": [], "curve": [], "tstats": tstats, "capped": False}
    drawing: dict[str, Path | None] = {"adapter": None}   # on-policy: the adapter drawing the batch in flight

    def keep(folder: Path) -> None:
        """With each checkpoint, the adapter that draws its next batch (none when no draw is pending)."""
        if drawing["adapter"] is not None:
            checkpoint.link_tree(drawing["adapter"], folder / "drawer")

    ck = checkpoint.Checkpoints(out, trainer, rng, checkpoint.fingerprint(a, init, prompts, val), lambda: run,
                                extra=keep if a.on_policy else None)
    batches = epoch_batches(prompts, a, rng)
    saved = ck.load()
    if saved and not checkpoint.replays(batches, saved["loop"]["steps"]):
        ck.discard("its batches do not replay")
        saved = None
    if saved:
        run.update(saved["loop"])
        ck.resume(saved)
    steps, curve, tstats = run["steps"], run["curve"], run["tstats"]

    def write_phase() -> None:
        (out / "phase.json").write_text(json.dumps({"train_items": tstats, "val_items": vstats, "curve": curve,
                                                    "steps": steps, "args": vars(a)}, indent=1, default=str))

    def evaluate(final: bool = False) -> None:
        """The validation scores now, into the curve, and a checkpoint after them when one is due where the phase goes
        on."""
        step = run["step"]
        point: dict[str, Any] = {"step": step, "epoch": round(step / steps_per_epoch, 2),
                                 "minutes": round(ck.minutes(), 1),
                                 "val": mean_scores(trainer, val, a.divergence, a.anchor_mu)}
        # against the fixed base model with the context: a moving teacher drifts toward the student, so only this says
        # whether the student moved toward the context's behaviour; with the teacher fixed (the base model or a saved
        # adapter) it is the same score
        point["val_fixed_teacher"] = (point["val"] if a.ema_decay is None
                                      else mean_scores(trainer, val, a.divergence, a.anchor_mu, base_teacher=True))
        v, f = point["val"], point["val_fixed_teacher"]
        print(f"step {step} epoch {point['epoch']} val div {v['teacher_div']:.5f} kl {v['teacher_kl']:.5f} | "
              f"fixed teacher div {f['teacher_div']:.5f} kl {f['teacher_kl']:.5f} | {point['minutes']} min", flush=True)
        curve.append(point)
        write_phase()
        if step and ck.due(step) and not final and step < len(batches):
            ck.save(step)

    def after_step() -> None:
        """The validation scores or the checkpoint a step's end is due, once (a resume ends where one was due)."""
        step = run["step"]
        if step % a.eval_every == 0:
            if not curve or curve[-1]["step"] != step:
                evaluate()
        elif ck.due(step) and step < len(batches):
            ck.save(step)

    live = pool = pending = name = None
    if a.on_policy:
        from ask_sd.server import Server
        server = Server(a.server)
        live = LiveAdapter(trainer, server, out, a.model)
        pool = ThreadPoolExecutor(max_workers=a.batch)
        # the batch in flight: the round's first, drawn by its starting student, or on resume the one the stop left,
        # drawn again by the student that drew it then
        pending = []
        if run["step"] < len(batches):
            if run["step"]:
                name = live.publish_saved(run["step"] - 1, ck.extra_dir(saved, "drawer"))
            else:
                name = live.publish(0)
            drawing["adapter"] = live.newest()
            pending = draw(pool, server, batches[run["step"]], name)
            if run["step"]:
                name = live.publish(run["step"])

    if not saved and not a.no_initial_eval:
        evaluate()
    elif saved and run["step"] and not run["capped"]:
        after_step()
    for k in range(run["step"], len(batches)):
        if run["capped"]:
            break
        ck.step(run["step"])   # SIGTERM in the last step's score or publish: the process's end
        batch = batches[k]
        record: dict[str, Any] = {"step": k + 1, "batch": checkpoint.batch_digest(batch)}
        if a.on_policy:
            t = time.time()
            fresh = [c for f in pending for c in f.result()]
            record["wait_s"] = round(time.time() - t, 1)
            # SIGTERM in the wait, the draws then likely failed with the server: the checkpoint is of the step before,
            # this batch to be drawn again by the same student
            ck.step(run["step"])
            if k + 1 < len(batches):
                # the next batch is drawn by the adapter as of now while this one trains: one step stale
                pending = draw(pool, server, batches[k + 1], name)
                drawing["adapter"] = live.newest()
            items, istats = build_items(tok, fresh, context, a.placement, a.max_seq_len)
            skipped = {key: istats[key] for key in ("too_long", "template_mismatch", "render_error")}
            record.update(drawn=len(fresh), draws_failed=len(batch) - len(fresh), items=len(items), **skipped,
                          retokenised=istats["retokenised"],
                          reply_tokens_mean=round(sum(len(c.reply_ids) for c in fresh) / max(1, len(fresh)), 1),
                          finish=dict(collections.Counter(c.finish_reason for c in fresh)))
            tstats["drawn"] += len(fresh)
            tstats["draws_failed"] += len(batch) - len(fresh)
            for key, v in {**skipped, "retokenised": istats["retokenised"]}.items():
                tstats[key] = tstats.get(key, 0) + v
            if not items:
                raise SystemExit(f"step {k + 1}: no reply was drawn ({record})")
        else:
            items = batch
        t = time.time()
        # no per-token credit logs: nothing here reads them, and they are a decode per reply token
        res = trainer.accumulate(items, rng=rng, divergence=a.divergence, credit_items=0)
        trainer.step()
        run["step"] = k + 1
        record.update(loss=round(res["loss_mean"], 5) if res["loss_mean"] is not None else None,
                      train_s=round(time.time() - t, 1))
        if "drift" in res:
            # the student's ratio to the anchor (the round's starting adapter) on the replies' own tokens: off-policy
            # the staleness of the played replies; on-policy how far the student has moved since the round began
            record["drift"] = res["drift"]
        steps.append(record)
        write_phase()
        if ck.minutes() > a.max_minutes:
            print(f"time cap {a.max_minutes} min at step {run['step']}", flush=True)
            run["capped"] = True
        ck.step(run["step"])   # SIGTERM: a checkpoint, and the process's end
        if not run["capped"]:
            after_step()
        if a.on_policy and not run["capped"] and run["step"] < len(batches):
            t = time.time()
            # the adapter that drew the batch just trained on is done; the one drawing the next batch stays
            live.retire(1)
            name = live.publish(run["step"])
            record["publish_s"] = round(time.time() - t, 1)
        print(json.dumps(record), flush=True)
    if pool is not None:
        # a time cap leaves the next batch's draws running: they finish before their adapter is unloaded and deleted
        concurrent.futures.wait(pending)
        pool.shutdown()
    if not curve or curve[-1]["step"] != run["step"]:   # --no-initial-eval and no score due yet: one now
        evaluate(final=True)
    write_phase()
    ck.write_adapter(out / "adapter", init)   # the adapter as training left it
    if a.ema_decay is not None:
        trainer.save_ema(out / "ema")
    if live is not None:
        live.retire(0)
    summary = {"steps": run["step"], "epochs_done": round(run["step"] / steps_per_epoch, 2),
               "time_capped": run["capped"], "items": tstats, "last": curve[-1], **ck.record()}
    (out / "DONE").write_text(json.dumps(summary))
    ck.clear()
    print("done", json.dumps(summary), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=loop.MODEL, help="the base model the adapter is trained on (Hugging Face id)")
    p.add_argument("--objective", type=Objective, choices=list(Objective), default=Objective.sdft,
                   help="sdft: distil the context (needs --val-run and --context-file); pg: learn from the plays' "
                        "outcomes; opd: learn from a teacher's per-token scores of the plays (needs "
                        "--train-runs' own opd_scores.json); sft: imitate every reply of the plays")
    p.add_argument("--train-runs", nargs="+", required=True)
    p.add_argument("--val-run", nargs="+", default=None, help="held-out plays (sft: several are read as one set)")
    p.add_argument("--val-max-items", type=int, default=0, dest="val_max_items",
                   help="sft: the held-out set as whole games drawn with --seed until their calls reach this (0: all)")
    p.add_argument("--eval-steps", type=int, default=0, dest="eval_steps",
                   help="sft with --val-run: measure the held-out loss every this many steps, --patience counting "
                        "measurements, rather than after each pass (0)")
    p.add_argument("--context-file", default=None)
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
    p.add_argument("--lora-targets", nargs="+", default=list(LORA_TARGETS), dest="lora_targets",
                   help="the projections the adapter trains, by module name as PEFT matches it; the default, ask-sd's, "
                        "is every linear layer of a Llama-shaped block, which leaves Qwen3.5's Gated DeltaNet layers "
                        "frozen: add in_proj_qkv in_proj_z out_proj to train their mixing too. An --init-adapter "
                        "trained on fewer loads into the modules it has, and the rest start fresh")
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
    p.add_argument("--chain-calls", type=int, default=CHAIN_CALLS, dest="chain_calls",
                   help="--chains/pg/opd: the most calls of one chain a step's piece holds, whatever --gpus "
                        "(CHAIN_CALLS)")
    p.add_argument("--per-call", action="store_true",
                   help="pg/opd: pass each call as its own item (policy_gradient) instead of reading a game's calls "
                        "as chains (policy_gradient_chains, the default for these two objectives), for the "
                        "equivalence check and as a fallback")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (a smoke run, a cycle's block)")
    p.add_argument("--patience", type=int, default=0,
                   help="sft with --val-run: stop once this many measurements in a row (passes, or every --eval-steps "
                        "steps) have not lowered the held-out loss, the best adapter kept (0: run every pass of "
                        "--epochs)")
    p.add_argument("--no-initial-eval", action="store_true",
                   help="skip the validation before the first step (a block continuing from the previous block's end)")
    p.add_argument("--dry-run", action="store_true", help="build the items with the tokenizer only and report their counts")
    a = p.parse_args()
    init = Path(a.init_adapter).expanduser() if a.init_adapter else None

    train_calls, compactions = [], 0
    for r in a.train_runs:
        calls, left_out = load_calls(Path(r).expanduser())
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
        if a.dry_run:
            policy_gradient_dry_run(a, train_calls, lambda tok: pg_items(tok, train_calls, a.max_seq_len), "games",
                                    compactions)
            return
        trainer = make_trainer(a)
        items, games = pg_items(trainer.tokenizer, train_calls, a.max_seq_len)
        # a synchronous round: the adapter training starts from is the one that played the round
        fit(a, trainer, PolicyGradient(a, Objective.pg, items, games, "games", played_by=init), out, init, rng)
        return
    if a.objective is Objective.opd:
        scores: dict[tuple[str, int, int], Any] = {}
        for r in a.train_runs:
            scores.update(load_opd_scores(Path(r).expanduser(), a.model))
        if a.dry_run:
            policy_gradient_dry_run(a, train_calls, opd_build(a, train_calls, scores), "items", compactions)
            return
        trainer = make_trainer(a)
        items, stats = opd_build(a, train_calls, scores)(trainer.tokenizer)
        # OPD's rounds are synchronous: the adapter training starts from is the one that played the round
        fit(a, trainer, PolicyGradient(a, Objective.opd, items, stats, "items", played_by=init), out, init, rng)
        return
    if a.objective is Objective.sft:
        if a.dry_run:
            sft_dry_run(a, train_calls, compactions)
            return
        trainer = make_trainer(a)
        held = held_out_calls(a.val_run, a.val_max_items, a.seed) if a.val_run else []
        fit(a, trainer, Imitation(a, trainer.tokenizer, train_calls, held), out, init, rng)
        return
    if not (a.val_run and a.context_file):
        p.error("--objective sdft needs --val-run and --context-file")
    if a.teacher_adapter and a.ema_decay is not None:
        p.error("--teacher-adapter is the teacher in place of the moving average: it needs --ema-decay none")
    context = Path(a.context_file).expanduser().read_text()
    val_calls = held_out_calls(a.val_run)
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

    train_sdft(a, make_trainer(a), train_calls, compactions, out, init, rng, context=context, val_calls=val_calls)


if __name__ == "__main__":
    main()
