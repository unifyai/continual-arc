"""The trained rows' three updates, all a phase leaves to its row (:func:`continual_arc_training.phase.fit`): the units
its steps are planned over, each step's gradient, and its held-out loss.

- :class:`Imitation` (DAgger): each labelled call's reply is its target, the oracle's label at that point as the student
  writes it (the label set's call: its prompt as the local template renders it), raised by maximum likelihood, every
  token weighing alike. Held out: the held-out targets' negative log-likelihood per token.
- :class:`SelfDistillation` (OPSD): DAgger's units, and at each the current student's own reply, drawn fresh from the
  model server it is published to after every step (:class:`LiveAdapter`; a step's replies are drawn while the step
  before it trains, so at most one step stale), graded by the frozen base model reading the call's label, DAgger's
  target, as a demonstration (:func:`demo`, at the end of the call's last message): the forward KL over the full
  vocabulary at each reply position but the first ``skip_first``, averaged over the reply, each reply weighing alike
  (SDFT's loss and reduction). Held out: that loss over the replies the held-out games played.
- :class:`PolicyGradient` (PG): REINFORCE with no baseline (:func:`pg_items`): a game's reward is 1 if it solved its
  instance and 0 if not, and is its advantage, so a failure pushes nothing and only a won game's calls are units, each
  raised through PPO's clipped ratio against the adapter that played it, whose reply log-probabilities each unit
  carries (:func:`attach_behaviour`). Its rule reads neither the benchmark's task ids nor the cost, whose worth plays
  out over the stream (a demonstration bought now pays off when its task returns), where no game's own reward can
  credit it. Held out: the same loss over the held-out games' wins, their negative log-likelihood per token logged
  beside it.

The training prompts are a play's calls made as the agent (:func:`continual_arc_training.calls.load_calls`), never
prime-agent's compaction calls."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import random
import shutil
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from continual_arc_training import checkpoints
from continual_arc_training.calls import Call, render_prompt_ids, resample, with_note
from continual_arc_training.chains import TokenItem, as_chains, chain_item, common_prefix, distill_chains
from continual_arc_training.server import Server, served_adapter

# SDFT's teacher prompt (as SDFT's paper, section 3, and its code word it): the demonstration after the question, in
# the question's own turn; for a harness call, the end of its last message, the one the reply answers
SDFT_DEMO = "\n\nThis is an example for a response to the question:\n{demo}\nNow answer with a response of your own, including the thinking process."
# a fresh reply's length cap: replies run to a few thousand tokens (Qwen3.5-9B: median 566, p90 2,068), and the longest
# reply of a batch sets how long the batch takes to draw
DRAW_MAX_TOKENS = 8192
# a publish that fails (a server restarting comes back in a few minutes) is retried this often
PUBLISH_RETRIES, PUBLISH_BACKOFF_S = 20, 30


def place(meta: dict[str, Any]) -> tuple[str, int, int]:
    """A call's game and place in it (its run, user and index)."""
    return meta["run"], meta["user"], meta["i"]


def sft_items(tok: Any, calls: list[Call], max_seq_len: int) -> tuple[list[TokenItem], dict[str, Any]]:
    """One item per call, its prompt then its reply as it ended (:meth:`Call.played_reply`), the reply the target; a
    call longer than the window is skipped and counted."""
    stats: dict[str, Any] = {"calls": len(calls), "items": 0, "too_long": 0}
    items = []
    for call in calls:
        reply = call.played_reply(tok.eos_token_id)
        ids = list(call.prompt_ids) + reply
        if len(ids) > max_seq_len:
            stats["too_long"] += 1
            continue
        items.append(TokenItem(ids, ids, len(reply), label=dict(call.meta)))
    stats["items"] = len(items)
    return items, stats


def played_items(tok: Any, calls: list[Call], max_seq_len: int) -> list[TokenItem]:
    """Every call's reply as played after its prompt as served, one item a call, a call longer than the window left
    out: what :func:`behaviour_logps` scores."""
    return sft_items(tok, calls, max_seq_len)[0]


def pg_items(tok: Any, calls: list[Call], max_seq_len: int) -> tuple[list[TokenItem], dict[str, Any]]:
    """Plain policy gradient's items (the module docstring): each won game's calls, its prompt as served and its reply
    as played, with advantage 1; a call longer than the window is skipped and counted (``too_long``), and a game whose
    outcome was never recorded is left out and counted (``unscored``). No baseline is subtracted: with wins as rare as
    the stream's, any mean of the run's games sits above every failure, so every token a failure played would be pushed
    down, the tool calls' own format with the rest. ``trained`` lists every won game trained on."""
    eos = tok.eos_token_id
    episodes: dict[tuple[str, int], list[Call]] = collections.defaultdict(list)
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
            reply = call.played_reply(eos)
            ids = list(call.prompt_ids) + reply
            if len(ids) > max_seq_len:
                stats["too_long"] += 1
                continue
            items.append(TokenItem(ids, ids, len(reply), label=dict(call.meta), advantage=1.0))
    stats["items"] = len(items)
    return items, stats


def demo(tok: Any, call: Call) -> str:
    """OPSD's demonstration for a labelled call (a label set's call: its reply the DAgger row's target there), as its
    teacher reads it at the end of the call's last message (:data:`SDFT_DEMO`): the target as text, its thinking opened
    as the template's generation prompt opens it, its end of turn left off."""
    opened = tok.decode(list(call.prompt_ids[-4:])).endswith("<think>\n")
    return SDFT_DEMO.format(demo=("<think>\n" if opened else "") + tok.decode(list(call.reply_ids[:-1])))


def splice_note(served: list[int], local: list[int], noted: list[int]) -> list[int] | None:
    """The teacher's prompt ids: ``served`` (a call's prompt as the server tokenised it) with the note that ``noted``
    (the local rendering with the note) adds to ``local`` (the local rendering without it), put in by position. The
    served ids can split an earlier turn's text otherwise than a fresh rendering would (they hold the replies as the
    model sampled them, and it can sample a split the tokenizer would not make), so the two renderings are not compared
    whole. They agree up to where the note starts, and what follows that point in ``local`` (the last message's end and
    the generation prompt) ends ``served`` too; the note goes in there, before it. None when ``served`` does not end
    that way."""
    cut = common_prefix(local, noted)
    tail = local[cut:]
    if len(tail) > len(served) or served[len(served) - len(tail):] != tail:
        return None
    return served[:len(served) - len(tail)] + noted[cut:]


def distill_items(tok: Any, calls: list[Call], max_seq_len: int) -> tuple[list[TokenItem], dict[str, Any]]:
    """Self-distillation's items, one per call carrying its demonstration (``meta["demo"]``, :func:`demo`): the
    student's ids are the call's prompt as served and its reply; the teacher's, the served prompt with the demonstration
    at the end of its last message, put in where the local template puts it (:func:`splice_note`), and the same reply.
    A call the local template cannot render (``render_error``) or whose served prompt does not end as the template's
    does (``template_mismatch``) is skipped and counted, one whose earlier turns the served ids split otherwise than a
    fresh rendering would is counted in ``retokenised``, and one longer than the window is skipped and counted."""
    stats: dict[str, Any] = {"calls": len(calls), "items": 0, "template_mismatch": 0, "too_long": 0, "render_error": 0,
                             "retokenised": 0}
    items = []
    for call in calls:
        try:
            local = render_prompt_ids(tok, call.request)
            noted = render_prompt_ids(tok, call.request, with_note(call.request["messages"], call.meta["demo"]))
        except Exception:  # noqa: BLE001  (a call the local template cannot render is skipped)
            stats["render_error"] += 1
            continue
        served = list(call.prompt_ids)
        teacher = splice_note(served, local, noted)
        if teacher is None:
            stats["template_mismatch"] += 1
            continue
        stats["retokenised"] += local != served
        reply = call.played_reply(tok.eos_token_id)
        item = TokenItem(served + reply, teacher + reply, len(reply), label=dict(call.meta))
        if max(len(item.student_ids), len(item.teacher_ids)) > max_seq_len:
            stats["too_long"] += 1
            continue
        items.append(item)
    stats["items"] = len(items)
    return items, stats


def behaviour_logps(trainer: Any, a: argparse.Namespace, items: list[TokenItem]) -> dict[tuple[str, int, int], list[float]]:
    """``items``' replies' token log-probabilities under the trainer's behaviour adapter (its configure_behaviour), read
    as chains, keyed by each call's run, user and place in its game; a call in a chain longer than the window is left
    out, as training leaves it out."""
    chains = as_chains(items, a.chain_calls, a.max_seq_len, a.chain_slack)
    got = trainer.behaviour_logps_chains(chains)["items"]
    labels = [label for c in chains for label in c.labels]
    return {place(label): logps for label, logps in zip(labels, got, strict=True) if logps is not None}


def attach_behaviour(items: list[TokenItem], logps: dict[tuple[str, int, int], list[float]]) -> list[TokenItem]:
    """Each item with its reply's log-probabilities under the adapter that played it (:func:`behaviour_logps`), which
    its ratios are taken against; an item none were taken for, or of another length, is refused (ValueError)."""
    for it in items:
        key = place(it.label)
        if key not in logps or len(logps[key]) != it.n:
            raise ValueError(f"attach_behaviour: no behaviour log-probabilities for {key} ({it.n} reply tokens)")
        it.behaviour_logp = logps[key]
    return items


def pg_record(res: dict[str, Any]) -> dict[str, Any]:
    """A policy-gradient step's record (:func:`continual_arc_training.phase.fit` adds the step's number and gradient
    norm): the items and tokens trained on, the played tokens' mean log-probability under the adapter as the step began
    (``logp_per_token``), the usual policy-gradient measures, the share of tokens the dual clip silenced, and the
    step's token of largest |log r| (``log_ratio_worst``), so that a step whose ratios ran away can be traced to its
    token from ``phase.json`` alone once the adapter that played it is pruned."""
    return {"items": res["n_examples"], "tokens": res["tokens"], "skipped_long": res["skipped_long"],
            "logp_per_token": (round(sum(x for x in res["logp"] if x is not None) / res["tokens"], 5)
                               if res["tokens"] else None),
            **{k: round(res[k], 6) if res[k] is not None else None
               for k in ("loss_mean", "ratio_mean", "clip_fraction", "dual_clip_fraction", "approx_kl",
                         "entropy_mean")},
            "log_ratio_worst": res["log_ratio_worst"]}


class LiveAdapter:
    """The student as the server serves it while OPSD trains: after every step its adapter is saved under ``out/live``
    and loaded under a new name, and the one before is retired once no draw uses it, so the server holds two of the
    student's at most (beside the adapters the runs play on: serve.sh's ``--max-loras``)."""

    def __init__(self, trainer: Any, server: Server, out: Path, model: str) -> None:
        self.trainer, self.server, self.model = trainer, server, model
        self.root = out / "live"
        # named for the run as well as the phase (out's last three parts), so runs sharing one server never take each
        # other's; each attempt's names are its own, since an attempt that died leaves its last adapters loaded and vLLM
        # refuses to load a name it already serves
        tag = "-".join(out.resolve().parts[-3:])
        for name in server.models():
            if name.startswith(tag + "-"):
                server.unload_adapter(name)
        shutil.rmtree(self.root, ignore_errors=True)
        self.prefix = f"{tag}-t{int(time.time())}"
        self.published: list[tuple[str, Path, Path]] = []   # (served name, its folder, the folder vLLM read), oldest first

    def publish(self, step: int) -> str:
        """The adapter as it is now, loaded on the server; its served name. Every rank holds the same adapter, so rank
        0's is the one saved."""
        path = self.root / f"s{step:04d}"
        shutil.rmtree(path, ignore_errors=True)
        self.trainer.model.save_pretrained(str(path))
        return self._load(step, path)

    def publish_saved(self, step: int, src: Path) -> str:
        """The adapter saved at ``src`` (a checkpoint's copy of the one that drew the batch in flight when the phase
        stopped), loaded on the server as the student of ``step``; its served name."""
        path = self.root / f"s{step:04d}"
        shutil.rmtree(path, ignore_errors=True)
        checkpoints.link_tree(src, path)
        return self._load(step, path)

    def newest(self) -> Path:
        """The newest published adapter's folder."""
        return self.published[-1][1]

    def _load(self, step: int, path: Path) -> str:
        served = served_adapter(self.model, path)
        name = f"{self.prefix}-s{step}"
        for attempt in range(PUBLISH_RETRIES + 1):
            try:
                self.server.load_adapter(name, served)
                break
            except OSError as error:   # a server that restarts refuses for a few minutes
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


def draw(pool: ThreadPoolExecutor, server: Server, calls: list[Call], model: str) -> list[Future]:
    """One fresh reply from ``model`` at each call's prompt, the call's own request otherwise (its sampling settings,
    tools and thinking), capped at :data:`DRAW_MAX_TOKENS`; the draws run while the current step trains."""

    def one(call: Call) -> list[Call]:
        try:
            return resample(server, call, model, 1, max_tokens=DRAW_MAX_TOKENS)
        except Exception as error:  # noqa: BLE001  (a reply cut off mid-read is a failed draw, counted, not fatal)
            print(f"a draw failed: {type(error).__name__}: {error}", flush=True)
            return []
    return [pool.submit(one, c) for c in calls]


class Update:
    """What a row's phase does that is the row's own: its ``units``, the items the steps are planned over, each step's
    gradient over them (:meth:`step`, accumulated for the optimizer step the phase takes), and its held-out loss
    (:meth:`held_out`) over ``held_items``. :meth:`start`, :meth:`stepped`, :meth:`keep`, :meth:`end` and
    :meth:`close` are what it needs around the steps (OPSD serves the student its replies are drawn from), by default
    nothing but keeping the plan. ``stats`` are its counts (JSON-able, kept with a checkpoint), under ``key`` in
    phase.json and DONE, and ``held_stats`` its held-out ones."""

    objective: str
    key = "items"
    units: list[TokenItem]
    held_items: list[Any]
    stats: dict[str, Any]
    held_stats: dict[str, Any] | None = None
    keep: Callable[[Path], None] | None = None   # what a checkpoint keeps beside the trainer's save, into its extra/

    @property
    def holds_out(self) -> bool:
        return bool(self.held_items)

    def start(self, trainer: Any, steps: list[list[list[TokenItem]]], out: Path, at: int, drawer: Path | None,
              pause: Callable[[], None]) -> None:
        """Before step ``at`` (0, or the step a checkpoint resumes after; ``drawer``, the checkpoint's kept adapter);
        ``pause`` is where SIGTERM may end the process between steps."""
        self.steps = steps

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        """Step ``k``'s gradient (``self.steps[k]``), accumulated; its record's fields."""
        raise NotImplementedError

    def stepped(self, trainer: Any, n: int) -> dict[str, Any]:
        """After the optimizer step of a phase that goes on (``n`` steps taken); its record's fields."""
        return {}

    def end(self) -> None:
        """After the last step, before the phase's last held-out measurement: what is still in flight finishes."""

    def close(self) -> None:
        """Once the phase's adapter is written: what the update served is taken down."""

    def held_out(self, trainer: Any) -> dict[str, Any]:
        """The held-out loss under the adapter as it is now, {"loss": ..., ...}."""
        raise NotImplementedError


class Imitation(Update):
    """DAgger's update (the module docstring): every call's reply is its target, raised by maximum likelihood over a
    step's chain pieces; a call longer than the window is left out (:func:`sft_items`). Its held-out loss is the held-out
    targets' negative log-likelihood per token."""

    objective = "sft"

    def __init__(self, a: argparse.Namespace, tok: Any, calls: list[Call], held: list[Call]) -> None:
        self.a = a
        self.units, self.stats = sft_items(tok, calls, a.max_seq_len)
        self.held_items, self.held_stats = sft_items(tok, held, a.max_seq_len)
        self.held_read = as_chains(self.held_items, a.chain_calls, a.max_seq_len, a.chain_slack)

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        res = trainer.sft_chains([chain_item(piece, self.a.chain_slack) for piece in self.steps[k]], rng=rng)
        # nll_per_token: the step's reply tokens' mean negative log-likelihood under the adapter as the step began
        return {"items": res["n_examples"], "tokens": res["tokens"],
                **{name: round(res[name], 6) if res[name] is not None else None
                   for name in ("nll_per_token", "loss_mean", "entropy_mean")},
                "read": {name: res["chains"][name] for name in ("chains", "tokens_read", "tokens_per_call")}}

    def held_out(self, trainer: Any) -> dict[str, Any]:
        res = [x for x in trainer.nll_chains(self.held_read)["items"] if x]
        return {"loss": -sum(x["logp"] for x in res) / max(1, sum(x["tokens"] for x in res))}


class SelfDistillation(Update):
    """OPSD's update (the module docstring): the units are DAgger's, the labelled calls as :class:`Imitation` takes
    them, so the same calls give OPSD the steps they give DAgger. At each, the current student's own reply is drawn
    fresh from the server (``a.server``), and the frozen base model reading the call's label as a demonstration is its
    teacher (:func:`distill_items`); a drawn reply whose student or teacher pass the window cannot hold is left out and
    counted (``drawn_too_long``). Its held-out loss is that loss over the replies the held-out games played (``held``:
    the plays of the held-out labelled calls, each carrying its label's demonstration), the same replies at every
    measurement. A checkpoint keeps the student that drew the batch in flight (:meth:`keep`), which draws it again on
    resume, so every batch is drawn by the student one step before the one it trains, as without the stop."""

    objective = "sdft"

    def __init__(self, a: argparse.Namespace, tok: Any, calls: list[Call], held: list[Call]) -> None:
        self.a, self.tok = a, tok
        self.units, self.stats = sft_items(tok, calls, a.max_seq_len)
        for call in calls:
            call.meta["demo"] = demo(tok, call)
        self.calls = {place(call.meta): call for call in calls}
        self.stats.update(drawn=0, draws_failed=0, drawn_too_long=0, template_mismatch=0, render_error=0,
                          retokenised=0)
        self.held_items = held
        items, self.held_stats = distill_items(tok, held, a.max_seq_len)
        self.held = distill_chains(items, a.chain_calls, a.max_seq_len, a.chain_slack)
        self.live: LiveAdapter | None = None
        self.drawer: Path | None = None   # the student drawing the batch in flight

    @property
    def holds_out(self) -> bool:
        return bool(self.held[0])   # a held-out call whose teacher pass the window cannot hold is no held-out item

    def start(self, trainer: Any, steps: list[list[list[TokenItem]]], out: Path, at: int, drawer: Path | None,
              pause: Callable[[], None]) -> None:
        super().start(trainer, steps, out, at, drawer, pause)
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
            checkpoints.link_tree(self.drawer, folder / "drawer")

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        t = time.time()
        fresh = [c for f in self.pending for c in f.result()]
        record: dict[str, Any] = {"wait_s": round(time.time() - t, 1)}
        # SIGTERM in the wait, the draws then likely failed with the server: the checkpoint is of the step before,
        # this batch to be drawn again by the same student
        self.pause()
        if k + 1 < len(self.steps):
            self.pending = self.ask(k + 1)   # drawn by the student as of now while this step trains: one step stale
        items, istats = distill_items(self.tok, fresh, self.a.max_seq_len)
        record.update(drawn=len(fresh), draws_failed=sum(len(piece) for piece in self.steps[k]) - len(fresh),
                      items=len(items), drawn_too_long=istats["too_long"],
                      **{name: istats[name] for name in ("template_mismatch", "render_error", "retokenised")},
                      reply_tokens_mean=round(sum(len(c.reply_ids) for c in fresh) / max(1, len(fresh)), 1),
                      finish=dict(collections.Counter(c.finish_reason for c in fresh)))
        for name in ("drawn", "draws_failed", "drawn_too_long", "template_mismatch", "render_error", "retokenised"):
            self.stats[name] += record[name]
        if not items:
            raise SystemExit(f"step {k + 1}: no reply was drawn ({record})")
        res = trainer.distill_chains(*distill_chains(items, self.a.chain_calls, self.a.max_seq_len, self.a.chain_slack),
                                     rng=rng, skip_first=self.a.skip_first)
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
        res = [x for x in trainer.distill_scores_chains(*self.held, skip_first=self.a.skip_first)["items"] if x]
        return {"loss": sum(x["loss"] for x in res) / max(1, len(res)), "replies": len(res)}


class PolicyGradient(Update):
    """PG's update (the module docstring): every unit's played reply raised through PPO's clipped ratio against the
    adapter that played it, by its advantage, over a step's chain pieces. Its held-out loss is the same loss over the
    held-out units' games, the held-out wins' negative log-likelihood per token logged beside it."""

    objective = "pg"
    key = "games"

    def __init__(self, a: argparse.Namespace, items: list[TokenItem], stats: dict[str, Any],
                 held: list[TokenItem]) -> None:
        self.a, self.units, self.stats = a, items, stats
        self.held_items = held
        self.held = as_chains(held, a.chain_calls, a.max_seq_len, a.chain_slack)
        self.wins = as_chains([it for it in held if it.label.get("solved")], a.chain_calls, a.max_seq_len,
                              a.chain_slack)
        self.held_stats = {"items": len(held)}

    def step(self, trainer: Any, k: int, rng: random.Random) -> dict[str, Any]:
        return pg_record(trainer.policy_gradient_chains([chain_item(piece, self.a.chain_slack)
                                                         for piece in self.steps[k]], rng=rng))

    def held_out(self, trainer: Any) -> dict[str, Any]:
        res = [x for x in trainer.policy_gradient_scores_chains(self.held)["items"] if x]
        point = {"loss": sum(x["loss"] for x in res) / max(1, len(res)), "spans": len(res)}
        if self.wins:
            nll = [x for x in trainer.nll_chains(self.wins)["items"] if x]
            point["wins_nll"] = round(-sum(x["logp"] for x in nll) / max(1, sum(x["tokens"] for x in nll)), 6)
        return point
