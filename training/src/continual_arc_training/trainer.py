"""A LoRA adapter on frozen base weights, trained a step at a time on chains of a game's calls
(:class:`continual_arc_training.chains.ChainItem`), each chain read in one pass (:mod:`continual_arc_training.branching`).

Each training pass accumulates its replies' gradients, every reply one item, and :meth:`Trainer.step` takes one
optimizer step over every item passed since the last: the summed gradient divided by their number, clipped to norm 1,
then AdamW. The passes, one per update of the paper's trained rows:

- :meth:`Trainer.sft_chains` (DAgger): each reply raised by maximum likelihood, its loss minus the sum of its token
  log-probabilities over a constant (:data:`PG_TOKEN_SCALE`), so every token weighs alike and a reply is not pushed
  harder for being short.
- :meth:`Trainer.distill_chains` (OPSD): each reply moved toward the frozen base model reading the reply's teacher
  prompt: the forward KL, KL(teacher || student), over the full vocabulary at each position of the reply but its first
  ``skip_first``, averaged over the reply, each reply weighing alike (SDFT's loss and reduction).
- :meth:`Trainer.policy_gradient_chains` (PG): each reply pushed by its advantage through PPO's clipped ratio against
  the adapter that played it, whose token log-probabilities the chain carries, dual-clipped under a negative advantage
  (:data:`PG_DUAL_CLIP`), the token sum over the same constant, with no KL term (Dr. GRPO's form).

Each has a forward-only score with the same loss, for a held-out measurement that touches nothing a step reads
(:meth:`Trainer.nll_chains`, :meth:`Trainer.distill_scores_chains`, :meth:`Trainer.policy_gradient_scores_chains`),
and :meth:`Trainer.behaviour_logps_chains` takes a reply's token log-probabilities under the adapter that played it
while that adapter still exists. Every full-vocabulary quantity is computed a chunk of positions at a time from the
final hidden states, the output layer and a float32 log-softmax, so full logits never exist for a whole reply.

Adapters are saved in PEFT's layout with the trainer's own copy of the weights and the optimizer's moments beside it,
so a phase that loads one carries on its moments. Inside a ``torch.distributed`` group the trainer is one rank of a
data-parallel step (:mod:`continual_arc_training.parallel`)."""
from __future__ import annotations

import contextlib
import math
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from continual_arc_training import branching
from continual_arc_training.chains import ChainItem
from continual_arc_training.parallel import (
    DataParallelError,
    DataParallelRank,
    data_parallel_guard,
    data_parallel_size,
    deal,
)

# The projections the adapter trains, matched as PEFT matches them (a module whose name ends in one): Qwen3.5's
# attention layers and every MLP. Its Gated DeltaNet layers name theirs in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and
# out_proj, so this list leaves the token mixing of three layers in four frozen.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# What a reply's summed token losses are divided by (Dr. GRPO's normaliser, about a reply's length: Qwen3.5-9B's
# harness replies run to a median of 566 tokens and a 90th percentile of 2,068); with the optimizer's normalisation it
# only decides when the gradient's norm clip at 1 binds.
PG_TOKEN_SCALE = 1024.0
# PPO's clip: the band 1 +- PG_CLIP around the behaviour adapter's probability of a token.
PG_CLIP = 0.2
# The dual clip (dual-clip PPO): under a negative advantage a token's objective is at least PG_DUAL_CLIP times the
# advantage, so a token whose ratio has run past it gives no gradient. PPO's clip bounds every other case but leaves this
# one linear in the ratio, and a token the behaviour adapter all but never samples (temperature 1, no top-p) can carry an
# enormous ratio once its probability moves, above all when the behaviour adapter is phases behind the one training:
# one step of 22,891 tokens once averaged a ratio of 21.3 and took almost all of its update from a few tokens.
PG_DUAL_CLIP = 3.0
TRAINER_WEIGHTS = "trainer_adapter.safetensors"
OPTIMIZER_STATE = "optimizer.pt"
CHUNK = 1024   # positions per chunk of a full-vocabulary computation


class Trainer(DataParallelRank):
    """A persistent LoRA adapter on the base model ``model_name`` (the module docstring)."""

    def __init__(self, model_name: str, *, rank: int, alpha: int, lr: float, device: str, max_seq_len: int,
                 grad_checkpointing: bool = True, lora_targets: tuple[str, ...] = LORA_TARGETS) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

        from continual_arc_training import kernels

        self.torch = torch
        self.device = device
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        AttentionInterface.register(kernels.ATTENTION, kernels.chain_attention)
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, trust_remote_code=True,
                                                    attn_implementation=kernels.ATTENTION).to(device)
        if grad_checkpointing:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        self.lora_config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                      target_modules=list(lora_targets))
        self.model = get_peft_model(base, self.lora_config)
        print(f"LoRA in bfloat16 on {kernels.bf16_lora(self.model)} projections", flush=True)
        self.model.print_trainable_parameters()
        branching.install(self.model)
        self._init_state = {k: v.detach().clone() for k, v in self.model.named_parameters() if v.requires_grad}
        self.optimizer = torch.optim.AdamW(self._trainable(), lr=lr)
        self.behaviour: dict[str, Any] | None = None   # the adapter that played the replies (configure_behaviour)
        self._pending_items = 0   # the items accumulated since the last step
        if data_parallel_size()[1] > 1:
            self.join_data_parallel()

    def reset(self) -> None:
        """Back to the untrained adapter, with a fresh optimizer; anything accumulated is dropped."""
        with self.torch.no_grad():
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    v.copy_(self._init_state[k])
        self.optimizer = self.torch.optim.AdamW(self._trainable(), lr=self.lr)
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_items = 0
        self.behaviour = None

    def _trainable(self) -> list[Any]:
        """The adapter's parameters, in the model's order (the order every rank's gradients are kept and summed in)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def configure_behaviour(self, adapter: Path | None) -> None:
        """The behaviour policy :meth:`behaviour_logps_chains` reads replies under: the adapter :meth:`save` wrote at
        ``adapter``, or with None the untrained adapter (the base model's own weights), the trainer's own weights left
        as they are. A reset or a load drops it."""
        if adapter is None:
            self.behaviour = {k: v.detach().float().clone() for k, v in self._init_state.items()}
            return
        from safetensors.torch import load_file

        tensors = load_file(str(Path(adapter) / TRAINER_WEIGHTS))
        self.behaviour = {k: tensors[k].to(self.device, dtype=self.torch.float32)
                          for k, v in self.model.named_parameters() if v.requires_grad}

    @contextlib.contextmanager
    def _adapter_weights(self, weights: dict[str, Any]) -> Any:
        """The adapter's parameters replaced by ``weights`` for the block, and the live ones put back after it. The block
        must build no graph: a parameter changed in place after a forward that saved it breaks that backward."""
        torch = self.torch
        params = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad]
        with torch.no_grad():
            live = [v.detach().clone() for _, v in params]
            for k, v in params:
                v.copy_(weights[k].to(v.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for (_, v), saved in zip(params, live, strict=True):
                    v.copy_(saved)

    # -- what every pass shares ----------------------------------------------------------------------------------------

    def _decoder_head(self) -> tuple[Any, Any]:
        """The base model's decoder and its output layer's weight (frozen: no gradient is computed for it)."""
        base_model = self.model.get_base_model()
        head = base_model.lm_head
        assert getattr(head, "bias", None) is None, "the losses assume an output layer without bias"
        return base_model.model, head.weight

    @staticmethod
    def _states(decoder: Any, packed: Any) -> Any:
        """The packed chain's final hidden states. Its branch plan must be active around this pass and, under gradient
        checkpointing, around the backward too, whose recomputation reads it (:func:`branching.active`)."""
        x, positions, mask, _, _ = packed
        return decoder(input_ids=x, position_ids=positions, attention_mask=mask).last_hidden_state[0]

    def _chunked(self, fn: Callable[..., Any], *tensors: Any) -> list[Any]:
        """``fn`` over ``tensors`` a chunk of positions at a time; under gradient each chunk is checkpointed, so its
        logits exist only while it is computed and again in its backward."""
        from torch.utils.checkpoint import checkpoint

        out = []
        for a in range(0, tensors[0].shape[0], CHUNK):
            part = [t[a:a + CHUNK] for t in tensors]
            if self.torch.is_grad_enabled() and part[0].requires_grad:
                out.append(checkpoint(fn, *part, use_reentrant=False))
            else:
                out.append(fn(*part))
        return out

    def _token_logp(self, h: Any, W: Any, ids: Any) -> Any:
        """log p(ids_t | prefix) at every position, float32, from the hidden states that produced the tokens."""
        torch = self.torch

        def piece(hc: Any, idc: Any) -> Any:
            return torch.log_softmax((hc @ W.T).float(), dim=-1).gather(-1, idc[:, None])[:, 0]

        return torch.cat(self._chunked(piece, h, ids))

    def _token_entropy(self, h: Any, W: Any) -> Any:
        """The entropy of the next-token distribution at each position, float32, without gradient."""
        torch = self.torch

        def piece(hc: Any) -> Any:
            lp = torch.log_softmax((hc @ W.T).float(), dim=-1)
            return -(lp.exp() * lp).sum(-1)

        with torch.no_grad():
            return torch.cat([piece(hc) for hc in h.split(CHUNK)])

    def _forward_kl(self, h_student: Any, h_teacher: Any, W: Any) -> Any:
        """KL(teacher || student) over the full vocabulary at each position, each side's logits the output layer on its
        hidden states (bfloat16) and its log-softmax in float32, averaged over the positions."""
        torch = self.torch

        def piece(hs: Any, ht: Any) -> Any:
            s = torch.log_softmax((hs @ W.T).float(), dim=-1)
            t = torch.log_softmax((ht @ W.T).float(), dim=-1)
            return (t.exp() * (t - s)).sum()

        return sum(self._chunked(piece, h_student, h_teacher)) / h_student.shape[0]

    def _chain_cost(self, item: ChainItem, passes: float) -> float:
        """A chain's estimated compute in token passes (the data-parallel deal's): its tokens and its attention pairs
        (the trunk causal, each reply over its prompt and itself), ``passes`` times (about four under gradient: forward,
        its recomputation and the backward)."""
        w = self._dp_pair_weight()
        N = len(item.trunk)
        pairs = N * N / 2 + sum(len(r) * point + len(r) * len(r) / 2 for point, r in item.branches)
        return passes * (item.length() + w * pairs)

    @staticmethod
    def _firsts(chains: list[ChainItem]) -> tuple[list[int], int]:
        """Each chain's first reply's place among all the replies given, and how many there are."""
        first, n = [], 0
        for c in chains:
            first.append(n)
            n += len(c.branches)
        return first, n

    # -- the training passes -----------------------------------------------------------------------------------------

    def _train_frame(self, chains: list[ChainItem], fits: list[bool], cost: Callable[[int], float], rng: Any,
                     lists: tuple[list, ...], shared: dict[str, int], pass_chain: Callable[[ChainItem, int], None]) -> int:
        """What every training pass shares: the chains that fit the window, in ``rng``'s order, each passed by
        ``pass_chain`` (its replies' gradients accumulated and their statistics appended to ``lists``), dealt between the
        ranks by ``cost`` (of chain i) and every rank's statistics gathered after; the replies passed join the step's
        count. Returns how many replies were skipped for a chain longer than the window."""
        self.model.train()
        first, _ = self._firsts(chains)
        packs = [i for i, ok in enumerate(fits) if ok]
        skipped = sum(len(c.branches) for c, ok in zip(chains, fits, strict=True) if not ok)
        rng.shuffle(packs)
        if not self._pending_items:
            self.optimizer.zero_grad(set_to_none=True)
        dp = (self._dp_begin([cost(i) for i in packs], sum(len(chains[i].branches) for i in packs), lists, (shared,))
              if self.dp_size > 1 else None)
        for pos in (dp.mine if dp else range(len(packs))):
            i = packs[pos]
            if dp:
                dp.mark(pos)
            pass_chain(chains[i], first[i])
            self._pending_items += len(chains[i].branches)
            shared["chains"] += 1
            shared["replies"] += len(chains[i].branches)
            shared["tokens_read"] += chains[i].length()
            shared["tokens_per_call"] += chains[i].per_call_tokens()
        if dp:
            self._dp_gather(dp)
        self.model.eval()
        return skipped

    @data_parallel_guard
    def sft_chains(self, chains: list[ChainItem], *, rng: Any) -> dict[str, Any]:
        """Supervised fine-tuning's pass (the module docstring): each reply's loss minus the sum of its token
        log-probabilities over :data:`PG_TOKEN_SCALE`. ``logp`` per reply in the chains' order (None for a reply in a
        chain longer than the window), ``nll_per_token`` (the replies' tokens' mean negative log-likelihood under the
        adapter as the pass began), ``entropy_mean`` and ``loss_mean``; ``chains`` counts the chains passed, their
        replies, the tokens read and the tokens their calls would have read one by one."""
        torch = self.torch
        stats: list[tuple[int, int, float, float, float]] = []   # per reply: place, tokens, logp, loss, entropy sum
        shared = {"chains": 0, "replies": 0, "tokens_read": 0, "tokens_per_call": 0}
        decoder, W = self._decoder_head()

        def pass_chain(c: ChainItem, first: int) -> None:
            packed = branching.pack(c, self.device)
            producing = packed[3]
            with branching.active(self.model, packed[4]):
                h = self._states(decoder, packed)
                losses = []
                for k in range(len(c.branches)):
                    reply = c.reply(k)
                    hk = h[producing[k]]
                    logp = self._token_logp(hk, W, torch.tensor(reply, device=self.device))
                    loss = -logp.sum() / PG_TOKEN_SCALE
                    losses.append(loss)
                    with torch.no_grad():
                        stats.append((first + k, len(reply), float(logp.double().sum()), float(loss),
                                      float(self._token_entropy(hk, W).double().sum())))
                sum(losses).backward()

        fits = [c.length() <= self.max_seq_len for c in chains]
        skipped = self._train_frame(chains, fits, lambda i: self._chain_cost(chains[i], 4.0), rng, (stats,), shared,
                                    pass_chain)
        _, n_calls = self._firsts(chains)
        logp: list[float | None] = [None] * n_calls
        tokens = logp_sum = loss_sum = entropy = 0.0
        for i, n, item_logp, loss, entropy_sum in stats:
            logp[i] = item_logp
            tokens += n
            logp_sum += item_logp
            loss_sum += loss
            entropy += entropy_sum
        return {"n_examples": n_calls, "skipped_long": skipped, "tokens": int(tokens), "chains": shared, "logp": logp,
                "loss_mean": loss_sum / len(stats) if stats else None,
                "nll_per_token": -logp_sum / tokens if tokens else None,
                "entropy_mean": entropy / tokens if tokens else None}

    @staticmethod
    def _check_teachers(chains: list[ChainItem], teacher_chains: list[ChainItem]) -> None:
        if len(teacher_chains) != len(chains):
            raise ValueError(f"{len(teacher_chains)} teacher chains for {len(chains)} chains")
        for c, t in zip(chains, teacher_chains, strict=True):
            if t.trunk != c.trunk or len(t.branches) != len(c.branches) \
                    or any(t.reply(k) != c.reply(k) for k in range(len(c.branches))):
                raise ValueError("a teacher chain shares its chain's trunk and every reply")

    def _teacher_states(self, decoder: Any, teacher: ChainItem) -> list[Any]:
        """The frozen base model's states that produced each reply of ``teacher``, read under its teacher prompts."""
        packed = branching.pack(teacher, self.device)
        with self.torch.no_grad(), self.model.disable_adapter(), branching.active(self.model, packed[4]):
            h = self._states(decoder, packed)
            return [h[idx] for idx in packed[3]]

    @data_parallel_guard
    def distill_chains(self, chains: list[ChainItem], teacher_chains: list[ChainItem], *, rng: Any,
                       skip_first: int = 0) -> dict[str, Any]:
        """Self-distillation's pass (the module docstring): per chain, its teacher chain (the same trunk and replies,
        each branch carrying the rest of its call's teacher prompt as its head) passes without gradient under the frozen
        base model, then the chain itself under the adapter with gradient. Each reply's loss is the forward KL at its
        positions after its first ``skip_first`` (all of them for a reply no longer than that), averaged over them. A
        chain either of whose passes is longer than the window is skipped (``skipped_long``, in replies).
        ``item_loss`` per reply in the chains' order; ``chains`` as :meth:`sft_chains`', the teacher's passes in
        ``teacher_tokens_read``."""
        self._check_teachers(chains, teacher_chains)
        teachers = {id(c): t for c, t in zip(chains, teacher_chains, strict=True)}
        stats: list[tuple[int, int, float]] = []   # per reply: place, tokens graded, loss
        shared = {"chains": 0, "replies": 0, "tokens_read": 0, "teacher_tokens_read": 0, "tokens_per_call": 0}
        decoder, W = self._decoder_head()

        def pass_chain(c: ChainItem, first: int) -> None:
            t = teachers[id(c)]
            hid_t = self._teacher_states(decoder, t)
            packed = branching.pack(c, self.device)
            producing = packed[3]
            with branching.active(self.model, packed[4]):
                h = self._states(decoder, packed)
                losses = []
                for k in range(len(c.branches)):
                    reply = c.reply(k)
                    s = skip_first if len(reply) > skip_first else 0
                    loss = self._forward_kl(h[producing[k]][s:], hid_t[k][s:], W)
                    losses.append(loss)
                    stats.append((first + k, len(reply) - s, float(loss.detach())))
                sum(losses).backward()
            shared["teacher_tokens_read"] += t.length()

        fits = [max(c.length(), t.length()) <= self.max_seq_len for c, t in zip(chains, teacher_chains, strict=True)]
        skipped = self._train_frame(
            chains, fits, lambda i: self._chain_cost(chains[i], 4.0) + self._chain_cost(teacher_chains[i], 1.0), rng,
            (stats,), shared, pass_chain)
        _, n_calls = self._firsts(chains)
        item_loss: list[float | None] = [None] * n_calls
        for i, _, loss in stats:
            item_loss[i] = loss
        return {"n_examples": n_calls, "skipped_long": skipped, "tokens": sum(n for _, n, _ in stats),
                "chains": shared, "item_loss": item_loss,
                "loss_mean": sum(loss for _, _, loss in stats) / len(stats) if stats else None}

    @staticmethod
    def _check_pg_chains(chains: list[ChainItem]) -> None:
        """Refuses (ValueError) chains the policy gradient cannot train or score: a branch with no advantage, and
        behaviour log-probabilities missing or not of their reply's length."""
        for i, c in enumerate(chains):
            for name, v in (("advantages", c.advantages), ("behaviour_logps", c.behaviour_logps)):
                if v is None or len(v) != len(c.branches):
                    raise ValueError(f"chain {i}: {'no' if v is None else len(v)} {name} for {len(c.branches)} branches")
            for k in range(len(c.branches)):
                if c.advantage(k) is None:
                    raise ValueError(f"chain {i} branch {k}: no advantage")
                if len(c.behaviour_logp(k)) != len(c.reply(k)):
                    raise ValueError(f"chain {i} branch {k}: {len(c.behaviour_logp(k))} behaviour log-probabilities for "
                                     f"{len(c.reply(k))} reply tokens")

    def _pg_reply(self, new: Any, old: Any, advantage: float, ids: Any, h: Any, W: Any) -> tuple[Any, tuple]:
        """One reply's term of the policy-gradient loss, to backpropagate, and its statistics. ``new`` and ``old`` are
        its tokens' log-probabilities under the adapter as it is and under the one that played it, ``ids`` its tokens
        and ``h`` the states that produced them. Each token adds ``-max(min(r A, clip(r, 1 - c, 1 + c) A), d A) /
        PG_TOKEN_SCALE`` where its advantage A is negative and ``-min(r A, clip(r, 1 - c, 1 + c) A) / PG_TOKEN_SCALE``
        elsewhere, r being exp(new - old), c :data:`PG_CLIP` and d :data:`PG_DUAL_CLIP`. Taking r as at most d gives
        exactly that: past it a positive advantage's minimum is already PPO's clipped side and a negative one's is
        ``d A``, neither moving with r, so the cap changes no value or gradient of the surrogate, and no ratio overflows.

        The statistics: the reply's log-probability, the sum of its tokens' ratios and of their KL estimates (from the
        true ratios, the cap left out), how many tokens PPO's clip and the dual clip silenced, its loss, the sum of the
        entropies at its positions, and its token of largest |log r| (the log-ratio, its position, its id and both
        log-probabilities)."""
        torch = self.torch
        log_ratio = new - old
        ratio = log_ratio.clamp(max=math.log(PG_DUAL_CLIP)).exp()
        unclipped, clipped_side = ratio * advantage, ratio.clamp(1.0 - PG_CLIP, 1.0 + PG_CLIP) * advantage
        loss = -torch.minimum(unclipped, clipped_side).sum() / PG_TOKEN_SCALE
        with torch.no_grad():
            clipped = float((clipped_side < unclipped).sum())
            dual = float((log_ratio > math.log(PG_DUAL_CLIP)).sum()) if advantage < 0 else 0.0
            true_log = log_ratio.double()
            true_ratio = true_log.exp()
            worst = int(log_ratio.abs().argmax())
            return loss, (float(new.double().sum()), float(true_ratio.sum()), clipped, dual,
                          float((true_ratio - 1.0 - true_log).sum()), float(loss),
                          float(self._token_entropy(h, W).double().sum()),
                          (float(log_ratio[worst]), worst, int(ids[worst]), float(new[worst]), float(old[worst])))

    @data_parallel_guard
    def policy_gradient_chains(self, chains: list[ChainItem], *, rng: Any) -> dict[str, Any]:
        """The policy gradient's pass (the module docstring): each branch a span with its own advantage, its ratios
        against the token log-probabilities the chain carries (``behaviour_logps``, taken under the adapter that played
        it). ``logp`` per branch in the chains' order; over the tokens, ``ratio_mean``, ``clip_fraction`` (the share of
        tokens whose clipped side was taken), ``dual_clip_fraction`` (the share the dual clip silenced), ``approx_kl``
        (the mean of ``r - 1 - log r``, an estimate of KL(behaviour || current) that is never negative) and
        ``entropy_mean``; ``log_ratio_worst``, the token of largest |log r| with its branch's label, which names the
        token behind a step whose ratios ran away after the adapter that played it is gone; ``loss_mean``."""
        torch = self.torch
        self._check_pg_chains(chains)
        stats: list[tuple] = []
        shared = {"chains": 0, "replies": 0, "tokens_read": 0, "tokens_per_call": 0}
        decoder, W = self._decoder_head()

        def pass_chain(c: ChainItem, first: int) -> None:
            packed = branching.pack(c, self.device)
            producing = packed[3]
            replies = [torch.tensor(c.reply(k), device=self.device) for k in range(len(c.branches))]
            with branching.active(self.model, packed[4]):
                h = self._states(decoder, packed)
                losses = []
                for k, ids in enumerate(replies):
                    old = torch.tensor(c.behaviour_logp(k), device=self.device, dtype=torch.float32)
                    new = self._token_logp(h[producing[k]], W, ids)
                    loss, reply_stats = self._pg_reply(new, old, c.advantage(k), ids, h[producing[k]], W)
                    losses.append(loss)
                    stats.append((first + k, len(ids), *reply_stats))
                sum(losses).backward()

        fits = [c.length() <= self.max_seq_len for c in chains]
        skipped = self._train_frame(chains, fits, lambda i: self._chain_cost(chains[i], 5.0), rng, (stats,), shared,
                                    pass_chain)
        labels = [c.labels[k] if c.labels else {} for c in chains for k in range(len(c.branches))]
        return {**self._pg_result(labels, stats), "skipped_long": skipped, "chains": shared}

    @staticmethod
    def _pg_result(labels: list[dict[str, Any]], stats: list[tuple]) -> dict[str, Any]:
        """:meth:`policy_gradient_chains`' result from its replies' statistics (:meth:`_pg_reply`, each after the
        reply's place among ``labels`` and its token count)."""
        logp: list[float | None] = [None] * len(labels)
        tokens = ratios = clipped_n = dual_n = kl = loss_sum = entropy = 0.0
        worst: dict[str, Any] | None = None
        for i, n, item_logp, ratio_sum, clipped, dual, kl_sum, loss, entropy_sum, (lr, position, token, lp, lp_b) \
                in stats:
            logp[i] = item_logp
            tokens += n
            ratios += ratio_sum
            clipped_n += clipped
            dual_n += dual
            kl += kl_sum
            loss_sum += loss
            entropy += entropy_sum
            if worst is None or abs(lr) > abs(worst["log_ratio"]):
                worst = {"label": labels[i], "position": position, "token": token, "log_ratio": lr, "logp": lp,
                         "logp_behaviour": lp_b}

        def per_token(total: float) -> float | None:
            return total / tokens if tokens else None

        return {"n_examples": len(labels), "tokens": int(tokens), "logp": logp,
                "loss_mean": loss_sum / len(stats) if stats else None, "ratio_mean": per_token(ratios),
                "clip_fraction": per_token(clipped_n), "dual_clip_fraction": per_token(dual_n),
                "approx_kl": per_token(kl), "entropy_mean": per_token(entropy), "log_ratio_worst": worst}

    # -- forward only ----------------------------------------------------------------------------------------------

    def _forward(self, chains: list[ChainItem], fits: list[bool], cost: Callable[[int], float],
                 score: Callable[[int, Any, Any], list[Any]], what: str) -> list[Any]:
        """What every forward-only pass shares: each chain that fits the window scored without gradient by
        ``score(i, decoder, W)``, a result per branch; per branch in the chains' order, None for a branch in a chain that
        does not fit. Nothing a step reads is touched. On several ranks each passes its share (dealt by ``cost`` of
        chain i) and every rank returns every branch's."""
        torch = self.torch
        first, n_calls = self._firsts(chains)
        packs = [i for i, ok in enumerate(fits) if ok]
        if self.dp_size > 1:
            owner, _ = deal([cost(i) for i in packs], self.dp_size)
            mine = [packs[k] for k, r in enumerate(owner) if r == self.dp_rank]
        else:
            mine = packs
        decoder, W = self._decoder_head()
        was_training = self.model.training
        self.model.eval()
        results: dict[int, Any] = {}
        error = None
        try:
            with torch.no_grad():
                for i in mine:
                    for k, res in enumerate(score(i, decoder, W)):
                        results[first[i] + k] = res
        except Exception:
            if self.dp_size == 1:
                raise
            error = traceback.format_exc()
        finally:
            if was_training:
                self.model.train()
        if self.dp_size > 1:
            gathered: list[Any] = [None] * self.dp_size
            torch.distributed.all_gather_object(gathered, {"results": results, "error": error})
            errors = [f"rank {r}: {g['error']}" for r, g in enumerate(gathered) if g["error"]]
            if errors:
                raise DataParallelError(f"{what} failed on " + "\n".join(errors))
            results = {i: res for g in gathered for i, res in g["results"].items()}
        return [results.get(i) for i in range(n_calls)]

    def nll_chains(self, chains: list[ChainItem]) -> dict[str, Any]:
        """Each reply's log-probability under the adapter as it is now: ``items``, per reply in the chains' order,
        {"logp", "tokens"}, None for a reply in a chain longer than the window."""
        torch = self.torch

        def score(i: int, decoder: Any, W: Any) -> list[dict[str, Any]]:
            c = chains[i]
            packed = branching.pack(c, self.device)
            with branching.active(self.model, packed[4]):
                h = self._states(decoder, packed)
            return [{"logp": float(self._token_logp(h[packed[3][k]], W, torch.tensor(c.reply(k), device=self.device))
                                   .double().sum()), "tokens": len(c.reply(k))} for k in range(len(c.branches))]

        fits = [c.length() <= self.max_seq_len for c in chains]
        return {"items": self._forward(chains, fits, lambda i: self._chain_cost(chains[i], 1.0), score, "nll_chains")}

    def distill_scores_chains(self, chains: list[ChainItem], teacher_chains: list[ChainItem], *,
                              skip_first: int = 0) -> dict[str, Any]:
        """Each reply's :meth:`distill_chains` loss under the adapter as it is now: ``items``, per reply in the chains'
        order, {"loss", "tokens"}, None for a reply in a chain either of whose passes is longer than the window."""
        self._check_teachers(chains, teacher_chains)

        def score(i: int, decoder: Any, W: Any) -> list[dict[str, Any]]:
            c = chains[i]
            hid_t = self._teacher_states(decoder, teacher_chains[i])
            packed = branching.pack(c, self.device)
            with branching.active(self.model, packed[4]):
                h = self._states(decoder, packed)
            out = []
            for k in range(len(c.branches)):
                reply = c.reply(k)
                s = skip_first if len(reply) > skip_first else 0
                out.append({"loss": float(self._forward_kl(h[packed[3][k]][s:], hid_t[k][s:], W)),
                            "tokens": len(reply) - s})
            return out

        fits = [max(c.length(), t.length()) <= self.max_seq_len for c, t in zip(chains, teacher_chains, strict=True)]
        return {"items": self._forward(
            chains, fits, lambda i: self._chain_cost(chains[i], 1.0) + self._chain_cost(teacher_chains[i], 1.0), score,
            "distill_scores_chains")}

    def policy_gradient_scores_chains(self, chains: list[ChainItem]) -> dict[str, Any]:
        """Each branch's :meth:`policy_gradient_chains` loss under the adapter as it is now: ``items``, per branch in the
        chains' order, {"loss", "tokens"}, None for a branch in a chain longer than the window."""
        torch = self.torch
        self._check_pg_chains(chains)

        def score(i: int, decoder: Any, W: Any) -> list[dict[str, Any]]:
            c = chains[i]
            packed = branching.pack(c, self.device)
            with branching.active(self.model, packed[4]):
                h = self._states(decoder, packed)
            out = []
            for k in range(len(c.branches)):
                ids = torch.tensor(c.reply(k), device=self.device)
                old = torch.tensor(c.behaviour_logp(k), device=self.device, dtype=torch.float32)
                new = self._token_logp(h[packed[3][k]], W, ids)
                loss, _ = self._pg_reply(new, old, c.advantage(k), ids, h[packed[3][k]], W)
                out.append({"loss": float(loss), "tokens": len(ids)})
            return out

        fits = [c.length() <= self.max_seq_len for c in chains]
        return {"items": self._forward(chains, fits, lambda i: self._chain_cost(chains[i], 2.0), score,
                                       "policy_gradient_scores_chains")}

    def behaviour_logps_chains(self, chains: list[ChainItem]) -> dict[str, Any]:
        """Each branch's reply token log-probabilities under the behaviour adapter (:meth:`configure_behaviour`), a list
        of floats: ``items``, per branch in the chains' order, None for a branch in a chain longer than the window."""
        torch = self.torch
        if self.behaviour is None:
            raise ValueError("behaviour_logps_chains needs configure_behaviour first")

        def score(i: int, decoder: Any, W: Any) -> list[list[float]]:
            c = chains[i]
            packed = branching.pack(c, self.device)
            with self._adapter_weights(self.behaviour), branching.active(self.model, packed[4]):
                h = self._states(decoder, packed)
                return [self._token_logp(h[packed[3][k]], W, torch.tensor(c.reply(k), device=self.device)).tolist()
                        for k in range(len(c.branches))]

        fits = [c.length() <= self.max_seq_len for c in chains]
        return {"items": self._forward(chains, fits, lambda i: self._chain_cost(chains[i], 1.0), score,
                                       "behaviour_logps_chains")}

    # -- the step, and the adapter on disk ------------------------------------------------------------------------------

    def step(self) -> dict[str, Any]:
        """One optimizer step over every item accumulated since the last: the summed gradient (every rank's) divided by
        their number, clipped to norm 1 (``grad_norm``: the norm before clipping). ``steps`` is 0, and nothing changes,
        when nothing was accumulated; ``identical`` says every rank holds the same adapter after it."""
        n = self._pending_items
        out: dict[str, Any] = {"steps": 0, "items": n}
        if not n:
            return out
        params = self._trainable()
        if self.dp_size > 1:
            self._dp_reduce()
        for p in params:
            if p.grad is not None:
                p.grad.div_(n)
        out["grad_norm"] = float(self.torch.nn.utils.clip_grad_norm_(params, 1.0))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_items = 0
        if self.dp_size > 1:
            out["identical"] = self._dp_verify()
        out["steps"] = 1
        return out

    def save(self, path: Path) -> None:
        """Save the adapter as PEFT writes it, and the trainer's own copy of its weights (parameter names as PEFT holds
        them) and the optimizer's moments, what a phase that starts from it carries on with. Rank 0 writes."""
        from safetensors.torch import save_file

        if self.dp_rank != 0:
            return
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        save_file({k: v.detach().to("cpu").contiguous() for k, v in self.model.named_parameters() if v.requires_grad},
                  str(path / TRAINER_WEIGHTS), metadata={"format": "pt"})
        self.torch.save(self.optimizer.state_dict(), path / OPTIMIZER_STATE)

    def load(self, path: Path) -> None:
        """Restore an adapter :meth:`save` wrote, its weights and optimizer moments. The moments and step counts carry
        on at this trainer's own learning rate, not the one the state was saved at. An adapter saved on fewer target
        modules than this trainer's loads into the modules it has, the others keeping their fresh start (B at zero, so
        the model computes as the saved adapter does) and no moments (:func:`_widen_optimizer_state`); a saved weight
        this trainer has no module for is an error."""
        from safetensors.torch import load_file

        tensors = load_file(str(path / TRAINER_WEIGHTS))
        names = [k for k, v in self.model.named_parameters() if v.requires_grad]
        unknown = tensors.keys() - set(names)
        if unknown or not tensors:
            raise ValueError(f"{path}: {len(unknown)} of {len(tensors)} saved adapter weights have no place in this "
                             f"trainer's adapter (lora targets {sorted(self.lora_config.target_modules)}), such as "
                             f"{sorted(unknown)[:1]}")
        with self.torch.no_grad():
            for k, v in self.model.named_parameters():
                if v.requires_grad and k in tensors:
                    v.copy_(tensors[k].to(v.device, dtype=v.dtype))
        state = path / OPTIMIZER_STATE
        if state.is_file():
            saved = self.torch.load(state, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(_widen_optimizer_state(saved, names, tensors.keys()))
            for group in self.optimizer.param_groups:   # load_state_dict also brings back the saved rate
                group["lr"] = self.lr
        self.behaviour = None


def _widen_optimizer_state(saved: dict[str, Any], names: list[str], saved_names: Any) -> dict[str, Any]:
    """An optimizer state saved over the adapter weights ``saved_names`` as one over ``names``, the trainer's own, which
    hold them all: each saved weight keeps its moments and step count under its new index, and the rest have none, so
    AdamW starts them as it starts any weight. A saved state numbers its weights in the trainer's order, which adding
    target modules keeps among the weights it already had."""
    kept = [n for n in names if n in saved_names]
    if len(kept) == len(names):
        return saved
    (group,) = saved["param_groups"]
    if len(group["params"]) != len(kept):
        raise ValueError(f"the saved optimizer holds {len(group['params'])} weights for {len(kept)} saved adapter weights")
    index = {n: i for i, n in enumerate(names)}
    name_of = dict(zip(group["params"], kept, strict=True))
    return {"state": {index[name_of[i]]: s for i, s in saved["state"].items()},
            "param_groups": [{**group, "params": list(range(len(names)))}]}
