"""The trainer (continual_arc_training.trainer) on a tiny random Qwen3.5 on a CPU: every pass over a chain gives each
reply what it has computed directly, alone after its own prompt (its log-probabilities, its loss and the gradient), and
the step, the save and the load keep the adapter and its optimizer as they should. Needs torch, transformers and peft
(skipped without them) and the tiny Llama tokenizer from the Hugging Face cache."""
from __future__ import annotations

import dataclasses
import math
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from continual_arc_training import branching, kernels  # noqa: E402
from continual_arc_training.chains import ChainItem  # noqa: E402
from continual_arc_training.trainer import PG_CLIP, PG_DUAL_CLIP, PG_TOKEN_SCALE, Trainer  # noqa: E402

TOKENIZER = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"   # the items are ids: any tokenizer will do
VOCAB = 1024


def _hybrid_model(path, lora_targets=None, grad_checkpointing=False):
    """A tiny random Qwen3.5 (three linear-attention layers, then a full-attention one, the 9B's pattern), saved in
    ``path`` and loaded through the Trainer as the 9B is; float32 and its adapter moved off its start, so the student
    differs from the base and every weight has a gradient."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if not (path / "config.json").exists():
        config = Qwen3_5TextConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16, intermediate_size=128, linear_num_key_heads=2, linear_num_value_heads=4,
                                   linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=VOCAB,
                                   layer_types=["linear_attention"] * 3 + ["full_attention"])
        torch.manual_seed(0)
        Qwen3_5ForCausalLM(config).save_pretrained(path)
        try:
            AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(path)
        except OSError as error:
            pytest.skip(f"tiny tokenizer unavailable: {error}")
    kwargs = {"lora_targets": lora_targets} if lora_targets else {}
    tr = Trainer(str(path), rank=4, alpha=8, lr=1e-2, device="cpu", max_seq_len=512,
                 grad_checkpointing=grad_checkpointing, **kwargs)
    tr.model.float()
    return tr


def _perturbed(tr, seed=1, scale=0.05):
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        for v in tr._trainable():
            v.add_(scale * torch.randn(v.shape, generator=g).to(v.dtype))
    return tr


@pytest.fixture
def model_dir(tmp_path):
    return tmp_path / "model"


def _trainer(model_dir, **kw):
    return _perturbed(_hybrid_model(model_dir, **kw))


def _grads(tr):
    return [None if p.grad is None else p.grad.detach().clone() for p in tr._trainable()]


def _zero(tr):
    tr.optimizer.zero_grad(set_to_none=True)
    tr._pending_items = 0


def _close(got, want, tol=1e-4):
    scale = max(float(w.abs().max()) for w in want if w is not None)
    assert max(float((a - b).abs().max()) for a, b in zip(got, want, strict=True) if b is not None) < tol * scale


def _a_chain(seed=0, heads=False):
    """A chain of four replies off one 300-token trunk, at points that cut it unevenly (one inside the short
    convolution's reach of the start, two adjacent, one at the trunk's end), with ``heads`` some of them leaving the
    trunk in their prompt's last ids; and each call's prompt and reply as it would be alone."""
    g = random.Random(seed)
    trunk = [g.randrange(1, VOCAB) for _ in range(300)]
    off = lambda i: (trunk[i] % (VOCAB - 1)) + 1   # noqa: E731  an id that is not the trunk's there
    plan = [(2, [off(2), off(3)], 7), (129, [off(129)], 20), (131, [off(131)], 5), (300, [], 12)] if heads else \
        [(2, [], 7), (130, [], 20), (131, [], 5), (300, [], 12)]
    branches, head_lens, calls = [], [], []
    for p, head, n in plan:
        reply = [g.randrange(1, VOCAB) for _ in range(n)]
        branches.append((p, head + reply))
        head_lens.append(len(head))
        calls.append((trunk[:p] + head, reply))
    return ChainItem(trunk, branches, heads=head_lens if heads else []), calls


def _logprobs(tr, prompt, reply, adapter=True):
    """The reply's full-vocabulary log-probabilities at its positions, the call passed alone."""
    ids = torch.tensor([prompt + reply])
    ctx = tr.model.disable_adapter() if not adapter else torch.enable_grad()
    with ctx:
        logits = tr.model(input_ids=ids).logits[0, len(prompt) - 1:-1]
    return torch.log_softmax(logits.float(), dim=-1)


def _reply_logp(lp, reply):
    return lp.gather(-1, torch.tensor(reply)[:, None])[:, 0]


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("heads", [False, True])
def test_a_chains_sft_is_its_calls_one_by_one(model_dir, checkpointing, heads):
    """One pass over a chain gives every reply the log-probability it has alone after its prompt and the accumulated
    gradient of its calls' losses (each minus the reply's summed log-probability over PG_TOKEN_SCALE), with gradient
    checkpointing too (the backward's recomputation reads the branch plan); a head's ids are read, not graded; each
    reply is one item in the step."""
    tr = _trainer(model_dir, grad_checkpointing=checkpointing)
    chain, calls = _a_chain(heads=heads)
    want_logp = []
    loss = 0
    for prompt, reply in calls:
        logp = _reply_logp(_logprobs(tr, prompt, reply), reply)
        want_logp.append(float(logp.detach().sum()))
        loss = loss - logp.sum() / PG_TOKEN_SCALE
    loss.backward()
    want = _grads(tr)
    _zero(tr)
    out = tr.sft_chains([chain], rng=random.Random(0))
    assert tr._pending_items == 4 and out["n_examples"] == 4 and out["skipped_long"] == 0 and out["tokens"] == 44
    assert out["logp"] == pytest.approx(want_logp, rel=1e-5)
    assert out["loss_mean"] == pytest.approx(-sum(want_logp) / PG_TOKEN_SCALE / 4, rel=1e-5)
    assert out["nll_per_token"] == pytest.approx(-sum(want_logp) / 44, rel=1e-5)
    read = 300 + sum(len(b) for _, b in chain.branches)
    assert out["chains"] == {"chains": 1, "replies": 4, "tokens_read": read,
                             "tokens_per_call": sum(len(p) + len(r) for p, r in calls)}
    _close(_grads(tr), want)
    assert [r["logp"] for r in tr.nll_chains([chain])["items"]] == pytest.approx(want_logp, rel=1e-5)
    taken = tr.step()
    assert taken["steps"] == 1 and taken["items"] == 4 and tr._pending_items == 0 and taken["grad_norm"] > 0


def _demo_chain(seed=5):
    """A chain of four replies and its teacher chain: each call's teacher prompt is its prompt with a note of nine ids
    inserted before its last few (a demonstration at the end of the last message, the turn's end after it), which the
    teacher chain carries as each branch's head; and each call's student and teacher prompts and reply."""
    g = random.Random(seed)
    trunk = [g.randrange(1, VOCAB) for _ in range(300)]
    branches, t_branches, t_heads, calls = [], [], [], []
    for p, n in [(4, 7), (130, 20), (131, 5), (300, 12)]:
        reply = [g.randrange(1, VOCAB) for _ in range(n)]
        note = [g.randrange(1, VOCAB) for _ in range(9)]
        cut = p - min(3, p - 1)
        t_prompt = trunk[:cut] + note + trunk[cut:p]
        branches.append((p, reply))
        t_branches.append((cut, t_prompt[cut:] + reply))
        t_heads.append(len(t_prompt) - cut)
        calls.append((trunk[:p], t_prompt, reply))
    return ChainItem(trunk, branches), ChainItem(trunk, t_branches, heads=t_heads), calls


def _forward_kl(student, teacher):
    """KL(teacher || student) over the full vocabulary at each position, averaged over the positions."""
    return (teacher.exp() * (teacher - student)).sum(-1).mean()


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("skip", [0, 2])
def test_a_chains_distillation_is_its_calls_one_by_one(model_dir, checkpointing, skip):
    """The teacher is the frozen base model reading each call's teacher prompt: every reply's loss (the forward KL at
    its positions after its first ``skip_first``, averaged over them) and the accumulated gradient are the calls' own,
    each computed alone; the forward-only score gives the same losses and touches nothing a step reads."""
    tr = _trainer(model_dir, grad_checkpointing=checkpointing)
    chain, teacher, calls = _demo_chain()
    want_loss, loss = [], 0
    for prompt, t_prompt, reply in calls:
        with torch.no_grad():
            t = _logprobs(tr, t_prompt, reply, adapter=False)[skip:]
        kl = _forward_kl(_logprobs(tr, prompt, reply)[skip:], t)
        want_loss.append(float(kl.detach()))
        loss = loss + kl
    loss.backward()
    want = _grads(tr)
    _zero(tr)
    scored = tr.distill_scores_chains([chain], [teacher], skip_first=skip)["items"]
    assert tr._pending_items == 0 and all(p.grad is None for p in tr._trainable())
    out = tr.distill_chains([chain], [teacher], rng=random.Random(0), skip_first=skip)
    assert tr._pending_items == 4 and out["n_examples"] == 4 and out["tokens"] == 44 - 4 * skip
    assert out["item_loss"] == pytest.approx(want_loss, rel=1e-4)
    assert [r["loss"] for r in scored] == pytest.approx(want_loss, rel=1e-4)
    assert out["chains"]["teacher_tokens_read"] == teacher.length()
    _close(_grads(tr), want)


def test_a_teacher_chain_must_share_its_chains_trunk_and_replies_and_a_long_one_is_left_out(model_dir):
    tr = _trainer(model_dir)
    chain, teacher, _ = _demo_chain()
    other = ChainItem(teacher.trunk, [(p, ids[:-1] + [(ids[-1] % (VOCAB - 1)) + 1]) for p, ids in teacher.branches],
                      heads=teacher.heads)
    with pytest.raises(ValueError, match="every reply"):
        tr.distill_chains([chain], [other], rng=random.Random(0))
    with pytest.raises(ValueError, match="teacher chains"):
        tr.distill_scores_chains([chain], [])
    long = ChainItem(list(range(1, 501)), [(500, list(range(1, 30)))])   # 529 tokens: past the 512 window
    out = tr.distill_scores_chains([chain, long], [teacher, long])
    assert out["items"][4] is None and all(r is not None for r in out["items"][:4])
    trained = tr.distill_chains([long], [long], rng=random.Random(0))
    assert trained["skipped_long"] == 1 and trained["item_loss"] == [None] and tr._pending_items == 0


def _pg_reference(tr, calls, advantages, old):
    """The policy gradient computed directly, each call alone: each token's ``-max(min(r A, clip(r) A), d A)`` under a
    negative advantage and ``-min(r A, clip(r) A)`` otherwise, over PG_TOKEN_SCALE, r = exp(new - old)."""
    total, logps, losses, ratios, clipped = 0, [], [], [], 0
    for (prompt, reply), adv, behaviour in zip(calls, advantages, old, strict=True):
        new = _reply_logp(_logprobs(tr, prompt, reply), reply)
        r = (new - torch.tensor(behaviour)).exp()
        obj = torch.minimum(r * adv, r.clamp(1 - PG_CLIP, 1 + PG_CLIP) * adv)
        clipped += int((r.clamp(1 - PG_CLIP, 1 + PG_CLIP) * adv < r * adv).sum())
        if adv < 0:
            obj = torch.maximum(obj, torch.full_like(obj, PG_DUAL_CLIP * adv))
        loss = -obj.sum() / PG_TOKEN_SCALE
        total = total + loss
        logps.append(float(new.detach().sum()))
        losses.append(float(loss.detach()))
        ratios += r.detach().tolist()
    total.backward()
    return logps, losses, {"ratio_mean": sum(ratios) / len(ratios), "clip_fraction": clipped / len(ratios)}


def _played(tr, calls):
    """Each reply's token log-probabilities under the untrained adapter (the base model's own weights), alone."""
    with torch.no_grad():
        return [_reply_logp(_logprobs(tr, p, r, adapter=False), r).tolist() for p, r in calls]


@pytest.mark.parametrize("heads", [False, True])
def test_a_chains_policy_gradient_is_its_calls_one_by_one_against_the_adapter_that_played(model_dir, heads):
    """The behaviour log-probabilities read through the chain under the untrained adapter are each reply's alone; the
    pass's log-probabilities, per-reply losses and accumulated gradient are the clipped surrogate's computed directly;
    the forward-only score gives the same losses and touches nothing a step reads."""
    tr = _trainer(model_dir)
    chain, calls = _a_chain(heads=heads)
    advantages = [0.6, -0.3, 0.4, -0.5]
    tr.configure_behaviour(None)
    played = tr.behaviour_logps_chains([chain])["items"]
    assert [len(x) for x in played] == [len(r) for _, r in calls]
    for got, want in zip(played, _played(tr, calls), strict=True):
        assert got == pytest.approx(want, rel=1e-5, abs=1e-6)
    chain = dataclasses.replace(chain, advantages=advantages, behaviour_logps=played,
                                labels=[{"i": i} for i in range(4)])
    _perturbed(tr, seed=7, scale=0.3)   # away from the adapter that played: a nontrivial ratio and some clipping
    want_logp, want_loss, measured = _pg_reference(tr, calls, advantages, played)
    want = _grads(tr)
    _zero(tr)
    scored = tr.policy_gradient_scores_chains([chain])["items"]
    assert tr._pending_items == 0 and all(p.grad is None for p in tr._trainable())
    out = tr.policy_gradient_chains([chain], rng=random.Random(0))
    assert tr._pending_items == 4 and out["n_examples"] == 4 and out["tokens"] == 44
    assert out["logp"] == pytest.approx(want_logp, rel=1e-5)
    assert [r["loss"] for r in scored] == pytest.approx(want_loss, rel=1e-4)
    assert out["loss_mean"] == pytest.approx(sum(want_loss) / 4, rel=1e-4)
    assert out["clip_fraction"] == pytest.approx(measured["clip_fraction"]) and out["clip_fraction"] > 0
    assert out["ratio_mean"] == pytest.approx(measured["ratio_mean"], rel=1e-4)
    assert out["log_ratio_worst"]["label"] in chain.labels
    _close(_grads(tr), want)


def test_far_from_the_adapter_that_played_the_dual_clip_silences_a_failures_runaway_tokens(model_dir):
    tr = _trainer(model_dir)
    with torch.no_grad():
        tr.model.get_base_model().lm_head.weight.mul_(30)   # peaked, so that the adapter's move shows in the ratios
    chain, calls = _a_chain()
    advantages = [-0.6, -0.3, 0.4, -0.5]
    tr.configure_behaviour(None)
    played = tr.behaviour_logps_chains([chain])["items"]
    _perturbed(tr, seed=5, scale=0.5)
    chain = dataclasses.replace(chain, advantages=advantages, behaviour_logps=played)
    _, want_loss, _ = _pg_reference(tr, calls, advantages, played)
    want = _grads(tr)
    _zero(tr)
    out = tr.policy_gradient_chains([chain], rng=random.Random(0))
    assert out["dual_clip_fraction"] > 0
    assert out["loss_mean"] == pytest.approx(sum(want_loss) / 4, rel=1e-4)
    _close(_grads(tr), want, tol=1e-3)


def test_a_token_ratio_past_float32s_range_still_gives_a_finite_step(model_dir):
    """A log-ratio past about 89 nats overflows exp in float32; the dual clip caps the ratio before exp, so the step
    stays finite, and the ratio's statistics come from the true log-ratios in float64."""
    tr = _trainer(model_dir)
    with torch.no_grad():
        tr.model.get_base_model().lm_head.weight.mul_(300)
    chain, _ = _a_chain()
    tr.configure_behaviour(None)
    played = tr.behaviour_logps_chains([chain])["items"]
    _perturbed(tr, seed=5, scale=0.5)
    chain = dataclasses.replace(chain, advantages=[-0.6, -0.3, 0.4, -0.5], behaviour_logps=played)
    out = tr.policy_gradient_chains([chain], rng=random.Random(0))
    assert abs(out["log_ratio_worst"]["log_ratio"]) > 89 and math.isfinite(out["loss_mean"])
    assert all(bool(torch.isfinite(g).all()) for g in _grads(tr) if g is not None)
    assert math.isfinite(tr.step()["grad_norm"])


def test_the_policy_gradient_refuses_a_chain_without_its_advantages_or_behaviour(model_dir):
    tr = _trainer(model_dir)
    chain, calls = _a_chain()
    played = [[0.0] * len(r) for _, r in calls]
    with pytest.raises(ValueError, match="advantages"):
        tr.policy_gradient_chains([dataclasses.replace(chain, behaviour_logps=played)], rng=random.Random(0))
    with pytest.raises(ValueError, match="advantages"):
        tr.policy_gradient_chains([dataclasses.replace(chain, advantages=[0.1] * 5, behaviour_logps=played)],
                                  rng=random.Random(0))
    with pytest.raises(ValueError, match="behaviour log-probabilities"):
        tr.policy_gradient_scores_chains([dataclasses.replace(chain, advantages=[0.1] * 4,
                                                              behaviour_logps=[x[:-1] for x in played])])
    with pytest.raises(ValueError, match="configure_behaviour"):
        tr.behaviour_logps_chains([chain])


def test_a_step_divides_the_summed_gradient_by_the_items_and_nothing_accumulated_is_no_step(model_dir):
    tr = _trainer(model_dir)
    assert tr.step() == {"steps": 0, "items": 0}
    chain, _ = _a_chain()
    tr.sft_chains([chain], rng=random.Random(0))
    summed = _grads(tr)
    before = [p.detach().clone() for p in tr._trainable()]
    norm = float(torch.cat([g.flatten() / 4 for g in summed if g is not None]).norm())
    taken = tr.step()
    assert taken["grad_norm"] == pytest.approx(norm, rel=1e-5)
    assert any(not torch.equal(a, p) for a, p in zip(before, tr._trainable(), strict=True))


def test_a_loaded_adapter_carries_its_moments_on_at_the_loading_trainers_rate(model_dir, tmp_path):
    """load restores a saved adapter's weights and its optimizer's moments and step counts, and keeps the loading
    trainer's own learning rate: at the saver's rate the next step is the saver's own next step."""
    chain, _ = _a_chain()
    saver = _trainer(model_dir)
    saver.sft_chains([chain], rng=random.Random(0))
    saver.step()
    saver.save(tmp_path / "p0001")
    assert {f.name for f in (tmp_path / "p0001").iterdir()} >= {"adapter_model.safetensors", "adapter_config.json",
                                                                "trainer_adapter.safetensors", "optimizer.pt"}
    start = [p.detach().clone() for p in saver._trainable()]
    saver.sft_chains([chain], rng=random.Random(1))
    saver.step()
    carried = [p.detach().clone() for p in saver._trainable()]
    for lr in (1e-2, 5e-2):
        loaded = _hybrid_model(model_dir)
        loaded.lr = lr
        loaded.reset()
        loaded.load(tmp_path / "p0001")
        assert all(group["lr"] == lr for group in loaded.optimizer.param_groups)
        assert all(torch.equal(a, b) for a, b in zip(loaded._trainable(), start, strict=True))
        loaded.sft_chains([chain], rng=random.Random(1))
        loaded.step()
        moved = loaded._trainable()
        if lr == 1e-2:
            assert all(torch.allclose(m, c, rtol=1e-5, atol=1e-7) for m, c in zip(moved, carried, strict=True))
        else:   # the same moments and gradient: AdamW moves every weight five times as far
            assert all(torch.allclose(m - s, 5 * (c - s), rtol=1e-3, atol=1e-6)
                       for m, s, c in zip(moved, start, carried, strict=True))
    loaded.reset()
    assert all(not p.any() for k, p in loaded.model.named_parameters() if p.requires_grad and "lora_B" in k)


def test_an_adapter_saved_on_fewer_target_modules_loads_into_a_wider_one(model_dir, tmp_path):
    """The saved weights and their moments load where they were and the added modules keep their fresh start, so the
    student computes as the saved one did; a saved weight the loading trainer has no module for is an error."""
    chain, calls = _a_chain()
    narrow = ("q_proj", "v_proj")
    saver = _trainer(model_dir, lora_targets=narrow)
    saver.sft_chains([chain], rng=random.Random(0))
    saver.step()
    saver.save(tmp_path / "narrow")
    wide = _hybrid_model(model_dir, lora_targets=(*narrow, "up_proj"))
    wide.load(tmp_path / "narrow")
    prompt, reply = calls[1]
    with torch.no_grad():
        assert torch.allclose(_logprobs(wide, prompt, reply), _logprobs(saver, prompt, reply), atol=1e-6)
    names = [k for k, v in wide.model.named_parameters() if v.requires_grad]
    assert {names[i] for i in wide.optimizer.state_dict()["state"]} == \
        {k for k, v in saver.model.named_parameters() if v.requires_grad}
    wide.sft_chains([chain], rng=random.Random(1))
    wide.step()
    assert any(p.any() for k, p in wide.model.named_parameters() if p.requires_grad and ".up_proj.lora_B" in k)
    wide.save(tmp_path / "wide")
    with pytest.raises(ValueError, match="no place"):
        _hybrid_model(model_dir, lora_targets=narrow).load(tmp_path / "wide")


def test_the_behaviour_adapter_is_a_saved_one_or_the_untrained_one_leaving_the_weights_alone(model_dir, tmp_path):
    tr = _trainer(model_dir)
    chain, calls = _a_chain()
    tr.save(tmp_path / "played")
    live = [p.detach().clone() for p in tr._trainable()]
    _perturbed(tr, seed=9)
    moved = [p.detach().clone() for p in tr._trainable()]
    tr.configure_behaviour(tmp_path / "played")
    played = tr.behaviour_logps_chains([chain])["items"]
    assert all(torch.equal(a, b) for a, b in zip(tr._trainable(), moved, strict=True))
    with torch.no_grad():
        for p, v in zip(tr._trainable(), live, strict=True):
            p.copy_(v)
        want = [_reply_logp(_logprobs(tr, pr, r), r).tolist() for pr, r in calls]
    for got, w in zip(played, want, strict=True):
        assert got == pytest.approx(w, rel=1e-5, abs=1e-6)
    tr.reset()
    assert tr.behaviour is None


def test_bf16_lora_computes_peft_s_own_update_in_the_inputs_dtype():
    """The patched LoRA forward is PEFT's in float32, and in bfloat16 within bfloat16's rounding; with dropout on, PEFT's
    own forward runs."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(16, 16))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["0"]))
    with torch.no_grad():
        for k, p in model.named_parameters():
            if "lora_B" in k:
                p.normal_()
    x = torch.randn(3, 5, 16)
    want = model(x)
    assert kernels.bf16_lora(model) == 1 and kernels.bf16_lora(model) == 0   # patched once
    assert torch.allclose(model(x), want, atol=1e-5)
    xb = x.to(torch.bfloat16)
    model.to(torch.bfloat16)
    for k, p in model.named_parameters():
        if "lora" in k:
            p.data = p.data.float()   # PEFT keeps the adapter in float32
    got = model(xb)
    assert got.dtype == torch.bfloat16 and torch.allclose(got.float(), want, atol=0.1, rtol=0.05)


def test_flex_attention_falls_back_to_tritons_flex_kernel_never_its_own_choice(monkeypatch):
    """A chain goes through FlashAttention's backend first; one it refuses goes, from then on, to Triton's
    FlexAttention kernel named outright, never to FlexAttention's own choice (its decoding kernel for a query under 128
    tokens has no backward); one Triton refuses raises."""
    from torch.nn.attention import flex_attention as flex

    seen = []

    def fake(q, k, v, block_mask, scale, enable_gqa, kernel_options):
        seen.append(dict(kernel_options))
        if kernel_options.get("BACKEND") != "TRITON":
            raise RuntimeError("refused")
        return q

    mask = flex.create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, 128, 128, device="cpu")
    q = torch.zeros(1, 2, 128, 4)
    monkeypatch.setattr(kernels, "_flex", fake)
    monkeypatch.setattr(kernels, "_flex_options", {"BACKEND": "FLASH"})
    out, _ = kernels.chain_attention(None, q, q, q, mask)
    assert seen == [{"BACKEND": "FLASH"}, {"BACKEND": "TRITON"}] and out.shape == (1, 128, 2, 4)
    kernels.chain_attention(None, q, q, q, mask)
    assert seen[-1] == {"BACKEND": "TRITON"} and len(seen) == 3   # Triton's from then on, FlashAttention's not retried
    monkeypatch.setattr(kernels, "_flex", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no kernel")))
    with pytest.raises(RuntimeError, match="no kernel"):
        kernels.chain_attention(None, q, q, q, mask)


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_a_chains_block_mask_is_create_block_masks_built_a_band_at_a_time(model_dir, monkeypatch):
    """On a GPU a chain's mask is FlexAttention's block mask, built a band of query blocks at a time: the same blocks as
    create_block_mask, however many bands; its mask_mod gives the dense mask a CPU uses; and through the block-mask
    path (eager FlexAttention on a CPU) the full-attention layer's states are the dense mask's."""
    from torch.nn.attention import flex_attention as flex

    trunk = list(range(1, 601))
    chain = ChainItem(trunk, [(600, list(range(800, 900))), (300, list(range(5, 65))),
                              (600, list(range(100, 190))), (450, [7, 8, 9])])
    x, pos, block, producing, plan = branching.pack(chain, "cpu", block=True)
    _, _, dense, _, _ = branching.pack(chain, "cpu")
    L = x.shape[1]
    n = -(-L // 128)
    assert torch.equal(flex.create_mask(block.mask_mod, None, None, L, L, "cpu"), dense)
    want = flex.create_block_mask(block.mask_mod, None, None, L, L, device="cpu")
    fields = ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
              "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices")
    for band in (1, 2 * 128 * 128 * n, 4 * 128 * 128 * n, 2 ** 28):   # 1, 2, 4 and all rows of blocks a band
        got = branching.block_mask(block.mask_mod, L, "cpu", band=band)
        assert all(torch.equal(getattr(got, f), getattr(want, f)) for f in fields), band
    assert int(want.full_kv_num_blocks.sum()) > 0 and int(want.kv_num_blocks.sum()) > 0   # full and partial blocks
    tr = _trainer(model_dir)
    monkeypatch.setattr(kernels, "_flex", flex.flex_attention)   # eager: no compile on a CPU
    monkeypatch.setattr(kernels, "_flex_options", {})
    decoder = tr.model.get_base_model().model
    with torch.no_grad(), branching.active(tr.model, plan):
        h_block = decoder(input_ids=x, position_ids=pos, attention_mask=block).last_hidden_state[0]
        h_dense = decoder(input_ids=x, position_ids=pos, attention_mask=dense).last_hidden_state[0]
    for idx in producing:
        assert torch.allclose(h_block[idx], h_dense[idx], atol=1e-5, rtol=1e-5)


def test_the_forward_kl_is_ligers_jsd_at_beta_zero_where_liger_is_installed(model_dir):
    """The distillation loss is the fused JSD loss with beta 0 the same runs once took from Liger's kernels (KL(teacher
    || student), the mean over positions): the same value from the same hidden states."""
    liger = pytest.importorskip("liger_kernel.transformers")
    tr = _trainer(model_dir)
    _, W = tr._decoder_head()
    g = torch.Generator().manual_seed(0)
    hs, ht = torch.randn(40, 64, generator=g), torch.randn(40, 64, generator=g)
    want = liger.LigerFusedLinearJSD(jsd_beta=0.0, temperature=1.0)(hs, W, ht, W)
    assert float(tr._forward_kl(hs, ht, W)) == pytest.approx(float(want), rel=1e-4)
