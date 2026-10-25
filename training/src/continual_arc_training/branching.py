"""A chain (:class:`continual_arc_training.chains.ChainItem`) through the model in one pass: the trunk, then each
branch, every branch's states those it would have alone after its own prompt.

The attention layers see the layout through a mask (:func:`pack`): the trunk is causal, and a branch attends to the
trunk before its point and to its own earlier ids, never to the trunk past its point or to another branch. Qwen3.5's
linear-attention layers (Gated DeltaNet) read their input in order through a running memory and a short convolution,
with no mask, so a branch would read the trunk past its point and the branches before it. While a :class:`BranchPlan`
is active (:func:`active`), such a layer reads the trunk in pieces cut at the branch points, each starting from the
memory the piece before ended with, and each branch from the memory at its point and the trunk's last
``conv_kernel_size - 1`` inputs before it, with gradients through the memory to the trunk. A reply's states are then
exactly those it has passed alone, to floating-point rounding. With no plan active the layer runs as loaded."""
from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from continual_arc_training.chains import ChainItem


@dataclass
class BranchPlan:
    """One packed chain's layout: the trunk's length, and per branch its point (the length of the prompt it continues,
    at most the trunk's) and its span in the packed sequence (start, length), the branches laid out in order after the
    trunk."""

    trunk_len: int
    points: list[int]
    spans: list[tuple[int, int]]

    def check(self, total: int) -> None:
        end = self.trunk_len
        for p, (s, n) in zip(self.points, self.spans, strict=True):
            if not 0 < p <= self.trunk_len or s != end or n < 1:
                raise ValueError(f"branch plan: point {p} or span ({s}, {n}) out of order (trunk {self.trunk_len})")
            end = s + n
        if end != total:
            raise ValueError(f"branch plan: the spans end at {end}, the sequence at {total}")


# FlexAttention's block size, both ways (create_block_mask's default)
BLOCK = 128


def block_mask(mask_mod: Any, length: int, device: Any, band: int = 2 ** 28) -> Any:
    """FlexAttention's block mask for ``mask_mod`` over ``length`` tokens, exactly as ``create_block_mask(mask_mod,
    None, None, length, length)`` builds it, but from a band of query blocks at a time holding at most ``band``
    query-key pairs: create_block_mask holds the whole mask, a padded copy and a 64-bit copy for the block sums, 11 to 12
    bytes a pair, some 37 GB at a 57,344-token window."""
    import torch
    from torch.nn.attention.flex_attention import BlockMask, create_mask

    n = -(-length // BLOCK)
    rows = max(1, band // (n * BLOCK * BLOCK))   # query blocks per band
    partial = torch.zeros(n, n, dtype=torch.int32, device=device)
    full = torch.zeros(n, n, dtype=torch.int32, device=device)
    for r in range(0, n, rows):
        q0, q1 = r * BLOCK, min(length, (r + rows) * BLOCK)
        m = create_mask(lambda b, h, q, kv, q0=q0: mask_mod(b, h, q + q0, kv), None, None, q1 - q0, length,
                        device)[0, 0]
        m = torch.nn.functional.pad(m, (0, n * BLOCK - length, 0, -(q1 - q0) % BLOCK))   # whole blocks, the rest masked
        s = m.view(-1, BLOCK, n, BLOCK).sum(dim=(1, 3), dtype=torch.int32)
        full[r:r + len(s)] = s == BLOCK * BLOCK
        partial[r:r + len(s)] = (s > 0) & (s < BLOCK * BLOCK)

    def ordered(dense: Any) -> tuple[Any, Any]:   # per query block, the count, then the blocks in order
        return (dense.sum(-1, dtype=torch.int32)[None, None],
                torch.argsort(dense, dim=-1, descending=True, stable=True).to(torch.int32)[None, None])

    return BlockMask.from_kv_blocks(*ordered(partial), *ordered(full), BLOCK_SIZE=(BLOCK, BLOCK), mask_mod=mask_mod,
                                    seq_lengths=(length, length))


def pack(item: ChainItem, device: Any, block: bool | None = None) -> tuple[Any, Any, Any, list[Any], BranchPlan]:
    """One sequence for a chain: the trunk, then each branch in turn, a branch's positions continuing from its point.
    The mask is FlexAttention's block mask on a GPU, or wherever ``block`` asks for it, else the same mask dense, for a
    CPU's scaled-dot-product attention. Returns the input ids, the position ids, the mask, for each branch the indices
    of the hidden states that produced its reply's ids (the state at its point's last id, then its own but last, a
    head's ids read and not graded), and the plan the linear-attention layers read."""
    import torch
    from torch.nn.attention.flex_attention import create_mask

    trunk_len = len(item.trunk)
    ids, pos, seg, plen, points, spans, producing = list(item.trunk), list(range(trunk_len)), [0] * trunk_len, [0], [], [], []
    if item.heads and len(item.heads) != len(item.branches):
        raise ValueError(f"ChainItem: {len(item.heads)} heads for {len(item.branches)} branches")
    for k, (point, branch) in enumerate(item.branches, start=1):
        head = item.head(k - 1)
        if not 0 < point <= trunk_len or not 0 <= head < len(branch):
            raise ValueError(f"ChainItem: branch {k} at {point} (trunk {trunk_len}, {len(branch)} ids, head {head})")
        start = len(ids)
        ids += branch
        pos += range(point, point + len(branch))
        seg += [k] * len(branch)
        plen.append(point)
        points.append(point)
        spans.append((start, len(branch)))
        producing.append(torch.tensor([point - 1, *range(start, start + len(branch) - 1)][head:], device=device))
    length = len(ids)
    seg_t, plen_t = torch.tensor(seg, device=device), torch.tensor(plen, device=device)

    def mask_mod(b: Any, h: Any, q: Any, kv: Any) -> Any:
        on_trunk = seg_t[q] == 0
        return (kv <= q) & (((seg_t[kv] == 0) & (on_trunk | (kv < plen_t[seg_t[q]])))
                            | ((seg_t[kv] == seg_t[q]) & ~on_trunk))

    if block is None:
        block = str(device).startswith("cuda")
    mask = block_mask(mask_mod, length, device) if block else create_mask(mask_mod, None, None, length, length, device)
    return (torch.tensor([ids], device=device), torch.tensor([pos], device=device), mask, producing,
            BranchPlan(trunk_len, points, spans))


def _is_linear_attention(mod: Any) -> bool:
    return all(hasattr(mod, a) for a in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "A_log",
                                         "dt_bias", "out_proj", "norm"))


def _kernel(mod: Any) -> Any:
    """The chunked gated delta rule the layer itself uses: its own (transformers 5.6: fla's kernel when installed),
    else its module's (later releases: the reference, swapped for fla's from the kernel hub when installed)."""
    k = getattr(mod, "chunk_gated_delta_rule", None)
    return k if k is not None else sys.modules[type(mod).__module__].torch_chunk_gated_delta_rule


def branch_forward(mod: Any, hidden_states: Any, plan: BranchPlan) -> Any:
    """A Gated DeltaNet layer's output over a packed chain (``hidden_states`` [1, L, hidden]) laid out by ``plan``."""
    import torch
    import torch.nn.functional as F

    if hidden_states.shape[0] != 1:
        raise ValueError("a branch plan packs one chain: batch 1")
    if getattr(mod, "activation", "silu") not in ("silu", "swish"):
        raise ValueError(f"branch_forward: activation {mod.activation!r}")
    L, N, K = hidden_states.shape[1], plan.trunk_len, mod.conv_kernel_size
    plan.check(L)
    kernel = _kernel(mod)
    pre = mod.in_proj_qkv(hidden_states).transpose(1, 2)       # [1, C, L], before the convolution
    z = mod.in_proj_z(hidden_states)
    b = mod.in_proj_b(hidden_states)
    a = mod.in_proj_a(hidden_states)

    def conv(x: Any) -> Any:                                   # the causal convolution, then SiLU, as the layer
        return F.silu(mod.conv1d(x)[:, :, :x.shape[-1]])

    def inputs(mixed: Any, b_: Any, a_: Any) -> tuple[Any, ...]:
        T = mixed.shape[-1]
        q, k, v = torch.split(mixed.transpose(1, 2), [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        q = q.reshape(1, T, -1, mod.head_k_dim)
        k = k.reshape(1, T, -1, mod.head_k_dim)
        v = v.reshape(1, T, -1, mod.head_v_dim)
        beta = b_.sigmoid()
        g = -mod.A_log.float().exp() * F.softplus(a_.float() + mod.dt_bias)
        rep = mod.num_v_heads // mod.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        return q, k, v, g, beta

    def rule(q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any, final: bool) -> tuple[Any, Any]:
        return kernel(q.contiguous(), k.contiguous(), v.contiguous(), g=g.contiguous(), beta=beta.contiguous(),
                      initial_state=state, output_final_state=final, use_qk_l2norm_in_kernel=True)

    # the trunk, in pieces cut at the branch points, each from the memory the one before ended with
    q, k, v, g, beta = inputs(conv(pre[:, :, :N]), b[:, :N], a[:, :N])
    outs, memory, state, start = [], {}, None, 0
    for cut in sorted({p for p in plan.points if p < N} | {N}):
        o, state = rule(q[:, start:cut], k[:, start:cut], v[:, start:cut], g[:, start:cut], beta[:, start:cut], state,
                        True)
        outs.append(o)
        memory[cut] = state
        start = cut
    # each branch, from the memory at its point and the trunk's convolution inputs just before it
    for p, (s, n) in zip(plan.points, plan.spans, strict=True):
        ctx = pre[:, :, max(0, p - (K - 1)):p]
        if ctx.shape[-1] < K - 1:                              # the trunk's own zero padding
            ctx = F.pad(ctx, (K - 1 - ctx.shape[-1], 0))
        mixed = conv(torch.cat([ctx, pre[:, :, s:s + n]], dim=-1))[:, :, K - 1:]
        qb, kb, vb, gb, bb = inputs(mixed, b[:, s:s + n], a[:, s:s + n])
        o, _ = rule(qb, kb, vb, gb, bb, memory[p], False)
        outs.append(o)
    core = torch.cat(outs, dim=1)
    core = mod.norm(core.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)).reshape(1, L, -1)
    return mod.out_proj(core)


def install(model: Any) -> int:
    """Make every linear-attention layer of ``model`` branch-aware (idempotent): it runs :func:`branch_forward` while a
    plan is active (:func:`active`), as loaded otherwise. The text model's linear-attention mask (transformers'
    ``_update_linear_attn_mask``, which compares the mask to 1 and cannot read a block mask) is left out while a plan is
    active, since the branch-aware layers do not read it. Returns how many layers were installed."""
    n = 0
    for mod in model.modules():
        if _is_linear_attention(mod) and not hasattr(mod, "_branch_forward_as_loaded"):
            mod._branch_forward_as_loaded = mod.forward
            mod._branch_plan = None

            def forward(hidden_states: Any, *args: Any, _mod: Any = mod, **kwargs: Any) -> Any:
                if _mod._branch_plan is None:
                    return _mod._branch_forward_as_loaded(hidden_states, *args, **kwargs)
                return branch_forward(_mod, hidden_states, _mod._branch_plan)

            mod.forward = forward
            n += 1
        elif hasattr(mod, "_update_linear_attn_mask") and not hasattr(mod, "_branch_mask_as_loaded"):
            mod._branch_mask_as_loaded = mod._update_linear_attn_mask
            mod._branch_on = False

            def update(*args: Any, _mod: Any = mod, **kwargs: Any) -> Any:
                return None if _mod._branch_on else _mod._branch_mask_as_loaded(*args, **kwargs)

            mod._update_linear_attn_mask = update
    return n


@contextlib.contextmanager
def active(model: Any, plan: BranchPlan) -> Iterator[None]:
    """``plan`` active in ``model``'s branch-aware layers for the block: the forward pass and, under gradient
    checkpointing, the backward's recomputation must both run inside it."""
    layers = [m for m in model.modules() if hasattr(m, "_branch_forward_as_loaded")]
    masks = [m for m in model.modules() if hasattr(m, "_branch_mask_as_loaded")]
    for m in layers:
        m._branch_plan = plan
    for m in masks:
        m._branch_on = True
    try:
        yield
    finally:
        for m in layers:
            m._branch_plan = None
        for m in masks:
            m._branch_on = False
