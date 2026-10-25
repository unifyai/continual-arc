"""The trainer's two departures from the model's own code on a GPU.

- **Attention under a chain's block mask** (:func:`chain_attention`, registered with transformers as
  :data:`ATTENTION`): FlexAttention when the mask is a block mask, which a chain on a GPU brings
  (:func:`continual_arc_training.branching.pack`), and scaled-dot-product attention otherwise, causal when there is no
  mask and following a dense 4D mask when one is given (a chain on a CPU). FlexAttention runs on FlashAttention's CuTe
  kernels where they are installed and on Triton's FlexAttention kernel once they refuse a call. The Triton kernel is
  named outright, never left to FlexAttention's own choice, which takes its decoding kernel for a query under 128
  tokens: a kernel with no backward, and none at all for Qwen3.5's 256-wide heads.
- **The adapter's matmuls in bfloat16** (:func:`bf16_lora`). PEFT keeps a bfloat16 model's LoRA weights in float32 and
  casts each projection's input to float32, so every adapter matmul runs in float32 beside float32 copies of its input
  and output. Here they run in bfloat16 with float32 master weights, as mixed-precision training does: the weights,
  their gradients, the optimizer and the saved adapter stay float32 in PEFT's layout. It changes the adapter's
  rounding, not its accuracy (on an H200 a step at 57k tokens ran 1.23 times faster, the update's cosine with PEFT's
  0.99986 against 0.99985 for PEFT's own repeat)."""
from __future__ import annotations

import importlib.util
import types
from typing import Any

ATTENTION = "chain_attention"
_TRITON = {"BACKEND": "TRITON"}
_flex: Any = None
_flex_options: dict[str, str] | None = None


def chain_attention(module: Any, query: Any, key: Any, value: Any, attention_mask: Any, dropout: float = 0.0,
                    scaling: float | None = None, **kwargs: Any) -> tuple[Any, None]:
    """The attention function the trainer's model runs with (the module docstring)."""
    from torch.nn.attention.flex_attention import BlockMask

    if isinstance(attention_mask, BlockMask):
        global _flex, _flex_options
        if _flex is None:
            import torch
            from torch.nn.attention.flex_attention import flex_attention

            _flex = torch.compile(flex_attention)
            _flex_options = {"BACKEND": "FLASH"} if importlib.util.find_spec("flash_attn.cute") else _TRITON
        gqa = query.shape[1] != key.shape[1]
        try:
            out = _flex(query, key, value, block_mask=attention_mask, scale=scaling, enable_gqa=gqa,
                        kernel_options=_flex_options)
        except Exception:
            if _flex_options == _TRITON:
                raise
            _flex_options = _TRITON   # the FlashAttention backend refused this call: Triton's kernel from now on
            out = _flex(query, key, value, block_mask=attention_mask, scale=scaling, enable_gqa=gqa,
                        kernel_options=_flex_options)
        return out.transpose(1, 2).contiguous(), None
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)


def _bf16_lora_forward(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """PEFT's LoRA Linear forward for a plain adapter (one active, no dropout, not merged) with the adapter's matmuls in
    the input's dtype: its float32 weights cast per call, the scaling folded into B, and the update added in the second
    matmul's epilogue (addmm, one rounding of the sum). Anything else goes to PEFT's own forward."""
    import torch
    import torch.nn.functional as F

    name = self.active_adapters[0] if len(self.active_adapters) == 1 else None
    if (args or kwargs or name is None or self.disable_adapters or self.merged or name not in self.lora_A
            or name in getattr(self, "lora_variant", {}) or not isinstance(self.lora_dropout[name], torch.nn.Identity)):
        return type(self).forward(self, x, *args, **kwargs)
    result = self.base_layer(x)
    h = F.linear(x, self.lora_A[name].weight.to(x.dtype))
    b = (self.lora_B[name].weight * self.scaling[name]).to(x.dtype)
    out = torch.addmm(result.reshape(-1, result.shape[-1]), h.reshape(-1, h.shape[-1]), b.t())
    return out.view(result.shape)


def bf16_lora(peft_model: Any) -> int:
    """Every PEFT LoRA Linear of ``peft_model`` with its adapter's matmuls in the input's dtype (the module docstring);
    how many were patched. The patches are instance attributes, so deleting one restores PEFT's forward."""
    from peft.tuners.lora import Linear as LoraLinear

    n = 0
    for module in peft_model.modules():
        if isinstance(module, LoraLinear) and "forward" not in module.__dict__:
            module.forward = types.MethodType(_bf16_lora_forward, module)
            n += 1
    return n
