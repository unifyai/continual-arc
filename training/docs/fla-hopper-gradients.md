# fla's wrong gradients on Hopper GPUs: the check of 14 Oct 2026

**No training run here was affected.** Every Qwen3.5 training run (30 Sep to 14 Oct 2026) went through fla's kernels
on Hopper GPUs, but with Triton 3.7.1, the release that fixes the bug. On 8 of the 9 nodes whose first training log was
kept, the affected step also ran on fla's TileLang kernel rather than Triton's (the ninth is q9grpo's, item 5 below).
No published number needs a note, and none of the runs the earlier experiments' write-up cites does.

The [glossary](glossary.md) explains fla, Triton, TileLang and Hopper.

## The bug

On Hopper GPUs (H100, H200, H20), Triton 3.4.0 up to 3.7.0 compiles one of fla's kernels wrongly: the backward step of
the gated delta rule that computes dq, dk, dw and dg (`chunk_bwd_dqkwg`). The dk it returns is wrong, and so is the dg
computed from it. In the test on issue #640, dk was off by up to 14.65 and dg by up to 5.47 at the affected tile size,
against under 0.02 at the other tile sizes.

| When | What |
| --- | --- |
| 12 Nov 2025 | Reported: [fla issue #640](https://github.com/fla-org/flash-linear-attention/issues/640), "GDN precision error in triton3.5 and h20". |
| 15 Apr 2026 | fla [#827](https://github.com/fla-org/flash-linear-attention/pull/827), in fla 0.5.0: a TileLang kernel for that step, used when tilelang is installed. Without tilelang, fla raises an error on Hopper with Triton 3.4.0 or later instead of running the broken kernel. |
| 27 Jun 2026 | fla [#983](https://github.com/fla-org/flash-linear-attention/pull/983), in fla 0.5.2: Triton 3.7.1 fixes the bug, so the error is kept for Triton 3.4.0 to 3.7.0 only. |

Qwen3.5-9B has 24 Gated DeltaNet layers out of 32. When fla is installed, transformers trains those layers through
fla's kernels, so every training step runs the affected backward step. fla before 0.5.0 has no such check.

## What our training ran on

Training runs on the GPU node in prime-rl's Python environment (`uv run --no-sync` from prime-rl's checkout,
`~/prime-rl` by default). The node's setup builds it: prime-rl checked out at commit `a1822f7` (18 Sep 2026), then
`uv sync --all-extras --all-packages`, which installs the versions in prime-rl's lockfile.

| Part | Version |
| --- | --- |
| GPUs | 8 × H100 by default, 8 × H200 when a run asks; Nebius. Both are Hopper. |
| torch | 2.13.0+cu130. torch 2.13.0 itself requires Triton 3.7.1. |
| Triton | 3.7.1 |
| fla (flash-linear-attention) | 0.5.2, built from fla's git at `a954753` (3 Jul 2026), which includes #983 |
| tilelang | 0.1.12 |
| transformers | 5.6.2 |
| causal-conv1d | not installed |

Without causal-conv1d, every training log says "The fast path is not available ... Falling back to torch
implementation". That is only the layers' short convolution. transformers still sends the gated delta rule itself to
fla whenever fla is installed (`modeling_qwen3_5.py` in transformers 5.6.2).

The model servers (`serve.sh`, vLLM 0.30.0 with torch 2.13.0 in `~/vllm-0.30.0`) only serve the games and never
compute gradients. The bug is in a backward step, so they are not part of this check.

## How we know

1. **The setup is pinned.** The node setup pinned prime-rl `a1822f7` from 21 Sep and did not change after 24 Sep, so
   every result of this period comes from that commit. No script here sets `PRIME_RL_COMMIT`, `FLA_TILELANG` or
   `FLA_DISABLE_BACKEND_DISPATCH`.
2. **Training goes through fla.** q9loop's first training attempt (30 Sep, `train-a1.oom-1.log`) ran out of memory
   inside `~/prime-rl/.venv/.../fla/ops/gated_delta_rule/chunk.py`, in its backward.
3. **The Triton is 3.7.x.** The training logs warn from `triton/language/core.py:2284`. That warning sits at that line
   in Triton 3.7.0 and 3.7.1 only (3.6.0 has no such warning; 3.8.0 has it at line 2625). The lockfile and torch
   2.13.0's own requirement both say 3.7.1.
4. **The affected step ran on TileLang.** The first training on each node compiles a TileLang kernel ("TileLang
   begins to compile kernel `kernel`", once per GPU process). Later trainings on the same node load it from
   TileLang's cache. In Qwen3.5's layers this step is the only one fla runs on TileLang, and by default only on a
   Hopper GPU with Triton 3.4.0 or later. So the log line also shows the GPUs were Hopper. Seen in: q9loop (30 Sep),
   q9cycles (2 Oct), q9climb and probe (7 Oct), overfit1 and overfit2 (8 Oct), pure1 (9 Oct), and fwd-comparison
   (13 Oct). round2 and round2b trained on fwd-comparison's node after it.
5. **q9grpo (5 Oct) shows no TileLang compile**, on a fresh node, and the reason is not known. Its gradients are right
   either way. With this fla, the Triton kernel under Triton 3.4.0 to 3.7.0 stops training with an error at the first
   backward, and q9grpo trained to the end. So its step ran on TileLang or on Triton 3.7.1's fixed kernel.
6. **Runs whose training logs were not kept** (probe2, dagger1 to dagger6, fwd-common, fwd-staged) came from the same
   drivers and the same setup.

## The runs the earlier experiments cite

The earlier experiments' write-up named these runs: q9loop, q9loop-v3, q9cycles, q9grpo, q9climb,
oracle, probe, probe2, dagger1, dagger4, dagger5, dagger6, codeloop-pilot, fwd-common, fwd-staged, fwd-comparison
(with round2), and flash-c1-val. oracle, codeloop-pilot and flash-c1-val trained nothing. dagger4 played L, which
dagger3 trained. All the training ran on the setup above, so none is affected.

## Other places that trained a model for this work

| Where | Verdict |
| --- | --- |
| K2 Horizon 7B runs (26 to 28 Sep: k2loop, k2lib, and the baselines' K2 runs) | Not affected: every K2 layer is full attention, so fla is never called. |
| The baselines' Qwen3.5-27B runs (23 to 25 Sep; RunPod and Nebius H200s) | Not affected: the same node setup and prime-rl `a1822f7`. |
| Laptop environments (the trainer's and the baselines' virtual environments) | Not affected: no CUDA GPU, so transformers never calls fla. |

## From now on

`train.py` writes `kernels.json` into every training folder before it trains, and `realtime.py`'s `train` process
writes it into the run's `train/` folder for the paper's trained rows: the package versions, the GPU, and the path the
gated delta rule's backward step takes ([kernel_stack.py](../kernel_stack.py)). `train.py` stops before loading the
model when the GPU is Hopper, fla is installed, Triton is 3.4.0 to 3.7.0, and TileLang cannot take the step. Before moving the node setup off prime-rl
`a1822f7`, or training anywhere new, run `python kernel_stack.py` there and read its `path`. A training job on a new
Hopper machine needs Triton 3.7.1 or later (torch 2.13 or later), or tilelang with fla 0.5.0 or later.
