# Continual-ARC: training

This package trains Continual-ARC's trained rows and plays the rows they are read against. A small open model,
**Qwen3.5-9B**, plays the `arc1-small` stream in the prime-agent harness under the benchmark's rules, while a LoRA
adapter is trained between rounds on what it played. Three updates are compared:

- **DAgger**: imitate a stronger model's reply (the oracle's) at every state the student met.
- **OPSD**: SDFT's self-distillation, the student's own fresh replies moved toward the frozen base model reading the
  oracle's reply as a demonstration.
- **PG**: policy gradient on the games the student won.

The same package plays the reference rows on the open models, the two frozen tests after the stream, and DAgger's
"training once at the end".

## What produced which rows

| Script (`scripts/`) | What it runs | Rows |
| --- | --- | --- |
| `realtime.sh` | A trained row on one 8-GPU node: the player, the labeller and the trainer side by side, round r played by the adapter trained on rounds 0..r-2 | DAgger, OPSD and PG at 1, 2 and 4 users, the stream's 200 rounds |
| `frozen.sh` | A trained run's two frozen tests on its own node once its stream has ended | The trained rows' held-out and seen tests |
| `refs.sh` | The untrained student on the trained rows' stack, from nothing, on both tests | Rows H0 and F on Qwen3.5-9B, the trained rows' references |
| `reference.sh` | A reference row with its model served by OpenRouter, no GPU | Rows h, h0, hpl, hg, ha, hpl0, hplg and hgpl (the paper's H, H0, H-E, H-T, H-A, H-E0, H-E-T-E and H-T-E) and the bare-model rows dc, f, m and m0, on GLM-5.3 Flash and Qwen3.5-9B |
| `ref-tests.sh` | Each reference row's two frozen tests from its end-of-stream memory | The reference rows' held-out and seen tests |
| `scratch.sh` | At a DAgger run's end: its last phase's weights alone on both tests, then that phase trained once more from the base model and tested | DAgger's training once at the end |
| `paper.sh` | A frozen row round by round on the node's own vLLM | Rows H0 and F over the held-out instances alone; smoke tests of a node |

`serve.sh` and `vllm-serve.sh` run the model server every node script uses; `lib.sh` holds what they share.

## How a trained run works

Three loops on one GPU node hand each other files in the run's folder ([how a run works](docs/how-a-run-works.md)):

1. **The player** plays round r on phase r-2's adapter, loaded into the node's vLLM server and checked to answer
   differently from the base model; prime-agent plays the round's instances through the benchmark's runner, every
   model call saved with the token ids the server saw and sampled.
2. **The labeller** (DAgger and OPSD) asks the oracle, GLM-5.3 Flash at medium effort through OpenRouter, for its own
   reply to every call the student made, read exactly as the student read it with the oracle's working rules at the
   end. One game in ten is held out.
3. **The trainer** trains phase k on rounds 0..k from phase k-1's adapter once round k is ready: nothing until 24
   games have been played, then a rank-128 LoRA adapter at LR 1e-4, 64 calls a step, its held-out loss measured every
   5 steps and the phase stopped once 2 measurements in a row have not improved on the best.

After the 200 rounds `frozen.sh` serves the run's last phase and plays the held-out test (one instance of each of 100
tasks the stream never uses) and the seen test (four new instances of each stream task), every test user starting from
the end-of-stream memory. The settings and the reason for each are in [settings](docs/settings.md).

A trained run is started on its node with:

```bash
ROW=dagger USERS=1 LR=1e-4 START_AFTER=24 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 \
    setsid nohup bash ~/continual-arc/training/scripts/realtime.sh > /dev/null 2>&1 < /dev/null &
# once its stream has ended, on the same node:
(setsid -f nohup bash ~/continual-arc/training/scripts/frozen.sh dagger-u1-lag1 0,1,2,3,4,5,6,7 \
    > /dev/null 2>&1 < /dev/null)
```

with `ROW` one of `dagger`, `opsd` and `pg` and `USERS` 1, 2 or 4. The DAgger runs take `DOSE=recent PATIENCE=4` once
their phases stall ([settings](docs/settings.md#the-dose)).

## What it needs

- **A node of eight H100s** (80 GB) for a trained run: four for the model server, four for the trainer. Its frozen
  tests run on the same node. Nothing here creates or sets up a machine; [how a run works](docs/how-a-run-works.md#what-a-node-needs)
  lists what a node must hold.
- **The node's environment**, a prime-rl checkout at commit `a1822f7` (`PRIME_RL`, default `~/prime-rl`), whose
  environment pins the trainer's GPU stack: torch 2.13.0, Triton 3.7.1, flash-linear-attention 0.5.2, tilelang 0.1.12
  and transformers 5.6.2. Install into it peft, prime-agent 0.9.6, and this repository's benchmark, baselines and this
  package. The loops run there with `uv run --no-sync`.
- **vLLM 0.30.0** in an environment of its own (`VLLM_VENV`, default `~/vllm-0.30.0`), which `vllm-serve.sh` installs
  on first use; bubblewrap for the harness's sandbox; `Qwen/Qwen3.5-9B` in the node's Hugging Face cache.
- **An OpenRouter key** for the oracle and the reference rows: `OPENROUTER_API_KEY` in the environment or in
  `baselines/.env`.
- **The reference rows need no GPU**: a machine that runs prime-agent, with the repository's own environment and the
  key.

The GPU stack is the node's environment rather than an extra of this package: its versions are pinned together by that
checkout, vLLM must sit apart from it, and none of it is needed to play, read or test a run. The package itself depends
on the benchmark, the baselines and httpx alone, and imports torch, transformers, peft and safetensors only inside
the functions that need them (the trainer, the label set's tokenizer, an adapter's files).

Every script finds the package, the benchmark and the baselines from where it sits; the node's folders are parameters:
`PRIME_RL`, `VLLM_VENV`, `RUNS` (default `~/runs`, where every run writes) and `BASELINES` (default the repository's
`baselines/`). The examples assume the repository is cloned at `~/continual-arc`.

## Tests

From the repository root, with this folder a member of its uv workspace:

```bash
uv run --package continual-arc-baselines --directory training pytest
```

or from this folder with any environment that has the benchmark and the baselines installed: `python -m pytest`. No
test needs a GPU, a model server or a network. The trainer's and the data-parallel tests (`test_trainer.py`,
`test_parallel.py`, and three in `test_server.py`) need torch, transformers, peft and safetensors and skip without them;
they build a tiny random Qwen3.5 on the CPU and read the tiny Llama tokenizer `trl-internal-testing/tiny-LlamaForCausalLM-3.2`
from the Hugging Face hub or its cache. One test compares the distillation loss with Liger's fused kernel and skips
without `liger_kernel`. The scripts' tests run them in a fake home with every node tool stubbed. Lint with `ruff check`.

## Documents

- [How a run works](docs/how-a-run-works.md): what runs where, the three loops, the frozen tests, the reference rows,
  and what a run leaves on disk.
- [Settings](docs/settings.md): every method setting and why it is what it is, and the software stack.
- [The code](docs/code.md): what each module and script does.
- [Glossary](docs/glossary.md): every term.
