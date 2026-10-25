# Continual-ARC: training

This folder holds the code that trained the paper's trained rows and played its rows on the open models: a small open
model, **Qwen3.5-9B**, learning from its own games on Continual-ARC's `arc1-small` stream, a LoRA adapter trained
between rounds while the stream goes on, in the same harness (prime-agent) and under the same rules as the untrained
rows. Three updates are compared: **DAgger** (imitate a stronger model's reply to each state the student met),
**OPSD** (SDFT's self-distillation toward the base model reading that reply as a demonstration) and **PG** (policy
gradient on the games the student won). The same code plays the reference rows the trained rows are read against, and
the two frozen tests after the stream.

It is a snapshot without history. The paper's nine trained runs ran on the code of 20 October 2026, which their nodes
kept to the end; this folder is the code as it stood at the paper's release, with the runs' settings as parameters. It
differs from what the runs ran in at least these ways: their phases trained without step checkpoints (`checkpoint.py`,
on here) and without a shared trainer pool (`gpupool.py`), and each row's phase ran through its own training code
rather than the one setup all three share here (`train.fit`; [docs/method-search.md](docs/method-search.md), "The
OPSD audit").

The records of the paper's runs are at [unifyai/continual-arc-runs](https://github.com/unifyai/continual-arc-runs).

## What produced which rows

| Entry point | What it ran | The paper's rows |
| --- | --- | --- |
| `realtime.sh` (`realtime.py`, `train.py`, `forward.py`) | A trained row on one 8-GPU node: the player, the labeller and the trainer side by side, round r played by the adapter trained on rounds 0..r-2 | DAgger, OPSD and PG at 1, 2 and 4 users (`dagger-u1-lag1` ... `pg-u4-lag1`): the stream's 200 rounds |
| `frozen.sh` (`frozen.py`, started by `frozen-launch.sh`) | A trained run's two frozen tests on its own node once its stream has ended | The trained rows' held-out and seen tests |
| `refs.sh` (`frozen.sh` with `ROW`, started by `refs-launch.sh`) | The untrained student on the same stack, from nothing, on both tests | Rows H0 and F on Qwen3.5-9B as the trained rows' references |
| `reference.sh` (`reference.py`) | A reference row, the model served by OpenRouter, no GPU | Rows H, H0, HPL, HG, HA, HPL0, HPLG, HGPL and the bare-model rows DC, F, M and M0, on GLM-5.3 Flash and on Qwen3.5-9B |
| `ref-tests.sh` (`frozen.py play --player`) | Each reference row's two frozen tests, from its end-of-stream memory | The reference rows' held-out and seen tests |
| `scratch.sh` (`scratch.py`, `grid-vals.sh`, started by `scratch-launch.sh`) | At a DAgger run's end: its last phase's weights alone on both tests, and its last phase trained once more from the base model, both adapters then tested | DAgger's "training once at the end" |
| `paper.sh` (`paper.py`) | A row round by round with no lag, on the node's own vLLM | Rows H0 and F on Qwen3.5-9B over the held-out instances alone, as references; smoke tests |

`plot_rows.py` and `paper_compare.py` read the rows' curves; `fleet.py` and `gpupool.py` lay runs out on nodes and
share a node's trainer GPUs; `paper_backup.sh` and `gcs_upload.py` copy a run off its node while it runs.
[docs/code.md](docs/code.md) says what every file does.

## How a run works

A trained run is three loops on one GPU node, handing each other files in the run's folder
([docs/how-a-run-works.md](docs/how-a-run-works.md)):

1. **The player** plays round r: `serve.sh` serves Qwen3.5-9B through vLLM with phase r-2's adapter loaded (a lag of
   one round, so the hardware never decides which model plays which round), checked to differ from the base model,
   and prime-agent plays the round's instances through the benchmark's runner, every model call saved as served.
2. **The labeller** (DAgger and OPSD) asks the oracle, GLM-5.3 Flash at medium effort through OpenRouter, for its own
   reply to every call the student made, read exactly as the student read it with the oracle's working rules at the
   end (c1 and the program library's procedure, which the student never reads). One game in ten is held out.
3. **The trainer** trains phase k on rounds 0..k from phase k-1's adapter once round k is ready: nothing until 24
   games have been played, then a LoRA adapter (rank 128, LR 1e-4, 64 calls a step), its held-out loss measured every
   5 steps and the phase stopped once 2 measurements in a row have not improved on the best. DAgger fine-tunes on the
   oracle's replies; OPSD distils, on its own fresh replies, the frozen base model reading the oracle's reply as a
   demonstration; PG raises the replies of the games it won through PPO's clipped surrogate against the adapter that
   played them.

After the 200 rounds, `frozen.sh` serves the run's last phase and plays the held-out test (one instance of each of 100
tasks the stream never uses) and the seen test (four new instances of each stream task), every test user starting
from the end-of-stream memory and nothing learnt during the test. The settings and why each was chosen are in
[docs/method-search.md](docs/method-search.md).

The paper's runs were started with:

```bash
ROW=dagger USERS=1 LR=1e-4 START_AFTER=24 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 \
    setsid nohup bash ~/continual-arc/training/realtime.sh > /dev/null 2>&1 < /dev/null &
# then, once its stream has ended, from a machine with the node's ssh key:
bash frozen-launch.sh dagger-u1-lag1 NODE 0,1,2,3,4,5,6,7
```

with `ROW` one of `dagger`, `opsd` and `pg` and `USERS` 1, 2 or 4; the DAgger runs were restarted with `DOSE=recent
PATIENCE=4` part of the way through ([docs/how-a-run-works.md](docs/how-a-run-works.md)).

## What it needs

- **GPUs.** A trained run takes a node of eight H100s (80 GB; the paper's ran on rented, mostly preemptible nodes),
  four for the model server and four for the trainer; its frozen tests run on the same node. The reference rows need
  no GPU: a CPU machine that runs prime-agent, with an OpenRouter key.
- **Machines are yours to provide.** unify rented and set up its nodes with scripts of its own (renting, leases,
  setup), which are not public; nothing here creates a machine. A node needs what the next items list,
  laid out as [docs/how-a-run-works.md](docs/how-a-run-works.md#what-a-node-needs) says.
- **prime-rl at commit `a1822f7`**, whose environment runs the loops and the trainer (torch 2.13.0, Triton 3.7.1,
  fla 0.5.2, tilelang 0.1.12, transformers 5.6.2, peft), with this repository's benchmark (`..`), the baselines
  (`../baselines`) and `ask_sd` installed into it.
- **vLLM 0.30.0** in an environment of its own, which `vllm-serve.sh` installs on first use; prime-agent 0.9.6 and
  bubblewrap for the harness; `Qwen/Qwen3.5-9B` in the node's Hugging Face cache.
- **`ask_sd`**, unify's trainer package: `ask_sd.trainer.Trainer` (LoRA training, data-parallel, and the supervised,
  self-distillation and policy-gradient losses), `ask_sd.calls`, `ask_sd.server`, `ask_sd.thoughts` and
  `ask_sd.parallel`. `train.py`, `realtime.py`'s trainer and parts of `paper.py` import it. It is not part of this
  repository, so the trainer cannot run without it; everything that only plays or reads runs does.
- **An OpenRouter key** for the oracle and the reference rows: `OPENROUTER_API_KEY` in the environment or in
  `../baselines/.env`.

Every script finds this folder, the benchmark and the baselines from where it sits; the node's own folders are
parameters: `PRIME_RL` (default `~/prime-rl`), `VLLM_VENV` (default `~/vllm-0.30.0`), `RUNS` (default `~/runs`, where
runs write) and `BASELINES` (default `../baselines`). The examples assume the repository is cloned at `~/continual-arc`.
A launcher that reaches a node takes its ssh destination as an argument, with no default.

## Tests

From this folder, with the benchmark and the baselines importable:

```bash
PYTHONPATH=.:../src:../baselines/src python -m pytest
bash test_frozen.sh && bash test_paper_backup.sh
```

No test needs a GPU, prime-rl or a network. On a laptop with neither torch nor `ask_sd`, 232 tests pass and 28 skip.
The skipped ones need:

- **`ask_sd`, with torch and transformers** (`ask_sd`'s training extra): `test_checkpoint.py`, `test_dev_loop.py`,
  `test_train_chains.py`, `test_train_records.py`, most of `test_paper.py`, the trainer's tests in `test_realtime.py`
  and one in `test_loop.py`. With `ask_sd` and its test helpers on the path, 365 pass and 14 skip.
- **ask-sd's own test helpers** (its `tests/` folder on `PYTHONPATH`): three DAgger phase tests in `test_dagger.py`.
- **the benchmark at v0.1.0**: one test in `test_loop.py` checks that `loop.py`'s validation split is the one the
  earlier experiments drew, which the benchmark's current task pools no longer give (the split itself is kept in
  `schedules/validation/`).

`test_overfit.py`'s card test reads the benchmark's Re-ARC data, which the benchmark fetches. The Python is linted
with `ruff check` under [ruff.toml](ruff.toml).

## Documents

- [docs/how-a-run-works.md](docs/how-a-run-works.md): what runs where, the three loops, the frozen tests, the
  reference rows, and what a run leaves on disk.
- [docs/method-search.md](docs/method-search.md): the search that set the trained rows' settings, each with the
  measurement behind it.
- [docs/code.md](docs/code.md): what each file does.
- [docs/glossary.md](docs/glossary.md): every term.
- [docs/fla-hopper-gradients.md](docs/fla-hopper-gradients.md): the software the trainer runs on, and the check that no
  run trained on fla's wrong gradients on Hopper GPUs.

Some modules hold code of the experiments that came before the paper's rows (September and October 2026: context
distillation, an offline DAgger, overfit tests and probes), because the rows import them; those experiments' own
scripts and records are not here.
