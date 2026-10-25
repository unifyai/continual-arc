# The code

What each module and script does. Every module is run as `python -m continual_arc_training.MODULE` where it has a
command line; [how a run works](how-a-run-works.md) says how they fit together.

## The package (`src/continual_arc_training/`)

### Playing

| Module | What it does |
| --- | --- |
| `play` | The player: a row's rounds through the benchmark's runner on the node's model server, one round a call, with the settings every row plays under (the harness, the action budget, the sampling, the window). The schedule and its cuts (`full_schedule`, `schedule_cut`, `epilogue_cut`, `wide_epilogue_cut`); each game saved as it ends under its round's `captured/` (`save_instance_hook`); H0's wipe before every round. Command line: one round of a row (`paper.sh`). |
| `references` | The reference rows with the model on OpenRouter: the two players, the harness rows (H, H0, HPL, HG, HA, HPL0, HPLG, HGPL; `hand_storage` gives a storer the turn after each verdict) and the bare-model rows (DC, F, M, M0; `keep_library` carries row M's memory between rounds). Command line: every round not yet played (`reference.sh`). |
| `memory` | A round's memory-use counters: each user's harness state snapshotted and diffed against the round before by area (programs, skills, the harness's store, notes, scratch), the reads of each area in the round's calls, and whether a game reused a saved program or solved without a demo; written to the round's `memory.json`. Command line: a round's summary line (`paper.sh`). |
| `frozen` | The two frozen tests: each test user seeded from its stream user's end-of-stream state (forked from a state log by the baselines' seeding, or copied with every path renamed, SQLite rows included), a trained run's test played on its server, a reference row's through its own player, a reference from nothing, a run's weights alone; the summary. Command line: `model`, `play`, `summary` (`frozen.sh`, `ref-tests.sh`). |

### The model calls and the server

| Module | What it does |
| --- | --- |
| `calls` | A play's model calls as training reads them (`Call`: the request, the prompt's ids as the server tokenised them, the reply's ids as it sampled them), read back from a play's captured games with each instance's outcome and cost (`load_calls`); the agent's own calls told apart from prime-agent's internal ones; a prompt rendered locally exactly as vLLM 0.30 renders it (`render_prompt_ids`); more replies drawn at a call's prompt (`resample`). |
| `server` | The model server as the rows talk to it: the client, with retries and an adapter loaded again after a restart (`Server`); the copy of an adapter under the checkpoint's names that vLLM loads (`served_adapter`); the check that a served name answers from its adapter's weights (`check_served_adapter`). Command line: `vllm-adapter`, the folder vLLM should load an adapter from (`lib.sh`). |

### Labelling

| Module | What it does |
| --- | --- |
| `oracle` | The oracle: its relay to OpenRouter, which alone holds the key and pins every request to one model and its hosts (`relay`); its client (`Oracle`); its working rules (`oracle_rules`); one point's label, asked again with a note when the reply names the rules, has no reasoning or makes no move (`ask_label`). Command line: `relay` (`realtime.sh`). |
| `labeller` | DAgger's and OPSD's labels: every call the student made as the agent as a point (`points`), the oracle's reply at each under the working rules and the round's budget (`label`), the label set rendered as the student writes each label (`label_set`, `student_message`, `render_reply`), and its split into a training side and a held-out side (`split`, `held_game`). |
| `rules/c1.md` | The playing rules the oracle labels under. |

### Training

| Module | What it does |
| --- | --- |
| `realtime` | The real-time driver of the trained rows: the player loop (each round on phase r-1-LAG's adapter, served and checked), the labeller loop, and the trainer loop, which runs each phase in a `train` process on the GPUs the node's pool gives the run; the dose, the held-out games, PG's behaviour log-probabilities, pruning. Command line: `player`, `labeller`, `trainer`, `train` (`realtime.sh`). |
| `phase` | A training phase, the same for every trained row but for its update: its steps planned before the first (`plan`), the held-out stop, the step checkpoints and the phase's record on disk (`fit`); the trainer it runs on (`make_trainer`). |
| `updates` | The three updates: `Imitation` (DAgger), `SelfDistillation` (OPSD, with the student it publishes after every step, `LiveAdapter`, and its fresh replies, `draw`) and `PolicyGradient` (PG); the items each trains on (`sft_items`, `distill_items`, `pg_items`) and PG's behaviour log-probabilities (`behaviour_logps`, `attach_behaviour`). |
| `chains` | A game's calls as chains: calls whose prompts each extend the last, read in one pass with each reply a branch at its prompt's end (`TokenItem`, `ChainItem`, `find_chains`); pieces that fit the window (`chain_pieces`, `distill_pieces`); a pass's steps built from whole chains (`chain_batches`). |
| `trainer` | The LoRA trainer: its training passes over chains (`sft_chains`, `distill_chains`, `policy_gradient_chains`), their forward-only scores (`nll_chains`, `distill_scores_chains`, `policy_gradient_scores_chains`), the behaviour log-probabilities (`behaviour_logps_chains`), the optimizer step, and the adapter saved and loaded with its optimizer's moments. |
| `branching` | A chain through the model in one pass: its packed sequence and attention mask (`pack`, `block_mask`), and the Gated DeltaNet layers made branch-aware, each branch read from the trunk's memory at its point (`install`, `active`, `branch_forward`). |
| `kernels` | The trainer's two departures from the model's own code on a GPU: attention under a chain's block mask (`chain_attention`) and the adapter's matmuls in bfloat16 (`bf16_lora`). |
| `parallel` | Data parallelism: a trainer as one rank of a group (`DataParallelRank`, which `Trainer` inherits: the deal by cost, the gathering, the gradient sum, the check that every rank holds the same adapter) and the trainer's methods over several GPUs from one process (`DataParallelTrainer`, with one worker process per other GPU). |
| `checkpoints` | A phase's step checkpoints: written on SIGTERM and every 25 steps, committed on a thread of their own, the best and the newest kept; a phase resumes only from its own (`fingerprint`) and on batches that replay as recorded. |
| `gpupool` | The trainer GPUs real-time runs on one node share: an even split among the runs with a phase to train, agreed through one state file under an flock. |
| `stack` | The kernels Qwen3.5's linear-attention layers train through on a machine, written to a run's `train/kernels.json`, and a stop before training on a stack known to compute wrong gradients. Command line: this machine's stack. |
| `scratch` | Training once at the end: a DAgger run's phase K trained again from the base model, and the summary comparing both adapters' held-out loss and frozen tests. Command line: `train`, `summary` (`scratch.sh`). |
| `schedules/` | The two frozen tests' schedules for `arc1-small` at seed 0, the benchmark's own. |

## The scripts (`scripts/`)

| Script | What it does |
| --- | --- |
| `lib.sh` | What the node scripts share: the folders (`PRIME_RL`, `VLLM_VENV`, `RUNS`, `BASELINES`), the model and its server's arguments, the server's health check and stop, the adapter folder vLLM loads. |
| `realtime.sh` | One trained row on one node: the relay and the server kept up, the driver's loops run side by side and restarted when they die. |
| `serve.sh` | The stream's model server on :8101, started as asked unless it already serves so, and adapters loaded into it. |
| `vllm-serve.sh` | A vLLM server from its own environment (installed on first use), up when it answers `/health`. |
| `frozen.sh` | A run's two frozen tests on its node once its stream has ended, each on a server of its own; or a reference from nothing (`ROW`), or the weights alone (`MEMORY=none`). |
| `refs.sh` | The trained rows' references from nothing (rows h0 and bare) once a run's own tests are done. |
| `paper.sh` | Row H0 or the bare model round by round on the node's own vLLM. |
| `reference.sh` | One reference row kept going until its stream is played. |
| `ref-tests.sh` | Every finished reference row's frozen tests, a few rows at once. |
| `scratch.sh` | Training once at the end of a DAgger run, with its weights-alone tests. |

## The tests (`tests/`)

One test module per module, each on hand-made data with play, the oracle, the server and training faked where they
would need a network or a GPU. `test_trainer.py` and `test_parallel.py` run the real trainer on a tiny random Qwen3.5
on the CPU and compare every pass over a chain with the same quantities computed directly, call by call;
`test_frozen.py` and `test_scripts.py` run the node scripts in a fake home with every node tool stubbed; `fakes.py`
holds the tokenizers and games the tests share.
