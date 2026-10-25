# What each file does

Every script starts with a comment (or a docstring) that explains it in full. This page is the short map. The shell
scripts and the Python they call run on the GPU node, but for the launchers (`frozen-launch.sh`, `refs-launch.sh`,
`scratch-launch.sh`) and `paper_backup.sh`, which run on another machine and reach the node over ssh, and the
reference rows' scripts, which run on a CPU machine. See [How a run works](how-a-run-works.md).

## The trained rows

| File | What it does |
| --- | --- |
| `realtime.sh` | One trained row, DAgger, OPSD or PG (`ROW`), on one node: starts the oracle's relay (DAgger and OPSD) and the model server and keeps `realtime.py`'s three loops running, each restarted when it dies, until the run is done. Its `started:` line in the run's `progress` records every setting. |
| `realtime.py` | The three loops: the **player** (round r on phase r-1-LAG's adapter, waited for and checked, `served.json` recording it), the **labeller** (`forward.py points`/`label` as games land, under the oracle's working rules, then `label-set` and `split --hold-every`) and the **trainer**, whose `train` process holds the model on the GPUs the node's pool gives the run (`gpupool.py`). Phase k trains on rounds 0..k through `train.fit` from one configuration (`train_args`), with the row's own update; `--dose` picks what a phase trains on (`aggregate`, `recent`, `replay`). |
| `train.py` | Training. `fit` runs a phase for every row alike: every pass planned before the first step (`plan`), whole chains of a game's calls cut to fit each GPU, the held-out loss every `--eval-steps` steps, the best adapter kept, the patience stop, the step checkpoints, `phase.json` and `DONE`. Only the update is the row's: `Imitation` (DAgger, `--objective sft`), `SelfDistillation` (OPSD) and `PolicyGradient` (PG, `--objective pg`: `pg_items`, REINFORCE on the won games). It also holds `--objective opd` (per-token distillation from a teacher's scores, not a row of the paper) and the earlier experiments' context distillation. `--lora-targets` names the projections the adapter trains (the default trains the attention and MLP projections, none of the Gated DeltaNet layers'). |
| `checkpoint.py` | A phase's step checkpoints: one on SIGTERM and one every `EVERY` (25) steps, each synced whole before `state.json` names it, so a phase cut short goes on from the last one on the same batches in the same order. |
| `gpupool.py` | The trainer GPUs runs on one node share: runs given the same `TRAIN_GPUS` share them as one pool, split evenly over the runs with a phase to train, under an flock in `RUNS/gpupool/<gpus>.json`; lists that overlap without being the same are refused. |
| `fleet.py` | Where real-time runs' GPUs go: `profile` reads a run's rounds from its files, `simulate` plays the lag-1 pipeline over them, `plan` lays out a set of runs on 8-GPU nodes from the fastest plan to the cheapest and picks one by `--deadline` or `--budget` (`--launch` prints each run's `realtime.sh` line), `bench-serve` and `bench-train` time a model server and a trainer. |
| `kernel_stack.py` | Writes `kernels.json` into each training folder (the package versions, the GPU, and which kernels train Qwen3.5's linear-attention layers), and stops training on a stack known to give wrong gradients ([fla-hopper-gradients.md](fla-hopper-gradients.md)). |
| `serve.sh` | The model server on port 8101 (vLLM 0.30, Qwen3.5-9B with its chat template and MTP speculative decoding), adapters loaded into it by name; a healthy server is kept only when it was launched as asked. |
| `vllm-serve.sh` | Starts a vLLM server from an environment of its own (`VLLM_VENV`), installing it on first use; `serve.sh`, `frozen.sh` and `grid-vals.sh` start every server through it. |
| `qwen35_prefill.jinja` | Qwen3.5's chat template, with a `reasoning_prefill` that lets text open the model's own reasoning. |

## Play

| File | What it does |
| --- | --- |
| `loop.py` | The plays every row shares: schedules, prime-agent's play of a served model with every call captured, the harness settings (`HARNESS`, `HARNESS_BUDGETS`: `default` is prime-agent's own budget, `block` the 40-call cap with the seconds raised close to the 1,800 s ceiling), the replay of recorded games (`replay_proxy`, for a round taken up from its last call), Qwen3.5's adapter-serving fix (`served_adapter`, `vllm-adapter`), and the paired comparison of two plays. |
| `paper.py` | One round of a row through the benchmark's runner (`play_round`, over `schedule_cut`, every instance captured under the round's own folder), the served-adapter check (`check-served`), the schedules (`full_schedule`, `served`, the epilogue's cuts), the rows' settings, and the resume from a round's last call (`Resume`, off: `RESUME_CALLS = False`). |
| `paper.sh` | The synchronous driver of one row on the node (H0, H, the bare model or PG): serve, check, play round r and, for PG, train it, every stage marked. |
| `memory.py` | Per-round memory-use counters for every row: a user's harness state diffed against the round before, by area (`programs/`, skills, the harness's notes, the rest), and a round's games scanned for a read of what was kept; `round_report` writes `rounds/rNNNN/memory.json`. |
| `render.py` | A play made readable (every game in full, a sample, and counts); its parsing of calls and actions is what the rows' code reads games with. |

## Labels

| File | What it does |
| --- | --- |
| `forward.py` | Labelling every call of a play: `points` (every student call as a labelling point), `label` (the oracle's reply at each, with its working rules at the end of the latest message and its reasoning required, asked again with a note saying what went wrong), `label-set` (each label as the student writes it), `split` (one game in `--hold-every` held out, `held_game`), `spent` and `game-measures`. |
| `dagger.py` | DAgger's data: a point's id (`point_id`, so a label is never read for another game's call), the student's target from the oracle's reply (`student_message`, `makes_move`), the replay dose (`replay_plan`), and the earlier experiments' training sets and reports. |
| `expert.py` | The oracle's client (`Expert`), which `forward.py label` asks through the relay; in the earlier experiments it also played the student's call right after each failure, inside the student's own game (DAgger's takeover). |
| `expert_relay.py` | The oracle's endpoint for the labeller: a proxy on the same machine that forwards to OpenRouter with the key it alone holds, pinned to one model and its hosts. |
| `oracle_hint.py` | `oracle_rules` and `library_rules`, the working rules the trained rows' oracle labels under (c1 and the program library's procedure, or c1 alone); and the earlier experiments' hinted plays. |
| `rules/c1.md`, `rules/c1e.md` | The playing rules. `c1` is what the trained rows' oracle reads; `c1e` replaces c1's first two rules with facts about the harness's limits and is not used by the paper's rows. |

## The frozen tests and the references

| File | What it does |
| --- | --- |
| `frozen.py` | The two frozen tests after a run's stream: `seed` (each test user from stream user i mod U's end-of-stream memory, its paths renamed, forked from its state log where the run keeps one), `play` (one round of 100 users through `paper.play_round`; `--player` for a reference row on OpenRouter, `--reference` and `--row` from an empty state, `--no-memory` for the weights alone), `model` and `summary`. |
| `frozen.sh` | A trained run's two tests on its own node once its stream has ended: two model servers on half of the GPUs each, the run's last phase loaded into both, both tests played side by side, then `summary.json`. `ROW` plays a reference from nothing; `ADAPTER`/`SERVED` another adapter; `MEMORY=none` the weights alone. |
| `frozen-launch.sh` | Copies the tools to a node and starts `frozen.sh` there, from another machine. |
| `refs.sh`, `refs-launch.sh` | The trained rows' references on the same stack, rows h0 and bare from nothing on both tests, played on a trained run's node once its tests are done, on a copy of the current code. |
| `reference.py` | The reference rows on OpenRouter, played as `paper.play_round` plays a row: `play` for the harness rows (h, h0, hpl, hg, ha, hpl0, hplg, hgpl) and `bare` for the bare-model rows (dc, f, m, m0), each for GLM-5.3 Flash or Qwen3.5-9B. |
| `reference.sh` | Keeps one reference row going on a reference machine, restarting `reference.py` when it stops. |
| `ref-tests.sh` | Every reference row's two frozen tests, on a worker that copies each row's memory from its machine, plays the tests with the row's own player and copies them back. |
| `schedules/` | The paper's two test schedules (`arc1-small` at seed 0, as the benchmark's `configs.test_schedule` makes them), and the 100 validation puzzles of the earlier experiments with their aliases (`validation/`). |

## At the stream's end

| File | What it does |
| --- | --- |
| `scratch.py` | The from-scratch retrain: DAgger's phase K trained from the base model on the run's own label sets, through the run's own `realtime.py` and `train.py`, and `summary` comparing it with the continual student. |
| `scratch.sh` | On a DAgger run's node once its runs and tests are done: the weights-alone tests, the retrain, both adapters on the validation puzzles, the fresh adapter's frozen tests, the summary; each stage marked. |
| `scratch-launch.sh` | Copies the tools and the continual student's phase-K adapter (from the run's backup in a bucket) to the node and starts `scratch.sh`. |
| `grid-vals.sh` | Several adapters' plays of the validation puzzles at once on one node, a server each, then their per-game measures. |

## Reading the rows

| File | What it does |
| --- | --- |
| `plot_rows.py` | The rows against each other from their run folders, partial runs included: per-round cost and solve rate, cost by a user's k-th exposure to a task (the benchmark's own `metrics.cost_by_user_exposure`), training curves from `phase.json` and the memory counters. One PNG and one CSV per figure. |
| `paper_compare.py` | A row's learning curve beside the paper's baseline runs, read from a checkout of the run records ([continual-arc-runs](https://github.com/unifyai/continual-arc-runs)). |
| `paper_backup.sh` | Backs a run up from its node while it runs: its small files to a local folder by one filtered rsync, and with `GCS=1` the whole run from the node to a bucket of yours. |
| `gcs_upload.py` | Uploads a folder from the node straight to a bucket with a short-lived token read on stdin, each file checked by MD5, a rerun sending only what changed. |

## Code of the earlier experiments that the rows import

The rows import these modules, so they are here; the scripts and records of the experiments they were written for
(Sep-Oct 2026, before the paper's rows) are not.

| File | What it does |
| --- | --- |
| `overfit.py`, `overfit_expert.py`, `overfit_cards.json` | The recall overfit: could on-policy distillation write five puzzles' rules into the base model? `oracle_hint.py` reads captured plays with `overfit.read_docs` and `agent_calls`. |
| `probe.py` | The same-state replay probes of the moments the student was stuck; `overfit.py` uses its `check_program`. |
| `analysis/trend.py` | Per-play behaviour counts, which `probe.py` imports. |

## Tests

Run them from this folder with `python -m pytest` (the Python tests) and `bash test_frozen.sh` and
`bash test_paper_backup.sh` (the shell tests); the README says what each needs.

| File | What it tests |
| --- | --- |
| `test_realtime.py` | The three loops' file handshakes, the lag, the warm-up, the doses, the oracle's rules kept to the end, `realtime.sh`'s settings and launch lines. |
| `test_train_chains.py`, `test_train_records.py`, `test_dev_loop.py` | How `train.py` finds chains and builds steps, `pg_items`, `opd_items`, `plan`, and what a policy-gradient step keeps in `phase.json`. |
| `test_checkpoint.py` | Step checkpoints: written whole, resumed on the same batches, refused when they do not match the phase. |
| `test_gpupool.py`, `test_fleet.py` | The pool's split and the trainer's decisions over it, and fleet.py's profile, simulator and planner. |
| `test_kernel_stack.py` | The stop on a stack with wrong gradients, and `kernels.json`. |
| `test_scratch.py` | The retrain's summary: both adapters' held-out loss and their plays compared instance by instance. |
| `test_paper.py` | `paper.py`'s schedule cuts, the round-sharded capture, a round's training reading its own games, the served-adapter check against a fake server, the resume from the last call, and `paper.sh`'s shell logic. |
| `test_reference.py` | The reference rows' settings and players. |
| `test_frozen.py`, `test_frozen.sh` | The frozen tests' seeding (paths renamed in text files, links and SQLite rows), the summary, the schedules against the benchmark's own; `frozen.sh` end to end in a fake home. |
| `test_loop.py`, `test_memory.py`, `test_plot_rows.py`, `test_paper_compare.py` | The shared play code, the memory counters, the plots and the comparison, on synthetic runs. |
| `test_forward.py`, `test_dagger.py`, `test_oracle_hint.py`, `test_c1_loader.py` | Labelling, the expert's trigger and view, DAgger's helpers and phases, the hinted plays' checks, and c1's grid loader (an earlier instance's log beside the current one). |
| `test_overfit.py` | The earlier overfit test's schedules, its cards' programs on fresh Re-ARC pairs, and its report. |
| `test_gcs_upload.py`, `test_paper_backup.sh` | The uploader against a fake bucket on localhost, and the backup against a fake node. |
