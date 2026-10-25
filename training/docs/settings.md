# Settings

Every setting the rows run under, where it is set, and why it is what it is. Each is a parameter: an environment
variable of the script that starts the run, or a constant of the module named. Terms are in the [glossary](glossary.md).

## The protocol

| Setting | Value | Where | Why |
| --- | --- | --- | --- |
| Stream | `arc1-small`, workstreams order, seed 0 | `play.STREAM_CONFIG`, `play.SEED` | The benchmark's canonical stream for the trained rows. |
| Users | 1, 2 or 4 | `realtime.sh` `USERS` | Users 0..N-1 of any larger number are exactly N users' own streams (`play.served`), so runs at 1, 2 and 4 users and one reference play over users 0-3 share every user's instances. |
| Rounds | 200 | `ROUNDS` | The stream itself; the schedule's epilogue that follows is played only by a run given more rounds, and no measure reads it. |
| Lag | 1 | `LAG` | Round r plays on the model trained on rounds 0..r-1-LAG, so the hardware decides how long a run takes but never which model plays a round, while play and training overlap. `LAG=0` is the strict synchronous protocol. |
| Frozen tests | held-out and seen, one round of 100 users each | `frozen.TESTS`, `schedules/` | The benchmark's own (`configs.test_schedule`), kept with the package so every run plays the same instances. |

## The harness and the model's sampling

Every harness row, trained or reference, plays prime-agent under the same settings (`play.HARNESS`, `play.BLOCK`).

| Setting | Value | Why |
| --- | --- | --- |
| End of an instance | the finish protocol (`end: finish`) | The harness saves what it wants after the verdict, then finishes. |
| An empty reply | asked again in the same conversation (`empty_reply: reprompt`) | A restart loses the conversation. |
| Compaction | at 49,152 tokens, keeping 8,000 | On the untrained Qwen3.5-9B, 48 games an arm, compacting at 48k solved 16 against 10 at 32k, and every call stays inside the trainer's 65,536 tokens. |
| Action budget | 40 model calls; 1,740 s an action and its follow-up; 6 calls in the follow-up | The call cap is meant to be the only thing that routinely stops a turn: at the harness's 1,800 s hard ceiling the ceiling would fire first and restart the turn instead of the budget's "decide now". |
| Window and replies | 131,072-token window, 32,768-token replies | The model's context, and room for long reasoning. |
| Sampling | temperature 1, top-p 1, no top-k | The model's own distribution, which every update trains on as played. |
| Thinking | on (`enable_thinking`) | The model card's thinking mode, at every call. |
| The bare model | the floors' settings: the finish protocol, empty replies asked again, wipes archived (`play.FLOOR`) | The bare-model rows are the benchmark's floors. |

## The model server

`lib.sh`'s `SERVE_ARGS`, used by `serve.sh` (the stream's server on :8101) and `frozen.sh` (the tests' servers).

| Setting | Value | Why |
| --- | --- | --- |
| vLLM | 0.30.0, in an environment of its own | It runs beside the trainer's pinned stack without changing it. |
| Parsers | `qwen3` reasoning, `qwen3_coder` tool calls | The model card's. |
| Content format | `string` | The chat template is given each message's text as one string; the local rendering (`calls.render_prompt_ids`) normalises messages the same way, so a prompt rendered for training is the one the server rendered. |
| `--language-model-only` | on | The checkpoint is multimodal; the rows use text alone. |
| Speculative decoding | the checkpoint's MTP head, 2 tokens a step | Rejection sampling keeps every sampled token's distribution the target's, adapter and all; with a trained adapter on four GPUs tensor-parallel, calls decoded 1.84 times faster one at a time and 1.53 times four at a time, greedy text unchanged. |
| Adapters | loaded at run time, rank up to 128, 12 at once, one API frontend | Runs that share a server each play on one while OPSD publishes two of its student's; fewer slots would swap them request by request. vLLM 0.30 refuses run-time loading with more than one frontend. |
| `SERVE_DP` | 1: one copy tensor-parallel over `SERVE_GPUS` | The runs keep few calls in flight, so a call's own decode speed sets the pace of play: four H100s as one copy decoded about twice as fast as four replicas. |
| An adapter's folder | a copy under the checkpoint's names (`server.served_adapter`) | vLLM serves Qwen3.5 through its multimodal class, whose names PEFT's text-only adapter does not match; a misnamed adapter is served silently as the base model. |
| The served-adapter check | temperature 0, 16 tokens, top 5 log-probabilities, a gap of at least 0.01 | Qwen3.5 opens with its reasoning, so 16 tokens leave the visible text empty for any weights; the base model asked twice gave a largest gap of exactly 0, an adapter after 8 small steps 0.375 with the same tokens. An adapter no step has touched (every LoRA B zero) is the base model and is not checked. |

## The oracle and the labels (DAgger and OPSD)

| Setting | Value | Where | Why |
| --- | --- | --- | --- |
| Oracle | GLM-5.3 Flash at medium effort, thinking on | `realtime.sh` `LABEL_MODEL`, `LABEL_EFFORT` | A much stronger model than the student, at a cost a whole run can afford. |
| Its hosts | gmicloud, parasail, fireworks, no fallback | `RELAY_PROVIDERS` | The relay pins every request to the one model and these hosts. |
| Working rules | c1 and the program library's procedure (`ORACLE_RULES=library`), or c1 alone (`c1`) | `oracle.oracle_rules` | The procedure reaches the student's weights through the labels alone, since the student never reads it; c1 alone makes the procedure the one difference between two oracles. |
| Where the rules go | the end of the call's last message, closed by a request to reason in its own words with the rules unspoken | `oracle.with_rules` | On one round's 380 moments this labelled 89% against 79% for the rules alone, and 45% with them in the system prompt, where the oracle mostly stopped reasoning. |
| Asked again | with a note saying what was wrong, as if for the first time | `oracle.ask_label` | Told what was wrong, the oracle labelled 96% of the same moments against 89% when merely asked again. |
| A label that names the rules or the note | asked again up to 3 times (`RULE_RETRIES`), then dropped | `oracle.MENTIONS_RULES`, `NOTE_LEAK` | The reply becomes the student's target, and the student never sees the rules. |
| A label with no reasoning, or no move | asked again up to 4 times (`NEED_THINKING`), then dropped | `oracle.THINK_MIN_CHARS` (50) | A student trained on labels with no reasoning stopped thinking; a reply that only says what it will do teaches the student to spend a turn on nothing. On obvious moves the oracle reasons in 9 or 10 tokens and explains itself in its reply, so its reasoning and its visible prose count together. |
| The student's target | all of the oracle's prose as the student's reasoning, only the action and tool calls in view | `labeller.student_message` | The student reasons in its thinking and acts in the text the harness reads; a student trained with the prose left in view wrote prose with no move in 16% of its replies, against 6%. |
| A round's labels | at least 40% of its points labelled (`MIN_LABELLED_PCT`), at most $20 a round and $3,000 a run | `realtime.sh` | A round under the minimum means a dead relay or key, or an oracle that will not think. |
| Reply cap and timeout | 32,768 tokens, 420 s, one retry | `oracle.MAX_TOKENS`, `TIMEOUT_S` | The budget covers the reasoning too (one moment took 15,000 tokens of it); the timeout sits under prime-agent's 600 s with room for one retry. |
| Held out | one game in 10 (`HOLD_EVERY`) | `labeller.held_game` | Alone, every 10th round's game from round 9; with 10 users or more, a different user's game each round, so no user's own stream is held out for good. |

## Training: one setup for every row

| Setting | Value | Where | Why |
| --- | --- | --- | --- |
| Warm-up | no phase trains until rounds 0..k hold 24 games (`START_AFTER=24`) | `realtime.warming_up` | A first phase on one round's four games could derail the student (82-97% of its replies made a move); on 24 games it moved as often as the base model, with no empty, looping or garbled reply. |
| Adapter | LoRA rank 128, alpha 128, on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | `trainer.LORA_TARGETS`, `--lora-rank` | The attention layers and every MLP; Qwen3.5's Gated DeltaNet projections are named otherwise, so their token mixing stays frozen. |
| Learning rate | 1e-4 (`LR`), AdamW at torch's defaults, the gradient clipped to norm 1 | `trainer.Trainer.step` | A step divides the summed gradient by the replies it read, so every reply weighs alike. |
| Batch | 64 calls a step (`BATCH`) | `--batch` | |
| Window | 65,536 tokens a chain piece | `--max-seq-len` | A chain that the window cannot hold is cut at call boundaries; a call longer than the window is left out and counted. |
| Chains | a head of at most 8 ids (`CHAIN_SLACK`), at most 16 calls a piece (`CHAIN_CALLS`) | `chains` | A game's calls whose prompts each extend the last are read in one pass. A prompt that differs from the trunk in its last few ids (an empty think block re-rendered) stays in the chain with those ids as an ungraded head. The piece cap keeps a step's plan the same on any number of GPUs. |
| The held-out stop | measured before the first step and every 5 steps (`EVAL_STEPS`), stopped after 2 measurements in a row without a new best (`PATIENCE`), the best adapter kept | `phase.fit` | Each phase keeps the adapter that generalised best on held-out games, its starting one while none improves on it. |
| Caps | at most 120 steps (`MAX_STEPS`), 20 passes (`--max-epochs`), 240 minutes (`--max-minutes`); one pass while there is no held-out game | `realtime.dose` | With nothing held out there is nothing to stop on, so a phase reads its data once. |
| Held-out set | whole held-out games drawn until their calls reach 1,000 (`--val-max-items`) | `realtime.whole_games` | The held-out sides grow every round; the cost of a measurement does not. |
| Seed | phase k plans its steps from `random.Random(k)` | `realtime.train_phase` | Every row draws the same steps from the same units; OPSD's steps are DAgger's. |

### The dose

`DOSE=aggregate` (the default) trains each phase on every round's training side so far, DAgger's own aggregate.
`DOSE=recent` trains on the last 10 rounds' (`realtime.RECENT_ROUNDS`) and `REPLAY_RATIO` (1) times as many labels from
whole earlier games drawn at random, under the same stop. In long DAgger runs most phases kept the adapter they started
from: the aggregate dose reshuffles every earlier round in every phase, so a phase stopped at step 10 reads almost only
rounds the adapter has fitted already, and the held-out loss rose a little in the first steps before falling. Played
past its stop on a stalled phase, the recent mix lowered the loss on the held-out set, on the next rounds' states and on
the recent rounds steadily from step 10 (by 0.8%, 1.3% and 11.2% by step 40, against 0.1%, 0.5% and 3.5% for the
aggregate). So DAgger runs take `DOSE=recent PATIENCE=4` once their phases stall; OPSD and PG train on the aggregate.

### Each update

| Update | Setting | Value | Why |
| --- | --- | --- | --- |
| all | a reply's loss | its summed token losses over 1,024 (`trainer.PG_TOKEN_SCALE`) | Every token weighs alike and a reply is not pushed harder for being short (Dr. GRPO's normaliser, about a reply's length: median 566 tokens, 90th percentile 2,068); with the optimizer's normalisation it only decides when the norm clip binds. |
| DAgger | target | the label set's reply after its prompt as the local template renders it, through its end of turn | `labeller.render_reply` renders the reply on its own after the prompt's text, as the model writes it after the served prompt. |
| OPSD | the teacher's prompt | the call's prompt as served, with SDFT's demonstration at the end of its last message (`updates.SDFT_DEMO`), put in by position | The served ids can split an earlier turn otherwise than a fresh rendering would, so the two are never compared whole (`updates.splice_note`). |
| OPSD | the loss | KL(teacher ‖ student) over the full vocabulary at each reply position but the first 3 (`SKIP_FIRST`), averaged over the reply, each reply weighing alike | SDFT's loss and reduction (its code leaves the first 3 ids ungraded). |
| OPSD | the student's replies | drawn fresh from the server at each step, capped at 8,192 tokens, the student published after every step and at most one step stale | Replies run to a few thousand tokens, and the longest of a batch sets how long the batch takes to draw. |
| PG | reward and advantage | 1 for a win, 0 for a failure, no baseline | With wins as rare as the stream's, any mean of the run's games sits above every failure, so a baseline would push down every token a failure played, the tool calls' own format with them; a cost-aware reward pays a fast, broken game as well as any failure. |
| PG | the ratio | PPO's clip at 0.2 against the adapter that played the round, dual-clipped at 3 under a negative advantage | Each win trains through the clip, so passing over it again every phase does not memorise it. The dual clip silences a token whose ratio ran away; one step of 22,891 tokens once averaged a ratio of 21.3, nearly all of its update from a few tokens. |
| PG | the held-out stop | PG's own loss over the held-out games, the held-out wins' negative log-likelihood logged beside it | A held-out failure weighs 0, so the stop runs once a held-out game has won. |

### On the GPU

| Setting | Value | Why |
| --- | --- | --- |
| A chain's attention | FlexAttention under the chain's block mask, built a band of query blocks at a time; FlashAttention's CuTe kernels where installed, else Triton's FlexAttention kernel named outright (`kernels`) | One pass reads a game's trunk once and each reply after its own prompt. create_block_mask holds the whole mask at 11 to 12 bytes a query-key pair, some 37 GB at a 57,344-token window. FlexAttention's own choice takes its decoding kernel for a query under 128 tokens, which has no backward. |
| The linear-attention layers | read through the chain's branch plan (`branching`) | Gated DeltaNet reads its input in order with no mask, so each branch starts from the trunk's memory at its point. |
| The adapter's matmuls | bfloat16, float32 master weights (`kernels.bf16_lora`) | On an H200 a step at 57k tokens ran 1.23 times faster; the update's cosine with PEFT's float32 one was 0.99986, against 0.99985 for PEFT's own repeat. |
| Full-vocabulary quantities | computed 1,024 positions at a time from the final hidden states, checkpointed per chunk | Full logits never exist for a whole reply. |
| Gradient checkpointing | on (`Trainer(grad_checkpointing=True)`) | Each layer's activations are recomputed in the backward, with the chain's branch plan active again (`branching.active`). |
| Data parallelism | every GPU holds the model and the adapter, chains dealt by estimated cost, gradients summed once a step, every rank checked to hold the same adapter after it (`parallel`) | Qwen3.5-9B fits whole on each GPU. |
| Step checkpoints | on SIGTERM and every 25 steps (`checkpoints.EVERY`) | A save holds training up about 4 s; a preemption gives 60 s of notice, and a stop with none loses at most 25 steps. |
| Compiled kernels | cached in `~/.cache/torchinductor` (`phase.INDUCTOR_CACHE`) | A machine that restarts keeps its boot disk but clears `/tmp`: a trainer whose cache was cleared spent 203, 106 and 59 s on its first three steps compiling, against 25, 25 and 18 s from the cache. |
| Pruning | phases older than the newest lag + 1 lose their resume files and vLLM copy, all but every 20th lose their adapter; DONE, phase.json and rounds.json stay | A phase's adapter and optimizer are 4.4 GB on Qwen3.5-9B at rank 128. |

## Training once at the end

`scratch` trains a DAgger run's phase K from the base model with the run's own trainer settings
(`scratch.TRAINER_ARGV`), so only the start and the stop differ from the continual student's phase K: the held-out loss
every 25 steps instead of 5, no step cap, at most 720 minutes, since a fresh adapter needs far more steps than a
warm-started phase.

## The reference rows' players

| Player | Model | Settings | Why |
| --- | --- | --- | --- |
| glm | GLM-5.3 Flash | medium effort, the provider's own sampling, on Z.ai then Fireworks, no fallback | The oracle playing the games itself. GMICloud's replies to prompts it has cached run to the length cap or come back empty. |
| qwen | Qwen3.5-9B | temperature 1, no top-p or top-k cutoff, thinking on; Together, then Venice, then DeepInfra, no fallback | The untrained base model. Together and Venice answered a call in about 3 s; DeepInfra was some ten times slower and refused calls as rate-limited. |

Both send a harness's developer prompt as a system one and every earlier reply's reasoning under OpenRouter's
`reasoning` key, and drop prime-agent's own effort, so the player's setting is the one that applies.

## The software stack

The trainer runs in the node's prime-rl environment at commit `a1822f7`:

| Part | Version |
| --- | --- |
| GPUs | 8 × H100 (Hopper) |
| torch | 2.13.0 (CUDA 13.0), which requires Triton 3.7.1 |
| Triton | 3.7.1 |
| flash-linear-attention | 0.5.2 |
| tilelang | 0.1.12 |
| transformers | 5.6.2 |
| peft | unpinned; the CPU tests pass with 0.21.2 |
| vLLM (the server only) | 0.30.0 with torch 2.13.0, in `~/vllm-0.30.0` |

Qwen3.5-9B has 24 Gated DeltaNet layers out of 32, and transformers trains them through flash-linear-attention (fla)
whenever fla is installed and a GPU is present. On Hopper GPUs, Triton 3.4.0 up to 3.7.0 compiles one fla kernel
wrongly: the gated delta rule's backward step for dq, dk, dw and dg (`chunk_bwd_dqkwg`), whose dk and dg come out wrong
(fla issue #640, fixed in Triton 3.7.1, fla #983). fla 0.5.0 and later send that step to a TileLang kernel when
tilelang is installed, and raise an error instead of running the broken kernel otherwise; fla before 0.5.0 runs it
without a word. The stack above is clear on both counts.

Every run's train process writes `train/kernels.json` (the package versions, the GPU and the path that step takes) and
stops before loading the model on a stack known to compute wrong gradients (`stack.record`). Before training anywhere
new, run `python -m continual_arc_training.stack` there and read its `path`. Without causal-conv1d installed, training
logs say the fast path is not available: that is only the layers' short convolution; the gated delta rule itself still
goes to fla.
