# Glossary

The words used across this folder, in plain terms. Grouped by topic; each group is in reading order. The benchmark's
own terms are defined in full in the [benchmark's README](../../README.md).

## The benchmark

| Word | Meaning |
| --- | --- |
| **Continual-ARC** | The benchmark. A learner solves a stream of ARC puzzles (small coloured grids with a hidden rule) and is charged for what it spends. |
| **Task** | One hidden grid rule. |
| **Instance** | One freshly generated puzzle for a task (a new test input). A task can have many instances. |
| **Game** | One instance played from start to end. |
| **User** (`u000`, `u001`, ...) | One player slot in a schedule. In the paper's streams a user plays one instance a round and keeps its own harness memory from round to round; in a one-off play of 100 puzzles, "100 users" means 100 games. |
| **Schedule** | The fixed list of instances a play goes through, round by round. |
| **Stream** | `arc1-small`'s 200 rounds over its 25 tasks (workstreams order, seed 0), served to 1, 2 or 4 users in the paper's rows. |
| **Demo** | An example input/output pair for the current task. The learner asks for one with `request_demos`; each costs 1. |
| **Submit** | The learner sends its answer grid. |
| **Verdict** | The environment's reply to a submission: CORRECT or INCORRECT. It never shows the right grid. |
| **Give-up** | The game ends as failed. The harness forces one after 3 replies with no usable action. |
| **Cost** | What a game spends: a solved instance costs its demo requests and wrong submissions; an unsolved one costs the whole budget, 16, however it failed. Lower is better. |
| **Frozen tests** | The two tests after the stream's 200 rounds, each one round of 100 users played at once by the system as the stream left it, its model and memory, with nothing learnt during the test and nothing passed from one test instance to another: the **held-out test**, one instance of each of 100 **held-out tasks** (ARC-AGI-1 evaluation tasks the stream never uses), and the **seen test**, four new instances of each of the stream's 25 tasks. |
| **Transfer (T)** | The benchmark's headline: the share of the budget a run saves on a frozen test's instances against the learner reset (row H0) on the same instances. Positive is better. |
| **Share learned** | How much of the cost a learner could save over the stream it actually saves, read against the learner reset: 0 is no better than the reset, 1 is every returning task answered free. |
| **Epilogue** | 100 rounds after the stream in which every user meets the held-out tasks while the system goes on learning. The schedule still holds it, but it was retired for the frozen tests and no measure reads it; the held-out test's instances are its user 0's. |
| **Demo-cache floor (DC) and stateless floor (F)** | The bare model behind the benchmark's protocol: DC with the demos it bought for a task shown again free when the task returns, F with nothing kept. They need no learning. |
| **Solves** | How many games out of a set end with a correct submission. |
| **prime-agent** | The harness every row plays in: it gives the model a Python sandbox and tools, keeps its own memory, and talks to the environment. |
| **Budget stop / "Decide now"** | The benchmark's harness settings stop a long action (40 model calls, or a time limit) and tell the model to act now. |
| **Compaction** | prime-agent summarises a long conversation to save space; the trained rows compact at 48k tokens, keeping 8k. |
| **Resume from the last call** | A round cut short played again without asking the model again for the calls it had already made: every call is kept as it ends (the round's **journal**), each user's memory is put back as the round began, and each game is **replayed** (its recorded replies served again) to its last call, then **handed over** to the model. A replay that meets a request unlike the recording's has **drifted**, and the model plays on from there. Off for the paper's runs. |
| **Session log / loader** | The conversation is saved to a file. The *loader* is code in the rules (c1's rule 4) that reads the demo grids from that file, so the model does not retype them by hand. |
| **State log** | Every user's harness memory at the end of every round, kept as one git commit in the user's `system/state.git` and listed by round in `system/state-log.jsonl` (the baselines' `CollectiveHarnessLearner`), so `git diff HEAD~1 HEAD` there is what the user's harness wrote in the latest round. A frozen test's user starts from its stream user's commit of the stream's last round and carries that history on. |
| **Aliases** | Hidden per-run task names (from a run's `alias-salt`). Two plays with the same aliases see identical prompts, so they can be compared puzzle by puzzle. |
| **Validation puzzles** | 100 ARC-AGI-1 evaluation tasks that are neither stream nor held-out tasks, one instance each ([schedules/validation](../schedules/validation)), used to choose between students in the earlier experiments and by `grid-vals.sh`. |

## Rows

| Word | Meaning |
| --- | --- |
| **Row** | One learner of the paper's tables: a model, a harness setting and, for the trained rows, an update. |
| **H, H0** | Qwen3.5-9B (or GLM-5.3 Flash) in prime-agent, untrained: H keeps the harness's memory, H0 wipes it before every instance (the learner reset). |
| **HPL, HPL0, HA** | H told the program library's procedure; HPL wiped before every instance; H asked after every verdict to save what could help later. |
| **HG, HPLG, HGPL** | The storage turn after each verdict driven by the teacher instead of the student: HG plain, HPLG with both told the procedure, HGPL with the teacher alone told it. |
| **M, M0** | The program library on the bare model: every submission a program, the last that reproduces all of a task's pairs kept and run first when the task returns; M0 wiped before every instance. |
| **DAgger, OPSD, PG** | The trained rows: a LoRA adapter of Qwen3.5-9B trained between rounds with a one-round lag (below). |
| **Program library's procedure** | The habit row HPL's rules tell a learner (the baselines' `PROGRAM_LIBRARY_RULES`): keep one Python program per task in `programs/`, run it first when the task comes back, and save or fix it after the verdict. DAgger's and OPSD's oracle labels under c1 and this procedure (`ORACLE_RULES=library`); the student is never told it. |

## Models

| Word | Meaning |
| --- | --- |
| **Base model** | The untrained student, **Qwen3.5-9B**. |
| **Adapter (LoRA)** | A small set of trained weights added on top of the base model (rank 128). Every trained student is the base model plus an adapter. |
| **Student** | The model being trained. |
| **Oracle / expert** | A much stronger outside model, **GLM-5.3 Flash** at medium effort, which labels the student's moves for DAgger and OPSD. |
| **Teacher** | The model whose behaviour the student learns to copy. For OPSD, the frozen base model reading the oracle's label as a demonstration. In the earlier experiments, the base model with extra text in its prompt. |
| **b6, the round-1 model** | Students of the earlier offline experiments (Sep-Oct 2026), before the paper's rows: b6 is Qwen3.5-9B trained to play without c1 in its prompt as it plays with it (context distillation; 26 to 30 of those experiments' 100 test puzzles in three plays, against the base model's 10 to 11), and the round-1 model is b6 trained on the oracle's reply at every step of 296 games (59 of the 100 validation puzzles). No paper row starts from them. |
| **c1** | The playing rules: 6 rules, 287 words ([rules/c1.md](../rules/c1.md)), the oracle's working rules in the paper's rows. **c1e** ([rules/c1e.md](../rules/c1e.md)) replaces its first two rules with facts about the harness's limits; the paper's rows do not use it. |

## Training methods

| Word | Meaning |
| --- | --- |
| **On-policy** | Training on states the current student itself reached, or replies it just wrote. |
| **DAgger** | The student plays; the oracle writes its own reply to every state the student met; the student trains on those replies by supervised fine-tuning; repeat. The paper's DAgger aggregates every round's labels so far, warm-starting each phase from the last. |
| **OPSD** (on-policy self-distillation) | DAgger's data and schedule with SDFT's loss: the student writes a reply, the frozen base model reading the oracle's label as a demonstration scores every token of it, and the student moves toward the teacher's distribution (forward KL). |
| **SDFT** | Self-distillation fine-tuning, the published method whose loss OPSD uses. |
| **PG** | Policy gradient from the outcome only: REINFORCE with no baseline, a win's reward 1 and a failure's 0, PPO's clipped surrogate against the adapter that played the round, dual-clipped. A failed game pushes nothing. |
| **OPD** | On-policy distillation from a separate teacher's per-token scores (`train.py --objective opd`); not a row of the paper. |
| **Phase** | One training of the adapter, after one round: phase k trains on rounds 0..k, from phase k-1's adapter, until its held-out loss stops improving. |
| **Lag** | How many rounds behind play the playing model is: round r is played by phase r-1-LAG; the paper's rows run at a lag of 1. |
| **Warm-up (`START_AFTER`)** | The games the base model plays before the first phase trains: 24 in the paper's rows. |
| **Dose** | What a phase trains on: `aggregate`, every round's data so far; `recent`, the last 10 rounds' and as many from whole earlier games; `replay`, a fixed new-plus-replay amount. |
| **Held-out stop** | One game in ten held out of training (`HOLD_EVERY`), its loss measured every 5 steps (`EVAL_STEPS`), the phase stopped once 2 measurements in a row (`PATIENCE`) have not improved on the best, the best adapter kept. |
| **Label / point** | A moment in a game (one player call) and the oracle's reply there. |
| **Hint / thought** | Help for one exact moment, in the earlier experiments: a *hint* is shown as a note; a *thought* is the same text placed as the opening of the model's own reasoning (`qwen35_prefill.jinja`). |
| **Rule card** | A written statement of one task's rule, with a Python program checked on fresh examples (`overfit_cards.json`), used by the earlier overfit tests. |

## Reading results

| Word | Meaning |
| --- | --- |
| **Play** | One run of a model over a schedule. |
| **Paired comparison** | Two plays of the same puzzles, compared puzzle by puzzle ("8 solved only by A, 11 only by B"). |
| **McNemar p** | The standard test for a paired comparison. Small p (below about 0.05) means the difference is unlikely to be luck. |
| **Noise** | Two plays of the same model differ by several solves. A difference under about 6 solves out of 100 is not clear. |
| **Resend** | Submitting a grid that was already judged INCORRECT. |
| **KL / divergence** | How different the student's next-token choices are from the teacher's. Lower means closer. A falling KL does not by itself mean more solves. |

## The training software

| Word | Meaning |
| --- | --- |
| **ask-sd** | unify's trainer package (`ask_sd`): the `Trainer` (LoRA through peft, data-parallel workers, the SFT, self-distillation and policy-gradient losses), the call format and the vLLM client. Not in this repository. |
| **prime-rl** | The environment the trainer and the loops run in on a node, pinned at commit `a1822f7`. |
| **Chains** | A game's calls whose prompts each extend the one before, read as one sequence with each call's reply as a branch at its prompt's end (`train.py`, `ask_sd.branching`), so a game is read once rather than once per call. |
| **Linear attention (Gated DeltaNet)** | The kind of layer that makes up 24 of Qwen3.5-9B's 32 layers. It keeps a fixed-size memory of the text so far instead of looking back at every earlier token. |
| **fla (flash-linear-attention)** | The library of fast GPU code that trains the linear-attention layers. |
| **Triton** | The compiler that turns fla's code into GPU code. Triton 3.4.0 to 3.7.0 gets one of fla's training steps wrong on Hopper GPUs ([fla-hopper-gradients.md](fla-hopper-gradients.md)). |
| **TileLang** | A second GPU compiler. fla uses it for the step Triton gets wrong. |
| **Hopper** | NVIDIA's GPU generation of the H100, H200 and H20. |
| **kernels.json** | Written by `train.py` into each training folder: the package versions, the GPU, and which code trained the linear-attention layers. |
| **Preemptible machine / preemption** | A machine rented at a spot price (8 x H100 at $6.32 an hour, against about $36 on demand) that the provider may stop when it needs the GPUs back: a *preemption*. It warns the machine 60 s before (SIGTERM), keeps its disk, and does not start it again itself. |
| **Step checkpoint** | A training phase's saved state (the adapter, the optimizer, where it is in its batches, its held-out curve), written at a preemption's SIGTERM and every `checkpoint.EVERY` steps, so a phase stopped by a preemption or a crash goes on from there instead of starting again (`checkpoint.py`). The paper's nine runs trained without it. |
| **Trainer pool / share** | The GPUs that the trainers of runs sharing a node take turns on (`realtime.sh`'s `TRAIN_GPUS`: runs given the same list share it). Each run with a phase to train gets an even *share*; a phase gives GPUs up to another run's phase at its next step checkpoint and takes more when its next phase starts (`gpupool.py`). A run alone on its GPUs holds all of them. |
| **Layout** | Which runs share a node, how many of its GPUs serve the model and how many train (`fleet.py plan`): its plans run from the fastest to the cheapest, and one is picked by a deadline or a budget. |
| **Kernel caches** | The GPU code compiled and tuned for a run's shapes, kept on disk so a restarted trainer need not compile it again: torch's (`~/.cache/torchinductor`), fla's tuning (`~/.triton/cache`), TileLang's (`~/.tilelang`) and the model server's (`~/.cache/vllm`). |
