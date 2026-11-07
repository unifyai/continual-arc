# Glossary

The words used across this package, in plain terms, grouped by topic. The benchmark's own terms are defined in full in
the benchmark's README at the repository root.

## The benchmark

| Word | Meaning |
| --- | --- |
| **Continual-ARC** | The benchmark. A learner solves a stream of ARC puzzles (small coloured grids with a hidden rule) and is charged for what it spends. |
| **Task** | One hidden grid rule. |
| **Instance** | One freshly generated puzzle for a task (a new test input). A task can have many instances. |
| **Game** | One instance played from start to end. |
| **User** (`u000`, `u001`, ...) | One player slot in a schedule. A user plays one instance a round, from the one harness memory every user shares (see State log). |
| **Schedule** | The fixed list of instances a play goes through, round by round. |
| **Stream** | `arc1`'s 200 rounds (workstreams order, seed 0), served to 1, 2 or 4 users, each over 25 tasks of their own. |
| **Epilogue** | 100 rounds after the stream in the schedule, in which every user meets the 100 held-out tasks once each. No measure reads it; the held-out test's instances are its user 0's. |
| **Demo** | An example input/output pair for the current task. The learner asks for one with `request_demos`; each costs 1. |
| **Submit** | The learner sends its answer grid. |
| **Verdict** | The environment's reply to a submission: CORRECT or INCORRECT. It never shows the right grid. |
| **Cost** | What a game spends: a solved instance costs its demo requests and wrong submissions; an unsolved one costs the whole budget, 16, however it failed. Lower is better. |
| **Frozen tests** | The two tests after the stream, each one round of 100 users played at once by the system as the stream left it, its model and memory, nothing learnt during the test and nothing passed from one test instance to another: the **held-out test**, one instance of each of 100 **held-out tasks** (ARC-AGI-1 evaluation tasks the stream never uses), and the **seen test**, four new instances of each of the stream's 25 tasks at one user, and at more the same 100 spread over every user's own tasks, one of each task once they bring more than 100. |
| **Transfer** | The benchmark's headline: the share of the budget a run saves on a frozen test's instances against the learner reset (row H0) on the same instances. |
| **Aliases** | Per-run task names (from a run's `alias-salt`), by which a harness's memory names tasks. Two plays with the same aliases see identical prompts. |
| **State log** | The harness memory at the end of every round, one git commit in user 0's `system/state.git`, listed by round in `system/state-log.jsonl` (the baselines' `CollectiveHarnessLearner`). At several users the memory is one for all of them: every user's game of a round starts from a copy of it, and what every user wrote is merged into it as the round ends, each round's merge recorded in `system/state-merges.jsonl`. |

## The harness

| Word | Meaning |
| --- | --- |
| **prime-agent** | The harness every harness row plays in: it gives the model a Python sandbox and tools, keeps its own memory, and talks to the environment. |
| **Finish protocol** | The harness's end of an instance: after the verdict it takes a turn to save what it wants, then finishes. |
| **Action budget / "Decide now"** | The harness stops a long action (40 model calls, or a time limit) and tells the model to act now. |
| **Compaction** | prime-agent summarises a long conversation to save space: at 48k tokens, keeping 8k. Its summarizer's calls are never trained on. |
| **Grid loader** | The shared guidance's loader reads grids from the current Prime Agent session log; other harnesses use their own conversation access. |
| **Expert guidance (E)** | One text, `baselines/.../systems/guidance.py:EXPERT_GUIDANCE`, shared by the DAgger teacher and guided harnesses: check demonstrations, retain programs and pairs, reuse verified programs, and save after the verdict. The student receives it only through teacher labels. |
| **Program library** | An implementation of the policy: keep a Python program and demonstrations per task, check the program against the demonstrations, and run it first when the task returns. |

## Rows

| Word | Meaning |
| --- | --- |
| **Row** | One learner of the results: a model, a harness setting and, for the trained rows, an update. |
| **H, H0** | The model in prime-agent, untrained: H keeps the harness's memory, H0 wipes it before every instance (the learner reset). |
| **HPL, HPL0, HA** | H given expert guidance; HPL wiped before every instance; H asked after every verdict to save what could help later. |
| **HG, HPLG, HGPL** | Qwen plays each instance from the state GLM held before it: HG without guidance, HPLG with both given E, HGPL with only GLM given E. |
| **DC, F** | The bare model behind the benchmark's protocol: DC (the demo-cache floor) with the demos bought for a task shown again when the task returns, one cache for the team, as every row's memory is one; F (the stateless floor, `bare`) with nothing kept. |
| **M, M0** | The program library on the bare model: every submission a program, the last that reproduces all of a task's pairs kept and run first when the task returns; M0 wiped before every instance. |
| **DAgger, OPSD, PG** | The trained rows: a LoRA adapter of Qwen3.5-9B trained between rounds with a one-round lag. |

## Models

| Word | Meaning |
| --- | --- |
| **Base model** | The untrained student, Qwen3.5-9B. |
| **Adapter (LoRA)** | A small set of trained weights added on top of the base model (rank 128). Every trained student is the base model plus an adapter. |
| **Student** | The model being trained. |
| **Oracle** | A much stronger outside model, GLM-5.3 Flash at medium effort, which labels the student's moves for DAgger and OPSD. |
| **Teacher** | For OPSD, the frozen base model reading the oracle's label as a demonstration, whose distribution the student moves toward. |
| **Player, storer** | A reference row's model (GLM-5.3 Flash or Qwen3.5-9B on OpenRouter); for HG, HPLG and HGPL, the model that takes the turn after each verdict. |

## Training

| Word | Meaning |
| --- | --- |
| **Phase** | One training of the adapter, after one round: phase k trains on rounds 0..k, from phase k-1's adapter, until its held-out loss stops improving. |
| **Lag** | How many rounds behind play the playing model is: round r is played by phase r-1-LAG. |
| **Warm-up** | The games the base model plays before the first phase trains (`START_AFTER`, 24). |
| **Dose** | What a phase trains on: `aggregate`, every round's so far; `recent`, the last 10 rounds' and as many from whole earlier games. |
| **Held-out stop** | One game in ten held out of training, its loss measured every 5 steps, the phase stopped once 6 measurements in a row have not improved on the best, the best adapter kept. |
| **Point / label** | A moment in a game (one call the student made as the agent) and the oracle's reply there. |
| **Label set** | A round's labels as training targets: each rendered as the student writes it after its prompt, a game of labels a file. |
| **DAgger** | The student plays; the oracle writes its own reply at every state the student met; the student is fine-tuned on those replies. |
| **OPSD** (on-policy self-distillation) | DAgger's data and steps with SDFT's loss: the student writes a fresh reply, and moves toward the frozen base model reading the oracle's label as a demonstration (forward KL over the full vocabulary). |
| **SDFT** | Self-distillation fine-tuning, the published method whose loss OPSD uses. |
| **PG** | Policy gradient from the outcome alone: REINFORCE with no baseline, a win's reward 1 and a failure's 0, PPO's clipped surrogate against the adapter that played the round, dual-clipped. |
| **Behaviour adapter** | The adapter that played a round, whose token log-probabilities PG's ratios are taken against. |
| **Dual clip** | Under a negative advantage, a token's objective is at least 3 times the advantage, so a token whose ratio has run past 3 gives no gradient. |
| **Chain** | A game's calls whose prompts each extend the one before, read as one sequence, each call's reply a branch at its prompt's end, so a game is read once rather than once per call. |
| **Trunk, branch, head** | A chain's longest prompt; one call's reply, at its prompt's end on the trunk; the few ids of a prompt that leave the trunk at its end, read and not graded. |
| **Step checkpoint** | A phase's saved state (the adapter, the optimizer, the random state, its held-out curve), written at SIGTERM and every 25 steps, so a phase stopped by a preemption or a crash goes on from there. |
| **Preemptible machine** | A machine rented at a spot price that the provider may take back, with SIGTERM 60 s before; it keeps its disk. |
| **Trainer pool / share** | The GPUs the trainers of runs sharing a node take turns on (`TRAIN_GPUS`); each run with a phase to train gets an even share. |

## The training software

| Word | Meaning |
| --- | --- |
| **prime-rl** | The environment the trainer and the loops run in on a node, at commit `a1822f7`. |
| **Linear attention (Gated DeltaNet)** | The kind of layer that makes up 24 of Qwen3.5-9B's 32 layers: a fixed-size memory of the text so far instead of a look back at every earlier token. |
| **fla (flash-linear-attention)** | The library of GPU kernels that trains the linear-attention layers. |
| **Triton, TileLang** | GPU compilers. Triton 3.4.0 to 3.7.0 gets one of fla's backward steps wrong on Hopper GPUs; fla uses TileLang for that step when it is installed. |
| **Hopper** | NVIDIA's GPU generation of the H100, H200 and H20. |
| **FlexAttention** | torch's attention with a programmable mask, which reads a chain under its block mask on a GPU. |
| **kernels.json** | Written into a run's `train/`: the package versions, the GPU, and the path the linear-attention layers' backward step takes. |
