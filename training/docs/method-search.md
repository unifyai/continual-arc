# The training-method search

This page records the search, from 19 to 24 October 2026, that set how the paper's trained rows train: DAgger, OPSD
and PG on Qwen3.5-9B, each a LoRA adapter trained between the stream's rounds with a fixed one-round lag
([How a run works](how-a-run-works.md)). Each section gives a setting and the measurement behind it. A run's name is its
folder's: `dagger-u4-lag1` is DAgger at 4 users with a lag of one round, and a suffix marks a run of the search
(`-s24`: training starts after 24 games; `-pf`: the paused runs' own settings; `-agg`: every round's data in every
phase). The paper's appendix on the rows on the open models gives the settings this page arrives at.

## What the method must be

unify set the bounds of the search before it began:

- **A general learning algorithm with a few hyperparameters**, such as a number of games to play before training
  starts, never a pipeline of stages or anything fitted to the puzzles.
- **Every trained row starts from the base model.** The earlier adapters (b6 and the round-1 model of the earlier
  offline experiments, which practised the stream's own tasks) are evidence about what learns, never a starting point.
- **The levers:** how many users the stream serves, when in the stream training starts, how the labels split into
  training and held-out sets, and how overfitting is kept away. The protocol's other settings (time limits, the reply
  cap, the compaction point) are open too where they hurt how well the rows learn.
- **One setup for every row.** OPSD is DAgger with only the distillation step different. Every method trains on every
  game so far, so that it has as much data as the sparse stream gives at every step. PG's training is sized by a
  held-out loss too, never by a fixed dose.
- **The rows are baselines.** The aim is a method that learns well enough, not a new method, so a setting such as the
  warm-up goes in only if it makes a meaningful difference.

## Where the search started

The paper's nine trained runs (DAgger, OPSD and PG at 1, 2 and 4 users) were paused on 19 Oct after 10 to 44 rounds,
none of them learning: the trained models solved 41 of 321 games, the untrained model 6 of 42. DAgger's students asked
for 6 to 7 demonstrations a game, copying the oracle's per-state asks; PG at one user had stopped asking and had no
gradient left, and at two and four users was sliding the same way; OPSD was flat. DAgger's held-out loss moved 3% at
most (one held-out game, whose swings between checks exceed any gain), and 4 of its 7 phases with a held-out game kept
their starting adapter.

The evidence that this model can learn this game was the earlier offline DAgger: the round-1 model, which solves 59 of
the 100 validation puzzles against 27 to 30 for its starting point b6 and 16 for the base model, is b6 trained once on
17,449 labelled calls of 296 games (about 2 passes, 2,169 steps at batch 16, LR 5e-5; about 2,500 labels already gave
47 of 99). The paused rows started from the base model instead, had a few hundred to 2,600 labels by round 10, trained a
phase a round from the last adapter at batch 64 and LR 1e-4 with a stop on one held-out game's loss, and labelled under
c1 and the program library's procedure with reasoning required, which the offline labeller did not require.

The search was a bracket between that known recipe and the paper's setting, points near both ends tried at once and
the gap halved toward whichever side failed. Learning meant solves and cost above the untrained student playing the
same stream (each run's own warm-up rounds), beyond noise, by 10 to 20% of the stream; offline, solves on the 100
validation puzzles against the base model's 16.

## Offline: what the data and the starting point do

Each cell trained on a nested subset of the round-1 model's labels at its recipe (2 passes, LR 5e-5, batch 16), from
the base model or from b6, then played the 100 validation puzzles:

| cell | start | labels | training | validation solves (of 100) |
| --- | --- | --- | --- | --- |
| control | the round-1 model itself, retested | (17,449) | none | 59, as when it was made |
| `base-10pct` | base | 1,857 (22 games) | 234 steps | 44 |
| `base-25pct` | base | 4,427 | 554 steps | 41 |
| `base-50pct` | base | 8,845 | 2 passes | 49 |
| `base-100pct` | base | 17,449 | 2,182 steps | 60 of 99 ended |
| `b6-10pct` | b6 | 1,857 | 2 passes | 49 |
| `live-base-5500` | base | 5,577 of the paused DAgger runs' own labels (65 games of the stream) | 698 steps | 38 |

So 22 games' labels already take the base model from 16 to 44, b6 adds nothing once all the data is used, and the
stream's own labels teach the base model nearly as well (38). That set the warm-up's size below.

## The grid loader in the oracle's rules mixed instances

The first online runs with a warm-up trained the `base-10pct` recipe at round 6 of a 4-user stream (1,727 labels, 2
passes, 216 steps), yet their trained rounds 7-11 solved 1 of 16 games against 4 of 28 untrained, asking for 5.2
demonstrations a game and making 7.5 of the 8 wrong submissions allowed. Three checks on the validation puzzles split
the causes. `base-10pct`'s adapter at the stream's sampling (temperature 1, no top-p or top-k cutoff) solved 45, and
under the stream's `block` budget 46, against 44: the stream's settings were harmless. The online run's first trained
adapter solved 29 offline, and 34 after five more phases: the model was fine, and the stream broke it.

The trained games showed how. c1's rule 4 (`rules/c1.md`, which the oracle labels under) gave a grid loader that read
every log under `sessions/` oldest first and kept each grid's first sighting. The offline plays gave each puzzle a fresh
user and so one log; the stream keeps memory, so a user's folder holds every earlier instance's log too (13 to 16 by
round 12), and the loader gave every instance its user's first instance's grids. Run from the newest instance it
returned a 10x9 test input for a task whose test input is 3x20, and a trained student submitted that very grid. The
base model rarely calls the loader, so it was barely hurt; the trained students had learnt it from the oracle's replies,
and so worked on the wrong puzzle, asked for demonstrations again and again (the loader never showed them the ones they
bought) and gave up.

The fix reads this instance's own log alone, `sessions/<id>.jsonl` beside `RLM_SESSION_DIR`; `test_c1_loader.py` holds
an earlier instance's log beside the current one. Every run labelled under the old rules started over or had its rounds
labelled again. In a long instance the log keeps only what compaction kept, so the loader reads what the conversation
still shows, which is what the model can see too.

## The labeller

The oracle is GLM-5.3 Flash at medium effort. It reads each player call exactly as the player read it, with its working
rules (c1 and the program library's procedure) at the end of the last message, and its reasoning is required.

**The prompt.** The same 380 moments (one round of a 4-user run) labelled three ways:

| arm | labelled | on the first ask | no reasoning | mentioned the rules |
| --- | --- | --- | --- | --- |
| A: the rules at the end of the last message | 302 (79%) | 129 (34%) | 43 | 27 |
| B: A with a closing nudge to reason in its own words and keep the rules unspoken | 337 (89%) | 162 (43%) | 21 | 16 |
| C: the rules at the end of the system prompt instead | 172 (45%) | 62 (16%) | 184 | 12 |

B is the oracle's prompt (`forward.GUIDE_TAIL`). A re-ask told what was wrong (a note at the end of the last message)
instead of the same request again labelled 96% of the moments within 8 asks against 89% (73% against 62% within two),
and none of the 193 replies taken after a note mentioned it, so every ask after the first says what went wrong
(`forward.ask_label`). On obvious moves the oracle reasoned in 9 or 10 tokens, often explaining itself in the visible
reply instead, so a label counts as reasoned when its reasoning and its visible prose together hold 50 characters; the
student's target takes all of that prose as its reasoning and keeps only the action and any tool calls in view
(`dagger.student_message`). With all of it the same moments labelled 97%. Moving the prose matters: a student trained
with the oracle's prose left in view wrote prose with no move in 16% of its replies, against 6% with it moved. A label
that announces code and runs none becomes an empty reply as a target once its prose moves, and a run whose targets held
such labels had students replying empty on 5% of calls against 2.6%; so a reply that makes no move is asked again with
a note, and one that still makes none is left out of DAgger's targets and OPSD's demonstrations (`dagger.makes_move`).

**A reflex to ask for a demo, from the first phase's games.** When the final runs first started, the 4-user DAgger run
broke at its first trained round under both the old labeller and the new one (under the old one, 85 calls, 23 with
no valid action, 10 empty, code in 5%). An offline ablation (not included here) trained label sets as a run's first phase trains (LR 1e-4,
batch 64, one pass, from the base model) and asked every adapter the same 400 held-out moments of the base model's play;
it ranked the runs' own first-phase adapters as their play did (base model 98% of replies making a move, the broken
students 68-86%). Their replies deep in a game, after demos had been shown, read "I'm missing demo pairs ... Request
demos", copying the oracle's reasoning without the condition it rests on, and how often a first-phase student asks for a
demo follows how many of its mid-game targets are demo requests (69% of targets: 46% of replies; 51%: 42%; 25%: 24%).
That share came from the games, not the labeller: a long game that never asks for a demo gets "request a demo" from the
oracle at almost every later moment, rightly, under any labeller. On the same 380 moments ten labellers' students landed
at 82-96% moves, about as far apart as one labeller's students across rounds (82-97%), so no labeller stands out.

**A first phase of 24 games trains a sound student; one of four games may not.** One 4-user run's six rounds of the
base model's play, trained as a first phase trains:

| first phase | games | labels | moves | runs code | asks for a demo | empty |
| --- | --- | --- | --- | --- | --- | --- |
| one round, each of the six in turn | 4 | 222-515 | 82-97% | 51-75% | 11-32% | 1-5% |
| rounds 0-2 | 12 | 1,117 | 98% | 66% | 29% | 0% |
| rounds 0-5 | 24 | 2,011 | 98% | 67% | 25% | 0% |

The 24-game student moves as often as the base model, with no empty, looping or garbled reply, so 24 games keeps a
margin: the method should not derail because of what its first few games happen to contain. On the same 24 games the
new labeller trains as sound a student as the old one (100% moves against 98%) and asks less for each label (1.86 asks
against 2.21). So every row starts training once 24 games have been played (`START_AFTER=24`: the base model plays
until rounds 0..k hold 24 games, and the first phase trains on all of them), and the labelled rows use the new labeller.

## PG broke its own reply format, and its reward

PG first had a cost-aware reward (solved, less half the game's cost as a share of the budget) with a baseline, the mean
reward of the run's last 32 games. Its three runs broke in phases that had no held-out game and so no stop: from round 5
or 6 on, none of their 80 games solved, with 1.0 to 1.7 actions a game against 3.4 to 5.3 before, and every later phase
started from, and nearly always kept, a broken adapter. Three things in the update broke them:

- **Every failure was pushed down.** With a win in the window every failure's advantage is negative, so a phase lowers
  the probability of nearly every token the model played: the push down outweighed the push up 1.6 to 9.5 times in
  every phase before the break.
- **The tokens every game shares sank past the clip, for good.** The most extreme tokens were the tool call's own and
  grid digits, at a log-probability of about 0 when played and -8 to -28 after. PPO's clip stops pushing a token down
  only where it was sampled, the adapter carries the push to every other place the token appears, and nothing brings a
  collapsed token back, since a win's pull on it is its advantage times its ratio, about e^-20.
- **The cost reward paid the broken format best.** After three unusable replies the harness gives the instance up at
  the cheapest a failure could then cost, so a broken game scored at least as well as any other failure.

The held-out stop then only froze what the update had broken, keeping the phase's starting adapter in 37 of the 41
phases it ran in. A per-game reward cannot credit what the stream rewards in any case: its score is a sum over 200
rounds, a demonstration bought now pays off when its task returns, the learning rule may not read task ids to link the
visits, and a run is one sample of the whole stream's score. So outcome-only PG can learn one thing here, to solve an
instance, from its own wins.

**PG is REINFORCE with no baseline on the binary reward:** a win's reward is 1 and a failure's 0, and a game's advantage
is its reward, so no failure is pushed down and nothing pays for failing fast; a failure carries no gradient, so its
calls leave the batches (`train.pg_items`). Each win still trains through PPO's clip against the adapter that played it,
so passing over every win again each phase does not memorise it, and the surrogate is dual-clipped. The held-out check
is PG's own loss on the held-out games, how much likelier the held-out wins have become, with the wins' negative
log-likelihood per token logged beside it. It has something to say only once a held-out game has won, about one game in
70 at the base model's 13-16% win rate, so the stop runs from then and until then a phase makes one pass over every win
so far. The first phase trained this way, on the 115 calls of two won warm-up games, ended 0.006 nats a token from the
model that played them, where the broken runs' phases had ended 2 to 16 nats away. A round played by a warm-up phase,
which trained nothing, is scored against the base model, whose weights that phase served.

## The benchmark's charge

The benchmark charges an unsolved instance the whole budget, 16, however it failed, and a solved one what it spent. It
had charged an unsolved instance the attempt cap plus its demonstration requests, so failing without asking cost 8 and
asking for every demonstration before failing 16; unify changed it so that the benchmark does not reward failing fast
without asking for demonstrations. The rules every learner reads say so, so the nine final runs started again from round
0 under them.

## The final runs

The nine final runs, DAgger, OPSD and PG at 1, 2 and 4 users, started from round 0 on 20 Oct, every row alike:

- **Schedule:** no training until 24 games have been played; then each phase on every round's data so far, one game in
  ten held out (`forward.held_game`), stopped on the row's own held-out loss measured every 5 steps with a patience of
  2, the best adapter kept, at most 120 steps; LR 1e-4, batch 64, `LAG=1`, the stream's 200 rounds.
- **Harness:** prime-agent compacting at 48k tokens and keeping 8k (`loop.HARNESS`), where the paused runs compacted at
  32k: a compaction probe on the untrained Qwen3.5-9B, 48 games an arm, solved 16 at 48k against 10 at 32k, every call
  still inside the trainer's 65,536 tokens. Empty replies are re-prompted in the same conversation.
- **Labeller:** as above, under c1 and the program library's procedure (`ORACLE_RULES=library`), which the student
  never reads. Later runs whose oracle reads c1 alone (`ORACLE_RULES=c1`) differ in that alone.

When they started, the search's DAgger runs had solved 1 to 16 of 19 to 44 trained games against 0 to 7 of 2 to 28
untrained, and its OPSD and PG runs on every round's data none of 31.

## Step checkpoints

The runs play on preemptible machines, which the provider takes back about once in 22 machine-hours, most often within
an hour of a machine's start. A phase cut short trained again from its start, up to about an hour once phases reach 120
steps. `checkpoint.py` lets it go on from where it was instead. It is on (`checkpoint.ON = True`) after the GPU check
below; the nine final runs train without it, their nodes keeping their own code to the end.

- **When:** on SIGTERM, the provider's notice 60 s before it stops the machine: the trainer finishes the step in
  flight, writes a checkpoint and exits. And every `checkpoint.EVERY` steps, 25, for a stop with no notice in time: a
  save holds training up 4 s, so these cost 0.4% to 0.8% of a row's training, and a stop with no notice loses at most
  25 steps.
- **What:** the adapter, its trainer copy and the optimizer (3.7 GB), every file synced to disk (on a thread of its own,
  14 to 22 s) before `state.json` names it, so a machine stopped outright leaves the last checkpoint whole; the random
  generator's state; every step's record, with a digest of its batch's calls; the held-out curve, its best and the
  patience count. OPSD also keeps the student that drew the batch in flight, which draws that batch again on resume.
- **Only its own:** a checkpoint whose settings, starting adapter or calls differ from the phase's, or whose batches do
  not replay as recorded, is dropped and the phase trained again whole. ask-sd's data-parallel workers ignore SIGTERM,
  so every rank gets through the step in flight.

On an on-demand 8x H100, each update trained a phase of a final run's own rounds for 15 steps on three GPUs, stopped
once step 8 was logged, by SIGKILL (a checkpoint every 5 steps) and by SIGTERM (every 25, as it ships): killed, every
update went on from step 5; sent SIGTERM, it checkpointed at step 9, the trainer out 44 to 62 s after the signal (OPSD's
the longest, for its draws), and a restart reached its first resumed step in 36 to 72 s. Every phase trained its 15
steps to its `DONE`, and each update's two tests took the same batches in the same order and
ended within the GPUs' own noise of each other: training on these GPUs is not reproducible to the bit across processes
(two uninterrupted runs of one phase ended 32% to 48% of their updates apart in an earlier test), and a resume either
matched the uninterrupted phase to the bit or differed at bf16's precision. A step longer than about 35 s can overrun the
60 s notice, as OPSD's often would; the phase then goes on from the checkpoint before. A restart compiles nothing again:
torch's kernel cache is kept in `~/.cache/torchinductor` (`train.INDUCTOR_CACHE`), since `/tmp` is cleared when a
machine restarts (a fresh trainer's first three steps took 203, 106 and 59 s with the cache empty, against 25, 25 and
18 s once warm), and vLLM's compiled graphs in `~/.cache/vllm` (a server restart 135 s, against 335 s cold).

## The OPSD audit

The one intended difference between the rows: DAgger raises the likelihood of the oracle's label at a call; OPSD draws
the student's own reply there and distils the frozen base model reading that same label as a demonstration (SDFT's
forward KL). Each row's own selection code was run over the same play and labels, with the code the runs' nodes run, and
found no bug in OPSD's data: the same calls on both sides in every round, training and held-out; each demonstration
equal to DAgger's target at the same call; the same held-out games; the teacher's prompt the student's with the
demonstration at the end of the last message and nothing else changed, re-tokenised calls included
(`train.splice_note`); the stop and the settings (`realtime.train_args`) one set for both rows; and the loss, ask-sd's
`Trainer.distill_chains`, the same as SDFT's own trainer's: the forward KL over the vocabulary at each position of a
reply past its first 3, averaged over the reply, the replies weighing alike.

Two small differences in the step plan (too-long calls left out at a different point, 4 of 2,241 calls in one phase,
and later passes planned at a different time) change nothing in expectation. On the same
calls and labels, OPSD's trained games solved 14%, 12% and 8% at four, two and one users against DAgger's 25%, 22% and
22%: what separates them is the update, or chance at these counts.

**One setup for every row.** `train.fit` runs a phase for DAgger, OPSD and PG alike: every pass planned before the
first step (`train.plan`), the held-out loss measured before the first step, every `--eval-steps` and after the last,
the best adapter kept, the patience stop, the time cap and the step checkpoints. Only the update differs:
`train.Imitation` (DAgger), `train.SelfDistillation` (OPSD) and `train.PolicyGradient` (PG). OPSD's units are DAgger's,
so the same steps, and its demonstrations are the label set's targets (`train.demo`), so no phase renders a prompt
again. The nine final runs keep their own code; runs started later use this setup.

## GPU layouts

`fleet.py` uses a node's GPUs as fully as possible given the cost, as a general tool for any set of runs. On the nine
final runs (`fleet.py profile`, 17 to 78 rounds in), a 4-user DAgger round took 22.3 minutes to play, 25.4 more for its
last labels and 23.2 to train on four GPUs, and a 4-user PG round 9.7 to play and 4.1 to train. A round plays (P, on the model server), waits for its labels (L, the oracle through OpenRouter, no GPU) and trains (T).
Under the one-round lag round r+2 waits for phase r, which waits for round r's labels, so a round takes about max(P, T,
(P + L + T) / 2) (`test_fleet.py` checks the simulator against it). The labelled rows' players waited for their phases
34-63% of their hours: they are held up by their labels and training, not by play, and PG barely waits at all.

The trainer is close to linear in its GPUs: over 11,000 steps of the runs on H100s, DAgger read 100k tokens in 20.7 s on
one GPU, 12.3 on two and 6.2 on four, and on six, kernels cached, its steps ran 1.38 times faster than on four (78% per
GPU of one GPU's speed), every step's loss agreeing to six digits across 2, 4 and 6 GPUs. So a phase is about the same
GPU-minutes on any number of GPUs. A model server's call (`fleet.py bench-serve`, captured games replayed against
`serve.sh`'s server) took 3.70 s per thousand tokens it wrote on four GPUs tensor-parallel with one user and 5.23 with
eight; two GPUs make a call 18-27% slower than four at the same load, one GPU 51-72%, and a game's play scales with a
call's time. OPSD's trainer draws 64 replies a step from the server, which slowed the other calls on a 2-GPU server from
6.15 s to 16.2 s per thousand tokens, though not OPSD's own play, which mostly waits for the very phase its
trainer draws for.

**The design.** Runs on a node share one server and, given the same `TRAIN_GPUS`, one trainer pool (`gpupool.py`): each
run's phases run in a `train` process on the GPUs the pool gives it, every run computing the same split, the GPUs evenly
over the runs with a phase to train. A phase in flight gives GPUs up to another run's phase at its next step checkpoint
and takes more only when its next phase starts. A phase's steps do not depend on its GPUs (chain pieces are cut at
`train.CHAIN_CALLS`, 16 calls, and a checkpoint's fingerprint leaves the GPUs out), so it resumes on any number of them.
A warm-up phase takes no GPU and saves no adapter. `fleet.py plan` simulates every layout of a set of runs on 8-GPU
nodes over their profiles and prints the plans from the fastest to the cheapest; with `--deadline` it picks the cheapest
done in time, with `--budget` the fastest within it.

For six later runs on preemptible 8x H100s at $6.32 an hour, a node each in the final runs' layout came to $6,243, the
last done in 197 h; `apart`, a node each with the server on 2 GPUs and the trainer on 6, $5,945 in the same 197 h; three
pairs sharing nodes $4,087 and 224 h. No layout gives every run far fewer GPUs without slowing it, since a trainer, when
it trains, holds up the next round. Done in time came first, so later runs take `apart`: `SERVE_GPUS=0,1 TRAIN_GPUS=2,3,4,5,6,7`, but for DAgger at 4 users, whose play is the most of any, which
keeps `0,1,2,3` and `4,5,6,7`.

## Why OPSD does not learn

DAgger and PG learnt and OPSD did not. On the slots both rows' trained adapters played (every run plays the same
instance in each round and user slot), OPSD solved 5 of 48, 5 of 22 and 1 of 10 at four, two and one users, against
DAgger's 24 of 48, 10 of 22 and 4 of 10, at a cost of 14.7, 12.9 and 14.7 against 9.4, 9.8 and 10.2.

**OPSD's students are not broken; they never take the oracle's habit.** None of their replies came back empty, looping
reasoning stayed as rare as in the warm-ups, they gave up as often, and code that does not compile rose in both rows
alike. What separates them is the move:

| | demos before the first submission | submissions a game | reasoning, median characters a call | calls writing an action inside a tool call |
| --- | --- | --- | --- | --- |
| base model (the six warm-ups) | 0.27-0.72 | 2.0-3.0 | 338-466 | 20-33% |
| DAgger, trained | 3.1-3.6 | 1.4-1.9 | 716-1,280 | 0.0-0.3% |
| OPSD, trained | 1.0-1.2 | 3.5-4.4 | 202-242 | 10-16% |

**Training is not the difference.** OPSD trained as much as DAgger (416 steps against 361 at four users), its first
phase moved the weights as far, and the held-out divergence to the teacher fell 25%, 62% and 49% in the three runs'
first phases with a held-out game: the students moved toward their teacher.

**What the teacher favours.** 113 calls of one run's first trained phase, drawn by the move DAgger's label makes there,
with both prompts as `train.build_items` makes them: the student's as served, the teacher's the same with SDFT's
demonstration at the end of the last message, which in a harness call is the tool's output. Each cell is the share of
replies that make the label's move:

| DAgger's label (calls) | base model | OPSD's teacher | the run's first trained adapter | a late adapter (phase 20) |
| --- | --- | --- | --- | --- |
| asks for a demo (35) | 1% (1 of 70) | 16% (11 of 70) | 14% (5 of 35) | 3% (1 of 35) |
| submits (35) | 7% (5 of 70) | 9% (6 of 70) | 20% (7 of 35) | 6% (2 of 35) |
| runs code (35) | 89% (62 of 70) | 96% (67 of 70) | 80% (28 of 35) | 97% (34 of 35) |

The teacher favours DAgger's target overwhelmingly as text, since the target stands word for word in its prompt, but
as a move only weakly. It reads the demonstration, inside the tool's output, as part of the conversation, often as a move
already made: where the label asks for a demo, 46% of its replies name the request in their reasoning but 16% make it;
where the label submits, 59% take the submission as made. The students learn this teacher, its words and not a move it
seldom makes: late in a run, where the label asks for a demo, half of the run's own replies name the request and none
makes it. Placed as a user message of its own the demonstration made the teacher ask in 36% and submit in 17%, and in
the system prompt 3% and 14%: no placement makes it take the label's move reliably.

So OPSD fails for its teacher, not its code or its training. SDFT's teacher prompt sets the demonstration as an example
response in the question's own turn; in an agent harness a reply answers the tool's output, so the example lands inside
it, and the 9B base model reads an assistant's reply there as part of the conversation, not as advice for the reply it
is about to write. More steps, another learning rate or another stop would only bring the students closer to the same
teacher. unify did no further work on OPSD's design, since it is a baseline: with DAgger's trained games solving 36-54%
against its warm-ups' 12-18%, PG's 21-42% against 8-19% and OPSD's 8-21% against 8-27%, `opsd-u2-lag1` and
`opsd-u4-lag1` were stopped after rounds 33 and 26, and `opsd-u1-lag1` plays the whole stream as the one whole OPSD
curve.

## The frozen tests after the stream

The stream ends at round 200, and the system is then tested twice, frozen and in parallel, each test one round of 100
users who all start from the end-of-stream state, with nothing learnt during the test and nothing passed from one test
instance to another: the held-out test, one instance of each of the 100 held-out tasks, and the seen test, four new
instances of each of the stream's 25 tasks at seeds no stream draws (the benchmark's `configs.test_schedule`). The
held-out test asks what the system carries to new tasks; the seen test what it keeps of the tasks it learnt. The
schedule's epilogue, 100 more rounds in which every user met the held-out tasks while the system went on learning, was
retired for these tests: nothing played in one counts.

A trained run's tests are played on its own node the moment its stream ends (`frozen.sh`): the run's phase 199, trained
on every round, served on half of the run's GPUs for each test, and test user i starting from stream user i mod U's
memory, copied and relocated. A copied prime-agent state names its own folder some thousand times, in its Python
environments, its daemon's records and its sessions, so `frozen.py` renames every absolute path to the stream user's
folder in the copy's text files and links to the test user's, and leaves binary files alone (the one kind seen to name
the folder is a session's saved Python namespace, which prime-agent reloads only into that same session).

## The from-scratch retrain

Each DAgger phase starts from the phase before's adapter, so by round 200 the continual student is some 175 short
warm-started updates, the earliest games trained on in every phase and the latest in one or two. Textbook DAgger
retrains on the aggregate from scratch at each iteration; this is the warm-started variant, and warm starts have been
found to generalise worse than fresh starts on the same data (Ash and Adams, "On Warm-Starting Neural Network
Training", NeurIPS 2020). So at the stream's end the run's phase 199 is trained once more from the base model
(`scratch.py`, `scratch.sh`): the same label sets of rounds 0-199, the same held-out games, loss, learning rate, batch
and patience, so only the start and the stop differ. The stop measures the held-out loss every 25 steps rather than 5,
with no step cap and at most 12 hours, since a fresh adapter needs far more steps (a pass over a final run's label sets
is about 230 steps at one user and 950 at four). The data was gathered by the continual student's own play, so the
retrain isolates how the run updates on its data, not what a student retrained at every phase would have gathered.

`scratch.py` trains through the run's own `realtime.py` and its `train_phase` with no starting adapter. `scratch.sh`
then plays both adapters frozen, the continual phase 199's and the fresh one, on the 100 validation puzzles (`grid-vals.sh`), and the fresh
adapter's two frozen tests from the run's own end-of-stream memory, beside the continual student's, so the two differ in
their weights alone. `summary.json` holds both adapters' held-out loss on the same held-out games and each play compared
instance by instance; with 100 puzzles a single run's difference is about plus or minus 9 points, so the signal is a gap
of the same sign at 1, 2 and 4 users. It runs on each DAgger run's own node once every run there is done: about 6 to
12 hours of a preemptible node a run.

## The weights-alone tests at the stream's end

A run's frozen test starts from its end-of-stream memory, so a solve there may come from what the weights learnt, from
the programs and notes the harness kept, or both. The weights-alone tests take the memory out: the run's phase 199
plays the held-out and the seen test from an empty harness state, under the run's own task names (its `alias-salt`,
since the weights may have learnt to know a task by its name, as a note does), into `scratch/weights/` (`frozen.py play
--no-memory`, `frozen.sh` with `MEMORY=none`). Each test then has three plays of the same 100 instances: the reset (row
h0, the base model from an empty state), the weights alone, and the run's own test. The weights alone against the reset
is what training put into the weights; the run's own test against the weights alone is what the memory adds. A gain on
the seen test with none on the held-out one says the weights remember the stream's rules; a gain on the held-out test,
general skill. It is `scratch.sh`'s first stage.

PG at 2 users gave the first answer. Its weights alone solved 31 of the 100 held-out tasks and 38 of the 100 seen
instances, against row h0's 13 and 18 (transfer +12.1 points [4.6, 19.5] and +15.3 [6.2, 24.8]); with its memory it
solved 23 and 33, the memory costing 5.1 [-12.1, 2.0] and 2.4 [-10.5, 5.9] points, within noise. So PG's gain is in its
weights, and nearly as large on tasks it never met as on the stream's own.

## Why DAgger's phases stop

Some 50 to 80 rounds into the final runs, most DAgger phases kept the adapter they started from: `dagger-u1-lag1` in 39
of its 53 phases with a held-out curve, `dagger-u2-lag1` in 41 of 56 and `dagger-u4-lag1` in 28 of 44. In those the
held-out loss rose by a median 0.006, 0.004 and 0.003 nats a token (0.5 to 1%) by step 5, the stop (patience 2,
measured every 5 steps) ended the phase at step 10, and in about half of them the loss was falling again at that step.
Each run's weights first moved in one long fit, in the first phase to train once 24 games were played (phase 6 at four
users, 11 at two and 23 at one), and since then in one phase in three or four.

The aggregate dose reshuffles every round's labels so far in every phase (8,360 at `dagger-u2-lag1`'s phase 66, a new
round adding 34 to 222), so the 640 labels of a phase stopped at step 10 are almost all rounds the adapter has fitted
in phase after phase. The held-out set is one game in ten of every round so far, 1,000 calls of them drawn a phase, the
newest round most often none. Both are the old rounds, and the states the student plays now are a few percent of each.

An offline check (not included here) played one stalled phase, `dagger-u2-lag1`'s phase 61, from the adapter it started
from and in the run's own code, past its stop to 40 steps, on two mixes: the run's own aggregate, whose first
measurements repeated the run's, and the recent mix, the last ten rounds' labels and as many from earlier games drawn at
random. It measured the loss on the run's held-out set, on the frontier (every label of the next six rounds, the states
the student meets next) and on the recent rounds. The frontier is harder than the old held-out set (0.6685 against
0.6328 at step 0), but the two move together. By step 40 the aggregate mix had lowered the held-out loss 0.1%, the frontier 0.5%
and the recent rounds 3.5%; the recent mix lowered all three more, and steadily from step 10: by 0.8%, 1.3% and 11.2%.
Both mixes rose on every set at steps 5 to 10 and improved after, so the stop would have ended both at step 10. So the
old held-out set is a slow signal rather than a blind one, and what held the runs back is a stop too impatient for the
rise of the first steps and an aggregate mix that dilutes the newest rounds.

So the three final DAgger runs went over to the recent dose (`DOSE=recent`: each phase the last `RECENT_ROUNDS`, 10,
rounds' labels and as many from whole earlier games, under the held-out stop) and a patience of 4, from phase 94 at one
user, 89 at two and 69 at four, by a restart of each run at a moment when no play or training would be lost (the
scripts that did it are not included; `realtime.sh` takes `DOSE` and `PATIENCE`). OPSD and PG train on every game so far
throughout.

## The switch test

After the switch the 4-user run's lead over row H (the untrained student keeping its own notes, on the same instances)
fell from about 52 points of budget around round 80 to about 38 by round 110, about 8 points of it the same tasks played
worse. The test asked whether the recent dose made the weights worse: each run's adapter from just before its switch (the last phase on
the old method that changed the weights: 93 at one user, 88 at two, 68 at four) against its latest (152, 146 and 109),
both played on the two frozen tests from an empty memory under the run's own task names, so only the weights differ and
both play the same instances. The rule set in advance: go back to the switch point and continue on the original method
if the 4-user run's latest weights play the seen test worse beyond twice the standard error.

Budget used on the same instances, lower is better (the 4-user arms played twice, averaged per instance):

| users | test | solved before | solved after | budget before | budget after | change |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | seen | 52% | 73% | 56.8% | 44.6% | -12.2 ± 3.1 |
| 4 | held-out | 24% | 30% | 81.2% | 76.8% | -4.4 ± 2.2 |
| 2 | seen | 44% | 64% | 65.5% | 48.4% | -17.1 ± 4.1 |
| 2 | held-out | 27% | 35% | 78.9% | 74.2% | -4.8 ± 3.2 |
| 1 | seen | 52% | 48% | 63.4% | 61.1% | -2.3 ± 3.5 |
| 1 | held-out | 28% | 24% | 79.9% | 79.8% | -0.2 ± 3.2 |

The recent dose made the weights better, not worse: the 4-user and 2-user runs' latest weights play the seen test better
by four standard errors, and the 1-user run's did not change, so the switch stays. The 4-user stream's fall after round
90 was not its weights getting worse.

The recent dose's window counts rounds, not games. It trains half of each phase on the last ten rounds, and a round
holds one game for each user, so the window held 5.6 of the 25 tasks on average at one user (3 to 9), 9.8 at two and
16.0 at four. At one user half of every phase is a few tasks that change from round to round, which fits the one-user
adapter gaining nothing where the others rose. It is one run of each, and the one-user run saw only 60 games on the
recent dose; a window counted in games, or one that holds every task, would test it.

## Open questions

Candidates from a literature review, not run (its first, a minimum amount of data before the first update, as in
DQN-style agents' replay-start sizes and DAgger's own warning about its first policies, is the 24-game warm-up):

1. Retraining from the starting adapter on the aggregate each phase (STaR, ReST-EM) against continuing with replay and
   the learning rate re-warmed (TiC-CLIP, Ibrahim et al. 2024).
2. A higher LoRA learning rate for short updates, at smaller batches ("LoRA Without Regret", Thinking Machines 2025;
   Biderman et al. 2024).
3. DAgger: capping the share of "ask" labels at the oracle's own on-policy rate, or weighting by action cost
   (AggreVaTe); labelling only where the student's move differs from the oracle's (LEAP).
4. PG: a lexicographic reward (cost only among solved games), a per-task baseline; mixing oracle data into the update
   (LUFFY); entropy and clip guards against collapse (RAGEN, DAPO).
5. OPSD: whether the base model shown GLM's reply plays better than alone (SDFT's own check); an EMA teacher instead of
   the frozen base (SDFT App. A.3); a teacher that makes the label's move in most of its replies.
6. On-policy distillation from GLM's per-token log-probabilities (Thinking Machines 2025; GKD).
7. The recent dose's window counted in games rather than rounds.
