# The capacity and task-count ablation

A point of interest beside the paper, not part of it (the author, 28 Oct 2026, about 00:50 PDT: "I think we should run it
right now"). Two questions:

1. **Capacity.** Trained to overfit on GLM-5.3-flash's own games of the paper's 25 stream tasks, does Qwen3.5-9B's
   seen-test score reach GLM's? GLM solves 94 of the arc1-small seed-0 seen test with its memory wiped and 96 with
   it; R's recipe's weights alone solve 76-79, but that recipe guards against storing rules (at most two passes,
   an early stop on tasks held out of training; [method-search.md](../../docs/method-search.md), "On the seen test
   every label source teaches alike").
2. **Task count.** Trained the same way on 100 and on 400 tasks, does the held-out test (100 ARC-AGI-1 evaluation
   tasks, which have no Re-ARC generators, so none is a training task) climb from today's 45-50?

the author's words: "generate as much data as we possibly can and keep training on all of that data until the training loss
stops going down ... there's no such thing as overfitting because the whole point is to overfit".

## The runs

Three standalone runs, each from the base model, on the benchmark's nested task sets at seed 0
(`continual_arc.configs.config_tasks`): **t25** (arc1-small's 25, the seen test's own), **t100** (arc1-medium's 100,
the 25 among them) and **t400** (arc1-full's 400, every Re-ARC task). All three are tested alike: the seen test (25
tasks, 4 fresh instances each) and the held-out test, both of the weights alone from an empty memory under the run's
own task names, and R's 100 validation puzzles.

| piece | what it does | where |
|---|---|---|
| [`generate.py`](generate.py) | GLM plays batch after batch of 100 games of the run's tasks, each a fresh user in a play of its own; each batch, once every game has ended, is labelled into a training set | `NODE-cap-gen` (GCP e2-standard-16), `~/runs/rt/cap-RUN` |
| [`capacity.py`](capacity.py) | the runs, their tasks, each batch's instances, the labels, the stop, the tests' cells | everywhere |
| [`trainer.py`](trainer.py) | each training set a phase: the loss on its games first, then one pass over every batch so far | Modal's 8 H100s, the deployed app `MODAL_CAPACITY` ([`modal_capacity.py`](modal_capacity.py)), the volume `MODAL_CAPACITY` |
| brain's `capacity` task | every twenty minutes: ships finished sets to the volume, starts a run's trainer when a set waits and none runs, keeps each generator at most three batches ahead and under a cap that halves when the paper's reference rows slow, stops it at its plateau or its 1,500 USD and copies its last phase's adapter to the bucket for its tests | the operator's host (`capacity.yml`); deleted from brain on 2 Nov 2026 once the ablation ended, so restore it from brain's git to run this again |

### The games

GLM-5.3-flash plays through OpenRouter (Z.ai alone, medium reasoning, the labeller's and the reference rows'
setting) in the student's own harness as its frozen tests play the student: the turn-based budget, `loop.HARNESS`'s
compaction at 49,152 tokens in the student's 131,072-token window, replies at prime-agent's own cap, one fresh user
an instance. That is the harness GLM's own h0 row solved 94 of the seen test in, and every call it makes is one the
student could be served and trained on. Its games are played by [`glm_val.py`](../generalisation/glm_val.py)
`--harness tests` on the turn-based bundle `code-dagger-rows-turns`, the code the tests play the student on, under
the run's alias salt (`~/runs/rt/cap-RUN/alias-salt`), so the student meets each task under the name its tests show
it ("Task id: task-..."). A batch deals the run's tasks evenly, a seeded order of them round after round: each of 25
tasks 4 times, each of 100 once, a quarter of 400 once each. Each game's instance has a seed of its own,
`derive_seed(0, "capacity", run, batch, game)`, which the seen test (`derive_seed(0, "seen-test", i)`) never draws.

The generator plays 4 games at once to start, halves that on any call asked again (a 429 or a 5xx: each game's
`retries.jsonl`), and adds 2 after 30 minutes without one, up to `~/runs/rt/cap-RUN/concurrency`. The OpenRouter key
`OPENROUTER_KEY_NAME` and GLM's providers also carry the paper's 63 reference rows, which must not slow, so brain's task sets that file
from the rows' own pace: it reads the GLM rows' last hour of calls on `NODE-ref-rows*` against the day before the
generators started (27 ms a reply token, 0.11% failed), halves the cap (from 8, at least 2) once they run 1.25 times
slower or fail 3 times as often, and raises it by 2 after two calm hours, to 16.

### The labels

Every call GLM made as the agent (prime-agent's compaction and internal calls left out) is a label, its own reply at
its own state (`capacity.label_batch`): forward.py's points over the batch's games, each point's reply written in
`forward.py label`'s layout, then `forward.py label-set`, the student's template. Each call's earlier replies are
shown as the student's own would stand in its conversation, as `dagger.student_message` makes a target (all prose in
the reasoning, only the action visible), with the student's chat-template settings, so that what the student is
trained on reads as its own play does. Solved and unsolved games alike: GLM's policy is the target.

### The training

Qwen3.5-9B from the base model, LoRA rank 128 on every attention and MLP projection (train.py's defaults), R's
recipe's LR 5e-5 and batch 16, chains read whole as the DAgger rows read them. Every task is in training: nothing held
out, no early stop, nothing chosen by a validation set. Each phase

1. measures the loss on the new batch, games no phase has trained on, under the adapter the last phase left (the base
   model before phase 0). This is the stop's signal: the loss on data already trained on keeps falling as the model
   learns GLM's exact text;
2. trains one pass over every batch so far, warm-started from the last phase's adapter and optimizer moments; once
   that is more than 16,000 calls, the new batch whole and a seeded draw of older games up to 16,000;
3. keeps its adapter and numbers (`RUN/phases/pKKKK/`, `state.json`, `phases.jsonl`, `steps.jsonl` in the volume).

A run has **plateaued** once three phases in a row have not come 1% under its best new-batch loss
(`capacity.plateaued`). Each Modal job trains the sets waiting when it starts, phase after phase, and ends, so nothing
is paid while GLM plays; Modal ends a job at 24 hours, and no phase starts that would end after 22.5.

## The tests

At a run's plateau (brain's `capacity` line says `PLATEAUED`), or at its cap, test its last phase's adapter, which
the task has copied to `results/runs/gpu-nodes/modal-capacity/cap-RUN/phases/pKKKK/adapter/` (every phase's stays in
the volume, `RUN/phases/pKKKK/adapter/`):

```bash
~/continual-arc-baselines/.venv/bin/python search/capacity/capacity.py cells t25 14
```

writes two cells into `search/generalisation/cells/`: `cap-t25-p0014-frozen` (the seen and held-out tests, the
weights alone from an empty memory under the run's task names, from the run's play config, which the hourly backup
copies off `NODE-cap-gen`) and `cap-t25-p0014-val` (R's 100 puzzles). Commit them and launch each on a machine
that plays, `NODE-turns-b` once its queue ends or a node of its own
([../generalisation/README.md](../generalisation/README.md), "Running a cell on any machine"). Then write its scores
into [docs/method-search.md](../../docs/method-search.md)'s paragraph on this ablation.

## Spend

About 0.04 USD a GLM game, so 4 USD a batch; about 7 minutes of 8 H100s a batch for each pass, so a phase's training
grows with the batches until the 16,000-call cap, about 70 minutes; Modal's 8 H100s about 35 USD an hour. Each run is
capped at 1,500 USD of GPU and GLM together (brain's task stops its generator and starts no job past it), unless the author
raises it.

## Running it again

```bash
TTL_HOURS=168 PURPOSE="..." CREATE_ONLY=1 bash ~/continual-arc-baselines/scripts/create-node.sh NODE-cap-gen e2-standard-16
LEASE="gcp NODE-cap-gen" bash search/capacity/launch.sh NODE_USER@<ip> t25 t100 t400
bash search/capacity/modal-capacity.sh deploy    # after any change to trainer.py or capacity.py
```

`launch.sh` again updates the tools and starts only the generators not running; a generator takes up where it stopped.
