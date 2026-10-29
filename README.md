<!-- ARXIV_ID_PENDING: replace every ARXIV_ID_PENDING in this file with the paper's arXiv id once its listing is out, the evening before the launch on 27 October 2026, then delete this line -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/continual-arc-readme-banner-dark.png">
    <img src="assets/continual-arc-readme-banner-light.png" alt="Continual-ARC, a unify benchmark" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://unify.ai/continual-arc">Explore every run</a> ·
  <a href="https://unify.ai/blog/why-we-built-continual-arc">Paper</a> ·
  <a href="https://unify.ai/blog/why-we-built-continual-arc">Blog post</a> ·
  <a href="https://unify.ai/continual-arc/leaderboard">Leaderboard</a> ·
  <a href="baselines">Baselines</a> ·
  <a href="training">Training</a>
</p>

# Continual-ARC

ARC-AGI measures how efficiently a system picks up a new skill from a few examples, but it serves each task once.
Continual-ARC serves a lifelong stream of ARC-AGI-1 tasks instead, in which the same tasks return
after dormant gaps with fresh instances while new tasks keep arriving. On each instance the learner may pay for one
more demonstration or submit an answer, and it may try again after a wrong one. On an instance it solves, a wrong
answer and a demonstration each cost one; an instance it does not solve costs the whole budget of sixteen, however it
failed, and we charge the learner the share of that budget it used.

Learning on the job can mean two things: remembering the rule of a task met before, and getting better at tasks never
met. So after the stream we freeze the learner and give it two tests. Each is one round in which every instance starts
from the learner's state at the end of the stream, so nothing learnt on one test instance reaches another. The seen
test serves fresh instances of the tasks the learner met, which measures memorisation. The held-out test serves 100
ARC-AGI-1 evaluation tasks it never met, which measures generalization, and it is the headline. We read each test as T,
the points of the budget a learner saves against itself reset before each instance, in its own harness and on the
same instances, so a stronger model scores no higher for being stronger. A learner may keep anything at all between
instances, which puts an agent harness keeping notes and a model updating its weights on the same scale.

This repository is the benchmark of the paper [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://unify.ai/blog/why-we-built-continual-arc): the environment, the schedules, the track and the measures.
The learners the paper evaluates, adapters for three agent harnesses among them, are in
[`baselines/`](baselines), and the training of its trained rows is in [`training/`](training). The records of its
GPT-6 Luna runs, with the trajectories behind them, are plain JSON in git at
[continual-arc-runs](https://github.com/unifyai/continual-arc-runs) and attached to the
[v0.1.0 release](https://github.com/unifyai/continual-arc/releases/tag/v0.1.0), and every task, figure, run and
trajectory of the paper can be explored and replayed at [unify.ai/continual-arc](https://unify.ai/continual-arc).

On GPT-6 Luna, its check on a frontier model, the paper ran Prime Agent, Hermes Agent and OpenClaw through the stream
with every learning mechanism they ship switched on, and again with everything they wrote deleted before every
instance. Their persistence saved no measurable budget at any reasoning effort, and left to decide what to keep, they
kept almost nothing that outlasted an instance: none saved a measurable share of the budget left to learn. A
deliberately naive program library, which keeps one tested program per task and runs it when the task returns, saved
about two thirds of it at every effort. Run your own learner through it and [submit the result](#submitting-a-result).

* **Any learner.** The learner is a black box with four methods. Weight updates, external memory, program libraries
  and plain prompting are all welcome.
* **Binary feedback.** Every submission is answered correct or incorrect, as ARC Prize answers it.
* **Demonstrations cost.** One more input-output pair of the task is available on request, charged like a wrong
  attempt.
* **Fresh instances.** Every test input and every demonstration of the stream and the seen test is freshly drawn
  from the task's Re-ARC generator. A cached answer never fits the next grid, and caching a demonstration is a choice
  with a price.
* **Composable interventions.** Optional plug-ins can insert tagged auxiliary episodes without changing the
  schedule's exposure or gap metadata.
* **Two frozen tests, one headline.** T on the held-out test, with T on the seen test and the share learned over the
  stream beside it. Compute, tokens and memory are logged, not scored.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
uv run continual-arc data fetch     # clones Re-ARC and ARC-AGI at pinned commits
uv run pytest
```

Data lives in `~/.cache/continual-arc` (override with `CONTINUAL_ARC_DATA`).

## The environment

A schedule is a list of `(task_id, instance_seed, phase, reset_before)` slots.
The environment serves one instance at a time and accepts three actions:

| action          | effect                                                                   | cost |
|-----------------|--------------------------------------------------------------------------|------|
| `Submit(grid)`  | correct → instance ends; wrong → one more attempt used                   | 1 if wrong |
| `RequestDemos()`| returns one more demonstration pair, up to eight per instance (a ninth request is refused at no cost): a fresh Re-ARC pair on the stream and the seen test; on the held-out test, whose tasks have no generators, one of the task's own pairs, and nothing after the last of them (the instance's demo cap is the pairs it has, and the observation shows it) | `demo_cost` (default 1) |
| `GiveUp()`      | instance ends as failed                                                  | the rest of the budget, up to `max_cost` |

An instance that ends unsolved, at the attempt cap or given up, is charged the whole budget, `max_cost`, whatever
it spent, so failing is never cheaper than asking; a solved instance is charged its wrong attempts and the
demonstrations it bought. The budget used is `100 * mean_cost / max_cost`, with
`max_cost = attempt_cap + demo_cost * demo_cap`, 8 + 8 = 16, and lower is better; the
`score` in `summary.json` and the report is its complement, `100 - budget used`.

## The two frozen tests

After the stream the learner plays two tests, each built by `configs.test_schedule`:

| test | instances on `arc1-small` | what it reads |
|---|---|---|
| `held-out` | one instance of each of the configuration's 100 held-out ARC-AGI-1 evaluation tasks, which the stream never serves | generalization: whether the stream made the learner better at tasks it never met; the headline |
| `seen` | four fresh instances of each of the stream's 25 tasks, at instance seeds no stream draws; after a stream of several users, the same 100 spread over every user's own tasks, and one of each task once they bring more than 100 ([regimes](#regimes)) | memorisation: what the learner kept of the rules it met |

Each test is one round in which every instance is a user of its own, so every instance meets the learner as the
stream left it, and nothing learnt on one test instance reaches another, since a learner changes what it carries only
in `end_round` ([regimes](#regimes)). A test takes about as long as one instance, and nothing tells the learner it has
begun. A learner that keeps a demonstration the moment it buys one breaks that contract, and on the seen test it would
pass the pair to another instance of the same task in the same round. A learner that cannot serve 100 users at once
plays a test narrower, `--width N` users a round: then every instance is marked `reset_before`, and the learner's
`reset` must restore its state at the end of the stream rather than wipe it. `runner.run(..., max_concurrent=k)`
serves a concurrent learner's round k users at a time instead.

Both tests are read with T, against the same learner reset before each instance, played on the same instances on its
own, apart from the learner's own test:

```
T = U(reset) - U(learner)
```

with U the budget used over the test's instances, in points of the budget. The reset plays the learner's own model in
its own harness, so a stronger model is not credited for being stronger, and the harness, the environment the learner
learns in, is part of the baseline rather than measured against a model without it. The bare model is not a reference;
a result also shows the bare model on the same instances, and a reset that plays below it is flagged beside its T.
The reasons, and what remains open, are in [docs/DESIGN.md](docs/DESIGN.md#why-transfer-is-the-headline).

```bash
continual-arc schedule make --config arc1-small --seed 0 --no-epilogue --out stream.json
continual-arc schedule make --config arc1-small --seed 0 --test held-out --out held-out.json
continual-arc schedule make --config arc1-small --seed 0 --test seen --out seen.json
```

Playing a test from where the stream left the learner is the driver's job, as restoring any saved state is: keep what
the learner carries at the end of the stream and give each test its own copy, in a run directory that holds the
stream's `alias-salt`, so that the learner meets the stream's tasks under the names it learnt them by. For a learner
whose state is plain Python:

```python
import copy
import shutil
from pathlib import Path

from continual_arc.runner import run
from continual_arc.schedule import Schedule

from my_pkg.learners import MyLearner

learner = MyLearner()
run(learner, Schedule.load("stream.json"), out_dir="runs/my-stream")
for test in ("held-out", "seen"):
    out = Path(f"runs/my-stream-{test}")
    out.mkdir(parents=True)
    shutil.copy("runs/my-stream/alias-salt", out)        # the names the stream gave its tasks
    run(copy.deepcopy(learner), Schedule.load(f"{test}.json"), out_dir=out)
```

A learner that holds a model client or a process saves what it keeps instead, and loads it into a new instance for
each test. The references play the same instances one at a time, `--width 1`, so that the runner resets the learner
before each, each into a directory of its own, and `continual-arc transfer` reads T from the two runs of a test, the learner's and its reset's,
with a 95 percent interval from a bootstrap over tasks and the share solved of each:

```bash
continual-arc schedule make --config arc1-small --seed 0 --test held-out --width 1 --out held-out-1.json
continual-arc schedule make --config arc1-small --seed 0 --test seen --width 1 --out seen-1.json
continual-arc run --learner my_pkg.learners:MyLearner --schedule held-out-1.json --out runs/my-reset-held-out
continual-arc run --learner my_pkg.learners:MyLearner --schedule seen-1.json --out runs/my-reset-seen
continual-arc transfer runs/my-stream-held-out runs/my-reset-held-out
continual-arc transfer --test seen runs/my-stream-seen runs/my-reset-seen
```

The bare model's runs, shown beside T, come from `continual_arc_baselines.systems.bare.BareModelLearner` in
[`baselines/`](baselines), played on the same width-1 schedules
(`arc-baselines run --schedule held-out-1.json`).

## The share learned

The budget used rewards whatever the learner brings to the stream, and most of that is its
model and its harness. So a learner is read against **itself reset**, the same model in the same
harness with its state wiped before every instance of the same stream, which carries nothing
from one instance to the next. The reset is the scale's zero. Its one is perfect memory on the
same system: the reset's own charge on each task's first exposure, and nothing on any return.
Between them lies the **learnable budget**, which shrinks as the system gets stronger, and the
**share learned** is the part of it a learner saves:

```
share learned = (U(reset) - U(learner)) / (U(reset) - U(perfect memory))
```

with U the budget used over the stream. A learner that keeps nothing scores 0, one that answers
every returning task at no cost scores 1, and one worse than its reset scores below 0. Keeping
the demonstrations a task was shown is a kind of memory, and a learner that does so scores above
0 for it. The learnable budget says how much a track leaves a system to learn: when it is small,
the system has outgrown the track and the next edition measures it. The share learned reads
memorisation while it is being built, and the seen test what is left of it at the end.

```bash
continual-arc schedule make --config arc1-small --seed 0 --no-epilogue --reset-each --out stream-reset.json
continual-arc run --learner my_pkg.learners:MyLearner --schedule stream-reset.json --out runs/my-stream-reset
continual-arc learned runs/my-stream runs/my-stream-reset
```

prints the share with a 95 percent interval from a bootstrap over tasks
(`continual_arc.metrics.share_learned`). `--reset-each` serves the same stream with the learner
wiped before every slot; a learner that keeps nothing between instances is its own reset. Each
of the run's instances is read against the reset's play of the same instance, or, for one the
reset never met (another user's, at more users than the reset served), against the reset's mean
on the task.


## The stream

The stream interleaves its tasks as work does: a few tasks are active at once, each is worked in
a spell of a few instances and then rests, and new tasks keep arriving, so a task can rest long
enough to be forgotten before it returns. One process generates it (`stream_schedule`):
exponential spells, log-normal rests and Poisson arrivals, at the parameters each configuration
sets, with every task served the same number of times. An order of instances never touches the
learner's state.

The same process can serve the stream's visits one task at a time (`--order sequential`) or
fully shuffled (`--order random`). The paper plays those two only to check that the order does
not change its findings (its Appendix I). They are not settings of the benchmark, and a run in
either is an ablation.

The stream can still end with the **epilogue** (`--no-epilogue` leaves it out): the held-out
tasks once each, met while the learner still learns. No measure reads it. A learner that
keeps learning through 100 held-out tasks mixes what it built on the stream with what it
builds on them, and the held-out test reads the first alone. `--order fresh` builds no stream
but the held-out tasks alone, met by a learner whose state is wiped before each, at the
held-out test's own instances, so it can stand in for the held-out test's reset reference.

## Regimes

A configuration is served in one of two regimes, and they differ in one number, `users`:
how many people the system serves at once, and so how much experience arrives between two
updates. It is a setting of the run, not of the configuration (`with_users(config, users)`,
`schedule make --users`). Any learner can run at any number of users, whether it keeps notes
or trains its weights, and learners are compared only at the same number. The canonical
numbers are the powers of two from 1 to 16 (`CANONICAL_USERS`): a low-data regime by design,
every update seeing at most 16 new instances, so that a learner that trains its weights between
rounds finishes a whole stream on one machine in days, not weeks. A track serves as many users
as its pool gives tasks of their own: 16 on ARC-AGI-1, 4 on ARC-AGI-2 and 1 on ARC-AGI-3.

| regime       | users          | picture                                                                 |
|--------------|----------------|-------------------------------------------------------------------------|
| `individual` | 1              | one system walking one person's stream, one instance at a time          |
| `collective` | 2, 4, 8 or 16  | one system serving a team at once and learning from all of them (`arc1-small` at 16 users: 200 rounds of 16 instances) |

In the collective regime every user brings tasks of their own (Dan, 28 Oct 2026: "one user has
25 puzzles, two users has 50, and four users have 100"): user `k` plays block `k` of the
track's nested task order (`configs.task_order`, `configs.user_tasks`), which on ARC-AGI-1 is
`arc1-small`'s 25 tasks sorted, then the 75 `arc1-medium` adds sorted, then the 300 `arc1-full`
adds sorted, so that users 0 to 3 bring `arc1-medium`'s 100 and users 0 to 15 all 400. Each user
walks the individual regime's process over their own tasks with their own draw of it: user 0
at seed `s` walks exactly the individual stream at seed `s`, and user `u`'s tasks and stream do
not depend on how many users there are, so a run at fewer users is a run at more cut to its
first users. Every user meets the same held-out tasks, in an order of their own. The held-out
test is the same at every number of users; the seen test covers every user's tasks, 100 new
instances spread evenly over them (two of each of 50 at two users, one of each of 100 at four)
and one of every task at eight and sixteen, instance `i` a task of user `i mod U`, whose state
test user `i` starts from, so it is made for the run's number of users (`schedule make --test
seen --users U`). So more users bring more tasks as well as more instances at once. The users are deliberately concurrent: a **round** is one instance per
user, all open at once, the round's records are committed together, and the learner's
`begin_round` and `end_round` hooks bracket it. The contract is that whatever the learner
carries between instances changes only in `end_round`, so every user of a round meets the
state the round began with. Every observation and record says whose work it is (`user`).

A learner that can serve the users of a round at the same time sets
`concurrent_users = True` and reports its counters per user (`user_resources(user)`): the
runner then serves every user's instance on a thread of its own, from its first action to
its verdict, at its own pace. Any other learner is served one action at a time, one action
on every open instance before a second on any. With one user every instance is a round of
one and both are the serial loop.

Exposure index and gap are counted in rounds over the whole team (two users meeting a
held-out task in the same round share both; a stream task is one user's alone), and each record also carries the user's own count
(`user_exposure_index`, `user_gap`); with one user the two agree, the times a task was
seen before and the steps since it was last seen. A round never spans a phase, so the
epilogue starts on a round boundary and a test is a round of its own. `records.jsonl` only ever holds whole rounds, and a
resumed run replays the interrupted round from its start. Add-ons are served one instance at
a time and need one user.

## Cutting a stream short, and testing it early

**`Schedule.first_rounds(r)`** is a literal prefix of a schedule's own slots: rounds 0 to
`r - 1`, the same tasks, instance seeds and users as the full schedule. It is a slice, not a
rebuild, because the stream process gives every task a supply fixed by the configuration's
full length, so regenerating a shorter stream from the same seed would not be this one's
prefix. A resumed run checks its schedule's prefix slot by slot (`runner.run(resume=True)`),
so a run stopped under `first_rounds(20)` resumes without replay under `first_rounds(60)`, or
under the uncut schedule.

```bash
# the first 20 rounds of the stream served to 16 users
continual-arc schedule make --config arc1-small --users 16 --seed 0 --no-epilogue --rounds 20 --out stream-r20.json
```

The frozen tests need none of the stream, only the configuration and seed, so a learner can
play them from its state at any round that was kept, as from the end of the stream.
`protocol_of` lists a cut stream as a deviation from the canonical protocol ("stream cut at
round 20"); `meta.config` is kept, so the label stays readable.

## The tracks and the canonical number

Continual-ARC has a track for each ARC-AGI edition, and every configuration name carries its track:

| track | configs | stream and seen test | held-out test |
|---|---|---|---|
| `arc1` | `arc1-smoke`, `arc1-small`, `arc1-medium`, `arc1-full` | Re-ARC generators over the ARC-AGI-1 training tasks | ARC-AGI-1 evaluation tasks |
| `arc2` | `arc2-smoke`, `arc2-small` | ARC-GEN generators over the ARC-AGI-2 training tasks new to the second edition that pass the screen, 117 of them ([docs/arc2.md](docs/arc2.md)) | the 114 ARC-AGI-2 evaluation tasks new to the second edition |
| `arc3` | `arc3-smoke`, `arc3-small` | fresh levels of the first-level rules of 15 of ARC-AGI-3's 25 public games, each game's from a generator and solver of our own ([docs/arc3.md](docs/arc3.md)) | fresh levels of the other 10 games |

Each track has one canonical protocol: its `small` configuration's stream and then its two
frozen tests, at the default environment settings, at each canonical number
of users. Its number is T on the held-out test, with T on the seen test and the share learned
over the stream beside it. Seeds 0, 1 and 2 are all canonical; the paper reports seed 0, and
the leaderboard compares runs at seed 0. Every run's `summary.json` and `meta.json` carry a `protocol` block that
names the track and says whether the run is canonical; any other configuration, order,
seed or environment setting is an ablation and the report labels it as such. A bare
size name such as `small` is refused: a configuration names its track.

## Quickstart

```bash
uv run continual-arc schedule make --config arc1-small --seed 0 --no-epilogue --out stream.json
uv run continual-arc schedule make --config arc1-small --seed 0 --test held-out --out held-out.json
uv run continual-arc schedule make --config arc1-small --seed 0 --test seen --out seen.json
uv run continual-arc schedule show stream.json --head 40
uv run continual-arc run --learner my_pkg.learners:MyLearner --schedule stream.json --out runs/my-stream
uv run continual-arc report runs/my-stream --plots
uv run continual-arc schedule make --config arc1-small --seed 0 --no-epilogue --reset-each --out stream-reset.json
uv run continual-arc run --learner my_pkg.learners:MyLearner --schedule stream-reset.json --out runs/my-stream-reset
uv run continual-arc learned runs/my-stream runs/my-stream-reset   # share learned over the stream
# each test played from the end of the stream, and the learner reset on it (the two frozen tests, above)
uv run continual-arc transfer runs/my-stream-held-out runs/my-reset-held-out   # the headline
uv run continual-arc transfer --test seen runs/my-stream-seen runs/my-reset-seen
```

## Writing a learner

```python
from continual_arc.env import LearnerObservation, Submit, RequestDemos, GiveUp
from continual_arc.learner import BaseLearner

class MyLearner(BaseLearner):
    def reset(self):                          # wipe persistent state
        ...
    def act(self, obs: LearnerObservation):   # obs.task_id (an alias), obs.input, obs.attempts_used, obs.demo_cap, ...
        return RequestDemos()                 # or Submit(grid) or GiveUp()
    def observe(self, obs, action, feedback):
        ...                                   # SubmitFeedback / DemosFeedback / GiveUpFeedback
    def end_round(self):                      # what it carries between instances changes only here
        ...
    def resources(self):                      # cumulative counters, logged per instance
        return {"tokens": ..., "state_bytes": ...}
```

The runner shows a learner only the instance (`LearnerObservation`): the task under a
name drawn for the run, the input, its own attempts and demonstration requests against
their caps, and its place in its own work (`step`, `round`, `user`), which it could count
itself. It is told nothing about the schedule: not the phase, how often the task has come before,
whether its state was wiped, or how long the stream is. The name is an alias of the
task's id (`task-3f2a9c1e`, or a codename with `--alias-style names`), salted per run
in the run directory's `alias-salt`, because the real id is a lookup key: a learner
with no model that runs the Re-ARC verifier named by the id, and finds each held-out
input among the public evaluation pairs, would use none of the `arc1-small` budget. The
records keep the real ids. Every learner gets this view, the baselines' own adapters
and trained rows included; `runner.run` takes the name style as `alias_style`.

The runner logs per-instance records (`records.jsonl`) and a summary
(`summary.json`) with the budget used, acquisition cost, retention by gap,
cost by exposure index, the learning curve, and the budget used on each test.


## What a learner may use

The stream is public and deterministic, so a learner could be built to know it. A
result is verified only once we have rerun it from its code and read the code
against these rules:

* **Nothing about ARC brought in.** No ARC, Re-ARC, ARC-GEN or other task data,
  answers or solvers inside the learner, its prompts or its starting memory, and no
  rebuilding of the schedule. What a model learned in pretraining is its prior, which
  the learner's own reset on the same model already carries.
* **Every model named.** Every model the learner calls is named in the submission and
  reached through the tracking proxy of
  [`baselines/`](baselines). The
  share learned and T read the learner against itself reset, on the same models. A
  model that only teaches it, such as a
  stronger model labeling its games for training, is part of its training: those calls
  count in its training compute, shown beside the score.
* **Learning only from what it was shown.** A learner learns from its observations and
  feedback. The records, with their real ids and exposures, are the benchmark's.
* **Anything else is allowed:** memory, programs, tools, a cache, training the
  weights. Compute is logged and shown beside the score, never in it, and a learner
  that trains reports its training compute apart from what it costs to serve an
  instance.

The alias stops the free lookup and a model's recall of a famous task id; it makes a
determined lookup cost a demonstration or two per task, since a learner can still
match a bought pair against every solver. The rules and the rerun stop that, and a
verified result is rerun on a held-out stream of new rules, which no lookup reaches
([verified results](#verified-results)).

## Plug-ins

A plug-in implements `continual_arc.addons.Plugin`. The runner calls hooks at run
start and end, episode start and end, every observation, every action, and every
feedback response. A hook can return a `PluginResponse` to transform the value
seen by the next plug-in or by the learner, enqueue auxiliary episodes, attach
diagnostic metadata, emit metrics and events, or stop after the current episode.

The environment still receives the final action and computes the result from its
own task data. Plug-ins cannot change the score directly. Auxiliary episodes have
`episode_type: "addon"` and appear in the separate `addons` summary block, while
the original schedule and primary benchmark metrics remain separate.

`Addon` and `BaseAddon` remain available for older episode-completion add-ons.
They are adapted automatically by `PluginManager`. New interventions should use
`Plugin` or `BasePlugin`, and multiple plug-ins can be passed through the
runner's `plugins` argument.

The command-line runner accepts repeatable `--plugin` and `--plugin-kwargs`
arguments. The arguments are matched by position:

```bash
uv run continual-arc run \
  --learner my_pkg.learners:MyLearner \
  --schedule stream.json \
  --plugin my_pkg.plugins:ReviewPlugin \
  --plugin-kwargs '{"initial_interval":8}'
```

## Layout

```
src/continual_arc/
  grids.py       grid validation and equality
  data.py        pinned fetch of each track's upstreams: Re-ARC and ARC-AGI-1; ARC-AGI-2, ARC-GEN and ARC-AGI-1; ARC-AGI-3's games
  rearc.py       seeded, verified access to the Re-ARC generators
  arcgen.py      seeded access to ARC-GEN's generators and the screened ARC-AGI-2 pool (arcgen_screen.json)
  tasks.py       task sources (generative Re-ARC and ARC-GEN, fixed ARC JSON) and registry
  arc3/          the ARC-AGI-3 track: the pinned games, a generator and solver per game, levels served fresh by Arc3Env
  schedule.py    Slot/Schedule, the one stream generator and its order presets, the phases and tests
  configs.py     named configurations, build_schedule() and test_schedule()
  env.py         the environment: observations (the slot's, and the learner's view of the instance), actions, feedback, cost accounting
  aliases.py     the per-run names a learner sees for tasks
  addons.py      plug-in protocols, composition, and intervention runtime
  learner.py     the learner protocol
  runner.py      run a learner over a schedule, write a run directory
  metrics.py     the budget used, transfer on the frozen tests, the share learned and the diagnostics
  report.py      text reports and plots
  cli.py         continual-arc command line
  testing.py     data-free task source and trivial learners for tests
docs/DESIGN.md   design rationale and decisions
baselines/       the learners the paper evaluates, its harness adapters and the tracking proxy (baselines/README.md)
training/        the training of the paper's trained rows on Qwen3.5-9B (training/README.md)
```

## Submitting a result

The [leaderboard](https://unify.ai/continual-arc/leaderboard) opens with the paper's runs and takes anyone's. A result
on the public stream is taken as reported. It is marked verified once we have rerun its code on the held-out rules,
which nobody outside unify has seen ([verified results](#verified-results)).

1. Run your learner on the track's canonical protocol at seed 0 (`arc1-small`): its stream with
   `continual-arc run`, then the two frozen tests from where the stream left it, and your learner reset: on the same
   stream, wiped before every instance (`schedule make --reset-each`, [the share learned](#the-share-learned)), and
   before each instance of the tests ([the two frozen tests](#the-two-frozen-tests)). To drive an agent harness, start
   from the adapters
   and the tracking proxy in [`baselines/`](baselines), which count
   every model call whatever the harness does. Read [what a learner may use](#what-a-learner-may-use) first.
2. Unless the leaderboard already has it, run the bare model you play with on the two tests, one instance at a time,
   from the baselines (`continual_arc_baselines.systems.bare.BareModelLearner`): two cheap runs, shown beside your T,
   which is flagged if your reset plays below them. The bare model's tests of every model the paper ran are in the
   run records.
3. Copy every run directory the runner wrote to `results/<track>/<name>/`, and add a `submission.json` to your
   learner's stream, naming your reset's stream and each test's three runs:

   ```json
   {
     "name": "my-learner, as shipped",
     "learner": "my-learner 1.4.0, with its memory switched on",
     "model": "GPT-6 Luna, medium effort",
     "code": "https://github.com/you/my-learner",
     "contact": "you@example.com",
     "reset": "results/arc1/my-learner-stream-reset",
     "tests": {
       "held-out": {"run": "results/arc1/my-learner-held-out", "reset": "results/arc1/my-learner-reset-held-out",
                    "bare": "results/arc1/bare-luna6-medium-held-out"},
       "seen": {"run": "results/arc1/my-learner-seen", "reset": "results/arc1/my-learner-reset-seen",
                "bare": "results/arc1/bare-luna6-medium-seen"}
     },
     "notes": "anything a reader needs to know to read the number"
   }
   ```

4. Check it: `uv run python scripts/check_result.py results/<track>/<name>`. It rebuilds every run's schedule from its
   configuration, order or test, and seed, checks that its records cover every slot and that its summary is the one
   the records give, that your tests were played under your stream's task names, that your reset is a run of your
   learner and the bare runs are the bare model's, and prints the numbers the leaderboard will show, T on the held-out
   test first, with the flag if your reset plays below the bare model.
5. Open a pull request. Questions go to this repository's issues.

### Verified results

We keep a held-out season of new rules: 50 ARC-style rules nobody outside unify has seen, each a generator of fresh
instances, checked against every public ARC task: 25 streamed over 200 instances, as `arc1-small` streams its 25
tasks, and 25 held out. To have a learner verified on it:

1. Make an OpenRouter key for this run alone, with a credit limit (at most 1,000 US dollars; a GPT-6 Luna learner's
   run and its reset have cost tens of dollars), and keep OpenRouter's Private Input & Output Logging off on its
   account, since it would show you the held-out tasks; a run with it on is void.
2. Encrypt the key with our public key, so that only the private half, which never leaves Secret Manager, can read it:

   ```bash
   curl -sO https://raw.githubusercontent.com/unifyai/continual-arc/main/verify/unify-verify.pem
   printf %s "$OPENROUTER_KEY" | openssl pkeyutl -encrypt -pubin -inkey unify-verify.pem -pkeyopt rsa_padding_mode:oaep -pkeyopt rsa_oaep_md:sha256 | base64
   ```

3. Open an issue with the [Verify my learner](https://github.com/unifyai/continual-arc/issues/new?template=verify.yml)
   form: your public GitHub repository and the commit to run, the import path of your learner class (installable from
   the repository with pip, implementing `continual_arc.learner`, and built with no arguments, so its settings are its
   class's defaults), the OpenRouter model it plays with and its reasoning effort, any other models it calls, and the
   encrypted key. A harness adapter on the baselines' `SystemLearner` depends on
   `continual-arc-baselines @ git+https://github.com/unifyai/continual-arc@v0.5.0#subdirectory=baselines` alone,
   which brings the benchmark with it.

Within about a quarter of an hour the issue says the run has started. We install your repository and run your learner,
whole, inside the sandbox the paper's runs used, which reaches nothing but the tracking proxy: the proxy makes every
model call on your key, and your code never holds it. At a seed of its own, your learner plays the season's stream and
then its frozen held-out test from where the stream left it, one instance of each held-out rule. A harness on the
baselines' `SystemLearner`, which keeps everything as files and holds one conversation, plays them one at a time, as
`arc-baselines test` plays a harness's tests: before each one we stop it and put its state directory back as the
stream left it. Any other learner plays them in one round, every rule a user of its own, so it keeps each user's work
apart and changes what it keeps only once the round ends, as the learner protocol asks. Your learner reset plays the
season's stream wiped before every instance, for the share learned, and the same test instances one at a time, wiped
before each, for T; the bare model it plays with plays the test instances too, and a reset that plays below it is
flagged beside T. The issue then gets T on the held-out test with its 95 percent interval, the share learned on the held-out
stream with its interval, the learnable budget, the budget used, the share solved and what each run's model calls
cost, and the result joins the leaderboard as verified, beside your public-stream result if you sent one. We delete
the key on our side; delete it at OpenRouter too. Only the numbers come back: the records, and anything your learner
kept, stay with us, since they show the held-out tasks.

You pay for your model calls, on your key; we pay for the machine. A run takes hours, like one on the public stream.
Learners that train their weights are not verified for now.

## Run records and trajectories

The [v0.1.0 release](https://github.com/unifyai/continual-arc/releases/tag/v0.1.0) carries the records of the 198
runs of the paper's GPT-6 Luna grid, each in the layout above, and the trajectories behind them,
`continual-arc-trajectories-v0.1.0.tar.gz`: every message the benchmark sent and every reply, the harnesses'
reasoning summaries and tool calls, and what each learner kept, file by file. Both are under CC BY 4.0, and any run
can be replayed instance by instance at [unify.ai/continual-arc/runs](https://unify.ai/continual-arc/runs). They were
played before the frozen tests, so each stream ends on a learning-on epilogue of 25 held-out tasks, as many as
`arc1-small` had until 20 October 2026, which no measure reads now. Each run rebuilds slot for slot from its own
configuration, and every one of their streams is the one `arc1-small` builds today.

## Citation

```bibtex
@article{lenton2026continualarc,
  title   = {Continual-ARC: Measuring Skill Acquisition over a Working Lifetime},
  author  = {Lenton, Daniel},
  note    = {Preprint},
  year    = {2026}
}
```

## License

Apache-2.0. Re-ARC is MIT and ARC-AGI is Apache-2.0; both are fetched, not vendored.

## About

Continual-ARC is made by [unify](https://unify.ai). unify is building continual learning for the agents companies
have already built, training the model to drive their own harness, so the agent gets faster and cheaper the more it
runs, on a model they own. We publish the research behind it.
