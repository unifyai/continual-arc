<!-- ARXIV_ID_PENDING: replace every ARXIV_ID_PENDING in this file with the paper's arXiv id once the listing is out (Monday 19 October 2026, 20:00 ET), then delete this line -->
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
  <a href="https://github.com/unifyai/continual-arc-baselines">Baselines</a>
</p>

# Continual-ARC

ARC-AGI measures how efficiently a system picks up a new skill from a few examples, but it serves each task once.
Continual-ARC serves a lifelong stream of tasks from the three ARC-AGI editions instead, in which the same tasks return
after dormant gaps with fresh instances while new tasks keep arriving. On each instance the learner may pay for one
more demonstration or submit an answer, and it may try again after a wrong one. A wrong answer and a demonstration
each cost one, against a cap of sixteen, and we score the stream by the share of that budget it used, so lower is
better. A learner may keep anything at all between instances, which puts an agent harness keeping notes and a model
updating its weights on the same scale.

This repository is the benchmark of the paper [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://unify.ai/blog/why-we-built-continual-arc): the environment, the schedules, the three tracks and the measures.
The learners the paper evaluates, adapters for three agent harnesses among them, are in
[continual-arc-baselines](https://github.com/unifyai/continual-arc-baselines). The records of every run it reports are
attached to the [v0.1.0 release](https://github.com/unifyai/continual-arc/releases/tag/v0.1.0), and every task,
figure, run and trajectory of the paper can be explored and replayed at
[unify.ai/continual-arc](https://unify.ai/continual-arc).

In the paper we ran Prime Agent, Hermes Agent and OpenClaw through the ARC-AGI-1 track with every learning mechanism
they ship switched on, and again with everything they wrote deleted before every instance. Their persistence saved no
measurable budget at any reasoning effort, and left to decide what to keep, they kept almost nothing that outlasted an
instance. A deliberately naive program library, which keeps one tested program per task and runs it when the task
returns, used 13 to 30 points less of the budget than every harness at every effort. Run your own learner through it
and [submit the result](#submitting-a-result).

* **Any learner.** The learner is a black box with four methods. Weight updates, external memory, program libraries
  and plain prompting are all welcome.
* **Binary feedback.** Every submission is answered correct or incorrect, as ARC Prize answers it.
* **Demonstrations cost.** One more input-output pair of the task is available on request, charged like a wrong
  attempt.
* **Fresh instances.** On ARC-AGI-1 every test input and every demonstration is freshly drawn from the task's Re-ARC
  generator, and on ARC-AGI-2 each visit of a task serves its own pairs in a frame no earlier visit used. A cached
  answer never fits the next grid, and caching a demonstration is a choice with a price.
* **Composable interventions.** Optional plug-ins can insert tagged auxiliary episodes without changing the
  schedule's exposure or gap metadata.
* **One number per track.** The share of the budget used over the track's stream, with the share solved beside it.
  Compute, tokens and memory are logged, not scored.

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
| `RequestDemos()`| returns one more demonstration pair, up to eight per instance (a ninth request is refused at no cost): a fresh Re-ARC pair on ARC-AGI-1; on a task without generators (the ARC-AGI-2 track and both epilogues) one of the task's own pairs, in the test's frame, and nothing after the last of them (the instance's demo cap is the pairs it has, and the observation shows it) | `demo_cost` (default 1) |
| `GiveUp()`      | instance ends as failed                                                  | remaining attempts up to the cap |

An instance that hits the attempt cap or is given up is charged all its attempts, as well as the
demonstrations it bought. The budget used is `100 * mean_cost / max_cost`, with
`max_cost = attempt_cap + demo_cost * demo_cap`, 8 + 8 = 16 on the grid tracks, and lower is better; the
`score` in `summary.json` and the report is its complement, `100 - budget used`.

## Modes

All modes are schedules over the same seeded task set, so results are comparable. One
process generates every stream (`stream_schedule`): a set of active tasks, each worked in
a spell, resting between spells, new tasks arriving over time. The modes are named
settings of its parameters, from blocked to structureless; a mode is an order of
presentation and never touches the learner's state.

| mode          | parameters                                                              | what the learner sees                                                    |
|---------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `sequential`  | one active task, one spell of all its instances, no return               | one task at a time, every instance of it, then the next (blocked)         |
| `workstreams` | a few active, exponential bursts, log-normal dormancy, Poisson arrivals  | bursts of a few tasks; tasks go dormant and return; new tasks keep arriving (canonical) |
| `random`      | every task active, spells never end                                      | a uniform draw over all tasks at every step (no temporal structure)       |

Each mode ends with an **epilogue** of never-seen evaluation tasks, once each, with
nothing reset. `fresh` is not a mode but the transfer reference: the epilogue tasks alone,
met by a learner whose state is wiped before each, so that forward transfer is the epilogue
cost after a lifetime minus the cost with no lifetime behind it.

## Regimes

A configuration is served in one of two regimes, and they differ in one number, the
configuration's `users`: how many people the system serves at once, and so how much experience
arrives between two updates. Any learner can run at any number of users, whether it keeps notes
or trains its weights, and learners are compared only at the same number.

| regime       | users    | picture                                                                 |
|--------------|----------|-------------------------------------------------------------------------|
| `individual` | 1        | one system walking one person's stream, one instance at a time (every config but one) |
| `collective` | hundreds | one system serving a team at once and learning from all of them (`arc1-collective`: 100 users, 200 rounds) |

In the collective regime every user walks a stream of their own over the same tasks (the
same process as the individual regime, each user their own draw of it; user 0 at seed `s`
walks exactly the individual stream at seed `s`) and then meets the held-out tasks in an
order of their own. The users are deliberately concurrent: a **round** is one instance per
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
task in the same round share both), and each record also carries the user's own count
(`user_exposure_index`, `user_gap`); with one user the two agree, the times a task was
seen before and the steps since it was last seen. A round never spans a phase, so the
epilogue starts on a round boundary. `records.jsonl` only ever holds whole rounds, and a
resumed run replays the interrupted round from its start. Add-ons and ARC-AGI-3 games are
served one instance at a time and need one user.

## Cutting a stream short, and reading its epilogue early

A collective configuration's own epilogue only comes after every user has finished the whole
stream (`arc1-collective`: round 200), which is too much to pay just to read transfer at an
earlier round. Two schedules serve that instead, both built from the same configuration and
seed so their task sets and instance seeds agree with the full schedule's:

* **`Schedule.first_rounds(r)`** is a literal prefix of a schedule's own slots: rounds 0 to
  `r - 1`, the same tasks, instance seeds and users as the full schedule. It is a slice, not a
  rebuild, because the stream process gives every task a supply fixed by the configuration's
  full length (`even_visits`), so regenerating a shorter stream from the same seed would not be
  this one's prefix. A resumed run checks its schedule's prefix slot by slot
  (`runner.run(resume=True)`), so a run stopped under `first_rounds(20)` resumes without
  replay under `first_rounds(60)`, or under the uncut schedule.
* **`epilogue_for_users(config, seed, users)`** is the held-out tasks alone, for the first
  `users` of the configuration's users, as a `users`-wide collective schedule: user `u` meets
  them in the exact order and instance seeds the full schedule's own epilogue would give them.
  The learner is not reset between its tasks (nothing is frozen for the epilogue). It needs
  none of the stream built first, only the configuration and seed.

The CLI exposes both from `schedule make`:

```bash
# the first 20 rounds of the collective stream (a cut, not the epilogue)
continual-arc schedule make --config arc1-collective --mode workstreams --seed 0 --rounds 20 --out stream-r20.json

# the held-out tasks alone, for users 0-9 of the 100 (no --mode: there is no stream here)
continual-arc schedule make --config arc1-collective --seed 0 --epilogue-users 10 --out epilogue-10.json
```

Restoring the learner's state at round 20 for the epilogue run is the driver's job, not the
benchmark's: run the cut stream, save whatever the learner carries, load it back before playing
the epilogue schedule. `protocol_of` lists both as deviations from the canonical protocol
("stream cut at round 20", "epilogue alone, users 0–9 of 100"); `meta.config` is kept on both,
so the label stays readable.

## Tracks and the canonical number

Continual-ARC is three tracks, one per ARC-AGI edition, and no track is the default.
Each edition gives signal for a different class of model (ARC-AGI-1 for small open
models, ARC-AGI-2 for frontier models, ARC-AGI-3 for interactive skill), so the tracks
are not commensurable and there is no pooled or averaged score across them. Every
configuration name carries its track:

| track | configs | stream | epilogue |
|---|---|---|---|
| `arc1` | `arc1-smoke`, `arc1-small`, `arc1-medium`, `arc1-full`, `arc1-collective` (100 users) | Re-ARC generators over the ARC-AGI-1 training tasks | ARC-AGI-1 evaluation tasks |
| `arc2` | `arc2-smoke`, `arc2-small` | a draw from the 100 of the 114 ARC-AGI-2 evaluation tasks new to the second edition whose labels allow eight distinct frames, each visit of a task in a frame of its own, drawn only from the augmentations the task's rule survives | the rest of the same 114, held out, unaugmented |
| `arc3` | `arc3-smoke`, `arc3-small` | ARC-AGI-3 public games, one level per instance | held-out games |

Each track has exactly one canonical number: the budget used on its `small` configuration in
workstreams mode with the epilogue, at the default environment settings. Seeds 0, 1 and 2 are all
canonical; the paper reports seed 0, and the leaderboard compares runs at seed 0. Every run's `summary.json` and `meta.json` carry a `protocol` block that
names the track and says whether the run is canonical; any other configuration, mode,
seed or environment setting is an ablation and the report labels it as such. A bare
size name such as `small` is refused, because that would make one track the default.

**ARC-AGI-2.** `arc2-smoke` and `arc2-small` have the same shapes as the `arc1` configs
over the second edition. Stream and epilogue both draw from the 114 evaluation tasks new
to ARC-AGI-2 (`continual_arc.data.arc2_new_task_ids`), the stream's tasks removed for the
epilogue. The public training set is not used: 767 of its 1000 tasks are ARC-AGI-1 tasks
(a stream drawn from it was an ARC-AGI-1 stream in practice, solved at first sight by a
model at 18% on the ARC-AGI-2 board) and its difficulty is uncalibrated, while the
evaluation set is what the leaderboard is calibrated against, so the board's number is
the right reference for this stream. ARC-AGI-2 has no generators, so the stream serves each task's public
pairs under seeded augmentations, the same for the test pair and its demos within an
instance. Each task is drawn only under the symmetries of the square its rule is unchanged
by, relabelling only the colours with no fixed role in it, from per-task labels of all 114
tasks (`continual_arc/arc2_invariances.json`, two independent labelling passes merged
conservatively), so the rule a learner kept from one instance is the rule of the next; the
k-th visit of a task is served in the k-th of its distinct frames, so no visit shows a grid
an earlier visit showed. The stream draws only the 100 tasks with at least eight distinct
frames (`configs.arc2_stream_pool`); the epilogue holds out any of the 114, unaugmented. An
instance serves only the task's own pairs. `continual-arc data fetch` pins ARC-AGI-2 too (and ARC-AGI-1, to know which ids to exclude).

**ARC-AGI-3.** `arc3-smoke` and `arc3-small` run the same modes over the
25 public ARC-AGI-3 games (interactive, 64x64 frames, seven actions plus a level
reset), pinned by version and run locally through the `arc-agi` toolkit
(`pip install "continual-arc[arc3]"`, Python 3.12 or newer, then
`continual-arc data fetch --datasets arc3`). A task is a game and an instance is one
of its levels: successive exposures climb through the levels and wrap, so recurrence
tests whether the skill was kept rather than whether an output was cached (levels are
fixed; there are no fresh inputs). The learner sends `Act(action, x, y)`,
`ResetLevel()` or `GiveUp()` and sees the frames, the game state and the available
actions. Cost of a solved level is actions plus resets divided by the published human
baseline for that level; a level that hits its action cap
(`cap_multiplier` times the baseline, at most `max_actions`) or is given up costs
`cap_multiplier`, which is also `max_cost`. The epilogue holds out games the stream
never selected. See `continual_arc/arc3/`.

## Quickstart

```bash
uv run continual-arc schedule make --config arc1-small --mode workstreams --seed 0 --out ws.json
uv run continual-arc schedule show ws.json --head 40
uv run continual-arc run --learner my_pkg.learners:MyLearner --schedule ws.json --out runs/my-ws
uv run continual-arc report runs/my-ws --plots
uv run continual-arc compare runs/my-ws runs/my-fresh        # forward transfer
```

## Writing a learner

```python
from continual_arc.env import Observation, Submit, RequestDemos, GiveUp
from continual_arc.learner import BaseLearner

class MyLearner(BaseLearner):
    def reset(self):                       # wipe persistent state (the fresh reference)
        ...
    def act(self, obs: Observation):       # obs.task_id, obs.input, obs.attempts_used, ...
        return RequestDemos()              # or Submit(grid) or GiveUp()
    def observe(self, obs, action, feedback):
        ...                                # SubmitFeedback / DemosFeedback / GiveUpFeedback
    def resources(self):                   # cumulative counters, logged per instance
        return {"tokens": ..., "state_bytes": ...}
```

The runner logs per-instance records (`records.jsonl`) and a summary
(`summary.json`) with the headline score, acquisition cost, retention by gap,
cost by exposure index, the learning curve, and the epilogue.

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
  --schedule ws.json \
  --plugin my_pkg.plugins:ReviewPlugin \
  --plugin-kwargs '{"initial_interval":8}'
```

## Layout

```
src/continual_arc/
  grids.py       grid validation and equality
  data.py        pinned fetch of Re-ARC and ARC-AGI-1
  rearc.py       seeded, verified access to the Re-ARC generators
  tasks.py       task sources (generative Re-ARC, fixed ARC JSON) and registry
  schedule.py    Slot/Schedule, the one stream generator and its mode presets, the epilogue
  configs.py     named configurations and build_schedule()
  env.py         the environment: observations, actions, feedback, cost accounting
  addons.py      plug-in protocols, composition, and intervention runtime
  arc3/          ARC-AGI-3: pinned game manifest, local game handle, and the level-per-instance environment
  learner.py     the learner protocol
  runner.py      run a learner over a schedule, write a run directory
  metrics.py     score and diagnostics
  report.py      text reports and plots
  cli.py         continual-arc command line
  testing.py     data-free task source and trivial learners for tests
docs/DESIGN.md   design rationale and decisions
```

## Submitting a result

The [leaderboard](https://unify.ai/continual-arc/leaderboard) opens with the paper's runs and takes anyone's. There is
no held-out set: the stream is public, every schedule is deterministic, and a result is taken as reported until we
rerun it, when it is marked as verified.

1. Run your learner on a track's canonical protocol, its `small` configuration in workstreams order at seed 0
   (`arc1-small`, `arc2-small` or `arc3-small`), and on the same configuration's `fresh` reference if you want your
   forward transfer read. To drive an agent harness, start from the adapters and the tracking proxy in
   [continual-arc-baselines](https://github.com/unifyai/continual-arc-baselines), which count every model call
   whatever the harness does.
2. Copy the run directory the runner wrote to `results/<track>/<name>/`, and add a `submission.json`:

   ```json
   {
     "name": "my-harness, as shipped",
     "learner": "my-harness 1.4.0, with its memory and skills switched on",
     "model": "GPT-6 Luna, medium effort",
     "code": "https://github.com/you/my-harness",
     "contact": "you@example.com",
     "notes": "anything a reader needs to know to read the number"
   }
   ```

3. Check it: `uv run python scripts/check_result.py results/<track>/<name>`. It rebuilds the schedule from the run's
   configuration, order and seed, checks that the records cover every slot in order and that the summary is the one
   the records give, and prints the numbers the leaderboard will show.
4. Open a pull request. Questions go to this repository's issues.

## Run records and trajectories

The [v0.1.0 release](https://github.com/unifyai/continual-arc/releases/tag/v0.1.0) carries the records of the 257
runs the paper reports, each in the layout above, and the trajectories behind them,
`continual-arc-trajectories-v0.1.0.tar.gz`: every message the benchmark sent and every reply, the harnesses'
reasoning summaries and tool calls, and what each learner kept, file by file. Both are under CC BY 4.0, and any run
can be replayed instance by instance at [unify.ai/continual-arc/runs](https://unify.ai/continual-arc/runs).

## Citation

```bibtex
@article{lenton2026continualarc,
  title   = {Continual-ARC: Measuring Skill Acquisition over a Working Lifetime},
  author  = {Lenton, Daniel},
  note    = {Preprint},
  year    = {2026}
}
```

## Licence

Apache-2.0. Re-ARC is MIT and ARC-AGI is Apache-2.0; both are fetched, not vendored.

## About

Continual-ARC is made by [unify](https://unify.ai). unify is building continual learning for the agents companies
have already built, training the model to drive their own harness, so the agent gets faster and cheaper the more it
runs, on a model they own. We publish the research behind it.
