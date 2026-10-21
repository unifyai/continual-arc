# Continual-ARC: design

## Why

ARC-AGI-1 tasks are out of distribution for all but frontier models, have a clean
binary reward, are easy for people and hard for models, and are abstract enough
that progress should transfer beyond any one vertical. Every existing continual
learning benchmark for agents (CL-Bench, SkillLearnBench, AgentCL, SWE-Bench-CL,
LifelongAgentBench) is built from hand-authored real-world tasks, so each has tens
to a few hundred instances and no way to make more. Re-ARC gives unbounded fresh
instances per task, which makes a long, recurring stream possible and makes
caching outputs pointless.

## Decisions

| decision | choice | why |
|---|---|---|
| base data | ARC-AGI-1 training tasks via Re-ARC; ARC-AGI-1 evaluation tasks for the held-out test | generators exist for all 400 training tasks; the evaluation set is genuinely held out |
| learners | any; four-method protocol | the point is a fair arena for memory-based and gradient-based methods |
| feedback | correct/incorrect only | matches ARC Prize scoring; keeps it close to RL |
| attempts | unlimited up to a cap; cost = wrong attempts | retention shows up as fewer attempts on recurrence |
| task identity | task id always exposed | isolates memory from recognition; recognition can be a later track |
| demos | on request, at a cost equal to one wrong attempt | makes remembering an economic decision |
| score | mean cost per instance = wrong attempts + demo requests | one number; compute/tokens/memory are plotted, not scored |
| headline | T on the held-out test: the budget a learner saves there against the stronger of itself reset and the bare model on the same instances; T on the seen test and the share learned on the stream beside it (20 and 21 Oct 2026, below) | no cache helps on a task never seen, both references play the learner's own model, and a learner cannot gain by playing dumb with an empty memory |
| track | one, ARC-AGI-1 (`arc1`), named in every config (`arc1-small`); a bare size names nothing | its Re-ARC generators give every stream task fresh, verified instances without end, so a return is never a lookup; one track gives a learner one number, and the held-out season that verifies results streams as `arc1-small` does. ARC-AGI-2 has no generators, so its stream rested on augmentation, and ARC-AGI-3 serves levels, not fresh instances of a rule (Dan, 19 Oct 2026) |
| canonical protocol | the `arc1-small` config's stream in workstreams mode, then its two frozen tests, default environment settings, seeds 0 to 2; everything else is an ablation and is labelled so in run metadata | researchers need one number to quote, and the knobs must not let a non-standard run pass as it |
| stream shape | bursty workstreams over a growing task set | mimics an office worker; tests retention across long gaps and re-activation |
| frozen tests | after the stream, two tests, each one round in which every instance is a user of its own from the learner's state at the stream's end: the held-out test, one instance of each of 100 never-seen evaluation tasks on `arc1-small` (25 until 20 Oct 2026), and the seen test, four fresh instances of each of the stream's 25 tasks (21 Oct 2026, below) | the held-out test reads generalisation and the seen test memorisation; a test during which the learner keeps learning would mix what it built on the stream with what it builds on the test |
| add-ons | optional run-time auxiliary episodes, tagged separately | permits adaptive interventions such as spaced review without making them part of the learner protocol |

## What a learner is shown

The task id, the input and the instance's own counters, and nothing about the stream: no position, no
length, no phase and no exposure count. Neither test is announced: a test instance looks like a stream instance.
Why (Dan, 20 Sep 2026): the stream simulates a new worker in an office, who is not told they are in a
fixed-length evaluation; showing the position invited end-game behaviour, and an exposure count did
the learner's remembering for it. Whether a task was met before is for the learner's own memory to say.

`Observation` still carries `step`, `n_steps`, `phase` and `exposure_index`: the runner and the
adapters key their bookkeeping on them, and the diagnostics use exposure from the schedule. The
adapters in `continual-arc-baselines` render prompts without them (`systems/protocol.py`), but nothing
in this package stops a Python learner from reading them.

## Modes as schedules

Every mode is a schedule and an order of presentation, nothing else: it never touches the learner's
state, and the environment does not know which mode it is serving. One process generates every stream
(`stream_schedule`, `StreamParams`). Up to `concurrency` tasks are active and each step serves one of
them, drawn uniformly; an active task runs for a spell (exponential with mean `burst_mean`, rounded to
whole instances), then rests for a log-normal number of steps (median `dormancy_median`, sigma
`dormancy_sigma`); never-seen tasks arrive as a Poisson process (mean interval `arrival_interval`) and
take priority for a free slot, so the seen set grows through the stream. The modes are named settings
of the process, from blocked to structureless:

* `sequential`: one task active, one spell of its whole supply, no return (blocked practice).
* `workstreams`: the process's defaults, a few tasks active in bursts, dormant between them, new
  tasks arriving; the mode of the canonical number.
* `random`: every task active from the first step, spells that never end (the i.i.d. control; a
  system's workstreams score minus its random score is what burstiness and dormancy cost it).

Every configuration gives each task the same supply of instances and every mode serves exactly that
supply, so the modes are one multiset of instances in three orders. Every mode can end with the same epilogue, the
held-out tasks once each, nothing reset, which no measure reads since the frozen tests. `fresh` is not a mode but the
epilogue alone, met by a learner whose state is wiped before each held-out task, which can stand in for the held-out
test's reset reference.

Why these three (Dan, 21 Sep 2026): the first design's `isolated` mode wiped the learner's state at task
boundaries, an intervention on the system dressed as a schedule, with no analogue in any job; it became
`sequential` with state kept, and the transfer reference became a run of its own. `batch` (every task
live from the first step) was an epoch, a training-loop artefact whose constant gap cannot produce the
retention-by-gap curve; `random` replaced it. The workstreams streams are frozen by
`tests/fixtures/stream-golden.json`, because running cells were launched on those exact orders.

## Metrics

* headline, T on the held-out test: `min(U(reset), U(floor)) − U(run)` over the test's instances, where U is the
  budget used, `100 · mean_cost / max_cost`, `run` is the learner's own frozen test, `reset` the same learner reset
  before each instance and `floor` the bare model (the stateless floor), each played on the same instances apart from
  the run's own test, with a bootstrap over tasks that chooses the reference again in each draw and the
  share solved of all three beside it (`metrics.transfer`, `continual-arc transfer`); the seen test is read the same
  way (`--test seen`)
* score: `100 · (1 − mean_cost / max_cost)`, `max_cost = attempt_cap + demo_cost · demo_cap`
* cost of an instance: its wrong attempts plus `demo_cost` per demo request when it is solved, and `max_cost` when it
  is not. Until 20 Oct 2026 an unsolved instance was charged the attempt cap plus its demo requests, so failing
  without asking cost 8 and asking for every demonstration before failing cost 16: a learner that never asked and
  never solved scored 50, and one that asked and still failed scored 0. Charging every failure the whole budget makes
  failing never cheaper than trying, and a demonstration paid for only by an instance it helped to solve.
  `continual-arc rescore` charges a run written before then this way.
* acquisition: mean cost on exposure index 0, stream phases only
* retention: mean cost on re-exposure bucketed by gap (1-10, 11-30, 31-100, 101-300, 301+)
* exposure curve: mean cost by exposure index (0, 1, 2, 3-5, 6-10, 11+)
* learning curve: cumulative and windowed mean cost
* share learned: the share of the learnable budget a run saves on the stream beyond a cache of its
  demonstrations, against the demo-cache floor on the same model and stream (`metrics.share_learned`)
* forward transfer against one reference: reference cost − run cost on a frozen test, matched by task and seed
  (`metrics.forward_transfer`, `continual-arc compare`)

Add-on episodes are included in total resource accounting and exposed in
`by_phase`, but primary benchmark metrics exclude them. The original schedule's
exposure indices and gaps are preserved when an add-on inserts an episode. An
add-on may schedule a fresh task instance, but it must not inspect or mutate
learner state; that boundary keeps interventions comparable across harnesses.

## Why transfer is the headline

Settled 20 Oct 2026, on 100 held-out tasks so that it can resolve a few points (25 tasks met once could not). The
share learned has two flaws that model progress makes worse. Each stream task recurs under its alias, and its
demonstrations hold everything needed to solve it, so writing one program per alias and keeping it is close to
optimal: the naive program library saves about two thirds of the learnable budget at every effort on GPT-6 Luna. And
a strong model leaves little to learn: the learnable budget falls from 38.6 points at low effort to 15.0 at high.
Transfer avoids both. A held-out task was never seen, so no cached program or answer helps on it, and both references
play the learner's own model, so the score is the learning, not the model.

A learner judged against its own reset copy alone could fake transfer by holding back whenever its memory is empty:
answer garbage until it has seen a stream, then try. That costs one stream instance and wins every comparison, since
the reset copy's memory is always empty. Against the stronger of the reset copy and the bare model, garbage gains
nothing. What is left is a learner that plays exactly as well as the bare model with an empty memory and uses its full
strategy after a stream: it is credited with its edge over the bare model on new tasks, and a design that keeps its
best strategy for after a warm-up earns the same without meaning to. On the public held-out tasks a learner could also
look answers up after its stream and play dumb before it; the private season has nothing to look up. A placebo
reference, the same learner after a stream of equal length that teaches nothing, would close the warm-up case too; it
was declined for now (20 Oct 2026), so verified runs rest on their public code and its review.

Settled 21 Oct 2026: both tests are frozen, and the learning-on epilogue they replace is discarded. A learner that
keeps learning while it meets 100 held-out tasks mixes what it built on the stream with what it builds on them, and
over 100 tasks the second can swamp the first; a frozen test reads the first alone. Each test is one round in which
every instance is a user of its own from the state at the stream's end, so it takes about the time of one instance. The
seen test is not a repeat of the stream: it serves instances the stream never served to a learner that can no longer
change, so it reads what was kept of each rule at the end, while the stream's share learned reads the same memory as
it is built. The stream is the learning curve and the tests are its end points; a test from an earlier round is
possible wherever that round's state was kept.

Transfer has a ceiling of its own: a model that already solves the held-out tasks with no stream behind it leaves
nothing to transfer to. When cheap models outgrow it, a later version is needed, not a different number.

## Plug-in interface

The runner accepts one or more objects implementing
`continual_arc.addons.Plugin`. `PluginManager` orders them by their optional
`priority` value and passes each hook the result produced by the previous plug-in.
This makes interventions composable without adding a harness-specific branch to
the benchmark.

```python
class Plugin(Protocol):
    name: str
    priority: int

    def on_run_start(self, context) -> PluginResponse | None: ...
    def on_episode_start(self, observation, context) -> PluginResponse | None: ...
    def on_observation(self, observation, context) -> PluginResponse | None: ...
    def on_action(self, observation, action, context) -> PluginResponse | None: ...
    def on_feedback(self, observation, action, feedback, context) -> PluginResponse | None: ...
    def on_episode_end(self, record, context) -> PluginResponse | None: ...
    def on_run_end(self, records, context) -> None: ...
    def summary(self) -> dict: ...
```

`PluginContext` is read-only. It contains the run id, original schedule,
environment configuration, current episode information, a benchmark-only clock,
and a `PluginRuntime`. The runtime lets a plug-in emit JSONL events, record
numeric metrics, and register artifact paths. It does not expose the learner,
its memory, harness processes, or weight files.

`PluginResponse` can replace the learner-facing observation, action, or feedback.
It can also return `EpisodeRequest` values, attach record metadata, and request a
stop after the current episode. Requests are queued until the episode finishes.
The benchmark still validates the final action, serves the task, computes raw
feedback, and writes the scored record. A plug-in that changes the learner view
or feedback is therefore an explicit intervention and must be named in the
experiment condition.

Plug-ins may implement only the hooks they need. `BasePlugin` supplies no-op
methods for the full interface, and `PluginManager` also accepts a plain object
with a subset of hook methods. This keeps small interventions small while
allowing more involved plug-ins to coordinate several lifecycle points.

An `EpisodeRequest` names a task, registered source, deterministic instance seed,
phase, owner, and metadata. The source may be registered before the run starts
and populated by a plug-in while the run is in progress; this supports
reproducible generated auxiliary tasks without putting their expected outputs in
the learner-facing observation. The environment inserts the request before the
next scheduled slot, or appends it after the schedule ends. Inserted episodes
are ordinary episodes from the learner's point of view. They are marked with
`episode_type: "addon"`, and their `Observation` includes that type and the
owning plug-in.

The environment keeps the request source alongside each inserted slot when it
writes an `InstanceRecord`. Thus dynamically registered sources have the same
provenance guarantees as the original schedule sources, while the original
schedule itself remains immutable.

Auxiliary episodes consume attempts, demo cost, model calls, and wall time.
`summary["benchmark"]` and the usual stream, retention, acquisition, and
epilogue metrics use only original benchmark records. `summary["addons"]` reports
auxiliary records, while `summary["overall"]` includes both. The original
schedule's exposure indices and gaps do not change when a plug-in inserts a
probe.

The old `Addon` and `BaseAddon` interfaces remain supported. They expose only
`on_instance_end` and are adapted as low-priority legacy plug-ins. New code
should use `BasePlugin`. A no-op plug-in is equivalent to omitting `plugins`, so
existing learners and runs keep their current behavior.

## Reference learners (run in `continual-arc-baselines`)

Two floors assume nothing beyond the interface, and every result is read against them:

1. Stateless: requests demonstrations on every instance and carries nothing between instances.
2. Demo cache: requests demonstrations once per task id and replays them to the stateless solver
   on every later instance of the task. Its cost at the last exposure equals its cost at the
   second, so it separates memory of a task from memory of its demonstrations; a system that has
   learned a task falls below it.

No floor may carry domain knowledge (Dan, 23 Sep 2026), so a program library, which assumes a task
is a pure function of its input and verifies code against the demonstrations, is a method under
test, not a floor. Harnesses, methods and weight-based learners are the standing experiments of
`continual-arc-baselines`.

## Open questions

* Task-agnostic track (no task id) once the task-aware version is established.
* Settled 27 Sep 2026: the ARC-AGI-1 epilogue serves only its tasks' own pairs too, so no track pads with
  transformed copies.
* Human baseline on a short workstreams schedule.
