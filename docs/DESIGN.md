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
| base data | ARC-AGI-1 training tasks via Re-ARC; ARC-AGI-1 evaluation tasks for the epilogue | generators exist for all 400 training tasks; the evaluation set is genuinely held out |
| learners | any; four-method protocol | the point is a fair arena for memory-based and gradient-based methods |
| feedback | correct/incorrect only | matches ARC Prize scoring; keeps it close to RL |
| attempts | unlimited up to a cap; cost = wrong attempts | retention shows up as fewer attempts on recurrence |
| task identity | task id always exposed | isolates memory from recognition; recognition can be a later track |
| demos | on request, at a cost equal to one wrong attempt | makes remembering an economic decision |
| score | mean cost per instance = wrong attempts + demo requests | one number; compute/tokens/memory are plotted, not scored |
| track | one, ARC-AGI-1 (`arc1`), named in every config (`arc1-small`); a bare size names nothing | its Re-ARC generators give every stream task fresh, verified instances without end, so a return is never a lookup; one track gives a learner one number, and the held-out season that verifies results is laid out as `arc1-small`. ARC-AGI-2 has no generators, so its stream rested on augmentation, and ARC-AGI-3 serves levels, not fresh instances of a rule (Dan, 19 Oct 2026) |
| canonical protocol | the `arc1-small` config, workstreams mode, epilogue on, default environment settings, mean over seeds 0 to 2; everything else is an ablation and is labelled so in run metadata | researchers need one number to quote, and the knobs must not let a non-standard run pass as it |
| stream shape | bursty workstreams over a growing task set | mimics an office worker; tests retention across long gaps and re-activation |
| epilogue | never-seen evaluation tasks after the stream | measures forward transfer, not just retention |
| add-ons | optional run-time auxiliary episodes, tagged separately | permits adaptive interventions such as spaced review without making them part of the learner protocol |

## What a learner is shown

The task id, the input and the instance's own counters, and nothing about the stream: no position, no
length, no phase and no exposure count. The epilogue is not announced and nothing is frozen for it.
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
supply, so the modes are one multiset of instances in three orders. Every mode ends with the same epilogue, the held-out tasks once each, nothing reset.
`fresh` is not a mode but the transfer reference: the epilogue alone, met by a learner whose state is
wiped before each held-out task.

Why these three (Dan, 21 Sep 2026): the first design's `isolated` mode wiped the learner's state at task
boundaries, an intervention on the system dressed as a schedule, with no analogue in any job; it became
`sequential` with state kept, and the transfer reference became a run of its own. `batch` (every task
live from the first step) was an epoch, a training-loop artefact whose constant gap cannot produce the
retention-by-gap curve; `random` replaced it. The workstreams streams are frozen by
`tests/fixtures/stream-golden.json`, because running cells were launched on those exact orders.

## Metrics

* headline: `score = 100 · (1 − mean_cost / max_cost)`, `max_cost = attempt_cap + demo_cost · demo_cap`
* acquisition: mean cost on exposure index 0, stream phases only
* retention: mean cost on re-exposure bucketed by gap (1-10, 11-30, 31-100, 101-300, 301+)
* exposure curve: mean cost by exposure index (0, 1, 2, 3-5, 6-10, 11+)
* learning curve: cumulative and windowed mean cost
* forward transfer: reference epilogue cost − run epilogue cost, matched by task and seed

Add-on episodes are included in total resource accounting and exposed in
`by_phase`, but primary benchmark metrics exclude them. The original schedule's
exposure indices and gaps are preserved when an add-on inserts an episode. An
add-on may schedule a fresh task instance, but it must not inspect or mutate
learner state; that boundary keeps interventions comparable across harnesses.

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
