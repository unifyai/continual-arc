# Continual-ARC: runs

The runs of the paper [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://arxiv.org/abs/2611.14505) (Daniel Lenton, Yusha Arif and Haris Mahmood, 2026;
[the post](https://unify.ai/blog/why-we-built-continual-arc)): its 228 runs on GPT-6 Luna on the ARC-AGI-1 track, of
which 120 are the ablation rows of its Appendix F, each with the two frozen tests played after its stream, and its
runs on the open models, Qwen3.5-9B and GLM-5.3 Flash (its Appendix H) and Kimi K3.1 Pro on the ARC-AGI-2 and
ARC-AGI-3 tracks (its Appendix K). For each GPT-6 Luna run there are its
records, from which the paper's tables and figures are computed, and its trajectories, of its stream and of every
play of its frozen tests: every message the benchmark sent, what the learner did inside each exchange (its reasoning
summaries, its messages and its tool calls with their output), its reply and the action read from it, and what the
learner kept, file by file and version by version. The
open-model runs are here by their records and their trajectories: each in `runs/` once its stream and both its tests
have finished, listed in `open-models.json`, and each still being played in `runs/live/`, with what it has played so
far. The runs the
paper's text reports by machine, those of the search that chose DAgger's configuration and of the other weight-based
methods' attempts, are in `runs/arc1/<machine>/`, listed in `machines.json`, and the first-sight readings of its
Appendix K in `runs/first-sight/`. Not here are
the three OpenClaw cells of the paper's Appendix E that played with memory search set, under the earlier charge,
whose records no copy holds, and the trajectories of 112 of the open-model runs' frozen-test plays, whose users' logs
no copy holds (below).

The GPT-6 Luna runs were written by the benchmark in this repository, running the learners in its
[`baselines/`](../baselines), whose `scripts/paper-grid.txt` launches each of them again. Of the 228, the site lists and replays the 114 that played the
track's own order or the held-out tasks fresh (`ws` and `fresh`, below), instance by instance, at
[continual-arc.unify.ai/runs](https://continual-arc.unify.ai/runs), where each instance links to its file here; the
other 114 are the paper's side experiment on the order (`seq` and `rnd`, its Appendix B), played outside the benchmark,
and are in these files alone.

These are the files of the benchmark's [v0.1.0 release](https://github.com/unifyai/continual-arc/releases/tag/v0.1.0)
assets, `continual-arc-runs-v0.1.0.tar.gz` and `continual-arc-trajectories-v0.1.0.tar.gz`, as plain JSON in git, with
what neither asset holds: the trajectories of the open-model runs and of the frozen tests' plays, the GPT-6 Luna
runs' later plays of the frozen tests and DAgger's training records, which the paper's tables and figures read
([`paper/README.md`](../paper/README.md#the-figures-and-tables)), the letters QETE `-letter` read with its teacher's
state, and first sight's readings: about 168,600 files and 14.0 GB on disk. Until 11 November 2026 they were a
repository of their own,
[continual-arc-runs](https://github.com/unifyai/continual-arc-runs), kept read-only with its history; its tag `v0.1.0`
is that repository as the release went out. Every path in the registries below, such as
`runs/arc1/H-prime-agent-luna6-low-ws0`, is read from this repository's root, and so are the commands under "Reading
them".

```bash
git clone https://github.com/unifyai/continual-arc   # the runs are in runs/
```

## Layout

- `runs.json`: the registry of the GPT-6 Luna runs, one entry per run: its directory (`path`), row, model preset,
  track, order (`mode`), seed and label, the harness for the harness rows other than prime-agent, and
  `"ablation": true` for the ablation rows.
- `open-models.json`: the registry of the open-model runs in the same format, with each run's number of users,
  `reset`, the run whose reference plays of the two tests it is read against, `twin`, the run whose stream is the
  zero of its share learned, and `pooled`, the resets whose plays of the held-out test, the same at any number of
  users, it is read against too.
- `machines.json`: the machines whose every run the paper's text reports, one entry per machine: its directory
  (`path`, under `runs/arc1/`), track and label, and `rows`, its runs laid under their rows' names in `runs/`, with
  their paths.
- `runs/live/arc1/<run>/` and `runs/live.json`: each open-model run still being played, laid again every hour: in the
  layout of `runs/<track>/`, the files its stream, and each frozen test it has begun, have written so far, and in
  `runs/live.json` its entry
  in `open-models.json`'s format with `played`, how many instances of the stream and of each test it has played out
  of their total. A run leaves `runs/live/` for `runs/arc1/` and `runs/open-models.json` once its stream and both its
  tests are complete.
- `runs/first-sight/`: how often a model solves a track's instances at first sight (Appendix K's table of first-sight
  solve rates, and its reading of Kimi K3.1 Pro's effort), written by
  [`experiments/first-sight/`](../experiments/first-sight). `arc2/` holds each reading of ARC-AGI-2's calibration
  sample, four fresh instances (two for Kimi-K3 and the effort reading) of each of the track's 25 stream tasks, every
  answer with what it cost: the table's `kimi-k3.1-pro-medium`, `kimi-k3-medium` and `qwen3.5-9b-medium`, the effort
  reading's `-on-moonshotai-mxfp4` at each effort and its `rerun-`, and the same sample read on other providers and
  settings, which the paper does not report. `arc3/arc3-small-held-out-seed0/` holds the bare model's play of
  ARC-AGI-3's 40 held-out levels, wiped before each, by model, in the benchmark's records, its `README.md` saying which
  folder is whose.
- `runs/<track>/<run>/`, one directory per run. Its records, as the benchmark wrote them:
  - `records.jsonl`: one line per instance, in the order served: the task, phase (stream or epilogue), exposure index
    and gap, whether it was solved, its cost, the demonstrations requested and wrong attempts, and its resources: the
    proxy's counters of model calls, tokens and charge (`llm_*`), used throughout the paper, and the harness's own
    reports (`system_*`), which are not comparable across harnesses.
  - `summary.json`: the score and diagnostics, with a `protocol` block that says whether the run is canonical.
  - `schedule.json`: every slot the run was served; `config.json`: the environment's settings; `meta.json`: the
    learner, the protocol, when the run started and when it was resumed.
  - `frozen/<test>/play/`: the same five files for each frozen test the run played after its stream, `seen` and
    `held-out`, each one round of 100 instances, every instance a user of its own starting from the learner's state
    at the end of the stream. A `fresh` run holds instead its reference plays,
    `seen-reference` and `held-out-reference`: the same instances played from an empty state. The runs whose tests the
    paper reads (its Table 12) played them again with fresh sampling, each workstreams run of a learner that keeps
    something three more times, `<test>-2` to `<test>-4`, and each `fresh` run of a reset its reference plays twice
    more, `<test>-reference-2` and `<test>-reference-3`. An open-model run holds
    every play of each test the paper reads, its own and, for a reset, its reference plays, the later ones as
    `<test>-K` and `<test>-reference-K`, and beside a play, in `<play>.replay/`, the games it played again after a
    context overrun, which take the place of the same games in it.
  - `QETE-prime-agent-qwen3.5-9b-ws0-u1-letter` holds what its student read: `letters/r<round>.md`, the letter
    GLM-5.3 Flash wrote its student after each round of its stream from round 99 on (`r199.md` the one ending every
    test user's system prompt; `sizes.jsonl` each letter's call, `counts.json` its tokens), and `teacher-state.json`,
    in `memory.json`'s format, the files of GLM-5.3 Flash's state the letters were written from, each version as its
    harness committed it after a round of its stream (row GE's): Prime Agent's store and what GLM-5.3 Flash wrote
    outside the harness's own folders, which [`teach.py`](../experiments/teachers-letter/teach.py) reads.
  - DAgger's runs, `QDE-qwen3.5-9b-ws0-u{1,2,4}`, hold their training records too, as the trainer wrote them: for each
    phase, one per round, `train/p<phase>/DONE` (the step it kept, and its minutes) and `rounds.json` (the rounds it
    trained on), and for each phase that trained, `phase.json` (its settings, each step's loss and its held-out
    measures); and for each round, `rounds/r<round>/labels/split/split.json`, how many of the round's games and
    labelled calls went to training and how many to the held-out measure. The adapters and the labels are not here.

  And, for the GPT-6 Luna runs and the open-model runs, its trajectories, one format for every learner, and for a
  GPT-6 Luna run and an open-model run, one that plays the tests alone (`frozen tests` in `open-models.json`) among
  them, those of each of its plays but the 112 below, beside the play's records in `frozen/<play>/play/`, its id that
  path under `runs/<track>/`, every instance a user of its own:
  - `run.json`: the run's instances in the order served, each with its task, phase, exposure index and gap, whether
    it was solved, its charge, the demonstrations it bought and its wrong attempts, the model calls, tokens and spend
    the tracking proxy counted, and `trajectory` and `steps` (whether the instance has a trajectory, and whether that
    holds the learner's own steps).
  - `i/<step>.json`: one instance's trajectory, `{run, step, task, alias, turns, elided?}`. `alias` is the task's name
    as the learner saw it. Each turn is one exchange: `kind` (`observation` for every message the benchmark sent,
    feedback included; `reprompt`; `verdict`; `finish_reprompt`; `background` for steps a harness ran on its own
    between exchanges; `program` for the program library's kept program, run before the model is asked;
    `program_ask` and `program_retry` for ARC-AGI-3's program library asked for a game's rules after an instance),
    `phase` (`act`, or `store` from the final verdict on), `t0` and `t1` (Unix seconds), `message`, `reply`, `action`
    (`{type, grid?, correct?}`; on ARC-AGI-3 a game's `act` or `reset`, and the program library's rules, `model`) and
    `steps`. A step is `reasoning` (the provider's reasoning summary), `text`, `tool`
    (`{name, input, output, error?, stored?, recalled?}`: `stored` lists the files the call wrote, `recalled` marks a
    call that read the learner's memory, skills or past sessions) or `event` (Prime Agent's memory digest and its
    automatic refinements). `agent` marks a step that ran outside the main conversation, in a subagent or a
    scheduled session such as OpenClaw's dreaming.
  - `memory.json`: what the learner kept, `{run, kind, note?, files}`, each file with its versions
    (`{step, phase, text, change?, final?}`): the text a write left, or the edit a call made, and the copy on disk at
    the end of the run. Prime Agent's one tool is a Python kernel, so each of its writes is read from the cell that
    made it, and where the cell built the text as it ran, `change` holds the cell itself. A play kept nothing, since
    nothing a test's user wrote reached another instance, and its `note` says what each user started from.

A GPT-6 Luna run's name is `<row>-<system>-<preset>-<order><seed>`. The rows are F, the two floors
(`BareModelLearner`, stateless, and `BareDemoCacheLearner`); M, the program library; H, a harness with every learning
mechanism it ships; H0, the same harness with its persistence switched off; and the ablation rows HA (asked to store),
HPL and HPL0 (told to keep a program library) and M0 (the program library with nothing kept). The presets are GPT-6
Luna at an effort (`luna6-low`, `luna6-medium` and `luna6-high`). The orders are `ws` (workstreams, the track's),
`seq` (sequential), `rnd` (random) and `fresh` (the held-out tasks alone, the learner reset before each, whose
reference plays are the frozen tests' zero for the row). An open-model run's name is
`<row>-<system>-<model>-ws0-u<users>`, with the paper's names for the rows (its Table 1), and a run that only the
paper's text reports adds what sets it apart (`-ceiling-<library>`, `-letter`, `-page-alone`, `-from-base`,
`-pilot`).

## How the runs were played

Every GPT-6 Luna run was played under the benchmark's charge, stated in the rules every learner read: an instance
solved costs what it spent, one for each wrong answer and each demonstration requested, and one that ends unsolved
the whole budget, 16. After its stream a workstreams run went on to the epilogue, the 100 held-out tasks played while
the learner could still learn, before the frozen tests replaced it: the floors, the program library and Prime Agent
played all of it, and OpenClaw and Hermes Agent were stopped part of the way in, some at its start, so their
`schedule.json` ends with the stream and their records with the epilogue's instances they played. The paper reports
nothing from the epilogue. The sequential and random runs played the stream alone, and every workstreams,
sequential and random run played the two frozen tests. The process of 32 runs ended after their last record
without writing `summary.json`, so theirs is computed from their records by the benchmark's `summarize`, the code
that wrote the others', and `meta.json`'s `summarized` says so.

## Reading them

```bash
pip install "continual-arc @ git+https://github.com/unifyai/continual-arc"
continual-arc report runs/arc1/H-prime-agent-luna6-low-ws0
# the share learned over the stream, against the same harness wiped before every instance
continual-arc learned runs/arc1/H-prime-agent-luna6-low-ws0 runs/arc1/H0-prime-agent-luna6-low-ws0
# T on each frozen test, against the reset's reference play of the same instances
continual-arc transfer runs/arc1/H-prime-agent-luna6-low-ws0/frozen/held-out/play \
    runs/arc1/H0-prime-agent-luna6-low-fresh0/frozen/held-out-reference/play
continual-arc transfer --test seen runs/arc1/H-prime-agent-luna6-low-ws0/frozen/seen/play \
    runs/arc1/H0-prime-agent-luna6-low-fresh0/frozen/seen-reference/play
```

A harness is read against its row H0, HPL against HPL0, the demo-cache floor against the stateless floor and the
program library against M0; a floor or a wiped row is read against its own reference play. Task ids in the records
are the benchmark's own; a learner saw them only under aliases drawn for its run.

## Where the trajectories come from, and what is missing

Each learner's exchanges are the adapter's log of the protocol, and their outcomes the benchmark's. A harness's own
steps come from its own store: Prime Agent's RPC event log, OpenClaw's transcript database and the sessions it
archived, Hermes Agent's session database. Each step is placed in the exchange whose time window holds it. In a
GPT-6 Luna run's frozen test every user kept its own exchanges, Prime Agent's RPC log and a store of its own, which
began as a copy of the stream's, or empty in a reference play: OpenClaw's and Hermes Agent's are read from the user's
commit of the test round, their steps from the play's start on. The tests kept no record of the benchmark's events, so
each action's answer is read from the benchmark's next message. The harnesses' steps are there for all but 107 of the
59,377 instances of these plays they acted on, the wiped rows' included, and those 107 say so (`elided`); an instance a
harness never acted on (a forced give-up after its restarts, 23 of them) or that the program library's kept program
solved before the model was asked (1,548) has no trajectory.

The rows that wipe the harness before every instance (H0, HPL0 and every `fresh` run) deleted OpenClaw's and Hermes
Agent's stores with everything else, so in those runs' streams their reasoning and tool calls survive for the last
instance alone; the benchmark's messages and the harnesses' replies are complete. Prime Agent's RPC log sat outside its
state, so its wiped runs are complete. The floors and the program library are the model alone behind the protocol, so
their replies are the whole of what they did, and the program library's programs are in them. Tool outputs and messages
longer than 60,000 characters are cut, with a marker saying how much. One entry naming software of ours on the
runners' `PATH` was taken out of the environments two harnesses printed. The open-model runs' frozen-test plays are read
as those of the runs that play the tests alone are: Kimi K3.1 Pro's from each user's own exchanges and RPC log, which
its machine's copy kept, and Qwen3.5-9B's and GLM-5.3 Flash's from the model calls the trained rows' engine captured,
but for DAgger's replays, read from their users' own logs. The 80 plays of the floors and the program library on
Qwen3.5-9B and GLM-5.3 Flash and the 32 of Kimi K3.1 Pro's floors kept no call's text and no user's log in any copy: the
users' logs stayed on machines since deleted, and the bucket's copies count each call's tokens alone, so these 112 plays
have their records alone.

The plays the trained rows' engine played, of the served tests, the page alone, DAgger's fits from the base model, the
library ceiling and the Qwen3.5-9B and GLM-5.3 Flash runs of a harness or of DAgger, kept no adapter log and no RPC log,
but for their replays: they kept every model call, its request as sent and its reply, so their exchanges and steps are
read from those calls, the benchmark's messages and the harness's notes from each request, the model's messages from its
reply, and a tool's error from its output. Read so where both were kept, QETE `-letter`'s held-out test, GB's stream and
three of the replays, the calls give 99.1 to 99.6% of the RPC log's steps, 98 to 100% of the adapter's replies and 1,415
of the 1,416 exchanges it logged; a turn the harness failed in, after which the adapter restarted it, is left out, as
the adapter's log leaves it out. On ARC-AGI-3 each reply's action is read by the benchmark's own parser for games, and
the program library's programs from its own copy of them after every round. In the trajectories of the 22 open-model
runs the site does not replay, and in the letters and the teacher's state, the runners' home folder is written
`/home/USER`; in those of the GPT-6 Luna runs' frozen tests and of the other open-model runs' frozen-test plays the
machines' home folder is written `/home/USER` and their names `HOST`, and the value of each key, token or secret an
environment printed is taken out.

## Citation

If you use the runs, please cite the paper; its BibTeX is in the [README](../README.md#citation).

## Licence

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`).
