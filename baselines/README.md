<!-- ARXIV_ID_PENDING: replace every ARXIV_ID_PENDING in this file with the paper's arXiv id once its listing is out, the evening before the launch on 27 October 2026, then delete this line -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/continual-arc-readme-banner-dark.png">
    <img src="../assets/continual-arc-readme-banner-light.png" alt="Continual-ARC, a unify benchmark" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://unify.ai/continual-arc">Explore every run</a> ·
  <a href="https://unify.ai/blog/why-we-built-continual-arc">Paper</a> ·
  <a href="https://unify.ai/blog/why-we-built-continual-arc">Blog post</a> ·
  <a href="https://unify.ai/continual-arc/leaderboard">Leaderboard</a> ·
  <a href="https://github.com/unifyai/continual-arc">Benchmark</a>
</p>

# continual-arc-baselines

The learners that the paper [*Continual-ARC: Measuring Skill Acquisition over a Working Lifetime*](https://unify.ai/blog/why-we-built-continual-arc)
evaluates on the [Continual-ARC](https://github.com/unifyai/continual-arc) benchmark, and what runs them: the
adapters that drive three agent harnesses through the benchmark's protocol, the learner that serves a team of users at
once, the sandbox the harnesses run in, the local proxy through which every model call passes, the frozen tests after a
stream, and the scripts that launch the paper's runs. This folder is the package `continual-arc-baselines` inside the
benchmark's repository, whose root is the benchmark; the training of the trained rows is beside it in
[`training/`](../training), which drives the learners here. The records of the paper's GPT-6 Luna runs are plain JSON
in git at [continual-arc-runs](https://github.com/unifyai/continual-arc-runs) and attached to the benchmark's
[v0.1.0 release](https://github.com/unifyai/continual-arc/releases/tag/v0.1.0).

## What is here

| path | contents |
|---|---|
| `src/continual_arc_baselines/systems/bare.py` | the model alone behind the harness protocol: the two floors of row F (`BareModelLearner`, stateless, and `BareDemoCacheLearner`, which shows a task's bought pairs again when it returns), the demo-cache floor at several users (`BareSharedDemoCacheLearner` over one `SharedDemoCache` for the team), and the program library of row PL (`BareProgramLibraryLearner`; row PL0 with `stateless=True`; the run records name these rows M and M0) with the sandboxed runner of its programs |
| `src/continual_arc_baselines/systems/prime_agent.py`, `hermes_agent.py`, `openclaw.py` | the harness adapters of row H: prime-agent over its RPC mode, hermes-agent and openclaw over the Agent Client Protocol (`acp.py`). Row H0 is `stateless=True`, row HA `ask_to_store=True` and row HPL `program_library=True` |
| `src/continual_arc_baselines/systems/base.py`, `protocol.py` | `SystemLearner`, the adapter base class, with its budgets, restarts, wipes and frozen resets; the messages a learner is shown and the lenient parser that reads an action from a reply |
| `src/continual_arc_baselines/systems/sandbox.py` | the sandbox each harness runs in: bubblewrap on Linux, Seatbelt on macOS |
| `src/continual_arc_baselines/systems/hosted.py`, `learner_host.py` | a submitted Python learner run whole inside the sandbox, for verified runs, and the host that serves it there |
| `src/continual_arc_baselines/collective.py` | `CollectiveHarnessLearner`: one system per user, every user's instance of a round served at once, each user behind a proxy of their own, with the hooks a trained row learns from (every instance's turns and model calls, captured whole) and `snapshot_state` / `restore_state` for a user's state |
| `src/continual_arc_baselines/proxy.py`, `tracking.py` | the OpenAI-compatible reverse proxy that records every model call, and the per-instance counters it feeds (`llm_*` in every record) |
| `src/continual_arc_baselines/cli.py` | `arc-baselines`: `run` a learner over a schedule, `test` a harness's stream, `fork` a run from any round, `install-submission` |
| `src/continual_arc_baselines/frozen.py` | the frozen tests after a stream, played one instance at a time from the stream's state put back, or as one round of users forked from the stream's last commit |
| `src/continual_arc_baselines/statelog.py`, `fork.py` | a system's state committed to git at the end of every round, and a run forked from any round's commit |
| `src/continual_arc_baselines/attempt.py`, `costs.py` | the attempt directory each run writes beside its run directory (manifest, event log, metrics, artifacts) and its cost journal |
| `scripts/launch-cell.sh` | launches one cell: one row at one preset, configuration, order and seed |
| `scripts/paper-grid.sh`, `scripts/paper-grid.txt` | every cell of the paper's GPT-6 Luna grid, one line each |
| `scripts/arc-leaderboard.py` | reads the ARC Prize leaderboard's published data, from which every model in the grid was chosen |

## The paper's rows and the learners behind them

| rows | learner | how they ran |
|---|---|---|
| GPT-6 Luna's rows, at low, medium and high effort: F, PL and PL0, and H, H0, H-A, H-E and H-E0 (HA, HPL and HPL0 in the run names) in each of prime-agent, hermes-agent and openclaw | `BareModelLearner` and `BareDemoCacheLearner` (F), `BareProgramLibraryLearner` (PL, PL0), `PrimeAgentLearner`, `HermesAgentLearner` and `OpenClawLearner` (the H rows) | one `arc-baselines run` a cell (`scripts/launch-cell.sh`), its stream alone; its frozen tests as one round of a hundred users of the cell's own system, each forked from the commit of the stream's end (`arc-baselines test --collective`) |
| the open models' rows: on Qwen3.5-9B, F, PL, H, H-E (given expert guidance, the program library's procedure), the rows in which the teacher writes the notes (H-T, H-T-E and H-E-T-E; hg, hgpl and hplg in the run names) and the trained rows (DAgger, OPSD, PG), each beside its wiped twin; on GLM-5.3 Flash, F and PL | `CollectiveHarnessLearner`, one system per user: `PrimeAgentLearner` for every harness row, `BareModelLearner` and `BareDemoCacheLearner` (`BareSharedDemoCacheLearner` at several users) for the floors, `BareProgramLibraryLearner` for the program library | driven round by round from [`training/`](../training), which serves the model, trains the adapters and lets the teacher write; every user's state committed at the end of every round, and the frozen tests played as one round of users forked from the stream's end (`frozen.seed`) |

## Install

Linux, with [bubblewrap](https://github.com/containers/bubblewrap) for the sandbox (macOS works with its own sandbox),
Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/unifyai/continual-arc
cd continual-arc/baselines
uv sync                                                # the benchmark from the same checkout, through uv's workspace
uv run continual-arc data fetch                        # Re-ARC and ARC-AGI-1 at pinned commits
uv run pytest                                          # no network or keys needed
```

Or, as a dependency of your own learner, which brings the benchmark from the same repository:

```bash
pip install "continual-arc-baselines @ git+https://github.com/unifyai/continual-arc@v0.4.0#subdirectory=baselines"
```

On Ubuntu 24.04, bubblewrap needs unprivileged user namespaces, which AppArmor restricts by default:
`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, as the paper's machines had.

Each harness row needs its harness, at the versions of the paper's Appendix D:

- prime-agent 0.9.5: `curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh`
- hermes-agent 0.20.2 (git `0c3452bf75ce6aecc603954d44c8d470d2a20c15`): a checkout of
  [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) at `~/hermes-agent` with
  `uv pip install -e ".[acp]"` in its virtual environment, or pass `command` to the adapter
- openclaw 2026.9.5: `curl -fsSL https://openclaw.ai/install-cli.sh | bash`, or the npm package. The adapter checks
  once a run that the installed openclaw accepts its configuration (`openclaw config validate`) and stops if it does not

## Run a learner

```bash
# a smoke run with a learner that never solves anything: no model, no key
uv run arc-baselines run --learner continual_arc.testing:GiveUpLearner --config arc1-smoke --seed 0

# the stateless floor on the ARC-AGI-1 track at low effort, as the paper ran it
export OPENROUTER_API_KEY=...
bash scripts/launch-cell.sh floors luna6-low workstreams 0 arc1-small
```

A run directory (`--out`) holds `records.jsonl` (one line per instance), `summary.json`, `schedule.json`,
`config.json` and `meta.json`, the files of every run in the run records, and `launches.jsonl`, the command line and
the system's description of every launch; `uv run continual-arc report <run directory>` prints its scores. Beside it,
under `.workbench/runs/` in the folder the run was launched from, the run's attempt holds its event log, its cost
journal and the system's own folder, its state and logs (`attempt.json` in the run directory names it).

`arc-baselines run` starts the tracking proxy for the run and exports its address as `OPENAI_BASE_URL`,
`OPENROUTER_BASE_URL` and `ARC_LLM_BASE_URL` with a placeholder key, so any system pointed at it is measured without
knowing it; the real key stays in the proxy. The proxy forwards each request unchanged except that it asks the
provider to report usage and adds the run's defaults (`--llm-defaults`) where the system set none.

To run your own system, subclass `SystemLearner` (`systems/base.py`), which speaks the benchmark's protocol and leaves
an adapter only `start`, `stop`, `ask`, `new_session` and `wipe_state`, and keep everything the system persists under
`self.state_dir`, which its frozen tests put back before every instance; or write any learner against the benchmark's
four-method interface that builds its client from `OPENAI_BASE_URL`. Every instance of a `SystemLearner` begins in a
conversation of its own (`new_session`), the first after any start included, and a stateless one (`stateless=True`) is
wiped before every instance, the first included.

`--stream-only` plays a schedule's stream and stops before its epilogue, a literal prefix of the schedule, for a
`--schedule` as for a `--config`; `--no-epilogue` builds the stream alone from `--config`.

## Test a harness

After its stream a learner plays two frozen tests, the held-out test and the seen test, every instance from its state
at the end of the stream, so that nothing learnt on one test instance reaches another (the benchmark's
[two frozen tests](https://github.com/unifyai/continual-arc#the-two-frozen-tests)). `arc-baselines test` plays a test
of a harness's stream, with the harness the stream's first launch built (`launches.jsonl`), so run it from the same
folder, with the key in the environment and, for an adapter of your own, the same `PYTHONPATH`. The test's records go
to `RUN/frozen/<the test>/play` (or `--out`), with the stream's `alias-salt`, so that the harness meets its tasks under
the names it learnt them by, and `frozen.json`, what the test was played from.

**One instance at a time**, the default and what a submission's harness plays. A harness keeps one conversation, so it
plays a test made with `--width 1`, and before every instance `arc-baselines test` stops it and puts its state
directory back as the stream left it, at the same path: every path the harness wrote into its own files, its databases
included, still names its own state, so nothing is rewritten, whatever the harness.

```bash
export OPENROUTER_API_KEY=...
LEARNER=continual_arc_baselines.systems.openclaw:OpenClawLearner
SETTINGS='{"model": "openai/gpt-6-luna", "end": "finish"}'
EFFORT='{"reasoning_effort": "medium"}'
uv run continual-arc schedule make --config arc1-small --seed 0 --test held-out --width 1 --out arc1-held-out0-1.json
uv run continual-arc schedule make --config arc1-small --seed 0 --test seen --width 1 --out arc1-seen0-1.json

# the stream alone, then each test from the state it left
uv run arc-baselines run --learner $LEARNER --learner-kwargs "$SETTINGS" --llm-defaults "$EFFORT" \
  --config arc1-small --seed 0 --no-epilogue --out runs/arc1/openclaw-ws0
uv run arc-baselines test runs/arc1/openclaw-ws0 --schedule arc1-held-out0-1.json   # runs/arc1/openclaw-ws0/frozen/held-out/play
uv run arc-baselines test runs/arc1/openclaw-ws0 --schedule arc1-seen0-1.json       # runs/arc1/openclaw-ws0/frozen/seen/play

# its reset reference: the same instances, the harness wiped before each
uv run arc-baselines run --learner $LEARNER --learner-kwargs "$SETTINGS" --llm-defaults "$EFFORT" \
  --schedule arc1-held-out0-1.json --out runs/arc1/openclaw-reset-held-out
uv run continual-arc transfer runs/arc1/openclaw-ws0/frozen/held-out/play runs/arc1/openclaw-reset-held-out

# the share learned over the stream: the same stream, the harness wiped before every instance
uv run continual-arc schedule make --config arc1-small --seed 0 --no-epilogue --reset-each \
  --out arc1-ws0-reset.json
uv run arc-baselines run --learner $LEARNER --learner-kwargs "$SETTINGS" --llm-defaults "$EFFORT" \
  --schedule arc1-ws0-reset.json --out runs/arc1/openclaw-ws0-reset
uv run continual-arc learned runs/arc1/openclaw-ws0 runs/arc1/openclaw-ws0-reset
```

The run must be a stream played whole with `--out` and nothing after it (`--no-epilogue`, or a `--schedule` made with
`--no-epilogue`), and the test must be of its configuration and seed. For the test the stream's own state directory is
set aside, renamed `system/stream-end-<instances played>` in the stream's attempt, and a copy of it is put in its place
before every instance. When the test ends, or is stopped with Ctrl-C, `kill` or a hangup, the copy is deleted and the
stream's directory renamed back. A test stopped harder, by `kill -9` or the machine going down, leaves it aside, and
the next test, or the stream taken up again with `--resume`, puts it back first. The test's logs and model-call journal
go to an attempt of its own. A test launched again goes on from where it stopped, unless the stream has gone on since,
which is refused. One process at a time plays a run, its stream or a test of it, so the two tests play one after the
other, each about half as long as the stream. The references need nothing of the stream's, so they can play beside
them, each in a run directory of its own.

The bare model, `BareModelLearner` on the tests' schedules, is no reference: its runs are shown beside T, which is
flagged if the reset plays below them. A test puts back the state directory alone, so an adapter of your own keeps
everything its harness persists under `self.state_dir` and its own logs outside it; on macOS the sandbox also lets a
harness write the user's temporary and cache folders, which nothing puts back, in a test as in its stream. A learner
that keeps what it learns in memory, as the demo-cache floor does, has no tests here: a Python learner plays them from
a copy of itself (the benchmark's README), or after its stream in the same process with `arc-baselines run --test
TEST.json`, one test a launch.

**As one round of users** (`--collective`), as the paper played its tests: every instance of a test made at its full
width is a user of its own, a hundred played at once on one machine, each forked from the commit of the stream's last
round. It needs a run that logged its state (below). Test user i starts from the run's user i mod U (a run of one
user: its one system), its state the stream's last commit with every absolute path that named the run's own system
folder renamed to the user's (in text files, links and SQLite databases); the users' objects share one store that reads
the run's. A run with no stream starts every user from nothing, in `RUN/frozen/<the test>-reference/play`.

```bash
uv run continual-arc schedule make --config arc1-small --seed 0 --test held-out --out arc1-held-out0.json
uv run arc-baselines test runs/arc1/openclaw-ws0-logged --schedule arc1-held-out0.json --collective
```

## Logged state and forks

`arc-baselines run --log-state` commits a `SystemLearner`'s whole state directory at the end of every round, into
`system/state.git` beside it, each round's commit listed in `system/state-log.jsonl`; git keeps a file once however
many rounds it lasts, so a run's every state costs about what its last one does. `CollectiveHarnessLearner` logs every
user's state this way, in `users/uNNN/system/`.

```bash
# the stream alone, its state committed at the end of every round
uv run arc-baselines run --learner $LEARNER --learner-kwargs "$SETTINGS" --llm-defaults "$EFFORT" \
  --config arc1-small --seed 0 --stream-only --log-state --out runs/arc1/openclaw-ws0-logged

# taken up again from the end of round 99, with the state its learner had then: prints the command that plays on
uv run arc-baselines fork runs/arc1/openclaw-ws0-logged --round 99 --out runs/arc1/openclaw-ws0-from-99
```

## Submit a result

The benchmark's README says how to [submit a result](https://github.com/unifyai/continual-arc#submitting-a-result):
the stream, the two tests and the references above, copied into its `results/`. A harness adapter on `SystemLearner`
depends on this package alone, which brings the benchmark with it.

For a verified run we rerun the submission's code on held-out rules with `HostedLearner` (`systems/hosted.py`), which
runs a submitted learner whole inside the sandbox, with the submission's own interpreter, behind the tracking proxy, so
the learner never holds a key and reads nothing of the held-out rules. `arc-baselines install-submission --repo URL
--commit SHA --venv DIR [--checkout DIR]` installs it into a new virtual environment, under the sandbox too, since
installing runs the package's build code. A hosted learner that keeps everything as files (`state_in_files`, as a
harness on `SystemLearner` does) plays `arc-baselines run --test` one instance at a time, its files put back before
each; any other plays the test in one round.

## Reproduce the paper's runs

`bash scripts/paper-grid.sh` launches every cell of the GPT-6 Luna grid on its stream alone: 198 runs on the ARC-AGI-1
track, at low, medium and high effort. Each writes `runs/<track>/<cell>/`, under the name its record carries in the run
records. Every cell is detached and long-lived, so launch a part at a time (`bash scripts/paper-grid.sh luna6-low`),
and `DRY_RUN=1` shows each cell's command first. The paper's 198 runs made about 1,500 USD of model calls through
OpenRouter, by the cost the proxy recorded on each instance. The models are sampled through provider APIs, so a rerun
differs from the paper's by about the spread the paper reports. A cell plays its stream alone (`--no-epilogue`), so a
harness cell's frozen tests can follow it with `arc-baselines test`, one instance at a time; `--collective`, as the
paper played them, needs the stream launched with `--log-state`. The paper's cells also played a learning-on epilogue,
which no measure reads since the frozen tests replaced it.

The paper's cells were launched between 27 and 29 September 2026, each row from the code of its launch day. The
learners here are that code as it stood once the last row was launched, with four differences: the rules a learner
reads say that an instance it does not solve costs the whole budget, as the benchmark charges it; the proxy limits the
upstream connections it keeps open (`max_connections`); a harness can put back a saved state when reset, which only its
frozen tests use; and the first instance after any start opens a conversation of its own, and a stateless row's is
wiped first, as every later instance's is, where the cells' streams did neither on the first instance of a launch, so
their records count no session there (and a stateless row no wipe). The cells launched before 17:11 UTC on 28
September ran under the earlier rule for empty replies (`empty_reply="restart"`), and `paper-grid.txt` launches them
under it.

The open-model rows are driven from [`training/`](../training), which builds each row's `CollectiveHarnessLearner`.

## Licence

Apache-2.0, like the benchmark. The baselines are part of [Continual-ARC](https://github.com/unifyai/continual-arc),
made by [unify](https://unify.ai).
