# Experiments

Every experiment the paper [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://arxiv.org/abs/2611.14505) reports, by its section, table or appendix and its rows: the code that
ran it, the command that launches it, and where its runs are under [`runs/`](../runs). Each run's directory is the
`path` of its entry in one of three registries, which resolves from the repository's root:
[`runs/runs.json`](../runs/runs.json) (the 228 GPT-6 Luna runs), [`runs/open-models.json`](../runs/open-models.json)
(the open models' runs) and [`runs/machines.json`](../runs/machines.json) (the machines whose every run the paper's
text reports). Each run's `meta.json` names its learner and when it ran.

The code is of two kinds:

- **Public**: the benchmark ([`src/continual_arc/`](../src/continual_arc)), the learners and the GPT-6 Luna grid's
  launcher ([`baselines/`](../baselines)) and the open models' rows ([`training/`](../training), whose
  [README](../training/README.md#what-each-script-runs) and [code map](../training/docs/code.md) say what each script
  and module does). Each entry below says whether it is the code that ran or an exact equivalent, and the commit where
  the code that ran differs from `main`.
- **Here**: the files that ran, where the public code does not hold them, in a folder of this directory each. They
  are copied from the private repositories the experiments ran from with only credentials, machine and host names,
  bucket names, home paths and people's names taken out, each replaced by a placeholder ([Placeholders](#placeholders)).
  They document what ran and with what settings; most drive a private engine of their day that `training/` replaced
  and do not run against `main` unchanged, which each entry says.

## Row names

The paper names a method by its model (Q Qwen3.5-9B, G GLM-5.3 Flash, K Kimi K3.1 Pro) and its method (its Table 1);
the scripts name the same methods by a row code:

| Paper | Method | Row code |
| --- | --- | --- |
| C0 (QC0, GC0, KC0) | the stateless floor | `f` (`BareModelLearner`) |
| C (QC, GC, KC) | the demo-cache floor | `dc` (`BareDemoCacheLearner`) |
| P (QP, GP, KP) | the program library | `m` (`BareProgramLibraryLearner`) |
| P0 (QP0, GP0, KP0) | the program library with its programs wiped before every instance | `m0` |
| Q, G, K | notes in prime-agent, as it ships | `h` |
| Q0, G0, K0 | prime-agent wiped before every instance | `h0` |
| QA, GA, KA | asked to store after every verdict | `ha` |
| QE, GE, KE | given expert guidance | `hpl` |
| QE0, GE0, KE0 | given expert guidance, wiped before every instance | `hpl0` |
| QT, QTE, QETE | Qwen3.5-9B playing from the notes GLM-5.3 Flash writes | `hg`, `hgpl`, `hplg` |
| QDE | Qwen3.5-9B's weights trained by DAgger, GLM-5.3 Flash teaching | `dagger` |
| GB, GB0 | the expert guidance's library kept as building blocks | `hpb`, `hpb0` |

GPT-6 Luna's rows are named in [`runs/README.md`](../runs/README.md): F (both floors), M and M0 (the program library),
H and H0 (a harness and its wiped twin), and the ablations HA, HPL and HPL0.

## ARC-AGI-1 on GPT-6 Luna

Section 5's Figure `fig:plane-luna`, Appendix I (the share learned, Table `tab:tests-luna`, the table on the
workstreams, sequential and random orders), Appendix B's side experiment on the order (`seq`, `rnd`) and Appendix F's
ablations (HA, HPL, HPL0, M0): the 228 runs of `runs/runs.json`, `"ablation": true` marking the 120 of Appendix F.

- **Code** (public, the code that ran, as [`baselines/README.md`](../baselines/README.md) says with the differences it
  lists): [`baselines/scripts/paper-grid.txt`](../baselines/scripts/paper-grid.txt), one line per cell with the
  empty-reply rule and the epilogue it ran under, launched by [`paper-grid.sh`](../baselines/scripts/paper-grid.sh)
  through [`launch-cell.sh`](../baselines/scripts/launch-cell.sh) (`arc-baselines run`); the frozen tests by
  `arc-baselines test --collective` from each cell's state at the stream's end.
- **Command**, from `baselines/`:

  ```bash
  DRY_RUN=1 bash scripts/paper-grid.sh            # every cell's run directory and command
  PILOT=25 bash scripts/paper-grid.sh 'luna6-low'  # a part at a time, stopped after 25 instances, as the paper's were
  RESUME=1 bash scripts/paper-grid.sh 'luna6-low'  # then continued
  ```

- **Runs**: `runs/arc1/<row>-<system>-luna6-<effort>-<order><seed>`, each in `runs/runs.json`.
- **Not run again**: the three openclaw cells of Appendix E that played with memory search set, under the earlier
  charge; no copy of their records survives.

## ARC-AGI-1 on the open models

Section 5's Table 1 and Figure `fig:notes-weights`, Appendix E (what persists), Appendix F (asked to store, expert
guidance, building blocks, the library ceiling, Table `tab:tests-open`) and Appendix H (the methods on the open
models, Tables `tab:tests-absolute` and `tab:tests-users`): the runs of `runs/open-models.json` on `arc1`.

### The reference rows (no GPU)

QC0, QC, QP, QP0, Q, Q0, QA, QE, QE0 on Qwen3.5-9B, and GC0, GC, GP, GP0, G, G0, GA, GE, GE0 on GLM-5.3 Flash, at one
user, and QC0, QC, QP, QP0 and Q0 at two and four.

- **Code** (public): [`training/scripts/reference.sh`](../training/scripts/reference.sh)
  (`continual_arc_training.references`), then each row's two frozen tests by
  [`ref-tests.sh`](../training/scripts/ref-tests.sh) (`continual_arc_training.frozen`). The expert guidance the
  E rows read is the consolidated text on `main`; the paper's rows read the text kept at
  [`9616cbd`](https://github.com/unifyai/continual-arc/tree/9616cbd2c1d5741142ca7f2590ecb3535028daf9) (c1 followed by
  the library procedure), so check out that commit to play QE, QE0, GE and GE0 as they ran.
- **Command**, on a machine with prime-agent 0.9.6, Node 22, bubblewrap and `OPENROUTER_API_KEY` in `baselines/.env`:

  ```bash
  PLAYER=qwen ROW=h USERS=1 setsid nohup bash training/scripts/reference.sh > /dev/null 2>&1 < /dev/null &
  #   PLAYER qwen|glm, ROW f|dc|m|m0|h|h0|ha|hpl|hpl0, USERS 1|2|4
  bash training/scripts/ref-tests.sh   # every finished reference row's frozen tests
  ```

- **Runs**: `runs/arc1/{QC0,QC,QP,QP0}-qwen3.5-9b-ws0-u{1,2,4}`, `runs/arc1/Q0-prime-agent-qwen3.5-9b-ws0-u{1,2,4}`,
  `runs/arc1/{Q,QA,QE,QE0}-prime-agent-qwen3.5-9b-ws0-u1`, `runs/arc1/{GC0,GC,GP,GP0}-glm-5.3-flash-ws0-u1`,
  `runs/arc1/{G,G0,GA,GE,GE0}-prime-agent-glm-5.3-flash-ws0-u1`.

### The teacher's notes (QT, QTE, QETE)

- **Code** (public): [`training/scripts/teacher-rows.sh`](../training/scripts/teacher-rows.sh)
  (`continual_arc_training.teacher_rows`), once the teacher's reference row (`glm-h-u1` for `hg`, `glm-hpl-u1` for
  the others) has played; QTE and QETE as they ran at `9616cbd`, as above.
- **Command**: `ROW=hg|hgpl|hplg setsid nohup bash training/scripts/teacher-rows.sh > /dev/null 2>&1 < /dev/null &`
- **Runs**: `runs/arc1/{QT,QTE,QETE}-prime-agent-qwen3.5-9b-ws0-u1`.

### DAgger (QDE) at one, two and four users

- **Code** (public): [`training/scripts/realtime.sh`](../training/scripts/realtime.sh) on one 8-GPU node (the
  player, the labeller and the trainer, `continual_arc_training.realtime`), then
  [`frozen.sh`](../training/scripts/frozen.sh) for both tests. The runs' teacher read the playing rules (c1) alone,
  as `9616cbd` keeps them; `main` gives it the consolidated expert guidance. The runs themselves were played by the
  private engine `training/` was consolidated from (the `-plays-a` and `-cross-fork2` machines in
  [`configuration-search/`](configuration-search/README.md)), whose settings these commands repeat.
- **Command**, on the node:

  ```bash
  ROW=dagger USERS=1 LR=1e-4 START_AFTER=24 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 \
      setsid nohup bash training/scripts/realtime.sh > /dev/null 2>&1 < /dev/null &
  bash training/scripts/frozen.sh dagger-u1-lag1 0,1,2,3,4,5,6,7   # once the stream has ended
  ```

- **Runs**: `runs/arc1/QDE-qwen3.5-9b-ws0-u{1,2,4}`.

### DAgger trained once from the base model (QDE `-from-base`, Appendix H)

- **Code** (here): [`configuration-search/frombase.sh`](configuration-search/frombase.sh), which runs
  [`longtrain.py`](configuration-search/longtrain.py) `--from-base` on all of a final DAgger run's labels, then the
  tests by [`trained-rows-resets/more-plays.sh`](trained-rows-resets/more-plays.sh). The public
  [`scratch.sh`](../training/scripts/scratch.sh) does the same training once at a run's end but is not what made these
  runs.
- **Command**: `bash experiments/configuration-search/frombase.sh RUN OUT GPUS EVAL_STEPS PATIENCE` on each final run's
  node.
- **Runs**: `runs/arc1/QDE-qwen3.5-9b-ws0-u{1,2,4}-from-base`.

### The trained rows' resets: the served tests and the page alone (Appendix H)

Q0 on the trained rows' own server, on the two tests alone at one, two and four users, and Q0 told only the page of
playing rules its teacher reads.

- **Code** (here): [`configuration-search/final-resets.sh`](configuration-search/final-resets.sh) with the private
  `frozen.sh` of 29 October 2026, [`trained-rows-resets/frozen.sh`](trained-rows-resets/frozen.sh), for the first
  plays; [`trained-rows-resets/more-plays.sh`](trained-rows-resets/more-plays.sh) for the later plays (`<test>-K`);
  [`trained-rows-resets/replay.sh`](trained-rows-resets/replay.sh) and [`replay.py`](trained-rows-resets/replay.py)
  for the games played again after a context overrun (`<play>.replay/`). The public successor is
  [`refs.sh`](../training/scripts/refs.sh) and `frozen.sh` with `ROW=h0`; `9616cbd` also keeps `ROW=h0c1`, the page
  alone, which `main` retired.
- **Command**: `(setsid -f nohup bash experiments/configuration-search/final-resets.sh > /dev/null 2>&1 < /dev/null)`
  on an 8-GPU node; `bash experiments/trained-rows-resets/more-plays.sh JOBS`;
  `bash experiments/trained-rows-resets/replay.sh LIST GPUS`. At `9616cbd`:
  `ROW=h0 bash training/scripts/frozen.sh qwen-h0-card 0,1,2,3` and `ROW=h0c1 ... qwen-h0c1-card`, with `USERS=2|4`
  for the seen test of several users.
- **Runs**: `runs/arc1/Q0-prime-agent-qwen3.5-9b-served-tests`, `-served-tests-u2`, `-served-tests-u4`,
  `runs/arc1/Q0-prime-agent-qwen3.5-9b-ws0-u1-page-alone`.

### The resets played in the same window (Q0 and QE0 `-same-window`)

- **Code** (here): [`teachers-letter/glm_harness.py`](teachers-letter/glm_harness.py) and
  [`glm_harness.sh`](teachers-letter/glm_harness.sh), rows `h0` and `hpl0` with no source, each stream played as two
  parts of 100 users. The nearest public code is `PLAYER=qwen ROW=h0|hpl0 USERS=1 bash training/scripts/reference.sh`,
  which plays the same rows round by round.
- **Command**: `python experiments/teachers-letter/glm_harness.py plan --run RUN --row h0 --streams > jobs`, then
  `bash experiments/teachers-letter/glm_harness.sh jobs` (and the same with `--row hpl0`).
- **Runs**: `runs/arc1/Q0-prime-agent-qwen3.5-9b-ws0-u1-same-window`,
  `runs/arc1/QE0-prime-agent-qwen3.5-9b-ws0-u1-same-window`.

### QETE with the teacher's letter (Appendix E)

- **Code** (here): [`teachers-letter/teach.py`](teachers-letter/teach.py) writes GLM-5.3 Flash's letter to its
  student after each round; [`glm_harness.py`](teachers-letter/glm_harness.py) and
  [`glm_harness.sh`](teachers-letter/glm_harness.sh) play the tests with the letter ending the student's system prompt.
- **Command**:

  ```bash
  OPENROUTER_API_KEY=... python experiments/teachers-letter/teach.py letters --source SRC --out OUT \
      --tokenizer tokenizer.json --par 24
  python experiments/teachers-letter/glm_harness.py plan --run RUN --row hpl --source hpl --teach letter > jobs-taught
  PAR=1 GAP=120 MAX_CONCURRENT=50 bash experiments/teachers-letter/glm_harness.sh jobs-taught
  ```

  `SRC` is GLM-5.3 Flash's `hpl` stream (`glm-hpl-u1`) with its `state.git`.
- **Runs**: `runs/arc1/QETE-prime-agent-qwen3.5-9b-ws0-u1-letter`.

### Expert guidance as building blocks (GB, GB0, Appendix F)

- **Code** (here): [`building-blocks/`](building-blocks): `baselines-blocks.patch` (the learner's side) and
  `reference-blocks.patch` (the reference rows' driver), [`run.sh`](building-blocks/run.sh), which plays both streams
  and then each test four times, and [`examples.py`](building-blocks/examples.py), the check of the in-context
  examples. The patches apply to the private trees the rows ran on, not to `main`.
- **Command**: `bash experiments/building-blocks/run.sh` (each stream as `PLAYER=glm ROW=hpb|hpb0 USERS=1` through the
  patched reference driver, then `frozen.py play` four times per test).
- **Runs**: `runs/arc1/GB-prime-agent-glm-5.3-flash-ws0-u1`, `runs/arc1/GB0-prime-agent-glm-5.3-flash-ws0-u1`.

### The library ceiling (GE0 `-ceiling-*`, Appendix F)

GE0 handed no library, the Re-ARC primitives the one-user stream's rules use (131 functions), or those the four-user
stream's use (153), on the tests alone.

- **Code** (here): [`library-ceiling/`](library-ceiling): `baselines.patch` (a superset of the building blocks' patch,
  adding the two primitives modules), `reference.patch`, [`primitives.py`](library-ceiling/primitives.py), which
  writes and checks the primitives modules, and [`run.sh`](library-ceiling/run.sh). `baselines.patch` carries code
  generated from Re-ARC's `dsl.py` (MIT licence, Michael Hodel).
- **Command**: `python experiments/library-ceiling/primitives.py write DIR`, then
  `bash experiments/library-ceiling/run.sh` (rows `hpl0`, `hpl0d1`, `hpl0d4`, two plays at a time).
- **Runs**: `runs/arc1/GE0-prime-agent-glm-5.3-flash-ws0-u1-ceiling-no-library`, `-ceiling-one-user-library`,
  `-ceiling-four-user-library`.

## The search that chose DAgger's configuration, and the other weight-based methods (Appendix H)

"How the configuration was chosen" and "Other weight-based methods" (policy gradient, PG; on-policy
self-distillation, OPSD): every run of the 20 machines in `runs/machines.json`, each machine's runs in its folder
under `runs/arc1/`, those laid under rows' names listed in its entry's `rows`.

- **Code** (here): [`configuration-search/`](configuration-search), whose [README](configuration-search/README.md)
  gives each machine's runs, the code that ran them and the command with its settings: the round-by-round and
  real-time DAgger, OPSD and PG rows of 17 to 20 October 2026, the validation grid (`grid-launch.sh`, `grid-exp.sh`,
  `grid-vals.sh`), the generalization cells (`generalisation/`, one JSON per cell), the switch test (`switch/`), the
  long and from-base training (`longtrain.py`, `modal_longtrain.py`), the final checks (`final_checks.py`), and the
  capacity runs whose adapters the `gen-cap-*` cells play (`capacity/`). The OPSD and PG machines ran the same engine
  and launcher as the search, so they are rows of the same table. The public `realtime.sh` (`ROW=dagger|opsd|pg`) is
  that engine's successor; it lacks the warm-up rows' `DOSE=replay` and PG's `PG_REWARD`.
- **Runs**: `runs/arc1/<machine>/<run>/`, the machines as `runs/machines.json` lists them.

## ARC-AGI-2 and ARC-AGI-3 on Kimi K3.1 Pro (Appendix K)

### The rows (Tables `tab:tracks-light` and `tab:arc3-pilot`, Figures `fig:plane-arc2` and `fig:plane-arc3`)

- **Code** (here): [`tracks-light/tracks-light.sh`](tracks-light/tracks-light.sh) plays the cells of one track on one
  CPU machine through the public baselines and benchmark at the commit it is set up with;
  [`tracks-light-floors.sh`](tracks-light/tracks-light-floors.sh) played the two floors;
  [`setup.sh`](tracks-light/setup.sh) brings a machine to the ready state. Every call is pinned to Moonshot AI.
- **Command**, with `CODE` a checkout of this repository and `OPENROUTER_API_KEY` set:

  | Rows | Benchmark commit | Setup | Launch on the machine |
  | --- | --- | --- | --- |
  | KC0, KC on ARC-AGI-2 and ARC-AGI-3 (28 Oct 2026) | `v0.4.0`, prime-agent 0.9.5 | `REF=v0.4.0 PRIME_AGENT_VERSION=0.9.5 CONFIG=arc2-small bash experiments/tracks-light/setup.sh HOST arc2` (`arc3-small`, `arc3` for the third) | `CAP_USD=1750 bash tracks-light-floors.sh arc2` (`arc3`) |
  | KP, KE, KE0 on ARC-AGI-2 | `v0.6.0` | `bash experiments/tracks-light/setup.sh HOST arc2` | `PREFIX=kimi-moonshot PROVIDER=moonshotai/mxfp4 PLAYS=2 CAP_USD=690 bash tracks-light.sh arc2 m hpl hpl0` |
  | KP0, KA, K, K0 on ARC-AGI-2 | `v0.6.0` | the same | `PREFIX=kimi-moonshot PROVIDER=moonshotai/mxfp4 PLAYS=2 CAP_USD=735 bash tracks-light.sh arc2 m0 ha h h0` |
  | KA, K, K0 on ARC-AGI-3 | `v0.6.0` | `bash experiments/tracks-light/setup.sh HOST arc3` | `PREFIX=kimi-moonshot PROVIDER=moonshotai/mxfp4 PLAYS=2 CAP_USD=450 bash tracks-light.sh arc3 ha h h0` |
  | KE pilot on ARC-AGI-3 (stream alone) | `1c499b7` | `REF=1c499b7 bash experiments/tracks-light/setup.sh HOST arc3` | `PREFIX=kimi-moonshot PROVIDER=moonshotai/mxfp4 PLAYS=0 CAP_USD=150 bash tracks-light.sh arc3 m hpl` (its `m` cell is not a paper row) |
  | KP pilot on ARC-AGI-3 (stream alone) | `9f7ef5e` | `REF=9f7ef5e bash experiments/tracks-light/setup.sh HOST arc3` | `PREFIX=kimi-moonshot PROVIDER=moonshotai/mxfp4 PLAYS=0 CAP_USD=118 bash tracks-light.sh arc3 m` |

  Each launch runs detached: `setsid nohup bash ~/loop/<script> ... > /dev/null 2>&1 < /dev/null &`.
- **Runs**: `runs/arc2/{KC0,KC,KP,KP0}-kimi-k3.1-pro-ws0-u1`,
  `runs/arc2/{K,K0,KA,KE,KE0}-prime-agent-kimi-k3.1-pro-ws0-u1`, `runs/arc3/{KC0,KC}-kimi-k3.1-pro-ws0-u1`,
  `runs/arc3/{K,K0,KA}-prime-agent-kimi-k3.1-pro-ws0-u1`, `runs/arc3/KP-kimi-k3.1-pro-ws0-u1-pilot`,
  `runs/arc3/KE-prime-agent-kimi-k3.1-pro-ws0-u1-pilot`.

### First sight (Table `tab:first-sight`, and Appendix K's reading of Kimi K3.1 Pro's effort)

- **Code**: the ARC-AGI-2 sample by the public
  [`scripts/arc2_calibration_sample.py`](../scripts/arc2_calibration_sample.py) (it ran when the configuration was
  named `arc2-small`; `main` differs only in that name); each model's answers by
  [`first-sight/arc2_first_sight.py`](first-sight/arc2_first_sight.py) (here, the public tasks' part of the private
  calibration script) and, on ARC-AGI-3, [`first-sight/arc3_first_sight.py`](first-sight/arc3_first_sight.py) (here),
  both on Modal.
- **Command**, with `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_ENVIRONMENT` and `OPENROUTER_API_KEY` set:

  ```bash
  uv run continual-arc data fetch --datasets arc2
  uv run python scripts/arc2_calibration_sample.py --out arc2-small-calibration-sample-seed0.json
  cd experiments/first-sight
  uv run --with modal python arc2_first_sight.py --items ../../arc2-small-calibration-sample-seed0.json \
      --model moonshotai/kimi-k3.1-pro --effort medium --instances 4
  #   Kimi K3: --model moonshotai/kimi-k3 --instances 2; Qwen3.5-9B: --model qwen/qwen3.5-9b --instances 4 --max-tokens 65536
  #   the effort reading: --instances 2 --effort low|medium|high --providers moonshotai/mxfp4
  uv run continual-arc data fetch --datasets arc3
  uv run --with modal python arc3_first_sight.py OUT medium 25 moonshotai/kimi-k3.1-pro moonshotai/kimi-k3
  ```

- **Runs**: not in `runs/`; see [What is not here](#what-is-not-here).

## Tables and figures

The tables and figures are computed from the runs' records by the paper's own scripts, which are not in this
repository yet: they read the open-model runs in the layout the machines that played them kept, which `runs/` does not
keep, so they cannot yet run from `runs/` ([`paper/README.md`](../paper/README.md#the-figures-and-tables) says which
of their inputs `runs/` holds). The paper's source, with every table as it sets it, is in [`paper/`](../paper).

## What is not here

- **The private set's rules (Appendix J).** The held-out rules of the private set, their generators and the code
  that wrote, screened, calibrated and selected them stay private by design, as a verified run needs them unseen.
- **The three openclaw cells with memory search (Appendix E).** No copy of their records survives.
- **The private engine the search drove.** The drivers in `configuration-search/` and `trained-rows-resets/`, and
  the patches in `building-blocks/` and `library-ceiling/`, run on the private training engine of each run's day,
  which `training/` consolidated; its revisions are not published.
- **The code that computes the tables and figures**, above.
- **First sight's records.** The answers behind Table `tab:first-sight` are kept with the private set's calibration
  and are not under `runs/`.
- **The letters and the teacher's state.** The letters QETE `-letter` read, and the `state.git` of GLM-5.3 Flash's
  stream they were written from, are not under `runs/`, which holds that stream's records, trajectories and
  `memory.json`.

## Placeholders

What was taken out of the files here, and what stands in its place:

| Placeholder | Stands for |
| --- | --- |
| `HOST`, `NODE_USER@<ip>` | the ssh destination of a rented machine |
| `NODE-<name>` | a rented machine, `<name>` the suffix of its folder under `runs/arc1/` |
| `REF_ROWS_NODE_A`, `REF_ROWS_NODE_B`, `REF_ROWS_NODE_E`, `REF_TESTS_NODE`, `RESETS_NODE` | the machines that played the reference rows, their tests and the resets |
| `NODE_USER`, `/home/USER` | the machines' login user and its home, which some scripts spell out |
| `ARTIFACTS_BUCKET`, `DATA_BUCKET`, `GCP_PROJECT` | the private Cloud Storage buckets and the Google Cloud project |
| `MODAL_APP`, `MODAL_CAPACITY`, `MODAL_JOBS_VOLUME`, `MODAL_HF_VOLUME`, `MODAL_ENVIRONMENT` | the Modal app, volumes and environment |
| `OPENROUTER_API_KEY`, `OPENROUTER_KEY_NAME` | the OpenRouter key, read from the environment, and its name |
| `~/benchmark-turns` | the private benchmark tree the building blocks and the ceiling played on |
| `R` | the researcher whose offline DAgger recipe the search compares against |
| "the author" | the paper's author, where a comment quotes a decision |
