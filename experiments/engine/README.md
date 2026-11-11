# The engine's revisions

The private training engine the paper's open-model experiments ran on (`continual-arc-training`, deployed on each
machine as `~/loop`), at every commit a play under [`runs/`](../../runs) ran code from, and the baselines package those
rows played with (`continual_arc_baselines` at `continual-arc-research` `1543c2e`), which the building blocks' and the
library ceiling's patches apply to. [`plays.tsv`](plays.tsv) gives, for each of the 564 plays that ran on them, the
trees it ran from, the code changed under it while it ran and how that is known. The engine was consolidated into
[`training/`](../../training), which runs the same methods on the public packages; these trees are what ran.

Each folder is named by the commit's first seven characters. The files were copied as the rest of
[`experiments/`](../README.md#placeholders) were, with the same placeholders and a few more: `NODE-<name>` for every
rented machine (`REF_ROWS_NODE_A` to `_E` and `REF_TESTS_NODE` for the reference rows' and their tests'), `<ip>` and
`<instance>` for an address and a cloud's instance id, "a GPU cloud" for the vendor the search rented from,
`~/benchmark-turns` and `~/benchmark-v2` for the benchmark trees beside the engine, `~/run-supervisor` for the run
supervisor's folder and `gen-cache` for a cell's cache. The author's quoted words are taken out of the dated
attributions in the comments, which keep their date.

## How a machine held the code

- **The engine tree**: `~/loop` on the search's and the trained rows' machines; `~/loop-turns` on the reference
  machines from 28 Oct 2026 (`refs-deploy.sh` at `ee7a4b3`); `~/loop-own` for the reference rows at two and four
  users; a code bundle's `loop/` for the generalisation cells, the switch test and the `cross-u4` plays.
- **Beside it**: the benchmark, the baselines and, for the trained rows, ask-sd, the trainer's package; and tools
  copied from another commit: `~/frozen-tools` (`frozen.sh`, `frozen.py`, `schedules/`, by `frozen-launch.sh` or
  `refs-launch.sh`), `~/scratch-tools` and `~/weights-tools`, a cell's `~/gen/tools` (the list
  `search/generalisation/tools.sh` copies), `~/glm-harness` (`glm_harness.py`, `glm_harness.sh`, `keep.sh`), and the
  later plays' launcher (`more-plays.sh`).
- **How it changed**: until 19 Oct the engine reached a machine as an rsync of a checkout, whose tracked files were
  clean at each sync; from then `search/deploy-code.sh` shipped the pushed main of the engine, the benchmark, the
  baselines and ask-sd together. A round or a phase reads the files as they are when it starts, so a sync during a run
  changed the code of its later rounds, and a few files were changed by hand ([`changes/`](changes)).

## What a folder holds

- **A whole tree** (40 folders, the commits a play's engine tree was): the drivers the experiments run and every file of
  the engine they import or name, as the commit held them. Left out are the engine's notes (`docs/`, the READMEs,
  `AGENTS.md`), its tests but `test_reference.py` (which the deploy checks run), and what no experiment runs: the
  earlier development loop's experiments, analyses and viewers.
- **Some files** (44 folders): what a play took from that commit beside another tree (a launcher's copies, a cell's
  tools, a driver), or what a sync carried under a running play. `plays.tsv` names the tree each sits beside.
- [`baselines-1543c2e/`](baselines-1543c2e): the modules of `src/continual_arc_baselines` the engine imports and theirs,
  with `pyproject.toml`. `secrets.py`, which reads keys from the company's Secret Manager, is left out: set
  `OPENROUTER_API_KEY` in the environment.
- [`changes/`](changes): the code changed by hand under some plays, which no commit holds, each a patch against the
  folder it was made in.
- [`plays.tsv`](plays.tsv): one line per play: `play` (its folder under `runs/`), `ran_on`, `started_utc`, `trees`
  (each a role and a folder: `loop`, `frozen-tools`, `cell-tools`, `driver` and so on), `changed_while_running` (each
  sync's time and folder), `hand_changes` (a file of `changes/`), `baselines` and `benchmark` (where the play's records
  name them) and `known_by`.

## Experiment by experiment

| Experiment ([`experiments/README.md`](../README.md)) | Engine | Beside it |
| --- | --- | --- |
| The reference rows at one user (QC0 to QE0, GC0 to GE0): streams from 01:31 UTC on 28 Oct 2026, tests from 29 Oct | [`ee7a4b3`](ee7a4b3) as `~/loop-turns` | baselines [`1543c2e`](baselines-1543c2e); the benchmark `continual-arc-internal` `060ba32`, whose `src/continual_arc/` and `tests/` are this repository's at `69b4e19`; `repl-guard.sh` ([`e86174d`](e86174d)), which ends a model-run REPL past 64 GB, beside most later test plays |
| The reference rows at two and four users (`-u2`, `-u4`), from 22:34 UTC on 28 Oct | [`b4b564a`](b4b564a) as `~/loop-own` | as above |
| The teacher's notes (QT, QTE, QETE) and the resets played in the same window, 5 Nov | [`ee7a4b3`](ee7a4b3) | the driver [`2d9c273`](2d9c273) as `~/glm-harness`; baselines and benchmark as above |
| QETE with the teacher's letter, 8 and 9 Nov | [`ee7a4b3`](ee7a4b3) | the driver [`def2b6f`](def2b6f); the letters by `teach.py`, rounds 99 to 101 at [`6d522a8`](6d522a8) and 102 to 199 at [`c6da762`](c6da762) (102 to 112 from an uncommitted copy the same as it, 108 to 112 through [`changes/letters_after_end.py`](changes/letters_after_end.py)), their counts by [`90ec5cd`](90ec5cd)'s |
| DAgger at one, two and four users: the streams, 28 Oct | one user [`1e7af11`](1e7af11) ([`b4b564a`](b4b564a) from 22:23:56 UTC), two and four [`b4b564a`](b4b564a) | ask-sd, not published |
| DAgger's tests, their later plays and replays, 2 to 7 Nov | [`b4b564a`](b4b564a) | frozen tools [`6a55104`](6a55104); the launcher [`1df4fdd`](1df4fdd); replays by [`19b4383`](19b4383)'s `replay.py` and [`4411d5f`](4411d5f)'s `replay.sh`; baselines [`1543c2e`](baselines-1543c2e), with [`changes/baselines-overrun-fix-on-1543c2e.patch`](changes/baselines-overrun-fix-on-1543c2e.patch) from 18:59 UTC on 4 Nov |
| DAgger from the base model (`-from-base`), 5 to 7 Nov | [`b4b564a`](b4b564a) | the fits by [`8d57163`](8d57163)'s `longtrain.py --from-base` and `frombase.sh`; `dagger-u4-longtrain` by [`760f4a4`](760f4a4)'s `longtrain.py` on Modal, from a bundle of [`b4b564a`](b4b564a); tests as above |
| The trained rows' resets (the served tests, the page alone) | 29 and 30 Oct [`caca3b8`](caca3b8); 1 Nov [`6a55104`](6a55104); 5 Nov [`b4b564a`](b4b564a) | from 5 Nov as DAgger's tests; replays as above |
| Expert guidance as building blocks (GB, GB0), 8 Nov | [`ee7a4b3`](ee7a4b3) with [`../building-blocks/reference-blocks.patch`](../building-blocks/reference-blocks.patch), as `~/loop-blocks` | [`9c9c662`](9c9c662)'s `reference.sh`, `keep.sh` and `blocks/run.sh` ([`9b470c8`](9b470c8)'s `blocks/run.sh` from 01:23 UTC, before the first test play); baselines [`1543c2e`](baselines-1543c2e) with [`../building-blocks/baselines-blocks.patch`](../building-blocks/baselines-blocks.patch) |
| The library ceiling, 8 and 9 Nov | [`ee7a4b3`](ee7a4b3) with [`../library-ceiling/reference.patch`](../library-ceiling/reference.patch), as `~/loop-ceiling` | [`d02af30`](d02af30)'s `keep.sh` and `ceiling/run.sh`; baselines [`1543c2e`](baselines-1543c2e) with [`../library-ceiling/baselines.patch`](../library-ceiling/baselines.patch) |
| The search that chose DAgger's configuration, and PG and OPSD (Appendix H) | below | below |

The search's machines, as [`../configuration-search/README.md`](../configuration-search/README.md) groups them:

| Runs | Engine at the start | Later, or beside |
| --- | --- | --- |
| The round-by-round rows of 17 and 18 Oct (`paper.sh`) | `dagger` [`807a58a`](807a58a); `dagger-u10`, `dagger-u1` [`69b7f2c`](69b7f2c); `dagger-u8` [`f65e45f`](f65e45f); `pg-u1`, `-u2`, `-u4` [`3c635f3`](3c635f3); `dagger-u2`, `dagger-u4` [`03dc9de`](03dc9de) | the syncs `plays.tsv` lists: [`2b68edd`](2b68edd), [`df44752`](df44752), [`f65e45f`](f65e45f), [`47c65fe`](47c65fe), [`ad1ffa9`](ad1ffa9), [`0ad0ac2`](0ad0ac2), [`eacc03c`](eacc03c), [`d87245f`](d87245f), [`6d37697`](6d37697), [`3c635f3`](3c635f3), [`03dc9de`](03dc9de) |
| The real-time rows of 19 Oct (`realtime.sh`): the first, the warm-up rows, the labeller rows | `opsd-u1-lag1`, `pg-u1-lag1` [`1706fbd`](1706fbd); `-s24` [`05a84cd`](05a84cd), [`575db21`](575db21), [`9459067`](9459067), [`7782731`](7782731); `-s24f` [`7098f9e`](7098f9e); `-pf` [`45ad50a`](45ad50a), [`20cd1e9`](20cd1e9), [`4495173`](4495173), [`625b0e4`](625b0e4); `-agg` [`8f77082`](8f77082), [`5495acc`](5495acc) | [`189df21`](189df21)'s `loop.py`, [`c3ad0a4`](c3ad0a4)'s `train.py`, [`e6dada1`](e6dada1), [`7098f9e`](7098f9e), [`2517a1b`](2517a1b) |
| The final real-time rows of 20 Oct, `{dagger,opsd,pg}-u{1,2,4}-lag1` | [`9f4da15`](9f4da15), [`e0b3dba`](e0b3dba), [`e9d6d04`](e9d6d04), [`45fd98f`](45fd98f), [`ec76734`](ec76734), [`41d3b9e`](41d3b9e), by machine in `plays.tsv` | PG's update on wins alone, [`145d277`](145d277)'s `train.py`, `realtime.py` and `realtime.sh`, and its warm-up fix ([`changes/pg-warmup-on-145d277.patch`](changes/pg-warmup-on-145d277.patch)) on the PG runs; OPSD's demo cache ([`changes/opsd-demo-cache-on-45fd98f.patch`](changes/opsd-demo-cache-on-45fd98f.patch)) on the OPSD runs; on 22 Oct the recent dose, which [`fda8f0c`](fda8f0c)'s `search/switch/switch-recent.py` wrote into each DAgger run's `realtime.py` and `realtime.sh` ([`changes/recent-dose-on-41d3b9e.patch`](changes/recent-dose-on-41d3b9e.patch) is its change to [`41d3b9e`](41d3b9e)'s) |
| The validation grid, 19 Oct | [`b0bad70`](b0bad70); `val-b10-block` [`7782731`](7782731) | `grid-exp.sh` of [`f119376`](f119376), [`eb16792`](eb16792) or [`5f90813`](5f90813) |
| The transfer references, 20 Oct | [`8be7d02`](8be7d02) | |
| The final rows' frozen, weights-alone and scratch plays, 20 to 28 Oct; `qwen-f`, `qwen-h0` | each run's own | frozen tools [`356c72c`](356c72c), weights tools [`a2622a5`](a2622a5), scratch tools [`b30cfc1`](b30cfc1), [`f2bb61a`](f2bb61a), [`22cb5c4`](22cb5c4); `qwen-f` and `qwen-h0` on [`356c72c`](356c72c) whole |
| The layout test, 20 Oct | `pooltest-u1-lag1` [`45fd98f`](45fd98f), `pooltest-u4-lag1` [`41d3b9e`](41d3b9e) | |
| The switch test, 24 Oct, and `cross-u4`, 25 Oct | the bundle `code-dagger-rows`: [`41d3b9e`](41d3b9e) with [`changes/recent-dose-on-41d3b9e.patch`](changes/recent-dose-on-41d3b9e.patch) | frozen tools [`e7e1280`](e7e1280) (`cross-u4`: [`c0eb55f`](c0eb55f)); `switch-run.sh` and `switch-recent.py` of [`fda8f0c`](fda8f0c) |
| The generalisation cells (`gen-*`), 25 Oct to 2 Nov | the bundle `code-dagger-rows-turns`: [`41d3b9e`](41d3b9e) with [`changes/turns-on-41d3b9e.patch`](changes/turns-on-41d3b9e.patch), from 20:50 UTC on 27 Oct [`changes/turns-weighting-on-41d3b9e.patch`](changes/turns-weighting-on-41d3b9e.patch) | each cell's tools: [`25fa3da`](25fa3da), [`7bc749c`](7bc749c), [`8db6287`](8db6287), [`c5d3a58`](c5d3a58), [`7c1bf44`](7c1bf44), [`22cb5c4`](22cb5c4), [`4aa4edf`](4aa4edf), [`797c7c1`](797c7c1), [`983b52e`](983b52e), [`8cf82ea`](8cf82ea), [`1e7af11`](1e7af11), [`b97030b`](b97030b), [`c685b9c`](c685b9c), [`576b473`](576b473), [`1daee98`](1daee98) |
| The gating run `dagger-u4-final`, 26 Oct | [`d66e5ef`](d66e5ef) | `search/final_checks.py`, which the run does not import, of [`7bc749c`](7bc749c), [`f5fb37d`](f5fb37d) and [`d3f2232`](d3f2232) |
| `pg-u1-final`, `h-u1`, `pg-u2-disjoint`, `h-u2`, 28 and 29 Oct | `pg-u1-final`, `h-u1` [`1e7af11`](1e7af11) ([`b4b564a`](b4b564a) from 22:23:56 UTC); the others [`b4b564a`](b4b564a) | |
| `dagger-u1-final`, `dagger-u2-disjoint`, `dagger-u4-disjoint` and their plays | as DAgger above | |

The search's plays name no baselines, benchmark or ask-sd: each was its repository's main (the baselines' branch
`turns` from 25 Oct) at the deploy that shipped the play's engine tree, and before 19 Oct the checkout rsynced with it.

## Running a driver against its revision

Each driver is run as [`experiments/README.md`](../README.md) and
[`configuration-search/README.md`](../configuration-search/README.md) give its command, with the engine folder where
the command expects its tree and the tools beside it, after any patch the table names:

```bash
E=experiments/engine
# the four-user DAgger stream, then its frozen tests
cp -R $E/b4b564a ~/loop
ROW=dagger USERS=4 LR=1e-4 START_AFTER=24 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 bash ~/loop/realtime.sh
mkdir -p ~/frozen-tools && cp -R $E/6a55104/frozen.sh $E/6a55104/frozen.py $E/6a55104/schedules ~/frozen-tools/
bash ~/frozen-tools/frozen.sh dagger-u4-disjoint 0,1,2,3,4,5,6,7
# GB's trees, as blocks/deploy.sh staged them
cp -R $E/ee7a4b3 ~/loop-blocks && git -C ~/loop-blocks apply "$PWD/experiments/building-blocks/reference-blocks.patch"
cp $E/9c9c662/reference.sh $E/9c9c662/keep.sh ~/loop-blocks/ && mkdir -p ~/loop-blocks/blocks && cp $E/9b470c8/blocks/run.sh ~/loop-blocks/blocks/
cp -R $E/baselines-1543c2e ~/cab-blocks && git -C ~/cab-blocks apply "$PWD/experiments/building-blocks/baselines-blocks.patch"
```

`git apply` works in a folder outside a repository. A tree also needs the packages it imports: the benchmark (the
public one at the commit given above for the open-model rows), the baselines (`baselines-1543c2e/src` on the path, with
`httpx`) and, for the trained rows, ask-sd and the baselines' `carc_stream` (three constants `loop.py` reads from
`carc_stream.team`), which are not published. Every top-level module of every whole tree imports with only
`baselines-1543c2e/src`, the public benchmark and `httpx` beside it (ask-sd's two importers, `train.py` and
`teacher_score.py`, aside), and every driver with a command line answers `--help`; the reference rows' deploy check,
`test_reference.py`, passes on `ee7a4b3` and `baselines-1543c2e` (40 tests), with the building blocks' patches (45)
and with the library ceiling's (50).

## How each commit is known

`known_by` in `plays.tsv` is `recorded` where a file the play left names or holds its code: the run supervisor's
`manifest.json` and `code.diff` (every play from 4 Nov), a cell's `tools-commit` and bundle checksum, a play's code
archive, or a tree copied off its machine before it ended, whose files hash to the commit. It is `deploy` where the
sessions that ran the experiment show the sync of a clean tree or of the pushed main before the play's start; GitHub's
record of the engine's pushes gives what main held at each moment.

## What is not here

- **ask-sd, the baselines of the search's days and `carc_stream`**, the private packages the engine imports besides
  `baselines-1543c2e`, and the bundles' copies of them.
- **The engine's later commits.** After the plays above its main holds work the paper does not report: the stream
  sweep for a review (6 to 9 Nov), the checks of where the four-user from-base fit's gain comes from (8 and 9 Nov) and
  more plays of the two-user fit (10 Nov).
- **The changes no copy keeps**: on 25 Oct a default in `realtime.sh` (the oracle relay's providers, set to Z.ai then
  Fireworks) was changed by hand on two DAgger runs' machines, read only by a relaunch neither run had; and the layout
  test's resumes ran on work in progress that was never pushed, recording no round.
- **The plays set aside**, and every other play not under `runs/`.
