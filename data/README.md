# Continual-ARC: the instances

Every instance the runs of the paper [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://arxiv.org/abs/2611.14505) were served on ARC-AGI-1, ARC-AGI-2 and ARC-AGI-3, as plain JSON: each
instance's grids or level, the schedules that served them and the ARC-AGI-1 tasks' own pairs, the grid tracks' the
same files [continual-arc.unify.ai](https://continual-arc.unify.ai) reads to show them. The runs themselves, their
records and trajectories, are in [`runs/`](../runs).

The same instances and tasks are on Hugging Face as
[`unifyai/continual-arc`](https://huggingface.co/datasets/unifyai/continual-arc), in Parquet with a split per schedule:
`load_dataset("unifyai/continual-arc", split="arc1_small_workstreams0")`, and ARC-AGI-3's levels in a config of their
own: `load_dataset("unifyai/continual-arc", "arc3", split="arc3_workstreams0")`.

- `tasks/arc1.json`: the 125 tasks of the ARC-AGI-1 track the GPT-6 Luna runs played, each with its own public pairs
  as ARC-AGI-1 publishes them (`train`, `test`), the split they come from (`split`) and the phase that serves them
  (`phases`): `stream` for the 25 tasks of the stream, from ARC-AGI-1's training set, and `epilogue` for the 100
  ARC-AGI-1 evaluation tasks of the held-out test, which the schedules serve after the stream as its epilogue.
- `schedules/<schedule>.json`: the twelve schedules the runs were served, each with its track, configuration, order
  (`mode`) and seed, and its slots in order: the task, its phase, its exposure index and gap, whether the learner was
  reset before it (`reset`), and on ARC-AGI-3 the seed its level is drawn from (`instanceSeed`). A schedule is named
  by its configuration, order and seed (`arc1-small-workstreams0`) and, past one user, by how many users it served,
  each with tasks of their own (`-u2`, `-u4`).
- `instances/<schedule>/<step>.json`: the instance served at each step of a schedule: its test `input`, the right
  `output`, and every demonstration it offers (`demos`, input-output pairs in the order a learner buys them, at most
  `demoCap`).
- On ARC-AGI-3 an instance is a fresh level of a game's first level, and its file holds the level: the `game` and its
  pinned `version`, the first `frame` (64 rows of 64 colors 0 to 15, top row first), the actions the game accepts
  (`availableActions`), the `baseline`, the actions a person needs for it (ARC Prize's count for the game's level 0,
  scaled by the solver's solution length for this level over level 0's), the `actionCap`, five times the baseline
  rounded up, and the solver's winning `solution`, each action `[id]`, or `[6, x, y]` for a click at column x and row
  y. Its `demos` are the three demonstrations a learner can buy, in the order bought, each the solver playing another
  fresh level of the game to its end: its first frame (`start`) and each action with the cells it changed (`steps`,
  each `{"action", "changes"}`, a change `[x, y, color]`).

| Schedule | Served to |
|---|---|
| `arc1-small-workstreams0` | the runs in the track's own order at seed 0 (`ws0` in their names): its stream, then the epilogue, the held-out tasks once each while the learner still learns, which no measure reads |
| `arc1-small-sequential0`, `arc1-small-random0` | the runs in the two other orders of the paper's Appendix B (`seq`, `rnd`) |
| `arc1-small-fresh0` | the held-out tasks alone, the learner wiped before each (`fresh`) |
| `arc1-small-workstreams0-u2`, `-u4` | the open-model runs of two and four users |
| `arc2-workstreams0` | the runs on ARC-AGI-2 (the paper's Appendix K) |
| `arc2-small-workstreams0` | the same stream, slot for slot and instance for instance, under the name the demonstration-cache runs `KC` and `KC0` were played under |
| `arc3-workstreams0` | the runs on ARC-AGI-3 (the paper's Appendix K), 120 levels of 15 games; `KC` and `KC0` played the same stream, slot for slot and level for level, under the name `arc3-small-workstreams` |
| `arc3-seen-test0`, `arc3-held-out-test0` | the two frozen tests the ARC-AGI-3 runs played after the stream, every level from the learner's state at the stream's end: four new levels of each of the stream's 15 games, and four levels of each of the 10 games the stream never serves |
| `arc3-small-fresh0` | the held-out test's 40 levels alone, the model wiped before each: the first-sight runs |

The runs name the first track's configuration `arc1-small`, as every run played before v0.6.0 does, when the
benchmark still named its configurations by size; its streams are the ones `arc1` builds today. Each run's own
`schedule.json` in `runs/` holds the slots it was served, and its `meta.json` its configuration, order, seed and
users, which name its schedule here.

The instances were written by serving each schedule through the benchmark's own environment
([`src/continual_arc/env.py`](../src/continual_arc/env.py)) with an oracle that buys every demonstration an instance
offers and then submits the right answer. The environment draws each instance and each demonstration from its slot's
seed alone, so these are exactly the pairs every learner on that schedule saw, and the benchmark draws them again from
the same schedule with the upstream data fetched (`uv run continual-arc data fetch`).

The ARC-AGI-3 levels were drawn again from each slot's seed by the source the environment serves them from
([`src/continual_arc/arc3/instances.py`](../src/continual_arc/arc3/instances.py)'s `LevelSource`) and checked against
every run on the track: the baseline, action cap and solution length in each of their 2,320 records, the first board
in each of the 1,426 transcripts that kept it, cell for cell, and the first board and length of each of the 2,102
demonstrations learners bought, and of the 190 the demonstration-cache floor `KC` showed again on later levels.
A level plays in its game, ARC Prize's MIT-licensed Python on `arcengine`, which the benchmark downloads at its pinned
version (`uv sync --extra arc3`, then `uv run continual-arc data fetch --datasets arc3`); the benchmark then draws the
level again from its seed (`LevelSource().instance(game, instanceSeed)`).

The grids come from ARC-AGI-1 and ARC-AGI-2 (Apache-2.0) and from the generators that draw their fresh instances,
Re-ARC's (MIT) and ARC-GEN's (Apache-2.0); each keeps its source's license. ARC-AGI-3's levels are drawn by this
repository's own generators (Apache-2.0), and their frames are the renderings of ARC Prize's games (MIT).

## Citation

If you use the data, please cite the paper; its BibTeX is in the [README](../README.md#citation).
