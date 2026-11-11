# Continual-ARC: the instances

Every instance the runs of the paper [*Continual-ARC: Measuring Skill Acquisition over a Working
Lifetime*](https://arxiv.org/abs/2611.14505) were served on ARC-AGI-1 and ARC-AGI-2, as plain JSON: each instance's
grids, the schedules that served them and the ARC-AGI-1 tasks' own pairs, the same files
[continual-arc.unify.ai](https://continual-arc.unify.ai) reads to show them. The runs themselves, their records and
trajectories, are in [`runs/`](../runs).

- `tasks/arc1.json`: the 125 tasks of the ARC-AGI-1 track the GPT-6 Luna runs played, each with its own public pairs
  as ARC-AGI-1 publishes them (`train`, `test`), the split they come from (`split`) and the phase that serves them
  (`phases`): `stream` for the 25 tasks of the stream, from ARC-AGI-1's training set, and `epilogue` for the 100
  ARC-AGI-1 evaluation tasks of the held-out test, which the schedules serve after the stream as its epilogue.
- `schedules/<schedule>.json`: the eight schedules the runs were served, each with its track, configuration, order
  (`mode`) and seed, and its slots in order: the task, its phase, its exposure index and gap, and whether the learner
  was reset before it (`reset`). A schedule is named by its configuration, order and seed (`arc1-small-workstreams0`)
  and, past one user, by how many users it served, each with tasks of their own (`-u2`, `-u4`).
- `instances/<schedule>/<step>.json`: the instance served at each step of a schedule: its test `input`, the right
  `output`, and every demonstration it offers (`demos`, input-output pairs in the order a learner buys them, at most
  `demoCap`).

| Schedule | Served to |
|---|---|
| `arc1-small-workstreams0` | the runs in the track's own order at seed 0 (`ws0` in their names): its stream, then the epilogue, the held-out tasks once each while the learner still learns, which no measure reads |
| `arc1-small-sequential0`, `arc1-small-random0` | the runs in the two other orders of the paper's Appendix B (`seq`, `rnd`) |
| `arc1-small-fresh0` | the held-out tasks alone, the learner wiped before each (`fresh`) |
| `arc1-small-workstreams0-u2`, `-u4` | the open-model runs of two and four users |
| `arc2-workstreams0` | the runs on ARC-AGI-2 (the paper's Appendix K) |
| `arc2-small-workstreams0` | the same stream, slot for slot and instance for instance, under the name the demonstration-cache runs `KC` and `KC0` were played under |

The runs name the first track's configuration `arc1-small`, as every run played before v0.6.0 does, when the
benchmark still named its configurations by size; its streams are the ones `arc1` builds today. Each run's own
`schedule.json` in `runs/` holds the slots it was served, and its `meta.json` its configuration, order, seed and
users, which name its schedule here.

The instances were written by serving each schedule through the benchmark's own environment
([`src/continual_arc/env.py`](../src/continual_arc/env.py)) with an oracle that buys every demonstration an instance
offers and then submits the right answer. The environment draws each instance and each demonstration from its slot's
seed alone, so these are exactly the pairs every learner on that schedule saw, and the benchmark draws them again from
the same schedule with the upstream data fetched (`uv run continual-arc data fetch`). ARC-AGI-3's games are not here:
the benchmark downloads them from ARC Prize at a pinned version.

The grids come from ARC-AGI-1 and ARC-AGI-2 (Apache-2.0) and from the generators that draw their fresh instances,
Re-ARC's (MIT) and ARC-GEN's (Apache-2.0); each keeps its source's license.
