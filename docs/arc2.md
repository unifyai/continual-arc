# Continual-ARC: the ARC-AGI-2 track

## Why

ARC-AGI-2's rules are harder than ARC-AGI-1's, and its evaluation tasks were calibrated on people (each was solved by
at least two human testers), so frontier models are far from saturating it, where they are close to it on the first
edition. A first ARC-AGI-2 track, removed on 19 Oct 2026, had no generators and so rested on augmentation: a
task's own pairs under the rotations, reflections and recolorings its rule survives, which gives a few dozen frames
of one task and no fresh instance. ARC-GEN (Moffitt 2025, github.com/google/arc-gen) has published generators for 500 ARC-AGI-2 training tasks
since 25 Mar 2026, so the second track keeps the first's protocol unchanged: a stream of fresh instances of tasks that
return after gaps, demonstrations on request, and the two frozen tests after it (Dan, 27 Oct 2026: "For now, we just
need to get the static variant working. For each of the 3 tracks.").

## Decisions

| decision | choice | why |
|---|---|---|
| stream tasks | the ARC-AGI-2 public training tasks new to the second edition whose ARC-GEN generator passes the screen below: 117 of the 172 ARC-GEN covers (`continual_arc/arcgen_screen.json`, `arcgen.pool()`) | 767 of the 1,000 training tasks are ARC-AGI-1 tasks (391 of its training tasks, 376 of its evaluation tasks), and a stream drawn from all of them was an ARC-AGI-1 stream, solved at first sight by a model at 18 percent on the ARC-AGI-2 leaderboard (23 Sep 2026); of the 233 new ones ARC-GEN covers 172, and none of the evaluation tasks |
| held-out test | the ARC-AGI-2 public evaluation tasks new to the second edition, 114 of 120, served with their own pairs (`FixedArcSource` over `arc2` as `arc2_eval`); `arc2-small` holds out 100 of them, drawn as the first track draws its 100 | as the first track serves the ARC-AGI-1 evaluation tasks: tasks no generator covers, whose own pairs are the only instances there are |
| instances and demonstrations | every test input and every demonstration of the stream and the seen test drawn fresh from the task's ARC-GEN generator, seeded by the slot (`ArcGenSource`); a demonstration never shares its input with the test or another demonstration | the first track's rule, with ARC-GEN in Re-ARC's place |
| configurations | `arc2-smoke` and `arc2-small`, the first track's smoke and small shapes: `arc2-small` 25 stream tasks of 8 instances each, 200 instances, the same workstreams parameters, 100 held-out tasks; the same charges and caps | one protocol on every grid track, so a learner runs on either unchanged and its numbers read the same way |
| canonical protocol | `arc2-small`'s stream, then its two frozen tests, seeds 0 to 2, 1 to 16 users (`canonical_protocol("arc2")`) | as the first track's; the tracks are read apart, never pooled |
| data | `continual-arc data fetch --datasets arc2` clones ARC-AGI-2, ARC-GEN (pinned at `a15cbdb`, 18 Apr 2026) and ARC-AGI, whose ids tell which ARC-AGI-2 tasks are new | nothing vendored, as for the first track |

ARC-GEN is under the Apache License 2.0, which allows the use, modification and redistribution of the code and of
what it generates, so instances and demonstrations served from it, and files of them such as a calibration sample,
can be published. ARC-AGI-2's tasks are under the Apache License 2.0 as well.

## Drawing an instance

ARC-GEN is one module per task, `tasks/task_<id>.py`, with `generate()`, which draws every parameter it is not given
through its `common` module's calls to Python's `random`, and `validate()`, which calls `generate` with the
parameters of the task's published pairs. `continual_arc.arcgen.ArcGen` loads a task's module from the pinned
checkout and hands `common` a `random.Random` of its own for each call, seeded by the call, so a pair is a pure
function of the task and the seed and nothing a learner does with the global `random` reaches it. It draws exactly
what ARC-GEN's own `random.seed(s)` and `generate()` draw.

Many generators redraw inside one call until a layout fits, and on some seeds some never stop. A call is therefore
stopped once it has taken ten million random numbers, counted rather than timed, so that it stops at the same point
on every machine. A call takes hundreds as a rule; the most any pool generator took in one of the screen's draws is
7.7 million (`342ae2ed`), and the most any took on average 0.36 million (`20fb2937`). A call that stops, raises or
makes a grid the benchmark refuses is drawn again at the next derived seed, as Re-ARC's are: the pool's generators
did none of these in any of the screen's draws, but a run draws far more pairs than the screen, and a draw that
failed every time a run resumed would stop that run for good.

## The screen

ARC-GEN ships no solver or verifier, so unlike Re-ARC's pairs a drawn pair is not checked against an independent
program as it is served. A generator enters the pool only if it passes a screen of two parts (`arcgen.screen_task`,
`scripts/screen_arcgen.py`), whose every reading is recorded in `src/continual_arc/arcgen_screen.json`.

**Fidelity.** ARC-GEN's `validate()` reproduces every published pair of the task, its train and test pairs, exactly,
at the pinned ARC-AGI-2 commit: the code that draws a fresh pair is the code that draws the task's own. This is the
check ARC-GEN's authors use, and all 172 generators pass it. And every draw keeps what all the task's published pairs
have alike of four traits a rule usually fixes: whether the output has the input's shape, the output's shape, its
size over the input's, and whether the output's colors are all among the input's. A draw that breaks one has left
the published rule. A yes-or-no trait counts only when every published pair has it: that no published output has
its input's shape, or that each adds a color, can be chance (an output whose size counts the input's layers can come
out the input's size). The check drops two generators that pass everything else, `342dd610` and `5792cb4d`, which in
2 and 6 percent of draws put an output color the input lacks, a cell another object hid when the input was drawn, so
the answer cannot be read from the input.

**Mechanical filters**, the held-out season's (the paper's appendix on it), over 1,000 draws at seeds of the
screen's own, more than the season's 300 because a run draws thousands of pairs of each task and a generator that
repeats its inputs or fails on a rare seed shows it only at that scale:

* every call finishes, within ten million random numbers and ten CPU seconds (a backstop for a loop that draws
  nothing), and raises nothing;
* every grid is valid and every input at least 3 by 3;
* at least 90 percent of inputs and 10 percent of outputs are distinct;
* the output differs from the input in at least 95 percent of draws, and is not always the input under one fixed
  rotation or reflection;
* five draws made again at the end come out as they did, so a pair depends on its seed alone.

The season's redraw filter (fewer than one draw in five redrawn) has no counterpart: ARC-GEN's generators redraw
inside a call, which the limit on random numbers bounds instead.

## What the screen found

ARC-GEN has 900 generators, 400 for the ARC-AGI-1 training tasks and 500 for ARC-AGI-2 training tasks; of those 500,
328 are ARC-AGI-1 evaluation tasks and 172 are among the 233 training tasks new to the second edition. All 172 were
screened at ARC-GEN `a15cbdb` and ARC-AGI-2 `f3283f7`, and 117 passed, the pool. How many passed each part:

| part of the screen | passed, of 172 |
|---|---|
| `validate()` reproduces every published pair | 172 |
| every draw keeps what the published pairs share | 169 |
| every call finishes and raises nothing (20 ran past ten million random numbers, 2 past ten CPU seconds, 3 raised) | 147 |
| every grid valid | 170 |
| every input at least 3 by 3 | 166 |
| at least 90 percent of inputs distinct | 153 |
| at least 10 percent of outputs distinct | 169 |
| the output differs from the input in at least 95 percent of draws | 167 |
| the output never always one rotation or reflection of the input | 172 |
| all of them: the pool | 117 |

Each failing generator's reasons, and every generator's readings, are in the record. The pool is far more than
`arc2-small`'s 25 stream tasks and `arc2-smoke`'s 8 need, so a larger configuration can be added on the same pool,
nested as the first track's are.

## Running it

```bash
uv run continual-arc data fetch --datasets arc2
uv run continual-arc schedule make --config arc2-small --seed 0 --no-epilogue --out arc2-stream.json
uv run continual-arc schedule make --config arc2-small --seed 0 --test held-out --out arc2-held-out.json
uv run continual-arc schedule make --config arc2-small --seed 0 --test seen --out arc2-seen.json
uv run python scripts/screen_arcgen.py              # the screen again, about twenty minutes on six cores
uv run python scripts/arc2_calibration_sample.py   # four instances of each stream task, three demonstrations each
```

A learner, a harness or a training loop that runs on the first track runs on this one unchanged: the text a learner
reads names no edition.

## What stays open

The stream's tasks are ARC-AGI-2 training tasks, since ARC-GEN covers none of the evaluation tasks, and the training
set was not calibrated on people as the evaluation sets were ("very easy to very difficult", in ARC Prize's words),
so the stream need not be as hard as the held-out test, which is the calibrated evaluation tasks. No person has
solved the generated instances either: the screen shows that a generator reproduces its task's published pairs and
keeps what they share, not that every draw is as clear as they are. The calibration sample
(`scripts/arc2_calibration_sample.py`) measures how often a model solves the stream's instances at first sight, so
that the paper can say how much is left to learn. The pool is frozen as the streams are: a new pool,
from a new ARC-GEN commit or a changed screen, is a new track. The drift mode, rules that change along a stream,
comes later.
