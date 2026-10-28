# Continual-ARC-3: the ARC-AGI-3 track

## Why

ARC-AGI-3's 25 public games are interactive: a 64x64 board of 16 colors that reacts to seven actions, with no
instructions, so a player must find out by acting what each game wants. Each game has several levels, and ARC Prize
publishes how many actions a person needed for each. They are MIT-licensed Python on ARC Prize's `arcengine`, and a
level is data: a list of sprites at places, with a few parameters, which the game's class reads when it is built. So a
level of our own plays unchanged in the real engine, and a generator of fresh levels gives a game what Re-ARC's
generators give an ARC-AGI-1 task: fresh instances of one rule, without end, so that a return is never a lookup.

The first version of the track (23 Sep to 19 Oct 2026) served the public levels themselves, one per instance, climbing
through each game's levels as it returned. A level met again was the same level, and the next level of a game adds
sprite types and changes its rules, so a return measured neither memory of a rule nor its use on fresh input. It was
taken out of version 1 on 19 Oct 2026. This version serves fresh levels of each game's first level instead (Dan, 27 Oct
2026), the static variant; rules that change over the stream, of which a game's later levels are a first case, are
the drift mode, which comes later.

## Decisions

| decision | choice | why |
|---|---|---|
| task | one game's rules at its first level, level 0: its sprite types, with level 0's own pixels and colors, and its level data | later levels add sprite types in 22 of the 25 games and change rule parameters in several (tr87's `alter_rules`, vc33's `Gravity`, ls20's `Fog`); they are left for the drift mode |
| instance | a fresh level of those rules, drawn from the slot's instance seed by the game's generator, `continual_arc.arc3.generators.<game>` (`GAME`, `generate(rng)`, optionally `solve(level)`); the public levels are never served | a cached solution never fits the next level |
| games in the track | the games whose generator and solver pass the gate (`levels.check` at 50 seeds), `arc3.games.TRACK_GAMES` | a game whose generator cannot be shown sound stays out, and the paper says how many passed |
| split | the track's games split once, at seed 0, into 60 percent stream games and the rest held out (`arc3.games.split`) | 25 games are too few to draw a held-out set per run seed; held-out games are generated too, so their instances are fresh and their demonstrations exist |
| difficulty | a level is kept only if the game's solver solves it within 0.5x to 2x of level 0's own solution length by the same solver | every instance of a game is about as hard |
| baseline | level 0's human action count (`metadata.json`'s `baseline_actions[0]`) times the instance's solution length over level 0's | the published count is for level 0 alone; the solver's lengths scale it to each fresh level |
| action cap | `ceil(5 x baseline)`, at most 1000; every action and every level reset counts toward it | the first version's cap |
| solved instance | `min(5, (actions + resets) / baseline)`, plus 1 for each demonstration | play is charged in baselines, as the first version charged it |
| failed instance | the cap reached unsolved: the whole budget, `max_cost = 5 + 1 x 3 = 8` | failing is never cheaper than trying, as on the ARC-AGI-1 track |
| give-up | none in the protocol; `GiveUp` is the runner's runaway guard and an adapter's last resort, charged as a failure | 20 Sep 2026: with a give-up, nearly every frontier model's instance ended in one, read as the cheapest way to "use fewer actions" |
| demonstration | the solver's play of another fresh level of the game, its seed derived from the instance's and the request's index, any layout equal to the instance's skipped; one unit, at most three an instance, at any time while the instance is open, leaving the level being played as it was | as on track 1, one more example of the rule at a price; asking before acting or after a game over, the moments the design named, are both allowed, and allowing it mid-level as well gives a learner nothing a request before acting would not, so the simplest rule is also fair |
| reproducibility | every solve is bounded by states expanded, never by time, and replayed in a fresh game before it is kept | a level is solved, or not, the same way on every machine |
| the cache | only solutions are kept on disk, one file a draw, keyed by game, pinned version, a digest of the generator module's source and the draw's seed, with a digest of the layout drawn; levels are drawn again from their seeds | generating is deterministic and under 2 s, solving can take a minute; the layout's digest catches a generator that is not deterministic before a stale solution is served |
| precompute | `continual-arc precompute --config arc3-small --seed 0` solves a configuration's every instance and its three demonstrations ahead of a run, in processes of its own | a run never waits on a solver |
| users | rounds of users as on track 1, every open instance a game of its own, so a frozen test is one round | the frozen tests need it; a collective stream works the same way |

## The protocol

A learner sees the game under the run's alias, the frames of the last engine call (the reset's one frame on a fresh
instance; an action that animates returns several, the last being the board), the game's state and the actions it
offers, its actions, resets and demonstrations against their caps, the instance's baseline, and the budget's figures
(`Arc3LearnerObservation`). It acts with `Act(action, x, y)` (ACTION1 to 7; ACTION6 a click at column x, row y),
`ResetLevel()` or `RequestDemos()`. An action that is not valid (an id outside 1 to 7, one the game does not offer, a
click outside the board) or that the engine ignores (the game is over) is charged and counted as invalid. Feedback on
an action is its verdict alone (the state, whether the level completed, whether it was valid, how many frames it
rendered); the board it left comes with the next observation, so that a run's event log does not hold every frame.

Records are `InstanceRecord`s, read by every measure unchanged: `wrong_attempts` holds the instance's actions (resets
excluded), `demo_requests` its demonstrations, `invalid_submissions` its invalid actions, and `metadata` its resets,
baseline, action cap and solution length. `summary.json`'s `max_cost` is 8 and `config` an `Arc3Config`; T and the
share learned read the budget used as on track 1.

The baselines' adapters render a game with `systems/arc3_protocol.py`: the rules once per conversation (the finish
protocol's facts only, with the storage note and a rules page after them), then one message per turn. A board is
4,160 characters, and shown after every action it would make a level of a few hundred actions cost a model millions
of tokens, so it is shown whole only at the start of a conversation, after a reset, and when listing its changes
would take more text than the board; otherwise only the cells changed since the board last shown, a line a row of
runs of one color (`y12: 20-24=0 30=c`). A reply may carry a sequence of up to 20 actions, played without another
message and stopped when the level completes, the game is over or an action is refused. A demonstration is shown as
its first board, each action with the cells it changed, and its last board: about 10,000 characters for a short
level, some 3,000 to 4,000 tokens. The demo-cache floor keeps the demonstrations it bought for a game and shows them
again, free, when the game returns.

## Configurations

| configuration | stream | seen test | held-out test |
|---|---|---|---|
| `arc3-smoke` | 2 stream games, 2 instances each | 4 fresh instances of each | 1 held-out game, 2 instances |
| `arc3-small` | every stream game, 8 fresh instances each, in the workstreams order | 4 fresh instances of each stream game | 4 fresh instances of each held-out game |

`arc3-small`'s stream is scaled from `arc1-small`'s to its length (three games active at once, bursts of three,
dormancy of median 16 steps, a new game every two). The held-out test is the epilogue's instances
(`BenchmarkConfig.held_out_per_task`, four), and the fresh order wipes the learner before every one of them.

## Open

* The drift mode: rules that change over the stream, with a game's later levels as its first source.
* A human baseline on fresh levels: the published counts are for level 0, scaled here by the solver's lengths.
* The program library and the trained rows do not play this track; a hosted submission's wire format carries grids
  alone.
