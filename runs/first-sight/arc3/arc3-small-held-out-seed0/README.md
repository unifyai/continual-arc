# ARC-AGI-3 first-sight calibration, 28 Oct 2026

The bare model (`BareModelLearner`, 64,000-token replies, medium effort through OpenRouter) on `arc3-small`'s 40
held-out levels at seed 0, in the fresh order, the learner wiped before each, played by `experiments/first-sight/arc3_first_sight.py`
on Modal from continual-arc `v0.4.0`. These are the numbers of the paper's Table `tab:first-sight` (Appendix K, ARC-AGI-3 columns).

| model | folder | levels | solved | mean charge (of 8) | cost |
|---|---|---|---|---|---|
| Kimi K3.1 Pro | `moonshotai_kimi-k3.1-pro/` | 40 | 10 | 6.75 | $8.56 |
| Kimi-K3 | `moonshotai_kimi-k3/` (29, stopped by its $25 cap) and `kimi-k3-rest/` (the other 11, one container a level, each checked against the first run's schedule) | 40 | 14 | 6.03 | $44.60 |
| Qwen3.5-9B | `qwen3.5-9b-partial/` | 0 of 1 | | | about 30 of its first level's 110 moves in 2.45 h, a quarter of its replies reasoned to the cap: no row |

Each folder holds the run's records, summary and schedule and the container's `calibration.json`; the full run
folders, their cost journals and event logs included, are the tarball `arc3-calibration-2026-10-28.tgz` in the private bucket
(`ARTIFACTS_BUCKET/results/calibration/arc3-small-2026-10-28/`), not here.
