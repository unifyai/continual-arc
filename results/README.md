# Results

Runs submitted to the [Continual-ARC leaderboard](https://continual-arc.unify.ai/leaderboard), one directory each:
`<track>/<name>/` holds a run directory the benchmark's runner wrote (`records.jsonl`, `summary.json`,
`schedule.json`, `config.json`, `meta.json`): a learner's stream, with a `submission.json` that names the learning
method, the model that plays it (`student`) and any model that teaches it (`teacher`), the learner reset on the same
stream, and the directories of its two frozen tests, their resets and the bare model's runs. How to run, check and submit
one is in the [README](../README.md#submitting-a-result).

The paper's own runs are not repeated here: every run it reports is in [`runs/`](../runs), and the leaderboard opens
with them.

## Citation

If you report results on Continual-ARC, please cite the paper; its BibTeX is in the [README](../README.md#citation).
