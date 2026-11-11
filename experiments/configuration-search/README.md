# The search that chose DAgger's configuration

The drivers of the search behind Appendix H's "How the configuration was chosen" (and the switch test, the resets
and the capacity runs played on the same machines) that `training/` does not hold. They are copied from the private
training repository's `search/`, `longtrain.py`, `frombase.sh`, `modal_longtrain.py` and `modal-longtrain.sh` as they stood on 11 November 2026, with only the
following taken out:

| Placeholder | Stands for |
| --- | --- |
| `ARTIFACTS_BUCKET`, `DATA_BUCKET` | the private Cloud Storage buckets that held the runs' products and backups |
| `GCP_PROJECT` | the Google Cloud project |
| `NODE-<name>` | one of the rented machines; `<name>` is the suffix of its folder under `runs/arc1/` (`runs/machines.json`) |
| `NODE_USER` | the machines' login user, whose home the loops' scripts spell out |
| `MODAL_APP`, `MODAL_CAPACITY`, `MODAL_JOBS_VOLUME`, `MODAL_HF_VOLUME` | the Modal app and volumes |
| `OPENROUTER_KEY_NAME` | the name of the OpenRouter key the generators shared |
| `R` | the researcher whose offline DAgger recipe, round-1 model and 100 validation puzzles the search compares against |
| `the author` | the paper's author, where a comment quotes a decision |

`127.0.0.1` is the oracle relay on the machine's own loopback. Credentials are read at run time by
`scripts/secret.sh` of the private research repository and none is in these files.

## What they drive

Every file here drives the training engine deployed on each machine as `~/loop`: the private repository's `loop.py`,
`paper.py`, `realtime.py`, `realtime.sh`, `train.py`, `frozen.py`, `frozen.sh`, `scratch.py`, `keep.sh` and their
neighbours at its `main` of the day each run started (`search/deploy-code.sh` put it there, or before 19 Oct 2026 an
rsync of a checkout). That engine was consolidated into `training/` (`continual_arc_training`) for the paper's
release; it is in [`../engine/`](../engine/README.md) at each commit a run started on or a sync put under it, with
each play's in [`../engine/plays.tsv`](../engine/plays.tsv). These drivers document what ran and with what settings;
they do not run against `training/` unchanged.

## Machine by machine

Run names are the folders under each machine's `runs/arc1/` folder; "public" means `training/scripts/`.

| Machine | Runs | Code | Command |
| --- | --- | --- | --- |
| `-paper-dagger`, `-paper-dagger-spot`, `-paper-dagger2`, `-paper-dagger2-spot`, `-paper-grpo`, `-paper-grpo-spot` | the round-by-round rows of 17-18 Oct 2026: `dagger` (100 users), `dagger-u1`, `-u2`, `-u4`, `-u8`, `-u10`, `pg-u1`, `-u2`, `-u4` | the private engine's `paper.sh` (public successor: `paper.sh`, which keeps rows h0 and bare alone) | `ROW=dagger\|pg LR=1e-4 USERS=N ROUNDS=R bash ~/loop/paper.sh` |
| `-paper-dagger`, `-paper-dagger-spot`, `-paper-dagger2`, `-paper-dagger2-spot`, `-paper-dagger3`, `-dagger-rules`, `-paper-grpo`, `-paper-grpo-spot`, `-paper-pg2`, `-paper-opsd`, `-paper-opsd-spot`, `-paper-opsd2` | the real-time rows of 19-20 Oct 2026: the warm-up rows `*-s24`, `*-s24f`, `pg-u4-lag1-sf`, the labeller rows `*-pf`, `*-pf-nudge`, `*-agg`, and the nine final rows `{dagger,opsd,pg}-u{1,2,4}-lag1` | the private engine's `realtime.sh` (public successor: `realtime.sh`, which has neither `DOSE=replay` nor `PG_REWARD`) | `ROW=dagger\|opsd\|pg USERS=N LR=1e-4 LAG=1 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 bash ~/loop/realtime.sh`; the warm-up rows with `START_AFTER=24 DOSE=replay` (OPSD's at LR 5e-5, batch 16); `pg-u4-lag1-sf` with `PG_REWARD=solve-first`; the nine final rows with `START_AFTER=24` and a held-out stop of patience 2 |
| the same machines | `val-ctrl-fwd2-final`, `val-base-10pct`, `-25pct`, `-50pct`, `val-b6-10pct`, `val-live-base-5500`, `val-s24p5`, `val-s24p10`, `val-u1p23`, `val-b10-T1`, `val-b10-block` | `grid-launch.sh`, `grid-exp.sh`, `grid-vals.sh` | `bash grid-launch.sh NODE_USER@<ip> NAME INIT TARGET` (INIT `base`, `b6`, `adapter:PATH` or `none`), which starts `grid-exp.sh` there; `bash grid-vals.sh NAME:INIT ...` for several checkpoints at once |
| the same machines | each final row's `frozen/`, `scratch/` and `weights/` plays | the private `frozen.sh` and `scratch.sh` (public successors: `frozen.sh`, `scratch.sh`) | `bash frozen.sh RUN GPUS` once the stream ended, `MEMORY=none` for the weights alone; `bash scratch.sh` for the last phase trained once more from the base model |
| `-paper-grpo-spot` | `qwen-f`, `qwen-h0` | the private `frozen.sh` with `ROW` (public successor: `frozen.sh` with `ROW`) | `ROW=bare\|h0 bash frozen.sh NAME GPUS`, a reference from nothing on both tests |
| `-layout-test` | `pooltest-u4-lag1` (a copy of `dagger-u4-lag1` from its phase 12, as DAgger), `pooltest-u1-lag1` (a copy of `dagger-u1-lag1` from its phase 25, continued as OPSD) | the private `realtime.sh` with `gpupool.py` (public: `realtime.sh`, `continual_arc_training.gpupool`) | two `realtime.sh` runs on one machine with `SERVE_GPUS=0,1` and the same `TRAIN_GPUS=2,3,4,5,6,7`, which their trainers share as one pool |
| `-transfer-refs` | `bare-u4`, `h0-u4` (`epilogue-u4-w100`) | the private `paper.sh`; public `paper.sh` plays the same | `ROW=bare\|h0 USERS=4 EPILOGUE=1 WIDTH=100 ROUNDS=4 bash training/scripts/paper.sh` |
| `-switch-a`, `-switch-b` | `switch-u4`, `switch-u1`, `switch-u2` (`before-*`, `after-*`, each a weights-alone play) | `switch/switch-run.sh`, `switch/switch-recent.py` | `bash switch/switch-run.sh USERS VALUES...` on the final DAgger run's machine (`DOSE=recent PATIENCE=4` from its next phase); the phases before and after it played with their weights alone by `frozen.sh` (`MEMORY=none`) |
| `-cross-fork2`, `-turns-b`, `-resets2` | `gen-card-val`, `gen-gate-val-1` to `-6`, `gen-turns-val-b`, `gen-u4final-frozen`, `gen-u4final-vals`, `gen-cap-t25-p0005-frozen`, `gen-cap-t25-p0005-val`, `gen-cap-t100-p0006-frozen`, `gen-cap-t400-p0007-frozen`, and `cross-u4` | `generalisation/` (`cell.py`, `launch.sh`, `setup.sh`, `bucket.py`, `replay.py`, `glm_val.py`, `tools.sh`, `modal_train.py`, `modal-train.sh`) with the cell of each name in `generalisation/cells/`; `cross-u4` was played by hand with the same tools on 25 Oct 2026 (`generalisation/README.md`) | `bash generalisation/launch.sh NODE_USER@<ip> CELL [STAGES]` |
| `-cross-fork2` | `dagger-u4-final` (the gating run of 26 Oct 2026), `pg-u1-final`, `pg-u2-disjoint` | the private `realtime.sh` in the final configuration, every round read by `final_checks.py` (public successor: `realtime.sh`) | `ROW=dagger USERS=4 LR=1e-4 START_AFTER=24 PATIENCE=6 bash ~/loop/realtime.sh`; `ROW=pg USERS=1\|2` likewise |
| `-cross-fork2` | `h-u1`, `h-u2` (row h, the untrained student keeping its notes, on the trained rows' own server) | the private `paper.sh` | `ROW=h USERS=1\|2 ROUNDS=200 bash ~/loop/paper.sh` |
| `-plays-a` | `dagger-u1-final`, `dagger-u2-disjoint`, `dagger-u4-disjoint` and their later plays `held-out-2`, `-3`, `seen-2`, `-3` | the private `realtime.sh` and `frozen.sh` in the final configuration (public successors: `realtime.sh` and `frozen.sh`; at commit `9616cbd` their teacher reads the playing rules alone, as these runs' did); [`../trained-rows-resets/more-plays.sh`](../trained-rows-resets/more-plays.sh) for the later plays | `ROW=dagger USERS=1\|2\|4 LR=1e-4 START_AFTER=24 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 bash training/scripts/realtime.sh`, then `bash training/scripts/frozen.sh dagger-uN-lag1 0,1,2,3,4,5,6,7`; `PURPOSE="..." bash ../trained-rows-resets/more-plays.sh JOBS` |
| `-plays-a` | `dagger-u1-scratch`, `dagger-u2-scratch`, `dagger-u4-scratch` (laid as the `-from-base` rows) | `frombase.sh`, `longtrain.py --from-base` | `PURPOSE="..." bash frombase.sh RUN OUT GPUS EVAL_STEPS PATIENCE` on each final run, then its tests by `../trained-rows-resets/more-plays.sh` |
| `-plays-a` | `dagger-u4-longtrain` | `longtrain.py` on Modal through `modal-longtrain.sh` and `modal_longtrain.py` | `TTL_HOURS=12 PURPOSE="..." bash modal-longtrain.sh dagger-u4-longtrain -- --run ~/runs/rt/dagger-u4-disjoint --out ~/runs/rt/dagger-u4-longtrain --gpus 8 --max-steps 1600 --patience 20`, then its tests by `../trained-rows-resets/more-plays.sh` |
| `-turns-b`, `-resets2` | `qwen-h0-card`, `qwen-h0-card-u2`, `qwen-h0c1-card`, `qwen-h0c1-card-u2`, `-u4` | `final-resets.sh` | `(setsid -f nohup bash ~/final-resets.sh > /dev/null 2>&1 < /dev/null)` on an 8-GPU machine |
| `-turns-b`, `-resets2` | the capacity runs whose adapters the `gen-cap-*` cells play | `capacity/` (`generate.py`, `capacity.py`, `trainer.py`, `launch.sh`, `start.sh`, `modal_capacity.py`, `modal-capacity.sh`) | `bash capacity/launch.sh NODE_USER@<ip> t25 t100 t400` (`capacity/README.md`) |
