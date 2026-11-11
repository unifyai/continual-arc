# The generalisation search

Where the paper's online DAgger loses what R's offline DAgger generalised. His round-1 model (b6, then one training
on GLM-5.3-flash's every move of 296 fresh games, 17,449 labels) solved 59 of his 100 validation puzzles, ARC-AGI-1
evaluation tasks the stream never serves; the DAgger rows' weights alone solve 24 to 35 of the paper's held-out test,
though dagger-u4-lag1 had as many labels by its phase 68. the author, 25 Oct 2026: binary-search between the two ("keep binary
searching down to work out where and why that's happening"). Each cell changes one thing between his recipe and the
rows'. The results are on the Continual-ARC page's Generalisation tab (the project's internal dashboard).

## The cells

| cell | kind | what it asks | stages |
|---|---|---|---|
| `r-frozen` | frozen | his round-1 model, his recipe on all and on a tenth of his labels, and dagger-u4-lag1's phase 17 on the paper's two frozen tests: does his 59 carry to the paper's yardstick? | play |
| `dagger-val` | val | eight of the DAgger rows' checkpoints on his 100 puzzles: are the rows weaker on his yardstick too? | play |
| `u4lab-17k` | recipe | his recipe on dagger-u4-lag1's own labels: the labels, or what trains on them? | train, play |
| `relab-u4` | relabel | the run's moments labelled as he labelled his (c1 alone, no reasoning required), then his recipe: the labeller? | label, train, play |
| `scratch-u4` | scratch | the run's own trainer once from the base model on all its labels: the stream's warm-started phases? | train, play |
| `replay-u4` | replay | the run's training from its switch to the recent dose replayed under the original rule: the switch? | train, play |
| `smoke` | recipe | the runner's check on a vendor it has not run on: training ends in minutes | train |

The first six first ran by hand on 25 Oct 2026 on `NODE-cross-test`, `-val`, `-train`, `-relab`, `-scratch` and
`-fork2`, with these tools, and the dashboard reads them there (`results/runs/gpu-nodes/NODE-cross-*/` in the
bucket). A cell launched from here writes everything under `~/runs/rt/gen-NAME` on its machine.

## Running a cell on any machine

1. Rent a machine with 8 NVIDIA GPUs from any vendor with continual-arc-research's create scripts (its
   `docs/RUNNING.md`, "Renting a machine"), bare: `create-node.sh <name> h100` (which leases and routes it),
   `runpod-create.sh`, `vast-create.sh`, `prime-create.sh` or Lambda's `node-up.sh`. No `-up` script is needed, since
   the machine is set up from the bucket alone.
2. Launch the cell:

   ```bash
   bash search/generalisation/launch.sh NODE_USER@<ip> u4lab-17k
   ```

   `PORT=<port>` for a pod's ssh port, `JUMP=NODE_USER@<gateway>` for a machine with no public address, and
   `LEASE="<vendor> <id>"` for a machine that `create-node.sh` did not make: it puts the operator's key on it and
   routes its lease, so that the operator's hourly backup copies `~/runs/rt` off it. A third argument runs some of the
   cell's stages (`train`, `play`, `label,train`), an earlier stage's products then coming from the bucket.
3. Follow `~/gen/setup.log` (setting a Lambda 8x A100 up took 3 minutes) and `~/runs/rt/gen-NAME/progress` on it. A
   stage that stops is resumed by launching the cell again, which skips what is done.
4. Each stage's products, the files a later stage reads, reach the bucket as `generalisation/NAME/STAGE.tgz` as it
   ends, through an upload session `launch.sh` left on the machine; the plays and everything else reach it through the
   hourly backup. End the machine through its lease once both are in the bucket.

The smoke cell checked it on 25 Oct 2026: a bare Lambda 8x A100, whose driver speaks only CUDA 12.8, set itself up in 3
minutes, trained (35 steps), handed its products to the bucket, then served the model and played its games under the
harness's bubblewrap; Modal's 8x H100 built the image, trained and handed its products over in the same way.

What runs where:

| stage | needs | runs on |
|---|---|---|
| label | the network; the OpenRouter key, which `launch.sh` writes to `~/.openrouter.env` | any machine |
| train | 8 GPUs | any 8-GPU machine of any vendor, and Modal |
| play | 8 GPUs, vLLM, the harness's bubblewrap sandbox and the user `NODE_USER` | VMs: a GPU cloud, Lambda, and Vast's or Prime Intellect's VM offers |

A play needs the user `NODE_USER` because the loops' scripts spell out `/home/NODE_USER` and the paths of the harness's
workspaces reach the models' prompts, so a play elsewhere would not be the same play. Containers, as RunPod's pods and
most of Vast's and Prime Intellect's offers are, run as root and mostly without the unprivileged user namespaces
bubblewrap needs; `setup.sh` checks the sandbox and writes the result to `~/gen/sandbox`, and `cell.py` refuses a play
where it does not work.

## Modal

```bash
TTL_HOURS=14 PURPOSE="<why, as detailed as a lease's>" bash search/generalisation/modal-train.sh scratch-u4
bash search/generalisation/launch.sh NODE_USER@<ip> scratch-u4 play
```

`modal-train.sh` runs a cell's train stage on Modal on `H100:8` (or `H200:8`), detached, in an image built from the
same bundle by the same `setup.sh`; its products reach the bucket as `generalisation/NAME/train.tgz`, and the play
stage then runs on a VM. Modal cannot play: its containers run under gVisor, where bubblewrap starts but brings no
loopback up in a new network namespace ("loopback: Failed RTM_NEWADDR"), and the harness's sandbox makes one for every
model-driven turn (checked on 25 Oct 2026; Modal's H100s run driver CUDA 13.3, so the CUDA 13 wheels need no compat
package). It is no cheaper: H100s at 3.95 USD a GPU-hour, about 35 USD an hour for eight with the CPU and memory, as
much as a GPU cloud on demand. It is for when no vendor has 8 GPUs to rent or a quota is full, one cell at a time, since the
workspace's plan runs 10 GPUs at once.

## A new cell

Copy the cell file nearest to the split, change what it asks, name the file after the cell, and check it:
`python3 search/generalisation/cell.py check search/generalisation/cells/NAME.json`. The kinds and their settings are in
`cell.py`'s docstring; a new kind is a class there. Its rows on the dashboard go in brain's
`brain/experiments/pages/carc_paper.py` (`SEARCH`), at `results/runs/gpu-nodes/<machine>/gen-NAME/` in the bucket.

## The code a cell runs

A cell trains and plays on a code bundle in the bucket, `bundles/generalisation/NAME.tgz`, named by the cell's
`bundle` field (`"bundle": "code-dagger-rows-turns"`); without one it is `code-dagger-rows`, so every cell written
before the field runs as it did:

| bundle | what it holds |
|---|---|
| `code-dagger-rows` (the default) | `~/loop`, `~/continual-arc`, `~/continual-arc-baselines` (the internal package) and `~/ask-sd` as dagger-u4-lag1's node held them, so that a cell differs from the rows only in what it is meant to change. Its plays cut an action at its seconds: the validations at prime-agent's 900 s, the frozen tests at the block's 1,740 s. |
| `code-dagger-rows-turns` | the same four trees with only the budget changed (25 Oct 2026): `loop/loop.py`'s `HARNESS_BUDGETS["turns"]` (40 model calls an action and 6 a follow-up, no seconds, a liveness guard of 6 hours a turn) as `loop.py play`'s default, `loop/paper.py`'s and `loop/reference.py`'s `play_round` playing it (so the frozen tests, which play through `paper.play_round`, are turn-based too), and the baselines' `systems/base.py` and `systems/prime_agent.py` ending an instance whose turn reaches the guard as an infrastructure failure (`system_infra_failures` on its record) instead of restarting it into a forced give-up, and `systems/prime_agent.py`'s wait for a busy agent before a prompt keeping a turn's idle guard (26 Oct 2026, the baselines' commit a8b6c90 on their branch `turns`): a Python cell that never returns, in a run left going between turns, had held three of a validation's games for hours and failed two validations at the 3-hour guard, and now costs ten minutes and a restart; and the same file's idle guard taking no notice of a cell's streamed output (27 Oct 2026, the baselines' commit 22c95b1 on `turns`), since a cell printing in a loop that never ended held three games for hours, the gating run's round 40 among them, and now costs the same. Since 27 Oct 2026 (MD5 64facdac) it also takes a loss weighting for R's recipe (the author, 27 Oct 2026: a loss normalised per turn rather than per token, so the oracle's long reasoning does not outweigh its moves): `ask-sd`'s `Trainer.sft` and `Trainer.sft_chains` take `weighting` (`ask_sd.trainer.token_weights`: `token`, every token alike as before; `turn`, every reply alike, its tokens sharing the step's mean reply length; `split`, every reply alike with half on its reasoning, the tokens before its `</think>`, and half on its move), `loop/train.py --weighting` hands it over and logs it in each step's record, and `realtime.train_args` keeps `token`; nothing else changed, and the held-out loss stays per token. The bundle before the first fix is `code-dagger-rows-turns-cf210701.tgz`, before the second `code-dagger-rows-turns-e857dc9b.tgz`, before the weighting `code-dagger-rows-turns-55bb601b.tgz`. No score made on it depends on how fast the model is served. |

`data-cache.tgz` beside them is the benchmark's data. `launch.sh` fetches the cell's bundle to `~/gen/bundle/NAME.tgz`,
`setup.sh` unpacks it once, into a machine that holds none of the four trees, and never onto a paper node, keeping its
MD5 in `~/gen/bundle/code.unpacked`, and `cell.py run` refuses any stage of a cell whose bundle is not the one
unpacked there, since a machine set up once is never set up again: a cell of another bundle needs a machine of its own.
Each cell records its bundle and its MD5 in `~/runs/rt/gen-NAME/bundle`. `modal-train.sh` builds its image from the
cell's bundle the same way. The tools around them (this folder, `grid-exp.sh`, `grid-vals.sh`, `frozen.sh`,
`frozen.py`, `scratch.py`) come from the checkout `launch.sh` runs in, whose commit each cell records in
`~/runs/rt/gen-NAME/tools-commit`, whichever bundle the cell runs on: `grid-exp.sh` and `grid-vals.sh` wait for
every game of a validation (none is stopped at a time, since a play stopped short writes no `records.jsonl`), stopping
a play only as a failure when nothing in it has moved for 3 hours or it has had to be launched more than 4 times, and
log how many games ended as infrastructure failures.

On `code-dagger-rows` the plays cut an action at its time budget, so how fast a model is served moves its score, and a
cell is compared only with numbers made the same way on the same GPUs. dagger-u4-lag1's phase 17 solved 42 of R's 99 puzzles on two
H200s a model (20 Oct 2026), 13 on one H100 a model with eight at once (`NODE-cross-val`'s first run, 25 Oct 2026),
fitting a quarter of the model calls into each game's budget, and 26 of the first 77 to end on two H100s a model,
against 38 of the same 77 on the H200s; each call generated as much either way. So `grid-vals.sh` serves each model on
two GPUs, four at a time, and a cell's validations are read beside references played in the same run's layout
(`NODE-cross-test` played R's models so, beside `NODE-cross-val`'s checkpoints). `grid-exp.sh`'s
validation of one model serves it on all eight GPUs, as R's were. The frozen tests take two GPUs an adapter, one a
test, the layout of the switch test, which the DAgger rows' weights-alone scores come from. On
`code-dagger-rows-turns` the layout sets only how long a play takes, so its numbers compare across GPUs and vendors.

Every process runs with `NCCL_NVLS_ENABLE=0`: NVLink SHARP fails NCCL's init on the a GPU cloud 8x H100 nodes made from 25
Oct 2026. Only training steps get `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, since a tensor-parallel vLLM
server that inherits it fails at graph capture.
