#!/usr/bin/env bash
# deploy-code.sh HOST: the pushed main of the benchmark, the baselines, ask-sd and continual-arc-training on HOST as
# ~/continual-arc, ~/continual-arc-baselines, ~/ask-sd and ~/loop (git archives, each unpacked into a new folder, and the
# four swapped in together once all are unpacked, so nothing reads a half-written tree or a mix of old and new). The
# four go as one: since the restricted learner view (19 Oct 2026) the runner shows a learner only the instance and
# CollectiveHarnessLearner takes the schedule, so loop.py and paper.py fail against an older benchmark or baselines, and
# an older loop.py fails against the newer ones. The node's environment (~/prime-rl, baselines
# scripts/remote-bootstrap.sh) has the benchmark, the baselines and ask-sd installed editable from these folders, so the
# swap is the install; the last step imports them there and checks the pair. A swap changes the code under every process
# the node starts afterwards, a running play's next round included.
set -euo pipefail
H=$1
REPOS=("continual-arc:continual-arc" "continual-arc-baselines:continual-arc-baselines" "ask-sd:ask-sd"
       "continual-arc-training:loop")
for spec in "${REPOS[@]}"; do
    repo=${spec%%:*}; dir=${spec##*:}
    git -C ~/"$repo" fetch -q origin
    git -C ~/"$repo" archive origin/main | ssh -o BatchMode=yes "$H" "rm -rf ~/$dir.new && mkdir ~/$dir.new && tar -C ~/$dir.new -xf -"
done
swap=""
for spec in "${REPOS[@]}"; do
    dir=${spec##*:}
    swap+="rm -rf ~/$dir.prev && { [ -d ~/$dir ] && mv ~/$dir ~/$dir.prev || true; } && mv ~/$dir.new ~/$dir && "
done
ssh -o BatchMode=yes "$H" "${swap}true"
ssh -o BatchMode=yes "$H" 'export PATH=$HOME/.local/bin:$PATH
grep -c PG_DUAL_CLIP ~/ask-sd/ask_sd/trainer.py | sed "s/^/ask-sd 836e7ed+ ok: /"
grep -c dual_clip_fraction ~/loop/train.py | sed "s/^/loop c3ad0a4+ ok: /"
cd ~/prime-rl && uv run --no-sync python -c "
import inspect
from continual_arc import runner
from continual_arc_baselines.collective import CollectiveHarnessLearner
assert \"view\" not in inspect.signature(runner.run).parameters, \"the benchmark still has the slot view\"
assert \"schedule\" in inspect.signature(CollectiveHarnessLearner).parameters, \"the baselines predate the learner view\"
print(\"learner view ok:\", runner.__file__, inspect.getfile(CollectiveHarnessLearner))"'
