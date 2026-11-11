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
# Each checkout, the folder it becomes on HOST, and the commit it ships. The baselines go at their branch turns: the
# package these loops import, as the tag internal-final keeps it (the last tree of their repository that held it; its
# main is the research fleet since, built on the public packages, which these loops predate), with the commits the
# turn-based budget (loop.HARNESS_BUDGETS["turns"]) relies on, on top: a turn that reaches prime-agent's liveness guard
# (timeout_s) ends its instance as an infrastructure failure (system_infra_failures), not a restart into a forced
# give-up (25 Oct 2026, the tag internal-turns), and a wait for a busy agent before a prompt keeps a turn's idle guard,
# so that a Python cell that never returns between turns costs ten minutes and a restart, not the six-hour guard
# (26 Oct 2026), and a cell's streamed output holds off no idle guard, so that one printing in a loop that never ends
# costs the same (27 Oct 2026).
REPOS=("continual-arc:continual-arc:origin/main" "continual-arc-baselines:continual-arc-baselines:origin/turns"
       "ask-sd:ask-sd:origin/main" "continual-arc-training:loop:origin/main")
for spec in "${REPOS[@]}"; do
    IFS=: read -r repo dir ref <<< "$spec"
    git -C ~/"$repo" fetch -q --tags origin
    git -C ~/"$repo" archive "$ref" | ssh -o BatchMode=yes "$H" "rm -rf ~/$dir.new && mkdir ~/$dir.new && tar -C ~/$dir.new -xf -"
done
swap=""
for spec in "${REPOS[@]}"; do
    IFS=: read -r _ dir _ <<< "$spec"
    # a folder's .env is the node's own, not git's (the baselines' one, from secret.sh install_node_env, holds the keys
    # the relay reads when it starts): it goes into the new tree, or the next relay to start on the node fails
    swap+="rm -rf ~/$dir.prev && { [ -f ~/$dir/.env ] && cp -p ~/$dir/.env ~/$dir.new/ || true; } && "
    swap+="{ [ -d ~/$dir ] && mv ~/$dir ~/$dir.prev || true; } && mv ~/$dir.new ~/$dir && "
done
ssh -o BatchMode=yes "$H" "${swap}true"
ssh -o BatchMode=yes "$H" 'export PATH=$HOME/.local/bin:$PATH
grep -c PG_DUAL_CLIP ~/ask-sd/ask_sd/trainer.py | sed "s/^/ask-sd 836e7ed+ ok: /"
grep -c dual_clip_fraction ~/loop/train.py | sed "s/^/loop c3ad0a4+ ok: /"
grep -c "class SystemStalled" ~/continual-arc-baselines/src/continual_arc_baselines/systems/base.py | sed "s/^/baselines turns (liveness guard) ok: /"
grep -c "while busy" ~/continual-arc-baselines/src/continual_arc_baselines/systems/prime_agent.py | sed "s/^/baselines turns (busy idle guard) ok: /"
cd ~/prime-rl && uv run --no-sync python -c "
import inspect
from continual_arc import runner
from continual_arc_baselines.collective import CollectiveHarnessLearner
assert \"view\" not in inspect.signature(runner.run).parameters, \"the benchmark still has the slot view\"
assert \"schedule\" in inspect.signature(CollectiveHarnessLearner).parameters, \"the baselines predate the learner view\"
print(\"learner view ok:\", runner.__file__, inspect.getfile(CollectiveHarnessLearner))"'
