#!/usr/bin/env bash
# frozen-launch.sh RUN DEST GPUS [JUMP]: start frozen.sh's two frozen tests of real-time run RUN on its node (frozen.py's
# docstring), from a checkout of this repository on a machine that holds the node's ssh key. DEST is the node's ssh
# destination (user@host, or a Host of the ssh configuration) and JUMP the gateway it is reached through, if any; there
# is no default for either. GPUS are the GPUs the tests take once the stream has ended, an even number, half for each
# test: all eight on a node of the run's own, its TRAIN_GPUS on a node whose model server another run still uses. It
# copies frozen.py, frozen.sh, vllm-serve.sh and schedules/ to the node's TOOLS folder (default frozen-tools, under the
# node's home) and starts frozen.sh there detached, with CODE the training folder the run played with on the node
# (NODE_CODE, default continual-arc/training under the node's home), so that the run's own code is never touched;
# frozen.sh waits for the run's stream to end and then plays the tests. While a frozen.sh for the run already runs
# there it only updates the copy. Launched again after the node stopped, the tests are taken up where they stopped.
# RUNS is the node's runs folder (default runs, under its home).
set -euo pipefail
RUN=${1:?RUN} DEST=${2:?DEST} GPUS=${3:?GPUS} JUMP=${4:-}
TOOLS=${TOOLS:-frozen-tools} NODE_CODE=${NODE_CODE:-continual-arc/training} RUNS=${RUNS:-runs}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${JUMP:+-J "$JUMP"})
ssh_() { "${SSH[@]}" "$DEST" "$@"; }
[[ $RUN =~ ^[a-z]+-u[0-9]+-lag[0-9]+$ ]] || { echo "RUN is a real-time run's name, ROW-uN-lagL, not $RUN" >&2; exit 1; }
[[ $GPUS =~ ^[0-7](,[0-7])+$ ]] || { echo "GPUS is a list of GPUs, 0,1,2,3 say, not $GPUS" >&2; exit 1; }
ssh_ "test -d ~/$RUNS/rt/$RUN" || { echo "$DEST has no run $RUN" >&2; exit 1; }
ssh_ "mkdir -p ~/$TOOLS"
rsync -a -e "${SSH[*]}" "$HERE/frozen.py" "$HERE/frozen.sh" "$HERE/vllm-serve.sh" "$HERE/schedules" "$DEST:$TOOLS/"
if ssh_ "pgrep -f '[f]rozen.sh $RUN ' > /dev/null"; then
    echo "frozen.sh already runs for $RUN on $DEST: its copy updated"; exit 0
fi
ssh_ "cd ~ && (CODE=~/$NODE_CODE RUNS=~/$RUNS setsid -f nohup bash ~/$TOOLS/frozen.sh $RUN $GPUS > /dev/null 2>&1 < /dev/null)"
sleep 5
ssh_ "tail -2 ~/$RUNS/rt/$RUN/frozen/progress"
