#!/usr/bin/env bash
# frozen-launch.sh RUN DEST GPUS [JUMP]: start frozen.sh's two frozen tests of real-time run RUN on its node (frozen.py's
# docstring), from a checkout of this repo on a machine with the node's ssh key (a laptop, or the operator's host,
# whose key every paper node takes). DEST is the node's ssh destination (NODE_USER@<ip>, or a Host of the ssh
# configuration) and JUMP the gateway it is reached through, if any (NODE_USER@<ip>). GPUS are the GPUs the tests
# take once the stream has ended, an even number, half for each test: all eight on a node of the run's own, its
# TRAIN_GPUS on a node whose model server another run still uses. It copies frozen.py, frozen.sh and schedules/ to the
# node's ~/frozen-tools and starts frozen.sh there detached, which waits for the run's stream to end and then plays the
# tests; while a frozen.sh for the run already runs there it only updates the copy. Launched again after the node
# stopped, the tests are taken up where they stopped. Brain's paper-restart task launches it again itself, from the
# "started:" line frozen.sh leaves in ~/runs/rt/RUN/frozen/progress, once it has started a node a GPU cloud stopped.
set -euo pipefail
RUN=${1:?RUN} DEST=${2:?DEST} GPUS=${3:?GPUS} JUMP=${4:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${JUMP:+-J "$JUMP"})
ssh_() { "${SSH[@]}" "$DEST" "$@"; }
[[ $RUN =~ ^[a-z]+-u[0-9]+-lag[0-9]+$ ]] || { echo "RUN is a real-time run's name, ROW-uN-lagL, not $RUN" >&2; exit 1; }
[[ $GPUS =~ ^[0-7](,[0-7])+$ ]] || { echo "GPUS is a list of GPUs, 0,1,2,3 say, not $GPUS" >&2; exit 1; }
ssh_ "test -d ~/runs/rt/$RUN" || { echo "$DEST has no run $RUN" >&2; exit 1; }
rsync -a -e "${SSH[*]}" "$HERE/frozen.py" "$HERE/frozen.sh" "$HERE/schedules" "$DEST:frozen-tools/"
if ssh_ "pgrep -f '[f]rozen.sh $RUN ' > /dev/null"; then
    echo "frozen.sh already runs for $RUN on $DEST: its copy updated"; exit 0
fi
ssh_ "cd ~ && (setsid -f nohup bash ~/frozen-tools/frozen.sh $RUN $GPUS > /dev/null 2>&1 < /dev/null)"
sleep 5
ssh_ "tail -2 ~/runs/rt/$RUN/frozen/progress"
