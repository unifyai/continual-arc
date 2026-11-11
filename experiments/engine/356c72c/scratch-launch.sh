#!/usr/bin/env bash
# scratch-launch.sh RUN NODE DEST [JUMP] [K]: start scratch.sh's from-scratch retrain of real-time run RUN on its own node
# (scratch.py's docstring), from a checkout of this repo on a machine with the automation gcloud configuration and the
# node's ssh key (the operator's host, whose key every paper node takes, or a laptop), as copy-off.py runs. NODE is the node's name in the bucket's backup
# (results/runs/gpu-nodes/NODE/RUN, e.g. NODE-paper-dagger-spot), DEST its ssh destination (NODE_USER@<ip>) and
# JUMP the gateway it is reached through, if any (NODE_USER@<ip>); K the phase, default 199, the stream's last.
# It refuses while a real-time run still plays or trains on the node, or before the run's frozen tests are done
# (frozen.sh, which takes the node's GPUs). It copies scratch.py, scratch.sh, search/grid-vals.sh, frozen.py, frozen.sh
# and schedules/ to the node's ~/scratch-tools, and the continual student's phase-K adapter, which the node pruned
# long ago, from the bucket to ~/runs/rt/RUN/scratch/continual; then starts scratch.sh detached. Running it again on a
# node where scratch.sh already ran resumes it at its first unfinished stage.
set -euo pipefail
RUN=${1:?RUN} NODE=${2:?NODE} DEST=${3:?DEST} JUMP=${4:-} K=${5:-199}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PK=p$(printf %04d "$K")
SRC=gs://ARTIFACTS_BUCKET/results/runs/gpu-nodes/$NODE/$RUN/train/$PK
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${JUMP:+-J "$JUMP"})
ssh_() { "${SSH[@]}" "$DEST" "$@"; }
gcs() { gcloud --configuration automation storage "$@"; }

if ssh_ 'pgrep -f "[r]ealtime\.(py|sh)" > /dev/null'; then
    echo "a real-time run still plays or trains on $DEST: not started" >&2; exit 1
fi
if ssh_ "pgrep -f '[f]rozen\.sh ' > /dev/null || [ ! -f ~/runs/rt/$RUN/frozen/done ]"; then
    echo "$RUN's frozen tests are not done on $DEST (frozen-launch.sh): not started" >&2; exit 1
fi
gcs ls "$SRC/DONE" > /dev/null && gcs ls "$SRC/adapter/" > /dev/null \
    || { echo "the bucket holds no whole $PK of $RUN at $SRC: wait for the next backup" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
gcs cp -r "$SRC/DONE" "$SRC/adapter" "$tmp/"
ssh_ "mkdir -p ~/scratch-tools ~/runs/rt/$RUN/scratch/continual"
rsync -a -e "${SSH[*]}" "$HERE/scratch.py" "$HERE/scratch.sh" "$HERE/search/grid-vals.sh" "$HERE/frozen.py" \
    "$HERE/frozen.sh" "$HERE/schedules" "$DEST:scratch-tools/"
rsync -a -e "${SSH[*]}" "$tmp/" "$DEST:runs/rt/$RUN/scratch/continual/"
ssh_ "cd ~ && (setsid -f nohup bash ~/scratch-tools/scratch.sh $RUN $K > /dev/null 2>&1 < /dev/null)"
sleep 10
ssh_ "tail -3 ~/runs/rt/$RUN/scratch/progress"
