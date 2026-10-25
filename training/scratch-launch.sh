#!/usr/bin/env bash
# scratch-launch.sh RUN SRC DEST [JUMP] [K]: start scratch.sh's weights-alone tests and from-scratch retrain of
# real-time run RUN on its own node (scratch.sh's and scratch.py's docstrings), from a checkout of this repository on a
# machine with gcloud signed in to the bucket that holds the run's backup and with the node's ssh key. SRC is the run's
# backup in that bucket (a gs:// folder holding the run's train/pNNNN, as paper_backup.sh GCS=1 writes it), DEST the
# node's ssh destination (user@host, or a Host of the ssh configuration) and JUMP the gateway it is reached through, if
# any; there is no default for any of them. K is the phase, default 199, the stream's last.
# It refuses while a real-time run still plays or trains on the node, or before the run's frozen tests are done
# (frozen.sh, which takes the node's GPUs). It copies scratch.py, scratch.sh, grid-vals.sh, frozen.py, frozen.sh,
# vllm-serve.sh and schedules/ to the node's TOOLS folder (default scratch-tools, under the node's home), and the
# continual student's phase-K adapter, which the node pruned long ago, from the backup to
# RUNS/rt/RUN/scratch/continual (RUNS default runs, under the node's home); then starts scratch.sh detached, with CODE
# the training folder the run trained with on the node (NODE_CODE, default continual-arc/training). Running it again on
# a node where scratch.sh already ran resumes it at its first unfinished stage.
set -euo pipefail
RUN=${1:?RUN} SRC=${2:?SRC} DEST=${3:?DEST} JUMP=${4:-} K=${5:-199}
TOOLS=${TOOLS:-scratch-tools} NODE_CODE=${NODE_CODE:-continual-arc/training} RUNS=${RUNS:-runs}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PK=p$(printf %04d "$K")
SRC=${SRC%/}/train/$PK
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${JUMP:+-J "$JUMP"})
ssh_() { "${SSH[@]}" "$DEST" "$@"; }

if ssh_ 'pgrep -f "[r]ealtime\.(py|sh)" > /dev/null'; then
    echo "a real-time run still plays or trains on $DEST: not started" >&2; exit 1
fi
if ssh_ "pgrep -f '[f]rozen\.sh ' > /dev/null || [ ! -f ~/$RUNS/rt/$RUN/frozen/done ]"; then
    echo "$RUN's frozen tests are not done on $DEST (frozen-launch.sh): not started" >&2; exit 1
fi
gcloud storage ls "$SRC/DONE" > /dev/null && gcloud storage ls "$SRC/adapter/" > /dev/null \
    || { echo "the backup holds no whole $PK of $RUN at $SRC" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
gcloud storage cp -r "$SRC/DONE" "$SRC/adapter" "$tmp/"
ssh_ "mkdir -p ~/$TOOLS ~/$RUNS/rt/$RUN/scratch/continual"
rsync -a -e "${SSH[*]}" "$HERE/scratch.py" "$HERE/scratch.sh" "$HERE/grid-vals.sh" "$HERE/frozen.py" \
    "$HERE/frozen.sh" "$HERE/vllm-serve.sh" "$HERE/schedules" "$DEST:$TOOLS/"
rsync -a -e "${SSH[*]}" "$tmp/" "$DEST:$RUNS/rt/$RUN/scratch/continual/"
ssh_ "cd ~ && (CODE=~/$NODE_CODE RUNS=~/$RUNS setsid -f nohup bash ~/$TOOLS/scratch.sh $RUN $K > /dev/null 2>&1 < /dev/null)"
sleep 10
ssh_ "tail -3 ~/$RUNS/rt/$RUN/scratch/progress"
