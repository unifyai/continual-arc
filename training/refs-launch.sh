#!/usr/bin/env bash
# refs-launch.sh RUN DEST GPUS [JUMP]: start refs.sh's references from nothing on the node of real-time run RUN, from a
# checkout of this repository on a machine that also has a checkout of ask-sd (ASK_SD, required: the trainer's
# package, which paper.py imports) and the node's ssh key. DEST is the node's ssh destination and JUMP its gateway, if
# any, as frozen-launch.sh takes them; GPUS the GPUs the references take once RUN's frozen tests are done. It copies
# this repository's pushed main and ask-sd's to the node as git archives, unpacked into MAIN (default
# frozen-tools/main, under the node's home; never over the node's own code), with frozen.py, frozen.sh, vllm-serve.sh,
# refs.sh and schedules/ into TOOLS (default frozen-tools), and starts refs.sh detached, which waits for RUN's tests and
# then plays rows h0 and bare into RUNS/paper/qwen-h0 and qwen-f (RUNS default runs, under the node's home).
# Launched again, it updates the copies and starts nothing while refs.sh runs.
set -euo pipefail
RUN=${1:?RUN} DEST=${2:?DEST} GPUS=${3:?GPUS} JUMP=${4:-}
ASK_SD=${ASK_SD:?set ASK_SD: a checkout of ask-sd}
TOOLS=${TOOLS:-frozen-tools} MAIN=${MAIN:-frozen-tools/main} RUNS=${RUNS:-runs}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${JUMP:+-J "$JUMP"})
ssh_() { "${SSH[@]}" "$DEST" "$@"; }
ssh_ "test -d ~/$RUNS/rt/$RUN" || { echo "$DEST has no run $RUN" >&2; exit 1; }
unpack() {   # unpack CHECKOUT DIR: CHECKOUT's pushed main, unpacked into MAIN/DIR on the node (DIR . for MAIN itself)
    git -C "$1" fetch -q origin
    git -C "$1" archive origin/main | ssh_ "mkdir -p ~/$MAIN.new/$2 && tar -C ~/$MAIN.new/$2 -xf -"
}
ssh_ "rm -rf ~/$MAIN.new"
unpack "$REPO" .
unpack "$ASK_SD" ask-sd
ssh_ "rm -rf ~/$MAIN && mv ~/$MAIN.new ~/$MAIN"
rsync -a -e "${SSH[*]}" "$HERE/frozen.py" "$HERE/frozen.sh" "$HERE/vllm-serve.sh" "$HERE/refs.sh" "$HERE/schedules" \
    "$DEST:$TOOLS/"
if ssh_ "pgrep -f '[r]efs.sh $RUN ' > /dev/null"; then echo "refs.sh already runs on $DEST: its copies updated"; exit 0; fi
ssh_ "cd ~ && (MAIN=~/$MAIN RUNS=~/$RUNS setsid -f nohup bash ~/$TOOLS/refs.sh $RUN $GPUS > /dev/null 2>&1 < /dev/null)"
echo "refs.sh started on $DEST: it waits for $RUN's frozen tests, then plays rows h0 and bare"
