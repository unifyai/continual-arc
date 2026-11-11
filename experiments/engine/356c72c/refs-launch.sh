#!/usr/bin/env bash
# refs-launch.sh RUN DEST GPUS [JUMP]: start refs.sh's references from nothing on the node of real-time run RUN, from a
# checkout of this repo on a machine that also has continual-arc, continual-arc-baselines and ask-sd checked out at
# ~/<repo> and the node's ssh key (a laptop). DEST is the node's ssh destination and JUMP its gateway, if any, as
# frozen-launch.sh takes them; GPUS the GPUs the references take once RUN's frozen tests are done. It copies the four
# repos' pushed main to the node as git archives, unpacked into ~/frozen-tools/main (never over the node's own code),
# with frozen.py, frozen.sh, refs.sh and schedules/ into ~/frozen-tools, and starts refs.sh detached, which waits for
# RUN's tests and then plays rows h0 and bare into ~/runs/paper/qwen-h0 and qwen-f. Launched again, it updates the
# copies and starts nothing while refs.sh runs.
set -euo pipefail
RUN=${1:?RUN} DEST=${2:?DEST} GPUS=${3:?GPUS} JUMP=${4:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${JUMP:+-J "$JUMP"})
ssh_() { "${SSH[@]}" "$DEST" "$@"; }
ssh_ "test -d ~/runs/rt/$RUN" || { echo "$DEST has no run $RUN" >&2; exit 1; }
for spec in continual-arc:continual-arc continual-arc-baselines:continual-arc-baselines ask-sd:ask-sd \
            continual-arc-training:loop; do
    repo=${spec%%:*} dir=${spec##*:}
    git -C ~/"$repo" fetch -q origin
    git -C ~/"$repo" archive origin/main | ssh_ "mkdir -p ~/frozen-tools/main && rm -rf ~/frozen-tools/main/$dir.new \
        && mkdir ~/frozen-tools/main/$dir.new && tar -C ~/frozen-tools/main/$dir.new -xf - && rm -rf ~/frozen-tools/main/$dir \
        && mv ~/frozen-tools/main/$dir.new ~/frozen-tools/main/$dir"
done
rsync -a -e "${SSH[*]}" "$HERE/frozen.py" "$HERE/frozen.sh" "$HERE/refs.sh" "$HERE/schedules" "$DEST:frozen-tools/"
if ssh_ "pgrep -f '[r]efs.sh $RUN ' > /dev/null"; then echo "refs.sh already runs on $DEST: its copies updated"; exit 0; fi
ssh_ "cd ~ && (setsid -f nohup bash ~/frozen-tools/refs.sh $RUN $GPUS > /dev/null 2>&1 < /dev/null)"
echo "refs.sh started on $DEST: it waits for $RUN's frozen tests, then plays rows h0 and bare"
