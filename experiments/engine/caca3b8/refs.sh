#!/usr/bin/env bash
# refs.sh RUN GPUS: the trained runs' references from nothing, rows h0 (the learner reset) and bare (the bare model),
# both frozen tests each, on the node of real-time run RUN once RUN's own frozen tests are done (its frozen/done):
# frozen.sh with ROW, one row after the other on GPUS, into ~/runs/paper/qwen-h0/frozen and ~/runs/paper/qwen-f/frozen
# (TEST-reference/play, where continual-arc-paper's registry reads the trained rows' references), on main's
# code, which refs-launch.sh unpacks beside the node's own into ~/frozen-tools/main and puts first on the path (paper.py's
# bare row is main's), so the node's code is never touched. The seen test's references are played on the stack the
# trained runs play on, as the held-out test's were on 20 Oct. The seen test is RUN's own (frozen.sh's SEEN_OF): a run
# whose users play tasks of their own (paper.py's disjoint users mode) has its own, whose references go into folders
# of their own, ~/runs/paper/qwen-h0-disjoint-u4 and qwen-f-disjoint-u4 at four users, both tests in each. Resumable:
# run again, a row whose tests are done is not played again.
# Usage (on the node): cd ~ && (setsid -f nohup bash ~/frozen-tools/refs.sh pg-u2-lag1 0,1,2,3,4,5,6,7 > /dev/null 2>&1 < /dev/null)
set -uo pipefail
RUN=$1 GPUS=$2
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) M=$HOME/frozen-tools/main
{   # read whole before it runs: bash reads a script as it goes
    until [ -f ~/runs/rt/"$RUN"/frozen/done ]; do sleep 60; done
    seen=$(python3 "$T/frozen.py" schedule --run ~/runs/rt/"$RUN" --test seen) || exit 1
    seen=$(basename "$seen" .json) && seen=${seen#arc1-small-seen-test-seed0}   # "" for arc1-small's own, -disjoint-uN
    for spec in h0:qwen-h0 bare:qwen-f; do
        ROW=${spec%%:*} SEEN_OF=~/runs/rt/$RUN CODE=$M/loop \
            PYPATH=$M/loop:$M/continual-arc/src:$M/continual-arc-baselines/src:$M/ask-sd \
            bash "$T/frozen.sh" "${spec#*:}$seen" "$GPUS" || exit 1
    done
    exit 0
}
