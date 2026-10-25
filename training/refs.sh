#!/usr/bin/env bash
# refs.sh RUN GPUS: the trained runs' references from nothing, rows h0 (the learner reset) and bare (the bare model),
# both frozen tests each, on the node of real-time run RUN once RUN's own frozen tests are done (its frozen/done):
# frozen.sh with ROW, one row after the other on GPUS, into RUNS/paper/qwen-h0/frozen and RUNS/paper/qwen-f/frozen
# (TEST-reference/play, where the paper's figures read the trained rows' references), on the current code, which
# refs-launch.sh unpacks beside the node's own into MAIN (default ~/frozen-tools/main) and puts first on the path
# (paper.py's bare row is the current code's), so the node's code is never touched. The seen test's references are
# played on the stack the trained runs play on, as the held-out test's were. Resumable: run again, a row whose tests
# are done is not played again.
# Usage (on the node): cd ~ && (setsid -f nohup bash ~/frozen-tools/refs.sh pg-u2-lag1 0,1,2,3,4,5,6,7 > /dev/null 2>&1 < /dev/null)
set -uo pipefail
RUN=$1 GPUS=$2
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) M=${MAIN:-$HOME/frozen-tools/main} RUNS=${RUNS:-$HOME/runs}
{   # read whole before it runs: bash reads a script as it goes
    until [ -f "$RUNS/rt/$RUN/frozen/done" ]; do sleep 60; done
    for spec in h0:qwen-h0 bare:qwen-f; do
        ROW=${spec%%:*} CODE=$M/training PYPATH=$M/training:$M/src:$M/baselines/src:$M/ask-sd RUNS=$RUNS \
            bash "$T/frozen.sh" "${spec#*:}" "$GPUS" || exit 1
    done
    exit 0
}
