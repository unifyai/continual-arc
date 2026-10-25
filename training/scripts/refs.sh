#!/usr/bin/env bash
# refs.sh RUN GPUS: the trained runs' references from nothing on the stack the trained runs play on, rows h0 (the
# learner reset) and bare (the bare model), both frozen tests each, on the node of real-time run RUN once RUN's own frozen
# tests are done (its frozen/done): frozen.sh with ROW, one row after the other on GPUS, into RUNS/paper/qwen-h0/frozen
# and RUNS/paper/qwen-f/frozen (TEST-reference/play). Resumable: a row whose tests are done is not played again.
# Usage (on the node): cd ~ && (setsid -f nohup bash ~/continual-arc/training/scripts/refs.sh pg-u2-lag1 0,1,2,3,4,5,6,7 \
#     > /dev/null 2>&1 < /dev/null)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=$1 GPUS=$2
{   # read whole before it runs: bash reads a script as it goes
    until [ -f "$RUNS/rt/$RUN/frozen/done" ]; do sleep "${POLL:-60}"; done
    for spec in h0:qwen-h0 bare:qwen-f; do
        ROW=${spec%%:*} RUNS=$RUNS bash "$HERE/frozen.sh" "${spec#*:}" "$GPUS" || exit 1
    done
    exit 0
}
