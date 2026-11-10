#!/usr/bin/env bash
# Launch every cell the paper reports, as launch-cell.sh lines read from paper-grid.txt (a floors line launches the
# two floor cells). Each cell runs detached, committing its state every round (LOG_STATE=1); the paper's cells ran on two
# 64-vCPU Linux machines, the workstreams order and the fresh references on one and the sequential and random orders on
# the other, so launch a part at a time on a machine of this size.
#
#   bash scripts/paper-grid.sh [PATTERN]     only the lines matching PATTERN (grep -E), e.g. 'luna6-low' or 'acp .* hermes'
#   DRY_RUN=1 bash scripts/paper-grid.sh     print every cell's run directory and command, launch nothing
#   PILOT=25 bash scripts/paper-grid.sh      stop every cell after 25 instances, as the paper's cells were launched;
#   RESUME=1 bash scripts/paper-grid.sh      then continue them
#
# Each line of paper-grid.txt is the empty-reply rule its cells ran under (launch-cell.sh, EMPTY_REPLY), then the
# epilogue they were launched with (EPILOGUE), then launch-cell.sh's arguments.
set -euo pipefail
here=$(dirname "$0")
grep -v '^#' "$here/paper-grid.txt" | grep -E -- "${1:-.}" | while read -r rule epilogue args; do
    # shellcheck disable=SC2086  # the arguments are whitespace-separated words
    EMPTY_REPLY=$rule EPILOGUE=$epilogue LOG_STATE=1 bash "$here/launch-cell.sh" $args
done
