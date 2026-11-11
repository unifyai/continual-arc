#!/usr/bin/env bash
# One reference row of the RL graph (reference.py's docstring) kept going on the reference machine: reference.py plays
# every round not yet played and is started again when it stops (every round resumes from the run's records), up to
# MAX_RESTARTS times, a minute apart. The run is ~/runs/paper/$PLAYER-$ROW-u$USERS (play/ the benchmark run; progress
# and play.log beside it), where brain's paper-backup task finds it on a routed machine. A compaction probe's arm sets
# COMPACT_AT and COMPACT_KEEP (reference.py's --compact-at/--compact-keep; 0 is prime-agent's own) and a TAG that ends
# its run's name ($PLAYER-$ROW-u$USERS-$TAG).
# Usage (on the machine, with ~/loop this repo and ~/continual-arc-baselines' .env holding OPENROUTER_API_KEY):
#   PLAYER=glm ROW=h USERS=1 setsid nohup bash ~/loop/reference.sh > /dev/null 2>&1 < /dev/null &
#   stop: this script's process group (kill -- -<its pid>)
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
PLAYER=${PLAYER:?set PLAYER: glm or qwen}; ROW=${ROW:?set ROW: h, h0 or hpl}; USERS=${USERS:-1}; ROUNDS=${ROUNDS:-200}
MAX_RESTARTS=${MAX_RESTARTS:-20}
L=~/runs/paper/$PLAYER-$ROW-u$USERS${TAG:+-$TAG}
COMPACT=(); [ -n "${COMPACT_AT:-}" ] && COMPACT+=(--compact-at "$COMPACT_AT"); [ -n "${COMPACT_KEEP:-}" ] && COMPACT+=(--compact-keep "$COMPACT_KEEP")
mkdir -p "$L"
say() { echo "$(date -u +%m-%d\ %H:%M) reference.sh: $*" >> "$L/progress"; }
{   # read whole before it runs: bash reads a script as it goes, and the code may be synced again while this runs
    say "started: $PLAYER plays row $ROW, $USERS users, $ROUNDS rounds${COMPACT_AT:+, compaction at $COMPACT_AT keeping ${COMPACT_KEEP:-8000}}"
    for i in $(seq 1 "$MAX_RESTARTS"); do
        if (cd ~/continual-arc-baselines && uv run --no-sync --env-file .env python ~/loop/reference.py play \
                --player "$PLAYER" --row "$ROW" --run-dir "$L/play" --users "$USERS" --rounds "$ROUNDS" "${COMPACT[@]}") \
                >> "$L/play.log" 2>&1; then
            say "done"; touch "$L/play.done"; exit 0
        fi
        say "reference.py stopped ($L/play.log): restart $i of $MAX_RESTARTS"; sleep 60
    done
    say "reference.py stopped $MAX_RESTARTS times: giving up"; exit 1
}
