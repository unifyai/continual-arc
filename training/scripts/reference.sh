#!/usr/bin/env bash
# One reference row (continual_arc_training.references) kept going on a reference machine: every round not yet played is
# played, and the driver is started again when it stops (every round resumes from the run's records), up to
# MAX_RESTARTS times, a minute apart. The run is RUNS/paper/$PLAYER-$ROW-u$USERS (play/ the benchmark run, progress and
# play.log beside it). ROW=dc, f, m or m0 plays one of the player's bare-model rows; MAX_TOKENS sets a bare-model row's
# reply cap (32,768 unless set; the demo-cache floors play at 128,000).
# A reference machine needs no GPU: the model is served by OpenRouter. It needs prime-agent 0.9.6, Node 22, bubblewrap,
# and an environment with this package, the benchmark and the baselines installed: the repository's own uv environment,
# run from BASELINES (default the repository's baselines/), whose .env holds OPENROUTER_API_KEY.
# Usage (on the machine): PLAYER=glm ROW=h USERS=1 setsid nohup bash ~/continual-arc/training/scripts/reference.sh \
#     > /dev/null 2>&1 < /dev/null &
#   stop: this script's process group (kill -- -<its pid>)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
PLAYER=${PLAYER:?set PLAYER: glm or qwen}; ROW=${ROW:?set ROW: h, h0, hpl, hg, ha, hpl0, hplg, hgpl, dc, f, m or m0}
USERS=${USERS:-1}; ROUNDS=${ROUNDS:-200}
MAX_RESTARTS=${MAX_RESTARTS:-20}
L=$RUNS/paper/$PLAYER-$ROW-u$USERS
case $ROW in
    dc|f|m|m0) PLAY=(bare --player "$PLAYER" --row "$ROW" --run-dir "$L/play" --users "$USERS" --rounds "$ROUNDS" ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"}) ;;
    *) PLAY=(play --player "$PLAYER" --row "$ROW" --run-dir "$L/play" --users "$USERS" --rounds "$ROUNDS") ;;
esac
mkdir -p "$L"
say() { echo "$(date -u +%m-%d\ %H:%M) reference.sh: $*" >> "$L/progress"; }
{   # read whole before it runs: bash reads a script as it goes, and the code may be pulled again while this runs
    say "started: $PLAYER plays row $ROW, $USERS users, $ROUNDS rounds${MAX_TOKENS:+, reply cap $MAX_TOKENS}, baselines $BASELINES"
    for i in $(seq 1 "$MAX_RESTARTS"); do
        if (cd "$BASELINES" && uv run --no-sync --env-file .env python -m continual_arc_training.references "${PLAY[@]}") \
                >> "$L/play.log" 2>&1; then
            say "done"; touch "$L/play.done"; exit 0
        fi
        say "the driver stopped ($L/play.log): restart $i of $MAX_RESTARTS"; sleep 60
    done
    say "the driver stopped $MAX_RESTARTS times: giving up"; exit 1
}
