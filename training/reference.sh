#!/usr/bin/env bash
# One reference row (reference.py's docstring) kept going on a reference machine: reference.py plays every round not yet
# played and is started again when it stops (every round resumes from the run's records), up to MAX_RESTARTS times, a
# minute apart. The run is RUNS/paper/$PLAYER-$ROW-u$USERS (RUNS default ~/runs; play/ the benchmark run, progress and
# play.log beside it). A compaction probe's arm sets COMPACT_AT and COMPACT_KEEP (reference.py's
# --compact-at/--compact-keep; 0 is prime-agent's own) and a TAG that ends its run's name ($PLAYER-$ROW-u$USERS-$TAG).
# MAX_TOKENS sets a bare-model row's reply cap (reference.py bare --max-tokens; 32,768 unless set, the demo-cache
# floors' 128,000), named in its started line. ROW=dc, f, m or m0 plays one of the player's bare-model rows instead
# (reference.py bare): the demo-cache floor, one cache for the team, the stateless floor, the program library, or the
# program library wiped before every instance. reference.py is the one beside this script.
# A reference machine needs no GPU: the model is served by OpenRouter. It needs prime-agent 0.9.6, Node 22,
# bubblewrap, and the baselines' environment. BASELINES (../baselines by default) is the baselines checkout whose
# venv runs reference.py and whose .env holds OPENROUTER_API_KEY: a row on newer code than another row on the same
# machine takes a checkout and venv of its own, since this script restarts reference.py from that venv whenever it
# stops.
# Usage (on the machine): PLAYER=glm ROW=h USERS=1 setsid nohup bash ~/continual-arc/training/reference.sh \
#     > /dev/null 2>&1 < /dev/null &
#   stop: this script's process group (kill -- -<its pid>)
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
PLAYER=${PLAYER:?set PLAYER: glm or qwen}; ROW=${ROW:?set ROW: h, h0, hpl, hg, ha, hpl0, hplg, hgpl, dc, f, m or m0}; USERS=${USERS:-1}; ROUNDS=${ROUNDS:-200}
MAX_RESTARTS=${MAX_RESTARTS:-20}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); B=${BASELINES:-$(dirname "$HERE")/baselines}
RUNS=${RUNS:-$HOME/runs}
L=$RUNS/paper/$PLAYER-$ROW-u$USERS${TAG:+-$TAG}
COMPACT=(); [ -n "${COMPACT_AT:-}" ] && COMPACT+=(--compact-at "$COMPACT_AT"); [ -n "${COMPACT_KEEP:-}" ] && COMPACT+=(--compact-keep "$COMPACT_KEEP")
case $ROW in
    dc|f|m|m0) PLAY=(bare --player "$PLAYER" --row "$ROW" --run-dir "$L/play" --users "$USERS" --rounds "$ROUNDS" ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"});;
    *) PLAY=(play --player "$PLAYER" --row "$ROW" --run-dir "$L/play" --users "$USERS" --rounds "$ROUNDS" "${COMPACT[@]}");;
esac
mkdir -p "$L"
say() { echo "$(date -u +%m-%d\ %H:%M) reference.sh: $*" >> "$L/progress"; }
{   # read whole before it runs: bash reads a script as it goes, and the code may be synced again while this runs
    say "started: $PLAYER plays row $ROW, $USERS users, $ROUNDS rounds${COMPACT_AT:+, compaction at $COMPACT_AT keeping ${COMPACT_KEEP:-8000}}${MAX_TOKENS:+, reply cap $MAX_TOKENS}, baselines $B"
    for i in $(seq 1 "$MAX_RESTARTS"); do
        if (cd "$B" && uv run --no-sync --env-file .env python "$HERE/reference.py" "${PLAY[@]}") \
                >> "$L/play.log" 2>&1; then
            say "done"; touch "$L/play.done"; exit 0
        fi
        say "reference.py stopped ($L/play.log): restart $i of $MAX_RESTARTS"; sleep 60
    done
    say "reference.py stopped $MAX_RESTARTS times: giving up"; exit 1
}
