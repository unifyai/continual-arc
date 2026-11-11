#!/usr/bin/env bash
# One reference row of the RL graph (reference.py's docstring) kept going on the reference machine: reference.py plays
# every round not yet played and is started again when it stops (every round resumes from the run's records), up to
# MAX_RESTARTS times, a minute apart. The run is ~/runs/paper/$PLAYER-$ROW-u$USERS (play/ the benchmark run; progress
# and play.log beside it), where brain's paper-backup task finds it on a routed machine. A compaction probe's arm sets
# COMPACT_AT and COMPACT_KEEP (reference.py's --compact-at/--compact-keep; 0 is prime-agent's own) and a TAG that ends
# its run's name ($PLAYER-$ROW-u$USERS-$TAG). MAX_TOKENS sets a bare-model row's reply cap (reference.py bare
# --max-tokens; 32,768 unless set, the demo-cache floors' 128,000 since 21 Oct 2026), named in its started line.
# ROW=dc, f, m or m0 plays one of the player's bare-model rows instead
# (reference.py bare): the demo-cache floor, one cache for the team, the stateless floor, the program library, or the
# program library wiped before every instance. reference.py is the one beside this script.
# BASELINES (~/continual-arc-baselines by default) is the baselines checkout whose venv runs reference.py and whose .env
# holds OPENROUTER_API_KEY: a row on newer code than another row on the same machine takes a checkout and venv of its own,
# since this script restarts reference.py from that venv whenever it stops (19 Oct 2026: REF_ROWS_NODE_A' floors play on
# the code from before the restricted learner view, the six rows on the code after it).
# Usage (on the machine, with ~/loop this repo):
#   PLAYER=glm ROW=h USERS=1 setsid nohup bash ~/loop/reference.sh > /dev/null 2>&1 < /dev/null &
#   stop: this script's process group (kill -- -<its pid>)
set -uo pipefail
for k in "$(dirname "${BASH_SOURCE[0]}")"/{,../}keep.sh; do [ -f "$k" ] && . "$k" && break; done   # the run supervisor (keep.sh)
type keep_run > /dev/null 2>&1 || { echo "refused: no keep.sh beside $0, the run supervisor's helper" >&2; exit 1; }
export PATH=$HOME/.local/bin:$PATH
PLAYER=${PLAYER:?set PLAYER: glm or qwen}; ROW=${ROW:?set ROW: h, h0, hpl, hg, ha, hpl0, hplg, hgpl, hpb, hpb0, dc, f, m or m0}; USERS=${USERS:-1}; ROUNDS=${ROUNDS:-200}
MAX_RESTARTS=${MAX_RESTARTS:-20}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); B=${BASELINES:-$HOME/continual-arc-baselines}
L=~/runs/paper/$PLAYER-$ROW-u$USERS${TAG:+-$TAG}
COMPACT=(); [ -n "${COMPACT_AT:-}" ] && COMPACT+=(--compact-at "$COMPACT_AT"); [ -n "${COMPACT_KEEP:-}" ] && COMPACT+=(--compact-keep "$COMPACT_KEEP")
case $ROW in
    dc|f|m|m0) PLAY=(bare --player "$PLAYER" --row "$ROW" --run-dir "$L/play" --users "$USERS" --rounds "$ROUNDS" ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"});;
    *) PLAY=(play --player "$PLAYER" --row "$ROW" --run-dir "$L/play" --users "$USERS" --rounds "$ROUNDS" "${COMPACT[@]}");;
esac
# the run under the supervisor, its folder L (keep.sh), with the baselines checkout it plays on
keep_run "$L" --resume --log "$L/progress" --also "$B" --vars "PLAYER ROW USERS ROUNDS TAG COMPACT_AT COMPACT_KEEP MAX_TOKENS \
    MAX_RESTARTS" -- "$@" || exit 1
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
