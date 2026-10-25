#!/usr/bin/env bash
# A frozen row on the node's own vLLM, round by round with no lag (continual_arc_training.play): H0 (the base model in
# prime-agent, wiped before every instance) or the bare model (ROW=h0 or bare), ROUNDS rounds from FIRST_ROUND (default
# 0), every stage marked, so a kill at any step resumes there. Round r: serve.sh serves the base model, the driver plays
# round r of arc1-small served to USERS users (default 100, the whole schedule's number: users 0..USERS-1 are exactly
# the 100's first USERS, round by round), and the round's memory counters go to the progress log. With EPILOGUE=1 the
# row plays the epilogue alone, served to EPILOGUE_USERS of the USERS users, and with WIDTH its instances dealt WIDTH to a
# round: the held-out test's instances are user 0's of the epilogue, and neither row carries anything from one instance
# to the next, so one play of users 0-3 serves every trained run at 1, 2 and 4 users as transfer's reference. The run is
# RUNS/paper/ROW (ROW-uUSERS for fewer than 100 users), its play under play/ or, with EPILOGUE=1, under
# epilogue-uUSERS[-wWIDTH]/. The server is shared with any other run on the node: the last to finish stops it.
# Usage (on the node), the trained runs' references over the held-out instances:
#   ROW=h0 USERS=4 EPILOGUE=1 EPILOGUE_USERS=4 WIDTH=100 ROUNDS=4 bash ~/continual-arc/training/scripts/paper.sh
#   ROW=bare USERS=4 EPILOGUE=1 EPILOGUE_USERS=4 WIDTH=100 ROUNDS=4 bash ~/continual-arc/training/scripts/paper.sh
# and a smoke test of the node's stack, H0 on a 4-user cut of the stream: ROW=h0 USERS=4 ROUNDS=3 bash .../paper.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
export RUNS
ROW=${ROW:?set ROW: h0 or bare}
case $ROW in h0 | bare) ;; *) echo "ROW must be h0 or bare (the trained rows run through realtime.sh), got $ROW"; exit 1 ;; esac
FIRST_ROUND=${FIRST_ROUND:-0}
ROUNDS=${ROUNDS:?set ROUNDS: how many rounds this call plays, from FIRST_ROUND}
USERS=${USERS:-100}
EPILOGUE=${EPILOGUE:-0}
EPILOGUE_USERS=${EPILOGUE_USERS:-$USERS}
WIDTH=${WIDTH:-}
SERVE_GPUS=${SERVE_GPUS:-0,1,2,3,4,5,6,7}; SERVE_DP=${SERVE_DP:-8}
# the epilogue is served to some of the configuration's users: fail here, not partway into a play
if [ "$EPILOGUE" = 1 ] && [ "$EPILOGUE_USERS" -gt "$USERS" ]; then
    echo "EPILOGUE_USERS ($EPILOGUE_USERS) must not be more than USERS ($USERS)"; exit 1
fi
SUFFIX=""; [ "$USERS" = 100 ] || SUFFIX="-u$USERS"
L=$RUNS/paper/$ROW$SUFFIX
if [ "$EPILOGUE" = 1 ]; then
    PLAY_DIR=$L/epilogue$SUFFIX${WIDTH:+-w$WIDTH}
    SCHED_FLAGS=(--epilogue-users "$EPILOGUE_USERS" --users "$USERS" ${WIDTH:+--width "$WIDTH"})
else
    PLAY_DIR=$L/play$SUFFIX
    SCHED_FLAGS=(--users "$USERS")
fi
mkdir -p "$L"
say() { echo "$(date -u +%H:%M) paper($ROW$SUFFIX): $*" >> "$L/progress"; }
round_dir() { printf "%s/rounds/r%04d" "$PLAY_DIR" "$1"; }

play_round() {   # play_round R: round R played on the base model, a crashed or killed play launched again (when nothing
                 # runs and the round is still short) up to twice before the round fails for good
    local r=$1 pattern="[c]ontinual_arc_training.play --row $ROW --run-dir $PLAY_DIR .*--round $r( |\$)" have=0 want=1
    local attempt mem
    [ -f "$PLAY_DIR/round-play$r.done" ] && return 0
    bash "$HERE/serve.sh" "$SERVE_GPUS" "$SERVE_DP" || { say "the server did not start"; return 1; }
    for attempt in 1 2 3; do
        if ! pgrep -f "$pattern" >/dev/null; then
            say "play round $r started (attempt $attempt)"
            (cd "$PRIME_RL" && setsid nohup uv run --no-sync python -m continual_arc_training.play --row "$ROW" \
                --run-dir "$PLAY_DIR" --base-url http://localhost:8101/v1 --served "$MODEL" --round "$r" \
                "${SCHED_FLAGS[@]}" >> "$L/play-round$r.log" 2>&1 < /dev/null &)
            sleep "${GRACE:-60}"   # the play's start, before pgrep can see it
        fi
        while pgrep -f "$pattern" >/dev/null; do sleep 15; done
        # the runner writes schedule.json for its cut before playing an instance: without it the launch died first
        if [ -f "$PLAY_DIR/schedule.json" ]; then
            want=$(python3 -c "import json;print(len(json.load(open('$PLAY_DIR/schedule.json'))['slots']))") || return 1
        fi
        have=$([ -f "$PLAY_DIR/records.jsonl" ] && wc -l < "$PLAY_DIR/records.jsonl" || echo 0)
        [ "$have" -ge "$want" ] && break
        say "play round $r short ($have/$want) after attempt $attempt: see $L/play-round$r.log"
    done
    [ "$have" -ge "$want" ] || { say "play round $r failed after $attempt attempts"; return 1; }
    touch "$PLAY_DIR/round-play$r.done"; say "play round $r done ($have/$want)"
    # the round's memory counters, which the play writes (a missing file never fails the round: the play logged why)
    mem=$(round_dir "$r")/memory.json
    if [ -f "$mem" ]; then
        say "$(PYTHONPATH=$SRC python3 -m continual_arc_training.memory line "$mem")"
    fi
    return 0
}

{   # read whole before it runs: bash reads a script as it goes, and the code may be pulled again while this runs
    LAST=$((FIRST_ROUND + ROUNDS - 1))
    [ -f "$PLAY_DIR/paper-$FIRST_ROUND-$LAST.done" ] && { say "already done"; exit 0; }
    for ((r = FIRST_ROUND; r <= LAST; r++)); do
        say "round $r"
        play_round "$r" || exit 1
    done
    # the server is shared with any other run on this node: the last to finish stops it (each paper.sh runs in a process
    # group of its own)
    ps -eo pgid=,args= | awk '$2 == "bash" && $3 ~ /\/paper\.sh$/ {print $1}' | sort -u | grep -vqx "$(ps -o pgid= $$ | tr -d ' ')" \
        || pgrep -f "[c]ontinual_arc_training.realtime" >/dev/null || pkill -f "[v]llm serve"
    touch "$PLAY_DIR/paper-$FIRST_ROUND-$LAST.done"
    say "done"
    exit 0
}
