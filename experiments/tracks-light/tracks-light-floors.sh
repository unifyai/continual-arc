#!/usr/bin/env bash
# The light rows of Continual-ARC's second and third tracks (the paper's Appendix K), kept going on one machine a
# track: TRACK's four cells at seed 0, one user, Kimi K3.1 Pro at medium effort through OpenRouter, each with the
# settings continual-arc's baselines/scripts/launch-cell.sh gives its row: the stateless floor (f, BareModelLearner),
# the demo-cache floor (dc, BareDemoCacheLearner, its cache capped on arc3 at an instance's three demonstrations),
# prime-agent with every learning mechanism it ships (h) and prime-agent wiped before every instance (h0). Each cell
# plays its stream once, its state committed every round (arc-baselines run --no-epilogue --log-state), then each
# frozen test four times, every play one round of users forked from the stream's last commit (arc-baselines test
# --collective), in turn: seen 1, held-out 1, seen 2 and so on, at most two rounds on the machine at once.
#
# The paper reads this script's two floors, KC0 (f) and KC (dc), on continual-arc v0.4.0, whose configurations
# arc2-small and arc3-small serve the same instances slot for slot as v0.6.0's arc2 and arc3; its h and h0 cells were
# played again as K and K0 by tracks-light.sh beside it, on v0.6.0 with prime-agent 0.9.6, since v0.4.0's adapter
# installed none of prime-agent's Python skills.
#
# A cell is ~/runs/paper/kimi-TRACK-ROW: progress, the cell's log, its two tests' schedules, stream/ the stream's run
# directory (its attempt, cost journal and system in .workbench/ beside it) and stream/frozen/<test>-p<k>/play each
# play. Everything resumes: the stream from its records, a play cut short from its seeding, and a play done is not
# played again, so running this again after a stop or a reboot carries on where the cells were.
#
# CAP_USD (required) stops every cell once the records of its streams' instances and its plays' rounds have cost more
# than it, read every five minutes; what is still playing then costs on top, an instance a stream and at most two
# rounds. The stop is ~/tracks-light/TRACK.stop, which also stops the cells by hand; delete it to let them go on.
#
# Needs ~/continual-arc, a checkout of continual-arc whose baselines are synced (uv sync in baselines/, the workspace's
# .venv), baselines/.env holding OPENROUTER_API_KEY, prime-agent 0.9.5 in ~/.local/bin, bubblewrap, and the track's
# data: for arc2 `continual-arc data fetch --datasets arc2`, for arc3 the pinned games in
# ~/.cache/continual-arc/arc-agi-3 (`continual-arc data fetch --datasets arc3`) and
# `continual-arc precompute --config arc3-small --seed 0` (REF=v0.4.0 PRIME_AGENT_VERSION=0.9.5 CONFIG=arc3-small
# bash setup.sh HOST arc3 does all of it, and REF=v0.4.0 PRIME_AGENT_VERSION=0.9.5 bash setup.sh HOST arc2).
# Usage (on the machine, this script at ~/loop/tracks-light-floors.sh):
#   CAP_USD=1750 setsid nohup bash ~/loop/tracks-light-floors.sh arc2 > /dev/null 2>&1 < /dev/null &
#   stop: touch ~/tracks-light/arc2.stop, or kill this script's process group (kill -- -<its pid>)
set -uo pipefail
TRACK=${1:?set TRACK: arc2 or arc3}; CAP=${CAP_USD:?set CAP_USD: the most the cells of this track may cost, in USD}
case $TRACK in arc2|arc3) ;; *) echo "TRACK is arc2 or arc3, not $TRACK" >&2; exit 1;; esac
MODEL=moonshotai/kimi-k3.1-pro; PLAYS=4; MAX_RESTARTS=${MAX_RESTARTS:-20}
export PATH=$HOME/.local/bin:$PATH
CA=$HOME/continual-arc; BIN=$CA/.venv/bin; RUNS=$HOME/runs/paper; HOLD=$HOME/tracks-light
STOP=$HOLD/$TRACK.stop; LOG=$HOLD/$TRACK.log
mkdir -p "$RUNS" "$HOLD"
set -a; . "$CA/baselines/.env"; set +a

# launch-cell.sh's settings for each row (its floors, harness and stateless-harness verbs), at medium effort
REPLY=', "empty_reply": "reprompt", "archive_wipes": true'
CACHE=""; [ "$TRACK" = arc3 ] && CACHE=', "cache_cap": 3'
BARE="{\"model\": \"$MODEL\", \"max_tokens\": 128000, \"end\": \"finish\"$REPLY"
PRIME="{\"model\": \"$MODEL\", \"thinking\": \"medium\", \"end\": \"finish\"$REPLY, \"action_budget_s\": 300, \"max_model_calls\": 15"
declare -A LEARNER=(
    [f]=continual_arc_baselines.systems.bare:BareModelLearner
    [dc]=continual_arc_baselines.systems.bare:BareDemoCacheLearner
    [h]=continual_arc_baselines.systems.prime_agent:PrimeAgentLearner
    [h0]=continual_arc_baselines.systems.prime_agent:PrimeAgentLearner)
declare -A KWARGS=([f]="$BARE}" [dc]="$BARE$CACHE}" [h]="$PRIME}" [h0]="$PRIME, \"stateless\": true}")
declare -A DEFAULTS=([f]='{"reasoning": {"effort": "medium"}}' [dc]='{"reasoning": {"effort": "medium"}}'
                     [h]='{"reasoning_effort": "medium"}' [h0]='{"reasoning_effort": "medium"}')

say() { echo "$(date -u +%m-%d\ %H:%M) $*" >> "$LOG"; }

play() {   # one play of a test, in whichever of the machine's two round slots is free first (flock -n: 75 when held)
    local test=$1 k=$2 slot rc
    while :; do
        for slot in 0 1; do
            flock -n -E 75 "$HOLD/round-$slot.lock" "$BIN/arc-baselines" test stream --schedule "$test.json" \
                --collective --out "stream/frozen/$test-p$k/play" >> cell.log 2>&1
            rc=$?; [ "$rc" != 75 ] && return "$rc"
        done
        sleep 30
    done
}

cell() {   # one cell: its stream, then its plays
    local row=$1 dir="$RUNS/kimi-$TRACK-$1" i k test
    note() { echo "$(date -u +%m-%d\ %H:%M) tracks-light.sh: $*" >> "$dir/progress"; }
    mkdir -p "$dir"; cd "$dir" || return 1
    note "started: kimi-$TRACK-$row, $MODEL, ${LEARNER[$row]} ${KWARGS[$row]}, defaults ${DEFAULTS[$row]}, cap $CAP USD for the track"
    for test in seen held-out; do
        [ -f "$test.json" ] || "$BIN/continual-arc" schedule make --config "$TRACK-small" --seed 0 --test "$test" \
            --out "$test.json" >> cell.log 2>&1 || { note "making the $test test failed (cell.log)"; return 1; }
    done
    if [ ! -f stream.done ]; then
        for i in $(seq 1 "$MAX_RESTARTS"); do
            [ -f "$STOP" ] && { note "stopped before the stream ($STOP)"; return 0; }
            if "$BIN/arc-baselines" run --learner "${LEARNER[$row]}" --learner-kwargs "${KWARGS[$row]}" \
                    --llm-defaults "${DEFAULTS[$row]}" --config "$TRACK-small" --order workstreams --seed 0 \
                    --no-epilogue --log-state --events --quiet --resume --out stream >> cell.log 2>&1; then
                touch stream.done; note "stream done"; break
            fi
            note "the stream stopped (cell.log): restart $i of $MAX_RESTARTS"; sleep 60
        done
        [ -f stream.done ] || { note "the stream stopped $MAX_RESTARTS times: giving up"; return 1; }
    fi
    for k in $(seq 1 "$PLAYS"); do
        for test in seen held-out; do
            [ -f "$test-p$k.done" ] && continue
            for i in $(seq 1 5); do
                [ -f "$STOP" ] && { note "stopped before $test play $k ($STOP)"; return 0; }
                if play "$test" "$k"; then touch "$test-p$k.done"; note "$test play $k done"; break; fi
                note "$test play $k stopped (cell.log): try $i of 5"; sleep 60
            done
            [ -f "$test-p$k.done" ] || { note "$test play $k failed 5 times: giving up"; return 1; }
        done
    done
    note "done: the stream and $PLAYS plays of each test"
}

spent() {   # what the track's records have cost so far: every stream instance and every play's round
    python3 - "$RUNS" "$TRACK" <<'PY'
import json, pathlib, sys
total = 0.0
for cell in pathlib.Path(sys.argv[1]).glob(f"kimi-{sys.argv[2]}-*"):
    for records in [cell / "stream" / "records.jsonl", *cell.glob("stream/frozen/*/play/records.jsonl")]:
        if records.is_file():
            for line in records.read_text().splitlines():
                try:
                    total += json.loads(line)["resources"].get("llm_cost_usd", 0.0)
                except (ValueError, KeyError):
                    pass   # a line still being written
print(f"{total:.2f}")
PY
}

{   # read whole before it runs: bash reads a script as it goes, and the script may be copied again while this runs
    say "started: $TRACK, cells f dc h h0, cap $CAP USD"
    pids=()
    for row in f dc h h0; do cell "$row" & pids+=($!); done
    while :; do
        alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        [ "$alive" = 0 ] && break
        cost=$(spent)
        if [ ! -f "$STOP" ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$cost" "$CAP"; then
            echo "$cost USD spent, over the cap of $CAP USD" > "$STOP"
            say "stopping: $cost USD spent, over the cap of $CAP USD"
            pkill -TERM -f "arc-baselines (run|test)"
        fi
        sleep 300
    done
    wait
    say "finished: $(spent) USD spent"
}
