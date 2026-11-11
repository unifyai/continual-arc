#!/usr/bin/env bash
# Rows of Continual-ARC's second and third tracks (the paper's Appendix K) on one machine: the ROWs given of TRACK
# (arc2 or arc3), each a cell at seed 0, one user, Kimi K3.1 Pro at medium effort through OpenRouter, with the settings
# continual-arc's baselines/scripts/launch-cell.sh gives its row, named by make_figures.py's row codes:
#   f     the stateless floor (BareModelLearner)
#   dc    the demo-cache floor (BareDemoCacheLearner, its cache capped on arc3 at an instance's three demonstrations)
#   m     the program library (BareProgramLibraryLearner); m0 the same, wiped before every instance (stateless)
#   h     prime-agent with every learning mechanism it ships; h0 the same, wiped before every instance
#   hpl   prime-agent told to keep a program library as m does (program_library); hpl0 the same, wiped
#   ha    prime-agent asked to store after every verdict (ask_to_store), read against h0
# The paper (Table tab:tracks-light) names them KC0 (f), KC (dc), KP (m), KP0 (m0), K (h), K0 (h0), KE (hpl), KE0 (hpl0)
# and KA (ha).
# On arc3 the program library's program is the game's rules, a world model (continual-arc's
# baselines/.../systems/world_model.py, from 1c499b7), and expert guidance is guidance.GAME_GUIDANCE, so m, m0, hpl
# and hpl0 play both tracks, m and m0 caching an arc3 game's demonstrations as dc does. Each
# cell plays its stream once, its state committed every round (arc-baselines run --no-epilogue --log-state), its model
# calls journaled whole (--capture-content), then each frozen test PLAYS times (4 unless set), every play one round of
# users forked from the stream's last commit (arc-baselines test --collective), in turn: seen 1, held-out 1, seen 2 and
# so on, at most SLOTS rounds (2 unless set) on the machine at once.
#
# PROVIDER, an OpenRouter provider slug, which may name one endpoint of a provider as moonshotai/mxfp4 does, sends every
# model call of the cells to it alone, with no fallback. Unpinned, OpenRouter spreads Kimi K3.1 Pro's calls over a dozen
# providers that do not play its effort alike, holding a test's plays on one of them for an hour or two by the session
# id every play of that test shares, so a row and its reset meet different servings (the paper's Appendix K).
#
# A cell is ~/runs/paper/PREFIX-TRACK-ROW (PREFIX kimi unless set): progress, the cell's log, its two tests' schedules,
# stream/ the stream's run directory (its attempt, cost journal and system in .workbench/ beside it) and
# stream/frozen/<test>-p<k>/play each play. Everything resumes: the stream from its records, a play cut short from its
# seeding, and a play done is not played again, so running this again after a stop or a reboot carries on where the
# cells were. Each cell is a background job of this script; the paper's cells each ran under unify's run
# supervisor instead, which recorded the code's commit and started the cell again after a reboot, and which plays
# nothing of its own.
#
# CAP_USD (required) stops every cell of TRACK and PREFIX on the machine once the records of their streams' instances and their plays'
# rounds have cost more than it, read every five minutes; what is still playing then costs on top, an instance a stream
# and at most SLOTS rounds. The stop is ~/tracks-light/TRACK.stop, which also stops the cells by hand; delete it to let
# them go on.
#
# Needs ~/continual-arc, a checkout of continual-arc whose baselines are synced (uv sync in baselines/, the workspace's
# .venv), baselines/.env holding OPENROUTER_API_KEY, prime-agent in ~/.local/bin, bubblewrap, and the track's data:
# for arc2 `continual-arc data fetch --datasets arc2`, for arc3 `continual-arc data fetch --datasets arc3` and
# `continual-arc precompute --config arc3 --seed 0` (setup.sh beside this script does all of it).
# Usage (on the machine, this script at ~/loop/tracks-light.sh):
#   CAP_USD=1200 SLOTS=3 setsid nohup bash ~/loop/tracks-light.sh arc2 m m0 hpl > /dev/null 2>&1 < /dev/null &
#   PREFIX=kimi-moonshot PROVIDER=moonshotai/mxfp4 PLAYS=2 CAP_USD=700 setsid nohup bash ... arc2 m hpl hpl0 ...
#   stop: touch ~/tracks-light/arc2.stop, or kill this script's process group (kill -- -<its pid>)
set -uo pipefail
TRACK=${1:?set TRACK: arc2 or arc3}; shift; ROWS=("$@"); CAP=${CAP_USD:?set CAP_USD: the most the cells on this machine may cost, in USD}
SLOTS=${SLOTS:-2}
case $TRACK in arc2|arc3) ;; *) echo "TRACK is arc2 or arc3, not $TRACK" >&2; exit 1;; esac
[ ${#ROWS[@]} -gt 0 ] || { echo "name the rows to play: f dc m m0 h h0 hpl hpl0 ha" >&2; exit 1; }
MODEL=moonshotai/kimi-k3.1-pro; PLAYS=${PLAYS:-4}; PREFIX=${PREFIX:-kimi}; PROVIDER=${PROVIDER:-}; MAX_RESTARTS=${MAX_RESTARTS:-20}
export PATH=$HOME/.local/bin:$PATH
CA=$HOME/continual-arc; BIN=$CA/.venv/bin; RUNS=$HOME/runs/paper; HOLD=$HOME/tracks-light
STOP=$HOLD/$TRACK.stop; LOG=$HOLD/$TRACK.log
mkdir -p "$RUNS" "$HOLD"
set -a; . "$CA/baselines/.env"; set +a

# launch-cell.sh's settings for each row (its floors, method, harness and their stateless-, ask- and pl- verbs), at
# medium effort, every call pinned to PROVIDER when it names one
REPLY=', "empty_reply": "reprompt", "archive_wipes": true'
CACHE=""; [ "$TRACK" = arc3 ] && CACHE=', "cache_cap": 3'
BARE="{\"model\": \"$MODEL\", \"max_tokens\": 128000, \"end\": \"finish\"$REPLY"
PRIME="{\"model\": \"$MODEL\", \"thinking\": \"medium\", \"end\": \"finish\"$REPLY, \"action_budget_s\": 300, \"max_model_calls\": 15"
PIN=""; [ -n "$PROVIDER" ] && PIN=", \"provider\": {\"order\": [\"$PROVIDER\"], \"allow_fallbacks\": false}"
BARE_EFFORT="{\"reasoning\": {\"effort\": \"medium\"}$PIN}"; PRIME_EFFORT="{\"reasoning_effort\": \"medium\"$PIN}"
B=continual_arc_baselines.systems.bare; P=continual_arc_baselines.systems.prime_agent:PrimeAgentLearner
declare -A LEARNER=([f]=$B:BareModelLearner [dc]=$B:BareDemoCacheLearner [m]=$B:BareProgramLibraryLearner
                    [m0]=$B:BareProgramLibraryLearner [h]=$P [h0]=$P [hpl]=$P [hpl0]=$P [ha]=$P)
declare -A KWARGS=([f]="$BARE}" [dc]="$BARE$CACHE}" [m]="$BARE$CACHE}" [m0]="$BARE$CACHE, \"stateless\": true}"
                   [h]="$PRIME}" [h0]="$PRIME, \"stateless\": true}" [hpl]="$PRIME, \"program_library\": true}"
                   [hpl0]="$PRIME, \"stateless\": true, \"program_library\": true}" [ha]="$PRIME, \"ask_to_store\": true}")
declare -A DEFAULTS=([f]=$BARE_EFFORT [dc]=$BARE_EFFORT [m]=$BARE_EFFORT [m0]=$BARE_EFFORT [h]=$PRIME_EFFORT
                     [h0]=$PRIME_EFFORT [hpl]=$PRIME_EFFORT [hpl0]=$PRIME_EFFORT [ha]=$PRIME_EFFORT)
for row in "${ROWS[@]}"; do
    [ -n "${LEARNER[$row]:-}" ] || { echo "no row $row: f dc m m0 h h0 hpl hpl0 ha" >&2; exit 1; }
done

say() { echo "$(date -u +%m-%d\ %H:%M) $*" >> "$LOG"; }

play() {   # one play of a test, in whichever of the machine's round slots is free first (flock -n: 75 when held)
    local test=$1 k=$2 slot rc
    while :; do
        for slot in $(seq 0 $((SLOTS - 1))); do
            flock -n -E 75 "$HOLD/round-$slot.lock" "$BIN/arc-baselines" test stream --schedule "$test.json" \
                --collective --out "stream/frozen/$test-p$k/play" >> cell.log 2>&1
            rc=$?; [ "$rc" != 75 ] && return "$rc"
        done
        sleep 30
    done
}

cell() {   # one cell: its stream, then its plays
    local row=$1 dir="$RUNS/$PREFIX-$TRACK-$1" i k test
    note() { echo "$(date -u +%m-%d\ %H:%M) tracks-light.sh: $*" >> "$dir/progress"; }
    mkdir -p "$dir"; cd "$dir" || return 1
    note "started: $PREFIX-$TRACK-$row, $MODEL, ${LEARNER[$row]} ${KWARGS[$row]}, defaults ${DEFAULTS[$row]}, cap $CAP USD for the machine"
    for test in seen held-out; do
        [ -f "$test.json" ] || "$BIN/continual-arc" schedule make --config "$TRACK" --seed 0 --test "$test" \
            --out "$test.json" >> cell.log 2>&1 || { note "making the $test test failed (cell.log)"; return 1; }
    done
    if [ ! -f stream.done ]; then
        for i in $(seq 1 "$MAX_RESTARTS"); do
            [ -f "$STOP" ] && { note "stopped before the stream ($STOP)"; return 0; }
            if "$BIN/arc-baselines" run --learner "${LEARNER[$row]}" --learner-kwargs "${KWARGS[$row]}" \
                    --llm-defaults "${DEFAULTS[$row]}" --config "$TRACK" --order workstreams --seed 0 \
                    --no-epilogue --log-state --events --capture-content --quiet --resume --out stream >> cell.log 2>&1; then
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

stop_cells() {   # every arc-baselines that this track's and prefix's cells run, known by its folder: other runs on the machine go on
    local pid
    for pid in $(pgrep -f "arc-baselines (run|test)"); do
        case $(readlink "/proc/$pid/cwd") in "$RUNS/$PREFIX-$TRACK-"*) kill -TERM "$pid";; esac
    done
}

spent() {   # what the machine's cells of this track and prefix have cost so far: every stream instance and every play's round
    python3 - "$RUNS" "$PREFIX-$TRACK" <<'PY'
import json, pathlib, sys
total = 0.0
for cell in pathlib.Path(sys.argv[1]).glob(f"{sys.argv[2]}-*"):
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
    say "started: $TRACK, cells ${ROWS[*]} as $PREFIX-$TRACK-ROW, $PLAYS plays per test, provider ${PROVIDER:-as OpenRouter routes}, cap $CAP USD, $SLOTS round slots"
    pids=()
    for row in "${ROWS[@]}"; do cell "$row" & pids+=($!); done
    while :; do
        alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        [ "$alive" = 0 ] && break
        cost=$(spent)
        if [ ! -f "$STOP" ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$cost" "$CAP"; then
            echo "$cost USD spent, over the cap of $CAP USD" > "$STOP"
            say "stopping: $cost USD spent, over the cap of $CAP USD"
            stop_cells
        fi
        sleep 300
    done
    wait
    say "finished: $(spent) USD spent"
}
