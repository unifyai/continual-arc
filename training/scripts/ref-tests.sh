#!/usr/bin/env bash
# ref-tests.sh: every reference row's two frozen tests (continual_arc_training.frozen), on the reference machine that
# played the rows. Every POLL seconds (600) it looks in RUNS/paper for rows whose stream has ended (reference.sh's
# play.done) and whose tests are not there (frozen/done), and takes up to PAR of them at once (3), each in a process of
# its own:
#   play  both tests, one after the other, each up to three times, with the row's own player, into ROW/frozen/TEST/play;
#         a wiped row or a floor at one user (h0, hpl0, m0, f) both again from an empty state, into TEST-reference, the
#         second, independent play whose spread from the first is the noise of playing a test twice; GLM's row h at 1, 2
#         and 4 users again with Qwen playing from GLM's memory, into TEST-qwen
#   then  ROW/frozen/summary.json, the test users' copies of the memory deleted (every call, turn and game of the tests
#         stays), and ROW/frozen/done
# A row whose tests fail two passes in a row is left for a person, said so in the log. Resumable: a play whose round is
# recorded is not played again. Stop it with its whole process group (kill -- -PGID). The plays run from BASELINES's
# environment, with its .env (OPENROUTER_API_KEY), as reference.sh's do; its log is RUNS/paper/ref-tests.log.
# Usage (on the machine): (setsid nohup bash ~/continual-arc/training/scripts/ref-tests.sh > /dev/null 2>&1 < /dev/null &)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
POLL=${POLL:-600}
PAR=${PAR:-3}
P=$RUNS/paper
mkdir -p "$P/.ref-tests-failed"
say() { echo "$(date -u +%m-%d\ %H:%M) ref-tests.sh: $*" >> "$P/ref-tests.log"; }
# finished: the rows whose stream has ended and whose tests are not there, the earliest ended first; a folder set aside
# under a row's name with a suffix (qwen-h-u1.old) is no row, nor one with no stream
finished() {
    for r in "$P"/*; do
        r=${r##*/}
        case $r in *.*) continue ;; esac
        [ -d "$P/$r/play" ] && [ -f "$P/$r/play.done" ] && [ ! -f "$P/$r/frozen/done" ] \
            && echo "$(python3 -c "import os, sys; print(int(os.path.getmtime(sys.argv[1])))" "$P/$r/play.done") $r"
    done | sort -n | cut -d" " -f2
}
play() {   # play ROW TEST NAME OPTIONS...: one play into P/ROW/frozen/NAME/play
    local row=$1 t=$2 name=$3 i out=$P/$1/frozen
    shift 3
    mkdir -p "$out"
    for i in 1 2 3; do
        (cd "$BASELINES" && uv run --no-sync --env-file .env python -m continual_arc_training.frozen play \
            --run "$P/$row" --test "$t" --name "$name" "$@") >> "$out/$name.log" 2>&1 && return 0
        say "$row: $name stopped short, attempt $i ($out/$name.log)"
    done
    return 1
}
take() {   # take ROW: play its tests, then its summary and done
    local row=$1 player kind users t name options ok=1 plays=() spec
    read -r player kind users < <(python3 -c "import json, sys; d = json.load(open(sys.argv[1]))
print(d['player'], d['row'], json.load(open(sys.argv[2]))['users'])" "$P/$row/play/driver.json" \
        "$P/$row/play/schedule.json") || { say "$row: no driver.json"; return; }
    for t in held-out seen; do
        plays+=("$t|$t|--player $player")
        case $kind:$users in h0:1 | hpl0:1 | m0:1 | f:1) plays+=("$t|$t-reference|--player $player --reference") ;; esac
        case $row in glm-h-u[124]) plays+=("$t|$t-qwen|--player qwen") ;; esac
    done
    for spec in "${plays[@]}"; do
        IFS='|' read -r t name options <<< "$spec"
        say "$row: $name ($options)"
        play "$row" "$t" "$name" $options || { ok=0; continue; }
        say "$row: $name played: $(tail -1 "$P/$row/frozen/$name.log" | cut -c1-200)"
    done
    (cd "$BASELINES" && uv run --no-sync python -m continual_arc_training.frozen summary --run "$P/$row") \
        >> "$P/$row/frozen/frozen.log" 2>&1 || ok=0
    if [ "$ok" = 1 ]; then
        rm -rf "$P/$row"/frozen/*/play/users/*/system/state
        touch "$P/$row/frozen/done"
        say "$row: done"
    else
        echo $(( $(cat "$P/.ref-tests-failed/$row" 2>/dev/null || echo 0) + 1 )) > "$P/.ref-tests-failed/$row"
        say "$row: not done ($(cat "$P/.ref-tests-failed/$row") of 2 passes)"
    fi
}
{   # read whole before it runs: bash reads a script as it goes
    say "started: the reference rows' frozen tests in $P, every ${POLL}s, $PAR at once"
    declare -A taking   # row: the pid of the process playing it
    while true; do
        for row in $(finished); do
            [ "$(cat "$P/.ref-tests-failed/$row" 2>/dev/null || echo 0)" -ge 2 ] && continue
            [ -n "${taking[$row]:-}" ] && kill -0 "${taking[$row]}" 2>/dev/null && continue
            while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
            [ -f "$P/$row/frozen/done" ] && continue   # its process ended while this one waited for room
            take "$row" &
            taking[$row]=$!
        done
        [ "${ONCE:-0}" = 1 ] && { wait; exit 0; }
        sleep "$POLL"
    done
}
