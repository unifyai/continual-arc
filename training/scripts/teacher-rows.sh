#!/usr/bin/env bash
# teacher-rows.sh: one of the teacher's rows (continual_arc_training.teacher_rows), ROW hg, hgpl or hplg, once its teacher's
# reference row has played its stream (reference.sh, RUNS/paper/glm-h-u1 for hg and glm-hpl-u1 for the others). Its stream,
# two plays of 100 users, then both frozen tests PLAYS times (four), each play up to three times, started GAP seconds apart
# (120), since plays started together meet the model's providers' rate limits and a game whose calls fail plays differently.
# The row is RUNS/paper/qwen-$ROW-u1: frozen/NAME/play each play, play/ the stream as a row's is read, progress and play.log
# beside them, play.done and frozen/done once all are recorded. Resumable: a play whose round is recorded is not played
# again. The row's reset is reference.sh's PLAYER=qwen ROW=h0, or hpl0 for hplg, best at the same hours as this stream.
# It runs from BASELINES's environment, with its .env (OPENROUTER_API_KEY), as reference.sh does.
# Usage (on the machine): ROW=hplg setsid nohup bash ~/continual-arc/training/scripts/teacher-rows.sh \
#     > /dev/null 2>&1 < /dev/null &
#   stop: this script's process group (kill -- -<its pid>)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ROW=${ROW:?set ROW: hg, hgpl or hplg}
case $ROW in
    hg) TEACHER=$RUNS/paper/glm-h-u1 ;;
    hgpl | hplg) TEACHER=$RUNS/paper/glm-hpl-u1 ;;
    *) echo "ROW is hg, hgpl or hplg, not $ROW" >&2; exit 2 ;;
esac
GAP=${GAP:-120}; PLAYS=${PLAYS:-4}
L=$RUNS/paper/qwen-$ROW-u1
mkdir -p "$L/frozen"
say() { echo "$(date -u +%m-%d\ %H:%M) teacher-rows.sh: $*" >> "$L/progress"; }
py() { (cd "$BASELINES" && uv run --no-sync --env-file .env python -m continual_arc_training.teacher_rows "$@") >> "$L/play.log" 2>&1; }
take() {   # take NAME --first F | --test TEST
    local name=$1 i; shift
    for i in 1 2 3; do
        py play --row "$ROW" --teacher "$TEACHER" --run "$L" --name "$name" "$@" && { say "$name played"; return 0; }
        say "$name stopped short, attempt $i ($L/play.log)"
    done
    say "$name not played in three tries"; return 1
}
{   # read whole before it runs: bash reads a script as it goes, and the code may be pulled again while this runs
    [ -f "$TEACHER/play.done" ] || { say "its teacher $TEACHER has not played its stream"; exit 1; }
    say "started: $ROW from $TEACHER, $PLAYS plays of each test, $GAP s apart, baselines $BASELINES"
    pids=()
    for first in 000 100; do take "stream-$first" --first "$((10#$first))" & pids+=($!); sleep "$GAP"; done
    for k in $(seq 1 "$PLAYS"); do
        for t in held-out seen; do
            name=$t; [ "$k" = 1 ] || name=$t-$k
            take "$name" --test "$t" & pids+=($!); sleep "$GAP"
        done
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" = 0 ] || { say "a play was not played: left for a person"; exit 1; }
    py view --teacher "$TEACHER" --run "$L" && touch "$L/play.done" || { say "the stream's view failed ($L/play.log)"; exit 1; }
    (cd "$BASELINES" && uv run --no-sync --env-file .env python -m continual_arc_training.frozen summary --run "$L") >> "$L/play.log" 2>&1
    rm -rf "$L"/frozen/*/play/users/*/system/state
    touch "$L/frozen/done"; say "done"
}
