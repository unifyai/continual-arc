#!/usr/bin/env bash
# run.sh: GLM-5.3 Flash's building-block rows (README.md) on this machine, a clone of REF_TESTS_NODE' disk carrying
# deploy.sh's trees. The streams of rows hpb and hpb0 at one user and seed 0 on main's stream, each started by
# reference.sh beside this folder (~/loop-blocks) from BASELINES (~/cab-blocks), as reference.sh started main's rows on
# 28 Oct 2026, into ~/runs/paper/glm-ROW-u1-turns; then PLAYS (4) plays of each frozen test by each row: hpb0's from an
# empty state (TEST-reference, then TEST-reference-2 on), which wait only for its stream to have begun (its driver.json),
# and hpb's from its end-of-stream memory (TEST, then TEST-2 on), which wait for its stream's play.done. Up to PAR (1)
# plays at once, started at least GAP (300) seconds apart, so that beside the reviewers' sweep (sweep/README.md, at most
# about 300 games in flight) the model's providers meet the light load main's tests played under: a 429 storm turns
# solves into give-ups (analysis/glm-harness/storm-trace.md). Each play is a run of its own under the supervisor
# (keep.sh), tried up to three times as extra-tests.sh tries one. Refused without PURPOSE. Resumable: a stream whose
# play.done exists or whose supervisor runs is passed over, and a play whose round is recorded is not played again.
# Once every play is recorded, each run's frozen/summary.json and frozen/done. Its log is ~/runs/blocks.log, each
# stream's progress RUN/progress and each play's log RUN/frozen/NAME.log.
# Usage (on the machine): cd ~ && PURPOSE="..." setsid -f nohup bash ~/loop-blocks/blocks/run.sh > /dev/null 2>&1 < /dev/null
#   stop: kill -- -<its process group>, then each stream's reference.sh by its own group
set -uo pipefail
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); LOOP=$(dirname "$S")
export BASELINES=${BASELINES:-$HOME/cab-blocks}
PAR=${PAR:-1} GAP=${GAP:-300} POLL=${POLL:-120} PLAYS=${PLAYS:-4}
LOG=$HOME/runs/blocks.log
mkdir -p ~/runs/paper
say() { echo "$(date -u +%m-%d\ %H:%M) run.sh: $*" >> "$LOG"; }
. "$LOOP/keep.sh" || { say "refused: no keep.sh in $LOOP"; exit 1; }
keep_purpose 2>> "$LOG" || exit 1
export PATH=$HOME/.local/bin:$PATH
run_of() { echo "$HOME/runs/paper/glm-$1-u1-turns"; }
recorded() {   # recorded RUN NAME: the play's records hold its whole round, the test's 100 instances
    [ -f "$1/frozen/$2/play/records.jsonl" ] && [ "$(wc -l < "$1/frozen/$2/play/records.jsonl")" -ge 100 ]
}
stream() {   # stream ROW: its stream by reference.sh, unless it has ended or plays
    local L; L=$(run_of "$1")
    [ -f "$L/play.done" ] && { say "${L##*/}: its stream has ended"; return; }
    pgrep -f "runs.py launch .*--run=paper/${L##*/} " > /dev/null && { say "${L##*/}: its stream plays"; return; }
    (cd ~ && PLAYER=glm ROW=$1 USERS=1 TAG=turns setsid -f nohup bash "$LOOP/reference.sh" > /dev/null 2>&1 < /dev/null)
    say "${L##*/}: its stream started, from $LOOP and $BASELINES"
}
play() {   # play ROW TEST NAME [--reference]: one play into RUN/frozen/NAME/play, up to three tries
    local row=$1 t=$2 name=$3 ref=${4:-} L out sched i
    L=$(run_of "$row"); out=$L/frozen
    sched=$(python3 "$LOOP/frozen.py" schedule --run "$L" --test "$t") \
        || { say "${L##*/}: its $t test's schedule is not known"; return 1; }
    mkdir -p "$out"
    for i in 1 2 3; do
        say "${L##*/}: $name started${ref:+ from an empty state}, try $i"
        (cd "$BASELINES" && export PYTHONPATH=$LOOP && keep_exec "$out/$name" --resume --code "$LOOP" \
            --also "$BASELINES" --also "$HOME/benchmark-turns" -- uv run --no-sync --env-file .env python "$LOOP/frozen.py" \
            play --run "$L" --test "$t" --schedule "$sched" --name "$name" --player glm $ref) >> "$out/$name.log" 2>&1 \
            && { say "${L##*/}: $name played: $(tail -1 "$out/$name.log" | cut -c1-200)"; return 0; }
        say "${L##*/}: $name stopped short, try $i ($out/$name.log)"
    done
    say "${L##*/}: $name not played in three tries; left for a person"
    return 1
}
queue() {   # the plays in the order they are taken, a line each, "ROW TEST NAME [--reference]": hpb0's, which wait for
    # no stream to end, then hpb's; each play's held-out and seen tests side by side, so both tests' readings fill alike
    local k sfx t
    for k in $(seq 1 "$PLAYS"); do
        sfx=; [ "$k" = 1 ] || sfx=-$k
        for t in held-out seen; do echo "hpb0 $t $t-reference$sfx --reference"; done
    done
    for k in $(seq 1 "$PLAYS"); do
        sfx=; [ "$k" = 1 ] || sfx=-$k
        for t in held-out seen; do echo "hpb $t $t$sfx"; done
    done
}
{   # read whole before it runs: bash reads a script as it goes
    say "started: the streams of hpb and hpb0, then $PLAYS plays of each test by each, $PAR at once, $GAP s apart"
    stream hpb; sleep 30; stream hpb0
    while read -r row t name ref; do
        L=$(run_of "$row")
        recorded "$L" "$name" && continue
        case $row in hpb0) need=$L/play/driver.json ;; *) need=$L/play.done ;; esac
        [ -f "$need" ] || say "${L##*/}: $name waits for ${need#"$L"/}"
        until [ -f "$need" ]; do sleep "$POLL"; done
        while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
        # stdin from nothing: a play's commands would otherwise read the rest of the queue
        play "$row" "$t" "$name" $ref < /dev/null &
        sleep "$GAP"
    done < <(queue)
    wait
    for row in hpb hpb0; do
        L=$(run_of "$row")
        short=$(queue | while read -r r _ name _; do [ "$r" = "$row" ] && ! recorded "$L" "$name" && echo "$name"; done)
        [ -z "$short" ] || { say "${L##*/}: not done, short of $(echo $short)"; continue; }
        (cd "$BASELINES" && PYTHONPATH=$LOOP uv run --no-sync python "$LOOP/frozen.py" summary --run "$L") \
            >> "$L/frozen/frozen.log" 2>&1 && touch "$L/frozen/done" && say "${L##*/}: every play recorded, its summary written"
    done
    say "every play taken"
}
