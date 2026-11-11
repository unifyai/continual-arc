#!/usr/bin/env bash
# run.sh: the library ceiling (README.md) on this machine, a clone of REF_TESTS_NODE' disk carrying deploy.sh's trees.
# GLM-5.3 Flash's row hpl0 (main's GE0: c1 and the expert guidance, wiped before every instance) and the same given the
# Re-ARC primitives that the one-user stream's rules use (hpl0d1) or the four-user stream's (hpl0d4), each a run of its
# frozen tests alone, ~/runs/paper/glm-ROW-u1-ceiling, whose driver.json reference.py driver writes; then PLAYS (4)
# plays of the held-out test by each row, and of the seen test by hpl0 and hpl0d1, every play from an empty state:
# hpl0's named TEST-reference (to TEST-reference-4) and the others' TEST (to TEST-4), so that each library row reads
# against hpl0 as a row reads against its reset (continual-arc-paper's scripts/read_ceiling.py). Taken in rotation,
# every row's first play of each test before any row's second, so that the rows meet the same load, up to PAR (2) at
# once and at least GAP (300) seconds apart, beside the reviewers' sweep: a 429 storm turns solves into give-ups
# (analysis/glm-harness/storm-trace.md). Each play is a run of its own under the supervisor (keep.sh), from ~/loop-ceiling
# with BASELINES (~/cab-ceiling), tried up to three times as extra-tests.sh tries one. Refused without PURPOSE. Resumable:
# a play whose round is recorded is not played again. Once every play is recorded, each run's frozen/summary.json and
# frozen/done. Its log is ~/runs/ceiling.log, each play's RUN/frozen/NAME.log.
# Usage (on the machine): cd ~ && PURPOSE="..." setsid -f nohup bash ~/loop-ceiling/ceiling/run.sh > /dev/null 2>&1 < /dev/null
#   stop: kill -- -<its process group>
set -uo pipefail
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); LOOP=$(dirname "$S")
BASELINES=${BASELINES:-$HOME/cab-ceiling}
PAR=${PAR:-2} GAP=${GAP:-300} PLAYS=${PLAYS:-4}
ROWS=(hpl0 hpl0d1 hpl0d4)
SEEN=(hpl0 hpl0d1)   # the rows that play the seen test too: the one-user library's calibration against no library
LOG=$HOME/runs/ceiling.log
mkdir -p ~/runs/paper
say() { echo "$(date -u +%m-%d\ %H:%M) run.sh: $*" >> "$LOG"; }
. "$LOOP/keep.sh" || { say "refused: no keep.sh in $LOOP"; exit 1; }
keep_purpose 2>> "$LOG" || exit 1
export PATH=$HOME/.local/bin:$PATH
run_of() { echo "$HOME/runs/paper/glm-$1-u1-ceiling"; }
recorded() {   # recorded RUN NAME: the play's records hold its whole round, the test's 100 instances
    [ -f "$1/frozen/$2/play/records.jsonl" ] && [ "$(wc -l < "$1/frozen/$2/play/records.jsonl")" -ge 100 ]
}
play() {   # play ROW TEST NAME: one play from an empty state into RUN/frozen/NAME/play, up to three tries
    local row=$1 t=$2 name=$3 L out sched i
    L=$(run_of "$row"); out=$L/frozen
    # the one-user test's schedule, as ee7a4b3's extra-tests.sh and ref-tests.sh name it, that commit's frozen.py having no
    # schedule command (blocks/run.sh, 9b470c8)
    sched=$LOOP/schedules/arc1-small-$t-test-seed0.json
    [ -f "$sched" ] || { say "${L##*/}: the $t test's schedule is not in $LOOP/schedules"; return 1; }
    mkdir -p "$out"
    for i in 1 2 3; do
        say "${L##*/}: $name started, try $i"
        (cd "$BASELINES" && export PYTHONPATH=$LOOP && keep_exec "$out/$name" --resume --code "$LOOP" \
            --also "$BASELINES" --also "$HOME/benchmark-turns" -- uv run --no-sync --env-file .env python "$LOOP/frozen.py" \
            play --run "$L" --test "$t" --schedule "$sched" --name "$name" --player glm --reference) >> "$out/$name.log" 2>&1 \
            && { say "${L##*/}: $name played: $(tail -1 "$out/$name.log" | cut -c1-200)"; return 0; }
        say "${L##*/}: $name stopped short, try $i ($out/$name.log)"
    done
    say "${L##*/}: $name not played in three tries; left for a person"
    return 1
}
queue() {   # the plays in the order they are taken, a line each, "ROW TEST NAME"
    local k sfx t row
    for k in $(seq 1 "$PLAYS"); do
        sfx=; [ "$k" = 1 ] || sfx=-$k
        for t in held-out seen; do
            for row in "${ROWS[@]}"; do
                [ "$t" = seen ] && [[ " ${SEEN[*]} " != *" $row "* ]] && continue
                if [ "$row" = hpl0 ]; then echo "$row $t $t-reference$sfx"; else echo "$row $t $t$sfx"; fi
            done
        done
    done
}
{   # read whole before it runs: bash reads a script as it goes
    say "started: $PLAYS plays of the held-out test by ${ROWS[*]} and of the seen test by ${SEEN[*]}, $PAR at once, $GAP s apart"
    for row in "${ROWS[@]}"; do
        L=$(run_of "$row")
        [ -f "$L/play/driver.json" ] || (cd "$BASELINES" && PYTHONPATH=$LOOP uv run --no-sync python "$LOOP/reference.py" \
            driver --player glm --row "$row" --run-dir "$L/play") >> "$LOG" 2>&1 \
            || { say "${L##*/}: its driver.json was not written"; exit 1; }
    done
    while read -r row t name; do
        recorded "$(run_of "$row")" "$name" && continue
        while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
        # stdin from nothing: a play's commands would otherwise read the rest of the queue
        play "$row" "$t" "$name" < /dev/null &
        sleep "$GAP"
    done < <(queue)
    wait
    for row in "${ROWS[@]}"; do
        L=$(run_of "$row")
        short=$(queue | while read -r r _ name; do [ "$r" = "$row" ] && ! recorded "$L" "$name" && echo "$name"; done)
        [ -z "$short" ] || { say "${L##*/}: not done, short of $(echo $short)"; continue; }
        (cd "$BASELINES" && PYTHONPATH=$LOOP uv run --no-sync python "$LOOP/frozen.py" summary --run "$L") \
            >> "$L/frozen/frozen.log" 2>&1 && touch "$L/frozen/done" && say "${L##*/}: every play recorded, its summary written"
    done
    say "every play taken"
}
