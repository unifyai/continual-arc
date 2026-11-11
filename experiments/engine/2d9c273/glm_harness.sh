#!/usr/bin/env bash
# glm_harness.sh JOBS: the plays of Qwen in GLM's harness (glm_harness.py's docstring) that JOBS lists, a line each,
# "RUN SOURCE ROW NAME FIRST|TEST", SOURCE - for a wiped row (glm_harness.py plan), on a clone of REF_TESTS_NODE, the
# machine every reference row's tests played on, so that each play runs on the code and environment GLM's and Qwen's
# rows and their resets did: CODE (~/loop-turns), whose reference.py plays it, BASELINES (~/cab-turns), whose venv and
# .env run it, and CARC (~/benchmark-turns), the benchmark that venv holds. Up to PAR (7) play at once, started GAP (420)
# seconds apart in the order given, since plays started together meet the model's providers' 429s for their first half
# hour (extra-tests.sh). Each play is a run of its own under the supervisor (keep.sh), its folder RUN/frozen/NAME, tried
# up to three times; it is refused without a PURPOSE. Once both of a run's stream parts are recorded here,
# glm_harness.py view writes RUN/play and RUN/play.done; once every play JOBS lists for a run is recorded (and its view
# written, where it lists the stream), RUN/frozen/summary.json and RUN/frozen/done. A run's stream and its tests may
# play on different machines, each from its own JOBS. Resumable: a play whose round is recorded is not played again.
# Each RUN holds GLM's source in RUN/source/play before it starts (the laptop stages it, glm_harness.py's docstring).
# Stop it with its whole process group (kill -- -PGID). Its log is glm-harness.log beside it, and each run's own
# progress is RUN/progress.
# Usage (on the clone): (PURPOSE="..." OWNER=... setsid nohup bash ~/glm-harness/glm_harness.sh ~/glm-harness/jobs \
#     > /dev/null 2>&1 < /dev/null &)
set -uo pipefail
JOBS=$1
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$HOME/loop-turns}
BASELINES=${BASELINES:-$HOME/cab-turns}
CARC=${CARC:-$HOME/benchmark-turns}
PAR=${PAR:-7}
GAP=${GAP:-420}
. "$T/keep.sh" || exit 1
keep_purpose || exit 1
export PATH=$HOME/.local/bin:$PATH
say() { echo "$(date -u +%m-%d\ %H:%M) glm_harness.sh: $*" >> "$T/glm-harness.log"; }
note() { echo "$(date -u +%m-%d\ %H:%M) glm_harness.sh: $2" >> "$1/progress"; say "${1##*/}: $2"; }
py() { (cd "$BASELINES" && PYTHONPATH=$CODE uv run --no-sync --env-file .env python "$T/glm_harness.py" "$@"); }
recorded() {   # recorded RUN NAME: the play's records hold its whole round (100 instances)
    [ -f "$1/frozen/$2/play/records.jsonl" ] && [ "$(wc -l < "$1/frozen/$2/play/records.jsonl")" -ge 100 ]
}
take() {   # take RUN SOURCE ROW NAME FIRST|TEST
    local run=$1 source=$2 row=$3 name=$4 which=$5 out=$1/frozen i opt
    case $which in [0-9]*) opt=(--first "$which") ;; *) opt=(--test "$which") ;; esac
    [ "$source" = - ] || opt+=(--source "$source")
    recorded "$run" "$name" && { note "$run" "$name is recorded already"; return 0; }
    mkdir -p "$out"
    note "$run" "$name started (${opt[*]})"
    for i in 1 2 3; do
        (cd "$BASELINES" && export PYTHONPATH=$CODE && keep_exec "$out/$name" --resume --code "$T" --also "$CODE" \
            --also "$BASELINES" --also "$CARC" -- uv run --no-sync --env-file .env python "$T/glm_harness.py" play \
            --run "$run" --row "$row" --name "$name" "${opt[@]}") >> "$out/$name.log" 2>&1 \
            && { note "$run" "$name played: $(tail -1 "$out/$name.log" | cut -c1-200)"; return 0; }
        note "$run" "$name stopped short, attempt $i ($out/$name.log)"
        # a play cut short before its round was recorded: the attempt's calls and logs kept beside the play, its users'
        # working trees and store gone, and the next try seeds it afresh; one recorded only finishes (glm_harness.py)
        if [ -d "$out/$name/play" ] && ! recorded "$run" "$name"; then
            rm -rf "$out/$name/play/objects.git" "$out/$name"/play/users/*/system/state
            mv "$out/$name/play" "$out/$name/attempt-$i"
        fi
    done
    note "$run" "$name not played in three tries; left for a person"
    return 1
}
finish() {   # finish RUN: its stream's view once both parts are recorded here, its summary and done once every play
    # JOBS lists for it is, one process at a time
    exec 9> "$T/${1##*/}.lock" && flock 9
    local run=$1 name names
    names=$(awk -v r="$run" '$1 == r {print $4}' "$JOBS")
    if [ ! -f "$run/play.done" ] && recorded "$run" stream-000 && recorded "$run" stream-100; then
        py view --run "$run" >> "$run/frozen/view.log" 2>&1 && note "$run" "play/ written: $(tail -1 "$run/frozen/view.log")"
    fi
    for name in $names; do recorded "$run" "$name" || return 0; done
    case " $(echo $names) " in *" stream-"*) [ -f "$run/play.done" ] || return 0 ;; esac
    py summary --run "$run" >> "$run/frozen/frozen.log" 2>&1 && touch "$run/frozen/done" && note "$run" "done"
}
{   # read whole before it runs: bash reads a script as it goes, and the code may be synced again while this runs
    say "started: $(grep -c . "$JOBS") plays from $JOBS, $PAR at once, $GAP s apart, on $CODE, $BASELINES and $CARC"
    while read -r run source row name which; do
        [ -n "$run" ] || continue
        while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
        { take "$run" "$source" "$row" "$name" "$which"; (finish "$run"); } < /dev/null &
        sleep "$GAP"
    done < "$JOBS"
    wait
    for run in $(awk '{print $1}' "$JOBS" | sort -u); do (finish "$run"); done
    say "every play of $JOBS taken"
}
