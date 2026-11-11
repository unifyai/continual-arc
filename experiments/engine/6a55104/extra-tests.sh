#!/usr/bin/env bash
# extra-tests.sh JOBS: more plays of the reference rows' frozen tests (frozen.py's docstring), so that a row's T reads
# the mean of several plays of each test against the mean of several plays of its reset (continual-arc-paper's
# scripts/make_figures.py; the author, 26 Oct 2026). It runs on a clone of REF_TESTS_NODE, the machine every reference row's tests played on, so that each
# extra play is played by the same code and environment as the row's first, as ref-tests.sh takes them: CODE
# (~/loop-turns), whose frozen.py plays it, and BASELINES (~/cab-turns), whose venv and .env run it. JOBS holds one
# play a line, "HOST ROW TEST NAME [--reference]", started in the order given (every held-out play first) and at least
# GAP (5) seconds apart; up to PAR (3) play at once, each in a process of its own. Plays started together open with a
# burst of calls that the model's providers meet with 429s until they add capacity, about half an hour later, which
# slows them and under a budget in seconds cost them score, so many plays start GAP apart (fourteen Qwen3.5-9B plays
# started together on 26 Oct 2026 met 1,788 to 4,144 429s each, most in their first half hour):
#   copy  the row's memory from HOST to the row's own path here, once for all its plays, as ref-tests.sh copies it
#   play  frozen.py play --name NAME into ROW/frozen/NAME/play with the row's own player, from its end-of-stream state
#         or, with --reference, from an empty state, up to three tries, as ref-tests.sh plays
#   back  ROW/frozen/NAME and NAME.log into the row's frozen/ on HOST, its users' copies of the memory left out, where
#         the operator's hourly backup finds them; then this machine's copy of the play goes. The row's frozen/done
#         stays on HOST throughout, so ref-tests.sh never takes the row up again
# A play whose records are whole on HOST is not played again, so a run again takes up where it stopped. Stop it with
# its whole process group (kill -- -PGID). Its log is extra-tests.log beside it, in the folder it was copied into.
# Usage (on the clone): (setsid nohup bash ~/frozen-turns/extra-tests.sh ~/frozen-turns/jobs > /dev/null 2>&1 < /dev/null &)
set -uo pipefail
JOBS=$1
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$HOME/loop-turns}
BASELINES=${BASELINES:-$HOME/cab-turns}
PAR=${PAR:-3}
GAP=${GAP:-5}
export PATH=$HOME/.local/bin:$PATH
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30)
say() { echo "$(date -u +%m-%d\ %H:%M) extra-tests.sh: $*" >> "$T/extra-tests.log"; }
whole() {   # whole HOST ROW TEST NAME: the play's records on HOST hold every instance of the test
    local want   # every test is 100 instances, a row's own seen test as many as arc1-small's
    want=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['slots']))" \
        "$CODE/schedules/arc1-small-$3-test-seed0.json")
    [ "$("${SSH[@]}" "$1" "wc -l < runs/paper/$2/frozen/$4/play/records.jsonl 2>/dev/null || echo 0")" -ge "$want" ]
}
copy() {   # copy HOST ROW: the row's memory, once (ref-tests.sh's copy), under a lock its other plays here wait on
    (
        flock 9
        [ -f ~/runs/paper/"$2"/.copied ] && exit 0
        mkdir -p ~/runs/paper/"$2"/play
        rsync -a --delete -e "${SSH[*]}" --include=/alias-salt --include=/schedule.json --include=/driver.json \
            --include=/records.jsonl --include=/users/ --include='/users/*/' --include='/users/*/system/' \
            --include='/users/*/system/state/***' --include='/users/*/system/state.git/***' \
            --include='/users/*/system/state-log.jsonl' --include='/library/***' --include='/demo-cache/***' \
            --exclude='*' "$1:runs/paper/$2/play/" ~/runs/paper/"$2"/play/ && touch ~/runs/paper/"$2"/.copied
    ) 9> ~/runs/paper/"$2".lock
}
take() {   # take HOST ROW TEST NAME [--reference]
    local host=$1 row=$2 t=$3 name=$4 ref=${5:-} out=~/runs/paper/$2/frozen player sched i ok=0
    whole "$host" "$row" "$t" "$name" && { say "$row: $name is whole on $host already"; return; }
    "${SSH[@]}" "$host" "test -f runs/paper/$row/frozen/done" || { say "$row: its first tests are not done on $host"; return; }
    copy "$host" "$row" || { say "$row: the copy failed"; return; }
    player=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['player'])" ~/runs/paper/"$row"/play/driver.json)
    # the row's own test (frozen.py schedule): its seen test is of the tasks its users play
    sched=$(python3 "$CODE/frozen.py" schedule --run ~/runs/paper/"$row" --test "$t") \
        || { say "$row: its $t test's schedule is not known"; return; }
    mkdir -p "$out"
    say "$row: $name (--player $player $ref)"
    for i in 1 2 3; do
        (cd "$BASELINES" && PYTHONPATH=$CODE uv run --no-sync --env-file .env python "$CODE/frozen.py" play \
            --run ~/runs/paper/"$row" --test "$t" --schedule "$sched" \
            --name "$name" --player "$player" $ref) >> "$out/$name.log" 2>&1 && { ok=1; break; }
        say "$row: $name stopped short, attempt $i ($out/$name.log)"
    done
    [ "$ok" = 1 ] || { say "$row: $name not played in three tries; left here in $out"; return; }
    say "$row: $name played: $(tail -1 "$out/$name.log" | cut -c1-200)"
    if rsync -a -e "${SSH[*]}" --exclude='/*/play/users/*/system/state/' "$out/$name" "$out/$name.log" "$host:runs/paper/$row/frozen/"; then
        rm -rf "${out:?}/$name"
        say "$row: $name in its folder on $host"
    else
        say "$row: copying $name back failed; left here in $out"
    fi
}
{   # read whole before it runs: bash reads a script as it goes
    say "started: $(grep -c . "$JOBS") plays from $JOBS, $PAR at once, $GAP s apart, on $CODE and $BASELINES"
    mkdir -p ~/runs/paper
    while read -r host row t name ref; do
        [ -n "$host" ] || continue
        while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
        # stdin from nothing: ssh reads its standard input, which would eat the rest of JOBS
        take "$host" "$row" "$t" "$name" $ref < /dev/null &
        sleep "$GAP"
    done < "$JOBS"
    wait
    say "every play of $JOBS taken"
}
