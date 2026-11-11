#!/usr/bin/env bash
# ref-tests.sh HOST...: every reference row's two frozen tests (frozen.py's docstring), played on this machine, a worker
# with the code and environment the rows played on (a clone of a reference machine, with the memory a test needs: its
# 100 harness users take about 50 GB) and an ssh key every HOST takes. CODE (~/loop-turns) is this repository as the
# rows played it, whose frozen.py plays the tests and names each row's own (frozen.py schedule, on code from 28 Oct 2026
# on: a row whose users play tasks of their own has a seen test of its own) and whose modules it imports, and
# BASELINES (~/cab-turns) the baselines checkout whose venv runs it and whose .env holds the key: the final runs'
# trees (the author, 27 Oct 2026). ROWS
# (*-turns) is a pattern a row's name must match, so that rows played on other code are left to a worker of their own.
# Every POLL seconds (600) it looks on each HOST for such rows whose stream has ended (reference.sh's
# ~/runs/paper/ROW/play.done) and whose tests are not there (ROW/frozen/done), and takes up to PAR of them at once (3),
# each in a process of its own:
#   copy  the row's memory, to the same path here, so its absolute paths are the ones frozen.py renames: its play
#         folder's alias-salt, schedule.json, driver.json and records.jsonl, its users' harness states and, in a row
#         that commits them every round, their histories (state.git and state-log.jsonl, which frozen.py forks), and a
#         bare row's library/ and demo-cache/ (never its rounds' calls), mirrored exactly
#   play  both tests, one after the other, each up to three times (frozen.py play --player, the row's own player), into
#         ROW/frozen/TEST/play, where continual-arc-paper's registry reads them; a wiped row or a floor at one user
#         (h0, hpl0, m0, f) both again from an empty state, into TEST-reference, the second, independent play the
#         paper reads every row's noise from; GLM's row h at 1, 2 and 4 users again with Qwen playing from GLM's
#         memory, into TEST-qwen (Qwen in GLM's harness, the author, 20 Oct 2026)
#   back  ROW/frozen into the row's folder on its HOST, its users' copies of the memory left out (every call, turn and
#         game the test played stays), then ROW/frozen/done; the operator's hourly backup takes them from there
# then deletes its own copy. A row whose tests fail two passes in a row is left for a person, said so in the log.
# Resumable: a play whose round is recorded here is not played again, and a row with its frozen/done on its HOST is done.
# Stop it with its whole process group (kill -- -PGID), so no play of a row it took goes on beside the next instance's.
# Its log is ref-tests.log beside it, in the folder it was copied into to run (~/frozen-turns).
# Usage (on the worker): (setsid nohup bash ~/frozen-turns/ref-tests.sh NODE_USER@<ip> NODE_USER@<ip> \
#     NODE_USER@<ip> NODE_USER@<ip> > /dev/null 2>&1 < /dev/null &)
set -uo pipefail
HOSTS=("$@")
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$HOME/loop-turns}
BASELINES=${BASELINES:-$HOME/cab-turns}
ROWS=${ROWS:-*-turns}
POLL=${POLL:-600}
PAR=${PAR:-3}
export PATH=$HOME/.local/bin:$PATH
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30)
say() { echo "$(date -u +%m-%d\ %H:%M) ref-tests.sh: $*" >> "$T/ref-tests.log"; }
# finished HOST: its rows whose stream has ended and whose tests are not there, the earliest ended first; a folder set
# aside under a row's name with a suffix (qwen-h-u1.old-charge) is no row, nor one with no stream (glm_refs.sh's plays
# of the held-out instances alone, which are references)
finished() {
    "${SSH[@]}" "$1" 'cd ~/runs/paper 2>/dev/null && for r in *; do case $r in *.*) continue ;; esac
        [ -d "$r/play" ] && [ -f "$r/play.done" ] && [ ! -f "$r/frozen/done" ] && echo "$(stat -c %Y "$r/play.done") $r"
        done | sort -n | cut -d" " -f2'
}
copy() {   # copy HOST ROW
    mkdir -p ~/runs/paper/"$2"/play
    rsync -a --delete -e "${SSH[*]}" --include=/alias-salt --include=/schedule.json --include=/driver.json \
        --include=/records.jsonl --include=/users/ --include='/users/*/' --include='/users/*/system/' \
        --include='/users/*/system/state/***' --include='/users/*/system/state.git/***' \
        --include='/users/*/system/state-log.jsonl' --include='/library/***' --include='/demo-cache/***' --exclude='*' \
        "$1:runs/paper/$2/play/" ~/runs/paper/"$2"/play/
}
play() {   # play ROW TEST NAME FROZEN.PY-OPTIONS...: one play into ~/runs/paper/ROW/frozen/NAME/play
    local row=$1 t=$2 name=$3 i out=~/runs/paper/$1/frozen sched
    shift 3
    mkdir -p "$out"
    # the row's own test (frozen.py schedule): its seen test is of the tasks its users play
    sched=$(python3 "$CODE/frozen.py" schedule --run ~/runs/paper/"$row" --test "$t") \
        || { say "$row: its $t test's schedule is not known"; return 1; }
    for i in 1 2 3; do
        (cd "$BASELINES" && PYTHONPATH=$CODE uv run --no-sync --env-file .env python "$CODE/frozen.py" play \
            --run ~/runs/paper/"$row" --test "$t" --schedule "$sched" \
            --name "$name" "$@") >> "$out/$name.log" 2>&1 && return 0
        say "$row: $name stopped short, attempt $i ($out/$name.log)"
    done
    return 1
}
back() {   # back HOST ROW: its frozen/, the users' copies of the memory left out
    rsync -a -e "${SSH[*]}" --exclude='/*/play/users/*/system/state/' ~/runs/paper/"$2"/frozen/ "$1:runs/paper/$2/frozen/"
}
take() {   # take HOST ROW: copy its memory, play its tests, copy them back
    local host=$1 row=$2 player kind users spec t name options ok=1 plays=()   # plays: name|frozen.py options
    say "$row on $host: copying its memory"
    copy "$host" "$row" || { say "$row: the copy failed"; return; }
    read -r player kind users < <(python3 -c "import json, sys; d = json.load(open(sys.argv[1]))
print(d['player'], d['row'], json.load(open(sys.argv[2]))['users'])" ~/runs/paper/"$row"/play/driver.json \
        ~/runs/paper/"$row"/play/schedule.json) || { say "$row: no driver.json"; return; }
    for t in held-out seen; do
        plays+=("$t|$t|--player $player")
        case $kind:$users in h0:1 | hpl0:1 | m0:1 | f:1) plays+=("$t|$t-reference|--player $player --reference") ;; esac
        case $player:$kind in glm:h) plays+=("$t|$t-qwen|--player qwen") ;; esac
    done
    for spec in "${plays[@]}"; do
        IFS='|' read -r t name options <<< "$spec"
        say "$row: $name ($options)"
        play "$row" "$t" "$name" $options || { ok=0; continue; }
        say "$row: $name played: $(tail -1 ~/runs/paper/"$row"/frozen/"$name".log | cut -c1-200)"
    done
    (cd "$BASELINES" && PYTHONPATH=$CODE uv run --no-sync python "$CODE/frozen.py" summary --run ~/runs/paper/"$row") \
        >> ~/runs/paper/"$row"/frozen/frozen.log 2>&1
    back "$host" "$row" || ok=0
    if [ "$ok" = 1 ] && "${SSH[@]}" "$host" "touch runs/paper/$row/frozen/done"; then
        say "$row: done, its tests in its folder on $host"
        rm -rf ~/runs/paper/"$row"
    else
        mkdir -p "$T/failed"; echo $(( $(cat "$T/failed/$row" 2>/dev/null || echo 0) + 1 )) > "$T/failed/$row"
        say "$row: not done ($(cat "$T/failed/$row") of 2 passes)"
    fi
}
{   # read whole before it runs: bash reads a script as it goes
    say "started: the frozen tests of the rows $ROWS on ${HOSTS[*]}, every ${POLL}s, $PAR at once, on $CODE and $BASELINES"
    declare -A taking   # row: the pid of the process playing it
    while true; do
        for host in "${HOSTS[@]}"; do
            for row in $(finished "$host"); do
                [[ $row == $ROWS ]] || continue
                [ "$(cat "$T/failed/$row" 2>/dev/null || echo 0)" -ge 2 ] && continue
                [ -n "${taking[$row]:-}" ] && kill -0 "${taking[$row]}" 2>/dev/null && continue
                while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
                # a row whose process ended while this one waited for room has its frozen/done by now
                "${SSH[@]}" "$host" "test ! -f runs/paper/$row/frozen/done" || continue
                take "$host" "$row" &
                taking[$row]=$!
            done
        done
        sleep "$POLL"
    done
}
