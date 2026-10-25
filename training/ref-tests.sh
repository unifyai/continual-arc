#!/usr/bin/env bash
# ref-tests.sh HOST...: every reference row's two frozen tests (frozen.py's docstring), played on this machine, a worker
# with the reference machines' own code and environment (a copy of one of them, with the memory a test needs: its 100
# harness users take about 50 GB) and an ssh key every HOST takes. Each HOST is a reference machine's ssh destination
# (user@host or a Host of the ssh configuration); there is no default. Every POLL seconds (600) it looks on each HOST for
# rows whose stream has ended (reference.sh's RUNS/paper/ROW/play.done) and whose tests are not there
# (ROW/frozen/done), and takes up to PAR of them at once (3), each in a process of its own:
#   copy  the row's memory, to the same path here, so its absolute paths are the ones frozen.py renames: its play
#         folder's alias-salt, schedule.json, driver.json and records.jsonl, its users' harness states and, in a row
#         that commits them every round, their histories (state.git and state-log.jsonl, which frozen.py forks), and a
#         bare row's library/ and demo-cache/ (never its rounds' calls), mirrored exactly
#   play  both tests, one after the other, each up to three times (frozen.py play --player, the row's own player), into
#         ROW/frozen/TEST/play, where the paper's figures read them; a wiped row or a floor at one user (h0, hpl0, m0,
#         f) both again from an empty state, into TEST-reference, the second, independent play the paper reads every
#         row's noise from; GLM's row h at 1, 2 and 4 users again with Qwen playing from GLM's memory, into TEST-qwen
#         (Qwen in GLM's harness)
#   back  ROW/frozen into the row's folder on its HOST, its users' copies of the memory left out (every call, turn and
#         game the test played stays), then ROW/frozen/done
# then deletes its own copy. A row whose tests fail two passes in a row is left for a person, said so in the log.
# Resumable: a play whose round is recorded here is not played again, and a row with its frozen/done on its HOST is done.
# Stop it with its whole process group (kill -- -PGID), so no play of a row it took goes on beside the next instance's.
# CODE is the training code the rows played with (this folder by default) and BASELINES the baselines checkout whose
# venv and .env (OPENROUTER_API_KEY) they played from (default ../baselines beside CODE); RUNS (default ~/runs) is the
# runs folder, the same path here and on every HOST. Its log is ref-tests.log beside this script.
# Usage (on the worker): (setsid nohup bash ~/continual-arc/training/ref-tests.sh HOST... > /dev/null 2>&1 < /dev/null &)
set -uo pipefail
HOSTS=("$@")
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$T} RUNS=${RUNS:-$HOME/runs}
BASELINES=${BASELINES:-$(dirname "$CODE")/baselines}
[ "${#HOSTS[@]}" -gt 0 ] || { echo "usage: ref-tests.sh HOST..." >&2; exit 2; }
POLL=${POLL:-600}
PAR=${PAR:-3}
export PATH=$HOME/.local/bin:$PATH
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30)
say() { echo "$(date -u +%m-%d\ %H:%M) ref-tests.sh: $*" >> "$T/ref-tests.log"; }
# finished HOST: its rows whose stream has ended and whose tests are not there, the earliest ended first; a folder set
# aside under a row's name with a suffix (qwen-h-u1.old-charge) is no row, nor one with no stream (a play of the
# held-out instances alone, which is a reference)
finished() {
    "${SSH[@]}" "$1" "cd $RUNS/paper 2>/dev/null"' && for r in *; do case $r in *.*) continue ;; esac
        [ -d "$r/play" ] && [ -f "$r/play.done" ] && [ ! -f "$r/frozen/done" ] && echo "$(stat -c %Y "$r/play.done") $r"
        done | sort -n | cut -d" " -f2'
}
copy() {   # copy HOST ROW
    mkdir -p "$RUNS/paper/$2/play"
    rsync -a --delete -e "${SSH[*]}" --include=/alias-salt --include=/schedule.json --include=/driver.json \
        --include=/records.jsonl --include=/users/ --include='/users/*/' --include='/users/*/system/' \
        --include='/users/*/system/state/***' --include='/users/*/system/state.git/***' \
        --include='/users/*/system/state-log.jsonl' --include='/library/***' --include='/demo-cache/***' --exclude='*' \
        "$1:$RUNS/paper/$2/play/" "$RUNS/paper/$2/play/"
}
play() {   # play ROW TEST NAME FROZEN.PY-OPTIONS...: one play into RUNS/paper/ROW/frozen/NAME/play
    local row=$1 t=$2 name=$3 i out=$RUNS/paper/$1/frozen
    shift 3
    mkdir -p "$out"
    for i in 1 2 3; do
        (cd "$BASELINES" && PYTHONPATH=$CODE uv run --no-sync --env-file .env python "$T/frozen.py" play \
            --run "$RUNS/paper/$row" --test "$t" --schedule "$T/schedules/arc1-small-$t-test-seed0.json" \
            --name "$name" "$@") >> "$out/$name.log" 2>&1 && return 0
        say "$row: $name stopped short, attempt $i ($out/$name.log)"
    done
    return 1
}
back() {   # back HOST ROW: its frozen/, the users' copies of the memory left out
    rsync -a -e "${SSH[*]}" --exclude='/*/play/users/*/system/state/' "$RUNS/paper/$2/frozen/" "$1:$RUNS/paper/$2/frozen/"
}
take() {   # take HOST ROW: copy its memory, play its tests, copy them back
    local host=$1 row=$2 player kind users spec t name options ok=1 plays=()   # plays: name|frozen.py options
    say "$row on $host: copying its memory"
    copy "$host" "$row" || { say "$row: the copy failed"; return; }
    read -r player kind users < <(python3 -c "import json, sys; d = json.load(open(sys.argv[1]))
print(d['player'], d['row'], json.load(open(sys.argv[2]))['users'])" "$RUNS/paper/$row/play/driver.json" \
        "$RUNS/paper/$row/play/schedule.json") || { say "$row: no driver.json"; return; }
    for t in held-out seen; do
        plays+=("$t|$t|--player $player")
        case $kind:$users in h0:1 | hpl0:1 | m0:1 | f:1) plays+=("$t|$t-reference|--player $player --reference") ;; esac
        case $row in glm-h-u[124]) plays+=("$t|$t-qwen|--player qwen") ;; esac
    done
    for spec in "${plays[@]}"; do
        IFS='|' read -r t name options <<< "$spec"
        say "$row: $name ($options)"
        play "$row" "$t" "$name" $options || { ok=0; continue; }
        say "$row: $name played: $(tail -1 "$RUNS/paper/$row/frozen/$name.log" | cut -c1-200)"
    done
    (cd "$BASELINES" && PYTHONPATH=$CODE uv run --no-sync python "$T/frozen.py" summary --run "$RUNS/paper/$row") \
        >> "$RUNS/paper/$row/frozen/frozen.log" 2>&1
    back "$host" "$row" || ok=0
    if [ "$ok" = 1 ] && "${SSH[@]}" "$host" "touch $RUNS/paper/$row/frozen/done"; then
        say "$row: done, its tests in its folder on $host"
        rm -rf "${RUNS:?}/paper/$row"
    else
        mkdir -p "$T/failed"; echo $(( $(cat "$T/failed/$row" 2>/dev/null || echo 0) + 1 )) > "$T/failed/$row"
        say "$row: not done ($(cat "$T/failed/$row") of 2 passes)"
    fi
}
{   # read whole before it runs: bash reads a script as it goes
    say "started: the reference rows' frozen tests on ${HOSTS[*]}, every ${POLL}s, $PAR at once"
    declare -A taking   # row: the pid of the process playing it
    while true; do
        for host in "${HOSTS[@]}"; do
            for row in $(finished "$host"); do
                [ "$(cat "$T/failed/$row" 2>/dev/null || echo 0)" -ge 2 ] && continue
                [ -n "${taking[$row]:-}" ] && kill -0 "${taking[$row]}" 2>/dev/null && continue
                while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
                # a row whose process ended while this one waited for room has its frozen/done by now
                "${SSH[@]}" "$host" "test ! -f $RUNS/paper/$row/frozen/done" || continue
                take "$host" "$row" &
                taking[$row]=$!
            done
        done
        sleep "$POLL"
    done
}
