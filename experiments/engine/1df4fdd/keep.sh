#!/usr/bin/env bash
# keep.sh: every run this repository starts leaves one folder holding everything it produced and its whole record, for
# good (Experiments Keep Everything, ~/global-agent-rules/rules/experiments-keep-everything.md), through
# continual-arc-research's run supervisor, continual_arc_fleet/runs.py, whose docstring is the manual. A run's folder
# is its own under ~/runs (~/runs/rt/NAME, ~/runs/paper/ROW, a test's frozen/ inside its run's), and the run is that
# folder's path under ~/runs, so that its place in the bucket, gs://DATA_BUCKET/runs/continual-arc-training/<run>/,
# is its place on the machine, a run inside another's folder included: brain's run store copies every folder holding a
# manifest.json there from a routed machine. The supervisor refuses a run that would not leave its folder: no PURPOSE
# (at least 6 words and 40 characters, as detailed as a lease's), no record of the code's commit (a git checkout, or the
# SYNCED.json that `runs.py synced` writes beside a copy with no history), or a folder that already holds a run's
# manifest when the launcher does not resume. Otherwise it writes manifest.json, code.diff, packages.txt and
# launch.json, runs the command with CARC_RUN_DIR naming the folder and its console whole in console.log (and on
# stdout, as it was), and records its end, its cost and what the folder holds; a run that kept no *.jsonl is failed.
#
# On a machine, a launcher sources this file and names its folder:
#   keep_run DIR [OPTION...] -- "$@"
# Unless CARC_RUN_DIR is set (the launcher already runs under the supervisor: its own, or that of a run it is a part of,
# as scratch.sh's frozen.sh and grid-vals.sh are, whose outputs scratch.sh keeps in its folder), it runs the launcher
# again under the supervisor, bash "$0" "$@" in the folder it was started from, in this process (exec: the launcher's
# pid is the supervisor's, which passes a signal on to the launcher). A driver runs one command of its as a run with
#   keep_exec DIR [OPTION...] -- COMMAND...
# or, from Python or any shell, bash keep.sh DIR [OPTION...] -- COMMAND.... OPTIONs:
#   --resume       the launcher takes a run up where it stopped: a second launch goes on in the same folder, and one
#                  without PURPOSE keeps the purpose the folder's manifest holds (brain's paper-restart passes none);
#                  a folder that holds a run begun before runs kept a manifest is taken up with a purpose saying none
#                  was given
#   --log FILE     a refusal is written there too (the launcher's progress), since a detached launcher's stderr is lost
#   --vars "N..."  those variables' values kept in the manifest's config
#   --code DIR     the code whose record the run keeps, and the supervisor's folder (this file's folder unless given)
#   --also DIR     another tree the run's code comes from: its commit (SYNCED.json, or git) kept in the config
# A launcher that also writes outside its folder (grid-exp.sh's ~/exp/NAME) gathers those folders into it as it ends,
# when it is the run itself: keep_own DIR && trap 'keep_gather DIR NAME SRC...' EXIT.
#
# On a laptop, a launcher that starts a run on a machine sources it too:
#   keep_purpose                    refuses, as the supervisor would, unless PURPOSE says why the run is made
#   keep_env                        PURPOSE and OWNER as words for the machine's command line
#   keep_stage DIR CHECKOUT [REF]   into DIR, for a copy of CHECKOUT's code on a machine: SYNCED.json and SYNCED.diff
#                                   (of REF's tree when given, as git archive sends it, else of the working tree as it
#                                   is copied) and runs.py, and for a working-tree copy keep.sh
# The supervisor is the first runs.py with `launch` of: CARC_RUNS_PY; CARC_BASELINES/src/continual_arc_fleet/runs.py
# (continual-arc-research's checkout, ~/continual-arc-baselines unless set: a laptop's, a research node's); this file's
# folder (a tools folder keep_stage made); ~/run-supervisor/runs.py (where deploy-code.sh and refs-deploy.sh put it, on
# a node whose ~/continual-arc-baselines is the baselines' turns tree).

KEEP_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KEEP_REPO=continual-arc-training

keep_runs_py() {   # the supervisor's path, or nothing and 1
    local c
    for c in "${CARC_RUNS_PY:-}" "${CARC_BASELINES:-$HOME/continual-arc-baselines}/src/continual_arc_fleet/runs.py" \
             "$KEEP_HERE/runs.py" "$HOME/run-supervisor/runs.py"; do
        [ -n "$c" ] && [ -f "$c" ] && grep -q "def launch_from" "$c" && { echo "$c"; return 0; }
    done
    return 1
}

keep_say() {   # keep_say LOG MESSAGE: on stderr, and in LOG when one is given
    echo "$2" >&2
    [ -n "$1" ] && { mkdir -p "$(dirname "$1")" && echo "$(date -u +%m-%d\ %H:%M) keep.sh: $2" >> "$1"; } 2>/dev/null
    return 0
}

keep_commit() {   # keep_commit DIR: the commit of the code in DIR, from its SYNCED.json or git
    if [ -f "$1/SYNCED.json" ]; then
        python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d.get('commit', '') + ('-dirty' if d.get('dirty') else ''))" "$1/SYNCED.json"
    else
        git -C "$1" rev-parse HEAD 2>/dev/null || echo "no record"
    fi
}

keep_exec() {   # keep_exec DIR [OPTION...] -- COMMAND...: COMMAND as the run in DIR, in this process
    local dir=$1 log="" code=$KEEP_HERE resume=0 runs purpose run why v d
    local flags=()
    shift
    while [ $# -gt 0 ] && [ "$1" != -- ]; do
        case $1 in
            --resume) resume=1 ;;
            --log) log=$2; shift ;;
            --code) code=$2; shift ;;
            --vars) for v in $2; do flags+=("--config=$v=${!v:-}"); done; shift ;;
            --also) d=$2; flags+=("--config=code:$d=$(keep_commit "$d")"); shift ;;
            *) keep_say "$log" "keep.sh: unknown option $1"; return 2 ;;
        esac
        shift
    done
    [ "${1:-}" = -- ] && shift
    [ $# -gt 0 ] || { keep_say "$log" "keep.sh: no command"; return 2; }
    if [ -n "${CARC_RUN_DIR:-}" ]; then exec "$@"; fi
    case $dir in /*) ;; *) dir=$PWD/$dir ;; esac
    case $dir in
        "$HOME"/runs/?*) run=${dir#"$HOME"/runs/}; run=${run%/} ;;
        *) keep_say "$log" "refused: a run's folder is under ~/runs, where brain's run store finds it, not $dir"; return 1 ;;
    esac
    runs=$(keep_runs_py) || {
        keep_say "$log" "refused: no run supervisor (continual-arc-research's runs.py with launch): sync it to this machine, or set CARC_RUNS_PY"
        return 1; }
    purpose=${PURPOSE:-}
    if [ -z "$purpose" ] && [ "$resume" = 1 ] && [ -f "$dir/manifest.json" ]; then
        purpose=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('purpose', ''))" "$dir/manifest.json")
    elif [ -z "$purpose" ] && [ "$resume" = 1 ] && [ -d "$dir" ] \
            && [ -n "$(find "$dir" -mindepth 1 -maxdepth 1 ! -name progress -print -quit 2>/dev/null)" ]; then
        # a run that began before runs kept a manifest, taken up again without PURPOSE (brain's paper-restart): its
        # record from now on says so rather than refuse it, since no reason was ever recorded for it to keep
        purpose="not given: $run began before its runs kept a manifest, and ${KEEP_BY:-$(basename "$0")} on $(hostname) took it up again without PURPOSE"
    fi
    [ "$resume" = 1 ] && flags+=(--resume)
    # the command runs in the folder the launcher was started from; the supervisor's own is the code's
    flags=("--repo=$KEEP_REPO" "--run=$run" "--dir=$dir" "--cwd=$code" "--purpose=$purpose" --here
           "--launched-by=${KEEP_BY:-$(basename "$0")}" ${flags[@]+"${flags[@]}"}
           -- bash -c 'cd -- "$1" && shift && exec "$@"' keep.sh "$PWD" "$@")
    if ! why=$(python3 "$runs" launch --check "${flags[@]}" 2>&1 >/dev/null); then
        keep_say "$log" "refused, the run would not leave its whole folder ($dir): $(echo "$why" | paste -sd ';' -)"
        return 1
    fi
    exec python3 "$runs" launch "${flags[@]}"
}

keep_run() {   # keep_run DIR [OPTION...] -- "$@": the launcher itself again, as the run in DIR, unless it is one already
    [ -n "${CARC_RUN_DIR:-}" ] && return 0
    local dir=$1 script=$0 opts=()
    shift
    while [ $# -gt 0 ] && [ "$1" != -- ]; do opts+=("$1"); shift; done
    [ "${1:-}" = -- ] && shift
    case $script in /*) ;; *) script=$PWD/$script ;; esac
    KEEP_BY=$(basename "$script") keep_exec "$dir" ${opts[@]+"${opts[@]}"} -- bash "$script" "$@"
}

keep_purpose() {   # refuses unless PURPOSE says why the run is made, by the supervisor's own rule (its --check)
    local runs tmp why
    runs=$(keep_runs_py) || { echo "refused: no run supervisor; set CARC_BASELINES to continual-arc-research's checkout" >&2; return 1; }
    tmp=$(mktemp -d)
    why=$(python3 "$runs" launch --check "--repo=$KEEP_REPO" --run=purpose "--dir=$tmp/run" "--cwd=$KEEP_HERE" \
          "--purpose=${PURPOSE:-}" -- true 2>&1 >/dev/null | grep "purpose")
    rm -rf "$tmp"
    [ -z "$why" ] || { echo "refused: set PURPOSE. $why" >&2; return 1; }
}

keep_env() {   # PURPOSE=... OWNER=..., quoted for a remote shell's command line
    printf 'PURPOSE=%q OWNER=%q' "${PURPOSE:-}" "${OWNER:-${USER:-unknown}}"
}

keep_stage() {   # keep_stage DIR CHECKOUT [REF]
    local out=$1 checkout=$2 ref=${3:-} runs tmp rc=0
    runs=$(keep_runs_py) || { echo "no run supervisor; set CARC_BASELINES to continual-arc-research's checkout" >&2; return 1; }
    mkdir -p "$out" && cp "$runs" "$out/runs.py" || return 1
    if [ -n "$ref" ]; then
        tmp=$(mktemp -d)
        git -C "$checkout" worktree add -q --detach "$tmp/tree" "$ref" && python3 "$runs" synced "$tmp/tree" "$out" || rc=1
        git -C "$checkout" worktree remove --force "$tmp/tree" 2>/dev/null; rm -rf "$tmp"
        return $rc
    fi
    cp "$KEEP_HERE/keep.sh" "$out/keep.sh" && python3 "$runs" synced "$checkout" "$out"
}

keep_own() {   # keep_own DIR: whether this process runs as the run in DIR itself, not as a part of another run
    [ -n "${CARC_RUN_DIR:-}" ] && [ -d "$1" ] && [ "$(cd "$CARC_RUN_DIR" && pwd -P)" = "$(cd "$1" && pwd -P)" ]
}

keep_gather() {   # keep_gather DIR NAME SRC [NAME SRC...]: each SRC, a folder a run writes outside its own, into DIR/NAME,
    # as hard links (no disk spent, every file whole), afresh each time
    local dir=$1; shift
    while [ $# -ge 2 ]; do
        if [ -d "$2" ]; then rm -rf "${dir:?}/$1" && { cp -al "$2" "$dir/$1" 2>/dev/null || cp -a "$2" "$dir/$1"; }; fi
        shift 2
    done
    return 0
}

# as a command: bash keep.sh DIR [OPTION...] -- COMMAND...
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    keep_exec "$@"
    exit $?
fi
