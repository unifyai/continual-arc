#!/usr/bin/env bash
# more-plays.sh JOBS: more plays of trained runs' frozen tests and of their resets' (frozen.py's docstring), several at
# once on one node, each test play on a vLLM server of its own, so that a run's plays K = 2, 3, 4 go side by side where
# frozen.sh plays them one after another: one frozen.sh holds a run's whole frozen/ folder, and its end deletes every
# play's test users' memory there, a play still going included. Each line of JOBS is one test play, started GAP (20)
# seconds after the one before:
#   own     RUN TEST NAME GPUS                      a trained run's play from its end-of-stream memory on its last
#                                                   phase's adapter (frozen.py model), served under the name its own
#                                                   tests used, RUN/frozen/NAME
#   reset   RUN TEST NAME GPUS SEEN_OF              row h0 from nothing on the base model (frozen.sh's ROW=h0), its seen
#                                                   test the run SEEN_OF's own, RUN/frozen/NAME
#   adapter RUN TEST NAME GPUS OUT ADAPTER SERVED   RUN's play from its end-of-stream memory on another adapter, served
#                                                   as SERVED, into OUT/frozen/NAME (frozen.sh's ADAPTER and SERVED)
# RUN, SEEN_OF and OUT are folders under ~/runs (rt/dagger-u4-disjoint, paper/qwen-h0-card-u2); TEST is held-out or
# seen; NAME the play's folder (held-out-2, seen-reference-3); GPUS the server's, one copy tensor-parallel over them, as
# frozen.sh serves each test on half the GPUs it is given. The server is frozen.sh's: serve.sh's SERVE_ARGS and
# LORA_ARGS, MAX_MODEL_LEN 131072 and GPU_MEM 0.85, on port 8120 plus its first GPU, the adapter loaded as frozen.sh
# loads it (loop.py vllm-adapter); a play whose server stops answering has it started again. The play is frozen.py play
# from FROZEN (the node's ~/frozen-tools, frozen.sh's own copy of it), from ~/prime-rl with PYPATH on the path (CODE,
# the node's ~/loop, and OVERRUN, the harness with the context-window fix, ahead of the installed package, as DAgger's
# tests have played since 4 Nov 2026: ~/overrun-fix/README.md), each play a run of its own under the supervisor
# (keep.sh), its folder the play's, with code.tgz, every tree it played on, beside it. A play is tried up to three times,
# each from the end-of-stream state afresh; once its records are whole its folder gets NAME.done beside it, its test
# users' copies of the memory go as frozen.sh's end deletes them, and its server stops. Then OUT/frozen/summary.json is
# not touched: these plays are read from their own folders (continual-arc-paper's make_figures.py).
# PURPOSE (why the plays are made, as detailed as a lease's) is required. Follow it in ~/runs/more-plays.log; each play's
# own log is RUN/frozen/NAME.log, and its folder's live.jsonl gets a row every two minutes (its state, the
# games still playing, its records and solves once whole), which brain's Experiments page streams. Resumable: run
# again, a play whose NAME.done exists is passed over; more-plays.sh JOBS --live writes the rows alone.
# Usage (on the node): cd ~ && PURPOSE="..." setsid -f nohup bash ~/more-plays/more-plays.sh ~/more-plays/jobs \
#   > /dev/null 2>&1 < /dev/null
set -uo pipefail
JOBS=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for k in "$T/keep.sh" "$HOME/loop/keep.sh"; do [ -f "$k" ] && . "$k" && break; done
type keep_exec > /dev/null 2>&1 || { echo "refused: no keep.sh beside $0, the run supervisor's helper" >&2; exit 1; }
keep_purpose || exit 1
CODE=${CODE:-$HOME/loop} FROZEN=${FROZEN:-$HOME/frozen-tools} OVERRUN=${OVERRUN:-$HOME/overrun-fix/src}
PYPATH=${PYPATH:-$CODE:$OVERRUN}
GAP=${GAP:-20} POLL=${POLL:-60}
MODEL=Qwen/Qwen3.5-9B
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 NCCL_NVLS_ENABLE=0
LOG=$HOME/runs/more-plays.log
mkdir -p "$HOME/runs"
say() { echo "$(date -u +%m-%d\ %H:%M) more-plays.sh: $*" >> "$LOG"; }
# serve.sh's launch for Qwen3.5 (its SERVE_ARGS and LORA_ARGS, one API frontend so that adapters load at run time),
# as frozen.sh passes it
SERVE_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --chat-template-content-format string --language-model-only --chat-template "$CODE/qwen35_prefill.jinja"
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-lora --max-lora-rank 128 --max-loras 12
    --api-server-count 1)
up() { for _ in 1 2 3; do curl -fsS -m 10 "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done; return 1; }
stop() {   # stop PORT: its server's process group, while its pid is still vLLM's
    local pid
    pid=$(cat "$HOME/runs/vllm-$1.pid" 2>/dev/null)
    [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
    return 0
}
idle() {   # every GPU of $1 with next to nothing on it
    local g
    for g in ${1//,/ }; do
        [ "$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] || return 1
    done
}
serve() {   # serve PORT GPUS SERVED VLLM_DIR LOGFILE: a server answering on PORT, the adapter (if any) loaded into it
    local port=$1 gpus=$2 served=$3 dir=$4 log=$5
    if ! up "$port"; then
        stop "$port"
        [ -f "$HOME/runs/vllm-$port.log" ] && mv "$HOME/runs/vllm-$port.log" "$HOME/runs/vllm-$port.$(date -u +%m%d-%H%M%S).log"
        (cd ~/continual-arc-baselines && VLLM_ALLOW_RUNTIME_LORA_UPDATING=True MAX_MODEL_LEN=131072 GPU_MEM=0.85 \
            bash scripts/vllm-serve.sh "$MODEL" "$gpus" "$port" "${SERVE_ARGS[@]}") >> "$log" 2>&1 || return 1
    fi
    [ -z "$dir" ] && return 0
    curl -fsS "localhost:$port/v1/models" | grep -q "\"$served\"" && return 0
    curl -fsS -X POST "localhost:$port/v1/load_lora_adapter" -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"$served\", \"lora_path\": \"$dir\"}" >> "$log" 2>&1
}
records() { [ -f "$1/play/records.jsonl" ] && wc -l < "$1/play/records.jsonl" || echo 0; }
play() {   # play KIND RUN TEST NAME GPUS [SEEN_OF | OUT ADAPTER SERVED]
    local kind=$1 run=$2 t=$3 name=$4 gpus=$5 r=$HOME/runs/$2 out who=() model=() sched of served adapter dir="" want i ok=0
    local port=$(( 8120 + ${gpus%%,*} ))
    out=$r/frozen
    case $kind in
        own) read -r served adapter < <(cd ~/prime-rl && PYTHONPATH=$PYPATH uv run --no-sync python "$FROZEN/frozen.py" \
                 model --run "$r" 2>> "$LOG")
             [ -n "${adapter:-}" ] || { say "$run $name: the run has no last phase to test"; return 1; }
             who=(--run "$r") model=(--served "$served" --adapter "$adapter") of=$r ;;
        reset) served=$MODEL who=(--row h0) model=(--served "$MODEL") of=$HOME/runs/$6 ;;
        adapter) out=$HOME/runs/$6/frozen served=$8 adapter=$7
             who=(--run "$r") model=(--served "$served" --adapter "$adapter") of=$r ;;
        *) say "$run $name: unknown kind $kind"; return 1 ;;
    esac
    mkdir -p "$out"
    [ -f "$out/$name.done" ] && { say "$run $name: done already"; return 0; }
    sched=$(python3 "$FROZEN/frozen.py" schedule --test "$t" --run "$of") || { say "$run $name: no $t schedule ($of)"; return 1; }
    want=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['slots']))" "$sched")
    if [ -n "${adapter:-}" ]; then
        # the directory vLLM loads it from: for Qwen3.5 a copy under the checkpoint's names (loop.py served_adapter)
        dir=$(~/vllm-0.30.0/bin/python "$CODE/loop.py" vllm-adapter --model "$MODEL" "$adapter" 2>> "$out/$name.serve.log") \
            || { say "$run $name: preparing $adapter for vLLM failed ($out/$name.serve.log)"; return 1; }
    fi
    for _ in $(seq 60); do idle "$gpus" && break; up "$port" && break; sleep 30; done
    serve "$port" "$gpus" "$served" "$dir" "$out/$name.serve.log" || { say "$run $name: its server did not start"; return 1; }
    say "$run $name: serving $served on GPUs $gpus (:$port), $t test of $want instances"
    for i in 1 2 3; do
        up "$port" || serve "$port" "$gpus" "$served" "$dir" "$out/$name.serve.log" \
            || { say "$run $name: its server did not start again"; break; }
        if [ -d "$out/$name/play" ] && [ "$(records "$out/$name")" -lt "$want" ]; then
            # a play cut short is set aside whole before the test is played again from its seeding (the frozen.py
            # of the node's tests deletes it), its users' copies of the memory left out as a recorded test's are
            aside=$out/$name/play.stopped-$(date -u +%Y%m%dT%H%M%S)
            mv "$out/$name/play" "$aside" && rm -rf "$aside"/users/*/system/state
            say "$run $name: attempt $i plays it again from its seeding, the cut play set aside in $aside"
        fi
        mkdir -p "$out/$name"
        tar czf "$out/$name/code.tgz" -C "$HOME" --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
            "$(basename "$FROZEN")" "$(basename "$CODE")" overrun-fix continual-arc-baselines continual-arc 2>/dev/null
        say "$run $name: attempt $i"
        (cd ~/prime-rl && export PYTHONPATH=$PYPATH && keep_exec "$out/$name" --resume --code "$T" --also "$CODE" \
            --also "$FROZEN" --also "$HOME/overrun-fix" --also "$HOME/continual-arc-baselines" \
            -- uv run --no-sync python "$FROZEN/frozen.py" play "${who[@]}" --test "$t" --out "$out" --name "$name" \
            --schedule "$sched" --base-url "http://localhost:$port/v1" "${model[@]}") >> "$out/$name.log" 2>&1 < /dev/null
        [ "$(records "$out/$name")" -ge "$want" ] && { ok=1; break; }
        say "$run $name: stopped short at $(records "$out/$name") of $want records ($out/$name.log)"
    done
    stop "$port"
    [ "$ok" = 1 ] || { say "$run $name: not played whole in three tries"; return 1; }
    rm -rf "$out/$name"/play/users/*/system/state
    touch "$out/$name.done"
    say "$run $name done: $(python3 -c "import json, sys; rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; print(len(rs), 'records,', sum(bool(r.get('solved')) for r in rs), 'solved')" "$out/$name/play/records.jsonl")"
}
since() { tac "$LOG" | sed '/more-plays.sh: started: /q'; }   # the log since the launch now going
live() {   # live JOBS: a row a play every LIVE_S (120) seconds in its folder's live.jsonl, until every play is over
    local kind run t name gpus out a b c over agents
    while true; do
        over=1
        agents=$(pgrep -af -- '--mode rpc' 2>/dev/null)
        while read -r kind run t name gpus a b c; do
            [ -n "$kind" ] || continue
            case $kind in \#*) continue ;; esac
            out=$HOME/runs/$run/frozen; [ "$kind" = adapter ] && out=$HOME/runs/$a/frozen
            python3 - "$out" "$name" "$kind" "$run" "$t" "$(grep -o -- "$out/$name/play/users/u[0-9]*" <<< "$agents" | sort -u | grep -c .)" \
                "$(since | grep -c "$run $name: not played whole")" "$(hostname)" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, name, kind, run, test, playing, failed, host = sys.argv[1:]
play = Path(out) / name / "play"
records = [json.loads(line) for line in (play / "records.jsonl").read_text().splitlines() if line.strip()] \
    if (play / "records.jsonl").exists() else []
done = (Path(out) / f"{name}.done").exists()
state = ("done" if done else "failed" if int(failed) else "playing" if int(playing) else
         "seeding" if (play / "users").is_dir() else "waiting")
row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": host, "kind": kind, "run": run,
       "test": test, "name": name, "out": out, "state": state, "playing": int(playing),
       "seeded": len(list((play / "users").glob("u*"))) if (play / "users").is_dir() else 0,
       "records": len(records), "solved": sum(bool(r.get("solved")) for r in records),
       "cost": round(sum(r.get("cost", 0) for r in records) / len(records), 3) if records else None}
(Path(out) / name).mkdir(parents=True, exist_ok=True)
with open(Path(out) / name / "live.jsonl", "a") as fp:
    fp.write(json.dumps(row) + "\n")
PY
            [ -f "$out/$name.done" ] || since | grep -q "$run $name: not played whole" || over=0
        done < "$1"
        [ "$over" = 1 ] && return 0
        sleep "${LIVE_S:-120}"
    done
}
[ "${2:-}" = --live ] && { live "$JOBS"; exit 0; }   # more-plays.sh JOBS --live: the rows alone, for plays already going
{   # read whole before it runs: bash reads a script as it goes
    sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0   # the harness's bubblewrap (frozen.sh's)
    live "$JOBS" &
    say "started: $(grep -c . "$JOBS") plays from $JOBS, $GAP s apart, importing from $PYPATH, frozen.py from $FROZEN"
    while read -r kind rest; do
        [ -n "$kind" ] || continue
        case $kind in \#*) continue ;; esac
        # stdin from nothing: a play's commands would otherwise read the rest of JOBS
        play "$kind" $rest < /dev/null &
        sleep "$GAP"
    done < "$JOBS"
    wait
    say "every play of $JOBS taken"
}
