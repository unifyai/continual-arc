#!/usr/bin/env bash
# frozen.sh RUN GPUS: the two frozen tests of real-time run RUN (frozen.py's docstring) on its own node, once its stream
# has ended: every loop of realtime.sh's done and its last line "done", written once it has stopped the node's model
# server if no other run there still uses it. Each test has a model server of its own on half of GPUS, one copy
# tensor-parallel over them as the stream's was, launched as serve.sh launches it (its SERVE_ARGS and LORA_ARGS) but
# on a port of its own, 8120 plus its first GPU, so that no other run's server on the node is ever touched; the run's
# last phase's adapter is loaded into both, and frozen.py plays the two tests side by side, each started again from
# its seeding, up to three times, should it stop before its round is recorded, each into OUT/TEST/play (frozen.py's
# docstring). Then OUT/summary.json, the test users' copies of the memory deleted (the stream's own is the run's, and
# every call, turn and game of the test stays), the two servers stopped, and OUT/done. Detached and resumable: run
# again, it takes up where it stopped.
# ADAPTER and SERVED (both or neither) play another adapter, served under that name, instead of the run's last phase
# (scratch.sh's from-scratch retrain), with OUT another folder; OUT defaults to RUNS/rt/RUN/frozen. ROW (h0 or bare,
# paper.py's transfer references) plays that reference from nothing on the base model instead, into
# OUT/TEST-reference/play, with no run to wait for: RUN then names its folder, RUNS/paper/RUN, OUT defaulting to its
# frozen/. MEMORY=none plays the adapter (the run's last phase, or ADAPTER) from an empty harness memory under the
# run's own task names instead, its weights alone (frozen.py --no-memory), into OUT/TEST-weights, OUT defaulting to
# RUNS/rt/RUN/weights (scratch.sh's first stage). CODE is the training code the tests play on (the folder this script
# is in by default; frozen-launch.sh copies this script, frozen.py, vllm-serve.sh and schedules/ to a folder of their
# own on the node and points CODE at the training folder the run played with), and PYPATH the Python path they import
# from (CODE by default): refs.sh points both at a copy of the current code beside the node's. POLL is how often it
# looks, in seconds (60). PRIME_RL (default ~/prime-rl), VLLM_VENV (default ~/vllm-0.30.0) and RUNS (default ~/runs) are the
# node's folders.
# Usage (on the node):
#   cd ~ && (setsid -f nohup bash ~/continual-arc/training/frozen.sh pg-u2-lag1 0,1,2,3,4,5,6,7 > /dev/null 2>&1 < /dev/null)
#   cd ~ && (ROW=bare setsid -f nohup bash ~/continual-arc/training/frozen.sh bare-ref 0,1,2,3,4,5,6,7 > /dev/null 2>&1 < /dev/null)
set -uo pipefail
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRIME_RL=${PRIME_RL:-$HOME/prime-rl} VLLM_VENV=${VLLM_VENV:-$HOME/vllm-0.30.0} RUNS=${RUNS:-$HOME/runs}
RUN=$1 GPUS=$2 ROW=${ROW:-} CODE=${CODE:-$T} MEMORY=${MEMORY:-}
PYPATH=${PYPATH:-$CODE}
R=$RUNS/rt/$RUN
[ -n "$ROW" ] && R=$RUNS/paper/$RUN
FOLDER=frozen; [ "$MEMORY" = none ] && FOLDER=weights
OUT=${OUT:-$R/$FOLDER} ADAPTER=${ADAPTER:-} SERVED=${SERVED:-}
TESTS=(held-out seen) NAMES=(held-out seen)
[ -n "$ROW" ] && NAMES=(held-out-reference seen-reference)
[ "$MEMORY" = none ] && NAMES=(held-out-weights seen-weights)
POLL=${POLL:-60}
MODEL=Qwen/Qwen3.5-9B
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1
mkdir -p "$OUT"
say() { echo "$(date -u +%m-%d\ %H:%M) frozen.sh: $*" >> "$OUT/progress"; }
# serve.sh's launch for Qwen3.5 (its SERVE_ARGS and LORA_ARGS, one API frontend so that adapters load at run time)
SERVE_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --chat-template-content-format string --language-model-only --chat-template "$CODE/qwen35_prefill.jinja"
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-lora --max-lora-rank 128 --max-loras 12
    --api-server-count 1)
case ${RUN%%-*} in pg) LOOPS="player trainer" ;; *) LOOPS="player labeller trainer" ;; esac
# the stream has ended: every loop done, and realtime.sh's last word "done", said after it stopped what it stops
ended() {
    local loop
    for loop in $LOOPS; do [ -f "$R/$loop.done" ] || return 1; done
    grep "realtime.sh: " "$R/progress" | tail -1 | grep -q "realtime.sh: done$"
}
idle() {   # every GPU of $1 with next to nothing on it
    local g
    for g in ${1//,/ }; do
        [ "$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] || return 1
    done
}
gpus=(${GPUS//,/ })
half=$(( ${#gpus[@]} / 2 ))
shares=("$(IFS=,; echo "${gpus[*]:0:$half}")" "$(IFS=,; echo "${gpus[*]:$half}")")
ports=($(( 8120 + gpus[0] )) $(( 8120 + gpus[half] )))
# up PORT: three tries ten seconds apart, so a server busy for a moment is never taken for a dead one (serve.sh's)
up() { for _ in 1 2 3; do curl -fsS -m 10 "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done; return 1; }
stop() {   # stop PORT: its server's process group, while its pid is still vLLM's
    local pid
    pid=$(cat "$RUNS/vllm-$1.pid" 2>/dev/null)
    [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
    return 0
}
serve() {   # serve I: test I's server answering, the adapter (if any) loaded into it
    local port=${ports[$1]}
    if ! up "$port"; then
        stop "$port"
        [ -f "$RUNS/vllm-$port.log" ] && mv "$RUNS/vllm-$port.log" "$RUNS/vllm-$port.$(date -u +%m%d-%H%M%S).log"
        (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True MAX_MODEL_LEN=131072 GPU_MEM=0.85 VLLM_VENV=$VLLM_VENV RUNS=$RUNS \
            bash "$T/vllm-serve.sh" "$MODEL" "${shares[$1]}" "$port" "${SERVE_ARGS[@]}") >> "$OUT/serve.log" 2>&1 \
            || return 1
    fi
    [ -n "$ROW" ] && return 0
    curl -fsS "localhost:$port/v1/models" | grep -q "\"$SERVED\"" && return 0
    curl -fsS -X POST "localhost:$port/v1/load_lora_adapter" -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"$SERVED\", \"lora_path\": \"$VLLM_DIR\"}" >> "$OUT/serve.log" 2>&1
}
playing() { pgrep -f "[f]rozen.py play --(run|row) [^ ]+ --test $1 --out $OUT( |$)" > /dev/null; }
records() { [ -f "$OUT/$1/play/records.jsonl" ] && wc -l < "$OUT/$1/play/records.jsonl" || echo 0; }
py() { (cd "$PRIME_RL" && PYTHONPATH=$PYPATH uv run --no-sync python "$T/frozen.py" "$@"); }

{   # read whole before it runs: bash reads a script as it goes
    [ "${#gpus[@]}" -ge 2 ] && [ $(( ${#gpus[@]} % 2 )) = 0 ] \
        || { say "GPUS ($GPUS) must be an even number of GPUs, half for each test"; exit 1; }
    [ "${ADAPTER:+set}" = "${SERVED:+set}" ] || { say "ADAPTER and SERVED go together"; exit 1; }
    [ -z "$ROW" ] || [ -z "$ADAPTER" ] || { say "a reference from nothing plays the base model, no ADAPTER"; exit 1; }
    [ -z "$MEMORY" ] || [ "$MEMORY" = none ] || { say "MEMORY is none or unset, not $MEMORY"; exit 1; }
    [ -z "$ROW" ] || [ -z "$MEMORY" ] || { say "a reference from nothing has no memory to leave out"; exit 1; }
    [ -f "$OUT/done" ] && exit 0
    say "started: the frozen tests of $RUN on GPUs $GPUS${ADAPTER:+, $SERVED from $ADAPTER}${ROW:+, row $ROW from nothing}${MEMORY:+, its weights alone from an empty memory}, into $OUT"
    if [ -n "$ROW" ]; then
        SERVED=$MODEL who=(--row "$ROW") model=(--served "$MODEL")
    elif [ -z "$ADAPTER" ]; then
        if ! ended; then
            say "waiting for $RUN's stream to end"
            until ended; do sleep "$POLL"; done
        fi
        read -r SERVED ADAPTER < <(py model --run "$R" 2>> "$OUT/frozen.log")
        [ -n "$ADAPTER" ] || { say "the run has no last phase to test ($OUT/frozen.log)"; exit 1; }
    fi
    if [ -z "$ROW" ]; then
        who=(--run "$R") model=(--served "$SERVED" --adapter "$ADAPTER")
        [ "$MEMORY" = none ] && model+=(--no-memory)
        # the directory vLLM loads it from: for Qwen3.5 a copy under the checkpoint's names (loop.py served_adapter)
        VLLM_DIR=$("$VLLM_VENV/bin/python" "$CODE/loop.py" vllm-adapter --model "$MODEL" "$ADAPTER" 2>> "$OUT/serve.log") \
            || { say "preparing $ADAPTER for vLLM failed ($OUT/serve.log)"; exit 1; }
    fi
    for _ in $(seq 60); do idle "$GPUS" && break; sleep 30; done
    idle "$GPUS" || { say "GPUs $GPUS are still in use after 30 minutes"; exit 1; }
    # the harness's bubblewrap sandbox needs unprivileged user namespaces
    sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0
    pids=(); for i in 0 1; do serve "$i" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { say "a server did not start ($OUT/serve.log)"; exit 1; }; done
    say "serving $SERVED on GPUs ${shares[0]} (:${ports[0]}, ${NAMES[0]}) and ${shares[1]} (:${ports[1]}, ${NAMES[1]})"
    scheds=() wants=() tries=(0 0)
    for t in "${TESTS[@]}"; do
        sched=$T/schedules/arc1-small-$t-test-seed0.json
        scheds+=("$sched")
        wants+=("$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['slots']))" "$sched")")
    done
    while true; do
        left=0
        for i in 0 1; do
            t=${TESTS[$i]} n=${NAMES[$i]}
            [ -f "$OUT/$n.done" ] && continue
            # the runner writes the round's records before the play has closed its harnesses
            if ! playing "$t" && [ "$(records "$n")" -ge "${wants[$i]}" ]; then
                touch "$OUT/$n.done"; say "$n done: $(tail -1 "$OUT/$n.log" | cut -c1-200)"; continue
            fi
            left=1
            up "${ports[$i]}" || { say "the $n server does not answer: starting it again"; serve "$i" || { say "it did not start"; exit 1; }; }
            playing "$t" && continue
            tries[$i]=$(( tries[i] + 1 ))
            [ "${tries[$i]}" -gt 3 ] && { say "the $n play stopped three times ($OUT/$n.log): stopping"; exit 1; }
            say "playing $n, attempt ${tries[$i]}"
            (cd "$PRIME_RL" && PYTHONPATH=$PYPATH setsid nohup uv run --no-sync python "$T/frozen.py" play "${who[@]}" \
                --test "$t" --out "$OUT" --schedule "${scheds[$i]}" --base-url "http://localhost:${ports[$i]}/v1" \
                "${model[@]}" >> "$OUT/$n.log" 2>&1 < /dev/null &)
        done
        [ "$left" = 0 ] && break
        sleep "$POLL"
    done
    of=(--out "$OUT"); [ -n "$ROW" ] || of=(--run "$R" --out "$OUT")
    py summary "${of[@]}" >> "$OUT/frozen.log" 2>&1 || { say "the summary failed ($OUT/frozen.log)"; exit 1; }
    say "summary: $(tail -1 "$OUT/frozen.log" | cut -c1-300)"
    rm -rf "$OUT"/*/play/users/*/system/state
    for port in "${ports[@]}"; do stop "$port"; done
    touch "$OUT/done"
    say "done"
    exit 0
}
