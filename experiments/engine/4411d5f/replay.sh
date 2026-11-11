#!/usr/bin/env bash
# replay.sh LIST GPUS: the replays LIST names (replay.py's docstring: frozen-test games the context overrun cut, each
# played again from where the server refused it, on the harness with the fix), on a node once nothing else uses it: no
# frozen.sh running there and every GPU of GPUS idle, so that a run's own tests, which wait for its stream, are never
# raced for the GPUs. Two model servers of their own on half of GPUS each, launched as frozen.sh launches a test's
# (its SERVE_ARGS, the same 131,072-token window: the fix is the proxy's, not the server's) on its ports, 8120 plus
# their first GPU; every adapter LIST names loaded into both; then every replay played at once, each on the server
# with fewer users given it so far, each started again from its seeding up to three times should it stop before its
# round is recorded (replay.py play), into DIR/play, and left out after that (DIR/failed). Then the replay users' working trees deleted (every one is in its
# repository, as frozen.sh leaves a test), the servers stopped, and OUT/done. Detached and resumable: run again, it
# plays what is left.
# LIST holds a line per replay: its folder (replay.py plan's --out), the name its model is served under and the
# adapter's own folder, or - for a reference played on the base model. OUT is LIST's folder: progress, serve.log and
# each replay's DIR.log. CODE is the run's code (~/loop) and PYPATH what the replays import from, the fixed harness
# ahead of the installed one (~/loop:~/overrun-fix/src, ~/overrun-fix/README.md); replay.py, the node's frozen.py and
# everything replay.py imports beside it must be in the folder this script is in.
# Usage (on the node): cd ~ && (setsid -f nohup bash ~/overrun-fix/tools/replay.sh ~/replay/replays.txt 0,1,2,3,4,5,6,7 \
#   > /dev/null 2>&1 < /dev/null)
set -uo pipefail
LIST=$(realpath "$1") GPUS=$2 CODE=${CODE:-$HOME/loop} PYPATH=${PYPATH:-$HOME/loop:$HOME/overrun-fix/src}
OUT=$(dirname "$LIST") T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) POLL=${POLL:-60}
MODEL=Qwen/Qwen3.5-9B
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 NCCL_NVLS_ENABLE=0
say() { echo "$(date -u +%m-%d\ %H:%M) replay.sh: $*" >> "$OUT/progress"; }
SERVE_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --chat-template-content-format string --language-model-only --chat-template "$CODE/qwen35_prefill.jinja"
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-lora --max-lora-rank 128 --max-loras 12
    --api-server-count 1)
gpus=(${GPUS//,/ })
half=$(( ${#gpus[@]} / 2 ))
shares=("$(IFS=,; echo "${gpus[*]:0:$half}")" "$(IFS=,; echo "${gpus[*]:$half}")")
ports=($(( 8120 + gpus[0] )) $(( 8120 + gpus[half] )))
idle() { local g; for g in ${GPUS//,/ }; do
    [ "$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] || return 1; done; }
free() { ! pgrep -f '[f]rozen\.sh ' > /dev/null && idle; }
up() { for _ in 1 2 3; do curl -fsS -m 10 "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done; return 1; }
stop() { local pid; pid=$(cat ~/runs/vllm-"$1".pid 2>/dev/null)
    [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15; return 0; }
declare -A VLLM_DIR=()
serve() {   # serve I: server I answering, every adapter LIST names loaded into it
    local port=${ports[$1]} name
    if ! up "$port"; then
        stop "$port"
        [ -f ~/runs/vllm-"$port".log ] && mv ~/runs/vllm-"$port".log ~/runs/vllm-"$port".$(date -u +%m%d-%H%M%S).log
        (cd ~/continual-arc-baselines && VLLM_ALLOW_RUNTIME_LORA_UPDATING=True MAX_MODEL_LEN=131072 GPU_MEM=0.85 \
            bash scripts/vllm-serve.sh "$MODEL" "${shares[$1]}" "$port" "${SERVE_ARGS[@]}") >> "$OUT/serve.log" 2>&1 \
            || return 1
    fi
    for name in "${!VLLM_DIR[@]}"; do
        curl -fsS "localhost:$port/v1/models" | grep -q "\"$name\"" && continue
        curl -fsS -X POST "localhost:$port/v1/load_lora_adapter" -H 'Content-Type: application/json' \
            -d "{\"lora_name\": \"$name\", \"lora_path\": \"${VLLM_DIR[$name]}\"}" >> "$OUT/serve.log" 2>&1 || return 1
    done
}
playing() { pgrep -f "[r]eplay\.py play --dir $1 " > /dev/null; }
wanted() { python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['users'])" "$1/schedule.json"; }
records() { [ -f "$1/play/records.jsonl" ] && wc -l < "$1/play/records.jsonl" || echo 0; }

{   # read whole before it runs: bash reads a script as it goes
    [ "${#gpus[@]}" -ge 2 ] && [ $(( ${#gpus[@]} % 2 )) = 0 ] || { say "GPUS ($GPUS) must be an even number"; exit 1; }
    [ -f "$OUT/done" ] && exit 0
    mapfile -t lines < <(grep -v '^\s*\(#\|$\)' "$LIST")
    say "started: ${#lines[@]} replays from $LIST on GPUs $GPUS, importing from $PYPATH"
    for line in "${lines[@]}"; do
        read -r dir name adapter <<< "$line"
        [ -f "$dir/replay.json" ] && [ -f "$dir/plan.jsonl" ] || { say "$dir is not a planned replay"; exit 1; }
        [ "$adapter" = - ] && continue
        [ -n "${VLLM_DIR[$name]:-}" ] && continue
        VLLM_DIR[$name]=$(~/vllm-0.30.0/bin/python "$CODE/loop.py" vllm-adapter --model "$MODEL" "$adapter" \
            2>> "$OUT/serve.log") || { say "preparing $adapter for vLLM failed ($OUT/serve.log)"; exit 1; }
    done
    if ! free; then
        say "waiting for the node: no frozen.sh running and GPUs $GPUS idle"
        until free; do sleep "$POLL"; done
    fi
    sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0   # the harness's bubblewrap (frozen.sh)
    pids=(); for i in 0 1; do serve "$i" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { say "a server did not start ($OUT/serve.log)"; exit 1; }; done
    say "serving ${!VLLM_DIR[*]} and $MODEL on GPUs ${shares[0]} (:${ports[0]}) and ${shares[1]} (:${ports[1]})"
    declare -A tries=() server=()
    load=(0 0)
    for line in "${lines[@]}"; do
        read -r dir name adapter <<< "$line"
        i=0; [ "${load[1]}" -lt "${load[0]}" ] && i=1
        server[$dir]=$i; load[$i]=$(( load[i] + $(wanted "$dir") ))
    done
    while true; do
        left=0
        for line in "${lines[@]}"; do
            read -r dir name adapter <<< "$line"
            [ -f "$dir/done" ] || [ -f "$dir/failed" ] && continue
            if ! playing "$dir" && [ "$(records "$dir")" -ge "$(wanted "$dir")" ]; then
                touch "$dir/done"; say "$dir done: $(tail -1 "$dir.log" | cut -c1-200)"; continue
            fi
            left=1
            i=${server[$dir]}
            up "${ports[$i]}" || { say "server $i does not answer: starting it again"; serve "$i" || { say "it did not start"; exit 1; }; }
            playing "$dir" && continue
            tries[$dir]=$(( ${tries[$dir]:-0} + 1 ))
            if [ "${tries[$dir]}" -gt 3 ]; then
                touch "$dir/failed"; say "the replay $dir stopped three times ($dir.log): left out"; continue
            fi
            say "playing $dir on :${ports[$i]}, attempt ${tries[$dir]}"
            model=(--served "$name"); [ "$adapter" = - ] || model+=(--adapter "$adapter")
            (cd ~/prime-rl && PYTHONPATH=$PYPATH setsid nohup uv run --no-sync python "$T/replay.py" play --dir "$dir" \
                --base-url "http://localhost:${ports[$i]}/v1" "${model[@]}" >> "$dir.log" 2>&1 < /dev/null &)
        done
        [ "$left" = 0 ] && break
        sleep "$POLL"
    done
    for line in "${lines[@]}"; do read -r dir _ <<< "$line"; rm -rf "$dir"/play/users/*/system/state; done
    for port in "${ports[@]}"; do stop "$port"; done
    touch "$OUT/done"
    say "done"
    exit 0
}
