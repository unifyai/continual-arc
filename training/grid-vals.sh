#!/usr/bin/env bash
# grid-vals.sh NAME:INIT ...: several models' validations on the 100 validation puzzles of the earlier experiments at
# once on this node, each played and measured alike (the validation schedule and its aliases, the default budget, no
# memory between puzzles; a play stopped 60 minutes after 95 of its games have ended or 6 hours in, then its per-game
# measures, forward.py game-measures), so that N models take about one validation's time on one node: a single
# cost-conscious comparison of a few adapters, not a validation curve. Each model has a server of its own: one copy,
# tensor-parallel over an even share of the 8 GPUs (N divides 8), on port 8101 plus its place in the arguments, started
# with serve.sh's arguments, so every model's calls run under the same load and every game's calls find its
# conversation in its server's cache. A game's calls each resend its whole conversation (about 20k tokens): with four
# models on one server of 8 data-parallel copies a call lands on any copy, only about a quarter of each prompt was
# found in a copy's cache, and calls ran at a third of a validation's speed; with five models on one copy sharded over
# all 8 GPUs the cache held, but decoding a 9B model split 8 ways was slower still.
# INIT is adapter:PATH (that adapter, served as NAME) or none (the base model). SCHED and ALIAS play another schedule
# instead, with that run's aliases (default the validation schedule, schedules/validation/schedule.json beside this
# script, and its aliases, the alias-salt in schedules/validation). Detached and resumable: EXP/NAME (EXP default
# ~/exp) holds each model's markers, progress, logs and val-measures.json, EXP/vals the driver's, RUNS/vllm-PORT.log
# each server's (RUNS default ~/runs), and RUNS/loop/val-NAME each play. CODE is the training code it plays with
# (default this script's folder: loop.py, forward.py, qwen35_prefill.jinja and vllm-serve.sh), PRIME_RL (default
# ~/prime-rl) and VLLM_VENV (default ~/vllm-0.30.0) the node's environments.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$T} PRIME_RL=${PRIME_RL:-$HOME/prime-rl} VLLM_VENV=${VLLM_VENV:-$HOME/vllm-0.30.0}
RUNS=${RUNS:-$HOME/runs} EXP=${EXP:-$HOME/exp}
V=$EXP/vals
SCHED=${SCHED:-$T/schedules/validation/schedule.json} ALIAS=${ALIAS:-$T/schedules/validation}
mkdir -p "$V"
say() { echo "$(date -u +%m-%dT%H:%M) vals: $*" >> "$V/progress"; }
log() { echo "$(date -u +%m-%dT%H:%M) $1: $2" >> "$EXP/$1/progress"; }
[ $(( 8 % $# )) = 0 ] || { say "$# models do not share 8 GPUs evenly"; exit 1; }
K=$(( 8 / $# ))
MODEL=Qwen/Qwen3.5-9B
# serve.sh's launch for Qwen3.5 (its SERVE_ARGS and LORA_ARGS, one API frontend so adapters load at run time)
SERVE_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --chat-template-content-format string --language-model-only --chat-template "$CODE/qwen35_prefill.jinja"
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-lora --max-lora-rank 128 --max-loras 12
    --api-server-count 1)
names=(); models=(); paths=()
for spec in "$@"; do
    name=${spec%%:*}; init=${spec#*:}
    case $init in
        none) models+=("$MODEL"); paths+=("") ;;
        adapter:*) models+=("$name"); paths+=("${init#adapter:}") ;;
        *) say "INIT must be adapter:PATH or none ($spec)"; exit 1 ;;
    esac
    i=${#names[@]}; names+=("$name")
    mkdir -p "$EXP/$name"
    log "$name" "started: INIT $init, one of $# models, served alone on GPUs $(seq -s, $(( i * K )) $(( i * K + K - 1 ))), port $(( 8101 + i ))"
done
want=$(python3 -c "import json; print(len(json.load(open('$SCHED'))['slots']))")
records() { [ -f "$RUNS/loop/val-$1/records.jsonl" ] && wc -l < "$RUNS/loop/val-$1/records.jsonl" || echo 0; }
# up PORT: three tries ten seconds apart, so a server busy for a moment is never taken for a dead one (serve.sh's)
up() { for _ in 1 2 3; do curl -fsS -m 10 "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done; return 1; }
# serve I: model I's server answering and its adapter loaded. One that stopped answering is stopped first (its process
# group, while its pid is still vLLM's) and its log kept, as serve.sh does
serve() {
    local port=$(( 8101 + $1 )) pid dir
    if ! up "$port"; then
        pid=$(cat "$RUNS/vllm-$port.pid" 2>/dev/null)
        [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
        [ -f "$RUNS/vllm-$port.log" ] && mv "$RUNS/vllm-$port.log" "$RUNS/vllm-$port.$(date -u +%m%d-%H%M%S).log"
        (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True MAX_MODEL_LEN=131072 GPU_MEM=0.85 VLLM_VENV=$VLLM_VENV RUNS=$RUNS \
            bash "$T/vllm-serve.sh" "$MODEL" "$(seq -s, $(( $1 * K )) $(( $1 * K + K - 1 )))" "$port" "${SERVE_ARGS[@]}") \
            >> "$V/serve.log" 2>&1 || return 1
    fi
    [ -z "${paths[$1]}" ] && return 0
    curl -fsS "localhost:$port/v1/models" | grep -q "\"${names[$1]}\"" && return 0
    # the directory vLLM loads it from: for Qwen3.5 a copy under the checkpoint's names (loop.py served_adapter)
    dir=$("$VLLM_VENV/bin/python" "$CODE/loop.py" vllm-adapter --model "$MODEL" "${paths[$1]}") || return 1
    curl -fsS -X POST "localhost:$port/v1/load_lora_adapter" -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"${names[$1]}\", \"lora_path\": \"$dir\"}" >> "$V/serve.log" 2>&1
}
# play I: model I's play, resumed from its records, unless it is running
play() {
    local n=${names[$1]} d=$RUNS/loop/val-${names[$1]}
    pgrep -f "[r]un-dir $d( |$)" > /dev/null && return 0
    # EXP/NAME/play.args, when the launch left one: more loop.py play options for that model's validation
    (cd "$PRIME_RL" && setsid nohup uv run --no-sync python "$CODE/loop.py" play --schedule "$SCHED" \
        --run-dir "$d" --base-url "http://localhost:$(( 8101 + $1 ))/v1" --model "${models[$1]}" \
        --alias-from "$ALIAS" $(cat "$EXP/$n/play.args" 2>/dev/null) >> "$d.log" 2>&1 < /dev/null &)
    echo "$(date -u +%H:%M) launched val-$n ($(records "$n")/$want)" >> "$V/play.log"
}
left() { for i in "${!names[@]}"; do [ -f "$EXP/${names[$i]}/val.done" ] || echo "$i"; done; }
if [ -n "$(left)" ]; then
    # the harness's bubblewrap sandbox needs unprivileged user namespaces
    sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0
    pids=(); for i in $(left); do serve "$i" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { say "a server did not start ($V/serve.log)"; exit 1; }; done
    say "serving $(for i in $(left); do echo "${names[$i]} on :$(( 8101 + i ))"; done | paste -sd, -)"
fi
start=$(date +%s)
declare -A since=()
while [ -n "$(left)" ]; do
    for i in $(left); do
        n=${names[$i]} d=$RUNS/loop/val-${names[$i]}
        if [ "$(records "$n")" -ge "$want" ]; then touch "$EXP/$n/val.done"; log "$n" "val done"; continue; fi
        up $(( 8101 + i )) || { say "$n's server does not answer: starting it again"; serve "$i" || { say "it did not start"; exit 1; }; }
        play "$i"
        have=$(ls "$d"/captured 2>/dev/null | grep -c '\.json\.gz$')
        if [ -z "${since[$n]:-}" ] && [ "$have" -ge 95 ]; then since[$n]=$(date +%s); log "$n" "$have/100 games ended: 60 minutes more"; fi
        if { [ -n "${since[$n]:-}" ] && [ $(( $(date +%s) - since[$n] )) -ge 3600 ]; } || [ $(( $(date +%s) - start )) -ge 21600 ]; then
            # this play alone: the others play on
            for _ in 1 2 3 4 5; do pkill -9 -f "[l]oop.py play .*--run-dir $d( |$)"; sleep 3; done
            log "$n" "stopped at the cutoff: $have/100 games ended"
            touch "$EXP/$n/val.done"; log "$n" "val done"
        fi
    done
    [ -n "$(left)" ] && sleep 60
done
cd "$PRIME_RL" || exit 1
for n in "${names[@]}"; do
    [ -f "$EXP/$n/measures.done" ] && continue
    uv run --no-sync python "$CODE/forward.py" game-measures "$RUNS/loop/val-$n" --out "$EXP/$n/val-measures.json" \
        > "$EXP/$n/measures.log" 2>&1 || { log "$n" "measures failed ($EXP/$n/measures.log)"; continue; }
    log "$n" "validation: $(tail -1 "$EXP/$n/measures.log" | cut -c1-400)"
    touch "$EXP/$n/measures.done"; log "$n" "measures done"
done
pkill -f "[v]llm serve"
say "finished"
