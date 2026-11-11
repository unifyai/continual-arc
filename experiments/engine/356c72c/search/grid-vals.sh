#!/usr/bin/env bash
# grid-vals.sh NAME:INIT ...: several models' validations on R's 100 puzzles at once on this node, each played and
# measured as grid-exp.sh's val and measures stages play one (her schedule and aliases, the default budget, no memory
# between puzzles; a play stopped 60 minutes after 95 of its games have ended or 6 hours in, then its per-game
# measures), so that N models take about one validation's time on one node (the author, 20 Oct 2026: one cost-conscious test
# of the DAgger final runs against the base model, not a validation curve). Each model has a server of its own: one
# copy, tensor-parallel over an even share of the 8 GPUs (N divides 8), on port 8101 plus its place in the arguments,
# started with serve.sh's arguments, so every model's calls run under the same load and every game's calls find its
# conversation in its server's cache. A game's calls each resend its whole conversation (about 20k tokens): with four
# models on one server of 8 data-parallel copies a call lands on any copy, only about a quarter of each prompt was
# found in a copy's cache, and calls ran at a third of a validation's speed; with five models on one copy sharded over
# all 8 GPUs the cache held, but decoding a 9B model split 8 ways was slower still (NODE-dagger-val, 20 Oct 2026).
# INIT is adapter:PATH (that adapter, served as NAME) or none (the base model). SCHED and ALIAS play another schedule
# instead, with that run's aliases (scratch.sh's held-out instances: default her schedule, ~/runs/loop/sched-val.json,
# and her aliases, ~/runs/loop/r0-val). Detached and resumable as grid-exp.sh is: ~/exp/NAME holds each model's markers,
# progress, logs and val-measures.json as grid-exp.sh leaves them, ~/exp/vals the driver's, ~/runs/vllm-PORT.log each
# server's.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1
V=~/exp/vals
SCHED=${SCHED:-$HOME/runs/loop/sched-val.json} ALIAS=${ALIAS:-$HOME/runs/loop/r0-val}
mkdir -p "$V"
say() { echo "$(date -u +%m-%dT%H:%M) vals: $*" >> "$V/progress"; }
log() { echo "$(date -u +%m-%dT%H:%M) $1: $2" >> ~/exp/"$1"/progress; }
[ $(( 8 % $# )) = 0 ] || { say "$# models do not share 8 GPUs evenly"; exit 1; }
K=$(( 8 / $# ))
MODEL=Qwen/Qwen3.5-9B
# serve.sh's launch for Qwen3.5 (its SERVE_ARGS and LORA_ARGS, one API frontend so adapters load at run time)
SERVE_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --chat-template-content-format string --language-model-only --chat-template "$HOME/loop/qwen35_prefill.jinja"
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
    mkdir -p ~/exp/"$name"
    log "$name" "started: INIT $init, one of $# models, served alone on GPUs $(seq -s, $(( i * K )) $(( i * K + K - 1 ))), port $(( 8101 + i ))"
done
want=$(python3 -c "import json; print(len(json.load(open('$SCHED'))['slots']))")
records() { [ -f ~/runs/loop/val-"$1"/records.jsonl ] && wc -l < ~/runs/loop/val-"$1"/records.jsonl || echo 0; }
# up PORT: three tries ten seconds apart, so a server busy for a moment is never taken for a dead one (serve.sh's)
up() { for _ in 1 2 3; do curl -fsS -m 10 "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done; return 1; }
# serve I: model I's server answering and its adapter loaded. One that stopped answering is stopped first (its process
# group, while its pid is still vLLM's) and its log kept, as serve.sh does
serve() {
    local port=$(( 8101 + $1 )) pid dir
    if ! up "$port"; then
        pid=$(cat ~/runs/vllm-"$port".pid 2>/dev/null)
        [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
        [ -f ~/runs/vllm-"$port".log ] && mv ~/runs/vllm-"$port".log ~/runs/vllm-"$port".$(date -u +%m%d-%H%M%S).log
        (cd ~/continual-arc-baselines && VLLM_ALLOW_RUNTIME_LORA_UPDATING=True MAX_MODEL_LEN=131072 GPU_MEM=0.85 \
            bash scripts/vllm-serve.sh "$MODEL" "$(seq -s, $(( $1 * K )) $(( $1 * K + K - 1 )))" "$port" "${SERVE_ARGS[@]}") \
            >> "$V/serve.log" 2>&1 || return 1
    fi
    [ -z "${paths[$1]}" ] && return 0
    curl -fsS "localhost:$port/v1/models" | grep -q "\"${names[$1]}\"" && return 0
    # the directory vLLM loads it from: for Qwen3.5 a copy under the checkpoint's names (loop.py served_adapter)
    dir=$(~/vllm-0.30.0/bin/python ~/loop/loop.py vllm-adapter --model "$MODEL" "${paths[$1]}") || return 1
    curl -fsS -X POST "localhost:$port/v1/load_lora_adapter" -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"${names[$1]}\", \"lora_path\": \"$dir\"}" >> "$V/serve.log" 2>&1
}
# play I: model I's play, resumed from its records (as resume.sh does), unless it is running
play() {
    local n=${names[$1]} d=$HOME/runs/loop/val-${names[$1]}
    pgrep -f "[r]un-dir $d( |$)" > /dev/null && return 0
    # ~/exp/NAME/play.args, when the launch left one: more loop.py play options for that model's validation
    (cd ~/prime-rl && setsid nohup uv run --no-sync python ~/loop/loop.py play --schedule "$SCHED" \
        --run-dir "$d" --base-url "http://localhost:$(( 8101 + $1 ))/v1" --model "${models[$1]}" \
        --alias-from "$ALIAS" $(cat ~/exp/"$n"/play.args 2>/dev/null) >> "$d.log" 2>&1 < /dev/null &)
    echo "$(date -u +%H:%M) launched val-$n ($(records "$n")/$want)" >> "$V/play.log"
}
left() { for i in "${!names[@]}"; do [ -f ~/exp/"${names[$i]}"/val.done ] || echo "$i"; done; }
if [ -n "$(left)" ]; then
    # the sandbox setting resume.sh makes: the harness's bubblewrap needs unprivileged user namespaces
    sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0
    pids=(); for i in $(left); do serve "$i" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { say "a server did not start ($V/serve.log)"; exit 1; }; done
    say "serving $(for i in $(left); do echo "${names[$i]} on :$(( 8101 + i ))"; done | paste -sd, -)"
fi
start=$(date +%s)
declare -A since=()
while [ -n "$(left)" ]; do
    for i in $(left); do
        n=${names[$i]} d=$HOME/runs/loop/val-${names[$i]}
        if [ "$(records "$n")" -ge "$want" ]; then touch ~/exp/"$n"/val.done; log "$n" "val done"; continue; fi
        up $(( 8101 + i )) || { say "$n's server does not answer: starting it again"; serve "$i" || { say "it did not start"; exit 1; }; }
        play "$i"
        have=$(ls "$d"/captured 2>/dev/null | grep -c '\.json\.gz$')
        if [ -z "${since[$n]:-}" ] && [ "$have" -ge 95 ]; then since[$n]=$(date +%s); log "$n" "$have/100 games ended: 60 minutes more"; fi
        if { [ -n "${since[$n]:-}" ] && [ $(( $(date +%s) - since[$n] )) -ge 3600 ]; } || [ $(( $(date +%s) - start )) -ge 21600 ]; then
            # this play alone: the others play on
            for _ in 1 2 3 4 5; do pkill -9 -f "[l]oop/loop.py play .*--run-dir $d( |$)"; sleep 3; done
            log "$n" "stopped at the cutoff: $have/100 games ended"
            touch ~/exp/"$n"/val.done; log "$n" "val done"
        fi
    done
    [ -n "$(left)" ] && sleep 60
done
cd ~/prime-rl || exit 1
for n in "${names[@]}"; do
    [ -f ~/exp/"$n"/measures.done ] && continue
    uv run --no-sync python ~/loop/forward.py game-measures ~/runs/loop/val-"$n" --out ~/exp/"$n"/val-measures.json \
        > ~/exp/"$n"/measures.log 2>&1 || { log "$n" "measures failed (~/exp/$n/measures.log)"; continue; }
    log "$n" "validation: $(tail -1 ~/exp/"$n"/measures.log | cut -c1-400)"
    touch ~/exp/"$n"/measures.done; log "$n" "measures done"
done
pkill -f "[v]llm serve"
say "finished"
