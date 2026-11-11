#!/usr/bin/env bash
# grid-vals.sh NAME:INIT ...: several models' validations on R's 100 puzzles on this node, each played and measured
# as grid-exp.sh's val and measures stages play one (her schedule and aliases, the default budget, no memory between
# puzzles; a play stopped 60 minutes after 95 of its games have ended or 6 hours in, then its per-game measures), four
# at a time (the author, 20 Oct 2026: one cost-conscious test of the DAgger final runs against the base model, not a validation
# curve). Each model has a server of its own: one copy, tensor-parallel over two GPUs, on port 8101 plus its place in
# its wave of four, started with serve.sh's arguments, so every model's calls run under the same load and every game's
# calls find its conversation in its server's cache; a fifth model and on play in a later wave on the same four
# servers, each loading the next model's adapter as its last model's play ends.
# The serving sets the score, since the harness cuts an action at its time budget: dagger-u4-lag1's phase 17 made 97.5
# model calls a game and solved 42 of 99 on two H200s a model (NODE-dagger-val, 20 Oct 2026), and on 25 Oct
# (NODE-cross-val) 24.6 and 13 on one H100 a model, eight at once, while on two H100s a model, four at a time, it
# solved 26 of the first 77 puzzles to end, against 38 of the same 77 on the H200s; each call generated as much either
# way, it just took longer. So a model's score here compares only with others served the same way on the same GPUs.
# A game's calls each resend its whole
# conversation (about 20k tokens): with four models on one server of 8 data-parallel copies a call lands on any copy,
# only about a quarter of each prompt was found in a copy's cache, and calls ran at a third of a validation's speed;
# with five models on one copy sharded over all 8 GPUs the cache held, but decoding a 9B model split 8 ways was slower
# still (20 Oct 2026).
# INIT is adapter:PATH (that adapter, served as NAME) or none (the base model). SCHED and ALIAS play another schedule
# instead, with that run's aliases (scratch.sh's held-out instances: default her schedule, ~/runs/loop/sched-val.json,
# and her aliases, ~/runs/loop/r0-val). Detached and resumable as grid-exp.sh is: ~/exp/NAME holds each model's markers,
# progress, logs and val-measures.json as grid-exp.sh leaves them, ~/exp/vals the driver's, ~/runs/vllm-PORT.log each
# server's. NCCL runs without NVLink SHARP, whose init fails on the cloud provider's 8x H100 nodes made from 25 Oct 2026.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 NCCL_NVLS_ENABLE=0
# a tensor-parallel server's custom all-reduce cannot open expandable segments over CUDA IPC (scratch.sh's b30cfc1)
unset PYTORCH_CUDA_ALLOC_CONF
V=~/exp/vals
SCHED=${SCHED:-$HOME/runs/loop/sched-val.json} ALIAS=${ALIAS:-$HOME/runs/loop/r0-val}
WAVE=4
mkdir -p "$V"
say() { echo "$(date -u +%m-%dT%H:%M) vals: $*" >> "$V/progress"; }
log() { echo "$(date -u +%m-%dT%H:%M) $1: $2" >> ~/exp/"$1"/progress; }
MODEL=Qwen/Qwen3.5-9B
# serve.sh's launch for Qwen3.5 (its SERVE_ARGS and LORA_ARGS, one API frontend so adapters load at run time)
SERVE_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --chat-template-content-format string --language-model-only --chat-template "$HOME/loop/qwen35_prefill.jinja"
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-lora --max-lora-rank 128 --max-loras 12
    --api-server-count 1)
# model I's place in its wave: GPUs 2J and 2J+1 and port 8101+J
gpus() { local j=$(( $1 % WAVE )); echo "$(( 2 * j )),$(( 2 * j + 1 ))"; }
port() { echo $(( 8101 + $1 % WAVE )); }
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
    log "$name" "started: INIT $init, one of $# models, wave $(( i / WAVE + 1 )) of $(( ($# + WAVE - 1) / WAVE )), served alone on GPUs $(gpus "$i"), port $(port "$i")"
done
want=$(python3 -c "import json; print(len(json.load(open('$SCHED'))['slots']))")
records() { [ -f ~/runs/loop/val-"$1"/records.jsonl ] && wc -l < ~/runs/loop/val-"$1"/records.jsonl || echo 0; }
# up PORT: three tries ten seconds apart, so a server busy for a moment is never taken for a dead one (serve.sh's)
up() { for _ in 1 2 3; do curl -fsS -m 10 "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done; return 1; }
# serve I: model I's server answering and its adapter loaded. One that stopped answering is stopped first (its process
# group, while its pid is still vLLM's) and its log kept, as serve.sh does
serve() {
    local port pid dir
    port=$(port "$1")
    if ! up "$port"; then
        pid=$(cat ~/runs/vllm-"$port".pid 2>/dev/null)
        [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
        [ -f ~/runs/vllm-"$port".log ] && mv ~/runs/vllm-"$port".log ~/runs/vllm-"$port".$(date -u +%m%d-%H%M%S).log
        (cd ~/continual-arc-baselines && VLLM_ALLOW_RUNTIME_LORA_UPDATING=True MAX_MODEL_LEN=131072 GPU_MEM=0.85 \
            bash scripts/vllm-serve.sh "$MODEL" "$(gpus "$1")" "$port" "${SERVE_ARGS[@]}") >> "$V/serve.log" 2>&1 || return 1
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
        --run-dir "$d" --base-url "http://localhost:$(port "$1")/v1" --model "${models[$1]}" \
        --alias-from "$ALIAS" $(cat ~/exp/"$n"/play.args 2>/dev/null) >> "$d.log" 2>&1 < /dev/null &)
    echo "$(date -u +%H:%M) launched val-$n ($(records "$n")/$want)" >> "$V/play.log"
}
left() { for i in "$@"; do [ -f ~/exp/"${names[$i]}"/val.done ] || echo "$i"; done; }
# prune I: model I's ended play's users' copies of the harness's kernel venv, about 27 GB a play, which no measure reads
# and the backup never keeps; eight plays' worth fills an 8x H100 node's 500 GB disk before their measures
prune() {
    local d=$HOME/runs/loop/val-${names[$1]}
    while pgrep -f "[r]un-dir $d( |$)" > /dev/null; do sleep 10; done
    rm -rf "$d"/users/*/system/state/agent/kernel-venv
}
# the sandbox setting resume.sh makes: the harness's bubblewrap needs unprivileged user namespaces
sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0
for (( w = 0; w < ${#names[@]}; w += WAVE )); do
    wave=(); for (( i = w; i < w + WAVE && i < ${#names[@]}; i++ )); do wave+=("$i"); done
    [ -n "$(left "${wave[@]}")" ] || continue
    pids=(); for i in $(left "${wave[@]}"); do serve "$i" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { say "a server did not start ($V/serve.log)"; exit 1; }; done
    say "wave $(( w / WAVE + 1 )): serving $(for i in $(left "${wave[@]}"); do echo "${names[$i]} on :$(port "$i")"; done | paste -sd, -)"
    start=$(date +%s)
    declare -A since=()
    while [ -n "$(left "${wave[@]}")" ]; do
        for i in $(left "${wave[@]}"); do
            n=${names[$i]} d=$HOME/runs/loop/val-${names[$i]}
            if [ "$(records "$n")" -ge "$want" ]; then touch ~/exp/"$n"/val.done; log "$n" "val done"; continue; fi
            up "$(port "$i")" || { say "$n's server does not answer: starting it again"; serve "$i" || { say "it did not start"; exit 1; }; }
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
        [ -n "$(left "${wave[@]}")" ] && sleep 60
    done
    for i in "${wave[@]}"; do prune "$i"; done
done
cd ~/prime-rl || exit 1
for i in "${!names[@]}"; do
    n=${names[$i]}
    prune "$i"
    [ -f ~/exp/"$n"/measures.done ] && continue
    uv run --no-sync python ~/loop/forward.py game-measures ~/runs/loop/val-"$n" --out ~/exp/"$n"/val-measures.json \
        > ~/exp/"$n"/measures.log 2>&1 || { log "$n" "measures failed (~/exp/$n/measures.log)"; continue; }
    log "$n" "validation: $(tail -1 ~/exp/"$n"/measures.log | cut -c1-400)"
    touch ~/exp/"$n"/measures.done; log "$n" "measures done"
done
pkill -f "[v]llm serve"
say "finished"
