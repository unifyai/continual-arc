#!/usr/bin/env bash
# grid-vals.sh NAME:INIT ...: several models' validations on R's 100 puzzles on this node, each played and measured
# as grid-exp.sh's val and measures stages play one (her schedule and aliases, loop.py play's default budget, no memory
# between puzzles; every game waited for, then its per-game measures), four at a time (the author, 20 Oct 2026: one
# cost-conscious test of the DAgger final runs against the base model, not a validation curve). No play is stopped at a
# time (the author, 25 Oct 2026: "time should never be a gating factor"): one stopped short writes no records.jsonl, since the
# runner records a round only whole, and which of its games end first depends on how fast it is served. A play is
# stopped only as a failure, its val left undone and the others playing on: nothing in it has moved for LIVENESS minutes
# (180 unless set: no game ended and no model call ended, a hung play), or it had to be launched more than LAUNCHES
# times (4: a play that keeps dying). The driver then exits 1 once the rest are measured, and run again it plays the
# failed ones again. Each model has a server of its own: one copy, tensor-parallel over two GPUs, on port 8101 plus its
# place in its wave of four, started with serve.sh's arguments, so every model's calls run under the same load and every
# game's calls find its conversation in its server's cache; a fifth model and on play in a later wave on the same four
# servers, each loading the next model's adapter as its last model's play ends.
# Under a time budget the serving sets the score, since the harness cuts an action at its seconds: dagger-u4-lag1's
# phase 17 made 97.5 model calls a game and solved 42 of 99 on two H200s a model (NODE-dagger-val, 20 Oct 2026), and
# on 25 Oct (NODE-cross-val) 24.6 and 13 on one H100 a model, eight at once, while on two H100s a model, four at a
# time, it solved 26 of the first 77 puzzles to end, against 38 of the same 77 on the H200s; each call generated as much
# either way, it just took longer. So on code whose play cuts at seconds (a generalisation cell's default bundle), a
# model's score here compares only with others served the same way on the same GPUs; on the turn-based code (loop.py
# play's "turns", the turn-based bundle) the layout sets only how long the play takes.
# A game's calls each resend its whole
# conversation (about 20k tokens): with four models on one server of 8 data-parallel copies a call lands on any copy,
# only about a quarter of each prompt was found in a copy's cache, and calls ran at a third of a validation's speed;
# with five models on one copy sharded over all 8 GPUs the cache held, but decoding a 9B model split 8 ways was slower
# still (20 Oct 2026).
# INIT is adapter:PATH (that adapter, served as NAME) or none (the base model). SCHED and ALIAS play another schedule
# instead, with that run's aliases (scratch.sh's held-out instances: default her schedule, ~/runs/loop/sched-val.json,
# and her aliases, ~/runs/loop/r0-val). Detached and resumable as grid-exp.sh is: ~/exp/NAME holds each model's markers,
# progress, logs and val-measures.json as grid-exp.sh leaves them, ~/exp/vals the driver's, ~/runs/vllm-PORT.log each
# server's. NCCL runs without NVLink SHARP, whose init fails on the a GPU cloud 8x H100 nodes made from 25 Oct 2026.
set -uo pipefail
for k in "$(dirname "${BASH_SOURCE[0]}")"/{,../}keep.sh; do [ -f "$k" ] && . "$k" && break; done   # the run supervisor (keep.sh)
type keep_run > /dev/null 2>&1 || { echo "refused: no keep.sh beside $0, the run supervisor's helper" >&2; exit 1; }
# the validations under the supervisor (keep.sh): a part of the run that starts them (scratch.sh's, cell.py's), or a run
# of their own in ~/runs/vals/FIRST-SUM (the first model's name and a checksum of every argument), into which the driver's
# folder, each model's and each play gather as it ends
first=${1:-none}
VALS=~/runs/vals/${first%%:*}-$(printf '%s\n' "$@" | cksum | cut -d' ' -f1)
keep_run "$VALS" --resume --log ~/exp/vals/progress --vars "SCHED ALIAS LIVENESS LAUNCHES" -- "$@" || exit 1
if keep_own "$VALS"; then
    gather_vals() {
        local spec n pairs=(vals ~/exp/vals)
        for spec in "$@"; do n=${spec%%:*}; pairs+=("exp-$n" ~/exp/"$n" "val-$n" ~/runs/loop/val-"$n"); done
        keep_gather "$VALS" "${pairs[@]}"
    }
    trap 'gather_vals "$@"' EXIT
fi
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
# play I: model I's play, resumed from its records (as resume.sh does), unless it is running; each launch counted
play() {
    local n=${names[$1]} d=$HOME/runs/loop/val-${names[$1]}
    pgrep -f "[r]un-dir $d( |$)" > /dev/null && return 0
    launches[$n]=$(( ${launches[$n]:-0} + 1 ))
    # ~/exp/NAME/play.args, when the launch left one: more loop.py play options for that model's validation
    (cd ~/prime-rl && setsid nohup uv run --no-sync python ~/loop/loop.py play --schedule "$SCHED" \
        --run-dir "$d" --base-url "http://localhost:$(port "$1")/v1" --model "${models[$1]}" \
        --alias-from "$ALIAS" $(cat ~/exp/"$n"/play.args 2>/dev/null) >> "$d.log" 2>&1 < /dev/null &)
    echo "$(date -u +%H:%M) launched val-$n ($(records "$n")/$want)" >> "$V/play.log"
}
left() { for i in "$@"; do [ -f ~/exp/"${names[$i]}"/val.done ] || [ -f ~/exp/"${names[$i]}"/val.failed ] || echo "$i"; done; }
# moved RUN SINCE: the newest of SINCE and the ends of RUN's games and model calls (captured/uNNN.json.gz, written as a
# game ends, and users/uNNN/calls.jsonl, a line as each model call ends), in seconds since the epoch
moved() {
    local t
    t=$(find "$1/captured" "$1/users" -maxdepth 2 \( -name 'u*.json.gz' -o -name calls.jsonl \) -printf '%T@\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d. -f1)
    echo $(( ${t:-0} > $2 ? ${t:-0} : $2 ))
}
# the games of a play's records that the turn's liveness guard ended: infrastructure failures, not scores of the model's
infra() { python3 -c "import json, sys; print(sum(bool((json.loads(l).get('resources') or {}).get('system_infra_failures')) for l in open(sys.argv[1])))" "$1/records.jsonl"; }
# prune I: model I's ended play's users' copies of the harness's kernel venv, about 27 GB a play, which no measure reads
# and the backup never keeps; eight plays' worth fills an 8x H100 node's 500 GB disk before their measures
prune() {
    local d=$HOME/runs/loop/val-${names[$1]}
    while pgrep -f "[r]un-dir $d( |$)" > /dev/null; do sleep 10; done
    rm -rf "$d"/users/*/system/state/agent/kernel-venv
}
# the sandbox setting resume.sh makes: the harness's bubblewrap needs unprivileged user namespaces
sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0
# a failure from an earlier launch of the driver is played again
for n in "${names[@]}"; do rm -f ~/exp/"$n"/val.failed; done
declare -A launches=()
for (( w = 0; w < ${#names[@]}; w += WAVE )); do
    wave=(); for (( i = w; i < w + WAVE && i < ${#names[@]}; i++ )); do wave+=("$i"); done
    [ -n "$(left "${wave[@]}")" ] || continue
    pids=(); for i in $(left "${wave[@]}"); do serve "$i" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { say "a server did not start ($V/serve.log)"; exit 1; }; done
    say "wave $(( w / WAVE + 1 )): serving $(for i in $(left "${wave[@]}"); do echo "${names[$i]} on :$(port "$i")"; done | paste -sd, -)"
    start=$(date +%s)
    while [ -n "$(left "${wave[@]}")" ]; do
        for i in $(left "${wave[@]}"); do
            n=${names[$i]} d=$HOME/runs/loop/val-${names[$i]}
            # the runner writes the round's records before the play has closed its harnesses
            if [ "$(records "$n")" -ge "$want" ]; then
                touch ~/exp/"$n"/val.done; log "$n" "val done"
                k=$(infra "$d"); [ "$k" = 0 ] || log "$n" "$k games ended as infrastructure failures (the turn's liveness guard): not scores of the model's"
                continue
            fi
            up "$(port "$i")" || { say "$n's server does not answer: starting it again"; serve "$i" || { say "it did not start"; exit 1; }; }
            play "$i"
            have=$(ls "$d"/captured 2>/dev/null | grep -c '\.json\.gz$')
            quiet=$(( $(date +%s) - $(moved "$d" "$start") ))
            if [ "${launches[$n]:-0}" -gt "${LAUNCHES:-4}" ] || [ "$quiet" -ge $(( ${LIVENESS:-180} * 60 )) ]; then
                # this play alone: the others play on
                for _ in 1 2 3 4 5; do pkill -9 -f "[l]oop/loop.py play .*--run-dir $d( |$)"; sleep 3; done
                if [ "${launches[$n]:-0}" -gt "${LAUNCHES:-4}" ]; then why="the play was launched ${launches[$n]} times ($d.log)"
                else why="no game and no model call ended for $(( quiet / 60 )) minutes"; fi
                log "$n" "FAILED: $why, $have/100 games ended; stopped, val left undone (a liveness failure, not a result)"
                touch ~/exp/"$n"/val.failed
            fi
        done
        [ -n "$(left "${wave[@]}")" ] && sleep 60
    done
    for i in "${wave[@]}"; do prune "$i"; done
done
cd ~/prime-rl || exit 1
failed=()
for i in "${!names[@]}"; do
    n=${names[$i]}
    prune "$i"
    [ -f ~/exp/"$n"/val.failed ] && { failed+=("$n"); continue; }
    [ -f ~/exp/"$n"/measures.done ] && continue
    uv run --no-sync python ~/loop/forward.py game-measures ~/runs/loop/val-"$n" --out ~/exp/"$n"/val-measures.json \
        > ~/exp/"$n"/measures.log 2>&1 || { log "$n" "measures failed (~/exp/$n/measures.log)"; continue; }
    log "$n" "validation: $(tail -1 ~/exp/"$n"/measures.log | cut -c1-400)"
    touch ~/exp/"$n"/measures.done; log "$n" "measures done"
done
pkill -f "[v]llm serve"
[ "${#failed[@]}" = 0 ] || { say "finished but for ${failed[*]}, whose plays failed (their progress): run again to play them again"; exit 1; }
say "finished"
