#!/usr/bin/env bash
# The model server on :8101 (vLLM 0.30, started by vllm-serve.sh beside this script), with adapters loadable at run
# time: started on GPUS as DP replicas unless it already serves there (a server on other GPUs is stopped first), then
# each adapter NAME loaded from its PATH when not yet served. MODEL defaults to Qwen3.5-9B, and SERVE_ARGS to the
# Qwen3.5 architecture's: the model card's qwen3 parsers and the language model only (the checkpoint is multimodal);
# for Qwen3.5 models also their chat template with the loop's reasoning_prefill (qwen35_prefill.jinja: what play
# --context-placement thought sends; without it, the template's own rendering) and speculative decoding with the
# checkpoint's own MTP head drafting two tokens a step (rejection sampling keeps every sampled token's distribution
# the target's, adapter and all: with a trained DAgger adapter on GPUs 0-3 tensor-parallel, real player calls decoded
# 1.84x faster one at a time and 1.53x four at a time, about two tokens accepted a step, greedy text unchanged), and
# for any other model its own template, so no thought placement. Another architecture needs its own SERVE_ARGS.
# The arguments of the last call are kept in $RUNS/loop/server.args, and the server's launch (GPUs, replicas, serve
# arguments) in $RUNS/loop/server.launch: a healthy server is kept only when it was launched as asked (one started
# before a template change would silently ignore reasoning_prefill).
# VLLM_VENV (default ~/vllm-0.30.0) is the vLLM environment vllm-serve.sh installs and runs, RUNS (default ~/runs)
# where the server's files go.
# Usage (on the node): bash ~/continual-arc/training/serve.sh GPUS DP [NAME PATH]...
set -uo pipefail
ARGS="$*"
GPUS=$1; DP=$2; shift 2
export PATH=$HOME/.local/bin:$PATH
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-$HOME/runs}
VLLM_VENV=${VLLM_VENV:-$HOME/vllm-0.30.0}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
case $MODEL in
    Qwen/Qwen3.5-*) TEMPLATE="--chat-template $HERE/qwen35_prefill.jinja --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}" ;;
    *) TEMPLATE="" ;;
esac
SERVE_ARGS=${SERVE_ARGS:---reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --chat-template-content-format string --language-model-only $TEMPLATE}
L=$RUNS/loop
mkdir -p "$L"
# one call at a time on a node: the real-time runs' supervisors can all find a dead server at once, and two calls
# starting a server would stop each other's. The server is started with the lock's descriptor closed: a server that
# inherited it would hold the lock as long as it ran, and every later call would wait for ever
exec 9>>"$L/serve.lock"; flock 9
# up: three tries ten seconds apart, so a server busy for a moment is never taken for a dead one and stopped
healthy() { for _ in 1 2 3; do curl -fsS -m 10 localhost:8101/health >/dev/null 2>&1 && return 0; sleep 10; done; return 1; }
# twelve adapters at once (--max-loras): three runs on one server each play on one while, OPSD training on-policy
# beside its play (realtime.py; train.py --on-policy), each publishes two of its student's, nine in use at once, and
# fewer slots would swap them in and out request by request; part of the launch, so a server started
# with fewer is replaced
LORA_ARGS="--enable-lora --max-lora-rank ${LORA_RANK:-128} --max-loras 12"
LAUNCH="$GPUS $DP $MODEL $SERVE_ARGS $LORA_ARGS"
if ! healthy || [ "$(cat $L/server.launch 2>/dev/null)" != "$LAUNCH" ]; then
    pid=$(cat "$RUNS/vllm-8101.pid" 2>/dev/null)
    # the pid file outlives a reboot, after which its pid may be any process: its group is stopped only while it is vLLM
    [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
    pkill -f "[v]llm serve"; sleep 5
    # the last server's log is kept (vllm-serve.sh starts the log afresh): its end says why it died
    [ -f "$RUNS/vllm-8101.log" ] && mv "$RUNS/vllm-8101.log" "$RUNS/vllm-8101.$(date -u +%m%d-%H%M%S).log"
    # one API frontend: vLLM 0.30 refuses run-time adapter loading with more than one
    (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True DP=$DP MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072} GPU_MEM=0.85 VLLM_VENV=$VLLM_VENV \
        RUNS=$RUNS bash "$HERE/vllm-serve.sh" "$MODEL" "$GPUS" 8101 $SERVE_ARGS $LORA_ARGS \
        --api-server-count 1 >> "$L/server.log" 2>&1 9>&-) || { echo "the server did not start ($L/server.log)"; exit 1; }
    echo "$LAUNCH" > $L/server.launch
fi
while [ $# -ge 2 ]; do
    NAME=$1; ADAPTER=$2; shift 2
    curl -fsS localhost:8101/v1/models | grep -q "\"$NAME\"" && continue
    # the directory vLLM loads it from: for Qwen3.5 a copy under the checkpoint's names (loop.py served_adapter)
    ADAPTER=$("$VLLM_VENV/bin/python" "$HERE/loop.py" vllm-adapter --model "$MODEL" "$ADAPTER") \
        || { echo "preparing $ADAPTER for vLLM failed"; exit 1; }
    curl -fsS -X POST localhost:8101/v1/load_lora_adapter -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"$NAME\", \"lora_path\": \"$ADAPTER\"}" >> $L/server.log 2>&1 \
        || { echo "adapter $NAME did not load from $ADAPTER ($L/server.log)"; exit 1; }
done
echo "$ARGS" > $L/server.args
