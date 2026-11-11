#!/usr/bin/env bash
# The loop's model server on :8101 (vLLM 0.30 through vllm-serve.sh), with adapters loadable at run time: started on
# GPUS as DP replicas unless it already serves there (a server on other GPUs is stopped first), then each adapter NAME
# loaded from its PATH when not yet served. MODEL defaults to Qwen3.5-9B, and SERVE_ARGS to the Qwen3.5 architecture's:
# the model card's qwen3 parsers and the language model only (the checkpoint is multimodal); for Qwen3.5 models also
# their chat template with the loop's reasoning_prefill (qwen35_prefill.jinja: what play --context-placement thought
# sends; without it, the template's own rendering) and speculative decoding with the checkpoint's own MTP head drafting
# two tokens a step (rejection sampling keeps every sampled token's distribution the target's, adapter and all: on 18
# Oct 2026, with a trained DAgger adapter on GPUs 0-3 tensor-parallel, real player calls decoded 1.84x faster one at a
# time and 1.53x four at a time, about two tokens accepted a step, greedy text unchanged), and for any other
# (Qwen3.8-27B) the model's own template, so no thought placement. Another architecture needs its own SERVE_ARGS (K2
# Horizon's: continual-arc-baselines/scripts/k2-serve.sh).
# The arguments of the last call are kept in ~/runs/loop/server.args, from which resume.sh restarts a server that died
# (with every adapter of that call), and the server's launch (GPUs, replicas, serve arguments) in
# ~/runs/loop/server.launch: a healthy server is kept only when it was launched as asked (1 Oct: one started before the
# template change would silently ignore reasoning_prefill).
# Usage (on the node): bash ~/loop/serve.sh GPUS DP [NAME PATH]...
set -uo pipefail
ARGS="$*"
GPUS=$1; DP=$2; shift 2
export PATH=$HOME/.local/bin:$PATH
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
case $MODEL in
    Qwen/Qwen3.5-*) TEMPLATE="--chat-template $HOME/loop/qwen35_prefill.jinja --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}" ;;
    *) TEMPLATE="" ;;
esac
SERVE_ARGS=${SERVE_ARGS:---reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --chat-template-content-format string --language-model-only $TEMPLATE}
L=~/runs/loop
mkdir -p $L   # a fresh node's first server (17 Oct: the paper's DAgger node had played no dev loop to make it)
# one call at a time on a node: the real-time runs' supervisors can all find a dead server at once, and two calls
# starting a server would stop each other's. The server is started with the lock's descriptor closed: a server that
# inherited it would hold the lock as long as it ran, and every later call would wait for ever (18 Oct 2026)
exec 9>>"$L/serve.lock"; flock 9
# up: three tries ten seconds apart, so a server busy for a moment is never taken for a dead one and stopped
healthy() { for _ in 1 2 3; do curl -fsS -m 10 localhost:8101/health >/dev/null 2>&1 && return 0; sleep 10; done; return 1; }
# twelve adapters at once (--max-loras): three runs on one server each play on one while, OPSD training on-policy
# beside its play (realtime.py; train.py --on-policy), each publishes two of its student's, nine in use at once, and
# fewer slots would swap them in and out request by request (19 Oct 2026); part of the launch, so a server started
# with fewer is replaced
LORA_ARGS="--enable-lora --max-lora-rank ${LORA_RANK:-128} --max-loras 12"
LAUNCH="$GPUS $DP $MODEL $SERVE_ARGS $LORA_ARGS"
if ! healthy || [ "$(cat $L/server.launch 2>/dev/null)" != "$LAUNCH" ]; then
    pid=$(cat ~/runs/vllm-8101.pid 2>/dev/null)
    # the pid file outlives a reboot, after which its pid may be any process: its group is stopped only while it is vLLM
    [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
    pkill -f "[v]llm serve"; sleep 5
    # the last server's log is kept (vllm-serve.sh starts the log afresh): its end says why it died
    [ -f ~/runs/vllm-8101.log ] && mv ~/runs/vllm-8101.log ~/runs/vllm-8101.$(date -u +%m%d-%H%M%S).log
    # one API frontend: vLLM 0.30 refuses run-time adapter loading with more than one (k2-serve.sh)
    (cd ~/continual-arc-baselines && VLLM_ALLOW_RUNTIME_LORA_UPDATING=True DP=$DP MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072} GPU_MEM=0.85 \
        bash scripts/vllm-serve.sh "$MODEL" "$GPUS" 8101 $SERVE_ARGS $LORA_ARGS \
        --api-server-count 1 >> $L/server.log 2>&1 9>&-) || { echo "the server did not start ($L/server.log)"; exit 1; }
    echo "$LAUNCH" > $L/server.launch
fi
while [ $# -ge 2 ]; do
    NAME=$1; ADAPTER=$2; shift 2
    curl -fsS localhost:8101/v1/models | grep -q "\"$NAME\"" && continue
    # the directory vLLM loads it from: for Qwen3.5 a copy under the checkpoint's names (loop.py served_adapter)
    ADAPTER=$(~/vllm-0.30.0/bin/python ~/loop/loop.py vllm-adapter --model "$MODEL" "$ADAPTER") \
        || { echo "preparing $ADAPTER for vLLM failed"; exit 1; }
    curl -fsS -X POST localhost:8101/v1/load_lora_adapter -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"$NAME\", \"lora_path\": \"$ADAPTER\"}" >> $L/server.log 2>&1 \
        || { echo "adapter $NAME did not load from $ADAPTER ($L/server.log)"; exit 1; }
done
echo "$ARGS" > $L/server.args
