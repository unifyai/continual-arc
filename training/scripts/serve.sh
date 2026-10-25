#!/usr/bin/env bash
# The model server on :8101 (vLLM 0.30, started by vllm-serve.sh, with lib.sh's SERVE_ARGS), adapters loadable at run
# time: started on GPUS as DP replicas, each tensor-parallel over its share, unless it already serves there as asked
# (a server launched otherwise is stopped first), then each adapter NAME loaded from its PATH when not yet served. The
# server's launch (GPUs, replicas, arguments) is kept in $RUNS/loop/server.launch, and a healthy server is kept only when
# it was launched as asked; the arguments of the last call are in $RUNS/loop/server.args.
# Usage (on the node): bash serve.sh GPUS DP [NAME PATH]...
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ARGS="$*"
GPUS=$1; DP=$2; shift 2
L=$RUNS/loop
mkdir -p "$L"
# one call at a time on a node: the runs' supervisors can all find a dead server at once, and two calls starting a
# server would stop each other's. The server is started with the lock's descriptor closed: a server that inherited it
# would hold the lock as long as it ran, and every later call would wait for ever
exec 9>>"$L/serve.lock"; flock 9
LAUNCH="$GPUS $DP $MODEL ${SERVE_ARGS[*]}"
if ! up 8101 || [ "$(cat "$L/server.launch" 2>/dev/null)" != "$LAUNCH" ]; then
    stop_server 8101
    pkill -f "[v]llm serve"; sleep 5
    (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True DP=$DP MAX_MODEL_LEN=131072 GPU_MEM=0.85 VLLM_VENV=$VLLM_VENV RUNS=$RUNS \
        bash "$HERE/vllm-serve.sh" "$MODEL" "$GPUS" 8101 "${SERVE_ARGS[@]}" >> "$L/server.log" 2>&1 9>&-) \
        || { echo "the server did not start ($L/server.log)"; exit 1; }
    echo "$LAUNCH" > "$L/server.launch"
fi
while [ $# -ge 2 ]; do
    NAME=$1; ADAPTER=$2; shift 2
    curl -fsS localhost:8101/v1/models | grep -q "\"$NAME\"" && continue
    DIR=$(vllm_dir "$ADAPTER") || { echo "preparing $ADAPTER for vLLM failed"; exit 1; }
    curl -fsS -X POST localhost:8101/v1/load_lora_adapter -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"$NAME\", \"lora_path\": \"$DIR\"}" >> "$L/server.log" 2>&1 \
        || { echo "adapter $NAME did not load from $ADAPTER ($L/server.log)"; exit 1; }
done
echo "$ARGS" > "$L/server.args"
