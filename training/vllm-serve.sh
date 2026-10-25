#!/usr/bin/env bash
# A plain vLLM OpenAI-compatible server from an environment of its own (VLLM_VENV, default ~/vllm-<version>), apart
# from prime-rl's, so the server runs vLLM 0.30 while the trainer's environment stays as prime-rl pins it (vLLM 0.28).
# serve.sh, frozen.sh and grid-vals.sh start every model server through it. Installs the environment on first
# use, then waits until the server answers /health (up to 30 minutes).
# Usage (on a node): bash vllm-serve.sh <model> <gpus, comma list> <port> [extra vllm serve arguments...]
#   VLLM_VERSION (default 0.30.0), MAX_MODEL_LEN (default 131072), GPU_MEM (default 0.90); log RUNS/vllm-<port>.log,
#   RUNS defaulting to ~/runs
#   DP (default 1): data-parallel replicas; the GPUs are split among them and each replica is tensor-parallel
#   over its share (a small model on many GPUs serves many users faster as replicas than as one sharded copy)
set -uo pipefail
model=$1; gpus=$2; port=$3; shift 3
VER=${VLLM_VERSION:-0.30.0}; V=${VLLM_VENV:-$HOME/vllm-$VER}
RUNS=${RUNS:-$HOME/runs}
# the environment's own bin first: flashinfer compiles its sampling kernels at the first warmup with the ninja installed
# there (without it every engine dies at warmup, "No such file or directory: 'ninja'")
export PATH="$V/bin:$HOME/.local/bin:$PATH"
[ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ] && export CUDA_HOME=/usr/local/cuda
mkdir -p "$RUNS"
if [ ! -x "$V/bin/vllm" ]; then
    # torch pinned with it: a fresh resolve took torch 2.14.0 beside torchvision 0.28.0 (built for 2.13), and the
    # server died at import ("operator torchvision::nms does not exist"); VLLM_TORCH="" leaves it to the resolver
    uv venv -q --python 3.12 "$V" && uv pip install -q --python "$V/bin/python" "vllm==$VER" ${VLLM_TORCH-"torch==2.13.0"} \
        || { echo "vllm $VER install failed"; exit 1; }
fi
[ "${INSTALL_ONLY:-0}" = 1 ] && { echo "vllm $VER at $V"; exit 0; }
if curl -fsS -m 3 "http://localhost:$port/health" >/dev/null 2>&1; then echo "port $port is already serving"; exit 0; fi
n=$(( $(echo "$gpus" | tr -cd , | wc -c) + 1 )); dp=${DP:-1}; tp=$(( n / dp ))
[ $(( tp * dp )) -eq "$n" ] || { echo "DP=$dp does not divide the $n GPUs"; exit 1; }
log=$RUNS/vllm-$port.log
CUDA_VISIBLE_DEVICES=$gpus setsid nohup "$V/bin/vllm" serve "$model" --port "$port" --tensor-parallel-size "$tp" \
    --data-parallel-size "$dp" --max-model-len "${MAX_MODEL_LEN:-131072}" --gpu-memory-utilization "${GPU_MEM:-0.90}" \
    "$@" > "$log" 2>&1 < /dev/null &
echo $! > "$RUNS/vllm-$port.pid"
for i in $(seq 1 180); do
    curl -fsS -m 3 "http://localhost:$port/health" >/dev/null 2>&1 && { echo "serving $model on port $port (GPUs $gpus, tp $tp, dp $dp)"; exit 0; }
    kill -0 "$(cat "$RUNS/vllm-$port.pid")" 2>/dev/null || { echo "the $model server died:"; tail -30 "$log"; exit 1; }
    sleep 10
done
echo "$model not serving after 30 minutes:"; tail -30 "$log"; exit 1
