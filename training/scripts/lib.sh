# What the node scripts share, sourced by each: the folders, the model and its server's arguments, and the server's
# health and stop.
# Paths: HERE is the scripts' folder, TRAINING this package's root, SRC its source and BASELINES the baselines checkout
# (default the repository's baselines/); PRIME_RL (default ~/prime-rl) is the checkout whose environment runs the
# package (uv run --no-sync from it), VLLM_VENV (default ~/vllm-0.30.0) vLLM's own environment, and RUNS (default
# ~/runs) where every run writes.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAINING=$(dirname "$HERE")
SRC=$TRAINING/src
BASELINES=${BASELINES:-$(dirname "$TRAINING")/baselines}
PRIME_RL=${PRIME_RL:-$HOME/prime-rl}
VLLM_VENV=${VLLM_VENV:-$HOME/vllm-0.30.0}
RUNS=${RUNS:-$HOME/runs}
export PATH=$HOME/.local/bin:$PATH
MODEL=Qwen/Qwen3.5-9B
# The model server's arguments: the model card's reasoning and tool-call parsers, the language model alone (the
# checkpoint is multimodal), the chat template's content as one string, and speculative decoding with the checkpoint's
# own MTP head drafting two tokens a step. Rejection sampling keeps every sampled token's distribution the target's,
# adapter and all: with a trained adapter on four GPUs tensor-parallel, real calls decoded 1.84 times faster one at a time
# and 1.53 times four at a time, greedy text unchanged. Adapters load at run time, twelve at once at rank up to 128:
# runs that share a server each play on one while OPSD publishes two of its student's, and fewer slots would swap them in
# and out request by request. One API frontend, since vLLM 0.30 refuses run-time adapter loading with more.
SERVE_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --chat-template-content-format string --language-model-only
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
    --enable-lora --max-lora-rank 128 --max-loras 12 --api-server-count 1)

# up PORT: the server on PORT answers, three tries ten seconds apart, so a server busy for a moment is never taken for a
# dead one
up() { for _ in 1 2 3; do curl -fsS -m 10 "localhost:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done; return 1; }

# stop_server PORT: its server's process group stopped, while its pid is still vLLM's (the pid file outlives a reboot,
# after which its pid may be any process), and its log kept for why it stopped (vllm-serve.sh starts the log afresh)
stop_server() {
    local pid
    pid=$(cat "$RUNS/vllm-$1.pid" 2>/dev/null)
    [ -n "$pid" ] && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null && kill -- "-$pid" 2>/dev/null && sleep 15
    [ -f "$RUNS/vllm-$1.log" ] && mv "$RUNS/vllm-$1.log" "$RUNS/vllm-$1.$(date -u +%m%d-%H%M%S).log"
    return 0
}

# vllm_dir ADAPTER: the folder vLLM loads ADAPTER from (for Qwen3.5 a copy under the checkpoint's names), made with vLLM's
# own environment
vllm_dir() { PYTHONPATH=$SRC "$VLLM_VENV/bin/python" -m continual_arc_training.server vllm-adapter --model "$MODEL" "$1"; }
