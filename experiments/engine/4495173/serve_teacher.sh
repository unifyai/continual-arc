#!/usr/bin/env bash
# The OPD teacher's vLLM server, a different port from the student's :8101 so both can run on the same node between a
# round's play and its training. Frozen: no LoRA adapters, no chat template switch (teacher_score.py sends token ids
# straight to /v1/completions, so no template is ever read; Qwen3.8-27B's own template, left unused, would add a
# reasoning-effort line to the system prompt the student never saw). TEACHER_MODEL defaults to Qwen3.8-27B: 55.6 GB in
# BF16, tight on one 80 GB GPU. GPUS is vllm-serve.sh's own GPU list (e.g. "0,1,2,3,4,5,6,7"). DP (vLLM's
# data-parallel replica count) defaults to half the GPU count, so the teacher serves as several 2-GPU copies instead
# of one copy split across every GPU (vllm-serve.sh's own default, DP=1, which serve.sh also leaves at for the
# student): more copies in flight score more of a round's chains at once (teacher_score.py's concurrent --workers). A
# single GPU keeps DP=1. Override either with GPUS=N or DP=N. MAX_MODEL_LEN at least 131,072 (the longest recorded
# call: 129,482). Confirm these flags against continual-arc-baselines/scripts/vllm-serve.sh on the first node, as
# serve.sh's own comment asks for a new architecture (Qwen3.8-27B is `Qwen3_5ForConditionalGeneration`'s same class,
# so the node image should serve it as is).
#
# TEACHER_PREFIX_CACHE (off|align, default off): off passes none of the flags below, reproducing this script's
# server command from before this variable existed, exactly (vLLM 0.30.0 still defaults prefix caching on for this
# model either way, mamba_cache_mode "align"; off only means nothing ever asks to *read* it). align passes
# --enable-prefix-caching --mamba-cache-mode align explicitly (the teacher is hybrid: 16 full-attention layers and
# 48 linear-attention ones, so reusing the prefix also means reusing the linear layers' recurrent state; "align"
# checkpoints that state only at block-aligned scheduler-step boundaries). "all" (every block, not just at a step
# boundary) is not an accepted value here: vLLM 0.30.0 marks it deprecated and silently downgrades it to "align"
# for this model regardless, since Qwen3_5ForConditionalGeneration never sets supports_mamba_prefix_caching, so it
# would only restart the server for no change. align also installs the vLLM plugin (teacher_vllm_plugin/, `uv pip
# install -e`'d into vLLM's own environment) and sets TEACHER_READ_PREFIX_CACHE=1 in every process this script
# starts, which is what actually turns the plugin on (not vLLM's own VLLM_PLUGINS, an allowlist that would also
# stop this venv's LoRA resolver plugins from loading): a /v1/completions request can then opt into reading the
# cache per request, with vllm_xargs: {"read_prefix_cache": 1} (teacher_score.py's --read-prefix-cache); a request
# without that key is untouched either way, so turning this on changes nothing until a caller asks for it. Every
# invocation, off included, writes this launch's DP to ~/runs/loop/teacher.dp, which teacher_score.py's own
# --read-prefix-cache run reads to cross-check the DP it reads directly off the server's /metrics.
# Usage (on the node): bash ~/loop/serve_teacher.sh GPUS [PORT]
set -uo pipefail
GPUS=$1; PORT=${2:-8102}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this repo (teacher_vllm_plugin/)
export PATH=$HOME/.local/bin:$PATH
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3.8-27B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
# prompt_logprobs returns the full 248,320-entry vocabulary at every position: a lower ceiling than the student's
# server keeps that cost down per concurrent sequence, raised once the first round's throughput measurement
# (teacher_score.py) shows the server keeps up.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
TEACHER_PREFIX_CACHE=${TEACHER_PREFIX_CACHE:-off}
case "$TEACHER_PREFIX_CACHE" in
    off|align) ;;
    *) echo "TEACHER_PREFIX_CACHE must be off or align (got '$TEACHER_PREFIX_CACHE')" >&2; exit 1 ;;
esac
IFS=',' read -ra GPU_LIST <<< "$GPUS"
N=${#GPU_LIST[@]}
DP=${DP:-$(( N > 1 ? N / 2 : 1 ))}
L=~/runs/loop
VER=${VLLM_VERSION:-0.30.0}; V=$HOME/vllm-$VER   # vllm-serve.sh's own venv, read here only to install the plugin into it
healthy() { curl -fsS -m 3 "localhost:$PORT/health" >/dev/null 2>&1; }
if [ "$TEACHER_PREFIX_CACHE" = off ]; then
    CACHE_FLAGS=()   # nothing at all: this script's exact command from before TEACHER_PREFIX_CACHE existed
else
    CACHE_FLAGS=(--enable-prefix-caching --mamba-cache-mode "$TEACHER_PREFIX_CACHE")
    export TEACHER_READ_PREFIX_CACHE=1   # the plugin's own gate (teacher_vllm_plugin.py); not VLLM_PLUGINS, an
                                          # allowlist that would also stop this venv's LoRA resolver plugins loading
fi
LAUNCH="$GPUS $DP $TEACHER_MODEL $PORT $MAX_MODEL_LEN $TEACHER_PREFIX_CACHE"
if ! healthy || [ "$(cat $L/teacher.launch 2>/dev/null)" != "$LAUNCH" ]; then
    pid=$(cat "$HOME/runs/vllm-$PORT.pid" 2>/dev/null)
    [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null && sleep 15
    pkill -f "[v]llm serve.*--port $PORT"; sleep 5
    if [ "$TEACHER_PREFIX_CACHE" != off ]; then
        # vllm-serve.sh's own INSTALL_ONLY path: creates/updates the venv without launching anything, so it exists
        # for the plugin install below even on a node that has never served the teacher before.
        (cd ~/continual-arc-baselines && INSTALL_ONLY=1 bash scripts/vllm-serve.sh "$TEACHER_MODEL" "$GPUS" "$PORT" \
            >> $L/teacher.log 2>&1) || { echo "the teacher's vLLM environment did not install ($L/teacher.log)"; exit 1; }
        # the venv has no pip (vllm-serve.sh's own `uv venv`, no --seed); uv's own installer works on it directly,
        # and is idempotent across restarts on its own.
        uv pip install -q --python "$V/bin/python" -e "$REPO/teacher_vllm_plugin" \
            || { echo "the teacher's read-prefix-cache plugin failed to install"; exit 1; }
    fi
    (cd ~/continual-arc-baselines && DP=$DP GPU_MEM=0.85 MAX_MODEL_LEN=$MAX_MODEL_LEN \
        bash scripts/vllm-serve.sh "$TEACHER_MODEL" "$GPUS" "$PORT" --dtype bfloat16 \
        --max-num-seqs "$MAX_NUM_SEQS" --api-server-count 1 "${CACHE_FLAGS[@]}" >> $L/teacher.log 2>&1) \
        || { echo "the teacher server did not start ($L/teacher.log)"; exit 1; }
    echo "$LAUNCH" > $L/teacher.launch
fi
echo "$DP" > $L/teacher.dp   # always refreshed, even when the server was already up: round.sh/paper.sh's --dp-size
echo "teacher serving on :$PORT ($TEACHER_MODEL, $GPUS GPU(s), DP=$DP, context $MAX_MODEL_LEN, prefix-cache $TEACHER_PREFIX_CACHE)"
