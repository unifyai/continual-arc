#!/usr/bin/env bash
# frombase.sh RUN OUT GPUS EVAL_STEPS PATIENCE: longtrain.py --from-base on a node: DAgger's student trained once from the
# base model on every label of the real-time run ~/runs/rt/RUN, shuffled, with no phases, until its held-out loss turns,
# into ~/runs/rt/OUT, on the GPUs GPUS once nothing else holds them. Every setting is the run's last phase's, its held-out
# games among them, so the curve reads against the adapter the run ended with on the same games. It runs under the
# supervisor (keep.sh), so OUT is a run of its own, and on the node's own ~/loop, the code the run trained with. Run again
# after a stop, it goes on from its last checkpoint. The machine needs RUN's label sets and last phase
# (checks/manifests/train-RUN.txt, by search/generalisation/bucket.py get). PURPOSE (why, as detailed as a lease's) is
# required. MAX_MINUTES caps it, and a cap that stops it while its loss still falls leaves it unfinished: run it again.
# Beside it on the node: longtrain.py and what keep_stage leaves (keep.sh, runs.py, SYNCED.json, SYNCED.diff).
# Usage (on the node): cd ~ && PURPOSE="..." setsid -f nohup bash ~/frombase/frombase.sh dagger-u4-disjoint \
#   dagger-u4-scratch 2,3,6,7 200 8 > ~/runs/frombase-dagger-u4-scratch.out 2>&1 < /dev/null
set -uo pipefail
RUN=$1 OUT=$2 GPUS=$3 EVAL_STEPS=$4 PATIENCE=$5
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$T/keep.sh"
keep_purpose || exit 1
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$HOME/loop
free() {   # every GPU of GPUS with next to nothing on it
    local g
    for g in ${GPUS//,/ }; do
        [ "$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] || return 1
    done
}
until free; do sleep 30; done
mkdir -p "$HOME/runs/rt/$OUT"
n=$(( $(tr -cd , <<< "$GPUS" | wc -c) + 1 ))
cd ~/prime-rl && KEEP_BY=frombase.sh keep_exec "$HOME/runs/rt/$OUT" --resume --code "$T" --also "$HOME/loop" \
    --also "$HOME/ask-sd" --also "$HOME/continual-arc-baselines" -- uv run --no-sync python "$T/longtrain.py" \
    --run "$HOME/runs/rt/$RUN" --out "$HOME/runs/rt/$OUT" --gpus "$n" --from-base --eval-steps "$EVAL_STEPS" \
    --patience "$PATIENCE" --max-steps 1000000 --max-minutes "${MAX_MINUTES:-5760}"
