#!/usr/bin/env bash
# grid-exp.sh NAME INIT TARGET [LR BATCH EPOCHS SET HELD]: one cell of the DAgger search's data-size grid on this node
# (the author, 19 Oct 2026), detached, resumable (a marker per stage in ~/exp/NAME):
#   pool      a label set (SET, default ~/exp/stage1/set: R's round-1 set, GLM-5.3-flash's reply at every call of
#             b6's 296 games), its games taken whole in a seeded order until they hold TARGET examples, the puzzles HELD
#             (default her three) the early stopping's (forward.py match, as forward.sh's comparison arm)
#   train     SFT from INIT (base: Qwen3.5-9B itself; b6: ~/runs/loop/b1-6/adapter) at her settings (LR 5e-5, batch 16,
#             2 passes, early stop on the held-out puzzles, patience 1), 8 GPUs
#   val       the 100-puzzle validation play on her schedule and aliases (~/runs/loop/sched-val.json, r0-val), served on 8
#             replicas, stopped 60 minutes after 95 games have ended or 6 hours in, then its per-game measures
# INIT may be adapter:PATH (validate that adapter, no training) or none (validate the base model, no training).
# ~/exp/NAME/play.args, written before the launch, adds loop.py play options to the validation.
set -uo pipefail
NAME=$1 INIT=$2 TARGET=$3 LR=${4:-5e-5} BATCH=${5:-16} EPOCHS=${6:-2} SET=${7:-$HOME/exp/stage1/set}
HELD=${8:-7ddcd7ec,7e0986d6,f76d97a5}
O=~/exp/$NAME
mkdir -p "$O"
log() { echo "$(date -u +%m-%dT%H:%M) $NAME: $*" >> "$O/progress"; }
done_() { touch "$O/$1.done"; log "$1 done"; }
stage() { [ -f "$O/$1.done" ]; }
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/prime-rl || exit 1
PY="uv run --no-sync python"
gpus_free() {
    pkill -f "[v]llm serve"; sleep 20
    for _ in $(seq 12); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 10; done
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 10
}
case $INIT in
    base) init=(); adapter=$O/model/adapter; model=$NAME ;;
    b6) init=(--init-adapter "$HOME/runs/loop/b1-6/adapter"); adapter=$O/model/adapter; model=$NAME ;;
    adapter:*) adapter=${INIT#adapter:}; model=$NAME ;;
    none) adapter=""; model=Qwen/Qwen3.5-9B ;;
    *) log "INIT must be base, b6, adapter:PATH or none"; exit 1 ;;
esac
log "started: INIT $INIT, TARGET $TARGET, LR $LR, batch $BATCH, $EPOCHS passes, set $SET, held $HELD"
if [ "$INIT" = base ] || [ "$INIT" = b6 ]; then
    if ! stage pool; then
        rm -rf "$O/pool"
        $PY ~/loop/forward.py match "$SET" --target "$TARGET" --held "$HELD" --seed 0 \
            --out "$O/pool" >> "$O/pool.log" 2>&1 || { log "pool failed ($O/pool.log)"; exit 1; }
        log "pool: $($PY ~/loop/forward.py count "$O/pool/train-all") training examples, $($PY ~/loop/forward.py count "$O/pool/dev-all") held out"
        done_ pool
    fi
    if ! stage train; then
        gpus_free
        rm -rf "$O/model"
        $PY ~/loop/train.py --model Qwen/Qwen3.5-9B --objective sft --gpus 8 "${init[@]}" \
            --train-runs "$O/pool/train-all" --val-run "$O/pool/dev-all" --lr "$LR" --batch "$BATCH" \
            --epochs "$EPOCHS" --patience 1 --max-minutes 1440 --out "$O/model" > "$O/train.log" 2>&1 \
            || { log "training failed ($O/train.log)"; exit 1; }
        log "trained: $(cat "$O/model/DONE" | cut -c1-400)"
        done_ train
    fi
fi
if ! stage val; then
    gpus_free
    if [ -n "$adapter" ]; then bash ~/loop/serve.sh 0,1,2,3,4,5,6,7 8 "$model" "$adapter" >> "$O/serve.log" 2>&1
    else bash ~/loop/serve.sh 0,1,2,3,4,5,6,7 8 >> "$O/serve.log" 2>&1; fi || { log "the server did not start"; exit 1; }
    # O/play.args, when the launch left one: more loop.py play options for the validation (e.g. the stream's
    # --sampling {"temperature":1.0,"top_p":1.0,"top_k":-1}), one word each, as resume.sh passes them
    echo "--model $model --alias-from /home/NODE_USER/runs/loop/r0-val $(cat "$O/play.args" 2>/dev/null)" > ~/runs/loop/val-"$NAME".args
    start=$(date +%s); since=""
    while true; do
        bash ~/loop/resume.sh val-"$NAME":sched-val >> "$O/play.log" 2>&1; rc=$?
        [ "$rc" = 0 ] && break
        [ "$rc" = 3 ] || { log "resume.sh failed ($rc)"; exit 1; }
        have=$(ls ~/runs/loop/val-"$NAME"/captured 2>/dev/null | wc -l)
        if [ -z "$since" ] && [ "$have" -ge 95 ]; then since=$(date +%s); log "$have/100 games ended: 60 minutes more"; fi
        if { [ -n "$since" ] && [ $(( $(date +%s) - since )) -ge 3600 ]; } || [ $(( $(date +%s) - start )) -ge 21600 ]; then
            for _ in 1 2 3 4 5; do pkill -9 -f "[l]oop/loop.py play"; pkill -9 -f "[r]esume.sh"; sleep 3; done
            log "stopped at the cutoff: $have/100 games ended"
            break
        fi
        sleep 60
    done
    done_ val
fi
if ! stage measures; then
    $PY ~/loop/forward.py game-measures ~/runs/loop/val-"$NAME" --out "$O/val-measures.json" > "$O/measures.log" 2>&1 \
        || { log "measures failed ($O/measures.log)"; exit 1; }
    log "validation: $(tail -1 "$O/measures.log" | cut -c1-400)"
    done_ measures
fi
gpus_free
log "finished"
