#!/usr/bin/env bash
# grid-exp.sh NAME INIT TARGET [LR BATCH EPOCHS SET HELD]: one cell of the DAgger search's data-size grid on this node
# (the author, 19 Oct 2026), detached, resumable (a marker per stage in ~/exp/NAME):
#   pool      a label set (SET, default ~/exp/stage1/set: R's round-1 set, GLM-5.3-flash's reply at every call of
#             b6's 296 games), its games taken whole in a seeded order until they hold TARGET examples, the puzzles HELD
#             (default her three) the early stopping's (forward.py match, as forward.sh's comparison arm)
#   train     SFT from INIT (base: Qwen3.5-9B itself; b6: ~/runs/loop/b1-6/adapter) at her settings (LR 5e-5, batch 16,
#             2 passes, early stop on the held-out puzzles, patience 1), 8 GPUs
#   val       the 100-puzzle validation play on her schedule and aliases (~/runs/loop/sched-val.json, r0-val), served on
#             8 replicas, waited for until every game has ended and its records.jsonl is written, then its per-game
#             measures. No cutoff in time (the author, 25 Oct 2026: "time should never be a gating factor"): a play stopped
#             short writes no records.jsonl, since the runner records a round only whole (base-100pct's 60 of 99 left
#             none), and which games end first depends on how fast the model is served. Two guards, logged as failures
#             that leave the stage undone, so that launching the cell again plays it again: nothing has moved for
#             LIVENESS minutes (180 unless set; no game ended and no model call ended in that time, the play hung) or
#             the play had to be launched more than LAUNCHES times (4: a play that keeps dying). The play itself is
#             turn-based (loop.py play's default budget, 40 model calls an action and no seconds, where the code has it)
# INIT may be adapter:PATH (validate that adapter, no training) or none (validate the base model, no training).
# ~/exp/NAME/play.args, written before the launch, adds loop.py play options to the validation. UNTIL=train stops once
# the model is trained, for a machine that cannot play (search/generalisation's train stage, on Modal or a container
# without bubblewrap); run again without it, the cell goes on from its markers to the validation. WEIGHTING=turn or split
# trains with train.py's --weighting (every reply alike, or every reply alike with half on its reasoning and half on its
# move), on a bundle whose train.py takes it; unset, every token weighs alike.
set -uo pipefail
for k in "$(dirname "${BASH_SOURCE[0]}")"/{,../}keep.sh; do [ -f "$k" ] && . "$k" && break; done   # the run supervisor (keep.sh)
type keep_run > /dev/null 2>&1 || { echo "refused: no keep.sh beside $0, the run supervisor's helper" >&2; exit 1; }
NAME=$1 INIT=$2 TARGET=$3 LR=${4:-5e-5} BATCH=${5:-16} EPOCHS=${6:-2} SET=${7:-$HOME/exp/stage1/set}
HELD=${8:-7ddcd7ec,7e0986d6,f76d97a5}
O=~/exp/$NAME
# the cell under the supervisor (keep.sh): a part of the run that starts it (cell.py's), or a run of its own in
# ~/runs/exp/NAME, into which its folders here and its play gather as it ends
keep_run ~/runs/exp/"$NAME" --resume --log "$O/progress" --vars "NAME INIT TARGET LR BATCH EPOCHS SET HELD UNTIL WEIGHTING LIVENESS LAUNCHES" \
    -- "$@" || exit 1
keep_own ~/runs/exp/"$NAME" && trap 'keep_gather ~/runs/exp/"$NAME" exp "$O" play ~/runs/loop/val-"$NAME"' EXIT
mkdir -p "$O"
log() { echo "$(date -u +%m-%dT%H:%M) $NAME: $*" >> "$O/progress"; }
done_() { touch "$O/$1.done"; log "$1 done"; }
stage() { [ -f "$O/$1.done" ]; }
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/prime-rl || exit 1
PY="uv run --no-sync python"
# moved RUN SINCE: the newest of SINCE and the ends of RUN's games and model calls (captured/uNNN.json.gz, written as a
# game ends, and users/uNNN/calls.jsonl, a line as each model call ends), in seconds since the epoch
moved() {
    local t
    t=$(find "$1/captured" "$1/users" -maxdepth 2 \( -name 'u*.json.gz' -o -name calls.jsonl \) -printf '%T@\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d. -f1)
    echo $(( ${t:-0} > $2 ? ${t:-0} : $2 ))
}
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
log "started: INIT $INIT, TARGET $TARGET, LR $LR, batch $BATCH, $EPOCHS passes, set $SET, held $HELD, weighting ${WEIGHTING:-token}"
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
            --epochs "$EPOCHS" --patience 1 --max-minutes 1440 ${WEIGHTING:+--weighting "$WEIGHTING"} \
            --out "$O/model" > "$O/train.log" 2>&1 \
            || { log "training failed ($O/train.log)"; exit 1; }
        log "trained: $(cat "$O/model/DONE" | cut -c1-400)"
        done_ train
    fi
    [ "${UNTIL:-}" = train ] && { log "stopped once trained (UNTIL=train)"; exit 0; }
fi
if ! stage val; then
    gpus_free
    if [ -n "$adapter" ]; then bash ~/loop/serve.sh 0,1,2,3,4,5,6,7 8 "$model" "$adapter" >> "$O/serve.log" 2>&1
    else bash ~/loop/serve.sh 0,1,2,3,4,5,6,7 8 >> "$O/serve.log" 2>&1; fi || { log "the server did not start"; exit 1; }
    # O/play.args, when the launch left one: more loop.py play options for the validation (e.g. the stream's
    # --sampling {"temperature":1.0,"top_p":1.0,"top_k":-1}), one word each, as resume.sh passes them
    echo "--model $model --alias-from $HOME/runs/loop/r0-val $(cat "$O/play.args" 2>/dev/null)" > ~/runs/loop/val-"$NAME".args
    D=~/runs/loop/val-"$NAME" start=$(date +%s) launches=0
    while true; do
        bash ~/loop/resume.sh val-"$NAME":sched-val >> "$O/play.log.now" 2>&1; rc=$?
        grep -q "^relaunched" "$O/play.log.now" && launches=$(( launches + 1 ))
        cat "$O/play.log.now" >> "$O/play.log"; rm -f "$O/play.log.now"
        [ "$rc" = 0 ] && break
        [ "$rc" = 3 ] || { log "resume.sh failed ($rc)"; exit 1; }
        have=$(ls "$D"/captured 2>/dev/null | grep -c '\.json\.gz$')
        quiet=$(( $(date +%s) - $(moved "$D" "$start") ))
        if [ "$launches" -gt "${LAUNCHES:-4}" ] || [ "$quiet" -ge $(( ${LIVENESS:-180} * 60 )) ]; then
            for _ in 1 2 3 4 5; do pkill -9 -f "[l]oop/loop.py play"; pkill -9 -f "[r]esume.sh"; sleep 3; done
            if [ "$launches" -gt "${LAUNCHES:-4}" ]; then why="the play was launched $launches times ($O/play.log)"
            else why="no game and no model call ended for $(( quiet / 60 )) minutes"; fi
            log "FAILED: $why, $have/100 games ended; stopped, the stage left undone (a liveness failure, not a result)"
            exit 1
        fi
        sleep 60
    done
    done_ val
    infra=$(python3 -c "import json, sys; print(sum(bool((json.loads(l).get('resources') or {}).get('system_infra_failures')) for l in open(sys.argv[1])))" "$D/records.jsonl")
    [ "$infra" = 0 ] || log "$infra games ended as infrastructure failures (the turn's liveness guard): not scores of the model's"
fi
if ! stage measures; then
    $PY ~/loop/forward.py game-measures ~/runs/loop/val-"$NAME" --out "$O/val-measures.json" > "$O/measures.log" 2>&1 \
        || { log "measures failed ($O/measures.log)"; exit 1; }
    log "validation: $(tail -1 "$O/measures.log" | cut -c1-400)"
    done_ measures
fi
gpus_free
log "finished"
