#!/usr/bin/env bash
# The dev loop's round driver: the development set (loop.py schedule dev --round R: DEV_TASKS
# training tasks off arc1-small's stream and every epilogue, never the paper's own data) under the block's harness
# option (play --harness block: the 40-call cap, seconds raised to 1,740 s, just under the 1,800 s hard ceiling, on
# every play including the validation one), for either objective, on the node, resumable.
# Round k = FIRST_ROUND..FIRST_ROUND+ROUNDS-1:
#   - play k: the development stream's round FIRST_ROUND+k-1 (loop.py schedule dev --round: DEV_TASKS tasks x 4 users,
#     one instance each), played on all 8 GPUs by the newest adapter round(k-1) (the base model for k = 1);
#   - OPD only (OBJECTIVE=opd): stop the student's server, serve the teacher (serve_teacher.sh) and score the round's
#     games chain by chain (teacher_score.py --out, RL-03), then stop the teacher before training (score_opd below).
#     The scores land in round-play$k/opd_scores.json (written atomically, and only once teacher_score.py exits 0:
#     a failed or partly-dead scoring pass must not look done on the next resume), which train.py --objective opd
#     reads (train.opd_items) and refuses to train on if too much of it is missing;
#   - train k: train.py --objective $OBJECTIVE on round k's games (the server stopped first), as chains by default
#     (train.find_chains/chain_batches, item 8, needs ask-sd's per-token-advantages branch for both objectives: see
#     below) or, with PER_CALL=1, one item per call; one pass over all of the round's useful data (PG: every game
#     with an earlier one to weigh it against; OPD: every game's calls), clipped against round(k-1), into roundK; train.py's own --max-minutes
#     (TRAIN_MAX_MINUTES, forward.sh's 1,440-minute cap) stops an overlong pass, and the round's own DONE.time_capped
#     says so.
# After rounds in VAL_AFTER (default "3 5") the validation split is played by that round's adapter (round-val$k);
# VAL_AFTER always includes the last round asked for.
# Every stage is skipped once done (round$k/DONE, round-play$k.done, round-play$k/opd_scores.json, round-val$k.done),
# so a killed run resumes at the last finished stage: round.done at the very end.
# Inputs in ~/runs/loop: none beyond the base model and ~/prime-rl (no held-out task aliases are needed: the
# development set never meets the held-out test). Both objectives need ask-sd's per-token-advantages branch (RL-02,
# TokenItem.teacher_logp, TokenItem.advantage as a per-token list, and Trainer.policy_gradient_chains) on the node,
# ahead of its own merge to ask-sd main: chains are now the default path for pg as well as opd, and chain_item
# always sets ChainItem's newer fields for either. PER_CALL=1 is the only path that does not need it.
# Usage: OBJECTIVE=pg LR=1e-4 ROUNDS=5 bash ~/loop/round.sh
#   OBJECTIVE=opd TEACHER_MODEL=Qwen/Qwen3.8-27B LR=2e-5 ROUNDS=5 bash ~/loop/round.sh
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MODEL=${MODEL:-Qwen/Qwen3.5-9B}
L=~/runs/loop
OBJECTIVE=${OBJECTIVE:-pg}   # pg or opd (train.py --objective; both need ask-sd's per-token-advantages branch)
FIRST_ROUND=${FIRST_ROUND:-0}
ROUNDS=${ROUNDS:-5}            # the plan's "3 to 5 development rounds"
LR=${LR:?set LR, e.g. 2e-5 or 1e-4: the baselines plan grid}
BATCH=${BATCH:-64}
VAL_AFTER=${VAL_AFTER:-"3 5"}  # rounds after which the validation split is played
TRAIN_MAX_MINUTES=${TRAIN_MAX_MINUTES:-1440}   # forward.sh's cap: a safety limit, not a schedule
MAX_SEQ_LEN=${MAX_SEQ_LEN:-65536}   # train.py's own default; teacher_score.py scores to the same window (score_opd),
                                    # so the teacher never scores a call the trainer then throws away
PER_CALL=${PER_CALL:-0}        # 1: train.py --per-call (the equivalence check and fallback), in place of chains
HARNESS=${HARNESS:-block}      # play --harness: "block" is the RL baselines' option (decision 4); "default" reproduces
                                # the older loop's budget
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3.8-27B}   # OBJECTIVE=opd only (RL-03's serve_teacher.sh, teacher_score.py)
TEACHER_PORT=${TEACHER_PORT:-8102}
export TEACHER_PREFIX_CACHE=${TEACHER_PREFIX_CACHE:-off}   # off|align, serve_teacher.sh's own env var: exported,
                                                            # since serve_teacher.sh runs as a child bash (MODEL's
                                                            # own pattern); off unchanged
READ_PREFIX_CACHE=${READ_PREFIX_CACHE:-0}           # 1: teacher_score.py --read-prefix-cache for branch calls
SAMPLING='{"temperature":1.0,"top_p":1.0,"top_k":-1}'   # the full distribution the plan's shared settings ask for
ALL=0,1,2,3,4,5,6,7
say() { echo "$(date -u +%H:%M) round: $*" >> $L/progress; }
schedule() {   # schedule NAME ARGS...: write ~/runs/loop/NAME.json once
    [ -f "$L/$1.json" ] || (cd ~/prime-rl && uv run --no-sync python ~/loop/loop.py schedule "${@:2}" \
        --out "$L/$1.json") >> $L/progress 2>&1
}
plays() {   # plays NAME:SCHED ...: resume every listed play until each has one record per schedule slot
    while true; do
        bash ~/loop/resume.sh "$@" >> $L/progress 2>&1
        case $? in 0) return 0 ;; 3) sleep 120 ;; *) say "resume.sh failed"; return 1 ;; esac
    done
}
solved() {   # solved PLAY: "solved/records" of a play
    python3 -c "import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(f'{sum(r[\"solved\"] for r in rs)}/{len(rs)}')" \
        "$L/$1/records.jsonl" 2>/dev/null
}
stop_server() {   # the server off every GPU, before a block trains or the teacher serves there
    pid=$(cat ~/runs/vllm-8101.pid 2>/dev/null)
    [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null
    pkill -f "[v]llm serve"
    for i in $(seq 84); do
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
        [ "$busy" = 0 ] && return 0
        if [ "$i" = 24 ]; then   # 2 minutes on: a DP engine or worker slow to exit (the teacher's, 16 Oct)
            say "the GPUs still hold memory after 2 minutes: killing what is left of vLLM"
            pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"
        fi
        sleep 5
    done
    say "the GPUs did not free within 7 minutes: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"
    return 1
}
training() { pgrep -f "[t]rain.py .*--out $L/round$1( |$)" >/dev/null; }
play_round() {   # play_round K: round K's development stream play by round(K-1) (the base model for K = 1)
    local k=$1 model=$MODEL
    [ -f "$L/round-play$k.done" ] && return 0
    if [ "$k" = 1 ]; then
        bash ~/loop/serve.sh $ALL 8 || { say "server failed"; return 1; }
    else
        model=round$((k - 1))
        bash ~/loop/serve.sh $ALL 8 "$model" "$L/$model/adapter" || { say "server failed"; return 1; }
    fi
    echo "--model $model --sampling $SAMPLING --harness $HARNESS" > "$L/round-play$k.args"
    say "play round$k started ($model, dev round $((FIRST_ROUND + k - 1)))"
    plays "round-play$k:sched-round$k" || return 1
    touch "$L/round-play$k.done"; say "play round$k done: solved $(solved "round-play$k")"
}
score_opd() {   # score_opd K: round K's games scored by the teacher, chain by chain, into round-play$K/opd_scores.json
                # (OBJECTIVE=opd only; RL-03). The file is the stage's own resume marker (score_opd returns early once
                # it exists), so it is written at a temporary name and moved into place only once teacher_score.py
                # exits 0: a kill mid-score, or a teacher that scored too little of the round to pass its own 2%
                # check, must not look done on the next resume (teacher_score.py's own write is also atomic, but that
                # alone does not stop a *successful-looking* write from a failed run being mistaken for a good one).
                # Building OPD items from these scores and training on them is train_round's job (train.opd_items,
                # train.py --objective opd).
    local out="$L/round-play$1/opd_scores.json" tmp="$L/round-play$1/opd_scores.json.tmp" check=() cache_flag=()
    [ -f "$out" ] && return 0
    [ "$1" -gt 1 ] && check=(--skip-tokenizer-check)   # checked once, on round 1
    stop_server || return 1
    bash ~/loop/serve_teacher.sh $ALL "$TEACHER_PORT" || { say "teacher server failed"; return 1; }
    # serve_teacher.sh's own DP for this launch (~/runs/loop/teacher.dp, its own fixed path, not necessarily $L),
    # so a branch's read actually reaches the replica its trunk wrote to
    [ "$READ_PREFIX_CACHE" = 1 ] && cache_flag=(--read-prefix-cache \
        --dp-size "$(cat ~/runs/loop/teacher.dp 2>/dev/null || echo 1)")
    say "scoring round $1 with the teacher ($TEACHER_MODEL)"
    (cd ~/prime-rl && uv run --no-sync python ~/loop/teacher_score.py --vllm-url "http://localhost:$TEACHER_PORT/v1" \
        --model "$TEACHER_MODEL" --student-model "$MODEL" --train-runs "$L/round-play$1" --max-seq-len "$MAX_SEQ_LEN" \
        --out "$tmp" "${check[@]}" "${cache_flag[@]}" > "$L/score-round$1.log" 2>&1) \
        || { say "scoring round $1 failed ($L/score-round$1.log)"; rm -f "$tmp"; return 1; }
    mv "$tmp" "$out"
    pid=$(cat ~/runs/vllm-$TEACHER_PORT.pid 2>/dev/null)
    [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null
    pkill -f "[v]llm serve.*--port $TEACHER_PORT"
    say "round $1 scored: $(tail -2 "$L/score-round$1.log" | tr '\n' ' ')"
}
train_round() {   # train_round K: round K's training on all 8 GPUs, until its DONE (a relaunch waits for one running)
    local k=$1 init=() per_call=()
    [ -f "$L/round$k/DONE" ] && return 0
    [ "$PER_CALL" = 1 ] && per_call=(--per-call)
    if ! training "$k"; then
        stop_server || return 1
        [ "$k" -gt 1 ] && init=(--init-adapter "$L/round$((k - 1))/adapter")
        (cd ~/prime-rl && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup uv run --no-sync python \
            ~/loop/train.py --objective "$OBJECTIVE" --model "$MODEL" --gpus 8 --train-runs "$L/round-play$k" \
            --out "$L/round$k" --lr "$LR" --batch "$BATCH" --max-minutes "$TRAIN_MAX_MINUTES" \
            --max-seq-len "$MAX_SEQ_LEN" "${init[@]}" "${per_call[@]}" >> "$L/train-round$k.log" 2>&1 < /dev/null &)
        say "train round$k started"
        sleep 60
    fi
    while [ ! -f "$L/round$k/DONE" ]; do
        training "$k" || { sleep 15; [ -f "$L/round$k/DONE" ] || { say "train round$k ended without DONE ($L/train-round$k.log)"; return 1; }; }
        sleep 30
    done
    # a trainer that wrote DONE takes up to a minute to let go of its eight ranks, and the server starts next
    while pgrep -f "[t]rain.py " >/dev/null; do sleep 10; done
    say "train round$k done: $(python3 -c "
import json
d = json.load(open('$L/round$k/DONE'))
l = d['last'] or {}
capped = f\" (TIME-CAPPED after {d['minutes']} min)\" if d.get('time_capped') else ''
games = d.get('games')   # present for pg only (train_opd's own stats block is named 'items')
used = f'{games[\"wins\"]}/{games[\"episodes\"]} games won and trained on, {games[\"items\"]} calls, ' if games else ''
print(f'{used}{d[\"steps\"]} steps, clip {l.get(\"clip_fraction\")} kl {l.get(\"approx_kl\")} entropy {l.get(\"entropy_mean\")}{capped}')" 2>/dev/null)"
}
validate_round() {   # validate_round K: round K's adapter plays the validation split once, under the same harness as
                     # training (HARNESS): a different budget would make the selection signal score a play no paper
                     # run or training round actually uses
    [ -f "$L/round-val$1.done" ] && return 0
    bash ~/loop/serve.sh $ALL 8 "round$1" "$L/round$1/adapter" || { say "server failed"; return 1; }
    echo "--model round$1 --sampling $SAMPLING --harness $HARNESS" > "$L/round-val$1.args"
    say "validation play after round $1 started"
    plays "round-val$1:sched-val" || return 1
    touch "$L/round-val$1.done"; say "validation after round $1 done: solved $(solved "round-val$1")"
}
rounds() {   # rounds FROM TO: play and train rounds FROM..TO, then validate the ones VAL_AFTER asks for
    for ((k = $1; k <= $2; k++)); do
        schedule "sched-round$k" dev --round $((FIRST_ROUND + k - 1))
        if [ ! -f "$L/round$k/DONE" ]; then
            say "round $k"
            play_round "$k" || return 1
            if [ "$OBJECTIVE" = opd ]; then
                score_opd "$k" || return 1
            fi
            train_round "$k" || return 1
        fi
        if [[ " $VAL_AFTER $2 " == *" $k "* ]]; then
            validate_round "$k" || return 1
        fi
    done
}

{   # read whole before it runs: bash reads a script as it goes, and the code may be pushed again while this runs
[ -f $L/round.done ] && { say "already done"; exit 0; }
schedule sched-val validation
rounds 1 "$ROUNDS" || exit 1
pkill -f "[v]llm serve"
touch $L/round.done
say "done"
exit 0
}
