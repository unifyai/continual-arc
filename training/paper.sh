#!/usr/bin/env bash
# The paper driver: one row of the block (H0, PG, the "should" row H or the bare model) through the real
# paper rows' schedule (arc1-small served to 100 users, workstreams, seed 0, 300 rounds: the stream's 200, then the
# epilogue's 100; paper.py's full_schedule), one round per runner call, on the node. USERS (default 100, the
# full schedule's number) plays a cut instead: arc1-small served to that many users, 0..USERS-1 exactly the 100's
# configuration's own first USERS users, round by round (paper.py's served). The paper's rows run at 1, 2 and 4
# users. This is the synchronous protocol: the paper's trained rows, DAgger, OPSD and PG, run through realtime.sh
# instead, with a fixed one-round lag. Round r = FIRST_ROUND, ...:
#   - serve: H0, H and the bare model always serve the base model (frozen: nothing to load); PG serves round r-1's
#     adapter (the base model at r = 0), named "$ROW-r$((r-1))" (a smoke cut's own "$ROW-u$USERS-r$((r-1))") so it
#     never collides with another row's or another USERS cut's adapters on the server;
#   - check (trained rows, r > 0 only): the served-adapter check (paper.py check-served) once the adapter is loaded,
#     so a layer-naming mistake (loop.served_adapter's Qwen3.5 fix) is caught before a whole round plays on it; an
#     adapter no step has touched (a round with no trainable item, e.g. PG's first, with no earlier game) is the base
#     model's weights by construction and is skipped, with a line in the progress log;
#   - play r: paper.py play --row $ROW --round r, over the real schedule cut at r+1 rounds (Schedule.first_rounds)
#     or, with EPILOGUE=1, the K-user epilogue cut the same way; resume=True restores every whole round
#     already in $PLAY_DIR/records.jsonl and plays only round r (H0 wiped before every instance via baselines'
#     stateless mode; PG and H keep prime-agent's own memory, so a returning puzzle can get cheaper once a
#     trained row learns to store and fetch its program). Every instance lands under
#     $PLAY_DIR/rounds/r$r/captured/, sharded by round so a round trains on its own games alone (paper.py's module
#     docstring: the paper rows' users all play again every round under the same user ids, so a flat captured/
#     would have a later round silently overwrite an earlier one's files);
#   - train r (PG only): train.py --objective pg, on round r's games alone
#     ($PLAY_DIR/rounds/r$r), one pass over all of the round's useful data; from round r-1's adapter, into
#     $TRAIN_DIR/r$r;
#   - save: train.py's own adapter + optimizer under $TRAIN_DIR/r$r; round r+1 serves it;
#   - prune: the rounds before r-1 lose their vLLM copy, and their whole adapter unless it serves a KEEP_EVERY-th
#     round (prune_adapters).
# Every stage is marked (round-play$r.done, train$r/DONE), so a kill at
# any step resumes there: continual_arc.runner.run's own resume never replays a
# finished round, and train_round never starts a finished round's training twice. H0, H and the bare model never
# train: for them this is play-only.
# The epilogue at round R (EPILOGUE_ROUND, default 20; EPILOGUE_USERS, default 10), after a stream run of ROUNDS=R
# (rounds 0..R-1: EPILOGUE_ROUND pairs with the stream call's own ROUNDS, not with the last round index) is trained
# (EPILOGUE=1; FIRST_ROUND/ROUNDS then count the epilogue's own rounds, 0-indexed, not the stream's): a second,
# separate run ($PLAY_DIR/$TRAIN_DIR below point at epilogue-r$EPILOGUE_ROUND/ instead), serving at its own round 0
# the adapter that would have served round R of the stream -- train/r(R-1), the newest one a ROUNDS=R stream call
# leaves behind (INIT_ADAPTER) -- and training on through its own rounds the same way for a trained row; H0 plays it
# wiped, with no training, as T's fresh reference. USERS carries through to the epilogue too (its own run and
# directories get the same "-uN" suffix as the stream's, so INIT_ADAPTER always names an adapter the same USERS
# trained): the epilogue of a smoke cut stays a cut of the same size.
# Inputs: none beyond the base model and prime-rl's checkout (PRIME_RL, default ~/prime-rl) with this repository's
# packages installed in its environment.
# Everything this writes lives on the node's own disk, under RUNS/paper (RUNS default ~/runs): back it up while it runs
# (paper_backup.sh, which copies a run's small files off the node and, with GCS=1, the whole run to a bucket), since a
# rented node's disk goes with the node.
# Usage: ROW=h0 ROUNDS=20 bash ~/continual-arc/training/paper.sh
#   ROW=pg LR=1e-4 ROUNDS=20 bash ~/continual-arc/training/paper.sh
#   ROW=pg LR=1e-4 EPILOGUE=1 EPILOGUE_ROUND=20 EPILOGUE_USERS=10 ROUNDS=25 bash ~/continual-arc/training/paper.sh
#   ROW=h0 EPILOGUE=1 EPILOGUE_ROUND=20 EPILOGUE_USERS=10 ROUNDS=25 bash ~/continual-arc/training/paper.sh
# Transfer's two references for the trained runs at 1, 2 and 4 users (paper.py's module docstring): H0 (the learner
# reset) and the bare model (the floor), each on the epilogue alone for users 0-3, its 400 instances dealt WIDTH to a
# round (paper.py's wide_epilogue_cut: neither row keeps anything between instances), into
# RUNS/paper/$ROW-u4/epilogue-u4-w100 (a frozen row carries nothing from the stream, so its epilogue has no round R):
#   ROW=h0 USERS=4 EPILOGUE=1 EPILOGUE_USERS=4 WIDTH=100 ROUNDS=4 bash ~/continual-arc/training/paper.sh
#   ROW=bare USERS=4 EPILOGUE=1 EPILOGUE_USERS=4 WIDTH=100 ROUNDS=4 bash ~/continual-arc/training/paper.sh
#
# A GPU smoke test: H0 and PG on a 4-user cut of the paper rows' stream, 3 rounds, each a single node.
#   ROW=h0 USERS=4 ROUNDS=3 bash ~/continual-arc/training/paper.sh
#   ROW=pg LR=1e-4 USERS=4 ROUNDS=3 bash ~/continual-arc/training/paper.sh
# Check for each: the served-adapter check passes every round after the first; $PLAY_DIR/records.jsonl has exactly
# 4 x 3 records after round 2 (not 100 x 3: USERS=4 plays the real configuration's own first 4 users, round by
# round, never a different draw); a kill mid-round (during play_round) resumes at the same round, not the next; a
# kill mid-training (during train_round) resumes training, not a re-play. Then, with both rows' round 2
# adapters saved (ROUNDS=3: rounds 0, 1 and 2, the newest at train/r2):
#   ROW=pg LR=1e-4 USERS=4 EPILOGUE=1 EPILOGUE_ROUND=3 EPILOGUE_USERS=2 ROUNDS=1 bash ~/continual-arc/training/paper.sh
#   ROW=h0 USERS=4 EPILOGUE=1 EPILOGUE_ROUND=3 EPILOGUE_USERS=2 ROUNDS=1 bash ~/continual-arc/training/paper.sh
# play the 2-user epilogue's first round, PG starting from round 2's adapter (train/r2, EPILOGUE_ROUND-1) and H0
# wiped.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MODEL=${MODEL:-Qwen/Qwen3.5-9B}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRIME_RL=${PRIME_RL:-$HOME/prime-rl}
RUNS=${RUNS:-$HOME/runs}
export RUNS   # serve.sh keeps its files under RUNS too
ROW=${ROW:?set ROW: h0, pg, h or bare}
case $ROW in h0 | pg | h | bare) ;; *) echo "ROW must be h0, pg, h or bare (dagger and opsd: realtime.sh), got $ROW"; exit 1 ;; esac
FIRST_ROUND=${FIRST_ROUND:-0}
ROUNDS=${ROUNDS:?set ROUNDS: how many rounds this call plays, from FIRST_ROUND}
LR=${LR:-}
BATCH=${BATCH:-64}
TRAIN_MAX_MINUTES=${TRAIN_MAX_MINUTES:-1440}   # a safety cap, not a schedule
MAX_SEQ_LEN=${MAX_SEQ_LEN:-65536}   # train.py's own default
PER_CALL=${PER_CALL:-0}
EPILOGUE=${EPILOGUE:-0}
EPILOGUE_ROUND=${EPILOGUE_ROUND:-20}
EPILOGUE_USERS=${EPILOGUE_USERS:-10}
WIDTH=${WIDTH:-}   # EPILOGUE=1, ROW=h0 or bare: the epilogue alone's instances dealt WIDTH to a round, ROUNDS counting those
USERS=${USERS:-100}   # the paper rows' 100 (paper.py's PAPER_USERS); fewer (paper.py play --users) for a cheap smoke
ALL=0,1,2,3,4,5,6,7
# the server's GPUs and replicas, and the trainer's GPUs: all eight each by default, the server stopped for every
# round's training. With TRAIN_GPUS none of SERVE_GPUS', training runs beside the server and it stays up, so several
# cuts can share a node and its one server (each cut's own adapter names and directories): e.g. SERVE_GPUS=0,1,2,3
# SERVE_DP=4 and one trainer GPU per cut (three cuts at 1, 2 and 4 users on one node)
SERVE_GPUS=${SERVE_GPUS:-$ALL}; SERVE_DP=${SERVE_DP:-8}
TRAIN_GPUS=${TRAIN_GPUS:-$ALL}
case $ROW in pg) TRAINED=1 ;; *) TRAINED=0 ;; esac
if [ "$TRAINED" = 1 ] && [ -z "$LR" ]; then echo "ROW=$ROW trains: set LR (the tuned rate for this objective)"; exit 1; fi
# epilogue_for_users refuses EPILOGUE_USERS > the configuration's own users; fail here, not partway into a play
if [ "$EPILOGUE" = 1 ] && [ "$EPILOGUE_USERS" -gt "$USERS" ]; then
    echo "EPILOGUE_USERS ($EPILOGUE_USERS) must not be more than USERS ($USERS)"; exit 1
fi

SUFFIX=""; [ "$USERS" = 100 ] || SUFFIX="-u$USERS"   # a cut's own run, play and train directories, logs and
                                                      # adapter names, distinct from the 100-user paper run's
L=$RUNS/paper/$ROW$SUFFIX
if [ "$EPILOGUE" = 1 ]; then
    PLAY_DIR=$L/epilogue-r$EPILOGUE_ROUND$SUFFIX
    [ "$TRAINED" = 1 ] || PLAY_DIR=$L/epilogue$SUFFIX${WIDTH:+-w$WIDTH}   # a frozen row's starts from nothing, at
                                                                         # no round R
    TRAIN_DIR=$L/epilogue-train-r$EPILOGUE_ROUND$SUFFIX
    INIT_ADAPTER=$L/train$SUFFIX/r$((EPILOGUE_ROUND - 1))/adapter   # the adapter that would have served round R
                                                                    # (same USERS); unused for h0/h
    SCHED_FLAGS="--epilogue-users $EPILOGUE_USERS --users $USERS${WIDTH:+ --width $WIDTH}"
    ADAPTER_PREFIX=$ROW-epi$EPILOGUE_ROUND$SUFFIX-r   # distinct from the main stream's "$ROW-rN" names on the server
else
    PLAY_DIR=$L/play$SUFFIX
    TRAIN_DIR=$L/train$SUFFIX
    SCHED_FLAGS="--users $USERS"
    ADAPTER_PREFIX=$ROW$SUFFIX-r
fi
mkdir -p $L
say() { echo "$(date -u +%H:%M) paper($ROW$SUFFIX): $*" >> $L/progress; }
round_dir() { printf "$PLAY_DIR/rounds/r%04d" "$1"; }   # paper.py's round_dir: must match it exactly
stop_server() {
    pid=$(cat "$RUNS/vllm-8101.pid" 2>/dev/null)
    [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null
    pkill -f "[v]llm serve"
    for i in $(seq 84); do
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
        [ "$busy" = 0 ] && return 0
        if [ "$i" = 24 ]; then   # 2 minutes on: a DP engine or worker slow to exit
            say "the GPUs still hold memory after 2 minutes: killing what is left of vLLM"
            pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"
        fi
        sleep 5
    done
    say "the GPUs did not free within 7 minutes: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"
    return 1
}
training() { pgrep -f "[t]rain.py .*--out $(round_to_train_dir "$1")( |$)" >/dev/null; }
apart() {   # the trainer's GPUs are none of the server's, so training runs beside the server, which stays up
    local g
    for g in ${TRAIN_GPUS//,/ }; do case ",$SERVE_GPUS," in *",$g,"*) return 1 ;; esac; done
}
round_to_train_dir() { echo "$TRAIN_DIR/r$1"; }
KEEP_EVERY=${KEEP_EVERY:-20}   # every 20th round's serving adapter stays whole, resume files and all: the one an
                               # epilogue at round 20, 40, ... starts from (INIT_ADAPTER), as realtime.py keeps its own
prune_adapters() {   # prune_adapters R, once round R has trained: frees what the rounds before R-1 no longer need (4.4 GB a
                   # round on Qwen3.5-9B at rank 128, so 200 rounds of a cut, or several cuts on one node, outgrow its
                   # disk): each one's vLLM copy, and its whole adapter unless it serves a KEEP_EVERY-th round. DONE and
                   # phase.json, every round's record of its training, stay. Play never names an older adapter again:
                   # round R+1 serves round R's, and a resumed round R serves round R-1's, both kept.
    local k
    for ((k = 0; k < $1 - 1; k++)); do
        rm -rf "$TRAIN_DIR/r$k/adapter.vllm"
        (((k + 1) % KEEP_EVERY)) && rm -rf "$TRAIN_DIR/r$k/adapter"
    done
    return 0
}

serve_round() {   # serve_round R: the adapter round R's play needs: round (R-1)'s own for PG (the epilogue's
                  # round 0 instead needs the main stream's round-(EPILOGUE_ROUND-1) adapter, INIT_ADAPTER: nothing
                  # is frozen for a trained row there), the base model for H0/H always and for PG's own round 0
    local r=$1 name path
    case $ROW in
    h0 | h | bare) bash "$HERE/serve.sh" $SERVE_GPUS $SERVE_DP || { say "server failed"; return 1; }; echo "$MODEL" ;;
    *)
        if [ "$r" = 0 ] && [ "$EPILOGUE" = 0 ]; then
            bash "$HERE/serve.sh" $SERVE_GPUS $SERVE_DP || { say "server failed"; return 1; }
            echo "$MODEL"; return 0
        elif [ "$r" = 0 ]; then
            name=$ROW$SUFFIX-r$((EPILOGUE_ROUND - 1))
        else
            name=$ADAPTER_PREFIX$((r - 1))
        fi
        path=$(adapter_of "$r")
        bash "$HERE/serve.sh" $SERVE_GPUS $SERVE_DP "$name" "$path" || { say "server failed"; return 1; }
        echo "$name"
        ;;
    esac
}

adapter_of() {   # adapter_of R: the adapter directory serve_round loads for a trained row's round R > 0, or the
                 # epilogue's round 0 (INIT_ADAPTER)
    if [ "$1" = 0 ]; then echo "$INIT_ADAPTER"; else echo "$TRAIN_DIR/r$(($1 - 1))/adapter"; fi
}

check_served() {   # check_served NAME R: the served-adapter check (skipped for the base model itself: nothing to check;
                   # paper.py skips an adapter no step has touched, which is the base model's weights by construction)
    local name=$1 r=$2
    [ "$name" = "$MODEL" ] && return 0
    (cd "$PRIME_RL" && uv run --no-sync python "$HERE/paper.py" check-served --vllm-url http://localhost:8101/v1 \
        --base-model "$MODEL" --served "$name" --adapter "$(adapter_of "$r")") >> $L/progress 2>&1
}

play_round() {   # play_round R: play round R of $ROW's schedule under the served model serve_round gave. A crashed or
                 # killed launch is retried (relaunched only when nothing is running and the round is still short),
                 # up to twice, before the round fails for good.
    local r=$1 served pattern="[p]aper.py play .*--run-dir $PLAY_DIR .*--round $r( |\$)" have=0 want=1 attempt mem
    [ -f "$PLAY_DIR/round-play$r.done" ] && return 0
    served=$(serve_round "$r") || return 1
    check_served "$served" "$r" || { say "served-adapter check failed for round $r ($served)"; return 1; }
    for attempt in 1 2 3; do
        if ! pgrep -f "$pattern" >/dev/null; then
            say "play round $r started ($served, attempt $attempt)"
            (cd "$PRIME_RL" && setsid nohup uv run --no-sync python "$HERE/paper.py" play --row "$ROW" \
                --run-dir "$PLAY_DIR" --base-url http://localhost:8101/v1 --served "$served" --round "$r" \
                $SCHED_FLAGS >> $L/play-round$r.log 2>&1 < /dev/null &)
            sleep 60   # a startup grace: paper.py's own imports (continual_arc, torch-free here) take a few seconds,
                       # and polling pgrep right after the fork can catch it before the exec lands
        fi
        while pgrep -f "$pattern" >/dev/null; do sleep 15; done
        # the runner writes schedule.json for this call's own cut before playing a single instance; its absence
        # means the launch above died before that, which is certainly short and worth another attempt
        if [ -f "$PLAY_DIR/schedule.json" ]; then
            want=$(python3 -c "import json;print(len(json.load(open('$PLAY_DIR/schedule.json'))['slots']))") || return 1
        fi
        have=$([ -f "$PLAY_DIR/records.jsonl" ] && wc -l < "$PLAY_DIR/records.jsonl" || echo 0)
        [ "$have" -ge "$want" ] && break
        say "play round $r short ($have/$want) after attempt $attempt: see $L/play-round$r.log"
    done
    [ "$have" -ge "$want" ] || { say "play round $r failed after $attempt attempts"; return 1; }
    touch "$PLAY_DIR/round-play$r.done"; say "play round $r done ($have/$want)"
    # memory.py's round_report, written by paper.py play itself (every row alike, H0's wipe included): one
    # line in the progress log beside play's own. memory.json's own write already failed independently and never
    # cost the round its result if it is missing (paper._report_round_memory logs why and returns), so this must
    # never fail play_round either: the test sits in an if, and the function ends on an explicit return 0.
    mem=$(round_dir "$r")/memory.json
    if [ -f "$mem" ]; then
        say "$(PYTHONPATH="$HERE" python3 -c "
import json
import memory
print(memory.summary_line(json.load(open('$mem'))))")"
    fi
    return 0
}

train_round() {   # train_round R: round R's training (trained rows only), from round R-1's adapter
    local r=$1 out init=() runs=() flags=()
    out=$TRAIN_DIR/r$r
    if [ -f "$out/DONE" ]; then
        # a time-capped DONE is not a clean one pass over the round's data: retrain rather than call it done
        python3 -c "import json,sys; sys.exit(1 if json.load(open('$out/DONE')).get('time_capped') else 0)" \
            && return 0
        say "train round $r's earlier attempt was time-capped: retraining"
    fi
    # the round's own games, read as chains unless PER_CALL=1
    runs=("$(round_dir "$r")")
    [ "$PER_CALL" = 1 ] && flags=(--per-call)
    # a crashed or killed training is relaunched (play_round's shape), up to twice, before the round fails for good;
    # a relaunch starts the round's one pass over from the same starting adapter, so it trains exactly what a clean
    # first attempt would have
    for attempt in 1 2 3; do
        if ! training "$r"; then
            apart || stop_server || return 1
            if [ "$r" = 0 ]; then
                # the main stream's round 0 starts from the base model; the epilogue's round 0 starts from the stream's
                # round-R adapter instead (INIT_ADAPTER), since nothing is frozen for a trained row in the epilogue
                [ "$EPILOGUE" = 1 ] && init=(--init-adapter "$INIT_ADAPTER")
            else
                init=(--init-adapter "$TRAIN_DIR/r$((r - 1))/adapter")
            fi
            mkdir -p "$out"
            (cd "$PRIME_RL" && CUDA_VISIBLE_DEVICES=$TRAIN_GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                setsid nohup uv run --no-sync python \
                "$HERE/train.py" --objective pg --model "$MODEL" --gpus "$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)" \
                --train-runs "${runs[@]}" --out "$out" --lr "$LR" --batch "$BATCH" \
                --max-minutes "$TRAIN_MAX_MINUTES" --max-seq-len "$MAX_SEQ_LEN" \
                "${init[@]}" "${flags[@]}" >> "$L/train-round$r.log" 2>&1 < /dev/null &)
            say "train round $r started (attempt $attempt)"
            sleep 60
        fi
        while [ ! -f "$out/DONE" ] && training "$r"; do sleep 30; done
        [ -f "$out/DONE" ] || sleep 15   # DONE lands as train.py exits
        [ -f "$out/DONE" ] && break
        say "train round $r ended without DONE after attempt $attempt ($L/train-round$r.log)"
    done
    [ -f "$out/DONE" ] || { say "train round $r failed after $attempt attempts"; return 1; }
    while training "$r"; do sleep 10; done
    say "train round $r done: $(python3 -c "
import json
d = json.load(open('$out/DONE'))
l = d['last'] or {}
capped = f\" (TIME-CAPPED after {d['minutes']} min)\" if d.get('time_capped') else ''
games = d.get('games')     # the stats block of pg: the games, the won ones trained on
items = d.get('items')     # of opd: calls, items (kept), excluded, unscored, too_long
if games:
    block = f'{games[\"wins\"]}/{games[\"episodes\"]} games won and trained on, {games[\"items\"]} calls, '
elif items:
    block = (f'{items[\"items\"]}/{items[\"calls\"]} calls kept, {items.get(\"excluded\", 0)} excluded, '
             f'{items[\"unscored\"]} unscored, {items[\"too_long\"]} too long, ')
else:
    block = ''
print(f'{block}{d[\"steps\"]} steps, clip {l.get(\"clip_fraction\")} kl {l.get(\"approx_kl\")} '
      f'entropy {l.get(\"entropy_mean\")}{capped}')" 2>/dev/null)"
    # one pass over all of the round's useful data is the contract: a time-capped pass trained on only part of it,
    # so the round must not count as a clean one
    python3 -c "import json,sys; sys.exit(1 if json.load(open('$out/DONE')).get('time_capped') else 0)" \
        || { say "train round $r time-capped at $TRAIN_MAX_MINUTES min: not one full pass, failing the round"; return 1; }
}

rounds() {   # rounds FROM TO: play, and for a trained row train, every
            # round FROM..TO. Each stage checks its own "done" marker and returns at once when it
            # is already met, so always calling all of them here (rather than guessing from the markers first) costs
            # nothing on an already-finished round and never mistakes a time-capped train_round for one that need
            # not be called again.
    for ((r = $1; r <= $2; r++)); do
        say "round $r"
        play_round "$r" || return 1
        if [ "$TRAINED" = 1 ]; then
            train_round "$r" || return 1
            prune_adapters "$r"
        fi
    done
}

{   # read whole before it runs: bash reads a script as it goes, and the code may be pushed again while this runs
    LAST=$((FIRST_ROUND + ROUNDS - 1))
    [ -f "$PLAY_DIR/paper-$FIRST_ROUND-$LAST.done" ] && { say "already done"; exit 0; }
    rounds "$FIRST_ROUND" "$LAST" || exit 1
    # the server is shared with any other cut or run on this node: the last to finish stops it (each cut runs in a
    # process group of its own, its subshells inside it, so another cut is another group running paper.sh)
    ps -eo pgid=,args= | awk '$2 == "bash" && $3 ~ /\/paper\.sh$/ {print $1}' | sort -u | grep -vqx "$(ps -o pgid= $$ | tr -d ' ')" \
        || pgrep -f "[r]ealtime.py" >/dev/null || pkill -f "[v]llm serve"
    touch "$PLAY_DIR/paper-$FIRST_ROUND-$LAST.done"
    say "done"
    exit 0
}
