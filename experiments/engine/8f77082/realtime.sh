#!/usr/bin/env bash
# One of the paper's trained rows, DAgger, OPSD or PG (ROW), with a fixed lag (realtime.py's module docstring): round r
# is played by the model trained on rounds 0..r-1-LAG (LAG=1, the paper's; LAG=0, synchronous), play waiting for it
# when training is behind and training never waiting for play. The oracle's relay (DAgger and OPSD) and the model
# server stay up for the whole run, each started again within a pass when it is found down (the server with the adapter
# the run's round plays on), and realtime.py's player, labeller (DAgger and OPSD) and trainer run side by side, each
# restarted when it dies (every one resumes from its files) up to MAX_RESTARTS times, the run stopped once all of them
# are done. The server holds SERVE_GPUS (one copy tensor-parallel over them with SERVE_DP=1, the fastest per call at
# the few games a run keeps in flight), the trainer TRAIN_GPUS. Runs can share a node: the same SERVE_GPUS, so one
# server and one relay serve them all, and TRAIN_GPUS of their own. USERS serves arc1-small to that many users
# (paper.py's served: the benchmark's canonical numbers are 1, 2, 4, 8 and 16), ROUNDS rounds of it (all 200 by
# default). Each row trains at LR 1e-4 (DAgger's and PG's rate, and b6's for self-distillation).
# Inputs: the base model, ~/prime-rl, and for DAgger and OPSD the OpenRouter key bootstrap wrote to
# ~/continual-arc-baselines/.env, which the relay reads (AGENTS.md's dated exception).
# Usage (on the node): ROW=dagger USERS=1 LR=1e-4 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 \
#     setsid nohup bash ~/loop/realtime.sh > /dev/null 2>&1 < /dev/null &
#   progress: ~/runs/rt/$NAME/progress ($ROW-u$USERS-lag$LAG); stop: the run's realtime.sh, then its loops' process
#   groups (a pkill pattern in an ssh command line matches that command line too: run it from a script on the node)
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MODEL=${MODEL:-Qwen/Qwen3.5-9B}
ROW=${ROW:?set ROW: dagger, opsd or pg}
case $ROW in dagger | opsd | pg) ;; *) echo "ROW must be dagger, opsd or pg, got $ROW"; exit 1 ;; esac
USERS=${USERS:?set USERS: how many users to serve the stream to, 1 2 4 8 or 16}
ROUNDS=${ROUNDS:-200}
LR=${LR:?set LR: the learning rate of the trainer}
LAG=${LAG:-1}
NAME=${NAME:-$ROW-u$USERS-lag$LAG}
SERVE_GPUS=${SERVE_GPUS:-0,1,2,3}; SERVE_DP=${SERVE_DP:-1}
TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7}
# the label step (DAgger and OPSD): GLM-5.3-flash at medium effort through the relay, its reasoning required (asked
# again up to NEED_THINKING times), as the synchronous row's; DAgger and OPSD hold one game in HOLD_EVERY out for the
# held-out loss
LABEL_MODEL=${LABEL_MODEL:-z-ai/glm-5.3-flash}; LABEL_EFFORT=${LABEL_EFFORT:-medium}
RELAY_PORT=${RELAY_PORT:-18230}; RELAY_PROVIDERS=${RELAY_PROVIDERS:-gmicloud,parasail,fireworks}
NEED_THINKING=${NEED_THINKING:-4}; MIN_LABELLED_PCT=${MIN_LABELLED_PCT:-40}
LABEL_ROUND_BUDGET_USD=${LABEL_ROUND_BUDGET_USD:-20}; LABEL_RUN_BUDGET_USD=${LABEL_RUN_BUDGET_USD:-3000}
HOLD_EVERY=${HOLD_EVERY:-10}
# a DAgger or OPSD phase: from the last phase's best adapter on every labelled round's training side, the held-out loss
# measured every EVAL_STEPS steps, stopped once PATIENCE measurements in a row have not improved on it, at most
# MAX_STEPS steps. Two, not one (18 Oct 2026): with few users the held-out set is a game or two for the first rounds,
# and one noisy measurement ended every phase of the 1-user run and kept its starting adapter. Every 5 steps, not 10
# (19 Oct 2026): the best came at step 10 and the phase ran 20 steps more to confirm it, so a phase outlasted a round
EVAL_STEPS=${EVAL_STEPS:-5}; PATIENCE=${PATIENCE:-2}; MAX_STEPS=${MAX_STEPS:-120}; VAL_MAX_ITEMS=${VAL_MAX_ITEMS:-1000}
SKIP_FIRST=${SKIP_FIRST:-3}   # OPSD: SDFT's code leaves each reply's first 3 ids ungraded
# every row's learning-starts threshold: no phase trains before rounds 0..k have played START_AFTER games (every user's),
# the base model playing until then; DAgger's and OPSD's dose, the same for both: each phase every label so far
# (aggregate), or the first phase that trains every label so far for FIRST_EPOCHS passes and every later one its round's
# labels plus REPLAY_RATIO times as many from whole earlier games (replay)
START_AFTER=${START_AFTER:-0}; DOSE=${DOSE:-aggregate}; REPLAY_RATIO=${REPLAY_RATIO:-1}; FIRST_EPOCHS=${FIRST_EPOCHS:-2}
PG_REWARD=${PG_REWARD:-cost}   # PG: cost ranks every game, or solves only (solve-first: every failure scores 0)
BATCH=${BATCH:-64}
MAX_RESTARTS=${MAX_RESTARTS:-5}
C1=${C1:-$HOME/loop/rules/c1.md}
case $ROW in pg) LOOPS="player trainer" ;; *) LOOPS="player labeller trainer" ;; esac

L=~/runs/rt/$NAME
mkdir -p "$L"
say() { echo "$(date -u +%m-%d\ %H:%M) realtime.sh: $*" >> "$L/progress"; }
relay_up() { python3 -c "import socket; socket.create_connection(('127.0.0.1', $RELAY_PORT), 3)" 2>/dev/null; }
start_relay() {
    relay_up && return 0
    (cd ~/prime-rl && setsid nohup uv run --no-sync python ~/loop/expert_relay.py --port "$RELAY_PORT" \
        --model "$LABEL_MODEL" --providers "$RELAY_PROVIDERS" >> "$L/relay.log" 2>&1 < /dev/null &)
    for _ in $(seq 24); do sleep 5; relay_up && return 0; done
    say "the relay on :$RELAY_PORT did not start ($L/relay.log)"; return 1
}
serving() {   # this run's in-play adapter as serve.sh takes it (NAME PATH: the phase its newest round plays on, in
    # that round's served.json), nothing while the base model plays
    python3 - "$L" "$NAME" <<'PY'
import glob, json, sys
found = sorted(glob.glob(sys.argv[1] + "/play/rounds/r*/served.json"))
phase = json.load(open(found[-1]))["phase"] if found else None
if phase is not None:
    print(f"{sys.argv[2]}-p{phase:04d} {sys.argv[1]}/train/p{phase:04d}/adapter")
PY
}
keep_server() {   # the model server up and serving the run's in-play adapter: one that died under the runs (18 Oct
    # 2026, every run stalled until it was started by hand) is started again with it within a pass, and the round's
    # games carry on through prime-agent's retries. serve.sh takes one call at a time and stops a server only once
    # it has failed three health checks
    local adapter want
    adapter=$(serving); want=${adapter%% *}; want=${want:-$MODEL}
    curl -fsS -m 10 localhost:8101/v1/models 2>/dev/null | grep -q "\"$want\"" && return 0
    say "the model server is down or does not serve $want: serving it again"
    bash ~/loop/serve.sh "$SERVE_GPUS" "$SERVE_DP" $adapter >> "$L/serve.log" 2>&1 || say "serve.sh failed ($L/serve.log)"
}
running() { pgrep -f "[r]ealtime.py $1 --run $L " >/dev/null; }
launch() {   # launch LOOP: realtime.py's LOOP in the background, logging to $L/LOOP.log
    local loop=$1 args=(--run "$L" --rounds "$ROUNDS" --row "$ROW") env=()
    case $loop in
        player) args+=(--users "$USERS" --name "$NAME" --lag "$LAG" --serve-gpus "$SERVE_GPUS" --serve-dp "$SERVE_DP") ;;
        labeller) args+=(--c1 "$C1" --label-model "$LABEL_MODEL" --label-effort "$LABEL_EFFORT"
                         --relay-port "$RELAY_PORT" --need-thinking "$NEED_THINKING"
                         --label-round-budget "$LABEL_ROUND_BUDGET_USD" --label-run-budget "$LABEL_RUN_BUDGET_USD"
                         --min-labelled-pct "$MIN_LABELLED_PCT" --hold-every "$HOLD_EVERY") ;;
        trainer) args+=(--lag "$LAG" --lr "$LR" --batch "$BATCH" --gpus "$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)"
                        --eval-steps "$EVAL_STEPS" --patience "$PATIENCE" --max-steps "$MAX_STEPS" --hold-every "$HOLD_EVERY"
                        --val-max-items "$VAL_MAX_ITEMS" --skip-first "$SKIP_FIRST" --start-after "$START_AFTER"
                        --dose "$DOSE" --replay-ratio "$REPLAY_RATIO" --first-epochs "$FIRST_EPOCHS" --pg-reward "$PG_REWARD")
                 env=(CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) ;;
    esac
    (cd ~/prime-rl && setsid nohup env "${env[@]}" uv run --no-sync python ~/loop/realtime.py "$loop" "${args[@]}" \
        >> "$L/$loop.log" 2>&1 < /dev/null &)
}
labels() { [ "$ROW" != pg ]; }

{   # read whole before it runs: bash reads a script as it goes, and the code may be synced again while this runs
    for g in ${TRAIN_GPUS//,/ }; do   # OPSD draws its fresh replies from the server while it trains
        case ",$SERVE_GPUS," in *",$g,"*) say "TRAIN_GPUS ($TRAIN_GPUS) must be none of SERVE_GPUS ($SERVE_GPUS)"; exit 1 ;; esac
    done
    say "started: $ROW, $USERS users, $ROUNDS rounds, lag $LAG, LR $LR, serving on $SERVE_GPUS, training on $TRAIN_GPUS, batch $BATCH, start after $START_AFTER games, dose $DOSE (replay $REPLAY_RATIO, first $FIRST_EPOCHS passes), PG reward $PG_REWARD"
    if labels; then start_relay || exit 1; fi
    bash ~/loop/serve.sh "$SERVE_GPUS" "$SERVE_DP" || { say "the server did not start"; exit 1; }
    declare -A restarts=([player]=0 [labeller]=0 [trainer]=0) started=([player]=0 [labeller]=0 [trainer]=0)
    while true; do
        left=0
        for loop in $LOOPS; do
            [ -f "$L/$loop.done" ] && continue
            left=1
            running "$loop" && continue
            if [ "${started[$loop]}" = 1 ]; then
                restarts[$loop]=$((restarts[$loop] + 1))
                if [ "${restarts[$loop]}" -gt "$MAX_RESTARTS" ]; then
                    say "the $loop died $MAX_RESTARTS times ($L/$loop.log): stopping the run"
                    pkill -f "[r]ealtime.py [a-z]* --run $L "
                    exit 1
                fi
                say "the $loop stopped: restart ${restarts[$loop]} ($L/$loop.log)"
            fi
            started[$loop]=1
            launch "$loop"
        done
        [ "$left" = 0 ] && break
        if labels; then relay_up || start_relay; fi
        [ -f "$L/player.done" ] || keep_server
        sleep 30
    done
    # the relay and the server are shared with any other run on this node (every run's player loads its own adapters
    # into the one server): the last run to finish stops them
    if ! pgrep -af "[r]ealtime.py [a-z]* --run " | grep -qv -- "--run $L "; then
        pkill -f "[e]xpert_relay.py --port $RELAY_PORT"
        pkill -f "[v]llm serve"
    fi
    say "done"
    exit 0
}
