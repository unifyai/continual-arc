#!/usr/bin/env bash
# One of the paper's trained rows, DAgger, OPSD or PG (ROW), with a fixed lag (realtime.py's module docstring): round r
# is played by the model trained on rounds 0..r-1-LAG (LAG=1, the paper's; LAG=0, synchronous), play waiting for it
# when training is behind and training never waiting for play. The oracle's relay (DAgger and OPSD) and the model
# server stay up for the whole run, each started again within a pass when it is found down (the server with the adapter
# the run's round plays on), and realtime.py's player, labeller (DAgger and OPSD) and trainer run side by side, each
# restarted when it dies (every one resumes from its files) up to MAX_RESTARTS times, the run stopped once all of them
# are done. The server holds SERVE_GPUS (one copy tensor-parallel over them with SERVE_DP=1, the fastest per call at
# the few games a run keeps in flight), the trainer TRAIN_GPUS. Runs can share a node: the same SERVE_GPUS, so one
# server and one relay serve them all, and the same TRAIN_GPUS, which their trainers then share as one pool, each
# phase on an even share of the runs with a phase to train (gpupool.py), or TRAIN_GPUS of their own; lists that
# overlap but differ are refused. fleet.py plans which runs share a node and how many GPUs each part gets. USERS
# serves arc1-small to that many users (paper.py's served: the benchmark's canonical numbers are 1, 2, 4, 8 and 16),
# ROUNDS rounds of it: the stream's 200 by default, after which frozen.sh plays the run's two frozen tests. The
# schedule goes on with the epilogue's 100 rounds, which a run given ROUNDS=300 plays and trains on as the stream,
# and which no measure reads.
# Each row trains at LR 1e-4 (DAgger's and PG's rate, and b6's for self-distillation).
# Inputs: a GPU node set up as the README says (the base model on its disk, prime-rl's checkout at PRIME_RL with this
# repository's packages installed in its environment), and for DAgger and OPSD an OpenRouter key, OPENROUTER_API_KEY in
# the environment or in the baselines' .env (BASELINES/.env), which the relay reads.
# Paths: this folder is the one this script sits in; BASELINES (default ../baselines), PRIME_RL (default ~/prime-rl)
# and RUNS (default ~/runs, where every run writes) can be set.
# Usage (on the node): ROW=dagger USERS=1 LR=1e-4 SERVE_GPUS=0,1,2,3 SERVE_DP=1 TRAIN_GPUS=4,5,6,7 \
#     setsid nohup bash ~/continual-arc/training/realtime.sh > /dev/null 2>&1 < /dev/null &
#   progress: $RUNS/rt/$NAME/progress ($ROW-u$USERS-lag$LAG); stop: the run's realtime.sh, then its loops' process
#   groups (a pkill pattern in an ssh command line matches that command line too: run it from a script on the node).
#   With checkpoint.ON, a trainer given SIGTERM finishes its step, writes a step checkpoint and
#   exits, up to a step and some 20 s later, and its phase goes on from there when it starts again; SIGKILL stops it at
#   once, its phase going on from its checkpoint before
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MODEL=${MODEL:-Qwen/Qwen3.5-9B}
# the model is read from the node's disk alone (the node's setup downloads it): a run's train processes start whenever the
# pool hands it new GPUs, and six data-parallel workers starting at once, each looking the model up on the Hugging Face
# hub, were refused its files and the train process died
export HF_HUB_OFFLINE=1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASELINES=${BASELINES:-$(dirname "$HERE")/baselines}
PRIME_RL=${PRIME_RL:-$HOME/prime-rl}
RUNS=${RUNS:-$HOME/runs}
export BASELINES RUNS   # the relay reads BASELINES/.env, gpupool.py keeps the pool under RUNS
ROW=${ROW:?set ROW: dagger, opsd or pg}
case $ROW in dagger | opsd | pg) ;; *) echo "ROW must be dagger, opsd or pg, got $ROW"; exit 1 ;; esac
USERS=${USERS:?set USERS: how many users to serve the stream to, 1 2 4 8 or 16}
ROUNDS=${ROUNDS:-200}   # the stream's 200 rounds; paper.full_schedule's epilogue holds 100 more
LR=${LR:?set LR: the learning rate of the trainer}
LAG=${LAG:-1}
NAME=${NAME:-$ROW-u$USERS-lag$LAG}
SERVE_GPUS=${SERVE_GPUS:-0,1,2,3}; SERVE_DP=${SERVE_DP:-1}
TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7}
# the label step (DAgger and OPSD): GLM-5.3-flash at medium effort through the relay, its reasoning required (asked
# again up to NEED_THINKING times), as the synchronous row's; every row holds one game in HOLD_EVERY out for its
# held-out loss
LABEL_MODEL=${LABEL_MODEL:-z-ai/glm-5.3-flash}; LABEL_EFFORT=${LABEL_EFFORT:-medium}
RELAY_PORT=${RELAY_PORT:-18230}; RELAY_PROVIDERS=${RELAY_PROVIDERS:-gmicloud,parasail,fireworks}
NEED_THINKING=${NEED_THINKING:-4}; MIN_LABELLED_PCT=${MIN_LABELLED_PCT:-40}
LABEL_ROUND_BUDGET_USD=${LABEL_ROUND_BUDGET_USD:-20}; LABEL_RUN_BUDGET_USD=${LABEL_RUN_BUDGET_USD:-3000}
HOLD_EVERY=${HOLD_EVERY:-10}
# what the oracle is told, at the end of its copy of each call (oracle_hint.oracle_rules): C1 and the program library's
# procedure, row HPL's text, which the student is never told (library, the nine final runs'), or C1 alone (c1, the three
# rebuttal runs: an oracle not told the procedure either). A run keeps its oracle's rules to the end: a labeller started
# under others stops (realtime.py's cmd_labeller)
ORACLE_RULES=${ORACLE_RULES:-library}
case $ORACLE_RULES in library | c1) ;; *) echo "ORACLE_RULES must be library or c1, got $ORACLE_RULES"; exit 1 ;; esac
# a phase of any row: from the last phase's best adapter on every round's training side so far, the held-out loss
# measured every EVAL_STEPS steps, stopped once PATIENCE measurements in a row have not improved on it, at most
# MAX_STEPS steps. A patience of two, not one: with few users the held-out set is a game or two for the first rounds,
# and one noisy measurement ended every phase of a 1-user run and kept its starting adapter. Every 5 steps, not 10:
# the best came at step 10 and the phase ran 20 steps more to confirm it, so a phase outlasted a round
EVAL_STEPS=${EVAL_STEPS:-5}; PATIENCE=${PATIENCE:-2}; MAX_STEPS=${MAX_STEPS:-120}; VAL_MAX_ITEMS=${VAL_MAX_ITEMS:-1000}
SKIP_FIRST=${SKIP_FIRST:-3}   # OPSD: SDFT's code leaves each reply's first 3 ids ungraded
# every row's learning-starts threshold: no phase trains before rounds 0..k have played START_AFTER games (every user's),
# the base model playing until then; every row's dose, the same for all three: each phase every round's so far
# (aggregate), or the first phase that trains every label so far for FIRST_EPOCHS passes and every later one its round's
# labels plus REPLAY_RATIO times as many from whole earlier games (replay)
START_AFTER=${START_AFTER:-0}; DOSE=${DOSE:-aggregate}; REPLAY_RATIO=${REPLAY_RATIO:-1}; FIRST_EPOCHS=${FIRST_EPOCHS:-2}
BATCH=${BATCH:-64}
MAX_RESTARTS=${MAX_RESTARTS:-5}
C1=${C1:-$HERE/rules/c1.md}
case $ROW in pg) LOOPS="player trainer" ;; *) LOOPS="player labeller trainer" ;; esac

L=$RUNS/rt/$NAME
mkdir -p "$L"
say() { echo "$(date -u +%m-%d\ %H:%M) realtime.sh: $*" >> "$L/progress"; }
relay_up() { python3 -c "import socket; socket.create_connection(('127.0.0.1', $RELAY_PORT), 3)" 2>/dev/null; }
start_relay() {
    relay_up && return 0
    (cd "$PRIME_RL" && setsid nohup uv run --no-sync python "$HERE/expert_relay.py" --port "$RELAY_PORT" \
        --model "$LABEL_MODEL" --providers "$RELAY_PROVIDERS" >> "$L/relay.log" 2>&1 < /dev/null &)
    for _ in $(seq 24); do sleep 5; relay_up && return 0; done
    say "the relay on :$RELAY_PORT did not start ($L/relay.log)"; return 1
}
serving() {   # this run's in-play adapter as serve.sh takes it (NAME PATH: the phase its newest round plays on, in
    # that round's served.json), nothing while the base model plays (before a phase, or on a warm-up phase's, which
    # saves no adapter)
    python3 - "$L" "$NAME" <<'PY'
import glob, json, os, sys
found = sorted(glob.glob(sys.argv[1] + "/play/rounds/r*/served.json"))
phase = json.load(open(found[-1]))["phase"] if found else None
adapter = f"{sys.argv[1]}/train/p{phase:04d}/adapter" if phase is not None else ""
if adapter and os.path.isdir(adapter):
    print(f"{sys.argv[2]}-p{phase:04d} {adapter}")
PY
}
keep_server() {   # the model server up and serving the run's in-play adapter: one that dies under the runs (which
    # would otherwise stall every run until it was started by hand) is started again with it within a pass, and the
    # round's games carry on through prime-agent's retries. serve.sh takes one call at a time and stops a server only once
    # it has failed three health checks
    local adapter want
    adapter=$(serving); want=${adapter%% *}; want=${want:-$MODEL}
    curl -fsS -m 10 localhost:8101/v1/models 2>/dev/null | grep -q "\"$want\"" && return 0
    say "the model server is down or does not serve $want: serving it again"
    bash "$HERE/serve.sh" "$SERVE_GPUS" "$SERVE_DP" $adapter >> "$L/serve.log" 2>&1 || say "serve.sh failed ($L/serve.log)"
}
running() { pgrep -f "[r]ealtime.py $1 --run $L " >/dev/null; }
launch() {   # launch LOOP: realtime.py's LOOP in the background, logging to $L/LOOP.log
    local loop=$1 args=(--run "$L" --rounds "$ROUNDS" --row "$ROW") env=()
    case $loop in
        player) args+=(--users "$USERS" --name "$NAME" --lag "$LAG" --serve-gpus "$SERVE_GPUS" --serve-dp "$SERVE_DP") ;;
        labeller) args+=(--c1 "$C1" --oracle-rules "$ORACLE_RULES" --label-model "$LABEL_MODEL"
                         --label-effort "$LABEL_EFFORT" --relay-port "$RELAY_PORT" --need-thinking "$NEED_THINKING"
                         --label-round-budget "$LABEL_ROUND_BUDGET_USD" --label-run-budget "$LABEL_RUN_BUDGET_USD"
                         --min-labelled-pct "$MIN_LABELLED_PCT" --hold-every "$HOLD_EVERY") ;;
        trainer) args+=(--train-gpus "$TRAIN_GPUS" --lag "$LAG" --lr "$LR" --batch "$BATCH"
                        --eval-steps "$EVAL_STEPS" --patience "$PATIENCE" --max-steps "$MAX_STEPS" --hold-every "$HOLD_EVERY"
                        --val-max-items "$VAL_MAX_ITEMS" --skip-first "$SKIP_FIRST" --start-after "$START_AFTER"
                        --dose "$DOSE" --replay-ratio "$REPLAY_RATIO" --first-epochs "$FIRST_EPOCHS")
                 env=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) ;;
    esac
    (cd "$PRIME_RL" && setsid nohup env "${env[@]}" uv run --no-sync python "$HERE/realtime.py" "$loop" "${args[@]}" \
        >> "$L/$loop.log" 2>&1 < /dev/null &)
}
labels() { [ "$ROW" != pg ]; }

{   # read whole before it runs: bash reads a script as it goes, and the code may be synced again while this runs
    for g in ${TRAIN_GPUS//,/ }; do   # OPSD draws its fresh replies from the server while it trains
        case ",$SERVE_GPUS," in *",$g,"*) say "TRAIN_GPUS ($TRAIN_GPUS) must be none of SERVE_GPUS ($SERVE_GPUS)"; exit 1 ;; esac
    done
    # every value the run was started with, so that it can be started again with the same ones
    say "started: $ROW, $USERS users, $ROUNDS rounds, lag $LAG, LR $LR, serving on $SERVE_GPUS, training on $TRAIN_GPUS, batch $BATCH, start after $START_AFTER games, dose $DOSE (replay $REPLAY_RATIO, first $FIRST_EPOCHS passes)$(labels && echo ", oracle rules $ORACLE_RULES"), patience $PATIENCE"
    if labels; then start_relay || exit 1; fi
    bash "$HERE/serve.sh" "$SERVE_GPUS" "$SERVE_DP" || { say "the server did not start"; exit 1; }
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
