#!/usr/bin/env bash
# The real-time DAgger row on one node (realtime.py's module docstring): the oracle's relay and the model server stay
# up for the whole run, and realtime.py's player, labeller and trainer run side by side, each restarted when it dies
# (every one resumes from its files) up to MAX_RESTARTS times, the run stopped once all three are done. The server
# holds SERVE_GPUS (DP replicas: one game uses one replica at a time, so two serve ten users), the trainer
# TRAIN_GPUS. Two runs can share a node: the same SERVE_GPUS, so one server and one relay serve both, and TRAIN_GPUS
# of their own. USERS serves arc1-small to that many users (paper.py's served: the benchmark's canonical numbers are 1,
# 2, 4, 8 and 16), ROUNDS rounds of it (all 200 by default).
# Inputs: the base model, ~/prime-rl, and the OpenRouter key bootstrap wrote to ~/continual-arc-baselines/.env, which
# the relay reads (AGENTS.md's dated exception).
# Usage (on the node): USERS=1 LR=1e-4 setsid nohup bash ~/loop/realtime.sh > /dev/null 2>&1 < /dev/null &
#   progress: ~/runs/rt/dagger-u$USERS/progress; stop: pkill -f "[r]ealtime.sh"; pkill -f "[r]ealtime.py"
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MODEL=${MODEL:-Qwen/Qwen3.5-9B}
USERS=${USERS:?set USERS: how many users to serve the stream to, 1 2 4 8 or 16}
ROUNDS=${ROUNDS:-200}
LR=${LR:?set LR: the learning rate of the trainer}
NAME=${NAME:-dagger-u$USERS}
SERVE_GPUS=${SERVE_GPUS:-0,1}; SERVE_DP=${SERVE_DP:-2}
TRAIN_GPUS=${TRAIN_GPUS:-2,3,4,5,6,7}
# the label step: GLM-5.3-flash at medium effort through the relay, its reasoning required (asked again up to
# NEED_THINKING times), as the synchronous row's; one game in HOLD_EVERY held out for the held-out loss
LABEL_MODEL=${LABEL_MODEL:-z-ai/glm-5.3-flash}; LABEL_EFFORT=${LABEL_EFFORT:-medium}
RELAY_PORT=${RELAY_PORT:-18230}; RELAY_PROVIDERS=${RELAY_PROVIDERS:-gmicloud,parasail,fireworks}
NEED_THINKING=${NEED_THINKING:-4}; MIN_LABELLED_PCT=${MIN_LABELLED_PCT:-40}
LABEL_ROUND_BUDGET_USD=${LABEL_ROUND_BUDGET_USD:-20}; LABEL_RUN_BUDGET_USD=${LABEL_RUN_BUDGET_USD:-3000}
HOLD_EVERY=${HOLD_EVERY:-10}
# a phase: from the last phase's best adapter on every labelled round's training side, the held-out loss measured
# every EVAL_STEPS steps, stopped once PATIENCE measurements in a row have not improved on it, at most MAX_STEPS steps
EVAL_STEPS=${EVAL_STEPS:-10}; PATIENCE=${PATIENCE:-1}; MAX_STEPS=${MAX_STEPS:-120}; VAL_MAX_ITEMS=${VAL_MAX_ITEMS:-1000}
BATCH=${BATCH:-64}
MAX_RESTARTS=${MAX_RESTARTS:-5}
C1=${C1:-$HOME/loop/rules/c1.md}

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
running() { pgrep -f "[r]ealtime.py $1 --run $L " >/dev/null; }
launch() {   # launch LOOP: realtime.py's LOOP in the background, logging to $L/LOOP.log
    local loop=$1 args=(--run "$L" --rounds "$ROUNDS") env=()
    case $loop in
        player) args+=(--users "$USERS" --name "$NAME" --serve-gpus "$SERVE_GPUS" --serve-dp "$SERVE_DP") ;;
        labeller) args+=(--c1 "$C1" --label-model "$LABEL_MODEL" --label-effort "$LABEL_EFFORT"
                         --relay-port "$RELAY_PORT" --need-thinking "$NEED_THINKING"
                         --label-round-budget "$LABEL_ROUND_BUDGET_USD" --label-run-budget "$LABEL_RUN_BUDGET_USD"
                         --min-labelled-pct "$MIN_LABELLED_PCT" --hold-every "$HOLD_EVERY") ;;
        trainer) args+=(--lr "$LR" --batch "$BATCH" --gpus "$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)"
                        --eval-steps "$EVAL_STEPS" --patience "$PATIENCE" --max-steps "$MAX_STEPS"
                        --val-max-items "$VAL_MAX_ITEMS")
                 env=(CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) ;;
    esac
    (cd ~/prime-rl && setsid nohup env "${env[@]}" uv run --no-sync python ~/loop/realtime.py "$loop" "${args[@]}" \
        >> "$L/$loop.log" 2>&1 < /dev/null &)
}

{   # read whole before it runs: bash reads a script as it goes, and the code may be synced again while this runs
    say "started: $USERS users, $ROUNDS rounds, LR $LR, serving on $SERVE_GPUS, training on $TRAIN_GPUS"
    start_relay || exit 1
    bash ~/loop/serve.sh "$SERVE_GPUS" "$SERVE_DP" || { say "the server did not start"; exit 1; }
    declare -A restarts=([player]=0 [labeller]=0 [trainer]=0) started=([player]=0 [labeller]=0 [trainer]=0)
    while true; do
        left=0
        for loop in player labeller trainer; do
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
        relay_up || start_relay
        sleep 30
    done
    # the relay and the server are shared with any other run on this node (both runs' players load their own
    # adapters into the one server): the last run to finish stops them
    if ! pgrep -af "[r]ealtime.py [a-z]* --run " | grep -qv -- "--run $L "; then
        pkill -f "[e]xpert_relay.py --port $RELAY_PORT"
        pkill -f "[v]llm serve"
    fi
    say "done"
    exit 0
}
