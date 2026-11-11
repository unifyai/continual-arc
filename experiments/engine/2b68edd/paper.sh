#!/usr/bin/env bash
# The paper driver: one row of the block (H0, GRPO, OPSD, DAgger or the "should" row H) through the real
# `arc1-collective` schedule (100 users x 200 rounds, workstreams, seed 0; paper.py's full_schedule), one round per
# runner call, on the node (round.sh's shape, generalised to the paper's own data and rows). USERS (default 100, the
# real configuration) plays a cheaper cut instead: arc1-collective with that many users, 0..USERS-1 exactly the real
# configuration's own first USERS users, round by round (paper.py's collective_config); use it only to smoke-test
# the wiring below, never for a paper row. OPSD (RL-10) replaces standard OPD as the paper's teacher-scored trained
# row: its teacher is the frozen base model scoring each call under its own GLM-5.3-flash hint (oracle_hint.py
# watch, teacher_score.py --hints), not a 27B teacher's plain scores; train.py --objective opd (unchanged) trains
# on either alike. DAgger (17 Oct) is the method's own row: the same oracle, under the same playing and
# program-library rules, writes its own reply to every player call of the round (forward.py points/label/label-set,
# the dev loop's labelling), and train.py --objective sft trains on those replies. Round r = FIRST_ROUND,
# FIRST_ROUND+1, ...:
#   - serve: H0 and H always serve the base model (frozen: nothing to load); GRPO, OPSD and DAgger serve round r-1's
#     adapter (the base model at r = 0), named "$ROW-r$((r-1))" (a smoke cut's own "$ROW-u$USERS-r$((r-1))") so it
#     never collides with the dev loop's "round$k" or another row's or another USERS cut's adapters on the server;
#   - check (trained rows, r > 0 only): the served-adapter check (paper.py check-served) once the adapter is loaded,
#     so a layer-naming mistake (loop.served_adapter's Qwen3.5 fix) is caught before a whole round plays on it; an
#     adapter no step has touched (a round with no trainable item, e.g. GRPO with no mixed group) is the base
#     model's weights by construction and is skipped, with a line in the progress log;
#   - play r: paper.py play --row $ROW --round r, over the real schedule cut at r+1 rounds (Schedule.first_rounds)
#     or, with EPILOGUE=1, the K-user epilogue cut the same way; resume=True restores every whole round
#     already in $PLAY_DIR/records.jsonl and plays only round r (H0 wiped before every instance via baselines'
#     stateless mode; GRPO, OPSD and H keep prime-agent's own memory, so a returning puzzle can get cheaper once a
#     trained row learns to store and fetch its program). Every instance lands under
#     $PLAY_DIR/rounds/r$r/captured/, sharded by round so a round's own games never mix with another's in GRPO's
#     (run, task) groups (paper.py's module docstring: arc1-collective's 100 users all play again every round under
#     the same user ids, so a flat captured/ would have a later round silently overwrite an earlier one's files);
#   - ROW=opsd only, alongside play (network only, no GPU, so it cannot disturb it): a background watcher
#     (oracle_hint.py watch) hints each of the round's games as its captured file lands, into
#     round_dir(r)/hints/, finishing whatever is left once play ends;
#   - ROW=opsd only, after play: score round r's games under their hints with the frozen base model
#     (teacher_score.py --hints), read off the very same server play just used (serve.sh always still answers the
#     base model's own name too, adapter or none loaded beside it) -- nothing is stopped or served again -- into
#     $PLAY_DIR/rounds/r$r/opd_scores.json;
#   - ROW=dagger only, after play: label round r (label_round): the oracle's own reply to every player call of the
#     round's games, each read exactly as the player read it with the playing and program-library rules
#     (oracle_hint.py rules) at the end of its last message, its reasoning required (asked again up to
#     LABEL_NEED_THINKING times, else no label), through expert_relay.py on the node; then the label set
#     ($PLAY_DIR/rounds/r$r/labels/set, forward.py label-set's play folder of label targets), split into a training
#     side and a held-out side (split_round: VAL_FRAC of its games, drawn whole);
#   - train r (trained rows only): train.py --objective grpo or opd (for OPSD too: its training step is OPD's,
#     unchanged) on round r's games alone ($PLAY_DIR/rounds/r$r), one pass over all of the round's useful data, no cap
#     beyond the step size; or, for DAgger, sft on every round's training side so far (the aggregate), stopped when
#     the held-out loss on every round's held-out side so far turns (see the dagger settings below); from round r-1's
#     adapter, into $TRAIN_DIR/r$r;
#   - save: train.py's own adapter + optimizer under $TRAIN_DIR/r$r; round r+1 serves it.
# Every stage is marked as round.sh's are (round-play$r.done, hints.json, labels/set.json, labels/split/split.json,
# train$r/DONE, opd_scores.json), so a kill at any step resumes there: continual_arc.runner.run's own resume never replays a
# finished round, and train_round never starts a finished round's training twice. H0 and H never train: for them
# this is play-only.
# The epilogue at round R (EPILOGUE_ROUND, default 20; EPILOGUE_USERS, default 10), after a stream run of ROUNDS=R
# (rounds 0..R-1: EPILOGUE_ROUND pairs with the stream call's own ROUNDS, not with the last round index) is trained
# (EPILOGUE=1; FIRST_ROUND/ROUNDS then count the epilogue's own rounds, 0-indexed, not the stream's): a second,
# separate run ($PLAY_DIR/$TRAIN_DIR below point at epilogue-r$EPILOGUE_ROUND/ instead), serving at its own round 0
# the adapter that would have served round R of the stream -- train/r(R-1), the newest one a ROUNDS=R stream call
# leaves behind (INIT_ADAPTER) -- and training on through its own rounds the same way for a trained row; H0 plays it
# wiped, with no training, as T's fresh reference. USERS carries through to the epilogue too (its own run and
# directories get the same "-uN" suffix as the stream's, so INIT_ADAPTER always names an adapter the same USERS
# trained): the epilogue of a smoke cut stays a cut of the same size.
# Inputs: none beyond the base model and ~/prime-rl (as round.sh); ROW=opsd and ROW=dagger also need the OpenRouter
# key bootstrap already wrote to ~/continual-arc-baselines/.env on the node (AGENTS.md's dated exceptions).
# Usage: ROW=h0 ROUNDS=20 bash ~/loop/paper.sh
#   ROW=grpo LR=1e-4 ROUNDS=20 bash ~/loop/paper.sh
#   ROW=opsd LR=2e-5 ROUNDS=20 bash ~/loop/paper.sh
#   ROW=dagger LR=1e-4 ROUNDS=20 bash ~/loop/paper.sh
#   ROW=grpo LR=1e-4 EPILOGUE=1 EPILOGUE_ROUND=20 EPILOGUE_USERS=10 ROUNDS=25 bash ~/loop/paper.sh
#   ROW=h0 EPILOGUE=1 EPILOGUE_ROUND=20 EPILOGUE_USERS=10 ROUNDS=25 bash ~/loop/paper.sh
#
# The GPU smoke (deferred; none of this has run): H0 and GRPO on a 4-user cut of arc1-collective, 3 rounds, each a
# single node. OPSD gets none (R's go-ahead instead, once the code and CPU tests are reviewed): a CPU dry run
# on real captured games stands in (this ticket's own report), and the paper run itself waits for the go-ahead.
#   ROW=h0 USERS=4 ROUNDS=3 bash ~/loop/paper.sh
#   ROW=grpo LR=1e-4 USERS=4 ROUNDS=3 bash ~/loop/paper.sh
# Check for each: the served-adapter check passes every round after the first; $PLAY_DIR/records.jsonl has exactly
# 4 x 3 records after round 2 (not 100 x 3: USERS=4 plays the real configuration's own first 4 users, round by
# round, never a different draw); a kill mid-round (during play_round) resumes at the same round, not the next; a
# kill mid-training (during train_round) resumes training, not a re-play. Then, with both rows' round 2
# adapters saved (ROUNDS=3: rounds 0, 1 and 2, the newest at train/r2):
#   ROW=grpo LR=1e-4 USERS=4 EPILOGUE=1 EPILOGUE_ROUND=3 EPILOGUE_USERS=2 ROUNDS=1 bash ~/loop/paper.sh
#   ROW=h0 USERS=4 EPILOGUE=1 EPILOGUE_ROUND=3 EPILOGUE_USERS=2 ROUNDS=1 bash ~/loop/paper.sh
# play the 2-user epilogue's first round, GRPO starting from round 2's adapter (train/r2, EPILOGUE_ROUND-1) and H0
# wiped.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MODEL=${MODEL:-Qwen/Qwen3.5-9B}
ROW=${ROW:?set ROW: h0, grpo, opsd, dagger or h}
case $ROW in h0 | grpo | opsd | dagger | h) ;; *) echo "ROW must be h0, grpo, opsd, dagger or h, got $ROW"; exit 1 ;; esac
FIRST_ROUND=${FIRST_ROUND:-0}
ROUNDS=${ROUNDS:?set ROUNDS: how many rounds this call plays, from FIRST_ROUND}
LR=${LR:-}
BATCH=${BATCH:-64}
TRAIN_MAX_MINUTES=${TRAIN_MAX_MINUTES:-1440}   # a safety cap, not a schedule (forward.sh's own)
MAX_SEQ_LEN=${MAX_SEQ_LEN:-65536}   # train.py's own default; teacher_score.py scores to the same window
                                    # (score_opsd), so the teacher never scores a call the trainer then throws away
PER_CALL=${PER_CALL:-0}
# ROW=opsd only (RL-10): the hint step (oracle_hint.py watch), GLM-5.3-flash at medium effort straight from
# OpenRouter (AGENTS.md's dated exception: the OpenRouter key bootstrap already wrote to the node's own
# ~/continual-arc-baselines/.env, the hint step's only use of a key on a node, never logged or printed). C1 is the
# oracle's own THE PLAYING RULES text (the same c1 the pre-RL-10 rows distilled, not read by opsd's own training).
HINT_MODEL=${HINT_MODEL:-z-ai/glm-5.3-flash}; HINT_EFFORT=${HINT_EFFORT:-medium}
HINT_WORKERS=${HINT_WORKERS:-64}
HINT_ROUND_BUDGET_USD=${HINT_ROUND_BUDGET_USD:-40}; HINT_RUN_BUDGET_USD=${HINT_RUN_BUDGET_USD:-500}
HINT_NO_HINT_MAX_FRAC=${HINT_NO_HINT_MAX_FRAC:-0.05}   # the hint step's refusal: no usable hint for more of a round's
                                                       # player calls than this (oracle_hint.py --no-hint-max-frac)
C1=${C1:-$HOME/loop/rules/c1.md}
# ROW=dagger only: the label step (label_round), the same oracle at the same effort as OPSD's hints, reached through
# expert_relay.py on the node (it reads the key bootstrap wrote to ~/continual-arc-baselines/.env: AGENTS.md's dated
# exception) pinned to RELAY_PROVIDERS, the hosts that thought most often on 15 Oct first. Every label must carry its
# reasoning (the expert teaches how to think, not only what to do): one without is asked again up to
# LABEL_NEED_THINKING times, then dropped. The key has no credit cap, so forward.py label stops a round's new requests
# at LABEL_ROUND_BUDGET_USD, and a round whose run has already spent LABEL_RUN_BUDGET_USD is refused before it asks.
# Fewer than LABEL_MIN_PCT% of the round's points labelled (a dead relay or key, or an expert that will not think;
# 15 Oct: 22.5% never thought in 5 asks) fails the round.
LABEL_MODEL=${LABEL_MODEL:-$HINT_MODEL}; LABEL_EFFORT=${LABEL_EFFORT:-$HINT_EFFORT}
LABEL_WORKERS=${LABEL_WORKERS:-256}   # points in flight at once: a point asked again for its reasoning takes minutes
LABEL_NEED_THINKING=${LABEL_NEED_THINKING:-4}
LABEL_ROUND_BUDGET_USD=${LABEL_ROUND_BUDGET_USD:-60}; LABEL_RUN_BUDGET_USD=${LABEL_RUN_BUDGET_USD:-1000}
LABEL_MIN_PCT=${LABEL_MIN_PCT:-60}
RELAY_PORT=${RELAY_PORT:-18230}; RELAY_PROVIDERS=${RELAY_PROVIDERS:-gmicloud,parasail,fireworks}
# ROW=dagger only: DAgger's training (the author, 17 Oct 2026, as his 26 Sep team rows: train on the whole buffer until the
# held-out loss turns, then collect more). Each round's label set is split, VAL_FRAC of its games drawn whole with the
# round as the seed going to the held-out side, so the held-out set grows every round; then every round trains on
# every round's training side so far (DAgger's aggregate, uniform: the targets are the oracle's replies, which no
# student sampled, so there is no importance weight to correct for), from the last round's adapter, measuring the
# held-out loss on every round's held-out side so far (games drawn whole up to VAL_MAX_ITEMS calls) every EVAL_STEPS
# steps, and stopping once PATIENCE measurements in a row have not improved on it, the best adapter kept; at most
# SFT_MAX_STEPS steps and SFT_MAX_EPOCHS passes a round, so a round's training stays near an hour as the buffer grows.
VAL_FRAC=${VAL_FRAC:-0.1}; VAL_MAX_ITEMS=${VAL_MAX_ITEMS:-1000}
EVAL_STEPS=${EVAL_STEPS:-80}; PATIENCE=${PATIENCE:-1}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-320}; SFT_MAX_EPOCHS=${SFT_MAX_EPOCHS:-20}
EPILOGUE=${EPILOGUE:-0}
EPILOGUE_ROUND=${EPILOGUE_ROUND:-20}
EPILOGUE_USERS=${EPILOGUE_USERS:-10}
USERS=${USERS:-100}   # arc1-collective's own 100; a smaller cut (paper.py play --config-users) for a cheap smoke
ALL=0,1,2,3,4,5,6,7
case $ROW in grpo | opsd | dagger) TRAINED=1 ;; *) TRAINED=0 ;; esac
# train.py has no "opsd" or "dagger" objective: opsd trains exactly as opd does, dagger by sft on its labels
case $ROW in opsd) TRAIN_OBJECTIVE=opd ;; dagger) TRAIN_OBJECTIVE=sft ;; *) TRAIN_OBJECTIVE=$ROW ;; esac
if [ "$TRAINED" = 1 ] && [ -z "$LR" ]; then echo "ROW=$ROW trains: set LR (the tuned rate for this objective)"; exit 1; fi
# epilogue_for_users refuses EPILOGUE_USERS > the configuration's own users; fail here, not partway into a play
if [ "$EPILOGUE" = 1 ] && [ "$EPILOGUE_USERS" -gt "$USERS" ]; then
    echo "EPILOGUE_USERS ($EPILOGUE_USERS) must not be more than USERS ($USERS)"; exit 1
fi

L=~/runs/paper/$ROW
SUFFIX=""; [ "$USERS" = 100 ] || SUFFIX="-u$USERS"   # a smoke cut's own run, play and train directories and adapter
                                                      # names, distinct from the real 100-user paper run's
if [ "$EPILOGUE" = 1 ]; then
    PLAY_DIR=$L/epilogue-r$EPILOGUE_ROUND$SUFFIX
    TRAIN_DIR=$L/epilogue-train-r$EPILOGUE_ROUND$SUFFIX
    INIT_ADAPTER=$L/train$SUFFIX/r$((EPILOGUE_ROUND - 1))/adapter   # the adapter that would have served round R
                                                                    # (same USERS); unused for h0/h
    SCHED_FLAGS="--epilogue-users $EPILOGUE_USERS --config-users $USERS"
    ADAPTER_PREFIX=$ROW-epi$EPILOGUE_ROUND$SUFFIX-r   # distinct from the main stream's "$ROW-rN" names on the server
else
    PLAY_DIR=$L/play$SUFFIX
    TRAIN_DIR=$L/train$SUFFIX
    SCHED_FLAGS="--config-users $USERS"
    ADAPTER_PREFIX=$ROW$SUFFIX-r
fi
mkdir -p $L
say() { echo "$(date -u +%H:%M) paper($ROW): $*" >> $L/progress; }
round_dir() { printf "$PLAY_DIR/rounds/r%04d" "$1"; }   # paper.py's round_dir: must match it exactly
stop_server() {
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
training() { pgrep -f "[t]rain.py .*--out $(round_to_train_dir "$1")( |$)" >/dev/null; }
round_to_train_dir() { echo "$TRAIN_DIR/r$1"; }

serve_round() {   # serve_round R: the adapter round R's play needs: round (R-1)'s own for GRPO/OPSD (the epilogue's
                  # round 0 instead needs the main stream's round-(EPILOGUE_ROUND-1) adapter, INIT_ADAPTER: nothing
                  # is frozen for a trained row there), the base model for H0/H always and for GRPO/OPSD's own round 0
    local r=$1 name path
    case $ROW in
    h0 | h) bash ~/loop/serve.sh $ALL 8 || { say "server failed"; return 1; }; echo "$MODEL" ;;
    *)
        if [ "$r" = 0 ] && [ "$EPILOGUE" = 0 ]; then
            bash ~/loop/serve.sh $ALL 8 || { say "server failed"; return 1; }
            echo "$MODEL"; return 0
        elif [ "$r" = 0 ]; then
            name=$ROW$SUFFIX-r$((EPILOGUE_ROUND - 1))
        else
            name=$ADAPTER_PREFIX$((r - 1))
        fi
        path=$(adapter_of "$r")
        bash ~/loop/serve.sh $ALL 8 "$name" "$path" || { say "server failed"; return 1; }
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
    (cd ~/prime-rl && uv run --no-sync python ~/loop/paper.py check-served --vllm-url http://localhost:8101/v1 \
        --base-model "$MODEL" --served "$name" --adapter "$(adapter_of "$r")") >> $L/progress 2>&1
}

play_round() {   # play_round R: play round R of $ROW's schedule under the served model serve_round gave. A crashed or
                 # killed launch is retried (round.sh's resume.sh shape: relaunch only when nothing is running and
                 # the round is still short), up to twice, before the round fails for good.
    local r=$1 served pattern="[p]aper.py play .*--run-dir $PLAY_DIR .*--round $r( |\$)" have=0 want=1 attempt mem
    [ -f "$PLAY_DIR/round-play$r.done" ] && return 0
    served=$(serve_round "$r") || return 1
    check_served "$served" "$r" || { say "served-adapter check failed for round $r ($served)"; return 1; }
    for attempt in 1 2 3; do
        if ! pgrep -f "$pattern" >/dev/null; then
            say "play round $r started ($served, attempt $attempt)"
            (cd ~/prime-rl && setsid nohup uv run --no-sync python ~/loop/paper.py play --row "$ROW" \
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
        say "$(PYTHONPATH="$HOME/loop" python3 -c "
import json
import memory
print(memory.summary_line(json.load(open('$mem'))))")"
    fi
    return 0
}

start_hint_watcher() {   # start_hint_watcher R: a background watcher (oracle_hint.py watch) that hints round R's
                         # games as they land under round_dir(R)/captured/ (ROW=opsd only). Network only, no GPU,
                         # so it runs safely alongside play_round. Never double-launched if an earlier call's
                         # watcher is still running (a resumed paper.sh after a kill, say) -- it is left to finish
                         # on its own instead. Always launched otherwise, even when hints.json already exists:
                         # oracle_hint.py watch validates it itself (oracle_hint.stale_games, against the games
                         # round_dir/captured actually holds right now) and returns at once if it still holds, or
                         # moves it aside and redoes only what changed if it does not -- a bash-level "it exists,
                         # skip" here would trust a stale one exactly the way the 17 Oct re-review's blocker did.
    local r=$1 dir pattern
    dir=$(round_dir "$r")
    pattern="[o]racle_hint.py watch $dir( |\$)"
    pgrep -f "$pattern" >/dev/null && return 0
    mkdir -p "$dir"
    rm -f "$dir/play.done"   # a stale marker from an earlier, incomplete attempt must not look like play is done
    (cd ~/prime-rl && setsid nohup uv run --no-sync python ~/loop/oracle_hint.py watch "$dir" --c1 "$C1" \
        --model "$HINT_MODEL" --effort "$HINT_EFFORT" --workers "$HINT_WORKERS" --until "$dir/play.done" \
        --round-budget-usd "$HINT_ROUND_BUDGET_USD" --run-budget-usd "$HINT_RUN_BUDGET_USD" \
        --no-hint-max-frac "$HINT_NO_HINT_MAX_FRAC" \
        --run-budget-file "$PLAY_DIR/hints-cost.json" --parent-pid $$ > "$L/hints-round$r.log" 2>&1 < /dev/null &)
}

finish_hint_watcher() {   # finish_hint_watcher R: tells round R's watcher play is done (succeeded) and waits for
                         # hints.json (ROW=opsd only); the round fails if the watcher exits without it (the 5%
                         # rule, its budget guard, or hints.json failing its own re-check -- either way
                         # $L/hints-round$r.log has the reason). Never call this when play_round itself failed:
                         # kill_hint_watcher is that path's own, so a failed round can never produce a complete,
                         # trusted hints.json.
    local r=$1 dir
    dir=$(round_dir "$r")
    touch "$dir/play.done"
    while [ ! -f "$dir/hints.json" ] && pgrep -f "[o]racle_hint.py watch $dir( |\$)" >/dev/null; do sleep 10; done
    [ -f "$dir/hints.json" ] || { say "round $r's hints failed ($L/hints-round$r.log)"; return 1; }
    say "round $r hints: $(tail -1 "$L/hints-round$r.log")"
}

kill_hint_watcher() {   # kill_hint_watcher R: stops round R's watcher outright (ROW=opsd only, play_round failed
                        # for good): a failed round must never leave behind a hints.json a later resume would
                        # trust as this round's complete, authoritative hints (the 17 Oct re-review's blocker --
                        # finish_hint_watcher's own wait for one here would also keep paper.sh alive, so
                        # --parent-pid could never fire during it). pkill finding nothing (the watcher already
                        # exited on its own) is not a failure of this call.
    local r=$1 dir
    dir=$(round_dir "$r")
    pkill -f "[o]racle_hint.py watch $dir( |\$)" 2>/dev/null
    true
}

score_opsd() {   # score_opsd R: round R's games, under their own hints, scored by the frozen base model into
                 # round_dir(R)/opd_scores.json (ROW=opsd only). Reuses the server round R's own play served
                 # (serve.sh, port 8101): a LoRA-enabled vLLM always still answers the base model's own name too
                 # (the served-adapter check already depends on this), so nothing is stopped or served again --
                 # at round 0 the server is base-only anyway. serve_round "$r" at the top makes this stage
                 # self-sufficient on its own resume (a kill or reboot between play_round returning and this
                 # running leaves no server up): serve.sh is idempotent, so on the normal path (the server play
                 # just used is still healthy) this costs nothing. teacher_score.py --out writes atomically, but
                 # it still writes a complete-looking file even when it goes on to exit non-zero (too much of the
                 # round came back unscored): --out is a .tmp name here, and it is moved to the real
                 # opd_scores.json -- the stage's own resume marker -- only once the whole process exits 0, so a
                 # failed or partly-dead scoring pass never looks done on the next resume.
    local r=$1 out tmp
    out=$(round_dir "$r")/opd_scores.json
    tmp=$(round_dir "$r")/opd_scores.json.tmp
    [ -f "$out" ] && return 0
    serve_round "$r" >/dev/null || { say "server failed for round $r's scoring"; return 1; }
    say "scoring round $r with the base model ($MODEL), its own server still up"
    (cd ~/prime-rl && uv run --no-sync python ~/loop/teacher_score.py --vllm-url http://localhost:8101/v1 \
        --model "$MODEL" --student-model "$MODEL" --skip-tokenizer-check --train-runs "$(round_dir "$r")" \
        --hints "$(round_dir "$r")/hints.json" --max-seq-len "$MAX_SEQ_LEN" --out "$tmp" \
        > "$L/score-round$r.log" 2>&1) \
        || { say "scoring round $r failed ($L/score-round$r.log)"; rm -f "$tmp"; return 1; }
    mv "$tmp" "$out"
    say "round $r scored: $(tail -2 "$L/score-round$r.log" | tr '\n' ' ')"
}

relay_up() { python3 -c "import socket; socket.create_connection(('127.0.0.1', $RELAY_PORT), 3)" 2>/dev/null; }

start_relay() {   # the label step's endpoint (ROW=dagger only): expert_relay.py on this node, pinned to LABEL_MODEL
                  # and RELAY_PROVIDERS; left running for the next round once up
    relay_up && return 0
    (cd ~/prime-rl && setsid nohup uv run --no-sync python ~/loop/expert_relay.py --port "$RELAY_PORT" \
        --model "$LABEL_MODEL" --providers "$RELAY_PROVIDERS" >> "$L/relay.log" 2>&1 < /dev/null &)
    for _ in $(seq 24); do sleep 5; relay_up && return 0; done
    say "the relay on :$RELAY_PORT did not start ($L/relay.log)"; return 1
}

label_round() {   # label_round R: the oracle's own reply to every player call of round R's games, as a training set
                  # at round_dir(R)/labels/set (ROW=dagger only). Each call is read exactly as the player read it,
                  # with the playing and program-library rules OPSD's oracle reads (oracle_hint.py rules) at the end
                  # of its last message, so the two rows differ only in what the oracle writes. Resumable: forward.py
                  # label keeps every label already written, and labels/set.json, written last, is the stage's
                  # marker. The run's earlier label spend is checked before a single new request goes out.
    local r=$1 dir out spent
    dir=$(round_dir "$r"); out=$dir/labels
    [ -f "$out/set.json" ] && return 0
    mkdir -p "$out"
    spent=$(cd ~/prime-rl && uv run --no-sync python ~/loop/forward.py spent "$PLAY_DIR"/rounds/r*/labels) \
        || { say "reading the run's label spend failed"; return 1; }
    if python3 -c "import sys; sys.exit(0 if float('$spent') >= $LABEL_RUN_BUDGET_USD else 1)"; then
        say "round $r not labelled: the run's labels have spent \$$spent, over LABEL_RUN_BUDGET_USD ($LABEL_RUN_BUDGET_USD)"
        return 1
    fi
    start_relay || return 1
    (cd ~/prime-rl && uv run --no-sync python ~/loop/oracle_hint.py rules --c1 "$C1") > "$L/guidelines.md" \
        || { say "writing the label step's working rules failed"; return 1; }
    say "labelling round $r (run spend so far \$$spent)"
    (cd ~/prime-rl && uv run --no-sync python ~/loop/forward.py points "$dir" --out "$out") >> "$L/labels-round$r.log" 2>&1 \
        || { say "round $r's points failed ($L/labels-round$r.log)"; return 1; }
    # an attempt that labels too few points (the hosts down for a while, say) is tried again after a pause: it keeps
    # every label already written and asks only for the rest
    for attempt in 1 2 3; do
        (cd ~/prime-rl && uv run --no-sync python ~/loop/forward.py label --out "$out" --expert "$LABEL_MODEL" \
            --expert-url "http://127.0.0.1:$RELAY_PORT/v1" --budget "$LABEL_ROUND_BUDGET_USD" \
            --workers "$LABEL_WORKERS" --guidelines "$L/guidelines.md" --guidelines-at message --thinking \
            --reasoning "{\"enabled\": true, \"effort\": \"$LABEL_EFFORT\"}" --need-thinking "$LABEL_NEED_THINKING" \
            --min-labelled-pct "$LABEL_MIN_PCT") >> "$L/labels-round$r.log" 2>&1 && break
        say "round $r's labels, attempt $attempt: $(tail -1 "$L/labels-round$r.log")"
        [ "$attempt" = 3 ] && return 1
        start_relay || return 1
        sleep 300
    done
    say "round $r labels: $(tail -1 "$L/labels-round$r.log")"
    rm -rf "$out/set"   # a label set half-written by a killed attempt is rebuilt whole, never topped up
    (cd ~/prime-rl && uv run --no-sync python ~/loop/forward.py label-set --out "$out") >> "$L/labels-round$r.log" 2>&1 \
        || { say "round $r's label set failed ($L/labels-round$r.log)"; return 1; }
    say "round $r label set: $(tr -d ' \n' < "$out/set.json")"
}

side_of() { printf "$(round_dir "$1")/labels/split/%s-r%04d" "$2" "$1"; }   # side_of R train|dev: round R's side

split_round() {   # split_round R: round R's label set into its training and held-out sides (ROW=dagger only):
                  # VAL_FRAC of its games, drawn whole with R as the seed, held out, each side named for its round
                  # (side_of), since train.load_calls takes a folder's name as each of its games' run. split.json,
                  # written last, marks it done; a half-written split is made again whole.
    local r=$1 out
    out=$(round_dir "$r")/labels/split
    [ -f "$out/split.json" ] && return 0
    rm -rf "$out"
    (cd ~/prime-rl && uv run --no-sync python ~/loop/forward.py split "$(round_dir "$r")/labels/set" \
        --val-frac "$VAL_FRAC" --seed "$r" --sides "$(basename "$(side_of "$r" train)"),$(basename "$(side_of "$r" dev)")" \
        --out "$out") >> "$L/labels-round$r.log" 2>&1 || { say "round $r's split failed ($L/labels-round$r.log)"; return 1; }
    say "round $r split: $(tr -d ' \n' < "$out/split.json")"
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
    case $TRAIN_OBJECTIVE in
        # DAgger: every round's training side so far, measured on every round's held-out side so far (split_round),
        # read as chains (one pass per game, each label a branch: rules100's recipe) unless PER_CALL=1
        sft)
            local k vals=()
            for ((k = 0; k <= r; k++)); do
                [ -d "$(side_of "$k" train)" ] && [ -d "$(side_of "$k" dev)" ] \
                    || { say "train round $r: round $k has no split ($(side_of "$k" train))"; return 1; }
                runs+=("$(side_of "$k" train)"); vals+=("$(side_of "$k" dev)")
            done
            flags=(--val-run "${vals[@]}" --val-max-items "$VAL_MAX_ITEMS" --seed "$r" --eval-steps "$EVAL_STEPS"
                   --patience "$PATIENCE" --max-steps "$SFT_MAX_STEPS" --epochs "$SFT_MAX_EPOCHS")
            [ "$PER_CALL" = 1 ] || flags+=(--chains) ;;
        # GRPO and OPSD read the round's own games, as chains unless PER_CALL=1
        *) runs=("$(round_dir "$r")"); [ "$PER_CALL" = 1 ] && flags=(--per-call) ;;
    esac
    # a crashed or killed training is relaunched (play_round's shape), up to twice, before the round fails for good;
    # a relaunch starts the round's one pass over from the same starting adapter, so it trains exactly what a clean
    # first attempt would have
    for attempt in 1 2 3; do
        if ! training "$r"; then
            stop_server || return 1
            if [ "$r" = 0 ]; then
                # the main stream's round 0 starts from the base model; the epilogue's round 0 starts from the stream's
                # round-R adapter instead (INIT_ADAPTER), since nothing is frozen for a trained row in the epilogue
                [ "$EPILOGUE" = 1 ] && init=(--init-adapter "$INIT_ADAPTER")
            else
                init=(--init-adapter "$TRAIN_DIR/r$((r - 1))/adapter")
            fi
            mkdir -p "$out"
            (cd ~/prime-rl && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup uv run --no-sync python \
                ~/loop/train.py --objective "$TRAIN_OBJECTIVE" --model "$MODEL" --gpus 8 \
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
    while pgrep -f "[t]rain.py " >/dev/null; do sleep 10; done
    say "train round $r done: $(python3 -c "
import json
d = json.load(open('$out/DONE'))
l = d['last'] or {}
capped = f\" (TIME-CAPPED after {d['minutes']} min)\" if d.get('time_capped') else ''
groups = d.get('groups')   # the stats block of grpo
items = d.get('items')     # of opd and opsd: calls, items (kept), excluded (never asked, by design), unscored,
                           # too_long; of sft (dagger): calls, items (kept), too_long
if groups:
    block = f'{groups[\"mixed\"]}/{groups[\"groups\"]} groups mixed, {groups[\"items\"]} calls, '
elif items and 'unscored' in items:
    block = (f'{items[\"items\"]}/{items[\"calls\"]} calls kept, {items.get(\"excluded\", 0)} excluded, '
             f'{items[\"unscored\"]} unscored, {items[\"too_long\"]} too long, ')
elif items:
    block = f'{items[\"items\"]}/{items[\"calls\"]} labels kept, {items[\"too_long\"]} too long, '
else:
    block = ''
dev = [e[\"dev_nll\"] for e in d.get('dev') or []]
held = f', held-out {dev[0]} to {min(dev)} (best at step {d.get(\"best_step\")})' if dev else ''
step = (f'nll {l.get(\"nll_per_token\")}{held}' if d.get('objective') == 'sft'
        else f'clip {l.get(\"clip_fraction\")} kl {l.get(\"approx_kl\")}')
print(f'{block}{d[\"steps\"]} steps, {step} entropy {l.get(\"entropy_mean\")}{capped}')" 2>/dev/null)"
    # one pass over all of the round's useful data is the contract: a time-capped pass trained on only part of it,
    # so the round must not count as a clean one
    python3 -c "import json,sys; sys.exit(1 if json.load(open('$out/DONE')).get('time_capped') else 0)" \
        || { say "train round $r time-capped at $TRAIN_MAX_MINUTES min: not one full pass, failing the round"; return 1; }
}

rounds() {   # rounds FROM TO: play (hinting it live, for opsd), and for a trained row score (opsd) or label (dagger)
            # and train, every round FROM..TO. Each stage checks its own "done" marker and returns at once when it
            # is already met, so always calling all of them here (rather than guessing from the markers first) costs
            # nothing on an already-finished round and never mistakes a time-capped train_round for one that need
            # not be called again.
    for ((r = $1; r <= $2; r++)); do
        say "round $r"
        [ "$ROW" = opsd ] && { start_hint_watcher "$r" || return 1; }
        if ! play_round "$r"; then
            # the watcher must be stopped on this path, not finished: a failed round must never produce a
            # complete, trusted hints.json for a later resume to find (kill_hint_watcher's own comment)
            [ "$ROW" = opsd ] && kill_hint_watcher "$r"
            return 1
        fi
        if [ "$ROW" = opsd ]; then finish_hint_watcher "$r" || return 1; fi
        if [ "$TRAINED" = 1 ]; then
            [ "$ROW" = opsd ] && { score_opsd "$r" || return 1; }
            [ "$ROW" = dagger ] && { label_round "$r" && split_round "$r" || return 1; }
            train_round "$r" || return 1
        fi
    done
}

{   # read whole before it runs: bash reads a script as it goes, and the code may be pushed again while this runs
    LAST=$((FIRST_ROUND + ROUNDS - 1))
    [ -f "$PLAY_DIR/paper-$FIRST_ROUND-$LAST.done" ] && { say "already done"; exit 0; }
    rounds "$FIRST_ROUND" "$LAST" || exit 1
    pkill -f "[v]llm serve"
    pkill -f "[e]xpert_relay.py --port $RELAY_PORT"   # the label step's relay (dagger), if this run started one
    touch "$PLAY_DIR/paper-$FIRST_ROUND-$LAST.done"
    say "done"
    exit 0
}
