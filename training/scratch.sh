#!/usr/bin/env bash
# scratch.sh RUN [K]: the from-scratch retrain of real-time run RUN at its stream's end (scratch.py's docstring), and
# its weights-alone tests, on the run's own node once every run there is done, its frozen tests too, detached,
# resumable (a marker per stage in RUNS/rt/RUN/scratch):
#   weights the continual student's phase-K adapter's two frozen tests from an empty harness memory under the run's own
#          task names (frozen.sh with MEMORY=none, into scratch/weights), beside its tests from the end-of-stream memory,
#          which frozen.sh played into the run's frozen/: what the weights carry without what the harness kept. First,
#          so that it is played whatever becomes of the retrain
#   train  DAgger's phase K (default 199, the stream's last) from the base model into scratch/model, with the node's own
#          realtime.py and train.py (scratch.py train, CODE on the path), through scratch/view, whose play/ is the
#          run's, so nothing it logs reaches the run's own progress
#   val    the continual student's phase-K adapter (scratch/continual, copied there by scratch-launch.sh) and the fresh
#          one, frozen, each playing the 100 validation puzzles of the earlier experiments on four GPUs (grid-vals.sh)
#   test   the fresh adapter's two frozen tests from the run's end-of-stream harness memory (frozen.sh with ADAPTER,
#          into scratch/frozen), beside the continual student's own, which frozen.sh played into the run's frozen/
#   summary scratch/summary.json (scratch.py summary): both adapters' held-out loss on the same held-out games, their
#          plays of each test compared instance by instance, and the continual student's with its memory and without
# CODE is the training code the run trained with on the node (default continual-arc/training under the node's home),
# PRIME_RL (default ~/prime-rl) and RUNS (default ~/runs) the node's folders.
# Usage (on the node, from the folder scratch-launch.sh copies it to):
#   cd ~ && (setsid -f nohup bash ~/scratch-tools/scratch.sh dagger-u1-lag1 > /dev/null 2>&1 < /dev/null)
set -uo pipefail
RUN=$1 K=${2:-199}
CODE=${CODE:-$HOME/continual-arc/training} PRIME_RL=${PRIME_RL:-$HOME/prime-rl} RUNS=${RUNS:-$HOME/runs}
R=$RUNS/rt/$RUN O=$RUNS/rt/$RUN/scratch T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=${EXP:-$HOME/exp}   # grid-vals.sh's folder of per-model markers and measures
P=scratch-$RUN   # the grid-vals.sh names' prefix: EXP/$P-..., RUNS/loop/val-$P-...
export CODE PRIME_RL RUNS EXP
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1
mkdir -p "$O"
say() { echo "$(date -u +%m-%d\ %H:%M) scratch: $*" >> "$O/progress"; }
done_() { touch "$O/$1.done"; say "$1 done"; }
stage() { [ -f "$O/$1.done" ]; }
{   # read whole before it runs: bash reads a script as it goes
    say "started: $RUN's phase $K, its weights alone and then trained from the base model, beside $O/continual"
    if pgrep -f "[r]ealtime\.(py|sh)" > /dev/null; then say "a real-time run still plays or trains here: not started"; exit 1; fi
    if pgrep -f "[f]rozen\.sh " > /dev/null || [ ! -f "$R/frozen/done" ]; then
        say "the run's frozen tests are not done (frozen.sh, $R/frozen): not started"; exit 1
    fi
    if [ ! -f "$O/continual/DONE" ] || [ ! -d "$O/continual/adapter" ]; then
        say "no continual adapter in $O/continual: scratch-launch.sh copies train/p$(printf %04d "$K") there"; exit 1
    fi
    if ! stage weights; then
        ADAPTER=$O/continual/adapter SERVED=$P-continual MEMORY=none OUT=$O/weights bash "$T/frozen.sh" "$RUN" \
            0,1,2,3,4,5,6,7 && [ -f "$O/weights/done" ] || { say "the weights-alone tests failed ($O/weights/progress)"; exit 1; }
        done_ weights
    fi
    if ! stage train; then
        pkill -f "[v]llm serve"; sleep 20
        mkdir -p "$O/view" && ln -sfn "$R/play" "$O/view/play"
        rm -rf "$O/model"
        # expandable segments for training alone: vLLM's custom all-reduce cannot open their memory over CUDA IPC, so
        # a server started under them fails as it captures its graphs
        (cd "$PRIME_RL" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$CODE uv run --no-sync python \
            "$T/scratch.py" train --run "$O/view" \
            --phase "$K" --out "$O/model" --gpus 8) >> "$O/train.log" 2>&1 \
            || { say "training failed ($O/train.log)"; exit 1; }
        say "trained: $(tail -1 "$O/train.log" | cut -c1-200)"
        done_ train
    fi
    # play KIND: both adapters frozen through grid-vals.sh, then their plays and measures copied under $O
    play() {
        local kind=$1 who
        SCHED=${2:-} bash "$T/grid-vals.sh" "$P-$kind-continual:adapter:$O/continual/adapter" \
            "$P-$kind-scratch:adapter:$O/model/adapter" || return 1
        mkdir -p "$O/plays" "$O/exp"
        for who in continual scratch; do
            rm -rf "$O/plays/$kind-$who" "$O/exp/$kind-$who"
            cp -a "$RUNS/loop/val-$P-$kind-$who" "$O/plays/$kind-$who" && cp -a "$EXP/$P-$kind-$who" "$O/exp/$kind-$who" \
                || return 1
        done
    }
    if ! stage val; then
        play val || { say "the validation plays failed ($EXP/vals/progress)"; exit 1; }
        done_ val
    fi
    if ! stage test; then
        ADAPTER=$O/model/adapter SERVED=$P OUT=$O/frozen bash "$T/frozen.sh" "$RUN" 0,1,2,3,4,5,6,7 \
            && [ -f "$O/frozen/done" ] || { say "the frozen tests failed ($O/frozen/progress)"; exit 1; }
        done_ test
    fi
    if ! stage summary; then
        (cd "$PRIME_RL" && uv run --no-sync python "$T/scratch.py" summary "$O") >> "$O/summary.log" 2>&1 \
            || { say "the summary failed ($O/summary.log)"; exit 1; }
        say "summary: $(tail -1 "$O/summary.log" | cut -c1-400)"
        done_ summary
    fi
    say "done"
    exit 0
}
