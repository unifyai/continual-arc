#!/usr/bin/env bash
# scratch.sh RUN [K]: training once at the end of DAgger run RUN (continual_arc_training.scratch), and its weights-alone
# tests, on the run's own node once every run there and the run's frozen tests are done; detached and resumable, a
# marker per stage in RUNS/rt/RUN/scratch:
#   weights  the continual student's phase-K adapter's two frozen tests from an empty harness memory under the run's own
#            task names (frozen.sh with MEMORY=none, into scratch/weights), beside its tests from the end-of-stream
#            memory in the run's frozen/: what the weights carry without what the harness kept
#   train    DAgger's phase K (default 199, the stream's last) trained from the base model into scratch/model, through
#            scratch/view, whose play/ is the run's, so nothing it logs reaches the run's own progress
#   test     the fresh adapter's two frozen tests from the run's end-of-stream harness memory (frozen.sh with ADAPTER,
#            into scratch/frozen), beside the continual student's own in the run's frozen/
#   summary  scratch/summary.json: both adapters' held-out loss on the same held-out games, and their plays of each
#            test, and the continual student's with its memory and without, compared instance by instance
# The continual student's phase K is the run's own train/pK (scratch/continual links to it), which must still hold its
# adapter: the run's last phase always does. Paths: lib.sh's.
# Usage (on the node): cd ~ && (setsid -f nohup bash ~/continual-arc/training/scripts/scratch.sh dagger-u1-lag1 \
#     > /dev/null 2>&1 < /dev/null)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=$1 K=${2:-199}
R=$RUNS/rt/$RUN O=$RUNS/rt/$RUN/scratch
P=scratch-$RUN   # the adapters' served names' prefix
export RUNS HF_HUB_OFFLINE=1
mkdir -p "$O"
say() { echo "$(date -u +%m-%d\ %H:%M) scratch: $*" >> "$O/progress"; }
done_() { touch "$O/$1.done"; say "$1 done"; }
stage() { [ -f "$O/$1.done" ]; }
{   # read whole before it runs: bash reads a script as it goes
    say "started: $RUN's phase $K, its weights alone and then trained from the base model"
    if pgrep -f "[c]ontinual_arc_training.realtime|[r]ealtime\.sh" > /dev/null; then
        say "a real-time run still plays or trains here: not started"; exit 1
    fi
    if pgrep -f "[f]rozen\.sh " > /dev/null || [ ! -f "$R/frozen/done" ]; then
        say "the run's frozen tests are not done (frozen.sh, $R/frozen): not started"; exit 1
    fi
    continual=$R/train/p$(printf %04d "$K")
    if [ ! -f "$continual/DONE" ] || [ ! -d "$continual/adapter" ]; then
        say "$continual holds no adapter: phase $K's was pruned"; exit 1
    fi
    ln -sfn "$continual" "$O/continual"
    if ! stage weights; then
        ADAPTER=$O/continual/adapter SERVED=$P-continual MEMORY=none OUT=$O/weights bash "$HERE/frozen.sh" "$RUN" \
            0,1,2,3,4,5,6,7 && [ -f "$O/weights/done" ] || { say "the weights-alone tests failed ($O/weights/progress)"; exit 1; }
        done_ weights
    fi
    if ! stage train; then
        pkill -f "[v]llm serve"; sleep 20
        mkdir -p "$O/view" && ln -sfn "$R/play" "$O/view/play"
        rm -rf "$O/model"
        # expandable segments for training alone: vLLM's custom all-reduce cannot open their memory over CUDA IPC, so
        # a server started under them fails as it captures its graphs
        (cd "$PRIME_RL" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --no-sync python \
            -m continual_arc_training.scratch train --run "$O/view" --phase "$K" --out "$O/model" --gpus 8) \
            >> "$O/train.log" 2>&1 \
            || { say "training failed ($O/train.log)"; exit 1; }
        say "trained: $(tail -1 "$O/train.log" | cut -c1-200)"
        done_ train
    fi
    if ! stage test; then
        ADAPTER=$O/model/adapter SERVED=$P OUT=$O/frozen bash "$HERE/frozen.sh" "$RUN" 0,1,2,3,4,5,6,7 \
            && [ -f "$O/frozen/done" ] || { say "the frozen tests failed ($O/frozen/progress)"; exit 1; }
        done_ test
    fi
    if ! stage summary; then
        (cd "$PRIME_RL" && uv run --no-sync python -m continual_arc_training.scratch summary "$O") >> "$O/summary.log" 2>&1 \
            || { say "the summary failed ($O/summary.log)"; exit 1; }
        say "summary: $(tail -1 "$O/summary.log" | cut -c1-400)"
        done_ summary
    fi
    say "done"
    exit 0
}
