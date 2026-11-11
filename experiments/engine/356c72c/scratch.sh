#!/usr/bin/env bash
# scratch.sh RUN [K]: the from-scratch retrain of real-time run RUN at its stream's end (scratch.py's docstring; the author, 21
# Oct 2026), on the run's own node once every run there is done, its frozen tests too, detached, resumable (a marker
# per stage in ~/runs/rt/RUN/scratch, where the operator's hourly backup finds everything it writes):
#   train  DAgger's phase K (default 199, the stream's last) from the base model into scratch/model, with the node's own
#          realtime.py and train.py (scratch.py train), through scratch/view, whose play/ is the run's, so nothing it
#          logs reaches the run's own progress
#   val    the continual student's phase-K adapter (scratch/continual, copied from the bucket by scratch-launch.sh)
#          and the fresh one, frozen, each playing R's 100 validation puzzles on four GPUs (search/grid-vals.sh)
#   test   the fresh adapter's two frozen tests from the run's end-of-stream harness memory (frozen.sh with ADAPTER,
#          into scratch/frozen), beside the continual student's own, which frozen.sh played into the run's frozen/
#   summary scratch/summary.json (scratch.py summary): both adapters' held-out loss on the same held-out games, and
#          their plays of each test compared instance by instance
# Usage (on the node, from the folder scratch-launch.sh copies it to):
#   cd ~ && (setsid -f nohup bash ~/scratch-tools/scratch.sh dagger-u1-lag1 > /dev/null 2>&1 < /dev/null)
set -uo pipefail
RUN=$1 K=${2:-199}
R=~/runs/rt/$RUN O=~/runs/rt/$RUN/scratch T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=scratch-$RUN   # the grid-vals.sh names' prefix: ~/exp/$P-..., ~/runs/loop/val-$P-...
export PATH=$HOME/.local/bin:$PATH HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$O"
say() { echo "$(date -u +%m-%d\ %H:%M) scratch: $*" >> "$O/progress"; }
done_() { touch "$O/$1.done"; say "$1 done"; }
stage() { [ -f "$O/$1.done" ]; }
{   # read whole before it runs: bash reads a script as it goes
    say "started: $RUN's phase $K from the base model, beside $O/continual"
    if pgrep -f "[r]ealtime\.(py|sh)" > /dev/null; then say "a real-time run still plays or trains here: not started"; exit 1; fi
    if pgrep -f "[f]rozen\.sh " > /dev/null || [ ! -f "$R/frozen/done" ]; then
        say "the run's frozen tests are not done (frozen.sh, $R/frozen): not started"; exit 1
    fi
    if [ ! -f "$O/continual/DONE" ] || [ ! -d "$O/continual/adapter" ]; then
        say "no continual adapter in $O/continual: scratch-launch.sh copies train/p$(printf %04d "$K") there"; exit 1
    fi
    if ! stage train; then
        pkill -f "[v]llm serve"; sleep 20
        mkdir -p "$O/view" && ln -sfn "$R/play" "$O/view/play"
        rm -rf "$O/model"
        (cd ~/prime-rl && PYTHONPATH=$HOME/loop uv run --no-sync python "$T/scratch.py" train --run "$O/view" \
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
            cp -a ~/runs/loop/val-"$P-$kind-$who" "$O/plays/$kind-$who" && cp -a ~/exp/"$P-$kind-$who" "$O/exp/$kind-$who" \
                || return 1
        done
    }
    if ! stage val; then
        play val || { say "the validation plays failed (~/exp/vals/progress)"; exit 1; }
        done_ val
    fi
    if ! stage test; then
        ADAPTER=$O/model/adapter SERVED=$P OUT=$O/frozen bash "$T/frozen.sh" "$RUN" 0,1,2,3,4,5,6,7 \
            && [ -f "$O/frozen/done" ] || { say "the frozen tests failed ($O/frozen/progress)"; exit 1; }
        done_ test
    fi
    if ! stage summary; then
        (cd ~/prime-rl && uv run --no-sync python "$T/scratch.py" summary "$O") >> "$O/summary.log" 2>&1 \
            || { say "the summary failed ($O/summary.log)"; exit 1; }
        say "summary: $(tail -1 "$O/summary.log" | cut -c1-400)"
        done_ summary
    fi
    say "done"
    exit 0
}
