#!/usr/bin/env bash
# start.sh RUN...: on the generators' machine, from ~/gen/tools/capacity (launch.sh runs it, detached): setup.sh once
# (NO_VLLM=1 NO_MODEL=1, the turn-based bundle), then generate.py for each RUN not already generating, each detached,
# its output in ~/runs/rt/cap-RUN/generate.log.
set -uo pipefail
for k in "$(dirname "${BASH_SOURCE[0]}")"/{,../,../../}keep.sh; do [ -f "$k" ] && . "$k" && break; done   # the run supervisor
type keep_exec > /dev/null 2>&1 || { echo "refused: no keep.sh in ~/gen/tools, the run supervisor's helper" >&2; exit 1; }
keep_purpose || exit 1   # each generator is a run of its own (keep.sh), its folder ~/runs/rt/cap-RUN
export PATH=$HOME/.local/bin:$PATH
{ grep -q "setup done" ~/gen/setup.log 2>/dev/null && [ -s ~/gen/bundle/code.unpacked ]; } \
    || BUNDLE=code-dagger-rows-turns NO_VLLM=1 NO_MODEL=1 bash ~/gen/tools/setup.sh
{ tail -n 1 ~/gen/setup.log | grep -q "setup done" && grep -qx ok ~/gen/sandbox; } \
    || { echo "$(date -u +%m-%d\ %H:%M) the machine is not set up: ~/gen/setup.log, ~/gen/sandbox"; exit 1; }
for run in "$@"; do
    pgrep -f "[g]enerate.py $run\$" > /dev/null && { echo "$run is generating already"; continue; }
    mkdir -p ~/runs/rt/cap-"$run"
    (cd ~/prime-rl && PYTHONPATH=$HOME/loop setsid -f nohup bash "$KEEP_HERE/keep.sh" ~/runs/rt/cap-"$run" --resume \
        --log ~/runs/rt/cap-"$run"/progress --code "$KEEP_HERE" -- uv run --no-sync python ~/gen/tools/capacity/generate.py \
        "$run" >> ~/runs/rt/cap-"$run"/generate.log 2>&1 < /dev/null)
    echo "$(date -u +%m-%d\ %H:%M) started $run"
done
