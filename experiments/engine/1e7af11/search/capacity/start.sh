#!/usr/bin/env bash
# start.sh RUN...: on the generators' machine, from ~/gen/tools/capacity (launch.sh runs it, detached): setup.sh once
# (NO_VLLM=1 NO_MODEL=1, the turn-based bundle), then generate.py for each RUN not already generating, each detached,
# its output in ~/runs/rt/cap-RUN/generate.log.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
{ grep -q "setup done" ~/gen/setup.log 2>/dev/null && [ -s ~/gen/bundle/code.unpacked ]; } \
    || BUNDLE=code-dagger-rows-turns NO_VLLM=1 NO_MODEL=1 bash ~/gen/tools/setup.sh
{ tail -n 1 ~/gen/setup.log | grep -q "setup done" && grep -qx ok ~/gen/sandbox; } \
    || { echo "$(date -u +%m-%d\ %H:%M) the machine is not set up: ~/gen/setup.log, ~/gen/sandbox"; exit 1; }
for run in "$@"; do
    pgrep -f "[g]enerate.py $run\$" > /dev/null && { echo "$run is generating already"; continue; }
    mkdir -p ~/runs/rt/cap-"$run"
    (cd ~/prime-rl && PYTHONPATH=$HOME/loop setsid -f nohup uv run --no-sync python ~/gen/tools/capacity/generate.py \
        "$run" >> ~/runs/rt/cap-"$run"/generate.log 2>&1 < /dev/null)
    echo "$(date -u +%m-%d\ %H:%M) started $run"
done
