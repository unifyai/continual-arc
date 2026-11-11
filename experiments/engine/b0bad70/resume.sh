#!/usr/bin/env bash
# On the node after a restart: the sandbox setting, the server on :8101 as serve.sh last started it (its arguments in
# ~/runs/loop/server.args), then every listed play whose records are short of its schedule, resumed from its records.
# Exits 0 when every listed play is complete, 3 while any is short (running or just relaunched), 1 on an error.
# Usage: bash ~/loop/resume.sh NAME:SCHED ...
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
sudo sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0
if ! curl -fsS -m 5 localhost:8101/health >/dev/null 2>&1; then
    [ -f ~/runs/loop/server.args ] || { echo "no server on :8101 and no server.args: start one with serve.sh"; exit 1; }
    bash ~/loop/serve.sh $(cat ~/runs/loop/server.args) || exit 1
fi
cd ~/prime-rl
short=0
for spec in "$@"; do
    n=${spec%%:*}; s=${spec##*:}; d=~/runs/loop/$n
    want=$(python3 -c "import json;print(len(json.load(open('$HOME/runs/loop/$s.json'))['slots']))") || exit 1
    have=$( [ -f $d/records.jsonl ] && wc -l < $d/records.jsonl || echo 0)
    [ "$have" -lt "$want" ] && short=1
    if [ "$have" -lt "$want" ] && ! pgrep -f "[r]un-dir $d( |$)" >/dev/null; then
        extra=""; [ -f ~/runs/loop/$n.args ] && extra=$(cat ~/runs/loop/$n.args)
        setsid nohup uv run --no-sync python ~/loop/loop.py play --schedule ~/runs/loop/$s.json --run-dir $d $extra >> ~/runs/loop/$n.log 2>&1 < /dev/null &
        echo "relaunched $n ($have/$want)"
    fi
done
[ "$short" = 0 ] || exit 3
