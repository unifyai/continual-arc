#!/usr/bin/env bash
# repl-guard.sh: ends any prime-agent kernel REPL (python -m rlm.repl, the code a model runs inside an instance) whose
# memory passes LIMIT_GB, as the kernel OOM killer ended a 122 GB one on REF_TESTS_NODE on 31 Oct 2026, but before the
# machine swaps and slows every other play. The instance sees its tool call fail, as it would under the OOM killer.
LIMIT_GB=${LIMIT_GB:-64}
while true; do
    ps -eo pid=,rss=,args= | awk -v lim=$((LIMIT_GB*1048576)) '$3 ~ /python/ && /rlm\.repl/ && $2 > lim {print $1, $2}' |
    while read -r pid rss; do
        kill -9 "$pid" 2>/dev/null && echo "$(date -u +%m-%d\ %H:%M:%S) repl-guard.sh: ended rlm.repl $pid at $((rss/1048576)) GB ($(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null | cut -c1-80))" >> "$(dirname "$0")/repl-guard.log"
    done
    sleep 5
done
