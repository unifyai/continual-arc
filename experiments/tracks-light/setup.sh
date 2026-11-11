#!/usr/bin/env bash
# Brings a fresh CPU machine to tracks-light.sh's ready state for TRACK (arc2 or arc3), over ssh from a machine with a
# checkout of continual-arc at CODE and OPENROUTER_API_KEY in its environment: continual-arc at REF (v0.6.0 unless set)
# from the checkout CODE, unpacked into ~/continual-arc and its baselines synced; baselines/.env with the key, sent on
# ssh's stdin; prime-agent PRIME_AGENT_VERSION (0.9.6 unless set, the open models' rows' version in the paper's
# Appendix D, under which the adapter's Python skills were played); the track's data, fetched (arc3: its 25 pinned
# games through the arc-agi toolkit, then the CONFIG configuration, TRACK unless set, precomputed); and tracks-light.sh
# and tracks-light-floors.sh at ~/loop. Run again, it brings everything up to date and leaves the runs alone.
#
# The paper's rows were set up this way with REF=v0.6.0 (every row on ARC-AGI-2, and K, K0 and KA on ARC-AGI-3),
# REF=1c499b7 (ARC-AGI-3's KE pilot) and REF=9f7ef5e (ARC-AGI-3's KP pilot), and the floors KC0 and KC with
# REF=v0.4.0 PRIME_AGENT_VERSION=0.9.5 CONFIG=TRACK-small. The original script also registered the machine with
# unify's own fleet (its ssh keys, its lease, the hourly copy of ~/runs/paper to unify's storage, the key read from
# unify's secret store) and staged unify's run supervisor beside tracks-light.sh; none of that changes what a cell plays.
# Usage: CODE=<continual-arc checkout> OPENROUTER_API_KEY=... bash setup.sh HOST TRACK   (HOST an ssh destination)
set -euo pipefail
HOST=${1:?set HOST: the machine, as ssh names it}; TRACK=${2:?set TRACK: arc2 or arc3}
CODE=${CODE:?set CODE: a git checkout of unifyai/continual-arc}; REF=${REF:-v0.6.0}
PRIME_AGENT_VERSION=${PRIME_AGENT_VERSION:-0.9.6}; CONFIG=${CONFIG:-$TRACK}
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY: the OpenRouter key the cells play on}"
case $TRACK in arc2|arc3) ;; *) echo "TRACK is arc2 or arc3, not $TRACK" >&2; exit 1;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
on() { ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new "$HOST" "$@"; }
on 'command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1'
git -C "$CODE" archive "$REF" | on 'rm -rf ~/continual-arc.new && mkdir ~/continual-arc.new && tar -C ~/continual-arc.new -xf - \
    && { [ -f ~/continual-arc/baselines/.env ] && cp -p ~/continual-arc/baselines/.env ~/continual-arc.new/baselines/; \
         [ -d ~/continual-arc/.venv ] && mv ~/continual-arc/.venv ~/continual-arc.new/; true; } \
    && rm -rf ~/continual-arc && mv ~/continual-arc.new ~/continual-arc'
echo "continual-arc $(git -C "$CODE" rev-parse --short "$REF^{commit}") ($REF) unpacked"
on 'cd ~/continual-arc/baselines && ~/.local/bin/uv sync -q && echo "baselines synced"'
printf '%s\n' "$OPENROUTER_API_KEY" | on 'umask 077; read -r k; [ -n "$k" ] && printf "OPENROUTER_API_KEY=%s\n" "$k" > ~/continual-arc/baselines/.env && echo "key written"'
on "export PATH=\$HOME/.local/bin:\$PATH; [ \"\$(prime-agent --version 2>/dev/null | head -1)\" = $PRIME_AGENT_VERSION ] || curl --proto '=https' --proto-redir '=https' -fsSL https://app.primeintellect.ai/prime-agent/install.sh | PRIME_AGENT_VERSION=$PRIME_AGENT_VERSION sh > /dev/null 2>&1; echo \"prime-agent \$(prime-agent --version | head -1)\"; bwrap --ro-bind / / --dev /dev --proc /proc --unshare-pid --die-with-parent true && echo 'bubblewrap works'"
on "~/continual-arc/.venv/bin/continual-arc data fetch --datasets $TRACK > ~/data-fetch.log 2>&1 && tail -n 3 ~/data-fetch.log"
if [ "$TRACK" = arc3 ]; then
    on "cd ~ && ~/continual-arc/.venv/bin/continual-arc precompute --config $CONFIG --seed 0 > ~/precompute.log 2>&1; tail -n 3 ~/precompute.log"
fi
on "~/continual-arc/.venv/bin/continual-arc data status --datasets $TRACK 2>&1 | tail -n 3"
# on ssh's stdin and renamed into place, never rsync's quick check, which macOS's rsync was seen to skip a changed file
# by; the rename leaves a running script reading the copy it started from
for f in tracks-light.sh tracks-light-floors.sh; do
    on "mkdir -p ~/loop && cat > ~/loop/.$f.new && mv ~/loop/.$f.new ~/loop/$f" < "$HERE/$f" && echo "$f copied"
done
echo "READY $HOST $TRACK"
