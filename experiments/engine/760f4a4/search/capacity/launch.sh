#!/usr/bin/env bash
# launch.sh DEST RUN...: the capacity ablation's generators (generate.py, README.md) on the CPU machine at DEST
# (NODE_USER@IP), from a laptop or the operator's host with continual-arc-research's checkout at BASELINES
# (~/continual-arc-baselines). It
#   1. puts the generalisation search's tools on the machine (tools.sh into ~/gen/tools, capacity/ and glm_val.py
#      among them, swapped in whole),
#   2. has the machine fetch the turn-based bundle (code-dagger-rows-turns, the code the student's tests play on) and
#      the benchmark's data from the bucket on an hour's token of a research account (bucket.py get),
#   3. writes the OpenRouter key to ~/.openrouter.env by ssh's stdin, never on a command line,
#   4. with LEASE="gcp NAME", puts the operator's key on the machine and routes its lease (lease.py route), so that
#      the operator's hourly backup copies ~/runs/rt/cap-RUN and its capacity task reaches the machine,
#   5. runs start.sh there, detached: setup.sh once (NO_VLLM=1 NO_MODEL=1: the harness and prime-rl's environment, no
#      GPU software), then generate.py RUN for each RUN not already generating there.
# Launched again, it updates the tools and starts only the generators that are not running.
# PURPOSE (why, as detailed as a lease's) is required: each generator is a run under the run supervisor (keep.sh), which tools.sh
# ships with continual-arc-research's runs.py and the record of the tools' code.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,15p' "$0" >&2; exit 2; }
C=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$C/../.." && pwd)
. "$ROOT/keep.sh"
keep_purpose
G=$ROOT/search/generalisation
BASELINES=${BASELINES:-$HOME/continual-arc-baselines}
BUNDLE=code-dagger-rows-turns
DEST=$1; shift
for run in "$@"; do case $run in t25 | t100 | t400) ;; *) echo "RUN is t25, t100 or t400, not $run" >&2; exit 2 ;; esac; done
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new)
on() { "${SSH[@]}" "$DEST" "$@"; }

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
bash "$G/tools.sh" "$T"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$T" -czf - . | on 'rm -rf ~/gen/tools.new && mkdir -p ~/gen/tools.new && tar -C ~/gen/tools.new -xzf - \
    && rm -rf ~/gen/tools.old && { [ ! -d ~/gen/tools ] || mv ~/gen/tools ~/gen/tools.old; } && mv ~/gen/tools.new ~/gen/tools'
echo "tools: $(cat "$T/COMMIT")"
{
    bash "$BASELINES/scripts/secret.sh" --node-token
    echo "bundles/generalisation/$BUNDLE.tgz gen/bundle/$BUNDLE.tgz"
    echo "bundles/generalisation/data-cache.tgz gen/bundle/data.tgz"
} | on 'python3 ~/gen/tools/bucket.py get'
bash "$BASELINES/scripts/secret.sh" OPENROUTER_API_KEY | on 'umask 077; read -r k; [ -n "$k" ] && printf "OPENROUTER_API_KEY=%s\n" "$k" > ~/.openrouter.env'
if [ -n "${LEASE:-}" ]; then
    read -r vendor id <<< "$LEASE"
    key=$(cat "$BASELINES/infra/operator.pub")
    printf '%s\n' "$key" | on 'mkdir -p ~/.ssh && read -r k && { grep -qxF "$k" ~/.ssh/authorized_keys 2>/dev/null || echo "$k" >> ~/.ssh/authorized_keys; }'
    python3 "$BASELINES/scripts/lease.py" route "$vendor" "$id" --ssh "$DEST" --port 22
fi
on "mkdir -p ~/gen && ($(keep_env) setsid -f nohup bash ~/gen/tools/capacity/start.sh $* >> ~/gen/capacity-launch.log 2>&1 < /dev/null)"
cat <<EOF
generators for $* starting on $DEST (setup first if the machine is new). Follow them:
  ${SSH[*]} $DEST 'tail -n 3 ~/gen/setup.log; tail -n 3 ~/runs/rt/cap-*/progress'
EOF
