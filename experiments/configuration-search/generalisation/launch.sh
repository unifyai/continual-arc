#!/usr/bin/env bash
# launch.sh DEST CELL [STAGES]: one of the generalisation search's cells (CELL: cells/NAME.json, or NAME) on whatever
# machine answers ssh at DEST (user@host, or a Host of ~/.ssh/config; PORT and JUMP give a pod's port or a gateway),
# whichever vendor made it: a bare machine with 8 NVIDIA GPUs is enough, since it is set up from the bucket alone. It
#   1. puts this folder's tools on the machine (tools.sh, into ~/gen/tools, swapped in whole so that no running script
#      is rewritten under it),
#   2. fetches the cell's code bundle (its "bundle", code-dagger-rows unless named: the DAgger rows' code), the
#      benchmark's data and the cell's inputs from the bucket, which the machine downloads itself on an hour's token of
#      a research account (bucket.py get; it holds no credential),
#   3. leaves an upload session per stage for the stage's products (bucket.py session), and for a label stage the
#      OpenRouter key in ~/.openrouter.env, both by ssh's stdin, never on a command line,
#   4. with LEASE="VENDOR ID", puts the operator's key on the machine and routes its lease (lease.py route), so that the
#      operator's hourly backup copies ~/runs/rt, where the cell writes everything it finds,
#   5. starts setup.sh on the cell's bundle, unless the machine is set up and holds a bundle's trees (a machine whose
#      own trees were set aside, ~/gen/bundle/code.unpacked with them, is set up again), then cell.py run, detached,
#      which refuses a machine set up from another bundle, and says how to follow it.
# STAGES (label, train, play; default all the cell's) runs part of a cell here: an earlier stage that ran elsewhere
# comes as its products' tarball. Plays need the user NODE_USER and a working bubblewrap (VMs: a GPU cloud, Lambda), training
# any 8 GPUs, Modal's included (modal-train.sh). BASELINES is continual-arc-research's checkout (secret.sh, lease.py
# and the operator's key), ~/continual-arc-baselines unless set. README.md says more.
# PURPOSE (why, as detailed as a lease's) is required: the cell is a run under the run supervisor (keep.sh), which tools.sh
# ships with continual-arc-research's runs.py and the record of the tools' code.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,20p' "$0" >&2; exit 2; }
G=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$G/../.." && pwd)
. "$ROOT/keep.sh"
keep_purpose
BASELINES=${BASELINES:-$HOME/continual-arc-baselines}
DEST=$1 CELL=$2 PICK=${3:-}
[ -f "$CELL" ] || CELL=$G/cells/$CELL.json
NAME=$(basename "$CELL" .json)
STAGES=$(python3 "$G/cell.py" stages "$CELL" ${PICK:+--stages "$PICK"})
python3 "$G/cell.py" check "$CELL" > /dev/null
BUNDLE=$(python3 "$G/cell.py" bundle "$CELL")
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${PORT:+-p "$PORT"} ${JUMP:+-J "$JUMP"})
on() { "${SSH[@]}" "$DEST" "$@"; }
token() { bash "$BASELINES/scripts/secret.sh" --node-token; }

read -r home gpus < <(on 'echo "$HOME $(nvidia-smi -L 2>/dev/null | wc -l)"')
case ,$STAGES, in *,train,* | *,play,*) [ "$gpus" -ge 8 ] || { echo "$DEST has $gpus GPUs; train and play need 8"; exit 1; } ;; esac
case ,$STAGES, in *,play,*) [ "$home" = /home/NODE_USER ] \
    || { echo "$DEST runs as HOME $home: plays run as the user NODE_USER (the loops' scripts and the prompts hold /home/NODE_USER)"; exit 1; } ;; esac
if on "pgrep -f '[c]ell.py run .*cells/$NAME.json' > /dev/null"; then
    echo "$NAME is running on $DEST already: ssh to it and read ~/runs/rt/gen-$NAME/progress"; exit 1
fi

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
bash "$G/tools.sh" "$T"
cp "$CELL" "$T/cells/$NAME.json"
# no extended attributes: a Mac's provenance ones make GNU tar on the machine warn about every file
COPYFILE_DISABLE=1 tar --no-xattrs -C "$T" -czf - . | on 'rm -rf ~/gen/tools.new && mkdir -p ~/gen/tools.new && tar -C ~/gen/tools.new -xzf - \
    && rm -rf ~/gen/tools.old && { [ ! -d ~/gen/tools ] || mv ~/gen/tools ~/gen/tools.old; } && mv ~/gen/tools.new ~/gen/tools'
echo "tools: $(cat "$T/COMMIT")"

done_here=$(on "ls ~/runs/rt/gen-$NAME 2>/dev/null | sed -n 's/\.done\$//p' | paste -sd, -")
{
    token
    echo "bundles/generalisation/$BUNDLE.tgz gen/bundle/$BUNDLE.tgz"
    echo "bundles/generalisation/data-cache.tgz gen/bundle/data.tgz"
    python3 "$G/cell.py" inputs "$CELL" --stages "$STAGES" --done "$done_here"
} | on 'python3 ~/gen/tools/bucket.py get'

uploads="{"
for s in ${STAGES//,/ }; do
    url=$(token | PYTHONPATH=$ROOT python3 "$G/bucket.py" session "generalisation/$NAME/$s.tgz")
    uploads+="\"$s\": \"$url\","
done
printf '%s' "${uploads%,}}" | on "umask 077 && mkdir -p ~/gen/$NAME && cat > ~/gen/$NAME/uploads.json"
case ,$STAGES, in *,label,*)
    bash "$BASELINES/scripts/secret.sh" OPENROUTER_API_KEY | on 'umask 077; read -r k; [ -n "$k" ] && printf "OPENROUTER_API_KEY=%s\n" "$k" > ~/.openrouter.env'
    echo "the OpenRouter key is in ~/.openrouter.env" ;;
esac

if [ -n "${LEASE:-}" ]; then
    read -r vendor id <<< "$LEASE"
    key=$(cat "$BASELINES/infra/operator.pub")
    printf '%s\n' "$key" | on 'mkdir -p ~/.ssh && read -r k && { grep -qxF "$k" ~/.ssh/authorized_keys 2>/dev/null || echo "$k" >> ~/.ssh/authorized_keys; }'
    # where the machine answers, as ssh resolves DEST (a Host alias, PORT and JUMP included)
    u="" h="" p=22 j=""
    eval "$("${SSH[@]}" -G "$DEST" 2> /dev/null | awk '$1 == "user" {print "u=" $2} $1 == "hostname" {print "h=" $2}
                                           $1 == "port" {print "p=" $2} $1 == "proxyjump" && $2 != "none" {print "j=" $2}')"
    j=${j//[\[\]]/}   # newer OpenSSH's -G prints a jump host's address in brackets, which lease.py's --jump refuses
    python3 "$BASELINES/scripts/lease.py" route "$vendor" "$id" --ssh "$u@$h" --port "$p" ${j:+--jump "$j"}
fi

on "cd ~ && mkdir -p ~/gen/$NAME && ($(keep_env) setsid -f nohup bash -c '{ { grep -q \"setup done\" ~/gen/setup.log 2>/dev/null \
    && [ -s ~/gen/bundle/code.unpacked ]; } || BUNDLE=$BUNDLE bash ~/gen/tools/setup.sh; } \
    && tail -n 1 ~/gen/setup.log | grep -q \"setup done\" \
    && python3 ~/gen/tools/cell.py run ~/gen/tools/cells/$NAME.json --stages $STAGES' >> ~/gen/$NAME/launch.log 2>&1 < /dev/null)"
cat <<EOF
$NAME ($STAGES, on $BUNDLE) started on $DEST. Follow it:
  ${SSH[*]} $DEST 'tail -n 3 ~/gen/setup.log; tail -n 5 ~/runs/rt/gen-$NAME/progress'
Each stage's products reach the bucket as generalisation/$NAME/<stage>.tgz when it ends.
EOF
