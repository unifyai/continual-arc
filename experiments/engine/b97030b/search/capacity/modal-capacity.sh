#!/usr/bin/env bash
# modal-capacity.sh deploy | spawn RUN: the capacity ablation's trainer on Modal (modal_capacity.py), from a checkout
# with continual-arc-research's at BASELINES (~/continual-arc-baselines). deploy builds the app MODAL_CAPACITY in the
# research environment from the turn-based bundle (fetched from the bucket into ~/.cache/gen-cache) and this
# checkout's tools, so deploy again after a change to trainer.py or capacity.py. spawn starts RUN's phase job by hand,
# as brain's capacity task does each time a set waits (PURPOSE overrides the job's audit purpose). The workspace's token
# is read from Secret Manager into this process alone, as modal-train.sh reads it.
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,7p' "$0" >&2; exit 2; }
C=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$C/../.." && pwd)
BASELINES=${BASELINES:-$HOME/continual-arc-baselines}
. "$BASELINES/scripts/secret.sh"
MODAL_TOKEN_ID=$(secret MODAL_TOKEN_ID)
MODAL_TOKEN_SECRET=$(secret MODAL_TOKEN_SECRET)
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
case $1 in
deploy)
    CACHE=$HOME/.cache/gen-cache
    mkdir -p "$CACHE"
    {
        bash "$BASELINES/scripts/secret.sh" --node-token
        echo "bundles/generalisation/code-dagger-rows-turns.tgz $CACHE/code-dagger-rows-turns.tgz"
        echo "bundles/generalisation/data-cache.tgz $CACHE/data.tgz"
    } | PYTHONPATH=$ROOT python3 "$ROOT/search/generalisation/bucket.py" get
    cd "$C" && CAPACITY_DEPLOY=1 exec uv run --no-project --with 'modal>=1.6' modal deploy --env research modal_capacity.py ;;
spawn)
    RUN=${2:?RUN}
    PURPOSE=${PURPOSE:-"Continual-ARC capacity ablation, run $RUN: Qwen3.5-9B trained from the base model on GLM-5.3-flash's own games, one phase per waiting batch (measure the new batch's loss, then one pass over every batch so far), started by hand with search/capacity/modal-capacity.sh"}
    cd "$C" && exec uv run --no-project --with 'modal>=1.6' python -c "
import json, os, sys, modal
fn = modal.Function.from_name('MODAL_CAPACITY', 'phase', environment_name='research')
call = fn.spawn(sys.argv[1], {'name': 'cap-' + sys.argv[1], 'gpu': 'H100:8', 'timeout_hours': 24, 'owner': os.environ.get('USER', 'unknown'), 'purpose': sys.argv[2]})
print(json.dumps({'run': sys.argv[1], 'call': call.object_id}))" "$RUN" "$PURPOSE" ;;
*) echo "deploy or spawn RUN, not $1" >&2; exit 2 ;;
esac
