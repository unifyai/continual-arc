#!/usr/bin/env bash
# modal-train.sh CELL [GPU]: CELL's train stage on Modal (modal_train.py), on H100:8 unless GPU names another (H200:8),
# its products to the bucket as generalisation/NAME/train.tgz when it ends; then launch.sh DEST CELL play on a machine
# that plays. TTL_HOURS (at most 24, Modal's longest) and PURPOSE (as detailed as a lease's) are required. The cell's
# bundle (cell.py bundle) comes from the bucket into ~/.cache/gen-cache, which the image is built from, and is
# named to modal_train.py in BUNDLE, since the image is made as the module loads; the workspace's token is read
# from Secret Manager into this process alone, as continual-arc-research's modal-run.sh reads it. The job is detached:
# it runs on when this returns, and bash $BASELINES/scripts/modal.sh app stop <app id> ends it.
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
G=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$G/../.." && pwd)
BASELINES=${BASELINES:-$HOME/continual-arc-baselines}
C=$HOME/.cache/gen-cache
CELL=$1
[ -f "$CELL" ] || CELL=$G/cells/$CELL.json
BUNDLE=$(python3 "$G/cell.py" bundle "$CELL")
mkdir -p "$C"
{
    bash "$BASELINES/scripts/secret.sh" --node-token
    echo "bundles/generalisation/$BUNDLE.tgz $C/$BUNDLE.tgz"
    echo "bundles/generalisation/data-cache.tgz $C/data.tgz"
} | PYTHONPATH=$ROOT python3 "$G/bucket.py" get
. "$BASELINES/scripts/secret.sh"
MODAL_TOKEN_ID=$(secret MODAL_TOKEN_ID)
MODAL_TOKEN_SECRET=$(secret MODAL_TOKEN_SECRET)
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET BASELINES BUNDLE
exec uv run --no-project --with 'modal>=1.6' python "$G/modal_train.py" "$@"
