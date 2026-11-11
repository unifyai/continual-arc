#!/usr/bin/env bash
# modal-longtrain.sh JOB -- LONGTRAIN_ARGS...: longtrain.py on Modal (modal_longtrain.py), on H100:8 unless GPU names
# another. BUNDLE (the code bundle in the bucket the image is built from, fetched here into ~/.cache/gen-cache
# with the benchmark's data), MANIFEST (the bucket.py manifest of the run's inputs), TTL_HOURS (at most 24) and PURPOSE
# (as detailed as a lease's) are required; the workspace's token is read from Secret Manager into this process alone, as
# search/generalisation/modal-train.sh reads it. The job is detached: it runs on when this returns, and
# bash $BASELINES/scripts/modal.sh app stop <app id> ends it.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,7p' "$0" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")" && pwd)
BASELINES=${BASELINES:-$HOME/continual-arc-baselines}
C=$HOME/.cache/gen-cache
: "${BUNDLE:?BUNDLE names the code bundle}" "${MANIFEST:?MANIFEST is the inputs manifest}"
mkdir -p "$C"
{
    bash "$BASELINES/scripts/secret.sh" --node-token
    echo "bundles/generalisation/$BUNDLE.tgz $C/$BUNDLE.tgz"
    echo "bundles/generalisation/data-cache.tgz $C/data.tgz"
} | PYTHONPATH=$ROOT python3 "$ROOT/search/generalisation/bucket.py" get
. "$BASELINES/scripts/secret.sh"
MODAL_TOKEN_ID=$(secret MODAL_TOKEN_ID)
MODAL_TOKEN_SECRET=$(secret MODAL_TOKEN_SECRET)
MANIFEST=$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET BASELINES BUNDLE MANIFEST
exec uv run --no-project --with 'modal>=1.6' python "$ROOT/modal_longtrain.py" "$@"
