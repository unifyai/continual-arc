#!/usr/bin/env bash
# tools.sh DIR: the files a machine runs the generalisation search's cells with, copied into DIR (launch.sh ships them
# to ~/gen/tools, modal_train.py mounts them there): this folder's tools and cells, glm_val.py, the capacity
# ablation's folder (search/capacity, as capacity/), grid-exp.sh and grid-vals.sh beside it, and from the checkout's
# top gcs_download.py and gcs_upload.py (bucket.py's helpers), scratch.py (the from-scratch retrain, whose trainer
# arguments replay.py takes), frozen.sh, frozen.py and the two frozen tests' schedules. COMMIT
# names the checkout's commit, with "-dirty" when its tracked files hold changes, which go as they are.
set -euo pipefail
G=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$G/../.." && pwd)
D=$1
mkdir -p "$D/cells" "$D/schedules"
cp "$G/bucket.py" "$G/cell.py" "$G/replay.py" "$G/setup.sh" "$G/glm_val.py" "$D/"
mkdir -p "$D/capacity" && cp "$ROOT"/search/capacity/*.py "$ROOT"/search/capacity/*.sh "$D/capacity/"
cp "$G"/cells/*.json "$D/cells/"
cp "$ROOT/search/grid-exp.sh" "$ROOT/search/grid-vals.sh" "$D/"
cp "$ROOT/gcs_download.py" "$ROOT/gcs_upload.py" "$ROOT/scratch.py" "$ROOT/frozen.sh" "$ROOT/frozen.py" "$D/"
cp "$ROOT/schedules/arc1-small-held-out-test-seed0.json" "$ROOT/schedules/arc1-small-seen-test-seed0.json" "$D/schedules/"
echo "$(git -C "$ROOT" rev-parse HEAD)$(git -C "$ROOT" diff --quiet HEAD || echo -dirty)" > "$D/COMMIT"
