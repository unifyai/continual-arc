#!/usr/bin/env bash
# Backs up one of the paper's runs from its node while it runs: a paper.sh row or cut (RUNS/paper/ROW, ROW-uN at N
# users) or a real-time run (realtime.sh's RUNS/rt/NAME). The small files come to a local folder, and with GCS=1 the
# whole run goes from the node straight to a bucket. A rented node's disk goes with the node, so run it every hour or
# so for every run on every node: everything a run writes lives only on the node until it is copied off.
#   GCS=1      uploads the run whole from the node itself, with gcs_upload.py and a short-lived token sent on stdin (no
#              bytes through this machine): every file but the per-user Python environments, caches, the server's
#              copies of the adapters and a phase's step checkpoints (gcs_upload.py's PRUNE), each user's
#              session-artifacts included. The uploader's code goes with the command, so nothing is written on the
#              node. Each object records the version of the file it holds, so a rerun reads and sends only what
#              changed. TRAJECTORY_BUCKET (gs://BUCKET/PREFIX, required) is where runs go, NAME (required) where this
#              one lands, $TRAJECTORY_BUCKET/runs/gpu-nodes/$NAME (for instance <node>/<run>), and TOKEN_CMD
#              (required) prints the token: an hour's OAuth token for that bucket alone, for the node to upload with.
#   ADAPTER=1  also pulls the newest adapter of each training folder (train*/rN, or a real-time run's train/pNNNN)
#              to the local folder, deleting older ones from the copy (their DONE and phase.json stay). It needs the
#              disk for it; GCS=1 is how an adapter is kept safe.
#   CAPTURED=1 also pulls captured/ to the local folder. The exception, not the rule: about 1 GB a round at 100 users.
# With neither ADAPTER=1 nor GCS=1 this only pulls small files, and says so when an adapter from an earlier
# ADAPTER=1 run is still taking up room in the copy (it never deletes one it was not asked to manage).
# A run's progress only ever grows, so a copy whose progress is not where the node's starts is of an earlier run under
# the same name (a run started over, its old folder renamed or deleted on the node): the local copy moves aside to
# LOCAL_DIR.replaced-<UTC time>, and with GCS=1 the bucket's to the same name, both kept whole, before this run's
# copy begins, so that two runs never mix in one folder.
# The small-file pull is one filtered rsync over ssh (with keepalives and a timeout): the run's progress, *.log,
# guidelines.md and its loops' *.done; every play folder's records.jsonl, summary.json, meta.json, config.json,
# driver.json, schedule.json, alias-salt and hints-cost.json; every round's memory.json, hints.json, the hints/ cache,
# hints-round-cost.json, served.json, *.done and its labels/, the labeller's output (points.jsonl, the oracle's reply
# to each call, set.json, split/split.json and its markers), a real-time run's and a paper.sh row's alike; and every
# training round's or phase's DONE, phase.json and rounds.json, with train/latest.json. Never opd_scores.json*, the
# users' own state or the label sets' copies of the calls. The labels are the bulk of it: about 0.5 MB a round at 1
# user and 5 MB at 8, so some 1 GB by round 200 of an 8-user run.
#   LEAN=1     pulls only what a git repository of run records would keep of a run: no logs, which grow every minute,
#              and none of the oracle's own words, its hints or labels (points.jsonl and labels/labels/), a megabyte or
#              more a round; all of them are in the bucket.
# Idempotent and incremental, safe to run every hour or more often.
# LOCAL_SRC and UPLOADER (tests only) stand in for the node and for gcs_upload.py: the run is read from
# $LOCAL_SRC/runs/$RUN with a plain local rsync, and $UPLOADER runs here instead of on a node.
# Usage: [GCS=1 TRAJECTORY_BUCKET=gs://... NAME=... TOKEN_CMD=...] [LEAN=1] [ADAPTER=1] [CAPTURED=1] \
#            bash paper_backup.sh NODE RUN LOCAL_DIR
#   NODE: the node's ssh destination (user@host, or a Host of SSH_CONFIG, an ssh config file); RUN: the run's folder
#   under the node's runs folder (REMOTE_RUNS, default runs under its home), paper/ROW[-uN] or rt/NAME
#   e.g. GCS=1 TRAJECTORY_BUCKET=gs://BUCKET/carc NAME=node-1/pg-u1-lag1 TOKEN_CMD=./bucket-token.sh \
#            bash paper_backup.sh node-1 rt/pg-u1-lag1 backups/pg-u1-lag1
# Prints one line of what changed, a second for the GCS part, and exits non-zero on failure.
set -euo pipefail
L=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NODE=${1:?set NODE: the ssh destination of the node, or a Host of SSH_CONFIG (LOCAL_SRC tests pass anything here, it is ignored)}
RUN=${2:?set RUN: the run folder under the runs folder of the node, paper/ROW[-uN] or rt/NAME}
REMOTE_RUNS=${REMOTE_RUNS:-runs}
LOCAL_DIR=${3:?set LOCAL_DIR: the local folder for this run}
[[ $RUN =~ ^(paper|rt)/[A-Za-z0-9._-]+$ ]] || { echo "paper_backup: RUN is paper/ROW[-uN] or rt/NAME, not $RUN" >&2; exit 1; }
mkdir -p "$LOCAL_DIR"
UPLOADER=${UPLOADER:-$L/gcs_upload.py}
# macOS has no timeout(1); there ssh's keepalives alone end a call to a node that stopped answering
command -v timeout > /dev/null || timeout() { shift; "$@"; }
if [ "${GCS:-0}" = 1 ]; then
    BUCKET=${TRAJECTORY_BUCKET:?set TRAJECTORY_BUCKET with GCS=1: gs://BUCKET/PREFIX, where runs go}
    NAME=${NAME:?set NAME with GCS=1: the run lands at runs/gpu-nodes/NAME, for instance <node>/<run>}
    TOKEN_CMD=${TOKEN_CMD:?set TOKEN_CMD with GCS=1: a command that prints a short-lived token for the bucket}
    rest=${BUCKET#gs://}; BUCKET_NAME=${rest%%/*}; PREFIX=${rest#*/}
    [ "$PREFIX" = "$rest" ] && PREFIX=""
    DEST=${PREFIX:+$PREFIX/}runs/gpu-nodes/$NAME
fi

if [ -n "${LOCAL_SRC:-}" ]; then
    SRC="$LOCAL_SRC/runs/$RUN"
    rs() { timeout "${RSYNC_TIMEOUT:-120}" rsync -ai "$@"; }
    list() { (cd "$SRC" 2>/dev/null && eval "{ $1; true; }"); }   # $1: a shell listing command, run from $SRC
    upload_cmd() { python3 "$UPLOADER" --src "$SRC" --bucket "$BUCKET_NAME" --dest "$DEST" --keep session-artifacts \
                       --witness progress --aside ".replaced-$STAMP"; }
else
    SSH="ssh ${SSH_CONFIG:+-F $SSH_CONFIG} -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
    SRC="$NODE:$REMOTE_RUNS/$RUN"
    rs() { timeout "${RSYNC_TIMEOUT:-300}" rsync -azi -e "$SSH" "$@"; }
    list() { timeout 60 $SSH "$NODE" "cd ~/$REMOTE_RUNS/$RUN 2>/dev/null && { $1; true; }"; }
    upload_cmd() {   # the uploader's code travels in the command (base64: no quoting to get wrong), never to a file
        local code
        code=$(python3 -c 'import base64, sys; print(base64.b64encode(open(sys.argv[1], "rb").read()).decode())' "$UPLOADER")
        $SSH "$NODE" "python3 -c \"import base64; exec(compile(base64.b64decode('$code'), 'gcs_upload.py', 'exec'))\"" \
            "--src ~/$REMOTE_RUNS/$RUN --bucket $BUCKET_NAME --dest $DEST --keep session-artifacts --witness progress" \
            "--aside .replaced-$STAMP"
    }
fi

counted() {   # gcs_upload.py's JSON line as a phrase for the summary
    python3 -c 'import json,sys
d = json.loads(sys.stdin.read().strip().splitlines()[-1])
print("%d new (%d bytes), %d already there%s%s%s" % (d["uploaded"], d["bytes"], d["same"],
      ", %d gone since listed" % d["gone"] if d["gone"] else "", ", %d FAILED" % d["failed"] if d["failed"] else "",
      "; an earlier run under this name moved aside to %s" % d["aside"] if d["aside"] else ""))'
}
starts() {   # whether file $2 begins with the whole of file $1
    python3 -c 'import sys; sys.exit(0 if open(sys.argv[2], "rb").read().startswith(open(sys.argv[1], "rb").read()) else 1)' "$1" "$2"
}

STAMP=$(date -u +%Y%m%dT%H%MZ)
changes=$(mktemp); theirs=$(mktemp); trap 'rm -f "$changes" "$theirs"' EXIT
notes=()

# 0. a copy of an earlier run under this name moves aside, kept (the header says why)
list 'cat progress 2> /dev/null' > "$theirs" || { echo "paper_backup $RUN: reading $NODE's progress failed" >&2; exit 1; }
if [ -f "$LOCAL_DIR/progress" ] && ! starts "$LOCAL_DIR/progress" "$theirs"; then
    mv "$LOCAL_DIR" "$LOCAL_DIR.replaced-$STAMP"; mkdir -p "$LOCAL_DIR"
    notes+=("an earlier run's copy moved aside to $(basename "$LOCAL_DIR").replaced-$STAMP")
fi

# 1. everything small. Patterns with no "/" match a name at any depth (rsync's own rule), so one pass covers every
#    play*/train* variant (USERS cuts, the epilogue) and both layouts without listing them.
filt=(--prune-empty-dirs --exclude 'opd_scores.json*' --exclude 'adapter/' --exclude 'adapter.vllm/' --exclude 'users/'
      --exclude 'labels/set/' --exclude 'labels/split/*/')
if [ "${LEAN:-0}" = 1 ]; then
    filt+=(--exclude '*.log' --exclude 'hints/' --exclude 'hints.json' --exclude 'labels/labels/' --exclude 'points.jsonl')
fi
if [ "${CAPTURED:-0}" = 1 ]; then filt+=(--include 'captured/***'); else filt+=(--exclude 'captured/'); fi
filt+=(--include '**/labels/***' --include '*/'
       --include 'progress' --include '*.log' --include 'guidelines.md' --include '*.done' --include 'records.jsonl'
       --include 'summary.json' --include 'meta.json' --include 'config.json' --include 'driver.json'
       --include 'schedule.json' --include 'alias-salt' --include 'hints-cost.json' --include 'hints-round-cost.json'
       --include 'hints.json' --include 'memory.json' --include '**/hints/***' --include 'served.json'
       --include 'DONE' --include 'phase.json' --include 'rounds.json' --include 'latest.json'
       --exclude '*')
rs "${filt[@]}" "$SRC/" "$LOCAL_DIR/" >> "$changes" \
    || { echo "paper_backup $RUN: pulling $NODE's small files failed" >&2; exit 1; }

# 2. the run whole, from the node to the bucket: one uploader call, which lists the bucket once and reads a file only
#    when the bucket does not hold its version
if [ "${GCS:-0}" = 1 ]; then
    uploaded=$(bash -c "$TOKEN_CMD" | upload_cmd) \
        || { echo "paper_backup $RUN: uploading it to the bucket failed: ${uploaded:-no answer}" >&2; exit 1; }
    gcs_note=$(printf '%s\n' "$uploaded" | counted)
fi

# 3. the newest adapter of each training folder to the local folder, with ADAPTER=1: a paper row's train*/rN (a USERS
#    cut and the epilogue each have their own folder) or a real-time run's train/pNNNN
if [ "${ADAPTER:-0}" = 1 ]; then
    for td in $(list 'ls -d train* epilogue-train-* 2>/dev/null'); do
        newest=$(list "ls -d '$td'/[rp][0-9]*/adapter 2>/dev/null" | sed -E 's#/adapter/?$##; s#^.*/##' \
            | awk '{n = substr($0, 2) + 0; print n, $0}' | sort -n | tail -1 | cut -d' ' -f2)
        [ -n "$newest" ] || continue
        mkdir -p "$LOCAL_DIR/$td/$newest"
        rs "$SRC/$td/$newest/adapter/" "$LOCAL_DIR/$td/$newest/adapter/" >> "$changes" \
            || { echo "paper_backup $RUN: pulling $td/$newest's adapter from $NODE failed" >&2; exit 1; }
        for old in "$LOCAL_DIR/$td"/[rp][0-9]*/adapter; do
            [ -d "$old" ] || continue
            kept=$(basename "$(dirname "$old")")
            if [ "$kept" != "$newest" ]; then
                rm -rf "$old"
                notes+=("pruned $td/$kept/adapter (superseded by $newest)")
            fi
        done
    done
fi

# an adapter an earlier ADAPTER=1 run left in the copy is reported, never deleted unasked
if [ "${ADAPTER:-0}" != 1 ] && [ -n "$(find "$LOCAL_DIR" -maxdepth 3 -type d -name adapter 2>/dev/null)" ]; then
    notes+=("$(du -sh "$LOCAL_DIR" | awk '{print $1}') on disk, adapters included: rm -rf $LOCAL_DIR/*/[rp][0-9]*/adapter")
fi

join() { local sep=$1 out="" x; shift; for x in "$@"; do out="${out:+$out$sep}$x"; done; printf '%s' "$out"; }

n=$(grep -c . "$changes" 2>/dev/null || true); n=${n:-0}
summary="$n file(s) changed"
[ "${#notes[@]}" -eq 0 ] || summary="$summary; $(join '; ' "${notes[@]}")"
echo "$(date -u +%H:%M) paper_backup($RUN): $summary"
[ "${GCS:-0}" != 1 ] || echo "$(date -u +%H:%M) paper_backup($RUN) to gs://$BUCKET_NAME/$DEST: $gcs_note"
