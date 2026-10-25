#!/usr/bin/env bash
# Tests paper_backup.sh end to end against a fake local "node" (LOCAL_SRC, a plain directory rsync'd to a plain
# directory instead of over ssh), with no node and no bucket, for both layouts: a paper.sh cut (runs/paper/pg-u1)
# and a real-time DAgger run (runs/rt/dagger-u1). The default pulls each one's small files, the labeller's output
# among them, and leaves the adapters, the captured calls, the users' state and the label sets' copies on the node;
# LEAN=1 leaves out the logs, the hints and the labels too; GCS=1 refuses to run without TRAJECTORY_BUCKET, NAME or
# TOKEN_CMD, then asks TOKEN_CMD for a token and hands gcs_upload.py (both stubbed, the uploader as UPLOADER) the
# whole run, session-artifacts kept, under runs/gpu-nodes/NAME; ADAPTER=1 pulls the newest adapter of each layout and
# deletes an older one while a default run only reports it; a rerun with nothing new changes nothing; CAPTURED=1 pulls
# the raw calls; a run started over under the same name moves the earlier run's copy aside, kept, and asks the
# uploader to set the bucket's aside too; and a RUN outside paper/ and rt/ is refused.
# Usage: bash test_paper_backup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
NODE=$T/node; LOCAL=$T/local
fail=0
check() { if eval "$1"; then :; else echo "FAIL: $2"; fail=1; fi; }

# a paper.sh cut at 1 user: its play and train folders carry the cut's suffix
P=$NODE/runs/paper/pg-u1
mkdir -p "$P/play-u1/rounds/r0000/captured" "$P/play-u1/rounds/r0000/hints" "$P/train-u1/r0/adapter" \
    "$P/play-u1/users/u000/system/state/sessions"
echo '[]' > "$P/play-u1/records.jsonl"
for f in meta summary driver schedule config hints-cost; do echo '{}' > "$P/play-u1/$f.json"; done
echo salt > "$P/play-u1/alias-salt"; touch "$P/play-u1/round-play0.done"
echo '{"a":1}' > "$P/play-u1/rounds/r0000/memory.json"
echo '{"b":1}' > "$P/play-u1/rounds/r0000/hints.json"
echo '{"c":1}' > "$P/play-u1/rounds/r0000/hints-round-cost.json"
echo hinted > "$P/play-u1/rounds/r0000/hints/u000.json"
echo big-call > "$P/play-u1/rounds/r0000/captured/u000.json.gz"
echo big-scores > "$P/play-u1/rounds/r0000/opd_scores.json"
echo session > "$P/play-u1/users/u000/system/state/sessions/s.jsonl"
mkdir -p "$P/play-u1/rounds/r0000/labels/labels"
echo reply > "$P/play-u1/rounds/r0000/labels/labels/r0000-u000-c0000-cd.json"
echo point > "$P/play-u1/rounds/r0000/labels/points.jsonl"; echo done > "$P/play-u1/rounds/r0000/labels/labelled"
echo "round 0 done" > "$P/train-u1/r0/DONE"; echo '{"step":1}' > "$P/train-u1/r0/phase.json"
echo weights-r0 > "$P/train-u1/r0/adapter/model.bin"
echo progress-line > "$P/progress"; echo log-line > "$P/play-round0.log"

# a real-time run: one play folder, the labeller's files under each round, the trainer's phases
R=$NODE/runs/rt/dagger-u1
D=$R/play/rounds/r0000
mkdir -p "$D/captured" "$D/labels/labels" "$D/labels/set" "$D/labels/split/train-r0000" "$R/train/p0000/adapter" \
    "$R/train/p0000/adapter.vllm" "$R/play/users/u000/system/state/agent/kernel-venv"
echo '[]' > "$R/play/records.jsonl"; echo '{}' > "$R/play/config.json"
echo '{"phase":null}' > "$D/served.json"; touch "$D/play.done"; echo '{"m":1}' > "$D/memory.json"
echo call > "$D/captured/u000.json.gz"
echo oracle-reply > "$D/labels/labels/r0000-u000-c0001-ab.json"; echo point > "$D/labels/points.jsonl"
echo '{}' > "$D/labels/set.json"; echo copy > "$D/labels/set/u000.json.gz"
echo '{"games":1}' > "$D/labels/split/split.json"; echo '[]' > "$D/labels/split/train-r0000/records.jsonl"
echo '{"phase":0}' > "$R/train/latest.json"
echo done > "$R/train/p0000/DONE"; echo '{}' > "$R/train/p0000/phase.json"; echo '{"rounds":1}' > "$R/train/p0000/rounds.json"
echo weights-p0 > "$R/train/p0000/adapter/adapter_model.safetensors"; echo served > "$R/train/p0000/adapter.vllm/x"
echo venv > "$R/play/users/u000/system/state/agent/kernel-venv/python"
echo c1 > "$R/guidelines.md"; touch "$R/player.done"; echo progress-line > "$R/progress"; echo log > "$R/trainer.log"

# the token command and gcs_upload.py (the upload), stubbed: the uploader records every call and answers with the one
# JSON line the real one prints, so the GCS path runs end to end without a node or a bucket
cat > "$T/token.sh" <<'EOF'
echo "TOKEN $*" >> "$CALLS"
echo "fake-token"
EOF
BUCKET=(TRAJECTORY_BUCKET=gs://test-bucket/results "TOKEN_CMD=bash $T/token.sh --for-the-bucket")
cat > "$T/fake_gcs_upload.py" <<'EOF'
import json, os, sys
token = sys.stdin.readline().strip()
with open(os.environ["CALLS"], "a") as fp:
    fp.write("UPLOAD token=%s args=%s\n" % (token, " ".join(sys.argv[1:])))
aside = sys.argv[sys.argv.index("--dest") + 1] + sys.argv[sys.argv.index("--aside") + 1] if os.environ.get("ASIDE") else None
print(json.dumps({"files": 5, "uploaded": 4, "same": 0, "gone": 1, "failed": 0, "bytes": 40, "aside": aside}))
EOF
export CALLS=$T/calls.log; : > "$CALLS"
run() {  # run RUN [env...]: paper_backup.sh against the fake node, into $LOCAL/<the run's name>
    local run=$1; shift
    env LOCAL_SRC="$NODE" UPLOADER="$T/fake_gcs_upload.py" "$@" \
        bash paper_backup.sh fake-node "$run" "$LOCAL/${run#*/}"
}

echo "== default: a paper cut's small files only =="
out=$(run paper/pg-u1)
echo "$out"
C=$LOCAL/pg-u1
check '[ -n "$(echo "$out" | grep "paper_backup(paper/pg-u1)")" ]' "prints a one-line summary"
for f in play-u1/records.jsonl play-u1/meta.json play-u1/config.json play-u1/round-play0.done \
    play-u1/rounds/r0000/memory.json play-u1/rounds/r0000/hints.json play-u1/rounds/r0000/hints-round-cost.json \
    play-u1/rounds/r0000/hints/u000.json progress play-round0.log train-u1/r0/DONE train-u1/r0/phase.json \
    play-u1/rounds/r0000/labels/labels/r0000-u000-c0000-cd.json play-u1/rounds/r0000/labels/points.jsonl \
    play-u1/rounds/r0000/labels/labelled; do
    check "[ -f '$C/$f' ]" "pulls the cut's $f"
done
check '[ ! -e "$C/train-u1/r0/adapter" ]' "never pulls the adapter by default"
check '[ ! -e "$C/play-u1/rounds/r0000/captured" ]' "never pulls captured/ by default"
check '[ ! -e "$C/play-u1/rounds/r0000/opd_scores.json" ]' "never pulls opd_scores.json"
check '[ ! -e "$C/play-u1/users" ]' "never pulls the users' own state"
check '[ ! -s "$CALLS" ]' "no token or upload call without GCS=1"

echo "== default: a real-time run's small files only =="
out=$(run rt/dagger-u1)
echo "$out"
C=$LOCAL/dagger-u1
for f in play/records.jsonl play/config.json play/rounds/r0000/served.json play/rounds/r0000/play.done \
    play/rounds/r0000/memory.json play/rounds/r0000/labels/set.json play/rounds/r0000/labels/split/split.json \
    play/rounds/r0000/labels/labels/r0000-u000-c0001-ab.json play/rounds/r0000/labels/points.jsonl \
    train/latest.json train/p0000/DONE train/p0000/phase.json train/p0000/rounds.json guidelines.md player.done \
    progress trainer.log; do
    check "[ -f '$C/$f' ]" "pulls the real-time run's $f"
done
check '[ ! -e "$C/train/p0000/adapter" ] && [ ! -e "$C/train/p0000/adapter.vllm" ]' "never pulls a phase's adapter"
check '[ ! -e "$C/play/rounds/r0000/labels/set" ] && [ ! -e "$C/play/rounds/r0000/labels/split/train-r0000" ]' \
    "never pulls the label sets' copies of the calls"
check '[ ! -e "$C/play/rounds/r0000/captured" ] && [ ! -e "$C/play/users" ]' "never pulls captured/ or the users' state"

echo "== LEAN=1: what a repository of run records keeps, no logs and none of the oracle's hints or labels =="
for run in paper/pg-u1 rt/dagger-u1; do
    env LOCAL_SRC="$NODE" LEAN=1 bash paper_backup.sh fake-node "$run" "$T/lean/${run#*/}" > /dev/null
done
C=$T/lean
for f in pg-u1/progress pg-u1/play-u1/records.jsonl pg-u1/play-u1/hints-cost.json pg-u1/play-u1/rounds/r0000/memory.json \
    pg-u1/play-u1/rounds/r0000/hints-round-cost.json pg-u1/play-u1/rounds/r0000/labels/labelled \
    dagger-u1/play/records.jsonl dagger-u1/play/rounds/r0000/served.json dagger-u1/play/rounds/r0000/labels/set.json \
    dagger-u1/play/rounds/r0000/labels/split/split.json dagger-u1/train/p0000/phase.json; do
    check "[ -f '$C/$f' ]" "LEAN=1 keeps $f"
done
check '[ -z "$(find "$C" -name "*.log" -o -name hints -o -name hints.json -o -name points.jsonl)" ]' \
    "LEAN=1 leaves out the logs, the hints and the label points"
check '[ ! -e "$C/pg-u1/play-u1/rounds/r0000/labels/labels" ] && [ ! -e "$C/dagger-u1/play/rounds/r0000/labels/labels" ]' \
    "LEAN=1 leaves out the oracle's labels"

echo "== GCS=1 needs a bucket, NAME and a token command =="
if run rt/dagger-u1 GCS=1 NAME=node-1/dagger-u1 "TOKEN_CMD=bash $T/token.sh" > "$T/out" 2>&1; then
    check false "GCS=1 without TRAJECTORY_BUCKET must fail"
else
    check '[ -n "$(grep TRAJECTORY_BUCKET "$T/out")" ]' "says TRAJECTORY_BUCKET is needed"
fi
if run rt/dagger-u1 GCS=1 "${BUCKET[@]}" > "$T/out" 2>&1; then
    check false "GCS=1 without NAME must fail"
else
    check '[ -n "$(grep NAME "$T/out")" ]' "says NAME is needed"
fi
if run rt/dagger-u1 GCS=1 TRAJECTORY_BUCKET=gs://test-bucket/results NAME=node-1/dagger-u1 > "$T/out" 2>&1; then
    check false "GCS=1 without TOKEN_CMD must fail"
else
    check '[ -n "$(grep TOKEN_CMD "$T/out")" ]' "says TOKEN_CMD is needed"
fi
check '[ ! -s "$CALLS" ]' "and asks for no token"

echo "== GCS=1: the whole run, from the node to the bucket =="
out=$(run rt/dagger-u1 GCS=1 "${BUCKET[@]}" NAME=node-1/dagger-u1)
echo "$out"; echo "-- calls --"; cat "$CALLS"
check '[ -n "$(echo "$out" | grep "to gs://test-bucket/results/runs/gpu-nodes/node-1/dagger-u1: 4 new (40 bytes), 0 already there, 1 gone since listed")" ]' \
    "a second line for the GCS part, under runs/gpu-nodes/NAME"
check '[ "$(grep -c "^UPLOAD" "$CALLS")" = 1 ]' "one upload call for the whole run"
check '[ -n "$(grep -F -- "--src $NODE/runs/rt/dagger-u1 --bucket test-bucket --dest results/runs/gpu-nodes/node-1/dagger-u1 --keep session-artifacts --witness progress --aside .replaced-" "$CALLS")" ]' \
    "uploads the run folder whole, session-artifacts kept, an earlier run under the name set aside"
check '[ -n "$(grep -Fx "TOKEN --for-the-bucket" "$CALLS")" ]' "the token comes from TOKEN_CMD"
check '[ -n "$(grep -F "token=fake-token" "$CALLS")" ]' "the token reaches the uploader on stdin"
check '[ -z "$(grep "^UPLOAD" "$CALLS" | sed "s/^UPLOAD token=[^ ]* //" | grep -F fake-token)" ]' \
    "the token is never an argument"
check '[ ! -e "$LOCAL/dagger-u1/train/p0000/adapter" ]' "GCS=1 still keeps the adapter off the local copy"

echo "== ADAPTER=1: the newest adapter of each layout =="
mkdir -p "$R/train/p0001/adapter"; echo weights-p1 > "$R/train/p0001/adapter/adapter_model.safetensors"
mkdir -p "$P/train-u1/r1/adapter"; echo weights-r1 > "$P/train-u1/r1/adapter/model.bin"
out=$(run rt/dagger-u1 ADAPTER=1); echo "$out"
check '[ -f "$LOCAL/dagger-u1/train/p0001/adapter/adapter_model.safetensors" ]' "pulls the newest phase's adapter"
check '[ ! -e "$LOCAL/dagger-u1/train/p0000/adapter" ]' "and not an older phase's"
out=$(run paper/pg-u1 ADAPTER=1); echo "$out"
check '[ -f "$LOCAL/pg-u1/train-u1/r1/adapter/model.bin" ]' "pulls the cut's newest round's adapter"
mkdir -p "$P/train-u1/r2/adapter"; echo weights-r2 > "$P/train-u1/r2/adapter/model.bin"
echo "round 2 done" > "$P/train-u1/r2/DONE"
out=$(run paper/pg-u1 ADAPTER=1); echo "$out"
check '[ -n "$(echo "$out" | grep "pruned train-u1/r1/adapter (superseded by r2)")" ]' "reports the pruned round"
check '[ -f "$LOCAL/pg-u1/train-u1/r2/adapter/model.bin" ] && [ ! -e "$LOCAL/pg-u1/train-u1/r1/adapter" ]' \
    "replaces the superseded adapter in the copy"
check '[ -f "$LOCAL/pg-u1/train-u1/r0/DONE" ] && [ -f "$LOCAL/pg-u1/train-u1/r0/phase.json" ]' \
    "keeps an older round's DONE and phase.json"

echo "== a default run reports, but never deletes, an adapter an ADAPTER=1 run left behind =="
out=$(run paper/pg-u1); echo "$out"
check '[ -n "$(echo "$out" | grep "rm -rf")" ]' "says how to reclaim the disk"
check '[ -f "$LOCAL/pg-u1/train-u1/r2/adapter/model.bin" ]' "leaves the adapter it was not asked to manage"

# straight after the run above, with nothing touched in between: deleting anything from the copy first would change
# its parent directory mtime, which rsync then syncs back, and that is one honest change, not a flake
echo "== rerun with nothing new changes nothing =="
out=$(run paper/pg-u1); echo "$out"
check '[ -z "$(echo "$out" | grep pruned)" ]' "no pruning on an idempotent rerun"
check '[ -n "$(echo "$out" | grep -E ": 0 file")" ]' "nothing changed on an idempotent rerun"

echo "== CAPTURED=1 pulls the raw calls to the local copy (the exception) =="
out=$(run paper/pg-u1 CAPTURED=1); echo "$out"
check '[ -f "$LOCAL/pg-u1/play-u1/rounds/r0000/captured/u000.json.gz" ]' "CAPTURED=1 pulls the raw calls"

echo "== a run started over under the same name: the copy of the earlier one moves aside, kept =="
rm -rf "$R"; mkdir -p "$R/play"
echo "10-19 01:00 realtime.sh: started" > "$R/progress"; echo '{"round":0}' > "$R/play/records.jsonl"
: > "$CALLS"
out=$(run rt/dagger-u1 GCS=1 "${BUCKET[@]}" NAME=node-1/dagger-u1 ASIDE=1); echo "$out"
aside=$(ls -d "$LOCAL"/dagger-u1.replaced-* 2>/dev/null | head -1)
check '[ -n "$aside" ] && [ -f "$aside/train/p0000/phase.json" ] && [ -f "$aside/play/rounds/r0000/served.json" ]' \
    "the earlier run's copy is kept whole, beside"
check '[ ! -e "$LOCAL/dagger-u1/train" ] && [ "$(cat "$LOCAL/dagger-u1/progress")" = "10-19 01:00 realtime.sh: started" ]' \
    "the new run's copy holds the new run alone"
check '[ -n "$(echo "$out" | grep "moved aside to $(basename "$aside")")" ]' "says the copy moved aside"
check '[ -n "$(echo "$out" | grep "an earlier run under this name moved aside to results/runs/gpu-nodes/node-1/dagger-u1.replaced-")" ]' \
    "says the bucket's moved aside too"
echo "more" >> "$R/progress"
out=$(run rt/dagger-u1); echo "$out"
check '[ "$(ls -d "$LOCAL"/dagger-u1.replaced-* | wc -l | tr -d " ")" = 1 ]' "a run that goes on is not moved again"

echo "== a RUN outside paper/ and rt/ is refused =="
if run loop/x > "$T/out" 2>&1; then check false "RUN loop/x must be refused"; else
    check '[ -n "$(grep "paper/ROW" "$T/out")" ]' "says what RUN may be"; fi

if [ "$fail" = 0 ]; then echo PASS; else echo FAILED; exit 1; fi
