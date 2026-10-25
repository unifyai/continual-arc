#!/usr/bin/env bash
# Tests frozen.sh end to end in a fake home (HOME, with every node tool it calls stubbed on its PATH: nvidia-smi, curl,
# sudo, setsid, uv, vLLM's python and vllm-serve.sh beside it), with no GPU and no model: it waits while the run's
# stream goes on; once realtime.sh has said "done", it serves the run's last phase on two servers of its own, one on
# each half of GPUS at 8120 plus its first GPU, loads the adapter into each, plays both tests, plays the held-out test
# again after its first play stops short, writes the summary, deletes the test users' copies of the memory, and leaves
# test/done; run again, it does nothing. A reference from nothing serves the base model and loads nothing, and the
# weights alone (MEMORY=none) play another adapter with --no-memory into TEST-weights, which a reference refuses. The
# plays and the model are frozen.py's own off any node, so the summary is frozen.py's real one over the stub plays'
# records.
# Usage: bash test_frozen.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE=$(pwd)
F=$(mktemp -d); trap 'pkill -f "[f]rozen.sh pg-u1-lag1" 2>/dev/null; rm -rf "$F"' EXIT
fail=0
check() { if eval "$1"; then :; else echo "FAIL: $2"; fail=1; fi; }
wait_for() { for _ in $(seq 120); do eval "$1" && return 0; sleep 1; done; echo "FAIL: timed out waiting for $2"; fail=1; }

R=$F/runs/rt/pg-u1-lag1
mkdir -p "$R/play" "$R/train/p0002/adapter" "$F/.local/bin" "$F/vllm-0.30.0/bin" "$F/prime-rl" "$F/tools/schedules"
cp frozen.py frozen.sh "$F/tools/"
for t in held-out seen; do   # each test cut to three instances
    python3 -c "import json, sys; d = json.load(open(sys.argv[1])); d['slots'] = d['slots'][:3]; d['users'] = 3
json.dump(d, open(sys.argv[2], 'w'))" "schedules/arc1-small-$t-test-seed0.json" "$F/tools/schedules/arc1-small-$t-test-seed0.json"
done
echo "10-21 01:46 realtime.sh: started: pg, 1 users, 3 rounds" > "$R/progress"

stub() { printf '#!/usr/bin/env bash\n%s\n' "$2" > "$F/.local/bin/$1"; chmod +x "$F/.local/bin/$1"; }
stub nvidia-smi 'echo 0'
stub sudo 'exit 0'
stub setsid 'exec "$@"'
# a server answers once vllm-serve.sh has started it, and serves the adapters loaded into it
stub curl 'url=""; data=""; while [ $# -gt 0 ]; do case $1 in -d) data=$2; shift ;; localhost:*) url=$1 ;; esac; shift; done
port=${url#localhost:}; port=${port%%/*}
case $url in
    */health) [ -f "$HOME/up-$port" ] ;;
    */v1/models) echo "{\"data\": [$(cat "$HOME/loaded-$port" 2>/dev/null)]}" ;;
    */v1/load_lora_adapter) name=$(echo "$data" | sed "s/.*lora_name\": \"\([^\"]*\)\".*/\1/"); echo "{\"id\": \"$name\"}" >> "$HOME/loaded-$port"
        echo "$data" >> "$HOME/loads" ;;
esac'
cat > "$F/tools/vllm-serve.sh" <<'EOF'
echo "$@" >> "$HOME/served"; touch "$HOME/up-$3"; echo 1 > "$HOME/runs/vllm-$3.pid"
EOF
printf '#!/usr/bin/env bash\necho "$HOME/adapter.vllm"\n' > "$F/vllm-0.30.0/bin/python"; chmod +x "$F/vllm-0.30.0/bin/python"
# uv run --no-sync python frozen.py CMD ...: the model and the summary are frozen.py's own; a play writes its test's
# records after a moment, and the held-out test's first play stops short without them
stub uv 'shift 3; py=$1; cmd=$2
case $cmd in
    model) echo "pg-u1-lag1-p0002 $HOME/runs/rt/pg-u1-lag1/train/p0002/adapter" ;;
    summary) exec python3 "$@" ;;
    play) echo "$@" >> "$HOME/plays"
        ref=""; while [ $# -gt 0 ]; do case $1 in --test) t=$2 ;; --out) out=$2 ;; --schedule) s=$2 ;; --row) ref=-reference ;; --no-memory) ref=-weights ;; esac; shift; done
        sleep 2
        if [ "$t" = held-out ] && [ ! -f "$out/held-out-once" ]; then touch "$out/held-out-once"; exit 1; fi
        p=$out/$t$ref/play
        mkdir -p "$p/users/u000/system/state/agent/kernel-venv" "$p/users/u000/system/state/sessions"
        echo "{\"test\": \"$t\"}" > "$p/frozen.json"
        n=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))[\"slots\"]))" "$s")
        for i in $(seq "$n"); do echo "{\"solved\": $([ "$i" = 1 ] && echo true || echo false), \"cost\": $i}"; done > "$p/records.jsonl"
        echo "{\"test\": \"$t\", \"records\": $n}" ;;
esac'

HOME=$F POLL=1 bash "$F/tools/frozen.sh" pg-u1-lag1 6,7 &
T=$R/frozen
wait_for 'grep -q "waiting for pg-u1-lag1" "$T/progress" 2>/dev/null' "the wait for the stream"
sleep 3
check '[ ! -f "$F/served" ]' "nothing is served while the stream goes on"
touch "$R/player.done"; sleep 2
check '[ ! -f "$F/served" ]' "a stream whose loops are done is still going until realtime.sh says done"
touch "$R/trainer.done"; echo "10-21 20:00 realtime.sh: done" >> "$R/progress"
wait_for '[ -f "$T/done" ]' "the tests to end"
wait
check 'grep -q "^Qwen/Qwen3.5-9B 6 8126 " "$F/served" && grep -q "^Qwen/Qwen3.5-9B 7 8127 " "$F/served"' \
    "a server on each half of the GPUs, at 8120 plus its first GPU ($(cat "$F/served" 2>/dev/null | cut -c1-40))"
check '[ "$(grep -c "\"lora_name\": \"pg-u1-lag1-p0002\", \"lora_path\": \"$F/adapter.vllm\"" "$F/loads")" = 2 ]' \
    "the last phase loaded into both servers from vLLM's copy"
check 'grep -q "playing held-out, attempt 2" "$T/progress" && ! grep -q "playing seen, attempt 2" "$T/progress"' \
    "the held-out test played again after it stopped short, the seen test once"
check 'grep -q "serving pg-u1-lag1-p0002 on GPUs 6 (:8126, held-out) and 7 (:8127, seen)" "$T/progress"' "the serving line"
check '[ -f "$T/held-out.done" ] && [ -f "$T/seen.done" ]' "each test's marker"
check 'python3 -c "import json, sys; d = json.load(open(sys.argv[1]))[\"tests\"]
sys.exit(not (d[\"held-out\"][\"instances\"] == 3 and d[\"seen\"][\"solved\"] == 1 and d[\"seen\"][\"mean_cost\"] == 2.0))" \
    "$T/summary.json"' "the summary of both tests"
check '[ ! -d "$T/held-out/play/users/u000/system/state" ] && [ -f "$T/held-out/play/frozen.json" ]' \
    "the test users' copies of the memory deleted, the test kept"
before=$(wc -l < "$T/progress")
HOME=$F POLL=1 bash "$F/tools/frozen.sh" pg-u1-lag1 6,7
check '[ "$(wc -l < "$T/progress")" = "$before" ]' "a second run once the tests are done does nothing"

# a reference from nothing: no run to wait for, the base model served and nothing loaded, row bare played
rm -f "$F/served" "$F/loads" "$F"/up-* "$F"/loaded-*
HOME=$F POLL=1 ROW=bare bash "$F/tools/frozen.sh" qwen-f 6,7
B=$F/runs/paper/qwen-f/frozen
check '[ -f "$B/done" ] && [ -f "$B/held-out-reference.done" ] && [ -f "$B/seen-reference.done" ] \
    && [ -f "$B/seen-reference/play/records.jsonl" ]' "the reference's plays, into TEST-reference/play"
check 'grep -q "row bare from nothing" "$B/progress" && grep -q "serving Qwen/Qwen3.5-9B on GPUs 6" "$B/progress"' \
    "the reference's lines"
check '[ ! -f "$F/loads" ] && [ "$(grep -c . "$F/served")" = 2 ]' "two servers of the base model, nothing loaded"
check 'python3 -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(not (d[\"run\"] is None and d[\"tests\"][\"seen-reference\"][\"instances\"] == 3))" \
    "$B/summary.json"' "the reference's summary, of no run"

# the weights alone: another adapter served under its own name, played from an empty memory into TEST-weights
rm -f "$F/served" "$F/loads" "$F/plays" "$F"/up-* "$F"/loaded-*
HOME=$F POLL=1 MEMORY=none ADAPTER=$F/continual/adapter SERVED=scratch-pg-u1-lag1-continual OUT=$R/scratch/weights \
    bash "$F/tools/frozen.sh" pg-u1-lag1 6,7
S=$R/scratch/weights
check '[ -f "$S/done" ] && [ -f "$S/held-out-weights.done" ] && [ -f "$S/seen-weights.done" ] \
    && [ -f "$S/seen-weights/play/records.jsonl" ]' "the weights-alone plays, into TEST-weights/play"
check 'grep -q "its weights alone from an empty memory, into $S" "$S/progress" \
    && grep -q "serving scratch-pg-u1-lag1-continual on GPUs 6 (:8126, held-out-weights) and 7 (:8127, seen-weights)" "$S/progress"' \
    "the weights-alone lines"
check '[ "$(grep -c "\"lora_name\": \"scratch-pg-u1-lag1-continual\"" "$F/loads")" = 2 ]' "the adapter loaded into both servers"
check '[ "$(grep -c -- "--run $R --test .* --adapter $F/continual/adapter --no-memory$" "$F/plays")" = 3 ]' \
    "every play of the run's, on the adapter, with --no-memory (the held-out test twice)"
check '! grep -q "weights" "$R/frozen/progress"' "the run's own tests untouched"
HOME=$F POLL=1 ROW=h0 MEMORY=none bash "$F/tools/frozen.sh" qwen-h0 6,7 && fail=1
check 'grep -q "a reference from nothing has no memory to leave out" "$F/runs/paper/qwen-h0/weights/progress"' \
    "a reference with MEMORY=none refused"

[ "$fail" = 0 ] && echo "test_frozen.sh: all passed" || { echo "test_frozen.sh: FAILED"; exit 1; }
