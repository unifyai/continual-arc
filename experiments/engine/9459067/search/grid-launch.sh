#!/usr/bin/env bash
# grid-launch.sh HOST NAME INIT TARGET [LR BATCH EPOCHS]: puts what one grid cell needs on HOST, straight from the bucket
# (the laptop's automation account signs each download, the node fetches it), and starts grid-exp.sh there detached:
# R's round-1 label set as ~/exp/stage1, b6 as ~/runs/loop/b1-6 when INIT is b6, her validation schedule and alias
# salt, and continual-arc-training's and ask-sd's main (deploy-code.sh) unless the node has them already.
set -euo pipefail
H=$1; shift
S=$(cd "$(dirname "$0")" && pwd)   # search/: grid-exp.sh and deploy-code.sh beside this
B=ARTIFACTS_BUCKET
G=results/runs/gpu-nodes
V=$HOME/continual-arc-results/runs/gpu-nodes/NODE-fwd-comparison/val-fwd2-comparison
token() { CLOUDSDK_ACTIVE_CONFIG_NAME=automation gcloud auth print-access-token 2>/dev/null; }
fetch() {  # fetch OBJECT NODE_FILE: the object to the node, its md5 checked against the bucket's
    local obj want got
    obj=$(python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" "$1")
    want=$(CLOUDSDK_ACTIVE_CONFIG_NAME=automation gcloud storage objects describe "gs://$B/$1" --format='value(md5_hash)')
    token | ssh -o BatchMode=yes "$H" "read -r tok; mkdir -p \$(dirname $2); curl -fsS -H \"Authorization: Bearer \$tok\" -o $2 \
        'https://storage.googleapis.com/storage/v1/b/$B/o/$obj?alt=media'"
    got=$(ssh -o BatchMode=yes "$H" "openssl md5 -binary $2 | base64")
    [ "$got" = "$want" ] || { echo "$1: md5 $got on the node, $want in the bucket"; exit 1; }
}
ssh -o BatchMode=yes "$H" 'grep -q dual_clip_fraction ~/loop/train.py && test -f ~/loop/dagger_rules.py' 2>/dev/null \
    || bash "$S/deploy-code.sh" "$H"
if ! ssh -o BatchMode=yes "$H" 'test -d ~/exp/stage1/set'; then
    fetch "$G/NODE-fwd-comparison/labels/comparison-stage1.tar.gz" /tmp/stage1.tgz
    ssh -o BatchMode=yes "$H" 'mkdir -p ~/exp && tar -C ~/exp -xzf /tmp/stage1.tgz && rm -f /tmp/stage1.tgz && test -d ~/exp/stage1/set'
fi
if [ "$2" = b6 ] && ! ssh -o BatchMode=yes "$H" 'test -f ~/runs/loop/b1-6/adapter/trainer_adapter.safetensors'; then
    fetch "$G/NODE-q9cycles/adapters/b6.tar" /tmp/b6.tar
    ssh -o BatchMode=yes "$H" "mkdir -p ~/runs/loop && rm -rf ~/runs/loop/b1-6 && tar -C ~/runs/loop --exclude live --exclude adapter.vllm \
        --transform 's,^b6\(/\|$\),b1-6\1,' -xf /tmp/b6.tar && test -f ~/runs/loop/b1-6/adapter/trainer_adapter.safetensors && rm -f /tmp/b6.tar"
fi
if [[ "$2" == adapter:*fwd2-final* ]] && ! ssh -o BatchMode=yes "$H" 'test -f ~/exp/fwd2-final/adapter/adapter_model.safetensors'; then
    for f in DONE phase.json adapter/README.md adapter/adapter_config.json adapter/adapter_model.safetensors \
            adapter/trainer_adapter.safetensors adapter/trainer_state.json; do
        fetch "$G/NODE-fwd-comparison/adapters/fwd2-final/$f" "/home/NODE_USER/exp/fwd2-final/$f"
    done
fi
ssh -o BatchMode=yes "$H" "mkdir -p ~/runs/loop/r0-val ~/exp/$1"
scp -q -o BatchMode=yes "$V/schedule.json" "${H}:runs/loop/sched-val.json"
scp -q -o BatchMode=yes "$V/alias-salt" "${H}:runs/loop/r0-val/alias-salt"
# a copy of its own per cell: bash reads a script as it runs, so rewriting a shared one would corrupt a running cell
scp -q -o BatchMode=yes "$S/grid-exp.sh" "${H}:exp/$1/grid-exp.sh"
ssh -o BatchMode=yes "$H" "setsid -f nohup bash ~/exp/$1/grid-exp.sh $* > /dev/null 2>&1 < /dev/null; echo 'started grid-exp.sh $*'"
