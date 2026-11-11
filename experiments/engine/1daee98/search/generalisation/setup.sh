#!/usr/bin/env bash
# setup.sh (on any machine, as the user the cells run as): the environment the generalisation search's cells run in,
# built from what launch.sh fetched into ~/gen/bundle and nothing else of ours:
#   BUNDLE.tgz  the code bundle the cell names (cell.py's "bundle"; BUNDLE, code-dagger-rows unless set: the DAgger
#             rows' own code as their nodes held it, ~/loop, ~/continual-arc, ~/continual-arc-baselines at the internal
#             package and ~/ask-sd, so that every cell trains and plays on the code the paper's rows ran, or a variant
#             of those trees such as code-dagger-rows-turns, the same with turn-based plays); unpacked once, into a
#             machine that holds none of the four (a paper node's own are never replaced), its MD5 kept in
#             ~/gen/bundle/code.unpacked, by which cell.py refuses a cell of another bundle here
#   data.tgz  the benchmark's data, into ~/.cache/continual-arc
# then prime-rl's environment with the four installed (the bundled baselines' remote-bootstrap.sh, which knows each
# vendor's quirks: root without sudo, Lambda's CUDA 12.8 driver, images without Python's headers), Qwen3.5-9B, vLLM
# 0.30.0's own environment, and the harness at the paper nodes' release with its bubblewrap sandbox checked
# (team-node-setup.sh, prime-agent 0.9.6). TRAIN_ONLY=1 leaves out vLLM and the harness (a machine that only trains,
# such as Modal's image), NO_MODEL=1 the model (Modal keeps it in a volume), NO_VLLM=1 vLLM alone (a CPU machine that
# plays only models served elsewhere, the capacity ablation's generator). Idempotent: each step's last line goes to
# ~/gen/setup.log, which ends "setup done", and ~/gen/sandbox says "ok" where plays can run, or why not.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
BUNDLE=${BUNDLE:-code-dagger-rows}
mkdir -p ~/gen
say() { echo "$(date -u +%H:%M:%S) $*" >> ~/gen/setup.log; }
{   # read whole before it runs: bash reads a script as it goes
    say "setup started"
    want=$(md5sum ~/gen/bundle/"$BUNDLE".tgz | cut -d' ' -f1)
    [ -n "$want" ] || { say "the bundle $BUNDLE is not in ~/gen/bundle"; exit 1; }
    have=$(cut -d' ' -f1 ~/gen/bundle/code.unpacked 2>/dev/null)
    if [ -z "$have" ]; then
        for d in loop continual-arc continual-arc-baselines ask-sd; do
            [ -e ~/$d ] && { say "~/$d is here already and not from the bundle: this machine runs other work, set up nothing"; exit 1; }
        done
        tar -C ~ -xzf ~/gen/bundle/"$BUNDLE".tgz || { say "the bundle did not unpack"; exit 1; }
        echo "$want $BUNDLE" > ~/gen/bundle/code.unpacked
        say "code: the trees of the bundle $BUNDLE ($want)"
    elif [ "$have" != "$want" ]; then
        say "another bundle ($(cat ~/gen/bundle/code.unpacked)) is unpacked here, not $BUNDLE: a cell runs on one bundle, so use a fresh machine"; exit 1
    fi
    if [ ! -d ~/.cache/continual-arc ]; then
        mkdir -p ~/.cache && tar -C ~/.cache -xzf ~/gen/bundle/data.tgz || { say "the data did not unpack"; exit 1; }
        say "data: $(du -sh ~/.cache/continual-arc | cut -f1)"
    fi
    command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    pids=()
    if [ "${NO_MODEL:-}" != 1 ]; then
        (HF_HUB_ENABLE_HF_TRANSFER=1 uv run --with huggingface_hub --with hf_transfer python -c \
            "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))" > ~/gen/model.log 2>&1
         say "model: $(tail -1 ~/gen/model.log)") & pids+=($!)
    fi
    if [ "${TRAIN_ONLY:-}" != 1 ] && [ "${NO_VLLM:-}" != 1 ]; then
        (INSTALL_ONLY=1 bash ~/continual-arc-baselines/scripts/vllm-serve.sh Qwen/Qwen3.5-9B 0 0 > ~/gen/vllm-install.log 2>&1
         say "vllm: $(tail -1 ~/gen/vllm-install.log)") & pids+=($!)
    fi
    bash ~/continual-arc-baselines/scripts/remote-bootstrap.sh > ~/gen/bootstrap.log 2>&1
    say "bootstrap: $(tail -1 ~/gen/bootstrap.log)"
    grep -q "bootstrap done" ~/gen/bootstrap.log || { say "prime-rl's environment failed (~/gen/bootstrap.log)"; exit 1; }
    if [ "${TRAIN_ONLY:-}" != 1 ]; then
        PRIME_AGENT_VERSION=0.9.6 bash ~/continual-arc-baselines/scripts/team-node-setup.sh > ~/gen/team-setup.log 2>&1
        say "harness: $(grep -E 'bubblewrap|prime-agent|imports ok' ~/gen/team-setup.log | tr '\n' ' ')"
        if grep -q "^bubblewrap works" ~/gen/team-setup.log && grep -q "team driver imports ok" ~/gen/team-setup.log; then
            echo ok > ~/gen/sandbox
        else
            grep -E "BUBBLEWRAP FAILS|MISSING|Error" ~/gen/team-setup.log | head -1 > ~/gen/sandbox
            [ -s ~/gen/sandbox ] || echo "team-node-setup.sh did not finish (~/gen/team-setup.log)" > ~/gen/sandbox
        fi
    else
        echo "TRAIN_ONLY: no harness set up here" > ~/gen/sandbox
    fi
    for p in "${pids[@]}"; do wait "$p"; done
    say "sandbox: $(cat ~/gen/sandbox)"
    say "setup done"
}
