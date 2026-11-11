#!/usr/bin/env bash
# Brings a fresh CPU machine to the reference rows' ready state (reference.py's docstring), as REF_ROWS_NODE_A and
# NODE-compact-probe were made on 19 Oct 2026: this laptop's key and the operator's in its metadata, so brain's
# paper-backup reaches it; continual-arc-baselines, its data and .env (node-sync.sh --data, install_node_env), whose
# venv runs reference.py; ~/loop from this checkout; prime-agent 0.9.6 as on the RL nodes; and api-node-bootstrap.sh
# (Node 22, bubblewrap, the venv). Make the machine first with continual-arc-baselines' create-node.sh (CREATE_ONLY=1, with
# TTL_HOURS and PURPOSE, e2-standard-4 for six rows), then route its lease (lease.py route gcp NAME --ssh NODE_USER@IP)
# so the backup finds it. Runs from a laptop with gh signed in and ~/continual-arc-baselines checked out.
# Usage: bash reference_setup.sh NAME IP [ZONE]
set -uo pipefail
NAME=${1:?set NAME: the machine name}; IP=${2:?set IP: its address}; ZONE=${3:-us-central1-a}
H=NODE_USER@$IP; LOOP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); B=$HOME/continual-arc-baselines
cd "$B"
K=$(mktemp); printf 'NODE_USER:%s\nNODE_USER:%s\n' "$(cat ~/.ssh/id_ed25519.pub)" "$(cat infra/operator.pub)" > "$K"
CLOUDSDK_ACTIVE_CONFIG_NAME=automation gcloud compute instances add-metadata "$NAME" --zone "$ZONE" \
    --project GCP_PROJECT --metadata-from-file ssh-keys="$K" && echo "metadata keys set"
rm -f "$K"
bash scripts/node-sync.sh "$H" --data
. scripts/secret.sh; install_node_env "$H" && echo "env installed"
rsync -az -e ssh --exclude .git --exclude __pycache__ "$LOOP/" "${H}:loop/" && echo "loop synced"
ssh "$H" 'curl --proto "=https" --proto-redir "=https" -fsSL https://app.primeintellect.ai/prime-agent/install.sh | PRIME_AGENT_VERSION=0.9.6 sh >/dev/null 2>&1; echo "prime-agent $($HOME/.local/bin/prime-agent --version 2>/dev/null | head -1)"'
gh auth token | ssh "$H" 'IFS= read -r GH_TOKEN; export GH_TOKEN; bash ~/continual-arc-baselines/scripts/api-node-bootstrap.sh 2>&1 | tail -12'
ssh "$H" 'export PATH=$HOME/.local/bin:$PATH; prime-agent --version | head -1; node -v; bwrap --ro-bind / / --dev /dev --proc /proc --unshare-pid --die-with-parent true && echo "bubblewrap works"; cd ~/loop && ~/continual-arc-baselines/.venv/bin/python -m pytest -q -p no:cacheprovider test_reference.py 2>&1 | tail -1'
