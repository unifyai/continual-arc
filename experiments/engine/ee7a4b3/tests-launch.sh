#!/usr/bin/env bash
# tests-launch.sh DEST SPEC...: more plays of trained runs' frozen tests and of their reset (frozen.sh PLAY=K), on a
# bare machine with 8 NVIDIA GPUs whose own runs are gone, set up from the bucket alone as a generalisation cell is
# (search/generalisation/README.md): the DAgger rows' code bundle (code-dagger-rows, the frozen tests under the block's
# 1,740 s, as every paper row's first tests played), the benchmark's data, and of each run what its tests start from,
# its play config, its users' harness states at the stream's end and its last phase's adapter, at the run's own path
# ~/runs/rt/RUN, since a harness names its folder by that path. Each SPEC is NODE/RUN:K,K,... (the run's folder under
# the bucket's results/runs/gpu-nodes, and the plays to make of it), or h0:K,K,... for the trained runs' reset, the base
# model in row h0's harness from nothing (refs.sh's ~/runs/paper/qwen-h0); a run may come in several SPECs, so that
# its plays interleave with another's. The plays run one after another in the order given, each
# play's two tests side by side on four GPUs each, frozen.sh's layout, which every trained row's first tests and their
# reset's played in but PG's at one user (one GPU a test). The operator's hourly backup copies ~/runs off the machine
# once its lease has a route (create-node.sh routes it). A play named RUN:K on a line of ~/runs/tests.skip there is
# passed over, so that a second machine can take the end of the plan. JUMP=NODE_USER@<gateway> for a machine with no public address;
# BASELINES is continual-arc-research's checkout (secret.sh), ~/continual-arc-baselines unless set.
# Follow it: ssh DEST 'tail ~/gen/setup.log ~/runs/tests.log'; each run's frozen/progress says where its plays are.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,16p' "$0" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
BASELINES=${BASELINES:-$HOME/continual-arc-baselines}
DEST=$1; shift
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new ${JUMP:+-J "$JUMP"})
on() { "${SSH[@]}" "$DEST" "$@"; }
G=results/runs/gpu-nodes
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
bash "$HERE/search/generalisation/tools.sh" "$T"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$T" -czf - . | on 'rm -rf ~/gen/tools.new && mkdir -p ~/gen/tools.new && tar -C ~/gen/tools.new -xzf - \
    && rm -rf ~/gen/tools.old && { [ ! -d ~/gen/tools ] || mv ~/gen/tools ~/gen/tools.old; } && mv ~/gen/tools.new ~/gen/tools'
echo "tools: $(cat "$T/COMMIT")"
plan=()
{
    bash "$BASELINES/scripts/secret.sh" --node-token
    echo "bundles/generalisation/code-dagger-rows.tgz gen/bundle/code-dagger-rows.tgz"
    echo "bundles/generalisation/data-cache.tgz gen/bundle/data.tgz"
    for src in $(printf '%s\n' "${@%%:*}" | awk '!seen[$0]++'); do
        [ "$src" = h0 ] && continue
        run=${src#*/}
        for f in alias-salt config.json driver.json meta.json schedule.json; do echo "$G/$src/play/$f runs/rt/$run/play/$f"; done
        echo "$G/$src/play/users/ runs/rt/$run/play/users"
        for f in adapter_config.json adapter_model.safetensors; do
            echo "$G/$src/train/p0199/adapter/$f runs/rt/$run/train/p0199/adapter/$f"
        done
    done
} | on 'python3 ~/gen/tools/bucket.py get'
for spec in "$@"; do
    src=${spec%%:*}
    IFS=, read -ra plays <<< "${spec#*:}"
    for k in "${plays[@]}"; do plan+=("${src#*/}:$k"); done
done
printf '%s\n' "${plan[@]}" | on 'mkdir -p ~/runs && cat > ~/runs/tests.plan'
on "cd ~ && (setsid -f nohup bash -c '{ grep -q \"setup done\" ~/gen/setup.log 2>/dev/null || BUNDLE=code-dagger-rows bash ~/gen/tools/setup.sh; } \
    && tail -n 1 ~/gen/setup.log | grep -q \"setup done\" && grep -qx ok ~/gen/sandbox && \
    while IFS=: read -r run k; do
        grep -qx \"\$run:\$k\" ~/runs/tests.skip 2>/dev/null && continue
        echo \"\$(date -u +%m-%d\\ %H:%M) \$run play \$k\" >> ~/runs/tests.log
        if [ \"\$run\" = h0 ]; then
            ROW=h0 PLAY=\$k CODE=\$HOME/loop NCCL_NVLS_ENABLE=0 bash ~/gen/tools/frozen.sh qwen-h0 0,1,2,3,4,5,6,7 < /dev/null
        else
            ADAPTER=\$HOME/runs/rt/\$run/train/p0199/adapter SERVED=\$run-p0199 PLAY=\$k CODE=\$HOME/loop NCCL_NVLS_ENABLE=0 \
                bash ~/gen/tools/frozen.sh \$run 0,1,2,3,4,5,6,7 < /dev/null
        fi || echo \"\$(date -u +%m-%d\\ %H:%M) \$run play \$k stopped (its frozen/progress)\" >> ~/runs/tests.log
    done < ~/runs/tests.plan; echo \"\$(date -u +%m-%d\\ %H:%M) every play of the plan taken\" >> ~/runs/tests.log' \
    >> ~/runs/tests.launch.log 2>&1 < /dev/null)"
echo "started on $DEST: ${plan[*]}"
