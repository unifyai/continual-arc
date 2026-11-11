#!/usr/bin/env bash
# refs-deploy.sh HOST...: the final runs' code on each reference machine (the author, 27 Oct 2026: every reference row again
# from round 0 on the trained runs' code, turn-based), as trees of its own beside the ones its earlier rows play on
# (~/loop-v2, ~/cab-v2, ~/benchmark-v2), so that no running row's code changes under it: the pushed main of
# continual-arc-internal as ~/benchmark-turns, the baselines' branch turns as ~/cab-turns and this repository's main as
# ~/loop-turns, the three search/deploy-code.sh ships to the trained runs' nodes. Each is a git archive unpacked into a
# new folder, and the three are swapped in together once all are unpacked, so nothing reads a mix of old and new.
# ~/cab-turns keeps its .venv and .env across a deploy; the first deploy makes them from ~/cab-v2's, the venv's
# editable installs pointed at the new trees, since the reference machines are clones of one disk with no other way to
# build one (reference_setup.sh). Then it checks on each HOST that the venv imports the new trees and that
# test_reference.py passes there. Runs from a laptop with the three repositories checked out at ~/<repo>; a row starts
# on them with reference.sh's BASELINES=~/cab-turns from ~/loop-turns, ref-tests.sh's and extra-tests.sh's defaults.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: refs-deploy.sh HOST..." >&2; exit 1; }
REPOS=("continual-arc:benchmark-turns:origin/main" "continual-arc-baselines:cab-turns:origin/turns"
       "continual-arc-training:loop-turns:origin/main")
for spec in "${REPOS[@]}"; do
    IFS=: read -r repo _ _ <<< "$spec"
    git -C ~/"$repo" fetch -q origin
done
for H in "$@"; do
    for spec in "${REPOS[@]}"; do
        IFS=: read -r repo dir ref <<< "$spec"
        git -C ~/"$repo" archive "$ref" | ssh -o BatchMode=yes "$H" "rm -rf ~/$dir.new && mkdir ~/$dir.new && tar -C ~/$dir.new -xf -"
    done
    ssh -o BatchMode=yes "$H" 'set -e
for keep in .venv .env; do
    if [ -e ~/cab-turns/$keep ]; then mv ~/cab-turns/$keep ~/cab-turns.new/; else cp -a ~/cab-v2/$keep ~/cab-turns.new/; fi
done
site=$(echo ~/cab-turns.new/.venv/lib/python3*/site-packages)
echo /home/NODE_USER/benchmark-turns/src > "$site/_editable_impl_continual_arc.pth"
echo /home/NODE_USER/cab-turns/src > "$site/_editable_impl_continual_arc_baselines.pth"
for dir in benchmark-turns cab-turns loop-turns; do rm -rf ~/$dir.prev; [ -d ~/$dir ] && mv ~/$dir ~/$dir.prev; mv ~/$dir.new ~/$dir; done
rm -rf ~/benchmark-turns.prev ~/cab-turns.prev ~/loop-turns.prev
export PATH=$HOME/.local/bin:$PATH
cd ~/cab-turns && uv run --no-sync python -c "
import continual_arc, continual_arc_baselines
for m in (continual_arc, continual_arc_baselines):
    assert \"-turns/\" in m.__file__, m.__file__
print(\"imports\", continual_arc.__file__, continual_arc_baselines.__file__)"
cd ~/loop-turns && ~/cab-turns/.venv/bin/python -m pytest -q -p no:cacheprovider test_reference.py 2>&1 | tail -1'
    echo "$H: the final runs' trees deployed"
done
