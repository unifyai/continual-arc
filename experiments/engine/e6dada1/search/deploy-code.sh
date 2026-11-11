#!/usr/bin/env bash
# deploy-code.sh HOST: continual-arc-training's and ask-sd's pushed main on HOST as ~/loop and ~/ask-sd (git archives,
# unpacked into a new folder and swapped in, so nothing reads a half-written tree).
set -euo pipefail
H=$1
git -C ~/continual-arc-training fetch -q origin && git -C ~/ask-sd fetch -q origin
git -C ~/continual-arc-training archive origin/main | ssh -o BatchMode=yes "$H" 'rm -rf ~/loop.new && mkdir ~/loop.new && tar -C ~/loop.new -xf - && rm -rf ~/loop.prev && { [ -d ~/loop ] && mv ~/loop ~/loop.prev || true; } && mv ~/loop.new ~/loop'
git -C ~/ask-sd archive origin/main | ssh -o BatchMode=yes "$H" 'rm -rf ~/ask-sd.new && mkdir ~/ask-sd.new && tar -C ~/ask-sd.new -xf - && rm -rf ~/ask-sd.prev && { [ -d ~/ask-sd ] && mv ~/ask-sd ~/ask-sd.prev || true; } && mv ~/ask-sd.new ~/ask-sd && grep -c PG_DUAL_CLIP ~/ask-sd/ask_sd/trainer.py | sed "s/^/ask-sd 836e7ed+ ok: /"; grep -c dual_clip_fraction ~/loop/train.py | sed "s/^/loop c3ad0a4+ ok: /"'
