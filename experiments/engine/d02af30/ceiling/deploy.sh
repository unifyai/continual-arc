#!/usr/bin/env bash
# deploy.sh HOST: the library ceiling's code on its machine (README.md, "Launching"), a clone of the snapshot
# NODE-glm-harness-20261105 made by sweep/up.sh, such as NODE-blocks, then what it must show before anything starts.
# HOST is user@host as this machine reaches it. Two trees of their own, beside the snapshot's and blocks/'s, which stay
# as their runs played them:
#   ~/cab-ceiling  continual-arc-research 1543c2e, the baselines main's GLM rows played on (the snapshot's ~/cab-turns),
#                  with baselines.patch: base.SystemLearner's primitives and protocol.PRIMITIVES_NOTE, the modules of
#                  systems/primitives/, and the building-block guidance before them, as on that repository's branch turns;
#                  and ~/cab-turns' .venv and .env, the venv's editable install of continual_arc_baselines pointed at it,
#                  as refs-deploy.sh gave each new tree the venv before it
#   ~/loop-ceiling this repository's ee7a4b3, the reference rows' code (the snapshot's ~/loop-turns), with
#                  reference.patch (rows hpl0d1 and hpl0d4, reference.py driver, and blocks/'s rows before them), and
#                  this checkout's keep.sh and ceiling/run.sh
# each with keep_stage's record of its commit and patch (SYNCED.json and SYNCED.diff), which the run supervisor keeps
# with every run, and runs.py; put in whole beside an old copy and swapped in, never while a run plays from it. Then
# sweep/records.sh writes the record of the snapshot's own trees (~/benchmark-turns, which every play imports, among them),
# and the machine shows: the venv importing continual_arc from ~/benchmark-turns and continual_arc_baselines from
# ~/cab-ceiling with both modules of primitives, the rows' tests passing there, prime-agent's version (main's rows played 0.9.6), the
# first 12 hex digits of the sha256 of the OpenRouter key in .env (the `OPENROUTER_KEY_NAME` key's, which main's rows played on), the
# room, the run supervisor and repl-guard.sh. Runs from this checkout at main, with CARC_BASELINES naming
# continual-arc-research's checkout and INTERNAL continual-arc-internal's clone (sweep/records.sh's).
set -euo pipefail
H=${1:?usage: deploy.sh HOST}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(dirname "$S")
: "${CARC_BASELINES:?set CARC_BASELINES to the continual-arc-research checkout}"
: "${INTERNAL:?set INTERNAL to the continual-arc-internal clone}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new)
. "$R/keep.sh"
D=$(mktemp -d "$HOME/ceiling-deploy.XXXXXX")
trap 'git -C "$CARC_BASELINES" worktree remove --force "$D/cab" 2> /dev/null; git -C "$R" worktree remove --force "$D/loop" 2> /dev/null; rm -rf "$D"' EXIT
"${SSH[@]}" "$H" 'pgrep -f "[l]oop-ceiling/" > /dev/null' && { echo "refused: a run plays from ~/loop-ceiling on $H" >&2; exit 1; }

# the two trees, staged here as they will run there
git -C "$CARC_BASELINES" cat-file -e "1543c2e^{commit}" 2> /dev/null || git -C "$CARC_BASELINES" fetch -q origin
git -C "$CARC_BASELINES" worktree add -q --detach "$D/cab" 1543c2e
git -C "$D/cab" apply "$S/baselines.patch"
keep_stage "$D/cab-record" "$D/cab" > /dev/null
git -C "$R" worktree add -q --detach "$D/loop" ee7a4b3
git -C "$D/loop" apply "$S/reference.patch"
cp "$R/keep.sh" "$D/loop/" && mkdir "$D/loop/ceiling" && cp "$S/run.sh" "$D/loop/ceiling/"
keep_stage "$D/loop-record" "$D/loop" > /dev/null
put() {   # put TREE RECORD DIR: the staged tree and its record in ~/DIR.new on HOST
    tar -C "$1" --exclude=.git -cf - . | "${SSH[@]}" "$H" "rm -rf ~/$3.new && mkdir ~/$3.new && tar -C ~/$3.new -xf -"
    tar -C "$2" -cf - SYNCED.json SYNCED.diff runs.py | "${SSH[@]}" "$H" "tar -C ~/$3.new -xf -"
    echo "$H: ~/$3 staged, SYNCED.json $(tr -d '\n' < "$2/SYNCED.json" | cut -c1-120)"
}
put "$D/cab" "$D/cab-record" cab-ceiling
put "$D/loop" "$D/loop-record" loop-ceiling
"${SSH[@]}" "$H" 'set -e
for keep in .venv .env; do
    if [ -e ~/cab-ceiling/$keep ]; then mv ~/cab-ceiling/$keep ~/cab-ceiling.new/; else cp -a ~/cab-turns/$keep ~/cab-ceiling.new/; fi
done
site=$(echo ~/cab-ceiling.new/.venv/lib/python3*/site-packages)
echo /home/NODE_USER/cab-ceiling/src > "$site/_editable_impl_continual_arc_baselines.pth"
for dir in cab-ceiling loop-ceiling; do
    rm -rf ~/$dir.prev; if [ -d ~/$dir ]; then mv ~/$dir ~/$dir.prev; fi; mv ~/$dir.new ~/$dir; rm -rf ~/$dir.prev
done'

# the checks: the snapshot's trees and their records, then what the machine shows besides them
rc=0; "$R/sweep/records.sh" "$H" || rc=1
IFS= read -r -d '' CHECK <<'REMOTE' || true
export PATH=$HOME/.local/bin:$PATH
cd ~/cab-ceiling || exit 1
uv run --no-sync python -c 'import continual_arc, continual_arc_baselines
from continual_arc_baselines.systems import base, protocol
assert "/benchmark-turns/" in continual_arc.__file__ and "/cab-ceiling/" in continual_arc_baselines.__file__
assert {p.name for p in base.PRIMITIVES.iterdir()} >= {"stream_u1.py", "stream_u4.py"} and protocol.PRIMITIVES_NOTE
print("  venv imports:", continual_arc.__file__, continual_arc_baselines.__file__, "with both modules of primitives")' < /dev/null 2>&1 | tail -1
(cd ~/loop-ceiling && ~/cab-ceiling/.venv/bin/python -m pytest -q -p no:cacheprovider test_reference.py < /dev/null 2>&1 |
    tail -1 | sed 's/^/  test_reference.py: /')
echo "  prime-agent: $(command -v prime-agent || echo "not on PATH"), $(prime-agent --version < /dev/null 2>&1 | head -1)"
python3 -c 'import hashlib, re
keys = [m.group(1).strip().strip("\"'"'"'") for m in re.finditer(r"^OPENROUTER_API_KEY=(.*)$", open(".env").read(), re.M)]
print("  key: OPENROUTER_API_KEY", ", ".join(hashlib.sha256(k.encode()).hexdigest()[:12] for k in keys) or "absent")'
echo "  room: $(free -g | awk '/^Mem:/ {print $7}') GB of memory free, $(df -BG --output=avail ~ | tail -1 | tr -d ' ') of disk"
echo "  supervisor: $(bash -c '. ~/loop-ceiling/keep.sh && keep_runs_py' || echo none)"
if [ -f ~/frozen-turns/repl-guard.sh ]; then echo "  repl-guard.sh: there"; else echo "  repl-guard.sh: missing"; fi
REMOTE
echo "== $H"
"${SSH[@]}" "$H" "bash -c $(printf %q "$CHECK")"
[ "$rc" = 0 ] || { echo "records.sh: a tree differs from its commit or names another: trace it before a run starts" >&2; exit 1; }
