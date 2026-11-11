#!/usr/bin/env bash
# final-resets.sh: the final trained rows' resets at Qwen3.5-9B's model-card sampling (paper.SAMPLING; the author's go of 28 Oct
# 2026), on an 8-GPU node whose ~/loop is main (search/deploy-code.sh) and whose frozen.sh is copied to ~/frozen-tools:
# h0, the untrained student wiped before every instance, and h0c1, the same told c1 alone (paper.TOLD_ROWS), each from
# nothing by frozen.sh, side by side (h0's plays on GPUs 0-3, h0c1's on 4-7, frozen.sh giving each test half). Four
# plays each of the one-user tests into ~/runs/paper/qwen-ROW-card (frozen/, then PLAY=2, 3 and 4), then one play each
# of the 2- and 4-user users mode's seen tests into ~/runs/paper/qwen-ROW-card-uN, SEEN_OF ~/runs/paper/seen-of-uN: a
# folder whose play/schedule.json is a copy of any 2- or 4-user run's in that mode (frozen.py schedule reads the mode
# and the users from it, and every such run has the one seen test), which must be there before the script starts.
# Resumable: a play whose done marker is there is passed over. Log ~/runs/resets.log.
# Usage (on the node): cd ~ && (setsid -f nohup bash ~/final-resets.sh > /dev/null 2>&1 < /dev/null)
set -u
log() { echo "$(date -u '+%m-%d %H:%M') $*" >> ~/runs/resets.log; }
for users in 2 4; do
    [ -f ~/runs/paper/seen-of-u$users/play/schedule.json ] || { log "no ~/runs/paper/seen-of-u$users/play/schedule.json"; exit 1; }
done
play() {   # play NAME GPUS DONE [VAR=VALUE...]: one frozen.sh play of NAME on GPUS, waited for until its DONE marker
    local name=$1 gpus=$2 done=$3; shift 3
    [ -f "$done" ] && return 0
    (cd ~ && env "$@" setsid -f nohup bash ~/frozen-tools/frozen.sh "$name" "$gpus" > /dev/null 2>&1 < /dev/null)
    log "started $name on $gpus ($*)"
    until [ -f "$done" ]; do sleep 60; done
    log "done $name ($*)"
}
chain() {   # chain ROW GPUS: every play of ROW, one after another on GPUS
    local row=$1 gpus=$2 out=$HOME/runs/paper/qwen-$1-card/frozen
    play "qwen-$row-card" "$gpus" "$out/done" ROW=$row
    for k in 2 3 4; do play "qwen-$row-card" "$gpus" "$out/done-$k" ROW=$row PLAY=$k; done
    for users in 2 4; do
        play "qwen-$row-card-u$users" "$gpus" "$HOME/runs/paper/qwen-$row-card-u$users/frozen/done" ROW=$row \
            SEEN_OF=$HOME/runs/paper/seen-of-u$users
    done
}
chain h0 0,1,2,3 &
chain h0c1 4,5,6,7 &
wait
log "every reset played"
