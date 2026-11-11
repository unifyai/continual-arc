#!/usr/bin/env bash
# switch-run.sh USERS VALUES... (on a DAgger final run's node): the run dagger-u<USERS>-lag1 goes on from its next phase
# on the recent dose and a held-out stop of patience 4 (the author, 21 Oct 2026; continual-arc-training method-search.md,
# "Why DAgger's phases stop"). The run is stopped (~/stop-run.sh, which leaves the server and the relay up), its ~/loop
# given the recent dose (switch-recent.py, its files kept as *.pre-recent), and the run started again through brain's
# realtime_runs.py with VALUES (its last "started:" line's, any changed) and DOSE=recent PATIENCE=4. Every loop resumes
# from its files, the trainer from train/latest.json, so the run goes on where it was. ~/loop is patched only once the
# run's realtime.sh is gone, since bash reads a running script from its file as it goes; should the patch fail, ~/loop
# is put back and the run started again as it was.
# The stop waits for a moment that loses nothing: the player waiting for a phase and the trainer for a round's labels
# (about ten such moments a day per run, 21 Oct). After an hour without one, a moment the player waits will do (the
# phase in training then trains again from its start, on the recent dose); after three hours, any moment.
# Log: /tmp/switch-dagger-u<USERS>.log.
set -u
users=$1; shift
run=dagger-u$users-lag1
top=/home/NODE_USER/runs/rt/$run
progress=$top/progress
say() { echo "$(date -u '+%m-%d %H:%M') $*"; }
state() {   # idle: neither play nor training would be lost; training: the player waits, a phase trains; busy: a round plays
    local play phase
    play=$(grep "realtime: play round" "$progress" | tail -1)
    phase=$(grep -E "realtime: phase [0-9]+ (started|done)" "$progress" | tail -1)
    case $play in *" waits for phase "*) ;; *) echo busy; return ;; esac
    case $phase in *" done:"*) echo idle ;; *) echo training ;; esac
}
groups() { ps -eo pgid=,args= | awk -v r="--run $top " 'index($0, r) && $0 ~ /realtime\.py/ {print $1}' | sort -u; }
alive() {   # processes left in the loops' groups, a trainer's data-parallel workers included
    local n=0 g
    for g in $loop_groups; do n=$((n + $(ps -eo pgid= | awk -v g="$g" '$1 == g' | wc -l))); done
    echo $n
}
say "waiting for the loops to pause"
for i in $(seq 1 1080); do   # every 10 s for three hours
    s=$(state)
    [ "$s" = idle ] && break
    [ "$s" = training ] && [ "$i" -gt 360 ] && break
    sleep 10
done
say "stopping ($(state)) after: $(grep "realtime: play round" "$progress" | tail -1 | cut -c1-110) / $(grep -E "realtime: phase [0-9]+ (started|done)" "$progress" | tail -1 | cut -c1-110)"
loop_groups=$(groups)
bash ~/stop-run.sh dagger "$users"
for i in $(seq 1 24); do [ "$(alive)" = 0 ] && break; sleep 5; done
if [ "$(alive)" != 0 ]; then   # a process that outlived SIGTERM for two minutes
    for g in $loop_groups; do kill -9 -- "-$g" 2> /dev/null; done
    sleep 10
fi
say "processes left of the loops' groups ($(echo $loop_groups)): $(alive)"
if { grep -q "RECENT_ROUNDS" ~/loop/realtime.py || python3 /tmp/switch-recent.py ~/loop; } &&
        python3 -c "import ast; ast.parse(open('/home/NODE_USER/loop/realtime.py').read())" && bash -n ~/loop/realtime.sh; then
    python3 /tmp/realtime_runs.py launch "$run" "$@" DOSE=recent PATIENCE=4
else
    for f in realtime.py realtime.sh; do [ -f ~/loop/$f.pre-recent ] && cp -p ~/loop/$f.pre-recent ~/loop/$f; done
    say "the patch failed: ~/loop put back, the run started again as it was"
    python3 /tmp/realtime_runs.py launch "$run" "$@" DOSE=aggregate
fi
say "started: $(grep "realtime.sh: started:" "$progress" | tail -1 | cut -c1-300)"
