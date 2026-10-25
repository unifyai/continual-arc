"""Tests for gpupool.py, the trainer GPUs runs on one node share, and for realtime.py's trainer's decisions over it
(realtime.tick): who gets which GPU, and when a train process is stopped or started. Run with ``python3 -m pytest
test_gpupool.py -q``."""
from __future__ import annotations

import os

import pytest

import gpupool
import realtime

POOL = [2, 3, 4, 5, 6, 7]


def run(want: bool, held: list[int] = (), busy: bool = False) -> dict:
    return {"want": want, "busy": busy, "held": list(held), "pid": None, "seen": 0.0}


def test_a_run_alone_gets_the_whole_pool_and_keeps_it_while_idle_since_nobody_else_wants_it():
    assert gpupool.split(POOL, {"a": run(True)}) == {"a": POOL}
    assert gpupool.split(POOL, {"a": run(False, POOL)}) == {"a": []}   # its idle process's GPUs are free for others


def test_the_pool_goes_evenly_to_the_runs_that_want_it_each_keeping_what_it_holds():
    share = gpupool.split(POOL, {"a": run(True, [5, 6, 7]), "b": run(True, [2, 3, 4])})
    assert share == {"a": [5, 6, 7], "b": [2, 3, 4]}
    share = gpupool.split(POOL, {"a": run(True), "b": run(True), "c": run(True)})
    assert sorted(len(g) for g in share.values()) == [2, 2, 2] and sorted(sum(share.values(), [])) == POOL


def test_an_odd_gpu_goes_to_the_run_holding_more_and_a_newcomer_takes_free_ones_first():
    share = gpupool.split([1, 2, 3, 4, 5, 6, 7], {"a": run(True, [1, 2, 3, 4]), "b": run(True)})
    assert share == {"a": [1, 2, 3, 4], "b": [5, 6, 7]}


def test_a_phase_in_flight_gives_gpus_up_to_a_newcomer_but_never_takes_more_until_its_next_phase():
    # a holds all six mid-phase; b's phase is ready: a's share falls to three, b gets the other three
    share = gpupool.split(POOL, {"a": run(True, POOL, busy=True), "b": run(True)})
    assert share == {"a": [2, 3, 4], "b": [5, 6, 7]}
    # b's phase ended and b is idle: a, mid-phase on three, stays on three (it grows when its next phase starts)
    share = gpupool.split(POOL, {"a": run(True, [2, 3, 4], busy=True), "b": run(False, [5, 6, 7])})
    assert share == {"a": [2, 3, 4], "b": []}
    # c starts while a is held to its three mid-phase: c takes the other three, not half of what is left after a's cap
    share = gpupool.split(POOL, {"a": run(True, [2, 3], busy=True), "c": run(True)})
    assert share == {"a": [2, 3], "c": [4, 5, 6, 7]}


def test_a_list_overlapping_another_pools_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(gpupool, "_alive", lambda pid: pid == os.getpid())   # this process stands for a train process
    gpupool.Pool([4, 5, 6, 7], "a", root=tmp_path).update(True, False, [4, 5, 6, 7], os.getpid())
    with pytest.raises(SystemExit, match="overlaps the pool 4-5-6-7"):
        gpupool.Pool([6, 7], "b", root=tmp_path)
    gpupool.Pool([0, 1], "c", root=tmp_path)   # disjoint: a pool of its own
    gpupool.Pool([4, 5, 6, 7], "d", root=tmp_path)   # the same list: the same pool


def test_every_run_sees_the_same_split_and_a_run_gone_for_good_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(gpupool, "_alive", lambda pid: pid == os.getpid())   # this process stands for a train process
    a, b = gpupool.Pool(POOL, "a", root=tmp_path), gpupool.Pool(POOL, "b", root=tmp_path)
    a.update(True, True, POOL, os.getpid())
    view = b.update(True, False, [], None)
    assert view["share"] == {"a": [2, 3, 4], "b": [5, 6, 7]}
    assert b.held_by_others([5, 6, 7], view["runs"]) == {5, 6, 7}   # b waits until a's process lets them go
    assert b.share_if_starting() == [5, 6, 7]
    assert b.entry()["want"] and a.entry()["busy"]
    monkeypatch.setattr(gpupool.time, "time", lambda: 1e12)   # long after a last wrote, its process gone
    monkeypatch.setattr(gpupool, "_alive", lambda pid: False)
    assert b.update(True, False, [], None)["share"] == {"b": POOL}


def test_a_claim_fails_on_gpus_another_runs_live_process_holds_so_no_gpu_is_started_on_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(gpupool, "_alive", lambda pid: pid in (1, 2))
    a, b = gpupool.Pool(POOL, "a", root=tmp_path), gpupool.Pool(POOL, "b", root=tmp_path)
    assert a.claim([2, 3, 4], 1)
    assert not b.claim([4, 5, 6], 2)   # b's split, made before a's claim, gave it GPU 4
    assert b.claim([5, 6, 7], 2) and b.entry()["held"] == [5, 6, 7]
    assert b.update(True, False, [5, 6, 7], 2)["share"] == {"a": [2, 3, 4], "b": [5, 6, 7]}   # each keeps its claim


def test_a_pid_that_is_no_realtime_process_is_not_a_live_train_process():
    """After a machine's restart the state file's pids may be anyone's: only a realtime.py process holds GPUs."""
    assert not gpupool._alive(os.getpid()) and not gpupool._alive(None)


def tick(**kw) -> str:
    args = dict(run="a", share={"a": [2, 3, 4], "b": [5, 6, 7]}, held=[2, 3, 4, 5, 6, 7], busy=True, want=True,
                alive=True, asked=False, others_hold=set(), checkpoints=True)
    args.update(kw)
    return realtime.tick(**args)


def test_the_trainer_stops_its_process_when_another_runs_share_takes_its_gpus():
    assert tick() == "stop"                          # mid-phase, with step checkpoints: at its next one
    assert tick(checkpoints=False) == "wait"         # without, the phase would start over: it ends at its phase's end
    assert tick(busy=False, checkpoints=False) == "stop"   # between phases: nothing to lose
    assert tick(asked=True) == "wait"                # asked once already
    assert tick(held=[2, 3, 4]) == "wait"            # nobody wants what it holds


def test_the_trainer_starts_a_process_on_its_share_once_no_other_process_holds_any_of_it():
    assert tick(alive=False, held=[], others_hold={5}) == "start"
    assert tick(run="b", alive=False, held=[], others_hold={5}) == "wait"
    assert tick(alive=False, held=[], want=False) == "wait"
    assert tick(alive=False, held=[], share={"a": [], "b": POOL}) == "wait"
