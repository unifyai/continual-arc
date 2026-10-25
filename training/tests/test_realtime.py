"""The real-time driver's loops (continual_arc_training.realtime) on hand-made run folders, with play, the oracle and
training faked: which model plays each round under the fixed lag, which games each labelling pass asks about and what
the oracle is told, and what each row's phases train on; and realtime.sh's settings, as its "started:" line records
them."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import time
import types
from pathlib import Path

import pytest

from continual_arc_training import checkpoints, gpupool, labeller, oracle, phase, play, realtime, server, stack, updates
from continual_arc_training.calls import Call

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _split(run: Path, r: int, train_games: int, dev_games: int) -> None:
    d = realtime.split_dir(run, r)
    d.mkdir(parents=True)
    t, v = realtime.sides(r)
    (d / "split.json").write_text(json.dumps({t: {"games": train_games}, v: {"games": dev_games}}))


def _finish_phase(run: Path, phase_: int) -> None:
    """A phase the trainer has finished, as close_phase leaves it."""
    out = realtime.phase_dir(run, phase_)
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    realtime.write_json(out / "rounds.json", {"rounds": phase_ + 1})


def test_ready_rounds_counts_from_round_0_in_a_row_by_each_rows_own_marker(tmp_path):
    _split(tmp_path, 0, 1, 0)
    _split(tmp_path, 1, 0, 1)
    _split(tmp_path, 3, 1, 0)
    # a round is labelled once its label set is split, DAgger's and OPSD's alike; round 2 is not, so round 3 waits
    assert realtime.ready_rounds(tmp_path, 10, "dagger") == realtime.ready_rounds(tmp_path, 10, "opsd") == 2
    (realtime.round_dir(tmp_path, 0) / "play.done").touch()
    assert realtime.ready_rounds(tmp_path, 10, "pg") == 1      # PG needs only the round played


def test_phase_runs_skips_empty_sides(tmp_path):
    _split(tmp_path, 0, 1, 0)
    _split(tmp_path, 1, 0, 1)
    train_runs, dev_runs = realtime.phase_runs(tmp_path, 2)
    assert {r: Path(p).name for r, p in train_runs.items()} == {0: "train-r0000"}
    assert [Path(p).name for p in dev_runs] == ["dev-r0001"]


def _trainer_args(run: Path, rounds: int, row: str = "dagger", lag: int = 1) -> argparse.Namespace:
    return argparse.Namespace(run=str(run), rounds=rounds, row=row, lag=lag, lr=1e-4, batch=64, gpus=4, lora_rank=128,
                              max_seq_len=1024, eval_steps=5, patience=2, max_steps=120, max_epochs=20,
                              val_max_items=1000, skip_first=3, max_minutes=240, start_after=0, dose="aggregate",
                              replay_ratio=1.0, hold_every=10, train_gpus="4,5,6,7")


def test_the_training_settings_build_the_trainer_every_row_trains_on(monkeypatch, tmp_path):
    """The settings carry every field phase.make_trainer reads, the adapter's projections among them."""
    from continual_arc_training import parallel, trainer
    seen = {}
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(parallel, "DataParallelTrainer", lambda model, **kw: seen.update(model=model, **kw))
    a = _trainer_args(tmp_path, 10)
    phase.make_trainer(realtime.train_args(a))
    assert seen["model"] == play.MODEL and seen["lora_targets"] == tuple(trainer.LORA_TARGETS)
    assert (seen["rank"], seen["alpha"], seen["max_seq_len"], seen["world_size"]) == (128, 128, 1024, 4)


def test_the_retrain_from_scratch_parses_the_dagger_runs_settings_with_its_own_stop(tmp_path):
    from continual_arc_training import scratch
    a = scratch.trainer_args(tmp_path, 8)
    assert (a.row, a.lr, a.batch, a.patience, a.skip_first, a.start_after, a.dose) == \
           ("dagger", 1e-4, 64, 2, 3, 24, "aggregate")
    assert (a.gpus, a.train_gpus, a.eval_steps, a.max_steps, a.max_minutes) == \
           (8, "0,1,2,3,4,5,6,7", scratch.EVAL_STEPS, scratch.MAX_STEPS, scratch.MINUTES)


def _train(monkeypatch, a: argparse.Namespace) -> None:
    """What a run's trainer and its train process do for a run whose rounds are all ready, in this process: the
    warm-up's phases written by the trainer (no GPU), then every other phase trained by one train process on the run's
    whole pool, which no other run shares."""
    run = Path(a.run)
    monkeypatch.setattr(gpupool, "ROOT", run / "gpupool")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", a.train_gpus)
    while realtime.next_phase(run) < a.rounds and realtime.ready(run, realtime.next_phase(run), a.row) \
            and realtime.warming_up(run, realtime.next_phase(run), a.start_after):
        realtime.skip_phase(a, run, realtime.next_phase(run))
    realtime.cmd_train(a)


def _fake_training(monkeypatch, phases: list):
    """phase.fit and the rows' updates faked: each phase records which update its row built, from which calls and
    held-out calls, the settings fit was given and the step checkpoint it went on from; everything else is the
    driver's own."""
    monkeypatch.setattr(checkpoints, "catch_sigterm", lambda: None)

    def save(path):   # the adapter as the trainer holds it: after reset, the untrained one
        path.mkdir(parents=True)
        (path / "untrained").touch()

    trainer = types.SimpleNamespace(tokenizer="the tokenizer", reset=lambda: None, save=save)
    monkeypatch.setattr(phase, "make_trainer", lambda a: trainer)
    monkeypatch.setattr(stack, "record", lambda out: None)

    def labelled(kind, done):
        return lambda a, tok, calls, held: types.SimpleNamespace(kind=kind, calls=calls, held=held, units=calls,
                                                                 done=done, stats={})

    monkeypatch.setattr(updates, "Imitation", labelled("sft", {"steps": 3, "minutes": 0.1, "held_out": [],
                                                               "best_step": None}))
    monkeypatch.setattr(updates, "SelfDistillation", labelled("sdft", {"items": {"drawn": 64}, "steps": 1,
                                                                       "minutes": 0.1, "held_out": [0.5, 0.4],
                                                                       "best_step": 5}))
    monkeypatch.setattr(realtime, "plays", lambda run, calls, tok: [f"{c.meta['run']}'s play" for c in calls])

    def fit(a, trainer_, update, out, init, rng):
        phases.append({"kind": update.kind, "trainer": trainer_, "calls": update.calls, "held": update.held,
                       "init": init, "seed": a.seed, "epochs": a.epochs, "eval_steps": a.eval_steps,
                       "patience": a.patience, "max_steps": a.max_steps, "skip_first": a.skip_first,
                       "checkpoint": checkpoints.saved_step(out)})
        (out / "adapter").mkdir()
        return update.done

    monkeypatch.setattr(phase, "fit", fit)
    return trainer


def _labelled(monkeypatch, run: Path, sides: list[tuple[int, int]], calls: int = 2) -> None:
    """Rounds 0.. labelled and split as the labeller leaves them, round r's sides holding sides[r]'s training and
    held-out games, each game ``calls`` labels (load_calls reading a side)."""
    games = {}
    for r, counts in enumerate(sides):
        _split(run, r, *counts)
        games.update(zip(realtime.sides(r), counts, strict=True))
    monkeypatch.setattr(realtime, "load_calls", lambda path: ([types.SimpleNamespace(
        meta={"run": path.name, "user": u, "i": i}) for u in range(games[path.name]) for i in range(calls)], 0))


def _runs(calls: list) -> list:
    return [c.meta["run"] for c in calls]


def test_dagger_trains_one_round_a_phase_on_every_round_so_far_from_the_last_phases_adapter(tmp_path, monkeypatch):
    phases = []
    trainer = _fake_training(monkeypatch, phases)
    _labelled(monkeypatch, tmp_path, [(1, 0), (1, 1), (1, 0)])   # rounds 1 and 2 both labelled: still a phase each
    _train(monkeypatch, _trainer_args(tmp_path, 3))
    assert [p["kind"] for p in phases] == ["sft"] * 3
    assert [_runs(p["calls"]) for p in phases] == [["train-r0000"] * 2, ["train-r0000"] * 2 + ["train-r0001"] * 2,
                                                    ["train-r0000"] * 2 + ["train-r0001"] * 2 + ["train-r0002"] * 2]
    assert all(p["trainer"] is trainer for p in phases)   # one resident trainer
    assert [p["init"] for p in phases] == [None, realtime.phase_dir(tmp_path, 0) / "adapter",
                                           realtime.phase_dir(tmp_path, 1) / "adapter"]
    assert phases[0]["held"] == [] and phases[0]["epochs"] == 1   # no held-out game yet: one pass
    assert _runs(phases[2]["held"]) == ["dev-r0001"] * 2 and phases[2]["epochs"] == 20
    assert [p["seed"] for p in phases] == [0, 1, 2] and phases[2]["eval_steps"] == 5
    assert realtime.latest(tmp_path) == {"phase": 2, "adapter": str(realtime.phase_dir(tmp_path, 2) / "adapter"),
                                         "rounds": 3}
    assert realtime.finished(tmp_path, 1)["rounds"] == 2 and (tmp_path / "trainer.done").exists()
    progress = (tmp_path / "progress").read_text()
    assert "phase 0 started: rounds 0-0, 2 labels, 0 held-out games, 1 pass(es), no held-out stop\n" in progress
    assert "phase 2 started: rounds 0-2, 6 labels, 1 held-out games, 20 pass(es)\n" in progress
    assert "phase 2 done: rounds 0-2, 3 steps in 0.1 min\n" in progress


def _played(run: Path, r: int, games: int) -> None:
    captured = realtime.round_dir(run, r) / "captured"
    captured.mkdir(parents=True, exist_ok=True)
    for u in range(games):
        (captured / f"u{u:03d}.json.gz").touch()


def test_no_phase_trains_before_start_after_games_have_been_played_and_the_base_model_plays_until_then(tmp_path,
                                                                                                        monkeypatch):
    phases = []
    _fake_training(monkeypatch, phases)
    _labelled(monkeypatch, tmp_path, [(2, 0)] * 4, calls=1)
    for r in range(4):
        _played(tmp_path, r, 2)
    a = _trainer_args(tmp_path, 4)
    a.start_after = 5   # 2, 4, 6, 8 games played after rounds 0..3: phases 0 and 1 train nothing
    _train(monkeypatch, a)
    assert [sorted(set(_runs(p["calls"]))) for p in phases] == [["train-r0000", "train-r0001", "train-r0002"],
                                                                 ["train-r0000", "train-r0001", "train-r0002",
                                                                  "train-r0003"]]
    # a warm-up phase saves no adapter (the base model plays its round), and the first that trains starts from the base
    assert not (realtime.phase_dir(tmp_path, 0) / "adapter").exists()
    assert realtime.finished(tmp_path, 1)["adapter"] is None
    assert json.loads((realtime.phase_dir(tmp_path, 1) / "DONE").read_text())["skipped"]
    assert phases[0]["init"] is None
    progress = (tmp_path / "progress").read_text()
    assert "phase 0 done: rounds 0-0, not trained: 2 games played, training starts at 5" in progress
    assert "phase 1 done: rounds 0-1, not trained: 4 games played, training starts at 5" in progress
    assert "phase 2 started: rounds 0-2" in progress


def test_the_recent_dose_trains_the_last_rounds_with_as_many_earlier_labels_from_whole_games_under_the_stop(
        tmp_path, monkeypatch):
    phases = []
    _fake_training(monkeypatch, phases)
    monkeypatch.setattr(realtime, "RECENT_ROUNDS", 2)
    # each round's training side: two games of 3 labels each; round 0 holds a game out
    _labelled(monkeypatch, tmp_path, [(2, 1), (2, 0), (2, 0), (2, 0), (2, 0)], calls=3)
    for r in range(5):
        _played(tmp_path, r, 3)
    a = _trainer_args(tmp_path, 5)
    a.dose, a.start_after, a.replay_ratio = "recent", 4, 0.5
    _train(monkeypatch, a)
    last = phases[-1]   # phase 4: rounds 3 and 4, and half as many earlier labels as two whole games of rounds 0-2
    recent = [c for c in last["calls"] if c.meta["run"] in ("train-r0003", "train-r0004")]
    earlier = [c for c in last["calls"] if c.meta["run"] not in ("train-r0003", "train-r0004")]
    assert len(recent) == 12 and len(earlier) == 6
    assert len({(c.meta["run"], c.meta["user"]) for c in earlier}) == 2
    assert last["epochs"] == a.max_epochs and _runs(last["held"]) == ["dev-r0000"] * 3   # under the held-out stop


@pytest.mark.parametrize("dose", ["aggregate", "recent"])
def test_opsd_trains_daggers_calls_held_out_games_passes_and_stop_with_its_own_update_alone(tmp_path, monkeypatch,
                                                                                             dose):
    """OPSD's phase is DAgger's with SDFT's loss: the same label sets give it, phase by phase and under either dose,
    exactly DAgger's calls, held-out games (scored as they were played: realtime.plays), passes, seed, cap and stop, from
    the same phase's adapter; only the update differs (updates.SelfDistillation, its first --skip-first ids ungraded)."""
    phases, lines = {}, {}
    for row in ("dagger", "opsd"):
        phases[row] = []
        _fake_training(monkeypatch, phases[row])
        run = tmp_path / row
        _labelled(monkeypatch, run, [(2, 1), (2, 0), (2, 1), (2, 0), (2, 0)], calls=3)
        for r in range(5):
            _played(run, r, 3)
        a = _trainer_args(run, 5, row=row)
        a.dose, a.start_after = dose, 4
        _train(monkeypatch, a)
        lines[row] = [line.split(" realtime: ")[1] for line in (run / "progress").read_text().splitlines()]
    dagger, opsd = phases["dagger"], phases["opsd"]
    assert len(dagger) == len(opsd) == 4
    assert [p["kind"] for p in dagger] == ["sft"] * 4 and [p["kind"] for p in opsd] == ["sdft"] * 4
    for d, o in zip(dagger, opsd, strict=True):
        assert [c.meta for c in o["calls"]] == [c.meta for c in d["calls"]]
        assert o["held"] == [f"{c.meta['run']}'s play" for c in d["held"]]
        assert {k: o[k] for k in ("epochs", "seed", "eval_steps", "patience", "max_steps", "skip_first")} == \
               {k: d[k] for k in ("epochs", "seed", "eval_steps", "patience", "max_steps", "skip_first")}
        assert (o["init"] and o["init"].relative_to(tmp_path / "opsd")) == \
               (d["init"] and d["init"].relative_to(tmp_path / "dagger"))
    assert opsd[0]["skip_first"] == 3 and any(p["held"] for p in opsd)
    started = [[line for line in lines[row] if " started: " in line] for row in ("dagger", "opsd")]
    assert started[0] == started[1] and len(started[0]) == 4   # the same calls and held-out games, line for line
    assert "phase 4 done: rounds 0-4, 64 replies drawn, 1 steps in 0.1 min, held-out 0.5 to 0.4 (best at step 5)" in \
           lines["opsd"]


def _pg_fakes(monkeypatch, trainer, behaviour: list, won, users: int = 2, held_out=None):
    """PG's items and update faked over rounds whose games ``won`` names; the behaviour adapter each round's ratios are
    taken under recorded in ``behaviour``."""
    built = []
    monkeypatch.setattr(realtime, "load_calls", lambda path: ([types.SimpleNamespace(
        meta={"run": path.name, "user": u, "i": i, "round": int(path.name[1:]),
              "solved": (int(path.name[1:]), u) in won}) for u in range(users) for i in range(2)], 0))
    trainer.configure_behaviour = lambda adapter: behaviour.append("base" if adapter is None else adapter)
    monkeypatch.setattr(updates, "played_items", lambda tok, calls, max_len: calls)
    monkeypatch.setattr(updates, "behaviour_logps", lambda tr, a, calls: {
        (c.meta["run"], c.meta["user"], c.meta["i"]): [float(len(behaviour))] for c in calls})

    def games(calls):
        return sorted({(c.meta["run"], c.meta["user"]) for c in calls})

    def pg_items(tok, calls, max_len):
        built.append(games(calls))
        wins = [c for c in calls if c.meta["solved"]]
        return ([types.SimpleNamespace(label=c.meta, n=1, behaviour_logp=None) for c in wins],
                {"episodes": len(games(calls)), "wins": len(games(wins))})

    monkeypatch.setattr(updates, "pg_items", pg_items)
    monkeypatch.setattr(updates, "PolicyGradient", lambda a, items, stats, held: types.SimpleNamespace(
        kind="pg", calls=items, units=items, held=held, stats=stats,
        done={"steps": 1, "minutes": 0.1, "games": stats, "held_out": [0.3, 0.2] if held else [],
              "best_step": 5 if held else None}))
    return built


def test_pg_trains_on_every_won_game_so_far_holding_out_daggers_games_against_the_adapter_that_played_each(
        tmp_path, monkeypatch):
    """PG's phase is DAgger's with PG's loss: every round's games so far but those labeller.held_game holds out, of
    which the won games train (updates.pg_items), each call's ratios against its reply's log-probabilities under the
    adapter that played its round, taken once in the phase that first trains the round (the base model's, then phase
    0's for round 2 under lag 1), for won games alone, and read back from the round's pg_behaviour.json.gz in every
    later phase; a round with no win needs none. The held-out games' wins go to the held-out stop once a held-out game
    has won; a held-out failure, which PG's loss weighs 0, starts no stop."""
    phases, behaviour = [], []
    trainer = _fake_training(monkeypatch, phases)
    monkeypatch.setattr(server, "adapter_is_untrained", lambda d: (Path(d) / "untrained").exists())
    # each round: two users' games of 2 calls, user 0 winning rounds 0 and 2; hold_every 3 holds out
    # (round + user) % 3 == 2: round 1's user 1 (failed) and round 2's user 0 (won)
    built = _pg_fakes(monkeypatch, trainer, behaviour, {(0, 0), (2, 0)})
    for r, played in ((0, None), (1, None), (2, 0)):   # lag 1: round 2 is phase 0's
        rd = realtime.round_dir(tmp_path, r)
        rd.mkdir(parents=True)
        (rd / "play.done").touch()
        realtime.write_json(rd / "served.json", {"phase": played})
    a = _trainer_args(tmp_path, 3, row="pg")
    a.hold_every = 3
    _train(monkeypatch, a)

    def trained(items):
        return sorted({(i.label["run"], i.label["user"]) for i in items})

    first, second, third = phases
    assert built[0] == [("r0000", 0), ("r0000", 1)] and trained(first["calls"]) == [("r0000", 0)]
    assert first["held"] == [] and first["epochs"] == 1
    # round 1's held-out game failed: no held-out stop yet, one pass
    assert built[2] == [("r0000", 0), ("r0000", 1), ("r0001", 0)] and second["held"] == []
    assert second["epochs"] == 1
    assert built[4] == [("r0000", 0), ("r0000", 1), ("r0001", 0), ("r0002", 1)]
    assert trained(third["calls"]) == [("r0000", 0)] and trained(third["held"]) == [("r0002", 0)]
    assert third["epochs"] == 20 and third["max_steps"] == 120 and third["eval_steps"] == 5 and third["patience"] == 2
    assert third["init"] == realtime.phase_dir(tmp_path, 1) / "adapter"   # training goes on from the newest
    # each round's won calls' log-probabilities taken once, under the adapter that played it, and read back after
    assert behaviour == ["base", realtime.phase_dir(tmp_path, 0) / "adapter"]
    assert {i.label["run"]: i.behaviour_logp for i in third["calls"] + third["held"]} == {
        "r0000": [1.0], "r0002": [2.0]}
    assert [(realtime.round_dir(tmp_path, r) / "pg_behaviour.json.gz").exists() for r in range(3)] == [True, False,
                                                                                                         True]
    progress = (tmp_path / "progress").read_text()
    assert ("phase 1 started: rounds 0-1, 1 of 3 games won, 2 calls, 1 held-out games (0 won), 1 pass(es), "
            "no held-out stop\n") in progress
    assert "phase 2 started: rounds 0-2, 1 of 4 games won, 2 calls, 2 held-out games (1 won), 20 pass(es)\n" in progress
    assert "phase 2 done: rounds 0-2, 1/4 games won and trained on, 1 steps in 0.1 min, held-out 0.3 to 0.2 " \
           "(best at step 5)" in progress


def test_pg_takes_a_warm_up_rounds_ratios_against_the_base_model_though_prune_deleted_the_adapter_that_played_it(
        tmp_path, monkeypatch):
    """A round won in the warm-up is first trained in the first phase that trains, by when prune has deleted the
    untrained adapter that played it: its player trained nothing, so its ratios are taken against the base model,
    without reading that adapter. A round played by a phase that trained still takes its ratios against that phase's
    adapter."""
    phases, behaviour = [], []
    trainer = _fake_training(monkeypatch, phases)

    def adapter_is_untrained(d):   # the real one reads the adapter's weights: a pruned adapter is a missing file
        if not Path(d).is_dir():
            raise FileNotFoundError(Path(d) / "adapter_model.safetensors")
        return (Path(d) / "untrained").exists()

    monkeypatch.setattr(server, "adapter_is_untrained", adapter_is_untrained)
    # one user, one game a round, two calls a game; rounds 3, 5 and 6 won. start_after 5: phases 0-3 train nothing
    # (1 to 4 games played), phase 4 is the first that trains. Lag 1: round r is played by phase r-2, so round 3 by
    # phase 1 (pruned when phase 3 ends), round 5 by phase 3 (warm-up, still on disk) and round 6 by phase 4 (trained)
    _pg_fakes(monkeypatch, trainer, behaviour, {(3, 0), (5, 0), (6, 0)}, users=1)
    for r in range(7):
        _played(tmp_path, r, 1)
        (realtime.round_dir(tmp_path, r) / "play.done").touch()
        realtime.write_json(realtime.round_dir(tmp_path, r) / "served.json", {"phase": r - 2 if r >= 2 else None})
    a = _trainer_args(tmp_path, 7, row="pg")
    a.start_after = 5
    _train(monkeypatch, a)

    assert len(phases) == 3   # phases 4, 5 and 6 train
    assert not (realtime.phase_dir(tmp_path, 1) / "adapter").exists()   # pruned before phase 4 began
    assert behaviour == ["base", "base", realtime.phase_dir(tmp_path, 4) / "adapter"]   # rounds 3, 5 and 6
    assert {i.label["run"]: i.behaviour_logp for i in phases[2]["calls"]} == {"r0003": [1.0], "r0005": [2.0],
                                                                                "r0006": [3.0]}
    assert phases[0]["init"] is None   # the warm-up's phases saved no adapter: the base model
    assert "phase 4 done: rounds 0-4, 1/5 games won and trained on" in (tmp_path / "progress").read_text()


def test_a_phase_cut_short_goes_on_in_its_folder_and_one_past_its_done_is_not_trained_again(tmp_path, monkeypatch):
    """The trainer's restart: phase 0 trained to its DONE before the stop only has its rounds.json written; phase 1, cut
    short after its checkpoint at step 5, trains in its folder as it stands (phase.fit resumes from the checkpoint);
    phase 2, cut short before any checkpoint, trains again from an empty folder."""
    phases = []
    _fake_training(monkeypatch, phases)
    _labelled(monkeypatch, tmp_path, [(1, 0)] * 3, calls=1)
    p0, p1, p2 = (realtime.phase_dir(tmp_path, k) for k in range(3))
    (p0 / "adapter").mkdir(parents=True)
    (p0 / "DONE").write_text(json.dumps({"steps": 3, "minutes": 0.1, "held_out": [], "best_step": None}))
    (p1 / "checkpoints").mkdir(parents=True)
    (p1 / "checkpoints" / "state.json").write_text(json.dumps({"step": 5}))
    p2.mkdir(parents=True)
    (p2 / "phase.json").write_text("{}")   # a phase cut short before its first checkpoint
    _train(monkeypatch, _trainer_args(tmp_path, 3))
    assert [_runs(p["calls"]) for p in phases] == [["train-r0000", "train-r0001"],
                                                    ["train-r0000", "train-r0001", "train-r0002"]]
    assert [p["checkpoint"] for p in phases] == [5, None] and not (p2 / "phase.json").exists()
    assert all(realtime.finished(tmp_path, k) for k in range(3))
    assert "phase 1 goes on from its checkpoint at step 5" in (tmp_path / "progress").read_text()


def test_sigterm_while_the_trainer_waits_for_a_round_ends_it_with_nothing_to_save(tmp_path, monkeypatch):
    _fake_training(monkeypatch, [])

    def leave(step):
        raise SystemExit(f"left at {step}")

    monkeypatch.setattr(checkpoints, "leave", leave)
    monkeypatch.setattr(checkpoints, "stopping", lambda: True)
    with pytest.raises(SystemExit, match="left at None"):
        _train(monkeypatch, _trainer_args(tmp_path, 3))   # round 0 not labelled yet


def test_the_trainer_refuses_a_latest_phase_that_is_not_rounds_minus_one(tmp_path, monkeypatch):
    _fake_training(monkeypatch, [])
    (tmp_path / "train").mkdir()
    realtime.write_json(tmp_path / "train" / "latest.json", {"phase": 4, "adapter": "a", "rounds": 1})
    with pytest.raises(SystemExit, match="one round a phase"):
        _train(monkeypatch, _trainer_args(tmp_path, 3))


def test_a_held_out_label_is_scored_as_its_own_play_carrying_the_labels_demonstration(tmp_path):
    """OPSD scores its held-out games as they were played: realtime.plays takes each held-out label (a label set's
    call: its side's round, and from its call_id the play's user and the call's place in that game) to the play's own
    call, its served prompt and played reply, carrying the label's demonstration; a label whose play came back without
    its token ids has no played reply to score and is left out."""
    rd = realtime.round_dir(tmp_path, 6)
    (rd / "captured").mkdir(parents=True)
    exchanges = [{"call_id": f"id{i}", "request": {"messages": [{"role": "user", "content": f"move {i}"}]},
                  "response": {"prompt_token_ids": [1, 2, i], "choices": [{"token_ids": [7, 8, i],
                                                                            "finish_reason": "stop"}]}}
                 for i in range(3)]
    exchanges[2]["response"] = {"prompt_token_ids": [1, 2, 3], "choices": [{"finish_reason": "stop",
                                                                             "message": {"content": ""}}]}
    with gzip.open(rd / "captured" / "u002.json.gz", "wt") as fp:
        json.dump({"user": 2, "task_id": "t", "calls": exchanges}, fp)
    # the labels of user 2's calls 1 and 2, renumbered user 0 on round 6's held-out side, as the label set writes them
    labels = [Call({"messages": []}, [5, 6, 9, 10], [11, 12, 2], "stop",
                   {"run": "dev-r0006", "user": 0, "i": k, "call_id": f"{rd}-u002-c000{i}"})
              for k, i in enumerate((1, 2))]
    tok = types.SimpleNamespace(decode=lambda ids: "".join({10: "<think>\n"}.get(i, f"<{i}>") for i in ids))
    [own] = realtime.plays(tmp_path, labels, tok)
    assert (own.meta["run"], own.meta["user"], own.meta["i"]) == ("r0006", 2, 1)
    assert own.prompt_ids == [1, 2, 1] and own.reply_ids == [7, 8, 1]
    assert own.meta["demo"] == updates.SDFT_DEMO.format(demo="<think>\n<11><12>")


# -- the player --------------------------------------------------------------------------------------------------

def _player_args(run: Path, rounds: int, lag: int = 1) -> argparse.Namespace:
    return argparse.Namespace(run=str(run), rounds=rounds, row="dagger", users=2, name="dagger-u2-lag1", lag=lag,
                              serve_gpus="0,1,2,3", serve_dp=1)


def _fake_play(monkeypatch, served: list, during: dict | None = None):
    """play_round and serve faked: every round records its two games; ``during[r]``, if any, runs while round r plays
    (the trainer finishing a phase then)."""

    def serve(a, run_, phase_, current):
        name = realtime.adapter_name(a.name, phase_["phase"]) if phase_ else play.MODEL
        served.append(name)
        return name

    def play_round(row, played, schedule, *, base_url, served_name):
        played.mkdir(parents=True, exist_ok=True)
        with open(played / "records.jsonl", "a") as fp:
            for _ in range(2):
                fp.write("{}\n")
        (during or {}).get(schedule.round, lambda: None)()

    monkeypatch.setattr(realtime, "serve", serve)
    monkeypatch.setattr(play, "play_round", play_round)
    monkeypatch.setattr(play, "schedule_cut", lambda r, users: types.SimpleNamespace(round=r, slots=[0] * 2 * (r + 1)))


def test_the_player_plays_round_r_on_the_phase_trained_on_rounds_up_to_r_minus_2_never_a_newer_one(tmp_path,
                                                                                                monkeypatch):
    served = []
    # the trainer finishes phase 0 while round 1 plays, and phases 1 and 2 while round 2 plays (ahead of play)
    _fake_play(monkeypatch, served, {1: lambda: _finish_phase(tmp_path, 0),
                                     2: lambda: (_finish_phase(tmp_path, 1), _finish_phase(tmp_path, 2))})
    realtime.cmd_player(_player_args(tmp_path, 4))
    assert served == [play.MODEL, play.MODEL, "dagger-u2-lag1-p0000", "dagger-u2-lag1-p0001"]
    s3 = json.loads((realtime.round_dir(tmp_path, 3) / "served.json").read_text())
    assert s3["model"] == "dagger-u2-lag1-p0001" and s3["phase"] == 1 and s3["trained_rounds"] == 2 and s3["lag"] == 1
    s1 = json.loads((realtime.round_dir(tmp_path, 1) / "served.json").read_text())
    assert s1["phase"] is None and s1["trained_rounds"] == 0
    assert all((realtime.round_dir(tmp_path, r) / "play.done").exists() for r in range(4))
    assert (tmp_path / "player.done").exists()


def test_the_player_waits_for_its_phase_when_training_is_behind(tmp_path, monkeypatch):
    served, waits = [], []
    _fake_play(monkeypatch, served)

    def sleep(seconds):   # the trainer finishes phase 0 while round 2 waits for it
        waits.append(seconds)
        _finish_phase(tmp_path, 0)

    monkeypatch.setattr(realtime.time, "sleep", sleep)
    realtime.cmd_player(_player_args(tmp_path, 3))
    assert served[-1] == "dagger-u2-lag1-p0000" and len(waits) == 1
    assert "play round 2 waits for phase 0" in (tmp_path / "progress").read_text()


def test_the_player_with_lag_0_plays_round_r_on_phase_r_minus_1(tmp_path, monkeypatch):
    served = []
    _fake_play(monkeypatch, served, {0: lambda: _finish_phase(tmp_path, 0), 1: lambda: _finish_phase(tmp_path, 1)})
    realtime.cmd_player(_player_args(tmp_path, 3, lag=0))
    assert served == [play.MODEL, "dagger-u2-lag1-p0000", "dagger-u2-lag1-p0001"]


def test_the_player_fails_after_three_short_plays(tmp_path, monkeypatch):
    monkeypatch.setattr(realtime, "serve", lambda a, run, phase_, current: "base")
    monkeypatch.setattr(play, "play_round", lambda *args, **kw: None)   # never records anything
    monkeypatch.setattr(play, "schedule_cut", lambda r, users: types.SimpleNamespace(slots=[0, 0]))
    with pytest.raises(SystemExit):
        realtime.cmd_player(_player_args(tmp_path, 1))
    assert "failed after 3 attempts" in (tmp_path / "progress").read_text()


class _Server:
    """The model server as serve() sees it: what it serves, what it unloads, and the check's answers."""

    def __init__(self, served, differs):
        self.served, self.unloaded, self.differs = served, [], differs

    def models(self):
        return list(self.served)

    def unload_adapter(self, name):
        self.unloaded.append(name)


def test_serving_a_phase_loads_its_adapter_checks_it_and_unloads_the_runs_others(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(realtime.subprocess, "run", lambda argv, check: calls.append(argv[2:]))
    s = _Server(["Qwen/Qwen3.5-9B", "dagger-u2-lag1-p0000", "dagger-u2-lag1-p0001", "pg-u1-lag1-p0007"], True)
    monkeypatch.setattr(server, "Server", lambda url: s)
    monkeypatch.setattr(server, "adapter_is_untrained", lambda d: False)
    monkeypatch.setattr(server, "check_served_adapter",
                        lambda srv, base, name: {"answered": True, "differs": srv.differs})
    a = _player_args(tmp_path, 3)
    assert realtime.serve(a, tmp_path, None, "") == play.MODEL and calls == [["0,1,2,3", "1"]]
    adapter = str(tmp_path / "train" / "p0001" / "adapter")
    name = realtime.serve(a, tmp_path, {"phase": 1, "adapter": adapter}, play.MODEL)
    assert name == "dagger-u2-lag1-p0001" and calls[-1] == ["0,1,2,3", "1", name, adapter]
    assert s.unloaded == ["dagger-u2-lag1-p0000"]   # another run's adapters stay
    assert realtime.serve(a, tmp_path, {"phase": 1, "adapter": adapter}, name) == name and len(calls) == 2
    s.differs = False
    with pytest.raises(SystemExit):
        realtime.serve(a, tmp_path, {"phase": 2, "adapter": adapter}, name)
    assert s.unloaded[-1] == "dagger-u2-lag1-p0002"
    assert "phase 2's adapter failed the served-adapter check" in (tmp_path / "progress").read_text()


# -- the labeller ------------------------------------------------------------------------------------------------

def _labeller_args(run: Path, row: str = "dagger", rounds: int = 1,
                   oracle_rules: str = "library") -> argparse.Namespace:
    return argparse.Namespace(run=str(run), rounds=rounds, row=row, c1=str(run / "c1.md"), oracle_rules=oracle_rules,
                              label_model="z-ai/glm-5.3-flash", label_effort="medium", relay_port=18230,
                              label_workers=256, label_set_workers=4, need_thinking=4, label_round_budget=20.0,
                              label_run_budget=100.0, min_labelled_pct=40.0, hold_every=10)


def _fake_oracle(monkeypatch, asked: list, spent: float = 0.0):
    """The oracle and the labels faked: ``asked`` gets, per labelling pass, the users of the points it was given, its
    minimum labelled share and the working rules it labelled under."""
    monkeypatch.setattr(labeller, "spent", lambda dirs: spent)
    monkeypatch.setattr(labeller, "points", lambda plays: [{"play": p.name, "user": int(f.name[1:4]), "i": 0}
                                                           for p in plays for f in sorted((p / "captured").glob("u*"))])
    monkeypatch.setattr(oracle, "Oracle", lambda base_url, model, effort: types.SimpleNamespace(ask=None))

    def label(out, ask, rules, *, budget, workers, need_thinking, min_labelled_pct):
        pts = [json.loads(line) for line in open(out / "points.jsonl")]
        asked.append(([p["user"] for p in pts], min_labelled_pct, rules))

    monkeypatch.setattr(labeller, "label", label)

    def label_set(out, model, workers):
        (out / "set").mkdir()
        return {"examples": 1}

    def split(set_dir, out, round_, hold_every, sides):
        out.mkdir(parents=True)
        (out / "split.json").write_text(json.dumps({side: {"games": 1} for side in sides}))

    monkeypatch.setattr(labeller, "label_set", label_set)
    monkeypatch.setattr(labeller, "split", split)


def _land(run: Path, r: int, user: int) -> None:
    captured = realtime.round_dir(run, r) / "captured"
    captured.mkdir(parents=True, exist_ok=True)
    (captured / f"u{user:03d}.json.gz").write_text("")


def test_a_label_pass_asks_about_each_game_once_as_it_lands_then_the_whole_round_once_more(tmp_path, monkeypatch):
    asked = []
    _fake_oracle(monkeypatch, asked)
    a = _labeller_args(tmp_path)
    _land(tmp_path, 0, 0)
    games = realtime.label_pass(a, tmp_path, 0, "R", set(), final=False)
    _land(tmp_path, 0, 1)
    games = realtime.label_pass(a, tmp_path, 0, "R", games, final=False)
    realtime.label_pass(a, tmp_path, 0, "R", games, final=True)
    assert asked == [([0], 0, "R"), ([1], 0, "R"), ([0, 1], 40.0, "R")]   # the final pass alone holds the minimum
    points = [json.loads(line) for line in open(realtime.labels_dir(tmp_path, 0) / "points.jsonl")]
    assert [p["user"] for p in points] == [0, 1]   # the label set reads every game's points


def test_a_label_pass_refuses_once_the_run_has_spent_its_budget(tmp_path, monkeypatch):
    asked = []
    _fake_oracle(monkeypatch, asked, spent=100.0)
    _land(tmp_path, 0, 0)
    with pytest.raises(SystemExit):
        realtime.label_pass(_labeller_args(tmp_path), tmp_path, 0, "R", set(), final=True)
    assert asked == [] and "over 100.0" in (tmp_path / "progress").read_text()


def test_finishing_a_round_splits_its_label_set_under_the_rounds_own_names(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(labeller, "label_set", lambda out, model, workers: seen.update(set=(out, model)) or {})

    def split(set_dir, out, round_, hold_every, sides):
        seen["split"] = (set_dir, hold_every, round_, sides)
        out.mkdir(parents=True)
        (out / "split.json").write_text(json.dumps({"train-r0007": {"games": 1}, "dev-r0007": {"games": 0}}))

    monkeypatch.setattr(labeller, "split", split)
    done = realtime.finish_round(_labeller_args(tmp_path), tmp_path, 7)
    assert seen["set"] == (realtime.labels_dir(tmp_path, 7), play.MODEL)
    assert seen["split"] == (realtime.labels_dir(tmp_path, 7) / "set", 10, 7, ("train-r0007", "dev-r0007"))
    assert done["train-r0007"]["games"] == 1
    assert realtime.ready_rounds(tmp_path, 8, "dagger") == 0   # round 0 first: round 7 alone is not a run of rounds


def test_the_oracles_working_rules_are_c1_and_the_program_librarys_procedure_or_c1_alone():
    """DAgger's and OPSD's labels are written under c1 and the program library's procedure (row HPL's text), or, with
    ORACLE_RULES=c1, under c1 alone, so that the procedure is the one difference between the two oracles."""
    from continual_arc_baselines.systems.protocol import PROGRAM_LIBRARY_RULES
    c1 = "1. End every reply with one action line.\n"
    assert oracle.oracle_rules(c1, "library") == c1.strip() + "\n\n" + PROGRAM_LIBRARY_RULES.format(programs="programs")
    assert oracle.oracle_rules(c1, "c1") == c1.strip()


def _label_round(monkeypatch, run: Path, r: int, a: argparse.Namespace, asked: list) -> None:
    """Round ``r`` played and labelled by the labeller, the oracle faked."""
    _fake_oracle(monkeypatch, asked)
    _land(run, r, 0)
    (realtime.round_dir(run, r) / "play.done").touch()
    realtime.cmd_labeller(a)


@pytest.mark.parametrize("row", ["dagger", "opsd"])
def test_the_labeller_tells_the_oracle_its_rules_and_builds_daggers_split_for_both_labelled_rows(tmp_path, monkeypatch,
                                                                                                  row):
    """DAgger's and OPSD's labellers are one: the round's labels under the working rules, kept with the run, then its
    label set and split, which mark it labelled, so OPSD trains on DAgger's own label sets."""
    told = {}
    for rules in ("library", "c1"):
        run = tmp_path / rules
        run.mkdir()
        (run / "c1.md").write_text("1. End every reply with one action line.\n")
        told[rules] = []
        _label_round(monkeypatch, run, 0, _labeller_args(run, row, oracle_rules=rules), told[rules])
        assert (run / "guidelines.md").read_text() == told[rules][0][2]   # what the oracle read, kept with the run
        assert realtime.ready_rounds(run, 1, row) == 1 and (run / "labeller.done").exists()
        assert 'round 0 labelled: {"train-r0000":{"games":1},"dev-r0000":{"games":1}}' in \
            (run / "progress").read_text()
    assert told["library"][0][2] == oracle.oracle_rules("1. End every reply with one action line.\n", "library")
    assert told["c1"] == [([0], 40.0, "1. End every reply with one action line.")]


def test_a_labeller_started_again_under_other_oracle_rules_stops_before_it_labels(tmp_path, monkeypatch):
    """A run keeps its oracle's rules to the end: a relaunch that lost ORACLE_RULES=c1 would tell the oracle the
    procedure from its next round on, so its labeller stops there; started again under its own rules, it goes on."""
    (tmp_path / "c1.md").write_text("c1")
    told = []
    _label_round(monkeypatch, tmp_path, 0, _labeller_args(tmp_path, rounds=1, oracle_rules="c1"), told)
    with pytest.raises(SystemExit):
        _label_round(monkeypatch, tmp_path, 1, _labeller_args(tmp_path, rounds=2, oracle_rules="library"), told)
    assert [t[2] for t in told] == ["c1"] and (tmp_path / "guidelines.md").read_text() == "c1"
    assert "the labeller stops: guidelines.md" in (tmp_path / "progress").read_text()
    realtime.cmd_labeller(_labeller_args(tmp_path, rounds=2, oracle_rules="c1"))
    assert [t[2] for t in told] == ["c1", "c1"] and realtime.ready_rounds(tmp_path, 2, "dagger") == 2


# -- realtime.sh -------------------------------------------------------------------------------------------------

SETTINGS = {"USERS": "4", "LR": "1e-4", "SERVE_GPUS": "0,1", "TRAIN_GPUS": "2,3,4,5,6,7", "START_AFTER": "24"}


def _realtime_sh(env: dict[str, str], then: str, home: Path = Path("/home/u")) -> subprocess.CompletedProcess:
    """realtime.sh's settings, say(), labels() and launch() lifted out and run with ``env`` and HOME ``home``, then
    ``then``: the "started:" line (``started``, which say() prints, so nothing is started) or a command."""
    script = (SCRIPTS / "realtime.sh").read_text()
    settings = script[script.index("\n", script.index('source "$(dirname')) + 1:script.index("\nL=$RUNS/rt/")]
    labels = next(line for line in script.splitlines() if line.startswith("labels() {"))
    launch = script[script.index("launch() {"):script.index("\n}\n", script.index("launch() {")) + 2]
    started = next(line.strip() for line in script.splitlines() if line.strip().startswith('say "started: '))
    body = (f'source {SCRIPTS}/lib.sh\nsay() {{ echo "$*"; }}\n{labels}\n{launch}\n{settings}\nL={home}/run\n'
            f'{started if then == "started" else then}')
    return subprocess.run(["bash", "-c", body], env={"PATH": os.environ["PATH"], "HOME": str(home), **env},
                          capture_output=True, text=True)


def test_realtime_sh_names_the_oracles_rules_in_its_started_line_for_the_labelled_rows_alone():
    tail = "batch 64, start after 24 games, dose aggregate (replay 1)"
    assert _realtime_sh({**SETTINGS, "ROW": "dagger"}, "started").stdout.endswith(
        f"{tail}, oracle rules library, patience 2\n")
    assert _realtime_sh({**SETTINGS, "ROW": "opsd", "ORACLE_RULES": "c1"}, "started").stdout == (
        "started: opsd, 4 users, 200 rounds, lag 1, LR 1e-4, serving on 0,1, training on 2,3,4,5,6,7, "
        f"{tail}, oracle rules c1, patience 2\n")
    assert _realtime_sh({**SETTINGS, "ROW": "pg", "ORACLE_RULES": "c1"}, "started").stdout.endswith(
        f"{tail}, patience 2\n")
    refused = _realtime_sh({**SETTINGS, "ROW": "dagger", "ORACLE_RULES": "none"}, "started")
    assert refused.returncode == 1 and refused.stdout == "ORACLE_RULES must be library or c1, got none\n"


def test_a_runs_oracle_rules_are_read_back_from_its_last_started_line(tmp_path):
    """The rules a run's oracle labels under, read back from its progress: those realtime.sh's last "started:" line
    names (the driver's own "phase N started:" lines are not realtime.sh's), or the default for a line that names
    none."""
    before = ("10-20 01:16 realtime.sh: started: dagger, 1 users, 200 rounds, lag 1, LR 1e-4, serving on 0,1,2,3, "
              "training on 4,5, batch 64, start after 0 games, dose aggregate (replay 1)\n")
    phase_line = ("10-20 01:30 realtime: phase 1 started: rounds 0-1, 134 labels, 0 held-out games, 1 pass(es), "
                  "no held-out stop\n")
    (tmp_path / "progress").write_text(before + phase_line)
    assert realtime.started_oracle_rules(tmp_path) == "library"
    for env, rules in (({"ROW": "opsd", "ORACLE_RULES": "c1"}, "c1"), ({"ROW": "dagger"}, "library")):
        started = _realtime_sh({**SETTINGS, **env}, "started").stdout
        (tmp_path / "progress").write_text(f"{before}10-20 21:30 realtime.sh: {started}{phase_line}")
        assert realtime.started_oracle_rules(tmp_path) == rules


@pytest.mark.parametrize("loop", ["player", "labeller", "trainer"])
def test_realtime_sh_launches_each_loop_with_its_settings(tmp_path, monkeypatch, loop):
    """The arguments launch() gives the driver (uv and setsid stand-ins recording them) read by the driver's own
    parser, ORACLE_RULES reaching the labeller."""
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "setsid").write_text('#!/bin/sh\nexec "$@"\n')
    (tools / "uv").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {tmp_path}/argv.part && mv {tmp_path}/argv.part '
                              f'{tmp_path}/argv\n')
    for tool in tools.iterdir():
        tool.chmod(0o755)
    (tmp_path / "prime-rl").mkdir()
    (tmp_path / "run").mkdir()
    env = {**SETTINGS, "ROW": "opsd", "ORACLE_RULES": "c1", "PATH": f"{tools}:{os.environ['PATH']}"}
    assert _realtime_sh(env, f"launch {loop}", tmp_path).returncode == 0
    for _ in range(100):   # launch() leaves the driver in the background
        if (tmp_path / "argv").exists():
            break
        time.sleep(0.05)
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[:6] == ["run", "--no-sync", "python", "-m", "continual_arc_training.realtime", loop]
    seen = {}
    monkeypatch.setattr(realtime, f"cmd_{loop}", lambda a: seen.update(vars(a)))
    a = realtime.parser().parse_args(argv[5:])
    a.fn(a)
    assert (seen["row"], seen["run"], seen["rounds"]) == ("opsd", f"{tmp_path}/run", 200)
    if loop == "labeller":
        assert seen["oracle_rules"] == "c1" and seen["c1"] == str(oracle.RULES)
    if loop == "trainer":
        assert (seen["train_gpus"], seen["lr"], seen["start_after"], seen["dose"]) == ("2,3,4,5,6,7", 1e-4, 24,
                                                                                       "aggregate")
    if loop == "player":
        assert (seen["users"], seen["name"], seen["serve_gpus"]) == (4, "opsd-u4-lag1", "0,1")


# -- pruning -----------------------------------------------------------------------------------------------------

def _phases(root: Path, n: int) -> None:
    for k in range(n):
        a = root / f"p{k:04d}" / "adapter"
        a.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "adapter_config.json", *realtime.RESUME_FILES):
            (a / name).write_text("x")
        (root / f"p{k:04d}" / "adapter.vllm").mkdir()
        (root / f"p{k:04d}" / "DONE").write_text("{}")


def test_prune_keeps_the_newest_lag_plus_one_phases_the_served_one_and_every_20th_phases_adapter(tmp_path):
    _phases(tmp_path, 45)
    realtime.prune(tmp_path, 44, 41, 1)   # the round in play started on phase 41
    whole = {"adapter_model.safetensors", "adapter_config.json", *realtime.RESUME_FILES}
    for k in (41, 43, 44):
        assert {f.name for f in (tmp_path / f"p{k:04d}" / "adapter").iterdir()} == whole
        assert (tmp_path / f"p{k:04d}" / "adapter.vllm").exists()
    for k in (0, 20, 40):
        kept = tmp_path / f"p{k:04d}"
        assert {f.name for f in (kept / "adapter").iterdir()} == {"adapter_model.safetensors", "adapter_config.json"}
        assert not (kept / "adapter.vllm").exists()
    assert not (tmp_path / "p0021" / "adapter").exists() and (tmp_path / "p0021" / "DONE").exists()
    assert not (tmp_path / "p0042" / "adapter").exists()


def test_prune_with_a_longer_lag_keeps_the_phase_pg_will_clip_against(tmp_path):
    _phases(tmp_path, 10)
    realtime.prune(tmp_path, 9, None, 2)
    assert (tmp_path / "p0007" / "adapter" / "trainer_adapter.safetensors").exists()   # phase 10's round's player
    assert not (tmp_path / "p0006" / "adapter").exists()


def test_serving_is_the_phase_of_the_newest_round_the_player_started(tmp_path):
    assert realtime.serving(tmp_path) is None
    for r, k in ((0, None), (1, 0), (2, 3)):
        realtime.round_dir(tmp_path, r).mkdir(parents=True)
        realtime.write_json(realtime.round_dir(tmp_path, r) / "served.json", {"phase": k})
    assert realtime.serving(tmp_path) == 3
