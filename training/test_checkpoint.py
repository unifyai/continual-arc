"""Tests for checkpoint.py and the training phases' resume from it (train.fit, with each row's update: DAgger's
train.Imitation, PG's train.PolicyGradient and OPSD's on-policy train.SelfDistillation), with a trainer whose whole
state is two numbers: every step mixes in its batch's calls and a
draw of the phase's random generator (and, on-policy, the weights of the adapter that drew its replies), so a phase
killed or told to stop by SIGTERM, then resumed, ends with the adapter an uninterrupted phase gives only if it took
the same batches, the same draws and the same drawing adapters in the same order. Run with
``python3 -m pytest test_checkpoint.py -q`` with continual_arc and ask-sd importable."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import types
from pathlib import Path

import pytest

import checkpoint

train = pytest.importorskip("train", reason="needs ask_sd (train.py imports it)")
Call = pytest.importorskip("ask_sd.calls").Call


class Killed(Exception):
    """The process gone mid-step: SIGKILL, or the machine stopped outright."""


class Left(Exception):
    """The process's end after SIGTERM's checkpoint (checkpoint.leave)."""


class Inline:
    """checkpoint's commit thread, run inline: a test's steps take no time, so a stop the test makes right after a save
    would otherwise race the save's commit, which a real one 15 s later would not (one test below takes the real
    thread)."""

    def __init__(self, **kwargs) -> None:
        pass

    def submit(self, fn) -> concurrent.futures.Future:
        done = concurrent.futures.Future()
        done.set_result(fn())
        return done

    def shutdown(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _sigterm(monkeypatch):
    def leave(step):
        raise Left(step)

    monkeypatch.setattr(checkpoint, "leave", leave)
    monkeypatch.setattr(checkpoint, "ThreadPoolExecutor", Inline)
    monkeypatch.setattr(checkpoint, "ON", True)
    monkeypatch.setattr(checkpoint, "EVERY", 5)   # the held-out cadence here, so a check is a checkpoint too
    checkpoint._TERM.clear()
    yield
    checkpoint._TERM.clear()


def _value(doc) -> float:
    return int(hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:8], 16) / 2 ** 32


def _labels(chains) -> list:
    return [[[x["run"], x["user"], x["i"]] for x in c.labels] for c in chains]


class Weights:
    """A trainer whose adapter is one weight and whose optimizer one moment, saved and loaded exactly. Each pass mixes
    its batch's calls and a draw of the generator into the step, which moves the moment and the weight; the held-out
    loss falls as the weight nears 0.6 and climbs past it, so a held-out stop comes some steps in. ``kill_at``: the
    pass after that many steps raises Killed; ``term_at``: SIGTERM comes during that step."""

    tokenizer = types.SimpleNamespace(eos_token_id=2)

    def __init__(self, kill_at: int | None = None, term_at: int | None = None) -> None:
        self.w, self.m, self.pending, self.taken = 0.0, 0.0, 0.0, 0
        self.w0 = 0.0   # the untrained adapter, this process's own (its random A matrices)
        self.kill_at, self.term_at = kill_at, term_at

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.json").write_text(json.dumps({"w": self.w, "m": self.m}))

    def load(self, path: Path) -> None:
        d = json.loads((path / "weights.json").read_text())
        self.w, self.m = d["w"], d["m"]

    def reset(self) -> None:   # as Trainer.reset: back to this process's untrained adapter
        self.w, self.m = self.w0, 0.0

    def __getattr__(self, name):   # configure_*: nothing to set here
        if name.startswith("configure_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def mix(self, doc, rng: random.Random) -> None:
        if self.taken == self.kill_at:
            raise Killed
        self.pending = _value(doc) + rng.random()

    def step(self) -> dict:
        self.m = 0.9 * self.m + self.pending
        self.w += 0.01 * self.m
        self.taken += 1
        if self.taken == self.term_at:
            checkpoint._TERM.set()
        return {"grad_norm": 0.5}

    def loss(self) -> float:
        return (self.w - 0.6) ** 2 + 0.1

    # sft
    def sft_chains(self, chains, rng):
        self.mix(_labels(chains), rng)
        return {"n_examples": sum(len(c.labels) for c in chains), "tokens": 1, "nll_per_token": 0.1, "loss_mean": 0.1,
                "entropy_mean": 0.1, "chains": {"chains": len(chains), "tokens_read": 1, "tokens_per_call": 1}}

    def nll_chains(self, chains):
        return {"items": [{"logp": -self.loss(), "tokens": 1}]}

    # pg
    def policy_gradient_chains(self, chains, rng):
        self.mix(_labels(chains), rng)
        return {"n_examples": 1, "tokens": 1, "skipped_long": 0, "logp": [-1.0], "loss_mean": 0.1, "ratio_mean": 1.0,
                "clip_fraction": 0.0, "dual_clip_fraction": 0.0, "approx_kl": 0.0, "entropy_mean": 1.0,
                "log_ratio_worst": None}

    def policy_gradient_scores_chains(self, chains):
        return {"items": [{"loss": self.loss()}]}

    # sdft: a step's replies as drawn (the fakes' sdft_chain_pairs hands over the items), so whose they are counts
    def distill_chains(self, students, teachers, rng, divergence, skip_first):
        self.mix([[c.meta["user"], c.meta["i"], c.meta.get("drawn_by")] for c in students], rng)
        return {"loss_mean": 0.1, "skipped_long": 0, "tokens": 1,
                "chains": {"tokens_read": 1, "teacher_tokens_read": 1, "tokens_per_call": 1}}

    def distill_scores_chains(self, students, teachers, divergence, skip_first):
        return {"items": [{"loss": self.loss()}]}


def _games(n_games: int, calls: int, run: str = "r0000") -> list[Call]:
    """``n_games`` games of ``calls`` calls, each prompt extending the last (one chain a game)."""
    out = []
    for user in range(n_games):
        prompt = [5, 6, 7 + user]
        for i in range(calls):
            reply = [100 + 10 * user + i, 200 + i]
            out.append(Call({}, list(prompt), reply, "stop", {"run": run, "user": user, "i": i, "solved": user % 3 == 0}))
            prompt = prompt + reply + [300 + i]
    return out


def _weights(out: Path) -> dict:
    return json.loads((out / "adapter" / "weights.json").read_text())


def _phase(out: Path) -> tuple[list, list]:
    """The steps' batch digests and the held-out curve a phase's phase.json records."""
    doc = json.loads((out / "phase.json").read_text())
    curve = doc.get("dev") or doc.get("held_out") or doc.get("curve") or []
    return [s["batch"] for s in doc["steps"]], [{k: v for k, v in p.items() if k != "minutes"} for p in curve]


def _same(base: Path, other: Path) -> None:
    """``other`` (stopped and resumed) took ``base``'s batches and ends on its adapter and curve, exactly."""
    assert _phase(other) == _phase(base)
    assert _weights(other) == _weights(base)
    done, ref = json.loads((other / "DONE").read_text()), json.loads((base / "DONE").read_text())
    assert done["steps"] == ref["steps"] and done["best_step"] == ref["best_step"]
    assert not (other / "checkpoints").exists()


# -- checkpoint.py on its own -------------------------------------------------------------------------------------

def _ck(out: Path, trainer, rng=None, fp="fp", state=None):
    return checkpoint.Checkpoints(out, trainer, rng or random.Random(0), fp, state or (lambda: {"n": 1}))


def test_a_save_cut_short_is_never_read_and_goes_at_the_next_start(tmp_path):
    t = Weights()
    ck = _ck(tmp_path, t)
    t.w = 1.5
    ck.save(5)
    (tmp_path / "checkpoints" / ".c0002").mkdir()   # a save the machine's stop cut short
    (tmp_path / "checkpoints" / ".c0002" / "weights.json").write_text("{")
    again = _ck(tmp_path, Weights())
    saved = again.load()
    assert saved["step"] == 5 and saved["dir"] == "c0001"
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["c0001", "state.json"]
    again.resume(saved)
    assert again.trainer.w == 1.5 and again.resumed == [5]


def test_a_checkpoint_is_synced_and_committed_on_its_own_thread_while_training_goes_on(tmp_path, monkeypatch):
    """save() returns once the trainer's files are written: state.json names the checkpoint only after its folder is
    synced, on the commit thread, which the next save, the phase's end and SIGTERM's checkpoint wait for."""
    import threading
    monkeypatch.setattr(checkpoint, "ThreadPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    release = threading.Event()
    synced = []

    def slow_sync(folder):   # the disk's 15 s for 3.7 GB
        release.wait(10)
        synced.append(folder.name)

    monkeypatch.setattr(checkpoint, "_sync", slow_sync)
    t = Weights()
    ck = _ck(tmp_path, t)
    ck.save(5)
    assert synced == [] and checkpoint.saved_step(tmp_path) is None   # nothing committed yet: training goes on
    release.set()
    ck.wait()
    assert synced == [".c0001"] and checkpoint.saved_step(tmp_path) == 5
    release.clear()
    ck.save(10)
    threading.Timer(0.2, release.set).start()
    ck.write_adapter(tmp_path / "adapter", None)   # the phase's end waits for the commit in flight
    assert checkpoint.saved_step(tmp_path) == 10 and synced[-1] == "adapter.tmp"
    ck.clear()


def test_another_phases_checkpoint_is_dropped(tmp_path):
    _ck(tmp_path, Weights(), fp="phase 3").save(5)
    assert _ck(tmp_path, Weights(), fp="phase 3 after a deploy").load() is None
    assert not (tmp_path / "checkpoints").exists()


def test_the_best_and_the_newest_are_kept_and_nothing_else(tmp_path):
    t = Weights()
    ck = _ck(tmp_path, t)
    for step, best in ((5, True), (10, False), (15, False), (20, True), (25, False)):
        t.w = step
        ck.save(step, best=best)
        held = sorted(p.name for p in (tmp_path / "checkpoints").iterdir() if p.is_dir())
        assert held == sorted({ck.dir, ck.best})
    assert (ck.best, ck.dir) == ("c0004", "c0005")
    ck.write_adapter(tmp_path / "adapter", None)
    assert _weights(tmp_path) == {"w": 20, "m": 0.0}
    a, b = tmp_path / "adapter" / "weights.json", tmp_path / "checkpoints" / "c0004" / "weights.json"
    assert os.stat(a).st_ino == os.stat(b).st_ino   # linked, not copied


def test_a_second_save_at_the_same_step_rewrites_the_state_alone(tmp_path):
    t = Weights()
    run = {"evals": []}
    ck = _ck(tmp_path, t, state=lambda: run)
    ck.save(10)
    run["evals"].append({"step": 10})
    ck.save(10, best=True)   # the held-out check of a resume's first step: the trainer is where it was saved
    saved = _ck(tmp_path, Weights(), state=lambda: run).load()
    assert (saved["dir"], saved["best"], saved["loop"]) == ("c0001", "c0001", {"evals": [{"step": 10}]})


def test_the_untrained_start_is_saved_when_it_is_best_and_there_is_no_init(tmp_path):
    t = Weights()
    t.w = 0.25   # this process's own random A matrices
    ck = _ck(tmp_path, t)
    ck.start_is_best(None)
    t.w = 9.0
    ck.write_adapter(tmp_path / "adapter", None)
    assert _weights(tmp_path)["w"] == 0.25
    init = tmp_path / "init"
    Weights().save(init)
    ck2 = _ck(tmp_path / "b", t)
    ck2.start_is_best(init)   # init's adapter stays on disk while the phase runs: nothing to save
    assert not (tmp_path / "b" / "checkpoints").exists()
    ck2.write_adapter(tmp_path / "b" / "adapter", init)
    assert _weights(tmp_path / "b") == {"w": 0.0, "m": 0.0}


# -- DAgger (train.fit with train.Imitation) ------------------------------------------------------------------------

def _sft_args(**kw):
    d = dict(objective=train.Objective.sft, max_seq_len=10 ** 6, val_run=["held-out"], val_max_items=0, seed=0,
             chain_slack=train.CHAIN_SLACK, chain_calls=2, batch=4, gpus=2, chains=True, epochs=20, max_steps=120, max_minutes=240,
             eval_steps=5, patience=2, divergence="forward_kl")
    d.update(kw)
    return argparse.Namespace(**d)


def _sft(monkeypatch, out: Path, trainer, a, init=None):
    out.mkdir(exist_ok=True)
    held = _games(2, 3, "dev") if a.val_run else []
    return train.fit(a, trainer, train.Imitation(a, trainer.tokenizer, _games(8, 6), held), out, init,
                     random.Random(3))


@pytest.mark.parametrize("stop", ["kill", "sigterm"])
def test_a_dagger_phase_stopped_mid_way_resumes_on_the_same_batches_and_ends_on_the_same_adapter(tmp_path, monkeypatch,
                                                                                               stop):
    a = _sft_args()
    base = _sft(monkeypatch, tmp_path / "base", Weights(), a)
    assert base["steps"] > 12 and base["best_step"] not in (None, 0)   # a held-out stop some steps in
    out = tmp_path / stop
    if stop == "kill":   # mid-step 8: the checkpoint at step 5 is the last
        with pytest.raises(Killed):
            _sft(monkeypatch, out, Weights(kill_at=7), a)
        assert checkpoint.saved_step(out) == 5
    else:   # SIGTERM during step 7: a checkpoint after it, then the end
        with pytest.raises(Left):
            _sft(monkeypatch, out, Weights(term_at=7), a)
        assert checkpoint.saved_step(out) == 7
        checkpoint._TERM.clear()
    done = _sft(monkeypatch, out, Weights(), a)
    assert done["resumed"] == [5 if stop == "kill" else 7]
    _same(tmp_path / "base", out)


def test_a_dagger_phase_resumes_across_its_passes_without_a_held_out_set(tmp_path, monkeypatch):
    """No held-out game: two passes, each drawn where the last left the generator, checkpointed every 5 steps; a stop in
    the second pass replays that pass from where it was drawn."""
    a = _sft_args(val_run=None, epochs=2)
    base = _sft(monkeypatch, tmp_path / "base", Weights(), a)
    per_pass = base["steps"] // 2
    assert base["steps"] == 2 * per_pass and per_pass % 5   # a checkpoint mid-pass, the stop past the first pass
    out = tmp_path / "stopped"
    with pytest.raises(Killed):
        _sft(monkeypatch, out, Weights(kill_at=per_pass + 6), a)
    resumed_at = checkpoint.saved_step(out)
    assert per_pass < resumed_at <= per_pass + 6
    done = _sft(monkeypatch, out, Weights(), a)
    assert done["resumed"] == [resumed_at]
    _same(tmp_path / "base", out)


def test_sigterm_during_a_held_out_check_ends_the_phase_after_it(tmp_path, monkeypatch):
    a = _sft_args()
    trainer = Weights()

    def nll_chains(chains):   # SIGTERM comes while step 10's held-out loss is measured
        if trainer.taken == 10:
            checkpoint._TERM.set()
        return {"items": [{"logp": -trainer.loss(), "tokens": 1}]}

    trainer.nll_chains = nll_chains
    with pytest.raises(Left) as left:
        _sft(monkeypatch, tmp_path / "out", trainer, a)
    assert left.value.args == (10,) and trainer.taken == 10   # no step after the check
    assert checkpoint.saved_step(tmp_path / "out") == 10


class Climbing(Weights):
    """A trainer whose held-out loss only climbs: the adapter a phase starts from stays its best."""

    def loss(self) -> float:
        return self.w + 1.0


@pytest.mark.parametrize("init", [False, True])
def test_a_phase_whose_start_stays_best_ends_on_it_after_a_resume(tmp_path, monkeypatch, init):
    """The held-out stop never improves on the start: with an init, OUT/adapter is init's files, linked; with none,
    the untrained adapter of the process that began the phase, saved then, not the resumed process's own."""
    a = _sft_args()
    start = tmp_path / "init"
    first = Climbing()
    first.w = 0.125   # with no init: this process's random A matrices, the untrained adapter
    if init:
        first.save(start)
    with pytest.raises(Killed):
        _sft(monkeypatch, tmp_path / "out", Climbing(kill_at=7) if init else _with_w(Climbing(kill_at=7), 0.125), a,
             start if init else None)
    again = _with_w(Climbing(), 0.5)   # another process: other random A matrices
    done = _sft(monkeypatch, tmp_path / "out", again, a, start if init else None)
    assert done["resumed"] == [5] and done["best_step"] == 0 and done["steps"] == 10   # stopped by patience 2
    assert _weights(tmp_path / "out")["w"] == 0.125


def _with_w(trainer, w):
    trainer.w = trainer.w0 = w
    return trainer


def test_checkpoints_come_every_EVERY_steps_whatever_the_held_out_cadence(tmp_path, monkeypatch):
    """checkpoint.EVERY 10 against a held-out check every 5 steps: a check that does not improve writes no checkpoint
    unless one is due, so a phase killed at step 8 trains again whole, and one killed at 13 goes on from step 10."""
    monkeypatch.setattr(checkpoint, "EVERY", 10)
    a = _sft_args(patience=3)
    base = _sft(monkeypatch, tmp_path / "base", Climbing(), a)
    assert base["steps"] == 15 and base["best_step"] == 0   # checks at 5, 10 and 15 that do not improve
    for kill_at, resumed in ((8, []), (13, [10])):
        out = tmp_path / f"killed-{kill_at}"
        with pytest.raises(Killed):
            _sft(monkeypatch, out, Climbing(kill_at=kill_at), a)
        assert checkpoint.saved_step(out) == (resumed[0] if resumed else None)
        done = _sft(monkeypatch, out, Climbing(), a)
        assert done["resumed"] == resumed
        _same(tmp_path / "base", out)


def test_off_a_phase_keeps_only_its_best_and_trains_again_whole_after_a_stop(tmp_path, monkeypatch):
    """checkpoint.ON off, as the paper's nine launch runs' trainers trained: a phase writes a checkpoint only at a new
    best (the adapter it saved before), nothing at the checks that do not improve or between them, and a stopped phase
    trains again from its start; SIGTERM is not caught."""
    import signal
    monkeypatch.setattr(checkpoint, "ON", False)
    a = _sft_args()
    trainer = Weights(kill_at=12)
    saves = []
    save = trainer.save
    trainer.save = lambda path: (saves.append(trainer.taken), save(path))
    with pytest.raises(Killed):
        _sft(monkeypatch, tmp_path / "out", trainer, a)
    best = [e["step"] for e in json.loads((tmp_path / "out" / "phase.json").read_text())["held_out"] if e["best"]]
    assert saves == best and checkpoint.saved_step(tmp_path / "out") is None   # with no init, the untrained start too
    done = _sft(monkeypatch, tmp_path / "out", Weights(), a)
    assert done["resumed"] == []
    _sft(monkeypatch, tmp_path / "base", Weights(), a)
    _same(tmp_path / "base", tmp_path / "out")
    before = signal.getsignal(signal.SIGTERM)
    checkpoint.catch_sigterm()
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_resumed_phase_whose_batches_differ_trains_again_whole(tmp_path, monkeypatch):
    a = _sft_args()
    with pytest.raises(Killed):
        _sft(monkeypatch, tmp_path / "out", Weights(kill_at=7), a)
    state = tmp_path / "out" / "checkpoints" / "state.json"
    doc = json.loads(state.read_text())
    doc["loop"]["steps"][2]["batch"] = "another batch"   # as if the code that draws them had changed
    state.write_text(json.dumps(doc))
    done = _sft(monkeypatch, tmp_path / "out", Weights(), a)
    assert done["resumed"] == []
    _sft(monkeypatch, tmp_path / "base", Weights(), a)
    _same(tmp_path / "base", tmp_path / "out")


# -- PG (train.fit with train.PolicyGradient) ------------------------------------------------------------------------

def _pg_args(**kw):
    d = dict(objective="pg", patience=2, eval_steps=5, batch=4, gpus=2, chain_slack=train.CHAIN_SLACK, chain_calls=2,
             max_seq_len=10 ** 6, max_minutes=240, per_call=False, epochs=3, max_steps=120)
    d.update(kw)
    return argparse.Namespace(**d)


def _pg(out: Path, trainer, a, val=True):
    tok = Weights.tokenizer
    items, _ = train.sft_items(tok, _games(8, 6), a.max_seq_len)
    held, _ = train.sft_items(tok, _games(3, 3, "dev"), a.max_seq_len)
    out.mkdir(exist_ok=True)
    update = train.PolicyGradient(a, train.Objective.pg, items, {"episodes": 8}, "games", held if val else None)
    return train.fit(a, trainer, update, out, None, random.Random(5))


@pytest.mark.parametrize("stop", ["kill", "sigterm"])
@pytest.mark.parametrize("val", [True, False])
def test_a_pg_phase_stopped_mid_way_resumes_on_the_same_batches_and_ends_on_the_same_adapter(tmp_path, stop, val):
    a = _pg_args()
    base = _pg(tmp_path / "base", Weights(), a, val)
    assert base["steps"] > 12
    out = tmp_path / "stopped"
    if stop == "kill":
        with pytest.raises(Killed):
            _pg(out, Weights(kill_at=11), a, val)
        assert checkpoint.saved_step(out) == 10
    else:
        with pytest.raises(Left):
            _pg(out, Weights(term_at=12), a, val)
        assert checkpoint.saved_step(out) == 12
        checkpoint._TERM.clear()
    done = _pg(out, Weights(), a, val)
    assert done["resumed"] == [10 if stop == "kill" else 12]
    _same(tmp_path / "base", out)


# -- OPSD (train.fit with train.SelfDistillation, on-policy) -------------------------------------------------------

class Live:
    """train.LiveAdapter without a server: each published student a folder of the trainer's weights, its name the
    registry's key, which the fake draw reads the drawing student's weights from."""

    names: dict[str, Path] = {}

    def __init__(self, trainer, server, out, model):
        self.trainer, self.root, self.published = trainer, out / "live", []
        shutil.rmtree(self.root, ignore_errors=True)

    def publish(self, step):
        path = self.root / f"s{step:04d}"
        self.trainer.save(path)
        return self._load(step, path)

    def publish_saved(self, step, src):
        path = self.root / f"s{step:04d}"
        checkpoint.link_tree(src, path)
        return self._load(step, path)

    def _load(self, step, path):
        name = f"{path.parent.parent.name}-s{step}-{len(Live.names)}"
        Live.names[name] = path
        self.published.append((name, path))
        return name

    def newest(self):
        return self.published[-1][1]

    def retire(self, keep):
        while len(self.published) > keep:
            name, path = self.published.pop(0)
            shutil.rmtree(path)


def _draw(pool, server, calls, model):
    """Each call's fresh reply, made from the drawing student's weights, which the step then trains on."""
    w = json.loads((Live.names[model] / "weights.json").read_text())["w"]
    out = []
    for c in calls:
        f = concurrent.futures.Future()
        f.set_result([Call(c.request, c.prompt_ids, c.reply_ids, "stop", {**c.meta, "drawn_by": w})])
        out.append(f)
    return out


def _opsd_args(**kw):
    d = dict(server="http://fake", max_seq_len=10 ** 6, batch=4, gpus=2, chain_slack=train.CHAIN_SLACK, chain_calls=2,
             epochs=2,
             max_steps=120, patience=2, eval_steps=5, max_minutes=240, divergence="forward_kl", skip_first=3,
             model="m")
    d.update(kw)
    return argparse.Namespace(**d)


def _opsd(monkeypatch, out: Path, trainer, a, val=True):
    import ask_sd.server
    monkeypatch.setattr(ask_sd.server, "Server", lambda url: types.SimpleNamespace(base=url))
    monkeypatch.setattr(train, "LiveAdapter", Live)
    monkeypatch.setattr(train, "draw", _draw)
    monkeypatch.setattr(train, "build_items", lambda tok, calls, context, placement, max_len: (list(calls), {
        "calls": len(calls), "items": len(calls), "template_mismatch": 0, "too_long": 0, "render_error": 0,
        "retokenised": 0}))
    monkeypatch.setattr(train, "sdft_chain_pairs", lambda items, a: (items, items))
    monkeypatch.setattr(train, "demo", lambda tok, call: f"the label at {train.place(call.meta)}")
    out.mkdir(exist_ok=True)
    update = train.SelfDistillation(a, trainer.tokenizer, _games(8, 6), _games(2, 3, "dev") if val else [])
    return train.fit(a, trainer, update, out, None, random.Random(7))


@pytest.mark.parametrize("stop", ["kill", "sigterm"])
@pytest.mark.parametrize("val", [True, False])
def test_an_opsd_phase_resumes_with_each_batch_drawn_by_the_student_that_drew_it(tmp_path, monkeypatch, stop, val):
    a = _opsd_args()
    base = _opsd(monkeypatch, tmp_path / "base", Weights(), a, val)
    assert base["steps"] > 12
    out = tmp_path / "stopped"
    if stop == "kill":
        with pytest.raises(Killed):
            _opsd(monkeypatch, out, Weights(kill_at=8), a, val)
        assert checkpoint.saved_step(out) == 5
    else:
        with pytest.raises(Left):
            _opsd(monkeypatch, out, Weights(term_at=8), a, val)
        assert checkpoint.saved_step(out) == 8
        checkpoint._TERM.clear()
    # the checkpoint holds the student that drew its next batch: one step older than its own adapter
    drawer = json.loads(next((out / "checkpoints").glob("c*/extra/drawer/weights.json")).read_text())
    assert drawer["w"] != json.loads(next((out / "checkpoints").glob("c*/weights.json")).read_text())["w"]
    done = _opsd(monkeypatch, out, Weights(), a, val)
    assert done["resumed"] == [5 if stop == "kill" else 8]
    _same(tmp_path / "base", out)
    assert not (out / "live").exists() or not any((out / "live").iterdir())   # every live student retired


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("ends", ["time cap", "off the cadence"])
def test_an_opsd_phase_whose_last_measurement_is_its_best_keeps_it_and_retires_its_students(tmp_path, monkeypatch, on,
                                                                                           ends):
    # its best checkpoint keeps the student drawing the next batch (SelfDistillation.keep), so the students are
    # retired only once the phase's adapter is written
    monkeypatch.setattr(checkpoint, "ON", on)
    a = _opsd_args(max_minutes=-1) if ends == "time cap" else _opsd_args(max_steps=7)
    trainer = Weights()
    out = tmp_path / "o"
    done = _opsd(monkeypatch, out, trainer, a)
    assert done["steps"] == (1 if ends == "time cap" else 7) and done["best_step"] == done["steps"]
    assert _weights(out)["w"] == trainer.w
    assert not any((out / "live").iterdir())
