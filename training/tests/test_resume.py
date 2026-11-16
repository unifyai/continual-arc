"""A tiny trained run, all of it, stopped partway and resumed, against the same run never stopped (the shared rule's
"Every run resumes": a tiny run stopped and resumed matches one never stopped, record for record and weight for weight).

The real-time driver's own loops run side by side on one run folder as realtime.sh runs them, each in a thread of this
process: the player (``realtime.cmd_player``), the labeller (``cmd_labeller``, with the labeller's own passes, label
set and split) and the train process (``cmd_train``, with ``phase.fit`` and its step checkpoints). What needs a GPU, a
model server or an API is a stand-in that computes exactly: the play (each game a function of its round, its user and
the adapter it is served, so that a round played on another phase's adapter, or on other weights, plays otherwise and
changes every label and step after it), the oracle (each label a function of the call it answers) and the trainer (a
few float weights and their moments, each step pulling them by every call it trains on and a draw of the phase's
generator).

The stop is a node's: the train process killed mid-step in phase 3, after its step checkpoint at step 5, while the
player is partway through round 4 (one of its two games captured, the round not recorded), every other loop stopping at
its next wait. The run goes on from its files: round 4 played again, its unrecorded game set aside, the labels kept,
phase 3 from its checkpoint. Everything the protocol fixes, every record, game, label, label set, step and weight,
matches the run that never stopped; only what the clock gives (``at``, ``train_s``, ``minutes``) and the resume's own
record differ.

What cannot replay exactly lies outside the test: a real round samples its games from vLLM, and OPSD draws its replies
from it, so no two plays are alike; there a resume goes on from the same state, not to the same result.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import threading
import time
import types
from pathlib import Path

from fakes import RECORD_DEFAULTS, TurnTokenizer

from continual_arc_training import checkpoints, gpupool, labeller, oracle, phase, play, realtime, stack, updates
from continual_arc_training.calls import render_prompt_ids

USERS, ROUNDS, CALLS = 2, 6, 3
NAME = "dagger-u2-lag1"
SYSTEM = {"role": "system", "content": "You are a general purpose agent."}
TOK = TurnTokenizer()
CLOCK = {"at", "train_s", "minutes"}   # what a record reads from the clock, which no two runs share
WAIT_S = 60
STEP = updates.Imitation.step


class Stopped(BaseException):
    """A loop stopped with its node: a BaseException, so that no loop's own handler takes it for a failure it retries."""


class Killed(Stopped):
    """The train process killed mid-step."""


def digest(doc) -> float:
    return int(hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:12], 16) / 16 ** 12


class Node:
    """What the run's loops share: the node's stop, and in the run that is stopped the moment it comes, phase 3's
    seventh step once the player has captured the first of round 4's two games."""

    def __init__(self, stopped: bool) -> None:
        self.stop = threading.Event()
        self.mid_round = threading.Event()
        self.kill_at = (3, 7) if stopped else None
        self.cut_round = 4 if stopped else None

    def sleep(self, seconds: float) -> None:
        if self.stop.is_set():
            raise Stopped
        time.sleep(0.003)


class Model:
    """The trainer: three float weights, their two moments and a step count, all saved and loaded exactly. Each step
    pulls the weights toward a point every call of its batch moves (its place and its target's ids), as far as a draw
    of the phase's generator says, and moves them as Adam does; the held-out loss is their distance from the held-out
    calls' own point, so it falls, then climbs once the weights pass it, and a phase's held-out stop comes some steps
    in."""

    tokenizer = TOK
    LR = 0.04

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.w, self.m, self.v, self.t = [0.0] * 3, [0.0] * 3, [0.0] * 3, 0
        self.g, self.n = [0.0] * 3, 0

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.json").write_text(json.dumps({"w": self.w, "m": self.m, "v": self.v, "t": self.t}))

    def load(self, path: Path) -> None:
        d = json.loads((path / "weights.json").read_text())
        self.w, self.m, self.v, self.t = d["w"], d["m"], d["v"], d["t"]
        self.g, self.n = [0.0] * 3, 0

    @staticmethod
    def point(chains) -> list[float]:
        content = [[[x["run"], x["user"], x["i"]] for x in c.labels] + [c.reply(k) for k in range(len(c.branches))]
                   for c in chains]
        return [digest([content, j]) * 1.5 for j in range(3)]

    def sft_chains(self, chains, rng):
        target, pull = self.point(chains), 0.5 + rng.random()
        n = sum(len(c.labels) for c in chains)
        for j in range(3):
            self.g[j] += n * pull * (self.w[j] - target[j])
        self.n += n
        loss = sum((w - x) ** 2 for w, x in zip(self.w, target, strict=True))
        return {"n_examples": n, "tokens": n, "nll_per_token": loss, "loss_mean": loss, "entropy_mean": 0.1,
                "chains": {"chains": len(chains), "tokens_read": n, "tokens_per_call": 1.0}}

    def step(self) -> dict:
        if not self.n:
            return {"steps": 0, "items": 0}
        self.t += 1
        norm = math.sqrt(sum((g / self.n) ** 2 for g in self.g))
        for j in range(3):
            g = self.g[j] / self.n
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            m_hat, v_hat = self.m[j] / (1 - 0.9 ** self.t), self.v[j] / (1 - 0.999 ** self.t)
            self.w[j] -= self.LR * m_hat / (math.sqrt(v_hat) + 1e-8)
        items, self.g, self.n = self.n, [0.0] * 3, 0
        return {"steps": 1, "items": items, "grad_norm": norm}

    def nll_chains(self, chains):
        target = self.point(chains)
        return {"items": [{"logp": -sum((w - x) ** 2 for w, x in zip(self.w, target, strict=True)), "tokens": 1}]}


def adapter_weights(run: Path, served: str) -> list[float] | None:
    """The weights of the adapter a round is served under its name, None for the base model."""
    if served == play.MODEL:
        return None
    phase_ = int(served.rsplit("-p", 1)[1])
    return json.loads((realtime.phase_dir(run, phase_) / "adapter" / "weights.json").read_text())["w"]


def game(r: int, slot, served: str, weights: list[float] | None) -> dict:
    """User ``slot.user``'s game of round ``r`` as the player captures it: three calls, each reply a move the served
    adapter makes (a function of its weights), each later call's prompt holding the moves before it and the verdict on
    each."""
    style = digest([r, slot.user, slot.task_id, weights])
    messages = [SYSTEM, {"role": "user", "content": f"New instance of {slot.task_id}. Test input (1x1):\n{r}"}]
    calls = []
    for i in range(CALLS):
        reply = f'{{"action": "submit", "grid": [[{int(digest([style, i]) * 10)}]]}}'
        request = {"model": served, "messages": [dict(m) for m in messages],
                   "chat_template_kwargs": {"enable_thinking": True}}
        # a call's id is its play's own, as the server's is: a game played on other weights has other ids
        calls.append({"call_id": f"{int(style * 16 ** 6):06x}r{r}u{slot.user}c{i}", "turn": i, "phase": "turn", "status": "done",
                      "finish_reason": "stop", "request": request,
                      "response": {"prompt_token_ids": render_prompt_ids(TOK, request),
                                   "choices": [{"index": 0, "finish_reason": "stop", "token_ids": TOK.encode(reply),
                                                "message": {"role": "assistant", "content": reply}}]}})
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Your submission was INCORRECT. Wrong attempts so far: {i + 1}."}]
    return {"user": slot.user, "slot": slot.user, "round": r, "task_id": slot.task_id, "task_label": slot.task_id,
            "phase": slot.phase, "solved": digest([style, "solved"]) < 0.4, "turns": CALLS, "calls": calls}


def stand_ins(monkeypatch, run: Path, node: Node) -> None:
    """Everything but the driver's own loops, as the module's docstring says, for the run in ``run`` on ``node``."""
    from continual_arc.runner import prior_records, whole_rounds

    def play_round(row, played, schedule, *, base_url, served_name):
        """The runner's own resume, then the cut's last round: each game captured as it ends, the round's records
        written together once every game has, as the runner writes only whole rounds."""
        done = len(whole_rounds(prior_records(played), schedule))
        rounds = schedule.rounds()
        weights = adapter_weights(run, served_name)
        records = []
        for i in range(done, len(schedule.slots)):
            slot, r = schedule.slots[i], rounds[i]
            if r == node.cut_round and slot.user == 1:   # the node stops partway through the round
                node.cut_round = None
                node.mid_round.set()
                node.stop.wait(WAIT_S)
                raise Stopped
            if node.stop.is_set():
                raise Stopped
            doc = game(r, slot, served_name, weights)
            captured = play.round_dir(played, r) / "captured"
            captured.mkdir(parents=True, exist_ok=True)
            with gzip.open(captured / f"u{slot.user:03d}.json.gz.tmp", "wt") as fp:
                json.dump(doc, fp)
            (captured / f"u{slot.user:03d}.json.gz.tmp").rename(captured / f"u{slot.user:03d}.json.gz")
            records.append({**RECORD_DEFAULTS, "step": i, "round": r, "user": slot.user, "task_id": slot.task_id,
                            "instance_seed": slot.instance_seed, "phase": slot.phase, "solved": doc["solved"],
                            "cost": 1.0 if doc["solved"] else 16.0})
        with open(played / "records.jsonl", "a") as fp:
            fp.writelines(json.dumps(rec) + "\n" for rec in records)

    def serve(a, run_, phase_, current):
        return realtime.adapter_name(a.name, phase_["phase"]) if phase_ and phase_["adapter"] else play.MODEL

    def ask(request):
        if node.stop.is_set():
            raise Stopped
        h = digest(request["messages"])
        content = f'The fill follows the frame.\n{{"action": "submit", "grid": [[{int(h * 10)}]]}}'
        return {"choices": [{"message": {"content": content, "reasoning": f"rows repeat every {int(h * 7) + 1} cells"}}],
                "usage": {"cost": 0.001}}

    def stopping_step(self, trainer, k, rng):
        record = STEP(self, trainer, k, rng)
        if node.kill_at == (self.a.seed, k + 1):   # its gradient taken, its optimizer step not
            node.kill_at = None
            node.mid_round.wait(WAIT_S)
            node.stop.set()
            raise Killed
        return record

    monkeypatch.setattr(play, "play_round", play_round)
    monkeypatch.setattr(realtime, "serve", serve)
    monkeypatch.setattr(realtime, "time", types.SimpleNamespace(sleep=node.sleep))
    monkeypatch.setattr(oracle, "Oracle", lambda base_url, model, effort: types.SimpleNamespace(ask=ask))
    monkeypatch.setattr(labeller, "_games", {})   # each labeller process reads its games afresh
    monkeypatch.setattr(labeller, "_load_tokenizer", lambda model: setattr(labeller, "_tokenizer", TOK))
    monkeypatch.setattr(labeller.concurrent.futures, "ProcessPoolExecutor",
                        labeller.concurrent.futures.ThreadPoolExecutor)
    monkeypatch.setattr(updates.Imitation, "step", stopping_step)
    monkeypatch.setattr(phase, "make_trainer", lambda a: Model())
    monkeypatch.setattr(stack, "record", lambda root: None)
    monkeypatch.setattr(checkpoints, "catch_sigterm", lambda: None)   # a signal handler is the main thread's alone
    monkeypatch.setattr(realtime, "fcntl", types.SimpleNamespace(flock=lambda fd, op: None, LOCK_EX=0))
    monkeypatch.setattr(gpupool, "ROOT", run.parent / "gpupool")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


class Inline:
    """The checkpoints' commit thread, run inline: the stop comes right after a save, which a real one's 15 s would
    not race."""

    def __init__(self, **kwargs) -> None:
        pass

    def submit(self, fn):
        import concurrent.futures
        done = concurrent.futures.Future()
        done.set_result(fn())
        return done

    def shutdown(self) -> None:
        pass


def loops(run: Path) -> dict[str, argparse.Namespace]:
    """Each loop's settings, as realtime.sh launches it."""
    common = {"run": str(run), "rounds": ROUNDS, "row": "dagger"}
    return {
        "player": argparse.Namespace(**common, users=USERS, name=NAME, lag=1, serve_gpus="1", serve_dp=1),
        "labeller": argparse.Namespace(**common, guidance=None, label_model="z-ai/glm-5.3-flash", label_effort="medium",
                                       relay_port=18230, label_workers=4, label_set_workers=2, need_thinking=0,
                                       label_round_budget=20.0, label_run_budget=3000.0, min_labelled_pct=40.0,
                                       hold_every=3),
        "train": argparse.Namespace(**common, lag=1, lr=1e-4, batch=4, gpus=1, lora_rank=128, max_seq_len=65536,
                                    eval_steps=5, patience=2, max_steps=30, max_epochs=20, val_max_items=1000,
                                    skip_first=3, max_minutes=240, start_after=0, dose="aggregate", replay_ratio=1.0,
                                    hold_every=3, train_gpus="0")}


def drive(monkeypatch, run: Path, node: Node) -> dict[str, BaseException]:
    """The run's three loops side by side until each has ended, finished or stopped; what each one raised."""
    stand_ins(monkeypatch, run, node)
    run.mkdir(parents=True, exist_ok=True)
    raised: dict[str, BaseException] = {}
    fns = {"player": realtime.cmd_player, "labeller": realtime.cmd_labeller, "train": realtime.cmd_train}

    def go(name: str, a: argparse.Namespace) -> None:
        try:
            fns[name](a)
        except BaseException as error:  # noqa: BLE001  (the stop, or a failure the test reports)
            raised[name] = error
            node.stop.set()

    threads = [threading.Thread(target=go, args=item, daemon=True) for item in loops(run).items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT_S)
    alive = [t for t in threads if t.is_alive()]
    node.stop.set()
    assert not alive, f"loops still going: {raised}"
    return raised


def contents(run: Path) -> dict[str, object]:
    """Every file the run keeps, by its path, as JSON where it is JSON (gzipped or not, a line at a time for JSON
    Lines), what the clock gives left out: the run's whole record but its progress diary and its resumes."""
    out: dict[str, object] = {}
    for path in sorted(p for p in run.rglob("*") if p.is_file()):
        rel = str(path.relative_to(run))
        if rel in ("progress", "resumes.jsonl") or "/unrecorded/" in rel or rel.endswith(".tmp"):
            continue
        data = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
        if path.name.endswith(".jsonl"):
            out[rel] = [strip(json.loads(line)) for line in data.splitlines() if line.strip()]
        elif path.name.endswith((".json", ".json.gz")) or path.name in ("DONE",):
            out[rel] = strip(json.loads(data)) if data.strip() else data
        else:
            out[rel] = data
    return out


def strip(doc):
    if isinstance(doc, dict):
        return {k: strip(v) for k, v in doc.items() if k not in CLOCK}
    if isinstance(doc, list):
        return [strip(x) for x in doc]
    return doc


def test_a_run_stopped_partway_and_resumed_goes_on_exactly_as_one_never_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "ThreadPoolExecutor", Inline)
    monkeypatch.setattr(checkpoints, "EVERY", 5)
    run = tmp_path / "rt" / NAME
    assert drive(monkeypatch, run, Node(stopped=False)) == {}
    whole = run.rename(tmp_path / "whole")   # every path a record holds names the run's folder: the next run's is it

    raised = drive(monkeypatch, run, Node(stopped=True))
    assert isinstance(raised.pop("train"), Killed) and all(type(e) is Stopped for e in raised.values()), raised
    p3, r4 = realtime.phase_dir(run, 3), realtime.round_dir(run, 4)
    assert checkpoints.saved_step(p3) == 5 and not (p3 / "DONE").exists()
    assert sorted(f.name for f in (r4 / "captured").iterdir()) == ["u000.json.gz"] and not (r4 / "play.done").exists()
    assert len((run / "play" / "records.jsonl").read_text().splitlines()) == 4 * USERS   # rounds 0-3, whole
    assert realtime.latest(run)["phase"] == 2

    assert drive(monkeypatch, run, Node(stopped=False)) == {}
    assert all((run / f"{loop}.done").exists() for loop in ("player", "labeller", "trainer"))
    kept, again = contents(whole), contents(run)
    assert kept.keys() == again.keys()
    for rel in kept:
        if rel.endswith("DONE"):   # the steps a phase went on from are its resume's own record
            assert {k: v for k, v in again[rel].items() if k != "resumed"} == \
                   {k: v for k, v in kept[rel].items() if k != "resumed"}, rel
        else:
            assert again[rel] == kept[rel], rel
    final = realtime.phase_dir(run, ROUNDS - 1) / "adapter" / "weights.json"
    assert json.loads(final.read_text())["w"] != [0.0] * 3   # trained: the last phase's weights, weight for weight
    assert json.loads((p3 / "DONE").read_text())["resumed"] == [5]
    assert json.loads((realtime.phase_dir(whole, 3) / "DONE").read_text())["resumed"] == []
    assert (r4 / "unrecorded" / "1" / "captured" / "u000.json.gz").exists()   # the play the runner never recorded
    resumes = [{k: v for k, v in json.loads(line).items() if k != "at"}
               for line in (run / "resumes.jsonl").read_text().splitlines()]
    assert {"loop": "player", "round": 4} in resumes and {"loop": "labeller", "round": 4} in resumes
    assert {"loop": "train", "phase": 3, "step": 5} in resumes
    assert not (whole / "resumes.jsonl").exists()


def test_the_stand_in_trainer_trains_each_phase_some_steps_under_its_held_out_stop(tmp_path, monkeypatch):
    """The test above means something only if its phases train several steps and stop on the held-out loss: a phase that
    ended at once would resume trivially."""
    monkeypatch.setattr(checkpoints, "ThreadPoolExecutor", Inline)
    run = tmp_path / "rt" / NAME
    assert drive(monkeypatch, run, Node(stopped=False)) == {}
    done = [json.loads((realtime.phase_dir(run, k) / "DONE").read_text()) for k in range(ROUNDS)]
    assert done[0]["steps"] and not done[0]["held_out_stop"]   # round 0 holds no game out: one pass
    assert all(d["held_out_stop"] and d["steps"] >= 10 for d in done[1:]) and done[3]["best_step"]
    served = [json.loads((realtime.round_dir(run, r) / "served.json").read_text())["phase"] for r in range(ROUNDS)]
    assert served == [None, None, 0, 1, 2, 3]
