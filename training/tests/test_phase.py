"""A training phase (continual_arc_training.phase.fit) with stand-in trainers: the held-out stop and the best adapter,
DAgger and OPSD on the same steps, and PG's own held-out loss."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import types

from test_chains import _games

from continual_arc_training import phase, updates
from continual_arc_training.calls import Call
from continual_arc_training.chains import TokenItem


def test_fit_measures_the_held_out_loss_keeps_the_best_adapter_and_stops_on_patience(tmp_path):
    """The update's held-out loss before the first step and every --eval-steps steps, the adapter kept at each new best,
    so OUT/adapter is the best one's; training stops once --patience measurements in a row have not improved on it.
    With nothing held out it takes every planned step, a checkpoint every checkpoints.EVERY, and ends on the adapter
    as training left it."""
    held = iter([1.0, 0.9, 0.8, 0.85, 0.82, 0.7])   # steps 0, 5, 10, 15, 20: best at 10, then two worse
    state = {"step": 0, "saved": []}

    class Trainer:
        def reset(self):
            pass

        def step(self):
            state["step"] += 1
            return {"grad_norm": 0.1}

        def save(self, path):
            path.mkdir(parents=True, exist_ok=True)
            state["saved"].append(state["step"])
            (path / "step").write_text(str(state["step"]))

    class Update(updates.Update):
        objective = "sft"

        def __init__(self, n, held):
            self.units, self.stats, self.held_items = _games(n // 10, 10), {"items": n}, list(range(held))

        def step(self, trainer, k, rng):
            return {"items": sum(len(p) for p in self.steps[k])}

        def held_out(self, trainer):
            return {"loss": next(held)}

    a = argparse.Namespace(epochs=1, batch=2, gpus=1, max_seq_len=10**6, chain_slack=8, chain_calls=2, max_steps=None,
                           eval_steps=5, patience=2, max_minutes=60)
    (tmp_path / "held").mkdir()
    (tmp_path / "none").mkdir()
    done = phase.fit(a, Trainer(), Update(60, 2), tmp_path / "held", None, random.Random(0))
    assert done["steps"] == 20 and done["held_out_stop"] and done["best_step"] == 10
    assert done["held_out"] == [1.0, 0.9, 0.8, 0.85, 0.82]
    # each new best's adapter, the untrained one it started from (no init) the first; OUT/adapter is the best one's
    assert state["saved"] == [0, 5, 10] and (tmp_path / "held" / "adapter" / "step").read_text() == "10"
    record = json.loads((tmp_path / "held" / "phase.json").read_text())
    assert [p["step"] for p in record["held_out"]] == [0, 5, 10, 15, 20] and len(record["steps"]) == 20
    assert all(s["items"] == 2 and s["pass"] == 1 for s in record["steps"])
    assert json.loads((tmp_path / "held" / "DONE").read_text())["best_step"] == 10
    assert not (tmp_path / "held" / "checkpoints").exists()   # gone once DONE is written

    state.update(step=0, saved=[])
    done = phase.fit(a, Trainer(), Update(60, 0), tmp_path / "none", None, random.Random(0))
    assert done["steps"] == 30 and not done["held_out_stop"] and done["held_out"] == []
    assert state["saved"] == [25, 30] and (tmp_path / "none" / "adapter" / "step").read_text() == "30"


def _labelled_calls(n_games=4, calls=7, seed=0):
    """A label set's calls: per game, prompts that each extend the last, each call's reply its target, ending in the end
    of turn (2); one call's prompt alone is past the window."""
    g = random.Random(seed)
    out = []
    for game in range(n_games):
        prompt = [1000 + game]
        for i in range(calls):
            prompt = prompt + [g.randrange(10, 900) for _ in range(g.randrange(3, 9))]
            if (game, i) == (2, calls - 1):
                prompt = prompt + [3] * 400   # past the window, so DAgger leaves the call out
            target = [g.randrange(10, 900) for _ in range(g.randrange(2, 6))] + [2]
            out.append(Call({"messages": [{"role": "user", "content": f"{game}:{i}"}]}, list(prompt), target, "stop",
                            {"run": "train-r0003", "user": game, "i": i, "call_id": f"r0003-u{game:03d}-c{i:04d}"}))
    return out


def test_dagger_and_opsd_train_the_same_labelled_calls_in_the_same_steps(tmp_path, monkeypatch):
    """OPSD is DAgger with SDFT's loss, so the same labelled calls give the two updates the same steps: the same calls
    in each step of every pass, a call DAgger's window leaves out left out of OPSD's steps too, however differently each
    update draws on the phase's random numbers between steps (phase.plan draws every pass before the first step). At
    each step OPSD draws its student's replies at exactly the calls DAgger imitates there, each carrying its label's
    demonstration."""
    calls = _labelled_calls()
    tok = types.SimpleNamespace(eos_token_id=2, decode=lambda ids: " ".join(map(str, ids)))
    a = argparse.Namespace(epochs=3, batch=6, gpus=2, max_seq_len=300, chain_slack=8, chain_calls=3, max_steps=None,
                           eval_steps=5, patience=2, max_minutes=60, skip_first=3, server="http://127.0.0.1:9/v1",
                           model="m")
    trained = {"dagger": [], "opsd": []}

    class Trainer:
        def __getattr__(self, name):   # reset, save: nothing to do here
            return lambda *args, **kwargs: None

        def step(self):
            return {"grad_norm": 0.1}

        def sft_chains(self, chains, rng):
            trained["dagger"].append(sorted(updates.place(label) for c in chains for label in c.labels))
            rng.shuffle(list(range(len(chains))))   # the trainer's own draw between steps
            return {"n_examples": 1, "tokens": 1, "nll_per_token": 1.0, "loss_mean": 1.0, "entropy_mean": 1.0,
                    "chains": {"chains": 1, "tokens_read": 1, "tokens_per_call": 1}}

        def distill_chains(self, students, teachers, rng, skip_first):
            assert skip_first == 3
            rng.shuffle(list(range(len(students) + 7)))   # a draw of another size
            return {"skipped_long": 0, "tokens": 1, "loss_mean": 0.1,
                    "chains": {"tokens_read": 1, "teacher_tokens_read": 1, "tokens_per_call": 1}}

    asked = []

    def draw(pool, server, step_calls, model):
        asked.append(step_calls)
        trained["opsd"].append(sorted(updates.place(c.meta) for c in step_calls))
        done = []
        for c in step_calls:
            f = concurrent.futures.Future()
            f.set_result([Call(c.request, list(c.prompt_ids), [7, 7, 2], "stop", {**c.meta, "resampled": True})])
            done.append(f)
        return done

    class Live:
        def __init__(self, trainer, server, out, model):
            pass

        def publish(self, step):
            return f"s{step}"

        def newest(self):
            return None

        def retire(self, keep):
            pass

    def distill_items(tok_, fresh, max_len):
        return ([TokenItem(list(c.prompt_ids) + list(c.reply_ids), list(c.prompt_ids) + list(c.reply_ids),
                           len(c.reply_ids), label=dict(c.meta)) for c in fresh],
                {"too_long": 0, "template_mismatch": 0, "render_error": 0, "retokenised": 0})

    monkeypatch.setattr(updates, "draw", draw)
    monkeypatch.setattr(updates, "LiveAdapter", Live)
    monkeypatch.setattr(updates, "distill_items", distill_items)
    for name, update in (("dagger", updates.Imitation(a, tok, calls, [])),
                         ("opsd", updates.SelfDistillation(a, tok, calls, []))):
        (tmp_path / name).mkdir()
        phase.fit(a, Trainer(), update, tmp_path / name, None, random.Random(7))
    assert trained["opsd"] == trained["dagger"] and len(trained["dagger"]) > 3 * 27 // 6   # every pass, step by step
    assert ("train-r0003", 2, 6) not in {place for step in trained["opsd"] for place in step}
    assert sum(map(len, trained["dagger"])) == 3 * (len(calls) - 1)
    labels = {updates.place(c.meta): c for c in _labelled_calls()}
    assert all(c.meta["demo"] == updates.demo(tok, labels[updates.place(c.meta)]) for step in asked for c in step)


def test_policy_gradient_stops_on_its_own_held_out_loss_and_logs_the_held_out_wins(tmp_path):
    """PG's steps through the trainer's policy_gradient_chains, its own loss over the held-out games' chains
    (policy_gradient_scores_chains) the held-out loss the phase stops on, the held-out wins' negative log-likelihood
    per token (nll_chains) logged beside it."""
    held = iter([1.0, 0.9, 0.8, 0.85, 0.82, 0.7])
    state = {"stepped": 0, "wins": []}

    class Trainer:
        def __getattr__(self, name):   # reset, save: nothing to do here
            return lambda *args, **kwargs: None

        def step(self):
            state["stepped"] += 1
            return {"grad_norm": 0.1}

        def policy_gradient_chains(self, chains, rng):
            assert all(c.advantages == [1.0] * len(c.branches) for c in chains)
            return {"n_examples": 2, "tokens": 8, "skipped_long": 0, "logp": [-1.0, -1.0], "loss_mean": 0.1,
                    "ratio_mean": 1.0, "clip_fraction": 0.0, "dual_clip_fraction": 0.0, "approx_kl": 0.0,
                    "entropy_mean": 1.0, "log_ratio_worst": None}

        def policy_gradient_scores_chains(self, chains):
            return {"items": [{"loss": next(held), "tokens": 4}] + [None] * (len(chains) - 1)}

        def nll_chains(self, chains):
            state["wins"].append([label["user"] for c in chains for label in c.labels])
            return {"items": [{"logp": -6.0, "tokens": 3}]}

    items = _games(4, 30, advantage=lambda i: 1.0)
    for it in items:
        it.behaviour_logp = [-1.0] * it.n
    held_items = _games(2, 3, seed=1)
    for it in held_items:
        it.label["solved"] = it.label["user"] == 1
    a = argparse.Namespace(epochs=20, batch=4, gpus=1, max_seq_len=10**6, chain_slack=8, chain_calls=4, max_steps=None,
                           eval_steps=5, patience=2, max_minutes=60)
    update = updates.PolicyGradient(a, items, {"episodes": 4, "wins": 4}, held_items)
    done = phase.fit(a, Trainer(), update, tmp_path, None, random.Random(0))
    assert done["steps"] == 20 and done["held_out_stop"] and done["best_step"] == 10 and state["stepped"] == 20
    assert set(state["wins"][0]) == {1}   # the held-out game that solved, alone
    record = json.loads((tmp_path / "phase.json").read_text())
    assert [e["wins_nll"] for e in record["held_out"]] == [2.0] * 5 and record["games"] == {"episodes": 4, "wins": 4}
    assert record["objective"] == "pg" and done["games"] == {"episodes": 4, "wins": 4}
