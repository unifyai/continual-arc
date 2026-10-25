"""train.py --chains on a CPU: chains found in a game's calls, and steps built from whole chains."""
import collections
import random

import pytest

train = pytest.importorskip("train")
TokenItem = pytest.importorskip("ask_sd.trainer").TokenItem


def _games(n_games=6, calls=30, seed=0):
    """Fake training calls: per game, prompts that each extend the last, except at a 'summary' every 9 calls, where
    the prompt starts over; each call's reply is a few ids of its own."""
    g = random.Random(seed)
    items = []
    for game in range(n_games):
        prompt = [1000 + game]
        for i in range(calls):
            if i and i % 9 == 0:
                prompt = [2000 + game, i]                      # rewritten: a new chain starts
            prompt = prompt + [g.randrange(1, 900) for _ in range(g.randrange(5, 40))]
            reply = [g.randrange(1, 900) for _ in range(g.randrange(2, 8))]
            items.append(TokenItem(prompt + reply, prompt + reply, len(reply), label={"run": "r", "user": game, "i": i}))
    g.shuffle(items)
    return items


def test_chains_follow_each_game_in_call_order_and_break_where_the_prompt_starts_over():
    items = _games()
    chains = train.find_chains(items)
    assert sum(len(c) for c in chains) == len(items)
    assert len(chains) == 6 * 4                                # 30 calls cut at 9, 18, 27
    for c in chains:
        assert len({(x.label["user"]) for x in c}) == 1
        assert [x.label["i"] for x in c] == sorted(x.label["i"] for x in c)
        for a, b in zip(c, c[1:]):
            pa, pb = a.student_ids[:-a.n], b.student_ids[:-b.n]
            assert pb[:len(pa)] == pa


def test_a_chain_item_puts_each_reply_at_its_own_prompts_end():
    chain = train.find_chains(_games(1, 5))[0]
    item = train.chain_item(chain)
    last = chain[-1]
    assert item.trunk == last.student_ids[:-last.n]
    for (point, reply), c in zip(item.branches, chain):
        assert item.trunk[:point] + reply == c.student_ids


@pytest.mark.parametrize("batch,cap", [(64, 8), (16, 2), (10, 4)])
def test_steps_are_whole_chain_pieces_with_every_call_once(batch, cap):
    items = _games(8, 40)
    chains = train.find_chains(items)
    steps = train.chain_batches(chains, batch, cap, 2, 10_000, random.Random(0))
    seen = collections.Counter()
    # every step holds `batch` calls but each pass's last: at most one short step per pass
    assert sum(sum(len(p) for p in step) != batch for step in steps) <= 2
    for step in steps:
        assert sum(len(p) for p in step) <= batch
        for piece in step:
            assert 1 <= len(piece) <= cap
            keys = [(x.label["user"], x.label["i"]) for x in piece]
            assert len({u for u, _ in keys}) == 1 and [i for _, i in keys] == list(range(keys[0][1], keys[0][1] + len(keys)))
            seen.update(id(x) for x in piece)
    assert len(seen) == len(items) and set(seen.values()) == {2}   # two passes: every call exactly twice
    full = [s for s in steps if sum(len(p) for p in s) == batch]
    assert sum(len(s) for s in full) < sum(sum(len(p) for p in s) for s in full)   # fewer sequences than calls


def test_a_phases_steps_and_its_checkpoints_fingerprint_are_the_same_on_any_number_of_gpus():
    """A run's share of its node's pool (gpupool.py) changes between phases and within one, so a phase's plan must not
    depend on its GPUs: the same calls give the same steps, pieces and all, and a checkpoint taken on 6 GPUs resumes on
    3 (checkpoint.fingerprint leaves the GPUs out)."""
    import argparse
    import checkpoint
    items = _games(8, 40)
    plans, prints = [], set()
    for gpus in (1, 3, 6):
        a = argparse.Namespace(epochs=2, batch=16, gpus=gpus, chain_calls=4, chain_slack=8, max_seq_len=10_000,
                               max_steps=None)
        plans.append(train.plan(items, a, random.Random(7)))
        prints.add(checkpoint.fingerprint(a, None, items))
    assert plans[0] == plans[1] == plans[2] and len(prints) == 1
    assert max(len(piece) for _, step in plans[0] for piece in step) == 4


def test_steps_never_draw_single_calls_from_different_places():
    """The batching's point: a step's pieces are runs of consecutive calls of one chain, so a step of 64 calls on 8
    GPUs reads a few long pieces; drawing calls one by one (the per-call path) would give 64 pieces of one call."""
    steps = train.chain_batches(train.find_chains(_games(8, 40)), 64, 8, 1, 10_000, random.Random(1))
    lengths = [len(p) for s in steps[:-1] for p in s]
    assert sum(lengths) / len(lengths) > 4


def _quirk_games(n_games=3, calls=12, seed=0):
    """Fake calls whose prompts each end with a generation id (5) that the next prompt renders as another (6), as an
    empty think block's newlines tokenize as one id where the earlier prompt ended with one."""
    g = random.Random(seed)
    items = []
    for game in range(n_games):
        history = [3000 + game]
        for i in range(calls):
            history = history + [g.randrange(10, 900) for _ in range(g.randrange(5, 30))]
            prompt = history + [5]
            reply = [g.randrange(10, 900) for _ in range(g.randrange(2, 6))]
            items.append(TokenItem(prompt + reply, prompt + reply, len(reply), label={"run": "q", "user": game, "i": i}))
            history = history + [6] + reply
    return items


def test_a_prompt_that_differs_in_its_last_ids_stays_in_the_chain_with_a_head():
    items = _quirk_games()
    assert len(train.find_chains(items, slack=0)) == len(items)       # without slack every call is its own chain
    chains = train.find_chains(items)
    assert len(chains) == 3 and sum(len(c) for c in chains) == len(items)
    for chain in chains:
        for piece in train.chain_pieces(chain, 5, 10_000):
            item = train.chain_item(piece)
            assert item.heads == [1] * (len(piece) - 1) + [0]          # the trunk's own call needs no head
            for k, (c, (point, ids)) in enumerate(zip(piece, item.branches)):
                assert item.trunk[:point] + ids == c.student_ids and item.reply(k) == c.student_ids[-c.n:]
            assert item.length() <= len(item.trunk) + sum(c.n + 1 for c in piece)
    with pytest.raises(ValueError, match="slack"):
        train.chain_item(chains[0][:3], slack=0)                       # a head past the slack is refused


def test_a_piece_never_passes_the_window():
    items = _games(2, 30)
    for chain in train.find_chains(items):
        for piece in train.chain_pieces(chain, 64, 300):
            item = train.chain_item(piece)
            assert item.length() <= 300 or len(piece) == 1


def test_splice_note_puts_the_note_into_the_served_ids_where_the_template_puts_it():
    """train.splice_note: the teacher's prompt is the served ids with the note where the local renderings place it,
    when the served ids split an earlier turn otherwise than a fresh rendering (one id for two), when the note's first
    ids merge with the last message's last one, and when nothing differs; None when the served ids do not end as the
    template's do."""
    import train
    local = [1, 2, 3, 10, 11, 90, 91]             # ... the last message's content (10, 11), its end and the generation prompt
    noted = [1, 2, 3, 10, 11, 50, 51, 90, 91]     # the note (50, 51) at the content's end
    assert train.splice_note(local, local, noted) == noted
    served = [1, 23, 10, 11, 90, 91]               # an earlier turn's 2, 3 sampled as one id, 23
    assert train.splice_note(served, local, noted) == [1, 23, 10, 11, 50, 51, 90, 91]
    merged = [1, 2, 3, 10, 12, 51, 90, 91]         # the note's first id merged with the content's last (12 for 11, 50)
    assert train.splice_note(served, local, merged) == [1, 23, 10, 12, 51, 90, 91]
    assert train.splice_note([1, 23, 10, 11, 90, 92], local, noted) is None   # another generation prompt
    assert train.splice_note([91], local, noted) is None


def test_fit_measures_the_held_out_loss_keeps_the_best_adapter_and_stops_as_daggers_held_out_stop_does(tmp_path,
                                                                                                         monkeypatch):
    """train.fit, every row's phase: the update's held-out loss before the first step and every --eval-steps steps,
    the adapter kept at each new best (a checkpoint of it), so OUT/adapter is the best one's; training stops once
    --patience measurements in a row have not improved on it, and DONE says so. With nothing held out it takes every
    planned step and saves the adapter as training left it. Step checkpoints off, so the saves are these alone
    (test_checkpoint.py tests them on)."""
    import argparse
    import json
    import random
    import checkpoint
    monkeypatch.setattr(checkpoint, "ON", False)
    held = iter([1.0, 0.9, 0.8, 0.85, 0.82, 0.7])   # steps 0, 5, 10, 15, 20: best at 10, then two worse
    state = {"step": 0, "saved": []}

    class Trainer:
        def reset(self):
            pass

        def step(self):
            state["step"] += 1
            return {"grad_norm": 0.1}

        def save(self, path):
            state["saved"].append(state["step"])
            (path / "step").write_text(str(state["step"]))

    class Update(train.Update):
        objective, chains = train.Objective.sft, False

        def __init__(self, n, held):
            self.units, self.stats, self.held_items = list(range(n)), {"items": n}, list(range(held))

        def step(self, trainer, k, rng):
            return {"items": len(self.steps[k])}

        def held_out(self, trainer):
            return {"loss": next(held)}

    a = argparse.Namespace(epochs=1, batch=2, gpus=1, max_seq_len=10**6, chain_slack=8, chain_calls=2, max_steps=None,
                           eval_steps=5, patience=2, max_minutes=60)
    done = train.fit(a, Trainer(), Update(60, 2), tmp_path, None, random.Random(0))
    assert done["steps"] == 20 and done["held_out_stop"] and done["best_step"] == 10
    assert done["held_out"] == [1.0, 0.9, 0.8, 0.85, 0.82]
    # each new best's adapter, the untrained one it started from (no init) the first; OUT/adapter is the best one's
    assert state["saved"] == [0, 5, 10] and (tmp_path / "adapter" / "step").read_text() == "10"
    phase = json.loads((tmp_path / "phase.json").read_text())
    assert [p["step"] for p in phase["held_out"]] == [0, 5, 10, 15, 20] and len(phase["steps"]) == 20
    assert json.loads((tmp_path / "DONE").read_text())["best_step"] == 10
    assert not (tmp_path / "checkpoints").exists()   # gone once DONE is written

    state.update(step=0, saved=[])
    done = train.fit(a, Trainer(), Update(60, 0), tmp_path, None, random.Random(0))
    assert done["steps"] == 30 and not done["held_out_stop"] and done["held_out"] == []
    assert state["saved"] == [30] and (tmp_path / "adapter" / "step").read_text() == "30"


def _labelled_calls(n_games=4, calls=7, seed=0):
    """A label set's calls (ask_sd Calls): per game, prompts that each extend the last, each call's reply its
    target, ending in the end of turn (2); one call's prompt alone is past test_dagger_and_opsd's window."""
    from ask_sd.calls import Call
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
                            {"run": "train-r0003", "user": game, "i": i, "call_id": f"/p/r0003-u{game:03d}-c{i:04d}"}))
    return out


def test_dagger_and_opsd_train_the_same_labelled_calls_in_the_same_steps(tmp_path, monkeypatch):
    """OPSD is DAgger with SDFT's loss, so the same labelled calls give the two updates the same steps: the same calls
    in each step of every pass, a call DAgger's window leaves out left out of OPSD's steps too, however differently each
    update draws on the phase's random numbers between steps (train.plan draws every pass before the first step). At
    each step OPSD draws its student's replies at exactly the calls DAgger imitates there, each carrying its label's
    demonstration (train.demo)."""
    import argparse
    import concurrent.futures
    import types
    from ask_sd.calls import Call
    from ask_sd.trainer import TokenItem
    calls = _labelled_calls()
    tok = types.SimpleNamespace(eos_token_id=2, decode=lambda ids: " ".join(map(str, ids)))
    a = argparse.Namespace(epochs=3, batch=6, gpus=2, max_seq_len=300, chain_slack=8, chain_calls=3, max_steps=None,
                           eval_steps=5,
                           patience=2, max_minutes=60, chains=True, divergence="forward_kl", skip_first=3,
                           server="http://server/v1", model="m")
    trained = {"dagger": [], "opsd": []}

    class Trainer:
        tokenizer = tok

        def __getattr__(self, name):   # reset, configure_*: nothing to do here
            return lambda *args, **kwargs: None

        def step(self):
            return {"grad_norm": 0.1}

        def sft_chains(self, chains, rng):
            trained["dagger"].append(sorted(train.place(label) for c in chains for label in c.labels))
            rng.shuffle(list(range(len(chains))))   # the trainer's own draw between steps, as ask-sd's
            return {"n_examples": 1, "tokens": 1, "nll_per_token": 1.0, "loss_mean": 1.0, "entropy_mean": 1.0,
                    "chains": {"chains": 1, "tokens_read": 1, "tokens_per_call": 1}}

        def distill_chains(self, students, teachers, rng, divergence, skip_first):
            rng.shuffle(list(range(len(students) + 7)))   # a draw of another size: OPSD's packs are cut otherwise
            return {"skipped_long": 0, "tokens": 1, "loss_mean": 0.1,
                    "chains": {"tokens_read": 1, "teacher_tokens_read": 1, "tokens_per_call": 1}}

    asked = []

    def draw(pool, server, step_calls, model):
        asked.append(step_calls)
        trained["opsd"].append(sorted(train.place(c.meta) for c in step_calls))
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

    monkeypatch.setattr(train, "draw", draw)
    monkeypatch.setattr(train, "LiveAdapter", Live)
    monkeypatch.setattr(train, "build_items", lambda tok_, fresh, context, placement, max_len: (
        [TokenItem(list(c.prompt_ids) + list(c.reply_ids), list(c.prompt_ids) + list(c.reply_ids), len(c.reply_ids),
                   label=dict(c.meta)) for c in fresh],
        {"too_long": 0, "template_mismatch": 0, "render_error": 0, "retokenised": 0}))
    for name, update in (("dagger", train.Imitation(a, tok, calls, [])),
                         ("opsd", train.SelfDistillation(a, tok, calls, []))):
        (tmp_path / name).mkdir()
        train.fit(a, Trainer(), update, tmp_path / name, None, random.Random(7))
    assert trained["opsd"] == trained["dagger"] and len(trained["dagger"]) > 3 * 27 // 6   # every pass, step by step
    assert ("train-r0003", 2, 6) not in {place for step in trained["opsd"] for place in step}
    assert sum(map(len, trained["dagger"])) == 3 * (len(calls) - 1)
    labels = {train.place(c.meta): c for c in _labelled_calls()}
    assert all(c.meta["demo"] == train.demo(tok, labels[train.place(c.meta)]) for step in asked for c in step)


def test_policy_gradient_stops_on_its_own_held_out_loss_and_logs_the_held_out_wins(tmp_path):
    """train.PolicyGradient in train.fit: every step's chain pieces through Trainer.policy_gradient_chains, PG's own
    loss over the held-out games' chains (Trainer.policy_gradient_scores_chains) the held-out loss the phase stops on,
    the held-out wins' negative log-likelihood per token (Trainer.nll_chains) logged beside it."""
    import argparse
    import json
    held = iter([1.0, 0.9, 0.8, 0.85, 0.82, 0.7])
    state = {"stepped": 0, "wins": []}

    class Trainer:
        def __getattr__(self, name):   # reset, configure_behaviour: nothing to do here
            return lambda *args, **kwargs: None

        def step(self):
            state["stepped"] += 1
            return {"grad_norm": 0.1}

        def policy_gradient_chains(self, chains, rng):
            return {"n_examples": 2, "tokens": 8, "skipped_long": 0, "logp": [-1.0, -1.0], "loss_mean": 0.1,
                    "ratio_mean": 1.0, "clip_fraction": 0.0, "dual_clip_fraction": 0.0, "approx_kl": 0.0,
                    "entropy_mean": 1.0, "log_ratio_worst": None}

        def policy_gradient_scores_chains(self, chains):
            return {"items": [{"loss": next(held), "tokens": 4}] + [None] * (len(chains) - 1)}

        def nll_chains(self, chains):
            state["wins"].append([label["user"] for c in chains for label in c.labels])
            return {"items": [{"logp": -6.0, "tokens": 3}]}

    items = _games(4, 30)
    val = [it for it in _games(2, 3, seed=1)]
    for it in val:
        it.label["solved"] = it.label["user"] == 1
    a = argparse.Namespace(epochs=20, batch=4, gpus=1, max_seq_len=10**6, chain_slack=8, chain_calls=4, max_steps=None,
                           eval_steps=5, patience=2, max_minutes=60, per_call=False)
    update = train.PolicyGradient(a, train.Objective.pg, items, {"episodes": 4, "wins": 4}, "games", val)
    done = train.fit(a, Trainer(), update, tmp_path, None, random.Random(0))
    assert done["steps"] == 20 and done["held_out_stop"] and done["best_step"] == 10 and state["stepped"] == 20
    assert set(state["wins"][0]) == {1}   # the held-out game that solved, alone
    phase = json.loads((tmp_path / "phase.json").read_text())
    assert [e["wins_nll"] for e in phase["held_out"]] == [2.0] * 5 and phase["games"] == {"episodes": 4, "wins": 4}


def test_behaviour_logps_key_every_call_by_its_game_and_place_and_attach_behaviour_refuses_a_missing_one():
    """train.played_items and train.behaviour_logps: every call's reply (the end-of-turn token put back where it
    stopped) scored under the behaviour adapter through chains, keyed by run, user and place, whatever chain piece it
    landed in; attach_behaviour gives each item its own, and refuses an item with none or of another length."""
    import argparse
    import types
    import train
    from ask_sd.calls import Call
    calls = []
    for user in range(2):
        prompt = [5, 6, 7, 8 + user]
        for i in range(3):
            reply = [100 + 10 * user + i, 200 + i]
            calls.append(Call({}, list(prompt), reply, "stop", {"run": "r0004", "user": user, "i": i}))
            prompt = prompt + reply + [300 + i]   # the next prompt extends this one
    items = train.played_items(types.SimpleNamespace(eos_token_id=2), calls, max_seq_len=10**6)
    assert [it.n for it in items] == [3] * 6   # each reply ended by the end-of-turn token it stopped on

    class FakeTrainer:
        def token_logps_chains(self, chains, behaviour):
            assert behaviour
            return {"items": [[float(c.reply(k)[0])] * len(c.reply(k)) for c in chains for k in range(len(c.branches))]}

    a = argparse.Namespace(batch=2, gpus=1, chain_slack=8, chain_calls=2, max_seq_len=10**6)
    logps = train.behaviour_logps(FakeTrainer(), a, items)
    assert logps == {("r0004", u, i): [float(100 + 10 * u + i)] * 3 for u in range(2) for i in range(3)}
    fresh = train.played_items(types.SimpleNamespace(eos_token_id=2), calls, max_seq_len=10**6)
    assert [it.behaviour_logp for it in train.attach_behaviour(fresh, logps)] == [logps[k] for k in sorted(logps)]
    del logps[("r0004", 1, 2)]
    with pytest.raises(ValueError, match="no behaviour log-probabilities"):
        train.attach_behaviour(fresh, logps)
