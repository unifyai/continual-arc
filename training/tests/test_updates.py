"""The trained rows' updates' items (continual_arc_training.updates): what each trains on, the teacher's prompt with its
demonstration, the behaviour log-probabilities PG's ratios are taken against, and the student OPSD publishes."""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest
from fakes import TOK

from continual_arc_training import updates
from continual_arc_training.calls import Call, render_prompt_ids


def _call(prompt, reply, finish="stop", **meta):
    return Call({}, list(prompt), list(reply), finish, {"run": "r0001", "user": 0, "i": 0, "solved": True,
                                                         "round": 1, **meta})


def test_imitation_trains_every_calls_reply_as_it_ended_and_counts_a_call_past_the_window():
    tok = types.SimpleNamespace(eos_token_id=2)
    calls = [_call([5, 6], [7], "stop", i=0), _call([5, 6, 7, 8], [9], "length", i=1), _call([1] * 20, [9], i=2)]
    items, stats = updates.sft_items(tok, calls, max_seq_len=10)
    assert stats == {"calls": 3, "items": 2, "too_long": 1}
    assert [(it.student_ids, it.n) for it in items] == [([5, 6, 7, 2], 2), ([5, 6, 7, 8, 9], 1)]
    assert all(it.teacher_ids == it.student_ids and it.advantage is None for it in items)
    assert items[0].label == calls[0].meta and items[0].label is not calls[0].meta


def test_policy_gradient_trains_only_won_games_each_call_at_advantage_one():
    """REINFORCE with no baseline: a failure's calls push nothing, so only a win's are items, and a game whose outcome
    was never recorded is left out and counted."""
    tok = types.SimpleNamespace(eos_token_id=2)
    calls = [_call([1, 2], [3], user=0, i=0, solved=True), _call([1, 2, 3, 2, 4], [5], user=0, i=1, solved=True),
             _call([1, 9], [3], user=1, solved=False), _call([1, 8], [3], user=2, solved=None),
             _call([1, 7] * 10, [3], user=3, solved=True)]
    items, stats = updates.pg_items(tok, calls, max_seq_len=10)
    assert {k: v for k, v in stats.items() if k != "trained"} == {"episodes": 4, "unscored": 1, "wins": 2, "items": 2,
                                                                  "too_long": 1}
    assert stats["trained"] == [{"run": "r0001", "user": 0, "round": 1}, {"run": "r0001", "user": 3, "round": 1}]
    assert [it.student_ids for it in items] == [[1, 2, 3, 2], [1, 2, 3, 2, 4, 5, 2]]
    assert all(it.advantage == 1.0 for it in items)


def test_a_rounds_wins_alone_are_trained_from_its_own_games():
    tok = types.SimpleNamespace(eos_token_id=None)
    calls = [_call([1000 + u, 1], [10 + u], user=u, solved=s, round=1) for u, s in enumerate([True, True, False, False])]
    items, stats = updates.pg_items(tok, calls, max_seq_len=10**9)
    assert stats["episodes"] == 4 and stats["wins"] == 2
    assert {(it.label["round"], it.label["user"]) for it in items} == {(1, 0), (1, 1)}
    assert [it.n for it in items] == [1, 1]   # no end of turn to put back


def test_the_demonstration_is_the_label_as_text_its_thinking_opened_as_the_prompt_opens_it():
    """The generation prompt's last ids (Qwen3.5's: ``<|im_start|>``, ``assistant``, a newline, ``<think>\n``) say
    whether the template opened the thinking; the label's end of turn is left off."""
    vocab = ["<|im_start|>", "assistant", "\n", "<think>\n", "I see.", " Done.", "<|im_end|>"]
    tok = types.SimpleNamespace(decode=lambda ids: "".join(vocab[i] for i in ids))
    opened = Call({}, [0, 1, 2, 3], [4, 5, 6], "stop")
    assert updates.demo(tok, opened) == updates.SDFT_DEMO.format(demo="<think>\nI see. Done.")
    plain = Call({}, [2, 0, 1, 2], [5, 6], "stop")
    assert updates.demo(tok, plain) == updates.SDFT_DEMO.format(demo=" Done.")


def test_the_note_goes_into_the_served_ids_where_the_template_puts_it():
    """The teacher's prompt is the served ids with the note where the local renderings place it: when the served ids
    split an earlier turn otherwise than a fresh rendering (one id for two), when the note's first ids merge with the
    last message's last one, and when nothing differs; None when the served ids do not end as the template's do."""
    local = [1, 2, 3, 10, 11, 90, 91]             # the last message's content (10, 11), its end and the generation prompt
    noted = [1, 2, 3, 10, 11, 50, 51, 90, 91]     # the note (50, 51) at the content's end
    assert updates.splice_note(local, local, noted) == noted
    served = [1, 23, 10, 11, 90, 91]               # an earlier turn's 2, 3 sampled as one id, 23
    assert updates.splice_note(served, local, noted) == [1, 23, 10, 11, 50, 51, 90, 91]
    merged = [1, 2, 3, 10, 12, 51, 90, 91]         # the note's first id merged with the content's last (12 for 11, 50)
    assert updates.splice_note(served, local, merged) == [1, 23, 10, 12, 51, 90, 91]
    assert updates.splice_note([1, 23, 10, 11, 90, 92], local, noted) is None   # another generation prompt
    assert updates.splice_note([91], local, noted) is None


def test_a_distillation_item_reads_the_served_prompt_and_its_teacher_the_demonstration_at_the_last_messages_end():
    request = {"messages": [{"role": "system", "content": "Play."}, {"role": "user", "content": "The grid."}],
               "chat_template_kwargs": {"enable_thinking": True}}
    served = render_prompt_ids(TOK, request)
    reply = TOK.encode("Mine.")
    good = Call(request, served, reply, "stop", {"run": "r", "user": 0, "i": 0, "demo": " DEMO"})
    resplit = Call(request, [ord("#")] + served[2:], reply, "stop", {"run": "r", "user": 0, "i": 1, "demo": " DEMO"})
    mismatch = Call(request, served[:-1] + [ord("?")], reply, "stop", {"run": "r", "user": 0, "i": 2, "demo": " DEMO"})
    broken = Call({"messages": [{"content": "no role"}]}, served, reply, "stop",
                  {"run": "r", "user": 0, "i": 3, "demo": " DEMO"})
    items, stats = updates.distill_items(TOK, [good, resplit, mismatch, broken], max_seq_len=10**6)
    assert stats == {"calls": 4, "items": 2, "template_mismatch": 1, "too_long": 0, "render_error": 1,
                     "retokenised": 1}
    first = items[0]
    assert first.student_ids == served + reply + [TOK.eos_token_id] and first.n == len(reply) + 1
    assert TOK.decode(first.teacher_ids[:-first.n]) == TOK.decode(served).replace("The grid.", "The grid. DEMO")
    assert items[1].teacher_ids[:1] == [ord("#")] and items[1].teacher_ids[-first.n:] == first.student_ids[-first.n:]
    long_items, long_stats = updates.distill_items(TOK, [good], max_seq_len=len(served) + len(reply) + 1)
    assert long_items == [] and long_stats["too_long"] == 1   # the teacher's side is longer by the demonstration


def test_behaviour_logps_key_every_call_by_its_game_and_place_and_attach_behaviour_refuses_a_missing_one():
    calls = []
    for user in range(2):
        prompt = [5, 6, 7, 8 + user]
        for i in range(3):
            reply = [100 + 10 * user + i, 200 + i]
            calls.append(Call({}, list(prompt), reply, "stop", {"run": "r0004", "user": user, "i": i}))
            prompt = prompt + reply + [300 + i]   # the next prompt extends this one
    tok = types.SimpleNamespace(eos_token_id=2)
    items = updates.played_items(tok, calls, max_seq_len=10**6)
    assert [it.n for it in items] == [3] * 6   # each reply ended by the end-of-turn token it stopped on

    class FakeTrainer:
        def behaviour_logps_chains(self, chains):
            return {"items": [[float(c.reply(k)[0])] * len(c.reply(k)) for c in chains for k in range(len(c.branches))]}

    a = argparse.Namespace(chain_slack=8, chain_calls=2, max_seq_len=10**6)
    logps = updates.behaviour_logps(FakeTrainer(), a, items)
    assert logps == {("r0004", u, i): [float(100 + 10 * u + i)] * 3 for u in range(2) for i in range(3)}
    fresh = updates.played_items(tok, calls, max_seq_len=10**6)
    assert [it.behaviour_logp for it in updates.attach_behaviour(fresh, logps)] == [logps[k] for k in sorted(logps)]
    logps[("r0004", 0, 0)] = [0.0]
    with pytest.raises(ValueError, match="no behaviour log-probabilities"):
        updates.attach_behaviour(fresh, logps)
    del logps[("r0004", 0, 0)]
    with pytest.raises(ValueError, match="no behaviour log-probabilities"):
        updates.attach_behaviour(fresh, logps)


def test_a_policy_gradient_steps_record_keeps_the_dual_clip_and_the_worst_token():
    """A step whose ratios ran away names its call and token in phase.json after the adapter that played it is
    pruned."""
    worst = {"label": {"run": "r0021", "user": 1, "i": 57}, "position": 12, "token": 4242, "log_ratio": 13.05,
             "logp": -0.05, "logp_behaviour": -13.1}
    res = {"n_examples": 64, "tokens": 22891, "skipped_long": 0, "logp": [-0.2] * 64, "loss_mean": 1.95,
           "ratio_mean": 21.285371, "clip_fraction": 0.031716, "dual_clip_fraction": 1 / 22891, "approx_kl": 20.29,
           "entropy_mean": 0.188, "log_ratio_worst": worst}
    record = updates.pg_record(res)
    assert record["dual_clip_fraction"] == round(1 / 22891, 6) > 0 and record["log_ratio_worst"] == worst
    assert record["items"] == 64 and record["ratio_mean"] == 21.285371
    assert record["logp_per_token"] == round(-0.2 * 64 / 22891, 5)
    assert updates.pg_record({**res, "tokens": 0, "approx_kl": None})["logp_per_token"] is None


class FakeServer:
    """A model server that keeps the names it serves; ``refuse`` loads fail with a connection error first."""

    def __init__(self, serving=(), refuse=0):
        self.serving, self.refuse, self.events = list(serving), refuse, []

    def models(self):
        return list(self.serving)

    def load_adapter(self, name, path):
        if self.refuse:
            self.refuse -= 1
            raise ConnectionRefusedError("restarting")
        self.serving.append(name)
        self.events.append(("load", name, Path(path).name))

    def unload_adapter(self, name):
        self.serving.remove(name)
        self.events.append(("unload", name))


class SavingModel:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "adapter_model.safetensors").write_text("w")


def test_the_live_student_is_published_each_step_and_the_one_before_retired(tmp_path, monkeypatch):
    """Names a dead attempt of the same phase left loaded are unloaded first; each publish is a folder of its own
    loaded under a new name, retried while the server restarts; retiring keeps the newest alone."""
    monkeypatch.setattr(updates, "served_adapter", lambda model, path: path)
    monkeypatch.setattr(updates, "PUBLISH_BACKOFF_S", 0)
    out = tmp_path / "run" / "train" / "p0003"
    out.mkdir(parents=True)
    tag = "run-train-p0003"
    server = FakeServer([f"{tag}-t1-s4", "other-run-train-p0003-t1-s4", "base"], refuse=2)
    live = updates.LiveAdapter(types.SimpleNamespace(model=SavingModel()), server, out, "m")
    assert server.serving == ["other-run-train-p0003-t1-s4", "base"]
    first = live.publish(0)
    second = live.publish(1)
    assert first.startswith(tag + "-t") and first.endswith("-s0") and second.endswith("-s1")
    assert live.newest() == out / "live" / "s0001"
    live.retire(1)
    assert server.serving[-1:] == [second] and first not in server.serving
    assert not (out / "live" / "s0000").exists() and (out / "live" / "s0001").exists()
    saved = tmp_path / "kept"
    SavingModel().save_pretrained(saved)
    third = live.publish_saved(2, saved)
    assert (out / "live" / "s0002" / "adapter_model.safetensors").read_text() == "w" and third.endswith("-s2")
    live.retire(0)
    assert server.serving == ["other-run-train-p0003-t1-s4", "base"] and not list((out / "live").iterdir())
