"""paper.py's off-GPU pieces, all on synthetic data (no played run, no GPU, no served model needed):

- the paper rows' schedule (arc1-small served to 100 users), cut a round at a time (paper.schedule_cut: a literal
  prefix);
- the K-user epilogue, cut the same way (paper.epilogue_cut);
- the round-sharded capture (paper.save_instance_hook, paper.round_dir): the shape loop.py's own save_instance
  writes, and that a round's own train.load_calls/train.pg_items trains on that round's games alone, each weighed
  against the run's earlier games from the records.jsonl at the run's root, where a flat captured/ (one file per
  user, independent of round) would lose an earlier round's games, since the paper rows' users each play again
  every round under the same user ids;
- the served-adapter check (paper.check_served_adapter) against a fake server.

Run with ``python3 -m pytest test_paper.py -q``; needs ``continual_arc`` on the path (as every other test here
assumes) and, for the capture/pg_items tests, ``ask_sd``/``train`` (``pytest.importorskip``, as test_dev_loop.py).
"""
from __future__ import annotations

import collections
import gzip
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import paper

train = pytest.importorskip("train")
Call = pytest.importorskip("ask_sd.calls").Call


# -- the real schedule, cut a round at a time --------------------------------------------------------------

def test_schedule_cut_is_a_literal_prefix_of_the_full_schedule():
    full = paper.full_schedule()
    assert full.users == 100
    cut2 = paper.schedule_cut(1)   # round_ = 1: rounds 0 and 1, "+1" in first_rounds(round_ + 1)
    assert cut2.rounds()[-1] == 1
    assert cut2.slots == full.slots[:len(cut2)]   # a true slice, not a shorter rebuild
    cut1 = paper.schedule_cut(0)
    assert cut1.rounds()[-1] == 0
    assert cut1.slots == cut2.slots[:len(cut1)]    # cuts nest: round 0's cut is round 1's cut's own prefix


def test_epilogue_cut_is_the_held_out_tasks_for_k_users_only():
    from continual_arc.configs import epilogue_for_users

    full_epilogue = epilogue_for_users(paper.served(), paper.SEED, paper.DEFAULT_EPILOGUE_USERS)
    assert full_epilogue.users == paper.DEFAULT_EPILOGUE_USERS
    one_round = paper.epilogue_cut(round_=0)
    assert one_round.rounds()[-1] == 0
    assert one_round.slots == full_epilogue.slots[:len(one_round)]
    assert all(s.phase == "epilogue" for s in one_round.slots)


# -- the users a stream is served to ----------------------------------------------------------------------

def test_served_is_arc1_small_served_to_that_many_users_the_paper_rows_100_by_default():
    from continual_arc.configs import CANONICAL_USERS

    assert paper.served().users == paper.PAPER_USERS == 100 and paper.served().name == "arc1-small"
    eight = paper.served(8)
    assert eight.users == 8 and eight.name == "arc1-small" and 8 in CANONICAL_USERS


def test_a_smaller_user_count_gives_exactly_a_larger_ones_first_users_round_by_round():
    """paper.full_schedule(users=N)'s users are not a different draw: every field a user's own stream or epilogue
    order depends on (continual_arc.configs._user_seeds: the seed and the user's own index; config_tasks: the dataset
    and n_tasks) is unaffected by `users`, so the paper rows' first N users give this schedule exactly, round by round
    -- the property that lets runs at 1, 2, 4 and 8 users share every user's stream with the paper rows' 100."""
    full = paper.full_schedule()
    small = paper.full_schedule(users=4)
    assert small.users == 4
    full_rounds, small_rounds = full.rounds(), small.rounds()
    by_round_full: dict[int, list] = {}
    for slot, r in zip(full.slots, full_rounds):
        by_round_full.setdefault(r, []).append(slot)
    by_round_small: dict[int, list] = {}
    for slot, r in zip(small.slots, small_rounds):
        by_round_small.setdefault(r, []).append(slot)
    assert len(by_round_small) == len(by_round_full)   # the stream itself is not shortened, only narrowed
    for r, slots in by_round_small.items():
        assert slots == by_round_full[r][:4]


def test_schedule_cut_at_fewer_users_is_still_a_literal_prefix():
    cut = paper.schedule_cut(2, users=4)
    assert cut.users == 4 and cut.rounds()[-1] == 2
    full_small = paper.full_schedule(users=4)
    assert cut.slots == full_small.slots[:len(cut)]


def test_epilogue_cut_at_fewer_users_serves_only_those_users_epilogue():
    from continual_arc.configs import epilogue_for_users

    full_epilogue_small = epilogue_for_users(paper.served(4), paper.SEED, 2)
    cut = paper.epilogue_cut(round_=0, epilogue_users=2, users=4)
    assert cut.slots == full_epilogue_small.slots[:len(cut)]


def test_schedule_cut_runs_on_past_the_stream_into_the_epilogue():
    """A run plays the whole schedule: rounds 0-199 are the stream and 200-299 the epilogue, in which every user meets
    each of the 100 held-out tasks once, so a trained row's round 200 is a cut like any other and its epilogue is the
    full schedule's own."""
    full = paper.full_schedule(users=2)
    rounds = full.rounds()
    assert rounds[-1] == 299
    phase = {r: s.phase for s, r in zip(full.slots, rounds)}
    assert {phase[r] for r in range(200)} == {"stream"} and {phase[r] for r in range(200, 300)} == {"epilogue"}
    cut = paper.schedule_cut(200, users=2)
    assert cut.slots == full.slots[:len(cut)] and [s.phase for s in cut.slots[-3:]] == ["stream", "epilogue", "epilogue"]
    for u in range(2):
        held = [s.task_id for s in full.slots if s.user == u and s.phase == "epilogue"]
        assert len(held) == len(set(held)) == 100


def test_wide_epilogue_cut_deals_the_epilogue_alones_instances_width_at_a_time():
    """Transfer's references meet users 0-3's 400 held-out instances 20 to a round: every instance of the 4-user
    epilogue alone once, in its own order, so their round count is 20, not 100."""
    narrow = paper.epilogue_cut(99, epilogue_users=4, users=4)
    wide = paper.wide_epilogue_cut(19, width=20, epilogue_users=4, users=4)
    assert wide.users == 20 and wide.rounds()[-1] == 19
    assert [(s.task_id, s.instance_seed, s.phase) for s in wide.slots] == \
        [(s.task_id, s.instance_seed, s.phase) for s in narrow.slots]
    assert [s.user for s in wide.slots[:21]] == list(range(20)) + [0] and not any(s.reset_before for s in wide.slots)
    assert paper.wide_epilogue_cut(0, width=20, epilogue_users=4, users=4).slots == wide.slots[:20]
    with pytest.raises(ValueError, match="whole rounds"):
        paper.wide_epilogue_cut(0, width=30, epilogue_users=4, users=4)


def test_one_reference_play_over_users_0_to_3_serves_every_run_at_1_2_and_4_users():
    """Transfer matches a run's epilogue to its references by task and instance seed. User u's held-out instances are
    the same in a run at any number of users, in the epilogue alone (epilogue_cut, the references' schedule) and in the
    benchmark's fresh reference schedule, so H0 and the bare model, each played once over users 0-3, serve the 1-, 2-
    and 4-user runs alike."""
    from continual_arc.configs import build_schedule

    def held(slots):
        out: dict[int, list] = {}
        for s in slots:
            if s.phase == "epilogue":
                out.setdefault(s.user, []).append((s.task_id, s.instance_seed))
        return out

    reference = held(paper.epilogue_cut(99, epilogue_users=4, users=4).slots)
    assert sorted(reference) == [0, 1, 2, 3] and all(len(v) == 100 for v in reference.values())
    assert held(build_schedule(paper.served(4), "fresh", paper.SEED).slots) == reference
    for users in (1, 2, 4):
        assert held(paper.full_schedule(users=users).slots) == {u: reference[u] for u in range(users)}


# -- round-sharded capture ----------------------------------------------------------------------------------

def test_round_dir_is_four_digits_wide_and_distinct_per_round():
    d0, d1, d9999 = paper.round_dir(Path("/r"), 0), paper.round_dir(Path("/r"), 1), paper.round_dir(Path("/r"), 9999)
    assert d0.name == "r0000" and d1.name == "r0001" and d9999.name == "r9999"
    assert d0 != d1


def _fake_call(call_id, prompt, reply, finish="stop", started=0.0):
    return types.SimpleNamespace(call_id=call_id, turn=0, phase="play", status="done", finish_reason=finish, started=started,
                                 request={"messages": [{"role": "user", "content": "go"}]},
                                 response={"prompt_token_ids": prompt, "choices": [
                                     {"token_ids": reply, "finish_reason": finish}]})


def _fake_trace(user, round_, task_id, solved, seed):
    prompt = [1000 + user, round_]
    reply = [10 + seed]
    return types.SimpleNamespace(user=user, slot=0, round=round_, task_id=task_id, task_label=task_id, phase="stream",
                                 solved=solved, turns=1, calls=[_fake_call(0, prompt, reply)])


_RECORD_DEFAULTS = {"step": 0, "phase": "stream", "source": "fake", "exposure_index": 0, "gap": None,
                   "instance_seed": 0, "reset_before": False, "wrong_attempts": 0, "invalid_submissions": 0,
                   "demo_requests": 0, "gave_up": False}


def _write_records_jsonl(path: Path, entries: list[tuple[int, int, str, bool, float]]) -> None:
    """A minimal but valid records.jsonl (every InstanceRecord field continual_arc.runner.prior_records needs),
    one line per (user, round, task_id, solved, cost)."""
    lines = [json.dumps({**_RECORD_DEFAULTS, "user": u, "round": r, "task_id": t, "solved": s, "cost": c})
            for u, r, t, s, c in entries]
    path.write_text("\n".join(lines) + "\n")


def test_save_instance_hook_writes_loops_own_shape_under_round_dir(tmp_path):
    hook = paper.save_instance_hook(tmp_path, paper.played)
    hook(_fake_trace(user=3, round_=0, task_id="A", solved=True, seed=1))
    path = paper.round_dir(tmp_path, 0) / "captured" / "u003.json.gz"
    assert path.exists()
    with gzip.open(path, "rt") as fp:
        doc = json.load(fp)
    assert doc["user"] == 3 and doc["round"] == 0 and doc["task_id"] == "A" and doc["solved"] is True
    assert doc["calls"][0]["response"]["choices"][0]["token_ids"] == [11]


def test_save_instance_hook_never_overwrites_an_earlier_round_of_the_same_user(tmp_path):
    """The paper rows' 100 users all play again every round under the same user id: a flat captured/ (one file per
    user, as loop.py's own cmd_play writes for the dev loop's fresh-user-per-round probes) would have round 1's save
    overwrite round 0's. Sharding by round must keep both."""
    hook = paper.save_instance_hook(tmp_path, paper.played)
    hook(_fake_trace(user=0, round_=0, task_id="A", solved=True, seed=1))
    hook(_fake_trace(user=0, round_=1, task_id="A", solved=False, seed=2))
    r0 = paper.round_dir(tmp_path, 0) / "captured" / "u000.json.gz"
    r1 = paper.round_dir(tmp_path, 1) / "captured" / "u000.json.gz"
    assert r0.exists() and r1.exists()
    with gzip.open(r0, "rt") as fp:
        assert json.load(fp)["solved"] is True
    with gzip.open(r1, "rt") as fp:
        assert json.load(fp)["solved"] is False


def test_a_game_is_saved_in_the_order_its_calls_ended_unless_its_round_is_played_to_be_resumed(tmp_path):
    """The learner keeps a game's calls in the order they ended. A round played without RESUME_CALLS saves them in that
    order; a round played to be resumed saves them in the order they began (Resume.calls), the order its recording is
    replayed in, since a call cancelled at a cut ends after the next has begun."""
    t = _fake_trace(user=0, round_=0, task_id="A", solved=True, seed=1)
    t.calls = [_fake_call("b", [1], [2], started=2.0), _fake_call("a", [1], [3], started=1.0)]
    resume = types.SimpleNamespace(kept=paper.captured)
    paper.save_instance_hook(tmp_path / "off", paper.played)(t)
    paper.save_instance_hook(tmp_path / "on", lambda trace: paper.Resume.calls(resume, trace))(t)

    def saved(run_dir):
        with gzip.open(paper.round_dir(run_dir, 0) / "captured" / "u000.json.gz", "rt") as fp:
            return [c["call_id"] for c in json.load(fp)["calls"]]

    assert saved(tmp_path / "off") == ["b", "a"]
    assert saved(tmp_path / "on") == ["a", "b"]


def _write_flat(captured: Path, t) -> None:
    """``t`` saved into ``captured`` directly by user id alone, as a flat (unsharded) captured/ would: the shape a
    fix-free ``save_instance`` writes, named only ``uNNN.json.gz`` with no round in its path."""
    doc = {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
           "phase": t.phase, "solved": t.solved, "turns": t.turns,
           "calls": [{"call_id": c.call_id, "turn": c.turn, "phase": c.phase, "status": c.status,
                      "finish_reason": c.finish_reason, "request": c.request, "response": c.response}
                     for c in t.calls]}
    with gzip.open(captured / f"u{t.user:03d}.json.gz", "wt") as fp:
        json.dump(doc, fp, default=str)


def test_round_sharded_capture_trains_a_round_on_its_own_games(tmp_path):
    """Round 0 and round 1 reuse the same four users and the same task ("A", as the paper rows' own users do), with a
    different solve pattern each round. Reading each round's own directory on its own (as paper.sh's train step
    does, --train-runs one round at a time) must give that round's games alone, each with the cost found in the
    records.jsonl at the run's root (train.instance_records, two directories up), and train.pg_items must train that
    round's own wins alone."""
    hook = paper.save_instance_hook(tmp_path, paper.played)
    round0_solved = {0: True, 1: False, 2: False, 3: False}    # 1 of 4 solved
    round1_solved = {0: True, 1: True, 2: False, 3: False}     # 2 of 4 solved: a different pattern
    for user, solved in round0_solved.items():
        hook(_fake_trace(user=user, round_=0, task_id="A", solved=solved, seed=user))
    for user, solved in round1_solved.items():
        hook(_fake_trace(user=user, round_=1, task_id="A", solved=solved, seed=10 + user))
    # the captured doc carries no cost of its own: train.load_calls joins it in from records.jsonl
    _write_records_jsonl(tmp_path / "records.jsonl",
                         [(u, 0, "A", s, 8.0) for u, s in round0_solved.items()]
                         + [(u, 1, "A", s, 8.0) for u, s in round1_solved.items()])

    tok = types.SimpleNamespace(eos_token_id=None)
    dir0, dir1 = paper.round_dir(tmp_path, 0), paper.round_dir(tmp_path, 1)
    assert dir0.name != dir1.name
    calls0, _ = train.load_calls(dir0)
    calls1, _ = train.load_calls(dir1)
    assert {c.meta["run"] for c in calls0} == {dir0.name} and {c.meta["round"] for c in calls0} == {0}
    assert {c.meta["run"] for c in calls1} == {dir1.name} and {c.meta["round"] for c in calls1} == {1}
    records = train.instance_records(dir1)
    assert len(records) == 8 and train.instance_records(dir0) == records   # the run's own, both rounds

    assert {c.meta["cost"] for c in calls0 + calls1} == {8.0}   # joined in from the run's own records.jsonl

    items1, stats1 = train.pg_items(tok, calls1, max_seq_len=10**9)
    assert stats1["episodes"] == 4 and stats1["wins"] == 2   # round 1's own pattern
    assert {(it.label["round"], it.label["user"]) for it in items1} == {(1, 0), (1, 1)}
    items0, stats0 = train.pg_items(tok, calls0, max_seq_len=10**9)
    assert stats0["episodes"] == 4 and stats0["wins"] == 1
    assert {(it.label["round"], it.label["user"]) for it in items0} == {(0, 0)}
    assert all(it.advantage == 1.0 for it in items0 + items1)

    # The bug sharding fixes: a flat captured/ (one file per user, independent of round, as loop.py's own
    # save_instance writes for the dev loop's fresh-user-per-round probes) would have round 1's write for user 0
    # overwrite round 0's, since both rounds reuse the same user ids. Simulate that flat layout directly and show
    # train.load_calls then finds only round 1's 4 games, not 8: round 0's are gone.
    flat = tmp_path / "flat" / "captured"
    flat.mkdir(parents=True)
    for user, solved in round0_solved.items():
        _write_flat(flat, _fake_trace(user=user, round_=0, task_id="A", solved=solved, seed=user))
    for user, solved in round1_solved.items():
        _write_flat(flat, _fake_trace(user=user, round_=1, task_id="A", solved=solved, seed=10 + user))
    _write_records_jsonl(tmp_path / "flat" / "records.jsonl", [(u, 1, "A", s, 8.0) for u, s in round1_solved.items()])
    flat_calls, _ = train.load_calls(tmp_path / "flat")
    assert len(flat_calls) == 4 and {c.meta["round"] for c in flat_calls} == {1}   # round 1 overwrote round 0


def test_load_calls_reads_cost_from_records_jsonl_for_a_round_sharded_capture(tmp_path):
    """save_instance_hook's captured doc carries no cost of its own (continual_arc_baselines' InstanceTrace has
    none), so train.load_calls must join it in from the run's own records.jsonl (continual_arc.env.InstanceRecord),
    keyed by (user, round, task_id). A round-sharded capture's records.jsonl lives two directories up from
    round_dir, at the run's own root, not beside the round's own captured/."""
    hook = paper.save_instance_hook(tmp_path, paper.played)
    hook(_fake_trace(user=0, round_=2, task_id="A", solved=True, seed=1))
    hook(_fake_trace(user=1, round_=2, task_id="A", solved=False, seed=2))
    _write_records_jsonl(tmp_path / "records.jsonl",
                         [(0, 2, "A", True, 3.0), (1, 2, "A", False, 12.0),
                          (5, 1, "B", False, 99.0)])   # a different round: must not be picked up

    calls, _ = train.load_calls(paper.round_dir(tmp_path, 2))
    assert {c.meta["user"]: c.meta["cost"] for c in calls} == {0: 3.0, 1: 12.0}


def test_load_calls_reads_cost_from_records_jsonl_beside_a_flat_captured_dir(tmp_path):
    """A plain (unsharded) run directory keeps records.jsonl directly beside its own captured/, not two levels up:
    load_calls must check there first."""
    captured = tmp_path / "captured"
    captured.mkdir()
    _write_flat(captured, _fake_trace(user=0, round_=0, task_id="A", solved=True, seed=1))
    _write_records_jsonl(tmp_path / "records.jsonl", [(0, 0, "A", True, 5.0)])
    calls, _ = train.load_calls(tmp_path)
    assert calls[0].meta["cost"] == 5.0


def test_load_calls_prefers_the_captured_docs_own_cost_over_records_jsonl(tmp_path):
    """If a captured doc ever does carry its own cost (unlike today's save_instance_hook), load_calls must use it
    directly rather than consult records.jsonl at all."""
    captured = tmp_path / "captured"
    captured.mkdir()
    t = _fake_trace(user=0, round_=0, task_id="A", solved=True, seed=1)
    doc = {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
          "phase": t.phase, "solved": t.solved, "cost": 2.0, "turns": t.turns,
          "calls": [{"call_id": c.call_id, "turn": c.turn, "phase": c.phase, "status": c.status,
                     "finish_reason": c.finish_reason, "request": c.request, "response": c.response}
                    for c in t.calls]}
    with gzip.open(captured / "u000.json.gz", "wt") as fp:
        json.dump(doc, fp, default=str)
    (tmp_path / "records.jsonl").write_text(json.dumps({"user": 0, "round": 0, "task_id": "A", "cost": 999.0}) + "\n")
    calls, _ = train.load_calls(tmp_path)
    assert calls[0].meta["cost"] == 2.0


def test_held_out_calls_draws_whole_games_from_every_rounds_held_out_side(tmp_path):
    """The DAgger row's held-out set grows by a round's held-out games every round, and is measured on games drawn
    whole across all of them up to a cap: never a game's calls split between kept and dropped."""
    def side(name, games):
        captured = tmp_path / name / "captured"
        captured.mkdir(parents=True)
        for u, n in enumerate(games):
            calls = [_fake_call(i, [1000 + u, i], [10 + i]) for i in range(n)]
            doc = {"user": u, "task_id": f"{name}-{u}", "solved": True,
                   "calls": [{"call_id": c.call_id, "request": c.request, "response": c.response} for c in calls]}
            with gzip.open(captured / f"u{u:03d}.json.gz", "wt") as fp:
                json.dump(doc, fp)
        return str(tmp_path / name)

    runs = [side("dev-r0000", [2, 2]), side("dev-r0001", [2, 2])]
    assert len(train.held_out_calls(runs)) == 8
    drawn = train.held_out_calls(runs, max_items=3, seed=0)
    games = collections.Counter((c.meta["run"], c.meta["user"]) for c in drawn)
    assert len(drawn) == 4 and sorted(games.values()) == [2, 2]   # two whole games: the cap reached on the second
    assert drawn == train.held_out_calls(runs, max_items=3, seed=0)


def test_load_calls_reads_a_label_set_whose_games_are_no_played_instances(tmp_path):
    """The DAgger row trains (sft) on a round's label set, dagger.write_set's play folder of the expert's replies,
    whose games carry no round and no cost; load_calls must read them as any other calls, with no cost, never
    look one up (a KeyError on 'round' once stopped every label set from loading)."""
    import dagger
    t = _fake_trace(user=0, round_=0, task_id="A", solved=True, seed=1)
    c = t.calls[0]
    point = {"play": "r0000", "user": 0, "task_id": "A", "solved": True, "i": 0, "request": c.request}
    dagger.write_set(tmp_path, [(point, c.response["prompt_token_ids"], c.response["choices"][0]["token_ids"],
                                 {"role": "assistant", "content": "the expert's reply"})])
    calls, _ = train.load_calls(tmp_path)
    assert len(calls) == 1 and calls[0].meta["cost"] is None and calls[0].meta["task_id"] == "A"


# -- the served-adapter check -------------------------------------------------------------------------------

def _reply(tokens: list[str], shift: float = 0.0) -> list[dict]:
    """vLLM's chat ``logprobs.content`` for ``tokens``: each with its top candidates, the chosen one first; ``shift``
    moves every other candidate's log-probability, as a small weight update would."""
    return [{"token": t, "logprob": -0.001,
             "top_logprobs": [{"token": t, "logprob": -0.001}, {"token": "x", "logprob": -7.0 + shift}]}
            for t in tokens]


class _FakeServer:
    """vLLM's chat answer as Qwen3.5 gives it: the reply is all reasoning at 16 tokens, so ``content`` is None
    and only ``logprobs`` carries the tokens."""

    def __init__(self, replies: dict[str, list[dict]]) -> None:
        self.replies = replies
        self.requests: list[dict] = []

    def chat(self, payload):
        self.requests.append(payload)
        return {"choices": [{"message": {"content": None, "reasoning": "Thinking..."},
                             "logprobs": {"content": self.replies[payload["model"]]}}]}


def test_check_served_adapter_flags_a_served_name_that_answers_exactly_like_the_base():
    server = _FakeServer({"base": _reply(["Think", "ing"]), "round5": _reply(["Think", "ing"])})
    out = paper.check_served_adapter(server, "base", "round5", paper.SERVED_ADAPTER_PROMPT)
    assert out == {"base": "Thinking", "served": "Thinking", "answered": True, "logprob_gap_max": 0.0,
                   "differs": False}
    assert [r["model"] for r in server.requests] == ["base", "round5"]
    assert all(r["temperature"] == 0.0 and r["logprobs"] is True and r["top_logprobs"] == paper.SERVED_TOP_LOGPROBS
               for r in server.requests)


def test_check_served_adapter_passes_an_update_that_keeps_the_tokens_but_moves_the_candidates():
    # the case measured on a node: content None both ways, the same 16 tokens, the candidates' log-probabilities moved
    server = _FakeServer({"base": _reply(["Think", "ing"]),
                          "round5": _reply(["Think", "ing"], shift=2 * paper.SERVED_LOGPROB_GAP)})
    out = paper.check_served_adapter(server, "base", "round5", paper.SERVED_ADAPTER_PROMPT)
    assert out["differs"] is True and out["logprob_gap_max"] == pytest.approx(2 * paper.SERVED_LOGPROB_GAP)


def test_check_served_adapter_keeps_a_gap_under_the_bar_as_the_same_weights():
    server = _FakeServer({"base": _reply(["Think", "ing"]),
                          "round5": _reply(["Think", "ing"], shift=paper.SERVED_LOGPROB_GAP / 10)})
    assert paper.check_served_adapter(server, "base", "round5", paper.SERVED_ADAPTER_PROMPT)["differs"] is False


def test_check_served_adapter_passes_when_the_adapter_answers_with_other_tokens():
    server = _FakeServer({"base": _reply(["give", " up"]), "round5": _reply(["ask", " again"])})
    out = paper.check_served_adapter(server, "base", "round5", paper.SERVED_ADAPTER_PROMPT)
    assert out["differs"] is True and out["served"] == "ask again"


def test_check_served_adapter_reports_a_reply_with_no_tokens_as_unanswered():
    server = _FakeServer({"base": _reply(["Think"]), "round5": []})
    out = paper.check_served_adapter(server, "base", "round5", paper.SERVED_ADAPTER_PROMPT)
    assert out["answered"] is False and out["differs"] is False


def _write_adapter(path: Path, b_value: float) -> Path:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file({"base_model.model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4),
                                 "base_model.model.layers.0.q_proj.lora_B.weight": torch.full((4, 2), b_value)},
                                str(path / "adapter_model.safetensors"))
    return path


def test_adapter_is_untrained_reads_whether_every_lora_b_is_zero(tmp_path):
    assert paper.adapter_is_untrained(_write_adapter(tmp_path / "fresh", 0.0)) is True
    assert paper.adapter_is_untrained(_write_adapter(tmp_path / "stepped", 1e-6)) is False


def test_cmd_check_served_skips_an_untrained_adapter_without_asking_the_server(tmp_path, monkeypatch, capsys):
    adapter = _write_adapter(tmp_path / "r0" / "adapter", 0.0)
    fake_module = types.ModuleType("ask_sd.server")
    fake_module.Server = lambda url: pytest.fail("an untrained adapter must not reach the server")
    monkeypatch.setitem(sys.modules, "ask_sd.server", fake_module)
    a = types.SimpleNamespace(vllm_url="http://localhost:8101/v1", base_model="base", served="pg-r0",
                              max_tokens=16, adapter=str(adapter))
    paper.cmd_check_served(a)
    assert "skipped" in json.loads(capsys.readouterr().out)


def test_cmd_check_served_raises_when_the_check_fails(monkeypatch):
    class FakeAskSDServer:
        def __init__(self, url):
            self.url = url

        def chat(self, payload):
            return {"choices": [{"message": {"content": None}, "logprobs": {"content": _reply(["same"])}}]}

    fake_module = types.ModuleType("ask_sd.server")
    fake_module.Server = FakeAskSDServer
    monkeypatch.setitem(sys.modules, "ask_sd.server", fake_module)
    a = types.SimpleNamespace(vllm_url="http://localhost:8101/v1", base_model="base", served="round5", max_tokens=16,
                              adapter=None)
    with pytest.raises(SystemExit, match="answered exactly as the base model"):
        paper.cmd_check_served(a)


def test_cmd_check_served_says_so_when_no_tokens_came_back(monkeypatch):
    class FakeAskSDServer:
        def __init__(self, url):
            self.url = url

        def chat(self, payload):
            return None

    fake_module = types.ModuleType("ask_sd.server")
    fake_module.Server = FakeAskSDServer
    monkeypatch.setitem(sys.modules, "ask_sd.server", fake_module)
    a = types.SimpleNamespace(vllm_url="http://localhost:8101/v1", base_model="base", served="round5", max_tokens=16,
                              adapter=None)
    with pytest.raises(SystemExit, match="no tokens came back"):
        paper.cmd_check_served(a)


# -- play_round against the real CollectiveHarnessLearner/SystemLearner --------------------------------------
# Only the tests below need continual_arc_baselines (play_round's own imports); everything
# above runs without them. A module-level importorskip would skip this whole file wherever they are missing, so the
# check is a plain try/except instead, and only these tests carry the skip.

try:
    from continual_arc_baselines.systems.base import SystemLearner
    HAS_BASELINES = True
except ImportError:
    SystemLearner = object
    HAS_BASELINES = False

needs_baselines = pytest.mark.skipif(not HAS_BASELINES,
                                     reason="needs continual_arc_baselines")

_SYSTEM_LEARNER_PARAMS = (frozenset(inspect.signature(SystemLearner.__init__).parameters) - {"self"}
                         if HAS_BASELINES else frozenset())


class _StubSystem(SystemLearner):
    """A SystemLearner that answers every observation with unparseable text, so SystemLearner.act's own reprompt
    loop ends every instance in a forced give-up with no real model or harness process; and writes a file tagged
    ``round_tag`` into its state directory on every call, so wiping (or not) is directly observable. ``round_tag`` is
    set by the test before each play_round call, not derived from the stub's own state (``_instances_done`` resets
    to 0 every fresh process, so it cannot by itself tell two separate rounds' writes apart). Accepts, and silently
    drops, PrimeAgentLearner's own constructor kwargs (model, context_window, max_model_calls, ...; everything
    SystemLearner.__init__ itself does not name), so play_round's real kwargs construct this in its place unchanged,
    with PrimeAgentLearner monkeypatched to it."""
    name = "stub"
    round_tag = 0

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if k in _SYSTEM_LEARNER_PARAMS})

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def new_session(self) -> None:
        pass

    def ask(self, message, *, reprompt=False):
        state = self.workdir / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / f"round{self.round_tag}.md").write_text("memory")
        return "I don't know what to do."

    def wipe_state(self) -> None:
        import shutil
        shutil.rmtree(self.workdir / "state", ignore_errors=True)


def _tiny_collective_schedule(users=2, rounds=3, task_id="fake000"):
    """A hand-built collective schedule over continual_arc.testing's data-free FakeSource: ``users`` users, one
    instance per round, ``rounds`` rounds, all the same task (its rule does not matter: every instance here ends in
    a forced give-up). Paired with a registry of {"fake": FakeSource(...)}."""
    from continual_arc.schedule import Schedule, Slot

    slots = [Slot(task_id=task_id, instance_seed=r * 1000 + u, phase="stream", user=u)
             for r in range(rounds) for u in range(users)]
    return Schedule(name="wipe-test", seed=0, slots=slots, sources={task_id: "fake"}, users=users)


def _fake_registry(n_tasks=1):
    from continual_arc.tasks import TaskRegistry
    from continual_arc.testing import FakeSource
    return TaskRegistry({"fake": FakeSource(n_tasks=n_tasks)})


def _stream_then_epilogue(users=2):
    """A hand-built schedule shaped as the real one ends: two stream rounds over fake000, then two epilogue rounds in
    which each user meets the held-out fake001 and fake002 once each, in an order of their own. Paired with
    ``_fake_registry(3)``."""
    from continual_arc.schedule import Schedule, Slot

    slots = [Slot(task_id="fake000", instance_seed=r * 1000 + u, phase="stream", user=u)
             for r in range(2) for u in range(users)]
    slots += [Slot(task_id=f"fake00{1 + (r + u) % 2}", instance_seed=r * 1000 + u, phase="epilogue", user=u)
              for r in range(2, 4) for u in range(users)]
    return Schedule(name="epilogue-test", seed=0, slots=slots,
                    sources={f"fake00{i}": "fake" for i in range(3)}, users=users)


def _state_files(run_dir, user):
    state = run_dir / "users" / f"u{user:03d}" / "system" / "state"
    return sorted(p.name for p in state.iterdir()) if state.is_dir() else []


@needs_baselines
def test_play_round_wipes_stateless_rows_state_before_every_round(tmp_path, monkeypatch):
    """The finding this fixes: a play call plays exactly one instance per user, so SystemLearner's own in-process
    wipe (gated on _session_pending, set only at the end of the instance this call plays) never fires across
    separate processes. Three separate play_round calls, over the same run directory, simulate three separate
    `paper.py play` processes exactly (play_round builds a brand new CollectiveHarnessLearner, and so a brand new
    _StubSystem per user, every call; only the on-disk state directory persists between calls, as it does for a
    real harness). For a stateless row, each round's state directory must hold only that round's own file."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _StubSystem)
    full = _tiny_collective_schedule(users=2, rounds=3)
    registry = _fake_registry()
    run_dir = tmp_path / "h0"
    for r in range(3):
        _StubSystem.round_tag = r
        paper.play_round("h0", run_dir, full.first_rounds(r + 1), base_url="http://localhost:1/v1",
                         served_name="base", registry=registry)
        for u in range(2):
            assert _state_files(run_dir, u) == [f"round{r}.md"]   # only this round's file: wiped before it


@needs_baselines
def test_play_round_leaves_row_h_memory_across_rounds(tmp_path, monkeypatch):
    """Row H keeps prime-agent's own memory: played the same three separate calls, its state directory accumulates
    one file per round instead of being wiped."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _StubSystem)
    full = _tiny_collective_schedule(users=2, rounds=3)
    registry = _fake_registry()
    run_dir = tmp_path / "h"
    for r in range(3):
        _StubSystem.round_tag = r
        paper.play_round("h", run_dir, full.first_rounds(r + 1), base_url="http://localhost:1/v1",
                         served_name="base", registry=registry)
        for u in range(2):
            assert _state_files(run_dir, u) == [f"round{i}.md" for i in range(r + 1)]   # every round's, kept


@needs_baselines
def test_play_round_leaves_row_pg_memory_across_rounds(tmp_path, monkeypatch):
    """PG is trained but, unlike OPD, not wiped: a returning puzzle must be able to find what an earlier
    round stored, exactly as row H's memory is kept, so its state directory accumulates one file per round too."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _StubSystem)
    full = _tiny_collective_schedule(users=2, rounds=3)
    registry = _fake_registry()
    run_dir = tmp_path / "pg"
    for r in range(3):
        _StubSystem.round_tag = r
        paper.play_round("pg", run_dir, full.first_rounds(r + 1), base_url="http://localhost:1/v1",
                         served_name="base", registry=registry)
        for u in range(2):
            assert _state_files(run_dir, u) == [f"round{i}.md" for i in range(r + 1)]   # every round's, kept


@needs_baselines
def test_a_trained_row_plays_on_into_the_epilogue_with_its_memory_kept(tmp_path, monkeypatch):
    """A trained row's first epilogue round is a cut one round longer over the same run directory: every user's memory
    carries over from the stream, and the epilogue's games are recorded and captured by round as the stream's are."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _StubSystem)
    full, registry, run_dir = _stream_then_epilogue(users=2), _fake_registry(3), tmp_path / "dagger"
    for r in range(4):
        _StubSystem.round_tag = r
        paper.play_round("dagger", run_dir, full.first_rounds(r + 1), base_url="http://localhost:1/v1",
                         served_name="base", registry=registry)
        for u in range(2):
            assert _state_files(run_dir, u) == [f"round{i}.md" for i in range(r + 1)]   # the stream's memory kept
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in records] == ["stream"] * 4 + ["epilogue"] * 4
    assert [r["task_id"] for r in records[4:]] == ["fake001", "fake002", "fake002", "fake001"]
    for r in (2, 3):
        assert sorted(f.name for f in (paper.round_dir(run_dir, r) / "captured").iterdir()) == ["u000.json.gz",
                                                                                             "u001.json.gz"]


@needs_baselines
def test_play_round_refreshes_meta_json_protocol_each_call(tmp_path, monkeypatch):
    """continual_arc.runner.run reloads meta.json wholesale on a resumed call, freezing its protocol block at the
    first call's cut, while summary.json's is recomputed every time; play_round must rewrite meta.json's from the
    fresh one so a reader of meta.json alone is not misled."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _StubSystem)
    full = _tiny_collective_schedule(users=1, rounds=2)
    registry = _fake_registry()
    run_dir = tmp_path / "h0"
    paper.play_round("h0", run_dir, full.first_rounds(1), base_url="http://localhost:1/v1", served_name="base",
                     registry=registry)
    paper.play_round("h0", run_dir, full.first_rounds(2), base_url="http://localhost:1/v1", served_name="base",
                     registry=registry)
    meta = json.loads((run_dir / "meta.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert meta["protocol"]["deviations"] == summary["protocol"]["deviations"]
    assert "stream cut at round 1" not in " ".join(meta["protocol"]["deviations"])
    assert "stream cut at round 2" in " ".join(meta["protocol"]["deviations"])


@needs_baselines
def test_play_round_writes_driver_json(tmp_path, monkeypatch):
    """A published paper row should carry a record of the settings it played under (loop.py's own driver.json,
    which play_round had no equivalent of): the row, the served model, whether it was wiped, and the harness and
    sampling settings every row plays under."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _StubSystem)
    schedule = _tiny_collective_schedule(users=1, rounds=1)
    run_dir = tmp_path / "h0"
    paper.play_round("h0", run_dir, schedule.first_rounds(1), base_url="http://localhost:1/v1",
                     served_name="round0", registry=_fake_registry())
    driver = json.loads((run_dir / "driver.json").read_text())
    assert driver["row"] == "h0" and driver["served"] == "round0" and driver["stateless"] is True
    assert driver["model"] == paper.loop.MODEL
    assert driver["harness"]["compact_at"] == paper.loop.HARNESS["compact_at"]
    assert driver["harness_budget"]["max_model_calls"] == paper.loop.HARNESS_BUDGETS["block"]["max_model_calls"]
    assert driver["sampling"] == paper.SAMPLING


@needs_baselines
def test_play_round_passes_the_fixed_harness_and_stateless_flag_to_the_system(tmp_path, monkeypatch):
    """The block's harness settings, and the stateless row/row-H split, pinned at the system's own constructor: a
    capturing stand-in for PrimeAgentLearner records every kwarg play_round builds it with."""
    captured: list[dict] = []

    class _Capturing(_StubSystem):
        def __init__(self, **kwargs):
            captured.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _Capturing)
    schedule = _tiny_collective_schedule(users=1, rounds=1)
    registry = _fake_registry()

    paper.play_round("h0", tmp_path / "h0", schedule.first_rounds(1), base_url="http://localhost:1/v1",
                     served_name="base", registry=registry)
    paper.play_round("h", tmp_path / "h", schedule.first_rounds(1), base_url="http://localhost:1/v1",
                     served_name="base", registry=registry)

    h0_kwargs, h_kwargs = captured
    assert h0_kwargs["stateless"] is True and h_kwargs["stateless"] is False
    for kw in (h0_kwargs, h_kwargs):
        assert kw["max_model_calls"] == paper.loop.HARNESS_BUDGETS["block"]["max_model_calls"]
        assert kw["action_budget_s"] == paper.loop.HARNESS_BUDGETS["block"]["action_budget_s"]
        assert kw["compact_at"] == paper.loop.HARNESS["compact_at"] and kw["compact_keep"] == paper.loop.HARNESS["compact_keep"]
        assert kw["end"] == paper.loop.HARNESS["end"]
        assert kw["empty_reply"] == "reprompt"


class _StreamingModel:
    """A model server that streams every reply as vLLM does for ``"stream": true`` (server-sent events: the reply in one
    delta, then its finish), the same unparseable text each time, and keeps every request body it was sent."""

    def __init__(self, reply: str = "I am not sure what the rule is.") -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        model, self.bodies, lock = self, [], threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                with lock:
                    model.bodies.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta, finish in (({"role": "assistant", "content": reply}, None), ({}, "stop")):
                    chunk = {"id": "c", "object": "chat.completion.chunk", "model": body["model"],
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def stop(self) -> None:
        self.server.shutdown()


@needs_baselines
def test_play_round_bare_is_the_bare_model_at_the_floor_settings_on_the_served_model(tmp_path):
    """Row bare, transfer's floor reference: baselines' bare model at every paper floor's settings, asking the served
    base model at the rows' sampling and thinking with the rows' reply cap, every instance a conversation of its own
    and nothing kept between them."""
    model = _StreamingModel()
    schedule = _stream_then_epilogue(users=2)
    schedule.slots = [s for s in schedule.slots if s.phase == "epilogue"]
    run_dir = tmp_path / "bare"
    try:
        paper.play_round("bare", run_dir, schedule, base_url=model.url, served_name=paper.loop.MODEL,
                         registry=_fake_registry(3))
    finally:
        model.stop()
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    assert len(records) == 4 and {r["phase"] for r in records} == {"epilogue"} and not any(r["solved"] for r in records)
    assert model.bodies
    for body in model.bodies:
        assert body["model"] == paper.loop.MODEL and body["stream"] is True and body["max_tokens"] == paper.MAX_TOKENS
        assert {k: body[k] for k in paper.SAMPLING} == paper.SAMPLING
        assert body["chat_template_kwargs"] == paper.loop.TEMPLATE_KWARGS and "return_token_ids" not in body
    assert sum(len(b["messages"]) == 1 for b in model.bodies) == 4   # each instance starts a conversation afresh
    assert not any((run_dir / "users" / f"u{u:03d}" / "system" / "state").exists() and
                   any((run_dir / "users" / f"u{u:03d}" / "system" / "state").iterdir()) for u in range(2))
    driver = json.loads((run_dir / "driver.json").read_text())
    assert driver["row"] == "bare" and driver["floor"] == paper.FLOOR and driver["max_tokens"] == paper.MAX_TOKENS
    assert "harness" not in driver


class _Model:
    """A model server: every chat call it is sent gets a reply of fresh random text, so a game played twice by it differs
    wherever it was asked again, and ``generated`` counts the replies it made."""

    def __init__(self) -> None:
        import threading
        import uuid
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        model, self.generated, lock = self, 0, threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with lock:
                    model.generated += 1
                data = json.dumps({"id": uuid.uuid4().hex, "object": "chat.completion", "model": "m",
                                   "prompt_token_ids": [1, 2, 3],
                                   "choices": [{"index": 0, "token_ids": [4, 5],
                                                "message": {"role": "assistant", "content": f"r-{uuid.uuid4().hex[:8]}"},
                                                "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def stop(self) -> None:
        self.server.shutdown()


class _ModelStub(_StubSystem):
    """A _StubSystem that asks its model: every turn, ``calls_per_turn`` model calls through its proxy, each answering
    the conversation so far (the turn's message, then each reply and a tool result), each reply written to its memory
    (``notes.md`` in its state directory), and then the same unparseable answer, so its instance ends in a forced
    give-up. ``crash`` = (user, n) raises out of that user's n-th call before it is sent, as a node stopped mid-round,
    once ``crash_after`` exists (another user's game saved, say)."""
    calls_per_turn = 2
    crash: tuple[int, int] | None = None
    crash_after: Path | None = None
    made: collections.Counter = collections.Counter()
    proxies: dict = {}

    def ask(self, message, *, reprompt=False):
        import time
        import httpx
        user = int(self.workdir.parent.name[1:])
        _ModelStub.proxies.setdefault(user, set()).add(self.environ["OPENAI_BASE_URL"])
        state = self.workdir / "state"
        state.mkdir(parents=True, exist_ok=True)
        messages = [{"role": "system", "content": "a stub harness"}, {"role": "user", "content": message}]
        for k in range(self.calls_per_turn):
            _ModelStub.made[user] += 1
            if _ModelStub.crash == (user, _ModelStub.made[user]):
                while not _ModelStub.crash_after.exists():
                    time.sleep(0.05)
                raise RuntimeError("the node stops")
            r = httpx.post(self.environ["OPENAI_BASE_URL"] + "/chat/completions", timeout=30,
                           json={"model": "m", "messages": messages})
            reply = r.json()["choices"][0]["message"]["content"]
            with open(state / "notes.md", "a") as fp:
                fp.write(reply + "\n")
            messages += [{"role": "assistant", "content": reply}, {"role": "tool", "content": f"ok {k}"}]
        return "I don't know what to do."


def _captured_game(run_dir, round_, user):
    with gzip.open(paper.round_dir(run_dir, round_) / "captured" / f"u{user:03d}.json.gz", "rt") as fp:
        return json.load(fp)


def _replies(doc):
    return [c["response"]["choices"][0]["message"]["content"] for c in doc["calls"]]


def _notes(run_dir, user):
    return (run_dir / "users" / f"u{user:03d}" / "system" / "state" / "notes.md").read_text().split()


@needs_baselines
def test_a_round_cut_short_is_taken_up_from_its_last_model_call(tmp_path, monkeypatch):
    """Round 1 of a two-user row is cut short while user 1 plays, user 0's game already saved. The next attempt puts
    every user's memory back as the round began, replays user 0's game whole and user 1's to its last call, and has the
    model play user 1's on: the model is asked for the live calls alone, every replayed call is kept as recorded (its
    id and reply), each user's memory ends holding its game once, and the records hold whole rounds."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _ModelStub)
    monkeypatch.setattr(paper, "RESUME_CALLS", True)
    model = _Model()
    full, registry, run_dir = _tiny_collective_schedule(users=2, rounds=2), _fake_registry(), tmp_path / "pg"
    play = lambda: paper.play_round("pg", run_dir, full.first_rounds(2), base_url=model.url,  # noqa: E731
                                    served_name="base", registry=registry)
    try:
        paper.play_round("pg", run_dir, full.first_rounds(1), base_url=model.url, served_name="base",
                         registry=registry)
        before = [_notes(run_dir, u) for u in range(2)]
        _ModelStub.made.clear()
        _ModelStub.proxies.clear()
        _ModelStub.crash, _ModelStub.crash_after = (1, 3), paper.round_dir(run_dir, 1) / "captured" / "u000.json.gz"
        with pytest.raises(RuntimeError, match="the node stops"):
            play()
        first = _captured_game(run_dir, 1, 0)
        cut = [json.load(gzip.open(f, "rt")) for f in sorted((paper.journal_dir(run_dir, 1) / "a1" / "u001").glob("*.gz"))]
        assert len(cut) == 2 and len(_notes(run_dir, 1)) == len(before[1]) + 2
        assert sum(1 for _ in open(run_dir / "records.jsonl")) == 2                   # round 0 alone
        _ModelStub.crash = None
        generated = model.generated
        play()
    finally:
        _ModelStub.crash = None
        model.stop()
    out = json.loads((paper.round_dir(run_dir, 1) / "resume.json").read_text())
    counts = {int(u): c for u, c in out["counts"].items()}
    assert [a["replaying"] for a in out["attempts"]] == [{}, {"0": len(first["calls"]), "1": 2}]
    assert counts[0] == {"replayed": len(first["calls"])}
    assert counts[1]["replayed"] == 2 and counts[1]["handed_off"] == 1 and "drifted" not in counts[1]
    assert model.generated - generated == counts[1]["live"]                           # no generation for a replay
    again = _captured_game(run_dir, 1, 0)                                             # replayed whole, as recorded
    assert again["calls"] == first["calls"]
    second = _captured_game(run_dir, 1, 1)
    assert [c["call_id"] for c in second["calls"][:2]] == [c["call_id"] for c in cut]
    assert _replies(second)[:2] == [c["response"]["choices"][0]["message"]["content"] for c in cut]
    for u, doc in ((0, again), (1, second)):                                          # the memory: its game once
        assert _notes(run_dir, u) == before[u] + _replies(doc)
    assert sum(1 for _ in open(run_dir / "records.jsonl")) == 4
    assert all(len(urls) == 1 for urls in _ModelStub.proxies.values())   # each user's proxy at one port, both attempts
    assert not paper.journal_dir(run_dir, 1).exists()
    assert not any((run_dir / "users" / f"u{u:03d}" / "round-start").exists() for u in range(2))


@needs_baselines
def test_a_resumed_attempt_cut_short_too_hands_the_longest_recording_on(tmp_path, monkeypatch):
    """An attempt that is itself cut while it replays adds nothing to a recording: the third takes up the first's, and
    still replays every call the first made rather than asking the model again."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _ModelStub)
    monkeypatch.setattr(paper, "RESUME_CALLS", True)
    model = _Model()
    schedule, registry, run_dir = _tiny_collective_schedule(users=1, rounds=1), _fake_registry(), tmp_path / "pg"
    play = lambda: paper.play_round("pg", run_dir, schedule.first_rounds(1), base_url=model.url,  # noqa: E731
                                    served_name="base", registry=registry)
    _ModelStub.crash_after = tmp_path
    try:
        for n in (5, 2):   # the first attempt makes 4 calls; the second is cut in its replay, after 1
            _ModelStub.made.clear()
            _ModelStub.crash = (0, n)
            with pytest.raises(RuntimeError, match="the node stops"):
                play()
        _ModelStub.crash = None
        generated = model.generated
        play()
    finally:
        _ModelStub.crash = None
        model.stop()
    out = json.loads((paper.round_dir(run_dir, 0) / "resume.json").read_text())
    assert [a["replaying"] for a in out["attempts"]] == [{}, {"0": 4}, {"0": 4}]
    assert out["counts"]["0"]["replayed"] == 4 and model.generated - generated == out["counts"]["0"]["live"]


@needs_baselines
def test_a_round_already_recorded_is_not_put_back(tmp_path, monkeypatch):
    """Playing a round whose records are whole again (a player restarted after the round, before it marked it done)
    leaves its users' memory as the round left it: there is nothing to resume."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _ModelStub)
    monkeypatch.setattr(paper, "RESUME_CALLS", True)
    model = _Model()
    schedule, registry, run_dir = _tiny_collective_schedule(users=1, rounds=1), _fake_registry(), tmp_path / "pg"
    try:
        for _ in range(2):
            paper.play_round("pg", run_dir, schedule.first_rounds(1), base_url=model.url, served_name="base",
                             registry=registry)
    finally:
        model.stop()
    assert _notes(run_dir, 0) == _replies(_captured_game(run_dir, 0, 0))
    assert [a["attempt"] for a in json.loads((paper.round_dir(run_dir, 0) / "resume.json").read_text())["attempts"]] == [1]


@needs_baselines
def test_with_resume_calls_off_a_round_keeps_no_journal_and_no_snapshot(tmp_path, monkeypatch):
    """Off (the final runs' setting until it is switched on), a round plays as it always has: nothing kept per call,
    no snapshot of the users' memory, nothing replayed."""
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _ModelStub)
    monkeypatch.setattr(paper, "RESUME_CALLS", False)
    model = _Model()
    schedule, run_dir = _tiny_collective_schedule(users=1, rounds=1), tmp_path / "pg"
    try:
        paper.play_round("pg", run_dir, schedule.first_rounds(1), base_url=model.url, served_name="base",
                         registry=_fake_registry())
    finally:
        model.stop()
    rd = paper.round_dir(run_dir, 0)
    assert (rd / "captured" / "u000.json.gz").exists() and model.generated == len(_captured_game(run_dir, 0, 0)["calls"])
    assert not (rd / "journal").exists() and not (rd / "resume.json").exists() and not (rd / "replay.jsonl").exists()
    assert not (run_dir / "users" / "u000" / "round-start").exists()


def test_a_cancelled_call_takes_the_partial_reply_of_the_turn_row_it_was_cut_in(tmp_path):
    """A call the harness cancelled when it cut its turn at the budget is replayed held, streaming first the reply the
    harness had read: the first turn row logged after the call began, among the rows of the attempt it was made in; a
    call cancelled in a row that was not cut, or in the turn the attempt was stopped in, has none."""
    run_dir = tmp_path / "run"
    jd = paper.journal_dir(run_dir, 0)
    (jd / "a1" / "u000").mkdir(parents=True)
    (jd / "attempts.jsonl").write_text(json.dumps({"attempt": 1, "at": 100.0, "replaying": {}}) + "\n")

    def call(cid, started, message):
        doc = {"call_id": cid, "started": started, "status": "completed" if message else "cancelled", "latency_ms": 10,
               "request": {"messages": [{"role": "user", "content": cid}]},
               "response": {"choices": [{"message": {"role": "assistant", "content": cid}}]} if message else None}
        with gzip.open(jd / "a1" / "u000" / f"{started:.6f}-{cid}.json.gz", "wt") as fp:
            json.dump(doc, fp)
    for cid, started, message in (("a", 101.0, True), ("b", 102.0, False), ("c", 110.0, False), ("d", 120.0, False)):
        call(cid, started, message)
    turns = run_dir / "users" / "u000" / "system" / "turns.jsonl"
    turns.parent.mkdir(parents=True)
    rows = [{"t": 50.0, "outcome": {"stop_reason": "budget"}, "reply": "an earlier round's"},
            {"t": 105.0, "outcome": {"stop_reason": "budget"}, "reply": '{"action": "submit", "grid": [[1'},
            {"t": 112.0, "outcome": {"stop_reason": "reply"}, "reply": "done"}]
    turns.write_text("".join(json.dumps(r) + "\n" for r in rows))
    resume = types.SimpleNamespace(dir=jd, run_dir=run_dir, round=0,
                                   attempts=[json.loads((jd / "attempts.jsonl").read_text())])
    calls, ended = paper.Resume.recording(resume, 0)
    assert [(c["call_id"], c.get("partial")) for c in calls] == [
        ("a", None), ("b", '{"action": "submit", "grid": [[1'), ("c", ""), ("d", "")]
    assert not ended                                          # the game was cut: no ended mark in its attempt
    (jd / "a1" / "u000" / "ended").touch()
    assert paper.Resume.recording(resume, 0)[1]


def test_a_replayed_call_held_at_a_cut_is_kept_as_recorded_though_it_ends_with_no_response():
    """A replayed call the harness cancels when it cuts its turn ends with no response, so nothing in it says it was
    replayed: the proxy's own record of what it replayed does, and the call is kept as recorded, in its place."""
    recorded = {"call_id": "rec-42", "started": 100.0, "status": "cancelled", "request": {"messages": []},
                "response": None, "partial": ""}
    proxy = types.SimpleNamespace(replays={"live-7": "rec-42"}, recorded={"rec-42": recorded})
    resume = types.SimpleNamespace(proxy=proxy)
    held = types.SimpleNamespace(call_id="live-7", started=900.0, status="cancelled", response=None, request={})
    made = types.SimpleNamespace(call_id="live-8", started=901.0, status="completed", response={"choices": []},
                                 request={})
    assert paper.Resume.kept(resume, held) is recorded
    assert paper.Resume.kept(resume, made)["call_id"] == "live-8"


def test_an_attempt_that_replayed_a_game_makes_it_anew_only_once_its_replay_went_past_it(tmp_path):
    """The second attempt replayed part of a 3-call recording and made one call of its own, the model's code calling on
    the side, before it was cut: the first attempt's recording stands. Once the second attempt's replay of the game was
    handed over (or drifted), its journal is the game."""
    run_dir = tmp_path / "run"
    jd = paper.journal_dir(run_dir, 0)
    attempts = [{"attempt": 1, "at": 100.0, "replaying": {}}, {"attempt": 2, "at": 200.0, "replaying": {"0": 3}}]
    jd.mkdir(parents=True)
    (jd / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts))

    def call(attempt, cid, started):
        doc = {"call_id": cid, "started": started, "status": "completed", "latency_ms": 10,
               "request": {"messages": [{"role": "user", "content": cid}]},
               "response": {"choices": [{"message": {"role": "assistant", "content": cid}}]}}
        (jd / f"a{attempt}" / "u000").mkdir(parents=True, exist_ok=True)
        with gzip.open(jd / f"a{attempt}" / "u000" / f"{started:.6f}-{cid}.json.gz", "wt") as fp:
            json.dump(doc, fp)
    for cid, started in (("c0", 101.0), ("c1", 102.0), ("c2", 103.0)):
        call(1, cid, started)
    call(2, "c0", 101.0)                                       # replayed, kept as recorded
    call(2, "side", 205.0)                                     # the model's code, live
    resume = types.SimpleNamespace(dir=jd, run_dir=run_dir, round=0, attempts=attempts)
    assert [c["call_id"] for c in paper.Resume.recording(resume, 0)[0]] == ["c0", "c1", "c2"]
    rd = paper.round_dir(run_dir, 0)
    (rd / "replay.jsonl").write_text(json.dumps({"t": 206.0, "user": 0, "handed_off_at": 2}) + "\n")
    assert [c["call_id"] for c in paper.Resume.recording(resume, 0)[0]] == ["c0", "side"]


def test_resume_reads_the_state_and_turns_where_the_learner_keeps_them(tmp_path):
    """Resume finds a user's harness state and turn log by the learner's own layout, which it does not build itself."""
    if not HAS_BASELINES:
        pytest.skip("needs continual_arc_baselines")
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    learner = CollectiveHarnessLearner(lambda u: None, tmp_path, schedule=_tiny_collective_schedule(users=1, rounds=1),
                                       upstream="http://localhost:1/v1")
    assert learner.user_dir(3) == paper.user_dir(tmp_path, 3)
    assert learner.state_dir(3) == paper.user_dir(tmp_path, 3) / "system" / "state"


def test_cmd_play_dispatches_epilogue_vs_stream_by_epilogue_users(monkeypatch):
    """cmd_play plays the epilogue only when --epilogue-users is given; otherwise the stream, at the same round and
    seed, users passed through either way."""
    calls: list[tuple] = []
    monkeypatch.setattr(paper, "play_round", lambda row, run_dir, schedule, **kw: calls.append((row, schedule, kw))
                        or types.SimpleNamespace(records=[], summary={}))

    stream_schedule = object()
    epilogue_schedule = object()
    monkeypatch.setattr(paper, "schedule_cut", lambda round_, seed=paper.SEED, users=paper.PAPER_USERS: stream_schedule)
    monkeypatch.setattr(paper, "epilogue_cut",
                        lambda round_, epilogue_users=paper.DEFAULT_EPILOGUE_USERS, seed=paper.SEED, users=paper.PAPER_USERS:
                        epilogue_schedule)

    wide_schedule = object()
    monkeypatch.setattr(paper, "wide_epilogue_cut",
                        lambda round_, width, epilogue_users, seed=paper.SEED, users=paper.PAPER_USERS: wide_schedule)

    a = types.SimpleNamespace(row="pg", run_dir="/tmp/x", round=3, seed=0, users=paper.PAPER_USERS, epilogue_users=None,
                              width=None, base_url="http://localhost:1/v1", served="base")
    paper.cmd_play(a)
    assert calls[-1][1] is stream_schedule

    a.epilogue_users = 10
    paper.cmd_play(a)
    assert calls[-1][1] is epilogue_schedule

    a.width = 20   # dealt wider only for a row that keeps nothing between instances
    with pytest.raises(SystemExit, match="--width"):
        paper.cmd_play(a)
    for row in paper.TRANSFER_REFERENCES:
        a.row = row
        paper.cmd_play(a)
        assert calls[-1][:2] == (row, wide_schedule)
    a.epilogue_users = None
    with pytest.raises(SystemExit, match="--width"):
        paper.cmd_play(a)


def test_paper_sh_round_dir_matches_paper_pys_round_dir():
    """paper.sh's round_dir comment says "must match it exactly"; nothing enforces that. Lift the one-line shell
    function out (it is a pure function of $1 and $PLAY_DIR, no other setup needed) and render it for a sample
    round, against paper.round_dir rendering the same inputs: a drift here would otherwise train silently on an
    empty directory."""
    script = (Path(__file__).parent / "paper.sh").read_text()
    start = script.index("round_dir() {")
    fn = script[start:script.index("\n", start)]   # a trailing "# ..." comment: must end its own line, not a ;
    out = subprocess.run(["bash", "-c", f'PLAY_DIR=/tmp/play\n{fn}\nround_dir 7'],
                        capture_output=True, text=True, check=True).stdout
    assert out == str(paper.round_dir(Path("/tmp/play"), 7))


def _paper_sh_dirs(**env):
    """paper.sh's PLAY_DIR/TRAIN_DIR/INIT_ADAPTER/ADAPTER_PREFIX/SCHED_FLAGS block, run standalone with the given
    variables (ROW, TRAINED, USERS, EPILOGUE, EPILOGUE_ROUND, EPILOGUE_USERS, WIDTH), printed as a small JSON object. Exercises the real
    shell logic without the rest of paper.sh (serve.sh, train.py, ...), which these four variables never touch."""
    script = (Path(__file__).parent / "paper.sh").read_text()
    start = script.index('SUFFIX=""')
    block = script[start:script.index("\nfi", start) + len("\nfi")]
    # RUNS as paper.sh defaults it, which the block's run folder sits under
    assignments = "\n".join(f'{k}="{v}"' for k, v in {"RUNS": str(Path.home() / "runs"), **env}.items())
    print_vars = ('printf \'{"PLAY_DIR": "%s", "TRAIN_DIR": "%s", "INIT_ADAPTER": "%s", "ADAPTER_PREFIX": "%s", '
                 '"SCHED_FLAGS": "%s"}\' "$PLAY_DIR" "$TRAIN_DIR" "$INIT_ADAPTER" "$ADAPTER_PREFIX" "$SCHED_FLAGS"')
    out = subprocess.run(["bash", "-c", f"{assignments}\n{block}\n{print_vars}"],
                        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_paper_sh_epilogues_init_adapter_is_the_adapter_that_would_have_served_round_r():
    """EPILOGUE_ROUND=R pairs with a stream call's own ROUNDS=R (rounds 0..R-1): the epilogue's round 0 must start
    from train/r(R-1), the newest adapter such a stream run leaves behind, not train/rR, which a ROUNDS=R stream
    never writes (the off-by-one the review caught)."""
    dirs = _paper_sh_dirs(ROW="pg", USERS=100, EPILOGUE=1, EPILOGUE_ROUND=20, EPILOGUE_USERS=10)
    assert dirs["INIT_ADAPTER"] == str(Path.home() / "runs/paper/pg/train/r19/adapter")

    # the smoke's own convention: USERS=4, a ROUNDS=3 stream (rounds 0, 1, 2) pairs with EPILOGUE_ROUND=3 -> r2
    dirs = _paper_sh_dirs(ROW="pg", USERS=4, EPILOGUE=1, EPILOGUE_ROUND=3, EPILOGUE_USERS=2)
    assert dirs["INIT_ADAPTER"] == str(Path.home() / "runs/paper/pg-u4/train-u4/r2/adapter")


def test_paper_sh_plays_transfers_references_wide_each_into_a_folder_of_its_own():
    """The references' paper.sh call (its header): a frozen row's epilogue alone has no round R, so its folder is named
    by its users and width alone, and the width reaches paper.py; a trained row's keeps its round and plays narrow."""
    for row in paper.TRANSFER_REFERENCES:
        dirs = _paper_sh_dirs(ROW=row, TRAINED=0, USERS=4, EPILOGUE=1, EPILOGUE_ROUND=20, EPILOGUE_USERS=4, WIDTH=20)
        assert dirs["PLAY_DIR"] == str(Path.home() / f"runs/paper/{row}-u4/epilogue-u4-w20")
        assert dirs["SCHED_FLAGS"] == "--epilogue-users 4 --users 4 --width 20"
    dirs = _paper_sh_dirs(ROW="pg", TRAINED=1, USERS=4, EPILOGUE=1, EPILOGUE_ROUND=3, EPILOGUE_USERS=2)
    assert dirs["PLAY_DIR"] == str(Path.home() / "runs/paper/pg-u4/epilogue-r3-u4")
    assert dirs["SCHED_FLAGS"] == "--epilogue-users 2 --users 4"


def test_paper_sh_epilogue_users_over_users_is_refused_up_front():
    """epilogue_for_users raises for users > cfg.users; USERS=4 with the default EPILOGUE_USERS=10 must fail at the
    top of paper.sh, not partway into a play."""
    script = (Path(__file__).parent / "paper.sh").read_text()
    start = script.index("# epilogue_for_users refuses")
    guard = script[start:script.index("\nfi", start) + len("\nfi")]
    ok = subprocess.run(["bash", "-c", f'EPILOGUE=1\nUSERS=4\nEPILOGUE_USERS=2\n{guard}\necho survived'],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "survived" in ok.stdout

    refused = subprocess.run(["bash", "-c", f'EPILOGUE=1\nUSERS=4\nEPILOGUE_USERS=10\n{guard}\necho survived'],
                             capture_output=True, text=True)
    assert refused.returncode != 0 and "survived" not in refused.stdout


# -- the OPSD row: paper.py's row, and paper.sh's row-dependent shell logic ------------------------------

def test_paper_py_opsd_replaces_opd_as_the_paper_rows_trained_teacher_scored_row():
    assert paper.ROWS == ("h0", "pg", "opsd", "dagger", "h", "bare")
    assert "opd" not in paper.ROWS
    # opsd, dagger and pg keep prime-agent's own memory (the oracle's replies, which opsd and dagger learn from, coach
    # a program library across rounds): only h0, the control, is wiped
    assert set(paper.STATELESS_ROWS) == {"h0"}


def test_paper_sh_row_refuses_dagger_and_opsd_which_run_in_real_time_and_the_dropped_opd():
    script = (Path(__file__).parent / "paper.sh").read_text()
    start = script.index("case $ROW in h0 | pg | h | bare)")
    guard = script[start:script.index("\n", start)]
    for row, ok in (("h0", True), ("pg", True), ("opsd", False), ("dagger", False), ("h", True), ("bare", True),
                    ("opd", False)):
        out = subprocess.run(["bash", "-c", f'ROW="{row}"\n{guard}\necho survived'], capture_output=True, text=True)
        assert (out.returncode == 0 and "survived" in out.stdout) == ok, (row, out.stdout, out.stderr)


def test_paper_sh_trains_pg_alone_and_labels_nothing():
    """paper.sh's one trained row is PG (realtime.sh runs DAgger and OPSD, labeller and all): rounds() plays every
    round and PG then trains it, a failed play stopping the run; no label, hint or teacher-scoring step is left."""
    script = (Path(__file__).parent / "paper.sh").read_text()
    start = script.index("case $ROW in pg) TRAINED=1")
    block = script[start:script.index("esac", start) + len("esac")]
    for row, trained in (("pg", "1"), ("h0", "0"), ("h", "0"), ("bare", "0")):
        out = subprocess.run(["bash", "-c", f'ROW="{row}"\n{block}\nprintf "%s" "$TRAINED"'],
                             capture_output=True, text=True, check=True).stdout
        assert out == trained, (row, out)
    start = script.index("rounds() {")
    fn = script[start:script.index("\n}", start) + len("\n}")]
    assert 'play_round "$r" || return 1' in fn and fn.index('play_round "$r"') < fn.index('train_round "$r"')
    for gone in ("label_round", "start_relay", "--demos", "hint_watcher", "HINT_", "score_opsd", "teacher_score",
                 "opd_scores", "hints.json"):
        assert gone not in script, gone


def _paper_sh_fn(name):
    """A function of paper.sh, lifted out whole: to its closing brace, or to the line's end for a one-line one."""
    script = (Path(__file__).parent / "paper.sh").read_text()
    start = script.index(f"{name}() {{")
    line = script[start:script.index("\n", start)]
    return line if line.rstrip().endswith("}") or "; }" in line else script[start:script.index("\n}", start) + len("\n}")]


def _train_round_inputs(tmp_path, **env):
    """train_round's own per-objective inputs (what --train-runs reads and the flags beside it), run standalone as
    round 3 in a function of its own; the printed runs and flags."""
    fn = _paper_sh_fn("train_round")
    start = fn.index("    runs=(")
    block = fn[start:fn.index("\n", fn.index("\n", start) + 1)]
    play = tmp_path / "p"
    assignments = "\n".join(f'{k}="{v}"' for k, v in env.items())
    script = (f'round_dir() {{ printf "{play}/rounds/r%04d" "$1"; }}\n{assignments}\nf() {{\nr=3\n'
              f'local runs=() flags=()\n{block}\nprintf "%s|%s" "${{runs[*]}}" "${{flags[*]}}"\n}}\nf')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
    return out.replace(str(play), "/p")


def test_paper_sh_train_round_trains_pg_on_its_rounds_own_games(tmp_path):
    assert _train_round_inputs(tmp_path, PER_CALL=0) == "/p/rounds/r0003|"
    assert _train_round_inputs(tmp_path, PER_CALL=1) == "/p/rounds/r0003|--per-call"


def _train_summary(tmp_path, done):
    """train_round's closing progress line for a DONE file, the real inline Python run as paper.sh runs it."""
    fn = _paper_sh_fn("train_round")
    start = fn.index('    say "train round $r done:')
    line = fn[start:fn.index('2>/dev/null)"', start) + len('2>/dev/null)"')]
    out = tmp_path / "r3"
    out.mkdir(exist_ok=True)
    (out / "DONE").write_text(json.dumps(done))
    return subprocess.run(["bash", "-c", f'say() {{ printf "%s" "$1"; }}\nr=3\nout="{out}"\n{line}'],
                          capture_output=True, text=True, check=True).stdout


def test_paper_sh_train_summary_reads_each_objectives_own_stats(tmp_path):
    opd = {"objective": "opd", "steps": 2, "minutes": 1.0,
           "items": {"calls": 10, "items": 8, "excluded": 1, "unscored": 1, "too_long": 0},
           "last": {"clip_fraction": 0.1, "approx_kl": 0.01, "entropy_mean": 0.3}}
    assert _train_summary(tmp_path, opd) == ("train round 3 done: 8/10 calls kept, 1 excluded, 1 unscored, 0 too long, "
                                             "2 steps, clip 0.1 kl 0.01 entropy 0.3")
    pg = {"objective": "pg", "steps": 2, "minutes": 1.0,
          "games": {"episodes": 3, "unscored": 0, "wins": 2, "items": 40, "too_long": 0,
                    "trained": [{"run": "r0003", "user": 0, "round": 3}, {"run": "r0003", "user": 2, "round": 3}]},
          "last": {"clip_fraction": 0.1, "approx_kl": 0.01, "entropy_mean": 0.3}}
    assert _train_summary(tmp_path, pg) == ("train round 3 done: 2/3 games won and trained on, 40 calls, 2 steps, "
                                            "clip 0.1 kl 0.01 entropy 0.3")


def _play_round_memory_block():
    """paper.sh play_round()'s memory block (F4): the few lines after a round's success that log memory.json's
    summary, lifted out the same way _paper_sh_dirs lifts its own block -- a pure function of $PLAY_DIR, $r, $L
    and round_dir/say, which the memory block alone never needs to drive the real play_round (serve_round,
    check_served, the retry loop) to exercise."""
    script = (Path(__file__).parent / "paper.sh").read_text()
    start = script.index('# memory.py\'s round_report')
    return script[start:script.index("\n}", start)]


def test_paper_sh_memory_block_never_fails_the_round_when_memory_json_is_missing(tmp_path):
    """F4: a missing memory.json (paper._report_round_memory failed and already logged why) must not make
    play_round return non-zero after a round that played and recorded perfectly."""
    play_dir = tmp_path / "play"
    play_dir.mkdir()
    block = _play_round_memory_block()
    script = f'''
round_dir() {{ printf "{play_dir}/rounds/r%04d" "$1"; }}
say() {{ :; }}
run_block() {{
{block}
}}
PLAY_DIR="{play_dir}"
r=0
run_block
'''
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_paper_sh_memory_block_logs_summary_line_when_memory_json_is_present(tmp_path):
    """When memory.json is there, the block's one progress-log line is memory.summary_line's own, not a
    reimplementation of it (F4's smaller finding): run it for real and read what `say` was given."""
    play_dir = tmp_path / "play"
    round_dir = play_dir / "rounds" / "r0000"
    round_dir.mkdir(parents=True)
    report = {"round": 0, "users": {},
             "total": {"memory": {a: {"added": 1, "changed": 0, "removed": 0} for a in
                                  ("programs", "skills", "harness_store", "notes", "workspace_other", "other")},
                       "reads": {"programs": 0, "skills": 0, "harness_store": 0, "notes": 0},
                       "reused_program": 2, "solved_without_demo": 1}}
    (round_dir / "memory.json").write_text(json.dumps(report))
    said = tmp_path / "said.txt"
    block = _play_round_memory_block()
    # the block imports memory via PYTHONPATH="$HERE", paper.sh's own folder: this one, so the real line runs
    # unmodified against the real memory.py
    here = Path(__file__).resolve().parent
    script = f'''
round_dir() {{ printf "{play_dir}/rounds/r%04d" "$1"; }}
say() {{ printf '%s\\n' "$1" >> "{said}"; }}
run_block() {{
{block}
}}
PLAY_DIR="{play_dir}"
HERE="{here}"
r=0
run_block
'''
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    import memory as memory_module
    assert said.read_text().strip() == memory_module.summary_line(report)


def test_load_calls_skips_a_raw_completion_the_models_own_code_sent_through_the_proxy(tmp_path):
    """GRPO's round 1 on an earlier paper run: the model's own sandbox code posted a raw completion (a "prompt", no
    "messages") through the tracking proxy, captured beside the harness's chat calls with a response, and
    train.load_calls crashed on it in render.is_compaction (KeyError: 'messages') on every attempt. Such a call is no
    move of the agent's: it must be skipped, never trained or scored, the other calls of its game kept."""
    import gzip
    import render
    hook = paper.save_instance_hook(tmp_path, paper.played)
    hook(_fake_trace(user=0, round_=1, task_id="A", solved=True, seed=1))
    _write_records_jsonl(tmp_path / "records.jsonl", [(0, 1, "A", True, 3.0)])
    f = paper.round_dir(tmp_path, 1) / "captured" / "u000.json.gz"
    with gzip.open(f, "rt") as fp:
        doc = json.load(fp)
    raw = {"call_id": "raw", "turn": 0, "phase": "play", "status": "ok", "finish_reason": "length",
           "request": {"model": "Qwen/Qwen3.5-9B", "prompt": "def transform(g):"},
           "response": {"id": "x", "choices": [{"text": " return g", "finish_reason": "length"}], "usage": {}}}
    doc["calls"].append(raw)
    with gzip.open(f, "wt") as fp:
        json.dump(doc, fp)

    assert render.is_compaction(raw) is False
    calls, _ = train.load_calls(paper.round_dir(tmp_path, 1))
    assert len(calls) == len(doc["calls"]) - 1   # every chat call kept, the raw completion alone skipped
    assert all(c.meta["i"] != len(doc["calls"]) - 1 for c in calls)


def test_a_body_whose_messages_is_not_a_list_is_never_read_as_a_chat_call(tmp_path):
    """GRPO's round 10 on an earlier paper run: memory.py's _call_text raised KeyError: -1 on a captured call whose
    request held "messages" as a dict, a body the model's own sandbox code posted through the proxy. Every reader of
    captured calls must treat such a body as no chat call: train.load_calls skips it, render.is_compaction and
    overfit.chat_call say no, and memory's _call_text reads it without raising."""
    import gzip
    import memory
    import overfit
    import render
    hook = paper.save_instance_hook(tmp_path, paper.played)
    hook(_fake_trace(user=0, round_=1, task_id="A", solved=True, seed=1))
    _write_records_jsonl(tmp_path / "records.jsonl", [(0, 1, "A", True, 3.0)])
    f = paper.round_dir(tmp_path, 1) / "captured" / "u000.json.gz"
    with gzip.open(f, "rt") as fp:
        doc = json.load(fp)
    odd = {"call_id": "odd", "turn": 0, "phase": "play", "status": "ok", "finish_reason": "stop",
           "request": {"messages": {"role": "user", "content": "grid?"}},
           "response": {"id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}}
    doc["calls"].append(odd)
    with gzip.open(f, "wt") as fp:
        json.dump(doc, fp)

    assert render.is_compaction(odd) is False
    assert overfit.chat_call(odd) is False
    assert "ok" in memory._call_text(odd)
    calls, _ = train.load_calls(paper.round_dir(tmp_path, 1))
    assert len(calls) == len(doc["calls"]) - 1
