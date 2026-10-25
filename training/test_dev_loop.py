"""The off-GPU pieces of the development loop (the earlier experiments' round driver over a development set, not a row
of the paper) and of the trained rows' policy gradient, all on synthetic data (no played run, no GPU needed):

- the development task draw (loop.dev_tasks/dev_task_pool/dev_slots);
- the block's harness option (loop.HARNESS_BUDGETS);
- loop.py compare's forced-give-ups count;
- plain policy gradient on the won games (train.pg_items), what it leaves out and its counts, and a phase with no
  game won (train.fit with train.PolicyGradient);
- the chain finder and batching over PG-shaped items, not just SFT's;
- OPD's items from a fake teacher (train.opd_items, train.load_opd_scores; OPD is not a row of the paper);
- chain_item carrying PG's advantages and OPD's teacher_logps onto their branches.

Run with ``python3 -m pytest test_dev_loop.py -q``; needs ``continual_arc`` on the path, and, for the PG/OPD tests,
an ``ask_sd`` with per-token advantages (``TokenItem.teacher_logp``). No path surgery here (as in
``test_train_chains.py``): whatever ``ask_sd``/``train`` resolve to on ``sys.path`` is what runs, so ``PYTHONPATH``
picks the ask-sd under test. The OPD/chain tests *fail* (not skip) on an ask-sd without per-token advantages, since a
quietly skipped chain test can let a regression in the policy gradient's chain path through.
"""
import argparse
import collections
import json
import random
import types

import pytest

import loop

Call = pytest.importorskip("ask_sd.calls").Call
TokenItem = pytest.importorskip("ask_sd.trainer").TokenItem
train = pytest.importorskip("train")

HAS_PER_TOKEN_ADVANTAGES = "teacher_logp" in TokenItem.__dataclass_fields__


def _require_opd_trainer() -> None:
    """Fails (never skips) a test that needs ask-sd's per-token advantages (``TokenItem.teacher_logp``) when they
    are missing: see the module docstring for pointing ``PYTHONPATH`` at an ask-sd that has them."""
    if not HAS_PER_TOKEN_ADVANTAGES:
        pytest.fail("needs ask-sd's per-token advantages: TokenItem.teacher_logp is missing from the ask_sd on sys.path")


# -- the development task draw --------------------------------------------------------------------------

def test_dev_tasks_are_25_training_tasks_off_the_stream_and_every_epilogue():
    from continual_arc.configs import CANONICAL_SEEDS, build_schedule, get_config, stream_task_pool

    dev = loop.dev_tasks()
    assert len(dev) == len(set(dev)) == loop.DEV_TASKS
    stream_tasks = {x["task_id"] for x in loop.stream_slots(0)}
    assert not set(dev) & stream_tasks

    epilogues = {s.task_id for name in ("arc1-small", "arc1-medium") for seed in CANONICAL_SEEDS
                 for s in build_schedule(name, "workstreams", seed).slots if s.phase == "epilogue"}
    pool = set(stream_task_pool(get_config("arc1-small")))   # Re-ARC: ARC-1 training tasks
    # the training pool and every epilogue (ARC-1 evaluation tasks) never overlap by dataset alone, so "dev avoids
    # every epilogue" can't fail on its own; what can fail is dev_task_pool's own stream filter, checked directly
    assert not pool & epilogues
    assert set(loop.dev_task_pool()) == pool - stream_tasks
    assert not set(dev) & epilogues

    assert loop.dev_tasks() == dev   # a seeded sample of its own: the same draw every call


def test_dev_slots_are_stream_slots_shape_over_the_dev_tasks():
    dev = set(loop.dev_tasks())
    r0, r1 = loop.dev_slots(0), loop.dev_slots(1)
    assert len(r0) == 100 and {x["task_id"] for x in r0} == dev
    assert all(sum(x["task_id"] == t for x in r0) == 4 for t in dev)
    assert {x["task_id"] for x in r1} == dev
    seeds0 = {(x["task_id"], x["instance_seed"]) for x in r0}
    seeds1 = {(x["task_id"], x["instance_seed"]) for x in r1}
    assert not seeds0 & seeds1 and len(seeds0) == 100   # fresh instances every round
    assert sum((loop.dev_slots(0, user=u) for u in range(4)), []) == r0


# -- the block's harness option --------------------------------------------------------------------------

def test_harness_block_raises_the_action_and_follow_up_seconds_just_under_the_hard_ceiling():
    assert loop.HARNESS_BUDGETS["default"] == {}
    assert loop.HARNESS_BUDGETS["block"] == {
        "max_model_calls": 40, "action_budget_s": 1740.0, "reprompt_budget_s": 1740.0, "reprompt_max_model_calls": 6}
    # strictly under PrimeAgentLearner's default timeout_s (1,800.0): at or above it, the hard ceiling would fire
    # first and abort the turn with a restart instead of the budget's graceful decide-now
    budget = loop.HARNESS_BUDGETS["block"]
    assert budget["action_budget_s"] < 1800.0 and budget["reprompt_budget_s"] < 1800.0
    # no key of the block option overrides HARNESS's own (compact_at, compact_keep, end): the two are recorded and
    # applied side by side, never one replacing the other
    assert not set(loop.HARNESS_BUDGETS["block"]) & set(loop.HARNESS)


# -- compare's forced give-ups -----------------------------------------------------------------------------

def _write_records(run_dir, forced_give_ups):
    run_dir.mkdir(parents=True)
    with open(run_dir / "records.jsonl", "w") as fp:
        for i, forced in enumerate(forced_give_ups):
            rec = {"task_id": "t", "instance_seed": i, "cost": 1, "solved": False, "gave_up": bool(forced),
                   "demo_requests": 0, "wrong_attempts": 0,
                   "resources": {"llm_requests": 10, "system_parse_failures": 0, "system_forced_give_ups": forced}}
            fp.write(json.dumps(rec) + "\n")


def test_compare_counts_forced_give_ups_over_resources(tmp_path):
    base, new = tmp_path / "base", tmp_path / "new"
    _write_records(base, [0, 1, 0])
    _write_records(new, [1, 1, 2])
    out = loop.compare(base, new)
    assert out["base"]["forced_give_ups"] == 1
    assert out["new"]["forced_give_ups"] == 4


# -- plain policy gradient on the won games (train.pg_items) --------------------------------------------------

def _call(run, user, round_, task_id, i, solved, seed, cost=16.0):
    g = random.Random(seed)
    prompt = [g.randrange(1, 900) for _ in range(10)]
    reply = [g.randrange(1, 900) for _ in range(4)]
    return Call({}, prompt, reply, "stop", {"run": run, "user": user, "round": round_, "task_id": task_id, "i": i,
                                           "solved": solved, "cost": cost})


def _episode(run, user, round_, task_id, solved, cost=16.0, n_calls=2, seed=0):
    return [_call(run, user, round_, task_id, i, solved, seed * 100 + i, cost) for i in range(n_calls)]


def test_pg_items_train_the_won_games_alone_each_call_at_advantage_one():
    """REINFORCE with no baseline: every call of a won game is an item with advantage 1, its ids the prompt as served
    and the reply as played; a failed game's advantage is 0, so it pushes nothing and none of its calls is an item;
    a game never scored is left out and counted."""
    calls = (_episode("r0003", 0, 3, "D", True, cost=3.0, n_calls=3) + _episode("r0003", 1, 3, "E", False, seed=1)
             + [_call("r0003", 2, 3, "F", 0, None, seed=2)])
    tok = types.SimpleNamespace(eos_token_id=None)
    items, stats = train.pg_items(tok, calls, max_seq_len=10**9)

    won = [c for c in calls if c.meta["user"] == 0]
    assert [it.advantage for it in items] == [1.0] * 3
    assert [it.student_ids for it in items] == [list(c.prompt_ids) + list(c.reply_ids) for c in won]
    assert {k: stats[k] for k in ("episodes", "unscored", "wins", "items", "too_long")} == {
        "episodes": 3, "unscored": 1, "wins": 1, "items": 3, "too_long": 0}
    assert stats["trained"] == [{"run": "r0003", "user": 0, "round": 3}]


def test_pg_items_read_neither_the_task_nor_the_cost():
    """A game's advantage is its outcome alone: two wins alike in everything but their task and cost give the same
    items, and a failure gives none whatever it cost."""
    tok = types.SimpleNamespace(eos_token_id=None)
    cheap, _ = train.pg_items(tok, _episode("r", 0, 2, "A", True, cost=1.0), max_seq_len=10**9)
    dear, _ = train.pg_items(tok, _episode("r", 0, 2, "Z", True, cost=16.0), max_seq_len=10**9)
    assert [it.advantage for it in cheap] == [it.advantage for it in dear] == [1.0, 1.0]
    assert [it.student_ids for it in cheap] == [it.student_ids for it in dear]
    for cost in (8.0, 16.0):
        assert train.pg_items(tok, _episode("r", 0, 2, "A", False, cost=cost), max_seq_len=10**9)[0] == []


def test_pg_items_skip_a_call_longer_than_the_window():
    tok = types.SimpleNamespace(eos_token_id=None)
    items, stats = train.pg_items(tok, _episode("r", 0, 0, "A", True, n_calls=3), max_seq_len=13)
    assert items == [] and stats["too_long"] == 3 and stats["wins"] == 1


def test_train_pg_with_no_game_won_takes_no_step_and_keeps_the_adapter(tmp_path):
    """Before any game has won there is nothing to learn from: no item, no step, and the adapter is saved as it was,
    for the next round to play again."""
    saved = []
    trainer = _StubTrainer()
    trainer.save = saved.append
    calls = [c for u in range(4) for c in _episode("r", u, 1, "A", False, seed=u)]
    a = _opd_train_args(objective="pg")
    items, games = train.pg_items(trainer.tokenizer, calls, a.max_seq_len)
    done = train.fit(a, trainer, train.PolicyGradient(a, train.Objective.pg, items, games, "games"), tmp_path, None,
                     random.Random(0))
    assert done["steps"] == 0 and done["games"]["wins"] == 0 and done["games"]["episodes"] == 4
    # saved beside OUT/adapter, then put in its place
    assert saved == [tmp_path / "adapter.tmp"] and (tmp_path / "adapter").is_dir() and (tmp_path / "DONE").exists()


# -- the chain batching, over PG-shaped items ---------------------------------------------------------------

def _pg_games(n_users=16, calls=20, seed=0, run="r"):
    """Fake PG plays shaped like a real round (round 1, every user's game won, so every game is trained): ``n_users``
    users, every user's game a chain of ``calls`` calls, each prompt extending the last except every 9th
    (test_train_chains._games's own break point: a rewrite starts a new chain), so a game splits into several chains,
    as 76-83% of real consecutive calls do, not one chain per game by construction."""
    g = random.Random(seed)
    calls_out = []
    for user in range(n_users):
        prompt = [1000 + user]
        for i in range(calls):
            if i and i % 9 == 0:
                prompt = [2000 + user, i]   # rewritten: a new chain starts
            prompt = prompt + [g.randrange(1, 900) for _ in range(g.randrange(5, 40))]
            reply = [g.randrange(1, 900) for _ in range(g.randrange(2, 8))]
            calls_out.append(Call({}, prompt, reply, "stop",
                                  {"run": run, "user": user, "round": 1, "task_id": f"t{user % 4}", "i": i,
                                   "solved": True, "cost": float(1 + user % 4)}))
            prompt = prompt + reply
    g.shuffle(calls_out)
    return calls_out


def test_pg_items_chain_and_batch_the_same_way_sft_items_do():
    """find_chains and chain_batches don't care that an item is PG's (an advantage, and a round/task_id/
    solved in its label beside run/user/i): a round's PG items chain and batch exactly as sft_items' do
    (test_train_chains.py), every step whole chain pieces, every call used once a pass, far fewer sequences read than
    calls -- on imperfect chains (periodic breaks), so the "far fewer" claim is the batcher's doing, not true of any
    chain by construction."""
    _require_opd_trainer()   # chain_item(piece) below needs ChainItem.advantages: pg needs it too, not only opd
    calls = _pg_games()
    tok = types.SimpleNamespace(eos_token_id=None)
    items, stats = train.pg_items(tok, calls, max_seq_len=10**9)
    assert stats["wins"] == 16 and stats["items"] == len(items) == len(calls)   # every game won

    chains = train.find_chains(items)
    assert sum(len(c) for c in chains) == len(items)
    assert len(chains) > 16   # more than one chain per game: the periodic break actually broke them
    for c in chains:
        assert len({x.label["user"] for x in c}) == 1
        assert [x.label["i"] for x in c] == sorted(x.label["i"] for x in c)

    steps = train.chain_batches(chains, 64, 8, 2, 100_000, random.Random(0))
    seen = collections.Counter()
    assert sum(sum(len(p) for p in step) != 64 for step in steps) <= 2   # at most one short step per pass
    for step in steps:
        assert sum(len(p) for p in step) <= 64
        for piece in step:
            assert 1 <= len(piece) <= 8
            keys = [(x.label["user"], x.label["i"]) for x in piece]
            assert len({u for u, _ in keys}) == 1
            ci = train.chain_item(piece)   # every piece is a valid PG chain, the game's advantage on every branch
            assert ci.advantages == [piece[0].advantage] * len(piece) and ci.teacher_logps is None
            seen.update(id(x) for x in piece)
    assert len(seen) == len(items) and set(seen.values()) == {2}   # two passes: every call exactly twice
    lengths = [len(p) for s in steps[:-1] for p in s]
    assert sum(lengths) / len(lengths) > 2   # chains, not single calls: the batcher reads fewer sequences than calls


# -- OPD's items from a fake teacher -----------------------------------------------------------------------

def _opd_call(run, user, i, n_reply):
    return Call({}, [1, 2, 3], list(range(10, 10 + n_reply)), "length", {"run": run, "user": user, "i": i})


def test_load_opd_scores_keys_by_run_user_i(tmp_path):
    run_dir = tmp_path / "round-play1"
    run_dir.mkdir()
    (run_dir / "opd_scores.json").write_text(json.dumps({"model": "teacher", "student_model": "student",
                                                          "stats": {}, "items": [
        {"run": "r", "user": 0, "i": 0, "scores": [1.0, None]},
        {"run": "r", "user": 0, "i": 1, "scores": None}]}))
    scores = train.load_opd_scores(run_dir)
    assert scores[("r", 0, 0)] == [1.0, None]
    assert scores[("r", 0, 1)] is None
    assert train.load_opd_scores(run_dir, "student") == scores   # the matching student: no refusal
    with pytest.raises(ValueError, match="student"):
        train.load_opd_scores(run_dir, "a-different-student")


def test_load_opd_scores_with_no_recorded_student_is_not_refused(tmp_path):
    """A file from before this field existed, or one its scorer wrote without the student's name, has nothing to
    check against: load_opd_scores(..., model) must not refuse it on that account alone."""
    run_dir = tmp_path / "round-play1"
    run_dir.mkdir()
    (run_dir / "opd_scores.json").write_text(json.dumps({"model": "teacher", "stats": {},
                                                          "items": [{"run": "r", "user": 0, "i": 0, "scores": [1.0]}]}))
    assert train.load_opd_scores(run_dir, "any-student") == {("r", 0, 0): [1.0]}


def test_opd_items_carry_the_teachers_scores_and_drop_unscored_or_long_calls():
    _require_opd_trainer()
    calls = [_opd_call("r", 0, 0, 2), _opd_call("r", 0, 1, 2), _opd_call("r", 1, 0, 2)]
    scores = {("r", 0, 0): [1.0, -0.5], ("r", 0, 1): [-2.0, -3.0], ("r", 1, 0): None}   # user 1's chain failed to score
    tok = types.SimpleNamespace(eos_token_id=None)
    items, stats = train.opd_items(tok, calls, scores, max_seq_len=10**9)
    assert stats == {"calls": 3, "items": 2, "excluded": 0, "unscored": 1, "too_long": 0}
    by_key = {(it.label["user"], it.label["i"]): it for it in items}
    assert by_key[(0, 0)].teacher_logp == [1.0, -0.5] and by_key[(0, 1)].teacher_logp == [-2.0, -3.0]
    assert (1, 0) not in by_key
    assert all(it.advantage is None for it in items)   # OPD items carry no PG-style advantage

    too_long = train.opd_items(tok, calls, scores, max_seq_len=1)[1]
    assert too_long["too_long"] == 2 and too_long["items"] == 0   # every scored call is now past the window


def test_opd_items_excludes_a_call_never_given_to_the_teacher_without_blaming_it():
    """A call with no key in ``scores`` at all (one too long for the scorer's window) is not the same as one
    the teacher was asked and failed on (``scores[key] is None``): the first is ``excluded``, by design, and never
    counted toward ``unscored``, the signal that the teacher itself is dead or dying."""
    _require_opd_trainer()
    calls = [_opd_call("r", 0, 0, 2), _opd_call("r", 0, 1, 2), _opd_call("r", 0, 2, 2)]
    scores = {("r", 0, 0): [1.0, -0.5], ("r", 0, 1): None}   # call 2 has no key at all: never attempted
    tok = types.SimpleNamespace(eos_token_id=None)
    items, stats = train.opd_items(tok, calls, scores, max_seq_len=10**9)
    assert stats == {"calls": 3, "items": 1, "excluded": 1, "unscored": 1, "too_long": 0}


# -- OPD's phase refuses on a dead or partly-dead teacher (train.opd_build) ------------------------------------

class _StubTrainer:
    """Just enough of ask_sd's Trainer for an OPD phase to reach its items (and, if it does not refuse, to finish a
    zero-step round): no real model, no computation."""

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(eos_token_id=None)

    def load(self, path):
        pass

    def reset(self):
        pass

    def configure_behaviour(self, adapter=None, *, base=False):
        pass

    def save(self, path):
        pass


def _opd_train_args(**overrides):
    defaults = dict(dry_run=False, model="stub", objective="opd", per_call=False, chain_slack=train.CHAIN_SLACK,
                    chain_calls=8, batch=64, gpus=8, epochs=1, max_steps=None, max_seq_len=10**9, max_minutes=1.0, eval_steps=0,
                    patience=0)
    return argparse.Namespace(**{**defaults, **overrides})


def _train_opd(calls, scores, out):
    """An OPD phase as train.py's --objective opd runs one: opd_build's items, refused on a dead teacher, into fit."""
    a, trainer = _opd_train_args(), _StubTrainer()
    items, stats = train.opd_build(a, calls, scores)(trainer.tokenizer)
    return train.fit(a, trainer, train.PolicyGradient(a, train.Objective.opd, items, stats, "items"), out, None,
                     random.Random(0))


def test_train_opd_refuses_when_every_call_is_unscored(tmp_path, monkeypatch):
    calls = [_opd_call("r", 0, 0, 2)]
    with pytest.raises(SystemExit, match="no items to train on"):
        _train_opd(calls, {}, tmp_path)
    assert not (tmp_path / "DONE").exists()   # refused before anything was saved


def test_train_opd_refuses_when_too_many_calls_are_unscored(tmp_path, monkeypatch):
    calls = [_opd_call("r", 0, i, 2) for i in range(100)]
    # every call present (every one asked), 1 of 100 scored: 99% unscored, far over OPD_UNSCORED_MAX_FRAC -- a key
    # simply missing from scores would mean "never asked" (excluded), not "the teacher failed on it" (unscored)
    scores = {("r", 0, 0): [1.0, -0.5], **{("r", 0, i): None for i in range(1, 100)}}
    with pytest.raises(SystemExit, match="unscored"):
        _train_opd(calls, scores, tmp_path)
    assert not (tmp_path / "DONE").exists()


def test_train_opd_proceeds_when_the_unscored_share_is_within_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "plan", lambda units, a, rng, chains=True: [])   # no real step: prove no refusal
    calls = [_opd_call("r", 0, i, 2) for i in range(100)]
    # every call present (every one asked), 1 of 100 came back None: 1% unscored, within OPD_UNSCORED_MAX_FRAC
    scores = {("r", 0, i): [1.0, -0.5] for i in range(99)} | {("r", 0, 99): None}
    _train_opd(calls, scores, tmp_path)   # does not raise
    assert json.loads((tmp_path / "DONE").read_text())["steps"] == 0


def test_train_opd_counts_excluded_calls_apart_from_unscored_in_its_own_refusal(tmp_path, monkeypatch):
    """A round where most of the gap is calls never asked of the teacher (excluded, by design) and only a
    handful are genuine teacher failures must refuse on the genuine failures' own share, not the combined gap: a
    round the teacher never actually failed on must not refuse."""
    monkeypatch.setattr(train, "plan", lambda units, a, rng, chains=True: [])
    calls = [_opd_call("r", 0, i, 2) for i in range(100)]
    # 6 calls never asked (no key at all: excluded) and every one actually asked came back scored (0 unscored) --
    # the combined gap (6%) would wrongly trip a 2% check computed over every call load_calls kept
    scores = {("r", 0, i): [1.0, -0.5] for i in range(6, 100)}
    _train_opd(calls, scores, tmp_path)   # does not raise
    assert json.loads((tmp_path / "DONE").read_text())["items"]["excluded"] == 6


# -- chain_item carries PG's advantages and OPD's teacher_logps ------------------------------------------

def _games_with_signal(n_games, calls, kind=None, seed=0):
    """A game's calls, each prompt extending the last (:func:`test_train_chains._games`'s shape), every item carrying a
    PG-style scalar advantage (``kind="pg"``), an OPD-style per-token ``teacher_logp`` (``kind="opd"``), or neither
    (plain, as :func:`sft_items` builds them)."""
    g = random.Random(seed)
    items = []
    for game in range(n_games):
        prompt = [1000 + game]
        for i in range(calls):
            prompt = prompt + [g.randrange(1, 900) for _ in range(g.randrange(5, 40))]
            reply = [g.randrange(1, 900) for _ in range(g.randrange(2, 8))]
            kwargs = {"advantage": 0.75} if kind == "pg" else {"teacher_logp": [-1.0] * len(reply)} if kind == "opd" else {}
            items.append(TokenItem(prompt + reply, prompt + reply, len(reply), label={"run": "r", "user": game, "i": i},
                                   **kwargs))
            prompt = prompt + reply
    return items


def test_chain_item_carries_pg_advantages_and_opd_teacher_logps_onto_every_branch():
    _require_opd_trainer()
    pg_chain = train.find_chains(_games_with_signal(1, 6, "pg"))[0]
    item = train.chain_item(pg_chain)
    assert item.advantages == [0.75] * len(pg_chain) and item.teacher_logps is None

    opd_chain = train.find_chains(_games_with_signal(1, 6, "opd"))[0]
    item2 = train.chain_item(opd_chain)
    assert item2.teacher_logps == [[-1.0] * c.n for c in opd_chain] and item2.advantages is None


def test_chain_item_gives_each_branch_its_own_calls_advantage_not_the_games_whole_list():
    """A game's calls split into several chains (only 76-83% of consecutive calls chain: a rewrite breaks one, as
    test_train_chains._games's periodic break does), so a chain holds only some of a game's calls, never all of them.
    Each call here carries a distinct advantage; a chain_item that read the whole game's list instead of the one chain
    passed to it, or misaligned a branch with another chain's call, would be caught by a value out of place."""
    _require_opd_trainer()
    g = random.Random(0)
    items, prompt = [], [1000]
    for i in range(27):   # three chains of 9 calls each (the break point test_train_chains._games uses)
        if i and i % 9 == 0:
            prompt = [2000, i]   # rewritten: a new chain starts
        prompt = prompt + [g.randrange(1, 900) for _ in range(g.randrange(5, 40))]
        reply = [g.randrange(1, 900) for _ in range(g.randrange(2, 8))]
        items.append(TokenItem(prompt + reply, prompt + reply, len(reply), label={"run": "r", "user": 0, "i": i},
                               advantage=float(i)))
        prompt = prompt + reply
    chains = train.find_chains(items)
    assert len(chains) == 3
    for chain in chains:
        ci = train.chain_item(chain)
        assert ci.advantages == [float(c.label["i"]) for c in chain]   # each branch keeps its own call's value


def _quirk_games_with_signal(kind, n_games=3, calls=12, seed=0):
    """test_train_chains._quirk_games' shape (every call's own prompt ends with a marker the next prompt renders
    differently, the one-token quirk the slack is for), every item carrying a PG-style scalar advantage
    (``kind="pg"``) or an OPD-style per-token ``teacher_logp`` (``kind="opd"``)."""
    g = random.Random(seed)
    items = []
    for game in range(n_games):
        history = [3000 + game]
        for i in range(calls):
            history = history + [g.randrange(10, 900) for _ in range(g.randrange(5, 30))]
            prompt = history + [5]
            reply = [g.randrange(10, 900) for _ in range(g.randrange(2, 6))]
            kwargs = {"advantage": 0.75} if kind == "pg" else {"teacher_logp": [-1.0] * len(reply)}
            items.append(TokenItem(prompt + reply, prompt + reply, len(reply), label={"run": "q", "user": game, "i": i},
                                   **kwargs))
            history = history + [6] + reply
    return items


@pytest.mark.parametrize("kind", ["pg", "opd"])
def test_a_chain_with_heads_still_gives_each_branch_its_own_signal(kind):
    """The quirk the chain slack is for (a reply with no reasoning renders in the next prompt as an empty think
    block, a one-token difference the slack keeps in the chain as a head): chain_item must still give each branch its
    own call's advantage or teacher_logp, not the trunk's, not another branch's — the off-by-one a head makes easy."""
    _require_opd_trainer()
    for chain in train.find_chains(_quirk_games_with_signal(kind)):
        for piece in train.chain_pieces(chain, 5, 10_000):
            ci = train.chain_item(piece)
            assert ci.heads == [1] * (len(piece) - 1) + [0]
            for k, c in enumerate(piece):
                assert list(ci.reply(k)) == list(c.student_ids[-c.n:])
                if kind == "pg":
                    assert ci.advantage(k) == c.advantage
                else:
                    assert ci.teacher_logp(k) == c.teacher_logp and len(ci.teacher_logp(k)) == c.n


def test_chain_item_leaves_advantages_and_teacher_logps_none_for_plain_items():
    """sft_items' items carry neither: chain_item then builds ChainItem(trunk, branches, labels, heads) exactly. On an
    ask-sd whose ChainItem has no advantages or teacher_logps fields, that is a ChainItem with no such attributes at all
    (getattr's default), not an explicit None; on one with them, an explicit None either way."""
    chain = train.find_chains(_games_with_signal(1, 6))[0]
    item = train.chain_item(chain)
    assert getattr(item, "advantages", None) is None and getattr(item, "teacher_logps", None) is None


def _opd_games(n_games=8, calls=40, seed=0, run="r"):
    """Fake OPD plays shaped like a real round: ``n_games`` games, each a chain of ``calls`` calls, each prompt
    extending the last except every 9th (test_train_chains._games's own break point: a rewrite starts a new chain), so
    a game splits into several chains, as 76-83% of real consecutive calls do; a scores dict covering every call
    (:func:`train.opd_items`'s input, as the teacher would score a round it failed to score none of)."""
    g = random.Random(seed)
    calls_out, scores = [], {}
    for game in range(n_games):
        prompt = [1000 + game]
        for i in range(calls):
            if i and i % 9 == 0:
                prompt = [2000 + game, i]   # rewritten: a new chain starts
            prompt = prompt + [g.randrange(1, 900) for _ in range(g.randrange(5, 40))]
            reply = [g.randrange(1, 900) for _ in range(g.randrange(2, 8))]
            calls_out.append(Call({}, prompt, reply, "length", {"run": run, "user": game, "i": i}))
            scores[(run, game, i)] = [-1.0] * len(reply)
            prompt = prompt + reply
    g.shuffle(calls_out)
    return calls_out, scores


def test_opd_items_chain_and_batch_the_same_way_sft_items_do():
    """find_chains and chain_batches read opd_items' teacher_logp-carrying items the same way as sft_items' and
    pg_items' (test_train_chains.py, test_pg_items_chain_and_batch_the_same_way_sft_items_do): every step whole
    chain pieces, every call used once a pass, far fewer sequences read than calls — on imperfect chains (periodic
    breaks), so the "far fewer" claim is the batcher's doing, not true of any chain by construction."""
    _require_opd_trainer()
    calls, scores = _opd_games()
    tok = types.SimpleNamespace(eos_token_id=None)
    items, stats = train.opd_items(tok, calls, scores, max_seq_len=10**9)
    assert stats == {"calls": len(calls), "items": len(calls), "excluded": 0, "unscored": 0, "too_long": 0}

    chains = train.find_chains(items)
    assert sum(len(c) for c in chains) == len(items)
    assert len(chains) > 8   # more than one chain per game: the periodic break actually broke them
    for c in chains:
        assert len({x.label["user"] for x in c}) == 1
        assert [x.label["i"] for x in c] == sorted(x.label["i"] for x in c)

    steps = train.chain_batches(chains, 64, 8, 2, 100_000, random.Random(0))
    seen = collections.Counter()
    assert sum(sum(len(p) for p in step) != 64 for step in steps) <= 2   # at most one short step per pass
    for step in steps:
        assert sum(len(p) for p in step) <= 64
        for piece in step:
            assert 1 <= len(piece) <= 8
            ci = train.chain_item(piece)   # every piece is a valid OPD chain, teacher_logps carried through
            assert ci.teacher_logps == [[-1.0] * c.n for c in piece] and ci.advantages is None
            seen.update(id(x) for x in piece)
    assert len(seen) == len(items) and set(seen.values()) == {2}   # two passes: every call exactly twice
    lengths = [len(p) for s in steps[:-1] for p in s]
    assert sum(lengths) / len(lengths) > 2   # chains, not single calls: the batcher reads fewer sequences than calls


# -- train.plan: chains by default, --per-call the flat fallback ------------------------------------------------

def test_the_plan_is_single_calls_with_per_call_and_pieces_of_chains_by_default():
    """train.plan (every row's, pg's and opd's among them) gives steps of single calls with
    --per-call, and chain_batches' steps of chain pieces otherwise, the default."""
    items = _games_with_signal(4, 10, "pg")   # 4 games of 10 calls, each one unbroken chain

    a = argparse.Namespace(epochs=1, batch=8, max_steps=None, chain_slack=train.CHAIN_SLACK, chain_calls=1, gpus=8,
                           max_seq_len=10**9)
    per_call = [step for _, step in train.plan(items, a, random.Random(0), chains=False)]
    assert sum(len(s) for s in per_call) == len(items) and all(len(p) == 1 for s in per_call for p in s)
    assert all(isinstance(p[0], TokenItem) for s in per_call for p in s)

    chain_steps = [step for _, step in train.plan(items, a, random.Random(0))]
    assert sum(len(p) for s in chain_steps for p in s) == len(items)
    assert all(isinstance(p, list) and p and isinstance(p[0], TokenItem) for s in chain_steps for p in s)   # pieces
