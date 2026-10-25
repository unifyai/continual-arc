"""A game's calls as chains: chains found in a game's calls, pieces within the window, steps built from whole chains,
and a phase's plan the same on any number of GPUs."""
import argparse
import collections
import random

from continual_arc_training import checkpoints, phase
from continual_arc_training.chains import (
    CHAIN_SLACK,
    TokenItem,
    chain_batches,
    chain_item,
    chain_pieces,
    distill_chains,
    find_chains,
)


def _games(n_games=6, calls=30, seed=0, advantage=None):
    """Fake training calls: per game, prompts that each extend the last, except at a 'summary' every 9 calls, where the
    prompt starts over; each call's reply is a few ids of its own."""
    g = random.Random(seed)
    items = []
    for game in range(n_games):
        prompt = [1000 + game]
        for i in range(calls):
            if i and i % 9 == 0:
                prompt = [2000 + game, i]                      # rewritten: a new chain starts
            prompt = prompt + [g.randrange(1, 900) for _ in range(g.randrange(5, 40))]
            reply = [g.randrange(1, 900) for _ in range(g.randrange(2, 8))]
            items.append(TokenItem(prompt + reply, prompt + reply, len(reply), label={"run": "r", "user": game, "i": i},
                                   advantage=advantage(i) if advantage else None))
    g.shuffle(items)
    return items


def test_chains_follow_each_game_in_call_order_and_break_where_the_prompt_starts_over():
    items = _games()
    chains = find_chains(items)
    assert sum(len(c) for c in chains) == len(items)
    assert len(chains) == 6 * 4                                # 30 calls cut at 9, 18, 27
    for c in chains:
        assert len({x.label["user"] for x in c}) == 1
        assert [x.label["i"] for x in c] == sorted(x.label["i"] for x in c)
        for a, b in zip(c, c[1:], strict=False):
            pa, pb = a.student_ids[:-a.n], b.student_ids[:-b.n]
            assert pb[:len(pa)] == pa


def test_a_chain_item_puts_each_reply_at_its_own_prompts_end():
    chain = find_chains(_games(1, 5))[0]
    item = chain_item(chain)
    last = chain[-1]
    assert item.trunk == last.student_ids[:-last.n]
    for (point, reply), c in zip(item.branches, chain, strict=True):
        assert item.trunk[:point] + reply == c.student_ids
    assert item.advantages is None and item.behaviour_logps is None   # plain items carry neither


def test_a_chain_item_gives_each_branch_its_own_calls_advantage_and_behaviour():
    """A game splits into several chains: each branch keeps its own call's advantage and behaviour log-probabilities,
    never another chain's or another branch's."""
    items = _games(1, 27, advantage=float)
    for it in items:
        it.behaviour_logp = [-float(it.label["i"])] * it.n
    chains = find_chains(items)
    assert len(chains) == 3
    for chain in chains:
        ci = chain_item(chain)
        assert ci.advantages == [float(c.label["i"]) for c in chain]
        assert [ci.behaviour_logp(k) for k in range(len(chain))] == [c.behaviour_logp for c in chain]


def test_steps_are_whole_chain_pieces_with_every_call_once_a_pass():
    items = _games(8, 40)
    chains = find_chains(items)
    rng = random.Random(0)
    for batch, cap in ((64, 8), (16, 2), (10, 4)):
        steps = chain_batches(chains, batch, cap, 10_000, rng) + chain_batches(chains, batch, cap, 10_000, rng)
        seen = collections.Counter()
        # every step holds `batch` calls but each pass's last: at most one short step per pass
        assert sum(sum(len(p) for p in step) != batch for step in steps) <= 2
        for step in steps:
            assert sum(len(p) for p in step) <= batch
            for piece in step:
                assert 1 <= len(piece) <= cap
                keys = [(x.label["user"], x.label["i"]) for x in piece]
                assert len({u for u, _ in keys}) == 1
                assert [i for _, i in keys] == list(range(keys[0][1], keys[0][1] + len(keys)))
                seen.update(id(x) for x in piece)
        assert len(seen) == len(items) and set(seen.values()) == {2}   # two passes: every call exactly twice
        full = [s for s in steps if sum(len(p) for p in s) == batch]
        assert sum(len(s) for s in full) < sum(sum(len(p) for p in s) for s in full)   # fewer sequences than calls


def test_steps_never_draw_single_calls_from_different_places():
    """A step's pieces are runs of consecutive calls of one chain, so a step of 64 calls reads a few long pieces."""
    steps = chain_batches(find_chains(_games(8, 40)), 64, 8, 10_000, random.Random(1))
    lengths = [len(p) for s in steps[:-1] for p in s]
    assert sum(lengths) / len(lengths) > 4


def test_a_phases_plan_and_its_checkpoints_fingerprint_are_the_same_on_any_number_of_gpus():
    """A run's share of its node's pool changes between phases and within one, so a phase's plan must not depend on its
    GPUs: the same calls give the same steps, pieces and all, and a checkpoint taken on 6 GPUs resumes on 3."""
    items = _games(8, 40)
    plans, prints = [], set()
    for gpus in (1, 3, 6):
        a = argparse.Namespace(epochs=2, batch=16, gpus=gpus, chain_calls=4, chain_slack=8, max_seq_len=10_000,
                               max_steps=None)
        plans.append(phase.plan(items, a, random.Random(7)))
        prints.add(checkpoints.fingerprint(a, None, items))
    assert plans[0] == plans[1] == plans[2] and len(prints) == 1
    assert max(len(piece) for _, step in plans[0] for piece in step) == 4
    assert {p for p, _ in plans[0]} == {1, 2}


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
            items.append(TokenItem(prompt + reply, prompt + reply, len(reply), label={"run": "q", "user": game, "i": i},
                                   advantage=0.75))
            history = history + [6] + reply
    return items


def test_a_prompt_that_differs_in_its_last_ids_stays_in_the_chain_with_a_head():
    items = _quirk_games()
    assert len(find_chains(items, slack=0)) == len(items)       # without slack every call is its own chain
    chains = find_chains(items)
    assert len(chains) == 3 and sum(len(c) for c in chains) == len(items)
    for chain in chains:
        for piece in chain_pieces(chain, 5, 10_000):
            item = chain_item(piece)
            assert item.heads == [1] * (len(piece) - 1) + [0]          # the trunk's own call needs no head
            for k, (c, (point, ids)) in enumerate(zip(piece, item.branches, strict=True)):
                assert item.trunk[:point] + ids == c.student_ids and item.reply(k) == c.student_ids[-c.n:]
                assert item.advantage(k) == c.advantage
            assert item.length() <= len(item.trunk) + sum(c.n + 1 for c in piece)
    try:
        chain_item(chains[0][:3], slack=0)                       # a head past the slack is refused
    except ValueError as error:
        assert "slack" in str(error)
    else:
        raise AssertionError("a head past the slack was taken")


def test_a_piece_never_passes_the_window():
    for chain in find_chains(_games(2, 30)):
        for piece in chain_pieces(chain, 64, 300):
            assert chain_item(piece).length() <= 300 or len(piece) == 1


def test_a_distillation_piece_fits_both_passes_and_its_teacher_chain_carries_each_demonstration_as_a_head():
    """distill_chains: each piece's student pass and teacher pass both fit the window (but a call too long for it alone,
    a piece of its own), and the teacher chain's branch for each call is the rest of its teacher prompt (the
    demonstration and what follows it) before the same reply."""
    g = random.Random(4)
    items, prompt = [], [7]
    for i in range(12):
        prompt = prompt + [g.randrange(10, 900) for _ in range(g.randrange(20, 40))] + [8, 9]   # 8, 9: the turn's end
        reply = [g.randrange(10, 900) for _ in range(g.randrange(3, 9))]
        demo = [g.randrange(900, 999) for _ in range(15)]
        teacher = prompt[:-2] + demo + prompt[-2:]
        items.append(TokenItem(prompt + reply, teacher + reply, len(reply), label={"run": "d", "user": 0, "i": i}))
        prompt = prompt + reply
    students, teachers = distill_chains(items, 16, 400, CHAIN_SLACK)
    assert len(students) > 1 and sum(len(s.branches) for s in students) == len(items)
    assert any(len(s.branches) > 1 for s in students)
    seen = 0
    for s, t in zip(students, teachers, strict=True):
        assert (max(s.length(), t.length()) <= 400 or len(s.branches) == 1) and t.trunk == s.trunk
        for k in range(len(s.branches)):
            it = items[seen + k]
            point, ids = t.branches[k]
            assert t.trunk[:point] + ids == it.teacher_ids and t.reply(k) == s.reply(k) == it.student_ids[-it.n:]
        seen += len(s.branches)
