"""A game's calls read as chains. Each call of a game sends the conversation so far, so each prompt extends the one
before until a compaction summary rewrites the conversation. A run of such calls is one chain: the longest prompt is
its trunk, and every call's reply is a branch off the trunk at the end of its own prompt, so the trainer reads the
trunk once for all of them (:mod:`continual_arc_training.branching`) instead of once a call.

An item (:class:`TokenItem`) is one call's prompt and the reply it trains on, as token ids. :func:`find_chains` groups
a phase's items into chains, :func:`chain_pieces` cuts a chain into pieces a GPU can take within the window, and
:func:`chain_item` makes a piece one :class:`ChainItem`, the trainer's input."""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from typing import Any

# A call joins the chain of the call before it when its prompt begins with all but at most this many of that call's
# prompt ids. A reply with no reasoning renders in the next prompt as an empty think block, whose newlines tokenize as
# one id where the earlier prompt ended with one, so a prompt can leave the trunk in its last few ids; such a call
# branches at the last id the two share and reads the rest of its prompt ungraded (ChainItem.heads).
CHAIN_SLACK = 8
# The most calls of one chain a piece holds. Pieces are cut by this count and the window, never by the number of GPUs,
# so a phase's steps are the same on any number of them and its checkpoint resumes on any share of a node's pool
# (gpupool.py). A piece's tokens are bounded by the window either way.
CHAIN_CALLS = 16


@dataclass
class TokenItem:
    """One call's prompt and the reply it trains on, as token ids: ``student_ids`` the prompt then the reply, and
    ``teacher_ids`` the teacher's prompt then the same reply (the student's own for every update but OPSD's), the reply
    being the last ``n`` ids of both. ``label`` names the call (its game's run and user, and its place ``i``).
    ``advantage`` is the policy gradient's weight on the reply, and ``behaviour_logp`` the reply's token
    log-probabilities under the adapter that played it."""

    student_ids: list[int]
    teacher_ids: list[int]
    n: int
    label: dict[str, Any] = field(default_factory=dict)
    advantage: float | None = None
    behaviour_logp: list[float] | None = None


@dataclass
class ChainItem:
    """Several replies graded at prompts that each begin with the one before: ``trunk``, the longest of the prompts,
    and per branch its point (the length of the prompt it was graded at, ``trunk[:point]``) and its ids. One sequence
    passes them all: the trunk once, each branch after its point, seeing nothing past it.

    ``heads``: per branch, how many of its first ids end its prompt rather than begin its reply (none when empty), for
    a prompt that leaves the trunk in its last ids: its prompt is ``trunk[:point] + ids[:head]``. ``labels``,
    ``advantages`` and ``behaviour_logps``: per branch, as :class:`TokenItem`'s fields."""

    trunk: list[int]
    branches: list[tuple[int, list[int]]]
    labels: list[dict[str, Any]] = field(default_factory=list)
    heads: list[int] = field(default_factory=list)
    advantages: list[float] | None = None
    behaviour_logps: list[list[float]] | None = None

    def head(self, k: int) -> int:
        """How many of branch ``k``'s first ids are its prompt's, not its reply's."""
        return self.heads[k] if self.heads else 0

    def reply(self, k: int) -> list[int]:
        """Branch ``k``'s graded ids: its reply."""
        return self.branches[k][1][self.head(k):]

    def advantage(self, k: int) -> float | None:
        return self.advantages[k] if self.advantages else None

    def behaviour_logp(self, k: int) -> list[float] | None:
        return self.behaviour_logps[k] if self.behaviour_logps else None

    def length(self) -> int:
        """The packed sequence's length: the trunk and every branch."""
        return len(self.trunk) + sum(len(ids) for _, ids in self.branches)

    def per_call_tokens(self) -> int:
        """What its calls would read one by one: each prompt and its reply."""
        return sum(point + len(ids) for point, ids in self.branches)


def common_prefix(a: list[int], b: list[int]) -> int:
    """How many ids two lists share from their start."""
    n, i = min(len(a), len(b)), 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def find_chains(items: list[TokenItem], slack: int = CHAIN_SLACK) -> list[list[TokenItem]]:
    """``items`` as chains: within each game (an item's ``run`` and ``user``), in call order (``i``), a call joins the
    chain of the call before it when its prompt is at least as long and begins with that call's prompt but for at most
    its last ``slack`` ids. Every item is in exactly one chain, in order."""
    games: dict[tuple, list[TokenItem]] = collections.defaultdict(list)
    for it in items:
        games[(it.label.get("run"), it.label.get("user"))].append(it)
    chains: list[list[TokenItem]] = []
    for key in sorted(games, key=str):
        prev = None
        for it in sorted(games[key], key=lambda x: x.label.get("i", 0)):
            prompt = it.student_ids[:-it.n]
            base = len(prev) - slack if prev is not None else 0
            if prev is not None and len(prompt) >= len(prev) and base > 0 and prompt[:base] == prev[:base]:
                chains[-1].append(it)
            else:
                chains.append([it])
            prev = prompt
    return chains


def chain_item(calls: list[TokenItem], slack: int = CHAIN_SLACK) -> ChainItem:
    """Consecutive calls of one chain as one :class:`ChainItem`: the last call's prompt is the trunk, and each call's
    reply a branch at the last id its prompt shares with the trunk, the rest of its prompt (at most ``slack`` ids)
    carried ungraded before it. Each call's advantage and behaviour log-probabilities go onto its branch."""
    last = calls[-1]
    trunk = list(last.student_ids[:-last.n])
    branches, heads = [], []
    for c in calls:
        prompt = c.student_ids[:-c.n]
        point = max(0, len(prompt) - slack)
        if trunk[:point] != list(prompt[:point]):
            raise ValueError("chain_item: a call's prompt leaves the chain's last prompt by more than the slack")
        while point < len(prompt) and trunk[point] == prompt[point]:
            point += 1
        if point == 0:
            raise ValueError("chain_item: a call's prompt shares no id with the chain's last prompt")
        branches.append((point, list(prompt[point:]) + list(c.student_ids[-c.n:])))
        heads.append(len(prompt) - point)
    extra: dict[str, Any] = {}
    if any(c.advantage is not None for c in calls):
        extra["advantages"] = [c.advantage for c in calls]
    if any(c.behaviour_logp is not None for c in calls):
        extra["behaviour_logps"] = [c.behaviour_logp for c in calls]
    return ChainItem(trunk, branches, [c.label for c in calls], heads if any(heads) else [], **extra)


def chain_pieces(chain: list[TokenItem], max_calls: int, max_len: int,
                 slack: int = CHAIN_SLACK) -> list[list[TokenItem]]:
    """A chain cut at call boundaries into pieces of at most ``max_calls`` calls whose pass (the last call's prompt and
    every reply, each earlier one with a head of at most ``slack`` ids) is at most ``max_len`` tokens."""
    pieces: list[list[TokenItem]] = []
    cur: list[TokenItem] = []
    for c in chain:
        cand = cur + [c]
        size = len(c.student_ids) - c.n + sum(x.n for x in cand) + slack * (len(cand) - 1)
        if cur and (len(cand) > max_calls or size > max_len):
            pieces.append(cur)
            cand = [c]
        cur = cand
    if cur:
        pieces.append(cur)
    return pieces


def as_chains(items: list[TokenItem], max_calls: int, max_len: int, slack: int = CHAIN_SLACK) -> list[ChainItem]:
    """``items``' chains in pieces each GPU can take within the window, one :class:`ChainItem` a piece, and their items
    in the same order: how a held-out set or a scoring pass reads them."""
    return [chain_item(piece, slack) for chain in find_chains(items, slack)
            for piece in chain_pieces(chain, max_calls, max_len, slack)]


def chain_batches(chains: list[list[TokenItem]], batch: int, cap: int, max_len: int, rng: random.Random,
                  slack: int = CHAIN_SLACK) -> list[list[list[TokenItem]]]:
    """One pass over the chains' calls in steps of ``batch`` calls, each step made of whole chains drawn at random: the
    chains are shuffled (never single calls: two calls of one chain in one step share their prefix, two calls drawn
    apart share nothing), each is cut into pieces of at most ``cap`` calls within the window (:func:`chain_pieces`), and
    the pieces fill the steps in order, a piece split at a call boundary where a step fills up. Every call is in exactly
    one step, every step holds ``batch`` calls but the last, and a step is a list of pieces."""
    steps: list[list[list[TokenItem]]] = []
    order = chains[:]
    rng.shuffle(order)
    step: list[list[TokenItem]] = []
    room = batch
    for chain in order:
        for piece in chain_pieces(chain, cap, max_len, slack):
            while piece:
                take, piece = piece[:room], piece[room:]
                step.append(take)
                room -= len(take)
                if room == 0:
                    steps.append(step)
                    step, room = [], batch
    if step:
        steps.append(step)
    return steps


def teacher_chain_item(items: list[TokenItem], chain: ChainItem) -> ChainItem:
    """The teacher's chain over ``chain`` (:func:`chain_item` of the same ``items``): the same trunk, and each item's
    branch at the last id its teacher prompt shares with the trunk, the rest of that prompt (the demonstration and what
    follows it to the reply) carried ungraded as its head before the same reply."""
    branches, heads = [], []
    for it in items:
        t_prompt = list(it.teacher_ids[:-it.n])
        at = min(common_prefix(chain.trunk, t_prompt), len(t_prompt) - 1)
        branches.append((at, t_prompt[at:] + list(it.teacher_ids[-it.n:])))
        heads.append(len(t_prompt) - at)
    return ChainItem(chain.trunk, branches, [it.label for it in items], heads)


def distill_pieces(chain: list[TokenItem], max_calls: int, max_len: int,
                   slack: int = CHAIN_SLACK) -> list[list[TokenItem]]:
    """:func:`chain_pieces` for self-distillation: pieces whose student pass (as chain_pieces counts it) and teacher
    pass (the same trunk, and every reply after the rest of its teacher prompt) are both at most ``max_len`` tokens."""
    rest = {id(it): len(it.teacher_ids) - it.n - common_prefix(it.student_ids[:-it.n], it.teacher_ids[:-it.n]) + slack
            for it in chain}
    pieces: list[list[TokenItem]] = []
    cur: list[TokenItem] = []
    for c in chain:
        cand = cur + [c]
        trunk = len(c.student_ids) - c.n
        size = trunk + sum(x.n for x in cand) + max(slack * (len(cand) - 1), sum(rest[id(x)] for x in cand))
        if cur and (len(cand) > max_calls or size > max_len):
            pieces.append(cur)
            cand = [c]
        cur = cand
    if cur:
        pieces.append(cur)
    return pieces


def distill_chains(items: list[TokenItem], max_calls: int, max_len: int,
                   slack: int = CHAIN_SLACK) -> tuple[list[ChainItem], list[ChainItem]]:
    """Self-distillation items (each with its teacher prompt) as the trainer's distillation takes them: their games'
    chains cut so that both passes fit the window (:func:`distill_pieces`), each piece's chain and its teacher chain."""
    students, teachers = [], []
    for chain in find_chains(items, slack):
        for piece in distill_pieces(chain, max_calls, max_len, slack):
            student = chain_item(piece, slack)
            students.append(student)
            teachers.append(teacher_chain_item(piece, student))
    return students, teachers
