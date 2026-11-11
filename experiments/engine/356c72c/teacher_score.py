"""The OPD teacher's log-probability of the student's own tokens, read from a vLLM server (``serve_teacher.sh``).

ask-sd loads only one model (``trainer.py:204``), so the teacher runs as its own server and is asked with
``prompt_logprobs``, as ``ask_sd.thoughts.thought_logprob`` already does for a thought's sum and mean; this reads one
log-probability per reply token instead, for every reply token the student actually wrote, ``None`` where the server
gives none. Imports only ask-sd's client (``ask_sd.server.Server``, ``ask_sd.calls.Call``) and this repo's ``train``
module, for the one definition of a game's chains (``train.find_chains``, the same ``--chain-slack`` the trainer
reads with).

Scoring a call:

- :func:`score_call` sends one ``/v1/completions`` request per call: the prompt ids then the reply ids (with the
  end-of-turn token appended when the reply ended on ``stop`` or ``tool_calls`` and the server left it out, as
  ``ask_sd.calls.call_items`` grades it: :func:`train.played_reply`), ``max_tokens: 1``, ``prompt_logprobs`` set.

Scoring a game: about 76% of a game's calls start with the previous call's prompt and reply (``train.find_chains``).
:func:`score_items` groups a run's items into chains exactly as ``train.py --chains`` does, and :func:`score_chain`
reads each chain with one request over its last (longest) call, slicing every earlier call's reply out of the one
response. A call whose own prompt differs from the chain's trunk in its last few tokens (16 Oct: a reply with no
reasoning renders in the next prompt as an empty think block, whose newlines tokenize as one id where the earlier
prompt ended with one, so the trunk and this call's own prompt agree only up to ``--chain-slack`` ids of its end) is
a "branch": scored on its own, with its own prompt in front of it instead of the trunk's -- 65% of a round's calls,
by the 16 Oct measurement, almost all this one quirk, each resending its whole sequence (the trunk's shared part
included) for want of a server that would answer the shared part from what it already computed for the trunk.

``--read-prefix-cache`` asks the server to answer that shared part from its KV/state prefix cache instead (the
teacher server's own ``TEACHER_PREFIX_CACHE``, ``serve_teacher.sh``). vLLM 0.30.0's ``prompt_logprobs`` answer is
*always* the full prompt's length, hit or no hit: a cache hit of ``k`` tokens leaves rows ``1..k`` as uninitialised
memory rather than shortening the list (verified against the live server and the source: ``v1/outputs.py``'s
``LogprobsTensors.empty_cpu`` allocates for the whole prompt with ``torch.empty``, and ``gpu_model_runner.py``
only ever writes the chunk starting at ``num_computed_tokens``). So a hit cannot be *detected* by length after the
fact if it reaches into the reply -- it must be prevented by construction:

1. every request of a chain that reads the cache -- the trunk (which only ever writes) and every branch whose read
   :func:`score_chain` allows -- carries a ``cache_salt`` (:func:`_cache_salt`: a fresh uuid per
   :func:`score_items` call, or minted by :func:`score_chain` itself if none was given, plus a short hash of the
   chain's own trunk label), so a branch can only ever hit blocks this chain itself wrote in this run: never
   another chain's, and never a previous run's against a warm server;
2. before letting a branch read, :func:`score_chain` computes ``d``, its longest common prefix with the trunk and
   with every other item in the chain (seqs are known upfront, from the round's own captured calls, so this needs
   no request), and only sets ``vllm_xargs: {"read_prefix_cache": 1}`` when ``d <= len(prompt) - 1`` -- the real,
   block-aligned hit length is always `<= d` (content-addressed caching cannot hit further than the sequences
   genuinely agree), so that bound guarantees the hit never reaches ``len(prompt)``, the reply's first position.
   A branch that diverges inside its own reply (``d >= len(prompt)``, notably one scored after its trunk's request
   itself failed: its own full sequence can then be a literal prefix of a trunk that is already cached) is sent
   without the flag, computed in full, and counted (``cache_read_attempts``).
3. a backstop detector still runs: :func:`_slice_tail` refuses unless the answer is the full prompt's length, and
   a read whose reply still contains any ``None`` (with ``prompt_logprobs: 1``, the target token's own row always
   has a logprob when the position was actually computed, so a missing one means it wasn't) is redone as a plain
   standalone request, never reading the cache, and counted (``cache_fallbacks``). The scores themselves are exact
   either way construction or the backstop lets through: reading the cache changes which GPU work the server
   skips, never the maths behind a token's log-probability.

Trunk requests never set the read flag (every other item in the chain needs every one of its rows, which a hit
would leave uninitialised), but still carry the chain's own ``cache_salt`` and write to the cache regardless, since
writing was never gated on ``skip_reading_prefix_cache`` -- only reading was. With vLLM's DP router load-balancing
requests across replicas rather than routing by prefix, a chain's branches would otherwise mostly land on a
replica that never saw the trunk; every request a chain's read touches also carries ``X-data-parallel-rank``,
pinning it to one replica (a stable hash of the chain key, mod ``--dp-size``, read directly off the server's own
``/metrics`` when not given rather than trusted from the flag: see :func:`_dp_size_from_metrics`).
``ask_sd.server.Server.completion`` has no way to attach either header, so that path uses a small client of this
module's own (``_PinnedServer``); off by default, and the off path keeps ask-sd's own client, untouched.

Before scoring, :func:`check_tokenizer` refuses a teacher whose tokenizer does not give the student's ids: the
vocabulary and merges must agree for every id the student's tokenizer uses; the teacher may add further tokens above
that (Qwen3.8-27B's 7 audio tokens at ids 248070-248076, which the student never writes).

Usage (on a node with the teacher server up; reads one or more runs' calls chain by chain and prints tokens read,
requests sent and tokens per second; with ``--out``, also writes every item's (run, user, i) and its teacher scores,
the student model scored for, the run's own stats (``branch_requests``/``branch_tokens``/``cache_read_attempts``/
``cache_fallbacks`` included), seconds taken, whether ``--read-prefix-cache`` was set, and, when the server's
``/metrics`` was reachable, the prefix-cache tokens queried and hit over the run, to that path (atomically), for
round.sh's OPD scoring step, an OPD training step to read once it builds its items, and ``teacher_cache_check.py``'s
comparison; exits non-zero when more than :data:`TEACHER_UNSCORED_MAX_FRAC` of a round's calls came back unscored,
a dead or partly-dead teacher's only trace otherwise):
    uv run python teacher_score.py --vllm-url http://localhost:8102/v1 --model Qwen/Qwen3.8-27B \\
        --train-runs ~/runs/loop/r0-stream [--chain-slack 8] [--max-seq-len 65536] [--workers 32] \\
        [--read-prefix-cache [--dp-size 4]] [--out SCORES.json]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

log = logging.getLogger("teacher_score")

SCORE_WORKERS = 32  # a round of 100 games is about 2,000 chains; vLLM only batches requests that arrive together
BRANCH_WORKERS = 8  # a chain's own branch pool (score_chain): bounds how many of one chain's branches queue on the
                    # server at once, independent of score_items' outer --workers chains in flight

TEACHER_MAX_SEQ_LEN = 131_072  # the plan's context floor: the longest recorded call is 129,482 tokens; a caller that
                               # trains on a shorter window (train.py's --max-seq-len, 65,536 by default) should pass
                               # it here too, so the teacher never scores calls the trainer then throws away
# a chain whose scoring fails scores every one of its calls None (score_items); above this share of a round's calls,
# main refuses (a non-zero exit) rather than let a dead or partly-dead teacher pass as a quietly smaller round
TEACHER_UNSCORED_MAX_FRAC = 0.02


# -- scoring -----------------------------------------------------------------------------------------------------

# score_chain's own per-chain stats, zeroed: the shape score_items reduces over, and what a chain that raised
# before making any request is charged.
#   requests/prompt_tokens: every request sent, trunk's included.
#   branch_requests/branch_tokens: every request and prompt token spent on a branch item, a fallback retry
#     included -- so they always total requests/prompt_tokens minus one request and len(trunk) tokens (the trunk's).
#   cache_read_attempts: a branch's own first request that set the read flag, however it turned out.
#   cache_fallbacks: a cache_read_attempt redone as a plain standalone request because the backstop detector found
#     it did not cover the whole reply. Always 0 when read_prefix_cache is false.
_CHAIN_STATS_ZERO = {"requests": 0, "prompt_tokens": 0, "branch_requests": 0, "branch_tokens": 0,
                     "cache_read_attempts": 0, "cache_fallbacks": 0}


class _PinnedServer:
    """A minimal ``/v1/completions`` client for the ``--read-prefix-cache`` path only: ``ask_sd.server.Server`` has
    no way to attach a header, and DP pinning (``X-data-parallel-rank``, :func:`_dp_rank`) needs one on every
    request a chain's read touches, the trunk's write included -- it must land on the replica its branches will
    read from. The ``off`` path keeps using ``ask_sd.server.Server``, completely unchanged.

    Retries like ``ask_sd.server.Server`` does (``server.py``'s own ``_ask``): a 5xx or a connection failure is
    transient and retried with a flat backoff; an HTTP error below 500 is a configuration mistake, not a dropped
    connection, and is returned at once, logged, never retried -- an out-of-range ``X-data-parallel-rank``, an
    oversized or malformed ``cache_salt``, or a ``--vllm-url`` missing its ``/v1`` all answer with a 4xx, and
    without this a flat 30-retry/20s-backoff default turns any of them into a silent, request-by-request,
    multi-minute hang instead of an error in the log within seconds. ``base_url`` is accepted with or without a
    trailing ``/v1``, the same as ``ask_sd.server.Server``'s own convention."""

    def __init__(self, base_url: str) -> None:
        base = base_url.rstrip("/")
        self.root = base[:-3] if base.endswith("/v1") else base

    def completion(self, payload: dict[str, Any], *, headers: dict[str, str] | None = None,
                    retries: int = 30, backoff: float = 20.0) -> dict[str, Any] | None:
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        data = json.dumps(payload).encode()
        model = payload.get("model")
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(self.root + "/v1/completions", data=data, headers=req_headers,
                                             method="POST")
                with urllib.request.urlopen(req, timeout=1800) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as error:
                body = error.read()[:200]
                if error.code < 500 or attempt == retries:
                    log.warning("completion failed for %s: HTTP %s %s", model, error.code, body)
                    return None
                log.warning("completion for %s: HTTP %s %s; retry %d in %.0fs", model, error.code, body,
                           attempt + 1, backoff)
            except (urllib.error.URLError, OSError) as error:
                if attempt == retries:
                    log.warning("completion failed for %s: %s", model, error)
                    return None
                log.warning("completion for %s: %s; retry %d in %.0fs", model, error, attempt + 1, backoff)
            except ValueError as error:   # malformed JSON: a server bug, not a transient failure -- don't retry
                log.warning("completion returned non-JSON for %s: %s", model, error)
                return None
            time.sleep(backoff)
        return None


def _lcp(a: list[int], b: list[int]) -> int:
    """The number of leading ids ``a`` and ``b`` agree on."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _dp_rank(chain_key: str, dp_size: int) -> int:
    """A stable (not Python's own randomised ``hash()``) rank in ``[0, dp_size)`` for ``chain_key``, so every
    request of one chain's read lands on the same vLLM DP replica regardless of process or run."""
    if dp_size <= 1:
        return 0
    digest = hashlib.sha256(chain_key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % dp_size


def _cache_salt(run_nonce: str, chain_key: str) -> str:
    """``run_nonce`` plus a fixed-length hash of ``chain_key``, charset-safe and well under vLLM's own 128-
    character, ``@``/``/``/``\\``/NUL-free limit on ``cache_salt`` (``entrypoints/generate/base/protocol.py``'s
    ``validate_cache_salt``) regardless of how long a run's own name is: a raw ``chain_key`` (a run directory's
    name, a user and an index) can run past 128 characters on its own, and the one request a long run name would
    then make vLLM reject outright becomes, under this teacher's own retry policy, a silent multi-minute hang
    rather than an error (:class:`_PinnedServer`)."""
    return f"{run_nonce}:{hashlib.sha256(chain_key.encode()).hexdigest()[:16]}"


def _completion_rows(server: Any, model: str, prompt: list[int], *, read_prefix_cache: bool = False,
                      cache_salt: str | None = None, headers: dict[str, str] | None = None) -> list[Any] | None:
    """``prompt``'s own log-probabilities (vLLM's ``prompt_logprobs``: per position, the top-k tokens and always the
    prompt's own), always one row per id of ``prompt`` -- row 0 always ``None`` (no context to score the first id
    from), and, on a prefix-cache hit with ``read_prefix_cache`` set, every row the hit covered left as whatever
    vLLM's uninitialised buffer happened to hold (almost always no logprob for the row's own target id: see
    :func:`_slice_tail`). ``None`` when the server fails or answers with no such field. ``cache_salt`` and
    ``headers`` (``X-data-parallel-rank``) are the chain-scoped scoping and DP pinning :func:`score_chain` computes
    once per chain; ``headers`` only reaches the server when ``server`` understands the keyword (:class:`_PinnedServer`,
    never ``ask_sd.server.Server``, which this calls exactly as before when ``headers`` is ``None``)."""
    body = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}
    if read_prefix_cache:
        body["vllm_xargs"] = {"read_prefix_cache": 1}
    if cache_salt:
        body["cache_salt"] = cache_salt
    out = server.completion(body, headers=headers) if headers else server.completion(body)
    choices = (out or {}).get("choices") or []
    if not choices:
        return None
    rows = choices[0].get("prompt_logprobs")
    return rows if isinstance(rows, list) else None


def _logprob(entry: Any, tid: int) -> float | None:
    """One row of :func:`_completion_rows`' ``logprob`` for ``tid``; ``None`` when the row has none (or none for any
    id, vLLM's own ``prompt_logprobs: 0``) or isn't finite. A row vLLM never actually computed (a prefix-cache hit
    that reached it) is still a well-formed row, just over whatever token id its uninitialised memory happened to
    hold -- essentially never ``tid`` itself, out of a ~248k vocabulary -- so this reads as ``None`` here almost
    always, not as a missing or malformed row."""
    info = entry.get(str(tid)) if isinstance(entry, dict) else None
    if info is None and isinstance(entry, dict):
        info = entry.get(tid)
    lp = info.get("logprob") if isinstance(info, dict) else None
    return float(lp) if isinstance(lp, (int, float)) and math.isfinite(lp) else None


def _slice(rows: list[Any] | None, start: int, reply_ids: list[int]) -> list[float | None] | None:
    """``reply_ids``' log-probabilities out of ``rows`` (:func:`_completion_rows`), read from position ``start``;
    ``None`` per id the row has none for, and ``None`` for the whole reply when ``rows`` does not reach that far.
    Used only for the trunk's own rows, which (never reading the cache) are always the full, genuine answer."""
    if rows is None or len(rows) < start + len(reply_ids):
        return None
    return [_logprob(rows[pos], tid) for pos, tid in zip(range(start, start + len(reply_ids)), reply_ids, strict=True)]


def _slice_tail(rows: list[Any] | None, seq_len: int, reply_ids: list[int]) -> list[float | None] | None:
    """``reply_ids``' log-probabilities out of ``rows`` (:func:`_completion_rows`), read off the *end*: row
    ``len(rows) - len(reply_ids)`` is ``reply_ids[0]``, through the last row for ``reply_ids[-1]``. ``reply_ids`` is
    always the last ``len(reply_ids)`` ids of the ``seq_len``-token sequence that was sent as the prompt, and
    vLLM's answer is always exactly that many rows (never shortened by a cache hit: see the module docstring), so
    this is the right slice for a cache-read request and a plain one alike. Refuses (``None``) unless ``rows`` is
    exactly ``seq_len`` long -- a different length means something else is wrong, not a hit to read around.

    This alone cannot tell a genuine cache hit that reached the reply from a correct answer: that call is
    :func:`score_chain`'s (the construction in the module docstring) and :func:`_score_branch`'s (the ``None``
    backstop on a read attempt)."""
    if rows is None or len(rows) != seq_len:
        return None
    tail = rows[len(rows) - len(reply_ids):]
    return [_logprob(entry, tid) for entry, tid in zip(tail, reply_ids, strict=True)]


def score_call(server: Any, call: Any, model: str, *, eos_id: int | None = None) -> list[float | None] | None:
    """The teacher's log-probability of every id of ``call``'s reply as it is trained (:func:`train.played_reply`:
    the end-of-turn token appended when the reply ended on ``stop`` or ``tool_calls`` and the server left it out),
    one ``/v1/completions`` request: ``call.prompt_ids`` then the reply, ``max_tokens: 1``, ``prompt_logprobs`` set.
    One value per reply token, ``None`` where the server gives none; ``None`` for the whole call when the server's
    answer does not cover it."""
    reply = train.played_reply(call, eos_id)
    if not reply:
        return []
    prompt = list(call.prompt_ids) + reply
    return _slice(_completion_rows(server, model, prompt), len(call.prompt_ids), reply)


def _score_branch(server: Any, model: str, seq: list[int], reply: list[int], *, read_prefix_cache: bool,
                   cache_salt: str | None, headers: dict[str, str] | None) -> tuple[list[float | None] | None,
                                                                                     dict[str, int]]:
    """One branch item's reply (:func:`score_chain`: an item whose own full sequence, ``seq``, is not a literal
    prefix of the chain's trunk), scored on ``seq`` alone. ``read_prefix_cache`` is :func:`score_chain`'s own
    per-branch decision (``d <= len(prompt) - 1``, the module docstring's construction) to send
    ``vllm_xargs: {"read_prefix_cache": 1}``, not a blanket flag: a branch this call set to ``False`` is never
    retried here, exactly as every branch was scored before this option existed. For one sent ``True``, the
    backstop: a reply with any ``None`` in it (a genuine, uninitialised cache hit; see :func:`_slice_tail`) is
    redone as a plain standalone request, counted in ``cache_fallbacks``. Returns the scores and this one item's
    own delta onto :data:`_CHAIN_STATS_ZERO`'s shape."""
    rows = _completion_rows(server, model, seq, read_prefix_cache=read_prefix_cache, cache_salt=cache_salt,
                            headers=headers)
    delta = dict(_CHAIN_STATS_ZERO, requests=1, prompt_tokens=len(seq), branch_requests=1, branch_tokens=len(seq),
                cache_read_attempts=1 if read_prefix_cache else 0)
    scores = _slice_tail(rows, len(seq), reply)
    uncovered = scores is None or (read_prefix_cache and any(s is None for s in scores))
    if uncovered and read_prefix_cache:
        delta["cache_fallbacks"] = 1
        rows = _completion_rows(server, model, seq, read_prefix_cache=False, cache_salt=cache_salt, headers=headers)
        delta["requests"] += 1
        delta["prompt_tokens"] += len(seq)
        delta["branch_requests"] += 1
        delta["branch_tokens"] += len(seq)
        scores = _slice_tail(rows, len(seq), reply)
    return scores, delta


def score_chain(server: Any, model: str, chain: list[Any], *, read_prefix_cache: bool = False,
                run_nonce: str | None = None, dp_size: int = 1) -> tuple[list[list[float | None] | None],
                                                                          dict[str, int]]:
    """``chain`` (one of :func:`train.find_chains`' chains: items in call order, each a later call's prompt starting
    where the one before's prompt and reply end): one request over the chain's last item's full sequence (its
    ``student_ids``, the longest), and every earlier item's reply sliced out of it when that item's own full
    sequence is a literal prefix of it. An item whose own sequence is not a literal prefix (the chain's trunk
    diverges from it within ``--chain-slack`` ids of its end, or the one request failed) is a branch, scored by
    :func:`_score_branch` on its own, with its own prompt in front of it, not the trunk's; every chain's branch
    items are sent together, concurrently (bounded by :data:`BRANCH_WORKERS`), once the trunk request has returned
    (a chain's own, separate thread pool, so this never competes with :func:`score_items`' outer one for a worker
    to wait on -- though it does compete for the server, which is why the pool is bounded). Returns the per-item
    results, in the chain's order, and this chain's stats (:data:`_CHAIN_STATS_ZERO`'s shape).

    With ``read_prefix_cache``, every request this chain sends carries a ``cache_salt`` (:func:`_cache_salt`;
    ``run_nonce`` is minted here if the caller gave none, never left as a stable, reusable value), scoping the
    cache to this chain in this run (never another chain's blocks, never a previous run's against a warm
    server), and ``X-data-parallel-rank`` pinning it to one vLLM DP replica (:func:`_dp_rank` of the chain's own
    key, mod ``dp_size``) -- without it, vLLM's load-based DP router would mostly scatter a chain's branches
    away from the replica its trunk ran on, and a read could never hit. A branch only ever gets
    ``vllm_xargs: {"read_prefix_cache": 1}`` when its longest common prefix with the trunk and with every other
    item in the chain (``seqs``, known upfront -- no request needed to check) is ``<= len(prompt) - 1``: that
    bounds the real, block-aligned hit length below the reply's first position, by construction, before
    :func:`_score_branch`'s own backstop ever has to catch a miss. The trunk itself never reads (every other item
    needs every one of its rows), but still carries the salt and header, since its *write* is what a branch's read
    depends on.

    ``ask_sd.server.Server.completion`` has no ``timeout`` argument of its own; every request takes its default
    (1,800 seconds), ample even for the longest recorded call (129,482 tokens) under load."""
    trunk = list(chain[-1].student_ids)
    seqs = [list(item.student_ids) for item in chain]
    cache_salt = headers = None
    if read_prefix_cache:
        if run_nonce is None:   # never a stable salt by omission: see _cache_salt's own docstring
            run_nonce = uuid.uuid4().hex
        chain_key = ":".join(str(chain[-1].label[k]) for k in ("run", "user", "i"))
        cache_salt = _cache_salt(run_nonce, chain_key)
        headers = {"X-data-parallel-rank": str(_dp_rank(chain_key, dp_size))}
    trunk_rows = _completion_rows(server, model, trunk, cache_salt=cache_salt, headers=headers)
    stats = dict(_CHAIN_STATS_ZERO, requests=1, prompt_tokens=len(trunk))
    results: list[list[float | None] | None] = [None] * len(chain)
    branches: list[tuple[int, list[int], list[int], bool]] = []
    for i, item in enumerate(chain):
        seq = seqs[i]
        prompt, reply = seq[:-item.n] if item.n else seq, seq[-item.n:] if item.n else []
        if not reply:
            results[i] = []
            continue
        scores = _slice(trunk_rows, len(prompt), reply) if seq == trunk[:len(seq)] else None
        if scores is not None:
            results[i] = scores
            continue
        read_ok = False
        if read_prefix_cache:
            d = _lcp(seq, trunk)
            for j, other in enumerate(seqs):
                if j != i:
                    d = max(d, _lcp(seq, other))
            read_ok = d <= len(prompt) - 1
        branches.append((i, seq, reply, read_ok))
    if branches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(branches), BRANCH_WORKERS)) as pool:
            branch_results = pool.map(
                lambda b: _score_branch(server, model, b[1], b[2], read_prefix_cache=b[3], cache_salt=cache_salt,
                                        headers=headers),
                branches)
        for (i, _, _, _), (scores, delta) in zip(branches, branch_results, strict=True):
            results[i] = scores
            for key, value in delta.items():
                stats[key] += value
    return results, stats


def score_items(server: Any, model: str, items: list[Any], *, slack: int = train.CHAIN_SLACK,
                workers: int = SCORE_WORKERS, read_prefix_cache: bool = False, dp_size: int = 1
                ) -> tuple[list[list[float | None] | None], dict[str, int]]:
    """Teacher log-probabilities for every item's reply, in ``items``' own order: ``items`` (:func:`train.sft_items`,
    or any list of ``ask_sd.trainer.TokenItem`` labelled with ``run``, ``user`` and ``i``) grouped into chains by
    :func:`train.find_chains` with ``slack`` (the same the trainer reads ``--chain-slack`` with, so the scorer and
    the trainer read the same chains), and scored chain by chain (:func:`score_chain`) on up to ``workers`` threads
    at once, since vLLM only batches requests that arrive together: a round's chains sent one at a time would leave
    the server mostly idle. ``read_prefix_cache`` (with ``dp_size``, the teacher's own DP replica count) is passed
    through to every chain, sharing one fresh ``run_nonce`` (a uuid) for this whole call, so every chain's own
    ``cache_salt`` is scoped to this run alone. A chain whose scoring raises scores every one of its items ``None``
    (counted in ``stats["unscored"]``), the others unaffected. ``stats`` totals every chain's (:func:`score_chain`),
    plus ``calls``, ``chains`` and ``unscored`` (items scored ``None`` outright, not positions within a scored
    item); results come back in ``items``' own order regardless of which chain finishes first."""
    chains = train.find_chains(items, slack)
    run_nonce = uuid.uuid4().hex if read_prefix_cache else None

    def run(chain: list[Any]) -> tuple[list[list[float | None] | None], dict[str, int]]:
        try:
            return score_chain(server, model, chain, read_prefix_cache=read_prefix_cache, run_nonce=run_nonce,
                               dp_size=dp_size)
        except Exception:  # noqa: BLE001  (one chain's failure must not cost the others their scores)
            return [None] * len(chain), dict(_CHAIN_STATS_ZERO)

    scored: dict[int, list[float | None] | None] = {}
    stats = {"calls": len(items), "chains": len(chains), "unscored": 0, **_CHAIN_STATS_ZERO}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for chain, (results, chain_stats) in zip(chains, pool.map(run, chains), strict=True):
            for item, scores in zip(chain, results, strict=True):
                scored[id(item)] = scores
                stats["unscored"] += scores is None
            for key in _CHAIN_STATS_ZERO:
                stats[key] += chain_stats[key]
    return [scored[id(it)] for it in items], stats


def branch_calls(items: list[Any], *, slack: int = train.CHAIN_SLACK) -> set[tuple[str, int, int]]:
    """Which of ``items`` :func:`score_chain` would score as a branch -- its own full sequence is not a literal
    prefix of its chain's trunk, so a non-empty reply gets its own standalone request -- identified by their
    ``(run, user, i)`` label (:func:`train.load_calls`'s), with no server call at all: the same trunk-vs-item
    comparison :func:`score_chain` makes, over every chain :func:`train.find_chains` (``slack``) finds in ``items``.
    An item with an empty reply (``item.n == 0``) is neither: trivially scored ``[]``, nothing for the cache to
    answer or not. For a tool working from an already-scored ``--out`` file (``teacher_cache_check.py``'s reference
    sample of plain sliced calls, and its test population of branch calls) that has no per-item record of which was
    which."""
    out: set[tuple[str, int, int]] = set()
    for chain in train.find_chains(items, slack):
        trunk = list(chain[-1].student_ids)
        for item in chain:
            if item.n and list(item.student_ids) != trunk[:len(item.student_ids)]:
                out.add((item.label["run"], item.label["user"], item.label["i"]))
    return out


# -- the tokenizer guard -------------------------------------------------------------------------------------------

def _ids(tokenizer_json: dict[str, Any]) -> dict[int, str]:
    """A parsed ``tokenizer.json``'s ids to their tokens: the base vocabulary and the added tokens (the special
    tokens, which ``tokenizer.json``'s ``model.vocab`` leaves out)."""
    out = {i: t for t, i in tokenizer_json["model"]["vocab"].items()}
    for added in tokenizer_json.get("added_tokens") or []:
        out[added["id"]] = added["content"]
    return out


def tokenizer_mismatch(student: dict[str, Any], teacher: dict[str, Any]) -> str | None:
    """``None`` when ``teacher`` (a parsed ``tokenizer.json``) gives ``student``'s token ids for every id the
    student's own tokenizer uses: the same token at every id up to the student's highest, and the same merge rules
    (BPE: earlier rules first) over at least as many pairs as the student's. The teacher may add further tokens
    above the student's highest id (Qwen3.8-27B's 7 audio tokens at ids 248070-248076, which the student never
    writes) without being refused. One line saying what differs, otherwise."""
    s_ids, t_ids = _ids(student), _ids(teacher)
    used_max_id = max(s_ids)
    bad = [i for i in range(used_max_id + 1) if s_ids.get(i) != t_ids.get(i)]
    if bad:
        return (f"{len(bad)} of the student's {used_max_id + 1} ids give a different token in the teacher "
                f"(first: id {bad[0]}, student {s_ids.get(bad[0])!r}, teacher {t_ids.get(bad[0])!r})")
    s_merges = student["model"].get("merges") or []
    t_merges = teacher["model"].get("merges") or []
    if t_merges[:len(s_merges)] != s_merges:
        return "the merge rules differ"
    return None


def check_tokenizer(student: dict[str, Any], teacher: dict[str, Any]) -> None:
    """Raise ``ValueError`` when :func:`tokenizer_mismatch` finds one; otherwise do nothing."""
    reason = tokenizer_mismatch(student, teacher)
    if reason:
        raise ValueError(f"the teacher's tokenizer was refused: {reason}")


def load_tokenizer_json(repo: str) -> dict[str, Any]:
    """``repo``'s ``tokenizer.json``, parsed (:func:`tokenizer_mismatch`, :func:`check_tokenizer`)."""
    import json

    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(repo, "tokenizer.json")) as f:
        return json.load(f)


# -- throughput measurement (run on a GPU node, not here) -----------------------------------------------------------

def _metric_total(text: str, name: str) -> float:
    """The sum, across every line of a Prometheus text-format dump (``text``, e.g. a vLLM server's ``/metrics``),
    of the ``name``-Counter's value (one line per DP engine replica, each labelled ``engine``, summed into one
    total). ``prometheus_client.Counter`` renders ``name`` as ``{name}_total`` and also emits a same-prefixed
    ``{name}_created`` gauge line (the counter's creation timestamp, not a count); matched by the metric name
    alone, with any ``{...}`` labels stripped, so the ``_created`` line is never mistaken for a value to sum."""
    target = f"{name}_total"
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        if head.split("{", 1)[0] == target:
            try:
                total += float(value)
            except ValueError:
                continue
    return total


def _fetch_metrics(vllm_url: str) -> str | None:
    """The teacher server's own ``/metrics`` (Prometheus text format), or ``None`` when it can't be read (an
    older server, one started without it, or the server is down)."""
    try:
        base = vllm_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as resp:
            return resp.read().decode()
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _prefix_cache_metrics(text: str) -> tuple[float, float]:
    """``(queried_tokens, hit_tokens)`` from ``text`` (:func:`_fetch_metrics`; vLLM's ``vllm:prefix_cache_queries``
    and ``vllm:prefix_cache_hits`` counters, summed across every DP engine replica: ``v1/metrics/stats.py``'s
    ``PrefixCacheStats`` counts tokens queried/matched, not requests, and only for a request that actually looked
    the cache up -- ``kv_cache_manager.py`` early-returns otherwise). Server-wide: any other traffic to the same
    teacher between two reads of this (another scoring job, a probe) is counted too, same as this run's own."""
    return _metric_total(text, "vllm:prefix_cache_queries"), _metric_total(text, "vllm:prefix_cache_hits")


def _dp_size_from_metrics(text: str) -> int | None:
    """The teacher server's own DP replica count, read directly off ``text`` (:func:`_fetch_metrics`) rather than
    trusted from a flag: every per-engine metric line carries an ``engine="N"`` label
    (``v1/metrics/loggers.py``'s ``per_engine_labelvalues``), so the number of distinct values across the whole
    dump is the real DP. ``None`` when no such label appears at all (an older server, or a dump with no per-
    engine metrics)."""
    engines = set(re.findall(r'engine="(\d+)"', text))
    return len(engines) if engines else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--vllm-url", required=True, help="the teacher's server (serve_teacher.sh), e.g. http://localhost:8102/v1")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B", help="the teacher's served name")
    p.add_argument("--train-runs", nargs="+", required=True, help="published run folders (train.load_calls)")
    p.add_argument("--student-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--chain-slack", type=int, default=train.CHAIN_SLACK)
    p.add_argument("--max-seq-len", type=int, default=TEACHER_MAX_SEQ_LEN)
    p.add_argument("--workers", type=int, default=SCORE_WORKERS, help="chains scored at once (score_items)")
    p.add_argument("--read-prefix-cache", action="store_true",
                   help="let a branch (non-trunk-prefix) call's request read the server's prefix cache for the "
                        "part it shares with the trunk and its siblings, where that is safe by construction "
                        "(vllm_xargs: {'read_prefix_cache': 1}; needs the teacher server started with "
                        "TEACHER_PREFIX_CACHE=align, serve_teacher.sh); off by default")
    p.add_argument("--dp-size", type=int, default=None,
                   help="the teacher server's own DP replica count; with --read-prefix-cache, read directly off "
                        "the server's own /metrics when not given, and refused if given and it disagrees -- "
                        "needed to pin a chain's requests to the one replica its trunk ran on "
                        "(X-data-parallel-rank), since vLLM's DP router otherwise load-balances a chain's "
                        "branches away from it")
    p.add_argument("--skip-tokenizer-check", action="store_true", help="the first run on a node checks it once")
    p.add_argument("--out", default=None,
                   help="write every item's (run, user, i) and its teacher scores here (JSON), atomically, for an "
                        "OPD training step to read once it builds its items; the throughput line always prints "
                        "either way")
    a = p.parse_args()

    if not a.skip_tokenizer_check:
        check_tokenizer(load_tokenizer_json(a.student_model), load_tokenizer_json(a.model))

    from transformers import AutoTokenizer

    from ask_sd.server import Server
    tokenizer = AutoTokenizer.from_pretrained(a.student_model, trust_remote_code=True)
    calls = [c for r in a.train_runs for c in train.load_calls(Path(r).expanduser())[0]]
    server = _PinnedServer(a.vllm_url) if a.read_prefix_cache else Server(a.vllm_url)

    metrics_before_text = _fetch_metrics(a.vllm_url)
    metrics_before = _prefix_cache_metrics(metrics_before_text) if metrics_before_text else None

    items, item_stats = train.sft_items(tokenizer, calls, a.max_seq_len)
    dp_size = 1
    if a.read_prefix_cache:
        server_dp_size = _dp_size_from_metrics(metrics_before_text) if metrics_before_text else None
        if a.dp_size is not None and server_dp_size is not None and a.dp_size != server_dp_size:
            print(f"--dp-size {a.dp_size} disagrees with the server's own DP ({server_dp_size}, read off its "
                  f"/metrics' engine= labels): refusing rather than silently pinning to the wrong rank count",
                  file=sys.stderr)
            return 1
        if server_dp_size is not None:
            dp_size = server_dp_size
        elif a.dp_size is not None:
            dp_size = a.dp_size
        else:
            print("could not read the server's own DP from /metrics, and --dp-size was not given: every chain "
                  "will pin to the same replica (dp_size=1) -- pass --dp-size explicitly if the server's DP is "
                  "more than 1, or every other replica will sit idle with no error", file=sys.stderr)

    t0 = time.time()
    scores, stats = score_items(server, a.model, items, slack=a.chain_slack, workers=a.workers,
                                read_prefix_cache=a.read_prefix_cache, dp_size=dp_size)
    seconds = time.time() - t0
    out_items = [{"run": it.label["run"], "user": it.label["user"], "i": it.label["i"], "scores": sc}
                 for it, sc in zip(items, scores, strict=True)]

    metrics_after_text = _fetch_metrics(a.vllm_url)
    metrics_after = _prefix_cache_metrics(metrics_after_text) if metrics_after_text else None

    if a.out:
        metrics_delta = (None if metrics_before is None or metrics_after is None else
                         {"queries": metrics_after[0] - metrics_before[0], "hits": metrics_after[1] - metrics_before[1]})
        out = Path(a.out).expanduser()
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps({
            "model": a.model, "student_model": a.student_model, "stats": stats, "seconds": seconds,
            "read_prefix_cache": a.read_prefix_cache, "chain_slack": a.chain_slack,
            "max_seq_len": a.max_seq_len, "train_runs": a.train_runs, "prefix_cache_metrics": metrics_delta,
            "items": out_items}))
        tmp.replace(out)   # atomic: a kill mid-write leaves only the .tmp file, never a truncated --out

    print(f"calls: {item_stats['calls']} ({item_stats['too_long']} too long for {a.max_seq_len}), "
          f"items: {stats['calls']}, chains: {stats['chains']}, requests: {stats['requests']}, "
          f"unscored: {stats['unscored']}")
    print(f"branch requests: {stats['branch_requests']} ({stats['branch_tokens']} tokens), "
          f"cache read attempts: {stats['cache_read_attempts']}, cache fallbacks: {stats['cache_fallbacks']}")
    rate = f"{stats['prompt_tokens'] / seconds:.0f}" if seconds > 0 else "n/a"
    print(f"tokens read: {stats['prompt_tokens']}, seconds: {seconds:.1f}, tokens/second: {rate}")
    if metrics_before is not None and metrics_after is not None:
        d_queries, d_hits = metrics_after[0] - metrics_before[0], metrics_after[1] - metrics_before[1]
        hit_rate = f"{d_hits / d_queries:.1%}" if d_queries > 0 else "n/a"
        print(f"server prefix cache: {d_hits:.0f}/{d_queries:.0f} tokens hit ({hit_rate})")

    if not stats["calls"]:
        print("0 items scored: nothing to train on, whatever the cause", file=sys.stderr)
        return 1
    if stats["unscored"] / stats["calls"] > TEACHER_UNSCORED_MAX_FRAC:
        print(f"{stats['unscored']}/{stats['calls']} calls unscored, over {TEACHER_UNSCORED_MAX_FRAC:.0%}: "
              f"the teacher likely died or restarted partway through scoring", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
