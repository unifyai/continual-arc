"""Data parallelism (continual_arc_training.parallel): two ranks on the CPU (gloo) against one process, on the tiny
random Qwen3.5 of test_trainer. Needs torch, transformers and peft (skipped without them)."""
from __future__ import annotations

import dataclasses
import inspect
import os
import random
import signal
import time

import pytest

from continual_arc_training.parallel import deal


def test_chains_are_dealt_by_cost_the_same_on_every_rank():
    costs = [5.0, 40.0, 7.0, 7.0, 22.0, 3.0, 18.0, 1.0]
    owner, load = deal(costs, 3)
    assert owner == [2, 0, 2, 1, 1, 1, 2, 2] and load == [40.0, 32.0, 31.0]   # the costliest first, to the least loaded
    assert [sum(c for c, o in zip(costs, owner, strict=True) if o == r) for r in range(3)] == load
    assert deal(costs, 3) == (owner, load) and deal(costs, 1)[0] == [0] * len(costs)


def test_every_trainer_method_that_gathers_across_ranks_runs_on_every_rank():
    """A trainer method that deals work between ranks or gathers their results must be one of DataParallelTrainer's
    own: its __getattr__ sends anything else to rank 0 alone, which would then wait in the collective for ranks that
    were never asked."""
    pytest.importorskip("torch")
    from continual_arc_training.parallel import DataParallelTrainer
    from continual_arc_training.trainer import Trainer

    collective = ("all_gather_object", "_dp_begin", "_dp_gather", "_train_frame", "_forward(", "deal(")
    gathering = [name for name, fn in inspect.getmembers(Trainer, inspect.isfunction)
                 if not name.startswith("_") and any(marker in inspect.getsource(fn) for marker in collective)]
    assert set(gathering) >= {"sft_chains", "distill_chains", "policy_gradient_chains", "nll_chains",
                              "distill_scores_chains", "policy_gradient_scores_chains", "behaviour_logps_chains"}
    for name in gathering:
        assert name in vars(DataParallelTrainer), f"Trainer.{name} gathers across ranks but is not forwarded"


SETTINGS = dict(rank=4, alpha=8, lr=1e-2, max_seq_len=512, grad_checkpointing=False)


def _prepare(trainer) -> None:
    """Float32 and the adapter moved off its start alike on every rank and in one process."""
    import torch

    trainer.model.float()
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        for v in trainer._trainable():
            v.add_(0.05 * torch.randn(v.shape, generator=g).to(v.dtype))


def _fail_on_rank_1(trainer, on: bool) -> None:
    """Rank 1's chain packing raises (``on``), or no longer does."""
    from continual_arc_training import branching

    if trainer.dp_rank != 1:
        return
    if on:
        def broken(*args, **kwargs):
            raise RuntimeError("injected failure on rank 1")

        branching._pack_as_loaded, branching.pack = branching.pack, broken
    else:
        branching.pack = branching._pack_as_loaded


def _pending(trainer) -> int:
    return trainer._pending_items


def _chains(seed: int, vocab: int = 1024) -> list:
    """Two chains of different shape (different costs to deal between the ranks): a trunk and three branches, and a
    trunk and two."""
    from continual_arc_training.chains import ChainItem

    rng = random.Random(seed)
    ids = lambda n: [rng.randrange(1, vocab) for _ in range(n)]   # noqa: E731
    shapes = [(40, [(10, 7), (25, 3), (40, 11)]), (30, [(12, 5), (30, 4)])]
    return [ChainItem(ids(trunk_len), [(p, ids(n)) for p, n in points], labels=[{"chain": c, "k": k}
                                                                                 for k in range(len(points))])
            for c, (trunk_len, points) in enumerate(shapes)]


def _teacher_chains(chains: list, seed: int) -> list:
    """Each chain's teacher chain: the same trunk and replies, each branch carrying a note of five ids inserted before
    the last two ids of its prompt as its head."""
    from continual_arc_training.chains import ChainItem

    rng = random.Random(seed)
    out = []
    for c in chains:
        branches, heads = [], []
        for point, reply in c.branches:
            cut = max(1, point - 2)
            head = [rng.randrange(1, 1024) for _ in range(5)] + c.trunk[cut:point]
            branches.append((cut, head + reply))
            heads.append(len(head))
        out.append(ChainItem(c.trunk, branches, heads=heads))
    return out


def _session(tr):
    """What both setups run, in order: two SFT steps, two OPSD steps, the behaviour log-probabilities under the
    untrained adapter and two PG steps against them, then every forward-only score; each step's result, the adapter
    after it and the random generator's state."""
    rng = random.Random(0)
    out = []

    def took(name, res):
        taken = tr.step()
        out.append((name, res, taken, tr.adapter_state(), rng.getstate()))

    for seed in (1, 2):
        took("sft", tr.sft_chains(_chains(seed), rng=rng))
    for seed in (3, 4):
        chains = _chains(seed)
        took("opsd", tr.distill_chains(chains, _teacher_chains(chains, seed), rng=rng, skip_first=1))
    tr.configure_behaviour(None)
    for seed in (5, 6):
        chains = _chains(seed)
        played = tr.behaviour_logps_chains(chains)["items"]
        it = iter(played)
        chains = [dataclasses.replace(c, advantages=[0.5 - 0.4 * k for k in range(len(c.branches))],
                                      behaviour_logps=[next(it) for _ in c.branches]) for c in chains]
        took("pg", tr.policy_gradient_chains(chains, rng=rng))
    chains = _chains(7)
    teachers = _teacher_chains(chains, 7)
    played = tr.behaviour_logps_chains(chains)["items"]
    it = iter(played)
    pg = [dataclasses.replace(c, advantages=[0.3] * len(c.branches), behaviour_logps=[next(it) for _ in c.branches])
          for c in chains]
    scores = (played, [x["logp"] for x in tr.nll_chains(chains)["items"]],
              [x["loss"] for x in tr.policy_gradient_scores_chains(pg)["items"]],
              [x["loss"] for x in tr.distill_scores_chains(chains, teachers, skip_first=1)["items"]])
    return out, scores


def _max_diff(a: dict, b: dict) -> float:
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


STATS = {"sft": ("nll_per_token", "entropy_mean", "loss_mean"), "opsd": ("loss_mean",),
         "pg": ("ratio_mean", "clip_fraction", "approx_kl", "loss_mean", "entropy_mean")}


def test_two_ranks_take_every_step_and_score_one_process_takes(tmp_path):
    """DataParallelTrainer over two ranks (this process and one worker): each pass's result (every reply's statistics,
    in one process's order) is one process's, every rank holds the same adapter after each step and one process's (to
    float rounding: the gradients are summed in another order), the forward-only scores are one process's and leave
    nothing pending; rank 0 saves, every rank resets and loads; a pass in which rank 1 fails raises on both ranks with
    rank 1's error attached and takes no step; the worker ignores SIGTERM and exits when the front-end closes."""
    torch = pytest.importorskip("torch")
    import torch.distributed as dist
    from test_trainer import _hybrid_model

    from continual_arc_training.parallel import DataParallelError, DataParallelTrainer
    from continual_arc_training.trainer import Trainer

    model_dir = tmp_path / "model"
    _hybrid_model(model_dir)   # the model and tokenizer on disk
    torch.manual_seed(0)
    tr = Trainer(str(model_dir), device="cpu", **SETTINGS)
    _prepare(tr)
    init = tr.adapter_state()
    one, one_scores = _session(tr)
    del tr

    torch.manual_seed(0)
    front = DataParallelTrainer(str(model_dir), world_size=2, device="cpu", timeout_s=120, **SETTINGS)
    worker = front._workers[0][0]
    try:
        assert dist.get_world_size() == 2 and front.tokenizer is front.local.tokenizer
        untrained = front.every_rank("adapter_state")[0]
        front.every_rank(_prepare)
        start = front.every_rank("adapter_state")
        assert all(torch.equal(start[1][k], start[0][k]) and torch.equal(start[0][k], init[k]) for k in init)
        two, two_scores = _session(front)
        for (name, res1, taken1, after1, rng1), (_, res, taken, after, rng) in zip(one, two, strict=True):
            assert taken["identical"] and taken["steps"] == taken1["steps"] == 1 and taken["items"] == taken1["items"]
            assert res["tokens"] == res1["tokens"] and res["chains"] == res1["chains"], name
            for key in ("logp", "item_loss"):
                if key in res1:
                    assert res[key] == pytest.approx(res1[key], rel=1e-4, abs=1e-7), (name, key)
            assert [res[k] for k in STATS[name]] == pytest.approx([res1[k] for k in STATS[name]], rel=1e-4,
                                                                  abs=1e-7), name
            assert _max_diff(after, after1) < 1e-4 * _max_diff(after1, init) and rng == rng1
        states = front.every_rank("adapter_state")
        assert all(torch.equal(states[0][n], states[1][n]) for n in states[0])
        assert [v for x in two_scores[0] for v in x] == pytest.approx([v for x in one_scores[0] for v in x], rel=1e-5)
        for got, want in zip(two_scores[1:], one_scores[1:], strict=True):
            assert got == pytest.approx(want, rel=1e-4, abs=1e-7)
        assert front.every_rank(_pending) == [0, 0]

        saved = states[0]
        front.save(tmp_path / "p0001")
        assert (tmp_path / "p0001" / "trainer_adapter.safetensors").is_file()
        front.reset()
        back = front.every_rank("adapter_state")
        assert all(torch.equal(back[r][n], untrained[n].float()) for r in (0, 1) for n in saved)
        front.load(tmp_path / "p0001")
        back = front.every_rank("adapter_state")
        assert all(torch.equal(back[r][n], saved[n]) for r in (0, 1) for n in saved)

        front.every_rank(_fail_on_rank_1, True)
        with pytest.raises(DataParallelError, match="other rank failed") as failed:
            front.sft_chains(_chains(8), rng=random.Random(0))
        assert "injected failure on rank 1" in "".join(failed.value.__notes__)
        front.every_rank(_fail_on_rank_1, False)
        assert front.every_rank(_pending) == [0, 0]
        res = front.sft_chains(_chains(8), rng=random.Random(0))
        assert res["n_examples"] == 5 and front.step()["identical"]
        digests = front.every_rank("adapter_digest")
        assert digests[0] == digests[1]

        os.kill(worker.pid, signal.SIGTERM)   # a machine shutting down signals every process at once
        time.sleep(2)
        assert worker.is_alive()
        assert front.every_rank("adapter_digest")[1] == digests[1]
    finally:
        front.close()
    worker.join(timeout=60)
    assert worker.exitcode == 0 and not dist.is_initialized()
