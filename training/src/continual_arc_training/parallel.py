"""Data-parallel training: one optimizer step's chains split across several GPUs.

Qwen3.5-9B fits whole on every GPU, so data parallelism needs no sharding of the model: every rank holds the frozen
model and the adapter, passes its share of the step's chains, and right before the optimizer step the adapter's
gradients are summed across the ranks, once a step. Every rank then takes the same step from the same gradients, so
every rank holds the same adapter, which is checked after every step.

- :class:`DataParallelRank` (``Trainer`` inherits it): a trainer inside a ``torch.distributed`` group of several ranks
  takes rank 0's adapter, passes only its share of a call's chains (:func:`deal`: by estimated cost, since a chain's
  trunk runs from a few thousand to 65,536 tokens), after them gathers every reply's statistics in the order one
  process gives them, and in ``step`` sums the gradients before dividing by the count. Only rank 0 writes adapters.
  With one rank nothing changes.
- :class:`DataParallelTrainer`: the trainer's methods over several GPUs from one process: this process is rank 0, and
  one persistent worker process per other GPU holds its own trainer. Each call's chains go to every rank, the step is
  taken together, and rank 0's result (every reply's statistics) comes back.

A rank that raises in the middle of a call tells the others (:func:`data_parallel_guard`), so every rank raises and
the step is abandoned on every rank; a worker process that dies leaves the others waiting until the group's timeout.
"""
from __future__ import annotations

import atexit
import functools
import os
import pickle
import signal
import threading
import time
import traceback
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

# the size of the flat buffers the gradients are summed in (one all-reduce each)
BUCKET_BYTES = 256 << 20
# attention's cost per (query, key) pair relative to a token's, when the model's configuration does not give it
DEFAULT_PAIR_COST = 1 / 25_000


class DataParallelError(RuntimeError):
    """Another rank failed, or a worker process did."""


def data_parallel_size() -> tuple[int, int]:
    """(this process's rank, the number of ranks) of the ``torch.distributed`` group it is in; (0, 1) without one."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def deal(costs: list[float], ranks: int) -> tuple[list[int], list[float]]:
    """The rank each chain goes to: the most costly first, each to the rank with the least estimated work so far
    (longest processing time first), ties to the lower index and the lower rank, so every rank computes the same deal.
    Returns the owners and each rank's total."""
    load = [0.0] * ranks
    owner = [0] * len(costs)
    for i in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        r = min(range(ranks), key=lambda r: (load[r], r))
        owner[i] = r
        load[r] += costs[i]
    return owner, load


def data_parallel_guard(method: Callable) -> Callable:
    """For the trainer's passes that accumulate gradients: a rank that fails in a call's passes, before it has said it
    got through them, says so instead, so the other ranks, which will wait for it there, raise too instead of waiting
    until the group's timeout, and the step is abandoned on every rank."""

    @functools.wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            if self.dp_size > 1 and self._dp_pending:
                self._dp_flag(failed=True)
            raise

    return guarded


class _Step:
    """One data-parallel pass over some chains on one rank: the chains it passes (their positions in the call's order),
    and where each one's statistics begin in the per-reply lists, so every rank's can be put back in the order one
    process would have produced them; the counters the result sums over the chains, which every rank's are added
    into; and the replies of all the chains, which the step will count."""

    def __init__(self, mine: list[int], lists: tuple[list, ...], shared: tuple[dict[str, int], ...], items: int,
                 pending: int) -> None:
        self.mine, self.lists, self.shared = mine, lists, shared
        self.start = [len(x) for x in lists]
        self.shared0 = [dict(d) for d in shared]
        self.items, self.pending = items, pending
        self.marks: list[tuple[int, list[int]]] = []

    def mark(self, pos: int) -> None:
        self.marks.append((pos, [len(x) for x in self.lists]))


class DataParallelRank:
    """What the trainer does as one rank of a data-parallel group (``Trainer`` inherits it). Every rank is given every
    chain; :func:`deal` decides which of them it passes (:meth:`_dp_begin`); after the passes every reply's statistics
    and the reply count the step divides by are gathered (:meth:`_dp_gather`), right before the optimizer step the
    gradients are summed (:meth:`_dp_reduce`), and after it every rank's adapter is checked to be the same
    (:meth:`_dp_verify`)."""

    dp_rank, dp_size = 0, 1            # one rank: everything as before
    _dp_pending = False                # inside a call's passes: a failure must tell the other ranks
    _dp_pair_cost: float | None = None

    def join_data_parallel(self) -> None:
        """Train as one rank of the ``torch.distributed`` group this process is in: every rank takes rank 0's adapter
        (LoRA draws its A matrices at random), which :meth:`reset` then returns to; only rank 0 writes adapters."""
        torch = self.torch
        self.dp_rank, self.dp_size = data_parallel_size()
        if self.dp_size == 1:
            return
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
        self._dp_broadcast(optimizer=False)
        self._init_state = {k: v.detach().clone() for k, v in self.model.named_parameters() if v.requires_grad}

    def adapter_state(self) -> dict[str, Any]:
        """The adapter's weights (parameter name to a CPU copy)."""
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.named_parameters() if v.requires_grad}

    def adapter_digest(self) -> list[int]:
        """An exact digest of the adapter: per parameter, the sum of its values' bit patterns as integers. Two ranks'
        digests are equal when their adapters are bitwise equal (and, but for a contrived coincidence, only then)."""
        torch = self.torch
        sums = []
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    bits = p.detach().reshape(-1).view({4: torch.int32, 2: torch.int16, 8: torch.int64}[p.element_size()])
                    sums.append(bits.sum(dtype=torch.int64))
        return torch.stack(sums).tolist() if sums else []

    def _dp_begin(self, costs: list[float], items: int, lists: tuple[list, ...],
                  shared: tuple[dict[str, int], ...]) -> _Step:
        """This rank's share of chains whose estimated costs are given (in the call's order), holding ``items`` replies
        in all, which the step will count."""
        owner, _ = deal(costs, self.dp_size)
        self._dp_pending = True
        return _Step([i for i, r in enumerate(owner) if r == self.dp_rank], lists, shared, items, self._pending_items)

    def _dp_gather(self, step: _Step) -> None:
        """After this rank's passes: every rank got through its chains (else every rank raises), and every reply's
        statistics are gathered, in the order one process would have produced them, into the per-reply lists on every
        rank; the reply count the step divides by becomes the whole step's."""
        torch, dist = self.torch, self.torch.distributed
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        self._dp_check()
        ends = [m[1] for m in step.marks[1:]] + [[len(x) for x in step.lists]]
        segments = [(pos, [x[a:b] for x, a, b in zip(step.lists, starts, stops, strict=True)])
                    for (pos, starts), stops in zip(step.marks, ends, strict=True)]
        mine = {"segments": segments,
                "shared": [{k: d[k] - d0[k] for k in d} for d, d0 in zip(step.shared, step.shared0, strict=True)]}
        every: list[Any] = [None] * self.dp_size
        dist.all_gather_object(every, mine)
        segments = sorted((s for e in every for s in e["segments"]), key=lambda s: s[0])
        for j, x in enumerate(step.lists):
            del x[step.start[j]:]
            for _, parts in segments:
                x.extend(parts[j])
        for j, d in enumerate(step.shared):
            for k in d:
                d[k] = step.shared0[j][k] + sum(e["shared"][j][k] for e in every)
        self._pending_items = step.pending + step.items

    def _dp_reduce(self) -> None:
        """Right before the optimizer step: the adapter's gradients summed across ranks (a rank that passed no chain
        adds zeros)."""
        self._dp_check()
        self._dp_all_reduce_grads()

    def _dp_verify(self) -> bool:
        """After the step: whether every rank holds the same adapter (compared exactly), as it must, having taken the
        same step from the same gradients. If one ever did not, every rank takes rank 0's adapter and optimizer
        moments."""
        every: list[Any] = [None] * self.dp_size
        self.torch.distributed.all_gather_object(every, self.adapter_digest())
        identical = all(e == every[0] for e in every)
        if not identical:
            self._dp_broadcast(optimizer=True)
        return identical

    def _dp_check(self) -> None:
        """Once after a call's passes (whichever collective comes first): did every rank get through them?"""
        if self._dp_pending:
            self._dp_flag(failed=False)

    def _dp_flag(self, failed: bool) -> None:
        """Every rank says whether it failed in this call's passes; if one did, the step is abandoned on every rank
        (everything accumulated since the last step is dropped: the ranks' gradients cannot be told apart by call),
        and a rank that did not fail raises."""
        torch = self.torch
        flag = torch.tensor([1.0 if failed else 0.0], device=self.device)
        torch.distributed.all_reduce(flag)
        self._dp_pending = False
        n = int(flag.item())
        if n:
            self.optimizer.zero_grad(set_to_none=True)
            self._pending_items = 0
            if not failed:
                raise DataParallelError(f"{n} other rank{'s' if n > 1 else ''} failed in this call (their errors are "
                                        f"reported with them); the step it was part of is abandoned on every rank")

    def _dp_all_reduce_grads(self) -> None:
        """The adapter's gradients summed across ranks, in flat buffers of at most BUCKET_BYTES."""
        torch = self.torch
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        buckets: dict[Any, list[tuple[list[Any], list[int]]]] = {}   # per dtype: [(gradients, [their bytes])]
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            size = p.grad.numel() * p.grad.element_size()
            groups = buckets.setdefault(p.grad.dtype, [])
            if not groups or (groups[-1][0] and sum(groups[-1][1]) + size > BUCKET_BYTES):
                groups.append(([], []))
            groups[-1][0].append(p.grad)
            groups[-1][1].append(size)
        for groups in buckets.values():
            for grads, _ in groups:
                flat = _flatten_dense_tensors(grads)
                torch.distributed.all_reduce(flat)
                for g, summed in zip(grads, _unflatten_dense_tensors(flat, grads), strict=True):
                    g.copy_(summed)
                del flat

    def _dp_broadcast(self, optimizer: bool) -> None:
        """Every rank takes rank 0's adapter and, with ``optimizer``, its optimizer moments (every rank must have the
        same optimizer state keys: after a step, or none)."""
        torch = self.torch
        dist = torch.distributed
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)
            if optimizer:
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        state = self.optimizer.state.get(p, {})
                        for key in sorted(state):
                            if torch.is_tensor(state[key]):
                                buf = state[key].to(self.device)
                                dist.broadcast(buf, src=0)
                                state[key].copy_(buf)

    def _dp_pair_weight(self) -> float:
        """Attention's cost per (query, key) pair relative to the rest of the model's per token: 4 * heads * head_dim
        operations per pair per layer (the scores and the weighted values) against 2 per weight per token."""
        if self._dp_pair_cost is None:
            try:
                base = self.model.get_base_model()
                cfg = base.config
                heads = cfg.num_attention_heads
                head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
                weights = sum(p.numel() for name, p in base.named_parameters()
                              if "lora_" not in name and "embed" not in name and "lm_head" not in name)
                self._dp_pair_cost = 2.0 * cfg.num_hidden_layers * heads * head_dim / weights
            except AttributeError:
                self._dp_pair_cost = DEFAULT_PAIR_COST
        return self._dp_pair_cost


# -- the front-end: one process, several GPUs --------------------------------------------------------------------------


def _join_group(backend: str, port: int, rank: int, world_size: int, timeout_s: float, device: str,
                store: Any = None) -> None:
    import torch
    import torch.distributed as dist

    timeout = timedelta(seconds=timeout_s)
    if store is None:
        store = dist.TCPStore("127.0.0.1", port, world_size, is_master=False, timeout=timeout)
    extra = {}
    if device.startswith("cuda"):
        # the group binds to a device with an index (torch 2.13 refuses "cuda" alone): a trainer built on "cuda" is on
        # the current one
        bound = torch.device(device)
        extra["device_id"] = bound if bound.index is not None else torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group(backend, store=store, rank=rank, world_size=world_size, timeout=timeout, **extra)


def _group_failed(error: BaseException) -> bool:
    """Whether ``error`` is the process group's own (a collective, its connection or its store failed), after which no
    rank can count on another."""
    import torch.distributed as dist

    return isinstance(error, getattr(dist, "DistError", ()))


def _reply(conn: Any, status: str, value: Any) -> None:
    """A worker's reply, pickled plainly: tensors by value (``Connection.send`` would move them through torch's shared
    memory, whose manager process, on macOS, keeps the worker's exit from being seen for a minute)."""
    conn.send_bytes(pickle.dumps((status, value), protocol=pickle.HIGHEST_PROTOCOL))


def _end_with(parent: int) -> None:
    """A worker's watch on its front-end, process ``parent``: once that process is gone the worker ends at once, even
    in the middle of a call, where a step's collective would wait for rank 0 until the group's timeout, holding the
    worker's GPU from the trainer started next."""
    while os.getppid() == parent:
        time.sleep(1.0)
    os._exit(0)


def _worker(rank: int, world_size: int, device: str, backend: str, port: int, timeout_s: float, model_name: str,
            kwargs: dict[str, Any], conn: Any) -> None:
    """One rank of a :class:`DataParallelTrainer`, in its own process: builds its trainer, joins the group when told,
    then runs what the front-end sends (a trainer method, or a function of the trainer, with its arguments) until it is
    told to close or, within a second, once the front-end's process is gone (:func:`_end_with`). Ctrl-C and SIGTERM are
    the front-end's to handle: a preemptible machine sends SIGTERM to every process at once, and the front-end, which
    writes a step checkpoint when it comes, needs every rank through the step in flight first."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    threading.Thread(target=_end_with, args=(os.getppid(),), name="end-with-front-end", daemon=True).start()
    try:
        import torch

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        from continual_arc_training.trainer import Trainer

        trainer = Trainer(model_name, device=device, **kwargs)
    except BaseException:
        _reply(conn, "error", traceback.format_exc())
        return
    _reply(conn, "ok", None)
    while True:
        try:
            name, args, kw, want = pickle.loads(conn.recv_bytes())
        except (EOFError, OSError):
            return
        if name == "close":
            break
        try:
            if name == "join":
                _join_group(backend, port, rank, world_size, timeout_s, device)
                result = trainer.join_data_parallel()
            elif isinstance(name, str):
                result = getattr(trainer, name)(*args, **kw)
            else:
                result = name(trainer, *args, **kw)
            _reply(conn, "ok", result if want else None)
        except BaseException:
            _reply(conn, "error", traceback.format_exc())
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


class DataParallelTrainer:
    """The trainer's methods over ``world_size`` GPUs from one process: this process is rank 0 (its trainer,
    ``local``, on the first device) and one persistent worker process per other device holds its own (spawned, each
    loading the model while this one does). The passes, ``step``, ``reset``, ``configure_behaviour`` and ``load`` run
    on every rank (the chains and the random generator pickled to the workers; this process's generator advances as one
    process's would) and return rank 0's result, which holds every reply's statistics; ``save`` and everything else
    (``tokenizer``, ``model``) are rank 0's. NCCL on GPUs (ranks on cuda:0, cuda:1, ...), gloo on the CPU. With
    ``world_size`` 1 it is the trainer alone, in this process.

    The workers stop with ``close`` (at exit, too), or by themselves within a second of this process's end, mid-call
    too, and are terminated when this process is interrupted mid-call. A worker that dies mid-step leaves this one
    waiting until ``timeout_s``, the group's timeout; one that raises makes every rank raise, with its error attached."""

    def __init__(self, model_name: str, *, world_size: int, device: str = "cuda", timeout_s: float = 600.0,
                 **kwargs: Any) -> None:
        import torch

        from continual_arc_training.trainer import Trainer

        if world_size == 1:
            devices = [device]
        elif device.startswith("cuda"):
            devices = [f"cuda:{r}" for r in range(world_size)]
        else:
            devices = [device] * world_size
        self.world_size = world_size
        self._workers: list[tuple[Any, Any]] = []
        self._closed = False
        self._broken: str | None = None
        self._timeout_s = timeout_s
        if world_size == 1:
            self.local = Trainer(model_name, device=devices[0], **kwargs)
            return
        import torch.distributed as dist

        backend = "nccl" if devices[0].startswith("cuda") else "gloo"
        store = dist.TCPStore("127.0.0.1", 0, world_size, is_master=True, timeout=timedelta(seconds=timeout_s),
                              wait_for_workers=False)
        self._store = store
        ctx = torch.multiprocessing.get_context("spawn")
        atexit.register(self.close)
        try:
            for r in range(1, world_size):
                ours, theirs = ctx.Pipe()
                proc = ctx.Process(target=_worker, name=f"trainer-rank-{r}",
                                   args=(r, world_size, devices[r], backend, store.port, timeout_s, model_name,
                                         kwargs, theirs))
                proc.start()
                theirs.close()
                self._workers.append((proc, ours))
            if devices[0].startswith("cuda"):
                torch.cuda.set_device(torch.device(devices[0]))
            self.local = Trainer(model_name, device=devices[0], **kwargs)
            self._collect()                          # every worker's trainer is built
            self._send("join", (), {}, want=False)
            _join_group(backend, store.port, 0, world_size, timeout_s, devices[0], store=store)
            self.local.join_data_parallel()
            self._collect()
        except BaseException:
            self.close()
            raise

    # -- the trainer's methods that every rank runs ------------------------------------------------------------------

    def sft_chains(self, chains: list, **kwargs: Any) -> dict[str, Any]:
        return self._run("sft_chains", (chains,), kwargs)[0]

    def distill_chains(self, chains: list, teacher_chains: list, **kwargs: Any) -> dict[str, Any]:
        return self._run("distill_chains", (chains, teacher_chains), kwargs)[0]

    def policy_gradient_chains(self, chains: list, **kwargs: Any) -> dict[str, Any]:
        return self._run("policy_gradient_chains", (chains,), kwargs)[0]

    def nll_chains(self, chains: list) -> dict[str, Any]:
        return self._run("nll_chains", (chains,), {})[0]

    def distill_scores_chains(self, chains: list, teacher_chains: list, **kwargs: Any) -> dict[str, Any]:
        return self._run("distill_scores_chains", (chains, teacher_chains), kwargs)[0]

    def policy_gradient_scores_chains(self, chains: list) -> dict[str, Any]:
        return self._run("policy_gradient_scores_chains", (chains,), {})[0]

    def behaviour_logps_chains(self, chains: list) -> dict[str, Any]:
        return self._run("behaviour_logps_chains", (chains,), {})[0]

    def step(self) -> dict[str, Any]:
        return self._run("step", (), {})[0]

    def reset(self) -> None:
        self._run("reset", (), {})

    def configure_behaviour(self, adapter: Path | None) -> None:
        """``Trainer.configure_behaviour`` on every rank: each reads the same saved adapter from disk, or takes the
        untrained adapter every rank started from."""
        self._run("configure_behaviour", (adapter,), {})

    def load(self, path: Path) -> None:
        """Every rank reads the same saved adapter (one node: one filesystem)."""
        self._run("load", (Path(path),), {})

    def save(self, path: Path) -> None:
        self.local.save(path)

    def every_rank(self, fn: str | Callable, *args: Any, **kwargs: Any) -> list[Any]:
        """A trainer method (by name) or a function of the trainer (``fn(trainer, *args)``, importable by the workers)
        run on every rank; every rank's result, rank 0's first."""
        return self._run(fn, args, kwargs, want=True)

    def __getattr__(self, name: str) -> Any:
        if name in ("local", "_workers", "_closed", "_broken") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.local, name)

    # -- the workers -----------------------------------------------------------------------------------------------

    def _run(self, fn: str | Callable, args: tuple, kwargs: dict[str, Any], want: bool = False) -> list[Any]:
        """``fn`` on every rank: sent to the workers first, then run here as rank 0; every worker's reply is collected
        even when rank 0 fails (so the next call starts clean), and any failure raises here. A worker that has not
        replied within the group's timeout of rank 0's failure waits on it in a collective it will not reach, and a
        failure of the group itself leaves every worker so: the workers are then stopped and the trainer is broken,
        rather than waited for without end (NCCL's connection setup, which no timeout covers, can hold the other ranks
        for good after rank 0's first collective failed)."""
        call = getattr(self.local, fn) if isinstance(fn, str) else functools.partial(fn, self.local)
        if self.world_size == 1:
            return [call(*args, **kwargs)]
        self._send(fn, args, kwargs, want)
        try:
            mine = call(*args, **kwargs)
        except KeyboardInterrupt:
            self._broken = "interrupted"
            self._terminate()
            raise
        except BaseException as error:
            if _group_failed(error):
                self._broken = f"broken: rank 0's collective failed ({type(error).__name__})"
                self._terminate()
                raise
            try:
                self._collect(within=self._timeout_s)
            except DataParallelError as others:
                error.add_note(str(others))
            raise
        return [mine, *self._collect()]

    def _send(self, fn: str | Callable, args: tuple, kwargs: dict[str, Any], want: bool) -> None:
        if self._closed or self._broken:
            raise DataParallelError(f"the data-parallel trainer is {'closed' if self._closed else self._broken}")
        payload = pickle.dumps((fn, args, kwargs, want), protocol=pickle.HIGHEST_PROTOCOL)
        for proc, conn in self._workers:
            try:
                conn.send_bytes(payload)
            except OSError as error:
                self._broken = f"broken: a worker is gone (exit code {proc.exitcode}): {error}"
                raise DataParallelError(self._broken) from error

    def _collect(self, within: float | None = None) -> list[Any]:
        """Each worker's reply, in order; with ``within`` (seconds from now), a worker with none by then is taken to be
        stuck, and every worker is stopped (the trainer broken)."""
        deadline = None if within is None else time.time() + within
        results, errors = [], []
        for r, (proc, conn) in enumerate(self._workers, start=1):
            if deadline is not None and not conn.poll(max(0.0, deadline - time.time())):
                self._broken = f"broken: rank {r} did not reply within {within:.0f} s of rank 0's failure"
                self._terminate()
                errors.append(f"rank {r}: {self._broken}; every worker was stopped")
                break
            try:
                status, value = pickle.loads(conn.recv_bytes())
            except (EOFError, OSError):
                proc.join(timeout=5)
                self._broken = f"broken: rank {r}'s worker is gone (exit code {proc.exitcode})"
                status, value = "error", self._broken
            if status == "error":
                errors.append(f"rank {r}: {value}")
            results.append(value)
        if errors:
            raise DataParallelError("\n".join(errors))
        return results

    def close(self) -> None:
        """Stop the workers (each leaves the group and exits, as this process does) and leave the group; workers that
        do not stop within a minute, or any at all once a call was interrupted or a worker died, are terminated."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        if not self._broken:
            for _, conn in self._workers:
                try:
                    conn.send_bytes(pickle.dumps(("close", (), {}, False)))
                except OSError:
                    pass
            self._leave_group()
        self._terminate(wait_s=0 if self._broken else 60)
        for _, conn in self._workers:
            conn.close()
        self._leave_group()

    def _leave_group(self) -> None:
        if self.world_size > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()

    def _terminate(self, wait_s: float = 0) -> None:
        """Every worker gone: given ``wait_s`` to exit, then killed (a worker ignores SIGTERM: :func:`_worker`)."""
        for proc, _ in self._workers:
            proc.join(timeout=wait_s)
            if proc.is_alive():
                proc.kill()
        for proc, _ in self._workers:
            proc.join(timeout=10)

    def __enter__(self) -> DataParallelTrainer:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
