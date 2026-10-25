"""Sticky routing with spillover across a vLLM server's data-parallel replicas (25 Sep).

vLLM sends each request to its least-loaded replica, and each replica keeps a prefix cache of its own, so a harness
user's next call usually lands on a replica that has not seen most of their conversation and computes it again (25 Sep,
8 replicas: 59% of prompt tokens came from the cache, where prime-agent's append-only conversations on one replica would
give about 90%). :class:`ReplicaRouter` keeps each user on a home replica, named to vLLM in the ``X-data-parallel-rank``
header (:class:`continual_arc_baselines.proxy.LLMProxy`), for the whole of an instance.

What decides a replica is its active users, those in the middle of an instance, not how busy it is at the moment:
a replica free for a few seconds may serve many users about to call again.
- An instance starts on the replica with the fewest active users (its conversation is new, so nothing cached is lost).
- A user moves only when their replica has requests waiting, and then to a replica with none waiting and fewer active
  users than theirs, the one with the fewest: one recomputation of the conversation there, instead of a queue.
- :meth:`ReplicaRouter.release` marks a user's instance ended (the learner calls it).
Queues come from the server's ``/metrics``, polled about once a second; a call is placed when it is sent, since vLLM
cannot move a request already queued on a replica.
"""

from __future__ import annotations

import collections
import re
import threading
from typing import Any

import httpx

_METRIC = re.compile(r'^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)\{[^}]*engine="(\d+)"[^}]*\}\s+(\S+)')


def count_replicas(metrics_url: str, client: Any = None) -> int:
    """How many data-parallel replicas the server at ``metrics_url`` reports (0 when it cannot be read)."""
    try:
        text = (client or httpx.Client(timeout=10.0)).get(metrics_url).text
    except (httpx.HTTPError, OSError):
        return 0
    return len({m.group(2) for m in map(_METRIC.match, text.splitlines()) if m})


def parse_metrics(text: str, replicas: int) -> list[list[float]]:
    """Per replica: [requests waiting, requests running, KV cache use], from vLLM's Prometheus text."""
    load = [[0.0, 0.0, 0.0] for _ in range(replicas)]
    column = {"num_requests_waiting": 0, "num_requests_running": 1, "kv_cache_usage_perc": 2}
    for line in text.splitlines():
        m = _METRIC.match(line)
        if m and int(m.group(2)) < replicas:
            try:
                load[int(m.group(2))][column[m.group(1)]] = float(m.group(3))
            except ValueError:
                pass
    return load


class ReplicaRouter:
    """Which replica each user's next call goes to. Thread-safe; one per driver, shared by its users' proxies."""

    def __init__(self, metrics_url: str, replicas: int, *, poll_s: float = 1.0, client: Any = None) -> None:
        if replicas < 2:
            raise ValueError("routing needs at least two replicas")
        self.metrics_url, self.replicas, self.poll_s = metrics_url, replicas, poll_s
        self._client = client or httpx.Client(timeout=10.0)
        self._load = [[0.0, 0.0, 0.0] for _ in range(replicas)]
        self._home: dict[str, int] = {}
        self._active: set[str] = set()  # users in the middle of an instance
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"calls": 0, "instances": 0, "moves": 0, "polls": 0, "poll_failures": 0}

    def start(self) -> ReplicaRouter:
        self.poll()
        self._thread = threading.Thread(target=self._run, daemon=True, name="replica-router")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            self.poll()

    def poll(self) -> None:
        try:
            load = parse_metrics(self._client.get(self.metrics_url).text, self.replicas)
        except (httpx.HTTPError, OSError):
            self.stats["poll_failures"] += 1
            return
        with self._lock:
            self._load = load
            self.stats["polls"] += 1

    def _active_per_replica(self) -> collections.Counter:
        return collections.Counter(self._home[k] for k in self._active)

    def pick(self, key: str) -> int:
        with self._lock:
            self.stats["calls"] += 1
            load = self._load
            active = self._active_per_replica()
            if key not in self._active:  # a new instance: the replica with the fewest active users
                home = min(range(self.replicas), key=lambda r: (active[r], load[r][0] + load[r][1], r))
                self._home[key] = home
                self._active.add(key)
                self.stats["instances"] += 1
                return home
            home = self._home[key]
            if load[home][0] > 0:  # a queue here: a replica with none and fewer active users, the fewest
                free = [r for r in range(self.replicas)
                        if r != home and load[r][0] == 0 and active[r] < active[home]]
                if free:
                    best = min(free, key=lambda r: (active[r], load[r][1], r))
                    self._home[key] = best
                    self.stats["moves"] += 1
                    return best
            return home

    def release(self, key: str) -> None:
        """A user's instance has ended: they count as active nowhere until their next call."""
        with self._lock:
            self._active.discard(key)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            active = self._active_per_replica()
            return {**self.stats, "replicas": self.replicas,
                    "active_per_replica": [active[r] for r in range(self.replicas)]}
