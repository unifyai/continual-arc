import httpx

from continual_arc_baselines.proxy import LLMProxy
from continual_arc_baselines.routing import ReplicaRouter, count_replicas, parse_metrics

from .fake_upstream import FakeUpstream

METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="m"} 9.0
vllm:num_requests_running{engine="1",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting{engine="1",model_name="m"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.97
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.4
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 5.5e+07
"""


class FakeMetrics:
    def __init__(self, text: str) -> None:
        self.text = text

    def get(self, url: str) -> "FakeMetrics":
        return self


def router(text: str, replicas: int) -> ReplicaRouter:
    r = ReplicaRouter("http://server/metrics", replicas, client=FakeMetrics(text))
    r.poll()
    return r


def test_the_replicas_are_counted_from_the_servers_metrics():
    assert count_replicas("http://server/metrics", client=FakeMetrics(METRICS)) == 2
    assert count_replicas("http://server/metrics", client=FakeMetrics("")) == 0  # none reported: the server routes


def test_the_servers_metrics_give_each_replicas_load():
    assert parse_metrics(METRICS, 2) == [[2.0, 9.0, 0.97], [0.0, 3.0, 0.4]]
    assert parse_metrics(METRICS, 1) == [[2.0, 9.0, 0.97]]  # an engine beyond the replicas is ignored


def test_instances_start_where_fewest_users_are_active_and_stay_there():
    r = router("", 4)
    homes = [r.pick(f"u{i}") for i in range(8)]
    assert sorted(homes) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [r.pick(f"u{i}") for i in range(8)] == homes  # each user's calls stay on their replica
    for i in (0, 1, 2, 3, 4):  # five instances end...
        r.release(f"u{i}")
    assert r.describe()["active_per_replica"] == [0, 1, 1, 1]  # homes were 0, 1, 2, 3, 0, 1, 2, 3
    # ...and the next one starts on the replica with the fewest active users, wherever it ran before
    assert r.pick("u3") == 0 and r.stats["moves"] == 0 and r.stats["instances"] == 9


def test_a_user_moves_only_off_a_queue_to_a_free_replica_with_fewer_active_users():
    r = router(METRICS, 2)  # replica 0 has 2 waiting, replica 1 none
    for i in range(3):
        r._home[f"u{i}"] = 0
        r._active.add(f"u{i}")
    r._home["u9"] = 1
    r._active.add("u9")
    assert r.pick("u0") == 1 and r.stats["moves"] == 1  # 1 active user there against 3 here
    assert r.pick("u0") == 1  # the new home sticks
    crowded = router(METRICS, 2)
    for i in range(2):
        crowded._home[f"u{i}"] = 0
        crowded._active.add(f"u{i}")
    for i in range(5, 8):
        crowded._home[f"u{i}"] = 1
        crowded._active.add(f"u{i}")
    assert crowded.pick("u0") == 0  # free now, but more users active there: stay with the cache
    busy = router(METRICS.replace('engine="1",model_name="m"} 0.0', 'engine="1",model_name="m"} 1.0'), 2)
    busy._home["u0"] = 0
    busy._active.add("u0")
    assert busy.pick("u0") == 0  # every replica has a queue: stay


def test_the_proxy_names_the_replica_to_the_server():
    server = FakeUpstream().start()
    r = router("", 3)
    try:
        with LLMProxy(server.base_url, router=r, route_key="u7") as proxy:
            for _ in range(2):
                httpx.post(proxy.base_url + "/chat/completions", json={"model": "m", "messages": [
                    {"role": "user", "content": "hi"}]}, timeout=10)
    finally:
        server.stop()
    ranks = [q["rank"] for q in server.requests]
    assert len(set(ranks)) == 1 and ranks[0] in {"0", "1", "2"}
