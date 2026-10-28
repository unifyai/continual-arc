import pytest

from continual_arc.data import status
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource


def data_available(dataset: str = "arc1") -> bool:
    return all(s["ok"] for s in status((dataset,)).values())


needs_data = pytest.mark.skipif(not data_available(), reason="pinned data not fetched")
needs_arc2 = pytest.mark.skipif(not data_available("arc2"), reason="the second track's pinned data not fetched")


def arc3_available() -> bool:
    try:
        import arcengine  # noqa: F401
    except ImportError:
        return False
    return data_available("arc3")


needs_arc3 = pytest.mark.skipif(not arc3_available(), reason="arcengine or the pinned ARC-AGI-3 games missing")


@pytest.fixture
def fake_source():
    return FakeSource(n_tasks=12)


@pytest.fixture
def fake_registry(fake_source):
    reg = TaskRegistry({fake_source.name: fake_source})
    return reg


def as_users(sched, n):
    """A single stream laid out as ``n`` users per round, the users named by position, for tests
    of the round machinery (a configuration served to several users builds its own with interleave_users)."""
    import dataclasses

    sched.users = n
    rounds = sched.rounds()
    position = []
    for i, r in enumerate(rounds):
        position.append(0 if i == 0 or r != rounds[i - 1] else position[-1] + 1)
    sched.slots = [dataclasses.replace(x, user=u) for x, u in zip(sched.slots, position, strict=True)]
    return sched
