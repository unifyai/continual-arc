"""The system loop's own policies, exercised on a stub system."""
import threading
import time

from continual_arc.env import LearnerObservation, RequestDemos

from continual_arc_baselines.systems import base
from continual_arc_baselines.systems.base import SystemLearner, SystemUnavailable


class Stub(SystemLearner):
    """Answers from a script; records starts so a restart is visible."""

    def __init__(self, replies, **kwargs):
        super().__init__(**kwargs)
        self.replies = list(replies)
        self.starts = 0

    def start(self):
        self.starts += 1

    def stop(self):
        pass

    def ask(self, message, *, reprompt=False):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def new_session(self):
        pass

    def wipe_state(self):
        pass

    def describe(self):
        return {}


def test_empty_reply_restarts_the_system_like_a_crash():
    stub = Stub(["", '{"action": "request_demos"}'], max_restarts=2)
    assert stub._ask_resilient(lambda fresh: "obs") == '{"action": "request_demos"}'
    assert stub.stats["restarts"] == 1
    assert stub.starts == 2


class Resuming(Stub):
    """A stub whose start takes up the conversation it had, as openclaw's does, and whose first new session fails."""

    def __init__(self, replies, **kwargs):
        super().__init__(replies, **kwargs)
        self.failures, self.opened = 1, []

    def new_session(self):
        if self.failures:
            self.failures -= 1
            raise SystemUnavailable("the new session failed")
        self.opened.append(self.starts)


def test_an_instance_waits_for_its_conversation_through_a_restart(tmp_path):
    """An instance sends its first message to a conversation opened for it, the first instance a learner plays
    included: a new session that fails is opened again once the system has restarted, and only then is the instance
    sent."""
    stub = Resuming(['{"action": "request_demos"}'], max_restarts=2)
    stub.attach(tmp_path / "system", {})
    first = LearnerObservation(step=0, task_id="t1", input=[[0]], attempts_used=0, demo_requests_used=0,
                               attempt_cap=8, demo_cap=8)
    assert isinstance(stub.act(first), RequestDemos)
    assert stub.starts == 2 and stub.opened == [2]
    assert stub.stats["restarts"] == 1 and stub.stats["sessions"] == 1


def test_persistently_empty_system_is_unavailable_after_its_restarts():
    stub = Stub(["", "   ", "\n"], max_restarts=2)
    try:
        stub._ask_resilient(lambda fresh: "obs")
    except SystemUnavailable as error:
        assert "empty reply" in str(error)
    else:
        raise AssertionError("an empty system must be unavailable, not answering")
    assert stub.stats["restarts"] == 2


def test_a_process_starts_a_few_systems_at_a_time():
    """A collective round starts its hundred users' systems together: STARTS_AT_ONCE start at a time, every one starts,
    and a start that fails gives its turn up."""
    lock, starting, most = threading.Lock(), [0], [0]

    class Slow(Stub):
        def start(self):
            with lock:
                starting[0] += 1
                most[0] = max(most[0], starting[0])
            time.sleep(0.2)
            with lock:
                starting[0] -= 1
            super().start()
            if self.fails:
                raise SystemUnavailable("start failed")

    stubs = [Slow([]) for _ in range(3 * base.STARTS_AT_ONCE)]
    for i, stub in enumerate(stubs):
        stub.fails = i % 2 == 0

    def begin(stub):
        try:
            stub._ensure_started()
        except SystemUnavailable:
            pass

    threads = [threading.Thread(target=begin, args=(stub,)) for stub in stubs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert most[0] == base.STARTS_AT_ONCE
    assert all(stub.starts == 1 for stub in stubs) and [stub._started for stub in stubs] == [not s.fails for s in stubs]
