"""The learner interface.

A learner is any object with the methods below. The benchmark is method agnostic:
weight updates, external memory, program libraries and plain prompting are all
fine. The runner calls :meth:`reset` whenever the schedule says the learner's
persistent state must be wiped (the fresh reference), and reads :meth:`resources`
after every action to log, not score, whatever the learner chooses to report
(tokens, seconds, bytes of state, dollars). A learner that trains on its own
traces reports how many it has consumed under ``train_instances``: the data it
learns from is compute, plotted beside tokens and never part of the score, since
every instance is work the stream demanded and the only data the learner chooses
to buy is demonstrations, which the score already charges.

Instances are served in rounds, one instance per user of the schedule
(:attr:`continual_arc.schedule.Schedule.users`); every observation says whose work it
is (``obs.user``). The runner calls :meth:`begin_round` before the first action of a
round and :meth:`end_round` after its last instance has ended. The contract of the
collective regime is that what a learner carries between instances changes only in
:meth:`end_round`: the users of a round all meet the state the round began with, and
each user's working state within an instance is their own. In the individual regime
every instance is a round of one and the hooks bracket it.

The users of a round work at the same time. A learner that can serve them at the same
time says so with ``concurrent_users = True``: the runner then serves every user's
instance on a thread of its own, from its first action to its verdict, calling
:meth:`act` and :meth:`observe` for different users concurrently (never two calls for
one user at once), and the round ends when every user's instance has. Since instances
then overlap in time, such a learner reports its counters per user,
``user_resources(user) -> dict[str, float]`` (cumulative, like :meth:`resources`), and
an instance is charged its user's differences. Any other learner is served one action
at a time: one action on every open instance of the round before a second on any.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from continual_arc.env import Action, Feedback, Observation


@runtime_checkable
class Learner(Protocol):
    def reset(self) -> None:
        """Wipe all persistent state. Called at schedule-defined boundaries."""
        ...

    def act(self, obs: Observation) -> Action:
        """Choose the next action for the current instance."""
        ...

    def observe(self, obs: Observation, action: Action, feedback: Feedback) -> None:
        """Receive the environment's response to ``action``."""
        ...

    def resources(self) -> dict[str, float]:
        """Cumulative resource counters. The runner logs per-instance differences."""
        ...

    def begin_round(self, n_instances: int) -> None:
        """A round of ``n_instances`` is about to be served."""
        ...

    def end_round(self) -> None:
        """Every instance of the round has ended; persistent state may change now."""
        ...


class BaseLearner:
    """Convenience base with no-op hooks; subclasses need only ``act``."""

    concurrent_users = False
    """True for a learner that can serve a round's users at the same time (see the module docstring)."""

    def reset(self) -> None:
        pass

    def begin_round(self, n_instances: int) -> None:
        pass

    def end_round(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def observe(self, obs: Observation, action: Action, feedback: Feedback) -> None:
        pass

    def resources(self) -> dict[str, float]:
        return {}
