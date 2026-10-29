"""The learners, harness adapters and tracking proxy evaluated on the Continual-ARC benchmark.

``continual-arc`` is the benchmark: environment, schedules, metrics, with no model, no network and no tracking. This
package holds what the paper evaluates on it: the bare-model floors and the program library
(:mod:`continual_arc_baselines.systems.bare`), the adapters that drive agent harnesses through the benchmark's protocol
(:mod:`continual_arc_baselines.systems`), the learner that serves a team of users at once
(:mod:`continual_arc_baselines.collective`), the proxy through which every model call passes
(:mod:`continual_arc_baselines.proxy`), and ``arc-baselines`` (:mod:`continual_arc_baselines.cli`), which runs a learner
over a schedule, writing the run directory and its attempt (:mod:`continual_arc_baselines.attempt`), and plays a
harness's frozen tests (:mod:`continual_arc_baselines.frozen`).
"""

import inspect

from continual_arc import runner as _runner

__version__ = "0.5.0"

# The adapters pass a system each task's name as the benchmark's runner gives it. A runner that takes a ``view``
# argument shows a learner the whole slot, the real task id (a lookup key into the public solutions) among it, so
# loading against such a benchmark is refused rather than left to leak the id.
if "view" in inspect.signature(_runner.run).parameters:
    raise ImportError("continual-arc predates the restricted learner view: update it "
                      "(uv lock --upgrade-package continual-arc && uv sync)")
