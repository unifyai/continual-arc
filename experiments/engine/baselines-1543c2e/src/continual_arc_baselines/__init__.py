"""Baselines and experiment harness for Continual-ARC.

Three repositories, three responsibilities:

* ``continual-arc`` is the benchmark: environment, schedules, metrics. It has no
  model, no network and no tracking.
* ``continual-workbench`` is the shared execution and evidence layer: attempt
  directories, redacted logs, resource sampling, OpenRouter accounting, MLflow.
* this package glues them together and will hold the baseline learners.

The glue is deliberately thin. :mod:`continual_arc_baselines.evidence` turns a
continual-arc run into workbench metrics and artifacts;
:mod:`continual_arc_baselines.llm` wraps the workbench's OpenRouter client with a
retry policy and per-learner usage counters; :mod:`continual_arc_baselines.cli`
is the command that workbench experiment specs invoke.
"""

import inspect

from continual_arc import runner as _runner

__version__ = "0.1.0"

# The adapters pass a system each task's name as the benchmark's runner gives it, so under a runner that still shows a
# learner the whole slot, the real task id (a lookup key into the public solutions) would reach the system as it is.
# Each machine pins the benchmark in its own uv.lock, which git does not track, so loading against a benchmark from
# before the restricted learner view (continual-arc ad70ea6) is refused rather than left to leak the id.
if "view" in inspect.signature(_runner.run).parameters:
    raise ImportError("continual-arc predates the restricted learner view (continual-arc ad70ea6): update it "
                      "(uv lock --upgrade-package continual-arc && uv sync, or reinstall its checkout)")
