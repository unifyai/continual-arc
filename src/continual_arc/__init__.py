"""Continual-ARC: a continual learning benchmark built on ARC-AGI's three editions.

The benchmark is a single environment with a single interface. A *schedule* is a
sequence of (task_id, instance_seed) slots; the environment serves a freshly
generated test input for each slot and accepts two actions: submit a grid, or
request demonstration pairs. Feedback is correct/incorrect only. The score is the
number of wrong attempts plus the number of demo requests, averaged over the
stream. Everything else (tokens, wall-clock, memory) is logged but not scored.
"""

from continual_arc._version import __version__
from continual_arc.addons import (
    PHASE_ADDON_REVIEW,
    Addon,
    BaseAddon,
    BasePlugin,
    EpisodeRequest,
    Plugin,
    PluginContext,
    PluginManager,
    PluginResponse,
    PluginRuntime,
)
from continual_arc.env import ContinualArcEnv, EnvConfig
from continual_arc.grids import Grid, grids_equal, normalize_grid
from continual_arc.learner import Learner
from continual_arc.runner import run
from continual_arc.schedule import Schedule, Slot
from continual_arc.tasks import Pair, TaskRegistry

__all__ = [
    "__version__",
    "Addon",
    "BaseAddon",
    "BasePlugin",
    "ContinualArcEnv",
    "EnvConfig",
    "EpisodeRequest",
    "Grid",
    "Learner",
    "Pair",
    "PHASE_ADDON_REVIEW",
    "Plugin",
    "PluginContext",
    "PluginManager",
    "PluginResponse",
    "PluginRuntime",
    "Schedule",
    "Slot",
    "TaskRegistry",
    "grids_equal",
    "normalize_grid",
    "run",
]
