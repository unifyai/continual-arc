"""ARC-AGI-3 support: pinned interactive games served one level per instance.

:mod:`continual_arc.arc3.games` holds the manifest of the 25 public games and a
handle over the local engine; :mod:`continual_arc.arc3.env` is the environment
that the shared runner and metrics drive. The ``arc-agi`` toolkit is imported
only when a game is opened or fetched.
"""

from continual_arc.arc3.env import (
    SOURCE_ARC3,
    Act,
    ActFeedback,
    Arc3Config,
    Arc3Env,
    Arc3Observation,
    ResetFeedback,
    ResetLevel,
)
from continual_arc.arc3.games import GAMES, GameInfo, fetch, game_ids, open_game, status

__all__ = [
    "GAMES",
    "SOURCE_ARC3",
    "Act",
    "ActFeedback",
    "Arc3Config",
    "Arc3Env",
    "Arc3Observation",
    "GameInfo",
    "ResetFeedback",
    "ResetLevel",
    "fetch",
    "game_ids",
    "open_game",
    "status",
]
