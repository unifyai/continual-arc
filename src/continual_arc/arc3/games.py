"""The pinned ARC-AGI-3 games, the games the track serves, and their fetch.

ARC-AGI-3's 25 public games are MIT-licensed Python on ARC Prize's ``arcengine``, each pinned here by version so that a
run is reproducible: :func:`fetch` downloads every game at its pinned version through the ``arc-agi`` toolkit into
``<data root>/arc-agi-3/environment_files/<game>/<version>/`` (``<game>.py`` and ``metadata.json``), where
:func:`continual_arc.arc3.levels.games_dir` reads them.

A game enters the track (:data:`TRACK_GAMES`) only once its generator module passes the gate
(:func:`continual_arc.arc3.levels.check` at 50 seeds). The track's games are split once, at :data:`SPLIT_SEED`, into
stream games (:data:`STREAM_SHARE` of them) and held-out games, which the stream never serves: every configuration of
the track draws on this one split, so its held-out games stay held out whatever the run's seed.
"""

from __future__ import annotations

import logging
import random
import shutil

from continual_arc.seeds import derive_seed

SOURCE_ARC3 = "arc3"
"""Source name of every ARC-AGI-3 game in a schedule (stream, epilogue and tests)."""

# The 25 public games as the ARC-AGI-3 API listed them on 18 Sep 2026, pinned by version.
GAMES: dict[str, str] = {
    "ar25": "0c556536", "bp35": "0a0ad940", "cd82": "fb555c5d", "cn04": "2fe56bfb", "dc22": "fdcac232",
    "ft09": "0d8bbf25", "g50t": "5849a774", "ka59": "38d34dbb", "lf52": "271a04aa", "lp85": "305b61c3",
    "ls20": "9607627b", "m0r0": "492f87ba", "r11l": "495a7899", "re86": "8af5384d", "s5i5": "18d95033",
    "sb26": "7fbdac44", "sc25": "635fd71a", "sk48": "d8078629", "sp80": "589a99af", "su15": "1944f8ab",
    "tn36": "ef4dde99", "tr87": "cd924810", "tu93": "0768757b", "vc33": "5430563c", "wa30": "ee6fef47",
}

# The games whose generator modules (continual_arc.arc3.generators.<game>) pass the gate at 50 seeds: the track.
TRACK_GAMES: tuple[str, ...] = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0", "r11l", "re86",
    "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
)
SPLIT_SEED = 0
STREAM_SHARE = 0.6


def split(games: tuple[str, ...] = TRACK_GAMES, seed: int = SPLIT_SEED) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``games`` split into stream games, :data:`STREAM_SHARE` of them rounded, drawn at ``seed``, and held-out games,
    the rest; each sorted."""
    pool = sorted(games)
    stream = sorted(random.Random(derive_seed(seed, "arc3-split")).sample(pool, round(STREAM_SHARE * len(pool))))
    return tuple(stream), tuple(g for g in pool if g not in stream)


STREAM_GAMES, HELD_OUT_GAMES = split()


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _folder(game_id: str):
    from continual_arc.arc3.levels import games_dir

    return games_dir() / game_id / GAMES[game_id]


def _present(game_id: str) -> bool:
    folder = _folder(game_id)
    return (folder / "metadata.json").is_file() and (folder / f"{game_id}.py").is_file()


def status() -> dict[str, dict[str, object]]:
    """Per game, whether its pinned version is present, in :func:`continual_arc.data.status`'s shape: ``path``,
    ``present``, ``pinned`` (the version), ``head`` (the version found) and ``ok``."""
    out: dict[str, dict[str, object]] = {}
    for game_id, version in sorted(GAMES.items()):
        ok = _present(game_id)
        out[game_id] = {"path": str(_folder(game_id)), "present": ok, "pinned": version,
                        "head": version if ok else None, "ok": ok}
    return out


def fetch(force: bool = False) -> dict[str, str]:
    """Download every game at its pinned version, through the toolkit in its ``NORMAL`` mode, which fetches an
    anonymous key and the game's source from the ARC Prize server (``ARC_API_KEY`` is honored). Games already present
    are skipped unless ``force``. Returns game id -> folder."""
    import arc_agi

    from continual_arc.arc3.levels import games_dir

    games_dir().mkdir(parents=True, exist_ok=True)
    arcade = None
    out: dict[str, str] = {}
    for game_id, version in sorted(GAMES.items()):
        folder = _folder(game_id)
        if force and folder.exists():
            shutil.rmtree(folder)
        if not _present(game_id):
            if arcade is None:
                arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL,
                                        environments_dir=str(games_dir()), logger=logging.getLogger(__name__))
            if arcade.make(f"{game_id}-{version}") is None or not _present(game_id):
                raise RuntimeError(f"could not download ARC-AGI-3 game {game_id}-{version}")
        out[game_id] = str(folder)
    return out
