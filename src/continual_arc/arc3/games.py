"""The pinned ARC-AGI-3 games and a thin handle over the local engine.

ARC-AGI-3 games are interactive: a 64x64 grid of colours 0..15 that reacts to
seven actions (five simple, one with x/y coordinates, one more simple) plus a
reset. Each game has several levels; the ARC Prize publishes, per level, the
number of actions a human needed (``baseline_actions``). The 25 public games are
pinned here by version so a run is reproducible: :data:`GAMES` is the manifest,
:func:`fetch` downloads every game at its pinned version into the data root and
:func:`open_game` runs one locally, offline, through the ``arc-agi`` toolkit.

The toolkit (``pip install continual-arc[arc3]``, Python 3.12 or newer) is
imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continual_arc.data import data_root

ARC3_DIRNAME = "arc-agi-3"
ARC3_BASE_URL = "https://three.arcprize.org"

Frame = list[list[int]]
"""One 64x64 grid of colours 0..15."""

STATE_NOT_PLAYED = "NOT_PLAYED"
STATE_NOT_FINISHED = "NOT_FINISHED"
STATE_WIN = "WIN"
STATE_GAME_OVER = "GAME_OVER"

ACTION_IDS = (1, 2, 3, 4, 5, 6, 7)
COMPLEX_ACTION = 6
"""ACTION6 needs ``x`` and ``y`` in 0..63; every other action is simple."""


@dataclass(frozen=True)
class GameInfo:
    game_id: str
    version: str
    title: str
    tags: tuple[str, ...]
    baseline_actions: tuple[int, ...]
    """Human baseline: actions needed per level, as published by the ARC Prize."""

    @property
    def n_levels(self) -> int:
        return len(self.baseline_actions)

    @property
    def full_id(self) -> str:
        return f"{self.game_id}-{self.version}"


def _g(game_id: str, version: str, title: str, tags: tuple[str, ...], baseline: tuple[int, ...]) -> GameInfo:
    return GameInfo(game_id, version, title, tags, baseline)


# The 25 public games as listed by the ARC-AGI-3 API on 2026-09-18, pinned by version.
GAMES: dict[str, GameInfo] = {
    g.game_id: g
    for g in (
        _g("ar25", "0c556536", "AR25", ("keyboard_click",), (32, 50, 75, 37, 89, 159, 233, 73)),
        _g("bp35", "0a0ad940", "BP35", ("keyboard_click",), (21, 48, 44, 38, 33, 87, 86, 131, 163)),
        _g("cd82", "fb555c5d", "CD82", ("keyboard_click",), (55, 8, 41, 21, 23, 23)),
        _g("cn04", "2fe56bfb", "CN04", ("keyboard_click",), (29, 54, 85, 300, 208, 113)),
        _g("dc22", "fdcac232", "DC22", ("keyboard_click",), (59, 102, 67, 98, 324, 578)),
        _g("ft09", "0d8bbf25", "FT09", (), (43, 12, 23, 28, 65, 37)),
        _g("g50t", "5849a774", "G50T", ("keyboard",), (78, 175, 179, 230, 96, 54, 67)),
        _g("ka59", "38d34dbb", "KA59", ("keyboard_click",), (28, 109, 51, 51, 33, 132, 326)),
        _g("lf52", "271a04aa", "LF52", ("click",), (32, 81, 60, 71, 205, 148, 244, 109, 164, 225)),
        _g("lp85", "305b61c3", "LP85", ("click",), (17, 38, 31, 16, 41, 60, 26, 159)),
        _g("ls20", "9607627b", "LS20", ("keyboard",), (22, 123, 73, 84, 96, 192, 186)),
        _g("m0r0", "492f87ba", "M0R0", ("keyboard_click",), (30, 111, 203, 26, 500, 237)),
        _g("r11l", "495a7899", "R11L", ("click",), (22, 33, 51, 26, 52, 49)),
        _g("re86", "8af5384d", "RE86", ("keyboard_click",), (26, 42, 86, 108, 189, 139, 424, 241)),
        _g("s5i5", "18d95033", "S5I5", ("click",), (20, 89, 106, 54, 162, 38, 86, 83)),
        _g("sb26", "7fbdac44", "SB26", ("keyboard_click",), (18, 28, 18, 19, 31, 23, 58, 18)),
        _g("sc25", "635fd71a", "SC25", ("keyboard_click",), (36, 6, 32, 83, 143, 50)),
        _g("sk48", "d8078629", "SK48", ("keyboard_click",), (61, 177, 101, 103, 230, 181, 125, 92)),
        _g("sp80", "589a99af", "SP80", ("keyboard_click",), (39, 58, 25, 148, 96, 152)),
        _g("su15", "1944f8ab", "SU15", ("click",), (22, 42, 26, 115, 36, 31, 8, 40, 41)),
        _g("tn36", "ef4dde99", "TN36", ("click",), (32, 72, 26, 40, 30, 55, 62)),
        _g("tr87", "cd924810", "TR87", ("keyboard",), (54, 58, 40, 45, 71, 146)),
        _g("tu93", "0768757b", "TU93", ("keyboard_click",), (19, 16, 34, 42, 123, 80, 14, 23, 111)),
        _g("vc33", "5430563c", "VC33", ("click",), (7, 18, 44, 61, 131, 34, 152)),
        _g("wa30", "ee6fef47", "WA30", ("keyboard",), (71, 119, 183, 98, 368, 68, 79, 442, 415)),
    )
}


def game_ids() -> list[str]:
    return sorted(GAMES)


def get_game(game_id: str) -> GameInfo:
    """Look a game up by short id (``ls20``) or full id (``ls20-9607627b``)."""
    short = game_id.split("-", 1)[0]
    try:
        info = GAMES[short]
    except KeyError:
        raise KeyError(f"unknown ARC-AGI-3 game {game_id!r}; choose from {game_ids()}") from None
    if "-" in game_id and game_id != info.full_id:
        raise KeyError(f"{game_id!r} is not the pinned version {info.full_id!r}")
    return info


# ---------------------------------------------------------------------------
# data root
# ---------------------------------------------------------------------------


def arc3_dir() -> Path:
    return data_root() / ARC3_DIRNAME


def environments_dir() -> Path:
    """Where the toolkit keeps ``<game>/<version>/metadata.json`` and ``<game>.py``."""
    return arc3_dir() / "environment_files"


def game_dir(game_id: str) -> Path:
    info = get_game(game_id)
    return environments_dir() / info.game_id / info.version


def _present(info: GameInfo) -> bool:
    folder = environments_dir() / info.game_id / info.version
    return (folder / "metadata.json").is_file() and (folder / f"{info.game_id}.py").is_file()


def status() -> dict[str, dict[str, object]]:
    """Per game: whether the pinned version is present locally.

    The shape matches :func:`continual_arc.data.status` so the two can be merged:
    ``path``, ``present``, ``pinned`` (the version), ``head`` (the version found)
    and ``ok``.
    """
    out: dict[str, dict[str, object]] = {}
    for gid in game_ids():
        info = GAMES[gid]
        ok = _present(info)
        out[gid] = {
            "path": str(environments_dir() / info.game_id / info.version),
            "present": ok,
            "pinned": info.version,
            "head": info.version if ok else None,
            "ok": ok,
        }
    return out


def _toolkit():
    try:
        import arc_agi
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ARC-AGI-3 support needs the arc-agi toolkit (Python 3.12 or newer): "
            "pip install 'continual-arc[arc3]'"
        ) from e
    return arc_agi


def _logger() -> logging.Logger:
    return logging.getLogger("continual_arc.arc3")


def _write_metadata(info: GameInfo, folder: Path) -> None:
    """Pin ``metadata.json`` to the manifest so the offline scan finds the exact version."""
    import json

    path = folder / "metadata.json"
    meta: dict[str, Any] = {}
    if path.is_file():
        try:
            meta = json.loads(path.read_text())
        except ValueError:
            meta = {}
    meta.update(
        {
            "game_id": info.full_id,
            "title": info.title,
            "tags": list(info.tags),
            "baseline_actions": list(info.baseline_actions),
            "class_name": info.game_id[0].upper() + info.game_id[1:],
            "local_dir": str(folder),
        }
    )
    path.write_text(json.dumps(meta, indent=2))


def fetch(force: bool = False) -> dict[str, str]:
    """Download every manifest game at its pinned version. Returns game id -> path.

    Uses the toolkit in ``NORMAL`` mode, which fetches an anonymous key and
    the game source from the ARC Prize server (``ARC_API_KEY`` is honoured).
    Games already present are skipped unless ``force``.
    """
    arc_agi = _toolkit()
    env_dir = environments_dir()
    env_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    arcade = None
    for gid in game_ids():
        info = GAMES[gid]
        folder = env_dir / info.game_id / info.version
        if force and folder.exists():
            shutil.rmtree(folder)
        if not _present(info):
            if arcade is None:
                arcade = arc_agi.Arcade(
                    operation_mode=arc_agi.OperationMode.NORMAL,
                    environments_dir=str(env_dir),
                    logger=_logger(),
                )
            wrapper = arcade.make(info.full_id)
            if wrapper is None or not _present(info):
                raise RuntimeError(f"could not download ARC-AGI-3 game {info.full_id}")
            _write_metadata(info, folder)
        out[gid] = str(folder)
    return out


# ---------------------------------------------------------------------------
# game handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOutcome:
    frames: list[Frame]
    """Frames the engine rendered for this call; several when the action animates,
    none when the engine ignored the action (state was ``GAME_OVER`` or ``WIN``)."""
    state: str
    levels_completed: int
    level_completed: bool
    """The level started by :meth:`GameHandle.start_level` is now complete."""
    available_actions: list[int]
    ignored: bool


class GameHandle:
    """One local game, driven one level at a time.

    Wraps the toolkit's ``LocalEnvironmentWrapper`` in offline mode. The engine
    counts ``levels_completed`` from level 0, so :meth:`start_level` jumps to the
    level and sets the engine's score to the level index; completing the level
    then advances ``levels_completed`` to ``level + 1`` exactly as if the earlier
    levels had been played.
    """

    def __init__(self, wrapper: Any, info: GameInfo, seed: int):
        game = getattr(wrapper, "_game", None)
        if game is None:
            raise RuntimeError(f"toolkit failed to load game {info.full_id}")
        self._wrapper = wrapper
        self._game = game
        self.info = info
        self.game_id = info.game_id
        self.n_levels = info.n_levels
        self.baseline_actions = list(info.baseline_actions)
        self.seed = seed
        self.accepts_seed = _accepts_seed(type(game))
        self._level = 0
        self._last: StepOutcome | None = None

    # -- state ---------------------------------------------------------------

    @property
    def level(self) -> int:
        """Index of the level started by the last :meth:`start_level`."""
        return self._level

    @property
    def state(self) -> str:
        return self._last.state if self._last is not None else STATE_NOT_PLAYED

    @property
    def levels_completed(self) -> int:
        return self._last.levels_completed if self._last is not None else 0

    @property
    def available_actions(self) -> list[int]:
        return list(self._last.available_actions) if self._last is not None else []

    # -- driving --------------------------------------------------------------

    def start_level(self, level_index: int) -> Frame:
        """Jump to ``level_index`` clean and return its first frame.

        No action is spent: the engine is put on the level directly and the
        frame comes from a level reset routed through the engine, which is the
        same path a ``RESET`` takes mid-level (the game's own reset handling
        runs, and the action count stays at zero).
        """
        if not 0 <= level_index < self.n_levels:
            raise IndexError(f"level {level_index} out of range for {self.game_id} ({self.n_levels} levels)")
        game = self._game
        game.full_reset()
        game.set_level(level_index)
        game.level_reset()
        self._level = level_index
        outcome = self._engine_level_reset()
        if not outcome.frames:
            raise RuntimeError(f"{self.game_id}: no frame after starting level {level_index}")
        return outcome.frames[-1]

    def act(self, action_id: int, x: int | None = None, y: int | None = None) -> StepOutcome:
        """Send ``ACTION<action_id>`` (with ``x``/``y`` for ACTION6) and read the result."""
        from arcengine import GameAction

        if action_id not in ACTION_IDS:
            raise ValueError(f"action id must be in {ACTION_IDS}, got {action_id!r}")
        action = GameAction.from_id(action_id)
        data = {"x": int(x), "y": int(y)} if action_id == COMPLEX_ACTION else None
        if action_id == COMPLEX_ACTION and (x is None or y is None):
            raise ValueError("ACTION6 needs x and y")
        fd = self._wrapper.step(action, data=data)
        if fd is None:
            raise RuntimeError(f"{self.game_id}: engine failed on action {action_id}")
        return self._outcome(fd)

    def reset_level(self) -> StepOutcome:
        """Reload the current level. Never resets the whole game."""
        return self._engine_level_reset()

    def close(self) -> None:
        self._wrapper = None
        self._game = None

    # -- internals ------------------------------------------------------------

    def _engine_level_reset(self) -> StepOutcome:
        game = self._game
        # handle_reset() does a full reset (back to level 0, score 0) when the
        # action count is 0 or the state is WIN; both are made into a level reset.
        if game._state == _game_state().WIN:
            game.level_reset()
        game._action_count = max(1, game._action_count)
        game._score = self._level
        fd = self._wrapper.reset()
        if fd is None:
            raise RuntimeError(f"{self.game_id}: engine failed to reset level {self._level}")
        assert game.level_index == self._level, "level reset changed level"
        game._score = self._level
        fd.levels_completed = self._level
        return self._outcome(fd)

    def _outcome(self, fd: Any) -> StepOutcome:
        frames = [f.tolist() for f in fd.frame]
        state = fd.state.value if hasattr(fd.state, "value") else str(fd.state)
        out = StepOutcome(
            frames=frames,
            state=state,
            levels_completed=int(fd.levels_completed),
            level_completed=int(fd.levels_completed) > self._level or state == STATE_WIN,
            available_actions=[int(a) for a in fd.available_actions],
            ignored=not frames,
        )
        self._last = out
        return out


def _game_state():
    from arcengine import GameState

    return GameState


def _accepts_seed(cls: type) -> bool:
    import inspect

    try:
        return "seed" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False


def open_game(game_id: str, seed: int = 0) -> GameHandle:
    """Open a pinned game offline. ``seed`` reaches games whose constructor takes one."""
    arc_agi = _toolkit()
    info = get_game(game_id)
    if not _present(info):
        raise FileNotFoundError(
            f"ARC-AGI-3 game {info.full_id} not found under {environments_dir()}; "
            "run `continual-arc data fetch --datasets arc3`"
        )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(environments_dir()),
        logger=_logger(),
    )
    wrapper = arcade.make(info.full_id, seed=int(seed))
    if wrapper is None:
        raise RuntimeError(f"toolkit could not open {info.full_id} from {environments_dir()}")
    return GameHandle(wrapper, info, int(seed))
