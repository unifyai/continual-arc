"""World models of ARC-AGI-3 games: a game's rules written as a program, checked at no cost on every transition the
game has shown, and searched for the shortest sequence of actions that wins.

This is to a game what ``transform(grid)`` is to a grid task (:mod:`continual_arc_baselines.systems.bare`): the
program the program library keeps, checked on what the game has shown as a grid program is on its pairs. A program
defines ``step(state, action)``, the state after an action, and ``won(state)``; by default a state is the board
itself, 64 rows of 64 colors 0-15, top row first, and an action is ``[id]``, or ``[6, x, y]`` for a click. To carry
what a board does not show (a count, a history) a program defines ``init(board)``, its state at a level's first
board, and ``render(state)``, the board a state shows. A state whose board the program cannot tell, such as one after
an action whose outcome it does not model, renders as None (a program whose states are boards returns None from
``step``). It may define ``lost(state)``, a game over, ``clicks(state)``, the clicks worth trying, and ``key(state)``,
a value equal for states the search may treat as one.

A **trajectory** is what a learner saw of one level from its first board: the board, then steps, each the actions the
game applied with the cells they changed, and whether the level completed (``won``) or the game ended (``over``)
there, which ends it. A demonstration is one; so is each stretch of a learner's own play between resets
(:class:`Recorder`). :func:`check` runs a program from every trajectory's first board through every action played on
it: it must give each board the game showed, cell for cell, and say ``won``, and ``lost`` where it defines it, exactly
where the level completed or the game ended. The board after a win or a game over is never compared, so a program
need not draw an end screen; and running from each level's first board, never from a board midway, lets a program
track what the board hides. A state a program renders as None is never a win: no board is compared there and the
rest of that trajectory is not checked, but a level that completed there is a failure, so a program may leave an
outcome it cannot simulate unknown and still be held to where the level is won. :func:`plan` replays a level's trajectory so far
in the program and searches it breadth-first for the shortest winning sequence from there, over the actions the game
accepts and the clicks ``clicks(state)`` names (by default one cell of each one-colored patch of the board), states
merged by ``key``, never through a state that is lost or unknown, bounded by states expanded and guarded by a
deadline.

Standard library only, since it runs where the learner's code runs: the program library runs this file as the
script of a sandboxed process (``python world_model.py``, a JSON request on stdin and the answer on stdout), and the
expert guidance's adapter writes it into a harness's library directory as ``carc_games.py``, beside the instance's
boards (``instance.json``), where the agent imports :func:`load`, :func:`check_game`, :func:`plan_game` and
:func:`save`.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

SIZE = 64
MAX_STATES = 100_000
"""States a search expands at most."""
SECONDS = 60.0
"""The deadline of one check or one search, a guard on a program whose step is slow: the bound is ``MAX_STATES``."""
LISTED_ROWS = 24
"""Rows of differing cells a failure lists, of each side."""

Board = list[list[int]]


class ProgramError(Exception):
    """A program that cannot be used: it does not compile, lacks ``step`` or ``won``, or renders no board."""


# ---------------------------------------------------------------------------
# boards
# ---------------------------------------------------------------------------


def diff(before: Board, after: Board) -> list[list[int]]:
    """The cells ``after`` changed, ``[x, y, color]``, row by row."""
    return [[x, y, c] for y, (old, new) in enumerate(zip(before, after, strict=True))
            for x, (o, c) in enumerate(zip(old, new, strict=True)) if o != c]


def apply(board: Board, changes: list) -> Board:
    out = [list(row) for row in board]
    for x, y, c in changes:
        out[y][x] = c
    return out


def render_changes(cells) -> str:
    """Changed cells ``(x, y, color)``, one line a row, each a run of columns of one color: ``y12: 20-24=0 30=c``."""
    rows: dict[int, list[tuple[int, int]]] = {}
    for x, y, c in cells:
        rows.setdefault(y, []).append((x, c))
    lines = []
    for y in sorted(rows):
        runs: list[list[int]] = []   # [first x, last x, color]
        for x, c in sorted(rows[y]):
            if runs and runs[-1][1] == x - 1 and runs[-1][2] == c:
                runs[-1][1] = x
            else:
                runs.append([x, x, c])
        lines.append(f"y{y}: " + " ".join(f"{a}={c:x}" if a == b else f"{a}-{b}={c:x}" for a, b, c in runs))
    return "\n".join(lines)


def render_board(board: Board) -> str:
    return "\n".join("".join(format(c, "x") for c in row) for row in board)


def is_board(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == SIZE
            and all(isinstance(row, list) and len(row) == SIZE for row in value))


def as_board(value: Any) -> Board:
    """``value`` as a board, or ProgramError: 64 rows of 64 integers 0-15."""
    try:
        board = [[int(c) for c in row] for row in value]
    except (TypeError, ValueError):
        raise ProgramError("render(state) must return a board, 64 rows of 64 colors 0-15") from None
    if len(board) != SIZE or any(len(row) != SIZE for row in board) or any(c < 0 or c > 15 for r in board for c in r):
        raise ProgramError("render(state) must return a board, 64 rows of 64 colors 0-15")
    return board


def patches(board: Board) -> list[list[int]]:
    """One cell of each 4-connected one-colored patch of ``board``, the patch's cell nearest its center, in reading
    order: the clicks a search tries when a program names none."""
    seen = [[False] * SIZE for _ in range(SIZE)]
    out = []
    for y0 in range(SIZE):
        for x0 in range(SIZE):
            if seen[y0][x0]:
                continue
            color, cells, stack = board[y0][x0], [], [(x0, y0)]
            seen[y0][x0] = True
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < SIZE and 0 <= ny < SIZE and not seen[ny][nx] and board[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            cx = sum(x for x, _ in cells) / len(cells)
            cy = sum(y for _, y in cells) / len(cells)
            x, y = min(cells, key=lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2, p[1], p[0]))
            out.append([x, y])
    return sorted(out, key=lambda p: (p[1], p[0]))


def describe_action(action: list) -> str:
    return f"6 at ({action[1]}, {action[2]})" if action[0] == 6 and len(action) == 3 else str(action[0])


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------


def boards(trajectory: dict) -> list[Board]:
    """Every board of a trajectory: its first, then the one after each step (a step that won or ended the game
    changes nothing here, since its board is never compared)."""
    out = [trajectory["start"]]
    for step in trajectory["steps"]:
        out.append(apply(out[-1], step["changes"]))
    return out


def trajectory_of_demonstration(demo: dict, name: str) -> dict:
    """A demonstration as the benchmark serializes it (``Demonstration.to_json``: its first frame, then each action
    with the cells it changed) as a trajectory: one action a step, the last one winning."""
    steps = demo["steps"]
    return {"name": name, "start": demo["start"],
            "steps": [{"actions": [list(s["action"])], "changes": [list(c) for c in s["changes"]],
                       "won": i == len(steps) - 1, "over": False} for i, s in enumerate(steps)]}


class Recorder:
    """A learner's own play of one instance's level as trajectories. :meth:`board` gives a board the learner saw: the
    first opens a trajectory, and a later one closes a step of the actions applied since the last. :meth:`act` adds an
    action the game applied (not one it refused or ignored, which changed nothing): one that won closes its step at
    once, since its board is never compared, and one that ended the game closes its step at the next board. A reset
    ends the trajectory, and the next board opens one. Actions with no board after them, such as the last of an
    instance that reached its cap, are dropped.

    A learner that sees every board records a step per action; a conversation that shows the board only after each
    message records a step per message, of all the actions its sequence applied."""

    def __init__(self, name: str = "your play of this level"):
        self.name = name
        self.closed: list[dict] = []
        self.open: dict | None = None
        self.last: Board | None = None
        self.pending: list[list] = []
        self.over = False      # the game ended at the last action; its step closes at the next board
        self.ended = False     # the level completed or the game ended: nothing more until a reset
        self.resets = 0

    def board(self, frame: Board) -> None:
        if self.ended:
            return
        frame = [list(row) for row in frame]
        if self.open is None:
            name = self.name if not self.resets else f"{self.name} after reset {self.resets}"
            self.open = {"name": name, "start": frame, "steps": []}
        elif self.pending:
            self.open["steps"].append({"actions": self.pending, "changes": diff(self.last, frame), "won": False,
                                       "over": self.over})
            self.pending = []
            if self.over:
                self._end()
                return
        self.last = frame

    def act(self, action: list, *, won: bool = False, over: bool = False) -> None:
        if self.open is None or self.ended or self.over:
            return
        self.pending.append(list(action))
        if won:
            self.open["steps"].append({"actions": self.pending, "changes": [], "won": True, "over": False})
            self._end()
        elif over:
            self.over = True

    def reset(self) -> None:
        self._end()
        self.ended = False
        self.resets += 1

    def _end(self) -> None:
        if self.open is not None:
            self.closed.append(self.open)
        self.open, self.pending, self.over, self.last, self.ended = None, [], False, None, True

    @property
    def trajectories(self) -> list[dict]:
        """Every trajectory with a step, the open one included."""
        every = self.closed + ([self.open] if self.open is not None else [])
        return [t for t in every if t["steps"]]

    @property
    def current(self) -> dict | None:
        """The trajectory of the level as it stands, from its first board to the last board seen; None once the
        level completed or the game ended, until a reset."""
        return self.open


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted(((repr(k), _freeze(v)) for k, v in value.items()), key=lambda kv: kv[0]))
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class Program:
    """A model-written program, compiled, with every optional function's default."""

    def __init__(self, source: str):
        namespace: dict[str, Any] = {"__name__": "carc_program"}
        try:
            exec(compile(source, "program.py", "exec"), namespace)
        except Exception as error:   # whatever the model wrote, the failure is reported to it
            raise ProgramError(f"the program does not run: {type(error).__name__}: {error}") from None
        if not callable(namespace.get("step")) or not callable(namespace.get("won")):
            raise ProgramError("the program must define step and won")
        self.source = source
        self._step, self._won = namespace["step"], namespace["won"]
        self._init, self._render = namespace.get("init"), namespace.get("render")
        self._lost, self._clicks, self._key = namespace.get("lost"), namespace.get("clicks"), namespace.get("key")
        self.boards_are_states = self._init is None and self._render is None

    @staticmethod
    def _copy(state: Any) -> Any:
        return [list(row) for row in state] if is_board(state) else copy.deepcopy(state)

    def init(self, board: Board) -> Any:
        board = [list(row) for row in board]
        return self._init(board) if self._init is not None else board

    def step(self, state: Any, action: list) -> Any:
        return self._step(self._copy(state), list(action))

    def render(self, state: Any) -> Board | None:
        """The board ``state`` shows, None where the program cannot tell it."""
        board = self._render(state) if self._render is not None else state
        return None if board is None else as_board(board)

    def unknown(self, state: Any) -> bool:
        """Whether the program cannot tell ``state``'s board, without checking the board it gives."""
        return (self._render(state) if self._render is not None else state) is None

    def won(self, state: Any) -> bool:
        return bool(self._won(state))

    def lost(self, state: Any) -> bool:
        return self._lost is not None and bool(self._lost(state))

    @property
    def has_lost(self) -> bool:
        return self._lost is not None

    def clicks(self, state: Any) -> list[list[int]]:
        if self._clicks is None:
            return patches(self.render(state))
        return [[int(x), int(y)] for x, y in self._clicks(state) if 0 <= int(x) < SIZE and 0 <= int(y) < SIZE]

    def key(self, state: Any) -> Any:
        if self._key is not None:
            return _freeze(self._key(state))
        if self.boards_are_states:
            return tuple(map(tuple, state))
        return _freeze(state)


def compile_program(program: Program | str) -> Program:
    """A program from its source, or from the path of a ``.py`` file holding it."""
    if isinstance(program, Program):
        return program
    if program.rstrip().endswith(".py") and "\n" not in program.strip():
        program = Path(program).read_text()
    return Program(program)


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


def _failure(trajectory: dict, index: int, kind: str, step: int | None, **extra: Any) -> dict:
    """What :func:`check` found wrong: the trajectory (its position in the list and its name), the step (None for its
    first board), what kind of wrong, and what :func:`describe` needs to say it."""
    actions = [] if step is None else trajectory["steps"][step]["actions"]
    before = sum(len(s["actions"]) for s in trajectory["steps"][:step]) if step is not None else 0
    return {"trajectory": index, "name": trajectory.get("name", f"trajectory {index + 1}"), "step": step,
            "kind": kind, "actions": actions, "actions_before": before, **extra}


def check(program: Program | str, trajectories: list[dict], seconds: float = SECONDS) -> dict | None:
    """The first transition of ``trajectories`` that ``program`` gets wrong, or None when it reproduces them all (the
    module docstring). A program that cannot be compiled is a failure of kind ``program``."""
    try:
        program = compile_program(program)
    except ProgramError as error:
        return {"trajectory": None, "name": None, "step": None, "kind": "program", "error": str(error),
                "actions": [], "actions_before": 0}
    deadline = time.monotonic() + seconds
    for index, trajectory in enumerate(trajectories):
        shown = boards(trajectory)
        try:
            state = program.init(trajectory["start"])
            got = program.render(state)
            if got is None:
                return _failure(trajectory, index, "unknown start", None)
            if got != trajectory["start"]:
                return _failure(trajectory, index, "start", None, expected=trajectory["start"], got=got)
            if program.won(state):
                return _failure(trajectory, index, "won early", None)
        except Exception as error:   # the model's code: any error is the program's, reported to it
            return _failure(trajectory, index, "error", None, error=f"{type(error).__name__}: {error}")
        for k, step in enumerate(trajectory["steps"]):
            try:
                for action in step["actions"]:
                    state = program.step(state, action)
                if program.unknown(state):
                    # an outcome the program leaves unknown, which is never a win: nothing after it is checked
                    if step["won"]:
                        return _failure(trajectory, index, "not won", k)
                    break
                if step["won"]:
                    if not program.won(state):
                        return _failure(trajectory, index, "not won", k)
                    break
                if step["over"]:
                    if program.has_lost and not program.lost(state):
                        return _failure(trajectory, index, "not lost", k)
                    break
                if program.won(state):
                    return _failure(trajectory, index, "won early", k)
                if program.lost(state):
                    return _failure(trajectory, index, "lost early", k)
                got = program.render(state)
                if got != shown[k + 1]:
                    return _failure(trajectory, index, "board", k, before=shown[k], expected=shown[k + 1], got=got)
            except Exception as error:   # the model's code
                return _failure(trajectory, index, "error", k, error=f"{type(error).__name__}: {error}")
            if time.monotonic() > deadline:
                return _failure(trajectory, index, "slow", k, seconds=seconds)
    return None


def _where(failure: dict) -> str:
    step, n = failure["step"], len(failure["actions"])
    if step is None:
        return f"at the first board of {failure['name']}"
    first = failure["actions_before"] + 1
    played = ", ".join(describe_action(a) for a in failure["actions"])
    if n == 1:
        return f"after action {first} of {failure['name']} (action {played})"
    return f"after actions {first}-{first + n - 1} of {failure['name']} (actions {played})"


def _listed(cells: list) -> str:
    text = render_changes(cells).splitlines()
    if len(text) > LISTED_ROWS:
        text = text[:LISTED_ROWS] + [f"... and {len(text) - LISTED_ROWS} more rows"]
    return "\n".join(text)


def describe(failure: dict) -> str:
    """What a failure of :func:`check` says to the model that wrote the program."""
    kind = failure["kind"]
    if kind == "program":
        return f"Your program cannot be used: {failure['error']}."
    where = _where(failure)
    if kind in ("board", "start"):
        expected, got = failure["expected"], failure["got"]
        wrong = [(x, y) for y in range(SIZE) for x in range(SIZE) if expected[y][x] != got[y][x]]
        what = "the board your init and render give" if kind == "start" else "your program's board"
        return (f"Your program is wrong {where}: {what} differs from the game's in {len(wrong)} cells. Where they "
                f"differ, the game shows\n{_listed([[x, y, expected[y][x]] for x, y in wrong])}\nand your program "
                f"gives\n{_listed([[x, y, got[y][x]] for x, y in wrong])}")
    return {
        "unknown start": f"Your program renders no board {where}: init and render must give every level's first board.",
        "won early": f"Your program says won {where}, where the level had not completed.",
        "not won": f"The level completed {where}, but your program does not say won there.",
        "lost early": f"Your program says lost {where}, where the game went on.",
        "not lost": f"The game ended {where}, but your program does not say lost there.",
        "error": f"Your program raised an error {where}: {failure.get('error')}.",
        "slow": f"Your program was stopped {where}: it took over {failure.get('seconds', SECONDS):g} seconds to run "
                "through the transitions.",
    }[kind]


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


def plan(program: Program | str, history: dict, available: list[int], max_states: int = MAX_STATES,
         seconds: float = SECONDS) -> dict:
    """The shortest winning sequence from the board ``history`` ends at (a trajectory of the level so far), in
    ``program``: ``found``, the ``plan`` (actions), the ``changes`` each of its actions makes to the board before it as
    the program predicts them (none for the winning one, whose board is never compared), ``expanded`` (the states
    expanded), ``start`` (``replay`` when the program followed the level from its first board to the board now,
    ``board`` when it started from the board now because it did not) and, when nothing is found, ``reason``:
    ``exhausted`` (no winning sequence reachable), ``bound`` (``max_states`` expanded) or ``time``."""
    program = compile_program(program)
    now = boards(history)[-1]
    start = "replay"
    try:
        state = program.init(history["start"])
        for step in history["steps"]:
            for action in step["actions"]:
                state = program.step(state, action)
        if program.render(state) != now:   # None too: a replay that lost track of the level
            raise ProgramError("the replay does not reach the board now")
    except Exception:   # the model's code: a program that cannot follow the level starts from the board now
        state, start = program.init(now), "board"
    deadline = time.monotonic() + seconds
    simple = [[a] for a in sorted(set(available)) if a in (1, 2, 3, 4, 5)]
    clicking = 6 in available
    nodes: list[tuple[Any, int, list | None]] = [(state, -1, None)]   # state, parent, action
    seen = {program.key(state)}
    queue = deque([0])
    expanded = 0
    reason = "exhausted"
    while queue:
        index = queue.popleft()
        current = nodes[index][0]
        try:
            actions = simple + ([[6, x, y] for x, y in program.clicks(current)] if clicking else [])
        except Exception:   # the model's clicks: a state it cannot click in is searched with its simple actions
            actions = simple
        for action in actions:
            if expanded >= max_states:
                return {"found": False, "reason": "bound", "expanded": expanded, "start": start}
            if time.monotonic() > deadline:
                return {"found": False, "reason": "time", "expanded": expanded, "start": start}
            expanded += 1
            try:
                after = program.step(current, action)
                if program.unknown(after):
                    continue
                if program.won(after):
                    path = [action]
                    while index > 0:
                        _, index, taken = nodes[index]
                        path.append(taken)
                    path.reverse()
                    return {"found": True, "plan": path, "changes": _predictions(program, state, path),
                            "expanded": expanded, "start": start}
                if program.lost(after):
                    continue
                key = program.key(after)
            except Exception:   # the model's code: an action it cannot step is not searched
                continue
            if key in seen:
                continue
            seen.add(key)
            nodes.append((after, index, action))
            queue.append(len(nodes) - 1)
    return {"found": False, "reason": reason, "expanded": expanded, "start": start}


def _predictions(program: Program, state: Any, path: list) -> list[list]:
    """The changes each action of ``path`` but the last makes to the board before it, as ``program`` predicts."""
    out = []
    shown = program.render(state)
    for action in path[:-1]:
        state = program.step(state, action)
        board = program.render(state)
        out.append(diff(shown, board))
        shown = board
    return out


# ---------------------------------------------------------------------------
# the expert guidance's tools (imported by an agent as carc_games)
# ---------------------------------------------------------------------------


HERE = Path(__file__).resolve().parent
INSTANCE = "instance.json"


def load() -> dict:
    """This instance's game as its adapter wrote it beside this file: ``game`` (its id), ``available`` (the actions it
    accepts), ``board`` (the board now), ``demonstrations`` and ``play`` (this instance's trajectories, as the
    conversation showed them), ``current`` (the level as it stands, from its first board), ``kept`` (the trajectories
    kept for the game in ``<game>.json``) and ``all``, every trajectory of the game, to check a program on."""
    instance = json.loads((HERE / INSTANCE).read_text())
    path = HERE / f"{instance['game']}.json"
    kept = json.loads(path.read_text()) if path.exists() else []
    return {**instance, "kept": kept, "all": kept + instance["demonstrations"] + instance["play"]}


def _program_path(game: str) -> Path:
    return HERE / f"{game}.py"


def check_game(program: Program | str | None = None, trajectories: list[dict] | None = None) -> str | None:
    """None when ``program`` (a path, a source, or by default the game's ``<game>.py``) reproduces every trajectory
    of the game (by default :func:`load`'s ``all``), else what it gets wrong, the first failure."""
    game = load()
    program = program if program is not None else str(_program_path(game["game"]))
    failure = check(program, trajectories if trajectories is not None else game["all"])
    return None if failure is None else describe(failure)


def plan_game(program: Program | str | None = None, max_states: int = MAX_STATES, seconds: float = SECONDS) -> list[dict] | None:
    """The shortest winning sequence from the board now in ``program`` (by default the game's ``<game>.py``), followed
    from the level's first board, as the protocol's actions (``{"id": 4}``, ``{"id": 6, "x": 12, "y": 40}``), to send
    in sequences of up to 20; None when the search finds none."""
    game = load()
    program = program if program is not None else str(_program_path(game["game"]))
    history = game.get("current") or {"start": game["board"], "steps": []}
    found = plan(program, history, game["available"], max_states=max_states, seconds=seconds)
    if not found["found"]:
        return None
    return [{"id": a[0]} if len(a) == 1 else {"id": 6, "x": a[1], "y": a[2]} for a in found["plan"]]


def save() -> int:
    """This instance's demonstrations and play added to the game's ``<game>.json``, each trajectory once; how many
    were added."""
    game = load()
    kept = game["kept"]
    known = {json.dumps([t["start"], t["steps"]]) for t in kept}
    added = 0
    for trajectory in game["demonstrations"] + game["play"]:
        mark = json.dumps([trajectory["start"], trajectory["steps"]])
        if mark not in known:
            kept.append(trajectory)
            known.add(mark)
            added += 1
    (HERE / f"{game['game']}.json").write_text(json.dumps(kept))
    return added


# ---------------------------------------------------------------------------
# the sandboxed process
# ---------------------------------------------------------------------------


def serve(request: dict) -> dict:
    """One request of the program library's sandboxed process: ``check`` (``program``, ``trajectories``) or ``plan``
    (``program``, ``history``, ``available``, ``max_states``), with the deadline ``seconds``. A check that fails comes
    back with what :func:`describe` says of it."""
    seconds = float(request.get("seconds", SECONDS))
    if request["op"] == "check":
        failure = check(request["program"], request["trajectories"], seconds=seconds)
        if failure is None:
            return {"failure": None}
        slim = {k: v for k, v in failure.items() if k not in ("before", "expected", "got")}
        return {"failure": slim, "message": describe(failure)}
    if request["op"] == "plan":
        try:
            return plan(request["program"], request["history"], request["available"],
                        max_states=int(request.get("max_states", MAX_STATES)), seconds=seconds)
        except ProgramError as error:
            return {"found": False, "reason": "program", "error": str(error), "expanded": 0}
    raise ValueError(f"unknown op {request['op']!r}")


if __name__ == "__main__":
    print(json.dumps(serve(json.loads(sys.stdin.read()))))
