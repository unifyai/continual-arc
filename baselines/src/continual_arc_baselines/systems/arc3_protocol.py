"""The text protocol between the benchmark and a system for ARC-AGI-3 games, the third track.

Same shape as the grid protocol (:mod:`continual_arc_baselines.systems.protocol`): the rules once per conversation,
then one message per turn with the feedback since the last message, the board, and a request for one JSON action. A
turn may send a sequence of up to :data:`MAX_BATCH` actions, applied in order without another message (the adapter
queues them, :class:`continual_arc_baselines.systems.base.SystemLearner`), and stopped early when the level completes,
the game is over or an action is refused. There is no give-up action: an instance ends when its level completes or its
action cap is reached, so quitting is never cheaper than playing on (20 Sep 2026: nearly every frontier model's
instance ended in a give-up when the old track offered one).

The board is 64 by 64 cells of colors 0 to 15, one hexadecimal digit a cell, a row a line, 4,160 characters: shown
after every action it would make a level of a few hundred actions cost a model millions of tokens. So it is shown in
full only at the start of a conversation, after a reset, and when listing its changes would take more text than the
board; otherwise only the cells that changed since the board last shown (:func:`render_changes`), which an instance's
:class:`GameView` keeps. A demonstration (the solver's play of another fresh level of the game) is shown the same way:
its first board, each action with the cells it changed, and its last board.
"""

from __future__ import annotations

from typing import Any

from continual_arc.arc3.env import (
    ActFeedback,
    Arc3LearnerObservation,
    DemonstrationFeedback,
    ResetFeedback,
    ResetLevel,
)
from continual_arc.arc3.instances import COMPLEX_ACTION, Act, Demonstration, Frame, changes
from continual_arc.env import GiveUpFeedback, RequestDemos

from continual_arc_baselines.systems.protocol import (
    DEMO_ALIASES,
    FINISH_ALIASES,
    GIVE_UP_ALIASES,
    OVER_REPROMPT,
    Finish,
    ParseError,
    _clean,
    _loads,
    _normalise_name,
    objects_from_end,
)

MAX_BATCH = 20

_RULES = """\
You are being evaluated on Continual-ARC-3, a lifelong stream of ARC-AGI-3 games.
An instance is one level of a game: a fresh layout of the game's first level. You see
the board, 64x64 cells of colors 0-15 (one hex digit per cell, row by row, top row
first), and the actions the game accepts, and you must discover what the game wants by
acting. Games recur across the stream with fresh levels.{conversation}

On every message choose what to do and put it as a JSON object on the last line of
your reply:
  {{"action": "act", "id": 1}}                      one simple action (ids 1-5 and 7)
  {{"action": "act", "id": 6, "x": 12, "y": 40}}     one click at column x, row y (0-63)
  {{"action": "act", "ids": [1, 1, 2, 5]}}           a sequence of up to {max_batch} simple actions, applied
                                                  in order; it stops early if the level completes,
                                                  the game is over or an action is refused
  {{"actions": [{{"id": 6, "x": 3, "y": 4}}, {{"id": 1}}]}}   a sequence with clicks in it
  {{"action": "reset"}}                             reload the level, which a game over needs
  {{"action": "request_demo"}}                      a demonstration: the benchmark's solver playing
                                                  another fresh level of this game to its end{finish_line}
Actions 1-4 usually move (up, down, left, right), 5 usually selects or interacts, 6
clicks and 7 undoes where a game has it; what each does is the game's to define. You
see the board after each message, a sequence's last action included. After the first
board you are shown only the cells changed since the board last shown, row by row:
"y12: 20-24=0 30=c" means that in row 12 columns 20 to 24 are now color 0 and column 30
color c. The whole board is shown again after a reset, or when most of it changed.

Scoring: an instance's baseline is the actions a person needs for its level. Solved, an
instance costs the actions and resets you spent over its baseline, at most {cap:g}, plus
{demo_cost:g} for each demonstration (at most {demo_cap} an instance). Every action and reset
counts toward the instance's action cap, {cap:g} times its baseline, and an instance that
reaches the cap unsolved costs the whole budget, {max_cost:g}. The instance is over when its level
completes or the cap is reached. {ending}
"""

_KEEP = " Anything you learn about a game is worth keeping."
_CONVERSATION = " Each instance is a separate conversation: the next one starts without this one's\nmessages."
_FINISH_LINE = '\n  {"action": "finish"}                            end the instance, once it is over'
_ENDING = ("There is no way to abandon an instance and nothing is shared between instances: each\nis scored on its own, "
           "so keep playing until the level completes or the cap ends it.")
_ENDING_FINISH = ("There is no way to abandon an instance, and finish is refused until the instance\nis over. Each "
                  "instance is scored on its own, and lower cost is better.")


def rules(obs: Arc3LearnerObservation, finish: bool) -> str:
    """The rules a system reads at the start of a conversation (the finish protocol's facts only, with ``finish``),
    with the budget's figures from the observation."""
    return _RULES.format(conversation=_CONVERSATION if finish else _KEEP, max_batch=MAX_BATCH,
                         finish_line=_FINISH_LINE if finish else "",
                         cap=obs.max_cost - obs.demo_cost * obs.demo_cap, demo_cost=obs.demo_cost,
                         demo_cap=obs.demo_cap, max_cost=obs.max_cost, ending=_ENDING_FINISH if finish else _ENDING)


# ---------------------------------------------------------------------------
# boards
# ---------------------------------------------------------------------------


def render_frame(frame: Frame) -> str:
    return "\n".join("".join(format(c, "x") for c in row) for row in frame)


def render_changes(cells: tuple[tuple[int, int, int], ...]) -> str:
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


def render_board(frame: Frame, shown: Frame | None, what: str = "Board") -> str:
    """``frame`` in full, or the cells it changed since ``shown`` when that is the shorter text."""
    full = f"{what}:\n{render_frame(frame)}"
    if shown is None:
        return full
    diff = changes(shown, frame)
    if not diff:
        return f"{what}: unchanged since the board last shown."
    text = render_changes(diff)
    if len(text) >= len(full):
        return full
    return f"{what}: {len(diff)} cells changed since the board last shown:\n{text}"


def describe(action: Act) -> str:
    return f"{action.action} at ({action.x}, {action.y})" if action.action == COMPLEX_ACTION else str(action.action)


def render_demonstration(demo: Demonstration, k: int) -> str:
    """Demonstration ``k``: its first board, each action with the cells it changed, and its last board."""
    frames = demo.frames()
    lines = [f"Demonstration {k}: the benchmark's solver playing another fresh level of this game to its end, in "
             f"{len(demo.steps)} actions.", render_board(frames[0], None, "Its first board")]
    for i, step in enumerate(demo.steps, 1):
        board = render_board(frames[i], frames[i - 1], f"{i}. action {describe(step.action)}")
        lines.append(board.replace(" since the board last shown", ""))
    lines.append(render_board(frames[-1], None, "Its last board (the level completed)"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def _state(state: str) -> str:
    return {"GAME_OVER": " The game is over: reset to continue.", "WIN": " The level is completed."}.get(state, "")


def render_feedback(action: Any, feedback: Any) -> str:
    """One line for the feedback on one action: the board itself comes with the next message."""
    cap = " The action cap is reached; the instance is over." if getattr(feedback, "failed", False) else ""
    if isinstance(feedback, ActFeedback):
        name = f"Action {describe(action)}"
        if feedback.level_completed:
            return f"{name}: the level is COMPLETED. Instance solved."
        if feedback.ignored:
            return f"{name} was ignored, and charged: the game is over; reset to continue.{cap}"
        if not feedback.valid:
            return f"{name} is not one this game accepts here, and was charged.{cap}"
        return f"{name} applied.{_state(feedback.state)}{cap}"
    if isinstance(feedback, ResetFeedback):
        return f"Level reloaded (reset {feedback.resets_used}).{cap}"
    if isinstance(feedback, DemonstrationFeedback):
        if feedback.refused:
            return "Demonstration request REFUSED: the demo cap for this instance is reached."
        return render_demonstration(feedback.demonstration, feedback.demo_requests_used)
    if isinstance(feedback, GiveUpFeedback):
        return f"The instance ended unsolved. Charged {feedback.charged:g}."
    return f"Feedback: {feedback!r}"


class GameView:
    """One instance's conversation about a game: the protocol a system adapter calls for an observation of a game,
    in place of the grid protocol module (``render_observation``, ``render_feedback``, ``parse_action`` and the
    re-prompts), keeping the board it last showed so that the next message shows what changed."""

    REPROMPT = (
        "I could not find a valid action in your reply ({reason}). Reply again with exactly one of\n"
        '{{"action": "act", "id": N}} (with "x" and "y" for id 6), {{"action": "act", "ids": [...]}}, '
        '{{"action": "reset"}} or {{"action": "request_demo"}}\nas a JSON object on the last line.'
    )
    REPROMPT_FINISH = (
        "I could not find a valid action in your reply ({reason}). Reply again with exactly one of\n"
        '{{"action": "act", "id": N}} (with "x" and "y" for id 6), {{"action": "act", "ids": [...]}}, '
        '{{"action": "reset"}}, {{"action": "request_demo"}} or {{"action": "finish"}}\nas a JSON object on the last '
        "line."
    )
    OVER_REPROMPT = OVER_REPROMPT

    def __init__(self) -> None:
        self.shown: Frame | None = None
        self.resets = 0

    def render_observation(self, obs: Arc3LearnerObservation, *, pending: list[str], first_in_session: bool,
                           finish: bool = False, rules_note: str = "", preface: str = "", **_: Any) -> str:
        """The message for ``obs``. The grid protocol's ``demo_cost`` and ``max_cost`` are taken, and ignored: a
        game's observation carries its figures."""
        lines: list[str] = []
        if first_in_session:
            lines.append(rules(obs, finish))
            if finish and rules_note:
                lines.append(rules_note)
            lines.append("")
        if pending:
            lines.append("Feedback since your last action:")
            lines.extend(pending)
            lines.append("")
        if preface:
            lines.append(preface)
            lines.append("")
        lines.append(f"{'New instance' if obs.is_fresh else 'Same instance'}. Game id: {obs.task_id}.")
        lines.append(f"Available actions: {', '.join(str(a) for a in obs.available_actions)}. "
                     f"Actions used: {obs.actions_used + obs.resets_used}/{obs.action_cap} (baseline "
                     f"{obs.baseline_actions:.1f}). Demonstrations: {obs.demo_requests_used}/{obs.demo_cap}."
                     f"{_state(obs.state)}")
        board = obs.frames[-1]
        full = first_in_session or self.shown is None or obs.resets_used != self.resets
        if len(obs.frames) > 1:
            lines.append(f"(The last action played {len(obs.frames)} frames; the board is the last.)")
        lines.append(render_board(board, None if full else self.shown))
        self.shown, self.resets = board, obs.resets_used
        lines.append("Reply with exactly one action JSON on the last line.")
        return "\n".join(lines)

    render_feedback = staticmethod(render_feedback)

    @staticmethod
    def parse_action(text: str, *, finish: bool = False) -> Any:
        return parse_action(text, finish=finish)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


ACT_ALIASES = frozenset({"act", "action", "step", "press", "move", "click", "play"})
RESET_ALIASES = frozenset({"reset", "reset_level", "restart", "reload"})


def _one(spec: Any) -> Act:
    """An :class:`Act` from an id (an int, ``"3"`` or ``"ACTION3"``) or an object with an id and optional x and y."""
    if isinstance(spec, bool):
        raise ParseError(f"not an action: {spec!r}")
    if isinstance(spec, int):
        return Act(spec)
    if isinstance(spec, str):
        name = spec.strip().lower()
        if name.startswith("action") and name[6:].isdigit():
            return Act(int(name[6:]))
        if name.isdigit():
            return Act(int(name))
        raise ParseError(f"not an action: {spec!r}")
    if isinstance(spec, dict):
        ident = spec.get("id", spec.get("action_id", spec.get("action")))
        if isinstance(ident, str) and ident.strip().lower().startswith("action") and ident.strip()[6:].isdigit():
            ident = ident.strip()[6:]
        try:
            ident = int(ident)
        except (TypeError, ValueError):
            raise ParseError(f"an action needs an integer id, got {ident!r}") from None
        x, y = spec.get("x"), spec.get("y")
        try:
            return Act(ident, None if x is None else int(x), None if y is None else int(y))
        except (TypeError, ValueError):
            raise ParseError(f"a click needs integer x and y, got {x!r} and {y!r}") from None
    raise ParseError(f"not an action: {spec!r}")


def _sequence(specs: list) -> list[Act]:
    seq = [_one(s) for s in specs][:MAX_BATCH]
    if not seq:
        raise ParseError("empty action sequence")
    return seq


def parse_action(text: str, *, finish: bool = False) -> Any:
    """The action in a reply: an :class:`Act`, a list of them for a sequence (at most :data:`MAX_BATCH`), a
    :class:`ResetLevel`, a :class:`RequestDemos` or, with ``finish``, :class:`Finish`. Every brace-matched JSON object
    is read, the last first, leniently, as the grid protocol reads them; a give-up in any spelling is refused, since
    there is none."""
    text = _clean(text)
    if not text:
        raise ParseError("empty reply")
    last = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
    if finish and "{" not in last and _normalise_name(last) in FINISH_ALIASES:
        return Finish()
    unknown: list[str] = []
    for candidate in objects_from_end(text):
        obj = _loads(candidate)
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("actions"), list):
            return _sequence(obj["actions"])
        if "action" not in obj:
            continue
        name = _normalise_name(str(obj["action"]))
        if finish and name in FINISH_ALIASES:
            return Finish()
        if name.startswith("action") and name[6:].isdigit():   # {"action": "ACTION3"}
            return _one({**obj, "id": int(name[6:])})
        if name.isdigit():
            return _one({**obj, "id": int(name)})
        if name in ACT_ALIASES:
            if isinstance(obj.get("ids"), list):
                return _sequence(obj["ids"])
            return _one(obj)
        if name in RESET_ALIASES:
            return ResetLevel()
        if name in DEMO_ALIASES:
            return RequestDemos()
        if name in GIVE_UP_ALIASES:
            raise ParseError("there is no give_up in this game: the instance ends when the level completes or the "
                             "action cap is reached, so keep acting")
        unknown.append(str(obj["action"]))
    if unknown:
        raise ParseError(f"unknown action {unknown[0]!r}")
    raise ParseError("no action JSON object found in the reply")
