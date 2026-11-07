"""The expert guidance shared by every guided harness and the training oracle.

``EXPERT_GUIDANCE`` is the complete policy on the grid tracks, and ``GAME_GUIDANCE`` on an ARC-AGI-3 game: a playing
procedure (a demonstration first on a new game, each action it did not show tried once, the controls and the win
condition written into notes and read on a return) and the program library's procedure with a game's rules as the
program, checked and searched with the tools of :mod:`continual_arc_baselines.systems.world_model`, which the adapter
writes into the library directory as ``carc_games.py`` beside the instance's boards. Harness-specific paths are supplied
as runtime context, not substituted into or appended as another policy.
"""

from hashlib import sha256

PRIME_GRID_LOADER = r'''import os, re
from pathlib import Path

def load():
    session = Path(os.path.normpath(os.environ["RLM_SESSION_DIR"]))
    log = session.parent.parent / "sessions" / (session.name + ".jsonl")
    text = log.read_text().replace("\\n", "\n")
    grids = {}
    pattern = r"(Demo pair \d+ (?:input|output)|Test input) \((\d+)x(\d+)\):\n"
    for match in re.finditer(pattern, text):
        name, height, width = match[1], int(match[2]), int(match[3])
        rows = [row.split() for row in text[match.end():].split("\n")[:height]]
        if name not in grids and len(rows) == height and all(
            len(row) == width and all(cell in "0123456789" and len(cell) == 1 for cell in row)
            for row in rows
        ):
            grids[name] = [[int(cell) for cell in row] for row in rows]
    return grids

G = load()
print(sorted(G))'''

EXPERT_GUIDANCE = """\
Expert guidance

Actions. Your action is read only from the plain text of your reply, never from Python output or a tool call:
printing JSON or calling a tool named submit, request_demos, finish or answer does not submit an action. Keep each
turn to a few tool calls, then end your reply with exactly one action JSON on the last line:
{"action": "request_demos"}, {"action": "submit", "grid": [[0, 1], [2, 3]]}, or {"action": "finish"}.
Finish only after the instance is over. Replies without a valid action waste the turn; repeated invalid replies
end the instance charged in full. A submission has happened only when you receive its verdict.
When told "Decide now" or "could not find a valid action", make no further tool call: reply immediately with an
action. Submit only a program-produced grid already printed in this conversation, copying it exactly; otherwise
request a demonstration. If the instance is over, finish.

Library. Use the programs folder in the harness's workspace, or the library directory supplied with this instance.
Keep two files per task, named for its task id: <task id>.py defining transform(grid), which takes and returns a grid
as lists of lists of integers; and <task id>.json containing every received demonstration pair as a list of
{"input": grid, "output": grid} objects. Reading kept pairs is free, and they count as demonstrations you have.
At the start of an instance, look for the task's memory among those shown to you, then for its files if no memory
is shown. Load its kept pairs and program, if any, before requesting new demonstrations.

Grids. Never retype grids. Load the grids exactly as provided in the current conversation or its log, and load
cached pairs from the task's JSON file. In Prime Agent, when RLM_SESSION_DIR is available, the following loader reads
only this instance's log. Rerun it after each new demonstration. In other harnesses, use their tools to read the
current conversation's grids instead.
```python
""" + PRIME_GRID_LOADER + """
```

Solve. If you have no demonstration pair, cached or new, request one before submitting: the test input alone does
not establish the rule. If a kept transform exists, run it on every pair you have. If it reproduces them all, run it
on the test input and submit that output before requesting more demonstrations. Otherwise, or after a wrong answer, write or
repair transform and check it against every pair, cached and new; request another demonstration when needed.
These checks are free. As soon as every pair matches, print json.dumps(transform(test_input)) and submit that grid
instead of requesting more demonstrations. Submit only program-produced grids, and never resubmit a grid already
judged wrong in this instance.

Store. After the verdict, in one Python call if possible, add every received pair to the task's JSON file and save
the program that reproduces every pair to its Python file, replacing the older version. If no program reproduces
all pairs, leave the old program file unchanged. Keep one memory for the task, named for its task id, that lasts
between instances and records where the two files are and whether the program reproduces every kept pair. Then
finish.
"""

EXPERT_GUIDANCE_SHA256 = sha256(EXPERT_GUIDANCE.encode("utf-8")).hexdigest()

GAME_GUIDANCE = """\
Expert guidance for games

Actions. Your action is read only from the plain text of your reply, never from Python output or a tool call:
printing JSON or calling a tool named act, reset, request_demo or finish does not take an action. Keep each turn to a
few tool calls, then end your reply with exactly one action JSON on the last line: {"action": "act", "id": 1},
{"action": "act", "id": 6, "x": 12, "y": 40}, {"actions": [{"id": 1}, {"id": 6, "x": 3, "y": 4}]} for a sequence
of up to 20, {"action": "reset"}, {"action": "request_demo"}, or {"action": "finish"}. Finish only after the instance
is over. Replies without a valid action waste the turn; repeated invalid replies end the instance charged in full. When
told "Decide now" or "could not find a valid action", make no further tool call: reply immediately with an action. If
the instance is over, finish.

Library. Use the library directory supplied with this instance. Keep three files per game, named for its game id:
<game id>.md, notes on the game: what each action does, the objects by color and what they do, how the level is won
and lost, and any counter; <game id>.json, every trajectory of the game you have seen, as save() writes it; and
<game id>.py, the game's rules as a program (Model, below). At the start of an instance, look for the game's memory
among those shown to you, then for its files if no memory is shown, and read its notes and program before requesting a
demonstration.

Tools. Never retype a board. In Python, with LIBRARY the library directory: import sys; sys.path.insert(0, LIBRARY);
from carc_games import load, check_game, plan_game, save. load() gives this instance's game id, its available actions,
the board now, and every trajectory of the game: this instance's demonstrations and play as they were shown to you, and
those kept in <game id>.json. A trajectory is a level's first board and then each step: the actions played, the cells
they changed, and whether the level completed or the game ended there.

A new game. With no notes, request one demonstration first and read it: which action moved what, what changed on the
board, and what the last board shows. Then try once each available action the demonstration did not show, and write
the controls and the win condition into the notes before playing on.

Model. Write the game's rules in <game id>.py: step(board, action), the board after an action, and won(board), whether
a board shows the level completed, where a board is 64 rows of 64 colors 0-15, top row first, and an action is [id], or
[6, x, y] for a click. To carry what a board does not show, also define init(board), your state at a level's first
board, and render(state), the board a state shows; step and won then take that state. Where the program cannot tell
the board, such as after an action whose outcome it does not model, render may return None (or step, if the states are
boards): that board is not compared, nothing after it in that level is checked and the search never goes through it,
but it is never a win, so the check fails where the level completed. You may define lost(state), clicks(state), the
[x, y] clicks worth trying, and key(state), a value equal for states that are the same.
check_game() runs the program from every trajectory's first board through every action, at no cost, and returns None
when it gives every board and says won exactly where the level completed, else what it got wrong. Once it passes,
plan_game() searches it for the shortest sequence of actions that wins from the board now: play them, in sequences of
up to 20. If the game shows a board the program did not predict, repair the program, check it and plan again; if no
program passes, play from the notes.

Return. If the game's program passes check_game(), play the plan plan_game() gives from the first board before
requesting a demonstration; otherwise play from the notes, requesting a demonstration only if they do not say how the
level is won.

Store. After the verdict, in one Python call if possible: save() this instance's trajectories to <game id>.json,
update the notes with what you learned, and save the program to <game id>.py only if it passes check_game(), replacing
the older version; if none passes, leave the old program file unchanged. Keep one memory for the game, named for its
game id, that lasts between instances and records where its three files are and whether its program passes. Then
finish.
"""

GAME_GUIDANCE_SHA256 = sha256(GAME_GUIDANCE.encode("utf-8")).hexdigest()
