"""World models of ARC-AGI-3 games (systems/world_model.py): a program's free check on every transition a game has
shown, the search for a winning plan, the trajectories a learner records, and the tools an agent imports. The game
here is a toy defined in this file, so that the true rules and the program's are both in view; the real engine's games
are played in test_bare_games.py."""

import json
import subprocess
import sys

import pytest

from continual_arc_baselines.systems import world_model
from continual_arc_baselines.systems.world_model import (
    Recorder,
    apply,
    boards,
    check,
    describe,
    diff,
    patches,
    plan,
    serve,
)

BG, TIMER, PLAYER, GOAL, WALL, SWITCH = 0, 1, 3, 4, 5, 6
MOVES = {1: (0, -2), 2: (0, 2), 3: (-2, 0), 4: (2, 0)}


def square(x, y):
    return {(x + i, y + j) for i in (0, 1) for j in (0, 1)}


class Toy:
    """The toy game: a 2 x 2 player moves two cells a move; walls and the board's edge block it; entering the goal wins,
    with a switch only once the switch has been entered, which the board never shows (the switch is drawn again when
    the player leaves it); every action shortens the timer along the bottom row by one cell."""

    def __init__(self, player, goal, walls=(), switch=None):
        self.walls = set().union(*(square(*w) for w in walls)) if walls else set()
        self.goal, self.switch = square(*goal), square(*switch) if switch else set()
        self.pos, self.timer, self.open = player, 64, switch is None

    def board(self):
        board = [[BG] * 64 for _ in range(64)]
        board[63] = [TIMER] * self.timer + [BG] * (64 - self.timer)
        for cells, color in ((self.walls, WALL), (self.goal, GOAL), (self.switch, SWITCH), (square(*self.pos), PLAYER)):
            for x, y in cells:
                board[y][x] = color
        return board

    def step(self, action):
        dx, dy = MOVES[action[0]]
        to = (self.pos[0] + dx, self.pos[1] + dy)
        won = False
        if all(0 <= x < 64 and 0 <= y < 63 and (x, y) not in self.walls for x, y in square(*to)):
            if square(*to) & self.goal:
                won = self.open
                self.pos = to if won else self.pos
            else:
                self.pos = to
                self.open = self.open or bool(square(*to) & self.switch)
        self.timer -= 1
        return won


def play(toy, actions, name="t"):
    """A trajectory of ``toy``: one step per action, up to the one that wins."""
    start, steps = toy.board(), []
    for action in actions:
        before = toy.board()
        won = toy.step(action)
        steps.append({"actions": [action], "changes": [] if won else diff(before, toy.board()), "won": won,
                      "over": False})
        if won:
            break
    return {"name": name, "start": start, "steps": steps}


def level_a():
    return Toy(player=(10, 10), goal=(30, 10), walls=((20, 8), (20, 10), (20, 12)))


def level_b():
    return Toy(player=(40, 30), goal=(10, 30), walls=((24, 40),))


AROUND = [[4]] * 4 + [[1]] * 2 + [[4]] * 2 + [[2]] * 2 + [[4]] * 4   # level a: around the wall, into the goal
STRAIGHT = [[3]] * 15                                                # level b


RIGHT = """
def step(board, action):
    cells = [(x, y) for y in range(63) for x in range(64) if board[y][x] == 3]
    x, y = min(cells, key=lambda p: (p[1], p[0]))
    dx, dy = {1: (0, -2), 2: (0, 2), 3: (-2, 0), 4: (2, 0)}[action[0]]
    to = [(x + dx + i, y + dy + j) for i in (0, 1) for j in (0, 1)]
    if all(0 <= a < 64 and 0 <= b < 63 and board[b][a] != 5 for a, b in to):
        for i in (0, 1):
            for j in (0, 1):
                board[y + j][x + i] = 0
        for a, b in to:
            board[b][a] = 3
    left = sum(1 for c in board[63] if c == 1)
    board[63][left - 1] = 0
    return board

def won(board):
    return not any(4 in row for row in board)
"""

NO_TIMER = RIGHT.replace("    left = sum(1 for c in board[63] if c == 1)\n    board[63][left - 1] = 0\n", "")

def demos():
    return [play(level_a(), AROUND, "demonstration 1"), play(level_b(), STRAIGHT, "demonstration 2")]


def test_a_right_program_reproduces_every_transition_of_every_level():
    trajectories = demos()
    assert [len(t["steps"]) for t in trajectories] == [14, 15] and all(t["steps"][-1]["won"] for t in trajectories)
    assert check(RIGHT, trajectories) is None


def test_a_program_that_forgets_the_timer_is_shown_the_cells_it_got_wrong():
    failure = check(NO_TIMER, demos())
    assert failure["kind"] == "board" and failure["trajectory"] == 0 and failure["step"] == 0
    text = describe(failure)
    assert text.startswith("Your program is wrong after action 1 of demonstration 1 (action 4)")
    assert "differs from the game's in 1 cells" in text
    assert "the game shows\ny63: 63=0\nand your program gives\ny63: 63=1" in text


def test_won_is_read_exactly_where_the_level_completed():
    never = RIGHT.replace("return not any(4 in row for row in board)", "return False")
    failure = check(never, demos())
    assert failure["kind"] == "not won" and failure["step"] == 13
    assert describe(failure).startswith("The level completed after action 14 of demonstration 1")
    always = RIGHT.replace("return not any(4 in row for row in board)", "return True")
    assert check(always, demos())["kind"] == "won early"


def test_a_program_that_cannot_run_or_raises_is_told_so():
    assert check("def step(:", demos())["kind"] == "program"
    assert describe(check("def step(b, a):\n    return b\n", demos())) == (
        "Your program cannot be used: the program must define step and won.")
    failure = check(RIGHT.replace("[action[0]]", "[action[1]]"), demos())
    assert failure["kind"] == "error" and "IndexError" in describe(failure)


def test_the_search_finds_the_shortest_win_and_predicts_every_board_before_it():
    found = plan(RIGHT, {"start": level_a().board(), "steps": []}, [1, 2, 3, 4])
    assert found["found"] and found["start"] == "replay" and len(found["plan"]) == 14
    toy = level_a()
    for k, action in enumerate(found["plan"]):
        before = toy.board()
        won = toy.step(action)
        assert won == (k == 13)
        if not won:
            assert diff(before, toy.board()) == found["changes"][k]


def test_the_search_starts_where_the_level_stands_and_names_why_it_found_nothing():
    history = play(level_a(), [[4], [4], [4]])
    found = plan(RIGHT, history, [1, 2, 3, 4])
    assert found["found"] and found["start"] == "replay" and len(found["plan"]) == 11   # three of the 14 made
    walls = [(x, y) for x in (6, 14) for y in (6, 8, 10, 12, 14)] + [(x, y) for x in (8, 10, 12) for y in (6, 14)]
    boxed = Toy(player=(10, 10), goal=(30, 10), walls=walls)
    found = plan(RIGHT, {"start": boxed.board(), "steps": []}, [1, 2, 3, 4])
    assert not found["found"] and found["reason"] == "exhausted"
    found = plan(RIGHT, {"start": level_a().board(), "steps": []}, [1, 2, 3, 4], max_states=20)
    assert not found["found"] and found["reason"] == "bound" and found["expanded"] == 20


HIDDEN = """
def init(board):
    switch = [(x, y) for y in range(64) for x in range(64) if board[y][x] == 6]
    return {"board": board, "switch": switch, "open": not switch}

def render(state):
    return state["board"]

def step(state, action):
    board = state["board"]
    cells = [(x, y) for y in range(63) for x in range(64) if board[y][x] == 3]
    x, y = min(cells, key=lambda p: (p[1], p[0]))
    dx, dy = {1: (0, -2), 2: (0, 2), 3: (-2, 0), 4: (2, 0)}[action[0]]
    to = [(x + dx + i, y + dy + j) for i in (0, 1) for j in (0, 1)]
    if all(0 <= a < 64 and 0 <= b < 63 and board[b][a] != 5 for a, b in to):
        if state["open"] or not any(board[b][a] == 4 for a, b in to):
            if any((a, b) in state["switch"] for a, b in to):
                state["open"] = True
            for i in (0, 1):
                for j in (0, 1):
                    board[y + j][x + i] = 6 if (x + i, y + j) in state["switch"] else 0
            for a, b in to:
                board[b][a] = 3
    left = sum(1 for c in board[63] if c == 1)
    board[63][left - 1] = 0
    return state

def won(state):
    return not any(4 in row for row in state["board"])
"""


def switch_level():
    return Toy(player=(10, 10), goal=(14, 10), switch=(12, 20))


def test_a_program_carries_what_the_board_hides_from_the_level_s_first_board():
    # into the shut goal first, then down onto the switch, which is drawn again once the player leaves it, back up,
    # and into the goal: the board after the switch is the board before it, but for the timer
    trajectory = play(switch_level(), [[4], [4]] + [[2]] * 5 + [[1]] * 5 + [[4]], "your play")
    assert trajectory["steps"][-1]["won"] and len(trajectory["steps"]) == 13
    assert check(HIDDEN, [trajectory]) is None
    assert check(HIDDEN, demos()) is None                     # a level with no switch is open from the start
    shut = check(RIGHT, [trajectory])                         # the board alone cannot tell the goal is shut
    assert shut["kind"] == "board" and shut["step"] == 1
    found = plan(HIDDEN, {"start": switch_level().board(), "steps": []}, [1, 2, 3, 4])
    assert found["found"] and len(found["plan"]) == 12
    toy = switch_level()
    assert [toy.step(a) for a in found["plan"]][-1]
    later = plan(HIDDEN, play(switch_level(), [[4]]), [1, 2, 3, 4])   # followed from the level's first board
    assert later["start"] == "replay" and len(later["plan"]) == 11


def test_the_recorder_keeps_a_step_per_board_seen_and_ends_a_trajectory_at_a_win_or_a_reset():
    per_action, per_message, toy = Recorder(), Recorder(), level_a()
    per_action.board(toy.board())
    per_message.board(toy.board())
    for k, action in enumerate(AROUND):
        won = toy.step(action)
        per_action.act(action, won=won)
        per_message.act(action, won=won)
        if not won:
            per_action.board(toy.board())
            if k % 3 == 2:
                per_message.board(toy.board())   # a conversation shows the board only after each message
    (one,), (many,) = per_action.trajectories, per_message.trajectories
    assert [len(s["actions"]) for s in one["steps"]] == [1] * 14
    assert [len(s["actions"]) for s in many["steps"]] == [3] * 4 + [2]
    assert one["steps"][-1]["won"] and many["steps"][-1]["won"]
    assert per_action.current is None   # the level completed
    assert check(RIGHT, [one, many]) is None
    assert describe(check(NO_TIMER, [many])).startswith(
        "Your program is wrong after actions 1-3 of your play of this level (actions 4, 4, 4)")
    recorder, toy = Recorder(), level_a()
    recorder.board(toy.board())
    toy.step([4])
    recorder.act([4])
    recorder.board(toy.board())
    recorder.reset()
    toy = level_a()
    recorder.board(toy.board())
    toy.step([3])
    recorder.act([3])
    assert [t["name"] for t in recorder.trajectories] == ["your play of this level"]   # an action with no board yet
    recorder.board(toy.board())
    assert [t["name"] for t in recorder.trajectories] == ["your play of this level",
                                                          "your play of this level after reset 1"]
    assert recorder.current["start"] == level_a().board()


def test_a_game_over_is_read_by_lost_and_ends_the_trajectory():
    recorder, toy = Recorder(), level_a()
    start = toy.board()
    recorder.board(start)
    toy.step([4])
    recorder.act([4], over=True)
    recorder.board(toy.board())
    recorder.act([4])          # ignored by the game: nothing more until a reset
    recorder.board(toy.board())
    (trajectory,) = recorder.trajectories
    assert trajectory["steps"] == [{"actions": [[4]], "changes": diff(start, toy.board()), "won": False, "over": True}]
    assert recorder.current is None
    assert check(RIGHT, [trajectory]) is None   # a program without lost is not asked
    assert check(RIGHT + "\ndef lost(board):\n    return False\n", [trajectory])["kind"] == "not lost"


def test_clicks_default_to_one_cell_of_each_one_colored_patch():
    board = [[0] * 64 for _ in range(64)]
    for x in range(10, 13):
        for y in range(4, 7):
            board[y][x] = 2
    board[40][50] = 7
    cells = patches(board)
    assert len(cells) == 3 and [11, 5] in cells and [50, 40] in cells
    clicker = ("def step(board, action):\n    if action[0] == 6 and board[action[2]][action[1]] == 2:\n"
               "        board[action[2]][action[1]] = 9\n    return board\n\n"
               "def won(board):\n    return any(9 in row for row in board)\n")
    found = plan(clicker, {"start": board, "steps": []}, [6])
    assert found["found"] and found["plan"] == [[6, 11, 5]]
    named = clicker + "\ndef clicks(board):\n    return [[12, 6]]\n"
    assert plan(named, {"start": board, "steps": []}, [6])["plan"] == [[6, 12, 6]]


def test_the_tools_an_agent_imports_load_check_plan_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(world_model, "HERE", tmp_path)
    shown = play(level_a(), [[4], [4]], "your play of this level")
    (tmp_path / "instance.json").write_text(json.dumps({
        "game": "task-1", "available": [1, 2, 3, 4], "board": boards(shown)[-1],
        "demonstrations": demos()[1:], "play": [shown], "current": shown}))
    (tmp_path / "task-1.py").write_text(RIGHT)
    loaded = world_model.load()
    assert loaded["kept"] == [] and len(loaded["all"]) == 2
    assert world_model.check_game() is None
    assert world_model.check_game(NO_TIMER).startswith("Your program is wrong after action 1 of demonstration 2")
    moves = world_model.plan_game()
    toy = level_a()
    assert len(moves) == 12 and [toy.step([m["id"]]) for m in [{"id": 4}] * 2 + moves][-1]
    assert world_model.save() == 2 and world_model.save() == 0
    assert len(json.loads((tmp_path / "task-1.json").read_text())) == 2
    assert len(world_model.load()["all"]) == 4   # the kept two, and this instance's two again


def test_the_module_runs_alone_as_the_sandbox_s_script():
    request = {"op": "plan", "program": RIGHT, "history": {"start": level_b().board(), "steps": []},
               "available": [1, 2, 3, 4]}
    out = subprocess.run([sys.executable, "-I", "-S", world_model.__file__], input=json.dumps(request),
                         capture_output=True, text=True, check=True)
    answer = json.loads(out.stdout)
    assert answer["found"] and len(answer["plan"]) == 15
    assert serve({"op": "check", "program": NO_TIMER, "trajectories": demos()})["message"].startswith(
        "Your program is wrong after action 1")
    assert serve({"op": "check", "program": RIGHT, "trajectories": demos()}) == {"failure": None}


@pytest.mark.parametrize("changes", [[[1, 2, 3]], []])
def test_apply_and_diff_are_inverse(changes):
    board = [[0] * 64 for _ in range(64)]
    assert diff(board, apply(board, changes)) == [list(c) for c in changes]
