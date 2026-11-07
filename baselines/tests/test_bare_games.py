"""The program library on ARC-AGI-3 games (systems/bare.py, BareProgramLibraryLearner on a game): a world model checked
at no cost on every transition the game has shown, kept, and searched for a plan that is played with no model call, in
the real engine, on the tests' sp80 (tests/arc3_sp80.py). The model is scripted (FakeUpstream); its program is the
one below, written by hand for sp80, which reproduces every board the game shows, the timer along its top row
included."""

import json

import pytest
from continual_arc.arc3.env import Arc3Env
from continual_arc.arc3.instances import LevelSource
from continual_arc.runner import run
from continual_arc.schedule import Schedule, Slot

from continual_arc_baselines.proxy import LLMProxy
from continual_arc_baselines.systems.bare import (
    GAME_PROGRAM_ASK,
    GAME_PROGRAM_NOTE,
    BareProgramLibraryLearner,
)
from continual_arc_baselines.systems.world_model import check, plan, trajectory_of_demonstration

from .fake_upstream import FakeUpstream
from .test_arc3_protocol import GENERATORS, needs_arc3

SP80 = '''
def init(board):
    return {"board": board, "won": False, "lost": False}


def render(state):
    return state["board"]


def bar(board):
    cells = [(x, y) for y in range(1, 52) for x in range(64) if board[y][x] == 9]
    xs, ys = [x for x, _ in cells], [y for _, y in cells]
    return min(xs), min(ys), max(xs), max(ys)


def taken(board):
    left = sum(1 for c in board[0] if c == 14)
    return min(range(31), key=lambda k: abs(round(64 * (30 - k) / 30) - left))


def step(state, action):
    board = state["board"]
    x0, y0, x1, y1 = bar(board)
    a = action[0]
    if a in (1, 2, 3, 4):
        dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[a]
        if x0 + dx >= 0 and x1 + dx < 64 and y0 + dy >= 1 and y1 + dy < 52:
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    board[y][x] = 12
            for y in range(y0 + dy, y1 + dy + 1):
                for x in range(x0 + dx, x1 + dx + 1):
                    board[y][x] = 9
    elif a == 5:
        state["won" if x0 == 24 else "lost"] = True
    k = taken(board) + 1
    for x in range(round(64 * (30 - k) / 30), 64):
        board[0][x] = 0
    return state


def won(state):
    return state["won"]


def lost(state):
    return state["lost"]


def clicks(state):
    return []
'''

# right but for left moves, which it takes to be twice as long: a demonstration that only moves right cannot tell
LONG_LEFT = SP80.replace("3: (-4, 0)", "3: (-8, 0)")
# right but for the timer along the top row, which it never shortens
NO_TIMER = SP80.replace("    for x in range(round(64 * (30 - k) / 30), 64):\n        board[0][x] = 0\n", "")


def model(program: str) -> str:
    return f"The game's rules:\n```python{program}```\n{{\"action\": \"model\"}}"


# instance seed 100 draws a level whose bar starts right of the place that wins (solved by 3, 3, 3, 5), and whose first
# demonstration starts left of it (4, 4, 4, 5); seed 102's bar starts left of it (4, 4, 4, 4, 4, 5)
def schedule(*seeds: int) -> Schedule:
    return Schedule(name="sp80", seed=0, slots=[Slot("sp80", s, "stream") for s in seeds], sources={"sp80": "arc3"},
                    meta={"dataset": "arc3", "track": "arc3"})


def play(tmp_path, replies, sched, **kwargs):
    """``sched`` played by the program library answering with ``replies``, through the proxy: the run's records, the
    requests the model was sent, and the learner."""
    server = FakeUpstream(replies=replies).start()
    try:
        with LLMProxy(server.base_url, api_key="sk-real") as proxy:
            learner = BareProgramLibraryLearner("vendor/model", max_tokens=1000, sandbox="none", **kwargs)
            learner.attach(tmp_path / "system", proxy.environ())
            env = Arc3Env(sched, source=LevelSource(GENERATORS, cache=tmp_path / "solutions"))
            result = run(learner, sched, env=env)
            learner.close()
    finally:
        server.stop()
    return result, [r["body"] for r in server.requests if r["body"]], learner


def last(body) -> str:
    return body["messages"][-1]["content"]


@needs_arc3
def test_the_hand_written_program_reproduces_sp80_and_its_plan_wins_in_the_engine(tmp_path):
    from continual_arc.arc3 import levels

    source = LevelSource(GENERATORS, cache=tmp_path)
    demos = []
    for seed in range(100, 104):
        drawn = source.instance("sp80", seed)
        demos.append(trajectory_of_demonstration(source.demonstration("sp80", seed, 0, drawn.layout).to_json(),
                                                 f"demonstration {seed}"))
    assert check(SP80, demos) is None
    assert check(NO_TIMER, demos)["kind"] == "board"
    drawn = source.instance("sp80", 104)
    _, frame = levels.new_game("sp80", drawn.level)
    found = plan(SP80, {"start": [list(r) for r in frame.frame[-1]], "steps": []}, [1, 2, 3, 4, 5, 6])
    assert found["found"] and len(found["plan"]) == len(drawn.solution)
    from continual_arc.arc3.instances import Act

    assert levels.replay("sp80", drawn.level, [Act(*a).engine() for a in found["plan"]])


@needs_arc3
def test_a_kept_program_wins_the_game_s_return_with_no_model_call(tmp_path):
    result, bodies, learner = play(tmp_path, ['{"action": "request_demo"}', model(SP80)], schedule(100, 102))
    first, second = result.records
    assert first.solved and first.demo_requests == 1 and first.wrong_attempts == 4
    assert second.solved and second.demo_requests == 0 and second.wrong_attempts == 6
    assert second.cost == pytest.approx(6 / second.metadata["baseline"])
    assert len(bodies) == 2   # the demonstration and the program: the return asked nothing
    rules = GAME_PROGRAM_NOTE.format(max_checks=3, max_programs=6)
    assert rules.splitlines()[0] in bodies[0]["messages"][0]["content"]
    assert "kept for the game" in bodies[0]["messages"][0]["content"]
    assert learner.stats["programs_kept"] == 1 and learner.stats["plans"] == 2
    assert learner.stats["library_hits"] == 1 and learner.stats["program_asks"] == 0
    (game,) = learner.trajectories
    names = [t["name"] for t in learner.trajectories[game]]
    assert names == ["earlier demonstration 1", "your play of an earlier level of this game (1)",
                     "your play of an earlier level of this game (2)"]
    assert learner.library[game].strip() == SP80.strip()
    learner.state_dir.mkdir(parents=True)   # where a logged state is committed from (attach's log_state)
    learner.save_state()
    saved = json.loads((learner.state_dir / "trajectories.json").read_text())
    assert [t["name"] for t in saved[game]] == names


@needs_arc3
def test_a_board_the_program_did_not_predict_stops_its_plan_and_the_repair_is_checked_on_it(tmp_path):
    result, bodies, learner = play(tmp_path, [model(LONG_LEFT), model(SP80)], schedule(100))
    (record,) = result.records
    assert record.solved and record.demo_requests == 0
    assert learner.stats["surprises"] == 1 and learner.stats["programs_kept"] == 2 and learner.stats["plans"] == 2
    told = last(bodies[1])
    assert "The plan of your program stopped after its action" in told and "(action 3)" in told
    assert "the game showed a board your program did not predict" in told
    assert check(LONG_LEFT, list(learner.trajectories.values())[0])["kind"] == "board"   # the surprise is a check now
    assert [p.strip() for p in learner.library.values()] == [SP80.strip()]


@needs_arc3
def test_a_failed_check_goes_back_to_the_model_with_the_cells_it_got_wrong(tmp_path):
    replies = ['{"action": "request_demo"}', model(NO_TIMER), model(SP80)]
    result, bodies, learner = play(tmp_path, replies, schedule(100))
    (record,) = result.records
    assert record.solved and learner.stats["checks_failed"] == 1 and learner.stats["programs_kept"] == 1
    told = last(bodies[2])
    assert told.startswith("Your program is wrong after action 1 of demonstration 1 (action 4)")
    assert "\ny0: " in told and "Failed checks in this instance: 1 of 3." in told


@needs_arc3
def test_a_model_that_plays_without_a_program_is_asked_for_one_after_the_instance(tmp_path):
    replies = ['{"action": "act", "ids": [3, 3, 3, 5]}', model(SP80)]
    result, bodies, learner = play(tmp_path, replies, schedule(100, 102))
    assert [r.solved for r in result.records] == [True, True]
    assert len(bodies) == 2 and learner.stats["program_asks"] == 1
    asked = last(bodies[1])
    assert "Action 5: the level is COMPLETED. Instance solved." in asked and asked.endswith(GAME_PROGRAM_ASK)
    assert learner.stats["library_hits"] == 1 and result.records[1].wrong_attempts == 6


@needs_arc3
def test_the_stateless_library_keeps_nothing_and_is_not_asked_after_an_instance(tmp_path):
    result, bodies, learner = play(tmp_path, [model(SP80)], schedule(100, 102), stateless=True)
    assert [r.solved for r in result.records] == [True, True]
    assert len(bodies) == 2   # the second instance had to ask again
    first = bodies[0]["messages"][0]["content"]
    assert "Here you may also answer with the game's rules as a program." in first and "kept for the game" not in first
    assert learner.stats["program_asks"] == 0 and learner.stats["library_hits"] == 0


@needs_arc3
def test_the_kept_program_and_what_became_of_it_open_the_model_s_first_message(tmp_path):
    # kept on seed 102 (moves right only), its plan on seed 100 meets its first left move
    replies = [model(LONG_LEFT), model(SP80)]
    result, bodies, learner = play(tmp_path, replies, schedule(102, 100))
    assert [r.solved for r in result.records] == [True, True]
    assert learner.stats["library_runs"] == 1 and learner.stats["surprises"] == 1
    opening = bodies[1]["messages"][0]["content"]
    assert "The program kept for game task-" in opening and "its plan played first" in opening
    assert "The plan of your program stopped after its action" in opening
