"""The ARC-AGI-3 protocol (systems/arc3_protocol.py): boards shown whole once and then as the cells that changed,
demonstrations, parsing, and a scripted model playing a level of the real engine through the adapter base, to a win
and to the cap, with what each instance cost it in text."""

import json

import pytest
from continual_arc.arc3.env import (
    ActFeedback,
    Arc3Config,
    Arc3Env,
    Arc3LearnerObservation,
    DemonstrationFeedback,
    ResetFeedback,
    ResetLevel,
)
from continual_arc.arc3.instances import Act, LevelSource
from continual_arc.env import GiveUpFeedback, RequestDemos
from continual_arc.runner import run
from continual_arc.schedule import Schedule, Slot

from continual_arc_baselines.proxy import LLMProxy
from continual_arc_baselines.systems.arc3_protocol import (
    MAX_BATCH,
    GameView,
    parse_action,
    render_board,
    render_changes,
    render_demonstration,
    render_feedback,
)
from continual_arc_baselines.systems.protocol import Finish, ParseError
from continual_arc_baselines.tracking import instance_ended

from .fake_upstream import FakeUpstream


def arc3_available() -> bool:
    try:
        import arcengine  # noqa: F401
        from continual_arc.arc3 import games
    except ImportError:
        return False
    return all(s["ok"] for s in games.status().values())


needs_arc3 = pytest.mark.skipif(not arc3_available(), reason="arcengine or the pinned ARC-AGI-3 games missing")

GENERATORS = {"sp80": "tests.arc3_sp80"}
BOARD = 64 * 65 - 1   # characters of a whole board: 64 rows of 64 hex digits


def blank(color: int = 0) -> list[list[int]]:
    return [[color] * 64 for _ in range(64)]


def obs(frame=None, *, step=0, actions=0, resets=0, demos=0, state="NOT_FINISHED", frames=None):
    return Arc3LearnerObservation(step=step, task_id="task-0a1b2c3d", frames=frames or [frame or blank()], state=state,
                                  available_actions=[1, 2, 3, 4, 5, 6], actions_used=actions, resets_used=resets,
                                  action_cap=110, baseline_actions=22.0, demo_requests_used=demos, demo_cap=3,
                                  demo_cost=1.0, max_cost=8.0)


# -- boards ------------------------------------------------------------------------------------------------------------


def test_changes_are_rows_of_runs_of_one_color():
    cells = ((20, 12, 0), (21, 12, 0), (22, 12, 0), (30, 12, 12), (3, 2, 15))
    assert render_changes(cells) == "y2: 3=f\ny12: 20-22=0 30=c"


def test_a_board_is_whole_once_then_its_changes_and_whole_when_they_would_be_longer():
    a, b = blank(), blank()
    b[5][7] = 3
    assert render_board(a, None).startswith("Board:\n" + "0" * 64) and len(render_board(a, None)) == len("Board:\n") + BOARD
    assert render_board(b, a) == "Board: 1 cells changed since the board last shown:\ny5: 7=3"
    assert render_board(a, a) == "Board: unchanged since the board last shown."
    noise = [[(x * 7 + y * 3) % 16 for x in range(64)] for y in range(64)]
    assert render_board(noise, a) == render_board(noise, None)   # listing 4,000 changes is longer than the board


def test_the_game_view_shows_the_whole_board_at_the_start_and_after_a_reset_and_changes_between():
    view = GameView()
    first = view.render_observation(obs(), pending=[], first_in_session=True, finish=True, rules_note="NOTE")
    assert "Continual-ARC-3" in first and '{"action": "finish"}' in first and "NOTE" in first
    assert "at most 5, plus\n1 for each demonstration (at most 3 an instance)" in first
    assert "the whole budget, 8." in first and "Board:\n" + "0" * 64 in first
    assert "New instance. Game id: task-0a1b2c3d." in first and "Actions used: 0/110 (baseline 22.0)" in first
    moved = blank()
    moved[10][10:15] = [9] * 5
    second = view.render_observation(obs(moved, actions=1), pending=["Action 4 applied."], first_in_session=False)
    assert "Continual-ARC-3" not in second and "y10: 10-14=9" in second and "0" * 64 not in second
    assert "Feedback since your last action:\nAction 4 applied." in second and "Same instance" in second
    third = view.render_observation(obs(moved, actions=1, resets=1), pending=[], first_in_session=False)
    assert "Board:\n" + "0" * 64 in third   # after a reset, whole
    animated = view.render_observation(obs(frames=[blank(), moved], actions=2, resets=1), pending=[],
                                       first_in_session=False)
    assert "played 2 frames" in animated and "unchanged" in animated
    over = view.render_observation(obs(moved, actions=3, resets=1, state="GAME_OVER"), pending=[],
                                   first_in_session=False)
    assert "The game is over: reset to continue." in over
    plain = GameView().render_observation(obs(), pending=[], first_in_session=True)
    assert '"finish"' not in plain and "worth keeping" in plain


def test_feedback_is_one_line_an_action():
    assert render_feedback(Act(4), ActFeedback("NOT_FINISHED", False, 3, True, False)) == "Action 4 applied."
    assert "COMPLETED" in render_feedback(Act(5), ActFeedback("WIN", True, 4, True, False))
    assert "not one this game accepts" in render_feedback(Act(6, 70, 1), ActFeedback("NOT_FINISHED", False, 1, False, False))
    assert "(70, 1)" in render_feedback(Act(6, 70, 1), ActFeedback("NOT_FINISHED", False, 1, False, False))
    assert "ignored" in render_feedback(Act(1), ActFeedback("GAME_OVER", False, 2, False, True))
    assert "cap is reached" in render_feedback(ResetLevel(), ResetFeedback("NOT_FINISHED", 9, 1, True))
    assert "REFUSED" in render_feedback(RequestDemos(), DemonstrationFeedback(None, True, 3))


def test_an_arc3_instance_ends_on_a_completed_level_or_its_cap():
    assert instance_ended(ActFeedback("WIN", True, 4, True, False))
    assert instance_ended(ActFeedback("NOT_FINISHED", False, 9, True, False, failed=True))
    assert instance_ended(ResetFeedback("NOT_FINISHED", 8, 1, failed=True))
    assert instance_ended(GiveUpFeedback(charged=8))
    assert not instance_ended(ActFeedback("GAME_OVER", False, 3, True, False))
    assert not instance_ended(DemonstrationFeedback(None, False, 1))


# -- parsing -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reply, action", [
    ('I will move up.\n{"action": "act", "id": 1}', Act(1)),
    ('{"action": "act", "id": 6, "x": 12, "y": 40}', Act(6, 12, 40)),
    ('{"action": "ACTION3"}', Act(3)),
    ('{"action": "act", "id": "ACTION2"}', Act(2)),
    ('{"action": "reset"}', ResetLevel()),
    ('```json\n{"action": "request_demo"}\n```', RequestDemos()),
    ('{"action": "request_demos"}', RequestDemos()),
])
def test_one_action_is_read_from_a_reply(reply, action):
    assert parse_action(reply) == action


def test_a_sequence_is_read_up_to_its_cap_and_finish_only_under_the_finish_protocol():
    assert parse_action('{"action": "act", "ids": [1, 1, 2]}') == [Act(1), Act(1), Act(2)]
    assert parse_action('{"actions": [{"id": 6, "x": 3, "y": 4}, {"id": 1}]}') == [Act(6, 3, 4), Act(1)]
    assert len(parse_action(json.dumps({"action": "act", "ids": [1] * 50}))) == MAX_BATCH
    assert parse_action('{"action": "finish"}', finish=True) == Finish()
    assert parse_action("done", finish=True) == Finish()
    with pytest.raises(ParseError, match="unknown action"):
        parse_action('{"action": "finish"}')
    with pytest.raises(ParseError, match="no give_up"):
        parse_action('{"action": "give_up"}')
    with pytest.raises(ParseError):
        parse_action("I am not sure what to do.")
    with pytest.raises(ParseError, match="empty"):
        parse_action('{"action": "act", "ids": []}')


# -- the real engine ---------------------------------------------------------------------------------------------------


def schedule(n: int) -> Schedule:
    return Schedule(name="sp80", seed=0, slots=[Slot("sp80", 100 + i, "stream") for i in range(n)],
                    sources={"sp80": "arc3"}, meta={"dataset": "arc3", "track": "arc3"})


@needs_arc3
def test_a_demonstration_costs_a_learner_well_under_20k_tokens(tmp_path):
    source = LevelSource(GENERATORS, cache=tmp_path)
    drawn = source.instance("sp80", 5)
    text = render_demonstration(source.demonstration("sp80", 5, 0, drawn.layout), 1)
    assert text.startswith("Demonstration 1: the benchmark's solver") and "Its first board:\n" in text
    assert "1. action" in text and "Its last board (the level completed):\n" in text
    assert len(text) < 3 * BOARD + 4_000   # two boards and the changes: some 4k tokens of a 20k budget


def _play(tmp_path, replies, sched, config=None, learner_cls=None):
    """``sched`` played by the bare model (or ``learner_cls``) answering with ``replies``, through the proxy; the
    run's records, the requests the model was sent, and the learner."""
    from continual_arc_baselines.systems.bare import BareModelLearner

    learner_cls = learner_cls or BareModelLearner
    server = FakeUpstream(replies=replies).start()
    try:
        with LLMProxy(server.base_url, api_key="sk-real") as proxy:
            learner = learner_cls("vendor/model", max_tokens=1000, end="finish")
            learner.attach(tmp_path / "system", proxy.environ())
            env = Arc3Env(sched, config=config, source=LevelSource(GENERATORS, cache=tmp_path / "solutions"))
            result = run(learner, sched, env=env)
            learner.close()
    finally:
        server.stop()
    return result, [r["body"] for r in server.requests if r["body"]], learner


def _solution(tmp_path, seed: int) -> list[Act]:
    return list(LevelSource(GENERATORS, cache=tmp_path / "solutions").instance("sp80", seed).solution)


@needs_arc3
def test_a_scripted_model_buys_a_demonstration_and_plays_a_level_to_a_win_in_one_sequence(tmp_path):
    solution = _solution(tmp_path, 100)
    replies = ['First, an example.\n{"action": "request_demo"}',
               json.dumps({"actions": [{"id": a.action} for a in solution]})]
    result, bodies, learner = _play(tmp_path, replies, schedule(1))
    (record,) = result.records
    assert record.solved and record.demo_requests == 1 and record.wrong_attempts == len(solution)
    assert record.cost == pytest.approx(1 + len(solution) / record.metadata["baseline"])
    assert len(bodies) == 2 and learner.stats["batched_actions"] == len(solution) - 1
    first, second = (b["messages"][-1]["content"] for b in bodies)
    assert "Continual-ARC-3" in first and "Board:\n" in first
    assert "Demonstration 1: the benchmark's solver" in second and "Continual-ARC-3" not in second
    assert "Board: unchanged since the board last shown." in second   # a demonstration leaves the level as it was
    # the whole conversation: the rules and the first board, and a demonstration's two boards and changes
    assert sum(len(m["content"]) for m in bodies[-1]["messages"]) < 4 * BOARD + 2_000


@needs_arc3
def test_a_scripted_model_that_never_pours_reaches_the_cap_and_sees_only_changes(tmp_path):
    cfg = Arc3Config(cap_multiplier=0.5)
    result, bodies, learner = _play(tmp_path, ['{"action": "act", "ids": [3, 4]}'], schedule(1), config=cfg)
    (record,) = result.records
    assert not record.solved and record.cost == cfg.max_cost and record.wrong_attempts == record.metadata["action_cap"]
    assert len(bodies) == -(-record.metadata["action_cap"] // 2)
    later = [b["messages"][-1]["content"] for b in bodies[1:]]
    assert later and all("Board: " in m and "Board:\n" not in m for m in later)   # changes alone after the first
    assert max(len(m) for m in later) < BOARD // 4


@needs_arc3
def test_the_demo_cache_shows_the_demonstrations_it_bought_when_the_game_returns(tmp_path):
    from continual_arc_baselines.systems.bare import BareDemoCacheLearner

    one, two = _solution(tmp_path, 100), _solution(tmp_path, 101)
    replies = ['{"action": "request_demo"}', json.dumps({"action": "act", "ids": [a.action for a in one]}),
               json.dumps({"action": "act", "ids": [a.action for a in two]})]
    result, bodies, learner = _play(tmp_path, replies, schedule(2), learner_cls=BareDemoCacheLearner)
    assert [r.solved for r in result.records] == [True, True] and [r.demo_requests for r in result.records] == [1, 0]
    returning = bodies[2]["messages"][0]["content"]
    assert "Demonstrations of game task-" in returning and "Demonstration 1: the benchmark's solver" in returning
    assert result.records[1].cost == pytest.approx(len(two) / result.records[1].metadata["baseline"])
    learner.state_dir.mkdir(parents=True)   # where a logged state is committed from (attach's log_state)
    learner.save_state()
    learner.cache = {}
    learner.load_state()
    assert [type(d).__name__ for d in next(iter(learner.cache.values()))] == ["Demonstration"]


def test_the_module_names_one_game_view_per_instance():
    from continual_arc_baselines.systems.base import SystemLearner

    learner = SystemLearner()
    first = learner.protocol_for(obs())
    assert isinstance(first, GameView) and learner.protocol_for(obs(actions=1)) is first
    assert learner.protocol_for(obs(step=1)) is not first
