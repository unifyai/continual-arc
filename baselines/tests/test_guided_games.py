"""Expert guidance on ARC-AGI-3 games (base.py's program_library on a game, guidance.GAME_GUIDANCE): the game policy in
the rules, the world-model tools and the instance's boards written into the library directory before every message
(``carc_games.py`` and ``instance.json``), as the conversation has shown them, and the game's library ask after the
verdict. A scripted system plays the tests' sp80 in the real engine, and the tools are then run as an agent runs them,
in a Python of its own."""

import json
import subprocess
import sys
from pathlib import Path

from continual_arc.arc3.env import Arc3Env
from continual_arc.arc3.instances import LevelSource
from continual_arc.runner import run

from continual_arc_baselines.systems import protocol, world_model
from continual_arc_baselines.systems.base import SystemLearner
from continual_arc_baselines.systems.guidance import (
    EXPERT_GUIDANCE,
    GAME_GUIDANCE,
    GAME_GUIDANCE_SHA256,
)
from continual_arc_baselines.systems.world_model import boards, diff

from .test_arc3_protocol import GENERATORS, needs_arc3
from .test_bare_games import SP80, schedule


class Scripted(SystemLearner):
    """Answers each message with the next scripted reply, and keeps the instance's files as each message found them."""

    name = "scripted"

    def __init__(self, replies, workspace, **kwargs):
        super().__init__(end="finish", program_library=True, **kwargs)
        self.replies, self.sent, self.files, self.workspace = list(replies), [], [], workspace

    def start(self):
        pass

    def stop(self):
        pass

    def new_session(self):
        pass

    def wipe_state(self):
        pass

    def ask(self, message, *, reprompt=False):
        self.sent.append(message)
        self.files.append(json.loads((Path(self.programs_dir()) / world_model.INSTANCE).read_text()))
        return self.replies.pop(0) if self.replies else '{"action": "finish"}'


REPLIES = ['{"action": "request_demo"}', '{"action": "act", "ids": [3, 3]}',
           '{"actions": [{"id": 3}, {"id": 5}]}', 'Saved.\n{"action": "finish"}']


def play(tmp_path, **kwargs):
    learner = Scripted(REPLIES, tmp_path / "workspace", **kwargs)
    learner.attach(tmp_path / "system", {})
    sched = schedule(100)
    result = run(learner, sched, env=Arc3Env(sched, source=LevelSource(GENERATORS, cache=tmp_path / "solutions")))
    return result, learner


def tools(folder: Path, code: str) -> str:
    """``code`` run after importing the tools as the guidance says, in a Python of its own: its output."""
    script = f"import sys\nsys.path.insert(0, {str(folder)!r})\nfrom carc_games import load, check_game, plan_game, save\n{code}"
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip()


@needs_arc3
def test_a_guided_game_reads_the_game_policy_and_finds_its_boards_beside_the_tools(tmp_path):
    result, learner = play(tmp_path)
    (record,) = result.records
    assert record.solved and record.demo_requests == 1 and record.wrong_attempts == 4
    folder = tmp_path / "workspace" / "programs"
    first = learner.sent[0]
    assert first.count(GAME_GUIDANCE) == 1 and EXPERT_GUIDANCE not in first
    assert f"Expert-guidance library directory: {folder}\n\n" in first
    assert (folder / "carc_games.py").read_text() == Path(world_model.__file__).read_text()
    opening, after_demo, after_two, verdict = learner.files
    shown = opening["board"]
    assert opening["current"] == {"name": "your play of this level", "start": shown, "steps": []}
    assert opening["play"] == [] and opening["demonstrations"] == [] and opening["available"] == [1, 2, 3, 4, 5, 6]
    (demo,) = after_demo["demonstrations"]
    assert demo["name"] == "demonstration 1" and demo["steps"][-1]["won"]
    assert after_demo["play"] == []   # a demonstration leaves the level as it was
    (two,) = after_two["play"]
    assert two["steps"] == [{"actions": [[3], [3]], "changes": diff(shown, after_two["board"]), "won": False,
                             "over": False}]
    assert after_two["board"] != shown and boards(two)[-1] == after_two["board"]
    (whole,) = verdict["play"]
    assert [s["actions"] for s in whole["steps"]] == [[[3], [3]], [[3], [5]]] and whole["steps"][-1]["won"]
    assert verdict["current"] is None
    game = verdict["game"]
    ask = learner.sent[3]
    assert ask.startswith("Action 3 applied.\nAction 5: the level is COMPLETED. Instance solved.")
    assert ask.endswith(protocol.GAME_LIBRARY_ASK.format(
        note="", library=folder, trajectories=f"{folder}/{game}.json", notes=f"{folder}/{game}.md",
        program=f"{folder}/{game}.py", task=game))
    assert learner.stats["library_asks"] == 1 and learner.stats["finishes"] == 1
    assert learner.describe()["game_guidance_sha256"] == GAME_GUIDANCE_SHA256
    # the agent's own Python: the tools load the instance, check a program on it, and keep its trajectories
    (folder / f"{game}.py").write_text(SP80)
    assert tools(folder, "print(len(load()['all']))") == "2"
    assert tools(folder, "print(check_game())") == "None"
    assert tools(folder, "print(save(), save())") == "2 0"
    assert tools(folder, "print(len(load()['kept']))") == "2"


@needs_arc3
def test_the_tools_plan_from_the_board_the_conversation_shows_and_the_plan_wins(tmp_path):
    learner = Scripted(['{"action": "act", "ids": [3]}'], tmp_path / "workspace")
    learner.attach(tmp_path / "system", {})
    sched = schedule(100)
    env = Arc3Env(sched, source=LevelSource(GENERATORS, cache=tmp_path / "solutions"))
    from continual_arc.aliases import TaskAliases
    from continual_arc.runner import learner_view

    aliases = TaskAliases()
    obs = env.reset()
    shown = learner_view(obs, aliases)
    action = learner.act(shown)                    # the first message: one move left
    result = env.step(action)
    learner.observe(shown, action, result.feedback)
    shown = learner_view(result.observation, aliases)
    folder = tmp_path / "workspace" / "programs"
    learner.replies = ['{"action": "act", "ids": [3]}']
    learner.act(shown)                             # the second message writes the board after that move
    game = learner.files[-1]["game"]
    (folder / f"{game}.py").write_text(SP80)
    moves = json.loads(tools(folder, "import json; print(json.dumps(plan_game()))"))
    assert moves == [{"id": 3}, {"id": 3}, {"id": 5}]   # two more of the three moves, then the pour
    from continual_arc.arc3.instances import Act

    for move in moves:   # played in place of the scripted reply's move
        result = env.step(Act(move["id"]))
    assert result.feedback.level_completed


@needs_arc3
def test_the_wiped_twin_gets_the_same_policy_tools_and_ask(tmp_path):
    result, learner = play(tmp_path, stateless=True)
    assert result.records[0].solved
    assert learner.sent[0].count(GAME_GUIDANCE) == 1 and "Now save this game's library" in learner.sent[3]
    assert (tmp_path / "workspace" / "programs" / "carc_games.py").exists()
