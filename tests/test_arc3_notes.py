"""The ARC-AGI-3 learner of continual_arc.testing that checks a run end to end: what it plays, what it prints, the one
folder it keeps, and a reset that deletes that folder alone."""

import hashlib
import json

from continual_arc.arc3.env import ActFeedback, Arc3LearnerObservation, ResetLevel
from continual_arc.arc3.instances import Act
from continual_arc.env import GiveUp, GiveUpFeedback, RequestDemos
from continual_arc.testing import Arc3NotesLearner, files_digest

EMPTY = hashlib.sha256().hexdigest()


def obs(task: str = "task-a", actions: int = 0, resets: int = 0, demos: int = 0, available=(1, 6), step: int = 0):
    return Arc3LearnerObservation(step=step, task_id=task, frames=[[[0] * 64 for _ in range(64)]], state="NOT_FINISHED",
                                  available_actions=list(available), actions_used=actions, resets_used=resets,
                                  action_cap=50, baseline_actions=10.0, demo_requests_used=demos, demo_cap=3,
                                  demo_cost=1.0, max_cost=8.0)


def printed(err: str, kind: str) -> list[dict]:
    return [json.loads(line.split(" ", 1)[1]) for line in err.splitlines() if line.startswith(kind + " ")]


def test_it_buys_a_demonstration_of_a_new_game_then_acts_resets_and_gives_up(tmp_path, capsys):
    learner = Arc3NotesLearner()
    learner.attach(tmp_path, {})
    plays = [learner.act(obs()), learner.act(obs(demos=1)), learner.act(obs(demos=1, actions=1)),
             learner.act(obs(demos=1, actions=1, resets=1))]
    assert plays == [RequestDemos(), Act(6, 32, 32), ResetLevel(), GiveUp()]
    learner.observe(obs(demos=1, actions=1, resets=1), GiveUp(), GiveUpFeedback(8.0))
    # a game met before is played at once; a level completed ends the instance as a give-up does
    assert learner.act(obs(step=1, available=(1, 2))) == Act(1)
    learner.observe(obs(step=1, available=(1, 2)), Act(1), ActFeedback("WIN", True, 1, True, False))
    err = capsys.readouterr().err
    starts, ends = printed(err, "START"), printed(err, "END")
    assert [s["state"] for s in starts] == [EMPTY, ends[0]["state"]] and len(ends) == 2
    assert ends[1]["state"] == files_digest(tmp_path / "arc3-notes") != ends[0]["state"]
    assert json.loads((tmp_path / "arc3-notes" / "games" / "task-a.json").read_text()) == {"met": 2}
    assert sorted(p.name for p in (tmp_path / "arc3-notes" / "instances").iterdir()) == ["0000.json", "0001.json"]


def test_a_reset_deletes_its_folder_alone_and_given_no_directory_it_keeps_one_of_its_own(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine").write_text("not the learner's")
    learner = Arc3NotesLearner()
    learner.act(obs())
    assert learner.folder.exists() and not learner.folder.is_relative_to(tmp_path)
    learner.reset()
    assert not learner.folder.exists() and sorted(p.name for p in tmp_path.iterdir()) == ["mine"]
    attached = Arc3NotesLearner()
    attached.attach(tmp_path, {})
    attached.act(obs())
    attached.reset()
    attached.reset()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mine"] and attached.act(obs()) == RequestDemos()
    assert printed(capsys.readouterr().err, "START")[-1]["state"] == EMPTY
