"""A hosted learner on every track (systems/hosted.py, learner_host.py, remote.py): every message of an ARC-AGI-3 game
crosses the wire unchanged and the far side can send nothing the environment cannot count; grids as large as any track
serves cross it both ways, on arc2-smoke too; arc3-smoke plays through HostedLearner under bubblewrap, and from an arena,
as the in-process runner plays it, its frozen test one instance at a time from the files its stream left; and the bare
model a verified run plays beside the learner plays a game's frozen test through the verifier's own command."""

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from continual_arc.arc3.env import (
    ActFeedback,
    Arc3LearnerObservation,
    DemonstrationFeedback,
    ResetFeedback,
    ResetLevel,
)
from continual_arc.arc3.instances import Act, Demonstration, DemoStep, LevelSource
from continual_arc.configs import build_schedule
from continual_arc.configs import test_schedule as frozen_test
from continual_arc.data import status
from continual_arc.env import (
    DemosFeedback,
    GiveUp,
    GiveUpFeedback,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.schedule import (
    PHASE_HELD_OUT_TEST,
    TEST_HELD_OUT,
    TESTS,
    Schedule,
    Slot,
    stream_schedule,
)
from continual_arc.tasks import Pair, TaskRegistry
from continual_arc.testing import FakeSource, files_digest

from continual_arc_baselines.cli import main
from continual_arc_baselines.systems import learner_host
from continual_arc_baselines.systems.hosted import (
    HostedLearnerError,
    decode_action,
    encode_action,
    encode_feedback,
    encode_observation,
)
from continual_arc_baselines.systems.remote import RemoteLearner
from continual_arc_baselines.systems.sandbox import python_runtime_root

from .test_arc3_protocol import arc3_available
from .test_hosted import SECRET, submission_python
from .test_remote import FAKE_SSH

BWRAP = pytest.mark.skipif(platform.system() != "Linux" or shutil.which("bwrap") is None,
                           reason="bubblewrap on Linux only")
needs_arc3 = pytest.mark.skipif(not arc3_available(), reason="arcengine or the pinned ARC-AGI-3 games missing")
needs_arc2 = pytest.mark.skipif(not all(s["ok"] for s in status(("arc2",)).values()),
                                reason="the second track's pinned data not fetched")
HOSTED = "continual_arc_baselines.systems.hosted:HostedLearner"
NOTES = "continual_arc.testing:Arc3NotesLearner"
SEED = 0


@pytest.fixture
def upstream():
    from .fake_upstream import FakeUpstream

    server = FakeUpstream().start()
    yield server
    server.stop()


def board(color: int) -> list[list[int]]:
    return [[(color + x * y) % 16 for x in range(64)] for y in range(64)]


def wire(message: dict) -> dict:
    """``message`` as the other side reads it: a JSON line."""
    return json.loads(json.dumps(message))


GAME = Arc3LearnerObservation(
    step=3, task_id="task-0a1b2c3d", frames=[board(1), board(2)], state="NOT_FINISHED", available_actions=[1, 2, 6],
    actions_used=4, resets_used=1, action_cap=110, baseline_actions=22.5, demo_requests_used=1, demo_cap=3, demo_cost=1.0,
    max_cost=8.0, metadata={"note": "kept"}, round=3, user=0)
GRID = LearnerObservation(step=0, task_id="task-1", input=[[1, 2], [3, 4]], attempts_used=0, demo_requests_used=0,
                          attempt_cap=8, demo_cap=8)


# -- the wire format ----------------------------------------------------------------------------------------------------


def test_every_message_of_a_game_and_of_a_grid_crosses_the_wire_unchanged():
    """What the trusted side encodes, the learner's host decodes into the benchmark's own objects, and back."""
    for obs in (GAME, GRID):
        assert learner_host.observation(wire(encode_observation(obs))) == obs
    for obs, actions in ((GAME, (Act(1), Act(6, 0, 63), Act(7), ResetLevel(), RequestDemos(), GiveUp())),
                         (GRID, (Submit([[0] * 30 for _ in range(30)]), RequestDemos(), GiveUp()))):
        for action in actions:
            assert decode_action(wire(learner_host.encode_action(action)), obs) == action  # the learner's
            assert learner_host.action(wire(encode_action(action))) == action               # the one it observes
    shown = Demonstration(board(0), (DemoStep(Act(6, 3, 4), ((3, 4, 9), (4, 4, 9))), DemoStep(Act(1), ())))
    for feedback in (ActFeedback("NOT_FINISHED", False, 5, True, False, False, 3), ActFeedback("WIN", True, 6, True, False),
                     ActFeedback("GAME_OVER", False, 7, False, True, True, 0), ResetFeedback("NOT_FINISHED", 7, 2, True),
                     DemonstrationFeedback(shown, False, 2), DemonstrationFeedback(None, True, 3), GiveUpFeedback(8.0),
                     SubmitFeedback(False, True, 3, False), DemosFeedback([Pair([[1]], [[2]], "n")], False, 1)):
        assert learner_host.feedback(wire(encode_feedback(feedback))) == feedback


def test_a_learners_numpy_integers_cross_as_integers_and_its_floats_do_not():
    numpy = pytest.importorskip("numpy")
    sent = learner_host.encode_action(Act(numpy.int64(6), numpy.int64(3), numpy.int32(4)))
    assert decode_action(wire(sent), GAME) == Act(6, 3, 4) and type(sent["x"]) is int
    with pytest.raises(TypeError):
        learner_host.encode_action(Act(6, 3.0, 4.0))


@pytest.mark.parametrize("sent", [
    {"type": "Submit", "grid": [[1]]}, {"type": "Act"}, {"type": "Act", "action": "6"}, {"type": "Act", "action": True},
    {"type": "Act", "action": 6.0}, {"type": "Act", "action": None}, {"type": "Act", "action": 6, "x": 1.5, "y": 2},
    {"type": "Act", "action": 6, "x": [1], "y": 2}, {"type": "Act", "action": 6, "x": 1, "y": "2"},
    {"type": "Act", "action": 6, "x": False, "y": 2}, {"type": "Finish"}, {"type": None}, ["Act", 1], None, "GiveUp",
])
def test_a_far_side_cannot_answer_a_game_with_what_its_environment_cannot_count(sent):
    with pytest.raises(HostedLearnerError, match="not one of the actions on a game"):
        decode_action(sent, GAME)


@pytest.mark.parametrize("sent", [{"type": "Act", "action": 1}, {"type": "ResetLevel"}, {"type": "act"}])
def test_a_far_side_cannot_answer_a_grid_with_a_games_action(sent):
    with pytest.raises(HostedLearnerError, match="not one of the actions on a grid"):
        decode_action(sent, GRID)


def test_an_integer_the_game_cannot_play_is_passed_on_for_the_environment_to_count_and_nothing_else_is_read():
    sent = {"type": "Act", "action": 99, "x": -5, "y": 10**30, "note": "x" * 1000}
    assert decode_action(sent, GAME) == Act(99, -5, 10**30)
    assert decode_action({"type": "ResetLevel", "grid": [[1]]}, GAME) == ResetLevel()
    with pytest.raises(HostedLearnerError) as raised:
        decode_action({"type": "Act", "action": "x" * 10_000}, GAME)
    assert len(str(raised.value)) < 500   # what a far side sent is quoted short


# -- grids --------------------------------------------------------------------------------------------------------------

# A grid learner that sends and is sent the largest grids: one demonstration, then the demonstration's output and the input
# submitted, then a give-up; it notes the shapes it was shown.
GRIDS_TOY = '''
import json
from pathlib import Path

from continual_arc.env import DemosFeedback, GiveUp, RequestDemos, Submit
from continual_arc.learner import BaseLearner


class Learner(BaseLearner):
    def __init__(self):
        self.pairs, self.shapes = {}, []

    def act(self, obs):
        self.shapes.append([len(obs.input), len(obs.input[0])])
        Path("shapes.json").write_text(json.dumps(self.shapes))
        if obs.demo_requests_used == 0:
            return RequestDemos()
        if obs.attempts_used == 0 and self.pairs.get(obs.step):
            return Submit(self.pairs[obs.step][0].output)
        if obs.attempts_used < 2:
            return Submit(obs.input)
        return GiveUp()

    def observe(self, obs, action, feedback):
        if isinstance(feedback, DemosFeedback):
            self.pairs[obs.step] = feedback.pairs
'''


def grids_toy(root: Path) -> Path:
    (root / "grids_toy").mkdir(parents=True)
    (root / "grids_toy" / "__init__.py").write_text(GRIDS_TOY)
    return root


def played(out: Path) -> list[dict]:
    """A run directory's records with what varies from launch to launch (counters and seconds) left out."""
    return [{k: v for k, v in json.loads(line).items() if k != "resources"}
            for line in (out / "records.jsonl").read_text().splitlines()]


def large_registry() -> TaskRegistry:
    """FakeSource's tasks at 30 by 30, the largest grid of any track (``grids.MAX_DIM``)."""
    source = FakeSource(n_tasks=3, size=(30, 30))
    return TaskRegistry({source.name: source})


def run_both(tmp_path, monkeypatch, upstream, code: Path | None, learner: str, schedule_args: list[str], hosted: dict,
             test: Path | None = None) -> tuple[Path, Path]:
    """``learner`` played over the same schedule in this process and through HostedLearner (``hosted``, its keyword
    arguments beyond the learner and its ``code``); the two run directories."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    if code is not None:
        monkeypatch.syspath_prepend(str(code))
    common = [*schedule_args, "--upstream", upstream.base_url, "--quiet"]
    assert main(["run", "--learner", learner, *common, "--out", str(tmp_path / "in-process")]) == 0
    kwargs = {"learner": learner, **({"code": str(code)} if code is not None else {}), **hosted}
    assert main(["run", "--learner", HOSTED, "--learner-kwargs", json.dumps(kwargs), *common,
                 *(["--test", str(test)] if test else []), "--out", str(tmp_path / "hosted")]) == 0
    return tmp_path / "in-process", tmp_path / "hosted"


def system_of(run: Path) -> Path:
    return Path(json.loads((run / "attempt.json").read_text())["attempt_dir"]) / "system"


@pytest.mark.parametrize("backend", ["none", pytest.param("bwrap", marks=BWRAP)])
def test_grids_as_large_as_a_track_serves_cross_the_hosted_path_both_ways(tmp_path, monkeypatch, upstream, backend):
    """30 by 30, the largest grid of any track: served, shown in demonstrations, submitted, and scored as in this
    process."""
    source = FakeSource(n_tasks=3, size=(30, 30))
    stream_schedule(source.task_ids(), 6, seed=1, source=source.name).save(tmp_path / "schedule.json")
    python = sys.executable if backend == "none" else submission_python(tmp_path)
    inside, hosted = run_both(tmp_path, monkeypatch, upstream, grids_toy(tmp_path / "code"), "grids_toy:Learner",
                              ["--schedule", str(tmp_path / "schedule.json"), "--registry",
                               "tests.test_hosted_tracks:large_registry"],
                              {"python": python, "runtime": [str(python_runtime_root(Path(python)))], "sandbox": backend})
    assert played(hosted) == played(inside)
    assert [(r["demo_requests"], r["wrong_attempts"], r["gave_up"]) for r in played(hosted)] == [(1, 2, True)] * 6
    assert json.loads((system_of(hosted) / "state" / "shapes.json").read_text())[0] == [30, 30]


@needs_arc2
def test_arc2_smoke_plays_through_the_hosted_path_as_in_this_process(tmp_path, monkeypatch, upstream):
    inside, hosted = run_both(tmp_path, monkeypatch, upstream, grids_toy(tmp_path / "code"), "grids_toy:Learner",
                              ["--config", "arc2-smoke", "--seed", str(SEED), "--stream-only"],
                              {"python": sys.executable, "sandbox": "none"})
    records = played(hosted)
    assert len(records) == 120 and records == played(inside)
    shapes = json.loads((system_of(hosted) / "state" / "shapes.json").read_text())
    assert max(h * w for h, w in shapes) > 100   # grids larger than the toy's


# -- ARC-AGI-3 ----------------------------------------------------------------------------------------------------------

SOLUTIONS: Path | None = None
"""Where the levels of this module's arc3 runs are solved, ahead of them (``level_source``)."""


def level_source() -> LevelSource:
    """The pinned games' generators, their solutions kept where this module's runs keep them: the factory
    ``arc-baselines run --registry`` takes on ARC-AGI-3, as a verified run names its season's."""
    return LevelSource(cache=SOLUTIONS)


def frozen_tests(seed: int) -> Schedule:
    """arc3-smoke's two frozen tests at ``seed`` as one round, every instance a user of its own and in the held-out
    test's phase, as a verified run serves them to its learner after its stream."""
    drawn = [(s.task_id, s.instance_seed) for test in TESTS for s in frozen_test("arc3-smoke", seed, test).slots]
    sources = {task: "arc3" for task, _ in drawn}
    slots = [Slot(task, instance_seed, PHASE_HELD_OUT_TEST, user=i) for i, (task, instance_seed) in enumerate(drawn)]
    meta = {"dataset": "arc3", "track": "arc3", "test": TEST_HELD_OUT, "tests": list(TESTS), "width": len(drawn)}
    return Schedule(name="arc3-smoke-frozen-tests", seed=seed, users=len(drawn), sources=sources, slots=slots, meta=meta)


@pytest.fixture(scope="module")
def arc3_smoke(tmp_path_factory):
    """arc3-smoke at :data:`SEED`, every level of its stream and tests and their demonstrations solved ahead, as a
    verified run solves its own before any cell plays, in a process of its own, since precompute forks its workers and
    this one has started threads; the frozen tests' schedule."""
    global SOLUTIONS
    SOLUTIONS = tmp_path_factory.mktemp("solutions")
    tests = frozen_tests(SEED)
    instances = sorted({(s.task_id, s.instance_seed) for sched in (build_schedule("arc3-smoke", SEED), tests)
                        for s in sched.slots})
    solve = ("import json, sys\nfrom continual_arc.arc3.instances import LevelSource, precompute\n"
             "cache, instances = json.loads(sys.argv[1])\n"
             "print(precompute(LevelSource(cache=cache), [tuple(i) for i in instances], demos=3, workers=8))")
    done = subprocess.run([sys.executable, "-c", solve, json.dumps([str(SOLUTIONS), instances])], capture_output=True,
                          text=True, check=True)
    assert done.stdout.split()[-1] == str(len(instances))
    yield tests
    SOLUTIONS = None


ARC3_RUN = ["--config", "arc3-smoke", "--seed", str(SEED), "--registry", "tests.test_hosted_tracks:level_source"]


def check_frozen_play(tests: Schedule, tested: list[dict], log: str) -> None:
    """The learner's log and records: its stream's files grow, every instance of its frozen test began from the files
    its stream left and wrote to its copy, a game the stream never served was met afresh in each, and every instance
    ended."""
    starts = [json.loads(line[6:]) for line in log.splitlines() if line.startswith("START ")]
    ends = [json.loads(line[4:]) for line in log.splitlines() if line.startswith("END ")]
    stream = len(starts) - len(tests.slots)
    assert stream == 4 and len(ends) == len(starts)
    end = ends[stream - 1]["state"]
    assert len({s["state"] for s in starts[:stream]}) == stream
    assert {s["state"] for s in starts[stream:]} == {end} and all(e["state"] != end for e in ends[stream:])
    assert [(r["task_id"], r["instance_seed"]) for r in tested] == [(s.task_id, s.instance_seed) for s in tests.slots]
    assert all(r["user"] == 0 and r["reset_before"] and r["gave_up"] for r in tested)
    held_out = {s.task_id for s in frozen_test("arc3-smoke", SEED, TEST_HELD_OUT).slots}
    assert [r["demo_requests"] for r in tested] == [int(r["task_id"] in held_out) for r in tested]
    assert all(r["wrong_attempts"] == 1 and r["metadata"]["resets"] == 1 for r in tested)


@needs_arc3
@BWRAP
def test_arc3_smoke_plays_through_a_hosted_learner_under_bubblewrap_as_in_this_process(tmp_path, monkeypatch, upstream,
                                                                                        arc3_smoke):
    """Its stream as the runner plays it here, record for record, every action, reset and demonstration through the
    wire; then its two frozen tests, one instance at a time, each from the files its stream left."""
    arc3_smoke.save(tmp_path / "tests.json")
    python = submission_python(tmp_path)   # no arcengine: the learner's side never plays a level itself
    inside, hosted = run_both(tmp_path, monkeypatch, upstream, None, NOTES, [*ARC3_RUN, "--stream-only"],
                              {"python": python, "runtime": [str(python_runtime_root(Path(python)))], "sandbox": "bwrap"},
                              test=tmp_path / "tests.json")
    records = played(hosted)
    assert records == played(inside) and len(records) == 4
    assert [r["demo_requests"] for r in records] == [1, 0, 1, 0] and all(r["gave_up"] for r in records)
    system = system_of(hosted)
    log = (system / "learner.log").read_text()
    check_frozen_play(arc3_smoke, played(hosted / "frozen" / TEST_HELD_OUT / "play"), log)
    assert "sandbox bwrap" in log and log.count("[hosted] started continual_arc.testing.Arc3NotesLearner") == 1 + 1 + 10
    # the stream's files are back in their place, without anything the test wrote
    kept = system / "state" / "arc3-notes"
    assert not (system / "stream-end").exists() and len(list((kept / "instances").iterdir())) == 4
    ends = [json.loads(line[4:]) for line in log.splitlines() if line.startswith("END ")]
    assert files_digest(kept) == ends[3]["state"]


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    folder = tmp_path / "fake-bin"
    folder.mkdir()
    script = folder / "ssh"
    script.write_text(FAKE_SSH)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")


@needs_arc3
@BWRAP
def test_arc3_smoke_plays_from_an_arena_with_its_frozen_test_after_a_reboot_before_each_instance(
        tmp_path, monkeypatch, upstream, fake_ssh, arc3_smoke):
    arc3_smoke.save(tmp_path / "tests.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    reboots = []  # never a real one: ssh here is this machine
    monkeypatch.setattr(RemoteLearner, "_reboot", lambda self: reboots.append(self.arena))
    python = submission_python(tmp_path)
    kwargs = {"arena": "arena@test", "key": str(tmp_path / "key"), "known_hosts": str(tmp_path / "known"),
              "python": python, "learner": NOTES, "runtime": [str(python_runtime_root(Path(python)))],
              "arena_workdir": str(tmp_path / "arena" / "cell"), "arena_python": sys.executable, "timeout_s": 120.0,
              "startup_timeout_s": 120.0, "margin_s": 30.0}
    out = tmp_path / "run"
    assert main(["run", "--learner", "continual_arc_baselines.systems.remote:RemoteLearner", "--learner-kwargs",
                 json.dumps(kwargs), *ARC3_RUN, "--test", str(tmp_path / "tests.json"), "--test-out",
                 str(tmp_path / "tested"), "--upstream", upstream.base_url, "--quiet", "--out", str(out)]) == 0
    assert reboots == ["arena@test"] * len(arc3_smoke.slots)
    assert len(played(out)) == 4
    system = system_of(out)
    check_frozen_play(arc3_smoke, played(tmp_path / "tested"), (system / "learner.log").read_text())
    assert (system / "kept.tar").stat().st_size > 0
    instances = tmp_path / "arena" / "cell" / "state" / "arc3-notes" / "instances"
    assert len(list(instances.iterdir())) == 4   # the stream's files back on the arena, without the test's


@needs_arc3
def test_the_bare_model_plays_a_games_frozen_test_through_the_verifiers_own_command(tmp_path, monkeypatch, arc3_smoke):
    """``BareModelLearner`` at ``end="finish"`` as a verified run's bare cell launches it: its model confined at its
    effort, a spend ledger and prices, each instance of the test alone with a reset before it. A demonstration bought,
    an action, a reset, and an instance ended by replies that are no action; a bare model is sent no verdict."""
    from .fake_upstream import FakeUpstream

    alone = Schedule(name="alone", seed=SEED, users=1, sources=dict(arc3_smoke.sources), meta={**arc3_smoke.meta, "width": 1},
                     slots=[Slot(s.task_id, s.instance_seed, s.phase, reset_before=True) for s in arc3_smoke.slots[:2]])
    alone.save(tmp_path / "alone.json")
    replies = ['{"action": "request_demo"}', '{"action": "act", "id": 1}', '{"action": "reset"}', "finish"]
    server = FakeUpstream(replies=replies).start()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_KEY", SECRET)
    price = {"request": 0.0, "input": 0.0, "output": 0.0, "window": 1}
    try:
        assert main(["run", "--schedule", str(tmp_path / "alone.json"), "--registry", "tests.test_hosted_tracks:level_source",
                     "--learner", "continual_arc_baselines.systems.bare:BareModelLearner", "--learner-kwargs",
                     json.dumps({"model": "vendor/model", "max_tokens": 128000, "end": "finish"}),
                     "--upstream", server.base_url, "--upstream-key-env", "RUN_KEY", "--confine-models", "vendor/model=medium",
                     "--spend-ledger", str(tmp_path / "spend.ledger"), "--spend-cap", "5", "--prices",
                     json.dumps({"vendor/model": price}), "--name", "bare-0", "--out", str(tmp_path / "bare-0"), "--quiet"]) == 0
    finally:
        server.stop()
    first, second = played(tmp_path / "bare-0")
    assert (first["demo_requests"], first["wrong_attempts"], first["metadata"]["resets"], first["gave_up"]) == (1, 1, 1, True)
    assert (second["demo_requests"], second["wrong_attempts"], second["gave_up"]) == (0, 0, True)
    bodies = [r["body"] for r in server.requests if r["body"]]
    assert all(b["model"] == "vendor/model" and b["reasoning"] == {"effort": "medium"} for b in bodies)
    sent = [b["messages"][-1]["content"] for b in bodies]
    assert "Continual-ARC-3" in sent[0] and "Demonstration 1: the benchmark's solver" in sent[1]
    assert len(sent) == 4 + 2 + 1 + 2   # each instance's replies and the two re-prompts of a reply that is no action
    assert not any("The instance ended unsolved" in m for m in sent)
