"""reference.py off any network: each player's request settings, and that a reference row plays under exactly the
harness settings paper.play_round gives the trained rows (only the model and its serving differ), row hpl under row h's
with baselines' program library added, row hg under row h's with every user's turn after the verdict glm's, rows ha
and hpl0 as the paper's ablations, and the bare-model rows as the paper's floors and program library.

Run with ``python3 -m pytest test_reference.py -q``; needs ``continual_arc`` and ``continual_arc_baselines`` on the
path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import loop
import paper
import reference

ONE_USER = SimpleNamespace(users=1)
# the trained row whose harness settings each reference row adds to
PAPER_ROW = {"hpl": "h", "hg": "h", "ha": "h", "hpl0": "h0", "hplg": "h", "hgpl": "h"}


def test_players_play_at_the_trained_rows_settings():
    glm, qwen = reference.PLAYERS["glm"]["body"], reference.PLAYERS["qwen"]["body"]
    assert glm["reasoning"] == {"effort": "medium"}
    assert glm["provider"] == {"order": ["gmicloud", "parasail", "fireworks"], "allow_fallbacks": False}
    assert {k: qwen[k] for k in paper.SAMPLING} == {**paper.SAMPLING, "top_k": 0}
    assert qwen["reasoning"] == {"enabled": True}
    assert qwen["provider"] == {"order": ["together", "venice", "deepinfra"], "allow_fallbacks": False}


def _played(monkeypatch, fn, *args, **kw):
    """``fn`` (a play_round) with the learner, the runner and the env faked: the harness's kwargs and the learner's."""
    import continual_arc.runner as runner
    import continual_arc.tasks as tasks
    import continual_arc_baselines.collective as collective
    import continual_arc_baselines.systems.prime_agent as prime_agent
    seen: dict = {}

    class Learner:
        def __init__(self, make_system, run_dir, **kwargs):
            seen["learner"] = kwargs
            make_system(0)

        def _user(self, user):
            return SimpleNamespace(index=user, system=SimpleNamespace(_finish_instance=lambda obs, pending: None),
                                   proxy=SimpleNamespace(body_overrides={}))

        def close(self):
            pass

    monkeypatch.setattr(prime_agent, "PrimeAgentLearner", lambda **kwargs: seen.setdefault("harness", kwargs))
    monkeypatch.setattr(collective, "CollectiveHarnessLearner", Learner)
    monkeypatch.setattr(runner, "build_env", lambda *a, **k: None)
    monkeypatch.setattr(runner, "run", lambda *a, **k: SimpleNamespace(records=[], summary={"protocol": {}}))
    monkeypatch.setattr(tasks.TaskRegistry, "default", staticmethod(lambda: None))
    for name in ("_refresh_meta_protocol", "_report_round_memory", "_wipe_stateless_users"):
        monkeypatch.setattr(paper, name, lambda *a: None)
    fn(*args, **kw)
    return seen["harness"], seen["learner"]


@pytest.mark.parametrize("row", reference.REFERENCE_ROWS)
@pytest.mark.parametrize("player", list(reference.PLAYERS))
def test_a_reference_row_plays_the_trained_rows_harness(tmp_path, monkeypatch, row, player):
    harness, learner = _played(monkeypatch, reference.play_round, player, row, tmp_path / "ref", ONE_USER, api_key="k")
    trained, _ = _played(monkeypatch, paper.play_round, PAPER_ROW.get(row, row), tmp_path / "paper", ONE_USER,
                         base_url="http://x", served_name="m")
    assert {**harness, "model": None} == {**trained, "model": None, **reference.REFERENCE_ROWS[row]}
    assert learner["upstream"] == reference.OPENROUTER and learner["api_key"] == "k"
    assert learner["body_overrides"]["model"] == reference.PLAYERS[player]["model"]
    assert learner["declared_context_length"] == paper.CONTEXT_WINDOW
    assert learner["role_map"] == {"developer": "system"} and learner["body_drop"] == ("reasoning_effort",)


def test_row_hpl_is_told_the_program_library_and_keeps_its_memory(tmp_path, monkeypatch):
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    harness, _ = _played(monkeypatch, reference.play_round, "qwen", "hpl", tmp_path / "ref", None, api_key="k")
    system = PrimeAgentLearner(**harness)
    assert system.program_library and system.end == "finish" and not system.stateless


def test_rows_ha_and_hpl0_are_the_papers_ablations(tmp_path, monkeypatch):
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    ha, _ = _played(monkeypatch, reference.play_round, "qwen", "ha", tmp_path / "ha", ONE_USER, api_key="k")
    hpl0, _ = _played(monkeypatch, reference.play_round, "qwen", "hpl0", tmp_path / "hpl0", ONE_USER, api_key="k")
    asked, wiped = PrimeAgentLearner(**ha), PrimeAgentLearner(**hpl0)
    assert asked.ask_to_store and not asked.stateless and not asked.program_library
    assert wiped.program_library and wiped.stateless and not wiped.ask_to_store
    assert json.loads((tmp_path / "hpl0" / "driver.json").read_text())["stateless"]


@pytest.mark.parametrize("row,library,told", [("hg", False, False), ("hplg", True, False), ("hgpl", False, True)])
def test_rows_hg_hplg_and_hgpl_hand_every_users_turn_after_the_verdict_to_glm(tmp_path, monkeypatch, row, library,
                                                                               told):
    handed = []
    monkeypatch.setattr(reference, "hand_storage",
                        lambda user, storer, told=False: handed.append((user.index, storer, told)))
    harness, learner = _played(monkeypatch, reference.play_round, "qwen", row, tmp_path / row, SimpleNamespace(users=2),
                               api_key="k")
    glm = reference.overrides_of("glm")
    assert handed == [(0, glm, told), (1, glm, told)] and learner["body_overrides"] == reference.overrides_of("qwen")
    assert bool(harness.get("program_library")) == library
    driver = json.loads((tmp_path / row / "driver.json").read_text())
    assert driver["player"] == "qwen" and driver["storer"] == {"player": "glm", "overrides": glm, "told": told}
    handed.clear()
    _played(monkeypatch, reference.play_round, "qwen", "h", tmp_path / "h", SimpleNamespace(users=2), api_key="k")
    assert not handed and "storer" not in json.loads((tmp_path / "h" / "driver.json").read_text())


def test_the_turn_after_the_verdict_is_the_storers_and_the_next_play_the_players(tmp_path, monkeypatch):
    """A real row-hg harness: its close of an instance runs the finish protocol's turn with the storer's request fields
    on its proxy, and the player's are back after it, a failed turn included."""
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    harness, _ = _played(monkeypatch, reference.play_round, "qwen", "hg", tmp_path / "hg", ONE_USER, api_key="k")
    system = PrimeAgentLearner(**harness)
    player, glm = reference.overrides_of("qwen"), reference.overrides_of("glm")
    proxy = SimpleNamespace(body_overrides=player)
    during = []

    def finish(obs, pending):
        during.append(proxy.body_overrides)
        if len(during) == 2:
            raise RuntimeError("the harness died in the turn")

    system._finish_instance = finish
    reference.hand_storage(SimpleNamespace(system=system, proxy=proxy), glm)
    system._started = True
    system._close_instance(SimpleNamespace())
    assert during == [glm] and proxy.body_overrides is player
    with pytest.raises(RuntimeError):
        system._close_instance(SimpleNamespace())
    assert during == [glm, glm] and proxy.body_overrides is player


@pytest.mark.parametrize("told", [False, True])
def test_only_row_hgpls_storer_is_told_the_program_library_after_the_verdict(tmp_path, monkeypatch, told):
    """A real row-hgpl harness: the turn after the verdict carries the program library's procedure and the ask to save
    the task's program, as row hpl's player reads them, to the storer alone; row hg's carries the verdict alone, and
    neither player's harness is told the procedure."""
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    from continual_arc_baselines.systems.protocol import PROGRAM_LIBRARY_ASK, PROGRAM_LIBRARY_RULES
    row = "hgpl" if told else "hg"
    harness, _ = _played(monkeypatch, reference.play_round, "qwen", row, tmp_path / row, ONE_USER, api_key="k")
    system = PrimeAgentLearner(**harness)
    assert not system.program_library
    proxy = SimpleNamespace(body_overrides=reference.overrides_of("qwen"))
    seen = []
    system._finish_instance = lambda obs, pending: seen.append((proxy.body_overrides, pending))
    reference.hand_storage(SimpleNamespace(system=system, proxy=proxy), reference.overrides_of("glm"), told=told)
    system._started = True
    system.transcript.add_text("The verdict.")
    system._close_instance(SimpleNamespace(task_id="task-1"))
    ((overrides, pending),) = seen
    programs = system.programs_dir()
    lesson = PROGRAM_LIBRARY_RULES.format(programs=programs) + "\n\n" + PROGRAM_LIBRARY_ASK.format(
        path=f"{programs}/task-1.py")
    assert overrides == reference.overrides_of("glm") and pending[0] == "The verdict."
    assert pending == (["The verdict.", "", lesson] if told else ["The verdict."])


@pytest.mark.parametrize("at,keep", [(49_152, 8_000), (65_536, 8_000), (0, 0)])
def test_a_compaction_probe_changes_the_compaction_point_alone(tmp_path, monkeypatch, at, keep):
    probe, _ = _played(monkeypatch, reference.play_round, "qwen", "h", tmp_path / "probe", None, api_key="k",
                       harness=loop.harness_of(at, keep))
    rows, _ = _played(monkeypatch, reference.play_round, "qwen", "h", tmp_path / "row", None, api_key="k")
    assert (probe["compact_at"], probe["compact_keep"]) == (at or None, keep or None)
    assert {**probe, "compact_at": 0, "compact_keep": 0} == {**rows, "compact_at": 0, "compact_keep": 0}
    assert loop.harness_of(rows["compact_at"], rows["compact_keep"]) == loop.HARNESS


def _bare_played(monkeypatch, player, row, run_dir, schedule, rounds_done=0):
    """reference.play_bare_round with the collective learner, the runner and the env faked: the systems it made."""
    import continual_arc.runner as runner
    import continual_arc.tasks as tasks
    import continual_arc_baselines.collective as collective
    made = []

    class Learner:
        def __init__(self, make_system, run_dir, **kwargs):
            made.extend(make_system(user) for user in range(schedule.users))

        def _user(self, user):
            return SimpleNamespace(system=made[user])

        def close(self):
            pass

    monkeypatch.setattr(collective, "CollectiveHarnessLearner", Learner)
    monkeypatch.setattr(runner, "build_env", lambda *a, **k: None)
    monkeypatch.setattr(runner, "run", lambda *a, **k: SimpleNamespace(records=[], summary={}))
    monkeypatch.setattr(tasks.TaskRegistry, "default", staticmethod(lambda: None))
    reference.play_bare_round(player, row, run_dir, schedule, api_key="k", rounds_done=rounds_done)
    return made


@pytest.mark.parametrize("row,learner,stateless", [("f", "BareModelLearner", False),
                                                  ("m", "BareProgramLibraryLearner", False),
                                                  ("m0", "BareProgramLibraryLearner", True)])
@pytest.mark.parametrize("player", reference.PLAYERS)
def test_a_bare_row_is_the_papers_row_at_the_players_settings(player, row, learner, stateless, tmp_path, monkeypatch):
    (system,) = _bare_played(monkeypatch, player, row, tmp_path / "play", SimpleNamespace(users=1, rounds=lambda: [0]))
    assert type(system).__name__ == learner and system.stateless == stateless
    assert (system.model, system.max_tokens, system.end, system.empty_reply, system.archive_wipes) == \
        (reference.PLAYERS[player]["model"], paper.MAX_TOKENS, "finish", "reprompt", True)
    driver = json.loads((tmp_path / "play" / "driver.json").read_text())
    assert driver["row"] == row and driver["learner"].endswith(f":{learner}")


def test_row_m_takes_up_the_programs_and_pairs_the_round_before_kept(tmp_path, monkeypatch):
    from continual_arc.tasks import Pair
    from continual_arc_baselines.systems.bare import BareProgramLibraryLearner
    first = BareProgramLibraryLearner("m", **paper.FLOOR)
    first.library = {"t1": "def transform(grid):\n    return grid\n"}
    first.cache = {"t1": [Pair.from_json({"input": [[1]], "output": [[2]]})]}
    reference.keep_library(first, tmp_path / "play" / "library" / "r0004-u000.json")
    schedule = SimpleNamespace(users=1, rounds=lambda: [0, 1, 2, 3, 4, 5])
    (second,) = _bare_played(monkeypatch, "qwen", "m", tmp_path / "play", schedule, rounds_done=5)
    assert second.library == first.library
    assert {t: [p.to_json() for p in ps] for t, ps in second.cache.items()} == \
        {t: [p.to_json() for p in ps] for t, ps in first.cache.items()}
    assert (tmp_path / "play" / "library" / "r0005-u000.json").exists()


@pytest.mark.parametrize("player", reference.PLAYERS)
def test_the_floor_is_the_paper_floor_at_the_players_settings_with_one_cache_for_the_team(player, tmp_path,
                                                                                        monkeypatch):
    """reference.play_floor_round: every user's bare model is the paper's demo-cache floor (paper.FLOOR, the
    player's reply cap) on one team cache opened on the run's demo-cache/ after the rounds already played, the player's
    request settings its proxy's defaults; a round's end ends the cache's round, and a reset wipes it."""
    import continual_arc.runner as runner
    import continual_arc.tasks as tasks
    import continual_arc_baselines.collective as collective
    from continual_arc_baselines.systems.bare import BareSharedDemoCacheLearner
    seen: dict = {}

    class Learner:
        def __init__(self, make_system, run_dir, **kwargs):
            seen.update(kwargs, users=[make_system(0), make_system(1)])

        def close(self):
            pass

    monkeypatch.setattr(collective, "CollectiveHarnessLearner", Learner)
    monkeypatch.setattr(runner, "build_env", lambda *a, **k: None)
    monkeypatch.setattr(runner, "run", lambda *a, **k: SimpleNamespace(records=[], summary={}))
    monkeypatch.setattr(tasks.TaskRegistry, "default", staticmethod(lambda: None))
    (tmp_path / "play" / "demo-cache").mkdir(parents=True)
    (tmp_path / "play" / "demo-cache" / "r0000.json").write_text(json.dumps(
        [{"user": 1, "task": "t1", "pairs": [{"input": [[1]], "output": [[2]]}]}]))
    reference.play_bare_round(player, "dc", tmp_path / "play", None, api_key="k", rounds_done=1)
    spec = reference.PLAYERS[player]
    first, second = seen["users"]
    assert isinstance(first, BareSharedDemoCacheLearner) and (first.user, second.user) == (0, 1)
    assert first.shared is second.shared and first.cached_pairs("t1") == second.cached_pairs("t1") != []
    assert (first.model, first.max_tokens, first.end, first.empty_reply, first.archive_wipes) == \
        (spec["model"], paper.MAX_TOKENS, "finish", "reprompt", True)
    assert seen["upstream"] == reference.OPENROUTER and seen["api_key"] == "k"
    assert seen["body_defaults"] == spec["body"] and not seen["capture"]
    seen["round_hook"](None)
    assert (tmp_path / "play" / "demo-cache" / "r0001.json").exists()
    seen["reset_hook"]()
    assert first.cached_pairs("t1") == [] and not list((tmp_path / "play" / "demo-cache").glob("r*.json"))
    driver = json.loads((tmp_path / "play" / "driver.json").read_text())
    assert driver["row"] == "dc" and driver["learner"].endswith(":BareSharedDemoCacheLearner")


def test_floor_plays_the_one_user_rows_stream():
    from continual_arc.configs import build_schedule, get_config
    canonical = build_schedule(get_config(paper.STREAM_CONFIG), "workstreams", paper.SEED)
    rows = paper.full_schedule(paper.SEED, 1)
    assert [(s.user, s.phase, s.task_id, s.instance_seed) for s in canonical.slots] == \
        [(s.user, s.phase, s.task_id, s.instance_seed) for s in rows.slots]
