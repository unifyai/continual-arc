"""The reference rows (continual_arc_training.references) off any network: each player's request settings, and that a
reference row plays under exactly the harness settings play.play_round gives the trained rows (only the model and its
serving differ), row hpl under row h's with the baselines' program library added, rows ha and hpl0 as the ablations,
and the bare-model rows as the floors and program library."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from continual_arc_training import play
from continual_arc_training import references as reference

ONE_USER = SimpleNamespace(users=1)
# the row of play.play_round whose harness settings each reference row adds to: a trained row's (memory kept) or h0's
PAPER_ROW = {"h": "dagger", "h0": "h0", "hpl": "dagger", "ha": "dagger", "hpl0": "h0"}


def test_players_play_at_the_trained_rows_settings():
    glm, qwen = reference.PLAYERS["glm"]["body"], reference.PLAYERS["qwen"]["body"]
    assert glm["reasoning"] == {"effort": "medium"}
    assert glm["provider"] == {"order": ["z-ai", "fireworks"], "allow_fallbacks": False}
    assert {k: qwen[k] for k in ("temperature", "top_p", "top_k")} == {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    assert not {"min_p", "presence_penalty"} & set(qwen)   # OpenRouter rows sample the whole distribution
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
    for name in ("refresh_meta_protocol", "report_round_memory", "wipe_stateless_users"):
        monkeypatch.setattr(play, name, lambda *a: None)
    fn(*args, **kw)
    return seen["harness"], seen["learner"]


@pytest.mark.parametrize("row", reference.REFERENCE_ROWS)
@pytest.mark.parametrize("player", list(reference.PLAYERS))
def test_a_reference_row_plays_the_trained_rows_harness(tmp_path, monkeypatch, row, player):
    harness, learner = _played(monkeypatch, reference.play_round, player, row, tmp_path / "ref", ONE_USER, api_key="k")
    trained, _ = _played(monkeypatch, play.play_round, PAPER_ROW[row], tmp_path / "trained", ONE_USER,
                         base_url="http://127.0.0.1:8101/v1", served_name="m")
    assert {**harness, "model": None} == {**trained, "model": None, **reference.REFERENCE_ROWS[row]}
    assert learner["upstream"] == reference.OPENROUTER and learner["api_key"] == "k"
    assert learner["body_overrides"]["model"] == reference.PLAYERS[player]["model"]
    assert learner["declared_context_length"] == play.CONTEXT_WINDOW
    assert learner["role_map"] == {"developer": "system"} and learner["body_drop"] == ("reasoning_effort",)


def test_row_hpl_uses_shared_expert_guidance_and_keeps_its_memory(tmp_path, monkeypatch):
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    harness, _ = _played(monkeypatch, reference.play_round, "qwen", "hpl", tmp_path / "ref", None, api_key="k")
    system = PrimeAgentLearner(**harness)
    assert system.program_library and system.end == "finish" and not system.stateless
    assert system.rules_page == ""
    assert json.loads((tmp_path / "ref" / "driver.json").read_text())["guidance"] == "expert"


def test_rows_ha_and_hpl0_are_the_ablations(tmp_path, monkeypatch):
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    ha, _ = _played(monkeypatch, reference.play_round, "qwen", "ha", tmp_path / "ha", ONE_USER, api_key="k")
    hpl0, _ = _played(monkeypatch, reference.play_round, "qwen", "hpl0", tmp_path / "hpl0", ONE_USER, api_key="k")
    asked, wiped = PrimeAgentLearner(**ha), PrimeAgentLearner(**hpl0)
    assert asked.ask_to_store and not asked.stateless and not asked.program_library
    assert wiped.program_library and wiped.stateless and not wiped.ask_to_store
    assert json.loads((tmp_path / "hpl0" / "driver.json").read_text())["stateless"]


def _bare_played(monkeypatch, player, row, run_dir, schedule):
    """play_bare_round with the collective learner, the runner and the env faked: the systems it made."""
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
    reference.play_bare_round(player, row, run_dir, schedule, api_key="k")
    return made


@pytest.mark.parametrize("row,learner,stateless", [("f", "BareModelLearner", False),
                                                  ("dc", "BareDemoCacheLearner", False),
                                                  ("m", "BareProgramLibraryLearner", False),
                                                  ("m0", "BareProgramLibraryLearner", True)])
@pytest.mark.parametrize("player", reference.PLAYERS)
def test_a_bare_row_is_its_floor_at_the_players_settings(player, row, learner, stateless, tmp_path, monkeypatch):
    (system,) = _bare_played(monkeypatch, player, row, tmp_path / "play", SimpleNamespace(users=1, rounds=lambda: [0]))
    assert type(system).__name__ == learner and system.stateless == stateless
    assert (system.model, system.max_tokens, system.end, system.empty_reply, system.archive_wipes) == \
        (reference.PLAYERS[player]["model"], play.MAX_TOKENS, "finish", "reprompt", True)
    driver = json.loads((tmp_path / "play" / "driver.json").read_text())
    assert driver["row"] == row and driver["learner"].endswith(f":{learner}")


def test_row_m_takes_up_the_programs_and_pairs_its_one_state_holds(tmp_path):
    """Every round is a process of its own: row m's programs and pairs reach the next round's learner through the one
    state the row's users share, to which a round's end saves them and from which the next learner reads them back as
    it is attached."""
    from continual_arc.tasks import Pair
    from continual_arc_baselines.systems.bare import BareProgramLibraryLearner
    system = tmp_path / "play" / "users" / "u000" / "system"
    first = BareProgramLibraryLearner("m", **play.FLOOR)
    first.attach(system, {}, log_state=True)   # as the collective learner attaches the system that keeps the state
    first.library = {"t1": "def transform(grid):\n    return grid\n"}
    first.cache = {"t1": [Pair.from_json({"input": [[1]], "output": [[2]]})]}
    first.save_state()
    second = BareProgramLibraryLearner("m", **play.FLOOR)
    second.attach(system, {}, log_state=True)
    assert second.library == first.library
    assert {t: [p.to_json() for p in ps] for t, ps in second.cache.items()} == \
        {t: [p.to_json() for p in ps] for t, ps in first.cache.items()}


@pytest.mark.parametrize("player", reference.PLAYERS)
def test_a_bare_rows_users_play_behind_proxies_with_the_players_settings_and_no_cache_of_their_own(player, tmp_path,
                                                                                                   monkeypatch):
    """Row dc: every user's bare model is the demo-cache floor (play.FLOOR, the player's reply cap), the player's
    request settings its proxy's defaults, with no hook of the row's own: the team's cache is the users' one state, as
    every row's memory is (the baselines' CollectiveHarnessLearner)."""
    import continual_arc.runner as runner
    import continual_arc.tasks as tasks
    import continual_arc_baselines.collective as collective
    from continual_arc_baselines.systems.bare import BareDemoCacheLearner
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
    reference.play_bare_round(player, "dc", tmp_path / "play", None, api_key="k")
    spec = reference.PLAYERS[player]
    first, second = seen["users"]
    assert all(type(user) is BareDemoCacheLearner for user in (first, second)) and first is not second
    assert (first.model, first.max_tokens, first.end, first.empty_reply, first.archive_wipes) == \
        (spec["model"], play.MAX_TOKENS, "finish", "reprompt", True)
    assert seen["upstream"] == reference.OPENROUTER and seen["api_key"] == "k"
    assert seen["body_defaults"] == spec["body"] and not seen["capture"]
    assert not {"round_hook", "reset_hook"} & set(seen)
    assert not (tmp_path / "play" / "demo-cache").exists()


def test_the_floor_plays_the_one_user_rows_stream():
    from continual_arc.configs import build_schedule, get_config
    canonical = build_schedule(get_config(play.STREAM_CONFIG), play.SEED)
    rows = play.full_schedule(play.SEED, 1)
    assert [(s.user, s.phase, s.task_id, s.instance_seed) for s in canonical.slots] == \
        [(s.user, s.phase, s.task_id, s.instance_seed) for s in rows.slots]


def test_every_round_not_yet_played_is_played_in_turn(tmp_path, monkeypatch):
    played = []
    monkeypatch.setattr(reference, "play_round", lambda player, row, run_dir, schedule, *, api_key: played.append(
        (player, row, schedule.rounds()[-1], schedule.users, api_key)) or SimpleNamespace(records=[]))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    (tmp_path / "records.jsonl").write_text("")
    reference.cmd_play(SimpleNamespace(run_dir=str(tmp_path), users=2, rounds=3, player="qwen", row="h", seed=0))
    assert played == [("qwen", "h", r, 2, "k") for r in range(3)]


def test_a_dc_run_that_kept_its_cache_in_demo_cache_cannot_go_on(tmp_path):
    """Row dc kept its team's cache in the run's demo-cache/ before its cache was its users' one state, which holds
    none of it: such a run is refused rather than played on from an empty cache."""
    (tmp_path / "play" / "demo-cache").mkdir(parents=True)
    with pytest.raises(SystemExit, match="demo-cache"):
        reference.play_bare_round("qwen", "dc", tmp_path / "play", None, api_key="k")


@pytest.mark.parametrize("row", ["hpl", "hpl0"])
@pytest.mark.parametrize("saved", [{"told": "c1"}, {"expert_guidance_sha256": "another policy"}])
def test_guided_reference_refuses_old_policy_before_overwriting_driver(tmp_path, monkeypatch, row, saved):
    run = tmp_path / "ref"
    run.mkdir()
    driver = run / "driver.json"
    driver.write_text(json.dumps(saved))
    (run / "records.jsonl").write_text('{"step":0}\n')
    original = driver.read_text()
    with pytest.raises(ValueError, match="original code revision"):
        _played(monkeypatch, reference.play_round, "qwen", row, run, ONE_USER, api_key="k")
    assert driver.read_text() == original


def test_guided_reference_resumes_with_identical_recorded_guidance(tmp_path, monkeypatch):
    from continual_arc_baselines.systems.guidance import EXPERT_GUIDANCE_SHA256

    run = tmp_path / "ref"
    _played(monkeypatch, reference.play_round, "qwen", "hpl", run, ONE_USER, api_key="k")
    assert json.loads((run / "driver.json").read_text())["expert_guidance_sha256"] == EXPERT_GUIDANCE_SHA256
    (run / "records.jsonl").write_text('{"step":0}\n')
    _played(monkeypatch, reference.play_round, "qwen", "hpl", run, ONE_USER, api_key="k")
    assert json.loads((run / "driver.json").read_text())["expert_guidance_sha256"] == EXPERT_GUIDANCE_SHA256
