"""reference.py off any network: each player's request settings, and that a reference row plays under exactly the
harness settings paper.play_round gives the trained rows (only the model and its serving differ), row hpl under row h's
with baselines' program library added.

Run with ``python3 -m pytest test_reference.py -q``; needs ``continual_arc`` and ``continual_arc_baselines`` on the
path. carc_stream.team, whose vLLM constants paper.play_round reads (none of them a harness setting), is stubbed, since
it needs prime-rl's environment.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import loop
import paper
import reference


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

        def close(self):
            pass

    monkeypatch.setattr(prime_agent, "PrimeAgentLearner", lambda **kwargs: seen.setdefault("harness", kwargs))
    monkeypatch.setattr(collective, "CollectiveHarnessLearner", Learner)
    monkeypatch.setattr(runner, "build_env", lambda *a, **k: None)
    monkeypatch.setattr(runner, "run", lambda *a, **k: SimpleNamespace(records=[], summary={"protocol": {}}))
    monkeypatch.setattr(tasks.TaskRegistry, "default", staticmethod(lambda: None))
    for name in ("_refresh_meta_protocol", "_report_round_memory", "_wipe_stateless_users"):
        monkeypatch.setattr(paper, name, lambda *a: None)
    team = SimpleNamespace(THINKING_FIELD="reasoning_content", VLLM_DROP=(), VLLM_ROLES={})
    monkeypatch.setitem(sys.modules, "carc_stream", SimpleNamespace(team=team))
    monkeypatch.setitem(sys.modules, "carc_stream.team", team)
    fn(*args, **kw)
    return seen["harness"], seen["learner"]


@pytest.mark.parametrize("row", reference.REFERENCE_ROWS)
@pytest.mark.parametrize("player", list(reference.PLAYERS))
def test_a_reference_row_plays_the_trained_rows_harness(tmp_path, monkeypatch, row, player):
    harness, learner = _played(monkeypatch, reference.play_round, player, row, tmp_path / "ref", None, api_key="k")
    trained, _ = _played(monkeypatch, paper.play_round, "h" if row == "hpl" else row, tmp_path / "paper", None,
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


@pytest.mark.parametrize("at,keep", [(49_152, 8_000), (65_536, 8_000), (0, 0)])
def test_a_compaction_probe_changes_the_compaction_point_alone(tmp_path, monkeypatch, at, keep):
    probe, _ = _played(monkeypatch, reference.play_round, "qwen", "h", tmp_path / "probe", None, api_key="k",
                       harness=loop.harness_of(at, keep))
    rows, _ = _played(monkeypatch, reference.play_round, "qwen", "h", tmp_path / "row", None, api_key="k")
    assert (probe["compact_at"], probe["compact_keep"]) == (at or None, keep or None)
    assert {**probe, "compact_at": 0, "compact_keep": 0} == {**rows, "compact_at": 0, "compact_keep": 0}
    assert loop.harness_of(rows["compact_at"], rows["compact_keep"]) == loop.HARNESS


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
    reference.play_floor_round(player, tmp_path / "play", None, api_key="k", rounds_done=1)
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
