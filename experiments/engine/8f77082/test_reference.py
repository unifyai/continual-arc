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
    assert learner["declared_context_length"] == reference.CONTEXT_WINDOW
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
def test_floor_is_the_paper_floor_at_the_players_settings(player, tmp_path):
    from continual_arc_baselines.cli import build_parser
    args = build_parser().parse_args(reference.floor_argv(player, tmp_path / "play"))
    spec = reference.PLAYERS[player]
    assert args.learner == "continual_arc_baselines.systems.bare:BareDemoCacheLearner"
    assert json.loads(args.learner_kwargs) == {"model": spec["model"], "max_tokens": reference.MAX_TOKENS,
                                               "end": "finish", "empty_reply": "reprompt", "archive_wipes": True}
    assert json.loads(args.llm_defaults) == spec["body"]
    assert (args.config, args.mode, args.seed, args.out) == ("arc1-small", "workstreams", 0, str(tmp_path / "play"))
    assert args.resume and args.events


def test_floor_plays_the_one_user_rows_stream():
    from continual_arc.configs import build_schedule, get_config
    canonical = build_schedule(get_config(paper.STREAM_CONFIG), "workstreams", paper.SEED)
    rows = paper.full_schedule(paper.SEED, 1)
    assert [(s.user, s.phase, s.task_id, s.instance_seed) for s in canonical.slots] == \
        [(s.user, s.phase, s.task_id, s.instance_seed) for s in rows.slots]
