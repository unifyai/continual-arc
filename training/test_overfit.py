"""Checks for overfit.py on real inputs: the benchmark's own stream schedules, the five cards' programs on fresh
Re-ARC pairs (the benchmark's data fetched), and the report's arms.

Run: ``python -m pytest test_overfit.py -q`` (continual_arc and continual_arc_baselines importable).
"""
import sys
from pathlib import Path

import loop
import overfit

HERE = Path(__file__).resolve().parent
SALT = "overfit-test"   # any alias salt: the card must name the task by its alias, never by its id


def test_schedules_are_fresh_instances_of_the_five_tasks_only(tmp_path, monkeypatch):
    from continual_arc.schedule import Schedule
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["overfit.py", "schedule", "--users", "80", "81", "--out", str(out)])
    overfit.main()
    slots = Schedule.load(out).slots
    assert len(slots) == 10 and {s.task_id for s in slots} == set(overfit.TASKS)
    assert len({s.instance_seed for s in slots}) == 10 and len({s.user for s in slots}) == 10
    seeds = {(s["task_id"], s["instance_seed"]) for u in (80, 81) for s in loop.stream_slots(0, user=u)}
    assert {(s.task_id, s.instance_seed) for s in slots} <= seeds


def test_test_users_are_none_the_training_plays_take():
    trained = set(overfit.DATA_USERS) | {u for k in range(1, 8) for u in overfit.opsd_users(k)}
    assert not trained & set(overfit.TEST_USERS)
    assert max(trained | set(overfit.TEST_USERS)) < 100   # the collective configuration's users


def test_every_card_program_solves_fresh_pairs_and_the_card_names_the_alias():
    from continual_arc.aliases import TaskAliases
    from probe import check_program
    cards = overfit.load_cards(HERE / "overfit_cards.json")
    aliases = TaskAliases(salt=SALT, style="hex")
    for task in overfit.TASKS:
        assert all(check_program(cards[task]["program"], overfit.fresh_pairs(task, overfit.TEST_USERS))["ok"])
        text = overfit.card_text(task, cards, aliases)
        assert text.startswith(f"This is task {aliases.alias(task)}. {overfit.RECALL_MARK}")
        assert task not in text and "def transform(grid)" in text


def test_report_groups_the_teacher_plays_by_arm():
    assert overfit.arm_of("ov-teacher-88a10436") == "ov-teacher"
    assert overfit.arm_of("ov-s1-test") == "ov-s1-test"
