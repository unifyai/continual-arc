"""plot_rows.py on a tiny fake run folder: no GPU, no served model, no real paper run. ``continual_arc`` must be
on the path (as every other test here assumes, test_dev_loop.py's own docstring).

Run with ``python3 -m pytest test_plot_rows.py -q``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import plot_rows

pytest.importorskip("matplotlib")

_RECORD_DEFAULTS = {"step": 0, "phase": "stream", "source": "fake", "gap": None, "instance_seed": 0,
                   "reset_before": False, "wrong_attempts": 0, "invalid_submissions": 0, "demo_requests": 0,
                   "gave_up": False}


def _record(round_: int, user: int, task_id: str, cost: float, solved: bool, user_exposure_index: int = 0,
           exposure_index: int = 0) -> dict:
    return {**_RECORD_DEFAULTS, "round": round_, "user": user, "task_id": task_id, "cost": cost, "solved": solved,
           "user_exposure_index": user_exposure_index, "exposure_index": exposure_index}


def _write_records(play_dir: Path, records: list[dict]) -> None:
    play_dir.mkdir(parents=True, exist_ok=True)
    with (play_dir / "records.jsonl").open("w") as fp:
        for r in records:
            fp.write(json.dumps(r) + "\n")


def _write_memory_json(play_dir: Path, round_: int, added: int, changed: int, removed: int, reads: int,
                       reused_program: int, solved_without_demo: int = 0, scratch_added: int = 0) -> None:
    path = play_dir / "rounds" / f"r{round_:04d}"
    path.mkdir(parents=True, exist_ok=True)
    total = {"memory": {"skills": {"added": added, "changed": changed, "removed": removed},
                       "harness_store": {"added": 0, "changed": 0, "removed": 0},
                       "notes": {"added": 0, "changed": 0, "removed": 0},
                       "programs": {"added": 0, "changed": 0, "removed": 0},
                       "workspace_other": {"added": scratch_added, "changed": 0, "removed": 0},
                       "other": {"added": 0, "changed": 0, "removed": 0}},
            "reads": {"programs": reads, "skills": 0, "harness_store": 0, "notes": 0},
            "reused_program": reused_program, "solved_without_demo": solved_without_demo}
    (path / "memory.json").write_text(json.dumps({"round": round_, "users": {}, "total": total}))


def _write_phase_json(train_dir: Path, round_: int, steps: list[dict], stats_key: str, stats: dict) -> None:
    path = train_dir / f"r{round_}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "phase.json").write_text(json.dumps({"objective": stats_key, stats_key: stats, "steps": steps}))


def test_per_round_cost_and_solve_rate_groups_by_round():
    from continual_arc.runner import prior_records

    records = [_record(0, 0, "A", 2.0, True), _record(0, 1, "A", 4.0, False), _record(1, 0, "A", 1.0, True)]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        play_dir = Path(d) / "play"
        _write_records(play_dir, records)
        loaded = prior_records(play_dir)
    out = plot_rows.per_round_cost_and_solve_rate(loaded)
    assert out[0]["cost"] == pytest.approx(3.0) and out[0]["solve_rate"] == pytest.approx(0.5)
    assert out[1]["cost"] == pytest.approx(1.0) and out[1]["solve_rate"] == pytest.approx(1.0)


def test_per_round_cost_and_solve_rate_filters_out_the_epilogue_phase_like_cost_by_user_exposure_does():
    """F11: the per-round figure and the exposure figure beside it must be computed over the same record set.
    cost_by_user_exposure already filters through continual_arc.metrics' own _stream_records (episode_type
    "benchmark", phase != epilogue); an epilogue-phase record must not pull a round's mean cost here either."""
    from continual_arc.runner import prior_records

    stream = _record(0, 0, "A", 2.0, True)
    epilogue = {**_record(0, 1, "A", 100.0, False), "phase": "epilogue"}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        play_dir = Path(d) / "play"
        _write_records(play_dir, [stream, epilogue])
        loaded = prior_records(play_dir)
    out = plot_rows.per_round_cost_and_solve_rate(loaded)
    assert out[0]["cost"] == pytest.approx(2.0)   # the epilogue record (cost 100) never reaches the mean
    assert out[0]["solve_rate"] == pytest.approx(1.0)


def test_cost_by_user_exposure_uses_the_benchmarks_own_metrics_function():
    from continual_arc.metrics import cost_by_user_exposure as real_cost_by_user_exposure
    from continual_arc.runner import prior_records

    records = [_record(0, 0, "A", 2.0, True, user_exposure_index=0), _record(1, 0, "A", 1.0, True,
              user_exposure_index=1)]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        play_dir = Path(d) / "play"
        _write_records(play_dir, records)
        loaded = prior_records(play_dir)
    assert plot_rows.cost_by_user_exposure(loaded) == real_cost_by_user_exposure(loaded)


def test_memory_series_reads_every_rounds_memory_json(tmp_path):
    play_dir = tmp_path / "play"
    _write_memory_json(play_dir, 0, added=2, changed=0, removed=0, reads=1, reused_program=0)
    _write_memory_json(play_dir, 1, added=0, changed=1, removed=0, reads=3, reused_program=2,
                       solved_without_demo=1)
    series = plot_rows.memory_series(play_dir)
    assert series[0] == {"added": 2, "changed": 0, "removed": 0, "scratch_churn": 0, "reads": 1,
                         "reused_program": 0, "solved_without_demo": 0}
    assert series[1]["changed"] == 1 and series[1]["reads"] == 3 and series[1]["reused_program"] == 2
    assert series[1]["solved_without_demo"] == 1


def test_memory_series_reports_scratch_separately_from_the_deliberate_areas(tmp_path):
    """workspace_other is scratch: it must land in scratch_churn, never added/changed/removed."""
    play_dir = tmp_path / "play"
    _write_memory_json(play_dir, 0, added=1, changed=0, removed=0, reads=0, reused_program=0, scratch_added=50)
    series = plot_rows.memory_series(play_dir)
    assert series[0]["added"] == 1 and series[0]["scratch_churn"] == 50


def test_memory_series_skips_a_round_with_no_memory_json_yet(tmp_path):
    play_dir = tmp_path / "play"
    (play_dir / "rounds" / "r0000").mkdir(parents=True)   # the round folder exists, but play hasn't finished it
    assert plot_rows.memory_series(play_dir) == {}


def test_memory_series_on_a_run_with_no_rounds_at_all_is_empty(tmp_path):
    assert plot_rows.memory_series(tmp_path / "never-played") == {}


def test_training_series_reads_the_kept_item_count_for_pg_and_opd_alike(tmp_path):
    train_dir = tmp_path / "train"
    _write_phase_json(train_dir, 0, steps=[{"clip_fraction": 0.1, "approx_kl": 0.01, "entropy_mean": 1.5}],
                     stats_key="games", stats={"episodes": 3, "items": 40})
    _write_phase_json(train_dir, 1, steps=[{"clip_fraction": 0.2, "approx_kl": 0.02, "entropy_mean": 1.4},
                                          {"clip_fraction": 0.3, "approx_kl": 0.03, "entropy_mean": 1.3}],
                     stats_key="items", stats={"calls": 50, "items": 48})
    series = plot_rows.training_series(train_dir)
    assert series[0] == {"steps": 1, "clip_fraction": 0.1, "approx_kl": 0.01, "entropy": 1.5, "items": 40}
    assert series[1]["steps"] == 2 and series[1]["items"] == 48 and series[1]["clip_fraction"] == 0.3


def test_training_series_on_an_untrained_row_is_empty(tmp_path):
    assert plot_rows.training_series(tmp_path / "no-such-train-dir") == {}


def test_plot_runs_writes_a_png_and_a_csv_per_metric_on_a_tiny_fake_run_folder(tmp_path):
    """The script's own end-to-end smoke test: two rows, one trained and with memory counters, one frozen and
    wiped (no training, nothing persisting -- an empty memory series), on a run folder a fraction of a real one's
    size. Must not raise, and must produce every figure this module promises."""
    h0_dir = tmp_path / "h0" / "play"
    pg_dir = tmp_path / "pg" / "play"
    pg_train_dir = tmp_path / "pg" / "train"

    _write_records(h0_dir, [_record(0, 0, "A", 8.0, False), _record(0, 1, "A", 8.0, False)])
    _write_records(pg_dir, [_record(0, 0, "A", 2.0, True), _record(0, 1, "A", 4.0, False),
                             _record(1, 0, "A", 1.0, True, user_exposure_index=1)])
    _write_memory_json(pg_dir, 0, added=2, changed=0, removed=0, reads=1, reused_program=0)
    _write_memory_json(pg_dir, 1, added=0, changed=1, removed=0, reads=2, reused_program=1,
                       solved_without_demo=1)
    _write_phase_json(pg_train_dir, 0, steps=[{"clip_fraction": 0.1, "approx_kl": 0.01, "entropy_mean": 1.5}],
                     stats_key="games", stats={"episodes": 1, "items": 4})

    out_dir = tmp_path / "plots"
    plot_rows.plot_runs({"h0": h0_dir, "pg": pg_dir}, {"pg": pg_train_dir}, out_dir)

    expected = ["per_round_mean_cost", "per_round_solve_rate", "cost_by_user_exposure",
               "training_steps", "training_clip_fraction", "training_approx_kl", "training_entropy",
               "training_items", "memory_added", "memory_changed", "memory_removed", "memory_scratch_churn",
               "memory_reads", "memory_reused_program", "memory_solved_without_demo"]
    for name in expected:
        assert (out_dir / f"{name}.png").is_file(), f"missing {name}.png"
        assert (out_dir / f"{name}.csv").is_file(), f"missing {name}.csv"

    cost_csv = (out_dir / "per_round_mean_cost.csv").read_text()
    assert "h0" in cost_csv and "pg" in cost_csv

    import csv as csv_module
    with (out_dir / "training_items.csv").open() as fp:
        training_rows = list(csv_module.DictReader(fp))
    assert all(row["row"] == "pg" for row in training_rows)   # h0 never trains: no training_* row for it

    with (out_dir / "memory_added.csv").open() as fp:
        memory_rows = list(csv_module.DictReader(fp))
    assert {row["row"] for row in memory_rows} == {"pg"}   # h0 has no memory.json written in this fake run


def test_plot_runs_tolerates_a_run_with_nothing_played_yet(tmp_path):
    """A row just started (an empty play directory, no records.jsonl, no rounds/) must not crash the whole call:
    every other row's figures still come out."""
    empty_dir = tmp_path / "fresh" / "play"
    h0_dir = tmp_path / "h0" / "play"
    _write_records(h0_dir, [_record(0, 0, "A", 8.0, False)])

    out_dir = tmp_path / "plots"
    plot_rows.plot_runs({"h0": h0_dir, "fresh": empty_dir}, {}, out_dir)
    assert (out_dir / "per_round_mean_cost.png").is_file()
    import csv as csv_module
    with (out_dir / "per_round_mean_cost.csv").open() as fp:
        rows = list(csv_module.DictReader(fp))
    assert {row["row"] for row in rows} == {"h0"}


def test_color_of_is_stable_for_known_and_unknown_rows():
    order = ["h0", "pg", "opd", "h", "opsd", "weird-new-row"]
    assert plot_rows.color_of("h0", order) == plot_rows.ROW_COLORS["h0"]
    first = plot_rows.color_of("weird-new-row", order)
    assert first == plot_rows.color_of("weird-new-row", order)   # the same call twice: the same colour
