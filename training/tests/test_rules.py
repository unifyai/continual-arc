"""One expert guidance for teachers and harnesses; custom files replace it in full."""
import pytest
from continual_arc_baselines.systems.guidance import EXPERT_GUIDANCE

from continual_arc_training import oracle, realtime


def test_teacher_guidance_is_the_harness_guidance():
    assert oracle.guidance() == EXPERT_GUIDANCE


def test_custom_guidance_replaces_instead_of_extending_the_default(tmp_path):
    path = tmp_path / "custom.md"
    path.write_text("  A different policy.\n")
    assert oracle.guidance(path) == "A different policy."
    path.write_text(" \n")
    with pytest.raises(ValueError, match="guidance file is empty"):
        oracle.guidance(path)


def test_labeller_defaults_to_shared_guidance_and_rejects_old_modes():
    args = ["labeller", "--run", "/tmp/run", "--rounds", "1", "--row", "dagger", "--label-model", "test",
            "--relay-port", "18230", "--label-round-budget", "1", "--label-run-budget", "1"]
    assert realtime.parser().parse_args(args).guidance is None
    for old in (["--oracle-rules", "c1"], ["--c1", "rules/c1.md"]):
        with pytest.raises(SystemExit):
            realtime.parser().parse_args(args + old)
