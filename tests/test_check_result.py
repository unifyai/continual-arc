"""scripts/check_result.py's check that a submission's student and teacher are published checkpoints named exactly."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("check_result", Path(__file__).parents[1] / "scripts" / "check_result.py")
check_result = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_result)


@pytest.mark.parametrize("name", ["openai/gpt-6-luna", "z-ai/glm-5.3-flash", "qwen/qwen3.5-9b:free",
                                  "Qwen/Qwen3.5-9B@0123abc", "Qwen/Qwen3.5-9B@" + "a" * 40])
def test_published_names_pass(name):
    assert check_result.PUBLISHED.fullmatch(name)


@pytest.mark.parametrize("name", ["gpt-6-luna", "my sabotaged qwen", "Qwen/Qwen3.5-9B@main", "Qwen/Qwen3.5-9B@",
                                  "./weights/qwen", "/abs/path/model"])
def test_unpublished_names_fail(name):
    assert not check_result.PUBLISHED.fullmatch(name)
