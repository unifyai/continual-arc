"""glm_val.py's settings off any machine: each arm sends GLM-5.3-flash on Z.ai alone its reasoning setting, with the
reply cap and window its harness promises and the run's budget and compaction passed through. Run with
``python3 -m pytest search/generalisation/test_glm_val.py -q``."""
import pytest

import glm_val


def test_each_arm_sets_only_its_reasoning():
    budget, harness = {"max_model_calls": 40}, {"end": "finish"}
    seen = {}
    for arm in glm_val.ARMS:
        kwargs, overrides = glm_val.settings(arm, budget, harness)
        assert kwargs["max_tokens"] == 131_072 and kwargs["context_window"] == 262_144
        assert kwargs["max_model_calls"] == 40 and kwargs["end"] == "finish"
        assert overrides["model"] == "z-ai/glm-5.3-flash" and overrides["provider"]["order"] == ["z-ai"]
        assert overrides["max_completion_tokens"] == 131_072   # prime-agent's own request asks for 32,000 at most
        seen[arm] = overrides["reasoning"]
    assert seen == {"minimal": {"effort": "minimal"}, "medium": {"effort": "medium"},
                    "high": {"enabled": True, "effort": "high"}}
    with pytest.raises(SystemExit):
        glm_val.settings("off", budget, harness)


def test_the_tests_harness_is_the_students():
    """The capacity ablation's games: the student's window, and prime-agent's own reply cap, which the request keeps."""
    kwargs, overrides = glm_val.settings("medium", {"max_model_calls": 40}, {"compact_at": 49_152}, "tests")
    assert kwargs["context_window"] == 131_072 and kwargs["max_tokens"] == 32_768 and kwargs["compact_at"] == 49_152
    assert "max_completion_tokens" not in overrides and overrides["reasoning"] == {"effort": "medium"}
