"""glm_val.py's settings off any machine: each arm sends GLM-5.3-flash on Z.ai alone its reasoning setting, with the
reply cap and window the play promises and the run's budget and harness passed through. Run with
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
