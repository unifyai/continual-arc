"""The model server as the rows talk to it (continual_arc_training.server): the client's retries, the adapter copy vLLM
loads, and the check that a served name answers from its adapter's weights."""
from __future__ import annotations

import io
import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

from continual_arc_training import frozen, server
from continual_arc_training.server import (
    SERVED_LOGPROB_GAP,
    SERVED_TOP_LOGPROBS,
    Server,
    check_served_adapter,
    served_adapter,
)


def _http(code):
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "error", {}, io.BytesIO(b"body"))


def test_chat_retries_a_server_error_loads_a_forgotten_adapter_again_and_gives_up_on_a_client_error(monkeypatch):
    s = Server("http://x/v1")
    s.adapters["run-p0001"] = Path("/adapters/p0001")
    answers, paths = [_http(503), _http(404), {"choices": [{"message": {"content": "hi"}}]}], []

    def post(path, payload, timeout=1800):
        paths.append(path)
        if path == "/v1/load_lora_adapter":
            assert payload == {"lora_name": "run-p0001", "lora_path": "/adapters/p0001"}
            return {}
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(s, "_post", post)
    assert s.chat({"model": "run-p0001", "messages": []}, backoff=0) == {"choices": [{"message": {"content": "hi"}}]}
    assert paths == ["/v1/chat/completions"] * 2 + ["/v1/load_lora_adapter", "/v1/chat/completions"]
    answers[:] = [_http(400), "not json", _http(404)]
    assert s.chat({"model": "base"}, backoff=0) is None                 # a client error is final
    assert s.chat({"model": "base"}, backoff=0) is None                 # an answer that is no JSON object
    assert s.chat({"model": "base"}, backoff=0) is None                 # a 404 for a name this client never loaded
    answers[:] = [ConnectionResetError("reset")] * 3
    assert s.chat({"model": "base"}, retries=2, backoff=0) is None and answers == []


def test_unloading_an_adapter_the_server_no_longer_holds_is_no_error(monkeypatch):
    s = Server("http://x/v1")
    s.adapters["a"] = Path("/a")

    def post(path, payload, timeout=1800):
        raise _http(404)

    monkeypatch.setattr(s, "_post", post)
    s.unload_adapter("a")
    assert s.adapters == {}
    assert s.root == "http://x"


def _fake_hub(monkeypatch, tmp_path, architectures):
    """huggingface_hub's download answering every model's config with ``architectures``."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architectures": architectures}))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda model, name: str(config)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_an_adapter_for_a_model_served_as_multimodal_is_loaded_from_a_copy_under_the_checkpoints_names(tmp_path,
                                                                                                        monkeypatch):
    """vLLM reads Qwen3.5 through its multimodal class, so an adapter PEFT saved from the text-only model is served from
    a copy whose weights are renamed from model.layers to model.language_model.layers, one for one, the tensors and the
    config unchanged and the trainer's own files left as they were; another architecture's adapter is served as
    saved."""
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    src = tmp_path / "adapter"
    src.mkdir()
    weights = {f"{server.PEFT_LAYERS}{layer}.{module}.lora_{ab}.weight": torch.randn(4, 4)
               for layer, module in ((0, "mlp.down_proj"), (3, "self_attn.q_proj")) for ab in "AB"}
    st.save_file(weights, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({"r": 4, "target_modules": ["q_proj", "down_proj"]}))
    _fake_hub(monkeypatch, tmp_path, ["Qwen3_5ForConditionalGeneration"])
    served = served_adapter("Qwen/Qwen3.5-9B", src)
    assert served == tmp_path / "adapter.vllm"
    renamed = st.load_file(str(served / "adapter_model.safetensors"))
    assert sorted(renamed) == sorted(server.SERVED_LAYERS + k[len(server.PEFT_LAYERS):] for k in weights)
    assert all(torch.equal(renamed[server.SERVED_LAYERS + k[len(server.PEFT_LAYERS):]], v) for k, v in weights.items())
    assert (served / "adapter_config.json").read_text() == (src / "adapter_config.json").read_text()
    assert sorted(st.load_file(str(src / "adapter_model.safetensors"))) == sorted(weights)
    _fake_hub(monkeypatch, tmp_path, ["Qwen3ForCausalLM"])
    assert served_adapter("Qwen/Qwen3-8B", src) == src


def test_the_multimodal_class_is_read_from_the_checkpoints_architecture(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, tmp_path, ["Qwen3_5ForConditionalGeneration"])
    assert server.served_as_multimodal("any/name")
    _fake_hub(monkeypatch, tmp_path, [])
    assert not server.served_as_multimodal("Qwen/Qwen3.5-9B")


# -- the served-adapter check ------------------------------------------------------------------------------------

def _reply(tokens, shift=0.0):
    """vLLM's chat ``logprobs.content`` for ``tokens``: each with its top candidates, the chosen one first; ``shift``
    moves every other candidate's log-probability, as a small weight update would."""
    return [{"token": t, "logprob": -0.001,
             "top_logprobs": [{"token": t, "logprob": -0.001}, {"token": "x", "logprob": -7.0 + shift}]}
            for t in tokens]


class FakeServer:
    """vLLM's chat answer as Qwen3.5 gives it: the reply is all reasoning at 16 tokens, so ``content`` is None and only
    ``logprobs`` carries the tokens."""

    def __init__(self, replies):
        self.replies, self.requests = replies, []

    def chat(self, payload):
        self.requests.append(payload)
        return {"choices": [{"message": {"content": None, "reasoning": "Thinking..."},
                             "logprobs": {"content": self.replies[payload["model"]]}}]}


def test_a_served_name_that_answers_exactly_like_the_base_fails_the_check():
    s = FakeServer({"base": _reply(["Think", "ing"]), "round5": _reply(["Think", "ing"])})
    assert check_served_adapter(s, "base", "round5") == {"base": "Thinking", "served": "Thinking", "answered": True,
                                                         "logprob_gap_max": 0.0, "differs": False}
    assert [r["model"] for r in s.requests] == ["base", "round5"]
    assert all(r["temperature"] == 0.0 and r["logprobs"] is True and r["top_logprobs"] == SERVED_TOP_LOGPROBS
               and r["max_tokens"] == 16 for r in s.requests)


def test_an_update_that_keeps_the_tokens_but_moves_the_candidates_passes_and_a_smaller_move_does_not():
    s = FakeServer({"base": _reply(["Think", "ing"]), "round5": _reply(["Think", "ing"], shift=2 * SERVED_LOGPROB_GAP)})
    out = check_served_adapter(s, "base", "round5")
    assert out["differs"] is True and out["logprob_gap_max"] == pytest.approx(2 * SERVED_LOGPROB_GAP)
    s = FakeServer({"base": _reply(["Think", "ing"]), "round5": _reply(["Think", "ing"], shift=SERVED_LOGPROB_GAP / 10)})
    assert check_served_adapter(s, "base", "round5")["differs"] is False


def test_other_tokens_pass_and_no_tokens_say_nothing_about_the_weights():
    s = FakeServer({"base": _reply(["give", " up"]), "round5": _reply(["ask", " again"])})
    out = check_served_adapter(s, "base", "round5")
    assert out["differs"] is True and out["served"] == "ask again"
    out = check_served_adapter(FakeServer({"base": _reply(["Think"]), "round5": []}), "base", "round5")
    assert out["answered"] is False and out["differs"] is False


def _write_adapter(path, b_value):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    path.mkdir(parents=True, exist_ok=True)
    st.save_file({"base_model.model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4),
                  "base_model.model.layers.0.q_proj.lora_B.weight": torch.full((4, 2), b_value)},
                 str(path / "adapter_model.safetensors"))
    return path


def test_an_adapter_is_untrained_exactly_when_every_lora_b_is_zero(tmp_path):
    assert server.adapter_is_untrained(_write_adapter(tmp_path / "fresh", 0.0)) is True
    assert server.adapter_is_untrained(_write_adapter(tmp_path / "stepped", 1e-6)) is False


def test_a_frozen_test_skips_the_check_for_an_untrained_adapter_and_refuses_to_play_on_a_failed_one(tmp_path,
                                                                                                    monkeypatch):
    fresh = _write_adapter(tmp_path / "fresh", 0.0)
    monkeypatch.setattr(server, "Server", lambda url: pytest.fail("an untrained adapter must not reach the server"))
    frozen.check_served("http://127.0.0.1:8101/v1", "run-p0000", fresh)
    stepped = _write_adapter(tmp_path / "stepped", 1e-3)
    monkeypatch.setattr(server, "Server", lambda url: FakeServer({"Qwen/Qwen3.5-9B": _reply(["same"]),
                                                                  "run-p0003": _reply(["same"])}))
    with pytest.raises(SystemExit, match="failed the served-adapter check"):
        frozen.check_served("http://127.0.0.1:8101/v1", "run-p0003", stepped)
    monkeypatch.setattr(server, "Server", lambda url: FakeServer({"Qwen/Qwen3.5-9B": _reply(["same"]),
                                                                  "run-p0003": _reply(["other"])}))
    frozen.check_served("http://127.0.0.1:8101/v1", "run-p0003", stepped)
