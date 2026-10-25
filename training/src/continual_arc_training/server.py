"""The model server (serve.sh's vLLM, which loads LoRA adapters at run time) as the rows' code talks to it: the client
(:class:`Server`), the copy of an adapter vLLM loads (:func:`served_adapter`), and the check that a name the server
serves an adapter under answers from the adapter's weights (:func:`check_served_adapter`).

vLLM serves Qwen3.5 through its multimodal class, which maps the checkpoint's names (``model.language_model.layers``)
onto its modules, while PEFT on the text-only class the trainer loads writes ``model.layers``; such an adapter's
weights match no module, and vLLM then serves the base model under the adapter's name without a word. So an adapter is
loaded from a copy under the checkpoint's names, and every adapter a round plays on is checked before the round.

Run as ``python -m continual_arc_training.server vllm-adapter --model M PATH`` (serve.sh, frozen.sh, from vLLM's own
environment), it prints the folder vLLM should load the adapter saved at PATH from."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PEFT_LAYERS, SERVED_LAYERS = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
MULTIMODAL_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration",)
# The check's fixed prompt: one chat turn, nothing from the benchmark itself.
SERVED_ADAPTER_PROMPT = [{"role": "user",
                          "content": "In one short sentence, what should an agent do when it is unsure whether its "
                                     "last answer was correct?"}]
# The check reads the replies' tokens and, at each, the log-probabilities of the top SERVED_TOP_LOGPROBS candidates,
# never the reply's text: Qwen3.5 opens with its reasoning, which vLLM's reasoning parser moves out of ``content``, so 16
# tokens leave ``content`` empty for any weights. When both replies are the same tokens, the largest gap between a
# shared candidate's two log-probabilities must reach SERVED_LOGPROB_GAP: the base model asked twice gave a largest gap
# of exactly 0.0 (one request at a time, the same kernels), and an adapter after 8 small steps the same 16 tokens with
# a largest gap of 0.375, while the chosen tokens' own log-probabilities moved by under 1e-3.
SERVED_TOP_LOGPROBS = 5
SERVED_LOGPROB_GAP = 1e-2


class Server:
    """A vLLM OpenAI-compatible server at ``base_url`` (``.../v1``). Transient failures are retried, and a server
    restarted under the runs, which has forgotten the adapters loaded into it, has an adapter a request names loaded
    again."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.adapters: dict[str, Path] = {}   # every adapter loaded, for loading again after a restart

    def _post(self, path: str, payload: dict[str, Any], timeout: float = 1800) -> Any:
        req = urllib.request.Request(self.root + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        try:
            return json.loads(body)
        except ValueError:
            return body.decode(errors="replace")

    def load_adapter(self, name: str, path: Path) -> None:
        self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)}, timeout=600)
        self.adapters[name] = Path(path)

    def unload_adapter(self, name: str) -> None:
        """The adapter unloaded; one the server does not hold (it restarted) is no error."""
        self.adapters.pop(name, None)
        try:
            self._post("/v1/unload_lora_adapter", {"lora_name": name}, timeout=120)
        except OSError as error:
            log.warning("unloading %s: %s", name, error)

    def models(self) -> list[str]:
        """The names the server serves now: the base model's and every adapter's."""
        with urllib.request.urlopen(self.base + "/models", timeout=60) as resp:
            return [m["id"] for m in json.loads(resp.read())["data"]]

    def chat(self, payload: dict[str, Any], *, retries: int = 30, backoff: float = 20.0) -> dict[str, Any] | None:
        """One chat-completion request whose body is sent as given; the server's JSON answer, or None when the server
        fails for good or does not answer with JSON. Transient failures (5xx, connection resets, a server restart,
        which takes a few minutes) are retried with a flat backoff, ten minutes in all by default. A 404 for an adapter
        this client loaded (a restart) loads it again and retries."""
        model = payload.get("model")
        out: Any = None
        for attempt in range(retries + 1):
            try:
                out = self._post("/v1/chat/completions", payload)
                break
            except urllib.error.HTTPError as error:
                body = error.read()[:200]
                if error.code == 404 and model in self.adapters and attempt < retries:
                    log.warning("adapter %s unknown to the server (restarted?); loading it again", model)
                    try:
                        self.load_adapter(model, self.adapters[model])
                    except OSError as reload_error:
                        log.warning("loading %s again failed: %s", model, reload_error)
                    time.sleep(backoff)
                    continue
                if error.code < 500 or attempt == retries:
                    log.warning("completion failed for %s: HTTP %s %s", model, error.code, body)
                    return None
                log.warning("completion for %s: HTTP %s %s; retry %d in %.0fs", model, error.code, body, attempt + 1,
                            backoff)
            except OSError as error:   # a refused or reset connection, a timeout
                if attempt == retries:
                    log.warning("completion failed for %s: %s", model, error)
                    return None
                log.warning("completion for %s: %s; retry %d in %.0fs", model, error, attempt + 1, backoff)
            time.sleep(backoff)
        if not isinstance(out, dict):
            log.warning("completion returned non-JSON for %s: %r", model, str(out)[:200])
            return None
        return out


def served_as_multimodal(model: str) -> bool:
    """Whether vLLM serves ``model`` (a Hugging Face id) through its multimodal class: the checkpoint's architecture,
    read from its config (the local cache once downloaded), not its name."""
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    return any(a in MULTIMODAL_ARCHITECTURES for a in config.get("architectures") or [])


def served_adapter(model: str, src: Path) -> Path:
    """The folder vLLM should load the adapter saved at ``src`` from: ``src`` itself, or, for a model vLLM serves as
    multimodal, a copy beside it (``src.vllm``) with every weight under the checkpoint's names. The trainer's own files
    are left as they are."""
    if not served_as_multimodal(model):
        return src
    from safetensors.torch import load_file, save_file

    dst = src.with_name(src.name + ".vllm")
    dst.mkdir(exist_ok=True)
    shutil.copy(src / "adapter_config.json", dst)
    weights = load_file(str(src / "adapter_model.safetensors"))
    if not all(k.startswith(PEFT_LAYERS) for k in weights):
        raise ValueError(f"{src}: an adapter weight outside the language model's layers")
    save_file({SERVED_LAYERS + k[len(PEFT_LAYERS):]: v for k, v in weights.items()}, str(dst / "adapter_model.safetensors"))
    return dst


def adapter_is_untrained(adapter_dir: Path) -> bool:
    """Whether every ``lora_B`` matrix of the PEFT adapter in ``adapter_dir`` is exactly zero: PEFT starts B at zero, so
    an adapter no optimizer step has touched adds nothing and is the base model's own weights, which the served-adapter
    check cannot tell from a base model served under the adapter's name."""
    from safetensors import safe_open

    with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), "pt") as st:
        return all(not st.get_tensor(k).any() for k in st.keys() if "lora_B" in k)


def check_served_adapter(server: Any, base_model: str, served_name: str, max_tokens: int = 16) -> dict[str, Any]:
    """:data:`SERVED_ADAPTER_PROMPT` sent at temperature 0 to ``base_model`` and to ``served_name``: ``differs`` is false
    exactly when the server answered both from the same weights (the module docstring). Different tokens differ
    outright; the same tokens differ when a candidate both replies list moved by at least :data:`SERVED_LOGPROB_GAP`
    (``logprob_gap_max``, the largest such move). ``answered`` is false when either reply came back with no tokens (a
    failed request), which says nothing about the weights. ``server`` is a :class:`Server`, or anything with its
    ``chat``."""

    def reply(model: str) -> tuple[list[str], dict[tuple[int, str], float]]:
        out = server.chat({"model": model, "messages": SERVED_ADAPTER_PROMPT, "max_tokens": max_tokens,
                           "temperature": 0.0, "logprobs": True, "top_logprobs": SERVED_TOP_LOGPROBS})
        choice = ((out or {}).get("choices") or [{}])[0]
        content = ((choice.get("logprobs") or {}).get("content")) or []
        tops = {(i, c["token"]): float(c["logprob"]) for i, t in enumerate(content) for c in t.get("top_logprobs") or []}
        return [t["token"] for t in content], tops

    (base_tokens, base_tops), (served_tokens, served_tops) = reply(base_model), reply(served_name)
    out: dict[str, Any] = {"base": "".join(base_tokens), "served": "".join(served_tokens),
                           "answered": bool(base_tokens) and bool(served_tokens)}
    if not out["answered"]:
        return {**out, "differs": False}
    if base_tokens != served_tokens:
        return {**out, "differs": True}
    gaps = [abs(base_tops[k] - served_tops[k]) for k in base_tops.keys() & served_tops.keys()]
    out["logprob_gap_max"] = max(gaps) if gaps else None
    return {**out, "differs": bool(gaps) and max(gaps) >= SERVED_LOGPROB_GAP}


def main() -> None:
    p = argparse.ArgumentParser(description="the folder vLLM should load a saved adapter from")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vllm-adapter")
    v.add_argument("adapter")
    v.add_argument("--model", required=True)
    a = p.parse_args()
    print(served_adapter(a.model, Path(a.adapter).expanduser()))


if __name__ == "__main__":
    main()
