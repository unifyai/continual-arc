"""A vLLM general plugin (entry point group ``vllm.general_plugins``): a ``/v1/completions`` request whose
``vllm_xargs`` carries ``read_prefix_cache: 1`` gets ``SamplingParams.skip_reading_prefix_cache`` set to ``False``,
overriding vLLM's own default of ``True`` for any request with ``prompt_logprobs`` set (``sampling_params.py``'s
``__post_init__``, as of vLLM 0.30.0: "the output of prompt logprobs may be less than n_prompt_tokens"). A request
without that key is untouched.

Gated on ``TEACHER_READ_PREFIX_CACHE=1`` in its own process's environment, not on vLLM's ``VLLM_PLUGINS``: that is
an *allowlist*, and the teacher's venv also ships the LoRA filesystem/hub resolver plugins (shared with the
student's server, ``serve.sh``), which naming only this plugin there would silently stop loading. The plugin stays
installed in every process ``vllm.general_plugins`` loads into -- the API server, each engine-core process and
each worker (``vllm/plugins/__init__.py``'s ``load_general_plugins``) -- but it only ever patches a method the API
server calls, so patching it in a process that never calls it is harmless, just not literally a no-op there.

Patches after construction, not before: ``SamplingParams.skip_reading_prefix_cache`` is resolved from ``None`` to
``True`` in its own ``__post_init__``, which msgspec runs at construction time, before ``to_sampling_params``
returns -- there is no hook earlier than that. Plain attribute assignment on the returned object is enough,
because ``SamplingParams`` is a mutable ``msgspec.Struct`` (``dict=True``, not frozen), and the override survives
the msgpack round trip to the engine core: ``omit_defaults=True`` only omits a field equal to its *declared*
default (``None`` here), and ``False != None``, so the override is sent explicitly and the engine core's own
``__post_init__`` on decode leaves it alone (it only overwrites a still-``None`` value).

Installed into the teacher's vLLM environment by ``serve_teacher.sh`` (``uv pip install -e`` this directory, into
that environment specifically); ``TEACHER_PREFIX_CACHE=off`` (the default) never sets ``TEACHER_READ_PREFIX_CACHE``,
so even once installed the plugin stays inert there, and a request without the key is untouched either way.
"""
from __future__ import annotations

import os

_PATCHED_ATTR = "_teacher_read_prefix_cache_patched"


def register() -> None:
    if os.environ.get("TEACHER_READ_PREFIX_CACHE") != "1":
        return

    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    if getattr(CompletionRequest.to_sampling_params, _PATCHED_ATTR, False):
        return  # a process that calls load_general_plugins more than once (vLLM's own plugins are built for this)
    original = CompletionRequest.to_sampling_params

    def to_sampling_params(self, *args, **kwargs):
        sampling_params = original(self, *args, **kwargs)
        if self.vllm_xargs and self.vllm_xargs.get("read_prefix_cache"):
            sampling_params.skip_reading_prefix_cache = False
        return sampling_params

    setattr(to_sampling_params, _PATCHED_ATTR, True)
    CompletionRequest.to_sampling_params = to_sampling_params
