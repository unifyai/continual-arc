"""The kernels Qwen3.5's linear-attention layers train through on this machine, written into a run's ``train/`` folder
(``kernels.json``), and a stop before training on a stack known to compute wrong gradients.

Qwen3.5 puts three Gated DeltaNet (linear attention) layers before every full-attention layer: 24 of Qwen3.5-9B's 32.
transformers runs a Gated DeltaNet layer on flash-linear-attention (fla) whenever fla is installed and a GPU is
present, and on its own PyTorch loop otherwise. On Hopper GPUs (H100, H200, H20), Triton 3.4.0 up to 3.7.0 compiles one
fla kernel wrongly: the gated delta rule's backward step for dq, dk, dw and dg (``chunk_bwd_dqkwg``), whose dk and dg
come out wrong (fla issue #640; fixed in Triton 3.7.1, fla #983). fla 0.5.0 and later send that step to a TileLang
kernel when tilelang is installed, and raise an error instead of running the broken kernel otherwise; fla before 0.5.0
runs it without a word. The nodes' environment (torch 2.13.0, Triton 3.7.1, fla 0.5.2, tilelang 0.1.12) is clear on
both counts.

Nothing here imports torch or starts CUDA (the GPU comes from nvidia-smi), so the trainer's process calls
:func:`record` before its ranks start. ``python -m continual_arc_training.stack`` prints this machine's stack.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

# the distributions whose versions decide the path (fla's kernels are fla-core in releases that split the package)
PACKAGES = ("torch", "triton", "flash-linear-attention", "fla-core", "tilelang", "transformers", "peft")
# fla's switches: FLA_TILELANG=0 turns the TileLang kernel off, FLA_DISABLE_BACKEND_DISPATCH=1 every backend kernel
SWITCHES = ("FLA_TILELANG", "FLA_DISABLE_BACKEND_DISPATCH")
# the Triton releases that compile chunk_bwd_dqkwg wrongly on Hopper: from 3.4.0 up to, not including, 3.7.1
BROKEN_TRITON = ((3, 4, 0), (3, 7, 1))
# the first fla release with the TileLang kernel for that step and the error in place of the broken kernel (fla #827)
FLA_WITH_TILELANG = (0, 5, 0)
NOTE = "docs/settings.md, \"The software stack\""


def release(version: str | None) -> tuple[int, int, int] | None:
    """A version's release numbers, three of them ("2.13.0+cu130" -> (2, 13, 0), "9.0" -> (9, 0, 0)); None for none."""
    m = re.match(r"\d+(?:\.\d+)*", version or "")
    if not m:
        return None
    parts = [int(x) for x in m.group(0).split(".")] + [0, 0]
    return parts[0], parts[1], parts[2]


def versions() -> dict[str, str | None]:
    """The installed version of each of :data:`PACKAGES`, None where it is not installed."""
    found: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def gpu() -> dict[str, str] | None:
    """The first GPU's name and compute capability, from nvidia-smi (a node's GPUs are all one kind); None without one.
    A driver too old to report the capability gives the name alone."""
    for fields in ("name,compute_cap", "name"):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=60, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            name, _, cap = out.splitlines()[0].partition(",")
            return {"name": name.strip(), "compute_cap": cap.strip()}
    return None


def hopper(card: dict[str, str] | None) -> bool:
    """fla's own test (``fla.utils.IS_NVIDIA_HOPPER``): a name with "NVIDIA H" in it, or compute capability 9."""
    if not card:
        return False
    cap = release(card.get("compute_cap"))
    return "NVIDIA H" in card.get("name", "") or (cap is not None and cap[0] == 9)


def path(stack: dict[str, Any]) -> tuple[str, bool]:
    """What the Gated DeltaNet's backward step runs on, given a stack (:func:`describe`), and whether that gives wrong
    gradients (or, with fla 0.5.0 and later, fla's error at the first backward)."""
    v, env = stack["versions"], stack["env"]
    fla = release(v.get("fla-core") or v.get("flash-linear-attention"))
    if not stack["gpu"]:
        return "no GPU: transformers' PyTorch loop", False
    if fla is None:
        return "transformers' PyTorch loop (fla is not installed)", False
    triton = release(v.get("triton"))
    if not hopper(stack["gpu"]):
        return f"fla on Triton {v.get('triton')} (not a Hopper GPU)", False
    tilelang = (v.get("tilelang") is not None and fla >= FLA_WITH_TILELANG and env.get("FLA_TILELANG") != "0"
                and env.get("FLA_DISABLE_BACKEND_DISPATCH") != "1")
    if tilelang and triton is not None and triton >= BROKEN_TRITON[0]:
        return f"fla, with chunk_bwd_dqkwg on TileLang {v['tilelang']} (Triton {v.get('triton')})", False
    if triton is not None and BROKEN_TRITON[0] <= triton < BROKEN_TRITON[1]:
        return (f"fla on Triton {v.get('triton')} on a Hopper GPU without TileLang: chunk_bwd_dqkwg's dk and dg are "
                "wrong (fla #640)"), True
    return f"fla on Triton {v.get('triton')}", False


def describe() -> dict[str, Any]:
    """This machine's stack: the versions, the GPU, fla's switches where set, and the path :func:`path` finds."""
    stack: dict[str, Any] = {"versions": versions(), "gpu": gpu(),
                             "env": {k: os.environ[k] for k in SWITCHES if k in os.environ}}
    stack["path"], stack["wrong_gradients"] = path(stack)
    return stack


def record(out: Path) -> dict[str, Any]:
    """Write this machine's stack to ``out/kernels.json``, print its path, and stop before training (SystemExit) on a
    stack whose gradients are wrong."""
    stack = describe()
    (out / "kernels.json").write_text(json.dumps(stack, indent=1))
    print(f"kernels: {stack['path']}", flush=True)
    if stack["wrong_gradients"]:
        raise SystemExit(f"not training: {stack['path']}. Install Triton 3.7.1 or later (torch 2.13 or later) or "
                         f"tilelang with fla 0.5.0 or later; {NOTE} says why")
    return stack


if __name__ == "__main__":
    print(json.dumps(describe(), indent=1))
