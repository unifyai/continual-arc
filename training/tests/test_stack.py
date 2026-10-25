"""The path the Gated DeltaNet's backward step takes on a node's software stack (continual_arc_training.stack), and the
record a phase writes. No GPU or torch needed."""
import json

import pytest

from continual_arc_training import stack as software

H100 = {"name": "NVIDIA H100 80GB HBM3", "compute_cap": "9.0"}
# the nodes' environment (docs/settings.md, "The software stack")
NODES = {"torch": "2.13.0+cu130", "triton": "3.7.1", "flash-linear-attention": "0.5.2", "fla-core": None,
         "tilelang": "0.1.12", "transformers": "5.6.2"}


def stack(card=H100, env=None, **versions):
    return {"versions": {**NODES, **versions}, "gpu": card, "env": env or {}}


def test_the_nodes_run_the_step_on_tilelang():
    where, wrong = software.path(stack())
    assert not wrong and "TileLang 0.1.12" in where and "Triton 3.7.1" in where


def test_triton_3_7_1_is_fixed_without_tilelang():
    assert software.path(stack(tilelang=None)) == ("fla on Triton 3.7.1", False)


def test_torch_2_11_with_triton_3_6_and_no_tilelang_is_stopped():
    # torch 2.11.0 pins Triton 3.6.0, inside the broken range: with fla 0.5.2 and no tilelang, training must not start
    where, wrong = software.path(stack(torch="2.11.0", triton="3.6.0", tilelang=None))
    assert wrong and "fla #640" in where


@pytest.mark.parametrize("triton", ["3.4.0", "3.5.1", "3.7.0"])
def test_every_broken_triton_is_caught(triton):
    assert software.path(stack(triton=triton, tilelang=None))[1]


@pytest.mark.parametrize("triton", ["3.3.1", "3.7.1", "3.8.0"])
def test_the_releases_around_the_range_pass(triton):
    assert not software.path(stack(triton=triton, tilelang=None))[1]


def test_tilelang_covers_a_broken_triton_unless_switched_off():
    assert not software.path(stack(triton="3.7.0"))[1]
    assert software.path(stack(triton="3.7.0", env={"FLA_TILELANG": "0"}))[1]
    assert software.path(stack(triton="3.7.0", env={"FLA_DISABLE_BACKEND_DISPATCH": "1"}))[1]


def test_fla_before_0_5_has_no_tilelang_kernel():
    assert software.path(stack(triton="3.5.0", **{"flash-linear-attention": "0.4.2"}))[1]


def test_the_split_fla_package_counts():
    split = stack(triton="3.6.0", tilelang=None, **{"flash-linear-attention": None, "fla-core": "0.5.3"})
    assert software.path(split)[1]


def test_no_fla_runs_transformers_own_loop():
    # torch 2.11.0, Triton 3.6.0 and no fla: the Gated DeltaNet layers run on transformers' own loop
    where, wrong = software.path(stack(torch="2.11.0", triton="3.6.0", tilelang=None,
                                           **{"flash-linear-attention": None}))
    assert not wrong and "PyTorch loop" in where


@pytest.mark.parametrize("card", [{"name": "NVIDIA H200", "compute_cap": "9.0"},
                                  {"name": "NVIDIA H20", "compute_cap": ""},
                                  {"name": "NVIDIA GH200 480GB", "compute_cap": "9.0"}])
def test_hopper_cards(card):
    assert software.hopper(card)
    assert software.path(stack(card=card, triton="3.6.0", tilelang=None))[1]


@pytest.mark.parametrize("card", [{"name": "NVIDIA A100-SXM4-80GB", "compute_cap": "8.0"},
                                  {"name": "NVIDIA B200", "compute_cap": "10.0"}, None])
def test_other_cards_and_no_gpu_pass(card):
    assert not software.hopper(card)
    assert not software.path(stack(card=card, triton="3.6.0", tilelang=None))[1]


def test_release():
    assert software.release("2.13.0+cu130") == (2, 13, 0)
    assert software.release("9.0") == (9, 0, 0)
    assert software.release("3.7.1rc1") == (3, 7, 1)
    assert software.release(None) is None and software.release("") is None


def test_record_writes_the_stack_and_stops_on_wrong_gradients(tmp_path, monkeypatch):
    monkeypatch.setattr(software, "gpu", lambda: H100)
    monkeypatch.setattr(software, "versions", lambda: dict(NODES))
    software.record(tmp_path)
    kept = json.loads((tmp_path / "kernels.json").read_text())
    assert kept["versions"]["triton"] == "3.7.1" and not kept["wrong_gradients"] and kept["gpu"] == H100
    monkeypatch.setattr(software, "versions", lambda: {**NODES, "triton": "3.7.0", "tilelang": None})
    with pytest.raises(SystemExit, match="fla #640"):
        software.record(tmp_path)
    assert json.loads((tmp_path / "kernels.json").read_text())["wrong_gradients"]
