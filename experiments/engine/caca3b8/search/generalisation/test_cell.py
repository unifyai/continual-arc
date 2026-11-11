"""cell.py off any GPU: every cell file reads and plans, the manifests bucket.py fetches are well formed, the GPUs split as
frozen.sh and grid-vals.sh take them, stages run in order and resume from their markers or an earlier stage's products,
a run's label sets merge in the order the first cells merged them, and a cell runs on the bundle it names and nowhere
another bundle's trees are unpacked. Run with
``python3 -m pytest search/generalisation/test_cell.py -q``.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import cell

CELLS = sorted((Path(__file__).parent / "cells").glob("*.json"))


@pytest.mark.parametrize("path", CELLS, ids=lambda p: p.stem)
def test_every_cell_plans(path):
    c = cell.load(str(path))
    assert c.check()
    for stage in c.STAGES:
        for line in cell.inputs(c, [stage], done=set(c.STAGES[:c.STAGES.index(stage)])):
            source, dest, *flag = line.split()
            assert flag in ([], ["names"]) and not dest.startswith("/")
            assert source.startswith(("results/runs/gpu-nodes/", "generalisation/"))
            assert flag != ["names"] or source.endswith("/")
        assert c.products(stage)[0] == f"runs/rt/gen-{c.name}/{stage}.done"


def test_a_later_stage_alone_fetches_the_earlier_stages_products():
    c = cell.load(str(Path(__file__).parent / "cells/relab-u4.json"))
    lines = cell.inputs(c, ["play"], done={"label"})
    assert lines[0] == "generalisation/relab-u4/train.tgz gen/relab-u4/train.tgz"
    assert not any("label.tgz" in line for line in lines)
    assert "generalisation/relab-u4/label.tgz gen/relab-u4/label.tgz" in cell.inputs(c, ["play"], done=set())


def test_frozen_tests_take_two_gpus_an_adapter():
    assert cell.frozen_gpus(4) == ["0,1", "2,3", "4,5", "6,7"]
    assert cell.frozen_gpus(1) == ["0,1"]
    for n in (0, 5):
        with pytest.raises(cell.Failed):
            cell.frozen_gpus(n)


class Steps(cell.Cell):
    STAGES = ("first", "second")
    ran: list[str] = []

    def first(self):
        self.ran.append("first")
        (cell.HOME / self.work).mkdir(parents=True, exist_ok=True)
        (cell.HOME / self.work / "made").write_text("by first")

    def second(self):
        self.ran.append("second")
        assert (cell.HOME / self.work / "made").read_text() == "by first"

    def products(self, stage):
        return super().products(stage) + ([f"{self.work}/made"] if stage == "first" else [])


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(cell, "HOME", tmp_path)
    monkeypatch.setattr(cell, "gpus_here", lambda: 8)
    Steps.ran = []
    return tmp_path


def test_stages_run_in_order_and_resume(home):
    c = Steps({"name": "t", "kind": "steps"})
    cell.run(c, ["first", "second"])
    assert Steps.ran == ["first", "second"]
    cell.run(c, ["first", "second"])
    assert Steps.ran == ["first", "second"]   # both done: nothing runs again
    assert "stages first,second done" in (home / "runs/rt/gen-t/progress").read_text()


def test_a_stage_elsewhere_takes_the_earlier_ones_products_from_their_tarball(home, tmp_path_factory):
    there = tmp_path_factory.mktemp("there")
    cell.HOME = there
    c = Steps({"name": "t", "kind": "steps"})
    cell.run(c, ["first"])
    subprocess.run(["tar", "-C", str(there), "-czf", str(there / "first.tgz"), *c.products("first")], check=True)
    cell.HOME = home
    c = Steps({"name": "t", "kind": "steps"})
    with pytest.raises(cell.Failed, match="first has not run here"):
        cell.run(c, ["second"])
    (home / "gen/t").mkdir(parents=True)
    (there / "first.tgz").rename(home / "gen/t/first.tgz")
    cell.run(c, ["second"])
    assert Steps.ran == ["first", "second"] and (home / "runs/rt/gen-t/first.done").exists()


def test_no_upload_session_hands_nothing_over(home):
    c = Steps({"name": "t", "kind": "steps"})
    cell.run(c, ["first"])
    assert not (home / "gen/t/first.tgz").exists() and not (home / "runs/rt/gen-t/first.handed").exists()


def test_plays_only_as_ubuntu_with_a_sandbox(home):
    with pytest.raises(cell.Failed, match="user NODE_USER"):
        cell.ready_to_play()


def test_a_runs_label_sets_merge_round_by_round_dev_before_train(home, monkeypatch):
    c = cell.load(str(Path(__file__).parent / "cells/u4lab-17k.json"))
    for r in ("r0001", "r0000", "r0010"):
        for s in ("train", "dev"):
            (home / c.work / "rounds/play/rounds" / r / "labels/split" / f"{s}-{r}").mkdir(parents=True)
        (home / c.work / "rounds/play/rounds" / r / "labels/split/split.json").write_text("{}")
    seen = []
    monkeypatch.setattr(cell.Cell, "loop_py", lambda self, args, log, env=None: seen.append([str(a) for a in args]))
    c.merge()
    sets = [Path(a).name for a in seen[0][2:-2]]
    assert sets == ["dev-r0000", "train-r0000", "dev-r0001", "train-r0001", "dev-r0010", "train-r0010"]
    assert seen[0][-2:] == ["--out", str(home / c.work / "set")]
    c.merge()
    assert len(seen) == 1   # merged once


def test_cell_files_are_named_after_their_cells(tmp_path):
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps({"name": "smoke", "kind": "recipe"}))
    with pytest.raises(cell.Failed, match="named after its cell"):
        cell.load(str(bad))


def test_a_recipe_takes_a_label_sets_tarball(home):
    c = cell.Recipe({"name": "r", "kind": "recipe", "labels": {"set": "x/labels.tar.gz", "dir": "stage1/set"},
                     "target": 10, "lr": "5e-5", "batch": 16, "epochs": 2, "held": "a", "run": "y/z"})
    assert c.inputs_for("train") == ["results/runs/gpu-nodes/x/labels.tar.gz gen/r/labels.tgz"]
    made = home / "made"
    (made / "stage1/set").mkdir(parents=True)
    (made / "stage1/set/split.json").write_text("{}")
    (home / "gen/r").mkdir(parents=True)
    subprocess.run(["tar", "-C", str(made), "-czf", str(home / "gen/r/labels.tgz"), "stage1"], check=True)
    c.merge()
    assert (c.label_set() / "split.json").exists() and c.label_set() == home / "gen/r/stage1/set"


def test_a_val_or_frozen_cell_takes_the_adapter_a_recipe_cell_trained(home):
    assert cell.adapter_inputs("cell:r", "gen/v/adapters/a") == ["generalisation/r/train.tgz gen/v/adapters/a/train.tgz"]
    made = home / "made"
    (made / "exp/gen-r/model/adapter").mkdir(parents=True)
    (made / "exp/gen-r/model/adapter/adapter_config.json").write_text("{}")
    dest = home / "gen/v/adapters/a"
    dest.mkdir(parents=True)
    subprocess.run(["tar", "-C", str(made), "-czf", str(dest / "train.tgz"), "exp"], check=True)
    got = cell.adapter_dir("cell:r", dest)
    assert got == dest / "exp/gen-r/model/adapter" and (got / "adapter_config.json").exists()
    assert cell.adapter_dir("cell:r", dest) == got   # unpacked once
    assert cell.adapter_dir("node/run/train/p0001/adapter", dest) == dest


def test_a_recipes_weighting_reaches_grid_exp_and_a_bad_one_is_refused(home, monkeypatch):
    spec = {"name": "r", "kind": "recipe", "labels": {"run": "x/y", "rounds": [0, 1]}, "target": 10, "lr": "5e-5",
            "batch": 16, "epochs": 2, "held": "a", "run": "x/y"}
    seen = []
    for extra in ({}, {"weighting": "split"}):
        c = cell.Recipe({**spec, **extra})
        monkeypatch.setattr(c, "sh", lambda cmd, log, env=None: seen.append((env or {}).get("WEIGHTING")))
        c.grid_exp()
    assert seen == [None, "split"]
    assert "weighting split" in cell.Recipe({**spec, "weighting": "split"}).check()[0]
    with pytest.raises(cell.Failed, match="weighting must be one of"):
        cell.Recipe({**spec, "weighting": "turns"}).check()


def test_a_cell_runs_on_its_bundle_and_only_where_that_bundle_is_unpacked(home):
    spec = {"name": "t", "kind": "val", "adapters": {}}
    plain, turns = cell.Val(spec), cell.Val({**spec, "bundle": "code-dagger-rows-turns"})
    assert plain.bundle == "code-dagger-rows" and cell.bundle_object(plain) == "bundles/generalisation/code-dagger-rows.tgz"
    assert cell.bundle_object(turns) == "bundles/generalisation/code-dagger-rows-turns.tgz"
    assert cell.bundle_file(turns) == home / "gen/bundle/code-dagger-rows-turns.tgz"
    with pytest.raises(cell.Failed, match="not here"):
        cell.bundle_here(turns)
    (home / "gen/bundle").mkdir(parents=True)
    for name, data in (("code-dagger-rows", b"rows"), ("code-dagger-rows-turns", b"turns")):
        (home / "gen/bundle" / f"{name}.tgz").write_bytes(data)
    with pytest.raises(cell.Failed, match="no bundle"):
        cell.bundle_here(plain)
    # setup.sh unpacked the default bundle here (its MD5, then its name; a machine set up before the name has the MD5
    # alone): the default cell runs, the turn-based one is refused rather than run on the other bundle's code
    (home / "gen/bundle/code.unpacked").write_text(f"{cell.md5_of(home / 'gen/bundle/code-dagger-rows.tgz')} code-dagger-rows\n")
    assert cell.bundle_here(plain) == cell.md5_of(home / "gen/bundle/code-dagger-rows.tgz")
    with pytest.raises(cell.Failed, match="another bundle"):
        cell.bundle_here(turns)
    (home / "gen/bundle/code.unpacked").write_text(cell.md5_of(home / "gen/bundle/code-dagger-rows.tgz") + "\n")
    assert cell.bundle_here(plain)


def test_a_bundle_is_named_by_its_name_alone(tmp_path):
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({"name": "t", "kind": "val", "adapters": {}, "bundle": "../elsewhere/code"}))
    with pytest.raises(cell.Failed, match="by its name alone"):
        cell.load(str(bad))
    bad.write_text(json.dumps({"name": "t", "kind": "val", "adapters": {}, "bundle": "code-dagger-rows-turns"}))
    out = subprocess.run(["python3", str(Path(cell.__file__)), "bundle", str(bad)], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "code-dagger-rows-turns"


def test_a_val_adapter_of_null_plays_the_base_model(home, monkeypatch):
    c = cell.Val({"name": "t", "kind": "val", "adapters": {"base": None, "a": "node/run/adapter"}})
    assert cell.inputs(c, ["play"], done=set())[-2:] == [
        f"{cell.G}/node/run/adapter/{f} gen/t/adapters/a/{f}" for f in cell.ADAPTER]
    ran = []
    monkeypatch.setattr(cell.Cell, "sh", lambda self, argv, log: ran.append([str(x) for x in argv]))
    monkeypatch.setattr(cell.Cell, "keep_val", lambda self, name, to: None)
    c.play()
    assert ran[0][2:] == ["gen-t-base:none", f"gen-t-a:adapter:{home}/gen/t/adapters/a"]


def test_a_val_cells_sampling_reaches_each_models_play_as_one_word(home, monkeypatch):
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
    c = cell.Val({"name": "t", "kind": "val", "adapters": {"a": "node/run/adapter", "b": None}, "sampling": sampling})
    assert c.check()[0].endswith(json.dumps(sampling))
    monkeypatch.setattr(cell.Cell, "sh", lambda self, argv, log: None)
    monkeypatch.setattr(cell.Cell, "keep_val", lambda self, name, to: None)
    c.play()
    for label in ("a", "b"):
        words = (home / f"exp/gen-t-{label}/play.args").read_text().split()
        assert words[0] == "--sampling" and len(words) == 2 and json.loads(words[1]) == sampling
    with pytest.raises(cell.Failed, match="JSON object"):
        cell.Val({"name": "t", "kind": "val", "adapters": {}, "sampling": "top_p=0.95"}).check()


def test_a_relabel_points_each_moment_at_the_play_fetched_beside_it(home):
    c = cell.Relabel({"name": "t", "kind": "relabel", "run": "node/run", "rounds": [3, 4],
                      "labeller": {"guidelines": "g/c1.md"}})
    lines = cell.inputs(c, ["label"], done=set())
    assert f"{cell.G}/node/run/play/rounds/r0003/captured/ gen/t/play/rounds/r0003/captured" in lines   # whole games
    for r in ("r0003", "r0004"):
        (home / "gen/t/points").mkdir(parents=True, exist_ok=True)
        (home / f"gen/t/points/{r}.jsonl").write_text(json.dumps({"id": r, "play": f"/elsewhere/{r}"}) + "\n\n")
    points = [json.loads(line) for line in c.gather_points().read_text().splitlines()]
    assert [(p["id"], p["play"]) for p in points] == [(r, str(home / f"gen/t/play/rounds/{r}")) for r in ("r0003", "r0004")]
