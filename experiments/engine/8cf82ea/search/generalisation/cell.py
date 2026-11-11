"""The generalisation search's cells (README.md): each a JSON file in cells/, a kind and its settings, run on any machine
in stages, each resumable, its inputs fetched from the bucket and its products handed to it.

  inputs CELL [--stages S,..] [--done S,..]  the bucket inputs the stages need on a machine, one "SOURCE PATH [names]"
                                             a line (bucket.py get's manifest); an earlier stage that is not --done
                                             there comes as its products' tarball
  run CELL [--stages S,..]                   the stages here, in the cell's order (default: all of them), each skipped
                                             once done; ~/runs/rt/gen-NAME/progress is its log
  check CELL                                 the cell read and its plan printed, nothing run
  stages CELL [--stages S,..]                the stages chosen, in the cell's order, comma-separated (launch.sh's)
  bundle CELL                                the code bundle the cell runs on (launch.sh's and modal-train.sh's)

A stage that ends hands its products, the files a later stage reads, to the bucket as generalisation/NAME/STAGE.tgz
when ~/gen/NAME/uploads.json holds an upload session for it (launch.sh and modal_train.py make them), so the next
stage can run on another machine. Everything a cell finds is written under ~/runs/rt/gen-NAME, which the operator's
hourly backup copies off a routed machine: plays, measures, trained adapters and logs.

Kinds:
  frozen    the paper's two frozen tests from an empty memory (frozen.sh MEMORY=none), under a run's task names, with
            up to four adapters side by side
  val       R's 100 validation puzzles as she played them (grid-vals.sh), with any number of adapters, four at a
            time on two GPUs each; an adapter of null plays the base model
            (either kind's adapter is a path in the backup, or cell:NAME, what recipe cell NAME trained)
  recipe    R's recipe from the base model (grid-exp.sh: games taken whole until TARGET examples, train.py at her
            settings) on a run's own labels of some rounds, merged; then val and frozen of its adapter. ``weighting``
            (turn or split) weighs the loss's tokens otherwise than alike (grid-exp.sh's WEIGHTING)
  scratch   a run's own trainer once from the base model on every label of rounds 0..PHASE (scratch.py train); then
            val and frozen
  replay    a run's training replayed offline from one of its phases through another, under the rule it trained by
            before its switch (replay.py); then val of some of its phases and frozen of some, with the run's own beside
  relabel   a run's moments of some rounds labelled again by another labeller (forward.py label through the relay),
            then recipe on that set

Stages: label (the network and the OpenRouter key in ~/.openrouter.env), train (8 GPUs), play (8 GPUs, vLLM and the
harness's bubblewrap, as the user NODE_USER, since /home/NODE_USER is spelled out in the loops' scripts and reaches the
models' prompts in the paths of their workspaces). The stage commands are the loops' own, from the DAgger rows' code
(~/loop, setup.sh's bundle) and the tools beside this file.

Every kind takes ``bundle``, the code it runs on: BUNDLES/NAME.tgz in the bucket, ``code-dagger-rows`` (the DAgger
rows' own trees, whose plays cut an action at its seconds) unless it names another, such as ``code-dagger-rows-turns``
(the same trees with only the budget changed: turn-based plays, the turn's liveness guard an infrastructure failure).
A machine holds one bundle's trees, so a stage refuses to run where another's are unpacked (:func:`bundle_here`).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
TOOLS = Path(__file__).resolve().parent
G = "results/runs/gpu-nodes"   # where the backup keeps every node's runs, by node and run
PRODUCTS = "generalisation"    # each cell's stage products in the bucket: generalisation/NAME/STAGE.tgz
BUNDLES = "bundles/generalisation"   # the code bundles in the bucket, BUNDLES/NAME.tgz, each fetched to ~/gen/bundle
BUNDLE = "code-dagger-rows"          # a cell's bundle unless its "bundle" names another
BUNDLE_NAME = re.compile(r"[a-z0-9][a-z0-9.-]*")
R_VAL = f"{G}/NODE-fwd-comparison/val-fwd2-comparison"   # her validation's schedule and alias salt
PLAY_CONFIG = ("alias-salt", "config.json", "driver.json", "meta.json", "schedule.json")   # what frozen.py reads of a run
ADAPTER = ("adapter_config.json", "adapter_model.safetensors")   # what vLLM loads
WEIGHTINGS = ("token", "turn", "split")   # train.py's --weighting, the bundle's ask_sd.trainer.WEIGHTINGS
GPUS = 8
RELAY_PORT = 18231
# A machine's environment: the trainers' and servers' NCCL without NVLink SHARP, whose init fails on the cloud provider's 8x H100
# nodes made from 25 Oct 2026 (continual-arc-research's docs/RUNNING.md) and which nothing here needs
ENV = {**os.environ, "PATH": f"{HOME}/.local/bin:{os.environ['PATH']}", "HF_HUB_OFFLINE": "1", "NCCL_NVLS_ENABLE": "0"}
# A training step's allocator, the launchers' setting, never a server's: vLLM's custom all-reduce cannot open that
# memory over CUDA IPC ('custom_all_reduce.cuh:164 invalid argument'), so a tensor-parallel server inheriting it fails
TRAINING = {**ENV, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
# CUDA 13's forward-compatibility libraries, which remote-bootstrap.sh installs where the driver stops short of CUDA 13
# (Lambda's 570 speaks 12.8) and the CUDA 13 wheels then need on the library path; never with a CUDA 13 driver, whose
# own libraries are newer (Modal's image carries the package beside a 13.3 driver)
COMPAT = Path("/usr/local/cuda-13.0/compat")


class Failed(Exception):
    pass


def rounds(first: int, last: int) -> list[str]:
    return [f"r{k:04d}" for k in range(first, last + 1)]


def adapter_inputs(source: str, dest: str) -> list[str]:
    """An adapter's files for a val or frozen cell: from the backup (G/SOURCE), or for ``cell:NAME`` the adapter that
    recipe cell NAME trained, its train stage's products (unpacked where :func:`adapter_dir` reads them)."""
    if source.startswith("cell:"):
        return [f"{PRODUCTS}/{source[5:]}/train.tgz {dest}/train.tgz"]
    return [f"{G}/{source}/{f} {dest}/{f}" for f in ADAPTER]


def adapter_dir(source: str, dest: Path) -> Path:
    """The folder that holds ``source``'s adapter once :func:`adapter_inputs` has fetched it to ``dest``: ``dest``
    itself, or for ``cell:NAME`` the adapter in the train stage's products, unpacked there."""
    if not source.startswith("cell:"):
        return dest
    adapter = dest / "exp" / f"gen-{source[5:]}" / "model/adapter"
    if not adapter.exists():
        subprocess.run(["tar", "-C", str(dest), "-xzf", str(dest / "train.tgz")], check=True)
    return adapter


def frozen_gpus(n: int) -> list[str]:
    """Two GPUs for each of n adapters' frozen tests, one a test, n at most 4: the layout of the switch test, which the
    DAgger rows' weights-alone scores come from (frozen.sh on two GPUs an arm, 24 Oct 2026). The tests cut an action at
    its time budget, so a layout that serves faster scores higher, and a cell is compared only with its own layout."""
    if not 0 < n <= GPUS // 2:
        raise Failed(f"{n} adapters' frozen tests: 1 to {GPUS // 2} side by side, two GPUs each")
    return [f"{2 * i},{2 * i + 1}" for i in range(n)]


class Cell:
    """A cell: its name and settings, the stages its kind runs, their inputs and products, and the stages themselves."""

    STAGES: tuple[str, ...] = ()

    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec["name"]
        self.bundle = spec.get("bundle", BUNDLE)
        self.work = Path(f"gen/{self.name}")           # under ~: inputs and the tools' own folders
        self.out = HOME / "runs/rt" / f"gen-{self.name}"  # under ~: what the backup copies

    # what each kind says

    def inputs_for(self, stage: str) -> list[str]:
        return []

    def products(self, stage: str) -> list[str]:
        """Paths under ~ that later stages read, its marker always among them."""
        return [str((self.out / f"{stage}.done").relative_to(HOME))]

    def check(self) -> list[str]:
        return []

    # shared pieces

    def log(self, message: str) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "progress", "a") as f:
            f.write(f"{datetime.now(timezone.utc):%m-%d %H:%M} {self.name}: {message}\n")

    def sh(self, argv: list, logfile: Path, env: dict | None = None) -> None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a") as f:
            code = subprocess.run([str(a) for a in argv], cwd=HOME, env=env or ENV, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            what = next((Path(str(a)).name for a in argv if str(a).endswith((".sh", ".py"))), str(argv[0]))
            raise Failed(f"{what} exited {code} ({logfile})")

    def loop_py(self, args: list, logfile: Path, env: dict | None = None) -> None:
        """A Python tool in prime-rl's environment with the DAgger rows' code (~/loop) on the path, as the loops run."""
        self.sh(["bash", "-c", 'cd ~/prime-rl && PYTHONPATH=$HOME/loop exec uv run --no-sync python "$@"', "loop_py",
                 *map(str, args)], logfile, env)

    def run_inputs(self, run: str, first: int, last: int, dest: Path) -> list[str]:
        """A run's rounds FIRST..LAST as its trainer reads them, under DEST/play/rounds: each round's label sets
        (labels/split) and its captured games by name."""
        lines = []
        for r in rounds(first, last):
            src = f"{G}/{run}/play/rounds/{r}"
            lines += [f"{src}/labels/split/ {dest}/play/rounds/{r}/labels/split",
                      f"{src}/captured/ {dest}/play/rounds/{r}/captured names"]
        return lines

    def frozen(self, plays: dict[str, Path]) -> None:
        """frozen.sh for each adapter (label: folder), side by side on two GPUs each (frozen_gpus), from an empty memory
        under the task names of the run whose play config is this cell's play/ (play_config_inputs), into
        OUT/frozen-LABEL, or OUT/frozen for one adapter."""
        procs = []
        for (label, adapter), gpus in zip(plays.items(), frozen_gpus(len(plays))):
            out = self.out / (f"frozen-{label}" if len(plays) > 1 else "frozen")
            out.mkdir(parents=True, exist_ok=True)
            env = {**ENV, "OUT": str(out), "ADAPTER": str(adapter), "SERVED": f"gen-{label}", "MEMORY": "none",
                   "CODE": str(HOME / "loop")}
            with open(out / "frozen.sh.log", "a") as log:
                procs.append((out, subprocess.Popen(["bash", str(TOOLS / "frozen.sh"), f"gen-{self.name}", gpus],
                                                    cwd=HOME, env=env, stdout=log, stderr=subprocess.STDOUT)))
        for _, p in procs:
            p.wait()
        undone = [str(out) for out, _ in procs if not (out / "done").exists()]
        if undone:
            raise Failed(f"frozen tests not done: {', '.join(undone)} (their progress)")

    def play_config_inputs(self, run: str) -> list[str]:
        """The run's play config, which frozen.sh reads as this cell's run (~/runs/rt/gen-NAME/play)."""
        return [f"{G}/{run}/play/{f} {(self.out / 'play' / f).relative_to(HOME)}" for f in PLAY_CONFIG]

    def val_inputs(self) -> list[str]:
        """R's validation schedule and alias salt where grid-exp.sh and grid-vals.sh read them."""
        return [f"{R_VAL}/schedule.json runs/loop/sched-val.json", f"{R_VAL}/alias-salt runs/loop/r0-val/alias-salt"]

    def keep_val(self, name: str, to: str) -> None:
        """A validation play (grid-exp.sh's or grid-vals.sh's NAME) and its measures under OUT/TO, for the backup."""
        dest = self.out / to
        shutil.copytree(HOME / "runs/loop" / f"val-{name}", dest, symlinks=True, dirs_exist_ok=True)
        for f in ("progress", "val-measures.json"):
            if (HOME / "exp" / name / f).exists():
                shutil.copy2(HOME / "exp" / name / f, dest / f)

    def keep_adapter(self, model: Path, to: str) -> None:
        """A trained model's adapter and DONE under OUT/TO, for the backup."""
        dest = self.out / to
        (dest / "adapter").mkdir(parents=True, exist_ok=True)
        for f in ADAPTER:
            shutil.copy2(model / "adapter" / f, dest / "adapter" / f)
        for f in ("DONE", "phase.json"):
            if (model / f).exists():
                shutil.copy2(model / f, dest / f)


class Frozen(Cell):
    STAGES = ("play",)

    def check(self) -> list[str]:
        frozen_gpus(len(self.spec["adapters"]))
        return [f"frozen tests of {', '.join(self.spec['adapters'])} under {self.spec['run']}'s task names"]

    def inputs_for(self, stage: str) -> list[str]:
        lines = self.play_config_inputs(self.spec["run"])
        for label, source in self.spec["adapters"].items():
            lines += adapter_inputs(source, f"{self.work}/adapters/{label}")
        return lines

    def play(self) -> None:
        self.frozen({label: adapter_dir(source, HOME / self.work / "adapters" / label)
                     for label, source in self.spec["adapters"].items()})


class Val(Cell):
    STAGES = ("play",)

    def check(self) -> list[str]:
        return [f"R's 100 puzzles with {', '.join(self.spec['adapters'])}"]

    def inputs_for(self, stage: str) -> list[str]:
        lines = self.val_inputs()
        for label, source in self.spec["adapters"].items():
            if source:
                lines += adapter_inputs(source, f"{self.work}/adapters/{label}")
        return lines

    def play(self) -> None:
        # grid-vals.sh's NAME:INIT, INIT none for the base model (an adapter of null)
        specs = {label: f"gen-{self.name}-{label}:" + (f"adapter:{adapter_dir(source, HOME / self.work / 'adapters' / label)}"
                                                         if source else "none")
                 for label, source in self.spec["adapters"].items()}
        self.sh(["bash", TOOLS / "grid-vals.sh", *specs.values()], self.out / "grid-vals.log")
        for label in specs:
            self.keep_val(f"gen-{self.name}-{label}", f"val-{label}")


class Recipe(Cell):
    """R's recipe on a label set: ``labels`` names a run and its rounds [first, last], whose label sets merge, or a
    ``set``, a tarball of one in the bucket, and the ``dir`` it unpacks to (her round-1 labels:
    NODE-fwd-comparison/labels/comparison-stage1.tar.gz, stage1/set)."""

    STAGES = ("train", "play")

    @property
    def exp(self) -> str:
        return f"gen-{self.name}"

    def label_set(self) -> Path:
        return HOME / self.work / self.spec["labels"].get("dir", "set")

    def check(self) -> list[str]:
        s = self.spec
        labels = s.get("labels")
        src = ("the set its label stage makes" if labels is None
               else labels.get("set") or f"{labels['run']} rounds {labels['rounds']}")
        if s.get("weighting", "token") not in WEIGHTINGS:
            raise Failed(f"weighting must be one of {', '.join(WEIGHTINGS)}, not {s['weighting']!r}")
        return [f"R's recipe from the base model on {src}: {s['target']} examples, LR {s['lr']}, batch {s['batch']}, "
                f"{s['epochs']} passes, held out {s['held']}, weighting {s.get('weighting', 'token')}",
                f"frozen tests under {s['run']}'s task names"]

    def inputs_for(self, stage: str) -> list[str]:
        if stage == "train":
            src = self.spec["labels"]
            if "set" in src:
                return [f"{G}/{src['set']} {self.work}/labels.tgz"]
            return self.run_inputs(src["run"], *src["rounds"], self.work / "rounds")
        return self.val_inputs() + self.play_config_inputs(self.spec["run"])

    def products(self, stage: str) -> list[str]:
        if stage != "train":
            return super().products(stage)
        e = f"exp/{self.exp}"
        return super().products(stage) + [f"{e}/{f}" for f in ("pool.done", "train.done", "progress", "pool.log",
                                                                "train.log", "model/DONE")] + \
            [f"{e}/model/adapter/{f}" for f in ADAPTER]

    def grid_exp(self, until: str = "") -> None:
        """grid-exp.sh's cell, whose validation waits for every game."""
        s = self.spec
        env = {**ENV, **({"UNTIL": until} if until else {}), **({"WEIGHTING": s["weighting"]} if "weighting" in s else {})}
        self.sh(["bash", TOOLS / "grid-exp.sh", self.exp, "base", s["target"], s["lr"], s["batch"], s["epochs"],
                 self.label_set(), s["held"]], self.out / "grid-exp.log", env)

    def merge(self) -> None:
        """The run's label sets of its rounds merged into one (dagger.py merge, each round's dev and train in turn), or
        a label set's tarball unpacked."""
        if (HOME / self.work / "set.done").exists():
            return
        if "set" in self.spec["labels"]:
            subprocess.run(["tar", "-C", str(HOME / self.work), "-xzf", str(HOME / self.work / "labels.tgz")], check=True)
            (HOME / self.work / "set.done").touch()
            return
        src = HOME / self.work / "rounds" / "play" / "rounds"
        sets = sorted(str(p) for p in src.glob("r*/labels/split/*-r*") if p.name.split("-")[0] in ("train", "dev"))
        shutil.rmtree(self.label_set(), ignore_errors=True)
        self.loop_py([HOME / "loop/dagger.py", "merge", *sets, "--out", self.label_set()], self.out / "merge.log")
        (HOME / self.work / "set.done").touch()

    def train(self) -> None:
        self.merge()
        self.grid_exp(until="train")

    def play(self) -> None:
        self.grid_exp()
        model = HOME / "exp" / self.exp / "model"
        self.keep_adapter(model, "model")
        self.keep_val(self.exp, "val")
        self.frozen({self.name: model / "adapter"})


class Scratch(Cell):
    STAGES = ("train", "play")

    @property
    def model(self) -> Path:
        return HOME / self.work / "model"

    def check(self) -> list[str]:
        return [f"{self.spec['run']}'s trainer once from the base model on rounds 0-{self.spec['phase']}"]

    def inputs_for(self, stage: str) -> list[str]:
        if stage == "train":
            return self.run_inputs(self.spec["run"], 0, self.spec["phase"], self.work / "view")
        return self.val_inputs() + self.play_config_inputs(self.spec["run"])

    def products(self, stage: str) -> list[str]:
        m = str(self.model.relative_to(HOME))
        extra = [f"{m}/DONE", *(f"{m}/adapter/{f}" for f in ADAPTER)] if stage == "train" else []
        return super().products(stage) + extra

    def train(self) -> None:
        if (self.model / "DONE").exists():
            return
        self.loop_py([TOOLS / "scratch.py", "train", "--run", HOME / self.work / "view", "--phase", self.spec["phase"],
                      "--out", self.model, "--gpus", GPUS], self.out / "train.log", TRAINING)

    def play(self) -> None:
        exp = f"gen-{self.name}"
        self.sh(["bash", TOOLS / "grid-exp.sh", exp, f"adapter:{self.model / 'adapter'}", "0"], self.out / "grid-exp.log")
        self.keep_adapter(self.model, "model")
        self.keep_val(exp, "val")
        self.frozen({self.name: self.model / "adapter"})


class Replay(Cell):
    """``from`` and ``to``: the phase it starts from and the last it replays; ``vals``: its phases on R's puzzles;
    ``tests``: its phases on the frozen tests; ``own``: the run's own phases on them beside."""

    STAGES = ("train", "play")

    def phase(self, k: int) -> Path:
        return HOME / self.work / "phases" / f"p{k:04d}"

    def check(self) -> list[str]:
        s = self.spec
        frozen_gpus(len(s["tests"]) + len(s["own"]))
        return [f"{s['run']}'s phases {s['from'] + 1}-{s['to']} replayed from its phase {s['from']}",
                f"R's puzzles: phases {s['vals']}", f"frozen tests: phases {s['tests']}, the run's own {s['own']}"]

    def inputs_for(self, stage: str) -> list[str]:
        s = self.spec
        if stage == "train":
            start = f"{G}/{s['run']}/train/p{s['from']:04d}/ {self.phase(s['from']).relative_to(HOME)}"
            return self.run_inputs(s["run"], 0, s["to"], self.work / "view") + [start]
        lines = self.val_inputs() + self.play_config_inputs(s["run"])
        for k in s["own"]:
            lines += adapter_inputs(f"{s['run']}/train/p{k:04d}/adapter", f"{self.work}/own/p{k:04d}/adapter")
        return lines

    def products(self, stage: str) -> list[str]:
        if stage != "train":
            return super().products(stage)
        lines = []
        for k in sorted(set(self.spec["vals"]) | set(self.spec["tests"])):
            p = str(self.phase(k).relative_to(HOME))
            lines += [f"{p}/DONE", f"{p}/phase.json", *(f"{p}/adapter/{f}" for f in ADAPTER)]
        return super().products(stage) + lines

    def train(self) -> None:
        s = self.spec
        for attempt in range(3):
            try:
                self.loop_py([TOOLS / "replay.py", HOME / self.work / "view", HOME / self.work / "phases",
                              s["from"] + 1, s["to"], self.phase(s["from"])], self.out / "replay.log", TRAINING)
                break
            except Failed as e:
                self.log(f"the replay stopped short, attempt {attempt + 1}: {e}")
                time.sleep(60)
        if not (self.phase(s["to"]) / "DONE").exists():
            raise Failed("the replay stopped short three times")
        for k in sorted(set(s["vals"]) | set(s["tests"])):
            self.keep_adapter(self.phase(k), f"phases/p{k:04d}")

    def play(self) -> None:
        s = self.spec
        names = {k: f"gen-{self.name}-p{k:04d}" for k in s["vals"]}
        self.sh(["bash", TOOLS / "grid-vals.sh", *(f"{names[k]}:adapter:{self.phase(k) / 'adapter'}" for k in names)],
                self.out / "grid-vals.log")
        for k, name in names.items():
            self.keep_val(name, f"val-p{k:04d}")
        plays = {f"p{k:04d}": self.phase(k) / "adapter" for k in s["tests"]}
        plays |= {f"own-p{k:04d}": HOME / self.work / "own" / f"p{k:04d}" / "adapter" for k in s["own"]}
        self.frozen(plays)


class Relabel(Recipe):
    """``labeller``: forward.py label's settings, the relay's ``expert`` and ``providers`` and the ``guidelines``'
    object; then the recipe on the set it makes."""

    STAGES = ("label", "train", "play")

    @property
    def labels(self) -> Path:
        return HOME / self.work / "labels"

    def label_set(self) -> Path:
        return self.labels / "set"

    def check(self) -> list[str]:
        lab = self.spec["labeller"]
        return [f"{self.spec['run']}'s moments of rounds {self.spec['rounds']} labelled by {lab['expert']} through "
                f"{lab['providers']}, {lab['effort']} effort, guidelines {lab['guidelines']}"] + super().check()

    def inputs_for(self, stage: str) -> list[str]:
        if stage == "label":
            run, (first, last) = self.spec["run"], self.spec["rounds"]
            # each round's moments and the captured games they point into, whole (not by name, as the trainer counts
            # them): forward.py label reads each moment's calls from its game
            return [line for r in rounds(first, last)
                    for line in (f"{G}/{run}/play/rounds/{r}/labels/points.jsonl {self.work}/points/{r}.jsonl",
                                 f"{G}/{run}/play/rounds/{r}/captured/ {self.work}/play/rounds/{r}/captured")] \
                + [f"{G}/{self.spec['labeller']['guidelines']} {self.work}/guidelines.md"]
        return [] if stage == "train" else super().inputs_for(stage)

    def gather_points(self) -> Path:
        """LABELS/points.jsonl: the rounds' moments in order, each pointed at the copy of its round's play fetched here,
        since a moment names the play folder of the node that played it."""
        out = self.labels / "points.jsonl"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out.with_suffix(".jsonl.part"), "w") as f:
                for r in rounds(*self.spec["rounds"]):
                    play = str(HOME / self.work / "play" / "rounds" / r)
                    for line in (HOME / self.work / "points" / f"{r}.jsonl").read_text().splitlines():
                        if line.strip():
                            f.write(json.dumps({**json.loads(line), "play": play}) + "\n")
            out.with_suffix(".jsonl.part").rename(out)
        return out

    def products(self, stage: str) -> list[str]:
        if stage != "label":
            return super().products(stage)
        return Cell.products(self, stage) + [str(self.label_set().relative_to(HOME))]

    def relay_up(self) -> bool:
        import socket
        try:
            socket.create_connection(("127.0.0.1", RELAY_PORT), 3).close()
            return True
        except OSError:
            return False

    def label(self) -> None:
        lab, out = self.spec["labeller"], self.labels
        self.gather_points()
        key = dict(line.split("=", 1) for line in (HOME / ".openrouter.env").read_text().split())
        relay = None
        if not self.relay_up():
            relay = subprocess.Popen(
                ["bash", "-c", 'cd ~/prime-rl && exec uv run --no-sync python ~/loop/expert_relay.py "$@"', "relay",
                 "--port", str(RELAY_PORT), "--model", lab["expert"], "--providers", lab["providers"]],
                cwd=HOME, env={**ENV, **key}, stdout=open(self.out / "relay.log", "a"), stderr=subprocess.STDOUT,
                start_new_session=True)
            for _ in range(12):
                time.sleep(5)
                if self.relay_up():
                    break
            else:
                raise Failed(f"the relay did not start ({self.out / 'relay.log'})")
        try:
            if not (out / "label.done").exists():
                self.log(f"labelling {sum(1 for _ in open(out / 'points.jsonl'))} moments")
                self.loop_py([HOME / "loop/forward.py", "label", "--out", out, "--expert", lab["expert"],
                              "--expert-url", f"http://127.0.0.1:{RELAY_PORT}/v1", "--effort", lab["effort"],
                              "--budget", lab["budget"], "--workers", lab["workers"],
                              "--guidelines", HOME / self.work / "guidelines.md", "--guidelines-at", lab["guidelines_at"],
                              "--rule-retries", lab["rule_retries"], "--min-labelled-pct", lab["min_labelled_pct"]],
                             self.out / "label.log")
                (out / "label.done").touch()
        finally:
            if relay:
                os.killpg(relay.pid, 15)
        if not self.label_set().exists():
            self.loop_py([HOME / "loop/forward.py", "label-set", "--out", out], self.out / "label-set.log")

    def train(self) -> None:
        self.grid_exp(until="train")


KINDS = {"frozen": Frozen, "val": Val, "recipe": Recipe, "scratch": Scratch, "replay": Replay, "relabel": Relabel}


def load(path: str) -> Cell:
    spec = json.loads(Path(path).read_text())
    if spec.get("kind") not in KINDS:
        raise Failed(f"{path}: kind is one of {', '.join(KINDS)}")
    if Path(path).stem != spec.get("name"):
        raise Failed(f"{path}: the file is named after its cell, {spec.get('name')}")
    if not BUNDLE_NAME.fullmatch(spec.get("bundle", BUNDLE)):
        raise Failed(f"{path}: bundle names a tarball under {BUNDLES} by its name alone, not {spec['bundle']!r}")
    return KINDS[spec["kind"]](spec)


def bundle_object(cell: Cell) -> str:
    """The cell's bundle in the bucket."""
    return f"{BUNDLES}/{cell.bundle}.tgz"


def bundle_file(cell: Cell) -> Path:
    """Where launch.sh fetches the cell's bundle to, and setup.sh unpacks it from."""
    return HOME / "gen/bundle" / f"{cell.bundle}.tgz"


def md5_of(path: Path) -> str:
    import hashlib
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def bundle_here(cell: Cell) -> str:
    """The MD5 of the cell's bundle, refused unless the trees under ~ (~/loop and the rest) were unpacked from it:
    setup.sh's ~/gen/bundle/code.unpacked holds the MD5 of the bundle it unpacked, and a machine set up once is never
    set up again, so a cell launched on a machine of another bundle's would otherwise run that bundle's code without a
    word (a turn-based cell played on the default bundle's time budget)."""
    unpacked = HOME / "gen/bundle/code.unpacked"
    if not bundle_file(cell).exists():
        raise Failed(f"{cell.name}'s bundle is not here ({bundle_file(cell)}): launch.sh fetches it")
    want = md5_of(bundle_file(cell))
    have = unpacked.read_text().split()[0] if unpacked.exists() and unpacked.read_text().strip() else None
    if have != want:
        raise Failed(f"this machine's trees are from {'another bundle (' + have + ')' if have else 'no bundle'}, not "
                     f"{cell.name}'s {cell.bundle} ({want}): a machine holds one bundle, so use one set up from it")
    return want


def chosen(cell: Cell, stages: str | None) -> list[str]:
    picked = cell.STAGES if not stages else tuple(stages.split(","))
    unknown = set(picked) - set(cell.STAGES)
    if unknown:
        raise Failed(f"{cell.name} has no stage {', '.join(sorted(unknown))} (its stages: {', '.join(cell.STAGES)})")
    return [s for s in cell.STAGES if s in picked]


def tarball(cell: Cell, stage: str) -> Path:
    return HOME / cell.work / f"{stage}.tgz"


def inputs(cell: Cell, stages: list[str], done: set[str]) -> list[str]:
    """The manifest for STAGES on a machine where DONE are done: each stage's inputs, and the products of every
    earlier stage neither run nor done there."""
    first = cell.STAGES.index(stages[0])
    lines = [f"{PRODUCTS}/{cell.name}/{s}.tgz {tarball(cell, s).relative_to(HOME)}"
             for s in cell.STAGES[:first] if s not in done]
    for s in stages:
        lines += cell.inputs_for(s)
    return lines


def ready_to_play() -> None:
    """Plays run as the user NODE_USER on a machine whose bubblewrap works (setup.sh's ~/gen/sandbox)."""
    if HOME != Path("/home/NODE_USER"):
        raise Failed(f"plays run as the user NODE_USER, not with HOME {HOME}: the loops' scripts and the models' prompts "
                     "hold /home/NODE_USER")
    sandbox = (HOME / "gen/sandbox").read_text().strip() if (HOME / "gen/sandbox").exists() else "setup.sh has not run"
    if sandbox != "ok":
        raise Failed(f"no plays here, the harness's bubblewrap does not work: {sandbox}")


def gpus_here() -> int:
    return len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines())


def driver_cuda() -> int:
    """The CUDA version the driver speaks, as remote-bootstrap.sh reads it; 0 without a driver."""
    if not shutil.which("nvidia-smi"):
        return 0
    found = re.search(r"CUDA(?: UMD)? Version: (\d+)", subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    return int(found.group(1)) if found else 0


def with_compat() -> None:
    """Every stage's processes on CUDA 13's compatibility libraries where the driver needs them."""
    if COMPAT.is_dir() and 0 < driver_cuda() < 13:
        for env in (ENV, TRAINING):
            env["LD_LIBRARY_PATH"] = ":".join(filter(None, [str(COMPAT), env.get("LD_LIBRARY_PATH")]))


def hand_over(cell: Cell, stage: str) -> None:
    """The stage's products to the bucket, through the upload session launch.sh left for it, if any."""
    uploads = HOME / cell.work / "uploads.json"
    url = json.loads(uploads.read_text()).get(stage) if uploads.exists() else None
    if not url or (cell.out / f"{stage}.handed").exists():
        return
    paths = [p for p in cell.products(stage) if (HOME / p).exists()]
    subprocess.run(["tar", "-C", str(HOME), "-czf", str(tarball(cell, stage)), *paths], check=True)
    r = subprocess.run([sys.executable, str(TOOLS / "bucket.py"), "put", url, str(tarball(cell, stage))],
                       capture_output=True, text=True)
    if r.returncode:
        cell.log(f"{stage}'s products did not reach the bucket: {r.stderr.strip()[-300:]}")
        return
    (cell.out / f"{stage}.handed").touch()
    cell.log(f"{stage}'s products in the bucket: {PRODUCTS}/{cell.name}/{stage}.tgz ({len(paths)} files)")


def run(cell: Cell, stages: list[str]) -> None:
    with_compat()
    cell.out.mkdir(parents=True, exist_ok=True)
    (cell.out / "cell.json").write_text(json.dumps(cell.spec, indent=1) + "\n")
    if (TOOLS / "COMMIT").exists():
        shutil.copy2(TOOLS / "COMMIT", cell.out / "tools-commit")
    if any(s in ("label", "train", "play") for s in stages):
        (cell.out / "bundle").write_text(f"{bundle_object(cell)} {bundle_here(cell)}\n")
    for stage in stages:
        for earlier in cell.STAGES[:cell.STAGES.index(stage)]:
            if not (cell.out / f"{earlier}.done").exists():
                if not tarball(cell, earlier).exists():
                    raise Failed(f"{earlier} has not run here and its products are not here ({tarball(cell, earlier)})")
                subprocess.run(["tar", "-C", str(HOME), "-xzf", str(tarball(cell, earlier))], check=True)
                cell.log(f"{earlier}'s products unpacked from the bucket's tarball")
        if (cell.out / f"{stage}.done").exists():
            hand_over(cell, stage)
            continue
        if stage in ("train", "play") and gpus_here() < GPUS:
            raise Failed(f"{stage} needs {GPUS} GPUs; this machine has {gpus_here()}")
        if stage == "play":
            ready_to_play()
        cell.log(f"{stage} started")
        getattr(cell, stage)()
        (cell.out / f"{stage}.done").touch()
        cell.log(f"{stage} done")
        hand_over(cell, stage)
    cell.log(f"stages {','.join(stages)} done")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inputs", "run", "check", "stages", "bundle"):
        c = sub.add_parser(name)
        c.add_argument("cell", help="a cell's JSON file (cells/NAME.json)")
        c.add_argument("--stages", help="comma-separated, in the cell's order (default: all its stages)")
        if name == "inputs":
            c.add_argument("--done", default="", help="the stages already done on the machine")
    a = p.parse_args()
    cell = load(a.cell)
    stages = chosen(cell, a.stages)
    if a.cmd == "check":
        print(f"{cell.name}: {cell.spec['kind']}, stages {', '.join(cell.STAGES)}, on {bundle_object(cell)}")
        for line in cell.check():
            print(f"  {line}")
        for s in cell.STAGES:
            print(f"  {s}: {len(cell.inputs_for(s))} inputs, {len(cell.products(s))} products")
        return 0
    if a.cmd == "inputs":
        print("\n".join(inputs(cell, stages, set(filter(None, a.done.split(","))))))
        return 0
    if a.cmd == "stages":
        print(",".join(stages))
        return 0
    if a.cmd == "bundle":
        print(cell.bundle)
        return 0
    try:
        run(cell, stages)
    except Failed as e:
        cell.log(f"stopped: {e}")
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
