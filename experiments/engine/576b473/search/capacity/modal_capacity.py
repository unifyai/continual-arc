"""The capacity ablation's trainer on Modal (README.md): a deployed app, MODAL_CAPACITY in the research environment, whose
function ``phase`` trains a run's waiting sets as phases on 8 H100s (trainer.py) and ends, so that nothing is paid while
the generator plays the next batch. brain's capacity task ships each set into the volume MODAL_CAPACITY (RUN/sets/) and
spawns ``phase`` for a run with a set waiting and no job running; the job's adapters, state.json and steps.jsonl stay in
the volume under RUN/, committed every five minutes, which the task copies to the bucket and the dashboard reads.

The image is modal_train.py's, the DAgger rows' environment built from the turn-based bundle (setup.sh, TRAIN_ONLY=1
NO_MODEL=1) on CUDA 13.0's Ubuntu 24.04 image, with this checkout's tools at ~/gen/tools; Qwen3.5-9B is read from the
volume MODAL_HF_VOLUME. Every job leaves its audit record, its purpose and owner, in the volume MODAL_JOBS_VOLUME as modal_train.py's
do (cap-RUN-START/job.json), which the billing watcher reads. Modal ends a job at 24 hours; trainer.py starts no phase
that would end after 22.5, and the next job takes up the next phase.

  deploy:  bash search/capacity/modal-capacity.sh deploy       (the image from the bundle and this checkout's tools)
  spawn:   bash search/capacity/modal-capacity.sh spawn RUN    (what brain's task does, by hand)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import modal

C = Path(__file__).resolve().parent
ENVIRONMENT = "research"
APP = "MODAL_CAPACITY"
CAP, OUT, HF = "/cap", "/out", "/root/.cache/huggingface"
DAY = 24 * 60 * 60
COMMIT_EVERY = 300
CACHE = Path.home() / ".cache" / "gen-cache"   # modal-capacity.sh fetches the bundle here
BASE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
BUNDLE = "code-dagger-rows-turns"
GPU = "H100:8"
# every process's environment as cell.py's training steps have it: NCCL without NVLink SHARP, the model offline
TRAINING = {"HF_HUB_OFFLINE": "1", "NCCL_NVLS_ENABLE": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def image(tools: Path) -> modal.Image:
    return (modal.Image.from_registry(BASE, add_python="3.12")
            .apt_install("git", "curl", "wget", "ca-certificates", "build-essential", "procps")
            .env({"HOME": "/root", "PYTHONUNBUFFERED": "1",
                  "PATH": "/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
            .add_local_file(CACHE / f"{BUNDLE}.tgz", f"/root/gen/bundle/{BUNDLE}.tgz", copy=True)
            .add_local_file(CACHE / "data.tgz", "/root/gen/bundle/data.tgz", copy=True)
            .add_local_file(tools / "setup.sh", "/root/gen/setup.sh", copy=True)
            .run_commands(f"BUNDLE={BUNDLE} TRAIN_ONLY=1 NO_MODEL=1 bash /root/gen/setup.sh; cat /root/gen/setup.log; "
                          "grep -q 'setup done' /root/gen/setup.log")
            .add_local_dir(tools, "/root/gen/tools"))


TOOLS_DIR = Path(tempfile.mkdtemp()) if modal.is_local() and os.environ.get("CAPACITY_DEPLOY") else None
if TOOLS_DIR:
    subprocess.run(["bash", str(C.parent / "generalisation" / "tools.sh"), str(TOOLS_DIR)], check=True)
# Inside the container, and wherever the app is only looked up, this module is imported without the bundle; Modal runs
# the function it was deployed with, so the image there is a stand-in.
app = modal.App(APP, image=image(TOOLS_DIR) if TOOLS_DIR else modal.Image.debian_slim())
cap = modal.Volume.from_name("MODAL_CAPACITY", create_if_missing=True)
jobs = modal.Volume.from_name("MODAL_JOBS_VOLUME", create_if_missing=True)
hf = modal.Volume.from_name("MODAL_HF_VOLUME", create_if_missing=True)


@app.function(gpu=GPU, cpu=32.0, memory=256 * 1024, timeout=DAY, retries=0, single_use_containers=True,
              volumes={CAP: cap, OUT: jobs, HF: hf})
def phase(run: str, record: dict, folder: str = "") -> dict:
    """The run's waiting sets trained as phases (trainer.py), its folder in the volume (RUN, or ``folder`` for a check
    that must leave the run's own alone) committed as it goes."""
    started = f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}"
    job = Path(OUT) / f"cap-{run}-{started.replace(':', '')}"
    job.mkdir(parents=True, exist_ok=True)
    (job / "job.json").write_text(json.dumps({**record, "started": started,
                                              "call": modal.current_function_call_id()}, indent=1) + "\n")
    jobs.commit()
    root = Path(CAP) / (folder or run)
    root.mkdir(parents=True, exist_ok=True)
    ended = threading.Event()

    def commit() -> None:
        while not ended.wait(COMMIT_EVERY):
            cap.commit()

    threading.Thread(target=commit, daemon=True).start()
    try:
        cap.reload()
        code = subprocess.run(["bash", "-c", 'cd ~/prime-rl && PYTHONPATH=$HOME/loop exec uv run --no-sync python '
                               '~/gen/tools/capacity/trainer.py "$@"', "trainer", run, "--root", str(root)],
                              env={**os.environ, **TRAINING}).returncode
    finally:
        ended.set()
        cap.commit()
    state = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
    (job / "ended.json").write_text(json.dumps({"ended": f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}",
                                                "code": code, "phases": len(state.get("phases", []))}) + "\n")
    jobs.commit()
    if code:
        raise RuntimeError(f"{run}: trainer.py exited {code} (its progress: MODAL_CAPACITY {root.name}/progress)")
    return {"phases": len(state.get("phases", [])), "plateaued": state.get("plateaued", False)}
