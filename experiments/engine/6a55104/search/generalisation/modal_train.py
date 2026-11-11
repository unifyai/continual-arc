"""A cell's train stage on Modal (README.md, "Modal"): the stage alone, on 8 GPUs in one container, its products handed
to the bucket as generalisation/NAME/train.tgz for its play stage on a machine that plays (launch.sh DEST CELL play).
Modal cannot play: its containers run under gVisor, which brings no loopback up in a new network namespace, and the
harness's bubblewrap sandbox makes one for every model-driven turn ("loopback: Failed RTM_NEWADDR", 25 Oct 2026).

The image is the DAgger rows' environment as setup.sh builds it on any machine (TRAIN_ONLY=1, NO_MODEL=1), from the
cell's bundle in the bucket (its "bundle", which modal-train.sh fetches and names in BUNDLE, read as this module loads,
since the image is made then; main refuses a cell of another bundle), on CUDA 13.0's Ubuntu 24.04 image, the paper
nodes' system (Modal's driver speaks CUDA 13.3), with this checkout's tools (tools.sh) at ~/gen/tools. The container
fetches the stage's inputs itself on an hour's token of a research account, and keeps them and all it writes in the
volume MODAL_JOBS_VOLUME under the job's name, committed every five minutes, so that a container Modal restarts takes up where
it stopped; Qwen3.5-9B is kept in the volume MODAL_HF_VOLUME. Like continual-arc-research's modal-run.sh, a job is refused
without TTL_HOURS (at most Modal's 24) and a PURPOSE as detailed as a lease's, kept with its owner in job.json, the
job's audit record, which the billing watcher reads.

Usage (modal-train.sh, which reads the workspace's token from Secret Manager, runs it):
  TTL_HOURS=14 PURPOSE="..." bash search/generalisation/modal-train.sh scratch-u4 [H100:8]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import modal

G = Path(__file__).resolve().parent   # in the container, only this file, at /root
if modal.is_local():
    sys.path[:0] = [str(G), str(G.parents[1])]   # cell.py and bucket.py, and bucket.py's helpers at the checkout's top

ENVIRONMENT = "research"
OUT = "/out"   # the volume MODAL_JOBS_VOLUME, as modal-run.sh's jobs keep it: one folder per job, its job.json at the top
HF = "/root/.cache/huggingface"
DAY = 24 * 60 * 60
COMMIT_EVERY = 300
CACHE = Path.home() / ".cache" / "gen-cache"   # the bundle, which modal-train.sh fetches to build the image from
BASE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
MODEL = "Qwen/Qwen3.5-9B"


def image(tools: Path, bundle: str) -> modal.Image:
    return (modal.Image.from_registry(BASE, add_python="3.12")
            .apt_install("git", "curl", "wget", "ca-certificates", "build-essential", "procps")
            .env({"HOME": "/root", "PYTHONUNBUFFERED": "1",
                  "PATH": "/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
            .add_local_file(CACHE / f"{bundle}.tgz", f"/root/gen/bundle/{bundle}.tgz", copy=True)
            .add_local_file(CACHE / "data.tgz", "/root/gen/bundle/data.tgz", copy=True)
            .add_local_file(G / "setup.sh", "/root/gen/setup.sh", copy=True)
            .run_commands(f"BUNDLE={bundle} TRAIN_ONLY=1 NO_MODEL=1 bash /root/gen/setup.sh; cat /root/gen/setup.log; "
                          "grep -q 'setup done' /root/gen/setup.log")
            .add_local_dir(tools, "/root/gen/tools"))


TOOLS_DIR = Path(tempfile.mkdtemp()) if modal.is_local() else None
BUNDLE = os.environ.get("BUNDLE", "") if modal.is_local() else ""   # the cell's bundle, as modal-train.sh fetched it
if TOOLS_DIR:
    subprocess.run(["bash", str(G / "tools.sh"), str(TOOLS_DIR)], check=True)
# Inside the container this module is imported again without the laptop's files; Modal runs the function it was sent,
# so the image there is a stand-in.
app = modal.App("MODAL_APP", image=image(TOOLS_DIR, BUNDLE) if TOOLS_DIR and BUNDLE else modal.Image.debian_slim())
jobs = modal.Volume.from_name("MODAL_JOBS_VOLUME", create_if_missing=True)
hf = modal.Volume.from_name("MODAL_HF_VOLUME", create_if_missing=True)


@app.function(volumes={OUT: jobs, HF: hf}, timeout=DAY, retries=0, single_use_containers=True)
def train(job: str, name: str, manifest: list[str], token: str, upload: str, record: dict) -> None:
    home, out = Path.home(), Path(OUT) / job
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "job.json").exists():
        (out / "job.json").write_text(json.dumps({**record, "started": f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}"},
                                                 indent=1) + "\n")
    jobs.commit()
    # the cell's folders in the volume, where a restarted container finds them
    for rel in (f"gen/{name}", "exp", "runs"):
        (out / rel).mkdir(parents=True, exist_ok=True)
        if not (home / rel).is_symlink():
            (home / rel).parent.mkdir(parents=True, exist_ok=True)
            (home / rel).symlink_to(out / rel)
    (home / f"gen/{name}/uploads.json").write_text(json.dumps({"train": upload}))
    ended = threading.Event()

    def commit() -> None:
        while not ended.wait(COMMIT_EVERY):
            jobs.commit()

    threading.Thread(target=commit, daemon=True).start()
    tools = home / "gen/tools"
    try:
        if not (out / "inputs.done").exists():
            subprocess.run([sys.executable, str(tools / "bucket.py"), "get"], input="\n".join([token, *manifest]) + "\n",
                           text=True, check=True, cwd=tools)
            (out / "inputs.done").touch()
        if not list(Path(HF).glob("hub/models--Qwen--Qwen3.5-9B/snapshots/*/config.json")):
            subprocess.run(["bash", "-c", "HF_HUB_ENABLE_HF_TRANSFER=1 uv run --with huggingface_hub --with hf_transfer "
                            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')\""],
                           check=True, cwd=home)
            hf.commit()
        code = subprocess.run([sys.executable, str(tools / "cell.py"), "run", str(tools / f"cells/{name}.json"),
                               "--stages", "train"], cwd=home).returncode
    finally:
        ended.set()
        jobs.commit()
    if code:
        raise RuntimeError(f"{name}: the train stage exited {code} (its progress: {job}/runs/rt/gen-{name}/progress)")


def main() -> int:
    import bucket
    import cell

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cell", help="cells/NAME.json, or NAME")
    p.add_argument("gpu", nargs="?", default="H100:8", help="a Modal GPU with :8 (H100:8, H200:8)")
    a = p.parse_args()
    path = Path(a.cell) if Path(a.cell).is_file() else G / "cells" / f"{a.cell}.json"
    c = cell.load(str(path))
    if c.bundle != BUNDLE:
        p.error(f"{c.name} runs on the bundle {c.bundle}, and the image would be built from {BUNDLE or 'none'}: "
                "run it through modal-train.sh, which fetches the cell's bundle and names it in BUNDLE")
    shutil.copy2(path, TOOLS_DIR / "cells" / f"{c.name}.json")   # a cell not committed yet goes too, as launch.sh sends it
    if "train" not in c.STAGES:
        p.error(f"{c.name} has no train stage")
    if not a.gpu.endswith(":8"):
        p.error("the train stage takes 8 GPUs")
    try:
        hours = float(os.environ.get("TTL_HOURS", ""))
    except ValueError:
        p.error("TTL_HOURS is required: the timeout at which Modal ends the job, its expected length plus a margin")
    if not 0 < hours * 3600 <= DAY:
        p.error("TTL_HOURS must be above 0 and at most 24, Modal's longest timeout")
    purpose = os.environ.get("PURPOSE", "")
    baselines = Path(os.environ.get("BASELINES", Path.home() / "continual-arc-baselines"))
    sys.path.insert(0, str(baselines / "src"))
    from continual_arc_fleet import leases   # the fleet's rule for a lease's reason, which a Modal job's purpose keeps
    if leases.thin(purpose, c.name):
        p.error(f"PURPOSE is required: why the job runs, at least {leases.MIN_PURPOSE_WORDS} words and "
                f"{leases.MIN_PURPOSE_CHARS} characters, as detailed and explicit as a lease's (scripts/lease.py)")

    def token() -> str:
        return subprocess.run(["bash", str(baselines / "scripts/secret.sh"), "--node-token"], capture_output=True,
                              text=True, check=True).stdout.strip()

    job = f"gen-{c.name}-train"
    owner = os.environ.get("OWNER") or os.environ.get("USER") or "unknown"
    commit = (TOOLS_DIR / "COMMIT").read_text().strip()
    record = {"name": job, "gpu": a.gpu, "cpu": 32.0, "memory_gb": 256.0, "timeout_hours": hours,
              "command": f"cell.py run cells/{c.name}.json --stages train", "owner": owner, "purpose": purpose,
              "secrets": [], "revisions": {"continual-arc-training": commit, "bundle": cell.bundle_object(c)}}
    manifest = cell.inputs(c, ["train"], done=set())
    upload = bucket.new_session(token(), f"{cell.PRODUCTS}/{c.name}/train.tgz")
    fn = train.with_options(gpu=a.gpu, cpu=32.0, memory=256 * 1024, timeout=int(hours * 3600))
    with modal.enable_output(), app.run(name=job, detach=True, environment_name=ENVIRONMENT):
        app.set_tags({"owner": re.sub(r"[^A-Za-z0-9._-]", "-", owner)[:63], "job": job, "repo": "continual-arc-training"})
        call = fn.spawn(job, c.name, manifest, token(), upload, record)
        print(f"{job}: app {app.app_id}, call {call.object_id}, in Modal's {ENVIRONMENT} environment on {a.gpu}, ended by "
              f"Modal at {hours:g} h at the latest\n"
              f"  follow: bash {baselines}/scripts/modal.sh app logs {app.app_id}\n"
              f"  stop:   bash {baselines}/scripts/modal.sh app stop {app.app_id}\n"
              f"  then:   launch.sh DEST {c.name} play  (its products: generalisation/{c.name}/train.tgz)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
