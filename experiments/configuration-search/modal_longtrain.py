"""longtrain.py on Modal: a real-time run's last phase trained again from its own adapter with the stop moved, on 8 GPUs
in one container, where no machine of our own has them free (the author, 5 Nov 2026: check B of DAgger at more users, on a
day every vendor's 8x H100 and H200 were sold out).

The image is the DAgger rows' environment as search/generalisation/setup.sh builds it on any machine (TRAIN_ONLY=1,
NO_MODEL=1), from a code bundle in the bucket (BUNDLE, which must hold the code the run trained on: code-final-x-20261105
is the four trees of NODE-final-x, the node DAgger at four users trained and played on, its .env left out), on
CUDA 13.0's Ubuntu 24.04 image, with this checkout's tools (search/generalisation/tools.sh) and longtrain.py at
~/gen/tools, as search/generalisation/modal_train.py builds a cell's. The container fetches the run's label sets and its
last phase from the bucket itself on an hour's token of a research account (bucket.py get, a manifest given in
MANIFEST), and keeps them and all it writes in the volume MODAL_JOBS_VOLUME under the job's name, ~/runs a link into it, committed
every five minutes, so that a container Modal restarts takes up where it stopped (fit's step checkpoints); Qwen3.5-9B
is kept in the volume MODAL_HF_VOLUME. The job's folder is its run's, under continual-arc-research's run supervisor (manifest,
console and record), which brain's run store copies from the research environment's volumes. A job is refused without
TTL_HOURS (at most Modal's 24) and a PURPOSE as detailed as a lease's, kept with its owner in job.json, which the billing
watcher reads.

Usage (the workspace's token read from Secret Manager into this process alone, as modal-train.sh reads it):
  BUNDLE=code-final-x-20261105 MANIFEST=train-dagger-u4-disjoint.txt TTL_HOURS=12 PURPOSE="..." \\
      bash modal-longtrain.sh dagger-u4-longtrain -- --run ~/runs/rt/dagger-u4-disjoint \\
      --out ~/runs/rt/dagger-u4-longtrain --gpus 8 --max-steps 1600 --patience 20
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent   # in the container, only this file, at /root
G = ROOT / "search" / "generalisation"
ENVIRONMENT = "research"
OUT = "/out"   # the volume MODAL_JOBS_VOLUME, one folder per job, its job.json at the top
HF = "/root/.cache/huggingface"
DAY = 24 * 60 * 60
COMMIT_EVERY = 300
CACHE = Path.home() / ".cache" / "gen-cache"   # the bundle and the data, which the image is built from
BASE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
MODEL = "Qwen/Qwen3.5-9B"


def image(tools: Path, bundle: str) -> modal.Image:
    """search/generalisation/modal_train.py's image, with longtrain.py beside the tools."""
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
BUNDLE = os.environ.get("BUNDLE", "") if modal.is_local() else ""
if TOOLS_DIR:
    subprocess.run(["bash", str(G / "tools.sh"), str(TOOLS_DIR)], check=True)
    (TOOLS_DIR / "longtrain.py").write_text((ROOT / "longtrain.py").read_text())
app = modal.App("NODE-longtrain", image=image(TOOLS_DIR, BUNDLE) if TOOLS_DIR and BUNDLE else modal.Image.debian_slim())
jobs = modal.Volume.from_name("MODAL_JOBS_VOLUME", create_if_missing=True)
hf = modal.Volume.from_name("MODAL_HF_VOLUME", create_if_missing=True)


@app.function(volumes={OUT: jobs, HF: hf}, timeout=DAY, retries=0, single_use_containers=True)
def longtrain(job: str, manifest: list[str], token: str, record: dict, argv: list[str]) -> None:
    home, out = Path.home(), Path(OUT) / job
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "job.json").exists():
        (out / "job.json").write_text(json.dumps({**record, "started": f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}"},
                                                 indent=1) + "\n")
    (out / "runs").mkdir(exist_ok=True)
    if not (home / "runs").is_symlink():
        (home / "runs").symlink_to(out / "runs")
    jobs.commit()
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
        runs = supervisor(tools)
        folder = runs.start({
            "repo": "continual-arc-training", "run": f"modal/{job}", "dir": str(out), "purpose": record["purpose"],
            "owner": record["owner"], "cwd": str(tools), "resume": (out / "manifest.json").exists(),
            "command": ["bash", "-c", 'cd "$HOME/prime-rl" && export PYTHONPATH="$HOME/loop" HF_HUB_OFFLINE=1 '
                        '&& exec uv run --no-sync python "$0" "$@"', str(tools / "longtrain.py"), *argv],
            "config": {k: record[k] for k in ("gpu", "cpu", "memory_gb", "timeout_hours", "bundle")},
            "lease": {"vendor": "modal", "id": job, "timeout_hours": record["timeout_hours"]},
            "launched_from": {"by": "modal_longtrain.py", "revisions": record["revisions"]}},
            detach=False, echo=True)
        code = runs.read(folder).get("exit_code")
    finally:
        ended.set()
        jobs.commit()
    if code:
        raise RuntimeError(f"{job}: longtrain.py exited {code} (its console: {job}/console.log)")


def supervisor(tools: Path):
    """continual-arc-research's run supervisor, the runs.py tools.sh puts beside the tools."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("runs", tools / "runs.py")
    runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runs)
    return runs


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[2] != "--":
        sys.exit("usage: modal_longtrain.py JOB -- LONGTRAIN_ARGS...")
    job, argv = sys.argv[1], sys.argv[3:]
    try:
        hours = float(os.environ.get("TTL_HOURS", ""))
    except ValueError:
        sys.exit("TTL_HOURS is required: the timeout at which Modal ends the job, its expected length plus a margin")
    if not 0 < hours * 3600 <= DAY:
        sys.exit("TTL_HOURS must be above 0 and at most 24, Modal's longest timeout")
    if not BUNDLE:
        sys.exit("BUNDLE names the code bundle the image is built from (bundles/generalisation/BUNDLE.tgz)")
    purpose = os.environ.get("PURPOSE", "")
    baselines = Path(os.environ.get("BASELINES", Path.home() / "continual-arc-baselines"))
    sys.path.insert(0, str(baselines / "src"))
    from continual_arc_fleet import leases   # the fleet's rule for a lease's reason, which a Modal job's purpose keeps
    if leases.thin(purpose, job):
        sys.exit(f"PURPOSE is required: why the job runs, at least {leases.MIN_PURPOSE_WORDS} words and "
                 f"{leases.MIN_PURPOSE_CHARS} characters, as detailed and explicit as a lease's (scripts/lease.py)")
    manifest = [line for line in Path(os.environ["MANIFEST"]).read_text().splitlines() if line.strip()]
    token = subprocess.run(["bash", str(baselines / "scripts/secret.sh"), "--node-token"], capture_output=True,
                           text=True, check=True).stdout.strip()
    owner = os.environ.get("OWNER") or os.environ.get("USER") or "unknown"
    gpu = os.environ.get("GPU", "H100:8")
    record = {"name": job, "gpu": gpu, "cpu": 64.0, "memory_gb": 512.0, "timeout_hours": hours, "bundle": BUNDLE,
              "command": "longtrain.py " + " ".join(argv), "owner": owner, "purpose": purpose, "secrets": [],
              "revisions": {"continual-arc-training": (TOOLS_DIR / "COMMIT").read_text().strip(), "bundle": BUNDLE}}
    fn = longtrain.with_options(gpu=gpu, cpu=64.0, memory=512 * 1024, timeout=int(hours * 3600))
    with modal.enable_output(), app.run(name=job, detach=True, environment_name=ENVIRONMENT):
        app.set_tags({"owner": re.sub(r"[^A-Za-z0-9._-]", "-", owner)[:63], "job": job, "repo": "continual-arc-training"})
        call = fn.spawn(job, manifest, token, record, argv)
        print(f"{job}: app {app.app_id}, call {call.object_id}, in Modal's {ENVIRONMENT} environment on {gpu}, ended by "
              f"Modal at {hours:g} h at the latest\n"
              f"  follow: bash {baselines}/scripts/modal.sh app logs {app.app_id}\n"
              f"  stop:   bash {baselines}/scripts/modal.sh app stop {app.app_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
