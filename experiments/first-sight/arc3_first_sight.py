"""ARC-AGI-3 first-sight calibration on Modal: arc3-small's held-out test in the fresh order (every instance a fresh level
of one of the 10 held-out games, 4 a game, the learner wiped before each), played by the bare model on OpenRouter, each
container stopped once its calls cost more than its limit.

    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... OPENROUTER_API_KEY=... \
        uv run --with modal python arc3_first_sight.py OUT_DIR EFFORT LIMIT_USD MODEL [MODEL ...]

plays the whole test, one container a model, each model's run folder back as OUT_DIR/<model>.tgz;

        uv run --with modal python arc3_first_sight.py --slots SCHEDULE STEP[,STEP...] OUT_DIR EFFORT LIMIT_USD MODEL

plays the given slots of a schedule a run played (its run folder's schedule.json), one container a slot, each back as
OUT_DIR/<model>-step<STEP>.tgz: a model's levels that an earlier container's limit stopped.

The benchmark and its baselines are the public repository's release COMMIT; the pinned games are this machine's data root's
(``continual-arc data fetch --datasets arc3``). Modal runs it in the environment ``MODAL_ENVIRONMENT``, Modal's default
unless set. These are the ARC-AGI-3 columns of the paper's Table tab:first-sight.
"""
import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import modal

COMMIT = "v0.4.0"
REPO = "https://github.com/unifyai/continual-arc.git"
GAMES = Path.home() / ".cache/continual-arc/arc-agi-3"

app = modal.App("continual-arc-arc3-first-sight")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install(f"continual-arc-baselines @ git+{REPO}@{COMMIT}#subdirectory=baselines")  # brings continual-arc[arc3]
         .add_local_dir(str(GAMES), "/root/.cache/continual-arc/arc-agi-3"))
secret = modal.Secret.from_dict({"OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"]})


@app.function(image=image, secrets=[secret], timeout=5 * 3600, cpu=4, memory=8192)
def play(model: str, effort: str, limit: float, schedule: dict | None = None) -> bytes:
    import signal
    import subprocess
    import time

    work = Path("/tmp/work")
    work.mkdir()

    def spent() -> float:
        seen = {}
        for path in work.rglob("costs.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("account_charge") is not None:
                    seen[row.get("request_attempt_id") or row["source_id"]] = float(row["account_charge"])
        return sum(seen.values())

    kwargs = {"model": model, "max_tokens": 64000, "end": "finish", "empty_reply": "reprompt"}
    if schedule is None:
        served = ["--config", "arc3-small", "--order", "fresh", "--seed", "0"]
    else:
        (work / "schedule.json").write_text(json.dumps(schedule))
        served = ["--schedule", str(work / "schedule.json")]
    cmd = ["arc-baselines", "run", "--learner", "continual_arc_baselines.systems.bare:BareModelLearner",
           "--learner-kwargs", json.dumps(kwargs), "--llm-defaults", json.dumps({"reasoning_effort": effort}),
           *served, "--out", str(work / "run"), "--quiet"]
    log = open(work / "run.log", "w")
    proc = subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stopped = None
    while proc.poll() is None:
        if spent() > limit:
            stopped = f"stopped at {spent():.2f} USD, over the limit {limit}"
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            break
        time.sleep(20)
    (work / "calibration.json").write_text(json.dumps({"model": model, "effort": effort, "limit": limit,
                                                      "exit": proc.returncode, "stopped": stopped,
                                                      "spent": spent()}, indent=1))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(work, arcname=model.replace("/", "_"))
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", nargs=2, metavar=("SCHEDULE", "STEPS"))
    parser.add_argument("out", type=Path)
    parser.add_argument("effort")
    parser.add_argument("limit", type=float)
    parser.add_argument("models", nargs="+")
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = []  # (file name, model, schedule)
    if a.slots:
        played = json.loads(Path(a.slots[0]).read_text())
        for step in map(int, a.slots[1].split(",")):
            one = dict(played, name=f"{played['name']}-step{step}", slots=[played["slots"][step]])
            jobs += [(f"{m.replace('/', '_')}-step{step}", m, one) for m in a.models]
    else:
        jobs = [(m.replace("/", "_"), m, None) for m in a.models]
    with modal.enable_output(), app.run(environment_name=os.environ.get("MODAL_ENVIRONMENT")):
        calls = [(name, play.spawn(model, a.effort, a.limit, schedule)) for name, model, schedule in jobs]
        for name, call in calls:
            (a.out / f"{name}.tgz").write_bytes(call.get())
            print(f"{name}: back", flush=True)


if __name__ == "__main__":
    main()
