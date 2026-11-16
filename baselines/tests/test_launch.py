"""scripts/launch-cell.sh: a cell launched detached into its run directory, its console added to console.log there; a
cell that holds a run never launched from its start again, RESUME=1 going on with it (--resume) and skipping a cell that
holds none. A stand-in for uv takes the launch, so nothing is played."""

import os
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "launch-cell.sh"
FAKE_UV = """#!/usr/bin/env bash
# uv run ... python -m continual_arc_baselines run ... --out DIR ...: its arguments noted in DIR, which it marks as a run's
out=""; prev=""
for arg in "$@"; do [ "$prev" = "--out" ] && out=$arg; prev=$arg; done
echo "played with $*"
echo "$*" >> "$out/argv.txt"
echo '{}' > "$out/meta.json"
"""


def launch(tmp_path: Path, seed: int = 0, **env: str) -> str:
    """launch-cell.sh for one program-library cell of arc1-smoke at ``seed``, with ``env``, the stand-in uv first on the
    path (and a home of its own, whose tools the script puts before it)."""
    tools = tmp_path / "bin"
    tools.mkdir(exist_ok=True)
    (tools / "uv").write_text(FAKE_UV)
    (tools / "uv").chmod(0o755)
    environ = {**os.environ, "HOME": str(tmp_path / "home"), "PATH": f"{tools}:{os.environ['PATH']}",
               "RUNS": str(tmp_path / "runs"), **env}
    return subprocess.run(["bash", str(SCRIPT), "method", "luna6-low", "workstreams", str(seed), "arc1-smoke"],
                          env=environ, capture_output=True, text=True, check=True).stdout


def launched(cell: Path, n: int) -> list[str]:
    """The cell's first ``n`` launches as the stand-in noted them, waited for, since a cell runs detached."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (cell / "argv.txt").exists() and len(lines := (cell / "argv.txt").read_text().splitlines()) >= n:
            return lines
        time.sleep(0.05)
    raise AssertionError(f"{cell} was not launched {n} times")


def test_a_cell_is_never_launched_from_its_start_again_and_resume_goes_on_with_it(tmp_path):
    cell = tmp_path / "runs" / "arc1" / "M-BareProgramLibraryLearner-luna6-low-smoke-ws0"
    assert launch(tmp_path, PILOT="5").strip() == f"launched {cell}"
    first, = launched(cell, 1)
    assert "--stop-after 5" in first and "--resume" not in first and f"--out {cell}" in first
    assert launch(tmp_path).strip() == f"holds a run: {cell} (RESUME=1 goes on with it)"
    assert launch(tmp_path, RESUME="1", PILOT="5").strip() == f"relaunched {cell}"
    _, again = launched(cell, 2)
    assert again.endswith("--resume") and "--stop-after" not in again
    deadline = time.monotonic() + 10
    while (cell / "console.log").read_text().count("played with") < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (cell / "console.log").read_text().count("played with") == 2   # every launch's console, in one
    other = tmp_path / "runs" / "arc1" / "M-BareProgramLibraryLearner-luna6-low-smoke-ws1"
    assert launch(tmp_path, seed=1, RESUME="1").strip() == f"holds no run to resume: {other}"
    assert not other.exists()
