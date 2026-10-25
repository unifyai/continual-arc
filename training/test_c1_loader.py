"""c1's grid loader (rules/c1.md rule 4, also rules/c1e.md) on a user's session folder as prime-agent keeps
it with memory on: one log per instance, sessions/<id>.jsonl, each instance's RLM_SESSION_DIR session-artifacts/<id>."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import overfit_expert as ex

HERE = Path(__file__).resolve().parent


def _log(path: Path, test_input: str, pair: str) -> None:
    text = f"Demo pair 1 input (1x2):\n{pair}\nDemo pair 1 output (1x2):\n{pair}\nTest input (1x2):\n{test_input}\n"
    path.write_text(json.dumps({"text": text}) + "\n")


@pytest.mark.parametrize("rules", ["rules/c1.md", "rules/c1e.md"])
def test_the_loader_reads_this_instances_grids_not_an_earlier_instances(tmp_path, rules):
    state = tmp_path / "users" / "u000" / "system" / "state"
    (state / "sessions").mkdir(parents=True)
    _log(state / "sessions" / "older.jsonl", "7 7", "1 1")   # an earlier instance of another task, written first
    _log(state / "sessions" / "current.jsonl", "3 4", "2 2")
    (state / "session-artifacts" / "current").mkdir(parents=True)
    code = ex.c1_loader((HERE / rules).read_text()) + "\nimport json\nprint(json.dumps(load()))\n"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         env={"RLM_SESSION_DIR": str(state / "session-artifacts" / "current")})
    grids = json.loads(out.stdout.strip().splitlines()[-1])
    assert grids["Test input"] == [[3, 4]] and grids["Demo pair 1 input"] == [[2, 2]], out.stderr[-300:]
