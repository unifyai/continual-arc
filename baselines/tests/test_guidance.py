"""The executable grid-loading example in the shared expert guidance."""

import json
import os
import subprocess
import sys

from continual_arc_baselines.systems.guidance import EXPERT_GUIDANCE, PRIME_GRID_LOADER


def test_guidance_loader_reads_only_the_current_instance_and_refreshes_after_a_demo(tmp_path):
    sessions = tmp_path / "state" / "sessions"
    sessions.mkdir(parents=True)
    current = tmp_path / "state" / "session-artifacts" / "current"
    current.mkdir(parents=True)
    (sessions / "older.jsonl").write_text(json.dumps({"text": "Test input (1x2):\n7 7\n"}) + "\n")
    log = sessions / "current.jsonl"
    log.write_text(json.dumps({"text": "Test input (1x2):\n3 4\n"}) + "\n")
    code = PRIME_GRID_LOADER + "\nimport json\nprint(json.dumps(G))\n"

    def load():
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={**os.environ, "RLM_SESSION_DIR": str(current)}, timeout=10,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    assert load() == {"Test input": [[3, 4]]}
    with log.open("a") as stream:
        stream.write(json.dumps({"text": "Demo pair 1 input (1x2):\n1 2\n"
                                           "Demo pair 1 output (1x2):\n2 1\n"}) + "\n")
    assert load() == {"Test input": [[3, 4]], "Demo pair 1 input": [[1, 2]], "Demo pair 1 output": [[2, 1]]}
    assert EXPERT_GUIDANCE.count(PRIME_GRID_LOADER) == 1


def test_guidance_loader_rejects_malformed_grids(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    current = tmp_path / "session-artifacts" / "current"
    current.mkdir(parents=True)
    (sessions / "current.jsonl").write_text(json.dumps({"text":
        "Demo pair 1 input (1x2):\n1 12\n"
        "Demo pair 1 output (1x2):\n2\n"
        "Test input (1x2):\n3 4\n"}) + "\n")
    result = subprocess.run(
        [sys.executable, "-c", PRIME_GRID_LOADER + "\nimport json\nprint(json.dumps(G))\n"],
        capture_output=True, text=True, check=True, timeout=10,
        env={**os.environ, "RLM_SESSION_DIR": str(current)},
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"Test input": [[3, 4]]}
