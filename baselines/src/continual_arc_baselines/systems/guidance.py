"""The expert guidance shared by every guided harness and the training oracle.

``EXPERT_GUIDANCE`` is the complete policy. Harness-specific paths are supplied
as runtime context, not substituted into or appended as another policy.
"""

from hashlib import sha256

PRIME_GRID_LOADER = r'''import os, re
from pathlib import Path

def load():
    session = Path(os.path.normpath(os.environ["RLM_SESSION_DIR"]))
    log = session.parent.parent / "sessions" / (session.name + ".jsonl")
    text = log.read_text().replace("\\n", "\n")
    grids = {}
    pattern = r"(Demo pair \d+ (?:input|output)|Test input) \((\d+)x(\d+)\):\n"
    for match in re.finditer(pattern, text):
        name, height, width = match[1], int(match[2]), int(match[3])
        rows = [row.split() for row in text[match.end():].split("\n")[:height]]
        if name not in grids and len(rows) == height and all(
            len(row) == width and all(cell in "0123456789" and len(cell) == 1 for cell in row)
            for row in rows
        ):
            grids[name] = [[int(cell) for cell in row] for row in rows]
    return grids

G = load()
print(sorted(G))'''

EXPERT_GUIDANCE = """\
Expert guidance

Actions. Your action is read only from the plain text of your reply, never from Python output or a tool call:
printing JSON or calling a tool named submit, request_demos, finish or answer does not submit an action. Keep each
turn to a few tool calls, then end your reply with exactly one action JSON on the last line:
{"action": "request_demos"}, {"action": "submit", "grid": [[0, 1], [2, 3]]}, or {"action": "finish"}.
Finish only after the instance is over. Replies without a valid action waste the turn; repeated invalid replies
end the instance charged in full. A submission has happened only when you receive its verdict.
When told "Decide now" or "could not find a valid action", make no further tool call: reply immediately with an
action. Submit only a program-produced grid already printed in this conversation, copying it exactly; otherwise
request a demonstration. If the instance is over, finish.

Library. Use the programs folder in the harness's workspace, or the library directory supplied with this instance.
Keep two files per task, named for its task id: <task id>.py defining transform(grid), which takes and returns a grid
as lists of lists of integers; and <task id>.json containing every received demonstration pair as a list of
{"input": grid, "output": grid} objects. Reading kept pairs is free, and they count as demonstrations you have.
At the start of an instance, look for the task's memory among those shown to you, then for its files if no memory
is shown. Load its kept pairs and program, if any, before requesting new demonstrations.

Grids. Never retype grids. Load the grids exactly as provided in the current conversation or its log, and load
cached pairs from the task's JSON file. In Prime Agent, when RLM_SESSION_DIR is available, the following loader reads
only this instance's log. Rerun it after each new demonstration. In other harnesses, use their tools to read the
current conversation's grids instead.
```python
""" + PRIME_GRID_LOADER + """
```

Solve. If you have no demonstration pair, cached or new, request one before submitting: the test input alone does
not establish the rule. If a kept transform exists, run it on every pair you have. If it reproduces them all, run it
on the test input and submit that output before requesting more demonstrations. Otherwise, or after a wrong answer, write or
repair transform and check it against every pair, cached and new; request another demonstration when needed.
These checks are free. As soon as every pair matches, print json.dumps(transform(test_input)) and submit that grid
instead of requesting more demonstrations. Submit only program-produced grids, and never resubmit a grid already
judged wrong in this instance.

Store. After the verdict, in one Python call if possible, add every received pair to the task's JSON file and save
the program that reproduces every pair to its Python file, replacing the older version. If no program reproduces
all pairs, leave the old program file unchanged. Keep one memory for the task, named for its task id, that lasts
between instances and records where the two files are and whether the program reproduces every kept pair. Then
finish.
"""

EXPERT_GUIDANCE_SHA256 = sha256(EXPERT_GUIDANCE.encode("utf-8")).hexdigest()
