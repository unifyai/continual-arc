"""A round's memory-use counters (continual_arc_training.memory) on fake data: a hand-built state tree stands in for a
user's harness state, its commits per round (git, as continual_arc_baselines.statelog makes them) for the state's
history, and a hand-built captured game (the instance hook's shape) for a round's play."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from continual_arc_training import memory

# -- snapshot_state / area_of / diff_snapshot ----------------------------------------------------------------

def test_snapshot_state_reads_path_size_mtime_hash_and_skips_excluded(tmp_path):
    state = tmp_path / "state"
    (state / "agent" / "skills" / "solve-grids").mkdir(parents=True)
    (state / "agent" / "skills" / "solve-grids" / "SKILL.md").write_text("do the thing")
    (state / "workspace" / "node_modules" / "pkg").mkdir(parents=True)
    (state / "workspace" / "node_modules" / "pkg" / "index.js").write_text("module.exports = 1")
    (state / "workspace" / "programs").mkdir(parents=True)
    (state / "workspace" / "programs" / "p1.py").write_text("print(1)")

    snapshot = memory.snapshot_state(state)

    assert "agent/skills/solve-grids/SKILL.md" in snapshot
    assert not any(p.startswith("workspace/node_modules") for p in snapshot)   # excluded: a runtime, not what it learned
    entry = snapshot["agent/skills/solve-grids/SKILL.md"]
    assert entry["size"] == len("do the thing")
    assert entry["hash"] == memory._hash_file(state / "agent" / "skills" / "solve-grids" / "SKILL.md",
                                              memory.SNAPSHOT_MAX_FILE_BYTES)
    assert isinstance(entry["mtime"], float)


def test_snapshot_state_of_a_missing_directory_is_empty(tmp_path):
    assert memory.snapshot_state(tmp_path / "never-started") == {}


def test_snapshot_state_does_not_follow_the_home_prime_agent_link(tmp_path):
    """home/.prime/agent is a symlink to agent/ (prime_agent.py's own _link_home_agent_dir): os.walk defaults to
    followlinks=False, so a file under agent/skills/ must appear exactly once, under its real path, never a
    second time (and misclassified) under home/.prime/agent/skills/."""
    state = tmp_path / "state"
    (state / "agent" / "skills" / "a").mkdir(parents=True)
    (state / "agent" / "skills" / "a" / "SKILL.md").write_text("v1")
    (state / "home" / ".prime").mkdir(parents=True)
    (state / "home" / ".prime" / "agent").symlink_to(state / "agent", target_is_directory=True)

    snapshot = memory.snapshot_state(state)
    assert list(snapshot) == ["agent/skills/a/SKILL.md"]   # not also under home/.prime/agent/...


def test_area_of_classifies_every_known_path():
    assert memory.area_of("agent/skills/solve-grids/SKILL.md") == "skills"
    assert memory.area_of("agent/harness/memories.json") == "harness_store"
    assert memory.area_of("AGENTS.md") == "notes"
    assert memory.area_of("home/project/AGENTS.md") == "notes"
    assert memory.area_of("workspace/programs/p1.py") == "programs"
    assert memory.area_of("workspace/notes.md") == "workspace_other"
    assert memory.area_of("agent/models.json") == "other"   # bookkeeping config, not skills, not harness/
    assert memory.area_of("home/.bashrc") == "other"


def test_diff_snapshot_counts_added_changed_and_removed_per_area():
    previous = {
        "agent/skills/a/SKILL.md": {"size": 10, "mtime": 1.0, "hash": "h1"},
        "workspace/programs/old.py": {"size": 5, "mtime": 1.0, "hash": "h2"},
        "agent/harness/memories.json": {"size": 1, "mtime": 1.0, "hash": "h3"},
    }
    current = {
        "agent/skills/a/SKILL.md": {"size": 12, "mtime": 2.0, "hash": "h1-changed"},   # changed
        "workspace/programs/new.py": {"size": 7, "mtime": 2.0, "hash": "h4"},          # added; old.py removed
        "agent/harness/memories.json": {"size": 1, "mtime": 1.0, "hash": "h3"},        # unchanged: not counted
    }
    diff = memory.diff_snapshot(previous, current)
    assert diff["skills"] == {"added": 0, "changed": 1, "removed": 0}
    assert diff["programs"] == {"added": 1, "changed": 0, "removed": 1}
    assert diff["workspace_other"] == {"added": 0, "changed": 0, "removed": 0}
    assert diff["harness_store"] == {"added": 0, "changed": 0, "removed": 0}
    assert diff["notes"] == {"added": 0, "changed": 0, "removed": 0}
    assert diff["other"] == {"added": 0, "changed": 0, "removed": 0}


def test_diff_snapshot_against_nothing_counts_everything_as_added():
    current = {"agent/skills/a/SKILL.md": {"size": 1, "mtime": 1.0, "hash": "h"}}
    diff = memory.diff_snapshot({}, current)
    assert diff["skills"]["added"] == 1 and diff["skills"]["changed"] == 0 and diff["skills"]["removed"] == 0


def test_diff_snapshot_falls_back_to_size_and_mtime_when_a_hash_is_missing():
    """A file over SNAPSHOT_MAX_FILE_BYTES carries no hash; a change must still be caught."""
    previous = {"workspace/big.bin": {"size": 100, "mtime": 1.0, "hash": None}}
    current = {"workspace/big.bin": {"size": 100, "mtime": 2.0, "hash": None}}   # touched, same size, no hash
    assert memory.diff_snapshot(previous, current)["workspace_other"]["changed"] == 1
    unchanged = memory.diff_snapshot(previous, previous)
    assert unchanged["workspace_other"] == {"added": 0, "changed": 0, "removed": 0}


def test_snapshot_state_excludes_bookkeeping_that_changes_every_game(tmp_path):
    """A fake tree mirroring the real layout (checked against a harness row's run): session
    transcripts, the daemon's own command journals, its logs, its session leases, its per-session ledger, the
    supervisor's lock directory and the harness's own provider/model config (rewritten every process start, a
    fresh proxy port) all change every game by construction and must not show up as anything persisting, let
    alone as a save; a deliberate write under workspace/programs/ still does."""
    state = tmp_path / "state"
    bookkeeping = ("sessions/s1.jsonl", "session-artifacts/kernel-state.dill",
                  "session-artifacts/semantic-edges.jsonl", "agent/daemon-workers/w1.recovery.jsonl",
                  "agent/logs/daemon.log", "agent/session-leases/lease-1.json",
                  "agent/rlm-ledger/d8acaa0a4a41256a.jsonl", "home/.prime/supervisor-owners/owner-1",
                  "agent/auth.json", "agent/models.json", "agent/settings.json")
    for rel in bookkeeping:
        path = state / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changes every game")

    before = memory.snapshot_state(state)
    assert before == {}   # every bit of it is bookkeeping: nothing even lands in the snapshot

    (state / "workspace" / "programs").mkdir(parents=True)
    (state / "workspace" / "programs" / "t.py").write_text("print('solved')")
    for rel in ("sessions/s2.jsonl", "agent/logs/daemon2.log", "agent/models.json"):   # a second game's worth,
        path = state / rel                                                           # models.json a fresh port
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("yet more bookkeeping")

    after = memory.snapshot_state(state)
    diff = memory.diff_snapshot(before, after)
    assert diff["programs"] == {"added": 1, "changed": 0, "removed": 0}
    assert all(d == {"added": 0, "changed": 0, "removed": 0} for area, d in diff.items() if area != "programs")


# -- read_areas / count_reads / game_memory_outcome -----------------------------------------------------------

# A trimmed excerpt of a real captured call's system prompt (a harness row's run, round 0):
# prime-agent's bundled-skill catalogue lists each skill's *installed* path, which contains "agent/skills/" as a
# literal substring of "prime-**agent/skills/**" -- the measured false positive (F1): a plain substring search hit
# this in 3,400 of 3,861 calls, in every one of the 40 captured docs checked, though no user has an agent/skills/
# directory at all.
REAL_SYSTEM_PROMPT_EXCERPT = (
    "You have access to the following skill modules (pre-imported): `goal`, `rlm_heartbeat`, `agent_message`.\n"
    "<skill>\n"
    "  <name>goal</name>\n"
    "  <type>python</type>\n"
    "  <description>Start or mark complete a goal.</description>\n"
    "  <location>/home/user/.local/share/prime-agent/skills/goal/SKILL.md</location>\n"
    "</skill>\n"
    "<skill>\n"
    "  <name>rlm-heartbeat</name>\n"
    "  <type>python</type>\n"
    "  <python_import>rlm_heartbeat</python_import>\n"
    "  <location>/home/user/.local/share/prime-agent/skills/rlm-heartbeat/SKILL.md</location>\n"
    "</skill>\n"
)


TASK = "task-1a2b3c4d"
# A program a teacher kept for TASK, four of whose lines are marks (25 characters or more, not generic).
PROGRAM = '''import numpy as np


def transform(grid):
    """Fill every enclosed zero with the color that borders the most cells."""
    g = np.array(grid)
    border_color = int(np.bincount(g[g > 0]).argmax())
    for r, c in zip(*np.where(g == 0)):
        if 0 < r < g.shape[0] - 1 and 0 < c < g.shape[1] - 1:
            g[r, c] = border_color
    return g.tolist()
'''
# What the same task's storer writes over it with: it shares none of PROGRAM's marks.
REWRITTEN = '''def transform(grid):
    """Mirror the grid left to right, then paint its first row with the last row's colors."""
    out = [list(reversed(row)) for row in grid]
    out[0] = list(out[-1])
    return out
'''
MARKS = memory.marks(PROGRAM)


def _call(content: str | None = None, reasoning: str | None = None, tool_args: str | None = None,
          tool_name: str = "ipython", tool_results: tuple[str, ...] | list[str] = (), system: str | None = None,
          turn: int | None = 0, phase: str = "turn", tools: bool = True) -> dict:
    """One captured call (the shape the instance hook saves): ``content``/``reasoning``/``tool_args`` build the call's
    own reply (response.choices[0].message), ``tool_args`` a call of
    ``tool_name``; ``tool_results`` are the tool messages the request ends with, the results of the tool calls the
    reply before it made (as the real ipython-execution output is); ``system`` is the system prompt, present in every
    real call's request but never read -- here to prove it stays that way. By default the call is the student's own
    in play: phase "turn", with ipython offered (``tools``), as prime-agent's agent offers it; its context summarizer
    and its refinement offer none."""
    message: dict = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning"] = reasoning
    if tool_args is not None:
        message["tool_calls"] = [{"function": {"name": tool_name, "arguments": tool_args}}]
    messages = [{"role": "system", "content": system or "a generic system prompt, no markers"},
                {"role": "user", "content": "here is the puzzle: a game message, never read"}]
    if tool_results:
        messages.append({"role": "assistant", "tool_calls": [{"function": {"name": "ipython", "arguments": "{}"}}
                                                             for _ in tool_results]})
        messages += [{"role": "tool", "content": result} for result in tool_results]
    request: dict = {"messages": messages}
    if tools:
        request["tools"] = [{"type": "function", "function": {"name": "ipython"}}]
    return {"turn": turn, "phase": phase, "request": request, "response": {"choices": [{"message": message}]}}


# the areas read by a reference to them

def test_read_areas_finds_a_reference_in_the_reply_content():
    assert memory.read_areas(_call(content="opening agent/skills/solve-grids/SKILL.md now"), []) == {"skills"}


def test_read_areas_finds_a_reference_in_the_replys_reasoning():
    assert memory.read_areas(_call(reasoning="let me check AGENTS.md for what I stored last time"), []) == {"notes"}


def test_read_areas_finds_a_reference_in_a_tool_calls_arguments():
    assert memory.read_areas(_call(tool_args=json.dumps({"code": "open('agent/skills/solve-grids/SKILL.md')"})),
                             []) == {"skills"}


def test_read_areas_finds_a_reference_in_a_tool_result():
    """A tool result (the ipython execution output the request ends with) must count: this is where the content of a
    file actually read appears."""
    assert memory.read_areas(_call(tool_results=["loaded agent/harness/notes.json successfully"]), []) == \
        {"harness_store"}


def test_read_areas_never_counts_the_system_prompt_or_the_game_message():
    """The system prompt and the benchmark's own game message are both in every call's request, but neither is the
    model's own output or a tool result: a reference there, or a program's whole source, must never count."""
    system = PROGRAM + "\ngo read AGENTS.md and agent/skills/x/SKILL.md right now"
    assert memory.read_areas(_call(system=system, content="ok"), MARKS) == set()


def test_read_areas_rejects_prime_agents_bundled_skill_catalogue():
    """F1, reproduced directly: a real system prompt's bundled-skill catalogue lists each skill under
    .../prime-agent/skills/<name>/SKILL.md, which contains the literal substring "agent/skills/" -- but it must
    never register as a skills read, both because it is in the system prompt (never read at all) and because
    "prime-agent/skills/" is not "agent/skills/" (the anchored pattern)."""
    call = _call(system=REAL_SYSTEM_PROMPT_EXCERPT, content="ok, I see the puzzle")
    assert memory.read_areas(call, []) == set()
    # even if that same bundled-catalogue text somehow reached a field that is read (belt and suspenders: the
    # pattern itself, not only the field restriction, must reject "prime-agent/skills/")
    assert memory.read_areas(_call(content=REAL_SYSTEM_PROMPT_EXCERPT), []) == set()


def test_read_areas_accepts_the_users_own_skills_path_including_through_the_prime_link():
    assert memory.read_areas(_call(content="agent/skills/solve-grids/SKILL.md"), []) == {"skills"}
    assert memory.read_areas(_call(content="~/.prime/agent/skills/solve-grids/SKILL.md"), []) == {"skills"}


def test_read_areas_is_empty_with_no_reference():
    assert memory.read_areas(_call(content="just thinking out loud, nothing on disk"), MARKS) == set()


def test_read_areas_can_report_more_than_one_area_for_one_call():
    call = _call(content="AGENTS.md and agent/skills/x/SKILL.md both", tool_results=[PROGRAM])
    assert memory.read_areas(call, MARKS) == {"notes", "skills", "programs"}


# the program read by its source

def test_a_program_opened_through_a_path_built_with_os_path_join_is_a_read():
    """The student builds the path, so no call of the game spells out programs/<task>.py: the read is the program's
    source coming back in the tool result the next call answers."""
    code = json.dumps({"code": "import os\nprint(open(os.path.join(workspace, 'programs', task + '.py')).read())"})
    asked = _call(tool_args=code)
    shown = _call(content="the kept program fills the enclosed zeros", tool_results=[PROGRAM])
    assert memory.read_areas(asked, MARKS) == set()
    assert memory.read_areas(shown, MARKS) == {"programs"}


def test_a_path_string_alone_is_no_read():
    assert memory.read_areas(_call(content=f"I'll save this to programs/{TASK}.py once it works"), MARKS) == set()


def test_writing_over_the_program_is_no_read():
    """The student names the program's path when it writes over it, and the source in its tool call is its own new
    one, not the kept program coming back in a tool result."""
    wrote = _call(tool_args=json.dumps({"code": f"open('programs/{TASK}.py', 'w').write({REWRITTEN!r})"}))
    told = _call(content=f"saved programs/{TASK}.py", tool_results=["wrote 175 characters"])
    assert memory.read_areas(wrote, MARKS) == memory.read_areas(told, MARKS) == set()


def test_deleting_the_program_or_listing_its_folder_is_no_read():
    removed = _call(tool_args=json.dumps({"code": f"import os\nos.remove('programs/{TASK}.py')"}))
    listed = _call(tool_results=[f"['{TASK}.py', '{TASK}.json', 'task-99999999.py']"])
    assert memory.read_areas(removed, MARKS) == memory.read_areas(listed, MARKS) == set()


def test_a_call_to_a_tool_that_does_not_exist_is_no_read():
    """Only ipython is offered: a call of "bash" comes back an error, whatever path it names."""
    asked = _call(tool_args=json.dumps({"command": f"cat programs/{TASK}.py"}), tool_name="bash")
    refused = _call(tool_results=["Error: unknown tool 'bash'"])
    assert memory.read_areas(asked, MARKS) == memory.read_areas(refused, MARKS) == set()


def test_the_harnesss_context_summarizer_never_reads_a_program():
    """prime-agent's context summarizer calls the student's model, in play, with the conversation to summarize and no
    tools: the program's path in its summary and the program's source in its request are still no read, even with
    the source in a tool message its request ends with."""
    summary = _call(content=f"The agent opened programs/{TASK}.py and found a transform.", tool_results=[PROGRAM],
                    tools=False)
    assert memory.read_areas(summary, MARKS) == set()


def test_the_closing_turns_after_the_verdict_never_read():
    """A teacher's storer works in the closing turns (phase "after"), opening and writing the programs it keeps,
    once the game has drawn on all it will."""
    storer = _call(content=f"updating programs/{TASK}.py", tool_results=[PROGRAM], phase="after")
    assert memory.read_areas(storer, MARKS) == set()


def test_every_tool_result_a_request_ends_with_is_read():
    """A reply that makes several tool calls at once gets their results back as that many tool messages: the program
    can come back in any of them, not only the newest."""
    call = _call(tool_results=[PROGRAM, "[[0, 1], [1, 0]]", "Submitted."])
    assert memory.answered(call).startswith("import numpy as np")
    assert memory.read_areas(call, MARKS) == {"programs"}


def test_a_program_comes_back_only_with_two_of_its_marks_or_its_one():
    """One distinctive line alone, which another program can share, is not the program; a generic line never marks
    one."""
    assert len(MARKS) == 4
    assert memory.read_areas(_call(tool_results=[MARKS[1]]), MARKS) == set()
    assert memory.read_areas(_call(tool_results=["\n".join(MARKS[1:3])]), MARKS) == {"programs"}
    assert memory.read_areas(_call(tool_results=[MARKS[1]]), MARKS[1:2]) == {"programs"}
    assert memory.marks("from collections import Counter\n" + PROGRAM) == MARKS
    assert memory.read_areas(_call(tool_results=[PROGRAM]), []) == set()   # a program with no mark is never opened


def test_count_reads_sums_areas_over_every_call_of_a_game():
    calls = [_call(tool_results=[PROGRAM]), _call(content="nothing here"),
             _call(content="agent/skills/x/SKILL.md", tool_results=[PROGRAM])]
    assert memory.count_reads([memory.read_areas(c, MARKS) for c in calls]) == \
        {"programs": 2, "skills": 1, "harness_store": 0, "notes": 0}


def _turn(action_type: str) -> dict:
    return {"action": {"type": action_type}}


def test_game_memory_outcome_reused_program_true_at_or_before_the_first_submission():
    """The real action vocabulary is RequestDemos/Submit/GiveUp (never "Observation"): the program coming back at
    the same turn as the first Submit, or any turn before it, counts."""
    doc = {"turns": [_turn("RequestDemos"), _turn("Submit")],
           "calls": [_call(tool_results=[PROGRAM], turn=1), _call(content="ok", turn=2)]}
    call_areas = [memory.read_areas(c, MARKS) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is True


def test_game_memory_outcome_reused_program_false_when_the_read_is_after_the_first_submission():
    doc = {"turns": [_turn("Submit"), _turn("Submit")],
           "calls": [_call(content="just thinking", turn=0), _call(tool_results=[PROGRAM], turn=1)]}
    call_areas = [memory.read_areas(c, MARKS) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is False


def test_game_memory_outcome_reused_program_true_with_no_submission_at_all():
    """A game that read its program and then gave up without ever submitting still had it: there is no "before
    submitting" boundary to exclude it by."""
    doc = {"turns": [_turn("RequestDemos"), _turn("GiveUp")], "calls": [_call(tool_results=[PROGRAM], turn=0)]}
    call_areas = [memory.read_areas(c, MARKS) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is True


def test_game_memory_outcome_reused_program_false_with_no_program_read():
    doc = {"turns": [_turn("Submit")], "calls": [_call(content=f"programs/{TASK}.py", turn=0)]}
    call_areas = [memory.read_areas(c, MARKS) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is False


def test_game_memory_outcome_solved_without_demo():
    solved_no_demo = {"solved": True, "turns": [_turn("Submit")]}
    solved_with_demo = {"solved": True, "turns": [_turn("RequestDemos"), _turn("Submit")]}
    failed_no_demo = {"solved": False, "turns": [_turn("GiveUp")]}
    assert memory.game_memory_outcome(solved_no_demo, [])["solved_without_demo"] is True
    assert memory.game_memory_outcome(solved_with_demo, [])["solved_without_demo"] is False
    assert memory.game_memory_outcome(failed_no_demo, [])["solved_without_demo"] is False


# -- round_report / recount / summary_line, wired against a fake run folder ----------------------------------

# git with an identity and settings of its own, whatever the host's configuration (as baselines' StateLog runs it)
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def _round_dir(run_dir: Path, round_: int) -> Path:
    return run_dir / "rounds" / f"r{round_:04d}"


def _write_doc(run_dir: Path, round_: int, user: int, turns: list[dict], calls: list[dict]) -> None:
    captured = _round_dir(run_dir, round_) / "captured"
    captured.mkdir(parents=True, exist_ok=True)
    with gzip.open(captured / f"u{user:03d}.json.gz", "wt") as fp:
        json.dump({"task_label": TASK, "turns": turns, "calls": calls}, fp)


def _write_state(run_dir: Path, user: int, files: dict[str, str]) -> Path:
    state = memory.system_dir(run_dir, user) / "state"
    state.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = state / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        os.utime(path, (time.time(), time.time()))
    return state


def _commit_state(run_dir: Path, user: int, round_: int) -> None:
    """The user's state as round ``round_`` ended, committed as baselines' StateLog commits it: into the system
    folder's state.git, listed by round in its state-log.jsonl."""
    system = memory.system_dir(run_dir, user)
    repo, tree = system / "state.git", system / "state"
    if not repo.exists():
        subprocess.run(["git", "init", "--quiet", "--bare", str(repo)], env=GIT_ENV, check=True)
        subprocess.run(["git", f"--git-dir={repo}", "config", "core.bare", "false"], env=GIT_ENV, check=True)
    git = ["git", f"--git-dir={repo}", f"--work-tree={tree}"]
    subprocess.run([*git, "add", "--all", "--force", "."], cwd=tree, env=GIT_ENV, check=True)
    subprocess.run([*git, "commit", "--quiet", "--allow-empty", "-m", f"round {round_}"], cwd=tree, env=GIT_ENV,
                   check=True)
    commit = subprocess.run(["git", f"--git-dir={repo}", "rev-parse", "HEAD"], env=GIT_ENV, check=True,
                            capture_output=True, text=True).stdout.strip()
    with (system / "state-log.jsonl").open("a") as fp:
        fp.write(json.dumps({"round": round_, "commit": commit}) + "\n")


def test_kept_at_start_is_the_program_as_the_round_before_left_it(tmp_path):
    """The storer writes over the program after the verdict, so a round ends with another program in the folder than
    the one its game began with: a game is read against the state's commit at the end of the round before."""
    run_dir, path = tmp_path / "run", f"workspace/programs/{TASK}.py"
    _write_state(run_dir, 0, {path: PROGRAM})
    _commit_state(run_dir, 0, 0)
    _write_state(run_dir, 0, {path: REWRITTEN})
    _commit_state(run_dir, 0, 1)
    assert memory.kept_at_start(run_dir, 0, 1, path, stateless=False) == PROGRAM
    assert memory.kept_at_start(run_dir, 0, 2, path, stateless=False) == REWRITTEN
    assert memory.kept_at_start(run_dir, 0, 1, "workspace/programs/task-99999999.py", stateless=False) is None
    assert memory.kept_at_start(run_dir, 0, 0, path, stateless=False) is None   # a stream's round 0 begins from nothing
    assert memory.kept_at_start(run_dir, 0, 1, path, stateless=True) is None    # and so does a wiped row's every game


def test_forked_from_names_the_commit_a_play_began_from(tmp_path):
    """What a play's frozen.json says it began from, and the plays that began from nothing."""
    play = tmp_path / "play"
    play.mkdir()

    def write(doc: dict) -> None:
        (play / "frozen.json").write_text(json.dumps(doc))

    assert memory.forked_from(play, 0) is None                                           # a stream has no frozen.json
    write({"from": {"round": 199, "system": "/s", "commit": "c0", "relocated": 1, "databases": 0}})   # a frozen test
    assert memory.forked_from(play, 0) == "c0"
    write({"from": None})                                                             # a test of a run with no stream
    assert memory.forked_from(play, 0) is None
    write({"memory": "none: played from an empty state"})                             # a reference play
    assert memory.forked_from(play, 0) is None
    write({"memory": "every test user a copy of the run's one state, relocated", "relocated": {"files": 1}})
    with pytest.raises(ValueError, match="no history holds"):                         # a test of a run with no commits
        memory.forked_from(play, 0)


def _forked_play(run_dir: Path) -> None:
    """A frozen test of one user, its user forked from the commit of a state that kept PROGRAM, then writing over it in
    its game, as the baselines' seeding and the round's end leave it."""
    path = f"workspace/programs/{TASK}.py"
    _write_state(run_dir, 0, {path: PROGRAM})
    _commit_state(run_dir, 0, 199)
    commit = memory._committed(memory.system_dir(run_dir, 0), 199)
    (run_dir / "frozen.json").write_text(json.dumps({"from": {"round": 199, "system": "/stream", "commit": commit}}))
    _write_state(run_dir, 0, {path: REWRITTEN})
    _commit_state(run_dir, 0, 200)


def test_round_report_of_a_forked_play_reads_the_program_its_fork_began_with(tmp_path):
    """A frozen test is round 0 but begins from a kept state: its student opens the program it was forked with, and the
    program it writes over that one and prints is no kept program read."""
    run_dir = tmp_path / "held-out" / "play"
    _forked_play(run_dir)
    _write_doc(run_dir, 0, 0, turns=[_turn("Submit")], calls=[_call(tool_results=[PROGRAM], turn=0)])
    report = memory.round_report(run_dir, 0, users=1, round_dir_of=_round_dir, stateless=False)
    assert report["users"]["0"]["reads"]["programs"] == 1 and report["users"]["0"]["reused_program"] is True

    _write_doc(run_dir, 0, 0, turns=[_turn("Submit")], calls=[_call(tool_results=[REWRITTEN], turn=0)])
    report = memory.round_report(run_dir, 0, users=1, round_dir_of=_round_dir, stateless=False)
    assert report["users"]["0"]["reads"]["programs"] == 0 and report["users"]["0"]["reused_program"] is False


def test_round_report_a_wiped_rows_test_reads_no_program(tmp_path):
    """A wiped row's test is forked as any other's, but its wipe leaves its game nothing to read."""
    run_dir = tmp_path / "held-out" / "play"
    _forked_play(run_dir)
    _write_doc(run_dir, 0, 0, turns=[_turn("Submit")], calls=[_call(tool_results=[PROGRAM], turn=0)])
    report = memory.round_report(run_dir, 0, users=1, round_dir_of=_round_dir, stateless=True)
    assert report["users"]["0"]["reads"]["programs"] == 0


def test_round_report_writes_memory_json_with_per_user_and_total_counts(tmp_path):
    run_dir = tmp_path / "run"
    _write_state(run_dir, 0, {"agent/skills/a/SKILL.md": "v1"})
    _write_doc(run_dir, 0, 0, turns=[_turn("RequestDemos"), _turn("Submit")],
               calls=[_call(content="agent/skills/a/SKILL.md", turn=0)])

    report = memory.round_report(run_dir, 0, users=1, round_dir_of=_round_dir, stateless=False)

    assert report["round"] == 0
    assert report["users"]["0"]["memory"]["skills"]["added"] == 1
    assert report["users"]["0"]["reads"]["skills"] == 1
    assert report["total"]["memory"]["skills"]["added"] == 1
    assert "reused_program" in report["users"]["0"] and "solved_without_demo" in report["users"]["0"]
    assert (_round_dir(run_dir, 0) / "memory.json").is_file()
    assert (_round_dir(run_dir, 0) / "memory" / "u000.json").is_file()


def test_round_report_reads_the_program_each_game_began_with(tmp_path):
    """Two users kept the task's program in round 0. In round 1 user 0's student opened it before submitting, while
    user 1's wrote a program of its own over it and printed that back: only user 0 read its kept program. Both
    storers then wrote over the program, which is what the round's snapshot shows."""
    run_dir, path = tmp_path / "run", f"workspace/programs/{TASK}.py"
    for u in (0, 1):
        _write_state(run_dir, u, {path: PROGRAM})
        _commit_state(run_dir, u, 0)
        _write_doc(run_dir, 0, u, turns=[_turn("Submit")], calls=[_call(content="first visit", turn=0)])
    memory.round_report(run_dir, 0, users=2, round_dir_of=_round_dir, stateless=False)

    _write_doc(run_dir, 1, 0, turns=[_turn("Submit")],
               calls=[_call(tool_args=json.dumps({"code": "print(open(p).read())"}), turn=0),
                      _call(tool_results=[PROGRAM], turn=0),
                      _call(content=f"rewriting programs/{TASK}.py", tool_results=[PROGRAM], phase="after")])
    _write_doc(run_dir, 1, 1, turns=[_turn("Submit")],
               calls=[_call(tool_args=json.dumps({"code": f"open(p, 'w').write({REWRITTEN!r})"}), turn=0),
                      _call(tool_results=[REWRITTEN], turn=0)])
    for u in (0, 1):
        _write_state(run_dir, u, {path: REWRITTEN})
        _commit_state(run_dir, u, 1)

    report = memory.round_report(run_dir, 1, users=2, round_dir_of=_round_dir, stateless=False)

    assert report["users"]["0"]["reads"]["programs"] == 1 and report["users"]["0"]["reused_program"] is True
    assert report["users"]["1"]["reads"]["programs"] == 0 and report["users"]["1"]["reused_program"] is False
    assert report["total"]["reads"]["programs"] == 1 and report["total"]["reused_program"] == 1
    assert report["total"]["memory"]["programs"] == {"added": 0, "changed": 2, "removed": 0}


def test_round_report_of_a_team_reads_every_game_against_the_one_state_and_counts_what_each_user_wrote(tmp_path):
    """Two users who share one state (the baselines' CollectiveHarnessLearner): user 1's game opens the task's program
    that user 0's game kept in round 0, since every game of round 1 begins from the one state, user 0's commit of the
    round before; and each user's files are what the round's merge recorded them writing against that state, the
    harness's bookkeeping left out as a snapshot leaves it, with no snapshot of any user taken."""
    from continual_arc_baselines.shared import MERGES
    run_dir, path = tmp_path / "run", f"workspace/programs/{TASK}.py"
    _write_state(run_dir, 0, {path: PROGRAM})
    _commit_state(run_dir, 0, 0)
    written = [{"user": 0, "added": [], "changed": [path], "removed": []},
               {"user": 1, "added": ["agent/skills/grid/SKILL.md", "sessions/s1.jsonl"], "changed": [],
                "removed": ["workspace/scratch.json"]}]
    with (memory.system_dir(run_dir, 0) / MERGES).open("w") as fp:
        fp.write(json.dumps({"round": 1, "users": [{"user": u, "added": [], "changed": [], "removed": []}
                                                   for u in (0, 1)]}) + "\n")   # a first play cut short
        fp.write(json.dumps({"round": 1, "users": written}) + "\n")
    _write_doc(run_dir, 1, 0, turns=[_turn("Submit")], calls=[_call(content="no program here", turn=0)])
    _write_doc(run_dir, 1, 1, turns=[_turn("Submit")], calls=[_call(tool_results=[PROGRAM], turn=0)])

    report = memory.round_report(run_dir, 1, users=2, round_dir_of=_round_dir, stateless=False)

    assert report["users"]["1"]["reads"]["programs"] == 1 and report["users"]["1"]["reused_program"] is True
    assert report["users"]["0"]["reads"]["programs"] == 0
    assert report["users"]["0"]["memory"]["programs"] == {"added": 0, "changed": 1, "removed": 0}
    assert report["users"]["1"]["memory"]["skills"] == {"added": 1, "changed": 0, "removed": 0}
    assert report["users"]["1"]["memory"]["workspace_other"] == {"added": 0, "changed": 0, "removed": 1}
    assert sum(n for area in report["users"]["1"]["memory"].values() for n in area.values()) == 2
    assert not (_round_dir(run_dir, 1) / "memory" / "u001.json").exists()


def test_round_report_a_wiped_row_reads_no_program(tmp_path):
    """A wiped row's game begins from nothing, though the commit of the round before holds what its last game wrote
    (the wipe comes after the commit): a program like the last one, written again and printed, is no kept program
    read."""
    run_dir, path = tmp_path / "run", f"workspace/programs/{TASK}.py"
    _write_state(run_dir, 0, {path: PROGRAM})
    _commit_state(run_dir, 0, 0)
    _write_doc(run_dir, 1, 0, turns=[_turn("Submit")], calls=[_call(tool_results=[PROGRAM], turn=0)])
    report = memory.round_report(run_dir, 1, users=1, round_dir_of=_round_dir, stateless=True)
    assert report["users"]["0"]["reads"]["programs"] == 0 and report["users"]["0"]["reused_program"] is False


def test_round_report_a_wiped_rows_snapshot_shows_that_rounds_own_churn_but_nothing_unchanged(tmp_path):
    """A stateless row's harness is wiped and restarted *before the next* call, not before this snapshot
    (play.report_round_memory): round 1's own file is unrelated to round 0's (a fresh name, as
    a wipe-then-replay would give), so the diff against round 0's snapshot shows round 0's file removed and round
    1's added -- real churn, not an empty diff -- but changed stays 0 in every area, since no file ever survives
    into the next round unchanged. That is the invariant a wiped row's memory.json actually gives."""
    run_dir = tmp_path / "run"
    state = _write_state(run_dir, 0, {"agent/skills/a/SKILL.md": "round0"})
    _write_doc(run_dir, 0, 0, turns=[], calls=[])
    memory.round_report(run_dir, 0, users=1, round_dir_of=_round_dir, stateless=True)

    # the wipe: round 0's file is gone, replaced by a fresh one named for round 1
    (state / "agent" / "skills" / "a" / "SKILL.md").unlink()
    (state / "agent" / "skills" / "b").mkdir(parents=True)
    (state / "agent" / "skills" / "b" / "SKILL.md").write_text("round1")
    _write_doc(run_dir, 1, 0, turns=[], calls=[])

    report = memory.round_report(run_dir, 1, users=1, round_dir_of=_round_dir, stateless=True)
    assert report["users"]["0"]["memory"]["skills"] == {"added": 1, "changed": 0, "removed": 1}
    assert all(d["changed"] == 0 for d in report["users"]["0"]["memory"].values())   # the invariant a wipe gives


def test_recount_counts_a_played_runs_reads_again_and_keeps_its_files_counts(tmp_path):
    """A round whose memory.json counted a program's path string, the storer's own write among them: recount counts
    its reads again from its captured game and its users' state history, and keeps the files the round added,
    changed and removed as its snapshot recorded them."""
    run_dir, path = tmp_path / "run", f"workspace/programs/{TASK}.py"
    run_dir.mkdir()
    (run_dir / "driver.json").write_text(json.dumps({"row": "hgpl", "stateless": False}))
    _write_state(run_dir, 0, {path: PROGRAM})
    _commit_state(run_dir, 0, 0)
    _write_doc(run_dir, 1, 0, turns=[_turn("Submit"), _turn("RequestDemos"), _turn("Submit")],
               calls=[_call(content=f"let me look at programs/{TASK}.py", turn=0),
                      _call(tool_results=[PROGRAM], turn=2),
                      _call(content=f"saving programs/{TASK}.py", phase="after")])
    files = {area: {"added": 0, "changed": 0, "removed": 0} for area in memory.AREAS}
    files["programs"]["changed"] = 1
    recorded = {"round": 1, "users": {"0": {"memory": files, "reads": {"programs": 2, "skills": 0, "harness_store": 0,
                                                                       "notes": 0},
                                            "reused_program": True, "solved_without_demo": False}},
                "total": {"memory": files, "reads": {"programs": 2, "skills": 0, "harness_store": 0, "notes": 0},
                          "reused_program": 1, "solved_without_demo": 0}}
    (_round_dir(run_dir, 1) / "memory.json").write_text(json.dumps(recorded))

    [report] = memory.recount(run_dir)

    assert report["users"]["0"]["reads"]["programs"] == 1          # the one read, after the first submission
    assert report["users"]["0"]["reused_program"] is False
    assert report["users"]["0"]["memory"] == files
    assert report["total"]["reads"]["programs"] == 1 and report["total"]["reused_program"] == 0
    assert report["total"]["memory"] == files
    assert json.loads((_round_dir(run_dir, 1) / "memory.json").read_text()) == report


def test_summary_line_reports_deliberate_totals_only_and_the_two_game_signals():
    """workspace_other is scratch and must never land in the headline: a big workspace_other change here, with
    only small deliberate numbers, must not inflate added/changed/removed."""
    report = {"round": 3, "total": {"memory": {"skills": {"added": 1, "changed": 0, "removed": 0},
                                               "harness_store": {"added": 0, "changed": 2, "removed": 0},
                                               "notes": {"added": 0, "changed": 0, "removed": 0},
                                               "programs": {"added": 0, "changed": 0, "removed": 1},
                                               "workspace_other": {"added": 97, "changed": 100, "removed": 97},
                                               "other": {"added": 100, "changed": 100, "removed": 100}},
                                    "reads": {"programs": 2, "skills": 1, "harness_store": 0, "notes": 0},
                                    "reused_program": 4, "solved_without_demo": 2}}
    line = memory.summary_line(report)
    assert "round 3 memory" in line and "1 added" in line and "2 changed" in line and "1 removed" in line
    assert "97" not in line and "100" not in line   # scratch never reaches the headline
    assert "3 reads" in line
    assert "4 games reused a saved program" in line and "2 solved without a demo" in line


def _recorded(reads: int, reused: bool) -> dict:
    """A round's memory.json of one user, its reads as older code counted them."""
    files = {area: {"added": 0, "changed": 0, "removed": 0} for area in memory.AREAS}
    counts = {"programs": reads, "skills": 0, "harness_store": 0, "notes": 0}
    return {"round": 0, "users": {"0": {"memory": files, "reads": counts, "reused_program": reused,
                                        "solved_without_demo": False}},
            "total": {"memory": files, "reads": counts, "reused_program": int(reused), "solved_without_demo": 0}}


def test_recount_of_a_stored_forked_play_reads_the_history_it_is_given(tmp_path):
    """A stored play keeps none of its users' system folders: recount reads its fork's commit in the state.git it was
    forked from, and fails loudly without one, its memory.json left as it was."""
    run_dir = tmp_path / "held-out" / "play"
    _forked_play(run_dir)
    (run_dir / "driver.json").write_text(json.dumps({"row": "hplg", "stateless": False}))
    _write_doc(run_dir, 0, 0, turns=[_turn("Submit")], calls=[_call(tool_results=[PROGRAM], turn=0)])
    (_round_dir(run_dir, 0) / "memory.json").write_text(json.dumps(_recorded(7, False)))
    stored = tmp_path / "stream.git"
    os.rename(memory.system_dir(run_dir, 0) / "state.git", stored)

    with pytest.raises(subprocess.CalledProcessError):
        memory.recount(run_dir)
    assert json.loads((_round_dir(run_dir, 0) / "memory.json").read_text()) == _recorded(7, False)

    [report] = memory.recount(run_dir, stored)
    assert report["users"]["0"]["reads"]["programs"] == 1 and report["total"]["reused_program"] == 1


def test_recount_refuses_a_copy_without_its_games_and_writes_nothing(tmp_path):
    """A copy that kept a run's memory.json and driver.json but none of its games: recount would count every read 0, so
    it refuses, even when an earlier round could be read."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "driver.json").write_text(json.dumps({"row": "hgpl", "stateless": False}))
    _write_doc(run_dir, 0, 0, turns=[_turn("Submit")], calls=[_call(content="no program here", turn=0)])
    (_round_dir(run_dir, 0) / "memory.json").write_text(json.dumps(_recorded(7, True)))
    _round_dir(run_dir, 1).mkdir()
    (_round_dir(run_dir, 1) / "memory.json").write_text(json.dumps({**_recorded(3, True), "round": 1}))

    with pytest.raises(SystemExit, match="no captured games"):
        memory.recount(run_dir)
    assert json.loads((_round_dir(run_dir, 0) / "memory.json").read_text()) == _recorded(7, True)


def test_a_body_whose_messages_is_not_a_list_is_read_without_raising():
    """The model's own sandbox code can post any JSON body through the proxy, "messages" included as a dict: it answers
    no tool result, whatever the dict holds, and its text is the reply's alone."""
    odd = {"phase": "turn", "request": {"messages": {"role": "tool", "content": PROGRAM}, "tools": []},
           "response": {"choices": [{"message": {"content": "ok agent/skills/x/SKILL.md"}}]}}
    assert memory.answered(odd) == ""
    assert memory.read_areas(odd, MARKS) == {"skills"}
    assert memory.read_areas({"request": {"prompt": "x"}, "response": None}, MARKS) == set()


def test_a_play_whose_users_forked_commits_of_their_own_names_each_users(tmp_path):
    play = tmp_path / "play"
    play.mkdir()
    (play / "frozen.json").write_text(json.dumps({"from_users": [{"user": 0, "round": None, "commit": None},
                                                                 {"user": 1, "round": 4, "commit": "c4"}]}))
    assert memory.forked_from(play, 0) is None and memory.forked_from(play, 1) == "c4"
