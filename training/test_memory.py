"""memory.py's off-GPU pieces, all on fake data: a hand-built state tree stands in for a user's harness state
directory, and a hand-built captured doc (paper.save_instance_hook's shape) stands in for a round's play.

Run with ``python3 -m pytest test_memory.py -q``; no GPU, no served model, no continual_arc_baselines needed.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import memory


# -- snapshot_state / area_of / diff_snapshot ----------------------------------------------------------------

def test_snapshot_state_reads_path_size_mtime_hash_and_skips_excluded(tmp_path):
    state = tmp_path / "state"
    (state / "agent" / "skills" / "solve-grids").mkdir(parents=True)
    (state / "agent" / "skills" / "solve-grids" / "SKILL.md").write_text("do the thing")
    (state / "agent" / "kernel-venv" / "lib").mkdir(parents=True)
    (state / "agent" / "kernel-venv" / "lib" / "x.pyc").write_text("not real bytecode")
    (state / "workspace" / "programs").mkdir(parents=True)
    (state / "workspace" / "programs" / "p1.py").write_text("print(1)")

    snapshot = memory.snapshot_state(state)

    assert "agent/skills/solve-grids/SKILL.md" in snapshot
    assert not any(p.startswith("agent/kernel-venv") for p in snapshot)   # excluded: a runtime, not what it learned
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
    """A fake tree mirroring the real layout (checked against row H's run): session
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

# A trimmed excerpt of a real captured call's system prompt (row H's run, round 0):
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


def _call(content: str | None = None, reasoning: str | None = None, tool_args: str | None = None,
         thought: str | None = None, note: str | None = None, tool_result: str | None = None,
         system: str | None = None, turn: int | None = 0) -> dict:
    """One captured call (the raw dict shape paper.save_instance_hook/loop.save_instance write): ``content``/
    ``reasoning``/``tool_args`` build the call's own reply (response.choices[0].message); ``tool_result`` is the
    newest message already in the request, role "tool" (as the real ipython-execution output is, see
    test_memory.py's real-call inspection); ``system`` is the system prompt, present in every real call's request
    but never scanned -- here to prove it stays that way."""
    message: dict = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning"] = reasoning
    if tool_args is not None:
        message["tool_calls"] = [{"function": {"name": "ipython", "arguments": tool_args}}]
    messages = [{"role": "system", "content": system or "a generic system prompt, no markers"},
               {"role": "user", "content": "here is the puzzle: a game message, never scanned"}]
    if tool_result is not None:
        messages.append({"role": "tool", "content": tool_result})
    call = {"turn": turn, "request": {"messages": messages}, "response": {"choices": [{"message": message}]}}
    if thought:
        call["thought"] = thought
    if note:
        call["note"] = note
    return call


def test_read_areas_detects_a_marker_in_the_reply_content():
    assert memory.read_areas(_call(content="running programs/p1.py now")) == {"programs"}


def test_read_areas_detects_a_marker_in_the_replys_reasoning():
    assert memory.read_areas(_call(reasoning="let me check programs/solver.py for what I stored last time")) == \
        {"programs"}


def test_read_areas_detects_a_marker_in_a_tool_calls_arguments():
    assert memory.read_areas(_call(tool_args=json.dumps({"code": "open('agent/skills/solve-grids/SKILL.md')"}))) \
        == {"skills"}


def test_read_areas_detects_a_marker_in_a_tool_result():
    """A tool result (the ipython execution output, the newest message already in the request) must count: this
    is where the content of a programs/ or skills/ file actually read would appear."""
    assert memory.read_areas(_call(tool_result="loaded agent/harness/notes.json successfully")) == {"harness_store"}


def test_read_areas_never_counts_the_system_prompt_or_the_game_message():
    """The system prompt and the benchmark's own game message are both in every call's request, but neither is
    the model's own output or a tool result: a marker there must never count."""
    assert memory.read_areas(_call(system="go read programs/p1.py and agent/skills/x/SKILL.md right now",
                                   content="ok")) == set()


def test_read_areas_rejects_prime_agents_bundled_skill_catalogue():
    """F1, reproduced directly: a real system prompt's bundled-skill catalogue lists each skill under
    .../prime-agent/skills/<name>/SKILL.md, which contains the literal substring "agent/skills/" -- but it must
    never register as a skills read, both because it is in the system prompt (never scanned at all) and because
    "prime-agent/skills/" is not "agent/skills/" (the anchored pattern)."""
    call = _call(system=REAL_SYSTEM_PROMPT_EXCERPT, content="ok, I see the puzzle")
    assert memory.read_areas(call) == set()
    # even if that same bundled-catalogue text somehow reached a scanned field (belt and suspenders: the pattern
    # itself, not only the field restriction, must reject "prime-agent/skills/")
    assert memory.read_areas(_call(content=REAL_SYSTEM_PROMPT_EXCERPT)) == set()


def test_read_areas_accepts_the_users_own_skills_path_including_through_the_prime_link():
    assert memory.read_areas(_call(content="agent/skills/solve-grids/SKILL.md")) == {"skills"}
    assert memory.read_areas(_call(content="~/.prime/agent/skills/solve-grids/SKILL.md")) == {"skills"}


def test_read_areas_requires_an_actual_filename_for_programs():
    """Plan text alone ("I'll write this to programs/") must not count: programs/<name>.py needs a real filename
    immediately after the slash."""
    assert memory.read_areas(_call(content="I'll write this to programs/ once I figure out the rule")) == set()
    assert memory.read_areas(_call(content="saving to programs/t1.py now")) == {"programs"}


def test_read_areas_detects_a_marker_for_notes():
    assert memory.read_areas(_call(content="let me open AGENTS.md and see what it says")) == {"notes"}


def test_read_areas_is_empty_with_no_marker():
    assert memory.read_areas(_call(content="just thinking out loud, nothing on disk")) == set()


def test_read_areas_can_report_more_than_one_area_for_one_call():
    assert memory.read_areas(_call(content="programs/p1.py and agent/skills/x/SKILL.md both")) == {"programs",
                                                                                                    "skills"}


def test_count_reads_sums_areas_over_every_call_of_a_game():
    calls = [_call(content="programs/p1.py"), _call(content="nothing here"),
            _call(content="programs/p2.py and agent/skills/x/SKILL.md")]
    assert memory.count_reads(calls) == {"programs": 2, "skills": 1, "harness_store": 0, "notes": 0}


def _turn(action_type: str) -> dict:
    return {"action": {"type": action_type}}


def test_game_memory_outcome_reused_program_true_at_or_before_the_first_submission():
    """The real action vocabulary is RequestDemos/Submit/GiveUp (never "Observation"): a program read at the same
    turn as the first Submit, or any turn before it, counts."""
    doc = {"turns": [_turn("RequestDemos"), _turn("Submit")],
          "calls": [_call(content="running programs/p1.py", turn=1), _call(content="ok", turn=2)]}
    call_areas = [memory.read_areas(c) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is True


def test_game_memory_outcome_reused_program_false_when_the_read_is_after_the_first_submission():
    doc = {"turns": [_turn("Submit"), _turn("Submit")],
          "calls": [_call(content="just thinking", turn=0), _call(content="programs/p1.py now", turn=1)]}
    call_areas = [memory.read_areas(c) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is False


def test_game_memory_outcome_reused_program_true_with_no_submission_at_all():
    """A game that read a program and then gave up without ever submitting still reused it: there is no
    "before submitting" boundary to exclude it by."""
    doc = {"turns": [_turn("RequestDemos"), _turn("GiveUp")],
          "calls": [_call(content="programs/p1.py", turn=0)]}
    call_areas = [memory.read_areas(c) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is True


def test_game_memory_outcome_reused_program_false_with_no_program_read():
    doc = {"turns": [_turn("Submit")], "calls": [_call(content="just thinking", turn=0)]}
    call_areas = [memory.read_areas(c) for c in doc["calls"]]
    assert memory.game_memory_outcome(doc, call_areas)["reused_program"] is False


def test_game_memory_outcome_solved_without_demo():
    solved_no_demo = {"solved": True, "turns": [_turn("Submit")]}
    solved_with_demo = {"solved": True, "turns": [_turn("RequestDemos"), _turn("Submit")]}
    failed_no_demo = {"solved": False, "turns": [_turn("GiveUp")]}
    assert memory.game_memory_outcome(solved_no_demo, [])["solved_without_demo"] is True
    assert memory.game_memory_outcome(solved_with_demo, [])["solved_without_demo"] is False
    assert memory.game_memory_outcome(failed_no_demo, [])["solved_without_demo"] is False


# -- round_report / summary_line, wired against a fake run folder --------------------------------------------

def _round_dir(run_dir: Path, round_: int) -> Path:
    return run_dir / "rounds" / f"r{round_:04d}"


def _write_doc(run_dir: Path, round_: int, user: int, turns: list[dict], calls: list[dict]) -> None:
    captured = _round_dir(run_dir, round_) / "captured"
    captured.mkdir(parents=True, exist_ok=True)
    with gzip.open(captured / f"u{user:03d}.json.gz", "wt") as fp:
        json.dump({"turns": turns, "calls": calls}, fp)


def _write_state(run_dir: Path, user: int, files: dict[str, str]) -> Path:
    state = run_dir / "users" / f"u{user:03d}" / "system" / "state"
    state.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = state / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        os.utime(path, (time.time(), time.time()))
    return state


def test_round_report_writes_memory_json_with_per_user_and_total_counts(tmp_path):
    run_dir = tmp_path / "run"
    state0 = _write_state(run_dir, 0, {"agent/skills/a/SKILL.md": "v1"})
    _write_doc(run_dir, 0, 0, turns=[_turn("RequestDemos"), _turn("Submit")],
              calls=[_call(content="agent/skills/a/SKILL.md", turn=0)])

    report = memory.round_report(run_dir, 0, users=1, state_dir_of=lambda u: state0,
                                 round_dir_of=_round_dir)

    assert report["round"] == 0
    assert report["users"]["0"]["memory"]["skills"]["added"] == 1
    assert report["users"]["0"]["reads"]["skills"] == 1
    assert report["total"]["memory"]["skills"]["added"] == 1
    assert "reused_program" in report["users"]["0"] and "solved_without_demo" in report["users"]["0"]
    assert (_round_dir(run_dir, 0) / "memory.json").is_file()
    assert (_round_dir(run_dir, 0) / "memory" / "u000.json").is_file()


def test_round_report_a_wiped_rows_snapshot_shows_that_rounds_own_churn_but_nothing_unchanged(tmp_path):
    """A stateless row's harness is wiped and restarted *before the next* call, not before this snapshot
    (paper._report_round_memory's own docstring): round 1's own file is unrelated to round 0's (a fresh name, as
    a wipe-then-replay would give), so the diff against round 0's snapshot shows round 0's file removed and round
    1's added -- real churn, not an empty diff -- but changed stays 0 in every area, since no file ever survives
    into the next round unchanged. That is the invariant a wiped row's memory.json actually gives."""
    run_dir = tmp_path / "run"
    state = _write_state(run_dir, 0, {"agent/skills/a/SKILL.md": "round0"})
    _write_doc(run_dir, 0, 0, turns=[], calls=[])
    memory.round_report(run_dir, 0, users=1, state_dir_of=lambda u: state, round_dir_of=_round_dir)

    # the wipe: round 0's file is gone, replaced by a fresh one named for round 1
    (state / "agent" / "skills" / "a" / "SKILL.md").unlink()
    (state / "agent" / "skills" / "b").mkdir(parents=True)
    (state / "agent" / "skills" / "b" / "SKILL.md").write_text("round1")
    _write_doc(run_dir, 1, 0, turns=[], calls=[])

    report = memory.round_report(run_dir, 1, users=1, state_dir_of=lambda u: state, round_dir_of=_round_dir)
    assert report["users"]["0"]["memory"]["skills"] == {"added": 1, "changed": 0, "removed": 1}
    assert all(d["changed"] == 0 for d in report["users"]["0"]["memory"].values())   # the invariant a wipe gives


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
