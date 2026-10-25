import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from continual_arc.env import GiveUp, LearnerObservation, Submit

from continual_arc_baselines.systems.acp import ACPSystemLearner
from continual_arc_baselines.systems.base import SystemUnavailable
from continual_arc_baselines.systems.openclaw import OpenClawLearner

FAKE = Path(__file__).with_name("fake_acp_agent.py")


class FakeSystem(ACPSystemLearner):
    name = "fake"

    def default_command(self):
        return [sys.executable, str(FAKE)]

    def argv(self):
        return list(self.command)

    def prepare(self):
        (self.state_dir / "config.json").write_text("{}")

    def extra_env(self):
        return {"FAKE_ACP_STATE": str(self.state_dir / "agent"),
                "FAKE_ACP_REPLY": self.reply, "FAKE_ACP_SLEEP": "3",
                "PYTHONPATH": ":".join(sys.path)}

    def preserved(self):
        return ("config.json",)

    reply = json.dumps({"action": "submit", "grid": [[1]]})


def obs(step=0):
    return LearnerObservation(step=step, task_id="t1", input=[[0]], attempts_used=0, demo_requests_used=0,
                              attempt_cap=8, demo_cap=8)


@pytest.fixture
def system(tmp_path):
    s = FakeSystem(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60)
    s.attach(tmp_path / "system", {})
    yield s
    s.close()


def test_turn_roundtrip_and_state_wipe(system, tmp_path):
    action = system.act(obs())
    assert isinstance(action, Submit) and action.grid == [[1]]
    state = tmp_path / "system" / "state"
    assert (state / "agent" / "prompt-1.txt").read_text().startswith("You are being evaluated")
    # a second turn does not repeat the rules and carries the pending feedback
    from continual_arc.env import SubmitFeedback
    system.observe(obs(), action, SubmitFeedback(correct=False, valid=True, attempts_used=1, failed=False))
    system.act(obs())
    second = (state / "agent" / "prompt-2.txt").read_text()
    assert "INCORRECT" in second and not second.startswith("You are being evaluated")
    system.reset()  # stops, wipes, restarts lazily
    assert (state / "config.json").exists()
    assert not (state / "agent").exists()
    assert isinstance(system.act(obs()), Submit)
    assert (state / "agent" / "prompt-1.txt").exists()
    usage = system.resources()
    assert usage["system_turns"] == 3.0 and usage["system_resets"] == 1.0
    # the agent reports session totals (100, 200 before the reset; 100 after): totals, not their sum
    assert usage["system_input_tokens"] == 300.0 and usage["system_output_tokens"] == 30.0


def test_idle_timeout_becomes_a_restart(tmp_path):
    s = FakeSystem(model="m", sandbox="none", timeout_s=60, idle_timeout_s=1.0, startup_timeout_s=60)
    s.attach(tmp_path / "system", {})
    try:
        s._ensure_started()
        with pytest.raises(SystemUnavailable):
            s.ask("SLEEP please")
        assert s.stats["idle_timeouts"] == 1
    finally:
        s.close()


def test_hard_timeout_is_a_budget_hit_not_a_restart(tmp_path):
    """A prompt that overruns its budget is cancelled and re-prompted with the budget notice,
    as prime-agent's is; the system is working, so it is not restarted."""
    s = FakeSystem(model="m", sandbox="none", timeout_s=1.0, idle_timeout_s=60, startup_timeout_s=60,
                   max_restarts=1)
    s.attach(tmp_path / "system", {})
    try:
        s.reply = json.dumps({"action": "request_demos"})
        action = s.act(LearnerObservation(step=0, task_id="SLEEP", input=[[0]], attempts_used=0, demo_requests_used=0,
                                          attempt_cap=8, demo_cap=8))
        assert s.stats["budget_hits"] == 1 and s.stats["restarts"] == 0 and s.stats["forced_give_ups"] == 0
        assert not isinstance(action, GiveUp)
    finally:
        s.close()


# -- blank slate: no bundled skills, no channels -----------------------------

def test_hermes_starts_without_bundled_skills(tmp_path):
    from continual_arc_baselines.systems.hermes_agent import HermesAgentLearner
    s = HermesAgentLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    s.prepare()
    assert (s.hermes_home / ".no-bundled-skills").exists()
    config = json.loads((s.hermes_home / "config.yaml").read_text())
    # The post-turn memory and skill review stays at its shipped default (on): nothing overrides it.
    assert "background_review" not in (config.get("auxiliary") or {})


@pytest.fixture
def unchecked_openclaw(monkeypatch):
    """The config tests read what prepare() writes; whether an installed openclaw accepts it is check_config's test."""
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    monkeypatch.setattr(OpenClawLearner, "check_config", lambda self: None)


def test_openclaw_blocks_bundled_skills_and_channels(tmp_path, unchecked_openclaw):
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    s = OpenClawLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    s.prepare()
    config = json.loads(s.config_path.read_text())
    assert config["skills"]["allowBundled"] == ["__none__"]
    assert s.extra_env()["OPENCLAW_SKIP_CHANNELS"] == "1"


def test_openclaw_memory_search_is_set_only_when_asked(tmp_path, unchecked_openclaw):
    """Off (the grid cells of 27 and 28 Sep), memory search stays unset; on, it is keyword-only over the memory files
    and the session transcripts, and the run record says which."""
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    for on in (False, True):
        s = OpenClawLearner(model="m", sandbox="none", memory_search=on)
        s.attach(tmp_path / f"system-{on}", {})
        s.prepare()
        config = json.loads(s.config_path.read_text())
        assert s.describe()["memory_search"] is on
        if not on:
            assert "memory" not in config
            continue
        search = config["memory"]["search"]
        assert search["provider"] == "none" and search["experimental"]["sessionMemory"] is True
        assert search["sources"] == ["memory", "sessions"]


def test_openclaw_settles_a_run_its_dead_gateway_left_running(tmp_path, unchecked_openclaw):
    """A gateway killed mid-run leaves the bridge session running, which openclaw's restart recovery would
    replay at the next boot; prepare() closes that run as a clean stop leaves it, and leaves anything else."""
    import sqlite3

    from continual_arc_baselines.systems.openclaw import BRIDGE_KEY, OpenClawLearner
    s = OpenClawLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    db = s.openclaw_dir / "agents" / "main" / "agent" / "openclaw-agent.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("create table session_nodes (session_key text primary key, status text, entry_json text, updated_at integer)")
    stuck = {"sessionId": "s1", "status": "running", "abortedLastRun": False, "startedAt": 1000, "activeWriterRunId": "w",
             "lifecycleRunId": "r", "mainRestartRecovery": {"cycleId": "c", "chargedAttempts": 156}, "restartRecoveryRuns": [1]}
    con.execute("insert into session_nodes values (?, 'running', ?, 1)", (BRIDGE_KEY, json.dumps(stuck)))
    con.execute("insert into session_nodes values ('agent:main:cron:x', 'running', ?, 1)", (json.dumps({"status": "running"}),))
    con.commit()
    con.close()
    notes = []
    s.note = notes.append
    s.prepare()
    con = sqlite3.connect(db)
    status, entry, updated = con.execute("select status, entry_json, updated_at from session_nodes where session_key = ?", (BRIDGE_KEY,)).fetchone()
    entry = json.loads(entry)
    assert status == "done" and entry["status"] == "done" and entry["abortedLastRun"] is False and entry["endedAt"] >= 1000
    assert not {"mainRestartRecovery", "lifecycleRunId", "restartRecoveryRuns"} & set(entry)
    assert entry["activeWriterRunId"] == "w" and entry["sessionId"] == "s1"
    assert con.execute("select status from session_nodes where session_key = 'agent:main:cron:x'").fetchone()[0] == "running"
    con.close()
    assert notes and "156 attempts" in notes[0]
    s.prepare()                                     # a settled session is left as it is
    assert len(notes) == 1
    assert sqlite3.connect(db).execute("select updated_at from session_nodes where session_key = ?", (BRIDGE_KEY,)).fetchone()[0] == updated


def test_openclaw_refuses_a_release_that_rejects_its_config(tmp_path):
    """An openclaw whose ``config validate`` fails stops the run before its first start, with openclaw's reason,
    rather than letting every start fail on a closed connection; one that accepts the config is asked once a run."""
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    calls = tmp_path / "calls"
    script = 'echo "$@" >> "$0"; [ "$1" = --version ] && echo "OpenClaw 9.9.9" && exit 0; '
    rejecting = ["sh", "-c", script + 'echo \'skills.workshop: Unrecognized key: "approvalPolicy"\' >&2; exit 1', str(calls)]
    s = OpenClawLearner(model="m", sandbox="none", command=rejecting)
    s.attach(tmp_path / "rejecting", {})
    with pytest.raises(RuntimeError, match='OpenClaw 9.9.9 rejects the adapter.s config: .*"approvalPolicy"'):
        s.prepare()
    accepting = ["sh", "-c", script + "exit 0", str(calls)]
    s = OpenClawLearner(model="m", sandbox="none", command=accepting)
    s.attach(tmp_path / "accepting", {})
    s.prepare()
    s.prepare()
    assert calls.read_text().splitlines().count("config validate") == 2   # the rejecting one's and the accepting one's


def test_openclaw_keeps_every_gateway_start_in_its_log(tmp_path):
    """The launcher sends the gateway's output to its own stderr, the adapter's log, which every start appends to, so a
    gateway that never came up can still be read after the next start. It waits for the gateway as long as the
    adapter's start allows, but the minute left to the bridge, and ends a gateway it gave up on."""
    from continual_arc_baselines.systems.acp import free_port
    from continual_arc_baselines.systems.openclaw import LAUNCHER, OpenClawLearner
    fake = tmp_path / "openclaw"
    fake.write_text('#!/bin/sh\n[ "$1" = gateway ] || exit 2\necho "booting ($BOOT)"\necho $$ > "$PIDS/$$"\n'
                    '[ "$BOOT" = rejected ] && { echo "config rejected" >&2; exit 1; }\nexec sleep 60\n')
    fake.chmod(0o755)
    pids = tmp_path / "pids"
    pids.mkdir()
    log = tmp_path / "openclaw.log"
    env = {**os.environ, "OPENCLAW_GATEWAY_PORT": str(free_port()), "OPENCLAW_GATEWAY_TOKEN": "t", "PIDS": str(pids)}
    for boot, ready in (("slow", "2"), ("rejected", "30")):
        with log.open("a") as stderr:
            started = time.monotonic()
            out = subprocess.run(["sh", "-c", LAUNCHER, "openclaw-launcher", ready, str(fake)], env={**env, "BOOT": boot},
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr, timeout=30)
        assert out.returncode == 1 and out.stdout == b""   # the bridge's stdout, the ACP stream, carries none of it
        assert time.monotonic() - started < 10
    text = log.read_text()
    assert text.count("gateway starting at") == 2
    assert text.index("booting (slow)") < text.index("gateway not ready at") < text.index("booting (rejected)")
    assert text.index("config rejected") < text.index("gateway exited")
    deadline = time.monotonic() + 10
    while any(_alive(int(p.name)) for p in pids.iterdir()):
        assert time.monotonic() < deadline, "the launcher left its gateway running"
        time.sleep(0.1)
    s = OpenClawLearner(model="m", sandbox="none", startup_timeout_s=600)
    assert s.argv()[3:5] == ["openclaw-launcher", "540"]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_opencode_starts_without_plugins_or_external_skills(tmp_path):
    from continual_arc_baselines.systems.opencode import OpenCodeLearner
    s = OpenCodeLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    s.prepare()
    env = s.extra_env()
    assert env["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "1"
    # Both discovery switches stay off: enabling either makes opencode write into its read-only
    # install tree at start. Its private global config directory is what persists.
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1" and env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"


def test_stateless_row_wipes_everything_between_instances(tmp_path):
    """Row H0 on an ACP system: a correct submission ends the instance; the next fresh
    observation stops the agent, deletes its state and starts it again, so the second
    instance's first prompt is prompt-1 again and carries the rules, and nothing the first
    instance wrote survives."""
    from continual_arc.env import SubmitFeedback

    s = FakeSystem(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60, stateless=True)
    s.attach(tmp_path / "system", {})
    try:
        state = tmp_path / "system" / "state"
        first = s.act(obs(step=0))
        (state / "agent" / "memory.md").write_text("what the system learned")   # something it persisted
        s.observe(obs(step=0), first, SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False))
        assert s._session_pending
        second = s.act(obs(step=1))
        assert isinstance(second, Submit)
        assert not (state / "agent" / "memory.md").exists()
        assert not (state / "agent" / "prompt-2.txt").exists()
        assert (state / "agent" / "prompt-1.txt").read_text().startswith("You are being evaluated")
        assert (state / "config.json").exists()   # the adapter's configuration, rewritten by prepare()
        usage = s.resources()
        assert usage["system_wipes"] == 1.0 and usage["system_resets"] == 1.0 and usage["system_sessions"] == 1.0
    finally:
        s.close()


def test_stateless_logged_acp_system_is_wiped_before_its_first_instance_too(tmp_path):
    """Row H0 with its state logged: the first instance is wiped as every later one is, so what an instance played by
    an earlier process left (a collective test's user, a run forked from a round) is gone."""
    from continual_arc.env import SubmitFeedback

    s = FakeSystem(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60, stateless=True)
    state = tmp_path / "system" / "state"
    (state / "agent").mkdir(parents=True)
    (state / "agent" / "memory.md").write_text("what an earlier process's instance learned")
    s.attach(tmp_path / "system", {}, log_state=True)
    try:
        first = s.act(obs(step=0))
        assert not (state / "agent" / "memory.md").exists()
        s.observe(obs(step=0), first, SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False))
        s.act(obs(step=1))
        usage = s.resources()
        assert usage["system_wipes"] == 2.0 and usage["system_resets"] == 2.0 and usage["system_sessions"] == 2.0
    finally:
        s.close()


class KeptConversation(FakeSystem):
    """A harness whose one conversation outlives its process and is taken up again at every start, as openclaw's ACP
    bridge takes up its one fixed session, and whose new session is openclaw's own in-place ``/reset``."""

    new_session = OpenClawLearner.new_session

    def extra_env(self):
        return {**super().extra_env(), "FAKE_ACP_CONVERSATION": str(self.state_dir / "agent" / "conversation.json")}

    def conversation(self) -> list[str]:
        return json.loads((self.state_dir / "agent" / "conversation.json").read_text())


def test_every_instance_of_a_logged_stateful_system_has_a_conversation_of_its_own(tmp_path):
    """A logged learner made anew on a folder an earlier process wrote (a collective test's user forked from the
    stream's last round, a run forked from a round) plays its first instance in a new conversation, as it plays every
    later one, and not in the last conversation the harness's state holds, which openclaw's bridge takes up again at
    every start."""
    from continual_arc.env import SubmitFeedback

    solved = SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False)
    stream = KeptConversation(model="m", sandbox="none", startup_timeout_s=60, timeout_s=60)
    stream.attach(tmp_path / "system", {})
    try:
        stream.observe(obs(step=0), stream.act(obs(step=0)), solved)
    finally:
        stream.close()
    assert len(stream.conversation()) == 2   # the stream's last instance and its closing turn, kept in the state
    test = KeptConversation(model="m", sandbox="none", startup_timeout_s=60, timeout_s=60)
    test.attach(tmp_path / "system", {}, log_state=True)
    try:
        for step in (1, 2):
            action = test.act(obs(step=step))
            conversation = test.conversation()
            assert len(conversation) == 1 and conversation[0].startswith("You are being evaluated")
            test.observe(obs(step=step), action, solved)
        assert test.resources()["system_sessions"] == 2.0
    finally:
        test.close()


def test_openclaw_npm_fails_fast():
    """The gateway's boot-time plugin install must not retry against a registry it cannot reach."""
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    learner = OpenClawLearner.__new__(OpenClawLearner)
    learner._port = 1
    learner._token = "t"
    learner.workdir, learner._state_dir = Path("/x"), None
    env = learner.extra_env()
    assert env["npm_config_fetch_retries"] == "0"
