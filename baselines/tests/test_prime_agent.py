"""End to end: CLI -> proxy -> prime-agent adapter -> fake prime-agent -> fake upstream."""

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from venv import create as create_venv

import pytest
from continual_arc.schedule import epilogue_schedule, stream_schedule
from continual_arc.testing import FakeSource

from continual_arc_baselines.attempt import Attempt
from continual_arc_baselines.cli import main
from continual_arc_baselines.costs import CostJournal

FAKE = str(Path(__file__).with_name("fake_prime_agent.py"))
REPO = Path(__file__).resolve().parents[1]

# prime-agent's bundled skills as its releases ship them in skills/, each with whether it is a Python skill: 0.9.6's,
# and 0.9.5's, whose Linear and Notion integrations were Python skills where 0.9.6 has its markdown mcp skill
SKILLS_0_9_6 = {"agent-message": True, "agent-observe": True, "attach-image": True, "compact": True, "edit": True,
                "goal": True, "mcp": False, "prime-intellect": False, "refine": True, "rlm-heartbeat": True,
                "skill-creator": False, "websearch": True}
SKILLS_0_9_5 = {**{name: python for name, python in SKILLS_0_9_6.items() if name != "mcp"}, "linear": True,
                "notion": True}
# what the adapter installs from either: every Python skill but web search and the integrations, which its settings
# turn off
KERNEL_SKILLS = ["agent_message", "agent_observe", "attach_image", "compact", "edit", "goal", "refine", "rlm_heartbeat"]


def _release(folder: Path, skills: dict[str, bool]) -> Path:
    """A prime-agent release's folder as its tarball unpacks: the executable, here the fake run by this Python, and
    skills/, each skill a SKILL.md and a Python skill also a pyproject.toml and its package in src/."""
    folder.mkdir(parents=True)
    (folder / "prime-agent").write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(FAKE)} "$@"\n')
    (folder / "prime-agent").chmod(0o755)
    for name, python in skills.items():
        skill = folder / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: the {name} skill\n---\n")
        if python:
            (skill / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
            (skill / "src" / name.replace("-", "_")).mkdir(parents=True)
            (skill / "src" / name.replace("-", "_") / "__init__.py").write_text("")
    return folder


def install_rust_sh(prefix: Path, skills: dict[str, bool] = SKILLS_0_9_6) -> Path:
    """prime-agent as install-rust.sh installs it, which prime-agent's install.sh is whatever release it installs (the
    GPU nodes' 0.9.6): the release in <prefix>/share/prime-agent/, and a launcher script in <prefix>/bin that runs its
    executable. The launcher."""
    _release(prefix / "share" / "prime-agent", skills)
    launcher = prefix / "bin" / "prime-agent"
    launcher.parent.mkdir()
    launcher.write_text('#!/bin/sh\nexec "$(dirname "$0")/../share/prime-agent/prime-agent" "$@"\n')
    launcher.chmod(0o755)
    return launcher


def install_typescript(home: Path, skills: dict[str, bool] = SKILLS_0_9_5) -> Path:
    """prime-agent as its TypeScript installer installs it (0.9.5's own install.sh): the release in
    ~/.local/share/prime-agent/releases/<version>-<platform>-<digest>/, that folder's bin/prime-agent a link to the
    release's executable, and ~/.local/bin/prime-agent a link to that. The public link."""
    root = home / ".local" / "share" / "prime-agent"
    release = _release(root / "releases" / "0.9.5-linux-x64-bc4b0ed7", skills)
    (root / "bin").mkdir()
    (root / "bin" / "prime-agent").symlink_to(f"../releases/{release.name}/prime-agent")
    launcher = home / ".local" / "bin" / "prime-agent"
    launcher.parent.mkdir()
    launcher.symlink_to(root / "bin" / "prime-agent")
    return launcher


def editable(python: Path, skill: Path) -> None:
    """``skill`` installed into ``python``'s venv as an editable install leaves it: a .pth file naming its src/."""
    site = next(python.parent.parent.glob("lib/python*/site-packages"))
    (site / f"_{skill.name}.pth").write_text(f"{skill / 'src'}\n")


def kernel_venv(path: Path, skills: Path | None = None) -> Path:
    """A virtual environment for prime-agent's kernel, holding the Python skills of a release's ``skills`` folder when
    given, as prime-agent's own kernel setup leaves them."""
    create_venv(path, symlinks=True)
    for skill in sorted(skills.iterdir()) if skills else []:
        if (skill / "pyproject.toml").is_file():
            editable(path / "bin" / "python", skill)
    return path


def fake_uv(folder: Path, fails: str = "") -> Path:
    """``uv`` as the adapter calls it, ``uv pip install --quiet --python <python> -e <skill> ...``: every skill installed
    as an editable install leaves it and each call logged in ``uv.log`` beside it, or, given ``fails``, nothing installed
    and ``fails`` on stderr. Its folder."""
    folder.mkdir()
    (folder / "uv.py").write_text(f"""import sys
from pathlib import Path
args = sys.argv[1:]
with open({str(folder / "uv.log")!r}, "a") as log:
    log.write(" ".join(args) + "\\n")
if {fails!r}:
    sys.exit({fails!r})
site = next(Path(args[args.index("--python") + 1]).parent.parent.glob("lib/python*/site-packages"))
for skill in (Path(args[i + 1]) for i, arg in enumerate(args) if arg == "-e"):
    (site / f"_{{skill.name}}.pth").write_text(str(skill / "src") + "\\n")
""")
    (folder / "uv").write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(folder / "uv.py"))} "$@"\n')
    (folder / "uv").chmod(0o755)
    return folder


@pytest.fixture
def upstream():
    from .fake_upstream import FakeUpstream

    server = FakeUpstream().start()
    yield server
    server.stop()


def run_mode(tmp_path, monkeypatch, upstream, mode: str, extra_kwargs: dict | None = None,
             n_slots: int = 12):
    source = FakeSource(n_tasks=4)
    sched = tmp_path / f"{mode}.json"
    if mode == "fresh":
        # the transfer reference: every task met by a fresh system, three instances each
        epilogue_schedule(source.task_ids(), seed=1, source=source.name, reset_each=True, instances_per_task=3).save(sched)
    else:
        stream_schedule(source.task_ids(), n_slots, seed=1, source=source.name).save(sched)
    context = Attempt.create(f"pa {mode}", root=tmp_path / "wb")
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))
    monkeypatch.setenv("FAKE_AGENT_LOG", str(tmp_path / f"{mode}-commands.log"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-upstream")
    # the fake installed as the GPU nodes' prime-agent is, its kernel venv holding its bundled Python skills
    launcher = install_rust_sh(tmp_path / "prime-agent")
    venv = kernel_venv(tmp_path / "kernel-venv", tmp_path / "prime-agent" / "share" / "prime-agent" / "skills")
    kwargs = {"command": [str(launcher)], "kernel_venv": str(venv), "model": "vendor/model", "thinking": "low",
              "startup_timeout_s": 30, "timeout_s": 30, "sandbox": "none",
              # The child environment is an allowlist; the fake's own switches must be named.
              "passthrough_env": [k for k in os.environ if k.startswith("FAKE_AGENT_")],
              **(extra_kwargs or {})}
    code = main(["run", "--learner", "continual_arc_baselines.systems.prime_agent:PrimeAgentLearner",
                 "--learner-kwargs", json.dumps(kwargs), "--schedule", str(sched),
                 "--registry", "tests.fixtures:fake_registry",
                 "--upstream", upstream.base_url, "--llm-defaults", '{"reasoning_effort": "low"}',
                 "--quiet"])
    assert code == 0
    return context, tmp_path / f"{mode}-commands.log"


def records_of(context):
    return [json.loads(line) for line in (context.directory / "continual-arc" / "records.jsonl").read_text().splitlines()]


def adapter_log(context) -> str:
    return (context.directory / "system" / "adapter.log").read_text()


def test_workstreams_learns_and_is_measured_at_the_boundary(tmp_path, monkeypatch, upstream, capsys):
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["stream_score"] > 0
    records = records_of(context)
    # First exposure needs demos (cost 1); later exposures of a task are solved from memory (cost 0).
    first = [r for r in records if r["exposure_index"] == 0 and r["phase"] == "stream"]
    later = [r for r in records if r["exposure_index"] > 0 and r["phase"] == "stream"]
    assert first and all(r["cost"] == 1.0 for r in first)
    assert later and all(r["cost"] == 0.0 for r in later)
    # Proxy-measured usage lands in every record and in the cost journal: one model call per
    # action, plus the closing turn that carries the verdict (paid and journaled, never scored).
    assert all(r["resources"]["llm_requests"] == r["resources"]["actions"] + 1 for r in records)
    assert all(r["resources"]["llm_cost_usd"] > 0 for r in records)
    # one turn per action plus the closing turn that carries the instance's verdict
    assert all(r["resources"]["system_turns"] == r["resources"]["actions"] + 1 for r in records)
    assert all(r["resources"]["system_closings"] == 1 for r in records)
    summary = CostJournal(context.directory).summary()
    assert summary["entities"] == sum(r["resources"]["actions"] + 1 for r in records)
    # The system never saw the upstream key, and the run-level default reached the model call.
    sent = upstream.requests[0]
    assert sent["authorization"] == "Bearer sk-upstream" and sent["body"]["reasoning_effort"] == "low"
    params = next(json.loads(line)["payload"]["values"] for line in (context.directory / "events.jsonl").read_text().splitlines()
                  if json.loads(line)["kind"] == "parameters")
    assert params["model"] == "vendor/model" and params["system"]["system_version"] == "fake-prime-agent 0.0.1"
    assert params["proxy"]["enabled"] and params["proxy"]["upstream_key_present"]
    assert params["system"]["sandbox"] == "none" and params["alias_style"] == "hex"
    # No resets on the stream, one process for the whole run, and a fresh conversation for
    # every instance, the first included.
    assert commands.read_text().count("new_session") == len(records)
    assert all(r["resources"]["system_sessions"] == 1.0 for r in records)
    assert (context.directory / "system" / "state" / "agent" / "models.json").is_file()
    models = json.loads((context.directory / "system" / "state" / "agent" / "models.json").read_text())
    entry = next(iter(models["providers"].values()))["models"][0]
    assert entry["thinkingLevelMap"] == {"xhigh": "xhigh", "max": "max"}  # else pi-ai clamps both to high
    assert models["providers"]["arc-proxy"]["baseUrl"] == params["proxy"]["base_url"]


def test_system_never_sees_real_task_ids(tmp_path, monkeypatch, upstream):
    """Task ids are lookup keys into public solutions; systems get run-specific aliases."""
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    # Everything the system was told went through the proxy (the fake forwards its prompts).
    prompts = " ".join(json.dumps(r["body"]) for r in upstream.requests)
    assert "Task id: task-" in prompts
    for task_id in {r["task_id"] for r in records_of(context)}:
        assert task_id not in prompts
    # The records on disk keep the real ids, so analysis is unaffected.
    assert all(r["task_id"].startswith("fake") for r in records_of(context))


def test_fresh_reference_wipes_state_between_tasks(tmp_path, monkeypatch, upstream, capsys):
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "fresh")
    records = records_of(context)
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    for rows in by_task.values():
        # Within a task block the memory persists; the next block starts from nothing.
        assert rows[0]["cost"] == 1.0 and all(r["cost"] == 0.0 for r in rows[1:])
    resets = [r for r in records if r["reset_before"]]
    assert len(resets) == len(by_task)
    # A reset stops the process, wipes its directory and starts afresh (one get_state handshake per start).
    assert commands.read_text().count("get_state") == len(by_task)
    # The adapter's own forensics survive the wipes.
    assert (context.directory / "system" / "adapter.log").is_file()
    assert (context.directory / "system" / "rpc-events.jsonl").is_file()


def test_refine_is_invoked_between_instances(tmp_path, monkeypatch, upstream):
    _, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"refine_every": 2})
    lines = commands.read_text().splitlines()
    assert lines.count("prompt") > 12  # actions plus refine prompts
    assert "abort" not in lines  # refine_complete was recognised as the end of the turn


def test_home_agent_dir_is_the_agent_directory(tmp_path):
    """prime-agent's documented path for a personal skill (~/.prime/agent/skills/) is the one it loads: ~/.prime/agent
    links to the adapter's agent directory, relatively; a directory already there has its skills moved in and the rest
    kept aside; linking again changes nothing."""
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner

    learner = PrimeAgentLearner(command=[sys.executable, FAKE], sandbox="none")
    learner.workdir = tmp_path / "system"
    before = learner.home_dir / ".prime" / "agent" / "skills" / "arc-lib"
    before.mkdir(parents=True)
    (before / "SKILL.md").write_text("---\nname: arc-lib\ndescription: grid helpers\n---\n")
    learner.agent_dir.mkdir(parents=True)
    learner._link_home_agent_dir()
    link = learner.home_dir / ".prime" / "agent"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link)) and link.resolve() == learner.agent_dir.resolve()
    assert (learner.agent_dir / "skills" / "arc-lib" / "SKILL.md").exists() and (link / "skills" / "arc-lib").is_dir()
    assert [p.name.startswith("agent.before-link.") for p in (learner.home_dir / ".prime").iterdir() if p != link] == [True]
    learner._link_home_agent_dir()
    assert link.is_symlink() and link.resolve() == learner.agent_dir.resolve()


def test_the_finish_protocol_ends_every_instance_with_the_verdict_alone(tmp_path, monkeypatch, upstream):
    """end="finish" (the harness rows): the rules state facts only, prime-agent's one fact about its store among
    them; every instance ends with the verdict alone, and the agent ends it with finish."""
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"end": "finish"})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    verdicts = [t for t in turns if t["kind"] == "verdict"]
    assert len(verdicts) == len(records) == sum(r["resources"]["system_finishes"] for r in records)
    assert not any(t["kind"] in ("closing", "storage") for t in turns)
    assert all(t["message"].startswith("Your submission was") and "\n" not in t["message"] for t in verdicts)
    firsts = [t for t in turns if t["kind"] == "observation" and t.get("fresh")]
    assert firsts and all('{"action": "finish"}' in t["message"] and "global_=True" in t["message"]
                          and "worth keeping" not in t["message"] for t in firsts)


def test_the_stateless_row_ends_its_instances_the_same_way(tmp_path, monkeypatch, upstream):
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"end": "finish", "stateless": True})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    assert sum(t["kind"] == "verdict" for t in turns) == len(records) == sum(r["resources"]["system_finishes"] for r in records)


def test_a_storage_turn_ends_every_instance_with_the_verdict(tmp_path, monkeypatch, upstream):
    """storage_turn (the team rows): every instance ends with a turn of the agent's own in which to save what will help
    later, carrying the verdict and the fact about prime-agent its agent cannot know (entries without global_=True die
    with the session), in place of the closing turn."""
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"storage_turn": True})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    storage = [t for t in turns if t["kind"] == "storage"]
    assert len(storage) == len(records) == sum(r["resources"]["system_storage_turns"] for r in records)
    assert not any(t["kind"] == "closing" for t in turns)
    assert all(t["message"].startswith("Your submission was") and "only what you save outlasts it" in t["message"]
               and "global_=True" in t["message"] and t["message"].endswith("reply with the single word: done.")
               for t in storage)
    # the documented skills path is the loaded one in a real run too
    state = context.directory / "system" / "state"
    assert (state / "home" / ".prime" / "agent").resolve() == (state / "agent").resolve()


def test_refine_via_rpc_is_the_refinement_call_alone_with_the_verdict(tmp_path, monkeypatch, upstream):
    """refine_via rpc (the team rows): after every instance the adapter starts the global refinement itself, with the
    RPC refine command and the instance's verdict as its instructions. No closing turn and no turn asking the model to
    refine: the refinement's is the only model call after the instance's last action."""
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"refine_every": 1, "refine_via": "rpc"})
    records = records_of(context)
    lines = commands.read_text().splitlines()
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    assert lines.count("refine") == len(records) and "abort" not in lines
    assert lines.count("prompt") == sum(1 for t in turns if t["kind"] in ("observation", "reprompt"))
    assert not any(t["kind"] == "closing" for t in turns)
    refines = [t for t in turns if t["kind"] == "refine"]
    assert len(refines) == len(records) == sum(r["resources"]["system_refines"] for r in records)
    assert all(t["message"].startswith("Your submission was") and "This piece of work is finished" in t["message"]
               and t["reply"] == "fake global refine" for t in refines)
    # one model call per refinement, through the proxy, with the verdict in it
    asked = [r["body"]["messages"][-1]["content"] for r in upstream.requests if r["body"] and r["body"].get("messages")]
    assert sum(1 for m in asked if isinstance(m, str) and m.startswith("REFINE\nYour submission was")) == len(records)


def test_provider_error_retry_is_not_an_empty_reply(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_ERROR_FIRST", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    records = records_of(context)
    assert sum(r["resources"]["system_parse_failures"] for r in records) == 0
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert "waiting for prime-agent's retry" in adapter_log(context)
    events = [json.loads(line) for line in (context.directory / "system" / "rpc-events.jsonl").read_text().splitlines()]
    assert {"response", "agent_end", "agent_start"} <= {e["type"] for e in events}
    assert all(len(m["text"]) <= 2000 for e in events if e["type"] == "agent_end" for m in e["messages"])


def test_lost_closing_events_are_recovered_over_the_command_path(tmp_path, monkeypatch, upstream):
    """prime-agent 0.9.5's daemon relay drops the tail of a turn (turn_end, agent_end)
    while get_state still answers: the adapter notices the quiet stream, asks the worker,
    and reads the reply, in seconds instead of an idle timeout."""
    monkeypatch.setenv("FAKE_AGENT_LOSE_TAIL", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0.3, "idle_timeout_s": 20, "final_turn_grace_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_stalls"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert sum(r["resources"]["system_idle_timeouts"] for r in records) == 0
    assert all(r["solved"] for r in records)
    assert "recovered the reply over the command path" in adapter_log(context)
    lines = commands.read_text().splitlines()
    # The worker was asked over the command path; the reply itself came from the final
    # message_end already observed, so no abort and no restart were needed.
    assert lines.count("get_state") >= 2 and "abort" not in lines


def test_repeated_event_loss_reattaches_the_client(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_LOSE_TAIL", "2")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0.3, "idle_timeout_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_stalls"] for r in records) == 2
    assert sum(r["resources"]["system_restarts"] for r in records) == 1
    assert all(r["solved"] for r in records)
    assert "event attachment is dead" in adapter_log(context)
    assert commands.read_text().count("get_state") >= 3  # start, probes, restart


def test_provider_retry_delay_is_honoured(tmp_path, monkeypatch, upstream):
    """prime-agent's auto-retry backoff grows to minutes; the adapter waits for the retry it
    announces instead of declaring the turn empty after a fixed 15 s."""
    monkeypatch.setenv("FAKE_AGENT_RATE_LIMITED", "2")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"quiet_probe_s": 0.5})
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_parse_failures"] for r in records) == 0
    assert sum(r["resources"]["system_provider_retries"] for r in records) == 0
    assert sum(r["resources"]["system_stalls"] for r in records) == 0
    assert adapter_log(context).count("is retrying the model call") == 2


def test_provider_outage_resends_the_prompt_instead_of_giving_up(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_PROVIDER_DOWN", "2")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0.3, "provider_backoff_s": 0.2})
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert not any(r["gave_up"] for r in records)
    assert sum(r["resources"]["system_provider_retries"] for r in records) == 2
    # The previous turn's text was never mistaken for this turn's reply.
    assert sum(r["resources"]["system_stalls"] for r in records) == 0
    assert "re-sending the prompt" in adapter_log(context)


def test_final_reply_without_agent_end_is_taken_after_grace(tmp_path, monkeypatch, upstream):
    """The fallback when probing is disabled: a turn that ended on a final reply is taken
    after a short grace instead of waiting for an agent_end that never comes."""
    monkeypatch.setenv("FAKE_AGENT_STALL_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0, "final_turn_grace_s": 0.5, "idle_timeout_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_stalls"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert all(r["solved"] for r in records)
    assert "taking the reply" in adapter_log(context)
    assert "abort" in commands.read_text().splitlines()


def test_a_prompt_refused_while_prime_agent_compacts_waits_and_is_sent_again(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_COMPACT_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_restarts"] for r in records) == 0  # not taken for a dead system
    assert "busy (compacting)" in adapter_log(context)


def test_a_run_hung_between_turns_is_restarted_after_the_idle_timeout(tmp_path, monkeypatch, upstream):
    # a run left going after its turn sticks in a cell that never returns: the next prompt waits for it only while it
    # shows signs of life, then the system is restarted as a hung turn's is, long before the turn's hard ceiling
    monkeypatch.setenv("FAKE_AGENT_BUSY_HANG_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"idle_timeout_s": 1.5, "timeout_s": 60})
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_idle_timeouts"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 1
    assert "busy with no RPC event and no proxy traffic" in adapter_log(context)
    assert "abort" in commands.read_text().splitlines()


def test_a_cell_printing_forever_is_restarted_after_the_idle_timeout(tmp_path, monkeypatch, upstream):
    # a cell prints in a loop that never ends: its streamed output is no sign that the run moves on, so the turn is
    # aborted and the system restarted once nothing else has come for the idle timeout, long before the turn's hard
    # ceiling
    monkeypatch.setenv("FAKE_AGENT_PRINT_HANG_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"idle_timeout_s": 1.5, "timeout_s": 60})
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_idle_timeouts"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 1
    assert "no RPC event and no proxy traffic" in adapter_log(context)
    assert "abort" in commands.read_text().splitlines()


def test_a_run_stopped_to_compact_is_waited_for_not_taken_for_an_empty_reply(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_COMPACT_MIDRUN_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert sum(r["resources"]["system_parse_failures"] for r in records) == 0
    assert "stopped after a tool result to compact" in adapter_log(context)


def test_silent_system_is_restarted_after_idle_timeout(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_HANG_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"idle_timeout_s": 1.5, "timeout_s": 60})
    records = records_of(context)
    assert sum(r["resources"]["system_idle_timeouts"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 1
    assert all(r["solved"] for r in records)
    assert commands.read_text().count("get_state") == 2  # the original start and the restart
    assert "no RPC event and no proxy traffic" in adapter_log(context)


def test_action_budget_stops_an_endless_loop(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_LOOP_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"max_model_calls": 5, "idle_timeout_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_budget_hits"] for r in records) == 1
    assert sum(r["resources"]["system_reprompts"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert all(r["solved"] for r in records)
    assert "action budget exceeded" in adapter_log(context)
    # The aborted run's text had no action, so the system was asked once to decide.
    body = " ".join(json.dumps(r["body"]) for r in upstream.requests)
    assert "exceeded the time or model-call budget" in body


def test_dead_system_costs_actions_not_the_run(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_DIE", "1")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"max_restarts": 1}, n_slots=3)
    records = records_of(context)
    assert len(records) == 3 and all(r["gave_up"] for r in records)
    assert all(r["resources"]["system_forced_give_ups"] == 1 for r in records)
    assert all(r["resources"]["system_restarts"] == 1 for r in records)
    assert "this action is a give-up" in adapter_log(context)


def test_sandbox_none_must_be_explicit(tmp_path, monkeypatch):
    from continual_arc_baselines.systems.sandbox import SandboxUnavailable, resolve_backend

    monkeypatch.setattr("continual_arc_baselines.systems.sandbox.platform.system", lambda: "Linux")
    monkeypatch.setattr("continual_arc_baselines.systems.sandbox.shutil.which", lambda name: None)  # a Linux host without bubblewrap
    with pytest.raises(SandboxUnavailable):
        resolve_backend("auto")
    assert resolve_backend("none") == "none"


@pytest.mark.skipif(platform.system() != "Darwin", reason="seatbelt is macOS only")
def test_seatbelt_confines_the_system(tmp_path, monkeypatch, upstream):
    """Inside the sandbox the system cannot read the repository that launched it."""
    monkeypatch.setenv("FAKE_AGENT_PROBE", str(REPO / "pyproject.toml"))
    assert (REPO / "pyproject.toml").is_file()
    context, commands = run_mode(
        tmp_path, monkeypatch, upstream, "workstreams",
        {"sandbox": "seatbelt", "sandbox_read_paths": [sys.prefix, str(Path(FAKE).parent)]},
        n_slots=4,
    )
    lines = commands.read_text().splitlines()
    assert "probe:denied" in lines and "probe:ok" not in lines
    assert all(r["solved"] for r in records_of(context))  # the proxy was still reachable
    profile = (context.directory / "system" / "sandbox.sb").read_text()
    assert "(deny default)" in profile and "remote ip \"localhost:" in profile


def test_integrations_are_off(tmp_path):
    from continual_arc_baselines.systems.prime_agent import INTEGRATION_SKILLS, PrimeAgentLearner
    s = PrimeAgentLearner(model="m", sandbox="none", command=sys.executable)
    s.attach(tmp_path / "system", {})
    s._write_config()
    settings = json.loads((s.agent_dir / "settings.json").read_text())
    assert settings["bundledSkills"] == {"websearch": False}
    assert set(settings["skills"]) == {f"-{n}/SKILL.md" for n in INTEGRATION_SKILLS}
    assert "compaction" not in settings  # prime-agent's own threshold


def _configured(tmp_path):
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    s = PrimeAgentLearner(model="m", sandbox="none", command=sys.executable)
    s.attach(tmp_path / "system", {})
    s._write_config()
    return s


def test_the_provider_and_settings_are_written_afresh_at_every_start(tmp_path):
    """An agent's edit to settings.json lasts only until the next start, and a wipe leaves neither file for it to keep."""
    s = _configured(tmp_path)
    written = {name: (s.agent_dir / name).read_text() for name in ("models.json", "settings.json")}
    (s.agent_dir / "settings.json").write_text(json.dumps({"skills": ["+mine/SKILL.md"]}))
    (s.agent_dir / "models.json").write_text("{}")
    s._write_config()
    assert {name: (s.agent_dir / name).read_text() for name in written} == written
    s.wipe_state()
    assert not s.state_dir.exists() or not any(s.state_dir.iterdir())
    s._write_config()
    assert {name: (s.agent_dir / name).read_text() for name in written} == written


def test_the_adapter_follows_no_link_the_agent_leaves_in_its_state(tmp_path):
    """The adapter runs outside the sandbox: a link in the state at one of its own files is replaced, never written
    through, and a folder that leads out of the state stops the start with nothing written there."""
    from continual_arc_baselines.systems.base import SystemUnavailable
    host = tmp_path / "host"
    host.mkdir()
    victim = host / "victim.txt"
    victim.write_text("the host's own")
    s = _configured(tmp_path)
    for name in ("models.json", "settings.json"):
        (s.agent_dir / name).unlink()
        (s.agent_dir / name).symlink_to(victim)
    s._write_config()
    assert victim.read_text() == "the host's own"
    assert all((s.agent_dir / name).is_file() and not (s.agent_dir / name).is_symlink()
               for name in ("models.json", "settings.json"))
    for folder in (s.agent_dir, s.home_dir / ".prime"):
        s.wipe_state()
        folder.parent.mkdir(parents=True, exist_ok=True)
        folder.symlink_to(host, target_is_directory=True)
        with pytest.raises(SystemUnavailable, match="leads out of the agent's state"):
            s._write_config()
        assert sorted(p.name for p in host.iterdir()) == ["victim.txt"] and victim.read_text() == "the host's own"


def test_a_skills_link_in_home_prime_agent_is_moved_aside_not_followed(tmp_path):
    """A real ~/.prime/agent made before the link has its skills moved into the agent directory's, but a skills link
    there leads out of the state: it goes aside with the folder, and nothing it leads to moves."""
    host = tmp_path / "host"
    (host / "theirs").mkdir(parents=True)
    s = _configured(tmp_path)
    link = s.home_dir / ".prime" / "agent"
    link.unlink()
    link.mkdir()
    (link / "skills").symlink_to(host, target_is_directory=True)
    s._write_config()
    assert (host / "theirs").is_dir()
    assert link.is_symlink() and link.resolve() == s.agent_dir.resolve()
    assert not (s.agent_dir / "skills" / "theirs").exists()
    aside = next((s.home_dir / ".prime").glob("agent.before-link.*"))
    assert (aside / "skills").is_symlink()


def test_a_wipe_removes_a_link_as_a_link(tmp_path):
    """A wipe deletes the state's entries, an agent directory the agent made a link to the host as the link alone."""
    host = tmp_path / "host"
    host.mkdir()
    (host / "keep.txt").write_text("the host's own")
    s = _configured(tmp_path)
    shutil.rmtree(s.agent_dir)
    s.agent_dir.symlink_to(host, target_is_directory=True)
    s.wipe_state()
    assert (host / "keep.txt").read_text() == "the host's own"
    assert not os.path.lexists(s.agent_dir)


def test_compaction_can_come_earlier(tmp_path):
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    s = PrimeAgentLearner(model="m", sandbox="none", command=sys.executable, context_window=131_072, compact_at=49_152)
    s.attach(tmp_path / "system", {})
    s._write_config()
    # prime-agent compacts once a finished turn's context passes contextWindow - reserveTokens
    assert json.loads((s.agent_dir / "settings.json").read_text())["compaction"] == {"reserveTokens": 81_920}
    s = PrimeAgentLearner(model="m", sandbox="none", command=sys.executable, context_window=131_072, compact_at=32_768,
                          compact_keep=8_000)
    s.attach(tmp_path / "system2", {})
    s._write_config()
    assert json.loads((s.agent_dir / "settings.json").read_text())["compaction"] == {"reserveTokens": 98_304,
                                                                                     "keepRecentTokens": 8_000}
    with pytest.raises(ValueError):
        PrimeAgentLearner(model="m", context_window=131_072, compact_at=200_000)


def test_budget_abort_with_nothing_said_is_reprompted_not_restarted(tmp_path, monkeypatch, upstream):
    """A turn the adapter cut for budget can end with no text at all. That is not a dead
    system: the follow-up carries the budget notice and the instance goes on."""
    monkeypatch.setenv("FAKE_AGENT_LOOP_SILENT", "1")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"action_budget_s": 2, "reprompt_budget_s": 30})
    records = records_of(context)
    assert sum(r["resources"]["system_budget_hits"] for r in records) >= 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert "asking for a decision" in adapter_log(context)


def test_stateless_row_wipes_and_restarts_before_every_instance(tmp_path, monkeypatch, upstream):
    """Row H0: the harness is stopped, its state deleted and started again before each instance, the first included."""
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"stateless": True, "refine_every": 1, "storage_turn": True})
    records = records_of(context)
    assert len(records) > 2 and all(r["solved"] for r in records)
    # no storage turn either: the closing carries the verdict, as nothing saved could last
    assert sum(r["resources"]["system_storage_turns"] for r in records) == 0
    assert all(r["resources"]["system_wipes"] == r["resources"]["system_resets"] == 1 for r in records)
    assert sum(r["resources"]["system_refines"] for r in records) == 0   # nothing could survive it
    assert commands.read_text().count("get_state") == len(records)        # one start per instance
    assert commands.read_text().count("new_session") == 0
    # each start writes the provider configuration again after the wipe that took it, as it rebuilds the stores
    state = context.directory / "system" / "state" / "agent"
    assert (state / "models.json").is_file()
    params = next(json.loads(line)["payload"]["values"] for line in (context.directory / "events.jsonl").read_text().splitlines()
                  if json.loads(line)["kind"] == "parameters")
    assert params["system"]["stateless"] is True


def test_turn_log_records_every_exchange(tmp_path, monkeypatch, upstream):
    """turns.jsonl beside the run: one line per message the adapter sent, with the reply, the kind
    (observation, reprompt, closing, refine) and the instance it belonged to."""
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"refine_every": 1})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    kinds = {t["kind"] for t in turns}
    assert {"observation", "closing", "refine"} <= kinds
    assert sum(1 for t in turns if t["kind"] == "closing") == len(records)
    assert sum(1 for t in turns if t["kind"] == "refine") == len(records)
    assert sum(1 for t in turns if t["kind"] == "observation") == sum(r["resources"]["actions"] for r in records)
    first = next(t for t in turns if t["kind"] == "observation")
    assert first["message"].startswith("You are being evaluated") and first["reply"] and "task" in first
    assert all("t" in t and "reply" in t and "message" in t for t in turns)


def test_the_turn_budget_leaves_out_what_the_proxy_credits(tmp_path, monkeypatch):
    """A row whose proxy answers some calls itself spends part of each call outside the model: its
    proxy credits those seconds to the tracker (Tracker.credit_s), and the turn's time budget and hard timeout move out
    by what was credited during the turn, so the row is held to the same budget of model time as any other. A turn the
    model keeps working at is cut after its budget plus the credit, not after its budget alone; with no credit, as for
    every other row, after the budget. A negative credit is model time a replayed reply stands for without taking it
    (served at once): the turn is cut that much sooner."""
    import time as _time

    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    from continual_arc_baselines.tracking import Tracker

    class Crediting(Tracker):
        """Credits every second of the turn's first ``upto`` seconds, as a proxy busy in the pipeline would."""

        def __init__(self, upto: float) -> None:
            super().__init__()
            self.t0, self.upto = _time.monotonic(), upto

        @property
        def credit_s(self) -> float:
            return min(_time.monotonic() - self.t0, self.upto)

        @credit_s.setter
        def credit_s(self, value: float) -> None:
            pass

    class Debiting(Tracker):
        """Takes ``seconds`` off the turn's clock once the turn has begun, as a replayed reply's recorded generation
        does: the turn reads its starting credit first, then every later read has the debit."""

        def __init__(self, seconds: float) -> None:
            super().__init__()
            self.seconds, self.reads = seconds, 0

        @property
        def credit_s(self) -> float:
            self.reads += 1
            return 0.0 if self.reads == 1 else -self.seconds

        @credit_s.setter
        def credit_s(self, value: float) -> None:
            pass

    def cut_after(tracker, budget_s: float = 1.0) -> tuple[str, float, PrimeAgentLearner]:
        s = PrimeAgentLearner(model="m", action_budget_s=budget_s, timeout_s=60, idle_timeout_s=60, quiet_probe_s=0)
        s.tracker = tracker
        monkeypatch.setattr(s, "_send", lambda command: None)
        monkeypatch.setattr(s, "_wait_response", lambda command, timeout: {})
        monkeypatch.setattr(s, "_abort", lambda collect_s=0.0: "")
        monkeypatch.setattr(s, "_note", lambda message: None)
        s._events.put({"type": "message_start", "message": {"role": "assistant"}})   # the model at work, then quiet
        t0 = _time.monotonic()
        outcome, _ = s._turn("go", budget_s=budget_s, max_calls=40, until=("agent_end",))
        return outcome, _time.monotonic() - t0, s

    outcome, seconds, s = cut_after(Tracker())
    assert outcome == "budget" and 0.9 <= seconds < 2.5 and s.stats["budget_hits"] == 1
    outcome, seconds, s = cut_after(Crediting(upto=2.0))
    assert outcome == "budget" and 2.9 <= seconds < 5.0 and s.stats["budget_hits"] == 1   # 1 s of model time plus 2 credited
    outcome, seconds, s = cut_after(Debiting(2.5), budget_s=3.0)
    assert outcome == "budget" and 0.3 <= seconds < 1.5 and s.stats["budget_hits"] == 1   # 3 s less the 2.5 debited
    t = Tracker()
    t.credit(1.5)
    t.credit(-3.0)   # a replayed reply's recorded seconds: counted though not taken
    assert t.credit_s == -1.5


def test_prime_agent_0_9_6s_bundled_skills_are_where_install_rust_sh_puts_them(tmp_path, monkeypatch):
    """install-rust.sh's launcher is a script, which no link leads through: the install root is its prefix, and the
    skills are in the release in share/prime-agent/ there."""
    from continual_arc_baselines.systems.prime_agent import bundled_python_skills
    from continual_arc_baselines.systems.sandbox import install_root

    launcher = install_rust_sh(tmp_path / ".local")
    monkeypatch.setenv("PATH", f"{launcher.parent}{os.pathsep}{os.environ['PATH']}")
    root = install_root("prime-agent")
    assert root == (tmp_path / ".local").resolve()
    skills = root / "share" / "prime-agent" / "skills"
    assert bundled_python_skills(root) == {module: skills / module.replace("_", "-") for module in KERNEL_SKILLS}


def test_prime_agent_0_9_5s_bundled_skills_are_in_its_release_under_its_typescript_installer(tmp_path, monkeypatch):
    """The TypeScript installer's launcher links to the release's executable: the install root is the release, whose
    Linear and Notion Python skills are left out with web search."""
    from continual_arc_baselines.systems.prime_agent import bundled_python_skills
    from continual_arc_baselines.systems.sandbox import install_root

    launcher = install_typescript(tmp_path)
    monkeypatch.setenv("PATH", f"{launcher.parent}{os.pathsep}{os.environ['PATH']}")
    root = install_root("prime-agent")
    assert root == (tmp_path / ".local" / "share" / "prime-agent" / "releases" / "0.9.5-linux-x64-bc4b0ed7").resolve()
    assert bundled_python_skills(root) == {module: root / "skills" / module.replace("_", "-") for module in KERNEL_SKILLS}


def test_the_bundled_skills_a_kernel_venv_lacks_are_installed_then_imported(tmp_path, monkeypatch):
    """A kernel venv as prime-agent's installer makes it holds none of the skills: they are installed from the
    prime-agent on PATH, editable, once per venv in a process, and import there after."""
    from continual_arc_baselines.systems.prime_agent import install_kernel_skills

    launcher, uv = install_rust_sh(tmp_path / ".local"), fake_uv(tmp_path / "uv")
    monkeypatch.setenv("PATH", os.pathsep.join([str(launcher.parent), str(uv), os.environ["PATH"]]))
    venv = kernel_venv(tmp_path / "kernel-venv")
    python, imports = venv / "bin" / "python", ["-c", "import " + ", ".join(KERNEL_SKILLS)]
    assert subprocess.run([python, *imports], capture_output=True).returncode == 1
    install_kernel_skills(venv, "prime-agent")
    install_kernel_skills(venv, "prime-agent")
    skills = (tmp_path / ".local" / "share" / "prime-agent" / "skills").resolve()
    assert (uv / "uv.log").read_text().splitlines() == [
        f"pip install --quiet --python {python} " + " ".join(f"-e {skills / m.replace('_', '-')}" for m in KERNEL_SKILLS)]
    assert subprocess.run([python, *imports], capture_output=True).returncode == 0


@pytest.mark.parametrize("case", ["no skills", "no Python skills", "no kernel venv", "uv failing",
                                  "skills that do not import", "no prime-agent"])
def test_a_start_without_the_bundled_skills_in_the_kernel_stops_before_prime_agent_runs(tmp_path, monkeypatch, case):
    """Whatever keeps the skills from the kernel stops the start before prime-agent runs: an install with no skills, or
    none in Python, to install, no kernel venv to install them into, an install that fails or after which they still do
    not import, or no prime-agent on PATH."""
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner

    skills = {"no skills": {}, "no Python skills": {name: False for name in SKILLS_0_9_6}}.get(case, SKILLS_0_9_6)
    launcher = install_rust_sh(tmp_path / ".local", skills)
    if case == "skills that do not import":
        (tmp_path / ".local" / "share" / "prime-agent" / "skills" / "refine" / "src" / "refine" / "__init__.py").write_text(
            "import refine_dependency\n")
    uv = fake_uv(tmp_path / "uv", "error: no network" if case == "uv failing" else "")
    monkeypatch.setenv("PATH", os.pathsep.join([str(launcher.parent), str(uv), os.environ["PATH"]]))
    venv = tmp_path / "kernel-venv" if case == "no kernel venv" else kernel_venv(tmp_path / "kernel-venv")
    command = "prime-agent-nowhere" if case == "no prime-agent" else "prime-agent"
    learner = PrimeAgentLearner(command=command, kernel_venv=str(venv), sandbox="none", startup_timeout_s=30)
    learner.attach(tmp_path / "system", {})
    named = r"\(agent_message, .*, rlm_heartbeat\)"
    expected = {"no skills": "no bundled Python skills in", "no Python skills": "no bundled Python skills in",
                "no kernel venv": "no kernel venv at",
                "uv failing": rf"uv could not install prime-agent's bundled Python skills {named} into .*: error: no "
                              "network$",
                "skills that do not import": rf"(?s){named} do not import in .* after installing them: .*No module named "
                                             "'refine_dependency'$",
                "no prime-agent": "prime-agent-nowhere is not on PATH"}
    try:
        with pytest.raises(RuntimeError, match=expected[case]) as stopped:
            learner.start()
    finally:
        learner.stop()
    assert type(stopped.value) is RuntimeError   # not SystemUnavailable, which the adapter answers with restarts
    assert not (tmp_path / "system" / "prime-agent.log").exists()


def site(venv: Path) -> Path:
    return next(venv.glob("lib/python*/site-packages"))


def agent_skill(folder: Path, body: str = "def run():\n    return 'grid tools ran'\n", dependencies: str = "") -> Path:
    """A Python skill as an agent writes one: a SKILL.md, a pyproject.toml and its package in src/. Its build backend is
    its own (backend.py beside it, named by backend-path), which leaves BUILT beside it if anything ever runs it."""
    module = folder.name.replace("-", "_")
    (folder / "src" / module).mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {folder.name}\ndescription: what {folder.name} does\n---\n")
    (folder / "pyproject.toml").write_text(
        f'[project]\nname = "{folder.name}"\nversion = "0.1.0"\ndependencies = [{dependencies}]\n\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n')
    (folder / "backend.py").write_text(f"open({str(folder / 'BUILT')!r}, 'w').close()\n")
    (folder / "src" / module / "__init__.py").write_text(body)
    return folder


def kernel_learner(tmp_path, monkeypatch):
    """A factory of learners on the fake prime-agent with a kernel (FAKE_AGENT_KERNEL_LOG, read by kernel_starts), all
    sharing one kernel venv that holds the bundled skills as the adapter's install leaves them, with no network: uv,
    which nothing should call, fails. The shared venv."""
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner

    launcher, uv = install_rust_sh(tmp_path / ".local"), fake_uv(tmp_path / "uv", "error: no network")
    monkeypatch.setenv("PATH", os.pathsep.join([str(launcher.parent), str(uv), os.environ["PATH"]]))
    monkeypatch.setenv("FAKE_AGENT_KERNEL_LOG", str(tmp_path / "kernel.log"))
    venv = kernel_venv(tmp_path / "kernel-venv", tmp_path / ".local" / "share" / "prime-agent" / "skills")
    return lambda: PrimeAgentLearner(command="prime-agent", kernel_venv=str(venv), sandbox="none",
                                     startup_timeout_s=30, passthrough_env=["FAKE_AGENT_KERNEL_LOG"]), venv


def kernel_starts(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "kernel.log").read_text().splitlines()]


def imports(venv: Path, module: str) -> bool:
    return subprocess.run([venv / "bin" / "python", "-c", f"import {module}"], capture_output=True).returncode == 0


def files_of(*folders: Path) -> dict[Path, bytes]:
    """Every file in ``folders`` but Python's bytecode caches, which importing the bundled skills writes beside their
    sources (the adapter's check that they import does, and so does a test's)."""
    return {path: path.read_bytes() for folder in folders for path in sorted(folder.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


@pytest.mark.parametrize("root", ["agent/skills", "workspace/.prime/agent/skills", "workspace/.agents/skills",
                                  ".agents/skills", "home/.agents/skills"])
def test_a_python_skill_the_agent_writes_imports_in_the_kernel_from_its_next_session(tmp_path, monkeypatch, root):
    """A Python skill the agent writes during a session, in any folder of its state that prime-agent finds skills in,
    imports in the kernel from the next session on, as prime-agent's own kernel setup would install it, while nothing of
    it runs outside the kernel, its build backend never. It goes into the learner's own copy of the kernel venv beside
    the bundled skills, and nothing reaches the shared venv or prime-agent's install."""
    make, venv = kernel_learner(tmp_path, monkeypatch)
    learner = make()
    learner.attach(tmp_path / "system", {})
    try:
        learner.start()
        shared = files_of(venv, tmp_path / ".local")   # with the bundled skills installed, and the install's lock
        skill = agent_skill(learner.state_dir / root / "grid-tools")
        learner.new_session()
        learner.new_session()
    finally:
        learner.stop()
    ran = {"grid_tools": "grid tools ran"}
    assert kernel_starts(tmp_path) == [{"session": 0, "ran": {}, "disabled": []},
                                       {"session": 1, "ran": ran, "disabled": []},
                                       {"session": 2, "ran": ran, "disabled": []}]
    assert not (skill / "BUILT").exists() and not (tmp_path / "uv" / "uv.log").exists()
    assert (site(learner.own_venv) / "arc-agent-skills.pth").read_text() == f"{skill / 'src'}\n"
    assert imports(learner.own_venv, ", ".join(KERNEL_SKILLS))
    assert files_of(venv, tmp_path / ".local") == shared and not imports(venv, "grid_tools")
    assert "the agent's Python skills on the kernel's path: " + f"{root}/grid-tools/src" in (
        tmp_path / "system" / "adapter.log").read_text()


def test_what_the_kernel_imports_follows_the_skills_folder_as_each_session_starts(tmp_path, monkeypatch):
    """A skill that is no longer a Python skill, is removed or is wiped with the rest of the state (row H0's wipe before
    every instance) leaves nothing in the kernel from its next session, though the learner's venv outlasts the wipe; one
    there before the first start imports from the first session."""
    make, venv = kernel_learner(tmp_path, monkeypatch)
    learner = make()
    learner.attach(tmp_path / "system", {})
    skill = agent_skill(learner.agent_dir / "skills" / "grid-tools")
    importable = []
    try:
        learner.start()
        importable.append(imports(learner.own_venv, "grid_tools"))
        (skill / "pyproject.toml").rename(tmp_path / "pyproject.toml")   # no longer a Python skill
        learner.new_session()
        importable.append(imports(learner.own_venv, "grid_tools"))
        (tmp_path / "pyproject.toml").rename(skill / "pyproject.toml")
        learner.new_session()
        importable.append(imports(learner.own_venv, "grid_tools"))
        shutil.rmtree(skill)
        learner.new_session()
        importable.append(imports(learner.own_venv, "grid_tools"))
        agent_skill(skill)
        learner.new_session()
        importable.append(imports(learner.own_venv, "grid_tools"))
        before_the_wipe = site(learner.own_venv).stat().st_ino
        learner.stop()
        learner.wipe_state()
        learner.start()
        importable.append(imports(learner.own_venv, "grid_tools"))
    finally:
        learner.stop()
    assert importable == [True, False, True, False, True, False]
    assert site(learner.own_venv).stat().st_ino == before_the_wipe   # the same venv, not a new copy
    assert [start["ran"] for start in kernel_starts(tmp_path)] == [
        {"grid_tools": "grid tools ran"}, {}, {"grid_tools": "grid tools ran"}, {}, {"grid_tools": "grid tools ran"}, {}]
    assert not (site(learner.own_venv) / "arc-agent-skills.pth").exists()
    assert imports(learner.own_venv, ", ".join(KERNEL_SKILLS))
    assert not (site(venv) / "arc-agent-skills.pth").exists()


def test_a_state_that_comes_without_its_venv_has_its_skills_in_the_kernel_from_its_first_session(tmp_path, monkeypatch):
    """A state copied without the learner's venv, as a collective's snapshot is restored or a training snapshot is put
    back, has its Python skills in the kernel from the first session of the learner that plays it."""
    from continual_arc_baselines.collective import restore_state, snapshot_state

    make, _ = kernel_learner(tmp_path, monkeypatch)
    first = make()
    first.attach(tmp_path / "system", {})
    try:
        first.start()
        agent_skill(first.agent_dir / "skills" / "grid-tools")
    finally:
        first.stop()
    snapshot_state(first.state_dir, tmp_path / "snapshot", max_file_bytes=None)
    restore_state(tmp_path / "snapshot", tmp_path / "restored")
    second = make()
    second.attach(tmp_path / "second", {}, state_dir=tmp_path / "restored")
    try:
        second.start()
    finally:
        second.stop()
    assert kernel_starts(tmp_path)[-1] == {"session": 0, "ran": {"grid_tools": "grid tools ran"}, "disabled": []}
    assert (site(second.own_venv) / "arc-agent-skills.pth").read_text() == (
        f"{tmp_path / 'restored' / 'agent' / 'skills' / 'grid-tools' / 'src'}\n")


def test_a_skill_whose_dependency_the_venv_lacks_is_disabled_with_prime_agents_warning(tmp_path, monkeypatch):
    """Nothing is fetched: a skill that needs a package the kernel's venv does not hold fails to import, and prime-agent
    disables it with its warning, as it does any Python skill that does not import in the interpreter it is given; the
    agent's other skills import."""
    make, _ = kernel_learner(tmp_path, monkeypatch)
    learner = make()
    learner.attach(tmp_path / "system", {})
    try:
        learner.start()
        agent_skill(learner.agent_dir / "skills" / "grid-tools")
        plotter = agent_skill(learner.agent_dir / "skills" / "plotter", "import plotting_library\n",
                              dependencies='"plotting-library>=2"')
        learner.new_session()
    finally:
        learner.stop()
    assert kernel_starts(tmp_path)[-1] == {"session": 1, "ran": {"grid_tools": "grid tools ran"}, "disabled": ["plotter"]}
    assert ("Python skills unavailable in PRIME_AGENT_KERNEL_PYTHON and will be disabled: plotter (plotter)"
            in (tmp_path / "system" / "prime-agent.log").read_text())
    assert not (plotter / "BUILT").exists() and not (tmp_path / "uv" / "uv.log").exists()


def test_a_skill_reached_through_a_link_stays_out_of_the_kernel(tmp_path, monkeypatch):
    """The adapter reads the state with no link followed: a skill linked into the agent's skills from outside the state,
    and a skills folder that is a link to one outside it, stay out of the kernel, so nothing outside the state is put on
    its path; prime-agent, which follows links, finds them and disables them."""
    make, _ = kernel_learner(tmp_path, monkeypatch)
    learner = make()
    learner.attach(tmp_path / "system", {})
    linked = agent_skill(tmp_path / "outside" / "linked-skill")
    far = agent_skill(tmp_path / "far" / "skills" / "far-skill")
    try:
        learner.start()
        (learner.agent_dir / "skills").mkdir()
        (learner.agent_dir / "skills" / "linked-skill").symlink_to(linked)
        (learner.home_dir / ".agents").mkdir()
        (learner.home_dir / ".agents" / "skills").symlink_to(far.parent)
        learner.new_session()
    finally:
        learner.stop()
    assert kernel_starts(tmp_path)[-1] == {"session": 1, "ran": {}, "disabled": ["linked_skill", "far_skill"]}
    assert not (site(learner.own_venv) / "arc-agent-skills.pth").exists()


def test_a_skill_named_as_a_bundled_one_leaves_prime_agents_own_in_the_kernel(tmp_path, monkeypatch):
    """The agent's skills are on the kernel's path after prime-agent's bundled ones, so a package of the agent's that has
    a bundled skill's name (refine) leaves prime-agent's own in the kernel: the harness's mechanics stay as shipped."""
    make, _ = kernel_learner(tmp_path, monkeypatch)
    learner = make()
    learner.attach(tmp_path / "system", {})
    agent_skill(learner.agent_dir / "skills" / "refine", "def run():\n    return 'the agent wrote this refine'\n")
    try:
        learner.start()
    finally:
        learner.stop()
    assert kernel_starts(tmp_path) == [{"session": 0, "ran": {"refine": None}, "disabled": []}]


def test_the_kernel_runs_on_the_learners_own_venv_which_the_sandbox_binds_read_only(tmp_path, monkeypatch):
    """prime-agent is given the interpreter of the learner's own venv, in its workdir outside its state, which the
    sandbox binds in read-only, as it does the shared venv: the agent cannot write the venv its kernel runs on."""
    from continual_arc_baselines.systems import sandbox

    make, venv = kernel_learner(tmp_path, monkeypatch)
    learner = make()
    learner.attach(tmp_path / "system", {})
    try:
        learner.start()
        spec, env = learner._sandbox_spec(), learner._child_env()
    finally:
        learner.stop()
    own = learner.own_venv
    assert env["PRIME_AGENT_KERNEL_PYTHON"] == str(own / "bin" / "python")
    assert own.parent == learner.workdir and learner.state_dir not in own.parents
    cmd = sandbox.bwrap_command(learner._argv(), spec)
    ro = [(cmd[i + 1], cmd[i + 2]) for i, arg in enumerate(cmd[:-2]) if arg == "--ro-bind"]
    rw = [cmd[i + 1] for i, arg in enumerate(cmd[:-2]) if arg == "--bind"]
    own, venv = own.resolve(), venv.resolve()   # the sandbox binds the real paths
    assert (str(own), str(own)) in ro and (str(venv), str(venv)) in ro
    assert not any(own == Path(path) or Path(path) in own.parents for path in rw)


def test_the_agents_python_skills_are_read_off_its_state_with_no_link_followed(tmp_path):
    """agent_python_skill_sources: in each folder of the state that prime-agent finds skills in, the src/ of every skill
    holding a pyproject.toml and a src/ folder, in the order of the folders and by name within one, never what a skill's
    files say; a skill's own folders, dot-folders and node_modules are not searched; anything reached through a link,
    a file in a folder's place and a name that would not be one line of a .pth file are left out."""
    from continual_arc_baselines.systems.prime_agent import agent_python_skill_sources

    state, outside = tmp_path / "state", tmp_path / "outside"
    skills = state / "agent" / "skills"
    for folder in ("b-skill", "a-skill", "group/nested", "group/nested/inner", ".hidden/h", "node_modules/n",
                   "md-link", "src-link", "new\nline"):
        agent_skill(skills / folder)
    (skills / "markdown-only").mkdir()
    (skills / "markdown-only" / "SKILL.md").write_text("---\nname: markdown-only\ndescription: words\n---\n")
    (skills / "linked").symlink_to(agent_skill(outside / "linked"))
    (skills / "md-link" / "SKILL.md").unlink()
    (skills / "md-link" / "SKILL.md").symlink_to(outside / "linked" / "SKILL.md")
    (skills / "src-link" / "src").rename(skills / "src-link" / "source")
    (skills / "src-link" / "src").symlink_to("source")
    agent_skill(state / "home" / ".agents" / "skills" / "home-skill")
    (state / "workspace").mkdir()
    (state / "workspace" / ".agents").symlink_to(outside)
    agent_skill(outside / "skills" / "through-a-linked-folder")
    os.mkfifo(state / ".agents")
    roots = [Path("workspace/.prime/agent/skills"), Path("workspace/.agents/skills"), Path(".agents/skills"),
             Path("agent/skills"), Path("home/.agents/skills")]
    assert agent_python_skill_sources(state, roots) == [
        skills / "a-skill" / "src", skills / "b-skill" / "src", skills / "group" / "nested" / "src",
        state / "home" / ".agents" / "skills" / "home-skill" / "src"]
    assert agent_python_skill_sources(tmp_path / "never-started", roots) == []
