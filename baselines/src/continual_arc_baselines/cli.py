"""``arc-baselines``: run a learner over a Continual-ARC schedule, play a harness's frozen tests, fork a run, install a
submission.

``arc-baselines run`` builds the schedule (``--config``) or loads it (``--schedule``), plays the learner over it with
the benchmark's runner and writes the run directory (``--out``) and the attempt beside it
(:mod:`continual_arc_baselines.attempt`). Unless ``--no-proxy`` is given, a local OpenAI-compatible proxy is started
for the run and its address is exported as ``OPENAI_BASE_URL`` / ``OPENROUTER_BASE_URL`` / ``ARC_LLM_BASE_URL`` (with a
placeholder key) before the learner is built, so any system the learner spawns, and any client the learner builds from
the environment, is measured at the HTTP boundary without further wiring. ``--log-state`` commits a harness's state at
the end of every round (:mod:`continual_arc_baselines.statelog`), and ``--test`` plays a frozen test after the stream
in the same process, for a learner that is not a harness adapter (:func:`play_test`).

``arc-baselines test`` plays a frozen test of a harness's stream, one instance at a time from the state the stream
left or, with ``--collective``, as a round of users forked from the stream's last commit
(:mod:`continual_arc_baselines.frozen`). ``arc-baselines fork`` takes a run that logged its state up again from any
round (:mod:`continual_arc_baselines.fork`), and ``arc-baselines install-submission`` installs a submitted learner for
:class:`~continual_arc_baselines.systems.hosted.HostedLearner`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from continual_arc.aliases import ALIAS_STYLES


def import_object(spec: str, what: str) -> Any:
    """The object ``package.module:Name`` names, for the option ``--<what>``."""
    if ":" not in spec:
        raise SystemExit(f"--{what} must look like package.module:Name, got {spec!r}")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise SystemExit(f"{module_name} has no attribute {attr!r}") from None


def launch_settings(args) -> tuple[Any, dict[str, Any], Any, str]:
    """What a launch's arguments set: the environment's configuration, the learner's keyword arguments, the task
    registry (None for the benchmark's own) and the learner's display name."""
    from continual_arc.env import EnvConfig

    config = EnvConfig.from_json(json.loads(Path(args.env_config).read_text())) if args.env_config else EnvConfig()
    kwargs = json.loads(args.learner_kwargs) if args.learner_kwargs else {}
    registry = import_object(args.registry, "registry")() if args.registry else None
    return config, kwargs, registry, args.name or args.learner.rsplit(":", 1)[-1]


def _dataset_label(schedule) -> str:
    """Which task data the schedule draws on, from its provenance."""
    dataset = (schedule.meta or {}).get("dataset") or (schedule.meta or {}).get("config", {}).get("dataset")
    if dataset in (None, "arc1"):
        return "arc-agi-1+re-arc"
    if dataset == "arc2":
        return "arc-agi-2+arc-gen"
    return str(dataset)


def source_digest() -> str:
    """sha256 over this package's Python sources (first 16 hex digits): which code ran, where the copy that ran has no
    git history."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


def dependency_revisions(names: tuple[str, ...] = ("continual-arc",)) -> dict[str, str]:
    """The exact source each dependency was installed from, for the run record. The benchmark is not pinned, so this is
    what makes a run reproducible: a git install gives ``git+<url>@<commit>``, an editable checkout
    ``editable:<path>`` (honestly unpinned), anything else the version."""
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in names:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            out[name] = "not-installed"
            continue
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        url, vcs, dir_info = info.get("url", ""), info.get("vcs_info") or {}, info.get("dir_info") or {}
        if vcs.get("commit_id"):
            out[name] = f"git+{url}@{vcs['commit_id']}"
        elif dir_info.get("editable"):
            out[name] = f"editable:{url.removeprefix('file://')}"
        else:
            out[name] = f"{dist.version}"
    return out


def run_parameters(*, learner, learner_spec, learner_kwargs, schedule, config_name, order,
                   env_config, proxy_settings, alias_style) -> dict:
    """Everything worth recording about a run, in two layers: first the scalar keys by which runs are compared
    (programme, benchmark, protocol, method, model, seed, dataset, ...), then the full detail."""
    from continual_arc import __version__ as benchmark_version
    from continual_arc.configs import protocol_of

    revisions = dependency_revisions()
    arc_rev = revisions["continual-arc"]
    arc_commit = arc_rev.rsplit("@", 1)[-1][:12] if arc_rev.startswith("git+") else arc_rev
    system = dict(learner.describe()) if hasattr(learner, "describe") else {}
    model = system.get("model") or getattr(learner, "model", None)
    proto = protocol_of(schedule, env_config)
    return {
        "programme": "continual-arc",
        "benchmark": "continual-arc",
        "track": proto["track"],
        "canonical": proto["canonical"],
        "protocol": f"{config_name or schedule.name}/{order}",
        "protocol_version": f"{benchmark_version}+{arc_commit}",
        "method": learner_spec,
        "model": model,
        "seed": schedule.seed,
        "dataset": _dataset_label(schedule),
        # the full detail
        "learner_kwargs": learner_kwargs,
        "system": system,
        "alias_style": alias_style,
        "schedule": schedule.name,
        "n_slots": len(schedule),
        "env_config": env_config.to_json(),
        "proxy": proxy_settings,
        "dependencies": revisions,
        "protocol_detail": proto,
    }


def stream_of(schedule):
    """``schedule``'s stream alone: its rounds before the epilogue's first, a literal prefix
    (``Schedule.first_rounds``), so a stream-only run plays exactly the stream a full run plays and stops there."""
    epilogue = [r for r, slot in zip(schedule.rounds(), schedule.slots, strict=True) if slot.phase == "epilogue"]
    return schedule.first_rounds(min(epilogue)) if epilogue else schedule


def can_freeze(learner) -> bool:
    """Whether ``learner`` can play every instance of a test from the state its stream left: a hosted submission that
    keeps everything it learns as files (``HostedLearner.freeze``), such as a harness, which holds one conversation
    and so could not serve a round of users."""
    from continual_arc_baselines.systems.hosted import HostedLearner

    return isinstance(learner, HostedLearner) and learner.state_in_files


def play_test(learner, tracker, test, run_dir: Path, test_dir: Path, *, config, registry, name, alias_style,
              resume, progress):
    """Play ``test`` with ``learner``, the object that has just played the run's stream, so that it starts from the
    state the stream left it in, into ``test_dir``, a run directory of its own holding the run's ``alias-salt``, so
    that the learner meets its tasks under the names the stream gave them.

    One test a launch: once the test's round has ended the learner may change what it carries from what the test
    taught it, and a learner whose state lives in its own process (a hosted submission) has no snapshot to put back,
    so a second test played after the first would start from that state, not from the stream's end. A hosted learner
    frozen at the stream's end (``HostedLearner.freeze``) plays its test one instance at a time, its files put back
    before each. A harness adapter plays its tests with ``arc-baselines test`` instead."""
    from continual_arc.aliases import SALT_FILE
    from continual_arc.runner import build_env, run

    from continual_arc_baselines.tracking import TrackedLearner

    test_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / SALT_FILE, test_dir / SALT_FILE)
    return run(TrackedLearner(learner, tracker), test, config=config, out_dir=test_dir, registry=registry,
               progress=progress, learner_name=name, env=build_env(test, config, registry), resume=resume,
               alias_style=alias_style)


def cmd_run(args) -> int:
    from continual_arc.configs import build_schedule
    from continual_arc.schedule import Schedule

    from continual_arc_baselines.frozen import holding, load_test

    if bool(args.schedule) == bool(args.config):
        raise SystemExit("give exactly one of --schedule or --config")
    if args.schedule and args.no_epilogue:
        raise SystemExit("--no-epilogue builds the schedule from --config; a --schedule is played as it is")
    if args.test_out and not args.test:
        raise SystemExit("--test-out is the directory of a --test")
    if args.schedule:
        schedule = Schedule.load(args.schedule)
    else:
        schedule = build_schedule(args.config, args.seed, order=args.order, epilogue=not args.no_epilogue)
    test = load_test(args.test) if args.test else None
    if args.stream_only or test is not None:
        schedule = stream_of(schedule)
    if test is not None and "stream" not in schedule.phases():
        raise SystemExit(f"{args.test} is a test after a stream, and the schedule has none: play a test from nothing "
                         "as the run's --schedule")
    if not args.out:
        return play(args, schedule, test=test)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with holding(Path(args.out)):
        return play(args, schedule, test=test)


def play(args, schedule, *, frozen_from: tuple[Path, int] | None = None, test=None) -> int:
    """Run the learner ``args`` names over ``schedule``, writing the run directory and the attempt. A harness's stream
    taken up again first takes its state back from a test stopped hard (:func:`continual_arc_baselines.frozen.settle`).
    ``frozen_from``, the system folder of a harness's stream and how many instances it played, makes this that stream's
    frozen test (:func:`continual_arc_baselines.frozen.play_alone`): the harness keeps its logs in the test's own
    attempt and plays every instance from a copy of the stream's state, at the stream's own path. ``test``
    (``--test``) is a frozen test played after the stream, in the same process (:func:`play_test`)."""
    from continual_arc.report import text_report
    from continual_arc.runner import build_env, prior_records, run

    from continual_arc_baselines.attempt import Attempt
    from continual_arc_baselines.frozen import give_back, hold_aside, one_at_a_time, settle
    from continual_arc_baselines.proxy import LLMProxy
    from continual_arc_baselines.systems.base import SystemLearner
    from continual_arc_baselines.tracking import JournalSink, PilotStop, TrackedLearner, Tracker

    config, kwargs, registry, name = launch_settings(args)
    if args.resume and not args.out:
        raise SystemExit("--resume needs --out: the run directory to continue")
    prior = prior_records(args.out) if args.resume else []
    pointer = Path(args.out) / "attempt.json" if args.out else None
    attempt = None
    if prior and pointer is not None and pointer.is_file():
        # Continue in the attempt the interrupted launch used: the system's persisted
        # state (memories, skills, files) lives there and carries on.
        directory = Path(json.loads(pointer.read_text())["attempt_dir"])
        if directory.is_dir():
            attempt = Attempt(directory, max_cost=config.max_cost)
    if attempt is None:
        label = name if frozen_from is None else f"{name}-{schedule.meta['test']}"
        attempt = Attempt.from_environment(label, max_cost=config.max_cost)
    out_dir = Path(args.out) if args.out else attempt.directory / "continual-arc"
    if args.out:
        out_dir.mkdir(parents=True, exist_ok=True)
        pointer.write_text(json.dumps({"attempt_dir": str(attempt.directory)}))
    if prior:
        attempt.prime(prior)
        print(f"resuming {name}: {len(prior)} instances done, continuing from {len(prior) + 1}", file=sys.stderr)

    tracker = Tracker()
    proxy = None
    proxy_settings: dict = {"enabled": False}
    if not args.no_proxy:
        defaults = dict(json.loads(args.llm_defaults) if args.llm_defaults else {})
        # OpenRouter's sticky routing key: after a cached request it keeps the conversation on
        # the provider that served it, keyed by a hash of the messages unless a session_id is
        # given. Runs of one row open with identical prompts, so the hash would treat them
        # as one session; the attempt id keeps each run's routing and cache affinity its own.
        # Applied with setdefault, so a system that sets its own wins.
        defaults.setdefault("session_id", attempt.directory.name)
        tracker.subscribe(JournalSink(attempt, provider=args.provider, capture_content=args.capture_content))
        proxy = LLMProxy(args.upstream, api_key=os.environ.get(args.upstream_key_env) or None,
                         declared_context_length=args.declared_context_length or None,
                         tracker=tracker, body_defaults=defaults,
                         capture_content=args.capture_content).start()
        os.environ.update(proxy.environ())
        proxy_settings = {"enabled": True, "upstream": args.upstream, "provider": args.provider,
                          "body_defaults": defaults, "base_url": proxy.base_url,
                          "upstream_key_present": bool(proxy.api_key)}

    # imported once the proxy's address is in the environment, which a learner's module may read as it loads
    make_learner = import_object(args.learner, "learner")
    harness = isinstance(make_learner, type) and issubclass(make_learner, SystemLearner)
    refused = None
    if args.log_state and not harness:
        refused = f"--log-state commits a SystemLearner's state directory, and {args.learner} is no SystemLearner"
    elif test is not None and harness:
        refused = (f"{args.learner} is a harness adapter, which holds one conversation: play its stream alone, then "
                   "each test with arc-baselines test")
    if refused:
        if proxy is not None:
            proxy.stop()
        raise SystemExit(refused)
    learner = make_learner(**kwargs)
    if hasattr(learner, "attach"):
        where = ({"state_dir": frozen_from[0] / "state"} if frozen_from is not None
                 else {"log_state": True} if args.log_state else {})
        learner.attach(attempt.directory / "system", proxy.environ() if proxy else {}, tracker=tracker, **where)
    frozen = test is not None and can_freeze(learner)
    if frozen:
        test = one_at_a_time(test)
    params = run_parameters(
        learner=learner, learner_spec=args.learner, learner_kwargs=kwargs, schedule=schedule,
        config_name=args.config, order=args.order, env_config=config, proxy_settings=proxy_settings,
        alias_style=args.alias_style,
    )
    attempt.parameters(params)
    hold = out_dir / "pilot-hold.json"
    if args.out:
        # What ran, in the run directory itself, which travels with the records where the attempt may not: the
        # system's description (describe(): its version and settings) and this launch's command line, one line per
        # launch or resume.
        with (out_dir / "launches.jsonl").open("a") as fh:
            fh.write(json.dumps({"at": datetime.now(UTC).isoformat(timespec="seconds"),
                                 "resume": bool(args.resume), "prior_records": len(prior), "argv": args.argv,
                                 "system": params["system"], "learner_kwargs": kwargs,
                                 "dependencies": params["dependencies"], "baselines_src": source_digest()},
                                default=str) + "\n")
        if hold.exists() and not args.stop_after:
            hold.unlink()   # launched without --stop-after: the run is held no longer

    def show(record, n_steps):
        if not args.quiet:
            flag = "ok " if record.solved else ("GU " if record.gave_up else "X  ")
            cost = record.resources.get("llm_cost_usd", 0.0)
            print(
                f"[{record.step + 1:>6}/{n_steps}] {flag} {record.phase:9s} {record.task_id} "
                f"exp={record.exposure_index} cost={record.cost:.1f} usd={cost:.4f}",
                file=sys.stderr,
            )

    def progress(record, n_steps):
        attempt.progress(record, n_steps)
        show(record, n_steps)

    test_dir = None
    if test is not None:
        test_dir = Path(args.test_out) if args.test_out else out_dir / "frozen" / test.meta["test"] / "play"
    tested = None
    system = learner.state_dir.parent if isinstance(learner, SystemLearner) else None
    try:
        if frozen_from is not None:
            settle(system, frozen_from[1])
            learner.freeze(hold_aside(system, frozen_from[1]))
        elif prior and system is not None:
            settle(system, len(prior))
        result = run(
            TrackedLearner(learner, tracker, stop_after=args.stop_after, done=len(prior)), schedule, config=config,
            out_dir=out_dir, registry=registry, log_events=args.events, progress=progress, learner_name=name,
            env=build_env(schedule, config, registry), resume=args.resume, alias_style=args.alias_style,
        )
        if test is not None:
            if frozen:
                learner.freeze()
            # the attempt's metrics are the stream's, step by step; the test's records are its own directory's
            tested = play_test(learner, tracker, test, out_dir, test_dir, config=config, registry=registry,
                               name=name, alias_style=args.alias_style, resume=args.resume, progress=show)
    except PilotStop as stop:
        # stopped by --stop-after: held, and continued by a launch with --resume and without --stop-after
        hold.write_text(json.dumps({"stopped_after": stop.done, "at": datetime.now(UTC).isoformat(
            timespec="seconds"), "argv": args.argv}))
        print(f"pilot: {stop}; held ({hold})", file=sys.stderr)
        return 0
    finally:
        if hasattr(learner, "close"):
            learner.close()
        if frozen_from is not None and learner.frozen is not None:
            give_back(system, learner.frozen)   # the stream's state where the stream left it
        if proxy is not None:
            proxy.stop()
    emitted = attempt.finish(result)
    print(text_report(result.summary, title=f"{name} on {schedule.name}"), file=sys.stderr)
    if tested is not None:
        print(text_report(tested.summary, title=f"{name} on {test.name}"), file=sys.stderr)
    print(json.dumps({
        "run_dir": str(out_dir),
        "attempt_dir": str(attempt.directory),
        "inside_workbench": attempt.given,
        "overall_score": result.summary["overall"]["score"],
        "stream_score": result.summary["stream"]["score"],
        "epilogue_score": result.summary["epilogue"]["score"],
        "n_metrics_emitted": len(emitted),
        "llm": tracker.usage.as_resources(),
        **({} if tested is None else {"test": {"run_dir": str(test_dir), "test": test.meta["test"],
                                               "score": tested.summary["benchmark"]["score"]}}),
    }))
    return 0


def cmd_test(args) -> int:
    from continual_arc_baselines import frozen

    if args.collective:
        print(frozen.play_forks(args.run_dir, args.schedule, args.out))
        return 0
    return frozen.play_alone(args.run_dir, args.schedule, args.out, quiet=args.quiet, argv=args.argv)


def cmd_fork(args) -> int:
    from continual_arc.schedule import Schedule

    from continual_arc_baselines.fork import fork

    command = fork(args.run_dir, args.round, args.out, Schedule.load(args.schedule) if args.schedule else None)
    print(" ".join(shlex.quote(part) for part in command))
    return 0


def cmd_install_submission(args) -> int:
    from continual_arc_baselines.systems.hosted import install_submission

    installed = install_submission(args.repo, args.commit, args.venv, checkout=args.checkout, cache=args.cache,
                                   python=args.python, sandbox=args.sandbox)
    print(json.dumps(installed))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arc-baselines")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run a learner or system over a schedule, writing its run directory and attempt")
    r.add_argument("--learner", required=True, help="import path, e.g. continual_arc_baselines.systems.prime_agent:PrimeAgentLearner")
    r.add_argument("--learner-kwargs", help="JSON object of constructor kwargs")
    r.add_argument("--schedule", help="schedule JSON from `continual-arc schedule make`")
    r.add_argument("--config", help="named config to build a schedule from: <track>-<size>, e.g. arc1-small, arc1-medium, arc2-small (no default track)")
    r.add_argument("--order", default="workstreams",
                   help="workstreams, the benchmark's stream; sequential or random serve its visits in another order, "
                        "only for the check that the order does not change a finding (the paper's Appendix I), and "
                        "fresh the held-out tasks alone, the learner wiped before each")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-epilogue", action="store_true",
                   help="with --config, the stream alone, as `continual-arc schedule make --no-epilogue` builds it: "
                        "the run a harness's frozen tests follow (arc-baselines test)")
    r.add_argument("--env-config", help="EnvConfig JSON file (default: benchmark defaults)")
    r.add_argument("--registry", help="import path of a zero-arg TaskRegistry factory (testing)")
    r.add_argument("--out", help="run directory (default: <attempt dir>/continual-arc)")
    r.add_argument("--name", help="learner display name")
    r.add_argument("--events", action="store_true", help="also log every action and feedback")
    r.add_argument("--alias-style", choices=ALIAS_STYLES, default="hex",
                   help="the names the system sees tasks under, drawn for the run by the runner: task-3f2a9c1e (hex) "
                        "or a codename such as amber-heron (names)")
    r.add_argument("--resume", action="store_true",
                   help="continue an interrupted run in --out: finished instances are kept, the interrupted one is "
                        "replayed, and the system's persisted state (its attempt directory) is reused")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--stream-only", action="store_true", dest="stream_only",
                   help="play the schedule's stream alone and stop before its epilogue, a literal prefix of the "
                        "schedule (stream_of)")
    r.add_argument("--log-state", action="store_true", dest="log_state",
                   help="commit a SystemLearner's whole state directory at the end of every round (statelog), so the "
                        "run can be forked from any round (arc-baselines fork) and its tests played as a round of "
                        "users (arc-baselines test --collective)")
    r.add_argument("--test", metavar="TEST.json",
                   help="after the schedule's stream (no epilogue), play this frozen test (`continual-arc schedule make "
                        "--test`, one round) with the same learner, from the state the stream left it; a hosted "
                        "learner that keeps its state as files plays its instances one at a time, its files put back "
                        "before each (HostedLearner.freeze); one test a launch (play_test). A harness adapter's tests "
                        "are arc-baselines test's")
    r.add_argument("--test-out", help="the test's run directory (default: OUT/frozen/<the schedule's test>/play)")
    r.add_argument("--stop-after", type=int, metavar="N",
                   help="stop cleanly once N instances are done, leaving pilot-hold.json in the run directory; a "
                        "launch with --resume and without --stop-after continues it")
    g = r.add_argument_group("model-call tracking (the proxy)")
    g.add_argument("--no-proxy", action="store_true", help="do not start the tracking proxy")
    g.add_argument("--upstream", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible base URL the proxy forwards to")
    g.add_argument("--upstream-key-env", default="OPENROUTER_API_KEY",
                   help="name of the environment variable that holds the upstream API key")
    g.add_argument("--provider", default="openrouter", help="provider label for the cost journal")
    g.add_argument("--llm-defaults", help='JSON of chat-completion fields to set when absent, e.g. {"reasoning_effort": "xhigh"}')
    g.add_argument("--declared-context-length", type=int, default=262_144,
                   help="context length the proxy reports on GET /models, the figure every adapter also declares "
                        "in its harness's provider entry (0 passes the provider's figure through)")
    g.add_argument("--capture-content", action="store_true",
                   help="also journal request and response bodies (large; off by default)")
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("test", help="play a frozen test after a harness's stream, every instance from the state the "
                                    "stream left (continual_arc_baselines.frozen)")
    t.add_argument("run_dir", help="the stream's run directory, from arc-baselines run --out")
    t.add_argument("--schedule", required=True,
                   help="the test, from `continual-arc schedule make --test held-out|seen`: --width 1, or its full "
                        "width with --collective")
    t.add_argument("--collective", action="store_true",
                   help="play the test as one round, every instance a user of its own forked from the commit of the "
                        "stream's end, for a run that logged its state (--log-state); without it, one instance at a "
                        "time, the stream's state put back before each")
    t.add_argument("--out", help="the test's run directory (default: RUN_DIR/frozen/<the test>/play, or, with "
                                 "--collective, <the test>-reference/play for a run with no stream)")
    t.add_argument("--quiet", action="store_true")
    t.set_defaults(func=cmd_test)
    f = sub.add_parser("fork", help="take a run that logged its state up again from the end of one of its rounds, with "
                                    "the state its learner had then: prints the command line that plays on "
                                    "(continual_arc_baselines.fork)")
    f.add_argument("run_dir")
    f.add_argument("--round", type=int, required=True,
                   help="the round whose end state the new run starts from, such as the stream's last for another epilogue")
    f.add_argument("--out", required=True, help="the new run directory, which must not exist")
    f.add_argument("--schedule", help="the schedule to go on under, serving the carried rounds exactly as the run "
                                      "did (default: the run's own)")
    f.set_defaults(func=cmd_fork)
    i = sub.add_parser("install-submission",
                       help="install a submitted learner into a new virtual environment, under the sandbox, for "
                            "continual_arc_baselines.systems.hosted:HostedLearner")
    i.add_argument("--repo", required=True, help="the submission's git repository, e.g. https://github.com/org/repo")
    i.add_argument("--commit", required=True, help="the commit to install (its full SHA)")
    i.add_argument("--venv", required=True, help="the new virtual environment (a new or empty directory)")
    i.add_argument("--checkout", help="also clone the repository at the commit here: the run's \"code\"")
    i.add_argument("--cache", help="the install's own cache, HOME and TMPDIR (default: <venv>-cache)")
    i.add_argument("--python", default="3.12", help="the Python version for the environment (default 3.12)")
    i.add_argument("--sandbox", default="auto", help="auto | bwrap | seatbelt | none")
    i.set_defaults(func=cmd_install_submission)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)   # the command line as given, for launches.jsonl
    return args.func(args)
