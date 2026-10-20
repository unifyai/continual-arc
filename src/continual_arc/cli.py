"""Command line interface: ``continual-arc --help``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from continual_arc._version import __version__
from continual_arc.env import EnvConfig

ENV_KEYS = ("attempt_cap", "demo_cap", "demo_cost", "demos_per_request", "diff_lb", "diff_ub")


def _load_learner(spec: str, kwargs: dict):
    if ":" not in spec:
        raise SystemExit("--learner must look like package.module:ClassName")
    mod_name, cls_name = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    return cls(**kwargs)


def _load_plugins(specs: list[str], kwargs_values: list[str]) -> list:
    if len(kwargs_values) > len(specs):
        raise SystemExit("--plugin-kwargs cannot have more entries than --plugin")
    plugins = []
    for index, spec in enumerate(specs):
        if ":" not in spec:
            raise SystemExit("--plugin must look like package.module:ClassName")
        module_name, class_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        raw_kwargs = kwargs_values[index] if index < len(kwargs_values) else "{}"
        kwargs = json.loads(raw_kwargs)
        if not isinstance(kwargs, dict):
            raise SystemExit("each --plugin-kwargs value must be a JSON object")
        plugins.append(cls(**kwargs))
    return plugins


def _env_config(args, schedule) -> EnvConfig:
    """The env config for a schedule: its config's ``env`` block, then CLI overrides."""
    base = {}
    cfg_meta = (schedule.meta or {}).get("config") or {}
    if cfg_meta.get("env"):
        base = {k: v for k, v in cfg_meta["env"].items() if k in ENV_KEYS}
    for key in ENV_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            base[key] = val
    return EnvConfig(**base)


# -- commands -----------------------------------------------------------------


def cmd_data(args) -> int:
    from continual_arc import data

    datasets = tuple(args.datasets) if args.datasets else (data.DATASET_ARC1,)
    for ds in datasets:
        if ds not in data.DATASETS:
            raise SystemExit(f"unknown dataset {ds!r}; choose from {data.DATASETS}")
    if args.data_cmd == "fetch":
        paths = data.fetch(force=args.force, datasets=datasets)
        for k, v in paths.items():
            print(f"{k}: {v}")
        return 0
    st = data.status(datasets)
    ok = True
    for name, s in st.items():
        ok &= bool(s["ok"])
        print(f"{name}: {'ok' if s['ok'] else 'MISSING'}  {s['path']}  head={s['head']} pinned={s['pinned']}")
    return 0 if ok else 1


def cmd_schedule(args) -> int:
    from continual_arc.schedule import Schedule

    if args.schedule_cmd == "make":
        from continual_arc.configs import build_schedule, epilogue_for_users, with_users

        config = with_users(args.config, args.users)

        if args.epilogue_users is not None:
            if args.mode is not None:
                raise SystemExit("--epilogue-users builds the held-out epilogue alone; --mode does not apply to it")
            if args.rounds is not None:
                raise SystemExit("--epilogue-users and --rounds are not combined: cut the stream, or serve its "
                                 "epilogue to some users, not both in one schedule")
            if args.no_epilogue:
                raise SystemExit("--epilogue-users already means epilogue only; --no-epilogue does not apply to it")
            sched = epilogue_for_users(config, args.seed, args.epilogue_users)
        else:
            if args.mode is None:
                raise SystemExit("--mode is required unless --epilogue-users is given")
            sched = build_schedule(config, args.mode, args.seed, epilogue=not args.no_epilogue)
            if args.rounds is not None:
                sched = sched.first_rounds(args.rounds)
        out = Path(args.out) if args.out else Path(f"{sched.name}-seed{args.seed}.json")
        sched.save(out)
        print(f"wrote {out}")
        print(json.dumps(sched.summary(), indent=1))
        return 0
    sched = Schedule.load(args.path)
    print(json.dumps(sched.summary(), indent=1))
    if args.head:
        infos = sched.annotate()
        for i, (slot, info) in enumerate(zip(sched.slots[: args.head], infos, strict=False)):
            reset = " reset" if slot.reset_before else ""
            gap = "" if info.gap is None else f" gap={info.gap}"
            print(f"{i:6d}  {slot.phase:9s} {slot.task_id}  exp={info.exposure_index}{gap}{reset}")
    return 0


def cmd_run(args) -> int:
    from continual_arc.report import text_report
    from continual_arc.runner import run
    from continual_arc.schedule import Schedule

    sched = Schedule.load(args.schedule)
    config = _env_config(args, sched)
    kwargs = json.loads(args.learner_kwargs) if args.learner_kwargs else {}
    learner = _load_learner(args.learner, kwargs)
    plugins = _load_plugins(args.plugin or [], args.plugin_kwargs or [])

    def progress(record, n_steps):
        if args.quiet:
            return
        flag = "ok " if record.solved else ("GU " if record.gave_up else "X  ")
        print(
            f"[{record.step + 1:>6}/{n_steps}] {flag} {record.phase:9s} {record.task_id} "
            f"exp={record.exposure_index} cost={record.cost:.1f}",
            file=sys.stderr,
        )

    result = run(
        learner,
        sched,
        config=config,
        out_dir=args.out,
        log_events=args.events,
        progress=progress,
        learner_name=args.name or args.learner,
        plugins=plugins,
        alias_style=args.alias_style,
    )
    print(text_report(result.summary, title=f"{args.name or args.learner} on {sched.name}"))
    if args.out:
        print(f"run written to {args.out}")
    return 0


def cmd_report(args) -> int:
    from continual_arc.metrics import summarize
    from continual_arc.report import plot, text_report
    from continual_arc.runner import load_run

    sched, config, records = load_run(args.run_dir)
    from continual_arc.configs import protocol_of

    summary = summarize(records, config, window=args.window,
                        checkpoints=(sched.meta.get("config") or {}).get("checkpoints", ()))
    summary["protocol"] = protocol_of(sched, config)
    print(text_report(summary, title=f"{args.run_dir} ({sched.name})"))
    if args.plots:
        written = plot(summary, args.run_dir)
        for p in written:
            print(f"wrote {p}")
    return 0


def cmd_compare(args) -> int:
    from continual_arc.metrics import forward_transfer
    from continual_arc.report import transfer_report
    from continual_arc.runner import load_run

    _, config, run_records = load_run(args.run_dir)
    _, _, ref_records = load_run(args.reference_dir)
    transfer = forward_transfer(run_records, ref_records, config)
    print(transfer_report(transfer))
    if args.json:
        print(json.dumps(transfer, indent=1))
    return 0


def cmd_learned(args) -> int:
    from continual_arc.metrics import share_learned
    from continual_arc.report import learned_report
    from continual_arc.runner import load_run

    _, config, run_records = load_run(args.run_dir)
    _, _, floor_records = load_run(args.floor_dir)
    learned = share_learned(run_records, floor_records, config, reps=args.reps)
    print(learned_report(learned))
    if args.json:
        print(json.dumps(learned, indent=1))
    return 0


# -- parser -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="continual-arc", description="Continual-ARC benchmark")
    p.add_argument("--version", action="version", version=f"continual-arc {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="fetch or inspect the pinned upstream data")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    df = dsub.add_parser("fetch", help="clone Re-ARC and ARC-AGI at the pinned commits")
    df.add_argument("--force", action="store_true", help="re-clone even if present")
    df.add_argument("--datasets", nargs="+",
                    help="arc1, the track (the default); arc2 also clones ARC-AGI-2, whose public tasks new rules are "
                         "screened against")
    ds = dsub.add_parser("status", help="show whether the data is present and pinned")
    ds.add_argument("--datasets", nargs="+", help="arc1 (the default) and arc2")
    d.set_defaults(func=cmd_data)

    s = sub.add_parser("schedule", help="build or inspect schedules")
    ssub = s.add_subparsers(dest="schedule_cmd", required=True)
    sm = ssub.add_parser("make", help="build a schedule from a named config")
    sm.add_argument("--config", required=True, help="<track>-<size>: arc1-{smoke,small,medium,full}")
    sm.add_argument("--mode", help="workstreams | sequential | random | fresh (the transfer reference); required unless --epilogue-users is given")
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--users", type=int, default=1,
                    help="how many users the configuration is served to at once (with_users); 1, 2, 4, 8 and 16 are "
                         "the canonical numbers")
    sm.add_argument("--no-epilogue", action="store_true", help="omit the held-out epilogue")
    sm.add_argument("--rounds", type=int, help="cut the schedule to its first N rounds (stop a stream short)")
    sm.add_argument("--epilogue-users", type=int, dest="epilogue_users",
                    help="build the held-out epilogue alone, for the first K of the --users "
                         "(each in the order and instance seeds the full schedule's own epilogue gives them); "
                         "not combined with --mode, --rounds or --no-epilogue")
    sm.add_argument("--out", help="output path (default <name>-seed<seed>.json)")
    sv = ssub.add_parser("show", help="summarise a schedule file")
    sv.add_argument("path")
    sv.add_argument("--head", type=int, default=0, help="also print the first N slots")
    s.set_defaults(func=cmd_schedule)

    r = sub.add_parser("run", help="run a learner over a schedule")
    r.add_argument("--learner", required=True, help="import path, e.g. my_pkg.learners:MyLearner")
    r.add_argument("--learner-kwargs", help="JSON object of constructor kwargs")
    r.add_argument("--schedule", required=True, help="schedule JSON from `schedule make`")
    r.add_argument("--out", help="run directory to write")
    r.add_argument("--name", help="display name for the learner")
    r.add_argument("--events", action="store_true", help="also log every action and feedback")
    r.add_argument("--plugin", action="append", help="plug-in import path; repeat for composition")
    r.add_argument(
        "--plugin-kwargs", action="append", default=[],
        help="JSON constructor object for the corresponding --plugin",
    )
    r.add_argument("--alias-style", choices=("hex", "names"), default="hex", dest="alias_style",
                   help="how the learner sees each task's name: task-3f2a9c1e (hex) or a codename such as amber-heron")
    r.add_argument("--quiet", action="store_true")
    for key, typ in (
        ("attempt_cap", int), ("demo_cap", int), ("demo_cost", float),
        ("demos_per_request", int), ("diff_lb", float), ("diff_ub", float),
    ):
        r.add_argument(f"--{key.replace('_', '-')}", type=typ, default=None, dest=key)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="print the report for a run directory")
    rp.add_argument("run_dir")
    rp.add_argument("--plots", action="store_true", help="write PNG plots into the run directory")
    rp.add_argument("--window", type=int, default=50)
    rp.set_defaults(func=cmd_report)

    c = sub.add_parser("compare", help="forward transfer of a run against a reference run")
    c.add_argument("run_dir")
    c.add_argument("reference_dir")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compare)

    lr = sub.add_parser("learned", help="share of the learnable budget a run saves beyond the demo-cache floor "
                                        "on the same model and stream")
    lr.add_argument("run_dir")
    lr.add_argument("floor_dir", help="the demo-cache floor's run on the same model and schedule")
    lr.add_argument("--reps", type=int, default=2000, help="task resamples behind the 95%% interval")
    lr.add_argument("--json", action="store_true")
    lr.set_defaults(func=cmd_learned)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
