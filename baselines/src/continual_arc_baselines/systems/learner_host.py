"""The host of a submitted learner: the one program that runs inside the sandbox for it.

Standalone on purpose. It runs with the submission's own Python, in the submission's own virtual environment, where
the standard library and the public ``continual_arc`` package are installed and this package is not.
:class:`continual_arc_baselines.systems.hosted.HostedLearner` copies this file into the attempt directory and starts
it under the sandbox as ``python -I learner_host.py package.module:Class [--code DIR]``.

The host imports the class, builds it with no arguments, attaches it as ``arc-baselines run`` attaches every learner
that has ``attach`` (its working directory, the state directory, and its environment, which points at the proxy), and
serves it over JSON lines, one request read and one reply written per call::

    {"op": "reset"}                 {"op": "begin_round", "n": 1}      {"op": "end_round"}
    {"op": "act", "obs": {...}}                                  -> {"ok": true, "result": {"type": "Submit", "grid": [[1]]}}
    {"op": "observe", "obs": {...}, "action": {...}, "feedback": {...}}
    {"op": "resources"}                                          -> {"ok": true, "result": {"calls": 3.0}}
    {"op": "close"}

Observations are ``LearnerObservation.to_json()``; actions and feedback carry their class name under ``type``
(``Submit``, ``RequestDemos``, ``GiveUp``; ``SubmitFeedback``, ``DemosFeedback``, ``GiveUpFeedback``) beside their
fields. Before any request the host writes one line saying the learner was built, and whether it keeps everything it
learns as files (``state_in_files`` on its class, as a harness adapter on the baselines' ``SystemLearner`` does), which
lets a frozen test put it back by putting its files back. An exception in the learner comes back as
``{"ok": false, "error": ..., "traceback": ...}``; after one in building it, the host exits.

The channel lives on duplicates of the original stdin and stdout. File descriptor 0 then reads /dev/null and 1 writes
to stderr, so neither the learner's prints, nor a C extension's writes, nor a process it starts can corrupt a reply or
swallow a request. ``code`` (the submission's checkout) goes on the end of ``sys.path``, after the installed packages,
for a learner kept as a plain module at the top of its repository.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import continual_arc
from continual_arc.env import (
    DemosFeedback,
    GiveUp,
    GiveUpFeedback,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.tasks import Pair

ACTIONS = {cls.__name__: cls for cls in (Submit, RequestDemos, GiveUp)}
FEEDBACK = {cls.__name__: cls for cls in (SubmitFeedback, DemosFeedback, GiveUpFeedback)}


def _build(cls, data: dict):
    """``cls`` from the fields of ``data`` it has: the trusted side's benchmark may be newer than the submission's."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def observation(data: dict) -> LearnerObservation:
    return _build(LearnerObservation, data)


def action(data: dict):
    data = dict(data)
    return _build(ACTIONS[data.pop("type")], data)


def feedback(data: dict):
    data = dict(data)
    cls = FEEDBACK[data.pop("type")]
    if cls is DemosFeedback:
        data["pairs"] = [Pair(input=p["input"], output=p["output"], note=p.get("note")) for p in data["pairs"]]
    return _build(cls, data)


def encode_action(value) -> dict:
    if not isinstance(value, tuple(ACTIONS.values())):
        raise TypeError(f"act() returned {value!r}, not a Submit, RequestDemos or GiveUp")
    out = {"type": type(value).__name__}
    if isinstance(value, Submit):
        out["grid"] = value.grid
    return out


def _plain(value):
    """What ``json`` cannot encode itself: arrays and their scalars (``tolist``)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _failure(error: BaseException) -> dict:
    return {"ok": False, "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}


def _send(replies, message: dict) -> None:
    try:
        text = json.dumps(message, default=_plain)
    except (TypeError, ValueError) as error:
        text = json.dumps(_failure(error))
    replies.write(text + "\n")
    replies.flush()


def handle(learner, request: dict):
    op = request["op"]
    if op == "reset":
        return learner.reset()
    if op == "begin_round":
        return learner.begin_round(int(request["n"]))
    if op == "end_round":
        return learner.end_round()
    if op == "act":
        return encode_action(learner.act(observation(request["obs"])))
    if op == "observe":
        return learner.observe(observation(request["obs"]), action(request["action"]), feedback(request["feedback"]))
    if op == "resources":
        return {str(k): v for k, v in dict(learner.resources() or {}).items()}
    if op == "close":
        close = getattr(learner, "close", None)
        return close() if callable(close) else None
    raise ValueError(f"unknown op {op!r}")


def serve(learner, requests, replies) -> None:
    for line in requests:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            result = handle(learner, request)
        except Exception as error:
            _send(replies, _failure(error))
            continue
        _send(replies, {"ok": True, "result": result})
        if request["op"] == "close":
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a Continual-ARC learner over JSON lines on stdin and stdout.")
    parser.add_argument("learner", help="package.module:Class, built with no arguments")
    parser.add_argument("--code", help="the submission's checkout, put on the end of sys.path")
    args = parser.parse_args(argv)

    requests = os.fdopen(os.dup(0), "r", encoding="utf-8")
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, encoding="utf-8")
    sys.stdout = sys.stderr

    if args.code:
        sys.path.append(args.code)
    try:
        module_name, _, name = args.learner.partition(":")
        if not module_name or not name:
            raise ValueError(f"the learner must look like package.module:Class, got {args.learner!r}")
        learner = getattr(importlib.import_module(module_name), name)()
        attach = getattr(learner, "attach", None)
        if callable(attach):
            # an adapter names its files from the workdir it is given; with none it falls back to a relative path,
            # which a harness it starts in another folder, or in a sandbox of its own, cannot find
            attach(Path.cwd(), dict(os.environ))
    except Exception as error:
        _send(replies, _failure(error))
        return 1
    cls = type(learner)
    _send(replies, {"ok": True, "result": {
        "ready": True, "learner": f"{cls.__module__}.{cls.__qualname__}", "python": sys.version.split()[0],
        "continual_arc": continual_arc.__version__,
        "concurrent_users": bool(getattr(learner, "concurrent_users", False)),
        "state_in_files": bool(getattr(learner, "state_in_files", False))}})
    serve(learner, requests, replies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
