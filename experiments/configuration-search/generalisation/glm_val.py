"""glm_val.py ARM RUN_DIR [--schedule S] [--alias-salt F] [--harness val|tests]: GLM-5.3-flash itself, served through
OpenRouter, playing a schedule in the student's harness, a fresh user an instance, the turn-based budget, at one
reasoning setting (ARMS). By default R's 100 validation puzzles as the generalisation search's validations play a
model (loop.py play under grid-vals.sh: his schedule and task aliases, the harness at prime-agent's own compaction
point), so as to read whether GLM plays better or worse for its reasoning when neither a clock nor a reply cap binds
(the author, 26 Oct 2026: "does GLM 5.3 Flash also perform worse with reasoning turned on? ... assuming we don't have
timeouts or hard token caps"). The capacity ablation (search/capacity) plays GLM's training games with it, each a
schedule of one instance under its run's alias salt, in the harness of the student's frozen tests (``--harness tests``).

- Reasoning cannot be switched off: OpenRouter refuses ``reasoning.enabled: false`` and ``effort: none`` for this model
  ("Reasoning is mandatory for this endpoint", 26 Oct 2026), so the least is ``minimal``, which reasoned in 6 tokens
  over a sum; ``medium`` is R's labeller's setting and the reference rows'; ``high`` with the chat template's
  thinking flag is forward.py's ``THINKING_ON``, our labeller's.
- The harness (HARNESSES): ``val`` is his validation's, prime-agent's own compaction point, with no reply cap short of
  the provider's own: Z.ai's 131,072 tokens for the model (OpenRouter's endpoint list), in a 262,144-token window, so
  prime-agent compacts as late as a reply that long allows; prime-agent asks for at most 32,000 tokens a reply whatever
  its maxTokens (the cap every student's validation ran under), so the proxy's overrides set the request's
  max_completion_tokens itself. ``tests`` is the student's frozen tests' (the bundle's paper.play_round, and GLM's own
  h0 row, whose seen test solved 94 of 100): loop.HARNESS's compaction at 49,152 tokens in the student's
  131,072-token window, each reply at prime-agent's own cap, so that every call GLM makes is one the student could be
  served and trained on.
- Z.ai alone, every arm on one provider: Fireworks answered 429 to every call that night, and the paper's reference
  rows moved off GMICloud for its broken replies on 25 Oct.
- A rate-limited or failing call is asked again until answered (PatientProxy), so that API traffic never becomes an
  empty turn and a forced give-up, which the proxy's default six retries, about two minutes, would allow. Every call
  that was asked again adds a line to RUN_DIR/retries.jsonl (its time, retries and status), which the capacity
  ablation's generator backs off on.

Run on a machine set up from the turn-based bundle (setup.sh's trees: ~/loop and the rest), from ~/prime-rl, with the
OpenRouter key in ~/.openrouter.env (launch.sh's), which never reaches a command line:
  cd ~/prime-rl && setsid nohup uv run --no-sync --env-file ~/.openrouter.env python ~/gen/tools/glm_val.py minimal \\
      ~/runs/rt/glm-reasoning/val-minimal > ~/runs/rt/glm-reasoning/val-minimal.log 2>&1 < /dev/null &
Each finished game is captured whole in RUN_DIR/captured/uNNN.json.gz as loop.py play captures it, and the run's
records.jsonl is the benchmark's.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

HOME = Path.home()
MODEL = "z-ai/glm-5.3-flash"
ARMS: dict[str, dict[str, Any]] = {
    "minimal": {"reasoning": {"effort": "minimal"}},
    "medium": {"reasoning": {"effort": "medium"}},
    "high": {"reasoning": {"enabled": True, "effort": "high"}, "chat_template_kwargs": {"enable_thinking": True}},
}
PROVIDER = {"order": ["z-ai"], "allow_fallbacks": False}
# each harness's window and reply cap, and whether the cap goes in the request (the module's docstring); its
# compaction point is loop.harness_of(0, 0) for val and loop.HARNESS for tests
HARNESSES: dict[str, dict[str, Any]] = {
    "val": {"context_window": 262_144, "max_tokens": 131_072, "cap_in_request": True},
    "tests": {"context_window": 131_072, "max_tokens": 32_768, "cap_in_request": False},
}
UPSTREAM_RETRIES = 1_000   # at most a minute apart: about 16 hours of a throttled provider before a call fails
SCHEDULE = HOME / "runs/loop/sched-val.json"   # R's 100 puzzles, a fresh user each (cell.py's val_inputs)
ALIAS_SALT = HOME / "runs/loop/r0-val/alias-salt"


def settings(arm: str, budget: dict[str, Any], harness: dict[str, Any],
             name: str = "val") -> tuple[dict[str, Any], dict[str, Any]]:
    """prime-agent's settings and the request fields that replace the harness's own, for ``arm`` in harness ``name``
    (HARNESSES), whose compaction settings are ``harness``."""
    if arm not in ARMS:
        raise SystemExit(f"ARM is one of {', '.join(ARMS)}, not {arm!r}")
    h = HARNESSES[name]
    kwargs = {"model": MODEL, "context_window": h["context_window"], "max_tokens": h["max_tokens"], **harness, **budget}
    cap = {"max_completion_tokens": h["max_tokens"]} if h["cap_in_request"] else {}
    return kwargs, {"model": MODEL, "provider": PROVIDER, **cap, **ARMS[arm]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("arm", choices=ARMS)
    p.add_argument("run_dir")
    p.add_argument("--schedule", default=str(SCHEDULE), help="the schedule played (default: R's 100 puzzles)")
    p.add_argument("--alias-salt", default=str(ALIAS_SALT), dest="alias_salt",
                   help="the task names' salt, copied to RUN_DIR before its first play (default: his validation's)")
    p.add_argument("--harness", choices=HARNESSES, default="val")
    a = p.parse_args()
    run_dir = Path(a.run_dir).expanduser()
    sys.path.insert(0, str(HOME / "loop"))
    import loop
    import reference
    from continual_arc.env import EnvConfig
    from continual_arc.runner import build_env, run as run_stream
    from continual_arc.schedule import Schedule
    from continual_arc.tasks import TaskRegistry
    from continual_arc_baselines.collective import CollectiveHarnessLearner
    from continual_arc_baselines.proxy import LLMProxy
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner

    retried = threading.Lock()

    class PatientProxy(LLMProxy):
        """The proxy, asking a throttled or failing upstream again until it answers, each call asked again logged."""

        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, upstream_retries=UPSTREAM_RETRIES, **kw)
            emit = self.tracker.emit

            def logged(call: Any) -> None:
                n = int(call.response_headers.get("x-proxy-upstream-retries", 0))
                if n:
                    with retried, open(run_dir / "retries.jsonl", "a") as f:
                        f.write(json.dumps({"t": time.time(), "retries": n, "status": call.status_code}) + "\n")
                emit(call)

            self.tracker.emit = logged

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    captured = run_dir / "captured"
    captured.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "alias-salt").exists():
        shutil.copy(Path(a.alias_salt).expanduser(), run_dir / "alias-salt")
    budget_name = loop.run_budget(run_dir)
    if budget_name != "turns":
        raise SystemExit(f"{run_dir} plays under {budget_name!r}: these plays are turn-based, on the turn-based bundle")
    budget = loop.HARNESS_BUDGETS[budget_name]
    harness = loop.harness_of(0, 0) if a.harness == "val" else dict(loop.HARNESS)
    kwargs, overrides = settings(a.arm, budget, harness, a.harness)
    window = HARNESSES[a.harness]["context_window"]
    (run_dir / "driver.json").write_text(json.dumps(
        {"arm": a.arm, "served": f"openrouter:{MODEL}", "overrides": overrides, "harness": harness,
         "harness_name": a.harness, "harness_budget": {"name": budget_name, **budget}, "context_window": window,
         "max_tokens": kwargs["max_tokens"], "upstream_retries": UPSTREAM_RETRIES, "schedule": a.schedule}, indent=1))

    def save_instance(t: Any) -> None:
        doc = {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
               "phase": t.phase, "solved": getattr(t, "solved", None), "turns": t.turns,
               "calls": [{"call_id": c.call_id, "turn": c.turn, "phase": c.phase, "status": c.status,
                          "finish_reason": c.finish_reason, "request": c.request, "response": c.response}
                         for c in t.calls]}
        tmp = captured / f"u{t.user:03d}.json.gz.tmp"
        try:
            with gzip.open(tmp, "wt") as fp:
                json.dump(doc, fp, default=str)
            tmp.rename(captured / f"u{t.user:03d}.json.gz")
        except Exception:  # noqa: BLE001
            logging.exception("could not save instance of user %s", t.user)

    sched = Schedule.load(Path(a.schedule).expanduser())
    learner = CollectiveHarnessLearner(lambda user: PrimeAgentLearner(**kwargs), run_dir, schedule=sched,
                                       upstream=reference.OPENROUTER, api_key=os.environ["OPENROUTER_API_KEY"],
                                       body_overrides=overrides, body_drop=("reasoning_effort",),
                                       role_map={"developer": "system"}, thinking_field="reasoning",
                                       declared_context_length=window, capture=True, snapshot=False,
                                       instance_hook=save_instance, reset_hook=lambda: None, proxy=PatientProxy)
    env = build_env(sched, EnvConfig(), TaskRegistry.default())
    try:
        run_stream(learner, sched, EnvConfig(), out_dir=run_dir, env=env, resume=True, learner_name=f"glm-{a.arm}")
    finally:
        learner.close()


if __name__ == "__main__":
    main()
